"""Local web product UI for the JEV robot control loop."""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .contracts import SessionResult
from .events import CallbackEventSink, EventSink
from .ui import humanize_service_error

SessionRunner = Callable[[str, EventSink], SessionResult]
WristFrameSource = Callable[[], dict[str, Any]]
WEB_ROOT = Path(__file__).with_name("web")


class WebUiState:
    """Thread-safe event buffer shared by the robot worker and HTTP handlers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._running = False

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append({"event": event, "payload": payload})

    def snapshot(self, since: int) -> dict[str, Any]:
        with self._lock:
            cursor = max(0, min(since, len(self._events)))
            return {
                "events": self._events[cursor:],
                "cursor": len(self._events),
                "running": self._running,
            }

    def start(self, goal: str, run_session: SessionRunner) -> bool:
        with self._lock:
            if self._running:
                return False
            self._events.clear()
            self._running = True

        def worker() -> None:
            try:
                run_session(goal, CallbackEventSink(self.emit))
            except Exception as exc:
                self.emit(
                    "session_error",
                    {
                        "type": type(exc).__name__,
                        "message": humanize_service_error(exc),
                    },
                )
            finally:
                with self._lock:
                    self._running = False

        threading.Thread(target=worker, name="jev-product-session", daemon=True).start()
        return True


class WristCameraMonitor:
    """Continuously cache the newest wrist RGB frame without blocking HTTP requests."""

    def __init__(
        self,
        source: WristFrameSource | None,
        *,
        interval_s: float = 0.35,
    ) -> None:
        self._source = source
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: bytes | None = None
        self._sequence = 0
        self._status: dict[str, Any] = {
            "available": source is not None,
            "connected": False,
            "frame_available": False,
            "sequence": 0,
            "message": (
                "Waiting for the robot wrist camera."
                if source is not None
                else "This robot driver does not expose a wrist camera preview."
            ),
        }
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._source is None or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="jev-wrist-camera",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def frame(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._frame, self._sequence

    def _run(self) -> None:
        assert self._source is not None
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                packet = self._source()
                raw_frame = packet.get("rgb_bytes")
                if isinstance(raw_frame, (bytes, bytearray)) and raw_frame:
                    frame = bytes(raw_frame)
                else:
                    path = Path(str(packet.get("rgb_path") or ""))
                    if not path.is_file():
                        raise FileNotFoundError("The wrist RGB frame is not available yet.")
                    frame = path.read_bytes()
                    if not frame:
                        raise RuntimeError("The wrist RGB frame is empty.")
                with self._lock:
                    self._sequence += 1
                    self._frame = frame
                    self._status = {
                        "available": True,
                        "connected": True,
                        "frame_available": True,
                        "sequence": self._sequence,
                        "shape": packet.get("shape"),
                        "source": "robot_wrist_rgb",
                        "message": "Live wrist camera connected.",
                    }
            except Exception as exc:
                with self._lock:
                    self._status = {
                        **self._status,
                        "available": True,
                        "connected": False,
                        "message": str(exc)[:240],
                    }
            elapsed = time.perf_counter() - started
            self._stop.wait(max(0.05, self._interval_s - elapsed))


class ProductWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _handler(
    state: WebUiState,
    run_session: SessionRunner,
    camera: WristCameraMonitor,
) -> type[BaseHTTPRequestHandler]:
    assets = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/app.css": ("app.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "JEVProduct/1.0"

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send_json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_frame(self, body: bytes, sequence: int) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Wrist-Frame-Sequence", str(sequence))
            self.end_headers()
            self.wfile.write(body)

        def _send_asset(self, filename: str, content_type: str) -> None:
            path = WEB_ROOT / filename
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/events":
                raw_since = parse_qs(parsed.query).get("since", ["0"])[0]
                try:
                    since = int(raw_since)
                except ValueError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "since must be an integer"})
                    return
                self._send_json(HTTPStatus.OK, state.snapshot(since))
                return
            if parsed.path == "/api/camera/status":
                self._send_json(HTTPStatus.OK, camera.status())
                return
            if parsed.path == "/api/camera/frame":
                frame, sequence = camera.frame()
                if frame is None:
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "No wrist frame is available yet."},
                    )
                else:
                    self._send_frame(frame, sequence)
                return
            asset = assets.get(parsed.path)
            if asset is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_asset(*asset)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/sessions":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 64_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                goal = str(request.get("goal") or "").strip()
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Send a JSON object with a goal."},
                )
                return
            if not goal:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Enter a robot goal first."})
                return
            if not state.start(goal, run_session):
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": "A robot session is already running."},
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, {"status": "started", "goal": goal})

    return Handler


def run_web_ui(
    run_session: SessionRunner,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    wrist_frame_source: WristFrameSource | None = None,
) -> None:
    """Serve the operator UI locally until interrupted."""

    state = WebUiState()
    camera = WristCameraMonitor(wrist_frame_source)
    camera.start()
    server = ProductWebServer((host, port), _handler(state, run_session, camera))
    address, selected_port = server.server_address[:2]
    url = f"http://{address}:{selected_port}/"
    print(f"JEV Robot Control is ready at {url}", flush=True)
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        server.server_close()
