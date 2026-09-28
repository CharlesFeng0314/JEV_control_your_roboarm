"""Local web product UI for the JEV robot control loop."""

from __future__ import annotations

import json
import threading
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


class ProductWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _handler(
    state: WebUiState,
    run_session: SessionRunner,
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
) -> None:
    """Serve the operator UI locally until interrupted."""

    state = WebUiState()
    server = ProductWebServer((host, port), _handler(state, run_session))
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
        server.server_close()
