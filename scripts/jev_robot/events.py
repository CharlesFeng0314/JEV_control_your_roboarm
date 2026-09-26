"""Live event fan-out and immutable per-run JSONL recording."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .contracts import SessionResult, utc_now


class EventSink(Protocol):
    def emit(self, event: str, payload: dict[str, Any]) -> None: ...


class NullEventSink:
    def emit(self, event: str, payload: dict[str, Any]) -> None:
        del event, payload


class CallbackEventSink:
    def __init__(self, callback: Callable[[str, dict[str, Any]], None]):
        self.callback = callback

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.callback(event, payload)


class CompositeEventSink:
    def __init__(self, *sinks: EventSink):
        self.sinks = sinks

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        for sink in self.sinks:
            sink.emit(event, payload)


class JsonlRunRecorder:
    def __init__(
        self,
        root: Path,
        user_goal: str,
        *,
        scene_id: str | None = None,
        parent_run: Path | None = None,
    ):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        session: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_at": utc_now(),
            "user_goal": user_goal,
            "validation_scope": "end_to_end_product_loop",
        }
        if scene_id:
            session["scene_id"] = scene_id
        if parent_run is not None:
            session["parent_run"] = str(parent_run.resolve())
        (self.run_dir / "session.json").write_text(
            json.dumps(
                session,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        record = {"time": utc_now(), "event": event, "payload": payload}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fail(self, exc: BaseException) -> None:
        payload = {
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        self.emit("session_failed", payload)
        (self.run_dir / "failure.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def finish(self, result: SessionResult) -> None:
        (self.run_dir / "result.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
