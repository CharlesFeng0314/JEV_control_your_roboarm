"""Load an immutable failed product run as safe continuation context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ActionOutcome, JevDecision


@dataclass(frozen=True)
class ResumeContext:
    source_run: Path
    user_goal: str
    scene_id: str | None
    decisions: tuple[JevDecision, ...]
    outcomes: tuple[ActionOutcome, ...]
    post_action_observation_seen: bool


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _decision(payload: dict[str, Any]) -> JevDecision:
    return JevDecision(
        next_action=str(payload["next_action"]),
        confidence=float(payload.get("confidence", 0.0)),
        probabilities={
            str(key): float(value)
            for key, value in dict(payload.get("probabilities") or {}).items()
        },
        arguments=dict(payload.get("arguments") or {}),
        answers=dict(payload.get("answers") or {}),
        latency_ms=float(payload.get("latency_ms", 0.0)),
    )


def _outcome(payload: dict[str, Any]) -> ActionOutcome:
    return ActionOutcome(
        action=str(payload["action"]),
        success=bool(payload["success"]),
        message=str(payload.get("message") or ""),
        data=dict(payload.get("data") or {}),
    )


def load_resume_context(source_run: Path) -> ResumeContext:
    source = source_run.resolve()
    session_path = source / "session.json"
    events_path = source / "events.jsonl"
    failure_path = source / "failure.json"
    if not session_path.is_file() or not events_path.is_file():
        raise ValueError("Resume source must contain session.json and events.jsonl")
    if (source / "result.json").is_file():
        raise ValueError("Completed product runs cannot be resumed")
    if not failure_path.is_file():
        raise ValueError("Only an explicitly failed product run can be resumed")

    session = json.loads(session_path.read_text(encoding="utf-8"))
    records = _records(events_path)
    pending_actions: list[str] = []
    last_finished_index: int | None = None
    for index, record in enumerate(records):
        event = record.get("event")
        payload = dict(record.get("payload") or {})
        if event == "action_started":
            pending_actions.append(str(payload.get("action") or ""))
        elif event == "action_finished":
            if not pending_actions:
                raise ValueError("Run contains action_finished without action_started")
            started = pending_actions.pop(0)
            if started != str(payload.get("action") or ""):
                raise ValueError("Run contains mismatched action start/finish events")
            last_finished_index = index
    if pending_actions:
        raise ValueError(
            "Run stopped with an action in flight; inspect robot state before continuing"
        )

    decisions = tuple(
        _decision(dict(record["payload"]))
        for record in records
        if record.get("event") == "jev_choice"
    )
    outcomes = tuple(
        _outcome(dict(record["payload"]))
        for record in records
        if record.get("event") == "action_finished"
    )
    post_action_observation_seen = False
    if last_finished_index is not None:
        post_action_observation_seen = any(
            record.get("event") in {"scene_memory_updated", "given_that_ready"}
            for record in records[last_finished_index + 1 :]
        )

    scene_id = session.get("scene_id")
    if not scene_id:
        for record in reversed(records):
            if record.get("event") != "scene_memory_updated":
                continue
            memory = (record.get("payload") or {}).get("memory") or {}
            if memory.get("scene_id"):
                scene_id = str(memory["scene_id"])
                break

    return ResumeContext(
        source_run=source,
        user_goal=str(session["user_goal"]),
        scene_id=str(scene_id) if scene_id else None,
        decisions=decisions,
        outcomes=outcomes,
        post_action_observation_seen=post_action_observation_seen,
    )
