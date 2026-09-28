"""Stable contracts shared by the product UI, JEV loop, and robot drivers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SceneSnapshot:
    """Facts obtained through robot-mounted sensors and robot driver state."""

    sequence: int
    source: str
    facts: dict[str, Any] = field(default_factory=dict)
    visible_objects: tuple[dict[str, Any], ...] = ()
    unknown_regions: tuple[str, ...] = ()
    captured_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionOutcome:
    action: str
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JevDecision:
    next_action: str
    confidence: float
    probabilities: dict[str, float]
    arguments: dict[str, Any] = field(default_factory=dict)
    answers: dict[str, Any] = field(default_factory=dict)
    minor_questions: tuple[dict[str, Any], ...] = ()
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionResult:
    status: str
    message: str
    turns: int
    decisions: tuple[JevDecision, ...]
    outcomes: tuple[ActionOutcome, ...]
    completed_at: str = field(default_factory=utc_now)

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["completed"] = self.completed
        return value
