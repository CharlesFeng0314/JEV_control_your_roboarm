"""Stable data contracts shared by simulation actions and future Jev adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Pose3D:
    frame: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    label: str
    confidence: float
    pose: Pose3D
    source: str
    mask_area_px: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionResult:
    action: str
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FineMotionCommand:
    frame: str
    translation_axis: str
    translation_direction: str
    translation_step_m: float
    rotation_axis: str
    rotation_direction: str
    rotation_step_deg: float
    speed_profile: str
    stop_condition: str


@dataclass(frozen=True)
class GripperAdjustment:
    command: str
    stop_condition: str


@dataclass(frozen=True)
class ToolMotionCommand:
    primitive: str
    frame: str
    plane: str
    direction: str
    radius_m: float
    revolutions: float
    axial_step_m: float
    speed_profile: str
    stop_condition: str
