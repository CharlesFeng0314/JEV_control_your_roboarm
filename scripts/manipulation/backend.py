"""Backend boundary used by deterministic actions and high-level planners."""

from __future__ import annotations

from typing import Protocol

from contracts import (
    ActionResult,
    FineMotionCommand,
    GripperAdjustment,
    ObjectObservation,
    Pose3D,
    ToolMotionCommand,
)


class ManipulationBackend(Protocol):
    def observe(self, target_label: str) -> ObjectObservation: ...

    def pick(self, observation: ObjectObservation) -> ActionResult: ...

    def place(self, destination: Pose3D) -> ActionResult: ...

    def verify(self, observation: ObjectObservation, destination: Pose3D) -> ActionResult: ...

    def adjust_end_effector(self, command: FineMotionCommand) -> ActionResult: ...

    def adjust_gripper(self, command: GripperAdjustment) -> ActionResult: ...

    def execute_tool_motion(self, command: ToolMotionCommand) -> ActionResult: ...
