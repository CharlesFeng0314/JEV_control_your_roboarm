"""Adjust gripper aperture by one bounded increment."""

from backend import ManipulationBackend
from contracts import ActionResult, GripperAdjustment

_ALLOWED_COMMANDS = {"hold", "open", "close", "widen", "narrow"}
_ALLOWED_STOPS = {"position_reached", "contact", "force_limit", "visual_goal", "user_stop"}


def adjust_gripper(
    backend: ManipulationBackend,
    command: GripperAdjustment,
) -> ActionResult:
    if command.command not in _ALLOWED_COMMANDS:
        raise ValueError(f"Unsupported gripper command {command.command!r}")
    if not 0.0 <= command.step_m <= 0.005:
        raise ValueError("Gripper step must be within 0-5 mm per finger")
    if (command.command == "hold") != (command.step_m == 0.0):
        raise ValueError("Only hold may use a zero step; moving commands require a step")
    if command.stop_condition not in _ALLOWED_STOPS:
        raise ValueError(f"Unsupported gripper stop condition {command.stop_condition!r}")
    result = backend.adjust_gripper(command)
    if not result.success:
        raise RuntimeError(result.message)
    return result
