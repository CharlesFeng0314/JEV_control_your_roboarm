"""Open the fingers fully, or close them once toward contact."""

from backend import ManipulationBackend
from contracts import ActionResult, GripperAdjustment

_ALLOWED_COMMANDS = {"open", "close"}
_ALLOWED_STOPS = {"position_reached", "contact", "force_limit", "visual_goal", "user_stop"}


def adjust_gripper(
    backend: ManipulationBackend,
    command: GripperAdjustment,
) -> ActionResult:
    if command.command not in _ALLOWED_COMMANDS:
        raise ValueError(f"Unsupported gripper command {command.command!r}")
    if command.stop_condition not in _ALLOWED_STOPS:
        raise ValueError(f"Unsupported gripper stop condition {command.stop_condition!r}")
    result = backend.adjust_gripper(command)
    if not result.success:
        raise RuntimeError(result.message)
    return result
