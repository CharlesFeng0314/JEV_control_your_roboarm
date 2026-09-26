"""Execute one bounded Cartesian fine adjustment through the motion planner."""

from backend import ManipulationBackend
from contracts import ActionResult, FineMotionCommand

_ALLOWED_FRAMES = {"tool", "end_effector", "robot_base", "target_object"}
_ALLOWED_TRANSLATION_AXES = {"none", "x", "y", "z"}
_ALLOWED_TRANSLATION_DIRECTIONS = {"none", "positive", "negative"}
_ALLOWED_ROTATION_AXES = {"none", "roll", "pitch", "yaw"}
_ALLOWED_ROTATION_DIRECTIONS = {"none", "clockwise", "counterclockwise"}
_ALLOWED_SPEEDS = {"guarded", "slow", "normal"}
_ALLOWED_STOPS = {"visual_goal", "contact", "force_limit", "path_complete", "user_stop"}


def adjust_end_effector(
    backend: ManipulationBackend,
    command: FineMotionCommand,
) -> ActionResult:
    if command.frame not in _ALLOWED_FRAMES:
        raise ValueError(f"Unsupported fine-motion frame {command.frame!r}")
    if command.translation_axis not in _ALLOWED_TRANSLATION_AXES:
        raise ValueError(f"Unsupported translation axis {command.translation_axis!r}")
    if command.translation_direction not in _ALLOWED_TRANSLATION_DIRECTIONS:
        raise ValueError(f"Unsupported translation direction {command.translation_direction!r}")
    if not 0.0 <= command.translation_step_m <= 0.025:
        raise ValueError("Translation step must be within 0-25 mm")
    translation_disabled = command.translation_axis == "none"
    if translation_disabled != (command.translation_direction == "none"):
        raise ValueError("Translation axis and direction must enable or disable together")
    if translation_disabled != (command.translation_step_m == 0.0):
        raise ValueError("Translation axis and step must enable or disable together")
    if command.rotation_axis not in _ALLOWED_ROTATION_AXES:
        raise ValueError(f"Unsupported rotation axis {command.rotation_axis!r}")
    if command.rotation_direction not in _ALLOWED_ROTATION_DIRECTIONS:
        raise ValueError(f"Unsupported rotation direction {command.rotation_direction!r}")
    if not 0.0 <= command.rotation_step_deg <= 15.0:
        raise ValueError("Rotation step must be within 0-15 degrees")
    rotation_disabled = command.rotation_axis == "none"
    if rotation_disabled != (command.rotation_direction == "none"):
        raise ValueError("Rotation axis and direction must enable or disable together")
    if rotation_disabled != (command.rotation_step_deg == 0.0):
        raise ValueError("Rotation axis and step must enable or disable together")
    if command.speed_profile not in _ALLOWED_SPEEDS:
        raise ValueError(f"Unsupported speed profile {command.speed_profile!r}")
    if command.stop_condition not in _ALLOWED_STOPS:
        raise ValueError(f"Unsupported stop condition {command.stop_condition!r}")
    result = backend.adjust_end_effector(command)
    if not result.success:
        raise RuntimeError(result.message)
    return result
