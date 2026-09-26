"""Execute a bounded tool-relative motion primitive such as stirring."""

from backend import ManipulationBackend
from contracts import ActionResult, ToolMotionCommand

_ALLOWED_PRIMITIVES = {"hold", "linear", "arc", "circle"}
_ALLOWED_FRAMES = {"tool", "end_effector", "robot_base", "target_object"}
_ALLOWED_PLANES = {"xy", "xz", "yz"}
_ALLOWED_DIRECTIONS = {"none", "clockwise", "counterclockwise"}
_ALLOWED_SPEEDS = {"guarded", "slow", "normal"}
_ALLOWED_STOPS = {"visual_goal", "contact", "force_limit", "path_complete", "user_stop"}


def execute_tool_motion(
    backend: ManipulationBackend,
    command: ToolMotionCommand,
) -> ActionResult:
    if command.primitive not in _ALLOWED_PRIMITIVES:
        raise ValueError(f"Unsupported tool primitive {command.primitive!r}")
    if command.frame not in _ALLOWED_FRAMES:
        raise ValueError(f"Unsupported tool frame {command.frame!r}")
    if command.plane not in _ALLOWED_PLANES:
        raise ValueError(f"Unsupported tool plane {command.plane!r}")
    if command.direction not in _ALLOWED_DIRECTIONS:
        raise ValueError(f"Unsupported tool direction {command.direction!r}")
    if not 0.0 <= command.radius_m <= 0.03:
        raise ValueError("Tool path radius must be within 0-30 mm")
    if not 0.0 <= command.revolutions <= 3.0:
        raise ValueError("Tool path revolutions must be within 0-3")
    if not 0.0 <= command.axial_step_m <= 0.01:
        raise ValueError("Tool axial step must be within 0-10 mm")
    if command.primitive in {"arc", "circle"}:
        if command.direction == "none" or command.radius_m == 0.0:
            raise ValueError("Arc and circle motions require direction and positive radius")
        if command.revolutions == 0.0:
            raise ValueError("Arc and circle motions require a positive turn count")
    elif command.primitive == "linear":
        if command.direction != "none" or command.radius_m != 0.0:
            raise ValueError("Linear motion cannot specify circular direction or radius")
        if command.revolutions != 0.0 or command.axial_step_m == 0.0:
            raise ValueError("Linear motion requires only a positive axial step")
    elif any(
        (
            command.direction != "none",
            command.radius_m != 0.0,
            command.revolutions != 0.0,
            command.axial_step_m != 0.0,
        )
    ):
        raise ValueError("Hold motion cannot contain path displacement")
    if command.speed_profile not in _ALLOWED_SPEEDS:
        raise ValueError(f"Unsupported speed profile {command.speed_profile!r}")
    if command.stop_condition not in _ALLOWED_STOPS:
        raise ValueError(f"Unsupported stop condition {command.stop_condition!r}")
    result = backend.execute_tool_motion(command)
    if not result.success:
        raise RuntimeError(result.message)
    return result
