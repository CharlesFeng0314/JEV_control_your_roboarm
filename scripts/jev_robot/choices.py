"""Dynamic JEV questions for one decision turn in the robot control loop."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from typesafe_sdk import Choice, Noul, TypeSafeClient

from .contracts import JevDecision
from .manifest import ActionCatalog

DEFAULT_MODEL = "typesafe-ai/jev"
DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
CONTROL_CHOICES = {
    "finish_task": "Finish only when robot-sensor evidence verifies the user's whole goal.",
    "request_user": "Stop safely and ask the user when the goal or scene remains ambiguous.",
}
TRANSLATION_STEPS = {
    "none": 0.0,
    "micro": 0.001,
    "fine": 0.003,
    "small": 0.005,
    "medium": 0.010,
    "large": 0.025,
}
ROTATION_STEPS = {"none": 0.0, "micro": 1.0, "fine": 3.0, "small": 5.0, "medium": 15.0}
GRIPPER_STEPS = {"none": 0.0, "micro": 0.0005, "fine": 0.001, "small": 0.002, "medium": 0.005}
TOOL_RADII = {
    "none": 0.0,
    "micro": 0.003,
    "fine": 0.005,
    "small": 0.010,
    "medium": 0.020,
    "large": 0.030,
}
TOOL_REVOLUTIONS = {
    "none": 0.0,
    "quarter": 0.25,
    "half": 0.5,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
}
AXIAL_STEPS = {"none": 0.0, "micro": 0.001, "fine": 0.003, "small": 0.005, "medium": 0.010}


def _candidate_criteria(state: Mapping[str, Any], *, destination: bool) -> dict[str, str]:
    scene = state.get("current_scene") or {}
    visible = scene.get("visible_objects") or []
    memory = state.get("scene_memory") or {}
    remembered = memory.get("known_objects") or []
    role = "destination" if destination else "target"
    criteria: dict[str, str] = {}
    for index, item in enumerate(visible):
        candidate_id = str(item.get("object_id") or item.get("id") or f"candidate_{index}")
        label = str(item.get("description") or item.get("label") or candidate_id)
        attributes = item.get("attributes") or {}
        pose = item.get("pose") or item.get("position") or "pose unavailable"
        criteria[candidate_id] = (
            f"Visible {role} candidate {label}; attributes={attributes}; observed pose={pose}."
        )
    for index, item in enumerate(remembered):
        candidate_id = str(item.get("object_id") or f"remembered_{index}")
        if candidate_id in criteria:
            continue
        label = str(item.get("description") or item.get("label") or candidate_id)
        attributes = item.get("attributes") or {}
        pose = item.get("latest_pose") or "pose unavailable"
        stale = bool(item.get("stale", False))
        observations = int(item.get("observation_count", 0))
        criteria[candidate_id] = (
            f"Remembered {role} candidate {label}; attributes={attributes}; latest pose={pose}; "
            f"observations={observations}; stale={stale}. Re-observe before physical "
            "action when it is not currently visible or is stale."
        )
    prefix = "destination" if destination else "target"
    criteria[f"{prefix}_not_visible"] = f"The requested {prefix} is not currently visible."
    criteria[f"{prefix}_ambiguous"] = (
        f"Visible and remembered evidence does not identify one {prefix} confidently."
    )
    return criteria


def _perception_label_criteria(state: Mapping[str, Any]) -> dict[str, str]:
    capabilities = state.get("robot_capabilities") or {}
    raw = capabilities.get("perception_labels") or {}
    criteria = {
        str(label): str(description)
        for label, description in raw.items()
        if str(label).strip() and str(description).strip()
    }
    if criteria:
        return criteria
    return {
        "unknown_target": (
            "The requested object class is outside the currently advertised robot-vision "
            "vocabulary; ask the user instead of inventing a class."
        )
    }


def build_questions(
    state: Mapping[str, Any],
    catalog: ActionCatalog,
) -> dict[str, Noul | Choice]:
    action_criteria = catalog.choice_criteria()
    action_criteria.update(CONTROL_CHOICES)
    return {
        "next_action": Choice(
            instructions=(
                "Choose exactly one next high-level action. Use only current robot-sensor facts, "
                "the user goal, and recent action results. Do not invent scene objects or poses."
            ),
            criteria=action_criteria,
        ),
        "safe_to_continue": Noul(
            instructions=(
                "Can the selected next step be attempted through the listed robot actions without "
                "inventing coordinates, using simulator ground truth, or bypassing validation?"
            )
        ),
        "information_sufficient": Noul(
            instructions="Are current robot-mounted observations sufficient for a physical action?",
        ),
        "search_scope": Choice(
            instructions="If searching is needed, choose the smallest adequate search scope.",
            criteria={
                "current_view": "Do not move; inspect only the current robot-mounted camera view.",
                "narrow": "Inspect a bounded nearby subset of robot-mounted camera viewpoints.",
                "wide": "Inspect all configured safe robot-mounted camera viewpoints.",
            },
        ),
        "search_target": Choice(
            instructions=(
                "Choose the object class requested by the user for observe/search. This is only "
                "the robot vision vocabulary, not a claim that the object exists in the scene."
            ),
            criteria=_perception_label_criteria(state),
        ),
        "target_candidate": Choice(
            instructions="Ground the user's requested target in current visible evidence.",
            criteria=_candidate_criteria(state, destination=False),
        ),
        "destination_candidate": Choice(
            instructions="Ground the user's requested destination in current visible evidence.",
            criteria=_candidate_criteria(state, destination=True),
        ),
        "control_frame": Choice(
            instructions="Choose the coordinate frame for a bounded fine or tool motion.",
            criteria={
                "tool": "Move relative to the currently held tool.",
                "end_effector": "Move relative to the gripper/end-effector frame.",
                "robot_base": "Move along robot-base Cartesian axes.",
                "target_object": "Move relative to the currently grounded target object.",
            },
        ),
        "translation_axis": Choice(
            instructions="Choose at most one Cartesian translation degree of freedom.",
            criteria={
                "none": "No translation in this fine step.",
                "x": "Translate along the selected frame's X axis.",
                "y": "Translate along the selected frame's Y axis.",
                "z": "Translate along the selected frame's Z axis.",
            },
        ),
        "translation_direction": Choice(
            instructions="Choose the translation direction.",
            criteria={
                "none": "No translation.",
                "positive": "Move in the positive selected axis.",
                "negative": "Move in the negative selected axis.",
            },
        ),
        "translation_step": Choice(
            instructions="Choose the smallest adequate bounded translation increment.",
            criteria={
                "none": "0 mm.",
                "micro": "1 mm.",
                "fine": "3 mm.",
                "small": "5 mm.",
                "medium": "10 mm.",
                "large": "25 mm maximum.",
            },
        ),
        "rotation_axis": Choice(
            instructions="Choose at most one rotational degree of freedom.",
            criteria={
                "none": "No rotation.",
                "roll": "Rotate around tool/frame X.",
                "pitch": "Rotate around tool/frame Y.",
                "yaw": "Rotate around tool/frame Z.",
            },
        ),
        "rotation_direction": Choice(
            instructions="Choose the bounded rotation or circular path direction.",
            criteria={
                "none": "No rotation.",
                "clockwise": "Clockwise around the selected axis.",
                "counterclockwise": "Counterclockwise around the selected axis.",
            },
        ),
        "rotation_step": Choice(
            instructions="Choose the smallest adequate bounded angular increment.",
            criteria={
                "none": "0 degrees.",
                "micro": "1 degree.",
                "fine": "3 degrees.",
                "small": "5 degrees.",
                "medium": "15 degrees maximum.",
            },
        ),
        "gripper_command": Choice(
            instructions="Choose one bounded gripper adjustment.",
            criteria={
                "hold": "Keep the current aperture.",
                "open": "Open from a closed state.",
                "close": "Close toward contact.",
                "widen": "Increase aperture by one selected increment.",
                "narrow": "Decrease aperture by one selected increment.",
            },
        ),
        "gripper_step": Choice(
            instructions="Choose the per-finger gripper increment.",
            criteria={
                "none": "0 mm.",
                "micro": "0.5 mm.",
                "fine": "1 mm.",
                "small": "2 mm.",
                "medium": "5 mm maximum.",
            },
        ),
        "tool_primitive": Choice(
            instructions="Choose a bounded tool trajectory primitive.",
            criteria={
                "hold": "Hold tool pose.",
                "linear": "Move the tool along a straight segment.",
                "arc": "Move the tool along a bounded arc.",
                "circle": "Move the tool in a bounded circle, such as stirring.",
            },
        ),
        "tool_plane": Choice(
            instructions="Choose the plane for an arc or circular tool path.",
            criteria={
                "xy": "Circle/arc in XY.",
                "xz": "Circle/arc in XZ.",
                "yz": "Circle/arc in YZ.",
            },
        ),
        "tool_radius": Choice(
            instructions="Choose the smallest adequate tool path radius.",
            criteria={
                "none": "0 mm.",
                "micro": "3 mm.",
                "fine": "5 mm.",
                "small": "10 mm.",
                "medium": "20 mm.",
                "large": "30 mm maximum.",
            },
        ),
        "tool_revolutions": Choice(
            instructions="Choose a bounded arc fraction or number of revolutions.",
            criteria={
                "none": "0 turns.",
                "quarter": "0.25 turns.",
                "half": "0.5 turns.",
                "one": "1 turn.",
                "two": "2 turns.",
                "three": "3 turns maximum.",
            },
        ),
        "axial_step": Choice(
            instructions="Choose tool insertion or withdrawal distance per primitive.",
            criteria={
                "none": "0 mm.",
                "micro": "1 mm.",
                "fine": "3 mm.",
                "small": "5 mm.",
                "medium": "10 mm maximum.",
            },
        ),
        "speed_profile": Choice(
            instructions="Choose a validated motion speed profile.",
            criteria={
                "guarded": "Minimum speed with contact/force guard.",
                "slow": "Slow fine manipulation.",
                "normal": "Normal validated motion.",
            },
        ),
        "stop_condition": Choice(
            instructions="Choose the condition that terminates a fine/tool motion.",
            criteria={
                "visual_goal": "Stop when robot-mounted perception confirms the goal.",
                "contact": "Stop on detected contact.",
                "force_limit": "Stop at the validated force/torque limit.",
                "path_complete": "Stop when the bounded path completes.",
                "user_stop": "Stop if the user requests it.",
            },
        ),
        "gripper_stop_condition": Choice(
            instructions="Choose the condition that terminates gripper adjustment.",
            criteria={
                "position_reached": "Stop at the validated aperture.",
                "contact": "Stop on gripper contact.",
                "force_limit": "Stop at the validated grip force limit.",
                "visual_goal": "Stop when mounted vision confirms alignment.",
                "user_stop": "Stop if the user requests it.",
            },
        ),
    }


def bounded_arguments(action: str, answers: Mapping[str, Any]) -> dict[str, Any]:
    target = str(answers["target_candidate"].choice)
    destination = str(answers["destination_candidate"].choice)
    arguments: dict[str, Any] = {
        "target_ref": None if target.endswith(("_not_visible", "_ambiguous")) else target,
        "destination_ref": (
            None if destination.endswith(("_not_visible", "_ambiguous")) else destination
        ),
    }
    if action in {"search_object", "observe_object"}:
        arguments["target_label"] = str(answers["search_target"].choice)
        if action == "search_object":
            arguments["search_scope"] = str(answers["search_scope"].choice)
    elif action == "adjust_end_effector":
        arguments.update(
            {
                "frame": str(answers["control_frame"].choice),
                "translation_axis": str(answers["translation_axis"].choice),
                "translation_direction": str(answers["translation_direction"].choice),
                "translation_step_m": TRANSLATION_STEPS[str(answers["translation_step"].choice)],
                "rotation_axis": str(answers["rotation_axis"].choice),
                "rotation_direction": str(answers["rotation_direction"].choice),
                "rotation_step_deg": ROTATION_STEPS[str(answers["rotation_step"].choice)],
                "speed_profile": str(answers["speed_profile"].choice),
                "stop_condition": str(answers["stop_condition"].choice),
            }
        )
    elif action == "adjust_gripper":
        arguments.update(
            {
                "command": str(answers["gripper_command"].choice),
                "step_m": GRIPPER_STEPS[str(answers["gripper_step"].choice)],
                "stop_condition": str(answers["gripper_stop_condition"].choice),
            }
        )
    elif action == "execute_tool_motion":
        arguments.update(
            {
                "primitive": str(answers["tool_primitive"].choice),
                "frame": str(answers["control_frame"].choice),
                "plane": str(answers["tool_plane"].choice),
                "direction": str(answers["rotation_direction"].choice),
                "radius_m": TOOL_RADII[str(answers["tool_radius"].choice)],
                "revolutions": TOOL_REVOLUTIONS[str(answers["tool_revolutions"].choice)],
                "axial_step_m": AXIAL_STEPS[str(answers["axial_step"].choice)],
                "speed_profile": str(answers["speed_profile"].choice),
                "stop_condition": str(answers["stop_condition"].choice),
            }
        )
    return arguments


def _dump_answer(answer: Any) -> dict[str, Any]:
    if hasattr(answer, "model_dump"):
        return answer.model_dump(mode="json")
    return dict(vars(answer))


def guard_action(
    action: str,
    confidence: float,
    safe_score: float,
    perception_sufficiency: float,
    available_actions: set[str],
) -> str:
    """Keep the highest-ranked JEV Choice; probabilities are advisory telemetry."""
    del confidence, safe_score, perception_sufficiency, available_actions
    return action


class DecisionEngine(Protocol):
    def decide(self, state: Mapping[str, Any], catalog: ActionCatalog) -> JevDecision: ...


class JevDecisionEngine:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        if not api_key:
            raise ValueError("JEV API key must not be empty")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def decide(self, state: Mapping[str, Any], catalog: ActionCatalog) -> JevDecision:
        started = time.perf_counter()
        with TypeSafeClient(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
        ) as client:
            response = client.system_one(
                state=dict(state), questions=build_questions(state, catalog)
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        answers = response.answers
        next_answer = answers["next_action"]
        action = str(next_answer.choice)
        confidence = float(next_answer.confidence)
        safe = float(answers["safe_to_continue"].noul)
        sufficient = float(answers["information_sufficient"].noul)
        arguments = bounded_arguments(action, answers)
        dumped_answers = {key: _dump_answer(value) for key, value in answers.items()}
        dumped_answers["decision_trace"] = {
            "selected_action": action,
            "selection_policy": "execute_highest_ranked_jev_choice",
            "choice_confidence": confidence,
            "safety_score_advisory": safe,
            "information_sufficient_advisory": sufficient,
        }
        return JevDecision(
            next_action=action,
            confidence=confidence,
            probabilities={
                str(key): float(value) for key, value in next_answer.probabilities.items()
            },
            arguments=arguments,
            answers=dumped_answers,
            latency_ms=latency_ms,
        )


def load_api_key(path: Path) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        for name in ("AI_GATEWAY_API_KEY", "TYPESAFE_API_KEY"):
            prefix = f"{name}="
            if line.startswith(prefix) and line[len(prefix) :].strip():
                return line[len(prefix) :].strip()
    raise RuntimeError(f"No JEV API key is configured in {path}")
