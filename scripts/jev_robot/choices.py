"""Dynamic JEV questions for one decision turn in the robot control loop."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from typesafe_sdk import Choice, Noul, TypeSafeClient

from .contracts import JevDecision
from .manifest import ActionCatalog
from .prompting import build_parameter_state

DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
GATEWAY_MODEL = "typesafe-ai/jev"
GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
CONTROL_CHOICES = {
    "finish_task": "Finish only when robot-sensor evidence verifies the user's whole goal.",
    "request_user": (
        "Ask the user only when their language is ambiguous, conflicting perceived candidates "
        "cannot be resolved autonomously, or a completed wide robot-mounted search still cannot "
        "find a required object. A target missing from the current view alone requires search, "
        "not user clarification."
    ),
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
PLACEMENT_CLEARANCES = {"contact": 0.0, "close": 0.005, "safe": 0.020}


def _candidate_criteria(
    state: Mapping[str, Any],
    *,
    destination: bool,
    current_only: bool = False,
) -> dict[str, str]:
    scene = state.get("current_scene") or {}
    visible = scene.get("visible_objects") or []
    memory = state.get("scene_memory") or {}
    remembered = memory.get("known_objects") or []
    remembered_by_id = {
        str(item.get("object_id")): item for item in remembered if item.get("object_id")
    }
    bindings = memory.get("active_bindings") or {}
    binding = bindings.get("destination" if destination else "target") or {}
    committed_id = str(binding.get("object_id") or "")
    role = "destination" if destination else "target"
    criteria: dict[str, str] = {}
    for index, item in enumerate(visible):
        candidate_id = str(item.get("object_id") or item.get("id") or f"candidate_{index}")
        stable = remembered_by_id.get(candidate_id) or {}
        stable_label = str(stable.get("label") or item.get("label") or candidate_id)
        observed_label = str(item.get("description") or item.get("label") or candidate_id)
        attributes = item.get("attributes") or {}
        pose = item.get("pose") or item.get("position") or "pose unavailable"
        commitment = " This is the active goal binding." if candidate_id == committed_id else ""
        criteria[candidate_id] = (
            f"Visible {role} candidate with stable identity {stable_label!r}; current visual "
            f"description={observed_label!r}; attributes={attributes}; observed pose={pose}."
            + commitment
        )
    for index, item in enumerate(() if current_only else remembered):
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


def _search_scope_criteria(state: Mapping[str, Any]) -> dict[str, str]:
    recent = state.get("recent_action_results") or []
    failed_scopes = [
        str((item.get("data") or {}).get("search_scope"))
        for item in recent
        if item.get("action") == "search_object" and not item.get("success")
    ]
    history = ", ".join(failed_scopes) if failed_scopes else "none"
    history_note = f" Recent failed scopes: {history}."
    criteria = {
        "current_view": (
            "Do not move; inspect only the current robot-mounted camera view. Choose this only "
            "when this view has not already failed for the requested target." + history_note
        ),
        "narrow": (
            "Inspect a bounded nearby subset of robot-mounted camera viewpoints. Prefer this "
            "after current_view has failed for the requested target." + history_note
        ),
        "wide": (
            "Inspect all configured safe robot-mounted camera viewpoints. Prefer this after a "
            "narrow search has failed, or when the remembered scene remains substantially "
            "unobserved." + history_note
        ),
    }
    recovery = state.get("recovery_context") or {}
    if recovery.get("active"):
        criteria["current_view"] += (
            " A failed pick is under recovery, so this unchanged view cannot establish progress."
        )
        criteria["narrow"] += (
            " Do not choose this after a pick failure when a bounded wide recovery search has "
            "not been completed."
        )
        criteria["wide"] += (
            " Choose this once to seek a new view of the locked recovery target with "
            "grasp-feasible geometry."
        )
    return criteria


def build_questions(
    state: Mapping[str, Any],
    catalog: ActionCatalog,
) -> dict[str, Noul | Choice]:
    action_criteria = catalog.choice_criteria()
    if "search_object" in action_criteria:
        action_criteria["search_object"] += (
            " Use this when a required object is not visible in the current wrist view; expand "
            "the bounded robot-mounted search using recent failed-scope feedback before asking "
            "the user."
        )
    action_criteria.update(CONTROL_CHOICES)
    recent = state.get("recent_action_results") or []
    perception_progress = state.get("perception_progress") or {}
    repeated_without_progress = int(
        perception_progress.get("repeated_without_progress", 0)
    )
    if repeated_without_progress >= 2:
        evidence = (
            f" Robot feedback has repeated the same target geometry "
            f"{repeated_without_progress} times without material progress."
        )
        for action in ("search_object", "observe_object"):
            if action in action_criteria:
                action_criteria[action] += (
                    evidence
                    + " Do not repeat this perception action unless it can change the viewpoint "
                    "or resolve grasp feasibility."
                )
        action_criteria["request_user"] += (
            evidence
            + " Request operator review if no listed physical action can use the current facts."
        )
    recovery = state.get("recovery_context") or {}
    if recovery.get("active"):
        target_ref = recovery.get("target_ref") or "the locked target"
        geometry_ready = bool(recovery.get("geometry_ready"))
        if "observe_object" in action_criteria:
            action_criteria["observe_object"] += (
                " Do not repeat passive observation to recover a motion failure; it does not "
                "change the approach geometry."
            )
        if "adjust_gripper" in action_criteria:
            action_criteria["adjust_gripper"] += (
                " Do not close the gripper in free space while pick recovery is active."
            )
        if geometry_ready:
            if "pick_object" in action_criteria:
                action_criteria["pick_object"] += (
                    f" Recovery found grasp-feasible geometry for {target_ref}; retry this locked "
                    "target now instead of searching or observing again."
                )
            if "search_object" in action_criteria:
                action_criteria["search_object"] += (
                    " Do not search again because recovery geometry is already grasp-feasible."
                )
        else:
            if "search_object" in action_criteria:
                action_criteria["search_object"] += (
                    f" A pick failed for locked target {target_ref}. Perform at most one wide "
                    "recovery search for grasp-feasible geometry; semantic recognition alone is "
                    "not progress."
                )
            if "pick_object" in action_criteria:
                action_criteria["pick_object"] += (
                    " Retry only after recovery reports grasp-feasible geometry in the current "
                    "view."
                )
        action_criteria["request_user"] += (
            " Choose operator review if the bounded recovery search is exhausted without "
            "grasp-feasible geometry."
        )
    failed_wide_searches = sum(
        1
        for item in recent
        if item.get("action") == "search_object"
        and not item.get("success")
        and (item.get("data") or {}).get("search_scope") == "wide"
    )
    if failed_wide_searches:
        evidence = (
            f" Robot feedback contains {failed_wide_searches} completed failed wide search(es) "
            "with no accepted target."
        )
        action_criteria["search_object"] += (
            evidence
            + " Do not repeat wide search unless the scene, target description, or configured "
            "viewpoints changed."
        )
        action_criteria["request_user"] += (
            evidence
            + " With unchanged evidence, asking the user is the information-gaining next step."
        )
    return {
        "next_action": Choice(
            instructions=(
                "Choose exactly one next high-level action. Use only current robot-sensor facts, "
                "the user goal, and recent action results. Do not invent scene objects or poses. "
                "Missing current-view evidence is a reason to search autonomously, not by itself "
                "a reason to ask the user."
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
            instructions=(
                "Choose the smallest search scope that makes new perceptual progress. Do not "
                "repeat a failed scope when the robot feedback shows no new target evidence."
            ),
            criteria=_search_scope_criteria(state),
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
        "place_relation": Choice(
            instructions="Choose the spatial relation the placed target must satisfy.",
            criteria={
                "on": "Place the target on the destination's observed top surface.",
                "inside": "Place the target inside the observed destination bounds.",
                "next_to": "Place beside the destination using the smallest collision-free gap.",
                "left_of": "Place on the robot-base negative-X side of the destination.",
                "right_of": "Place on the robot-base positive-X side of the destination.",
                "in_front_of": "Place on the robot-base negative-Y side of the destination.",
                "behind": "Place on the robot-base positive-Y side of the destination.",
                "at_destination": "Place at the destination candidate's observed pose.",
            },
        ),
        "placement_clearance": Choice(
            instructions="Choose the smallest adequate geometric clearance.",
            criteria={
                "contact": "0 mm nominal gap; use for supported contact such as on/inside.",
                "close": "5 mm nominal gap.",
                "safe": "20 mm nominal gap for conservative separation.",
            },
        ),
        "placement_offset_axis": Choice(
            instructions="Choose one optional robot-base offset axis after relation solving.",
            criteria={
                "none": "No offset.",
                "x": "Offset along X.",
                "y": "Offset along Y.",
                "z": "Offset along Z.",
            },
        ),
        "placement_offset_direction": Choice(
            instructions="Choose the optional placement offset direction.",
            criteria={
                "none": "No offset.",
                "positive": "Positive axis.",
                "negative": "Negative axis.",
            },
        ),
        "placement_offset_step": Choice(
            instructions="Choose a bounded optional placement offset.",
            criteria={
                "none": "0 mm.",
                "micro": "1 mm.",
                "fine": "3 mm.",
                "small": "5 mm.",
                "medium": "10 mm.",
                "large": "25 mm maximum.",
            },
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
            instructions=(
                "Choose one complete gripper action. Finger motion is binary: grasp closes "
                "toward contact and release opens fully; never split it into incremental steps."
            ),
            criteria={
                "grasp": "Close once toward contact/force limit to grasp.",
                "release": "Open once to the validated full-open aperture to release.",
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


DECISION_QUESTION_KEYS = ("next_action", "safe_to_continue", "information_sufficient")
ACTION_QUESTION_KEYS: dict[str, tuple[str, ...]] = {
    "search_object": ("search_target", "search_scope"),
    "observe_object": ("search_target", "target_candidate"),
    "pick_object": ("target_candidate",),
    "place_object": (
        "target_candidate",
        "destination_candidate",
        "place_relation",
        "placement_clearance",
        "placement_offset_axis",
        "placement_offset_direction",
        "placement_offset_step",
    ),
    "verify_transfer": (
        "target_candidate",
        "destination_candidate",
        "place_relation",
        "placement_clearance",
        "placement_offset_axis",
        "placement_offset_direction",
        "placement_offset_step",
    ),
    "adjust_end_effector": (
        "target_candidate",
        "control_frame",
        "translation_axis",
        "translation_direction",
        "translation_step",
        "rotation_axis",
        "rotation_direction",
        "rotation_step",
        "speed_profile",
        "stop_condition",
    ),
    "adjust_gripper": ("gripper_command", "gripper_stop_condition"),
    "execute_tool_motion": (
        "target_candidate",
        "control_frame",
        "tool_primitive",
        "tool_plane",
        "rotation_direction",
        "tool_radius",
        "tool_revolutions",
        "axial_step",
        "speed_profile",
        "stop_condition",
    ),
}


def build_decision_questions(
    state: Mapping[str, Any], catalog: ActionCatalog
) -> dict[str, Noul | Choice]:
    all_questions = build_questions(state, catalog)
    return {key: all_questions[key] for key in DECISION_QUESTION_KEYS}


def build_argument_questions(
    action: str, state: Mapping[str, Any], catalog: ActionCatalog
) -> dict[str, Noul | Choice]:
    all_questions = build_questions(state, catalog)
    if "target_candidate" in ACTION_QUESTION_KEYS.get(action, ()):
        all_questions["target_candidate"] = Choice(
            instructions="Ground the user's requested target in stable object identity.",
            criteria=_candidate_criteria(
                state,
                destination=False,
                current_only=action == "pick_object",
            ),
        )
    if "destination_candidate" in ACTION_QUESTION_KEYS.get(action, ()):
        all_questions["destination_candidate"] = Choice(
            instructions="Ground the requested destination in stable object identity.",
            criteria=_candidate_criteria(
                state,
                destination=True,
                current_only=action in {"place_object", "verify_transfer"},
            ),
        )
    return {key: all_questions[key] for key in ACTION_QUESTION_KEYS.get(action, ())}


def _answer_choice(answers: Mapping[str, Any], key: str, default: str) -> str:
    answer = answers.get(key)
    return str(answer.choice) if answer is not None else default


def bounded_arguments(action: str, answers: Mapping[str, Any]) -> dict[str, Any]:
    target = _answer_choice(answers, "target_candidate", "target_not_visible")
    destination = _answer_choice(answers, "destination_candidate", "destination_not_visible")
    arguments: dict[str, Any] = {
        "target_ref": None if target.endswith(("_not_visible", "_ambiguous")) else target,
        "destination_ref": (
            None if destination.endswith(("_not_visible", "_ambiguous")) else destination
        ),
    }
    if action in {"search_object", "observe_object"}:
        arguments["target_label"] = _answer_choice(
            answers, "search_target", "unknown_target"
        )
        if action == "search_object":
            arguments["search_scope"] = _answer_choice(
                answers, "search_scope", "current_view"
            )
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
    elif action in {"place_object", "verify_transfer"}:
        arguments.update(
            {
                "relation": str(answers["place_relation"].choice),
                "clearance_m": PLACEMENT_CLEARANCES[str(answers["placement_clearance"].choice)],
                "offset_axis": str(answers["placement_offset_axis"].choice),
                "offset_direction": str(answers["placement_offset_direction"].choice),
                "offset_m": TRANSLATION_STEPS[str(answers["placement_offset_step"].choice)],
            }
        )
    return arguments


def _dump_answer(answer: Any) -> dict[str, Any]:
    if hasattr(answer, "model_dump"):
        return answer.model_dump(mode="json")
    return dict(vars(answer))


def _minor_question_payload(
    questions: Mapping[str, Noul | Choice], answers: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    payload = []
    for name, question in questions.items():
        answer = answers.get(name)
        if answer is None:
            continue
        dumped = _dump_answer(answer)
        criteria = getattr(question, "criteria", None) or {}
        probabilities = dumped.get("probabilities") or {}
        payload.append(
            {
                "name": name,
                "prompt": str(getattr(question, "instructions", "") or name),
                "selected": dumped.get("choice", dumped.get("noul")),
                "confidence": dumped.get("confidence"),
                "choices": [
                    {
                        "name": str(choice),
                        "description": str(description or ""),
                        "probability": probabilities.get(choice),
                    }
                    for choice, description in criteria.items()
                ],
            }
        )
    return tuple(payload)


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


@dataclass(frozen=True)
class JevApiSettings:
    api_key: str
    model: str
    base_url: str
    provider: str


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
            decision_response = client.system_one(
                state=dict(state), questions=build_decision_questions(state, catalog)
            )
            decision_answers = decision_response.answers
            action = str(decision_answers["next_action"].choice)
            argument_questions = build_argument_questions(action, state, catalog)
            argument_answers: Mapping[str, Any] = {}
            if argument_questions:
                parameter_state = build_parameter_state(state, action)
                argument_response = client.system_one(
                    state=parameter_state,
                    questions=argument_questions,
                )
                argument_answers = argument_response.answers
        latency_ms = (time.perf_counter() - started) * 1000.0
        answers = {**decision_answers, **argument_answers}
        next_answer = answers["next_action"]
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
            minor_questions=_minor_question_payload(argument_questions, argument_answers),
            latency_ms=latency_ms,
        )


def load_api_settings(path: Path) -> JevApiSettings:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        for name in ("AI_GATEWAY_API_KEY", "TYPESAFE_API_KEY"):
            prefix = f"{name}="
            if line.startswith(prefix) and line[len(prefix) :].strip():
                values[name] = line[len(prefix) :].strip()
    if values.get("TYPESAFE_API_KEY"):
        return JevApiSettings(
            api_key=values["TYPESAFE_API_KEY"],
            model=DEFAULT_MODEL,
            base_url=DEFAULT_BASE_URL,
            provider="typesafe_official",
        )
    if values.get("AI_GATEWAY_API_KEY"):
        return JevApiSettings(
            api_key=values["AI_GATEWAY_API_KEY"],
            model=GATEWAY_MODEL,
            base_url=GATEWAY_BASE_URL,
            provider="vercel_ai_gateway",
        )
    raise RuntimeError(f"No JEV API key is configured in {path}")


def load_api_key(path: Path) -> str:
    """Backward-compatible key-only loader."""
    return load_api_settings(path).api_key
