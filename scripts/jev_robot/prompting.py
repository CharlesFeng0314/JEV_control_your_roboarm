"""Build auditable given-that state without benchmark or simulator ground truth."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .contracts import ActionOutcome, SceneSnapshot
from .manifest import ActionCatalog

_PROMPT_OBJECT_KEYS = (
    "object_id",
    "id",
    "label",
    "description",
    "attributes",
    "confidence",
    "pose",
    "position",
    "bbox3d_world_m",
)
_PROMPT_ACTION_DATA_KEYS = (
    "target_label",
    "search_scope",
    "visited_views",
    "minimum_target_score",
    "best_target_score",
    "target_ref",
    "destination_ref",
    "relation",
    "phase",
    "failure_reason",
    "object_id",
    "recovery_active",
    "manipulation_ready",
    "search_exhausted",
    "new_view_evaluated",
    "failure_phase",
    "required_progress",
    "pick_attempts",
)
MAX_JEV_INPUT_TOKENS = 31_000
TARGET_PROMPT_CHARS = 24_000
DIRECT_GRIPPER_RANGE_M = 0.12


def summarize_scene_object(item: Mapping[str, Any]) -> dict[str, Any]:
    """Keep identity and pose. Drop score maps, masks, and grasp geometry."""
    summary: dict[str, Any] = {}
    for key in _PROMPT_OBJECT_KEYS:
        if key not in item:
            continue
        value = item[key]
        if key in {"attributes", "pose", "position", "bbox3d_world_m"}:
            if isinstance(value, (dict, list)):
                summary[key] = value
            continue
        if isinstance(value, (str, int, float, bool)):
            summary[key] = value
    return summary


def summarize_action_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Record what the robot did without replaying that turn's scene payload."""
    raw_data = record.get("data")
    data = raw_data if isinstance(raw_data, Mapping) else {}
    summary: dict[str, Any] = {
        "action": record.get("action"),
        "success": bool(record.get("success")),
        "message": str(record.get("message") or ""),
    }
    if record.get("time"):
        summary["time"] = record["time"]
    kept: dict[str, Any] = {}
    for key in _PROMPT_ACTION_DATA_KEYS:
        value = data.get(key)
        if isinstance(value, (str, int, float, bool)):
            kept[key] = value
    arguments = data.get("arguments")
    if isinstance(arguments, Mapping):
        flat = {
            str(key): value
            for key, value in arguments.items()
            if isinstance(value, (str, int, float, bool))
        }
        if flat:
            kept["arguments"] = flat
    if kept:
        summary["data"] = kept
    return summary


def build_given_that(
    user_goal: str,
    snapshot: SceneSnapshot,
    capabilities: Mapping[str, Any],
    catalog: ActionCatalog,
    history: Sequence[ActionOutcome],
    scene_memory: Mapping[str, Any] | None = None,
    recovery_context: Mapping[str, Any] | None = None,
    perception_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    goal = user_goal.strip()
    if not goal:
        raise ValueError("User goal must not be empty")
    prompt_scene = _prompt_scene(snapshot)
    prompt_memory = _prompt_memory(scene_memory)
    _apply_stable_identities(prompt_scene, prompt_memory)
    state = {
        "schema_version": 1,
        "user_goal": goal,
        "control_contract": {
            "facts_come_from": "robot_mounted_sensors_and_robot_driver",
            "simulator_ground_truth_allowed": False,
            "jev_selects": "next_high_level_action_and_bounded_arguments",
            "deterministic_code_owns": "validation_motion_planning_and_actuation",
        },
        "robot_capabilities": _prompt_capabilities(capabilities),
        "available_actions": [
            {
                "name": action.name,
                "description": action.description,
                "inputs": action.inputs,
            }
            for action in catalog.actions
        ],
        "current_scene": prompt_scene,
        "scene_memory": prompt_memory,
        "recent_action_results": [
            summarize_action_record(outcome.to_dict()) for outcome in history[-6:]
        ],
    }
    if recovery_context:
        state["recovery_context"] = dict(recovery_context)
    if perception_progress:
        state["perception_progress"] = dict(perception_progress)
    return _fit_prompt_budget(state)


def _prompt_scene(snapshot: SceneSnapshot) -> dict[str, Any]:
    hand = snapshot.facts.get("hand_position_m")
    hand_position = (
        [float(value) for value in hand]
        if isinstance(hand, (list, tuple)) and len(hand) == 3
        else None
    )
    objects = []
    for item in snapshot.visible_objects[:16]:
        summary = summarize_scene_object(item)
        pose = item.get("pose") or item.get("position") or {}
        position = pose.get("position_m") if isinstance(pose, Mapping) else None
        distance = None
        if hand_position is not None and isinstance(position, (list, tuple)) and len(position) == 3:
            distance = math.sqrt(
                sum(
                    (float(value) - hand_position[index]) ** 2
                    for index, value in enumerate(position)
                )
            )
        grasp = item.get("grasp") or {}
        geometry_state = (
            "anchored"
            if item.get("geometry_stabilized")
            else "incomplete"
            if item.get("partial_view")
            else "complete"
        )
        summary["pick_readiness"] = {
            "geometry_state": geometry_state,
            "grasp_geometry_feasible": (
                bool(grasp.get("feasible", False)) and geometry_state != "incomplete"
            ),
            "hand_to_target_m": None if distance is None else round(distance, 4),
            "within_gripper_contact_range": (
                None if distance is None else distance <= DIRECT_GRIPPER_RANGE_M
            ),
            "requires_approach": None if distance is None else distance > DIRECT_GRIPPER_RANGE_M,
        }
        objects.append(summary)
    return {
        "sequence": snapshot.sequence,
        "source": snapshot.source,
        "captured_at": snapshot.captured_at,
        "facts": _prompt_robot_facts(snapshot.facts),
        "visible_objects": objects,
        "unknown_regions": list(snapshot.unknown_regions),
    }


def _prompt_memory(scene_memory: Mapping[str, Any] | None) -> dict[str, Any]:
    """Replay fused object belief.

    Action history for the current robot goal is supplied separately as
    ``recent_action_results``. The scene file also keeps earlier goals' actions,
    and those logs are not copied into a new goal's prompt.
    """
    memory = dict(scene_memory or {})
    memory.pop("recent_actions", None)
    known = list(memory.get("known_objects") or [])
    known.sort(
        key=lambda item: (
            not bool(item.get("identity_locked")),
            not bool(item.get("currently_visible")),
            bool(item.get("stale")),
            -int(item.get("observation_count", 0)),
        )
    )
    memory["known_objects"] = known[:16]
    memory.pop("latest_robot_facts", None)
    return memory


def _apply_stable_identities(scene: dict[str, Any], memory: Mapping[str, Any]) -> None:
    remembered = {
        str(item.get("object_id")): item
        for item in memory.get("known_objects") or []
        if item.get("object_id")
    }
    for item in scene.get("visible_objects") or []:
        stable = remembered.get(str(item.get("object_id") or ""))
        if not stable or not stable.get("label"):
            continue
        observed_label = item.get("label")
        item["observed_label"] = observed_label
        item["stable_label"] = stable["label"]
        item["label"] = stable["label"]
        if observed_label != stable["label"]:
            item["description"] = stable.get("description") or stable["label"]
        item["identity_locked"] = bool(stable.get("identity_locked"))


def _prompt_capabilities(capabilities: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "driver",
        "persistent_scene",
        "perception",
        "perception_labels",
        "motion_planner",
        "supported_actions",
        "gripper_control",
        "ground_truth_feedback",
    }
    return {key: capabilities[key] for key in keep if key in capabilities}


def _prompt_robot_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("hand_position_m", "hand_orientation_wxyz", "gripper", "pick_recovery"):
        value = facts.get(key)
        if isinstance(value, (dict, list, tuple, str, int, float, bool)):
            result[key] = value
    return result


def _serialized_chars(state: Mapping[str, Any]) -> int:
    return len(json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str))


def _fit_prompt_budget(state: dict[str, Any]) -> dict[str, Any]:
    """Keep every JEV call comfortably below its 31k-token hard limit."""

    bounded = deepcopy(state)
    known = (bounded.get("scene_memory") or {}).get("known_objects") or []
    recent = bounded.get("recent_action_results") or []
    while _serialized_chars(bounded) > TARGET_PROMPT_CHARS and len(known) > 4:
        known.pop()
    while _serialized_chars(bounded) > TARGET_PROMPT_CHARS and len(recent) > 2:
        recent.pop(0)
    if _serialized_chars(bounded) > TARGET_PROMPT_CHARS:
        labels = (bounded.get("robot_capabilities") or {}).get("perception_labels")
        if isinstance(labels, dict):
            bounded["robot_capabilities"]["perception_labels"] = {
                str(label): str(label) for label in labels
            }
    chars = _serialized_chars(bounded)
    bounded["prompt_budget"] = {
        "hard_limit_tokens": MAX_JEV_INPUT_TOKENS,
        "target_char_budget": TARGET_PROMPT_CHARS,
        "serialized_chars": chars,
        "conservative_token_estimate": chars,
    }
    return bounded


def build_parameter_state(state: Mapping[str, Any], action: str) -> dict[str, Any]:
    """Build a small action-specific state for the second JEV call."""

    selected_actions = [
        item for item in state.get("available_actions") or [] if item.get("name") == action
    ]
    recent = [
        item
        for item in state.get("recent_action_results") or []
        if item.get("action") in {action, "search_object", "observe_object"}
    ][-4:]
    current_scene = deepcopy(state.get("current_scene") or {})
    memory = deepcopy(state.get("scene_memory") or {})
    capabilities = deepcopy(state.get("robot_capabilities") or {})
    if action == "adjust_gripper":
        current_scene["visible_objects"] = []
        memory["known_objects"] = [
            item for item in memory.get("known_objects") or [] if item.get("identity_locked")
        ]
        capabilities = {
            key: capabilities[key]
            for key in ("driver", "gripper_control")
            if key in capabilities
        }
    elif action == "search_object":
        for item in current_scene.get("visible_objects") or []:
            item.pop("pick_readiness", None)
    elif action in {"pick_object", "place_object", "verify_transfer"}:
        visible_ids = {
            str(item.get("object_id"))
            for item in current_scene.get("visible_objects") or []
            if item.get("object_id")
        }
        memory["known_objects"] = [
            item
            for item in memory.get("known_objects") or []
            if item.get("identity_locked") or str(item.get("object_id")) in visible_ids
        ]
        capabilities.pop("perception_labels", None)
    parameter_state = {
        "schema_version": state.get("schema_version", 1),
        "user_goal": state.get("user_goal"),
        "selected_next_action": action,
        "parameter_selection_contract": (
            "Select only bounded arguments for selected_next_action. Stable object ids and "
            "active goal bindings outrank transient visual labels. Use pick_readiness to tell "
            "whether approach motion is still required."
        ),
        "robot_capabilities": capabilities,
        "available_actions": selected_actions,
        "current_scene": current_scene,
        "scene_memory": memory,
        "recent_action_results": recent,
    }
    return _fit_prompt_budget(parameter_state)


def render_given_that(state: Mapping[str, Any]) -> str:
    """Human-readable representation shown in the product UI and run archive."""
    lines = ["GIVEN THAT", "", f"Goal: {state.get('user_goal')}"]
    capabilities = state.get("robot_capabilities") or {}
    perception = ", ".join(capabilities.get("perception") or []) or "unavailable"
    planner = capabilities.get("motion_planner") or "unavailable"
    lines.extend(
        [
            f"Robot: mounted perception={perception}; motion planner={planner}",
            "JEV chooses actions and bounded arguments; the driver validates and executes.",
            "",
            "Available actions:",
        ]
    )
    for action in state.get("available_actions") or []:
        lines.append(f"  - {action.get('name')}: {action.get('description')}")

    recovery = state.get("recovery_context") or {}
    if recovery.get("active"):
        lines.extend(
            [
                "",
                "Active pick recovery:",
                f"  - target={recovery.get('target_ref')}; "
                f"failure_phase={recovery.get('failure_phase')}; "
                f"perception_attempts={recovery.get('perception_attempts', 0)}",
                f"  - required progress: {recovery.get('required_progress')}",
            ]
        )

    def object_line(item: Mapping[str, Any], *, remembered: bool = False) -> str:
        identity = item.get("object_id") or item.get("id") or "unknown-id"
        description = item.get("description") or item.get("label") or "unknown object"
        pose = item.get("pose") or item.get("latest_pose") or {}
        position = pose.get("position_m") if isinstance(pose, Mapping) else None
        if isinstance(position, (list, tuple)) and len(position) == 3:
            location = ", ".join(f"{1000.0 * float(value):.0f}" for value in position)
            location = f"[{location}] mm"
        else:
            location = "position unavailable"
        confidence = item.get("confidence", item.get("latest_confidence"))
        confidence_text = "" if confidence is None else f"; confidence={float(confidence):.2f}"
        state_text = ""
        if remembered:
            state_text = (
                f"; visible={bool(item.get('currently_visible'))}; "
                f"stale={bool(item.get('stale'))}; observations={item.get('observation_count', 0)}"
            )
        return f"  - {identity}: {description}; position={location}{confidence_text}{state_text}"

    scene = state.get("current_scene") or {}
    visible = scene.get("visible_objects") or []
    lines.extend(["", f"Current wrist view (sequence {scene.get('sequence', '?')}):"])
    lines.extend(object_line(item) for item in visible)
    if not visible:
        lines.append("  - no object accepted in the current view")

    memory = state.get("scene_memory") or {}
    remembered = memory.get("known_objects") or []
    lines.extend(["", f"Persistent scene memory (revision {memory.get('revision', 0)}):"])
    lines.extend(object_line(item, remembered=True) for item in remembered)
    if not remembered:
        lines.append("  - empty")

    unknown = scene.get("unknown_regions") or memory.get("unknown_regions") or []
    lines.extend(["", "Unknown or unobserved: " + (", ".join(unknown) if unknown else "none")])
    results = state.get("recent_action_results") or []
    lines.append("")
    lines.append("Recent results:")
    for result in results[-5:]:
        status = "OK" if result.get("success") else "FAILED"
        lines.append(f"  - {result.get('action')}: {status} - {result.get('message', '')}")
    if not results:
        lines.append("  - none")
    return "\n".join(lines)
