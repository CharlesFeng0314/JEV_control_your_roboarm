"""Build auditable given-that state without benchmark or simulator ground truth."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import ActionOutcome, SceneSnapshot
from .manifest import ActionCatalog


def build_given_that(
    user_goal: str,
    snapshot: SceneSnapshot,
    capabilities: Mapping[str, Any],
    catalog: ActionCatalog,
    history: Sequence[ActionOutcome],
    scene_memory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    goal = user_goal.strip()
    if not goal:
        raise ValueError("User goal must not be empty")
    return {
        "schema_version": 1,
        "user_goal": goal,
        "control_contract": {
            "facts_come_from": "robot_mounted_sensors_and_robot_driver",
            "simulator_ground_truth_allowed": False,
            "jev_selects": "next_high_level_action_and_bounded_arguments",
            "deterministic_code_owns": "validation_motion_planning_and_actuation",
        },
        "robot_capabilities": dict(capabilities),
        "available_actions": [
            {
                "name": action.name,
                "description": action.description,
                "inputs": action.inputs,
            }
            for action in catalog.actions
        ],
        "current_scene": snapshot.to_dict(),
        "scene_memory": dict(scene_memory or {}),
        "recent_action_results": [outcome.to_dict() for outcome in history[-8:]],
    }


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
