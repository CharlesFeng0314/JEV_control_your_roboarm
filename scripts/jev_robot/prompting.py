"""Build auditable given-that state without benchmark or simulator ground truth."""

from __future__ import annotations

import json
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
    return "GIVEN THAT\n" + json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
