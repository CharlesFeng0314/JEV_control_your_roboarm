"""Multi-turn JEV/action loop shared by the CLI and product GUI."""

from __future__ import annotations

from .action_bridge import ActionBridge
from .choices import CONTROL_CHOICES, DecisionEngine
from .contracts import ActionOutcome, JevDecision, SessionResult
from .events import EventSink, NullEventSink
from .manifest import ActionCatalog
from .prompting import build_given_that, render_given_that
from .scene_memory import SceneMemory

PICK_RECOVERY_PERCEPTION_LIMIT = 3
PERCEPTION_STALL_LIMIT = 3
PERCEPTION_ACTIONS = {"search_object", "observe_object"}


class ProductOrchestrator:
    def __init__(
        self,
        catalog: ActionCatalog,
        engine: DecisionEngine,
        bridge: ActionBridge,
        *,
        sink: EventSink | None = None,
        max_turns: int = 40,
        memory: SceneMemory | None = None,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.catalog = catalog
        self.engine = engine
        self.bridge = bridge
        self.sink = sink or NullEventSink()
        self.memory = memory
        self.max_turns = max_turns

    @staticmethod
    def _end_to_end_ready(
        decisions: list[JevDecision],
        outcomes: list[ActionOutcome],
        post_action_observation_seen: bool,
    ) -> bool:
        acted = any(item.action != "finish_task" and item.success for item in outcomes)
        return len(decisions) >= 2 and acted and post_action_observation_seen

    def _result(
        self,
        status: str,
        message: str,
        decisions: list[JevDecision],
        outcomes: list[ActionOutcome],
    ) -> SessionResult:
        result = SessionResult(
            status=status,
            message=message,
            turns=len(decisions),
            decisions=tuple(decisions),
            outcomes=tuple(outcomes),
        )
        self.sink.emit("session_finished", result.to_dict())
        return result

    @staticmethod
    def _advance_pick_recovery(
        recovery: dict[str, object] | None,
        outcome: ActionOutcome,
    ) -> dict[str, object] | None:
        data = outcome.data
        if outcome.action == "pick_object":
            if outcome.success:
                return None
            return {
                "active": True,
                "target_ref": data.get("target_ref"),
                "target_label": data.get("target_label"),
                "failure_phase": data.get("failure_phase", data.get("phase", "UNKNOWN")),
                "failure_reason": data.get("failure_reason", outcome.message),
                "required_progress": data.get(
                    "required_progress", "new_view_with_grasp_feasible_geometry"
                ),
                "perception_attempts": 0,
                "geometry_ready": False,
                "exhausted": False,
            }
        if recovery is None or outcome.action not in PERCEPTION_ACTIONS:
            return recovery

        updated = dict(recovery)
        attempts = int(updated.get("perception_attempts", 0)) + 1
        geometry_was_ready = bool(updated.get("geometry_ready"))
        observation = data.get("observation")
        observation = observation if isinstance(observation, dict) else {}
        grasp = observation.get("grasp")
        target_ref = updated.get("target_ref")
        observed_refs = (
            observation.get("object_id") or observation.get("id"),
            data.get("target_ref"),
        )
        target_matches = not target_ref or all(
            not reference or reference == target_ref for reference in observed_refs
        )
        geometry_ready = bool(
            outcome.success
            and data.get("manipulation_ready", True)
            and target_matches
            and isinstance(grasp, dict)
            and grasp.get("feasible", False)
            and (not observation.get("partial_view") or observation.get("geometry_stabilized"))
        )
        updated.update(
            {
                "perception_attempts": attempts,
                "geometry_ready": geometry_ready,
                "last_perception_action": outcome.action,
                "last_perception_success": outcome.success,
            }
        )
        wide_exhausted = bool(
            outcome.action == "search_object"
            and data.get("search_scope") == "wide"
            and data.get("search_exhausted")
        )
        updated["exhausted"] = (
            wide_exhausted
            or (geometry_was_ready and geometry_ready)
            or (attempts >= PICK_RECOVERY_PERCEPTION_LIMIT and not geometry_ready)
        )
        return updated

    @classmethod
    def _recovery_from_history(
        cls, outcomes: list[ActionOutcome]
    ) -> dict[str, object] | None:
        recovery = None
        for outcome in outcomes:
            recovery = cls._advance_pick_recovery(recovery, outcome)
        return recovery

    @staticmethod
    def _perception_signature(outcome: ActionOutcome) -> dict[str, object] | None:
        if outcome.action not in PERCEPTION_ACTIONS:
            return None
        data = outcome.data
        observation = data.get("observation")
        if not isinstance(observation, dict):
            arguments = data.get("arguments")
            arguments = arguments if isinstance(arguments, dict) else {}
            return {
                "target_label": data.get("target_label", arguments.get("target_label")),
                "search_scope": data.get("search_scope", arguments.get("search_scope")),
                "object_id": None,
            }
        pose = observation.get("pose")
        position = pose.get("position_m") if isinstance(pose, dict) else None
        position_bucket = None
        if isinstance(position, (list, tuple)) and len(position) == 3:
            position_bucket = tuple(round(float(value) / 0.02) for value in position)
        grasp = observation.get("grasp")
        grasp = grasp if isinstance(grasp, dict) else {}
        raw_opening = grasp.get("required_opening_m")
        opening_bucket = (
            None if raw_opening is None else round(float(raw_opening) / 0.005)
        )
        return {
            "target_label": data.get("target_label"),
            "object_id": observation.get("object_id"),
            "position_bucket_20mm": position_bucket,
            "grasp_feasible": grasp.get("feasible"),
            "opening_bucket_5mm": opening_bucket,
        }

    @classmethod
    def _advance_perception_progress(
        cls,
        progress: dict[str, object] | None,
        outcome: ActionOutcome,
    ) -> dict[str, object] | None:
        signature = cls._perception_signature(outcome)
        if signature is None:
            return None
        repeated = 1
        if progress is not None and progress.get("signature") == signature:
            repeated = int(progress.get("repeated_without_progress", 1)) + 1
        return {
            "active": True,
            "signature": signature,
            "repeated_without_progress": repeated,
            "limit": PERCEPTION_STALL_LIMIT,
            "exhausted": repeated >= PERCEPTION_STALL_LIMIT,
        }

    @classmethod
    def _perception_progress_from_history(
        cls, outcomes: list[ActionOutcome]
    ) -> dict[str, object] | None:
        progress = None
        for outcome in outcomes:
            progress = cls._advance_perception_progress(progress, outcome)
        return progress

    def run(
        self,
        user_goal: str,
        *,
        initial_decisions: tuple[JevDecision, ...] = (),
        initial_outcomes: tuple[ActionOutcome, ...] = (),
        post_action_observation_seen: bool = False,
    ) -> SessionResult:
        decisions = list(initial_decisions)
        outcomes = list(initial_outcomes)
        pick_recovery = self._recovery_from_history(outcomes)
        perception_progress = self._perception_progress_from_history(outcomes)
        capabilities = self.bridge.driver.capabilities()
        self.sink.emit(
            "session_started",
            {
                "user_goal": user_goal,
                "continued_decisions": len(initial_decisions),
                "continued_outcomes": len(initial_outcomes),
            },
        )
        if self.memory is not None:
            self.memory.begin_goal(user_goal)
        first_turn = len(decisions) + 1
        for turn in range(first_turn, first_turn + self.max_turns):
            snapshot = self.bridge.driver.snapshot()
            if any(item.action != "finish_task" and item.success for item in outcomes):
                post_action_observation_seen = True
            memory_view = self.memory.observe(snapshot) if self.memory is not None else {}
            if self.memory is not None:
                self.sink.emit("scene_memory_updated", {"turn": turn, "memory": memory_view})
            state = build_given_that(
                user_goal,
                snapshot,
                capabilities,
                self.catalog,
                outcomes,
                memory_view,
                pick_recovery,
                perception_progress,
            )
            self.sink.emit(
                "given_that_ready",
                {"turn": turn, "state": state, "rendered": render_given_that(state)},
            )
            choices = [
                {
                    "name": action["name"],
                    "description": action["description"],
                }
                for action in state["available_actions"]
            ]
            choices.extend(
                {"name": name, "description": description}
                for name, description in CONTROL_CHOICES.items()
            )
            self.sink.emit(
                "jev_question",
                {
                    "turn": turn,
                    "question": (
                        "Given the goal and current sensor evidence, "
                        "what should the robot do next?"
                    ),
                    "choices": choices,
                },
            )
            decision = self.engine.decide(state, self.catalog)
            decisions.append(decision)
            self.sink.emit("jev_choice", {"turn": turn, **decision.to_dict()})
            if decision.next_action == "request_user":
                return self._result(
                    "needs_user",
                    "JEV requested user clarification or operator review.",
                    decisions,
                    outcomes,
                )
            if decision.next_action == "finish_task":
                if self._end_to_end_ready(
                    decisions,
                    outcomes,
                    post_action_observation_seen,
                ):
                    return self._result(
                        "completed",
                        "The complete JEV/action/robot-feedback loop was verified.",
                        decisions,
                        outcomes,
                    )
                rejected = ActionOutcome(
                    action="finish_task",
                    success=False,
                    message="Finish rejected: end-to-end robot-sensor verification is missing.",
                )
                outcomes.append(rejected)
                if self.memory is not None:
                    self.memory.record_outcome(rejected, goal=user_goal)
                self.sink.emit("finish_rejected", rejected.to_dict())
                continue
            if self.memory is not None:
                self.memory.bind_object(
                    "target", decision.arguments.get("target_ref"), goal=user_goal
                )
                self.memory.bind_object(
                    "destination", decision.arguments.get("destination_ref"), goal=user_goal
                )
            self.sink.emit(
                "action_started",
                {
                    "turn": turn,
                    "action": decision.next_action,
                    "arguments": decision.arguments,
                },
            )
            outcome = self.bridge.execute(decision)
            outcomes.append(outcome)
            pick_recovery = self._advance_pick_recovery(pick_recovery, outcome)
            perception_progress = self._advance_perception_progress(
                perception_progress, outcome
            )
            if self.memory is not None:
                self.memory.record_outcome(outcome, goal=user_goal)
            self.sink.emit("action_finished", {"turn": turn, **outcome.to_dict()})
            if pick_recovery is not None and pick_recovery.get("exhausted"):
                self.sink.emit("recovery_exhausted", {"turn": turn, **pick_recovery})
                return self._result(
                    "needs_user",
                    (
                        "Automatic pick recovery found no grasp-feasible geometry; "
                        "operator review is required instead of repeating search/observe."
                    ),
                    decisions,
                    outcomes,
                )
            if perception_progress is not None and perception_progress.get("exhausted"):
                self.sink.emit("perception_stalled", {"turn": turn, **perception_progress})
                return self._result(
                    "needs_user",
                    (
                        "Robot perception repeated the same target geometry without progress; "
                        "operator review is required instead of continuing search/observe."
                    ),
                    decisions,
                    outcomes,
                )
        return self._result(
            "max_turns",
            "The product loop reached its turn budget without end-to-end verification.",
            decisions,
            outcomes,
        )
