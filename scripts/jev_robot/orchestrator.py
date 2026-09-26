"""Multi-turn JEV/action loop shared by the CLI and product GUI."""

from __future__ import annotations

from .action_bridge import ActionBridge
from .choices import DecisionEngine
from .contracts import ActionOutcome, JevDecision, SessionResult
from .events import EventSink, NullEventSink
from .manifest import ActionCatalog
from .prompting import build_given_that, render_given_that
from .scene_memory import SceneMemory


class ProductOrchestrator:
    def __init__(
        self,
        catalog: ActionCatalog,
        engine: DecisionEngine,
        bridge: ActionBridge,
        *,
        sink: EventSink | None = None,
        max_turns: int = 20,
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
        capabilities = self.bridge.driver.capabilities()
        self.sink.emit(
            "session_started",
            {
                "user_goal": user_goal,
                "continued_decisions": len(initial_decisions),
                "continued_outcomes": len(initial_outcomes),
            },
        )
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
            )
            self.sink.emit(
                "given_that_ready",
                {"turn": turn, "state": state, "rendered": render_given_that(state)},
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
                    self.memory.record_outcome(rejected)
                self.sink.emit("finish_rejected", rejected.to_dict())
                continue
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
            if self.memory is not None:
                self.memory.record_outcome(outcome)
            self.sink.emit("action_finished", {"turn": turn, **outcome.to_dict()})
        return self._result(
            "max_turns",
            "The product loop reached its turn budget without end-to-end verification.",
            decisions,
            outcomes,
        )
