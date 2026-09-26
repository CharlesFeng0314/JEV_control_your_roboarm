"""The only product boundary allowed to invoke robot action capabilities."""

from __future__ import annotations

from typing import Any, Protocol

from .contracts import ActionOutcome, JevDecision, SceneSnapshot
from .manifest import ActionCatalog, ActionSpec


class RobotDriver(Protocol):
    """Implemented by Isaac and real-robot adapters with identical semantics."""

    def capabilities(self) -> dict[str, Any]: ...

    def snapshot(self) -> SceneSnapshot: ...

    def execute_action(self, action: ActionSpec, arguments: dict[str, Any]) -> ActionOutcome: ...


class ActionBridge:
    def __init__(self, catalog: ActionCatalog, driver: RobotDriver):
        self.catalog = catalog
        self.driver = driver

    def execute(self, decision: JevDecision) -> ActionOutcome:
        if decision.next_action in {"finish_task", "request_user"}:
            raise ValueError(f"{decision.next_action} is a control choice, not a robot action")
        action = self.catalog.require(decision.next_action)
        return self.driver.execute_action(action, dict(decision.arguments))
