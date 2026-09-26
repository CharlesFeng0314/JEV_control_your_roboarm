"""Load the single action catalog shared by JEV and robot action bridges."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "manipulation" / "action_manifest.json"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    implementation: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


class ActionCatalog:
    def __init__(self, actions: tuple[ActionSpec, ...]):
        if not actions:
            raise ValueError("Action catalog must not be empty")
        names = [action.name for action in actions]
        if len(names) != len(set(names)):
            raise ValueError("Action catalog contains duplicate names")
        self.actions = actions
        self._by_name = {action.name: action for action in actions}

    @classmethod
    def load(cls, path: Path = DEFAULT_MANIFEST) -> ActionCatalog:
        payload = json.loads(path.read_text(encoding="utf-8"))
        actions = tuple(
            ActionSpec(
                name=item["name"],
                description=item.get("description") or item["name"].replace("_", " "),
                implementation=item["implementation"],
                inputs=dict(item.get("inputs") or {}),
                outputs=dict(item.get("outputs") or {}),
            )
            for item in payload["actions"]
        )
        return cls(actions)

    def only(self, names: Iterable[str]) -> ActionCatalog:
        allowed = {str(name) for name in names}
        unknown = allowed.difference(self._by_name)
        if unknown:
            raise ValueError(f"Robot driver advertises unknown actions: {sorted(unknown)}")
        return ActionCatalog(tuple(action for action in self.actions if action.name in allowed))

    def require(self, name: str) -> ActionSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ValueError(f"JEV selected unknown action {name!r}") from exc

    def choice_criteria(self) -> dict[str, str]:
        return {action.name: action.description for action in self.actions}
