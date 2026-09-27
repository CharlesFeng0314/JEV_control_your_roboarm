"""Persistent scene belief built only from robot-mounted observations and action results."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .contracts import ActionOutcome, SceneSnapshot, utc_now
from .prompting import summarize_action_record, summarize_scene_object

FORBIDDEN_SOURCES = {"spectator_camera", "simulator_ground_truth", "usd_stage"}
_SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


def _object_key(item: dict[str, Any]) -> str:
    explicit = item.get("object_id") or item.get("id")
    if explicit:
        return str(explicit)
    stable = {
        "label": item.get("label") or item.get("description"),
        "pose": item.get("pose") or item.get("position"),
    }
    digest = hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"perceived_{digest}"


class SceneMemory:
    """Durable controller memory; JEV receives a prompt view on every call."""

    def __init__(self, path: Path, scene_id: str, payload: dict[str, Any] | None = None):
        self.path = path
        self.scene_id = scene_id
        self.payload = payload or {
            "schema_version": 1,
            "scene_id": scene_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "revision": 0,
            "known_objects": {},
            "latest_robot_facts": {},
            "unknown_regions": [],
            "recent_actions": [],
        }

    @classmethod
    def open(cls, root: Path, scene_id: str) -> SceneMemory:
        safe_id = _SAFE_ID.sub("_", scene_id.strip()).strip("._")
        if not safe_id:
            raise ValueError("scene_id must contain a usable character")
        path = root / f"{safe_id}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("scene_id") != scene_id:
                raise ValueError("Scene memory id does not match the requested scene")
            memory = cls(path, scene_id, payload)
            if memory._compact_stored_actions():
                memory.save()
            return memory
        root.mkdir(parents=True, exist_ok=True)
        memory = cls(path, scene_id)
        memory.save()
        return memory

    def save(self) -> None:
        self.payload["updated_at"] = utc_now()
        self.path.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def observe(self, snapshot: SceneSnapshot) -> dict[str, Any]:
        if snapshot.source in FORBIDDEN_SOURCES:
            raise ValueError(f"Forbidden scene-memory source {snapshot.source!r}")
        if snapshot.facts.get("validation_only") is True:
            raise ValueError("Validation-only simulator facts cannot enter scene memory")
        now = snapshot.captured_at
        next_revision = int(self.payload["revision"]) + 1
        visible_ids: set[str] = set()
        objects = self.payload["known_objects"]
        for item in snapshot.visible_objects:
            candidate = dict(item)
            key = _object_key(candidate)
            visible_ids.add(key)
            previous = objects.get(key, {})
            observations = int(previous.get("observation_count", 0)) + 1
            confidence = candidate.get("confidence")
            observed_label = candidate.get("label") or candidate.get("description") or "unknown"
            label_evidence = dict(previous.get("label_evidence", {}))
            label_evidence[observed_label] = int(label_evidence.get(observed_label, 0)) + 1
            fused_label = max(label_evidence, key=label_evidence.get)
            pose_history = list(previous.get("pose_history", []))
            observed_pose = candidate.get("pose") or candidate.get("position")
            if observed_pose is not None:
                pose_history.append(
                    {
                        "captured_at": now,
                        "pose": observed_pose,
                        "confidence": confidence,
                    }
                )
                pose_history = pose_history[-20:]
            record = {
                **previous,
                "object_id": key,
                "label": fused_label,
                "label_evidence": label_evidence,
                "latest_pose": observed_pose,
                "pose_history": pose_history,
                "latest_observation": summarize_scene_object(candidate),
                "first_seen_at": previous.get("first_seen_at") or now,
                "last_seen_at": now,
                "last_seen_revision": next_revision,
                "observation_count": observations,
                "currently_visible": True,
                "stale": False,
            }
            if confidence is not None:
                record["latest_confidence"] = float(confidence)
                record["best_confidence"] = max(
                    float(previous.get("best_confidence", 0.0)),
                    float(confidence),
                )
            objects[key] = record
        for key, record in objects.items():
            if key not in visible_ids:
                record["currently_visible"] = False
                last_seen_revision = int(record.get("last_seen_revision", next_revision))
                record["stale"] = next_revision - last_seen_revision >= 3
        self.payload["latest_robot_facts"] = dict(snapshot.facts)
        self.payload["unknown_regions"] = list(snapshot.unknown_regions)
        self.payload["revision"] = next_revision
        self.save()
        return self.prompt_view()

    def begin_goal(self, user_goal: str) -> None:
        """Start a robot goal. Earlier goals stay archived and leave the next prompt."""
        goal = user_goal.strip()
        if not goal:
            raise ValueError("User goal must not be empty")
        if self.payload.get("active_goal") == goal:
            return
        self.payload["active_goal"] = goal
        self.payload["goal_started_at"] = utc_now()
        self.payload["revision"] = int(self.payload["revision"]) + 1
        self.save()

    def record_outcome(self, outcome: ActionOutcome, *, goal: str | None = None) -> None:
        self._compact_stored_actions()
        record = summarize_action_record({"time": utc_now(), **outcome.to_dict()})
        active_goal = (goal or self.payload.get("active_goal") or "").strip()
        if active_goal:
            record["goal"] = active_goal
        actions = self.payload["recent_actions"]
        actions.append(record)
        del actions[:-50]
        self.payload["revision"] = int(self.payload["revision"]) + 1
        self.save()

    def _compact_stored_actions(self) -> bool:
        actions = list(self.payload.get("recent_actions") or [])
        compact = []
        for item in actions[-50:]:
            record = summarize_action_record(item)
            goal = str(item.get("goal") or "").strip()
            if goal:
                record["goal"] = goal
            compact.append(record)
        if compact == actions:
            return False
        self.payload["recent_actions"] = compact
        return True

    def prompt_view(self) -> dict[str, Any]:
        known = []
        for key in sorted(self.payload["known_objects"]):
            record = self.payload["known_objects"][key]
            known.append(
                {
                    "object_id": record["object_id"],
                    "label": record["label"],
                    "description": (record.get("latest_observation") or {}).get("description"),
                    "attributes": (record.get("latest_observation") or {}).get("attributes", {}),
                    "bbox3d_world_m": (record.get("latest_observation") or {}).get(
                        "bbox3d_world_m"
                    ),
                    "label_evidence": record.get("label_evidence", {}),
                    "latest_pose": record.get("latest_pose"),
                    "latest_confidence": record.get("latest_confidence"),
                    "best_confidence": record.get("best_confidence"),
                    "observation_count": record["observation_count"],
                    "currently_visible": record["currently_visible"],
                    "stale": record["stale"],
                    "last_seen_at": record["last_seen_at"],
                    "pose_observation_count": len(record.get("pose_history", [])),
                }
            )
        return {
            "scene_id": self.scene_id,
            "revision": self.payload["revision"],
            "memory_semantics": (
                "Controller-owned persistent belief resent to stateless JEV on every decision."
            ),
            "known_objects": known,
            "latest_robot_facts": self.payload["latest_robot_facts"],
            "unknown_regions": self.payload["unknown_regions"],
        }
