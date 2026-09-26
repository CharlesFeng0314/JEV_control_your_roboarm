"""Product driver for an already-running Isaac scene host.

The driver uses only the arm interface exposed by the persistent simulation
process. It never calls validation-only object-state or simulator-ground-truth
commands.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from scripts.manipulation.perception.wrist_semantics import WristRgbdSemanticPerception

from ..contracts import ActionOutcome, SceneSnapshot
from ..manifest import ActionSpec

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 47631
RpcTransport = Callable[[dict[str, Any], float], dict[str, Any]]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_CONFIG = (
    PROJECT_ROOT
    / "scripts"
    / "manipulation"
    / "config"
    / "tabletop_household_franka_wrist_rgbd_v1.json"
)
_BUNDLED_CONFIG = (
    PROJECT_ROOT / "scripts" / "manipulation" / "perception" / "wrist_rgbd.json"
)
DEFAULT_CONFIG = _LOCAL_CONFIG if _LOCAL_CONFIG.is_file() else _BUNDLED_CONFIG
ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
SEARCH_ARM_POSES = (
    (0.0, -0.45, 0.0, -2.15, 0.0, 2.35, 0.8),
    (-0.65, -0.45, 0.0, -2.15, 0.0, 2.35, 0.8),
    (0.65, -0.45, 0.0, -2.15, 0.0, 2.35, 0.8),
    (-1.1, -0.55, 0.0, -2.05, 0.0, 2.2, 0.8),
    (1.1, -0.55, 0.0, -2.05, 0.0, 2.2, 0.8),
    (0.0, -0.8, 0.0, -2.35, 0.0, 2.55, 0.8),
    (-0.4, -0.85, 0.0, -2.2, 0.0, 2.45, 0.8),
    (0.4, -0.85, 0.0, -2.2, 0.0, 2.45, 0.8),
)
SEARCH_SCOPE_VIEW_COUNT = {
    "current_view": 0,
    "narrow": 4,
    "wide": len(SEARCH_ARM_POSES),
}
CACHE_MATCH_RADIUS_M = 0.12
LIFTED_PHASES = {"APPROACH_PLACE", "DESCEND_PLACE", "RELEASE", "RETREAT", "DONE"}
MAX_MANIPULATION_STEPS = 3600
PLACEMENT_VERIFY_TOLERANCE_M = 0.06


class PerceptionProvider(Protocol):
    @property
    def label_criteria(self) -> dict[str, str]: ...

    @property
    def minimum_target_score(self) -> float: ...

    def detect(self, packet: dict[str, Any]) -> list[dict[str, Any]]: ...


def _socket_rpc(
    host: str,
    port: int,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=5.0) as connection:
        connection.settimeout(timeout)
        connection.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            chunk = connection.recv(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    line = b"".join(chunks).split(b"\n", 1)[0]
    if not line:
        raise ConnectionError("Isaac scene host returned no response")
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Isaac scene host response must be a JSON object")
    return value


class IsaacRpcDriver:
    """Connect JEV product actions to the persistent Isaac arm interface."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        transport: RpcTransport | None = None,
        perception: PerceptionProvider | None = None,
    ):
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self.host = host
        self.port = int(port)
        self._transport = transport
        self._perception = perception or WristRgbdSemanticPerception(DEFAULT_CONFIG, PROJECT_ROOT)
        self._sequence = 0
        self._sensor_sequence = 0
        self._recording = False
        self._run_dir: Path | None = None
        self._visible_objects: tuple[dict[str, Any], ...] = ()
        self._object_tracks: dict[str, dict[str, Any]] = {}
        self._next_object_id = 1
        self._held_target: dict[str, Any] | None = None
        self._active_destination: dict[str, Any] | None = None
        self._expected_place_position: tuple[float, float, float] | None = None

    def _call(self, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        if self._transport is None:
            response = _socket_rpc(self.host, self.port, payload, timeout)
        else:
            response = self._transport(dict(payload), timeout)
        if not response.get("ok"):
            message = response.get("error") or f"{payload.get('cmd')} failed"
            raise RuntimeError(str(message))
        return response

    def capabilities(self) -> dict[str, Any]:
        return {
            "driver": "isaac_rpc",
            "connection": {"host": self.host, "port": self.port},
            "persistent_scene": True,
            "perception": ["wrist_rgbd"],
            "perception_labels": self._perception.label_criteria,
            "motion_planner": "scene_host_owned",
            "supported_actions": [
                "search_object",
                "observe_object",
                "adjust_gripper",
                "pick_object",
                "place_object",
                "verify_transfer",
            ],
            "unsupported_until_rpc_extension": [
                "adjust_end_effector",
                "execute_tool_motion",
            ],
            "ground_truth_feedback": False,
        }

    def _archive_rgbd(self, packet: dict[str, Any]) -> dict[str, Any]:
        self._sensor_sequence += 1
        if self._run_dir is None:
            return packet
        artifact_dir = self._run_dir / "artifacts" / "wrist_rgbd"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        archived = dict(packet)
        for key, suffix in (("rgb_path", ".png"), ("depth_path", ".npy")):
            source = Path(str(packet.get(key) or ""))
            if not source.is_file():
                continue
            destination = artifact_dir / f"{self._sensor_sequence:06d}_{key}{suffix}"
            shutil.copy2(source, destination)
            archived[key] = str(destination)
        return archived

    def _capture_rgbd(self) -> dict[str, Any]:
        return self._archive_rgbd(self._call({"cmd": "rgbd"}, timeout=180.0))

    def snapshot(self) -> SceneSnapshot:
        state = self._call({"cmd": "state"})
        rgbd = self._capture_rgbd()
        visible_objects = self._track_observations(self._perception.detect(rgbd))
        self._sequence += 1
        facts = {
            "joint_names": list(state.get("joint_names") or []),
            "joint_positions": list(state.get("joint_positions") or []),
            "hand_position_m": list(state.get("hand_position_m") or []),
            "hand_orientation_wxyz": list(state.get("hand_orientation_wxyz") or []),
            "wrist_rgbd": {
                "rgb_path": rgbd.get("rgb_path"),
                "depth_path": rgbd.get("depth_path"),
                "intrinsics": rgbd.get("intrinsics"),
                "camera_pose": rgbd.get("pose"),
                "camera_orientation_wxyz": rgbd.get("orientation_wxyz"),
                "shape": rgbd.get("shape"),
            },
        }
        return SceneSnapshot(
            sequence=self._sequence,
            source="robot_mounted_sensors",
            facts=facts,
            visible_objects=visible_objects,
            unknown_regions=(() if visible_objects else ("workspace_not_interpreted",)),
        )

    @staticmethod
    def _position(observation: dict[str, Any]) -> tuple[float, float, float] | None:
        pose = observation.get("pose") or {}
        raw = pose.get("position_m") if isinstance(pose, dict) else None
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None
        return tuple(float(value) for value in raw)

    def _track_observations(self, observations: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        tracked = []
        for item in observations:
            observation = dict(item)
            position = self._position(observation)
            best_id = None
            best_distance = CACHE_MATCH_RADIUS_M
            if position is not None:
                for object_id, previous in self._object_tracks.items():
                    previous_position = self._position(previous)
                    if previous_position is None:
                        continue
                    distance = (
                        sum(
                            (current - old) ** 2
                            for current, old in zip(position, previous_position, strict=True)
                        )
                        ** 0.5
                    )
                    if distance <= best_distance:
                        best_id = object_id
                        best_distance = distance
            if best_id is None:
                best_id = f"wrist_object_{self._next_object_id:03d}"
                self._next_object_id += 1
            observation["object_id"] = best_id
            observation["sensor_sequence"] = self._sensor_sequence
            self._object_tracks[best_id] = observation
            tracked.append(observation)
        self._visible_objects = tuple(tracked)
        return self._visible_objects

    def _perceive_current_view(self) -> tuple[dict[str, Any], ...]:
        packet = self._capture_rgbd()
        return self._track_observations(self._perception.detect(packet))

    @staticmethod
    def _target_candidate(
        observations: tuple[dict[str, Any], ...], target_label: str
    ) -> tuple[dict[str, Any] | None, float]:
        best = None
        best_score = 0.0
        for observation in observations:
            scores = observation.get("semantic_scores") or {}
            score = float(scores.get(target_label, 0.0))
            if score > best_score:
                best = observation
                best_score = score
        return best, best_score

    def _search_object(
        self,
        arguments: dict[str, Any],
        *,
        action_name: str = "search_object",
    ) -> ActionOutcome:
        target_label = str(arguments.get("target_label") or "")
        search_scope = str(arguments.get("search_scope") or "current_view")
        if target_label not in self._perception.label_criteria:
            raise ValueError(f"Unsupported wrist-vision target label {target_label!r}")
        if search_scope not in SEARCH_SCOPE_VIEW_COUNT:
            raise ValueError(f"Unsupported search scope {search_scope!r}")

        observations = self._perceive_current_view()
        candidate, score = self._target_candidate(observations, target_label)
        best_candidate, best_score = candidate, score
        visited_views = 1
        threshold = self._perception.minimum_target_score
        for pose in SEARCH_ARM_POSES[: SEARCH_SCOPE_VIEW_COUNT[search_scope]]:
            if candidate is not None and score >= threshold:
                break
            targets = {name: float(value) for name, value in zip(ARM_JOINTS, pose, strict=True)}
            self._call(
                {
                    "cmd": "set_joints",
                    "targets": targets,
                    "steps": 220,
                    "tolerance": 0.06,
                },
                timeout=180.0,
            )
            observations = self._perceive_current_view()
            candidate, score = self._target_candidate(observations, target_label)
            if score > best_score:
                best_candidate, best_score = candidate, score
            visited_views += 1

        accepted = candidate is not None and score >= threshold
        data = {
            "target_label": target_label,
            "search_scope": search_scope,
            "visited_views": visited_views,
            "minimum_target_score": threshold,
            "best_target_score": best_score,
            "observation": candidate if accepted else best_candidate,
            "visible_objects": list(observations),
        }
        return ActionOutcome(
            action=action_name,
            success=accepted,
            message=(
                f"Wrist RGB-D found {target_label!r} with score {score:.3f}"
                if accepted
                else (f"Wrist RGB-D did not accept {target_label!r}; best score {best_score:.3f}")
            ),
            data=data,
        )

    @staticmethod
    def _finger_width(state: dict[str, Any]) -> float:
        names = [str(value) for value in state.get("joint_names") or []]
        positions = [float(value) for value in state.get("joint_positions") or []]
        values = []
        for name in ("panda_finger_joint1", "panda_finger_joint2"):
            try:
                values.append(positions[names.index(name)])
            except (ValueError, IndexError) as exc:
                raise RuntimeError(f"Isaac state is missing {name}") from exc
        return sum(values) / len(values)

    def _adjust_gripper(self, arguments: dict[str, Any]) -> ActionOutcome:
        command = str(arguments.get("command") or "")
        step_m = float(arguments.get("step_m") or 0.0)
        if command not in {"hold", "open", "close", "widen", "narrow"}:
            raise ValueError(f"Unsupported gripper command {command!r}")
        if not 0.0 <= step_m <= 0.005:
            raise ValueError("Gripper step must be within 0-5 mm per finger")
        if (command == "hold") != (step_m == 0.0):
            raise ValueError("Only hold may use a zero step; moving commands require a step")

        state = self._call({"cmd": "state"})
        current_m = self._finger_width(state)
        direction = 0.0 if command == "hold" else 1.0
        if command in {"close", "narrow"}:
            direction = -1.0
        target_m = min(0.04, max(0.0, current_m + direction * step_m))
        if command != "hold" and abs(target_m - current_m) < 1e-6:
            return ActionOutcome(
                action="adjust_gripper",
                success=False,
                message="Gripper is already at the requested mechanical limit",
                data={
                    "command": command,
                    "step_m": step_m,
                    "current_per_finger_m": current_m,
                    "target_per_finger_m": target_m,
                    "stop_condition": arguments.get("stop_condition"),
                },
            )
        if command != "hold":
            self._call(
                {"cmd": "set_gripper", "width": target_m, "steps": 40},
                timeout=180.0,
            )
        return ActionOutcome(
            action="adjust_gripper",
            success=True,
            message=(
                "Gripper held at its current aperture"
                if command == "hold"
                else f"Gripper moved from {current_m:.4f} m to {target_m:.4f} m per finger"
            ),
            data={
                "command": command,
                "step_m": step_m,
                "current_per_finger_m": current_m,
                "target_per_finger_m": target_m,
                "stop_condition": arguments.get("stop_condition"),
            },
        )

    def _require_track(
        self,
        object_ref: object,
        *,
        role: str,
        fresh: bool,
    ) -> dict[str, Any]:
        reference = str(object_ref or "")
        observation = self._object_tracks.get(reference)
        if observation is None:
            raise ValueError(f"{role} reference {reference!r} is not in wrist scene memory")
        if fresh and int(observation.get("sensor_sequence", -1)) != self._sensor_sequence:
            raise ValueError(
                f"{role} reference {reference!r} is not from the latest wrist RGB-D frame"
            )
        return observation

    @staticmethod
    def _bounds(observation: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        raw = observation.get("bbox3d_world_m")
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in raw)
        ):
            raise ValueError("Wrist observation has no 3D bounds")
        return (
            tuple(float(value) for value in raw[0]),
            tuple(float(value) for value in raw[1]),
        )

    @classmethod
    def _stack_place_position(
        cls,
        target: dict[str, Any],
        destination: dict[str, Any],
    ) -> tuple[float, float, float]:
        target_min, target_max = cls._bounds(target)
        destination_min, destination_max = cls._bounds(destination)
        del destination_min
        target_height = max(target_max[2] - target_min[2], 0.015)
        center = cls._position(destination)
        if center is None:
            raise ValueError("Destination has no wrist-observed position")
        return (
            center[0],
            center[1],
            destination_max[2] + target_height / 2.0 + 0.005,
        )

    @staticmethod
    def _failed_motion(action: str, state: dict[str, Any]) -> ActionOutcome:
        return ActionOutcome(
            action=action,
            success=False,
            message=str(state.get("failure_reason") or "Robot motion did not complete"),
            data={"robot_state": state},
        )

    def _pick_object(self, arguments: dict[str, Any]) -> ActionOutcome:
        if self._held_target is not None:
            return ActionOutcome(
                action="pick_object",
                success=False,
                message="The gripper already has an active held-object belief",
                data={"held_target_ref": self._held_target.get("object_id")},
            )
        target = self._require_track(arguments.get("target_ref"), role="Target", fresh=True)
        grasp = target.get("grasp") or {}
        if not grasp.get("feasible", False):
            return ActionOutcome(
                action="pick_object",
                success=False,
                message="Wrist RGB-D geometry says the target exceeds the gripper opening",
                data={"target_ref": target["object_id"], "grasp": grasp},
            )
        self._call(
            {
                "cmd": "set_pick",
                "position": list(grasp["grasp_position_world_m"]),
                "orientation_wxyz": list(grasp["grasp_orientation_wxyz"]),
                "z_offset": float(grasp["grasp_z_offset_from_centroid_m"]),
            }
        )
        self._call({"cmd": "reset_manipulation"}, timeout=180.0)
        self._call(
            {
                "cmd": "set_pregrasp_verified",
                "verified": True,
                "evidence": {
                    "source": target.get("source"),
                    "sensor_sequence": target.get("sensor_sequence"),
                    "target_ref": target.get("object_id"),
                },
            }
        )
        last_state: dict[str, Any] = {}
        for _ in range(MAX_MANIPULATION_STEPS):
            last_state = self._call({"cmd": "manipulation_step"}, timeout=60.0)
            if last_state.get("failed"):
                return self._failed_motion("pick_object", last_state)
            if str(last_state.get("phase")) in LIFTED_PHASES:
                self._held_target = dict(target)
                return ActionOutcome(
                    action="pick_object",
                    success=True,
                    message="Target grasped and lifted using wrist-observed geometry",
                    data={
                        "target_ref": target["object_id"],
                        "phase": last_state.get("phase"),
                        "finger_positions_m": last_state.get("finger_positions_m"),
                        "grasp": grasp,
                    },
                )
            if not last_state.get("alive", True):
                break
        return self._failed_motion("pick_object", last_state)

    def _place_object(self, arguments: dict[str, Any]) -> ActionOutcome:
        if self._held_target is None:
            return ActionOutcome(
                action="place_object",
                success=False,
                message="No held-object belief is available; pick an observed target first",
            )
        destination = self._require_track(
            arguments.get("destination_ref"),
            role="Destination",
            fresh=True,
        )
        place_position = self._stack_place_position(self._held_target, destination)
        self._call({"cmd": "set_place", "position": list(place_position)})
        last_state: dict[str, Any] = {}
        for _ in range(MAX_MANIPULATION_STEPS):
            last_state = self._call({"cmd": "manipulation_step"}, timeout=60.0)
            if last_state.get("failed"):
                return self._failed_motion("place_object", last_state)
            if last_state.get("done"):
                self._active_destination = dict(destination)
                self._expected_place_position = place_position
                return ActionOutcome(
                    action="place_object",
                    success=True,
                    message="Held object released at the wrist-observed destination surface",
                    data={
                        "target_ref": self._held_target["object_id"],
                        "destination_ref": destination["object_id"],
                        "expected_place_position_m": list(place_position),
                        "phase": last_state.get("phase"),
                    },
                )
            if not last_state.get("alive", True):
                break
        return self._failed_motion("place_object", last_state)

    def _verify_transfer(self, arguments: dict[str, Any]) -> ActionOutcome:
        if self._held_target is None or self._expected_place_position is None:
            return ActionOutcome(
                action="verify_transfer",
                success=False,
                message="No completed pick/place action is available to verify",
            )
        requested_destination = str(arguments.get("destination_ref") or "")
        if (
            self._active_destination is not None
            and requested_destination
            and requested_destination != self._active_destination.get("object_id")
        ):
            raise ValueError("Verification destination differs from the executed place action")

        observations = self._perceive_current_view()
        target_label = str(self._held_target.get("label") or "")
        expected = self._expected_place_position
        best: dict[str, Any] | None = None
        best_score = -1.0
        best_error = float("inf")
        for observation in observations:
            scores = observation.get("semantic_scores") or {}
            score = float(scores.get(target_label, 0.0))
            position = self._position(observation)
            if position is None:
                continue
            error = sum(
                (current - goal) ** 2
                for current, goal in zip(position, expected, strict=True)
            ) ** 0.5
            ranking = score - min(error, 1.0)
            if ranking > best_score - min(best_error, 1.0):
                best, best_score, best_error = observation, score, error

        semantic_ok = best is not None and best_score >= self._perception.minimum_target_score
        spatial_ok = best is not None and best_error <= PLACEMENT_VERIFY_TOLERANCE_M
        success = semantic_ok and spatial_ok
        if success:
            self._held_target = None
        return ActionOutcome(
            action="verify_transfer",
            success=success,
            message=(
                "Wrist RGB-D verified the requested target at the destination"
                if success
                else "Wrist RGB-D did not verify the target/destination spatial relation"
            ),
            data={
                "target_label": target_label,
                "destination_ref": (
                    None
                    if self._active_destination is None
                    else self._active_destination.get("object_id")
                ),
                "expected_place_position_m": list(expected),
                "observed_target": best,
                "semantic_score": best_score,
                "position_error_m": best_error,
                "tolerance_m": PLACEMENT_VERIFY_TOLERANCE_M,
                "visible_objects": list(observations),
                "source": "robot_mounted_wrist_rgbd",
            },
        )

    def execute_action(
        self,
        action: ActionSpec,
        arguments: dict[str, Any],
    ) -> ActionOutcome:
        if action.name == "adjust_gripper":
            return self._adjust_gripper(arguments)
        if action.name == "search_object":
            return self._search_object(arguments)
        if action.name == "observe_object":
            return self._search_object(arguments, action_name="observe_object")
        if action.name == "pick_object":
            return self._pick_object(arguments)
        if action.name == "place_object":
            return self._place_object(arguments)
        if action.name == "verify_transfer":
            return self._verify_transfer(arguments)
        return ActionOutcome(
            action=action.name,
            success=False,
            message=f"Isaac RPC driver does not yet expose {action.name!r}",
            data={"supported_actions": self.capabilities()["supported_actions"]},
        )

    def start_session(self, run_dir: Path) -> None:
        if self._recording:
            raise RuntimeError("Isaac recording is already active")
        self._call({"cmd": "begin_record", "run_dir": str(run_dir)})
        self._recording = True
        self._run_dir = run_dir

    def end_session(self) -> None:
        if not self._recording:
            return
        try:
            self._call({"cmd": "end_record"})
        finally:
            self._recording = False
            self._run_dir = None


def create_driver() -> IsaacRpcDriver:
    host = os.environ.get("JEV_ISAAC_RPC_HOST", DEFAULT_HOST)
    port = int(os.environ.get("JEV_ISAAC_RPC_PORT", str(DEFAULT_PORT)))
    return IsaacRpcDriver(host, port)
