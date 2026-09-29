"""Runtime wrist RGB-D object descriptions using geometry and local CLIP."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .rgbd_geometry import (
    cluster_table_objects,
    depth_to_mm,
    gripper_camera_local_matrix,
    mask_geometry,
    remove_gripper,
    world_maps,
)
from .semantic_fusion import CausalSemanticFusion

PROMPT_BANK = {
    "bowl": [
        "a photo of a bowl",
        "a round ceramic bowl on a table",
        "an empty household bowl viewed by a robot",
    ],
    "sugar_box": [
        "a photo of a box of sugar",
        "a rectangular sugar carton",
        "the YCB sugar box household object",
    ],
    "cracker_box": [
        "a photo of a cracker box",
        "a tall rectangular box of crackers",
        "the YCB cracker box household object",
    ],
    "soup_can": [
        "a photo of a tomato soup can",
        "a red and white cylindrical food can",
        "the YCB tomato soup can household object",
    ],
    "mug": [
        "a photo of a coffee mug with a handle",
        "a ceramic drinking mug",
        "a household cup with a side handle",
    ],
    "water_bottle": [
        "a photo of a plastic water bottle",
        "a tall reusable drinking bottle",
        "a water bottle with a narrow neck",
    ],
    "coffee_jar": [
        "a photo of a coffee jar with a lid",
        "a cylindrical jar of instant coffee",
        "a short household food jar",
    ],
    "spoon": [
        "a photo of a metal spoon",
        "a thin silver eating spoon",
        "a spoon lying flat on a table",
    ],
    "cube": [
        "a photo of a plain white cube",
        "a photo of a blue cube shaped block",
        "a small cube shaped block",
        "a geometric white cube on a table",
    ],
}
YOLO_PROMPTS = {
    "bowl": "bowl",
    "sugar_box": "sugar box",
    "cracker_box": "cracker box",
    "soup_can": "tomato soup can",
    "mug": "red mug",
    "water_bottle": "blue water bottle",
    "coffee_jar": "coffee jar",
    "spoon": "metal spoon",
    "cube": "white cube",
}
LABEL_CRITERIA = {
    label: "Robot wrist-vision vocabulary: " + "; ".join(prompts)
    for label, prompts in PROMPT_BANK.items()
}


class _SemanticWorkerClient:
    """Keep GPU models isolated from the control process's scientific DLLs."""

    def __init__(
        self,
        project_root: Path,
        clip_model: str,
        yolo_model: str,
        *,
        device: str,
        confidence: float,
        imgsz: int,
    ):
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.manipulation.perception.semantic_worker",
                "--clip-model",
                clip_model,
                "--yolo-model",
                yolo_model,
                "--device",
                device,
                "--confidence",
                str(confidence),
                "--imgsz",
                str(imgsz),
            ],
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creationflags,
        )
        ready = self._read_response()
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"Semantic worker did not become ready: {ready}")
        self.device = str(ready.get("device") or "unknown")
        self.gpu = str(ready.get("gpu") or "unknown")
        self.torch_version = str(ready.get("torch") or "unknown")

    def _read_response(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("CLIP worker stdout is unavailable")
        line = self._process.stdout.readline()
        if line:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
            raise TypeError("CLIP worker returned a non-object response")
        detail = ""
        if self._process.stderr is not None:
            detail = self._process.stderr.read().strip()
        raise RuntimeError(
            f"CLIP worker exited before returning a response (code={self._process.poll()}): "
            f"{detail[-1000:]}"
        )

    @staticmethod
    def _encode_png(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def analyze(
        self, rgb: np.ndarray, crops: list[Image.Image]
    ) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
        if self._process.stdin is None:
            raise RuntimeError("Semantic worker stdin is unavailable")
        self._process.stdin.write(
            json.dumps(
                {
                    "prompts": PROMPT_BANK,
                    "detector_prompts": YOLO_PROMPTS,
                    "rgb_png_base64": self._encode_png(Image.fromarray(rgb)),
                    "crops_png_base64": [self._encode_png(crop) for crop in crops],
                },
                ensure_ascii=True,
            )
            + "\n"
        )
        self._process.stdin.flush()
        response = self._read_response()
        if not response.get("ok"):
            raise RuntimeError(
                f"Semantic worker failed: {response.get('error_type')}: {response.get('error')}"
            )
        return (
            list(response["labels"]),
            np.asarray(response["probabilities"], dtype=np.float64),
            list(response.get("detections") or []),
        )

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.write('{"command":"close"}\n')
                self._process.stdin.flush()
            self._process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()


def _dominant_color(rgb: np.ndarray, mask: np.ndarray) -> str:
    """Return a coarse, prompt-friendly color from segmented RGB pixels."""
    pixels = rgb[mask]
    if not len(pixels):
        return "unknown"
    values = pixels.astype(np.float64) / 255.0
    maximum = values.max(axis=1)
    minimum = values.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    chromatic = values[(saturation > 0.28) & (maximum > 0.16)]
    if len(chromatic) >= max(12, len(values) // 12):
        red, green, blue = np.median(chromatic, axis=0)
        if red > 1.22 * max(green, blue):
            return "red"
        if green > 1.18 * max(red, blue):
            return "green"
        if blue > 1.18 * max(red, green):
            return "blue"
        if red > 0.75 and green > 0.55 and blue < 0.45:
            return "yellow"
    luminance = float(np.median(maximum))
    if luminance > 0.78:
        return "white"
    if luminance < 0.22:
        return "black"
    return "gray"


def _grasp_geometry(
    geometry: dict[str, Any],
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: dict[str, Any],
    pose: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Derive a bounded top-down grasp from wrist RGB-D geometry."""
    _camera, world = world_maps(depth_mm, intrinsics, pose)
    footprint = world[mask & (depth_mm > 0), :2]
    if len(footprint) < 8:
        raise ValueError("Object mask has too few points for grasp geometry")
    footprint_center, long_axis, long_extent, short_extent = _minimum_area_footprint(footprint)
    yaw = float(np.arctan2(long_axis[1], long_axis[0])) + np.pi
    orientation = [0.0, float(np.cos(yaw / 2.0)), float(np.sin(yaw / 2.0)), 0.0]

    bounds = np.asarray(geometry["bbox3d_world_m"], dtype=np.float64)
    center = 0.5 * (bounds[0] + bounds[1])
    center[:2] = footprint_center
    table_z = float(config["table_top_z_m"])
    if config["perception"].get("infer_upright_center_z_from_table", False):
        center[2] = 0.5 * (table_z + float(bounds[1, 2]))
    height = max(float(bounds[1, 2] - table_z), 0.015)
    grasp_position = center.copy()
    grasp_position[2] = (
        table_z + float(config["perception"]["upright_grasp_height_fraction"]) * height
    )
    # Keep the palm above tall objects; only the usable finger pads enter the grasp.
    maximum_insertion = float(config["perception"].get("maximum_top_grasp_depth_m", 0.025))
    grasp_position[2] = max(grasp_position[2], float(bounds[1, 2]) - maximum_insertion)
    clearance = 0.008
    return {
        "center_world_m": center.tolist(),
        "grasp_position_world_m": grasp_position.tolist(),
        "grasp_orientation_wxyz": orientation,
        "grasp_z_offset_from_centroid_m": float(grasp_position[2] - center[2]),
        "required_opening_m": short_extent + clearance,
        "estimated_contact_span_m": short_extent,
        "long_extent_m": long_extent,
        "feasible": short_extent + clearance <= 0.08,
        "approach": "top_down",
        "insertion_depth_m": float(bounds[1, 2] - grasp_position[2]),
    }


def _minimum_area_footprint(
    footprint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate box axes without PCA's bias toward the most visible face."""
    center, (width, height), angle_degrees = cv2.minAreaRect(
        np.asarray(footprint, dtype=np.float32)
    )
    angle = float(np.deg2rad(angle_degrees))
    if height > width:
        angle += np.pi / 2.0
        long_extent, short_extent = float(height), float(width)
    else:
        long_extent, short_extent = float(width), float(height)
    long_axis = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
    return np.asarray(center, dtype=np.float64), long_axis, long_extent, short_extent


def _masked_crop(rgb: np.ndarray, mask: np.ndarray, bbox: list[int]) -> Image.Image:
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    span = max(x2 - x1, y2 - y1)
    padding = max(6, int(round(span * 0.22)))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    crop = rgb[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    muted = np.full_like(crop, 116)
    muted[crop_mask] = crop[crop_mask]
    side = max(muted.shape[:2])
    canvas = np.full((side, side, 3), 116, dtype=np.uint8)
    y_offset = (side - muted.shape[0]) // 2
    x_offset = (side - muted.shape[1]) // 2
    canvas[y_offset : y_offset + muted.shape[0], x_offset : x_offset + muted.shape[1]] = muted
    return Image.fromarray(canvas)


def _box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _label_clusters(
    objects: list[dict[str, Any]], masks: list[np.ndarray], detections: list[dict[str, Any]]
) -> None:
    """Attach sparse YOLO evidence; geometry alone decides whether an object exists."""
    if not objects or not detections:
        return
    scores = np.zeros((len(objects), len(detections)), dtype=np.float64)
    for row, (obj, mask) in enumerate(zip(objects, masks, strict=True)):
        area = max(1, int(np.count_nonzero(mask)))
        for col, detection in enumerate(detections):
            x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
            x1, x2 = sorted((max(0, x1), min(mask.shape[1], x2)))
            y1, y2 = sorted((max(0, y1), min(mask.shape[0], y2)))
            inside = int(np.count_nonzero(mask[y1:y2, x1:x2])) if x2 > x1 and y2 > y1 else 0
            coverage = inside / area
            scores[row, col] = 0.78 * coverage + 0.22 * _box_iou(
                obj["bbox_xyxy"], detection["bbox_xyxy"]
            )
    rows, cols = linear_sum_assignment(-scores)
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if scores[row, col] < 0.20:
            continue
        objects[row]["detector_label"] = detections[col]["label"]
        objects[row]["detector_score"] = float(detections[col]["score"] * scores[row, col])


class WristRgbdSemanticPerception:
    """Detect all current-view objects without reading scene object metadata."""

    def __init__(self, config_path: Path, project_root: Path):
        self.project_root = project_root
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self._runtime: _SemanticWorkerClient | None = None
        self._fusion = CausalSemanticFusion(
            list(PROMPT_BANK), yolo_weight=float(self.config["perception"].get("yolo_weight", 2.0))
        )
        if self.config["perception"].get("eager_load_semantics", False):
            # Load the model runtime before the first simulator RPC. On Windows, DLLs loaded by
            # the RPC/scientific stack can otherwise make a later torch shm.dll load unreliable.
            self._semantic_runtime()

    @property
    def label_criteria(self) -> dict[str, str]:
        return dict(LABEL_CRITERIA)

    @property
    def minimum_target_score(self) -> float:
        return float(self.config["perception"]["minimum_target_semantic_score"])

    def _semantic_runtime(self) -> _SemanticWorkerClient:
        if self._runtime is not None:
            return self._runtime
        perception = self.config["perception"]
        clip_ref = str(perception["clip_model"])
        yolo_ref = str(perception["yolo_model"])
        clip_candidate = (self.project_root / clip_ref).resolve()
        yolo_candidate = (self.project_root / yolo_ref).resolve()
        self._runtime = _SemanticWorkerClient(
            self.project_root,
            str(clip_candidate) if clip_candidate.is_file() else clip_ref,
            str(yolo_candidate) if yolo_candidate.is_file() else yolo_ref,
            device=str(perception.get("device", "cuda:0")),
            confidence=float(perception.get("yolo_confidence", 0.02)),
            imgsz=int(perception.get("yolo_imgsz", 640)),
        )
        return self._runtime

    def fuse_tracked_observation(
        self, object_id: str, observation: dict[str, Any]
    ) -> dict[str, Any]:
        return self._fusion.update(object_id, observation)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def detect(self, packet: dict[str, Any]) -> list[dict[str, Any]]:
        rgb = np.asarray(Image.open(packet["rgb_path"]).convert("RGB"))
        depth = np.load(packet["depth_path"]).astype(np.float32)
        pose = {
            **packet["pose"],
            "orientation_wxyz": packet["orientation_wxyz"],
        }
        camera = self.config["wrist_camera"]
        perception = self.config["perception"]
        depth_mm = depth_to_mm(
            depth,
            float(camera["depth_near_m"]),
            float(camera["depth_far_m"]),
        )
        local = gripper_camera_local_matrix(
            camera["mount_translation_m"],
            camera["look_at_in_parent_m"],
        )
        depth_mm = remove_gripper(
            depth_mm,
            packet["intrinsics"],
            pose,
            packet["orientation_wxyz"],
            local,
            camera["gripper_exclusion_box_min_m"],
            camera["gripper_exclusion_box_max_m"],
        )
        masks = cluster_table_objects(
            depth_mm,
            packet["intrinsics"],
            pose,
            table_z_m=float(self.config["table_top_z_m"]),
            pixel_stride=int(perception["pixel_stride"]),
            eps_m=float(perception["dbscan_eps_m"]),
            min_samples=int(perception["dbscan_min_samples"]),
        )
        candidates = []
        candidate_masks = []
        for mask in masks:
            geometry = mask_geometry(mask, depth_mm, packet["intrinsics"], pose)
            if geometry is None:
                continue
            if geometry["area_px"] < int(perception["minimum_mask_area_px"]):
                continue
            candidates.append(geometry)
            candidate_masks.append(mask)
        if not candidates:
            return []

        crops = [
            _masked_crop(rgb, mask, candidate["bbox_xyxy"])
            for candidate, mask in zip(candidates, candidate_masks, strict=True)
        ]
        labels, probabilities, detections = self._semantic_runtime().analyze(rgb, crops)
        _label_clusters(candidates, candidate_masks, detections)

        observations = []
        for candidate, mask, probability in zip(
            candidates, candidate_masks, probabilities, strict=True
        ):
            scores = probability.astype(np.float64)
            scores /= max(float(scores.sum()), 1e-9)
            best = int(np.argmax(scores))
            grasp = _grasp_geometry(
                candidate,
                mask,
                depth_mm,
                packet["intrinsics"],
                pose,
                self.config,
            )
            color = _dominant_color(rgb, mask)
            x1, y1, x2, y2 = candidate["bbox_xyxy"]
            partial_view = x1 <= 2 or y1 <= 2 or x2 >= rgb.shape[1] - 2 or y2 >= rgb.shape[0] - 2
            observation = {
                "label": labels[best],
                "description": f"{color} {labels[best]}",
                "attributes": {"color": color},
                "confidence": float(scores[best]),
                "pose": {
                    "frame": "world",
                    "position_m": grasp["center_world_m"],
                },
                "source": "wrist_rgbd_world_cluster_semantic_view",
                "mask_area_px": int(candidate["area_px"]),
                "partial_view": bool(partial_view),
                "semantic_scores": {
                    label: float(scores[index]) for index, label in enumerate(labels)
                },
                "view_semantic_scores": {
                    label: float(scores[index]) for index, label in enumerate(labels)
                },
                "bbox_xyxy": candidate["bbox_xyxy"],
                "bbox3d_world_m": candidate["bbox3d_world_m"],
                "grasp": grasp,
            }
            if "detector_label" in candidate:
                observation["detector_label"] = candidate["detector_label"]
                observation["detector_score"] = candidate["detector_score"]
            observations.append(observation)
        return observations
