"""Runtime wrist RGB-D object descriptions using geometry and local CLIP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .rgbd_geometry import (
    cluster_table_objects,
    depth_to_mm,
    gripper_camera_local_matrix,
    mask_geometry,
    remove_gripper,
    world_maps,
)

PROMPT_BANK = {
    "bowl": ["a bowl", "a round ceramic bowl on a table"],
    "sugar_box": ["a box of sugar", "a rectangular sugar carton"],
    "cracker_box": ["a cracker box", "a tall rectangular box of crackers"],
    "soup_can": ["a tomato soup can", "a red and white cylindrical food can"],
    "mug": ["a coffee mug with a handle", "a ceramic drinking mug"],
    "water_bottle": ["a plastic water bottle", "a tall reusable drinking bottle"],
    "coffee_jar": ["a coffee jar with a lid", "a short household food jar"],
    "spoon": ["a metal spoon", "a thin silver eating spoon"],
    "cube": ["a plain cube", "a small cube shaped block"],
}
LABEL_CRITERIA = {
    label: "Robot wrist-vision vocabulary: " + "; ".join(prompts)
    for label, prompts in PROMPT_BANK.items()
}


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
    centered = footprint - np.median(footprint, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    short_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float64)
    long_extent = float(np.ptp(centered @ long_axis))
    short_extent = float(np.ptp(centered @ short_axis))
    if short_extent > long_extent:
        long_extent, short_extent = short_extent, long_extent
        long_axis = short_axis
    yaw = float(np.arctan2(long_axis[1], long_axis[0])) + np.pi
    orientation = [0.0, float(np.cos(yaw / 2.0)), float(np.sin(yaw / 2.0)), 0.0]

    bounds = np.asarray(geometry["bbox3d_world_m"], dtype=np.float64)
    center = 0.5 * (bounds[0] + bounds[1])
    eye = np.asarray(pose["eye_world_m"], dtype=np.float64)
    forward = np.asarray(pose["target_world_m"], dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    center += forward * float(config["perception"]["visible_surface_center_bias_m"])
    table_z = float(config["table_top_z_m"])
    if config["perception"].get("infer_upright_center_z_from_table", False):
        center[2] = 0.5 * (table_z + float(bounds[1, 2]))
    height = max(float(bounds[1, 2] - table_z), 0.015)
    grasp_position = center.copy()
    grasp_position[2] = table_z + float(
        config["perception"]["upright_grasp_height_fraction"]
    ) * height
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
    }


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


def _shape_prior(geometry: dict[str, Any], labels: list[str]) -> np.ndarray:
    bounds = np.asarray(geometry["bbox3d_world_m"], dtype=np.float64)
    dimensions = np.maximum(bounds[1] - bounds[0], 1e-4)
    ordered = np.sort(dimensions)
    thinness = ordered[0] / ordered[-1]
    elongation = ordered[-1] / ordered[1]
    height_ratio = dimensions[2] / max(dimensions[0], dimensions[1])
    prior = np.zeros(len(labels), dtype=np.float64)
    index = {label: position for position, label in enumerate(labels)}
    if thinness > 0.62:
        prior[index["cube"]] += 0.20
    if elongation > 2.8 and thinness < 0.24:
        prior[index["spoon"]] += 0.24
    if height_ratio > 2.2:
        prior[index["water_bottle"]] += 0.14
    if height_ratio < 0.55:
        prior[index["bowl"]] += 0.14
    if 0.65 < height_ratio < 2.0:
        prior[index["soup_can"]] += 0.04
        prior[index["coffee_jar"]] += 0.04
    return prior


class WristRgbdSemanticPerception:
    """Detect all current-view objects without reading scene object metadata."""

    def __init__(self, config_path: Path, project_root: Path):
        self.project_root = project_root
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self._runtime: tuple[Any, ...] | None = None

    @property
    def label_criteria(self) -> dict[str, str]:
        return dict(LABEL_CRITERIA)

    @property
    def minimum_target_score(self) -> float:
        return float(self.config["perception"]["minimum_target_semantic_score"])

    def _clip_runtime(self) -> tuple[Any, ...]:
        if self._runtime is not None:
            return self._runtime
        import clip
        import torch

        clip_ref = str(self.config["perception"]["clip_model"])
        candidate = (self.project_root / clip_ref).resolve()
        source = str(candidate) if candidate.is_file() else clip_ref
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(source, device=device)
        model.eval()
        labels = list(PROMPT_BANK)
        features = []
        with torch.inference_mode():
            for label in labels:
                tokens = clip.tokenize(PROMPT_BANK[label]).to(device)
                encoded = model.encode_text(tokens)
                encoded /= encoded.norm(dim=-1, keepdim=True)
                mean = encoded.mean(dim=0)
                features.append(mean / mean.norm())
        self._runtime = (
            model,
            preprocess,
            labels,
            torch.stack(features),
            device,
            torch,
        )
        return self._runtime

    def detect(self, packet: dict[str, Any]) -> list[dict[str, Any]]:
        rgb = np.asarray(Image.open(packet["rgb_path"]).convert("RGB"))
        depth = np.load(packet["depth_path"]).astype(np.float32)
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
            packet["pose"],
            packet["orientation_wxyz"],
            local,
            camera["gripper_exclusion_box_min_m"],
            camera["gripper_exclusion_box_max_m"],
        )
        masks = cluster_table_objects(
            depth_mm,
            packet["intrinsics"],
            packet["pose"],
            table_z_m=float(self.config["table_top_z_m"]),
            pixel_stride=int(perception["pixel_stride"]),
            eps_m=float(perception["dbscan_eps_m"]),
            min_samples=int(perception["dbscan_min_samples"]),
        )
        candidates = []
        candidate_masks = []
        for mask in masks:
            geometry = mask_geometry(mask, depth_mm, packet["intrinsics"], packet["pose"])
            if geometry is None:
                continue
            if geometry["area_px"] < int(perception["minimum_mask_area_px"]):
                continue
            candidates.append(geometry)
            candidate_masks.append(mask)
        if not candidates:
            return []

        model, preprocess, labels, text_features, device, torch = self._clip_runtime()
        crops = [
            _masked_crop(rgb, mask, candidate["bbox_xyxy"])
            for candidate, mask in zip(candidates, candidate_masks, strict=True)
        ]
        batch = torch.stack([preprocess(crop) for crop in crops]).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        observations = []
        for candidate, mask, probability in zip(
            candidates, candidate_masks, probabilities, strict=True
        ):
            scores = probability.float().cpu().numpy().astype(np.float64)
            scores += _shape_prior(candidate, labels)
            scores /= scores.sum()
            best = int(np.argmax(scores))
            grasp = _grasp_geometry(
                candidate,
                mask,
                depth_mm,
                packet["intrinsics"],
                packet["pose"],
                self.config,
            )
            color = _dominant_color(rgb, mask)
            observations.append(
                {
                    "label": labels[best],
                    "description": f"{color} {labels[best]}",
                    "attributes": {"color": color},
                    "confidence": float(scores[best]),
                    "pose": {
                        "frame": "robot_base",
                        "position_m": grasp["center_world_m"],
                    },
                    "source": "wrist_rgbd_geometry_clip",
                    "mask_area_px": int(candidate["area_px"]),
                    "semantic_scores": {
                        label: float(scores[index]) for index, label in enumerate(labels)
                    },
                    "bbox_xyxy": candidate["bbox_xyxy"],
                    "bbox3d_world_m": candidate["bbox3d_world_m"],
                    "grasp": grasp,
                }
            )
        return observations
