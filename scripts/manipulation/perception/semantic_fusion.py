"""Causal multi-view semantic memory for geometry-defined object tracks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _shape_prior(observation: dict[str, Any], labels: list[str]) -> np.ndarray:
    bounds = np.asarray(observation["bbox3d_world_m"], dtype=np.float64)
    dimensions = np.maximum(bounds[1] - bounds[0], 1e-4)
    ordered = np.sort(dimensions)
    thinness = ordered[0] / ordered[-1]
    elongation = ordered[-1] / ordered[1]
    height_ratio = dimensions[2] / max(dimensions[0], dimensions[1])
    prior = np.zeros(len(labels), dtype=np.float64)
    index = {label: position for position, label in enumerate(labels)}
    if "cube" in index and thinness > 0.62:
        prior[index["cube"]] += 0.20
    if "spoon" in index and elongation > 2.8 and thinness < 0.24:
        prior[index["spoon"]] += 0.24
    if "water_bottle" in index and height_ratio > 2.2:
        prior[index["water_bottle"]] += 0.14
    if "bowl" in index and height_ratio < 0.55:
        prior[index["bowl"]] += 0.14
    if 0.65 < height_ratio < 2.0:
        for label in ("soup_can", "coffee_jar"):
            if label in index:
                prior[index[label]] += 0.04
    return prior


class CausalSemanticFusion:
    """Accumulate CLIP, geometry and YOLO evidence without inventing objects."""

    def __init__(self, labels: list[str], *, yolo_weight: float = 2.0):
        self.labels = list(labels)
        self._index = {label: index for index, label in enumerate(self.labels)}
        self._yolo_weight = float(yolo_weight)
        self._evidence: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.labels), dtype=np.float64)
        )
        self._view_weight: dict[str, float] = defaultdict(float)

    def update(self, track_id: str, observation: dict[str, Any]) -> dict[str, Any]:
        result = dict(observation)
        raw = observation.get("view_semantic_scores") or observation.get("semantic_scores") or {}
        probability = np.asarray(
            [float(raw.get(label, 0.0)) for label in self.labels], dtype=np.float64
        )
        total = float(probability.sum())
        if total <= 0.0:
            return result
        probability /= total
        area = max(10, int(observation.get("mask_area_px", 10)))
        view_weight = min(2.0, max(0.5, float(np.log10(area)) - 1.0))
        self._evidence[track_id] += probability * view_weight
        self._evidence[track_id] += _shape_prior(observation, self.labels)
        detector_label = observation.get("detector_label")
        if detector_label in self._index:
            detector_weight = self._yolo_weight * float(observation.get("detector_score", 0.0))
            self._evidence[track_id][self._index[str(detector_label)]] += detector_weight
        self._view_weight[track_id] += view_weight
        normalized = self._evidence[track_id] / max(float(self._evidence[track_id].sum()), 1e-9)
        best = int(np.argmax(normalized))
        label = self.labels[best]
        color = str((result.get("attributes") or {}).get("color") or "unknown")
        result.update(
            {
                "label": label,
                "description": f"{color} {label}",
                "confidence": float(normalized[best]),
                "semantic_scores": {
                    name: float(normalized[index]) for index, name in enumerate(self.labels)
                },
                "semantic_label_source": "causal_multi_view_clip_yolo",
                "semantic_views_weight": float(self._view_weight[track_id]),
                "source": "wrist_rgbd_world_cluster_semantic_fusion",
            }
        )
        return result
