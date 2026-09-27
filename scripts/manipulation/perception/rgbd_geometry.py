"""Geometry-only RGB-D clustering with no simulator ground-truth access."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def quaternion_matrix(quaternion_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion_wxyz]
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def gripper_camera_local_matrix(
    translation: list[float],
    look_at: list[float],
) -> np.ndarray:
    eye = np.asarray(translation, dtype=np.float64)
    target = np.asarray(look_at, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    up_hint = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    if abs(float(np.dot(up_hint, z_axis))) > 0.95:
        up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = up_hint - np.dot(up_hint, z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    local = np.eye(4, dtype=np.float64)
    local[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    local[:3, 3] = eye
    return local


def camera_axes(pose: dict[str, Any]) -> tuple[np.ndarray, ...]:
    eye = np.asarray(pose["eye_world_m"], dtype=np.float64)
    orientation = pose.get("orientation_wxyz")
    if isinstance(orientation, (list, tuple)) and len(orientation) == 4:
        rotation = quaternion_matrix(orientation)
        right = rotation[:, 0]
        down = -rotation[:, 1]
        forward = -rotation[:, 2]
        return eye, right, down, forward
    target = np.asarray(pose["target_world_m"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return eye, right, down, forward


def pixels_to_camera(
    xs: np.ndarray,
    ys: np.ndarray,
    depths_m: np.ndarray,
    intrinsics: dict[str, Any],
) -> np.ndarray:
    x = (xs - float(intrinsics["cx"])) * depths_m / float(intrinsics["fx"])
    y = (ys - float(intrinsics["cy"])) * depths_m / float(intrinsics["fy"])
    return np.stack((x, y, depths_m), axis=-1)


def camera_to_world(points: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
    eye, right, down, forward = camera_axes(pose)
    return (
        eye[None, :]
        + points[:, 0:1] * right[None, :]
        + points[:, 1:2] * down[None, :]
        + points[:, 2:3] * forward[None, :]
    )


def world_maps(
    depth_mm: np.ndarray,
    intrinsics: dict[str, Any],
    pose: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_mm.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    depths_m = depth_mm.astype(np.float32) / 1000.0
    camera = pixels_to_camera(xs, ys, depths_m, intrinsics)
    eye, right, down, forward = camera_axes(pose)
    world = (
        eye[None, None, :]
        + camera[..., 0:1] * right[None, None, :]
        + camera[..., 1:2] * down[None, None, :]
        + camera[..., 2:3] * forward[None, None, :]
    )
    return camera, world


def mask_geometry(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: dict[str, Any],
    pose: dict[str, Any],
) -> dict[str, Any] | None:
    valid = (mask > 0) & (depth_mm > 0)
    ys, xs = np.nonzero(valid)
    if xs.size < 8:
        return None
    depths_m = depth_mm[ys, xs].astype(np.float64) / 1000.0
    camera = pixels_to_camera(xs.astype(float), ys.astype(float), depths_m, intrinsics)
    world = camera_to_world(camera, pose)
    camera_min = np.percentile(camera, 2, axis=0)
    camera_max = np.percentile(camera, 98, axis=0)
    world_min = np.percentile(world, 2, axis=0)
    world_max = np.percentile(world, 98, axis=0)
    median_depth = float(np.median(depths_m))
    near = np.abs(depths_m - median_depth) <= max(0.025, 0.04 * median_depth)
    source = world[near] if np.count_nonzero(near) >= 8 else world
    point_world = np.median(source, axis=0)
    return {
        "area_px": int(xs.size),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centroid_uv": [float(xs.mean()), float(ys.mean())],
        "median_depth_m": median_depth,
        "point_world_m": point_world.tolist(),
        "bbox3d_camera_m": [camera_min.tolist(), camera_max.tolist()],
        "bbox3d_world_m": [world_min.tolist(), world_max.tolist()],
    }


def depth_to_mm(depth: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > near_m) & (depth < far_m)
    result = np.zeros(depth.shape, dtype=np.uint16)
    result[valid] = np.clip(depth[valid] * 1000.0, 1, 65535).astype(np.uint16)
    return result


def remove_gripper(
    depth_mm: np.ndarray,
    intrinsics: dict[str, Any],
    pose: dict[str, Any],
    camera_orientation_wxyz: list[float],
    camera_local: np.ndarray,
    box_min: list[float],
    box_max: list[float],
) -> np.ndarray:
    oriented_pose = {**pose, "orientation_wxyz": camera_orientation_wxyz}
    _camera, world = world_maps(depth_mm, intrinsics, oriented_pose)
    camera_world = np.eye(4, dtype=np.float64)
    camera_world[:3, :3] = quaternion_matrix(camera_orientation_wxyz)
    camera_world[:3, 3] = np.asarray(pose["eye_world_m"], dtype=np.float64)
    hand_from_world = np.linalg.inv(camera_world @ np.linalg.inv(camera_local))
    flat = world.reshape(-1, 3).astype(np.float64, copy=False)
    local_points = (hand_from_world[:3, :3] @ flat.T).T + hand_from_world[:3, 3]
    local_points = local_points.reshape(world.shape)
    lower = np.asarray(box_min, dtype=np.float64)
    upper = np.asarray(box_max, dtype=np.float64)
    inside = np.all(local_points >= lower, axis=-1) & np.all(local_points <= upper, axis=-1)
    kept = depth_mm.copy()
    kept[inside] = 0
    return kept


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _merge_nearby_clusters(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Merge DBSCAN fragments using the rule from the winning benchmark method."""
    cluster_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    if not cluster_ids:
        return labels
    bounds = {
        cluster_id: (
            points[labels == cluster_id].min(axis=0),
            points[labels == cluster_id].max(axis=0),
        )
        for cluster_id in cluster_ids
    }
    parent = list(range(max(cluster_ids) + 1))
    for index, left in enumerate(cluster_ids):
        for right in cluster_ids[index + 1 :]:
            minimum_left, maximum_left = bounds[left]
            minimum_right, maximum_right = bounds[right]
            gap = np.maximum(
                0.0,
                np.maximum(minimum_left - maximum_right, minimum_right - maximum_left),
            )
            if float(np.linalg.norm(gap)) < 0.032:
                left_root, right_root = _find(parent, left), _find(parent, right)
                if left_root != right_root:
                    parent[right_root] = left_root
    roots = {cluster_id: _find(parent, cluster_id) for cluster_id in cluster_ids}
    root_order = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    merged = labels.copy()
    for cluster_id, root in roots.items():
        merged[labels == cluster_id] = root_order[root]
    return merged


def cluster_table_objects(
    depth_mm: np.ndarray,
    intrinsics: dict[str, Any],
    pose: dict[str, Any],
    *,
    table_z_m: float,
    pixel_stride: int,
    eps_m: float,
    min_samples: int,
) -> list[np.ndarray]:
    _camera, world = world_maps(depth_mm, intrinsics, pose)
    foreground = (
        (depth_mm > 0)
        & (world[..., 2] > table_z_m + 0.012)
        & (world[..., 2] < table_z_m + 0.48)
        & (np.abs(world[..., 0]) < 0.92)
        & (np.abs(world[..., 1]) < 0.60)
    )
    ys, xs = np.nonzero(foreground)
    if xs.size == 0:
        return []
    sample_use = (xs % pixel_stride == 0) & (ys % pixel_stride == 0)
    sample_points = world[ys[sample_use], xs[sample_use]]
    if len(sample_points) < min_samples:
        return []
    sample_labels = DBSCAN(
        eps=eps_m,
        min_samples=min_samples,
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit_predict(sample_points)
    sample_labels = _merge_nearby_clusters(sample_points, sample_labels)
    good_labels = []
    for label in sorted(int(value) for value in np.unique(sample_labels) if value >= 0):
        points = sample_points[sample_labels == label]
        extent = points.max(axis=0) - points.min(axis=0)
        if len(points) < 12 or extent[2] < 0.006 or float(np.max(extent)) > 0.55:
            continue
        good_labels.append(label)
    if not good_labels:
        return []
    selected = np.isin(sample_labels, good_labels)
    tree = cKDTree(sample_points[selected])
    kept_labels = sample_labels[selected]
    full_points = world[ys, xs]
    distances, nearest = tree.query(full_points, k=1, workers=-1)
    full_labels = np.full(len(full_points), -1, dtype=np.int32)
    close = distances <= max(0.045, eps_m * 1.35)
    full_labels[close] = kept_labels[nearest[close]]
    masks = []
    for label in good_labels:
        use = full_labels == label
        if int(np.count_nonzero(use)) < 30:
            continue
        mask = np.zeros(depth_mm.shape, dtype=np.uint8)
        mask[ys[use], xs[use]] = 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if int(np.count_nonzero(mask)) >= 30:
            masks.append(mask.astype(bool))
    return masks
