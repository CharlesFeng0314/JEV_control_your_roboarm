"""Geometry-only RGB-D clustering with no simulator ground-truth access."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree


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
    _camera, world = world_maps(depth_mm, intrinsics, pose)
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
    voxel_size = eps_m * 0.72
    voxel_keys = np.floor(sample_points / voxel_size).astype(np.int32)
    unique_keys, inverse, counts = np.unique(
        voxel_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    key_to_index = {tuple(key): index for index, key in enumerate(unique_keys.tolist())}
    visited = np.zeros(len(unique_keys), dtype=bool)
    sample_labels = np.full(len(sample_points), -1, dtype=np.int32)
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    next_label = 0
    for start in range(len(unique_keys)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            key = unique_keys[current]
            for offset in neighbor_offsets:
                neighbor = key_to_index.get(tuple((key + offset).tolist()))
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if int(np.sum(counts[component])) < max(12, min_samples):
            continue
        in_component = np.isin(inverse, component)
        points = sample_points[in_component]
        extent = points.max(axis=0) - points.min(axis=0)
        if extent[2] < 0.006 or float(np.max(extent)) > 0.55:
            continue
        sample_labels[in_component] = next_label
        next_label += 1
    if next_label == 0:
        return []
    selected = sample_labels >= 0
    tree = cKDTree(sample_points[selected])
    kept_labels = sample_labels[selected]
    full_points = world[ys, xs]
    distances, nearest = tree.query(full_points, k=1, workers=-1)
    full_labels = np.full(len(full_points), -1, dtype=np.int32)
    close = distances <= max(0.045, eps_m * 1.35)
    full_labels[close] = kept_labels[nearest[close]]
    masks = []
    for label in range(next_label):
        use = full_labels == label
        if int(np.count_nonzero(use)) < 30:
            continue
        mask = np.zeros(depth_mm.shape, dtype=np.uint8)
        mask[ys[use], xs[use]] = 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if int(np.count_nonzero(mask)) >= 30:
            masks.append(mask.astype(bool))
    return masks
