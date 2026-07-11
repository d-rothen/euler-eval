"""Camera-calibration extraction from euler-loading samples.

MultiModalDataset samples carry calibration either directly (a ``(3, 3)`` /
``(4, 4)`` tensor per sample) or wrapped in hierarchical-modality dicts and
nested calibration payloads (e.g. a MUSES ``calib.json`` with named
``lidar2rgb`` transforms).  This module owns the tolerant extraction logic
shared by the CLI evaluators (:mod:`euler_eval.evaluate`) and the
programmatic validation interface (:mod:`euler_eval.validation`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np

from .data import (
    compose_sensor_to_camera_extrinsics,
    to_numpy_extrinsics,
    to_numpy_intrinsics,
)

__all__ = [
    "get_first_hierarchical_value",
    "iter_hierarchical_values",
    "get_sample_intrinsics",
    "get_sample_pointcloud_to_camera_extrinsics",
    "extract_wrapped_extrinsics",
    "extract_named_extrinsics",
    "extract_direct_source_to_camera",
    "DIRECT_SOURCE_TO_CAMERA_KEYS",
    "SOURCE_SENSOR_POSE_KEYS",
    "CAMERA_SENSOR_POSE_KEYS",
    "MATRIX_WRAPPER_KEYS",
]


def get_first_hierarchical_value(sample: dict, key: str):
    """Return ``sample[key]``, unwrapping hierarchical ``{file_id: value}`` dicts."""
    data = sample.get(key)
    if data is None:
        return None
    if isinstance(data, dict):
        if not data:
            return None
        return next(iter(data.values()))
    return data


def iter_hierarchical_values(sample: dict, key: str) -> list:
    """Return candidate values for ``sample[key]`` (dict itself plus entries)."""
    data = sample.get(key)
    if data is None:
        return []
    if isinstance(data, dict):
        return [data, *data.values()]
    return [data]


def get_sample_intrinsics(sample: dict) -> Optional[np.ndarray]:
    """Extract a ``(3, 3)`` intrinsics matrix from sample calibration data."""
    K_data = get_first_hierarchical_value(sample, "intrinsics")
    if K_data is None:
        K_data = get_first_hierarchical_value(sample, "calibration")
    if K_data is None:
        return None
    return to_numpy_intrinsics(K_data)


DIRECT_SOURCE_TO_CAMERA_KEYS = (
    "lidar2rgb",
    "lidar2camera",
    "lidar2cam",
    "lidar_to_rgb",
    "lidar_to_camera",
    "lidar_to_cam",
    "source2rgb",
    "source2camera",
    "source_to_rgb",
    "source_to_camera",
    "sensor2rgb",
    "sensor2camera",
    "sensor_to_rgb",
    "sensor_to_camera",
)
SOURCE_SENSOR_POSE_KEYS = (
    "lidar",
    "lidar_pose",
    "lidar_extrinsics",
    "source",
    "source_sensor",
    "source_extrinsics",
    "sensor",
    "sensor_extrinsics",
    "lidar2gnss",
    "lidar2ego",
    "lidar2vehicle",
    "lidar2world",
    "lidar_to_gnss",
    "lidar_to_ego",
    "lidar_to_vehicle",
    "lidar_to_world",
)
CAMERA_SENSOR_POSE_KEYS = (
    "rgb",
    "camera",
    "cam",
    "frame_camera",
    "rgb_pose",
    "camera_pose",
    "camera_extrinsics",
    "target",
    "target_sensor",
    "target_extrinsics",
    "rgb2gnss",
    "camera2gnss",
    "cam2gnss",
    "frame_camera2gnss",
    "rgb2ego",
    "camera2ego",
    "cam2ego",
    "frame_camera2ego",
    "rgb2vehicle",
    "camera2vehicle",
    "cam2vehicle",
    "frame_camera2vehicle",
    "rgb2world",
    "camera2world",
    "cam2world",
    "frame_camera2world",
)
MATRIX_WRAPPER_KEYS = ("T", "transform", "matrix", "extrinsics", "pose")


def _as_extrinsics_or_none(data) -> Optional[np.ndarray]:
    try:
        return to_numpy_extrinsics(data)
    except (TypeError, ValueError):
        return None


def extract_wrapped_extrinsics(data) -> Optional[np.ndarray]:
    """Extract a ``(4, 4)`` transform from a value or common wrapper mappings."""
    matrix = _as_extrinsics_or_none(data)
    if matrix is not None:
        return matrix

    if not isinstance(data, Mapping):
        return None

    for key in MATRIX_WRAPPER_KEYS:
        if key in data:
            matrix = extract_wrapped_extrinsics(data[key])
            if matrix is not None:
                return matrix

    if len(data) == 1:
        return extract_wrapped_extrinsics(next(iter(data.values())))

    return None


def extract_named_extrinsics(data, names: tuple[str, ...]) -> Optional[np.ndarray]:
    """Extract a transform stored under one of ``names`` in a calibration payload."""
    if not isinstance(data, Mapping):
        return extract_wrapped_extrinsics(data)

    for name in names:
        if name in data:
            matrix = extract_wrapped_extrinsics(data[name])
            if matrix is not None:
                return matrix

    for key in ("extrinsics", "sensors", "calibration", "poses"):
        if key in data and isinstance(data[key], Mapping):
            matrix = extract_named_extrinsics(data[key], names)
            if matrix is not None:
                return matrix

    if len(data) == 1:
        return extract_named_extrinsics(next(iter(data.values())), names)

    return None


def extract_direct_source_to_camera(data) -> Optional[np.ndarray]:
    """Extract a direct source→camera transform (e.g. MUSES ``lidar2rgb``)."""
    matrix = _as_extrinsics_or_none(data)
    if matrix is not None:
        return matrix

    if not isinstance(data, Mapping):
        return None

    for key in DIRECT_SOURCE_TO_CAMERA_KEYS:
        if key in data:
            matrix = extract_wrapped_extrinsics(data[key])
            if matrix is not None:
                return matrix

    for key in ("extrinsics", "calibration"):
        if key in data and isinstance(data[key], Mapping):
            matrix = extract_direct_source_to_camera(data[key])
            if matrix is not None:
                return matrix

    if len(data) == 1:
        return extract_direct_source_to_camera(next(iter(data.values())))

    return None


def get_sample_pointcloud_to_camera_extrinsics(
    sample: dict,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Extract or compose the transform used for sparse point projection.

    The common MUSES path supplies a direct ``lidar2rgb`` transform via
    ``camera_extrinsics``.  Some datasets instead expose separate lidar and
    camera sensor poses in a shared frame; in that case we compose them to the
    same source-to-camera transform expected by the projector.
    """
    lidar_pose = None
    for value in iter_hierarchical_values(sample, "lidar_extrinsics"):
        lidar_pose = extract_named_extrinsics(value, SOURCE_SENSOR_POSE_KEYS)
        if lidar_pose is not None:
            break

    camera_values = iter_hierarchical_values(sample, "camera_extrinsics")
    if not camera_values:
        camera_values = iter_hierarchical_values(sample, "extrinsics")

    if lidar_pose is not None:
        for value in camera_values:
            camera_pose = extract_named_extrinsics(value, CAMERA_SENSOR_POSE_KEYS)
            if camera_pose is not None:
                return (
                    compose_sensor_to_camera_extrinsics(lidar_pose, camera_pose),
                    "composed_lidar_and_camera_sensor_poses",
                )
        return None, None

    for value in camera_values:
        direct = extract_direct_source_to_camera(value)
        if direct is not None:
            return direct, "direct_source_to_camera"

    for value in camera_values:
        lidar_pose = extract_named_extrinsics(value, SOURCE_SENSOR_POSE_KEYS)
        camera_pose = extract_named_extrinsics(value, CAMERA_SENSOR_POSE_KEYS)
        if lidar_pose is not None and camera_pose is not None:
            return (
                compose_sensor_to_camera_extrinsics(lidar_pose, camera_pose),
                "composed_lidar_and_camera_sensor_poses",
            )

    return None, None
