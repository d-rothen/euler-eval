"""Normal-consistency metric for depth maps.

Surface normals are derived from the **unprojected** depth map: every pixel is
turned into a 3D camera-frame point with the camera intrinsics, and the normal
is the cross product of central-difference tangents — the same estimator the
``points_3d`` modality uses (:func:`.points3d_geometry.points_to_normals`).

Depth maps store either planar (Z) or radial (Euclidean) distance; pass
``is_radial`` accordingly so the unprojection is exact in both cases.

Intrinsics may be given as a ``(3, 3)`` camera matrix, an ``{fx, fy, cx, cy}``
mapping, or a scalar focal length in pixels.  When they are unknown, a pinhole
camera with the principal point at the image centre and ``fx = fy = width``
(≈ 53° horizontal field of view) is assumed; metric metadata reports which of
the two happened via ``intrinsics_source``.

Both sides of a comparison are unprojected with the same camera, so the metric
stays meaningful under an assumed camera — but only real intrinsics make the
normals true camera-frame normals.
"""

from collections.abc import Mapping
from typing import Optional, Union

import numpy as np
from scipy import ndimage

from .points3d_geometry import points_to_normals
from .utils import angles_between_unit_vectors

# Focal length assumed when no intrinsics are available, as a multiple of the
# image width. 1.0 * width corresponds to a ~53° horizontal field of view.
ASSUMED_FOCAL_LENGTH_FACTOR = 1.0


def resolve_intrinsics(
    shape: tuple[int, int],
    intrinsics: Optional[Union[np.ndarray, Mapping, float]] = None,
) -> tuple[dict, str]:
    """Resolve pinhole intrinsics for a depth map of *shape*.

    Args:
        shape: ``(height, width)`` of the depth map the intrinsics describe.
        intrinsics: A ``(3, 3)`` camera matrix, an ``{fx, fy, cx, cy}`` mapping,
            a scalar focal length in pixels, or ``None``.

    Returns:
        ``({"fx", "fy", "cx", "cy"}, source)`` where *source* is ``"sample"``
        for usable supplied intrinsics and ``"assumed"`` for the fallback
        pinhole camera.
    """
    height, width = int(shape[0]), int(shape[1])
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    assumed = {
        "fx": float(width) * ASSUMED_FOCAL_LENGTH_FACTOR,
        "fy": float(width) * ASSUMED_FOCAL_LENGTH_FACTOR,
        "cx": center_x,
        "cy": center_y,
    }

    values = _extract_intrinsics(intrinsics, center_x, center_y)
    if values is None:
        return assumed, "assumed"
    return values, "sample"


def _extract_intrinsics(
    intrinsics: Optional[Union[np.ndarray, Mapping, float]],
    center_x: float,
    center_y: float,
) -> Optional[dict]:
    """Return usable ``{fx, fy, cx, cy}`` from *intrinsics*, or ``None``."""
    if intrinsics is None:
        return None

    if isinstance(intrinsics, Mapping):
        values = {
            "fx": intrinsics.get("fx"),
            "fy": intrinsics.get("fy"),
            "cx": intrinsics.get("cx", center_x),
            "cy": intrinsics.get("cy", center_y),
        }
    else:
        matrix = np.asarray(intrinsics, dtype=np.float64)
        if matrix.ndim == 0:  # scalar focal length in pixels
            values = {
                "fx": matrix.item(),
                "fy": matrix.item(),
                "cx": center_x,
                "cy": center_y,
            }
        elif matrix.ndim == 2 and matrix.shape[0] >= 3 and matrix.shape[1] >= 3:
            values = {
                "fx": matrix[0, 0],
                "fy": matrix[1, 1],
                "cx": matrix[0, 2],
                "cy": matrix[1, 2],
            }
        else:
            return None

    try:
        resolved = {key: float(value) for key, value in values.items()}
    except (TypeError, ValueError):
        return None

    if not all(np.isfinite(value) for value in resolved.values()):
        return None
    if resolved["fx"] <= 0.0 or resolved["fy"] <= 0.0:
        return None
    return resolved


def depth_to_points(
    depth: np.ndarray,
    *,
    intrinsics: Optional[Union[np.ndarray, Mapping, float]] = None,
    is_radial: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Unproject a depth map into a camera-frame point map.

    For pixel ``(u, v)`` with normalized ray direction
    ``d = ((u - cx) / fx, (v - cy) / fy, 1)``, the 3D point is ``z · d`` where
    ``z`` is the planar depth.  Radial depth ``r`` is the length of ``z · d``,
    so it converts back with ``z = r / ‖d‖`` — the exact inverse of
    :func:`euler_eval.metrics.utils.convert_planar_to_radial`.

    Args:
        depth: ``(H, W)`` depth map in metres.
        intrinsics: Camera matrix, ``{fx, fy, cx, cy}`` mapping, scalar focal
            length, or ``None`` for the assumed pinhole camera.
        is_radial: True when *depth* stores Euclidean distance from the camera
            centre rather than planar Z.
        dtype: Working precision; float32 matches the depth maps the evaluator
            carries and halves the cost at 1080p and above.

    Returns:
        ``(H, W, 3)`` camera-frame point map in metres.
    """
    depth = np.asarray(depth, dtype=dtype)
    if depth.ndim != 2:
        raise ValueError(f"depth must be a (H, W) array, got shape {depth.shape}")

    height, width = depth.shape
    values, _ = resolve_intrinsics((height, width), intrinsics)

    u, v = np.meshgrid(
        np.arange(width, dtype=dtype),
        np.arange(height, dtype=dtype),
    )
    x = (u - values["cx"]) / values["fx"]
    y = (v - values["cy"]) / values["fy"]

    if is_radial:
        z = depth / np.sqrt(x * x + y * y + 1.0)
    else:
        z = depth

    return np.stack([x * z, y * z, z], axis=-1)


def depth_to_normals(
    depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    *,
    intrinsics: Optional[Union[np.ndarray, Mapping, float]] = None,
    is_radial: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Compute camera-frame surface normals from a depth map.

    The depth map is unprojected with :func:`depth_to_points` and normals are
    taken as ``n = (∂P/∂u) × (∂P/∂v)`` over central differences, normalized to
    unit length.  Normals point away from the camera (``+z`` for a
    fronto-parallel surface).

    Args:
        depth: ``(H, W)`` depth map in metres.
        valid_mask: Optional ``(H, W)`` bool mask; invalid pixels are zeroed.
        intrinsics: Camera intrinsics; see :func:`resolve_intrinsics`.
        is_radial: True when *depth* stores radial rather than planar depth.
        dtype: Working precision for the unprojection and the tangents.

    Returns:
        ``(H, W, 3)`` unit-normal map (``0`` at invalid pixels).
    """
    if valid_mask is None:
        depth_array = np.asarray(depth)
        valid_mask = (depth_array > 0) & np.isfinite(depth_array)

    points = depth_to_points(
        depth, intrinsics=intrinsics, is_radial=is_radial, dtype=dtype
    )
    return points_to_normals(points, valid_mask, dtype=dtype)


def _normal_angles(
    normals_pred: np.ndarray,
    normals_gt: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Per-pixel angle (degrees) between two normal maps over *valid_mask*."""
    return angles_between_unit_vectors(
        normals_pred[valid_mask], normals_gt[valid_mask]
    )


def _erode_valid_mask(valid_mask: np.ndarray) -> np.ndarray:
    """Erode by the finite-difference stencil so borders do not contribute."""
    kernel = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(valid_mask, kernel)
    return eroded if eroded is not None else valid_mask


def compute_normal_angles(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    *,
    intrinsics: Optional[Union[np.ndarray, Mapping, float]] = None,
    is_radial: bool = False,
    dtype: np.dtype = np.float32,
    return_metadata: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
    """Per-pixel angular errors between predicted and GT surface normals.

    Both depth maps are unprojected with the *same* camera, so the comparison
    is well posed even when intrinsics have to be assumed.

    Args:
        depth_pred: Predicted depth map in metres.
        depth_gt: Ground-truth depth map in metres.
        valid_mask: Optional mask of valid pixels to consider.
        intrinsics: Camera intrinsics; see :func:`resolve_intrinsics`.
        is_radial: True when both depth maps store radial depth.
        dtype: Working precision for the unprojection and the tangents.
        return_metadata: If True, also return a metadata dict for sanity
            checking.

    Returns:
        Array of angular errors in degrees over the eroded valid mask.  If
        *return_metadata* is True, returns ``(angles, metadata)``.
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    initial_valid_count = int(np.sum(valid_mask))

    resolved, source = resolve_intrinsics(np.shape(depth_gt)[:2], intrinsics)
    normals_pred = depth_to_normals(
        depth_pred, valid_mask, intrinsics=resolved, is_radial=is_radial, dtype=dtype
    )
    normals_gt = depth_to_normals(
        depth_gt, valid_mask, intrinsics=resolved, is_radial=is_radial, dtype=dtype
    )

    valid_mask = _erode_valid_mask(valid_mask)
    valid_after_erosion = int(np.sum(valid_mask))

    metadata = {
        "valid_pixels_before_erosion": initial_valid_count,
        "valid_pixels_after_erosion": valid_after_erosion,
        "focal_length_used": resolved["fx"],
        "intrinsics_used": dict(resolved),
        "intrinsics_source": source,
        "depth_is_radial": bool(is_radial),
        "mean_angle": None,
    }

    if not valid_mask.any():
        if return_metadata:
            return np.array([]), metadata
        return np.array([])

    angles = _normal_angles(normals_pred, normals_gt, valid_mask)
    metadata["mean_angle"] = float(np.mean(angles))

    if return_metadata:
        return angles, metadata
    return angles


def aggregate_normal_consistency(
    angle_arrays: list[np.ndarray],
) -> dict:
    """Aggregate normal consistency from multiple depth map pairs.

    Args:
        angle_arrays: List of per-pixel angular error arrays.

    Returns:
        Dictionary with aggregated statistics.
    """
    all_angles = np.concatenate([a for a in angle_arrays if len(a) > 0])

    if len(all_angles) == 0:
        return {
            "mean_angle": float("nan"),
            "median_angle": float("nan"),
            "percent_below_11_25": float("nan"),
            "percent_below_22_5": float("nan"),
            "percent_below_30": float("nan"),
        }

    return {
        "mean_angle": float(np.mean(all_angles)),
        "median_angle": float(np.median(all_angles)),
        "percent_below_11_25": float(np.mean(all_angles < 11.25) * 100),
        "percent_below_22_5": float(np.mean(all_angles < 22.5) * 100),
        "percent_below_30": float(np.mean(all_angles < 30.0) * 100),
    }
