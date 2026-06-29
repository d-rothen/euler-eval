"""Geometric (surface/structure) metrics for the ``points_3d`` modality.

Unlike the depth ``normal_consistency`` metric — which assumes a focal length
of ``1.0`` and estimates normals in image space — these metrics derive surface
normals **directly from the 3D point map** via the cross product of
central-difference tangents, so they reflect true camera-frame geometry.  A 3D
discontinuity F1 (the point-map analog of depth edge F1) is also provided.
"""

from typing import Optional, Union

import numpy as np
from scipy import ndimage


def points_to_normals(
    points: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate per-pixel surface normals from a 3D point map.

    Computes tangents with central differences along the image axes and takes
    their cross product, ``n = (∂P/∂u) × (∂P/∂v)``, normalized to unit length.

    Args:
        points: ``(H, W, 3)`` camera-frame point map in metres.
        valid_mask: Optional ``(H, W)`` bool mask; invalid pixels are zeroed.

    Returns:
        ``(H, W, 3)`` unit-normal map (``0`` at invalid pixels).
    """
    P = np.asarray(points, dtype=np.float64)
    height, width = P.shape[:2]

    dpdu = np.zeros_like(P)
    dpdv = np.zeros_like(P)
    if width >= 3:
        dpdu[:, 1:-1, :] = P[:, 2:, :] - P[:, :-2, :]
    if height >= 3:
        dpdv[1:-1, :, :] = P[2:, :, :] - P[:-2, :, :]

    normals = np.cross(dpdu, dpdv)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    normals = normals / norm

    if valid_mask is not None:
        normals[~valid_mask] = 0.0
    return normals.astype(np.float32)


def compute_point_normal_angles(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    return_metadata: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
    """Per-pixel angular error between point-map normals (degrees).

    Args:
        points_pred: ``(H, W, 3)`` predicted point map.
        points_gt: ``(H, W, 3)`` ground-truth point map.
        valid_mask: Optional ``(H, W)`` bool mask of valid pixels.
        return_metadata: If True, also return a metadata dict.

    Returns:
        1-D array of normal angular errors (degrees) over the eroded valid
        mask.  If *return_metadata* is True, returns ``(angles, metadata)``.
    """
    if valid_mask is None:
        valid_mask = (
            np.isfinite(points_gt).all(axis=-1)
            & np.isfinite(points_pred).all(axis=-1)
            & (np.linalg.norm(points_gt, axis=-1) > 0)
        )

    initial_valid = int(np.sum(valid_mask))

    normals_pred = points_to_normals(points_pred, valid_mask)
    normals_gt = points_to_normals(points_gt, valid_mask)

    # Erode to avoid border artefacts from the finite-difference stencil.
    kernel = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(valid_mask, kernel)
    valid_mask = eroded if eroded is not None else valid_mask

    metadata = {
        "valid_pixels_before_erosion": initial_valid,
        "valid_pixels_after_erosion": int(np.sum(valid_mask)),
        "mean_angle": None,
    }

    if not valid_mask.any():
        if return_metadata:
            return np.array([]), metadata
        return np.array([])

    dots = np.sum(normals_pred * normals_gt, axis=2)
    dots = np.clip(dots[valid_mask], -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))

    metadata["mean_angle"] = float(np.mean(angles))
    if return_metadata:
        return angles, metadata
    return angles


def _max_neighbor_distance(points: np.ndarray) -> np.ndarray:
    """Max 4-neighbour 3D distance per pixel for a ``(H, W, 3)`` map."""
    P = np.asarray(points, dtype=np.float64)
    height, width = P.shape[:2]
    max_dist = np.zeros((height, width), dtype=np.float64)

    if width >= 2:
        horiz = np.linalg.norm(P[:, 1:, :] - P[:, :-1, :], axis=-1)
        max_dist[:, :-1] = np.maximum(max_dist[:, :-1], horiz)
        max_dist[:, 1:] = np.maximum(max_dist[:, 1:], horiz)
    if height >= 2:
        vert = np.linalg.norm(P[1:, :, :] - P[:-1, :, :], axis=-1)
        max_dist[:-1, :] = np.maximum(max_dist[:-1, :], vert)
        max_dist[1:, :] = np.maximum(max_dist[1:, :], vert)
    return max_dist


def _detect_point_edges(
    points: np.ndarray,
    valid_mask: np.ndarray,
    rel_threshold: float,
) -> np.ndarray:
    """Boolean 3D discontinuity map: neighbour distance > ``rel_threshold·‖P‖``."""
    max_dist = _max_neighbor_distance(points)
    local_norm = np.linalg.norm(np.asarray(points, dtype=np.float64), axis=-1)
    edges = (max_dist > rel_threshold * np.maximum(local_norm, 1e-8)) & valid_mask
    return edges


def compute_point_edge_f1(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    rel_threshold: float = 0.1,
    pixel_tolerance: int = 1,
) -> dict:
    """3D discontinuity F1 between predicted and GT point maps.

    Marks an edge where the maximum 4-neighbour 3D distance exceeds
    ``rel_threshold`` times the local point norm, then matches predicted and
    GT edges with a dilation tolerance (the point-map analog of the depth
    ``relative`` edge F1).

    Returns:
        Dict with ``precision``, ``recall``, ``f1``.
    """
    if valid_mask is None:
        valid_mask = (
            np.isfinite(points_gt).all(axis=-1)
            & np.isfinite(points_pred).all(axis=-1)
            & (np.linalg.norm(points_gt, axis=-1) > 0)
        )

    pred_edges = _detect_point_edges(points_pred, valid_mask, rel_threshold)
    gt_edges = _detect_point_edges(points_gt, valid_mask, rel_threshold)

    if pixel_tolerance > 0:
        struct = ndimage.generate_binary_structure(2, 2)
        gt_dil = ndimage.binary_dilation(
            gt_edges, structure=struct, iterations=pixel_tolerance
        )
        pred_dil = ndimage.binary_dilation(
            pred_edges, structure=struct, iterations=pixel_tolerance
        )
    else:
        gt_dil = gt_edges
        pred_dil = pred_edges

    n_pred = int(np.sum(pred_edges))
    n_gt = int(np.sum(gt_edges))

    tp_precision = int(np.sum(pred_edges & gt_dil))
    tp_recall = int(np.sum(gt_edges & pred_dil))

    precision = tp_precision / n_pred if n_pred > 0 else 0.0
    recall = tp_recall / n_gt if n_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_pred_edges": n_pred,
        "num_gt_edges": n_gt,
    }
