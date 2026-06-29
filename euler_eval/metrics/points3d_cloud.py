"""Set-level surface agreement (Chamfer / F-score) for ``points_3d``.

A correspondence-free complement to the per-pixel Euclidean metrics: it asks
whether the predicted surface matches the GT surface *as a set of points*,
regardless of which pixel each point landed on.  This is robust to camera
mis-parameterization (which slides points along an otherwise-correct surface)
and is the standard 3D-reconstruction metric family (Chamfer in DUSt3R/MASt3R,
F-score in DTU / Tanks-and-Temples).

F-score is the bounded, outlier-robust headline; raw Chamfer (reported with
separate accuracy/completeness and a median) is the diagnostic.  Point clouds
are deterministically subsampled to bound the KD-tree cost.
"""

from typing import Optional

import numpy as np

# F-score distance thresholds (metres).
FSCORE_THRESHOLDS = (0.05, 0.1, 0.25, 0.5)
# Default cap on points per cloud before the KD-tree queries.
DEFAULT_MAX_POINTS = 50000


def threshold_key(tau: float) -> str:
    """JSON-safe F-score threshold key, e.g. ``0.1 -> "tau_0_1"``."""
    return "tau_" + ("%g" % tau).replace(".", "_")


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically subsample to at most *max_points* via uniform stride."""
    n = points.shape[0]
    if max_points is None or n <= max_points:
        return points
    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    return points[idx]


def compute_cloud_distance_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_points: int = DEFAULT_MAX_POINTS,
    thresholds: tuple = FSCORE_THRESHOLDS,
) -> Optional[dict]:
    """Chamfer distance and F-score between two 3D point sets.

    Args:
        pred_points: ``(N, 3)`` predicted points.
        gt_points: ``(M, 3)`` ground-truth points.
        max_points: Per-cloud subsample cap (``None`` to disable).
        thresholds: F-score distance thresholds in metres.

    Returns:
        Dict with a ``chamfer`` block (``accuracy``, ``completeness``,
        ``distance``, ``median``) and an ``fscore`` block keyed by threshold
        (``precision``, ``recall``, ``f1``).  ``None`` if either set is empty.
    """
    from scipy.spatial import cKDTree

    pred = np.asarray(pred_points, dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return None

    pred = _subsample(pred, max_points)
    gt = _subsample(gt, max_points)

    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    # accuracy: pred -> nearest gt ; completeness: gt -> nearest pred
    dist_pred_to_gt, _ = tree_gt.query(pred, k=1)
    dist_gt_to_pred, _ = tree_pred.query(gt, k=1)

    accuracy = float(np.mean(dist_pred_to_gt))
    completeness = float(np.mean(dist_gt_to_pred))
    pooled = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])

    chamfer = {
        "accuracy": accuracy,
        "completeness": completeness,
        "distance": float(0.5 * (accuracy + completeness)),
        "median": float(np.median(pooled)),
    }

    fscore = {}
    for tau in thresholds:
        precision = float(np.mean(dist_pred_to_gt < tau))
        recall = float(np.mean(dist_gt_to_pred < tau))
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fscore[threshold_key(tau)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {"chamfer": chamfer, "fscore": fscore}
