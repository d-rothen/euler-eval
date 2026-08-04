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


def compute_fscore_auc(
    dist_pred_to_gt: np.ndarray,
    dist_gt_to_pred: np.ndarray,
    max_threshold: float,
) -> Optional[float]:
    """Return the normalized area under the point-cloud F1 curve.

    ``F_A`` integrates the symmetric nearest-neighbour F1 score over all
    distance tolerances from zero through ``max_threshold``.  The empirical
    precision and recall curves are step functions, so this implementation
    integrates their harmonic mean exactly rather than choosing a sampling
    resolution.  The result is reported as a percentage in ``[0, 100]``.
    """
    pred_dist = np.asarray(dist_pred_to_gt, dtype=np.float64).reshape(-1)
    gt_dist = np.asarray(dist_gt_to_pred, dtype=np.float64).reshape(-1)
    if (
        pred_dist.size == 0
        or gt_dist.size == 0
        or not np.isfinite(max_threshold)
        or max_threshold <= 0
    ):
        return None

    pred_dist = np.sort(pred_dist[np.isfinite(pred_dist)])
    gt_dist = np.sort(gt_dist[np.isfinite(gt_dist)])
    if pred_dist.size == 0 or gt_dist.size == 0:
        return None

    events = np.unique(
        np.concatenate(
            [
                np.array([0.0, max_threshold], dtype=np.float64),
                pred_dist[(pred_dist >= 0.0) & (pred_dist < max_threshold)],
                gt_dist[(gt_dist >= 0.0) & (gt_dist < max_threshold)],
            ]
        )
    )
    left = events[:-1]
    widths = np.diff(events)
    # Immediately to the right of an event, all distances equal to that event
    # satisfy the paper's threshold test. Point values at the discontinuities
    # have zero measure and therefore do not affect the integral.
    precision = np.searchsorted(pred_dist, left, side="right") / pred_dist.size
    recall = np.searchsorted(gt_dist, left, side="right") / gt_dist.size
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.sum(f1 * widths) / max_threshold * 100.0)


def compute_cloud_distance_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_points: int = DEFAULT_MAX_POINTS,
    thresholds: tuple = FSCORE_THRESHOLDS,
    fscore_auc_max_threshold: Optional[float] = None,
) -> Optional[dict]:
    """Chamfer distance and F-score between two 3D point sets.

    Args:
        pred_points: ``(N, 3)`` predicted points.
        gt_points: ``(M, 3)`` ground-truth points.
        max_points: Per-cloud subsample cap (``None`` to disable).
        thresholds: F-score distance thresholds in metres.
        fscore_auc_max_threshold: When provided, also compute ``f_a``, the
            normalized F1 AUC through this distance threshold, as a percentage.

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

    result = {"chamfer": chamfer, "fscore": fscore}
    if fscore_auc_max_threshold is not None:
        f_a = compute_fscore_auc(
            dist_pred_to_gt,
            dist_gt_to_pred,
            fscore_auc_max_threshold,
        )
        if f_a is not None:
            result["f_a"] = f_a
    return result


def compute_sparse_cloud_distance_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_points: int = DEFAULT_MAX_POINTS,
    thresholds: tuple = FSCORE_THRESHOLDS,
    fscore_auc_max_threshold: Optional[float] = None,
) -> Optional[dict]:
    """Directed (GT→pred) cloud agreement for a sparse GT point set.

    When the ground truth is a sparse LiDAR cloud but the prediction is a
    *dense* surface, the symmetric Chamfer distance is misleading: a perfectly
    correct dense prediction still has many legitimate points far from any
    sparse GT return, so the ``pred→gt`` (accuracy / precision) side is
    dominated by that sparsity rather than by error.  This function therefore
    reports only the meaningful ``gt→pred`` direction — **completeness** (is
    every GT return covered by a nearby predicted point?) and the matching
    **recall** at each distance threshold.

    The result reuses the dense ``cloud_distance`` schema so the two paths share
    metric descriptions: ``chamfer.completeness`` / ``chamfer.median`` carry the
    GT→pred nearest-neighbour statistics and ``fscore.tau_<τ>.recall`` the
    coverage fractions.

    Args:
        pred_points: ``(N, 3)`` predicted (dense) points.
        gt_points: ``(M, 3)`` ground-truth (sparse) points.
        max_points: Per-cloud subsample cap (``None`` to disable).
        thresholds: Recall distance thresholds in metres.
        fscore_auc_max_threshold: When provided, compute symmetric ``f_a`` as
            the normalized F1 AUC through this distance threshold.  The fixed
            threshold metrics remain directed because the GT is sparse.

    Returns:
        Dict with a ``chamfer`` block (``completeness``, ``median``) and an
        ``fscore`` block keyed by threshold (``recall`` only).  ``None`` if
        either set is empty.
    """
    from scipy.spatial import cKDTree

    pred = np.asarray(pred_points, dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return None

    pred = _subsample(pred, max_points)
    gt = _subsample(gt, max_points)

    tree_pred = cKDTree(pred)
    # completeness: each GT point to its nearest predicted point.
    dist_gt_to_pred, _ = tree_pred.query(gt, k=1)

    chamfer = {
        "completeness": float(np.mean(dist_gt_to_pred)),
        "median": float(np.median(dist_gt_to_pred)),
    }

    fscore = {}
    for tau in thresholds:
        recall = float(np.mean(dist_gt_to_pred < tau))
        fscore[threshold_key(tau)] = {"recall": recall}

    result = {"chamfer": chamfer, "fscore": fscore}
    if fscore_auc_max_threshold is not None:
        # F_A is defined from symmetric F1, so it additionally needs the
        # prediction-to-GT distances even though the sparse diagnostic leaves
        # above intentionally remain completeness/recall-only.
        tree_gt = cKDTree(gt)
        dist_pred_to_gt, _ = tree_gt.query(pred, k=1)
        f_a = compute_fscore_auc(
            dist_pred_to_gt,
            dist_gt_to_pred,
            fscore_auc_max_threshold,
        )
        if f_a is not None:
            result["f_a"] = f_a
    return result
