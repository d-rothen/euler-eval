"""Euclidean 3D agreement metrics for the ``points_3d`` modality.

Operates on per-pixel 3D point maps.  Given the per-point Euclidean error
``e_i = ‖pred_i − gt_i‖`` and the relative error ``rel_i = e_i / ‖gt_i‖``
(both produced by :func:`euler_eval.data.decompose_point_errors`), this module
provides per-image summaries and the threshold-based 3D accuracy
("δ-accuracy") scores.

These are the 3D generalization of the depth AbsRel / RMSE / δ metrics.
"""

from typing import Optional

import numpy as np

# Absolute distance thresholds (metres) for 3D accuracy.
ABS_THRESHOLDS = (0.05, 0.1, 0.25, 0.5, 1.0)
# Relative distance thresholds (fraction of GT range) for 3D accuracy.
REL_THRESHOLDS = (0.05, 0.1, 0.25)


def threshold_key(prefix: str, tau: float) -> str:
    """Build a JSON-safe metric key, e.g. ``("acc", 0.1) -> "acc_0_1"``."""
    return f"{prefix}_" + ("%g" % tau).replace(".", "_")


def compute_point_error_metrics(
    euclidean: np.ndarray,
    relative: np.ndarray,
) -> Optional[dict]:
    """Per-image Euclidean 3D agreement metrics.

    Args:
        euclidean: ``(N,)`` per-point Euclidean errors in metres.
        relative: ``(N,)`` per-point relative errors (``e / ‖gt‖``).

    Returns:
        Dict of scalar metrics, or ``None`` if there are no valid points.
    """
    e = np.asarray(euclidean, dtype=np.float64).reshape(-1)
    rel = np.asarray(relative, dtype=np.float64).reshape(-1)
    if e.size == 0:
        return None

    metrics = {
        "mae3d": float(np.mean(e)),
        "rmse3d": float(np.sqrt(np.mean(e**2))),
        "median3d": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "rel_median": float(np.median(rel)),
        "rel_p90": float(np.percentile(rel, 90)),
    }
    for tau in ABS_THRESHOLDS:
        metrics[threshold_key("acc", tau)] = float(np.mean(e < tau) * 100.0)
    for tau in REL_THRESHOLDS:
        metrics[threshold_key("acc_rel", tau)] = float(np.mean(rel < tau) * 100.0)
    return metrics
