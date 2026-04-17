"""Standard monocular depth metrics with explicit dataset reductions."""

from __future__ import annotations

import numpy as np
from typing import Optional


STANDARD_DEPTH_METRIC_KEYS = (
    "absrel",
    "sqrel",
    "mae",
    "rmse",
    "rmse_log",
    "log10",
    "silog",
    "delta1",
    "delta2",
    "delta3",
)


def _resolve_valid_mask(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)
    return valid_mask


def _nan_metrics() -> dict[str, float]:
    return {key: float("nan") for key in STANDARD_DEPTH_METRIC_KEYS}


def _empty_pool_stats() -> dict[str, float]:
    return {
        "count": 0,
        "sum_absrel": 0.0,
        "sum_sqrel": 0.0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_log10_abs": 0.0,
        "sum_log_diff": 0.0,
        "sum_log_diff_sq": 0.0,
        "delta1_hits": 0,
        "delta2_hits": 0,
        "delta3_hits": 0,
    }


def compute_standard_depth_metrics(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute standard monocular depth metrics for one image pair.

    Returns both per-image scalar metrics and pooled sufficient statistics so
    dataset reducers can report image-wise and global pixel-pool variants
    without re-scanning the pixels.
    """
    valid_mask = _resolve_valid_mask(depth_pred, depth_gt, valid_mask)
    if not valid_mask.any():
        return _nan_metrics(), _empty_pool_stats()

    pred_valid = depth_pred[valid_mask].astype(np.float64, copy=False)
    gt_valid = depth_gt[valid_mask].astype(np.float64, copy=False)

    diff = pred_valid - gt_valid
    abs_diff = np.abs(diff)
    sq_diff = diff * diff
    absrel = abs_diff / gt_valid
    sqrel = sq_diff / gt_valid

    log_pred = np.log(pred_valid)
    log_gt = np.log(gt_valid)
    log_diff = log_pred - log_gt
    log_diff_sq = log_diff * log_diff
    abs_log10_diff = np.abs(np.log10(pred_valid) - np.log10(gt_valid))

    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta1_hits = int(np.sum(ratio < 1.25))
    delta2_hits = int(np.sum(ratio < (1.25**2)))
    delta3_hits = int(np.sum(ratio < (1.25**3)))

    mean_log_diff = float(np.mean(log_diff))
    mean_log_diff_sq = float(np.mean(log_diff_sq))
    silog_sq = max(mean_log_diff_sq - mean_log_diff**2, 0.0)

    metrics = {
        "absrel": float(np.mean(absrel)),
        "sqrel": float(np.mean(sqrel)),
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(sq_diff))),
        "rmse_log": float(np.sqrt(np.mean(log_diff_sq))),
        "log10": float(np.mean(abs_log10_diff)),
        "silog": float(np.sqrt(silog_sq)),
        "delta1": float(delta1_hits / pred_valid.size),
        "delta2": float(delta2_hits / pred_valid.size),
        "delta3": float(delta3_hits / pred_valid.size),
    }
    pool_stats = {
        "count": int(pred_valid.size),
        "sum_absrel": float(np.sum(absrel)),
        "sum_sqrel": float(np.sum(sqrel)),
        "sum_abs": float(np.sum(abs_diff)),
        "sum_sq": float(np.sum(sq_diff)),
        "sum_log10_abs": float(np.sum(abs_log10_diff)),
        "sum_log_diff": float(np.sum(log_diff)),
        "sum_log_diff_sq": float(np.sum(log_diff_sq)),
        "delta1_hits": delta1_hits,
        "delta2_hits": delta2_hits,
        "delta3_hits": delta3_hits,
    }
    return metrics, pool_stats


def init_standard_depth_store() -> dict:
    """Create a reducer store for dataset-level standard depth summaries."""
    return {
        "per_image": {key: [] for key in STANDARD_DEPTH_METRIC_KEYS},
        "pool": _empty_pool_stats(),
    }


def append_standard_depth_metrics(
    store: dict,
    metrics: dict[str, float],
    pool_stats: dict[str, float],
) -> None:
    """Update a reducer store with one image's standard metrics."""
    if int(pool_stats["count"]) <= 0:
        return

    for key in STANDARD_DEPTH_METRIC_KEYS:
        value = float(metrics[key])
        if np.isfinite(value):
            store["per_image"][key].append(value)

    pool = store["pool"]
    pool["count"] += int(pool_stats["count"])
    pool["sum_absrel"] += float(pool_stats["sum_absrel"])
    pool["sum_sqrel"] += float(pool_stats["sum_sqrel"])
    pool["sum_abs"] += float(pool_stats["sum_abs"])
    pool["sum_sq"] += float(pool_stats["sum_sq"])
    pool["sum_log10_abs"] += float(pool_stats["sum_log10_abs"])
    pool["sum_log_diff"] += float(pool_stats["sum_log_diff"])
    pool["sum_log_diff_sq"] += float(pool_stats["sum_log_diff_sq"])
    pool["delta1_hits"] += int(pool_stats["delta1_hits"])
    pool["delta2_hits"] += int(pool_stats["delta2_hits"])
    pool["delta3_hits"] += int(pool_stats["delta3_hits"])


def summarize_standard_depth_store(store: dict) -> dict[str, dict[str, float]]:
    """Build image-wise and pooled summaries from a reducer store."""

    def _reduce_per_image(reducer) -> dict[str, float]:
        summary = {}
        for key in STANDARD_DEPTH_METRIC_KEYS:
            values = store["per_image"][key]
            summary[key] = (
                float(reducer(np.asarray(values, dtype=np.float64)))
                if values
                else float("nan")
            )
        return summary

    pool = store["pool"]
    count = int(pool["count"])
    if count <= 0:
        pixel_pool = _nan_metrics()
    else:
        mean_log_diff = pool["sum_log_diff"] / count
        mean_log_diff_sq = pool["sum_log_diff_sq"] / count
        silog_sq = max(mean_log_diff_sq - mean_log_diff**2, 0.0)
        pixel_pool = {
            "absrel": float(pool["sum_absrel"] / count),
            "sqrel": float(pool["sum_sqrel"] / count),
            "mae": float(pool["sum_abs"] / count),
            "rmse": float(np.sqrt(pool["sum_sq"] / count)),
            "rmse_log": float(np.sqrt(pool["sum_log_diff_sq"] / count)),
            "log10": float(pool["sum_log10_abs"] / count),
            "silog": float(np.sqrt(silog_sq)),
            "delta1": float(pool["delta1_hits"] / count),
            "delta2": float(pool["delta2_hits"] / count),
            "delta3": float(pool["delta3_hits"] / count),
        }

    return {
        "image_mean": _reduce_per_image(np.mean),
        "image_median": _reduce_per_image(np.median),
        "pixel_pool": pixel_pool,
    }
