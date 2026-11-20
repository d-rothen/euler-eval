"""Depth-binned photometric error metrics for RGB images."""


import numpy as np

from .utils import get_depth_bins


def compute_depth_binned_mae(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    depth: np.ndarray,
    near_threshold: float = 1.0,
    far_threshold: float = 5.0,
) -> dict:
    """Compute MAE over depth bins (near/mid/far).

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        depth: Depth map in meters, shape (H, W).
        near_threshold: Depth threshold for near bin.
        far_threshold: Depth threshold for far bin.

    Returns:
        Dictionary with MAE for 'near', 'mid', 'far', and 'all' regions;
        ``NaN`` for a region without valid depth (no observation, not 0).
    """
    bins = get_depth_bins(depth, near_threshold, far_threshold)

    # Compute per-pixel absolute error
    abs_error = np.abs(img_pred - img_gt).mean(axis=-1)  # Average over RGB

    results = {}

    for bin_name, mask in bins.items():
        results[bin_name] = _masked_mean(abs_error, mask)

    # Overall MAE
    valid_mask = bins["near"] | bins["mid"] | bins["far"]
    results["all"] = _masked_mean(abs_error, valid_mask)

    return results


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of *values* under *mask*; ``NaN`` for an empty mask.

    An empty bin has no observation. It used to score ``0.0``, which entered
    the dataset mean as a perfect value (``rgb.depth_bins.policy`` ``v3``).
    """
    if mask.any():
        return float(np.mean(values[mask]))
    return float("nan")


def compute_depth_binned_mse(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    depth: np.ndarray,
    near_threshold: float = 1.0,
    far_threshold: float = 5.0,
) -> dict:
    """Compute MSE over depth bins (near/mid/far).

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        depth: Depth map in meters, shape (H, W).
        near_threshold: Depth threshold for near bin.
        far_threshold: Depth threshold for far bin.

    Returns:
        Dictionary with MSE for 'near', 'mid', 'far', and 'all' regions;
        ``NaN`` for a region without valid depth (no observation, not 0).
    """
    bins = get_depth_bins(depth, near_threshold, far_threshold)

    # Compute per-pixel squared error
    sq_error = ((img_pred - img_gt) ** 2).mean(axis=-1)  # Average over RGB

    results = {}

    for bin_name, mask in bins.items():
        results[bin_name] = _masked_mean(sq_error, mask)

    # Overall MSE
    valid_mask = bins["near"] | bins["mid"] | bins["far"]
    results["all"] = _masked_mean(sq_error, valid_mask)

    return results


def compute_depth_binned_photometric_error(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    depth: np.ndarray,
    near_threshold: float = 1.0,
    far_threshold: float = 5.0,
) -> dict:
    """Compute both MAE and MSE over depth bins.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        depth: Depth map in meters, shape (H, W).
        near_threshold: Depth threshold for near bin.
        far_threshold: Depth threshold for far bin.

    Returns:
        Dictionary with 'mae' and 'mse' sub-dictionaries.
    """
    return {
        "mae": compute_depth_binned_mae(
            img_pred, img_gt, depth, near_threshold, far_threshold
        ),
        "mse": compute_depth_binned_mse(
            img_pred, img_gt, depth, near_threshold, far_threshold
        ),
    }


def aggregate_depth_binned_errors(
    results: list[dict],
) -> dict:
    """Aggregate depth-binned errors from multiple image pairs.

    Args:
        results: List of per-image depth-binned error dictionaries.

    Returns:
        Aggregated dictionary with mean errors per bin.
    """
    aggregated = {
        "mae": {"near": [], "mid": [], "far": [], "all": []},
        "mse": {"near": [], "mid": [], "far": [], "all": []},
    }

    for r in results:
        for metric in ["mae", "mse"]:
            for bin_name in ["near", "mid", "far", "all"]:
                val = r[metric][bin_name]
                if np.isfinite(val):
                    aggregated[metric][bin_name].append(val)

    final = {"mae": {}, "mse": {}}
    for metric in ["mae", "mse"]:
        for bin_name in ["near", "mid", "far", "all"]:
            values = aggregated[metric][bin_name]
            if values:
                final[metric][bin_name] = float(np.mean(values))
            else:
                final[metric][bin_name] = float("nan")

    return final
