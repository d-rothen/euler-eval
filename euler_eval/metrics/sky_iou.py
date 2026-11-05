"""Sky segmentation IoU between a predicted and a ground-truth sky mask.

Per sample, with prediction sky ``P``, GT sky ``G`` and evaluation domain
``D``::

    I = |P ∩ G ∩ D|    U = |(P ∪ G) ∩ D|    T_p = |P ∩ D|    T_g = |G ∩ D|

IoU is ``I / U``, precision ``I / T_p`` and recall ``I / T_g``. Each is
undefined when its denominator is zero. Dataset reductions mirror
:mod:`euler_eval.metrics.depth_standard`: ``image_mean`` and ``image_median``
reduce the defined per-image values, and ``pixel_pool`` divides the summed
counts. A sample whose value is undefined is left out of the image reducers
but still adds its (zero) counts to the pooled sums.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

SKY_IOU_METRIC_KEYS = ("iou", "precision", "recall")

# Counts kept per sample; ``pixels`` is |D|, used for sky-fraction provenance.
SKY_IOU_COUNT_KEYS = ("intersection", "union", "pred_sky", "gt_sky", "pixels")

# Denominator of each metric, as a key of the counts dict.
_DENOMINATORS = {"iou": "union", "precision": "pred_sky", "recall": "gt_sky"}


def _as_bool_mask(mask, name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D (H, W) mask, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def sky_iou_counts(
    pred_sky: np.ndarray,
    gt_sky: np.ndarray,
    domain: Optional[np.ndarray] = None,
) -> dict[str, int]:
    """Count the pixels that one sample adds to the sky IoU.

    Args:
        pred_sky: Predicted sky ``(H, W)``; True = sky.
        gt_sky: Ground-truth sky ``(H, W)`` on the same grid; True = sky.
        domain: Optional ``(H, W)`` bool mask restricting the evaluated
            pixels. ``None`` evaluates the full grid.

    Returns:
        ``intersection`` (I), ``union`` (U), ``pred_sky`` (T_p),
        ``gt_sky`` (T_g) and ``pixels`` (|D|) as Python ints.
    """
    pred = _as_bool_mask(pred_sky, "pred_sky")
    gt = _as_bool_mask(gt_sky, "gt_sky")
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred_sky {pred.shape} and gt_sky {gt.shape} must share a grid"
        )
    if domain is None:
        pixels = pred.size
    else:
        dom = _as_bool_mask(domain, "domain")
        if dom.shape != pred.shape:
            raise ValueError(
                f"domain {dom.shape} must match the sky masks {pred.shape}"
            )
        pred = pred & dom
        gt = gt & dom
        pixels = int(np.count_nonzero(dom))
    return {
        "intersection": int(np.count_nonzero(pred & gt)),
        "union": int(np.count_nonzero(pred | gt)),
        "pred_sky": int(np.count_nonzero(pred)),
        "gt_sky": int(np.count_nonzero(gt)),
        "pixels": int(pixels),
    }


def sky_iou_from_counts(counts: dict) -> dict[str, Optional[float]]:
    """Return IoU, precision and recall for *counts*; ``None`` when undefined."""
    intersection = counts["intersection"]
    return {
        key: (
            float(intersection / counts[denominator])
            if counts[denominator] > 0
            else None
        )
        for key, denominator in _DENOMINATORS.items()
    }


def implicit_sky_from_depth(depth: np.ndarray, threshold: float) -> np.ndarray:
    """Treat depth at or beyond *threshold* as predicted sky.

    The test is ``depth >= threshold``, so a regressor that clamps exactly at
    the cap counts as sky. ``+inf`` is sky; NaN, ``-inf`` and values ``<= 0``
    are not (the threshold is positive).

    Args:
        depth: Predicted metric depth ``(H, W)`` in meters.
        threshold: Finite positive depth in meters.

    Returns:
        ``(H, W)`` bool mask, True = sky.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a finite positive value")
    arr = np.asarray(depth)
    if arr.ndim != 2:
        raise ValueError(f"depth must be a 2D (H, W) map, got shape {arr.shape}")
    with np.errstate(invalid="ignore"):
        return np.asarray(arr >= threshold, dtype=bool)


def init_sky_iou_store() -> dict:
    """Create a reducer store for one sky IoU variant."""
    return {
        "per_image": {key: [] for key in SKY_IOU_METRIC_KEYS},
        "pool": {key: 0 for key in SKY_IOU_COUNT_KEYS},
        "samples": 0,
        "samples_undefined_iou": 0,
    }


def append_sky_iou_store(store: dict, counts: dict) -> dict[str, Optional[float]]:
    """Add one sample's counts to *store* and return its per-image values."""
    values = sky_iou_from_counts(counts)
    for key, value in values.items():
        if value is not None:
            store["per_image"][key].append(value)
    pool = store["pool"]
    for key in SKY_IOU_COUNT_KEYS:
        pool[key] += int(counts[key])
    store["samples"] += 1
    if values["iou"] is None:
        store["samples_undefined_iou"] += 1
    return values


def summarize_sky_iou_store(store: dict) -> dict[str, dict[str, float]]:
    """Reduce *store* to ``pixel_pool``, ``image_mean`` and ``image_median``.

    Undefined reductions are NaN, which the CLI drops from ``eval.json``.
    """

    def _reduce_per_image(reducer) -> dict[str, float]:
        return {
            key: (
                float(reducer(np.asarray(values, dtype=np.float64)))
                if values
                else float("nan")
            )
            for key, values in store["per_image"].items()
        }

    pooled = sky_iou_from_counts(store["pool"])
    return {
        "pixel_pool": {
            key: float("nan") if value is None else value
            for key, value in pooled.items()
        },
        "image_mean": _reduce_per_image(np.mean),
        "image_median": _reduce_per_image(np.median),
    }
