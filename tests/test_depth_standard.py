"""Tests for standard monocular depth metrics and reducers."""

import numpy as np

from euler_eval.metrics.depth_standard import (
    STANDARD_DEPTH_METRIC_KEYS,
    append_standard_depth_metrics,
    compute_standard_depth_metrics,
    init_standard_depth_store,
    summarize_standard_depth_store,
)


def test_compute_standard_depth_metrics_matches_reference_formulas():
    pred = np.array([[1.0, 1.4, 3.0]], dtype=np.float32)
    gt = np.array([[1.0, 1.0, 2.0]], dtype=np.float32)

    metrics, pool = compute_standard_depth_metrics(pred, gt)

    pred_valid = pred.reshape(-1).astype(np.float64)
    gt_valid = gt.reshape(-1).astype(np.float64)
    diff = pred_valid - gt_valid
    abs_diff = np.abs(diff)
    sq_diff = diff * diff
    log_diff = np.log(pred_valid) - np.log(gt_valid)
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)

    assert np.isclose(metrics["absrel"], np.mean(abs_diff / gt_valid))
    assert np.isclose(metrics["sqrel"], np.mean(sq_diff / gt_valid))
    assert np.isclose(metrics["mae"], np.mean(abs_diff))
    assert np.isclose(metrics["rmse"], np.sqrt(np.mean(sq_diff)))
    assert np.isclose(metrics["rmse_log"], np.sqrt(np.mean(log_diff**2)))
    assert np.isclose(
        metrics["log10"],
        np.mean(np.abs(np.log10(pred_valid) - np.log10(gt_valid))),
    )
    assert np.isclose(
        metrics["silog"],
        np.sqrt(np.mean(log_diff**2) - np.mean(log_diff) ** 2),
    )
    assert np.isclose(metrics["delta1"], np.mean(ratio < 1.25))
    assert np.isclose(metrics["delta2"], np.mean(ratio < (1.25**2)))
    assert np.isclose(metrics["delta3"], np.mean(ratio < (1.25**3)))

    assert pool["count"] == 3
    assert pool["delta1_hits"] == 1
    assert pool["delta2_hits"] == 3
    assert pool["delta3_hits"] == 3


def test_standard_depth_store_reports_image_and_pixel_reductions():
    samples = [
        (
            np.array([[1.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 2.0]], dtype=np.float32),
        ),
        (
            np.array([[2.0, 4.0, 8.0]], dtype=np.float32),
            np.array([[1.0, 2.0, 4.0]], dtype=np.float32),
        ),
    ]

    store = init_standard_depth_store()
    per_image = []
    pooled_counts = []

    for pred, gt in samples:
        metrics, pool = compute_standard_depth_metrics(pred, gt)
        append_standard_depth_metrics(store, metrics, pool)
        per_image.append(metrics)
        pooled_counts.append(pool["count"])

    summary = summarize_standard_depth_store(store)

    for key in STANDARD_DEPTH_METRIC_KEYS:
        values = np.array([m[key] for m in per_image], dtype=np.float64)
        assert np.isclose(summary["image_mean"][key], np.mean(values))
        assert np.isclose(summary["image_median"][key], np.median(values))

    pooled_absrel_num = sum(
        m["absrel"] * c for m, c in zip(per_image, pooled_counts)
    )
    pooled_delta1_num = sum(
        m["delta1"] * c for m, c in zip(per_image, pooled_counts)
    )
    pooled_total = sum(pooled_counts)

    assert np.isclose(summary["pixel_pool"]["absrel"], pooled_absrel_num / pooled_total)
    assert np.isclose(summary["pixel_pool"]["delta1"], pooled_delta1_num / pooled_total)
