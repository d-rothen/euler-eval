"""RGB depth bins: an empty bin is an outcome, not a perfect score."""

import math

import numpy as np
import pytest

from euler_eval.metrics import (
    aggregate_depth_binned_errors,
    compute_depth_binned_photometric_error,
)


def test_an_empty_fixed_depth_bin_has_no_value():
    pred = np.zeros((4, 4, 3), dtype=np.float32)
    gt = np.full((4, 4, 3), 0.5, dtype=np.float32)
    depth = np.full((4, 4), 3.0, dtype=np.float32)  # every pixel in (1, 5]

    result = compute_depth_binned_photometric_error(pred, gt, depth)

    for metric in ("mae", "mse"):
        assert math.isnan(result[metric]["near"])
        assert math.isnan(result[metric]["far"])
    assert result["mae"]["mid"] == pytest.approx(0.5)
    assert result["mae"]["all"] == pytest.approx(0.5)


def test_without_valid_depth_every_fixed_bin_is_empty():
    pred = np.zeros((2, 2, 3), dtype=np.float32)
    result = compute_depth_binned_photometric_error(pred, pred, np.zeros((2, 2)))
    assert all(math.isnan(value) for value in result["mae"].values())


def test_empty_fixed_bins_stay_out_of_the_dataset_mean():
    pred = np.zeros((2, 2, 3), dtype=np.float32)
    near_only = compute_depth_binned_photometric_error(
        pred, np.full((2, 2, 3), 0.2, np.float32), np.full((2, 2), 0.5)
    )
    far_only = compute_depth_binned_photometric_error(
        pred, np.full((2, 2, 3), 0.6, np.float32), np.full((2, 2), 9.0)
    )

    aggregated = aggregate_depth_binned_errors([near_only, far_only])

    # Each bin averages the images that have it (a 0.0 would halve them).
    assert aggregated["mae"]["near"] == pytest.approx(0.2)
    assert aggregated["mae"]["far"] == pytest.approx(0.6)
    assert math.isnan(aggregated["mae"]["mid"])
    assert aggregated["mae"]["all"] == pytest.approx(0.4)
