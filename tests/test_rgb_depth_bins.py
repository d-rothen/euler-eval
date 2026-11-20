"""RGB depth bins: an empty bin is an outcome, not a perfect score."""

import math

import numpy as np
import pytest

import euler_eval.evaluate as eval_mod
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


class _Dataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]

    def modality_paths(self):
        return {"gt": "gt", "pred": "pred", "gt_depth": "gt_depth"}


class _DummyLPIPS:
    def __init__(self, device="cpu"):
        pass

    def compute(self, prediction, ground_truth):
        return float(np.mean(np.abs(prediction - ground_truth)))


class _DummyFID:
    def __init__(self, device="cpu"):
        pass

    def compute_rgb_fid(self, *args, **kwargs):
        return 0.0


def test_rgb_benchmark_bins_skip_empty_bins_and_are_written_per_file(monkeypatch):
    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _DummyFID)
    monkeypatch.setattr(eval_mod, "tqdm", lambda iterable, *args, **kwargs: iterable)
    samples = []
    # Benchmark [1, 80]: near < 13.3 m, far >= 39.6 m.
    for index, (depth, error) in enumerate(((5.0, 0.2), (60.0, 0.6))):
        samples.append({
            "id": f"f{index}",
            "full_id": f"/scene/f{index}",
            "pred": np.zeros((8, 8, 3), dtype=np.float32),
            "gt": np.full((8, 8, 3), error, dtype=np.float32),
            "gt_depth": np.full((8, 8), depth, dtype=np.float32),
        })

    result = eval_mod.evaluate_rgb_samples(
        _Dataset(samples),
        depth_meta={"radial_depth": True, "sparse": False, "invalid_sentinel": None},
        device="cpu",
        benchmark_depth_range=(1.0, 80.0),
    )

    benchmark = result["rgb_benchmark"]
    assert benchmark["mae"]["near"] == pytest.approx(0.2)
    assert benchmark["mae"]["far"] == pytest.approx(0.6)
    assert math.isnan(benchmark["mae"]["mid"])
    assert benchmark["mae"]["all"] == pytest.approx(0.4)
    files = result["per_file_metrics"]["children"]["scene"]["files"]
    first = files[0]["metrics"]["rgb"]
    assert first["near"] == pytest.approx({"mae": 0.2, "mse": 0.04})
    assert first["all"] == pytest.approx({"mae": 0.2, "mse": 0.04})
    assert "mid" not in first and "far" not in first
    assert set(files[1]["metrics"]["rgb"]) >= {"all", "far"}
