"""Tests for composable domain metric sets and dehazing metrics."""

from __future__ import annotations

import numpy as np
import pytest

import euler_eval.evaluate as eval_mod
import euler_eval.metric_sets as metric_sets_mod
from euler_eval.cli import _RGB_EVAL_DESCRIPTIONS, _rgb_eval_axes
from euler_eval.metric_sets import (
    MetricSet,
    NoReferenceRGBMetric,
    available_domains,
    resolve_metric_sets,
)
from euler_eval.metrics import compute_fade, compute_niqe


def _pattern_image(height: int, width: int) -> np.ndarray:
    rows, columns = np.indices((height, width))
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (rows * 3 + columns * 5) % 256
    image[..., 1] = (rows * 7 + columns * 11 + 23) % 256
    image[..., 2] = (rows * 13 + columns * 17 + 101) % 256
    return image.astype(np.float32) / 255.0


def test_metric_set_resolution_is_additive_and_deduplicated():
    assert available_domains() == ("dehazing",)
    assert [item.name for item in resolve_metric_sets()] == ["core"]
    assert [item.name for item in resolve_metric_sets(("dehazing", "dehazing"))] == [
        "core",
        "dehazing",
    ]


def test_unknown_domain_is_rejected():
    with pytest.raises(ValueError, match="Unknown evaluation domain 'underwater'"):
        resolve_metric_sets(("underwater",))


def test_dehazing_domain_extends_rgb_namespace_and_descriptions():
    default_categories = _rgb_eval_axes()["category"].values
    dehazing_categories = _rgb_eval_axes(domains=("dehazing",))["category"].values

    assert "dehazing" not in default_categories
    assert "dehazing" in dehazing_categories
    assert _RGB_EVAL_DESCRIPTIONS["niqe"].is_higher_better is False
    assert _RGB_EVAL_DESCRIPTIONS["fade"].is_higher_better is False


def test_fade_matches_reference_aligned_regression_fixture():
    rows, columns = np.indices((16, 16))
    image = np.empty((16, 16, 3), dtype=np.uint8)
    image[..., 0] = (rows * 17 + columns * 11) % 256
    image[..., 1] = (rows * 7 + columns * 19 + 23) % 256
    image[..., 2] = (rows * 29 + columns * 3 + 101) % 256

    score = compute_fade(image.astype(np.float32) / 255.0)

    assert score == pytest.approx(0.1813003412312952, abs=1e-12)


def test_fade_increases_when_contrast_and_saturation_are_suppressed():
    clear = _pattern_image(192, 192)
    hazy = clear * 0.25 + 0.75 * 0.8

    assert compute_fade(hazy) > compute_fade(clear)


def test_niqe_is_finite_and_deterministic_for_multiblock_image():
    image = _pattern_image(192, 192)

    assert compute_niqe(image) == pytest.approx(34.00505456054468, rel=1e-9)


@pytest.mark.parametrize("metric", [compute_niqe, compute_fade])
def test_no_reference_metrics_reject_values_outside_unit_range(metric):
    image = _pattern_image(192, 192)
    image[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        metric(image)


def test_niqe_rejects_images_smaller_than_reference_patch():
    with pytest.raises(ValueError, match="at least 96x96"):
        compute_niqe(np.zeros((95, 96, 3), dtype=np.float32))


class _DummyRGBDataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]

    def modality_paths(self):
        return {"gt": "gt", "pred": "pred"}


class _DummyLPIPS:
    def __init__(self, device="cpu"):
        self.device = device

    def compute(self, prediction, ground_truth):
        return float(np.mean(np.abs(prediction - ground_truth)))


class _DummyFID:
    def __init__(self, device="cpu"):
        self.device = device

    def compute_rgb_fid(self, *args, **kwargs):
        return 0.0


def test_rgb_evaluator_adds_domain_metrics_to_aggregate_and_per_file(
    monkeypatch,
):
    dummy_set = MetricSet(
        name="dehazing",
        category="dehazing",
        rgb_no_reference=(
            NoReferenceRGBMetric(
                name="niqe",
                display_name="NIQE",
                compute=lambda image: float(np.mean(image)),
                is_higher_better=False,
            ),
            NoReferenceRGBMetric(
                name="fade",
                display_name="FADE",
                compute=lambda image: float(1.0 - np.mean(image)),
                is_higher_better=False,
            ),
        ),
    )
    monkeypatch.setitem(metric_sets_mod.DOMAIN_METRIC_SETS, "dehazing", dummy_set)
    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _DummyFID)
    monkeypatch.setattr(eval_mod, "tqdm", lambda iterable, *args, **kwargs: iterable)

    samples = []
    for index, value in enumerate((0.25, 0.75), start=1):
        image = np.full((16, 16, 3), value, dtype=np.float32)
        samples.append(
            {
                "id": f"{index:05d}",
                "full_id": f"/scene/{index:05d}",
                "gt": image.copy(),
                "pred": image,
                # Everything is sky. The domain metrics must still see the
                # unmasked prediction rather than an all-zero fill image.
                "segmentation": np.ones((16, 16), dtype=bool),
            }
        )

    result = eval_mod.evaluate_rgb_samples(
        _DummyRGBDataset(samples),
        device="cpu",
        sky_mask_enabled=True,
        domains=("dehazing",),
    )

    assert result["rgb"]["dehazing"] == pytest.approx({"niqe": 0.5, "fade": 0.5})
    files = result["per_file_metrics"]["children"]["scene"]["files"]
    assert files[0]["metrics"]["rgb"]["dehazing"] == pytest.approx(
        {"niqe": 0.25, "fade": 0.75}
    )
    assert files[1]["metrics"]["rgb"]["dehazing"] == pytest.approx(
        {"niqe": 0.75, "fade": 0.25}
    )
