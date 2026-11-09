"""Tests for gt.input_rgb: the degraded input as an RGB evaluation reference."""

from __future__ import annotations

import contextlib
import json
import warnings

import numpy as np
import pytest

import euler_eval.evaluate as eval_mod
import euler_eval.metric_sets as metric_sets_mod
from euler_eval import cli, data
from euler_eval.cli import _RGB_EVAL_DESCRIPTIONS, _rgb_eval_axes
from euler_eval.data import align_by_upscaling, classify_upscaling_alignment
from euler_eval.metric_sets import (
    DEHAZING_METRIC_SET,
    InputReferenceRGBMetric,
    MetricOutput,
    MetricSet,
    NoReferenceRGBMetric,
)
from euler_eval.metrics import load_hautiere
from tests.test_data_builders import _assert_modality, _install_captured_dataset
from tests.test_protocol import _index

# ---------------------------------------------------------------------------
# Dataset construction and alignment
# ---------------------------------------------------------------------------


def test_rgb_builder_pairs_the_input_with_the_prediction(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_rgb_eval_dataset(
        gt_rgb_path=None,
        pred_rgb_path="/predictions/dehazed",
        input_rgb_path="/datasets/hazy",
        input_rgb_split="test",
    )

    assert set(dataset.modalities) == {"input", "pred"}
    _assert_modality(dataset.modalities["input"], key="rgb", split="test")
    _assert_modality(dataset.modalities["pred"], key="rgb", used_as="output")


def test_rgb_builder_needs_a_reference(monkeypatch):
    _install_captured_dataset(monkeypatch)

    with pytest.raises(ValueError, match="gt.rgb or gt.input_rgb"):
        data.build_rgb_eval_dataset(gt_rgb_path=None, pred_rgb_path="/predictions")


@pytest.mark.parametrize(
    "pred_shape, input_shape, method",
    [
        ((8, 12), (8, 12), "none"),
        ((8, 12), (4, 6), "upscale_input"),
        ((4, 6), (8, 12), "upscale_prediction"),
        ((6, 4), (4, 6), "upscale_input"),  # equal pixel counts: pred's grid
    ],
)
def test_the_smaller_image_is_upscaled(pred_shape, input_shape, method):
    pred = np.random.default_rng(0).uniform(size=(*pred_shape, 3)).astype(np.float32)
    input_image = np.random.default_rng(1).uniform(size=(*input_shape, 3))
    input_image = input_image.astype(np.float32)

    assert classify_upscaling_alignment(pred.shape, input_image.shape) == method
    with pytest.warns(UserWarning) if pred_shape == (6, 4) else _no_warning():
        aligned_pred, aligned_input = align_by_upscaling(pred, input_image)

    larger = pred if method != "upscale_prediction" else input_image
    assert aligned_pred.shape == aligned_input.shape == larger.shape
    if method == "none":
        assert aligned_pred is pred and aligned_input is input_image
    elif method == "upscale_input":
        assert aligned_pred is pred
    else:
        assert aligned_input is input_image
    assert 0.0 <= aligned_pred.min() and aligned_pred.max() <= 1.0


@contextlib.contextmanager
def _no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class _Dataset:
    def __init__(self, samples, modalities):
        self._samples = samples
        self._modalities = modalities

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]

    def modality_paths(self):
        return {name: name for name in self._modalities}


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


class _Forbidden:
    def __init__(self, *args, **kwargs):
        raise AssertionError("full-reference metric initialised without gt.rgb")


def _samples(*, gt=True, input_rgb=True, pred_shape=(16, 16), input_shape=(16, 16)):
    samples = []
    for index, value in enumerate((0.25, 0.75), start=1):
        sample = {
            "id": f"{index:05d}",
            "full_id": f"/scene/{index:05d}",
            "pred": np.full((*pred_shape, 3), value, dtype=np.float32),
        }
        if gt:
            sample["gt"] = np.full((*pred_shape, 3), value, dtype=np.float32)
        if input_rgb:
            sample["input"] = np.full((*input_shape, 3), 0.5, dtype=np.float32)
        samples.append(sample)
    return samples


@pytest.fixture
def domain(monkeypatch):
    """A dehazing set with one no-reference and one input-referenced metric."""
    calls = []

    def load(device):
        def score(pred, input_image):
            calls.append((device, pred.shape, input_image.shape))
            return {
                "gain": float(np.mean(pred) / np.mean(input_image)),
                "loss": float(np.mean(input_image) - np.mean(pred)),
            }

        return score

    metric_set = MetricSet(
        name="dehazing",
        category="dehazing",
        rgb_no_reference=(
            NoReferenceRGBMetric(
                name="brightness",
                display_name="Brightness",
                is_higher_better=True,
                compute=lambda image: float(np.mean(image)),
            ),
        ),
        rgb_input_reference=(
            InputReferenceRGBMetric(
                name="pair",
                outputs=(
                    MetricOutput("gain", "Gain", True),
                    MetricOutput("loss", "Loss", False),
                ),
                load=load,
            ),
        ),
    )
    monkeypatch.setitem(metric_sets_mod.DOMAIN_METRIC_SETS, "dehazing", metric_set)
    monkeypatch.setattr(eval_mod, "tqdm", lambda iterable, *args, **kwargs: iterable)
    return calls


def test_input_only_evaluation_skips_full_reference_metrics(monkeypatch, domain):
    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _Forbidden)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _Forbidden)
    monkeypatch.setattr(eval_mod, "GPUImageMetricsBatcher", _Forbidden)

    result = eval_mod.evaluate_rgb_samples(
        _Dataset(_samples(gt=False), ("input", "pred")),
        device="cpu",
        domains=("dehazing",),
    )

    assert result["rgb"] == {
        "dehazing": pytest.approx({"brightness": 0.5, "gain": 1.0, "loss": 0.0})
    }
    assert result["rgb_benchmark"] is None
    files = result["per_file_metrics"]["children"]["scene"]["files"]
    assert files[0]["metrics"]["rgb"] == {
        "dehazing": pytest.approx({"brightness": 0.25, "gain": 0.5, "loss": 0.25})
    }
    assert result["spatial_info"]["gt_dimensions"] is None
    assert result["spatial_info"]["input_alignment"] == "none"
    assert [call[0] for call in domain] == ["cpu", "cpu"]


def test_both_references_compute_every_metric(monkeypatch, domain):
    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _DummyFID)

    result = eval_mod.evaluate_rgb_samples(
        _Dataset(_samples(), ("gt", "input", "pred")),
        device="cpu",
        domains=("dehazing",),
    )

    assert {"image_quality", "edge_f1", "dehazing"} <= set(result["rgb"])
    assert set(result["rgb"]["dehazing"]) == {"brightness", "gain", "loss"}


def test_without_input_the_input_referenced_metrics_are_absent(
    monkeypatch, capsys, domain
):
    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _DummyFID)

    result = eval_mod.evaluate_rgb_samples(
        _Dataset(_samples(input_rgb=False), ("gt", "pred")),
        device="cpu",
        domains=("dehazing",),
    )

    assert result["rgb"]["dehazing"] == pytest.approx({"brightness": 0.5})
    assert domain == []
    assert "Gain, Loss need gt.input_rgb" in capsys.readouterr().out


def test_input_referenced_metrics_see_the_upscaled_pair(monkeypatch, domain):
    result = eval_mod.evaluate_rgb_samples(
        _Dataset(_samples(gt=False, pred_shape=(8, 8)), ("input", "pred")),
        device="cpu",
        domains=("dehazing",),
    )

    assert {call[1:] for call in domain} == {((16, 16, 3), (16, 16, 3))}
    assert result["spatial_info"]["input_alignment"] == "upscale_prediction"
    assert result["spatial_info"]["evaluated_dimensions"] == {"height": 8, "width": 8}


def test_evaluation_needs_a_reference(domain):
    with pytest.raises(ValueError, match="needs a 'gt'"):
        eval_mod.evaluate_rgb_samples(
            _Dataset(_samples(gt=False, input_rgb=False), ("pred",)),
            device="cpu",
        )


# ---------------------------------------------------------------------------
# Metric registry and CLI
# ---------------------------------------------------------------------------


def test_dehazing_scores_hautiere_against_the_input():
    (metric,) = DEHAZING_METRIC_SET.rgb_input_reference
    assert metric.load is load_hautiere
    assert [output.name for output in metric.outputs] == [
        "hautiere_e",
        "hautiere_r",
        "hautiere_sigma",
    ]
    assert _RGB_EVAL_DESCRIPTIONS["hautiere_e"].is_higher_better is True
    assert _RGB_EVAL_DESCRIPTIONS["hautiere_r"].is_higher_better is True
    sigma = _RGB_EVAL_DESCRIPTIONS["hautiere_sigma"]
    assert sigma.is_higher_better is False
    assert (sigma.scale, sigma.min_value, sigma.max_value) == ("percentage", 0, 100)


def test_rgb_axes_without_gt_rgb_declare_only_domain_categories():
    axes = _rgb_eval_axes(domains=("dehazing",), full_reference=False)

    assert axes["category"].values == ("dehazing",)
    assert "image_quality" in _rgb_eval_axes()["category"].values


def test_gt_input_rgb_alone_is_a_valid_gt_section(tmp_path):
    hazy = tmp_path / "hazy"
    hazy.mkdir()

    cli.validate_gt_config({"input_rgb": {"path": str(hazy)}})

    with pytest.raises(ValueError, match="gt.input_rgb.path does not exist"):
        cli.validate_gt_config({"input_rgb": {"path": str(tmp_path / "missing")}})
    with pytest.raises(ValueError, match="'input_rgb.path'"):
        cli.validate_gt_config({})


def test_input_rgb_is_not_an_experiment_modality():
    with pytest.raises(ValueError, match="'input_rgb' is a GT input"):
        cli.validate_experiments_config([{"name": "hazy", "modalities": ["input_rgb"]}])


def test_cli_evaluates_rgb_against_the_input_alone(monkeypatch, tmp_path):
    hazy = _index(tmp_path / "hazy", name="hazy_inputs", key="rgb", scope="rgb")
    calibration = _index(
        tmp_path / "calib", name="calib", key="calibration", scope="calibration"
    )
    depth = tmp_path / "depth"
    depth.mkdir()
    pred = tmp_path / "pred"
    pred.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "gt": {
                    "input_rgb": {"path": str(hazy)},
                    "depth": {"path": str(depth)},
                    "calibration": {"path": str(calibration)},
                },
                "datasets": [{"name": "model", "rgb": {"path": str(pred)}}],
            }
        )
    )

    built = {}

    class _Stub(list):
        def modality_paths(self):
            return {"input": "input", "pred": "pred"}

    def build_rgb(**kwargs):
        built.update(kwargs)
        return _Stub([object()] * 2)

    evaluated = {}

    def evaluate_rgb(**kwargs):
        evaluated.update(kwargs)
        return {
            "rgb": {"dehazing": {"hautiere_e": 0.5}},
            "dataset_info": {"num_pairs": 2},
            "spatial_info": {
                "method": "none",
                "evaluated_dimensions": {"height": 4, "width": 6},
                "input_dimensions": {"height": 8, "width": 12},
                "input_alignment": "upscale_prediction",
            },
            "per_file_metrics": {},
        }

    monkeypatch.setattr(cli, "build_rgb_eval_dataset", build_rgb)
    monkeypatch.setattr(cli, "get_rgb_metadata", lambda _ds: {"rgb_range": [0, 1]})
    monkeypatch.setattr(cli, "evaluate_rgb_samples", evaluate_rgb)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "euler-eval",
            str(config_path),
            "--device",
            "cpu",
            "--no-sanity-check",
            "--skip-depth",
        ],
    )
    cli.main()

    assert built["gt_rgb_path"] is None
    assert built["input_rgb_path"] == str(hazy)
    assert built["gt_depth_path"] is None
    assert built["calibration_path"] is None
    assert built["segmentation_path"] is None

    saved = json.loads((pred / ".eval" / "eval.json").read_text())
    assert saved["rgb"]["eval"] == {"dehazing": {"hautiere_e": 0.5}}
    assert saved["metricSet"]["axes"]["category"]["values"] == []
    parameters = saved["protocol"]["parameters"]
    assert saved["protocol"]["schemaVersion"] == "1.3"
    assert parameters["reference.input_rgb.dataset"] == "hazy_inputs"
    assert parameters["reference.gt.dataset"] is None
    assert parameters["reference.calibration.dataset"] is None
    assert parameters["method.input_alignment"] == "upscale_prediction"
    assert parameters["method.fid_backend"] is None
    assert parameters["method.spatial_alignment"] is None
    assert saved["meta"]["input"]["dimensions"] == {"height": 8, "width": 12}
