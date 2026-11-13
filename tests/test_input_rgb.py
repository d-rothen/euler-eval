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


def _ramp(height: int, width: int) -> np.ndarray:
    """An RGB image that is linear in x and y, every pixel distinct."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([xx / width, yy / height, (xx + 2 * yy) / 200.0], axis=-1)


def _grid(size, affine) -> data.SourceGrid:
    return data.SourceGrid.from_value({"size": list(size), "affine": affine})


def test_a_cropped_prediction_meets_the_input_pixels_it_covers():
    # 21x30 input, prediction cropped to the multiple of 8 below each side,
    # as a latent model does: auto takes it for the top-left crop it is.
    hazy = _ramp(21, 30)
    pred = hazy[:16, :24].copy()

    assert data.classify_input_alignment(pred.shape, hazy.shape) == "crop_input"
    with _no_warning():
        aligned_pred, aligned_input, method = data.align_with_input(pred, hazy)

    assert method == data.INPUT_CROP_METHOD == "crop_input"
    assert aligned_pred is pred
    np.testing.assert_array_equal(aligned_input, pred)


def test_a_recorded_crop_meets_the_input_at_its_origin():
    hazy = _ramp(21, 30)
    pred = hazy[2:18, 3:27].copy()  # a crop off the top-left corner
    grid = _grid((21, 30), [[1, 0, -3], [0, 1, -2]])

    # Without the record, auto can only guess the top-left corner.
    _, guessed, _ = data.align_with_input(pred, hazy)
    assert not np.array_equal(guessed, pred)

    assert (
        data.classify_input_alignment(pred.shape, hazy.shape, spatial_alignment=grid)
        == "crop_input"
    )
    _, aligned_input, method = data.align_with_input(
        pred, hazy, spatial_alignment=grid
    )

    assert method == "crop_input"
    np.testing.assert_array_equal(aligned_input, pred)


@pytest.mark.parametrize(
    "spatial_alignment",
    [
        "resize",
        # A recorded whole-image resize of 21x30 to 16x24, pixel centres aligned.
        _grid((21, 30), [[0.8, 0, -0.1], [0, 16 / 21, 8 / 21 - 0.5]]),
    ],
    ids=["explicit", "recorded"],
)
def test_a_resized_prediction_still_upscales_the_smaller_image(spatial_alignment):
    hazy = _ramp(21, 30)
    pred = _ramp(16, 24)

    assert (
        data.classify_input_alignment(
            pred.shape, hazy.shape, spatial_alignment=spatial_alignment
        )
        == "upscale_prediction"
    )
    aligned_pred, aligned_input, method = data.align_with_input(
        pred, hazy, spatial_alignment=spatial_alignment
    )

    assert method == "upscale_prediction"
    expected_pred, expected_input = align_by_upscaling(pred, hazy)
    np.testing.assert_array_equal(aligned_pred, expected_pred)
    assert aligned_input is hazy and expected_input is hazy


def test_a_recorded_warp_resamples_the_input_onto_the_prediction_grid():
    # Halve 20x40 to 10x20, then keep the top-left 8x16: neither a crop nor a
    # whole-image resize of the input.
    hazy = _ramp(20, 40)
    grid = _grid((20, 40), [[0.5, 0, -0.25], [0, 0.5, -0.25]])

    _, aligned_input, method = data.align_with_input(
        np.zeros((8, 16, 3), np.float32), hazy, spatial_alignment=grid
    )

    assert method == data.INPUT_WARP_METHOD == "warp_input"
    # Prediction pixel (x', y') covers input pixels 2x'..2x'+1; on a linear
    # image their average is the ramp at the block centre.
    yy, xx = np.mgrid[0:8, 0:16].astype(np.float32)
    expected = np.stack(
        [(2 * xx + 0.5) / 40, (2 * yy + 0.5) / 20, (2 * xx + 0.5 + 2 * (2 * yy + 0.5)) / 200.0],
        axis=-1,
    )
    np.testing.assert_allclose(aligned_input, expected, atol=1e-5)


def test_an_explicit_crop_cannot_be_larger_than_the_input():
    with pytest.raises(ValueError, match="no larger than the input"):
        data.align_with_input(
            _ramp(24, 32), _ramp(21, 30), spatial_alignment="crop"
        )


def test_a_grid_for_another_source_falls_back_to_the_shape_guess():
    hazy = _ramp(21, 30)
    grid = _grid((40, 60), [[1, 0, -3], [0, 1, -2]])

    with pytest.warns(UserWarning, match=r"source, but the input is 21x30"):
        _, aligned_input, method = data.align_with_input(
            hazy[:16, :24], hazy, spatial_alignment=grid
        )

    assert method == "crop_input"
    np.testing.assert_array_equal(aligned_input, hazy[:16, :24])


def test_an_unchanged_crop_scores_no_change_on_hautiere():
    """Scoring the input itself is the documented e = 0, r = 1, σ = 0 check;
    it must hold for a model that crops its input to a multiple of 8, which
    stretched by upscaling used to read as a large gradient gain."""
    from tests.test_hautiere import _pattern_rgb

    hazy = (_pattern_rgb(52, 70) * 0.4 + 0.5).astype(np.float32)
    pred = hazy[:48, :64].copy()

    aligned_pred, aligned_input, _ = data.align_with_input(pred, hazy)
    scores = load_hautiere("cpu")(aligned_pred, aligned_input)

    assert scores == {"hautiere_e": 0.0, "hautiere_r": 1.0, "hautiere_sigma": 0.0}


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


def _cropped_samples(top: int, left: int, *, record: bool) -> list[dict]:
    """Input-only samples whose prediction is the input's 16x24 crop at
    ``(top, left)``, unchanged, optionally recording that as its source_grid."""
    samples = []
    for index in (1, 2):
        hazy = _ramp(21, 30) * (0.5 + 0.25 * index)
        sample = {
            "id": f"{index:05d}",
            "full_id": f"/scene/{index:05d}",
            "pred": hazy[top : top + 16, left : left + 24].copy(),
            "input": hazy,
        }
        if record:
            grid = {"size": [21, 30], "affine": [[1, 0, -left], [0, 1, -top]]}
            sample["attributes"] = {"pred": {"source_grid": grid}}
        samples.append(sample)
    return samples


def test_input_referenced_metrics_see_the_input_pixels_a_crop_covers(domain):
    result = eval_mod.evaluate_rgb_samples(
        _Dataset(_cropped_samples(0, 0, record=False), ("input", "pred")),
        device="cpu",
        domains=("dehazing",),
    )

    assert {call[1:] for call in domain} == {((16, 24, 3), (16, 24, 3))}
    # The prediction is the input's pixels, unchanged.
    assert result["rgb"]["dehazing"]["gain"] == pytest.approx(1.0, abs=1e-6)
    assert result["rgb"]["dehazing"]["loss"] == pytest.approx(0.0, abs=1e-6)
    assert result["spatial_info"]["mode"] == "auto"
    assert result["spatial_info"]["input_alignment"] == "crop_input"


def test_input_referenced_metrics_follow_a_recorded_source_grid(domain):
    recorded = eval_mod.evaluate_rgb_samples(
        _Dataset(_cropped_samples(2, 3, record=True), ("input", "pred")),
        device="cpu",
        domains=("dehazing",),
    )

    assert recorded["spatial_info"]["mode"] == "source_grid"
    assert recorded["spatial_info"]["input_alignment"] == "crop_input"
    assert recorded["rgb"]["dehazing"]["loss"] == pytest.approx(0.0, abs=1e-6)

    # An explicit mode still wins, and its top-left crop misplaces these.
    guessed = eval_mod.evaluate_rgb_samples(
        _Dataset(_cropped_samples(2, 3, record=True), ("input", "pred")),
        device="cpu",
        domains=("dehazing",),
        spatial_alignment="crop",
    )

    assert guessed["spatial_info"]["mode"] == "crop"
    assert guessed["spatial_info"]["input_alignment"] == "crop_input"
    assert abs(guessed["rgb"]["dehazing"]["loss"]) > 0.01


def test_a_source_grid_written_with_the_prediction_reaches_the_input_alignment(
    tmp_path, domain
):
    """End to end: a dataset head's ``source_grid``, and one file's overriding
    it, as euler-inference writes them, are read through euler-loading."""
    ds_crawler = pytest.importorskip("ds_crawler")
    from PIL import Image

    from tests.test_hautiere import _pattern_rgb

    def writer(root, meta):
        return ds_crawler.DatasetWriter(
            root, name=root.name, type="rgb", separator=None,
            euler_train={"used_as": "target", "modality_type": "rgb"},
            meta={"range": [0, 255], **meta},
            euler_loading={"loader": "generic_dense_depth", "function": "rgb"},
        )

    def crop_grid(top, left):
        affine = [[1, 0, -left], [0, 1, -top]]
        return {"source": "rgb", "size": [40, 60], "affine": affine}

    # 24x40 crops of 40x60 inputs, which the shapes alone would call a resize.
    hazy_writer = writer(tmp_path / "hazy", {})
    pred_writer = writer(tmp_path / "pred", {"source_grid": crop_grid(8, 10)})
    for k, (top, left) in enumerate([(8, 10), (4, 6)]):
        hazy = (_pattern_rgb(40, 60) * (100 + 50 * k)).astype(np.uint8)
        Image.fromarray(hazy).save(hazy_writer.get_path(f"seq/{k:04d}", f"{k:04d}.png"))
        attributes = {"source_grid": crop_grid(top, left)} if k == 1 else None
        path = pred_writer.get_path(f"seq/{k:04d}", f"{k:04d}.png", attributes=attributes)
        Image.fromarray(hazy[top : top + 24, left : left + 40]).save(path)
    hazy_writer.save_index()
    pred_writer.save_index()

    dataset = data.build_rgb_eval_dataset(
        gt_rgb_path=None,
        pred_rgb_path=str(tmp_path / "pred"),
        input_rgb_path=str(tmp_path / "hazy"),
    )
    result = eval_mod.evaluate_rgb_samples(
        dataset, device="cpu", num_workers=0, domains=("dehazing",)
    )

    assert result["spatial_info"]["mode"] == "source_grid"
    assert result["spatial_info"]["input_alignment"] == "crop_input"
    assert {call[1:] for call in domain} == {((24, 40, 3), (24, 40, 3))}
    losses = _metric_values(result["per_file_metrics"], "loss")
    assert len(losses) == 2 and max(abs(value) for value in losses) == 0.0


def _metric_values(tree, name: str) -> list[float]:
    """Every per-file value of metric *name* under ``per_file_metrics``."""
    if isinstance(tree, dict):
        found = [float(tree[name])] if isinstance(tree.get(name), (int, float)) else []
        return found + [
            value
            for key, child in tree.items()
            if key != name
            for value in _metric_values(child, name)
        ]
    if isinstance(tree, list):
        return [value for child in tree for value in _metric_values(child, name)]
    return []


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
