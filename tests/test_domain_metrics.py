"""Tests for composable domain metric sets and dehazing metrics."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import euler_eval.evaluate as eval_mod
import euler_eval.metric_sets as metric_sets_mod
from euler_eval import cli
from euler_eval.cli import _RGB_EVAL_DESCRIPTIONS, _rgb_eval_axes
from euler_eval.metric_sets import (
    DEHAZING_METRIC_SET,
    MetricSet,
    NoReferenceRGBMetric,
    available_domains,
    check_metric_set_requirements,
    metric_set_package_versions,
    resolve_metric_sets,
)
from euler_eval.metrics import (
    PyIQAScorer,
    compute_fade,
    compute_niqe,
    load_clipiqa,
    load_musiq,
)


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
    assert _RGB_EVAL_DESCRIPTIONS["musiq"].is_higher_better is True
    assert _RGB_EVAL_DESCRIPTIONS["musiq"].display_name == "MUSIQ"
    assert _RGB_EVAL_DESCRIPTIONS["clipiqa"].is_higher_better is True
    assert _RGB_EVAL_DESCRIPTIONS["clipiqa"].display_name == "CLIP-IQA"
    assert _RGB_EVAL_DESCRIPTIONS["clipiqa"].min_value == 0.0
    assert _RGB_EVAL_DESCRIPTIONS["clipiqa"].max_value == 1.0


def test_dehazing_domain_lists_learned_metrics_after_natural_scene_metrics():
    metrics = DEHAZING_METRIC_SET.rgb_no_reference
    assert [metric.name for metric in metrics] == ["niqe", "fade", "musiq", "clipiqa"]
    by_name = {metric.name: metric for metric in metrics}
    assert by_name["musiq"].load is load_musiq
    assert by_name["clipiqa"].load is load_clipiqa
    assert by_name["musiq"].requires == ("pyiqa",)
    assert by_name["clipiqa"].requires == ("pyiqa",)
    assert DEHAZING_METRIC_SET.extra == "dehazing"


def test_no_reference_metric_needs_exactly_one_of_compute_or_load():
    with pytest.raises(ValueError, match="exactly one of 'compute' or 'load'"):
        NoReferenceRGBMetric(name="x", display_name="X", is_higher_better=True)
    with pytest.raises(ValueError, match="exactly one of 'compute' or 'load'"):
        NoReferenceRGBMetric(
            name="x",
            display_name="X",
            is_higher_better=True,
            compute=lambda image: 0.0,
            load=lambda device: (lambda image: 0.0),
        )


def test_bind_returns_compute_or_loads_on_device():
    def compute(image):
        return 1.0

    stateless = NoReferenceRGBMetric(
        name="a", display_name="A", is_higher_better=True, compute=compute
    )
    devices = []
    loaded = NoReferenceRGBMetric(
        name="b",
        display_name="B",
        is_higher_better=True,
        load=lambda device: devices.append(device) or compute,
    )

    assert stateless.bind("cuda") is compute
    assert loaded.bind("cuda:1") is compute
    assert devices == ["cuda:1"]


def _metric_set_requiring(*packages: str, extra: str | None = "dehazing") -> MetricSet:
    return MetricSet(
        name="dehazing",
        category="dehazing",
        rgb_no_reference=tuple(
            NoReferenceRGBMetric(
                name=f"m{index}",
                display_name=f"M{index}",
                is_higher_better=True,
                load=lambda device: (lambda image: 0.0),
                requires=(package,),
            )
            for index, package in enumerate(packages)
        ),
        extra=extra,
    )


def test_requirement_check_names_missing_package_and_extra():
    missing = "euler_eval_definitely_missing_package"
    with pytest.raises(ValueError) as exc:
        check_metric_set_requirements(
            (metric_sets_mod.CORE_METRIC_SET, _metric_set_requiring(missing, missing))
        )

    message = str(exc.value)
    assert f"--domain dehazing needs the '{missing}' package for M0, M1" in message
    assert 'pip install "euler-eval[dehazing]"' in message


def test_requirement_check_without_extra_suggests_package():
    missing = "euler_eval_definitely_missing_package"
    with pytest.raises(ValueError, match=f"pip install {missing}"):
        check_metric_set_requirements((_metric_set_requiring(missing, extra=None),))


def test_requirement_check_accepts_installed_packages():
    check_metric_set_requirements(
        (metric_sets_mod.CORE_METRIC_SET, _metric_set_requiring("numpy"))
    )


def test_package_versions_report_installed_and_missing_packages():
    versions = metric_set_package_versions(
        (_metric_set_requiring("numpy", "euler_eval_definitely_missing_package"),)
    )

    assert versions == {
        "euler_eval_definitely_missing_package": None,
        "numpy": np.__version__,
    }
    assert metric_set_package_versions(resolve_metric_sets()) == {}


def test_cli_rejects_domain_whose_package_is_missing_before_work(monkeypatch):
    missing = "euler_eval_definitely_missing_package"
    monkeypatch.setitem(
        metric_sets_mod.DOMAIN_METRIC_SETS, "dehazing", _metric_set_requiring(missing)
    )
    args = cli._build_parser().parse_args(["config.json", "--domain", "dehazing"])

    with pytest.raises(ValueError, match=f"needs the '{missing}' package"):
        cli._prepare_run(args, {})

    args = cli._build_parser().parse_args(["config.json"])
    assert [item.name for item in cli._prepare_run(args, {}).metric_sets] == ["core"]


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


class _FakeIQAMetric:
    lower_better = False

    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.inputs = []
        self.inference_mode = []

    def __call__(self, tensor):
        self.inputs.append(tensor.clone())
        self.inference_mode.append(torch.is_inference_mode_enabled())
        return tensor.mean().reshape(1, 1)


@pytest.fixture
def fake_pyiqa(monkeypatch):
    created = []
    module = types.ModuleType("pyiqa")

    def create_metric(metric_name, device=None):
        metric = _FakeIQAMetric(metric_name, device)
        created.append(metric)
        return metric

    module.create_metric = create_metric
    monkeypatch.setitem(sys.modules, "pyiqa", module)
    return created


@pytest.mark.parametrize(
    "loader, metric_name", [(load_musiq, "musiq"), (load_clipiqa, "clipiqa")]
)
def test_pyiqa_loaders_create_the_named_metric_on_device(
    fake_pyiqa, loader, metric_name
):
    scorer = loader("cpu")

    assert isinstance(scorer, PyIQAScorer)
    assert [metric.name for metric in fake_pyiqa] == [metric_name]
    assert fake_pyiqa[0].device == torch.device("cpu")
    assert scorer.lower_better is False


def test_pyiqa_scorer_feeds_one_nchw_float_tensor_in_inference_mode(fake_pyiqa):
    image = _pattern_image(24, 40)
    image[0, 0] = 1.0 + 5e-7  # within the rounding tolerance

    score = PyIQAScorer("musiq", "cpu")(image)

    metric = fake_pyiqa[0]
    (tensor,) = metric.inputs
    assert tensor.shape == (1, 3, 24, 40)
    assert tensor.dtype == torch.float32
    assert metric.inference_mode == [True]
    assert float(tensor.max()) <= 1.0
    expected = np.clip(image, 0.0, 1.0).transpose(2, 0, 1)
    np.testing.assert_allclose(tensor[0].numpy(), expected, rtol=0, atol=1e-7)
    assert isinstance(score, float)
    assert score == pytest.approx(float(np.clip(image, 0.0, 1.0).mean()), rel=1e-6)


def test_pyiqa_scorer_rejects_values_outside_unit_range(fake_pyiqa):
    image = _pattern_image(24, 40)
    image[0, 0, 0] = 2.0

    with pytest.raises(
        ValueError, match=r"pyiqa metrics expect RGB values in the \[0, 1\]"
    ):
        PyIQAScorer("clipiqa", "cpu")(image)


def test_pyiqa_scorer_explains_how_to_install_pyiqa(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyiqa", None)

    with pytest.raises(ImportError, match=r'pip install "euler-eval\[dehazing\]"'):
        load_musiq("cpu")


def _pyiqa_weights_cached() -> bool:
    hub = Path(torch.hub.get_dir())
    return (hub / "pyiqa" / "musiq_koniq_ckpt-e95806b9.pth").is_file() and (
        hub / "clip" / "RN50.pt"
    ).is_file()


@pytest.mark.parametrize(
    "loader, low, high", [(load_musiq, 0.0, 100.0), (load_clipiqa, 0.0, 1.0)]
)
def test_pyiqa_scorer_matches_pyiqa_scoring_the_png_file(tmp_path, loader, low, high):
    """Run the real networks when pyiqa and its weights are available."""
    pytest.importorskip("pyiqa")
    if not _pyiqa_weights_cached():
        pytest.skip("pyiqa weights are not cached; run euler-eval.init")
    from PIL import Image

    image = np.round(_pattern_image(240, 320) * 255.0).astype(np.uint8)
    path = tmp_path / "restored.png"
    Image.fromarray(image).save(path)

    scorer = loader("cpu")
    score = scorer(image.astype(np.float32) / 255.0)
    with torch.inference_mode():
        expected = scorer._metric(str(path)).item()

    assert score == pytest.approx(expected, rel=1e-6)
    assert low <= score <= high


def test_rgb_evaluator_loads_model_metrics_once_on_its_device(monkeypatch):
    devices = []

    def load(device):
        devices.append(device)
        return lambda image: float(np.max(image))

    dummy_set = MetricSet(
        name="dehazing",
        category="dehazing",
        rgb_no_reference=(
            NoReferenceRGBMetric(
                name="musiq",
                display_name="MUSIQ",
                load=load,
                is_higher_better=True,
            ),
        ),
    )
    monkeypatch.setitem(metric_sets_mod.DOMAIN_METRIC_SETS, "dehazing", dummy_set)
    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _DummyFID)
    monkeypatch.setattr(eval_mod, "tqdm", lambda iterable, *args, **kwargs: iterable)

    samples = [
        {
            "id": f"{index:05d}",
            "full_id": f"/scene/{index:05d}",
            "gt": np.full((16, 16, 3), value, dtype=np.float32),
            "pred": np.full((16, 16, 3), value, dtype=np.float32),
        }
        for index, value in enumerate((0.2, 0.6), start=1)
    ]

    result = eval_mod.evaluate_rgb_samples(
        _DummyRGBDataset(samples), device="cpu", domains=("dehazing",)
    )

    assert devices == ["cpu"]
    assert result["rgb"]["dehazing"] == pytest.approx({"musiq": 0.4})


def test_rgb_evaluator_stops_when_a_model_metric_cannot_load(monkeypatch):
    def load(device):
        raise OSError("no network")

    dummy_set = MetricSet(
        name="dehazing",
        category="dehazing",
        rgb_no_reference=(
            NoReferenceRGBMetric(
                name="musiq", display_name="MUSIQ", load=load, is_higher_better=True
            ),
        ),
    )
    monkeypatch.setitem(metric_sets_mod.DOMAIN_METRIC_SETS, "dehazing", dummy_set)
    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _DummyFID)
    image = np.zeros((16, 16, 3), dtype=np.float32)
    dataset = _DummyRGBDataset(
        [{"id": "00001", "full_id": "/scene/00001", "gt": image, "pred": image}]
    )

    with pytest.raises(
        RuntimeError, match=r"Failed to load dehazing\.musiq: no network"
    ):
        eval_mod.evaluate_rgb_samples(dataset, device="cpu", domains=("dehazing",))
