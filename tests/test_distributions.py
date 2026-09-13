"""Histogram boundary, masking, aggregation, and eval.json contract tests."""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from euler_eval import cli, evaluate as evaluation
from euler_eval.distribution_contract import validate_distribution_contract
from euler_eval.distributions import (
    DistributionConfig,
    ErrorDistribution,
    bin_values,
    depth_error_magnitudes,
)

from .distribution_contract_utils import assert_distribution_contract


def test_log_bins_include_zero_boundaries_overflow_and_ignore_invalid_values():
    config = DistributionConfig(n_bins=5, min_error=0.01, max_error=10)
    edges = config.bin_edges()
    np.testing.assert_allclose(edges, [0, 0.01, 0.1, 1, 10, np.inf])
    values = [0, 0.005, *edges[1:-1], 100, -1, np.nan, np.inf, -np.inf]
    counts = bin_values(values, edges)
    assert counts.tolist() == [2, 1, 1, 1, 2]
    assert counts.dtype == np.int64
    description = json.loads(json.dumps(config.binning_description(), allow_nan=False))
    assert description["binEdges"][-1] is None
    assert len(description["binEdges"]) == config.n_bins + 1


@pytest.mark.parametrize("minimum", [0, 0.01])
def test_linear_bins_and_empty_or_perfect_samples(minimum):
    config = DistributionConfig(
        n_bins=5, scale="linear", min_error=minimum, max_error=4
    )
    store = ErrorDistribution(config)
    assert store.add([])["rmse"]["distribution"] == [0] * 5
    assert store.add(np.zeros(7))["rmse"]["distribution"] == [7, 0, 0, 0, 0]
    assert store.summary()["pixel_pool"]["rmse"]["distribution"] == [7, 0, 0, 0, 0]
    assert store.edges.size == 6
    np.testing.assert_allclose(
        np.diff(store.edges[int(minimum > 0) : -1]),
        (4 - minimum) / (4 - int(minimum > 0)),
    )


def test_weighted_bins_can_sum_error_by_a_different_axis():
    depths = [0, 1, 2, 5, np.nan, 1]
    errors = [0.25, 0.5, 2, 10, 7, np.inf]
    np.testing.assert_array_equal(
        bin_values(depths, [0, 2, 5, np.inf], weights=errors), [0.75, 2, 10]
    )
    with pytest.raises(ValueError, match="same size"):
        bin_values([1, 2], [0, 3], weights=[1])
    with pytest.raises(ValueError, match="strictly increasing"):
        bin_values([1], [0, 0, 3])


@pytest.mark.parametrize(
    "settings",
    [
        {"n_bins": 0},
        {"n_bins": 2},
        {"n_bins": 3.5},
        {"n_bins": True},
        {"scale": "sqrt"},
        {"min_error": 0},
        {"min_error": -1},
        {"min_error": 10, "max_error": 1},
        {"min_error": 1, "max_error": 1},
        {"max_error": np.inf},
        {"min_error": np.nan},
        {"max_error": "10"},
        {"max_error": True},
        {"min_error": 1, "max_error": np.nextafter(1.0, 2.0)},
    ],
)
def test_invalid_bin_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        DistributionConfig(**settings)


def test_depth_magnitudes_use_valid_pixels_and_avoid_float32_square_overflow():
    gt = np.array([1, 1, 1, 0, np.nan, 1], dtype=np.float32)
    pred = np.array([1, 1e30, -1, 2, 2, np.inf], dtype=np.float32)
    values = depth_error_magnitudes(pred, gt)
    assert values.size == 2
    assert np.isfinite(values).all()
    assert bin_values(values, [0, 1, 10, np.inf]).tolist() == [1, 0, 1]


@pytest.fixture
def light_depth_backends(monkeypatch):
    """Stub only the learned metrics; run real depth and histogram computations."""
    monkeypatch.setattr(
        evaluation,
        "LPIPSMetric",
        lambda **kw: SimpleNamespace(
            compute_batch=lambda preds, gts, **kw: [0.0] * len(preds),
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "FIDKIDMetric",
        lambda **kw: SimpleNamespace(
            compute_fid=lambda *a, **kw: 0.0,
            compute_kid=lambda *a, **kw: (0.0, 0.0),
        ),
    )


def _depth_samples():
    samples = []
    for i, errors in enumerate(([0, 0.5, 1, 2, 3], [1], [])):
        gt = np.full((8, 8), 10.0, dtype=np.float32)
        pred = np.zeros_like(gt)
        pred.flat[: len(errors)] = 10 + np.array(errors)
        # A large sky error, invalid GT, invalid predictions, and an empty image.
        sky = np.zeros_like(gt, dtype=bool)
        sky[1, 0] = True
        pred[1, 0] = 110
        gt[1, 1] = np.nan
        pred[1, 1:4] = [12, np.inf, -1]
        samples.append(
            {
                "id": str(i),
                "full_id": f"/scene/{i}",
                "gt": gt,
                "pred": pred,
                "segmentation": sky,
            }
        )
    return samples


def _counts(branch, metric="rmse", pooled=False):
    block = branch["distributions"]
    if pooled:
        block = block["pixel_pool"]
    return block[metric]["distribution"]["values"]


def _files(results):
    return results["per_file_metrics"]["children"]["scene"]["files"]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in (multiply|subtract):RuntimeWarning"
)
def test_dense_depth_counts_match_valid_pixel_pool_and_batch_path(
    light_depth_backends, monkeypatch, batched
):
    # Exercise deferred callbacks on CPU, including the empty sample's tail flush.
    monkeypatch.setattr(
        evaluation.GPUDepthMetricsBatcher, "is_available", lambda device: batched
    )
    config = DistributionConfig(n_bins=4, scale="linear", min_error=0, max_error=3)
    result = evaluation.evaluate_depth_samples(
        _depth_samples(),
        is_radial=True,
        device="cpu",
        num_workers=0,
        alignment_mode="none",
        sky_mask_enabled=True,
        batch_size=2,
        distribution_config=config,
        benchmark_depth_range=(1, 20),
    )
    samples = [_counts(f["metrics"]["depth"]) for f in _files(result)]
    assert samples == [[2, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]]
    assert _counts(result["depth"], pooled=True) == [2, 2, 1, 1]
    np.testing.assert_array_equal(
        np.sum(samples, axis=0), _counts(result["depth"], pooled=True)
    )
    assert _counts(result["depth_benchmark"]["metric"]["all"], pooled=True) == [
        2,
        2,
        1,
        1,
    ]
    # Existing scalar metric paths survive when distributions are enabled.
    assert isinstance(
        _files(result)[0]["metrics"]["depth"]["depth_metrics"]["rmse"], float
    )
    assert isinstance(result["depth"]["standard"]["pixel_pool"]["rmse"], float)
    assert_distribution_contract(result)


def test_depth_distribution_follows_alignment_and_sky_cap(light_depth_backends):
    gt = np.arange(64, dtype=np.float32).reshape(8, 8) + 10
    pred = gt * 2
    config = DistributionConfig(n_bins=4, scale="linear", min_error=0, max_error=3)
    result = evaluation.evaluate_depth_samples(
        [{"id": "0", "gt": gt, "pred": pred}],
        is_radial=True,
        device="cpu",
        num_workers=0,
        alignment_mode="affine",
        sky_depth=20,
        distribution_config=config,
    )
    assert _counts(result["depth_native"], pooled=True) == [54, 1, 1, 8]
    assert _counts(result["depth_metric"], pooled=True) == [64, 0, 0, 0]
    assert _counts(result["depth"], pooled=True) == [64, 0, 0, 0]


def _sparse_sample(points=False):
    cloud = np.array([[0, 0, 2], [4, 0, 2], [4, 0, 4]], dtype=np.float32)
    pred = np.zeros((8, 8, 3) if points else (8, 8), dtype=np.float32)
    if points:
        pred[0, 0] = [0, 0, 2]
        pred[0, 2] = [4, 0, 5]  # Euclidean error 3.
    else:
        pred[0, 0] = 2
        pred[0, 2] = np.linalg.norm(cloud[1]) + 3
    # Third projection is masked sky; unobserved pixels never contribute.
    sky = np.zeros((8, 8), dtype=bool)
    sky[0, 1] = True
    pred[0, 1] = 100
    return {
        "id": "0",
        "full_id": "/scene/0",
        "gt": cloud,
        "pred": pred,
        "segmentation": sky,
        "intrinsics": np.eye(3, dtype=np.float32),
        "camera_extrinsics": np.eye(4, dtype=np.float32),
    }


@pytest.mark.parametrize("points", [False, True])
def test_sparse_histograms_count_only_evaluated_correspondences(points):
    config = DistributionConfig(n_bins=4, scale="linear", min_error=0, max_error=2)
    kwargs = dict(
        num_workers=0,
        alignment_mode="none",
        sky_mask_enabled=True,
        distribution_config=config,
    )
    sample = _sparse_sample(points)
    if points:
        result = evaluation.evaluate_points_3d_sparse_samples(
            [sample], pred_is_depth=False, **kwargs
        )
        key, metric = "points_3d", "rmse3d"
    else:
        result = evaluation.evaluate_sparse_depth_samples(
            [sample], pred_is_radial=True, **kwargs
        )
        key, metric = "sparse_depth", "rmse"
    assert _counts(result[key], metric, pooled=True) == [1, 0, 0, 1]
    assert _counts(_files(result)[0]["metrics"][key], metric) == [1, 0, 0, 1]
    assert (
        result["distribution_info"]["binSets"]["error_magnitude"]
        == config.binning_description()
    )


def test_pointmap_distribution_uses_euclidean_magnitudes_and_both_spaces():
    gt = np.zeros((8, 8, 3), dtype=np.float32)
    gt[..., 2] = 10
    pred = gt * 2
    config = DistributionConfig(n_bins=4, scale="linear", min_error=0, max_error=3)
    samples = [{"id": "0", "full_id": "/scene/0", "gt": gt, "pred": pred}]
    result = evaluation.evaluate_points_3d_samples(
        samples,
        num_workers=0,
        alignment_mode="scale",
        distribution_config=config,
    )
    assert _counts(result["points_3d_native"], "rmse3d", pooled=True) == [0, 0, 0, 64]
    assert _counts(result["points_3d_metric"], "rmse3d", pooled=True) == [64, 0, 0, 0]
    assert _counts(_files(result)[0]["metrics"]["points_3d"], "rmse3d") == [64, 0, 0, 0]


@pytest.mark.parametrize(
    "kind", ["depth", "sparse_depth", "points_3d", "points_3d_sparse"]
)
def test_cli_serializes_typed_values_axes_and_shared_bin_definitions(
    tmp_path, monkeypatch, light_depth_backends, kind
):
    sparse = "sparse" in kind
    points = "points_3d" in kind
    prediction_key = "points_3d" if points else "depth"
    root = "points3d" if points else "sparsedepth" if sparse else "depth"
    metric = "rmse3d" if points else "rmse"
    gt_key = "sparse_depth" if sparse else prediction_key
    gt = {gt_key: {"path": str(tmp_path)}}
    if sparse:
        gt.update(
            {k: {"path": str(tmp_path)} for k in ("intrinsics", "camera_extrinsics")}
        )
    output = tmp_path / "result.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("preserved.txt", "keep")
    config = {
        "gt": gt,
        "datasets": [
            {
                "name": "model",
                prediction_key: {"path": str(tmp_path)},
                "output_file": str(output / "eval.json"),
            }
        ],
        "distributions": {
            "n_bins": 8,
            "scale": "linear",
            "min_error": 0,
            "max_error": 3,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    if sparse:
        sample = _sparse_sample(points)
    elif points:
        gt_map = np.zeros((8, 8, 3), dtype=np.float32)
        gt_map[..., 2] = 10
        sample = {"id": "0", "full_id": "/scene/0", "gt": gt_map, "pred": gt_map.copy()}
    else:
        sample = {
            "id": "0",
            "full_id": "/scene/0",
            "gt": np.full((8, 8), 10.0),
            "pred": np.full((8, 8), 10.0),
        }
    monkeypatch.setattr(cli, f"build_{kind}_eval_dataset", lambda **kw: [sample])
    metadata_kind = "points_3d" if points else kind
    monkeypatch.setattr(
        cli,
        f"get_{metadata_kind}_metadata",
        lambda ds: {
            "radial_depth": True,
            "pred_radial_depth": True,
            "fov_domain": "sfov",
        },
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "euler-eval",
            str(config_path),
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--no-sanity-check",
            "--distribution-bins",
            "4",
        ]
        + ([] if points else ["--benchmark-depth-range", "1", "20"]),
    )
    cli.main()
    with zipfile.ZipFile(output) as archive:
        result = json.loads(archive.read("eval.json"))
        assert archive.read("preserved.txt") == b"keep"
    envelope = result["metricSet"]
    registry = envelope["metadata"]["distributions"]
    bins = registry["binSets"]["error_magnitude"]
    assert len(bins["binEdges"]) == 5  # CLI overrides the config setting.
    assert bins["binEdges"] == [0.0, 1.0, 2.0, 3.0, None]
    assert "distributions" in envelope["axes"]["category"]["values"]
    assert f"{metric}.distribution" in envelope["metricDescriptions"]
    space = "native" if points else "metric"
    aggregate = _counts(result[root]["eval"][space], metric, pooled=True)
    sample_counts = _counts(_files(result)[0]["metrics"][root]["eval"][space], metric)
    assert aggregate == sample_counts
    assert len(aggregate) == 4 and all(type(count) is int for count in aggregate)
    assert json.dumps(result).count('"binEdges"') == 1
    value = result[root]["eval"][space]["distributions"]["pixel_pool"][metric][
        "distribution"
    ]
    assert value["type"] == "distribution"
    definition = registry["definitions"][value["definitionId"]]
    assert definition["space"] == space
    assert definition["axes"]["x"]["unit"] == "m"
    validate_distribution_contract(registry, [value])
    if not points:
        benchmark = result[root]["eval"][space]["distributions"]["pixel_pool"]["all"]
        assert benchmark[metric]["distribution"]["values"] == aggregate
    json.dumps(result, allow_nan=False)
    assert_distribution_contract(result)


def test_distributions_are_opt_in_and_scalars_do_not_change(light_depth_backends):
    sample = {"id": "0", "gt": np.full((8, 8), 10.0), "pred": np.full((8, 8), 12.0)}
    kwargs = dict(is_radial=True, device="cpu", num_workers=0, alignment_mode="none")
    baseline = evaluation.evaluate_depth_samples([sample], **kwargs)
    enabled = evaluation.evaluate_depth_samples(
        [sample], distribution_config=DistributionConfig(), **kwargs
    )
    assert "distribution_info" not in baseline
    assert "distributions" not in baseline["depth"]
    assert baseline["depth"] == {
        k: v for k, v in enabled["depth"].items() if k != "distributions"
    }
    assert "distributions" not in cli._depth_eval_axes()["category"].values
    args = SimpleNamespace(
        distributions=False,
        distribution_bins=None,
        distribution_scale=None,
        distribution_range=None,
    )
    assert cli._resolve_distribution_config(True, args) is None
    args.distributions = None
    assert cli._resolve_distribution_config(None, args) is None
    args.distributions = True
    assert cli._resolve_distribution_config(None, args) == DistributionConfig()


@pytest.mark.parametrize(
    "kind", ["depth", "sparse_depth", "points_3d", "points_3d_sparse"]
)
def test_uncalibrated_native_output_does_not_claim_metric_units(
    kind, light_depth_backends
):
    points = "points_3d" in kind
    if "sparse" in kind:
        sample = _sparse_sample(points)
    else:
        gt = np.full((8, 8, 3) if points else (8, 8), 10.0, dtype=np.float32)
        sample = {"id": "0", "gt": gt, "pred": gt / 2}
    kwargs = dict(
        num_workers=0,
        alignment_mode="none",
        input_space_hint="relative",
        distribution_config=DistributionConfig(n_bins=3),
    )
    if kind == "depth":
        kwargs.update(is_radial=True, device="cpu")
    elif kind == "sparse_depth":
        kwargs.update(pred_is_radial=True)
    elif kind == "points_3d_sparse":
        kwargs.update(pred_is_depth=False)
    result = getattr(evaluation, f"evaluate_{kind}_samples")([sample], **kwargs)
    assert_distribution_contract(result)
    assert result["space_info"]["emitted_spaces"] == ["native"]
    definitions = list(result["distribution_info"]["definitions"].values())
    assert len(definitions) == 1
    assert definitions[0]["space"] == "native"
    assert definitions[0]["axes"]["x"]["unit"] == "unspecified"


@pytest.mark.parametrize("value", [1, [], "rmse", {"bins": 10}, {"n_bins": -5}])
def test_invalid_json_distribution_settings(value):
    with pytest.raises(ValueError):
        DistributionConfig.from_config(value)


@pytest.mark.parametrize(
    "kind", ["depth", "sparse_depth", "points_3d", "points_3d_sparse"]
)
def test_no_valid_pixels_retains_zero_histograms(kind, light_depth_backends):
    points = "points_3d" in kind
    if "sparse" in kind:
        sample = _sparse_sample(points)
        sample["pred"][:] = 0
    else:
        values = np.zeros((8, 8, 3) if points else (8, 8), dtype=np.float32)
        sample = {"id": "0", "full_id": "/scene/0", "gt": values, "pred": values}
    kwargs = dict(
        num_workers=0,
        alignment_mode="none",
        distribution_config=DistributionConfig(n_bins=3),
    )
    if kind == "depth":
        kwargs.update(is_radial=True, device="cpu")
    elif kind == "sparse_depth":
        kwargs.update(pred_is_radial=True)
    elif kind == "points_3d_sparse":
        kwargs.update(pred_is_depth=False)
    result = getattr(evaluation, f"evaluate_{kind}_samples")([sample], **kwargs)
    key, metric = ("points_3d", "rmse3d") if points else (kind, "rmse")
    assert _counts(result[key], metric, pooled=True) == [0, 0, 0]
    assert _counts(_files(result)[0]["metrics"][key], metric) == [0, 0, 0]
    json.dumps(cli._clean_metric_tree(result), allow_nan=False)
    assert_distribution_contract(result)


@pytest.mark.parametrize(
    "flags",
    [
        ["--distribution-bins", "2"],
        ["--distribution-bins", "3.5"],
        ["--distribution-range", "0", "10"],
        ["--distribution-range", "1", "nan"],
        ["--distribution-scale", "sqrt"],
    ],
)
def test_cli_rejects_invalid_histogram_flags_before_loading_data(
    tmp_path, monkeypatch, flags
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "gt": {"depth": {"path": str(tmp_path)}},
                "datasets": [{"name": "model", "depth": {"path": str(tmp_path)}}],
            }
        )
    )

    def unexpected_load(**kwargs):
        pytest.fail("Invalid histogram settings must fail before loading data")

    monkeypatch.setattr(cli, "build_depth_eval_dataset", unexpected_load)
    monkeypatch.setattr(cli.sys, "argv", ["euler-eval", str(config_path), *flags])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
