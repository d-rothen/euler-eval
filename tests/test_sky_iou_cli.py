"""CLI tests for ``--sky-iou``: gating, invariance, output keys and config."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from ds_crawler import build_dataset_head
from ds_crawler.artifacts import save_output_artifacts
from euler_metric_naming import AxisDeclaration, decompose
from PIL import Image

from euler_eval import cli, data
from tests.test_depth_alignment_output import _patch_depth_metrics

SKY_LABEL = 7
FRAMES = ("first", "second")


def _write_index(root, *, scope, key, loader, function, meta, files):
    head = build_dataset_head(
        dataset={"name": "sky_iou_test"},
        modality={"key": key, "meta": meta},
        addons={"euler_loading": {
            "version": "1.0", "loader": loader, "function": function,
        }},
    )
    save_output_artifacts(root, {
        "contract": {"kind": "dataset_index", "version": "1.0"},
        "head_file": "dataset-head.json", "head": head,
        "generator": {"name": "tests"}, "indexing": {}, "execution": {},
        "index": {"files": files},
    }, metadata_scope=scope)


def _gt_sky(frame_id):
    # The bottom row is sky for "first"; "second" is flipped like the depth.
    sky = np.array([[False, False], [True, True]])
    return np.flip(sky) if frame_id == "second" else sky


def write_gt_segmentation(root, *, scope="segmentation", key="segmentation"):
    """Index per-frame class-id maps whose sky label is SKY_LABEL."""
    root = Path(root)
    root.mkdir(exist_ok=True)
    files = []
    for frame_id in FRAMES:
        labels = np.where(_gt_sky(frame_id), SKY_LABEL, 1).astype(np.uint8)
        path = f"{frame_id}_labels.npy"
        np.save(root / path, labels)
        files.append({"path": path, "id": frame_id})
    _write_index(
        root, scope=scope, key=key, loader="generic",
        function="semantic_segmentation",
        meta={"skyclass": [SKY_LABEL] * 3}, files=files,
    )
    return str(root)


def write_sky_mask(root, frames):
    """Index RGB sky masks (sky at (1, 1) for "first", (0, 0) for "second")."""
    root = Path(root)
    root.mkdir()
    color = [12, 34, 56]
    files = []
    for frame_id in frames:
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[(1, 1) if frame_id == "first" else (0, 0)] = color
        Image.fromarray(rgb).save(root / f"{frame_id}.png")
        files.append({"path": f"{frame_id}.png", "id": frame_id})
    _write_index(
        root, scope="sky_mask", key="sky_mask", loader="generic_dense_depth",
        function="sky_mask", meta={"skyclass": color}, files=files,
    )
    return str(root)


def write_depth(root, *, scope, source):
    """Index a copy of the depth files in *source* under metadata *scope*."""
    root = Path(root)
    root.mkdir()
    files = []
    for frame_id in FRAMES:
        np.save(root / f"{frame_id}.npy", np.load(Path(source) / f"{frame_id}.npy"))
        files.append({"path": f"{frame_id}.npy", "id": frame_id})
    _write_index(
        root, scope=scope, key=scope, loader="generic_dense_depth",
        function="depth", meta={"radial_depth": True}, files=files,
    )
    return str(root)


@pytest.fixture
def sky_iou_paths(indexed_sky_prediction, tmp_path):
    paths = dict(indexed_sky_prediction)
    paths["segmentation"] = write_gt_segmentation(tmp_path / "segmentation")
    return paths


def _config(tmp_path, paths, *, gt_extra=None, dataset_extra=None, sky_mask=True):
    gt = {"depth": {"path": paths["gt"]}}
    if "segmentation" in paths:
        gt["segmentation"] = {"path": paths["segmentation"]}
    gt.update(gt_extra or {})
    dataset = {"name": "sky_prediction", "depth": {"path": paths["pred"]}}
    if sky_mask:
        dataset["sky_mask"] = {"path": paths["sky_mask"]}
    dataset.update(dataset_extra or {})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"gt": gt, "datasets": [dataset]}))
    return config_path


def _run(monkeypatch, config_path, *flags, output_dir):
    argv = [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", "--depth-alignment", "none", *flags,
    ]
    monkeypatch.setattr(cli.sys, "argv", argv)
    cli.main()
    return json.loads((Path(output_dir) / "eval.json").read_text())


def _flatten(tree, prefix=""):
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = value
    return flat


def _per_file(saved):
    files = saved["per_file_metrics"]["files"]
    return {entry["id"]: entry["metrics"] for entry in files}


def _without_sky(per_file):
    stripped = copy.deepcopy(per_file)
    for metrics in stripped.values():
        metrics["depth"]["eval"].pop("sky_segmentation", None)
    return stripped


def _declared_axes(saved):
    return {
        name: AxisDeclaration.from_dict(axis)
        for name, axis in saved["metricSet"]["axes"].items()
    }


# ---------------------------------------------------------------------------
# Output and invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sky_depth", [None, 50.0])
def test_sky_iou_adds_keys_without_changing_the_depth_output(
    monkeypatch, capsys, tmp_path, sky_iou_paths, sky_depth
):
    _patch_depth_metrics(monkeypatch)
    paths = sky_iou_paths
    config_path = _config(tmp_path, paths)
    cap = ["--sky-depth", str(sky_depth)] if sky_depth is not None else []

    baseline = _run(monkeypatch, config_path, *cap, output_dir=paths["pred"])
    capsys.readouterr()
    scored = _run(monkeypatch, config_path, *cap, "--sky-iou", output_dir=paths["pred"])
    out = capsys.readouterr().out

    # Flag off: the pre-change values (see the existing sky_mask CLI test),
    # no sky keys, no new meta fields and the usual axes.
    expected_mae = 5.0 if sky_depth is not None else 37.25
    metric_tree = baseline["depth"]["eval"]["metric"]
    assert metric_tree["standard"]["pixel_pool"]["mae"] == expected_mae
    assert "sky_segmentation" not in baseline["depth"]["eval"]
    assert "sky_segmentation" not in baseline["meta"]
    assert "sky_iou" not in baseline["meta"]["eval_params"]
    assert baseline["metricSet"]["axes"] == {
        name: axis.to_dict() for name, axis in cli._depth_eval_axes().items()
    }
    assert "iou" not in baseline["metricSet"]["metricDescriptions"]

    # Flag on: identical depth trees, aggregate and per file.
    assert scored["depth"]["eval"]["metric"] == metric_tree
    assert _without_sky(_per_file(scored)) == _per_file(baseline)
    assert scored["dataset_info"] == baseline["dataset_info"]

    # Explicit: masks at (1, 1) / (0, 0) inside the GT sky rows.
    sky = scored["depth"]["eval"]["sky_segmentation"]
    assert sky["explicit"]["pixel_pool"] == {
        "iou": 0.5, "precision": 1.0, "recall": 0.5,
    }
    assert sky["explicit"]["image_median"]["iou"] == 0.5
    per_file = _per_file(scored)
    assert per_file["first"]["depth"]["eval"]["sky_segmentation"]["explicit"] == {
        "iou": 0.5, "precision": 1.0, "recall": 0.5,
    }
    meta = scored["meta"]["sky_segmentation"]
    assert meta["gt_source"] == "gt.segmentation"
    assert meta["explicit"]["available"] is True
    assert meta["samples"] == 2
    assert meta["gt_sky_fraction"] == 0.5
    assert scored["meta"]["eval_params"]["sky_iou"] is True
    assert "Sky IoU, pixel_pool, 'sky_prediction': explicit 0.5000" in out

    if sky_depth is None:
        assert set(sky) == {"explicit"}
        assert meta["implicit"] == {
            "available": False, "reason": "--sky-depth is not set",
        }
        assert "threshold_m" not in meta
    else:
        # depth >= 50: (1, 0) of "first" (80 m) and (0, 1) of "second".
        assert sky["implicit"]["pixel_pool"] == {
            "iou": 0.5, "precision": 1.0, "recall": 0.5,
        }
        assert meta["threshold_m"] == 50.0
        assert meta["implicit"]["rule"] == "depth >= threshold_m"
        assert "implicit (depth >= 50 m) 0.5000" in out


def test_every_written_key_decomposes_against_the_declared_axes(
    monkeypatch, tmp_path, sky_iou_paths
):
    _patch_depth_metrics(monkeypatch)
    paths = sky_iou_paths
    saved = _run(
        monkeypatch, _config(tmp_path, paths), "--sky-iou", "--sky-depth", "50",
        "--benchmark-depth-range", "1", "80", output_dir=paths["pred"],
    )
    axes = _declared_axes(saved)
    assert axes["space"].optional is True
    assert "sky_segmentation" in axes["category"].values
    assert axes["variant"].values == ("explicit", "implicit")
    assert sorted(axis.position for axis in axes.values()) == list(range(len(axes)))
    descriptions = saved["metricSet"]["metricDescriptions"]
    assert descriptions["iou"]["isHigherBetter"] is True
    assert (descriptions["recall"]["min"], descriptions["recall"]["max"]) == (0, 1)

    names = list(_flatten(saved["depth"], "depth"))
    for metrics in _per_file(saved).values():
        names.extend(_flatten(metrics["depth"], "depth"))
    sky_names = [name for name in names if ".sky_segmentation." in name]
    assert "depth.eval.sky_segmentation.implicit.pixel_pool.iou" in sky_names
    assert "depth.eval.sky_segmentation.explicit.recall" in sky_names  # per file

    for name in names:
        parsed = decompose(name, "depth.eval", axes)
        if name in sky_names:
            assert parsed.axes["space"] is None
            assert parsed.axes["category"] == "sky_segmentation"
            assert parsed.axes["variant"] in {"explicit", "implicit"}
            assert parsed.metric in {"iou", "precision", "recall"}
        else:
            assert parsed.axes["space"] in {"native", "metric"}
            assert parsed.axes["variant"] is None


def test_sky_iou_alone_loads_segmentation_and_sky_mask(
    monkeypatch, tmp_path, sky_iou_paths
):
    _patch_depth_metrics(monkeypatch)
    paths = sky_iou_paths
    config_path = _config(tmp_path, paths)
    loader_calls = []
    builder_calls = []
    real_loader = data._resolve_sky_mask_loader
    real_builder = cli.build_depth_eval_dataset

    def _loader(path, **kwargs):
        loader_calls.append(path)
        return real_loader(path, **kwargs)

    def _builder(**kwargs):
        builder_calls.append(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(data, "_resolve_sky_mask_loader", _loader)
    monkeypatch.setattr(cli, "build_depth_eval_dataset", _builder)

    _run(monkeypatch, config_path, output_dir=paths["pred"])
    assert loader_calls == []
    assert builder_calls[-1]["segmentation_path"] is None
    assert builder_calls[-1]["pred_sky_mask_path"] is None

    _run(monkeypatch, config_path, "--sky-iou", output_dir=paths["pred"])
    assert loader_calls == [paths["segmentation"]]
    assert builder_calls[-1]["segmentation_path"] == paths["segmentation"]
    assert builder_calls[-1]["pred_sky_mask_path"] == paths["sky_mask"]


def test_relative_prediction_without_sky_mask_is_skipped_with_warnings(
    monkeypatch, capsys, tmp_path, sky_iou_paths
):
    _patch_depth_metrics(monkeypatch)
    paths = sky_iou_paths
    relative = write_depth(
        tmp_path / "relative", scope="relative_depth", source=paths["pred"]
    )
    config_path = _config(tmp_path, paths, sky_mask=False)
    config = json.loads(config_path.read_text())
    entry = config["datasets"][0]
    del entry["depth"]
    entry["relative_depth"] = {"path": relative}
    config_path.write_text(json.dumps(config))

    saved = _run(
        monkeypatch, config_path, "--sky-iou", "--sky-depth", "50",
        output_dir=relative,
    )
    out = capsys.readouterr().out

    assert "implicit variant skipped for depth: prediction is not metric" in out
    assert "explicit variant skipped for depth: no sky_mask" in out
    assert "Sky IoU, pixel_pool, 'sky_prediction': no variant computed" in out
    assert "sky_segmentation" not in saved["depth"]["eval"]
    assert all(
        "sky_segmentation" not in metrics["depth"]["eval"]
        for metrics in _per_file(saved).values()
    )
    assert saved["metricSet"]["axes"] == {
        name: axis.to_dict() for name, axis in cli._depth_eval_axes().items()
    }
    meta = saved["meta"]["sky_segmentation"]
    assert meta["explicit"]["available"] is False
    assert meta["implicit"] == {
        "available": False, "reason": "prediction is not metric (relative)",
    }


def test_sky_iou_without_gt_segmentation_is_an_error(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction
):
    config_path = _config(tmp_path, indexed_sky_prediction)
    monkeypatch.setattr(
        cli.sys, "argv", ["euler-eval", str(config_path), "--sky-iou"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    assert "--sky-iou requires gt.segmentation" in capsys.readouterr().err


def test_sky_mask_that_lacks_frames_is_not_loaded_for_iou_alone(
    monkeypatch, capsys, tmp_path, sky_iou_paths
):
    _patch_depth_metrics(monkeypatch)
    paths = dict(sky_iou_paths)
    paths["sky_mask"] = write_sky_mask(tmp_path / "partial_mask", ["first"])
    config_path = _config(tmp_path, paths)

    baseline = _run(monkeypatch, config_path, output_dir=paths["pred"])
    capsys.readouterr()
    scored = _run(monkeypatch, config_path, "--sky-iou", output_dir=paths["pred"])

    assert "the sky_mask of 'sky_prediction' lacks 1 depth pairs" in (
        capsys.readouterr().out
    )
    assert scored["dataset_info"]["num_pairs"] == 2
    assert scored["depth"]["eval"] == baseline["depth"]["eval"]
    assert scored["meta"]["sky_segmentation"]["explicit"] == {
        "available": False, "reason": "sky_mask does not cover every depth pair",
    }
    assert "sky_mask" not in scored["meta"]["pred"]


def test_count_ids_dropped_by_reads_the_pairing_lookups():
    class _Dataset:
        _lookups = {
            "gt": {"a": 0, "b": 0, "c": 0},
            "pred": {"a": 0, "b": 0},
            "pred_sky_mask": {"a": 0, "z": 0},
        }

    assert data.count_ids_dropped_by(_Dataset(), "pred_sky_mask") == 1
    assert data.count_ids_dropped_by(_Dataset(), "missing") is None
    assert data.count_ids_dropped_by(object(), "pred_sky_mask") is None


# ---------------------------------------------------------------------------
# gt.semantic_segmentation on a multi-scope archive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_key", ["semantic_segmentation", "segmentation"])
def test_segmentation_scope_in_a_multi_scope_gt_archive(
    monkeypatch, tmp_path, indexed_sky_prediction, config_key
):
    # The GT depth directory also holds a "semantic_segmentation" scope, as a
    # MUSES archive does. Both config keys must find it.
    _patch_depth_metrics(monkeypatch)
    paths = dict(indexed_sky_prediction)
    write_gt_segmentation(
        paths["gt"], scope="semantic_segmentation", key="semantic_segmentation"
    )
    config_path = _config(
        tmp_path, paths, gt_extra={config_key: {"path": paths["gt"]}}
    )

    saved = _run(monkeypatch, config_path, "--sky-iou", output_dir=paths["pred"])

    sky = saved["depth"]["eval"]["sky_segmentation"]
    assert sky["explicit"]["pixel_pool"]["iou"] == 0.5
    assert saved["meta"]["sky_segmentation"]["gt_source"] == f"gt.{config_key}"


# ---------------------------------------------------------------------------
# Sparse depth serialization
# ---------------------------------------------------------------------------


def test_sparse_depth_writes_sky_iou_under_the_wire_safe_root(monkeypatch, tmp_path):
    for name in ("gt_sparse", "intrinsics", "extrinsics", "segmentation", "pred"):
        (tmp_path / name).mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {
            "sparse_depth": {"path": str(tmp_path / "gt_sparse")},
            "intrinsics": {"path": str(tmp_path / "intrinsics")},
            "camera_extrinsics": {"path": str(tmp_path / "extrinsics")},
            "semantic_segmentation": {"path": str(tmp_path / "segmentation")},
        },
        "datasets": [{"name": "model", "depth": {"path": str(tmp_path / "pred")}}],
    }))
    sky = {"iou": 0.75, "precision": 1.0, "recall": 0.75}
    results = {
        "sparse_depth_native": None,
        "sparse_depth_metric": {"standard": {"image_mean": {"absrel": 0.1}}},
        "sparse_depth_benchmark": None,
        "sky_segmentation": {
            "implicit": {"pixel_pool": sky, "image_mean": sky, "image_median": sky},
        },
        "sky_segmentation_info": {
            "explicit": {
                "available": False, "reason": "no sky_mask for this prediction",
            },
            "implicit": {"available": True, "source": "pred.depth", "samples": 1},
            "threshold_m": 80.0, "domain": "prediction_grid", "samples": 1,
            "samples_without_gt": 0, "gt_sky_fraction": 0.25,
        },
        "dataset_info": {"num_pairs": 1},
        "space_info": {
            "input_space_detected": "metric", "metric_space_source": "native",
            "calibration_mode": "auto_affine", "calibration_applied": False,
            "emitted_spaces": ["metric"], "canonical_space": "metric",
        },
        "spatial_info": {},
        "per_file_metrics": {"files": [{"id": "frame-1", "metrics": {
            "sparse_depth_metric": {"standard": {"absrel": 0.1}},
            "sky_segmentation": {"implicit": sky},
        }}]},
    }
    builder_kwargs = {}
    evaluate_kwargs = {}
    monkeypatch.setattr(
        cli, "build_sparse_depth_eval_dataset",
        lambda **kwargs: builder_kwargs.update(kwargs) or [object()],
    )
    monkeypatch.setattr(
        cli, "get_sparse_depth_metadata", lambda _ds: {"pred_radial_depth": False}
    )
    monkeypatch.setattr(
        cli, "evaluate_sparse_depth_samples",
        lambda **kwargs: evaluate_kwargs.update(kwargs) or copy.deepcopy(results),
    )
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", "--sky-iou", "--sky-depth", "80",
    ])
    cli.main()

    saved = json.loads((tmp_path / "pred" / "eval.json").read_text())
    assert evaluate_kwargs["sky_iou"] is True
    assert evaluate_kwargs["sky_mask_enabled"] is False
    assert builder_kwargs["segmentation_path"] == str(tmp_path / "segmentation")
    assert builder_kwargs["segmentation_modality_key"] == "semantic_segmentation"
    sky_tree = saved["sparsedepth"]["eval"]["sky_segmentation"]
    assert sky_tree["implicit"]["pixel_pool"] == sky
    assert saved["meta"]["sky_segmentation"]["gt_source"] == "gt.semantic_segmentation"
    assert saved["meta"]["sky_segmentation"]["threshold_m"] == 80.0
    file_metrics = saved["per_file_metrics"]["files"][0]["metrics"]
    assert file_metrics["sparsedepth"]["eval"]["sky_segmentation"]["implicit"] == sky

    axes = _declared_axes(saved)
    names = list(_flatten(saved["sparsedepth"], "sparsedepth"))
    names += list(_flatten(file_metrics["sparsedepth"], "sparsedepth"))
    assert "sparsedepth.eval.sky_segmentation.implicit.pixel_pool.iou" in names
    for name in names:
        parsed = decompose(name, "sparsedepth.eval", axes)
        if ".sky_segmentation." in name:
            assert (parsed.axes["category"], parsed.axes["variant"]) == (
                "sky_segmentation", "implicit",
            )
