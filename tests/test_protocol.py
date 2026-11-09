"""The ``protocol`` block: comparison keys for matching evaluations."""

import json
import shutil
from pathlib import Path

import pytest
from ds_crawler import build_dataset_head
from ds_crawler.artifacts import save_output_artifacts

from euler_eval import cli
from euler_eval.cli import default_eval_path
from euler_eval.config_paths import dataset_identity
from euler_eval.protocol import (
    METRICS_REVISION,
    SCHEMA_VERSION,
    build_protocol,
    reference_identities,
)
from tests.test_depth_alignment_output import _patch_depth_metrics
from tests.test_sky_iou_cli import write_gt_segmentation

NO_INPUT = {"dataset": None, "scope": None, "split": None}


def _reference_keys(*roles):
    keys = {"reference.gt.modality"}
    for role in ("gt", *roles):
        keys |= {f"reference.{role}.{field}" for field in ("dataset", "scope", "split")}
    return keys


SPARSE_INPUTS = ("intrinsics", "camera_extrinsics", "lidar_extrinsics", "segmentation")

#: The documented key set of each modality (docs/output.md).
EXPECTED_KEYS = {
    "depth": _reference_keys("calibration", "segmentation") | {
        "population.samples",
        "population.gt_sky_masking",
        "population.missing_segmentation",
        "population.pred_sky_masking",
        "population.evaluated_resolution",
        "population.benchmark_depth_range_m",
        "method.metrics_revision",
        "method.depth_alignment",
        "method.sky_depth_m",
        "method.pred_sky_mask",
        "method.spatial_alignment",
    },
    "sparsedepth": _reference_keys(*SPARSE_INPUTS) | {
        "population.samples",
        "population.gt_sky_masking",
        "population.missing_segmentation",
        "population.pred_sky_masking",
        "population.evaluated_resolution",
        "population.benchmark_depth_range_m",
        "method.metrics_revision",
        "method.depth_alignment",
        "method.sky_depth_m",
        "method.pred_sky_mask",
    },
    "rgb": _reference_keys("input_rgb", "depth", "calibration", "segmentation") | {
        "population.samples",
        "population.gt_sky_masking",
        "population.missing_segmentation",
        "population.evaluated_resolution",
        "population.benchmark_depth_range_m",
        "method.metrics_revision",
        "method.fid_backend",
        "method.spatial_alignment",
        "method.input_alignment",
    },
    "rays": _reference_keys("calibration") | {
        "population.samples",
        "population.evaluated_resolution",
        "method.metrics_revision",
        "method.spatial_alignment",
    },
    "points3d": _reference_keys("intrinsics", "calibration", "segmentation") | {
        "population.samples",
        "population.gt_sky_masking",
        "population.missing_segmentation",
        "population.evaluated_resolution",
        "method.metrics_revision",
        "method.points_3d_alignment",
        "method.spatial_alignment",
    },
    "points3d_sparse": _reference_keys(*SPARSE_INPUTS) | {
        "population.samples",
        "population.gt_sky_masking",
        "population.missing_segmentation",
        "population.evaluated_resolution",
        "method.metrics_revision",
        "method.points_3d_alignment",
    },
}


def _index(root, *, name, key, scope):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    head = build_dataset_head(
        dataset={"name": name},
        modality={"key": key, "meta": {}},
        addons={"euler_loading": {
            "version": "1.0", "loader": "generic_dense_depth", "function": key,
        }},
    )
    save_output_artifacts(root, {
        "contract": {"kind": "dataset_index", "version": "1.0"},
        "head_file": "dataset-head.json", "head": head,
        "generator": {"name": "tests"}, "indexing": {}, "execution": {},
        "index": {"files": [{"path": "a.npy", "id": "a"}]},
    }, metadata_scope=scope)
    return root


# ---------------------------------------------------------------------------
# Identity and block construction
# ---------------------------------------------------------------------------


class TestDatasetIdentity:
    def test_scoped_dataset(self, tmp_path):
        root = _index(tmp_path / "gt", name="MUSES Test", key="depth", scope="depth")
        assert dataset_identity(f"{root}:test", modality_key="depth") == {
            "dataset": "muses_test", "scope": "depth", "split": "test",
        }

    def test_root_level_metadata_has_no_scope(self, tmp_path):
        root = _index(tmp_path / "gt", name="kitti", key="depth", scope=None)
        assert dataset_identity(root, modality_key="depth") == {
            "dataset": "kitti", "scope": None, "split": None,
        }

    def test_without_metadata(self, tmp_path):
        assert dataset_identity(tmp_path, modality_key="depth", split="val") == {
            "dataset": None, "scope": None, "split": "val",
        }

    def test_does_not_depend_on_location(self, tmp_path):
        root = _index(tmp_path / "a" / "gt", name="kitti", key="depth", scope="depth")
        copy = tmp_path / "elsewhere" / "gt_copy"
        shutil.copytree(root, copy)
        assert dataset_identity(root, modality_key="depth") == dataset_identity(
            copy, modality_key="depth"
        )


class TestReferenceIdentities:
    def test_unconfigured_inputs_are_null(self, tmp_path):
        root = _index(tmp_path / "gt", name="kitti", key="depth", scope="depth")
        identities = reference_identities({"depth": {"path": str(root)}})
        assert identities["depth"]["dataset"] == "kitti"
        assert identities["calibration"] == NO_INPUT
        assert identities["segmentation"] == NO_INPUT

    def test_segmentation_alias(self, tmp_path):
        seg = write_gt_segmentation(
            tmp_path / "seg", scope="semantic_segmentation", key="semantic_segmentation"
        )
        identities = reference_identities({"semantic_segmentation": {"path": seg}})
        assert identities["segmentation"] == {
            "dataset": "sky_iou_test", "scope": "semantic_segmentation", "split": None,
        }


class TestBuildProtocol:
    def test_flat_dotted_parameters(self):
        identities = {"rays": {"dataset": "d", "scope": None, "split": "test"},
                      "calibration": dict(NO_INPUT)}
        protocol = build_protocol(
            identities, gt="rays", inputs=("calibration",),
            population={"samples": 3, "evaluated_resolution": (4, 6)},
            method={"spatial_alignment": "none"},
        )
        assert protocol == {
            "schemaVersion": SCHEMA_VERSION,
            "parameters": {
                "reference.gt.modality": "rays",
                "reference.gt.dataset": "d",
                "reference.gt.scope": None,
                "reference.gt.split": "test",
                "reference.calibration.dataset": None,
                "reference.calibration.scope": None,
                "reference.calibration.split": None,
                "population.samples": 3,
                "population.evaluated_resolution": [4, 6],
                "method.metrics_revision": METRICS_REVISION,
                "method.spatial_alignment": "none",
            },
        }

    def test_rejects_nested_values(self):
        with pytest.raises(TypeError, match="method.bad"):
            build_protocol(
                {"rays": dict(NO_INPUT)}, gt="rays", inputs=(),
                population={}, method={"bad": {"nested": 1}},
            )


# ---------------------------------------------------------------------------
# Dense depth, end to end
# ---------------------------------------------------------------------------


def _depth_config(tmp_path, paths):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {
            "depth": {"path": paths["gt"]},
            "segmentation": {"path": paths["segmentation"]},
        },
        "datasets": [{
            "name": "model",
            "depth": {"path": paths["pred"]},
            "sky_mask": {"path": paths["sky_mask"]},
        }],
    }))
    return config_path


@pytest.fixture
def depth_paths(tmp_path, indexed_sky_prediction):
    paths = dict(indexed_sky_prediction)
    paths["segmentation"] = write_gt_segmentation(tmp_path / "segmentation")
    return paths


def _run_depth(monkeypatch, config_path, pred, name, *flags):
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", "--depth-alignment", "none",
        "--eval-name", name, *flags,
    ])
    cli.main()
    out = default_eval_path(pred, modality_key="depth", eval_name=name)
    return json.loads(out.read_text())


def test_depth_parameters(monkeypatch, tmp_path, depth_paths):
    _patch_depth_metrics(monkeypatch)
    saved = _run_depth(
        monkeypatch, _depth_config(tmp_path, depth_paths), depth_paths["pred"], "base"
    )

    protocol = saved["protocol"]
    assert protocol["schemaVersion"] == SCHEMA_VERSION
    parameters = protocol["parameters"]
    assert set(parameters) == EXPECTED_KEYS["depth"]
    assert parameters["reference.gt.modality"] == "depth"
    assert parameters["reference.gt.dataset"] == "sky_test"
    assert parameters["reference.gt.scope"] == "depth"
    assert parameters["reference.gt.split"] is None
    assert parameters["reference.segmentation.dataset"] == "sky_iou_test"
    assert parameters["reference.calibration.dataset"] is None
    assert parameters["population.samples"] == 2
    assert parameters["population.gt_sky_masking"] is False
    assert parameters["population.missing_segmentation"] is None
    assert parameters["population.evaluated_resolution"] == [2, 2]
    assert parameters["population.benchmark_depth_range_m"] is None
    assert parameters["method.metrics_revision"] == METRICS_REVISION
    assert parameters["method.depth_alignment"] == "none"
    assert parameters["method.sky_depth_m"] is None
    assert parameters["method.pred_sky_mask"] is False
    assert parameters["method.spatial_alignment"] == "none"
    # Paths are provenance, not comparison keys.
    assert not any(depth_paths["gt"] in str(value) for value in parameters.values())


@pytest.mark.parametrize(
    ("flags", "changed"),
    [
        (("--depth-alignment", "affine"), {"method.depth_alignment"}),
        # The config has a sky_mask, which both settings then score with.
        (("--sky-depth", "50"), {"method.sky_depth_m", "method.pred_sky_mask"}),
        (
            ("--mask-sky",),
            {"population.gt_sky_masking", "population.missing_segmentation"},
        ),
        # Without --mask-sky, what happens to unsegmented frames changes nothing.
        (("--missing-segmentation", "unmasked"), set()),
        (
            ("--mask-sky-prediction",),
            {"population.pred_sky_masking", "method.pred_sky_mask"},
        ),
        (
            ("--benchmark-depth-range", "1", "80"),
            {"population.benchmark_depth_range_m"},
        ),
        # Settings that only add metrics or affect the run are not parameters.
        (("--sky-iou",), set()),
        (("--distributions",), set()),
        (("--batch-size", "1", "--num-workers", "1"), set()),
    ],
)
def test_a_setting_changes_only_its_own_parameter(
    monkeypatch, tmp_path, depth_paths, flags, changed
):
    _patch_depth_metrics(monkeypatch)
    config_path = _depth_config(tmp_path, depth_paths)
    pred = depth_paths["pred"]
    base = _run_depth(monkeypatch, config_path, pred, "base")["protocol"]["parameters"]
    other = _run_depth(monkeypatch, config_path, pred, "other", *flags)
    other = other["protocol"]["parameters"]

    assert set(other) == set(base)
    assert {key for key in base if base[key] != other[key]} == changed


def test_same_data_elsewhere_has_the_same_parameters(monkeypatch, tmp_path, depth_paths):
    _patch_depth_metrics(monkeypatch)
    here = _run_depth(
        monkeypatch, _depth_config(tmp_path, depth_paths), depth_paths["pred"], "eval"
    )

    moved = {}
    for key, path in depth_paths.items():
        moved[key] = str(tmp_path / "elsewhere" / key)
        shutil.copytree(path, moved[key], ignore=shutil.ignore_patterns(".eval"))
    moved_config = _depth_config(tmp_path / "elsewhere", moved)
    there = _run_depth(monkeypatch, moved_config, moved["pred"], "eval")

    assert there["meta"]["gt"]["path"] != here["meta"]["gt"]["path"]
    assert there["protocol"] == here["protocol"]


# ---------------------------------------------------------------------------
# RGB, rays and dense points_3d (evaluators stubbed)
# ---------------------------------------------------------------------------


class _StubDataset(list):
    def modality_paths(self):
        return {}


def _stub_results(root, metrics):
    return {
        root: metrics,
        "dataset_info": {"num_pairs": 3},
        "spatial_info": {
            "method": "resize",
            "evaluated_dimensions": {"height": 4, "width": 6},
        },
        "per_file_metrics": {},
    }


def test_rgb_rays_and_points_3d_parameters(monkeypatch, tmp_path):
    paths = {}
    for name in ("gt_rgb", "gt_rays", "gt_p3", "pred_rgb", "pred_rays", "pred_p3"):
        paths[name] = tmp_path / name
        paths[name].mkdir()
    calibration = _index(tmp_path / "calib", name="calib", key="calibration",
                         scope="calibration")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {
            "rgb": {"path": str(paths["gt_rgb"])},
            "rays": {"path": f"{paths['gt_rays']}:test"},
            "points_3d": {"path": str(paths["gt_p3"])},
            "calibration": {"path": str(calibration)},
        },
        "datasets": [{
            "name": "model",
            "rgb": {"path": str(paths["pred_rgb"])},
            "rays": {"path": str(paths["pred_rays"])},
            "points_3d": {"path": str(paths["pred_p3"])},
        }],
    }))

    dataset = _StubDataset([object()] * 3)
    for builder in ("build_rgb_eval_dataset", "build_rays_eval_dataset",
                    "build_points_3d_eval_dataset"):
        monkeypatch.setattr(cli, builder, lambda **kwargs: dataset)
    monkeypatch.setattr(cli, "get_rgb_metadata", lambda _ds: {"rgb_range": [0, 1]})
    monkeypatch.setattr(cli, "get_rays_metadata", lambda _ds: {"fov_domain": None})
    monkeypatch.setattr(cli, "get_points_3d_metadata", lambda _ds: {"fov_domain": None})
    monkeypatch.setattr(cli, "evaluate_rgb_samples", lambda **kw: _stub_results(
        "rgb", {"psnr": 30.0}))
    monkeypatch.setattr(cli, "evaluate_rays_samples", lambda **kw: _stub_results(
        "rays", {"angular_error": {"mean_angle": 1.0}}))
    monkeypatch.setattr(cli, "evaluate_points_3d_samples", lambda **kw: {
        **_stub_results("points_3d", None),
        "points_3d_native": {"point_error": {"image_mean": {"mae3d": 0.2}}},
        "space_info": {"canonical_space": "native"},
    })
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", "--rgb-fid-backend", "builtin",
        "--points-3d-alignment", "scale",
    ])
    cli.main()

    def parameters(pred):
        return json.loads((pred / ".eval" / "eval.json").read_text())[
            "protocol"]["parameters"]

    rgb = parameters(paths["pred_rgb"])
    assert set(rgb) == EXPECTED_KEYS["rgb"]
    assert rgb["method.fid_backend"] == "builtin"
    assert rgb["method.spatial_alignment"] == "resize"
    assert rgb["population.evaluated_resolution"] == [4, 6]
    assert rgb["population.samples"] == 3
    assert rgb["reference.calibration.dataset"] == "calib"
    assert rgb["reference.depth.dataset"] is None
    assert rgb["reference.input_rgb.dataset"] is None
    assert rgb["method.input_alignment"] is None

    rays = parameters(paths["pred_rays"])
    assert set(rays) == EXPECTED_KEYS["rays"]
    assert rays["reference.gt.split"] == "test"
    assert rays["reference.calibration.scope"] == "calibration"

    points_3d = parameters(paths["pred_p3"])
    assert set(points_3d) == EXPECTED_KEYS["points3d"]
    assert points_3d["reference.gt.modality"] == "points_3d"
    assert points_3d["method.points_3d_alignment"] == "scale"
