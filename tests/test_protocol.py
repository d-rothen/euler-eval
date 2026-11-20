"""The ``protocol`` block: the evaluation-protocol contract v2.

euler-eval writes euler-metric-naming's contract (``euler_metric_naming.contract``)
into ``protocol``. These tests check what it records, that every block is
valid, that consumers can recompute aggregates from the per-item values, and
that the verdicts of the contract come out of real evaluations.
"""

import copy
import importlib.util
import json
import math
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pytest
from ds_crawler import build_dataset_head
from ds_crawler.artifacts import save_output_artifacts
from euler_metric_naming.contract import (
    check_protocol,
    compare_files,
    definition_key,
    file_evidence,
    load_eval_file,
    manifest_hash,
    recompute_cell,
)

from euler_eval import cli
from euler_eval.cli import default_eval_path
from euler_eval.config_paths import dataset_identity
from euler_eval.protocol import (
    POLICIES,
    SCHEMA_VERSION,
    canonical_head,
    item_ids,
    reference_identities,
)
from tests.test_depth_alignment_output import _patch_depth_metrics
from tests.test_sky_iou_cli import write_gt_segmentation

NO_INPUT = {"dataset": None, "scope": None, "split": None}

_DEPTH_KEYS = {
    "depth.alignment.mode",
    "depth.alignment.applied",
    "depth.alignment.fit",
    "depth.alignment.fit_trim",
    "depth.alignment.fit_max_gt_percentile",
    "depth.input_space",
    "depth.cap.max_m",
    "depth.benchmark.range_m",
    "depth.invalid.policy",
    "depth.conversion",
    "sky.gt.masking",
    "sky.gt.missing_policy",
    "sky.pred.masking",
    "sky.pred.fill",
    "sky.iou",
    "spatial.policy",
    "spatial.method",
    "spatial.uniform",
}
_POINTS_3D_KEYS = {
    "points3d.alignment.mode",
    "points3d.alignment.applied",
    "points3d.f_a.threshold_m",
    "rays.fov_policy",
    "rays.rho_a.threshold_deg",
    "sky.gt.masking",
    "sky.gt.missing_policy",
    "spatial.policy",
    "spatial.method",
    "spatial.uniform",
    "spatial.mask_resample",
}

#: The documented parameter keys of each modality (docs/output.md).
EXPECTED_KEYS = {
    "depth": _DEPTH_KEYS | {"spatial.resize_target", "spatial.mask_resample"},
    "sparsedepth": _DEPTH_KEYS | {"spatial.mask_resample"},
    "rgb": {
        "sky.gt.masking",
        "sky.gt.missing_policy",
        "spatial.policy",
        "spatial.method",
        "spatial.uniform",
        "spatial.resize_target",
        "spatial.mask_resample",
        "spatial.gt_resample",
        "rgb.fid.backend",
        "rgb.input.alignment",
        "rgb.input.alignment_policy",
        "rgb.depth_bins.policy",
        "rgb.depth_bins.edges_m",
        "depth.benchmark.range_m",
        "depth.invalid.policy",
    },
    "rays": {
        "spatial.policy",
        "spatial.method",
        "spatial.uniform",
        "spatial.resize_target",
        "spatial.gt_resample",
        "rays.fov_policy",
        "rays.rho_a.threshold_deg",
    },
    "points3d": _POINTS_3D_KEYS | {"spatial.resize_target"},
    "points3d_sparse": _POINTS_3D_KEYS,
}

FIXTURES = (
    Path(importlib.util.find_spec("euler_metric_naming").origin).parent
    / "contract"
    / "fixtures"
)
PROTOCOL_SCHEMA = FIXTURES.parent / "schema" / "protocol-2.schema.json"


def assert_valid_protocol(protocol):
    """The block passes the contract's validator and, when available, its schema."""
    check_protocol(protocol)
    assert protocol["schemaVersion"] == SCHEMA_VERSION
    described = set(protocol.get("descriptors", {}))
    keys = (
        set(protocol["parameters"])
        | set(protocol.get("observations", {}))
        | {f"reference.{role}" for role in protocol.get("references", {})}
    )
    assert keys <= described, keys - described
    try:
        import jsonschema
    except ImportError:
        return
    schema = json.loads(PROTOCOL_SCHEMA.read_text())
    jsonschema.Draft202012Validator(schema).validate(protocol)


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


def write_depth_dataset(root, frames, *, name="synthetic", meta=None, attributes=None):
    """Index ``{(scene, frame_id): depth}`` as a dense depth dataset."""
    root = Path(root)
    root.mkdir(parents=True)
    tree: dict = {}
    for (scene, frame_id), depth in frames.items():
        path = f"{scene}/{frame_id}.npy"
        (root / scene).mkdir(exist_ok=True)
        np.save(root / path, np.asarray(depth, dtype=np.float32))
        node = tree.setdefault("children", {}).setdefault(scene, {"files": []})
        node["files"].append({"path": path, "id": frame_id})
    dataset = {"name": name}
    if attributes:
        dataset["attributes"] = attributes
    head = build_dataset_head(
        dataset=dataset,
        modality={"key": "depth", "meta": {"radial_depth": True, **(meta or {})}},
        addons={"euler_loading": {
            "version": "1.0", "loader": "generic_dense_depth", "function": "depth",
        }},
    )
    save_output_artifacts(root, {
        "contract": {"kind": "dataset_index", "version": "1.0"},
        "head_file": "dataset-head.json", "head": head,
        "generator": {"name": "tests"}, "indexing": {}, "execution": {},
        "index": tree,
    }, metadata_scope="depth")
    return str(root)


# ---------------------------------------------------------------------------
# Identity helpers
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
        copy_root = tmp_path / "elsewhere" / "gt_copy"
        shutil.copytree(root, copy_root)
        assert dataset_identity(root, modality_key="depth") == dataset_identity(
            copy_root, modality_key="depth"
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


def test_canonical_head_drops_paths_and_timestamps():
    head = {
        "dataset": {"id": "d", "created_at": "2026-01-01"},
        "modality": {"key": "depth", "meta": {"source_path": "/x", "crop": {"boxes": []}}},
        "addons": {"run": {"path": "/tmp/x", "timestamp": 3, "version": "1"}},
    }
    assert canonical_head(head) == {
        "dataset": {"id": "d"},
        "modality": {"key": "depth", "meta": {"crop": {"boxes": []}}},
        "addons": {"run": {"version": "1"}},
    }


def test_item_ids_are_tree_paths_in_nfc():
    decomposed = unicodedata.normalize("NFD", "café")
    tree = {
        "children": {"scene": {"files": [{"id": decomposed, "metrics": {}}]}},
        "files": [{"id": "top", "metrics": {}}],
    }
    assert item_ids(tree) == ["scene~café", "top"]


# ---------------------------------------------------------------------------
# Dense depth, end to end
# ---------------------------------------------------------------------------


def _depth_config(tmp_path, paths, *, sky_mask=True):
    config_path = tmp_path / "config.json"
    dataset = {"name": "model", "depth": {"path": paths["pred"]}}
    if sky_mask:
        dataset["sky_mask"] = {"path": paths["sky_mask"]}
    config_path.write_text(json.dumps({
        "gt": {
            "depth": {"path": paths["gt"]},
            "segmentation": {"path": paths["segmentation"]},
        },
        "datasets": [dataset],
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


def test_depth_protocol(monkeypatch, tmp_path, depth_paths):
    _patch_depth_metrics(monkeypatch)
    saved = _run_depth(
        monkeypatch, _depth_config(tmp_path, depth_paths), depth_paths["pred"], "base"
    )

    protocol = saved["protocol"]
    assert_valid_protocol(protocol)
    assert protocol["task"] == "depth.dense"
    assert protocol["vocabularies"] == {
        "euler.core": "1.0", "euler.depth": "1.1", "euler.sky": "1.0",
    }
    parameters = protocol["parameters"]
    assert set(parameters) == EXPECTED_KEYS["depth"]
    assert parameters["depth.alignment.mode"] == "none"
    assert parameters["depth.alignment.applied"] is False
    assert parameters["depth.cap.max_m"] is None
    assert parameters["sky.gt.masking"] is False
    assert parameters["sky.pred.fill"] is False
    assert parameters["spatial.method"] == "none"
    assert parameters["spatial.uniform"] is True
    for key, value in POLICIES.items():
        if key in parameters:
            assert parameters[key] == value

    # Only inputs the evaluation read are references: segmentation is not
    # configured to be read without --mask-sky / --sky-iou.
    references = protocol["references"]
    assert set(references) == {"gt"}
    gt = references["gt"]
    assert gt["modality"] == "depth"
    assert gt["dataset"] == {
        "id": "sky_test", "variant": None, "split": None, "scope": "depth",
    }
    assert gt["manifest"] == {"hash": manifest_hash(["first", "second"]), "count": 2}
    assert gt["identity"].startswith("sha256:")
    assert "view" not in gt

    population = protocol["population"]
    ids = item_ids(saved["per_file_metrics"])
    assert population["count"] == 2
    assert population["manifest"] == manifest_hash(ids)
    assert population["idScheme"] == "euler.item_id/1"
    # The sky_mask is not read without --sky-depth / --mask-sky-prediction.
    assert population["accounting"] == {"reference": 2, "prediction": 2, "paired": 2}
    assert protocol["items"]["ids"] == ids

    observations = protocol["observations"]
    assert observations["domain.region"] == {"frame": gt["identity"], "box": [0, 0, 2, 2]}
    assert observations["prediction.resample"] == [1, 1]
    assert observations["depth.cap.clipped"] == 0.0
    assert protocol["axes"]["space"] == {"role": "space"}
    assert protocol["axes"]["reduction"] == {"role": "reduction"}
    assert "absrel" in protocol["metrics"]
    # Paths are provenance, not part of the protocol.
    assert depth_paths["gt"] not in json.dumps(protocol)


def test_segmentation_is_a_reference_when_read(monkeypatch, tmp_path, depth_paths):
    _patch_depth_metrics(monkeypatch)
    saved = _run_depth(
        monkeypatch, _depth_config(tmp_path, depth_paths), depth_paths["pred"],
        "masked", "--mask-sky",
    )
    protocol = saved["protocol"]
    assert_valid_protocol(protocol)
    segmentation = protocol["references"]["segmentation"]
    assert segmentation["dataset"]["id"] == "sky_iou_test"
    assert segmentation["manifest"]["count"] == 2
    assert protocol["parameters"]["sky.gt.masking"] is True
    assert protocol["parameters"]["sky.gt.missing_policy"] == "drop"
    assert protocol["observations"]["sky.gt.unsegmented"] == 0
    assert protocol["population"]["accounting"]["dropped"]["segmentation"] == 0


@pytest.mark.parametrize(
    ("flags", "changed"),
    [
        (
            ("--depth-alignment", "affine"),
            {
                "depth.alignment.mode",
                "depth.alignment.applied",
                "depth.alignment.fit",
                "depth.alignment.fit_trim",
            },
        ),
        # The config has a sky_mask, which both settings then score with.
        (("--sky-depth", "50"), {"depth.cap.max_m", "sky.pred.fill"}),
        (("--mask-sky",), {"sky.gt.masking", "sky.gt.missing_policy"}),
        # Without --mask-sky, what happens to unsegmented frames changes nothing.
        (("--missing-segmentation", "unmasked"), set()),
        (("--mask-sky-prediction",), {"sky.pred.masking", "sky.pred.fill"}),
        (("--benchmark-depth-range", "1", "80"), {"depth.benchmark.range_m"}),
        (("--sky-iou",), {"sky.iou"}),
        # Settings that only affect the run are not parameters.
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
    assert_valid_protocol(other["protocol"])
    other = other["protocol"]["parameters"]

    assert set(other) == set(base)
    assert {key for key in base if base[key] != other[key]} == changed


def test_same_data_elsewhere_has_the_same_protocol(monkeypatch, tmp_path, depth_paths):
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
    assert definition_key(load_eval_file(there).contract) == definition_key(
        load_eval_file(here).contract
    )


def test_an_inert_depth_cap_is_comparable(monkeypatch, tmp_path, depth_paths):
    """depth.cap.max_m is inert when it clipped nothing (inertIf on both sides)."""
    _patch_depth_metrics(monkeypatch)
    config_path = _depth_config(tmp_path, depth_paths, sky_mask=False)
    pred = depth_paths["pred"]
    uncapped = load_eval_file(_run_depth(monkeypatch, config_path, pred, "uncapped"))
    loose = load_eval_file(
        _run_depth(monkeypatch, config_path, pred, "loose", "--sky-depth", "1000")
    )
    tight = load_eval_file(
        _run_depth(monkeypatch, config_path, pred, "tight", "--sky-depth", "50")
    )
    metric = "depth.eval.metric.standard.image_mean.absrel"

    assert loose.contract["observations"]["depth.cap.clipped"] == 0.0
    assert compare_files(uncapped, loose, metric).label == "same protocol"
    # The GT reaches 100 m: a 50 m cap clips it.
    assert tight.contract["observations"]["depth.cap.clipped"] > 0
    verdict = compare_files(uncapped, tight, metric)
    assert verdict.label == "not comparable · depth cap differs"


# ---------------------------------------------------------------------------
# Per-item values: strata, weights and recomputed aggregates
# ---------------------------------------------------------------------------

FRAMES = [("scene-1", "f0"), ("scene-1", "f1"), ("scene-2", "f0"), ("scene-2", "f1")]


def _synthetic_depth(shape=(12, 12)):
    rng = np.random.default_rng(7)
    gt = {key: rng.uniform(2.0, 70.0, size=shape) for key in FRAMES}
    pred = {
        key: depth * rng.uniform(0.7, 1.3, size=shape) for key, depth in gt.items()
    }
    return gt, pred


def _run_synthetic(monkeypatch, tmp_path, gt_path, pred_frames, name, *flags):
    pred_path = write_depth_dataset(tmp_path / f"pred_{name}", pred_frames, name=name)
    config_path = tmp_path / f"config_{name}.json"
    config_path.write_text(json.dumps({
        "gt": {"depth": {"path": gt_path}},
        "datasets": [{"name": name, "depth": {"path": pred_path}}],
    }))
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", "--depth-alignment", "none",
        "--benchmark-depth-range", "1", "80", *flags,
    ])
    cli.main()
    out = default_eval_path(pred_path, modality_key="depth", eval_name="eval")
    return json.loads(out.read_text())


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    _patch_depth_metrics(monkeypatch)
    gt, pred = _synthetic_depth()
    gt_path = write_depth_dataset(tmp_path / "gt", gt, name="synthetic_gt")

    def run(name, frames=None, *flags):
        return _run_synthetic(
            monkeypatch, tmp_path, gt_path, pred if frames is None else frames,
            name, *flags,
        )

    run.gt = gt
    run.pred = pred
    return run


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def test_per_file_strata_and_weights_recompute_every_aggregate(synthetic):
    saved = synthetic("full")
    assert_valid_protocol(saved["protocol"])
    loaded = load_eval_file(saved)
    contract = loaded.contract

    ids = ["scene-1~f0", "scene-1~f1", "scene-2~f0", "scene-2~f1"]
    assert contract["items"]["ids"] == ids
    series = contract["items"]["series"]
    for bin_name in ("", "all.", "near.", "mid.", "far."):
        assert f"depth.eval.metric.standard.{bin_name}count" in series
    # Each item's bin counts add up to its benchmark (all) count.
    for index in range(len(ids)):
        parts = sum(
            series[f"depth.eval.metric.standard.{b}.count"][index]
            for b in ("near", "mid", "far")
        )
        assert parts == series["depth.eval.metric.standard.all.count"][index]

    per_file = loaded.per_file["scene-1~f0"]
    for bin_name in ("all", "near", "mid", "far"):
        assert f"depth.eval.metric.standard.{bin_name}.absrel" in per_file

    checked = 0
    for reduction in ("image_mean", "image_median", "pixel_pool"):
        for bin_name in ("", "all.", "near.", "mid.", "far."):
            for base in ("absrel", "sqrel", "mae", "rmse", "rmse_log", "log10",
                         "delta1", "delta2", "delta3", "silog"):
                name = f"depth.eval.metric.standard.{reduction}.{bin_name}{base}"
                if name not in loaded.metrics:
                    continue
                pool = contract["metrics"][base]["pool"]["form"]
                if reduction == "pixel_pool" and pool == "none":
                    assert recompute_cell(contract, loaded, name) is None
                    continue
                got = recompute_cell(contract, loaded, name)
                assert got is not None, name
                assert _close(got, loaded.metrics[name]), (name, got, loaded.metrics[name])
                checked += 1
    assert checked >= 3 * 5 * 9 - 5  # every poolable cell of every bin


def test_cropped_prediction_is_comparable_with_a_domain_difference(synthetic):
    full = load_eval_file(synthetic("full"))
    # A latent model's 8x8 crop of the 12x12 GT grid (vae_crop under auto).
    cropped = load_eval_file(
        synthetic("crop", {key: depth[:8, :8] for key, depth in synthetic.pred.items()})
    )
    assert cropped.contract["parameters"]["spatial.method"] == "vae_crop"
    assert cropped.contract["observations"]["domain.region"]["box"] == [0, 0, 8, 8]

    verdict = compare_files(full, cropped, "depth.eval.metric.standard.image_mean.absrel")
    assert verdict.tier == "comparable"
    # 1 - IoU(12x12, 8x8) = 1 - 64/144
    assert verdict.label == "comparable · domain differs by 55.6 %"
    assert verdict.codes == ["domain"]


def test_a_subset_of_files_is_compared_on_the_shared_files(synthetic):
    full_saved = synthetic("full")
    subset_frames = {key: synthetic.pred[key] for key in FRAMES[:3]}
    subset_saved = synthetic("subset", subset_frames)
    full, subset = load_eval_file(full_saved), load_eval_file(subset_saved)
    assert subset.contract["population"]["count"] == 3

    for name in (
        "depth.eval.metric.standard.image_mean.absrel",
        "depth.eval.metric.standard.image_median.rmse",
        "depth.eval.metric.standard.pixel_pool.absrel",
        "depth.eval.metric.standard.pixel_pool.near.rmse",
        "depth.eval.metric.standard.image_mean.far.delta1",
        # No reduction segment: the definition's implicit reducer (image_mean).
        "depth.eval.metric.image_quality.psnr",
    ):
        verdict = compare_files(full, subset, name)
        assert verdict.tier == "paired", (name, verdict.label)
        assert verdict.label == "compared on 3 shared files"
        assert verdict.shared == 3
        shared = file_evidence(full)[0] & file_evidence(subset)[0]
        on_full = recompute_cell(full.contract, full, name, shared)
        on_subset = recompute_cell(subset.contract, subset, name, shared)
        assert _close(on_full, on_subset), name
        assert _close(on_subset, subset.metrics[name]), name

    # Dataset-level and pooled-quantile metrics cannot be recomputed.
    for name in (
        "depth.eval.metric.image_quality.fid",
        "depth.eval.metric.standard.pixel_pool.silog",
        "depth.eval.metric.depth_metrics.absrel.median",
    ):
        verdict = compare_files(full, subset, name)
        assert verdict.label == "comparable · computed on different files: 3 of 4 shared"


def _crop_head_meta(top, left, height, width, source=(20, 20)):
    box = {
        "top": top, "left": left, "height": height, "width": width,
        "source_height": source[0], "source_width": source[1],
        "output_height": height, "output_width": width,
    }
    return {"camera": "cam", "crop": {"reference": "depth", "spec": {}, "boxes": [box]}}


def test_crops_of_one_source_are_comparable_with_a_region_difference(
    monkeypatch, tmp_path
):
    _patch_depth_metrics(monkeypatch)
    rng = np.random.default_rng(3)
    source = {key: rng.uniform(2.0, 70.0, size=(20, 20)) for key in FRAMES}
    runs = {}
    for name, (top, left) in {"a": (2, 2), "b": (4, 4)}.items():
        gt = {key: depth[top:top + 12, left:left + 12] for key, depth in source.items()}
        gt_path = write_depth_dataset(
            tmp_path / f"gt_{name}", gt, name="source",
            meta=_crop_head_meta(top, left, 12, 12),
            attributes={"variant": f"crop_{name}"},
        )
        pred = {key: depth * 1.1 for key, depth in gt.items()}
        runs[name] = load_eval_file(
            _run_synthetic(monkeypatch, tmp_path, gt_path, pred, name)
        )

    reference = runs["a"].contract["references"]["gt"]
    assert reference["dataset"]["variant"] == "crop_a"
    assert reference["view"] == {"source": "source/cam@20x20", "region": [2, 2, 12, 12]}
    assert runs["a"].contract["observations"]["domain.region"] == {
        "frame": "source/cam@20x20", "box": [2, 2, 12, 12],
    }
    assert reference["identity"] != runs["b"].contract["references"]["gt"]["identity"]

    verdict = compare_files(
        runs["a"], runs["b"], "depth.eval.metric.standard.image_mean.absrel"
    )
    # IoU of two 12x12 boxes offset by (2, 2): 100 / 188.
    assert verdict.label == "comparable · GT region differs by 46.8 %"
    assert verdict.codes == ["domain", "domain"]
    assert definition_key(runs["a"].contract) == definition_key(runs["b"].contract)


class _RgbDataset(list):
    def modality_paths(self):
        return {"gt": "gt", "pred": "pred", "gt_depth": "gt_depth"}


class _ConstantLPIPS:
    def __init__(self, device="cpu"):
        pass

    def compute(self, prediction, ground_truth):
        return float(np.mean(np.abs(prediction - ground_truth)))


class _ConstantFID:
    def __init__(self, device="cpu"):
        pass

    def compute_rgb_fid(self, *args, **kwargs):
        return 1.0


def _rgb_samples():
    rng = np.random.default_rng(11)
    samples = []
    for scene, frame_id in FRAMES:
        gt = rng.uniform(0.0, 1.0, size=(32, 32, 3)).astype(np.float32)
        noise = rng.normal(0.0, 0.05, size=gt.shape).astype(np.float32)
        samples.append({
            "id": frame_id,
            "full_id": f"/{scene}/{frame_id}",
            "gt": gt,
            "pred": np.clip(gt + noise, 0.0, 1.0),
            "gt_depth": rng.uniform(0.5, 60.0, size=(32, 32)).astype(np.float32),
        })
    return samples


def test_rgb_on_a_subset_of_files_is_compared_on_the_shared_files(
    monkeypatch, tmp_path
):
    import euler_eval.evaluate as eval_mod

    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _ConstantLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _ConstantFID)
    monkeypatch.setattr(eval_mod, "tqdm", lambda iterable, *args, **kwargs: iterable)
    monkeypatch.setattr(cli, "get_rgb_metadata", lambda ds: {"rgb_range": [0, 1]})
    monkeypatch.setattr(cli, "get_depth_metadata", lambda ds, modality="gt": {
        "radial_depth": True, "sparse": False, "invalid_sentinel": None,
    })
    samples = _rgb_samples()

    def run(name, chosen):
        pred = tmp_path / f"pred_{name}"
        pred.mkdir()
        for gt_dir in ("gt_rgb", "gt_depth"):
            (tmp_path / gt_dir).mkdir(exist_ok=True)
        config_path = tmp_path / f"config_{name}.json"
        config_path.write_text(json.dumps({
            "gt": {
                "rgb": {"path": str(tmp_path / "gt_rgb")},
                "depth": {"path": str(tmp_path / "gt_depth")},
            },
            "datasets": [{"name": name, "rgb": {"path": str(pred)}}],
        }))
        monkeypatch.setattr(
            cli, "build_rgb_eval_dataset", lambda **kw: _RgbDataset(chosen)
        )
        monkeypatch.setattr(cli.sys, "argv", [
            "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
            "--no-sanity-check", "--benchmark-depth-range", "1", "80",
        ])
        cli.main()
        saved = json.loads((pred / ".eval" / "eval.json").read_text())
        assert_valid_protocol(saved["protocol"])
        return saved

    full_saved = run("full", samples)
    subset_saved = run("subset", samples[:3])
    full, subset = load_eval_file(full_saved), load_eval_file(subset_saved)

    bins = full.contract["stratifications"]["rgb.depth_bins"]
    assert bins["edges"] == [0, 1, 5, None]
    assert full.contract["metrics"]["psnr"]["reducer"] == "image_mean"
    assert full.contract["metrics"]["fid"]["reducer"] == "dataset"

    shared = file_evidence(full)[0] & file_evidence(subset)[0]
    assert len(shared) == 3
    for name in (
        "rgb.eval.image_quality.psnr",
        "rgb.eval.image_quality.ssim",
        "rgb.eval.near.mae",
        "rgb.eval.all.mse",
        "rgb.eval.depth_binned_photometric.mae.far",
        "rgb.eval.high_frequency.relative_diff",
    ):
        verdict = compare_files(full, subset, name)
        assert verdict.label == "compared on 3 shared files", (name, verdict.label)
        assert verdict.shared == 3
        # The per-item values reproduce each file's aggregate, and both files
        # agree on the shared files.
        for loaded in (full, subset):
            got = recompute_cell(loaded.contract, loaded, name)
            assert got == pytest.approx(loaded.metrics[name], rel=1e-9), name
        assert recompute_cell(full.contract, full, name, shared) == pytest.approx(
            recompute_cell(subset.contract, subset, name, shared), rel=1e-9
        ), name

    for name in ("rgb.eval.image_quality.fid", "rgb.eval.edge_f1.f1"):
        verdict = compare_files(full, subset, name)
        assert verdict.label == (
            "comparable · computed on different files: 3 of 4 shared"
        ), name


# ---------------------------------------------------------------------------
# Conformance with euler-metric-naming's fixtures
# ---------------------------------------------------------------------------

VERDICTS = json.loads((FIXTURES / "verdicts.json").read_text())["cases"]


@pytest.mark.parametrize("case", VERDICTS, ids=[case["name"] for case in VERDICTS])
def test_contract_fixture_verdicts(case):
    file_a = load_eval_file(FIXTURES / "evals" / case["a"])
    file_b = load_eval_file(FIXTURES / "evals" / case["b"])
    verdict = compare_files(file_a, file_b, case["metric"])
    assert verdict.tier == case["tier"]
    assert verdict.label == case["label"]
    assert verdict.codes == case["codes"]
    assert verdict.shared == case["shared"]


# ---------------------------------------------------------------------------
# RGB, rays and dense points_3d (evaluators stubbed)
# ---------------------------------------------------------------------------


class _StubDataset(list):
    def modality_paths(self):
        return {"gt": "gt", "pred": "pred"}

    def hierarchical_modality_paths(self):
        return {"calibration": "calibration"}


def _stub_results(root, metrics):
    return {
        root: metrics,
        "dataset_info": {"num_pairs": 3, "threshold_deg": 20.0},
        "spatial_info": {
            "method": "resize",
            "evaluated_dimensions": {"height": 4, "width": 6},
            "gt_resample": [0.5, 0.5],
            "uniform": True,
        },
        "per_file_metrics": {
            "files": [{"id": f"f{i}", "metrics": {root: metrics}} for i in range(3)]
        },
    }


def test_rgb_rays_and_points_3d_protocols(monkeypatch, tmp_path):
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
        "rgb", {"image_quality": {"psnr": 30.0}}))
    monkeypatch.setattr(cli, "evaluate_rays_samples", lambda **kw: _stub_results(
        "rays", {"angular_error": {"mean_angle": 1.0}}))
    monkeypatch.setattr(cli, "evaluate_points_3d_samples", lambda **kw: {
        **_stub_results("points_3d", None),
        "points_3d_native": {"point_error": {"image_mean": {"mae3d": 0.2}}},
        "space_info": {"canonical_space": "native", "calibration_applied": True},
    })
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", "--rgb-fid-backend", "builtin",
        "--points-3d-alignment", "scale",
    ])
    cli.main()

    def protocol(pred):
        saved = json.loads((pred / ".eval" / "eval.json").read_text())
        assert_valid_protocol(saved["protocol"])
        return saved["protocol"]

    rgb = protocol(paths["pred_rgb"])
    assert rgb["task"] == "rgb.restoration"
    assert set(rgb["parameters"]) == EXPECTED_KEYS["rgb"]
    assert rgb["parameters"]["rgb.fid.backend"] == "builtin"
    assert rgb["parameters"]["spatial.method"] == "resize"
    assert rgb["parameters"]["spatial.gt_resample"] == [0.5, 0.5]
    assert rgb["parameters"]["rgb.depth_bins.policy"] == "v3"
    assert rgb["parameters"]["rgb.input.alignment"] is None
    assert rgb["observations"]["domain.grid"] == [4, 6]
    assert rgb["references"]["calibration"]["dataset"]["id"] == "calib"
    assert set(rgb["references"]) == {"gt", "calibration"}
    assert rgb["population"]["count"] == 3

    rays = protocol(paths["pred_rays"])
    assert rays["task"] == "rays.directions"
    assert set(rays["parameters"]) == EXPECTED_KEYS["rays"]
    assert rays["references"]["gt"]["dataset"]["split"] == "test"
    assert rays["references"]["calibration"]["dataset"]["scope"] == "calibration"
    assert rays["parameters"]["rays.rho_a.threshold_deg"] == 20.0

    points_3d = protocol(paths["pred_p3"])
    assert points_3d["task"] == "points3d.dense"
    assert set(points_3d["parameters"]) == EXPECTED_KEYS["points3d"]
    assert points_3d["references"]["gt"]["modality"] == "points_3d"
    assert points_3d["parameters"]["points3d.alignment.mode"] == "scale"
    assert points_3d["parameters"]["points3d.alignment.applied"] is True


def test_rgb_input_only_records_no_gt(monkeypatch, tmp_path):
    hazy = _index(tmp_path / "hazy", name="hazy_inputs", key="rgb", scope="rgb")
    pred = tmp_path / "pred"
    pred.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {"input_rgb": {"path": str(hazy)}},
        "datasets": [{"name": "model", "rgb": {"path": str(pred)}}],
    }))

    class _Stub(list):
        def modality_paths(self):
            return {"input": "input", "pred": "pred"}

    monkeypatch.setattr(cli, "build_rgb_eval_dataset", lambda **kw: _Stub([1, 2]))
    monkeypatch.setattr(cli, "get_rgb_metadata", lambda _ds: {"rgb_range": [0, 1]})
    monkeypatch.setattr(cli, "evaluate_rgb_samples", lambda **kw: {
        "rgb": {"dehazing": {"hautiere_e": 0.5}},
        "dataset_info": {"num_pairs": 2},
        "spatial_info": {"method": "none", "input_alignment": "crop_input"},
        "per_file_metrics": {"files": [
            {"id": "a", "metrics": {"rgb": {"dehazing": {"hautiere_e": 0.5}}}},
            {"id": "b", "metrics": {"rgb": {"dehazing": {"hautiere_e": 0.5}}}},
        ]},
    })
    monkeypatch.setattr(cli, "check_metric_set_requirements", lambda sets: None)
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--no-sanity-check",
        "--domain", "dehazing",
    ])
    cli.main()

    protocol = json.loads((pred / ".eval" / "eval.json").read_text())["protocol"]
    assert_valid_protocol(protocol)
    assert set(protocol["references"]) == {"input"}
    assert protocol["references"]["input"]["dataset"]["id"] == "hazy_inputs"
    assert protocol["references"]["input"]["modality"] == "input_rgb"
    assert protocol["parameters"]["rgb.input.alignment"] == "crop_input"
    assert protocol["parameters"]["rgb.fid.backend"] is None
    assert protocol["parameters"]["spatial.method"] is None
    assert "spatial.gt_resample" not in protocol["parameters"]
    # hautiere_* is a vocabulary metric of euler.rgb.
    assert protocol["metrics"]["hautiere_e"]["vocabulary"] == "euler.rgb@1.0"


def test_protocol_block_survives_a_v1_reader():
    """Everything new is inside protocol; parameters stay flat scalars."""
    saved = json.loads((FIXTURES / "evals" / "d1.json").read_text())
    protocol = copy.deepcopy(saved["protocol"])
    for value in protocol["parameters"].values():
        assert value is None or isinstance(value, (bool, int, float, str, list))


def test_points_3d_pixel_pool_recomputes_from_per_item_counts(monkeypatch, tmp_path):
    rng = np.random.default_rng(5)
    samples = []
    for index, valid in enumerate((64, 40, 12)):
        gt = rng.uniform(1.0, 20.0, size=(8, 8, 3)).astype(np.float32)
        pred = gt + rng.normal(0.0, 0.3, size=gt.shape).astype(np.float32)
        gt.reshape(-1, 3)[valid:] = np.nan  # fewer valid points per image
        samples.append({
            "id": f"f{index}", "full_id": f"/scene/f{index}", "gt": gt, "pred": pred,
        })
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {"points_3d": {"path": str(tmp_path)}},
        "datasets": [{"name": "model", "points_3d": {"path": str(pred_dir)}}],
    }))
    monkeypatch.setattr(cli, "build_points_3d_eval_dataset", lambda **kw: samples)
    monkeypatch.setattr(cli, "get_points_3d_metadata", lambda ds: {"fov_domain": "sfov"})
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check",
    ])
    cli.main()

    saved = json.loads((pred_dir / ".eval" / "eval.json").read_text())
    assert_valid_protocol(saved["protocol"])
    loaded = load_eval_file(saved)
    counts = loaded.contract["items"]["series"]["points3d.eval.native.point_error.count"]
    assert counts == [64, 40, 12]
    for name in (
        "points3d.eval.native.point_error.image_mean.mae3d",
        "points3d.eval.native.point_error.image_median.rmse3d",
        "points3d.eval.native.point_error.pixel_pool.mae3d",
        "points3d.eval.native.error_decomposition.radial_mae",
        "points3d.eval.native.cloud_distance.chamfer.distance",
    ):
        got = recompute_cell(loaded.contract, loaded, name)
        assert got == pytest.approx(loaded.metrics[name], rel=1e-6), name
