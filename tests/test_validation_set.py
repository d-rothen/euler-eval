"""Validation sets: in-loop validation judged against the CLI (the decisive test).

The CLI evaluates synthetic predictions on disk; a :class:`DepthValidationSet`
over the same GT scores the same predictions in memory. The contract engine
of euler-metric-naming must then read ``same protocol`` for every cell the
pass writes, with equal values, and ``compared on N shared files`` with equal
recomputed values for a pass on a subset of the items.
"""

import copy
import json
import math

import numpy as np
import pytest
from euler_metric_naming.contract import (
    check_protocol,
    compare_files,
    file_evidence,
    load_eval_file,
    load_validation_pass,
    recompute_cell,
    validation_pass_problems,
    validation_set_problems,
)

from euler_eval import cli
from euler_eval.cli import default_eval_path
from euler_eval.validation import DepthValidationSet
from tests.test_depth_alignment_output import _patch_depth_metrics
from tests.test_protocol import FRAMES, _synthetic_depth, write_depth_dataset


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _cli(monkeypatch, tmp_path, gt_path, frames, name, *flags):
    pred_path = write_depth_dataset(tmp_path / f"pred_{name}", frames, name=name)
    config = tmp_path / f"config_{name}.json"
    config.write_text(json.dumps({
        "gt": {"depth": {"path": gt_path}},
        "datasets": [{"name": name, "depth": {"path": pred_path}}],
    }))
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", *flags,
    ])
    cli.main()
    out = default_eval_path(pred_path, modality_key="depth", eval_name="eval")
    return json.loads(out.read_text())


def _frame_key(sample):
    return (sample["full_id"].strip("/").split("/")[0], sample["id"])


def _score(validation_set, frames, *, masks=None):
    validation_pass = validation_set.new_pass()
    for index in validation_set.indices:
        sample = validation_set.dataset[index]
        key = _frame_key(sample)
        if key in frames:
            mask = None if masks is None else masks.get(key)
            validation_pass.add(sample, frames[key], valid_mask=mask)
    return validation_pass


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    _patch_depth_metrics(monkeypatch)
    gt, pred = _synthetic_depth()
    gt_path = write_depth_dataset(tmp_path / "gt", gt, name="synthetic_gt")
    return {"gt": gt, "pred": pred, "gt_path": gt_path}


def _assert_valid(validation_set, record):
    document = validation_set.declaration()
    assert validation_set_problems(document) == []
    check_protocol(document["protocol"])
    assert validation_pass_problems(document, record) == []
    return document


# -- the decisive test --------------------------------------------------------------------------


def test_in_loop_validation_reads_same_protocol_with_equal_values(
    monkeypatch, tmp_path, synthetic
):
    saved = _cli(
        monkeypatch, tmp_path, synthetic["gt_path"], synthetic["pred"], "full",
        "--depth-alignment", "none", "--benchmark-depth-range", "1", "80",
    )
    evaluated = load_eval_file(saved)
    validation_set = DepthValidationSet(
        "synthetic", depth_path=synthetic["gt_path"], benchmark_depth_range=(1, 80)
    )
    record = _score(validation_set, synthetic["pred"]).record(step=100, epoch=1)
    document = _assert_valid(validation_set, record)
    passed = load_validation_pass(document, record)

    assert "population" not in record  # the pass scored exactly the set's items
    assert passed.item_ids == evaluated.item_ids
    names = sorted(record["metrics"])
    assert len(names) == 3 * 10 * 5  # 3 reductions x 10 metrics x (no bin + 4 bins)
    for name in names:
        assert name in evaluated.metrics, name
        verdict = compare_files(passed, evaluated, name)
        assert verdict.label == "same protocol", (name, verdict.label)
        assert _close(record["metrics"][name], evaluated.metrics[name]), name
    # per-item values are the CLI's per-file values
    for item, values in evaluated.per_file.items():
        for name, value in values.items():
            if name in passed.per_file[item]:
                assert _close(passed.per_file[item][name], value), (item, name)


def test_a_pass_on_a_subset_is_compared_on_the_shared_files(monkeypatch, tmp_path, synthetic):
    saved = _cli(
        monkeypatch, tmp_path, synthetic["gt_path"], synthetic["pred"], "full",
        "--depth-alignment", "none", "--benchmark-depth-range", "1", "80",
    )
    evaluated = load_eval_file(saved)
    validation_set = DepthValidationSet(
        "synthetic", depth_path=synthetic["gt_path"], benchmark_depth_range=(1, 80)
    )
    subset = {key: synthetic["pred"][key] for key in FRAMES[:3]}
    record = _score(validation_set, subset).record(step=200)
    document = _assert_valid(validation_set, record)
    passed = load_validation_pass(document, record)
    assert record["population"]["ids"] == ["scene-1~f0", "scene-1~f1", "scene-2~f0"]

    shared = file_evidence(passed)[0] & file_evidence(evaluated)[0]
    paired = 0
    for name in sorted(record["metrics"]):
        verdict = compare_files(passed, evaluated, name)
        if verdict.tier == "paired":
            assert verdict.label == "compared on 3 shared files", name
            on_pass = recompute_cell(passed.contract, passed, name, shared)
            on_eval = recompute_cell(evaluated.contract, evaluated, name, shared)
            assert _close(on_pass, on_eval), (name, on_pass, on_eval)
            assert _close(on_pass, record["metrics"][name]), name
            paired += 1
        else:  # pooled quantities without a pool form (silog)
            assert ".pixel_pool." in name and name.endswith(".silog"), name
            assert verdict.label == "comparable · computed on different files: 3 of 4 shared"
    assert paired == 3 * 10 * 5 - 5


def test_affine_alignment_matches_the_cli(monkeypatch, tmp_path, synthetic):
    saved = _cli(
        monkeypatch, tmp_path, synthetic["gt_path"], synthetic["pred"], "affine",
        "--depth-alignment", "affine",
    )
    evaluated = load_eval_file(saved)
    validation_set = DepthValidationSet(
        "synthetic", depth_path=synthetic["gt_path"], alignment="affine"
    )
    record = _score(validation_set, synthetic["pred"]).record(step=300)
    document = _assert_valid(validation_set, record)
    passed = load_validation_pass(document, record)
    assert record["parameters"]["depth.alignment.applied"] is True
    for name in sorted(record["metrics"]):
        assert name.startswith("depth.eval.metric.standard."), name
        verdict = compare_files(passed, evaluated, name)
        assert verdict.label == "same protocol", (name, verdict.label)
        assert _close(record["metrics"][name], evaluated.metrics[name]), name


def test_a_depth_cap_matches_the_cli(monkeypatch, tmp_path, synthetic):
    saved = _cli(
        monkeypatch, tmp_path, synthetic["gt_path"], synthetic["pred"], "capped",
        "--depth-alignment", "none", "--sky-depth", "40",
    )
    evaluated = load_eval_file(saved)
    validation_set = DepthValidationSet("synthetic", depth_path=synthetic["gt_path"], sky_depth=40)
    record = _score(validation_set, synthetic["pred"]).record(step=400)
    document = _assert_valid(validation_set, record)
    passed = load_validation_pass(document, record)
    assert record["observations"]["depth.cap.clipped"] == pytest.approx(
        evaluated.contract["observations"]["depth.cap.clipped"]
    )
    assert record["observations"]["depth.cap.clipped"] > 0
    for name in sorted(record["metrics"]):
        verdict = compare_files(passed, evaluated, name)
        assert verdict.label == "same protocol", (name, verdict.label)
        assert _close(record["metrics"][name], evaluated.metrics[name]), name


# -- what the in-loop path does differently is recorded -------------------------------------------


def test_in_loop_only_masking_is_not_comparable(monkeypatch, tmp_path, synthetic):
    saved = _cli(
        monkeypatch, tmp_path, synthetic["gt_path"], synthetic["pred"], "full",
        "--depth-alignment", "none",
    )
    evaluated = load_eval_file(saved)
    name = "depth.eval.metric.standard.image_mean.absrel"

    bounded = DepthValidationSet(
        "bounded", depth_path=synthetic["gt_path"], valid_depth_range=(None, 50)
    )
    record = _score(bounded, synthetic["pred"]).record(step=1)
    passed = load_validation_pass(_assert_valid(bounded, record), record)
    verdict = compare_files(passed, evaluated, name)
    assert verdict.label == "not comparable · GT validity range differs"
    assert verdict.reasons[0].hint == "score both with the same GT validity range, or none"

    plain = DepthValidationSet("plain", depth_path=synthetic["gt_path"])
    masks = {key: np.ones_like(depth, dtype=bool) for key, depth in synthetic["gt"].items()}
    masks[FRAMES[0]][:2] = False
    record = _score(plain, synthetic["pred"], masks=masks).record(step=1)
    assert record["parameters"]["depth.extra_mask"] is True
    passed = load_validation_pass(_assert_valid(plain, record), record)
    assert compare_files(passed, evaluated, name).label == (
        "not comparable · extra validity mask differs"
    )


def test_two_passes_of_a_set_are_judged_like_evaluations(synthetic):
    validation_set = DepthValidationSet("synthetic", depth_path=synthetic["gt_path"])
    first = _score(validation_set, synthetic["pred"]).record(step=1)
    worse = {key: depth * 1.1 for key, depth in synthetic["pred"].items()}
    second = _score(validation_set, worse).record(step=2)
    document = validation_set.declaration()
    name = "depth.eval.metric.standard.image_mean.rmse"
    verdict = compare_files(
        load_validation_pass(document, first), load_validation_pass(document, second), name
    )
    assert verdict.label == "same protocol"
    assert second["metrics"][name] != first["metrics"][name]


# -- the set document and the record ---------------------------------------------------------------


def test_the_set_document(synthetic):
    validation_set = DepthValidationSet(
        "synthetic", depth_path=synthetic["gt_path"], label="synthetic, 4 frames",
        benchmark_depth_range=(1, 80), indices=[0, 1, 2],
    )
    document = validation_set.declaration()
    assert document["kind"] == "euler.validation_set"
    assert document["schemaVersion"] == "1.0"
    assert document["label"] == "synthetic, 4 frames"
    assert document["metricSet"]["metricNamespace"] == "depth.eval"
    protocol = document["protocol"]
    assert protocol["task"] == "depth.dense"
    assert protocol["items"] == {"ids": ["scene-1~f0", "scene-1~f1", "scene-2~f0"], "series": {}}
    assert protocol["population"]["count"] == 3
    for key in ("spatial.method", "depth.alignment.applied", "depth.extra_mask", "domain.region"):
        assert key in protocol["descriptors"], key
        assert key not in protocol["parameters"], key
    assert protocol["parameters"]["depth.gt.valid_range_m"] is None
    assert protocol["parameters"]["depth.cap.max_m"] is None
    assert "depth.gt_range" in protocol["stratifications"]
    assert set(protocol["metrics"]) >= {"absrel", "rmse", "delta1", "count"}
    assert document == validation_set.declaration()  # stable


def test_per_item_values_are_optional(synthetic):
    validation_set = DepthValidationSet(
        "synthetic", depth_path=synthetic["gt_path"], per_item=False
    )
    validation_pass = _score(validation_set, synthetic["pred"])
    assert "per_item" not in validation_pass.record(step=1)
    with_items = validation_pass.record(step=1, per_item=True)
    assert len(with_items["per_item"]["depth.eval.metric.standard.absrel"]) == 4
    assert with_items["series"]["depth.eval.metric.standard.count"][0] > 0


def test_processes_merge_into_one_pass(synthetic):
    validation_set = DepthValidationSet(
        "synthetic", depth_path=synthetic["gt_path"], benchmark_depth_range=(1, 80)
    )
    single = _score(validation_set, synthetic["pred"]).record(step=1, wall_time=0.0)
    halves = [validation_set.new_pass(), validation_set.new_pass()]
    for index in validation_set.indices:
        sample = validation_set.dataset[index]
        halves[index % 2].add(sample, synthetic["pred"][_frame_key(sample)])
    state = json.loads(json.dumps(halves[1].state()))
    merged = halves[0].merge(state).record(step=1, wall_time=0.0)
    assert merged.keys() == single.keys()
    for name, value in single["metrics"].items():
        assert _close(merged["metrics"][name], value), name
    assert merged["per_item"] == single["per_item"]
    assert merged["parameters"] == single["parameters"]


def test_a_sample_is_scored_once_per_pass(synthetic):
    validation_set = DepthValidationSet("synthetic", depth_path=synthetic["gt_path"])
    validation_pass = validation_set.new_pass()
    sample = validation_set.dataset[0]
    validation_pass.add(sample, synthetic["pred"][_frame_key(sample)])
    with pytest.raises(ValueError, match="already scored"):
        validation_pass.add(sample, synthetic["pred"][_frame_key(sample)])
    with pytest.raises(ValueError, match="at least one"):
        validation_set.new_pass().record(step=1)


def test_settings_from_an_evaluation(monkeypatch, tmp_path, synthetic):
    saved = _cli(
        monkeypatch, tmp_path, synthetic["gt_path"], synthetic["pred"], "full",
        "--depth-alignment", "none", "--benchmark-depth-range", "1", "80",
        "--sky-depth", "60",
    )
    validation_set = DepthValidationSet.from_protocol(
        "synthetic", saved, depth_path=synthetic["gt_path"]
    )
    assert validation_set.alignment == "none"
    assert validation_set.sky_depth == 60
    assert validation_set.benchmark_depth_range == (1, 80)
    masked = copy.deepcopy(saved)
    masked["protocol"]["parameters"]["sky.gt.masking"] = True
    with pytest.raises(ValueError, match="GT sky masking"):
        DepthValidationSet.from_protocol("synthetic", masked, depth_path=synthetic["gt_path"])


def test_bad_settings_are_refused(synthetic):
    with pytest.raises(ValueError, match="key"):
        DepthValidationSet("Bad Key", depth_path=synthetic["gt_path"])
    with pytest.raises(ValueError, match="exactly one"):
        DepthValidationSet("synthetic")
    with pytest.raises(ValueError, match="alignment"):
        DepthValidationSet("synthetic", depth_path=synthetic["gt_path"], alignment="scale")


# -- sparse GT ------------------------------------------------------------------------------------


class _SparseStubDataset:
    """A GT-only dataset of point clouds, shaped like euler-loading's samples."""

    def __init__(self, clouds):
        self._clouds = clouds

    def __len__(self):
        return len(self._clouds)

    def sample_full_id(self, index):
        return f"/scene/f{index}"

    def __getitem__(self, index):
        return {"sparse_depth": self._clouds[index], "id": f"f{index}", "full_id": f"/scene/f{index}"}


def test_a_sparse_set_scores_projected_points(tmp_path):
    from tests.test_validation import _synthetic_cloud_and_pred

    points, K, pred, _ = _synthetic_cloud_and_pred()
    dataset = _SparseStubDataset([points, points * 1.05])
    roots = {}
    for name in ("lidar", "intrinsics", "extrinsics"):
        roots[name] = str(tmp_path / name)
        (tmp_path / name).mkdir()
    validation_set = DepthValidationSet(
        "sparse", sparse_depth_path=roots["lidar"], intrinsics_path=roots["intrinsics"],
        camera_extrinsics_path=roots["extrinsics"], dataset=dataset,
        benchmark_depth_range=(1, 50),
    )
    validation_pass = validation_set.new_pass()
    for index in range(len(dataset)):
        validation_pass.add(dataset[index], pred, intrinsics=K, camera_extrinsics=np.eye(4))
    record = validation_pass.record(step=10)
    document = _assert_valid(validation_set, record)
    assert document["protocol"]["task"] == "depth.sparse"
    assert document["metricSet"]["metricNamespace"] == "sparsedepth.eval"
    assert document["protocol"]["parameters"]["depth.conversion"] == "planar_to_radial"
    assert record["observations"]["domain.grid"] == [48, 64]
    assert record["parameters"]["spatial.method"] == "pointcloud_projection"
    # the first frame is consistent with the prediction, the second is not
    absrel = record["per_item"]["sparsedepth.eval.metric.standard.absrel"]
    assert absrel[0] == pytest.approx(0.0, abs=1e-5) and absrel[1] > 0.01
    passed = load_validation_pass(document, record)
    again = load_validation_pass(document, validation_pass.record(step=20))
    name = "sparsedepth.eval.metric.standard.image_mean.absrel"
    assert compare_files(passed, again, name).label == "same protocol"
