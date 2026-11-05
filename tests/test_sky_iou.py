"""Tests for the opt-in sky segmentation IoU (``--sky-iou``)."""

import copy
import json

import numpy as np
import pytest

import euler_eval.evaluate as eval_mod
from euler_eval.metrics import (
    append_sky_iou_store,
    implicit_sky_from_depth,
    init_sky_iou_store,
    sky_iou_counts,
    sky_iou_from_counts,
    summarize_sky_iou_store,
)
from tests.test_depth_alignment_output import _DummyDepthDataset, _patch_depth_metrics
from tests.test_sparse_depth import _OneSampleSparseDataset

T, F = True, False


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------


def test_counts_and_scores_on_hand_made_masks():
    pred = np.array([[T, T, F], [T, F, F], [F, F, F]])
    gt = np.array([[T, T, T], [F, F, F], [F, F, F]])

    counts = sky_iou_counts(pred, gt)

    assert counts == {
        "intersection": 2, "union": 4, "pred_sky": 3, "gt_sky": 3, "pixels": 9,
    }
    scores = sky_iou_from_counts(counts)
    assert scores["iou"] == pytest.approx(0.5, abs=1e-9)
    assert scores["precision"] == pytest.approx(2 / 3, abs=1e-9)
    assert scores["recall"] == pytest.approx(2 / 3, abs=1e-9)


def test_counts_respect_a_domain():
    pred = np.array([[T, T, F], [T, F, F], [F, F, F]])
    gt = np.array([[T, T, T], [F, F, F], [F, F, F]])
    domain = np.ones((3, 3), dtype=bool)
    domain[0, 2] = False

    counts = sky_iou_counts(pred, gt, domain)

    assert counts == {
        "intersection": 2, "union": 3, "pred_sky": 3, "gt_sky": 2, "pixels": 8,
    }


def test_counts_reject_mismatched_grids():
    with pytest.raises(ValueError, match="share a grid"):
        sky_iou_counts(np.zeros((2, 2), bool), np.zeros((2, 3), bool))


def test_empty_union_is_undefined_and_only_adds_zeros_to_the_pool():
    store = init_sky_iou_store()
    empty = append_sky_iou_store(
        store, sky_iou_counts(np.zeros((4, 4), bool), np.zeros((4, 4), bool))
    )
    assert empty == {"iou": None, "precision": None, "recall": None}

    defined = append_sky_iou_store(
        store,
        sky_iou_counts(np.array([[T, F], [F, F]]), np.array([[T, T], [F, F]])),
    )
    assert defined == {"iou": 0.5, "precision": 1.0, "recall": 0.5}

    # The undefined sample is left out of the image reducers ...
    assert store["per_image"] == {"iou": [0.5], "precision": [1.0], "recall": [0.5]}
    assert store["samples"] == 2
    assert store["samples_undefined_iou"] == 1
    # ... but still adds (zero) counts and its pixels to the pooled sums.
    assert store["pool"] == {
        "intersection": 1, "union": 2, "pred_sky": 1, "gt_sky": 2, "pixels": 20,
    }
    summary = summarize_sky_iou_store(store)
    for reduction in ("pixel_pool", "image_mean", "image_median"):
        assert summary[reduction] == {"iou": 0.5, "precision": 1.0, "recall": 0.5}


def test_all_undefined_store_summarizes_to_nan():
    store = init_sky_iou_store()
    append_sky_iou_store(
        store, sky_iou_counts(np.zeros((2, 2), bool), np.zeros((2, 2), bool))
    )
    summary = summarize_sky_iou_store(store)
    for reduction in summary.values():
        assert all(np.isnan(value) for value in reduction.values())


def test_pixel_pool_is_ratio_of_sums_over_samples_of_different_sizes():
    store = init_sky_iou_store()
    # 2x2: I=1, U=2, T_p=2, T_g=1.
    append_sky_iou_store(
        store,
        sky_iou_counts(np.array([[T, T], [F, F]]), np.array([[T, F], [F, F]])),
    )
    # 3x4: I=3, U=8, T_p=3, T_g=8.
    pred = np.zeros((3, 4), dtype=bool)
    pred[0, :3] = True
    gt = np.zeros((3, 4), dtype=bool)
    gt[:2, :] = True
    append_sky_iou_store(store, sky_iou_counts(pred, gt))

    summary = summarize_sky_iou_store(store)

    pooled = summary["pixel_pool"]
    assert pooled["iou"] == pytest.approx((1 + 3) / (2 + 8), abs=1e-9)
    assert pooled["precision"] == pytest.approx((1 + 3) / (2 + 3), abs=1e-9)
    assert pooled["recall"] == pytest.approx((1 + 3) / (1 + 8), abs=1e-9)
    mean = summary["image_mean"]
    assert mean["iou"] == pytest.approx((1 / 2 + 3 / 8) / 2, abs=1e-9)
    assert mean["precision"] == pytest.approx((1 / 2 + 1) / 2, abs=1e-9)
    assert mean["recall"] == pytest.approx((1 + 3 / 8) / 2, abs=1e-9)
    assert summary["image_median"]["iou"] == pytest.approx(
        (1 / 2 + 3 / 8) / 2, abs=1e-9
    )
    # Pooled IoU weights by sky area, so it differs from the image mean.
    assert pooled["iou"] != pytest.approx(mean["iou"])


def test_implicit_sky_uses_greater_or_equal_and_handles_non_finite_values():
    c = 80.0
    depth = np.array(
        [
            [c, c + 1.0, c - 1e-3],
            [np.inf, np.nan, 0.0],
            [-1.0, -np.inf, 1e9],
        ],
        dtype=np.float32,
    )

    sky = implicit_sky_from_depth(depth, c)

    assert sky.dtype == bool
    np.testing.assert_array_equal(
        sky,
        [[T, T, F], [T, F, F], [F, F, T]],
    )


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.inf, np.nan])
def test_implicit_sky_requires_a_finite_positive_threshold(threshold):
    with pytest.raises(ValueError, match="finite positive"):
        implicit_sky_from_depth(np.ones((2, 2)), threshold)


def test_explicit_mask_and_gt_are_resized_with_nearest_neighbour():
    # 4x4 -> 2x2 nearest keeps the top-left pixel of each 2x2 block; an
    # area or linear rule would round block (0, 0) (1/4 sky) to no sky and
    # block (1, 1) (3/4 sky) to sky.
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True  # block (0, 0): only its top-left pixel
    mask[0:2, 2:4] = True  # block (0, 1): all sky
    mask[2:4, 2:4] = True
    mask[2, 2] = False  # block (1, 1): sky except its top-left pixel
    nearest = np.array([[T, T], [F, F]])
    assert np.array_equal(mask[::2, ::2], nearest)

    tracker = eval_mod._SkyIoU(sky_depth=None)
    gt = np.array([[T, F], [F, T]])
    values = tracker.add(
        {
            "id": "a",
            "pred_sky_mask": mask,
            "segmentation": np.kron(gt, np.ones((2, 2))),
        },
        np.ones((2, 2), dtype=np.float32),
        None,
    )

    # P = [[T, T], [F, F]] against G = [[T, F], [F, T]]: I = 1, U = 3.
    assert values == {
        "explicit": {"iou": pytest.approx(1 / 3), "precision": 0.5, "recall": 0.5}
    }
    assert tracker.stores["explicit"]["pool"]["pixels"] == 4


# ---------------------------------------------------------------------------
# GT extraction
# ---------------------------------------------------------------------------


def test_gt_sky_is_not_inverted_and_is_paired_by_sample_id():
    first = np.array([[T, F], [F, F]])
    second = np.array([[F, F], [F, T]])
    sample = {"id": "second", "segmentation": {"first": first, "second": second}}

    np.testing.assert_array_equal(eval_mod._get_gt_sky_mask(sample), second)
    # --mask-sky keeps its historical first-entry behaviour, inverted.
    np.testing.assert_array_equal(eval_mod._get_sky_mask(sample), ~first)


def test_gt_sky_uses_a_single_shared_entry_and_rejects_ambiguous_ones():
    mask = np.array([[T, F], [F, F]])
    assert np.array_equal(
        eval_mod._get_gt_sky_mask({"id": "x", "segmentation": {"scene": mask}}), mask
    )
    ambiguous = {"id": "x", "segmentation": {"a": mask, "b": mask}}
    assert eval_mod._get_gt_sky_mask(ambiguous) is None
    assert eval_mod._get_gt_sky_mask({"id": "x", "segmentation": {}}) is None
    assert eval_mod._get_gt_sky_mask({"id": "x"}) is None


# ---------------------------------------------------------------------------
# Dense depth evaluator
# ---------------------------------------------------------------------------

_GT_SKY = np.array([[T, T], [F, F]])


def _sample(
    *, pred=None, sky_mask=True, segmentation=True, sample_id="frame0", gt=None
):
    sample = {
        "id": sample_id,
        "gt": np.array([[100, 90], [30, 40]], dtype=np.float32) if gt is None else gt,
        "pred": (
            np.array([[60, 10], [50, 20]], dtype=np.float32) if pred is None else pred
        ),
    }
    if sky_mask is True:
        sample["pred_sky_mask"] = np.array([[T, T], [F, F]])
    elif sky_mask is not None:
        sample["pred_sky_mask"] = sky_mask
    if segmentation is True:
        sample["segmentation"] = {sample_id: _GT_SKY}
    elif segmentation is not None:
        sample["segmentation"] = segmentation
    return sample


def _evaluate(samples, **kwargs):
    kwargs.setdefault("alignment_mode", "none")
    return eval_mod.evaluate_depth_samples(
        _DummyDepthDataset(samples), is_radial=True, device="cpu", num_workers=0,
        **kwargs,
    )


def _strip_sky_iou(result):
    """Drop the --sky-iou additions so the remaining tree can be compared."""
    result = copy.deepcopy(result)
    result.pop("sky_segmentation", None)
    result.pop("sky_segmentation_info", None)

    def _walk(node):
        for entry in node.get("files", []):
            entry["metrics"].pop("sky_segmentation", None)
        for child in node.get("children", {}).values():
            _walk(child)

    _walk(result["per_file_metrics"])
    return result


def _canonical(tree) -> str:
    return json.dumps(tree, sort_keys=True, default=str)


def test_dense_evaluator_reports_both_variants_with_per_file_values(monkeypatch):
    _patch_depth_metrics(monkeypatch)
    result = _evaluate([_sample()], sky_depth=50.0, sky_iou=True)

    summary = result["sky_segmentation"]
    # Explicit mask equals GT sky.
    assert summary["explicit"]["pixel_pool"] == {
        "iou": 1.0, "precision": 1.0, "recall": 1.0,
    }
    # Implicit sky is depth >= 50 before capping: [[T, F], [T, F]] vs
    # [[T, T], [F, F]] gives I = 1, U = 3, T_p = 2, T_g = 2.
    implicit = summary["implicit"]["pixel_pool"]
    assert implicit["iou"] == pytest.approx(1 / 3, abs=1e-9)
    assert implicit["precision"] == pytest.approx(0.5, abs=1e-9)
    assert implicit["recall"] == pytest.approx(0.5, abs=1e-9)
    assert set(summary["implicit"]) == {"pixel_pool", "image_mean", "image_median"}

    per_file = result["per_file_metrics"]["files"][0]["metrics"]["sky_segmentation"]
    assert per_file["explicit"] == {"iou": 1.0, "precision": 1.0, "recall": 1.0}
    assert per_file["implicit"]["iou"] == pytest.approx(1 / 3)

    info = result["sky_segmentation_info"]
    assert info["explicit"]["available"] is True
    assert info["explicit"]["source"] == "pred.sky_mask"
    assert info["implicit"]["available"] is True
    assert info["implicit"]["rule"] == "depth >= threshold_m"
    assert info["threshold_m"] == 50.0
    assert info["domain"] == "prediction_grid"
    assert info["samples"] == 1
    assert info["samples_without_gt"] == 0
    assert info["gt_sky_fraction"] == 0.5
    assert info["explicit"]["pred_sky_fraction"] == 0.5
    assert info["implicit"]["pred_sky_fraction"] == 0.5


def test_value_at_the_cap_counts_as_implicit_sky(monkeypatch):
    _patch_depth_metrics(monkeypatch)
    pred = np.array([[50, 50], [10, 10]], dtype=np.float32)
    result = _evaluate(
        [_sample(pred=pred, sky_mask=None)], sky_depth=50.0, sky_iou=True
    )
    assert result["sky_segmentation"]["implicit"]["pixel_pool"]["iou"] == 1.0


def test_explicit_and_implicit_stay_separate_when_both_sources_exist(monkeypatch):
    # A sky_mask does not replace the depth-threshold variant, nor the other
    # way round: each variant reads only its own source.
    _patch_depth_metrics(monkeypatch)
    far = np.array([[200, 200], [200, 200]], dtype=np.float32)
    with_mask = _evaluate([_sample(pred=far)], sky_depth=50.0, sky_iou=True)
    without_mask = _evaluate(
        [_sample(pred=far, sky_mask=None)], sky_depth=50.0, sky_iou=True
    )

    assert with_mask["sky_segmentation"]["explicit"]["pixel_pool"]["iou"] == 1.0
    assert with_mask["sky_segmentation"]["implicit"]["pixel_pool"]["iou"] == 0.5
    assert "explicit" not in without_mask["sky_segmentation"]
    assert (
        without_mask["sky_segmentation"]["implicit"]
        == with_mask["sky_segmentation"]["implicit"]
    )


def test_implicit_reads_the_prediction_before_calibration(monkeypatch):
    # With --depth-alignment affine a metric prediction is still scored on its
    # raw values; calibration would move the 49 m pixel past the threshold.
    _patch_depth_metrics(monkeypatch)
    gt = np.array([[100, 100], [10, 10]], dtype=np.float32)
    pred = np.array([[49, 49], [5, 5]], dtype=np.float32)
    result = _evaluate(
        [_sample(pred=pred, gt=gt, sky_mask=None)],
        sky_depth=50.0, sky_iou=True, alignment_mode="affine",
    )
    assert result["space_info"]["calibration_applied"] is True
    implicit = result["sky_segmentation_info"]["implicit"]
    assert implicit["available"] is True
    assert implicit["pred_sky_fraction"] == 0.0


@pytest.mark.parametrize(
    "hint, reason",
    [("relative", "prediction is not metric (relative)"),
     ("affine", "prediction is not metric (affine)")],
)
def test_implicit_is_skipped_for_non_metric_predictions(
    monkeypatch, capsys, hint, reason
):
    _patch_depth_metrics(monkeypatch)
    result = _evaluate(
        [_sample(sky_mask=None)], sky_depth=50.0, sky_iou=True, input_space_hint=hint
    )

    assert result["sky_segmentation"] == {}
    assert result["sky_segmentation_info"]["implicit"] == {
        "available": False, "reason": reason,
    }
    assert result["sky_segmentation_info"]["threshold_m"] is None
    assert "sky_segmentation" not in result["per_file_metrics"]["files"][0]["metrics"]
    out = capsys.readouterr().out
    assert f"implicit variant skipped for depth: {reason}" in out
    assert "explicit variant skipped for depth: no sky_mask" in out


def test_without_sky_depth_only_the_explicit_variant_runs(monkeypatch, capsys):
    _patch_depth_metrics(monkeypatch)
    result = _evaluate([_sample()], sky_iou=True)

    assert set(result["sky_segmentation"]) == {"explicit"}
    assert result["sky_segmentation_info"]["implicit"] == {
        "available": False, "reason": "--sky-depth is not set",
    }
    assert "implicit variant skipped for depth: --sky-depth is not set" in (
        capsys.readouterr().out
    )


def test_warns_when_gt_has_no_sky(monkeypatch, capsys):
    _patch_depth_metrics(monkeypatch)
    no_sky = {"frame0": np.zeros((2, 2), dtype=bool)}
    result = _evaluate([_sample(segmentation=no_sky)], sky_iou=True)

    out = capsys.readouterr().out
    assert "GT segmentation has no sky in any of the 1 depth samples" in out
    assert "'skyclass'" in out
    assert result["sky_segmentation_info"]["gt_sky_fraction"] == 0.0


def test_warns_when_implicit_sky_is_empty(monkeypatch, capsys):
    _patch_depth_metrics(monkeypatch)
    near = np.array([[10, 10], [10, 10]], dtype=np.float32)
    result = _evaluate(
        [_sample(pred=near, sky_mask=None)], sky_depth=250.0, sky_iou=True
    )

    assert "no depth prediction reaches --sky-depth 250 m" in capsys.readouterr().out
    implicit = result["sky_segmentation"]["implicit"]
    assert implicit["pixel_pool"]["iou"] == 0.0
    assert implicit["pixel_pool"]["recall"] == 0.0
    assert np.isnan(implicit["pixel_pool"]["precision"])


def test_samples_without_their_own_gt_are_left_out_with_a_warning(
    monkeypatch, capsys
):
    _patch_depth_metrics(monkeypatch)
    paired = _sample(sample_id="a")
    unpaired = _sample(
        sample_id="b", segmentation={"x": _GT_SKY, "y": _GT_SKY}
    )
    result = _evaluate([paired, unpaired], sky_iou=True)

    info = result["sky_segmentation_info"]
    assert info["samples"] == 2
    assert info["samples_without_gt"] == 1
    assert info["explicit"]["samples"] == 1
    assert "1 of 2 depth samples have no GT segmentation of their own frame" in (
        capsys.readouterr().out
    )


_INVARIANCE_CASES = {
    "metric, no mask, no cap": dict(sky_mask=None),
    "metric, mask, no cap": dict(),
    "metric, mask, cap": dict(sky_depth=50.0),
    "metric, cap, sky masking": dict(sky_depth=50.0, sky_mask_enabled=True),
    "affine alignment, mask, cap": dict(sky_depth=50.0, alignment_mode="affine"),
    "relative, mask, cap": dict(
        sky_depth=50.0, alignment_mode="auto_affine", input_space_hint="relative"
    ),
    "benchmark range": dict(sky_depth=50.0, benchmark_depth_range=(1.0, 80.0)),
}


@pytest.mark.parametrize("case", sorted(_INVARIANCE_CASES))
def test_dense_depth_tree_is_identical_with_and_without_sky_iou(monkeypatch, case):
    _patch_depth_metrics(monkeypatch)
    options = dict(_INVARIANCE_CASES[case])
    with_mask = options.pop("sky_mask", True)
    samples = [
        _sample(sky_mask=with_mask, sample_id="a"),
        _sample(
            sky_mask=with_mask, sample_id="b",
            pred=np.array([[90, 5], [55, 25]], dtype=np.float32),
        ),
    ]

    baseline = _evaluate(copy.deepcopy(samples), **options)
    scored = _evaluate(copy.deepcopy(samples), sky_iou=True, **options)

    assert "sky_segmentation" not in baseline
    assert _canonical(_strip_sky_iou(scored)) == _canonical(baseline)


def test_sky_mask_without_sky_depth_matches_a_run_without_the_mask(monkeypatch):
    # The pre-change CLI never loaded sky_mask without --sky-depth.
    _patch_depth_metrics(monkeypatch)
    pre_change = _evaluate([_sample(sky_mask=None, segmentation=None)])
    scored = _evaluate([_sample()], sky_iou=True)
    assert _canonical(_strip_sky_iou(scored)) == _canonical(pre_change)


def test_sky_depth_values_are_unchanged_by_sky_iou(monkeypatch):
    _patch_depth_metrics(monkeypatch)
    capped = _evaluate([_sample()], sky_depth=50.0)
    scored = _evaluate([_sample()], sky_depth=50.0, sky_iou=True)
    assert capped["depth_metric"]["standard"]["pixel_pool"] == (
        scored["depth_metric"]["standard"]["pixel_pool"]
    )
    assert _canonical(_strip_sky_iou(scored)) == _canonical(capped)


def test_sky_iou_does_not_mutate_the_samples(monkeypatch):
    _patch_depth_metrics(monkeypatch)
    sample = _sample()
    before = copy.deepcopy(sample)
    _evaluate([sample], sky_depth=50.0, sky_iou=True, sky_mask_enabled=True)
    for key in ("gt", "pred", "pred_sky_mask"):
        np.testing.assert_array_equal(sample[key], before[key])
    np.testing.assert_array_equal(
        sample["segmentation"]["frame0"], before["segmentation"]["frame0"]
    )


# ---------------------------------------------------------------------------
# Sparse depth evaluator
# ---------------------------------------------------------------------------


def _sparse_sample(**extra):
    sample = _OneSampleSparseDataset()[0]
    sample["pred"] = sample["pred"].copy()
    sample["pred"][0, :] = 300.0  # top row beyond the cap
    sample.update(extra)
    return sample


def test_sparse_evaluator_reports_sky_iou_on_the_dense_grid():
    gt_sky = np.zeros((3, 3), dtype=bool)
    gt_sky[0, :] = True
    sky_mask = np.zeros((3, 3), dtype=bool)
    sky_mask[0, :2] = True
    sample = _sparse_sample(
        segmentation={"000000": gt_sky}, pred_sky_mask=sky_mask
    )

    result = eval_mod.evaluate_sparse_depth_samples(
        [sample], pred_is_radial=True, num_workers=0, alignment_mode="none",
        sky_depth=250.0, sky_iou=True,
    )

    summary = result["sky_segmentation"]
    assert summary["explicit"]["pixel_pool"]["iou"] == pytest.approx(2 / 3)
    assert summary["explicit"]["pixel_pool"]["precision"] == 1.0
    assert summary["implicit"]["pixel_pool"]["iou"] == 1.0
    per_file = result["per_file_metrics"]["files"][0]["metrics"]
    assert per_file["sky_segmentation"]["explicit"]["recall"] == pytest.approx(2 / 3)
    assert result["sky_segmentation_info"]["implicit"]["available"] is True


@pytest.mark.parametrize("sky_depth", [None, 250.0])
def test_sparse_depth_tree_is_identical_with_and_without_sky_iou(sky_depth):
    gt_sky = np.zeros((3, 3), dtype=bool)
    gt_sky[0, :] = True
    sky_mask = np.zeros((3, 3), dtype=bool)
    sky_mask[0, :] = True

    def _run(sample, **kwargs):
        return eval_mod.evaluate_sparse_depth_samples(
            [sample], pred_is_radial=True, num_workers=0, alignment_mode="none",
            sky_depth=sky_depth, benchmark_depth_range=(1.0, 5.0), **kwargs,
        )

    with_mask = _sparse_sample(segmentation={"000000": gt_sky}, pred_sky_mask=sky_mask)
    baseline = _run(
        _sparse_sample(pred_sky_mask=sky_mask)
        if sky_depth is not None
        else _sparse_sample()
    )
    scored = _run(with_mask, sky_iou=True)

    assert "sky_segmentation" in scored
    assert _canonical(_strip_sky_iou(scored)) == _canonical(baseline)
