"""Tests for ``--mask-sky-prediction``: drop a prediction's own sky from scoring."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import euler_eval.evaluate as eval_mod
from euler_eval import cli
from tests.test_depth_alignment_output import _DummyDepthDataset, _patch_depth_metrics
from tests.test_sky_iou import _canonical
from tests.test_sky_iou_cli import write_gt_segmentation, write_sky_mask
from tests.test_sparse_depth import _OneSampleSparseDataset

T, F = True, False


def _evaluate(samples, **kwargs):
    kwargs.setdefault("alignment_mode", "none")
    return eval_mod.evaluate_depth_samples(
        _DummyDepthDataset(copy.deepcopy(samples)), is_radial=True, device="cpu",
        num_workers=0, **kwargs,
    )


def _sample(**extra):
    # The top row is predicted sky, and the model is badly wrong there.
    sample = {
        "id": "frame0",
        "gt": np.array([[100, 90], [30, 40]], dtype=np.float32),
        "pred": np.array([[5, 5], [30, 40]], dtype=np.float32),
        "pred_sky_mask": np.array([[T, T], [F, F]]),
    }
    sample.update(extra)
    return sample


def _strip(result):
    result = copy.deepcopy(result)
    result.pop("pred_sky_masking", None)
    return result


def test_predicted_sky_is_excluded_from_every_metric(monkeypatch):
    _patch_depth_metrics(monkeypatch)
    scored = _evaluate([_sample()])
    masked = _evaluate([_sample()], pred_sky_mask_enabled=True)

    assert scored["depth_metric"]["standard"]["pixel_pool"]["mae"] == 45.0
    assert masked["depth_metric"]["standard"]["pixel_pool"]["mae"] == 0.0
    per_file = masked["per_file_metrics"]["files"][0]["metrics"]["depth_metric"]
    assert per_file["standard"]["mae"] == 0.0
    assert masked["pred_sky_masking"] == {
        "source": "pred.sky_mask",
        "samples": 1,
        "samples_with_sky_mask": 1,
        "gt_valid_pixels": 4,
        "excluded_gt_pixels": 2,
        "excluded_gt_fraction": 0.5,
    }
    assert "pred_sky_masking" not in scored


def test_benchmark_bins_exclude_predicted_sky(monkeypatch):
    _patch_depth_metrics(monkeypatch)
    masked = _evaluate(
        [_sample()], pred_sky_mask_enabled=True, benchmark_depth_range=(1.0, 200.0)
    )
    benchmark = masked["depth_benchmark"]["metric"]["all"]
    assert benchmark["standard"]["pixel_pool"]["mae"] == 0.0


def test_without_a_sky_mask_nothing_changes_and_it_warns(monkeypatch, capsys):
    _patch_depth_metrics(monkeypatch)
    sample = _sample()
    del sample["pred_sky_mask"]
    baseline = _evaluate([sample])
    capsys.readouterr()
    masked = _evaluate([sample], pred_sky_mask_enabled=True)

    assert _canonical(_strip(masked)) == _canonical(baseline)
    assert masked["pred_sky_masking"]["samples_with_sky_mask"] == 0
    assert masked["pred_sky_masking"]["excluded_gt_fraction"] == 0.0
    assert (
        "--mask-sky-prediction found no predicted sky_mask in any of the 1 depth "
        "samples"
    ) in capsys.readouterr().out


def test_combines_with_gt_sky_masking(monkeypatch):
    # Row 0 is predicted sky, row 1 is GT sky; only row 2 is scored.
    _patch_depth_metrics(monkeypatch)
    sample = {
        "id": "frame0",
        "gt": np.array([[100, 90], [500, 500], [30, 40]], dtype=np.float32),
        "pred": np.array([[5, 5], [1, 1], [31, 41]], dtype=np.float32),
        "pred_sky_mask": np.array([[T, T], [F, F], [F, F]]),
        "segmentation": np.array([[F, F], [T, T], [F, F]]),
    }
    masked = _evaluate([sample], pred_sky_mask_enabled=True, sky_mask_enabled=True)

    assert masked["depth_metric"]["standard"]["pixel_pool"]["mae"] == 1.0
    # GT sky is not counted as excluded by the prediction: 2 of 4 pixels.
    info = masked["pred_sky_masking"]
    assert (info["gt_valid_pixels"], info["excluded_gt_pixels"]) == (4, 2)


def test_predicted_sky_is_excluded_from_the_affine_fit(monkeypatch):
    # Off the sky, pred = gt / 2 exactly; the sky row would bend the fit.
    _patch_depth_metrics(monkeypatch)
    sample = {
        "id": "frame0",
        "gt": np.array([[100, 100, 100], [10, 20, 30]], dtype=np.float32),
        "pred": np.array([[1, 1, 1], [5, 10, 15]], dtype=np.float32),
        "pred_sky_mask": np.array([[T, T, T], [F, F, F]]),
    }
    scored = _evaluate([sample], alignment_mode="affine")
    masked = _evaluate([sample], alignment_mode="affine", pred_sky_mask_enabled=True)

    assert scored["depth_metric"]["standard"]["pixel_pool"]["mae"] > 1.0
    assert masked["depth_metric"]["standard"]["pixel_pool"]["mae"] == pytest.approx(
        0.0, abs=1e-4
    )


def test_sky_iou_is_unaffected_by_predicted_sky_masking(monkeypatch):
    _patch_depth_metrics(monkeypatch)
    sample = _sample(segmentation=np.array([[T, F], [F, F]]))
    scored = _evaluate([sample], sky_iou=True, sky_depth=50.0)
    masked = _evaluate(
        [sample], sky_iou=True, sky_depth=50.0, pred_sky_mask_enabled=True
    )
    # Undefined precision is NaN in both, so compare serialized forms.
    for key in ("sky_segmentation", "sky_segmentation_info"):
        assert _canonical(masked[key]) == _canonical(scored[key])


def test_sparse_points_inside_predicted_sky_are_not_evaluated():
    sample = _OneSampleSparseDataset()[0]
    pred_sky = np.zeros((3, 3), dtype=bool)
    pred_sky[1, 2] = True  # one of the two projected lidar points
    sample["pred_sky_mask"] = pred_sky

    def _run(**kwargs):
        return eval_mod.evaluate_sparse_depth_samples(
            [copy.deepcopy(sample)], pred_is_radial=True, num_workers=0,
            alignment_mode="none", **kwargs,
        )

    scored = _run()
    masked = _run(pred_sky_mask_enabled=True)

    assert scored["dataset_info"]["evaluated_pixels"] == 2
    assert masked["dataset_info"]["evaluated_pixels"] == 1
    per_file = masked["per_file_metrics"]["files"][0]["metrics"]["sparse_depth"]
    assert per_file["depth_metrics"]["valid_pixel_count"] == 1
    info = masked["pred_sky_masking"]
    assert (info["gt_valid_pixels"], info["excluded_gt_pixels"]) == (2, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _config(tmp_path, paths, *, sky_mask, segmentation=False):
    gt = {"depth": {"path": paths["gt"]}}
    if segmentation:
        gt["segmentation"] = {"path": paths["segmentation"]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": gt,
        "datasets": [{
            "name": "sky_prediction",
            "depth": {"path": paths["pred"]},
            "sky_mask": {"path": sky_mask},
        }],
    }))
    return config_path


def _run(monkeypatch, config_path, output_dir, *flags):
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", "--depth-alignment", "none", *flags,
    ])
    cli.main()
    return json.loads((Path(output_dir) / "eval.json").read_text())


def test_cli_masks_predicted_sky_without_sky_depth(
    monkeypatch, tmp_path, indexed_sky_prediction
):
    _patch_depth_metrics(monkeypatch)
    paths = indexed_sky_prediction
    config_path = _config(tmp_path, paths, sky_mask=paths["sky_mask"])

    baseline = _run(monkeypatch, config_path, paths["pred"])
    masked = _run(monkeypatch, config_path, paths["pred"], "--mask-sky-prediction")

    # Each frame's predicted sky pixel (a 99 m error) is dropped; the 50 m
    # error elsewhere remains: 2 x 50 m over 6 scored pixels.
    assert baseline["depth"]["eval"]["metric"]["standard"]["pixel_pool"]["mae"] == (
        37.25
    )
    assert masked["depth"]["eval"]["metric"]["standard"]["pixel_pool"][
        "mae"
    ] == pytest.approx(100 / 6)
    assert masked["meta"]["eval_params"]["mask_sky_prediction"] is True
    assert masked["meta"]["pred"]["sky_mask"] == {"path": paths["sky_mask"]}
    assert masked["meta"]["pred_sky_masking"]["excluded_gt_fraction"] == 0.25
    assert "sky_depth" not in masked["meta"]["eval_params"]
    assert "mask_sky_prediction" not in baseline["meta"]["eval_params"]
    assert "pred_sky_masking" not in baseline["meta"]


def test_cli_sky_mask_used_for_masking_is_not_dropped_for_coverage(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction
):
    # With --mask-sky-prediction the mask is a required input, so a partial
    # mask drops the frames it lacks (with a warning) instead of being ignored.
    _patch_depth_metrics(monkeypatch)
    paths = dict(indexed_sky_prediction)
    paths["segmentation"] = write_gt_segmentation(tmp_path / "segmentation")
    partial = write_sky_mask(tmp_path / "partial_mask", ["first"])
    config_path = _config(tmp_path, paths, sky_mask=partial, segmentation=True)

    saved = _run(
        monkeypatch, config_path, paths["pred"], "--mask-sky-prediction", "--sky-iou"
    )

    assert saved["dataset_info"]["num_pairs"] == 1
    assert (
        "the sky_mask of 'sky_prediction' lacks 1 pairs that the other inputs "
        "share; those pairs are not evaluated."
    ) in capsys.readouterr().out
    assert saved["meta"]["sky_segmentation"]["explicit"]["available"] is True
