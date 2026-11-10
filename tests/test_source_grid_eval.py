"""Depth evaluation follows a prediction's recorded ``source_grid``."""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

ds_crawler = pytest.importorskip("ds_crawler")

H, W = 40, 60
_META = {"radial_depth": True, "scale_to_meters": 1.0, "range": [0, 200]}
_LOADER = {"loader": "generic_dense_depth", "function": "depth"}


def _depth(k: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    depth = (5.0 + 0.02 * xx + 0.01 * yy + k).astype(np.float32)
    depth[(xx // 7) % 2 == 1] += 20.0  # sharp stripes, so a misplaced pixel shows
    return depth


def _writer(root, name, meta):
    return ds_crawler.DatasetWriter(
        root, name=name, type="depth", separator=None,
        euler_train={"used_as": "target", "modality_type": "depth"},
        meta=meta, euler_loading=_LOADER,
    )


def _crop_grid(top: int, left: int) -> dict:
    return {"source": "rgb", "size": [H, W], "affine": [[1, 0, -left], [0, 1, -top]]}


@pytest.fixture
def light_learned_metrics(monkeypatch):
    """Stub the learned image metrics; alignment and depth metrics run for real."""
    from euler_eval import evaluate

    monkeypatch.setattr(
        evaluate, "LPIPSMetric",
        lambda **kw: SimpleNamespace(compute_batch=lambda preds, gts, **kw: [0.0] * len(preds)),
    )
    monkeypatch.setattr(
        evaluate, "FIDKIDMetric",
        lambda **kw: SimpleNamespace(
            compute_fid=lambda *a, **kw: 0.0, compute_kid=lambda *a, **kw: (0.0, 0.0)
        ),
    )


def test_recorded_centre_crops_are_scored_on_the_pixels_they_cover(
    tmp_path, light_learned_metrics
):
    from euler_eval.data import build_depth_eval_dataset
    from euler_eval.evaluate import evaluate_depth_samples

    gt = _writer(tmp_path / "gt", "gt", _META)
    # The head records the usual crop; the second file overrides it.
    pred = _writer(tmp_path / "pred", "pred", {**_META, "source_grid": _crop_grid(2, 3)})
    for k, (top, left) in enumerate([(2, 3), (1, 5)]):
        depth = _depth(k)
        np.save(gt.get_path(f"seq/{k:04d}", f"{k:04d}.npy"), depth)
        attributes = {"source_grid": _crop_grid(top, left)} if k == 1 else None
        np.save(
            pred.get_path(f"seq/{k:04d}", f"{k:04d}.npy", attributes=attributes),
            depth[top : top + 32, left : left + 48],
        )
    gt.save_index()
    pred.save_index()

    def run(mode):
        dataset = build_depth_eval_dataset(str(tmp_path / "gt"), str(tmp_path / "pred"))
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate_depth_samples(
                dataset, is_radial=True, device="cpu", num_workers=0,
                alignment_mode="none", spatial_alignment=mode,
            )

    recorded = run("auto")
    assert recorded["spatial_info"]["mode"] == "source_grid"
    assert recorded["spatial_info"]["method"] == "crop"
    absrel = _absrel(recorded["per_file_metrics"])
    assert len(absrel) >= 2 and max(absrel) == 0.0

    # An explicit mode still wins, and the top-left guess misplaces these crops.
    guessed = run("crop")
    assert guessed["spatial_info"]["mode"] == "crop"
    assert max(_absrel(guessed["per_file_metrics"])) > 0.01


def _absrel(per_file) -> list[float]:
    """Every per-sample ``absrel`` value under ``per_file_metrics``."""
    found: list[float] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "absrel" and isinstance(value, (int, float)):
                    found.append(float(value))
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(per_file)
    return found


def test_declared_invalid_sentinels_are_not_scored(tmp_path, light_learned_metrics):
    from euler_eval.data import build_depth_eval_dataset, get_depth_metadata
    from euler_eval.evaluate import evaluate_depth_samples

    depth = _depth(0)
    gt_values = depth.copy()
    gt_values[:5] = 655.35  # GT rows the sensor did not measure, marked by a sentinel
    pred_values = depth.copy()
    pred_values[:, -6:] = 0.0  # columns the model did not predict, marked by 0

    gt = _writer(tmp_path / "gt", "gt", {**_META, "invalid": {"non_finite": False, "sentinel": 655.35}})
    pred = _writer(tmp_path / "pred", "pred", {**_META, "invalid": {"non_finite": False, "sentinel": 0}})
    np.save(gt.get_path("seq/0000", "0000.npy"), gt_values)
    np.save(pred.get_path("seq/0000", "0000.npy"), pred_values)
    gt.save_index()
    pred.save_index()

    dataset = build_depth_eval_dataset(str(tmp_path / "gt"), str(tmp_path / "pred"))
    assert get_depth_metadata(dataset)["invalid_sentinel"] == pytest.approx(655.35)
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluate_depth_samples(
            dataset, is_radial=True, device="cpu", num_workers=0, alignment_mode="none"
        )
    # Every scored pixel is a real GT measurement with a real prediction: exact.
    assert max(_absrel(result["per_file_metrics"])) == 0.0
