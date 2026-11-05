"""GT segmentation is paired with each frame by ID (``--mask-sky``, ``--sky-iou``).

Segmentation used to be a hierarchical modality, which hands every sample all
masks of its folder; ``--mask-sky`` then masked each frame with the folder's
first mask.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from euler_eval import cli, data
from tests.test_depth_alignment_output import _patch_depth_metrics
from tests.test_sky_iou_cli import _write_index

FRAMES = ("f0", "f1", "f2")
# Each frame's own sky rows on a 4x4 grid; they differ from frame to frame.
SKY_ROWS = {"f0": [0], "f1": [0, 1], "f2": [2]}
SKY_LABEL = 7


def _sky(frame_id):
    sky = np.zeros((4, 4), dtype=bool)
    sky[SKY_ROWS[frame_id], :] = True
    return sky


@pytest.fixture
def flat_frames(tmp_path):
    """Depth and per-frame segmentation in one flat folder each.

    GT sky is 1000 m and the prediction saturates at 100 m there, so a frame
    masked with its own sky has zero error.
    """

    def _write(segmentation_frames=FRAMES):
        paths = {}
        for name in ("gt", "pred", "segmentation"):
            root = tmp_path / name
            root.mkdir(exist_ok=True)
            paths[name] = str(root)
        for frame_id in FRAMES:
            sky = _sky(frame_id)
            np.save(
                tmp_path / "gt" / f"{frame_id}.npy",
                np.where(sky, 1000.0, 10.0).astype(np.float32),
            )
            np.save(
                tmp_path / "pred" / f"{frame_id}.npy",
                np.where(sky, 100.0, 10.0).astype(np.float32),
            )
            np.save(
                tmp_path / "segmentation" / f"{frame_id}.npy",
                np.where(sky, SKY_LABEL, 1).astype(np.uint8),
            )
        for name in ("gt", "pred"):
            _write_index(
                tmp_path / name, scope="depth", key="depth",
                loader="generic_dense_depth", function="depth",
                meta={"radial_depth": True},
                files=[{"path": f"{f}.npy", "id": f} for f in FRAMES],
            )
        _write_index(
            tmp_path / "segmentation", scope="segmentation", key="segmentation",
            loader="generic", function="semantic_segmentation",
            meta={"skyclass": [SKY_LABEL] * 3},
            files=[{"path": f"{f}.npy", "id": f} for f in segmentation_frames],
        )
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "gt": {
                "depth": {"path": paths["gt"]},
                "segmentation": {"path": paths["segmentation"]},
            },
            "datasets": [{"name": "model", "depth": {"path": paths["pred"]}}],
        }))
        return config_path, paths

    return _write


def _run(monkeypatch, config_path, output_dir, *flags):
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--no-sanity-check", "--depth-alignment", "none", *flags,
    ])
    cli.main()
    return json.loads((Path(output_dir) / "eval.json").read_text())


def _per_file_mae(saved):
    return {
        entry["id"]: entry["metrics"]["depth"]["eval"]["metric"]["standard"]["mae"]
        for entry in saved["per_file_metrics"]["files"]
    }


def test_mask_sky_masks_each_frame_with_its_own_segmentation(
    monkeypatch, flat_frames
):
    _patch_depth_metrics(monkeypatch)
    config_path, paths = flat_frames()

    saved = _run(monkeypatch, config_path, paths["pred"], "--mask-sky")

    # With the folder's first mask, f1 and f2 scored 300 m: their own sky
    # (900 m error on 4 of 12 unmasked pixels) stayed in the metrics.
    assert _per_file_mae(saved) == {"f0": 0.0, "f1": 0.0, "f2": 0.0}
    assert saved["dataset_info"]["num_pairs"] == 3


def test_frames_without_segmentation_are_dropped_with_a_warning(
    monkeypatch, capsys, flat_frames
):
    _patch_depth_metrics(monkeypatch)
    config_path, paths = flat_frames(segmentation_frames=("f0", "f1"))

    saved = _run(monkeypatch, config_path, paths["pred"], "--mask-sky")

    assert saved["dataset_info"]["num_pairs"] == 2
    assert set(_per_file_mae(saved)) == {"f0", "f1"}
    assert (
        "Warning: the GT segmentation of 'model' lacks 1 pairs that the other "
        "inputs share; those pairs are not evaluated."
    ) in capsys.readouterr().out


def test_partial_segmentation_read_only_for_sky_iou_keeps_every_pair(
    monkeypatch, capsys, flat_frames
):
    _patch_depth_metrics(monkeypatch)
    config_path, paths = flat_frames(segmentation_frames=("f0", "f1"))

    baseline = _run(monkeypatch, config_path, paths["pred"])
    capsys.readouterr()
    scored = _run(monkeypatch, config_path, paths["pred"], "--sky-iou")

    assert "the GT segmentation of 'model' lacks 1 evaluated pairs" in (
        capsys.readouterr().out
    )
    assert scored["dataset_info"]["num_pairs"] == 3
    assert scored["depth"]["eval"] == baseline["depth"]["eval"]
    reason = "GT segmentation does not cover every evaluated pair"
    meta = scored["meta"]["sky_segmentation"]
    assert meta["explicit"] == {"available": False, "reason": reason}
    assert meta["implicit"] == {"available": False, "reason": reason}


def test_inputs_missing_the_same_frame_both_count_it():
    class _Dataset:
        _lookups = {
            "gt": {"a": 0, "b": 0},
            "pred": {"a": 0, "b": 0},
            "segmentation": {"a": 0},
            "pred_sky_mask": {"a": 0},
        }

    optional = ("segmentation", "pred_sky_mask")
    for name in optional:
        assert data.count_ids_dropped_by(_Dataset(), name, optional=optional) == 1
    # Without naming the other optional input, its gap hides this one's.
    assert data.count_ids_dropped_by(_Dataset(), "segmentation") == 0


class _CapturedDataset:
    def __init__(self, *, modalities, hierarchical_modalities=None):
        self.modalities = modalities
        self.hierarchical_modalities = hierarchical_modalities or {}


@pytest.mark.parametrize(
    "build, kwargs",
    [
        (data.build_rgb_eval_dataset, dict(gt_rgb_path="/gt", pred_rgb_path="/p")),
        (
            data.build_points_3d_eval_dataset,
            dict(pred_points_3d_path="/p", gt_points_3d_path="/gt"),
        ),
        (
            data.build_points_3d_sparse_eval_dataset,
            dict(
                gt_sparse_depth_path="/gt", pred_points_3d_path="/p",
                intrinsics_path="/k", camera_extrinsics_path="/e",
            ),
        ),
    ],
)
def test_every_builder_pairs_segmentation_by_id(monkeypatch, build, kwargs):
    monkeypatch.setattr(data, "MultiModalDataset", _CapturedDataset)
    monkeypatch.setattr(
        data, "_resolve_sky_mask_loader", lambda path, *, modality_key: "sky"
    )
    dataset = build(**kwargs, segmentation_path="/seg")
    assert dataset.modalities["segmentation"].loader == "sky"
    assert "segmentation" not in dataset.hierarchical_modalities
