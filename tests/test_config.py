"""Tests for euler_eval.cli config validation."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from euler_eval.cli import (
    _gt_segmentation_entry,
    load_config,
    validate_dataset_entry,
    validate_gt_config,
)
from euler_eval.config_paths import resolve_gt_segmentation_path
from tests.test_sky_iou_cli import _write_index

# ---------------------------------------------------------------------------
# validate_gt_config
# ---------------------------------------------------------------------------


class TestValidateGtConfig:
    def test_valid_minimal(self, tmp_path):
        """Minimal valid GT config with rgb and depth paths."""
        rgb_path = tmp_path / "rgb"
        depth_path = tmp_path / "depth"
        rgb_path.mkdir()
        depth_path.mkdir()

        gt = {
            "rgb": {"path": str(rgb_path)},
            "depth": {"path": str(depth_path)},
        }
        validate_gt_config(gt)  # should not raise

    def test_valid_with_all_modalities(self, tmp_path):
        """Full GT config with all optional modalities."""
        for name in ("rgb", "depth", "segmentation", "calibration"):
            (tmp_path / name).mkdir()

        gt = {
            "rgb": {"path": str(tmp_path / "rgb")},
            "depth": {"path": str(tmp_path / "depth")},
            "segmentation": {"path": str(tmp_path / "segmentation")},
            "calibration": {"path": str(tmp_path / "calibration")},
        }
        validate_gt_config(gt)
        key, entry = _gt_segmentation_entry(gt)
        assert key == "segmentation"
        assert entry == gt["segmentation"]

    def test_depth_only_gt_is_valid(self, tmp_path):
        depth_path = tmp_path / "depth"
        depth_path.mkdir()
        gt = {"depth": {"path": str(depth_path)}}
        validate_gt_config(gt)

    def test_rgb_only_gt_is_valid(self, tmp_path):
        rgb_path = tmp_path / "rgb"
        rgb_path.mkdir()
        gt = {"rgb": {"path": str(rgb_path)}}
        validate_gt_config(gt)

    def test_inline_zip_split_gt_path_is_valid(self, tmp_path):
        rgb_path = tmp_path / "reference.zip"
        with zipfile.ZipFile(rgb_path, "w") as zf:
            zf.writestr("dummy.txt", "x")
        gt = {"rgb": {"path": f"{rgb_path}:fog_day"}}
        validate_gt_config(gt)

    def test_inline_scope_gt_path_is_valid(self, tmp_path):
        rgb_path = tmp_path / "reference.zip"
        with zipfile.ZipFile(rgb_path, "w") as zf:
            zf.writestr("dummy.txt", "x")
        gt = {"rgb": {"path": f"{rgb_path}#scope=rgb"}}
        validate_gt_config(gt)

    def test_inline_scope_intrinsics_gt_path_is_valid(self, tmp_path):
        depth_path = tmp_path / "depth"
        depth_path.mkdir()
        intrinsics_path = tmp_path / "frame_camera_trainvaltest.zip"
        with zipfile.ZipFile(intrinsics_path, "w") as zf:
            zf.writestr("dummy.txt", "x")
        gt = {
            "depth": {"path": str(depth_path)},
            "intrinsics": {"path": f"{intrinsics_path}#scope=intrinsics"},
        }
        validate_gt_config(gt)

    def test_inline_split_and_scope_gt_path_is_valid(self, tmp_path):
        rgb_path = tmp_path / "reference.zip"
        with zipfile.ZipFile(rgb_path, "w") as zf:
            zf.writestr("dummy.txt", "x")
        gt = {"rgb": {"path": f"{rgb_path}:fog_day#scope=rgb"}}
        validate_gt_config(gt)

    def test_sparse_depth_requires_projection_modalities(self, tmp_path):
        sparse_path = tmp_path / "sparse_depth"
        sparse_path.mkdir()
        gt = {"sparse_depth": {"path": str(sparse_path)}}
        with pytest.raises(ValueError, match="gt.intrinsics.path"):
            validate_gt_config(gt)

    def test_sparse_depth_gt_is_valid_with_projection_modalities(self, tmp_path):
        for name in ("sparse_depth", "intrinsics", "camera_extrinsics"):
            (tmp_path / name).mkdir()
        gt = {
            "sparse_depth": {"path": str(tmp_path / "sparse_depth")},
            "intrinsics": {"path": str(tmp_path / "intrinsics")},
            "camera_extrinsics": {"path": str(tmp_path / "camera_extrinsics")},
        }
        validate_gt_config(gt)

    def test_sparse_depth_gt_accepts_optional_lidar_extrinsics(self, tmp_path):
        for name in (
            "sparse_depth",
            "intrinsics",
            "camera_extrinsics",
            "lidar_extrinsics",
        ):
            (tmp_path / name).mkdir()
        gt = {
            "sparse_depth": {"path": str(tmp_path / "sparse_depth")},
            "intrinsics": {"path": str(tmp_path / "intrinsics")},
            "camera_extrinsics": {"path": str(tmp_path / "camera_extrinsics")},
            "lidar_extrinsics": {"path": str(tmp_path / "lidar_extrinsics")},
        }
        validate_gt_config(gt)

    def test_nonexistent_path(self, tmp_path):
        rgb_path = tmp_path / "rgb"
        rgb_path.mkdir()
        gt = {
            "rgb": {"path": str(rgb_path)},
            "depth": {"path": str(tmp_path / "nonexistent")},
        }
        with pytest.raises(ValueError, match="does not exist"):
            validate_gt_config(gt)

    def test_semantic_segmentation_alias_is_accepted(self, tmp_path):
        for name in ("depth", "labels"):
            (tmp_path / name).mkdir()
        gt = {
            "depth": {"path": str(tmp_path / "depth")},
            "semantic_segmentation": {"path": str(tmp_path / "labels")},
        }
        validate_gt_config(gt)
        key, entry = _gt_segmentation_entry(gt)
        assert key == "semantic_segmentation"
        assert entry == gt["semantic_segmentation"]

    def test_semantic_segmentation_alias_path_must_exist(self, tmp_path):
        (tmp_path / "depth").mkdir()
        gt = {
            "depth": {"path": str(tmp_path / "depth")},
            "semantic_segmentation": {"path": str(tmp_path / "missing")},
        }
        with pytest.raises(ValueError, match="gt.semantic_segmentation.path"):
            validate_gt_config(gt)

    def test_segmentation_and_its_alias_together_are_an_error(self, tmp_path):
        for name in ("depth", "seg"):
            (tmp_path / name).mkdir()
        gt = {
            "depth": {"path": str(tmp_path / "depth")},
            "segmentation": {"path": str(tmp_path / "seg")},
            "semantic_segmentation": {"path": str(tmp_path / "seg")},
        }
        with pytest.raises(ValueError, match="configure only one"):
            validate_gt_config(gt)
        with pytest.raises(ValueError, match="aliases"):
            _gt_segmentation_entry(gt)


# ---------------------------------------------------------------------------
# validate_dataset_entry
# ---------------------------------------------------------------------------


class TestValidateDatasetEntry:
    @pytest.mark.parametrize("depth_key", ["depth", "relative_depth", "affine_depth"])
    def test_valid_prediction_sky_mask(self, tmp_path, depth_key):
        archive = tmp_path / "predictions.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("dummy.txt", "x")
        validate_dataset_entry({
            "name": "with_sky",
            depth_key: {"path": str(tmp_path)},
            "sky_mask": {"path": f"{archive}:test#scope=sky_mask"},
        }, 0)

    def test_prediction_sky_mask_path_must_exist(self, tmp_path):
        with pytest.raises(ValueError, match=r"datasets\[2\].sky_mask.path does not exist"):
            validate_dataset_entry({
                "name": "with_sky",
                "depth": {"path": str(tmp_path)},
                "sky_mask": {"path": str(tmp_path / "missing")},
            }, 2)

    @pytest.mark.parametrize("mask_entry", [None, {}, "mask", {"path": ""}])
    def test_prediction_sky_mask_requires_path(self, tmp_path, mask_entry):
        with pytest.raises(ValueError, match="sky_mask must have a 'path'"):
            validate_dataset_entry({
                "name": "with_sky",
                "depth": {"path": str(tmp_path)},
                "sky_mask": mask_entry,
            }, 0)

    def test_prediction_sky_mask_requires_dense_depth(self, tmp_path):
        with pytest.raises(ValueError, match="sky_mask requires a dense depth"):
            validate_dataset_entry({
                "name": "with_sky",
                "rgb": {"path": str(tmp_path)},
                "sky_mask": {"path": str(tmp_path)},
            }, 0)

    def test_valid_depth_only(self, tmp_path):
        depth_path = tmp_path / "depth"
        depth_path.mkdir()
        entry = {
            "name": "model_a",
            "depth": {"path": str(depth_path)},
        }
        validate_dataset_entry(entry, 0)

    def test_valid_relative_depth_only(self, tmp_path):
        depth_path = tmp_path / "relative_depth"
        depth_path.mkdir()
        entry = {
            "name": "model_a",
            "relative_depth": {"path": str(depth_path)},
        }
        validate_dataset_entry(entry, 0)

    def test_valid_affine_depth_only(self, tmp_path):
        depth_path = tmp_path / "affine_depth"
        depth_path.mkdir()
        entry = {
            "name": "model_a",
            "affine_depth": {"path": str(depth_path)},
        }
        validate_dataset_entry(entry, 0)

    def test_multiple_depth_prediction_entries_are_rejected(self, tmp_path):
        for name in ("depth", "relative_depth"):
            (tmp_path / name).mkdir()
        entry = {
            "name": "model_a",
            "depth": {"path": str(tmp_path / "depth")},
            "relative_depth": {"path": str(tmp_path / "relative_depth")},
        }
        with pytest.raises(ValueError, match="multiple dense depth entries"):
            validate_dataset_entry(entry, 0)

    def test_valid_rgb_only(self, tmp_path):
        rgb_path = tmp_path / "rgb"
        rgb_path.mkdir()
        entry = {
            "name": "model_a",
            "rgb": {"path": str(rgb_path)},
        }
        validate_dataset_entry(entry, 0)

    def test_valid_both(self, tmp_path):
        for name in ("rgb", "depth"):
            (tmp_path / name).mkdir()
        entry = {
            "name": "model_a",
            "rgb": {"path": str(tmp_path / "rgb")},
            "depth": {"path": str(tmp_path / "depth")},
        }
        validate_dataset_entry(entry, 0)

    def test_missing_name(self, tmp_path):
        depth_path = tmp_path / "depth"
        depth_path.mkdir()
        entry = {"depth": {"path": str(depth_path)}}
        with pytest.raises(ValueError, match="must have a 'name'"):
            validate_dataset_entry(entry, 0)

    def test_no_modalities(self):
        entry = {"name": "empty"}
        with pytest.raises(ValueError, match="at least"):
            validate_dataset_entry(entry, 0)

    def test_nonexistent_depth_path(self, tmp_path):
        entry = {
            "name": "bad",
            "depth": {"path": str(tmp_path / "nonexistent")},
        }
        with pytest.raises(ValueError, match="does not exist"):
            validate_dataset_entry(entry, 0)

    def test_inline_zip_split_dataset_path_is_valid(self, tmp_path):
        rgb_path = tmp_path / "prediction.zip"
        with zipfile.ZipFile(rgb_path, "w") as zf:
            zf.writestr("dummy.txt", "x")
        entry = {
            "name": "model_a",
            "rgb": {"path": f"{rgb_path}:fog_day"},
        }
        validate_dataset_entry(entry, 0)

    def test_index_in_error_message(self, tmp_path):
        entry = {"name": "test"}
        with pytest.raises(ValueError, match=r"datasets\[3\]"):
            validate_dataset_entry(entry, 3)


# ---------------------------------------------------------------------------
# load_config  (end-to-end validation)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def _write_config(self, tmp_path, config_dict):
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
        return str(config_path)

    def test_valid_config(self, tmp_path):
        for name in ("gt_rgb", "gt_depth", "pred_depth"):
            (tmp_path / name).mkdir()

        config = {
            "gt": {
                "rgb": {"path": str(tmp_path / "gt_rgb")},
                "depth": {"path": str(tmp_path / "gt_depth")},
            },
            "datasets": [
                {
                    "name": "test_model",
                    "depth": {"path": str(tmp_path / "pred_depth")},
                }
            ],
        }
        path = self._write_config(tmp_path, config)
        result = load_config(path)
        assert result["gt"]["rgb"]["path"] == str(tmp_path / "gt_rgb")
        assert len(result["datasets"]) == 1

    def test_missing_gt_section(self, tmp_path):
        config = {"datasets": [{"name": "x", "depth": {"path": "/fake"}}]}
        path = self._write_config(tmp_path, config)
        with pytest.raises(ValueError, match="'gt' section"):
            load_config(path)

    def test_missing_datasets_section(self, tmp_path):
        for name in ("rgb", "depth"):
            (tmp_path / name).mkdir()
        config = {
            "gt": {
                "rgb": {"path": str(tmp_path / "rgb")},
                "depth": {"path": str(tmp_path / "depth")},
            },
        }
        path = self._write_config(tmp_path, config)
        with pytest.raises(ValueError, match="'datasets'"):
            load_config(path)

    def test_empty_datasets_array(self, tmp_path):
        for name in ("rgb", "depth"):
            (tmp_path / name).mkdir()
        config = {
            "gt": {
                "rgb": {"path": str(tmp_path / "rgb")},
                "depth": {"path": str(tmp_path / "depth")},
            },
            "datasets": [],
        }
        path = self._write_config(tmp_path, config)
        with pytest.raises(ValueError, match="'datasets'"):
            load_config(path)

    def test_invalid_json(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("not json{{{")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(str(config_path))

    def test_multiple_datasets(self, tmp_path):
        for name in ("gt_rgb", "gt_depth", "pred_a_rgb", "pred_b_depth"):
            (tmp_path / name).mkdir()

        config = {
            "gt": {
                "rgb": {"path": str(tmp_path / "gt_rgb")},
                "depth": {"path": str(tmp_path / "gt_depth")},
            },
            "datasets": [
                {
                    "name": "model_a",
                    "rgb": {"path": str(tmp_path / "pred_a_rgb")},
                },
                {
                    "name": "model_b",
                    "depth": {"path": str(tmp_path / "pred_b_depth")},
                },
            ],
        }
        path = self._write_config(tmp_path, config)
        result = load_config(path)
        assert len(result["datasets"]) == 2
        assert result["datasets"][0]["name"] == "model_a"
        assert result["datasets"][1]["name"] == "model_b"


# ---------------------------------------------------------------------------
# resolve_gt_segmentation_path (multi-scope archives)
# ---------------------------------------------------------------------------


def _scoped_archive(root: Path, scopes: dict[str, str]) -> Path:
    """Write one metadata scope per ``{scope: head modality key}``."""
    root.mkdir()
    np.save(root / "a.npy", np.zeros((2, 2), dtype=np.uint8))
    for scope, key in scopes.items():
        _write_index(
            root, scope=scope, key=key, loader="generic", function="map_2d",
            meta={}, files=[{"path": "a.npy", "id": "a"}],
        )
    return root


def _zip(root: Path) -> Path:
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    return archive


class TestResolveGtSegmentationPath:
    @pytest.mark.parametrize("as_zip", [False, True])
    def test_canonical_key_finds_the_semantic_segmentation_scope(
        self, tmp_path, as_zip
    ):
        root = _scoped_archive(
            tmp_path / "muses",
            {"semantic_segmentation": "semantic_segmentation", "depth": "depth"},
        )
        path = str(_zip(root) if as_zip else root)
        assert (
            resolve_gt_segmentation_path(path, config_key="segmentation")
            == f"{path}#scope=semantic_segmentation"
        )
        # A scope named after the config key keeps today's behaviour.
        assert (
            resolve_gt_segmentation_path(path, config_key="semantic_segmentation")
            == path
        )

    def test_alias_finds_a_segmentation_scope(self, tmp_path):
        root = _scoped_archive(
            tmp_path / "ds", {"segmentation": "segmentation", "rgb": "rgb"}
        )
        assert resolve_gt_segmentation_path(
            str(root), config_key="semantic_segmentation"
        ) == f"{root}#scope=segmentation"

    def test_scope_is_matched_by_its_head_modality_key(self, tmp_path):
        root = _scoped_archive(
            tmp_path / "ds", {"labels": "semantic_segmentation", "depth": "depth"}
        )
        assert resolve_gt_segmentation_path(
            str(root), config_key="segmentation"
        ) == f"{root}#scope=labels"

    def test_single_scope_is_selected(self, tmp_path):
        root = _scoped_archive(tmp_path / "ds", {"labels": "labels"})
        assert resolve_gt_segmentation_path(
            f"{root}:val", config_key="segmentation"
        ) == f"{root}:val#scope=labels"

    def test_ambiguous_scopes_are_an_error(self, tmp_path):
        root = _scoped_archive(
            tmp_path / "ds",
            {
                "segmentation_coarse": "segmentation",
                "segmentation_fine": "semantic_segmentation",
                "depth": "depth",
            },
        )
        with pytest.raises(ValueError, match="#scope="):
            resolve_gt_segmentation_path(str(root), config_key="segmentation")

    def test_unrelated_scopes_are_an_error(self, tmp_path):
        root = _scoped_archive(tmp_path / "ds", {"rgb": "rgb", "depth": "depth"})
        with pytest.raises(ValueError, match="rgb"):
            resolve_gt_segmentation_path(str(root), config_key="segmentation")

    def test_root_level_metadata_is_left_to_the_legacy_fallback(self, tmp_path):
        root = _scoped_archive(tmp_path / "ds", {"depth": "depth"})
        _write_index(
            root, scope=None, key="segmentation", loader="generic",
            function="map_2d", meta={}, files=[{"path": "a.npy", "id": "a"}],
        )
        assert resolve_gt_segmentation_path(
            str(root), config_key="segmentation"
        ) == str(root)

    def test_explicit_scope_and_unscoped_layouts_are_unchanged(self, tmp_path):
        root = _scoped_archive(
            tmp_path / "ds", {"semantic_segmentation": "semantic_segmentation"}
        )
        inline = f"{root}#scope=semantic_segmentation"
        assert resolve_gt_segmentation_path(inline, config_key="segmentation") == (
            inline
        )
        plain = tmp_path / "plain"
        plain.mkdir()
        assert resolve_gt_segmentation_path(str(plain), config_key="segmentation") == (
            str(plain)
        )
