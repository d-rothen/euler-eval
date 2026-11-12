"""Tests for save_results with zip and directory multiplexing."""

import json
import zipfile
from pathlib import Path

import pytest

from euler_eval import cli
from euler_eval.cli import (
    _find_zip_ancestor,
    default_eval_path,
    save_results,
    validate_eval_name,
)
from euler_eval.zip_results import save_json_to_zip
from tests.test_depth_alignment_output import _patch_depth_metrics

# ---------------------------------------------------------------------------
# _find_zip_ancestor
# ---------------------------------------------------------------------------


class TestFindZipAncestor:
    def test_no_zip_in_path(self, tmp_path):
        p = tmp_path / "some" / "dir" / "eval.json"
        assert _find_zip_ancestor(p) == (None, None)

    def test_zip_file_detected(self, tmp_path):
        zp = tmp_path / "dataset.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "hello")

        p = zp / "eval.json"
        zip_path, internal = _find_zip_ancestor(p)
        assert zip_path == zp
        assert internal == "eval.json"

    def test_zip_with_nested_internal_path(self, tmp_path):
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "hello")

        p = zp / "sub" / "dir" / "eval.json"
        zip_path, internal = _find_zip_ancestor(p)
        assert zip_path == zp
        assert internal == "sub/dir/eval.json"

    def test_zip_suffix_but_not_a_file(self, tmp_path):
        """A directory named .zip should not be detected."""
        fake_zip = tmp_path / "not_a.zip"
        fake_zip.mkdir()
        p = fake_zip / "eval.json"
        assert _find_zip_ancestor(p) == (None, None)

    def test_zip_is_the_leaf(self, tmp_path):
        """If the path itself ends in .zip with no child, internal is empty."""
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "hello")
        zip_path, internal = _find_zip_ancestor(zp)
        assert zip_path == zp
        assert internal == ""


# ---------------------------------------------------------------------------
# save_json_to_zip
# ---------------------------------------------------------------------------


class TestSaveJsonToZip:
    def test_append_to_zip(self, tmp_path):
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("existing.txt", "keep me")

        data = {"score": 0.95}
        save_json_to_zip(zp, "eval.json", data)

        with zipfile.ZipFile(zp, "r") as zf:
            assert "existing.txt" in zf.namelist()
            assert "eval.json" in zf.namelist()
            result = json.loads(zf.read("eval.json"))
            assert result == data

    def test_replace_existing_entry(self, tmp_path):
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("existing.txt", "keep me")
            zf.writestr("eval.json", '{"old": true}')

        new_data = {"new": True, "score": 1.0}
        save_json_to_zip(zp, "eval.json", new_data)

        with zipfile.ZipFile(zp, "r") as zf:
            names = zf.namelist()
            # No duplicates
            assert names.count("eval.json") == 1
            assert "existing.txt" in names
            result = json.loads(zf.read("eval.json"))
            assert result == new_data

    def test_preserves_other_entries(self, tmp_path):
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("a.txt", "aaa")
            zf.writestr("b.txt", "bbb")
            zf.writestr("eval.json", '{"old": true}')

        save_json_to_zip(zp, "eval.json", {"replaced": True})

        with zipfile.ZipFile(zp, "r") as zf:
            assert zf.read("a.txt") == b"aaa"
            assert zf.read("b.txt") == b"bbb"
            assert json.loads(zf.read("eval.json")) == {"replaced": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scoped_dir(root, *scopes):
    """A dataset directory with ds-crawler metadata under each scope."""
    for scope in scopes:
        meta = root / ".ds_crawler" / scope
        meta.mkdir(parents=True)
        (meta / "index.json").write_text("{}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _unscoped_dir(root):
    """A dataset directory with root-level ds-crawler metadata only."""
    meta = root / ".ds_crawler"
    meta.mkdir(parents=True)
    (meta / "index.json").write_text("{}")
    return root


def _zip(path, *names):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("image_001.png", b"\x89PNG")
        for name in names:
            zf.writestr(name, "{}")
    return path


# ---------------------------------------------------------------------------
# default_eval_path
# ---------------------------------------------------------------------------


class TestDefaultEvalPath:
    def test_unscoped_dataset_uses_the_eval_folder(self, tmp_path):
        root = _unscoped_dir(tmp_path / "pred")
        assert default_eval_path(root, modality_key="depth") == (
            root / ".eval" / "eval.json"
        )

    def test_dataset_without_metadata_uses_the_eval_folder(self, tmp_path):
        root = tmp_path / "pred"
        assert default_eval_path(root, modality_key="depth") == (
            root / ".eval" / "eval.json"
        )

    def test_scope_named_like_the_config_key(self, tmp_path):
        root = _scoped_dir(tmp_path / "pred", "depth", "rgb")
        assert default_eval_path(root, modality_key="depth") == (
            root / ".eval" / "depth" / "eval.json"
        )
        assert default_eval_path(root, modality_key="rgb") == (
            root / ".eval" / "rgb" / "eval.json"
        )

    def test_inline_scope_selector_wins_over_the_key(self, tmp_path):
        root = _scoped_dir(tmp_path / "pred", "lidar_depth")
        assert default_eval_path(
            f"{root}#scope=lidar_depth", modality_key="depth"
        ) == root / ".eval" / "lidar_depth" / "eval.json"

    def test_scope_without_metadata_falls_back_like_euler_loading(self, tmp_path):
        """A scope that holds no metadata loads from the root .ds_crawler/."""
        root = _unscoped_dir(tmp_path / "pred")
        assert default_eval_path(
            f"{root}#scope=depth", modality_key="depth"
        ) == root / ".eval" / "eval.json"

    def test_split_adds_a_split_folder(self, tmp_path):
        root = _scoped_dir(tmp_path / "pred", "depth")
        assert default_eval_path(f"{root}:test", modality_key="depth") == (
            root / ".eval" / "depth" / "split_test" / "eval.json"
        )
        assert default_eval_path(root, modality_key="depth", split="val") == (
            root / ".eval" / "depth" / "split_val" / "eval.json"
        )

    def test_unscoped_split(self, tmp_path):
        root = _unscoped_dir(tmp_path / "pred")
        assert default_eval_path(f"{root}:test", modality_key="depth") == (
            root / ".eval" / "split_test" / "eval.json"
        )

    def test_eval_name_names_the_file(self, tmp_path):
        root = _scoped_dir(tmp_path / "pred", "depth")
        assert default_eval_path(
            f"{root}:test", modality_key="depth", eval_name="no_sky"
        ) == root / ".eval" / "depth" / "split_test" / "no_sky.json"

    def test_scoped_zip(self, tmp_path):
        zp = _zip(tmp_path / "pred.zip", ".ds_crawler/depth/index.json")
        assert default_eval_path(f"{zp}:test#scope=depth", modality_key="rgb") == (
            zp / ".eval" / "depth" / "split_test" / "eval.json"
        )


# ---------------------------------------------------------------------------
# save_results  (directory path)
# ---------------------------------------------------------------------------


class TestSaveResultsDirectory:
    def test_saves_to_depth_path(self, tmp_path):
        depth_dir = _scoped_dir(tmp_path / "pred_depth", "depth")

        results = {"depth": {"rmse": 0.5}}
        config = {"name": "model", "depth": {"path": str(depth_dir)}}
        out = save_results(results, config)

        assert out == depth_dir / ".eval" / "depth" / "eval.json"
        assert json.loads(out.read_text()) == results
        assert not (depth_dir / "eval.json").exists()

    def test_relative_depth_uses_its_own_scope(self, tmp_path):
        depth_dir = _scoped_dir(tmp_path / "pred", "relative_depth")

        results = {"depth": {"rmse": 0.5}}
        config = {"name": "model", "relative_depth": {"path": str(depth_dir)}}
        out = save_results(results, config, modality="depth")

        assert out == depth_dir / ".eval" / "relative_depth" / "eval.json"
        assert json.loads(out.read_text()) == results

    def test_saves_to_rgb_path_when_no_depth(self, tmp_path):
        rgb_dir = _unscoped_dir(tmp_path / "pred_rgb")

        config = {"name": "model", "rgb": {"path": str(rgb_dir)}}
        out = save_results({"rgb": {"psnr": 30.0}}, config)

        assert out == rgb_dir / ".eval" / "eval.json"

    def test_explicit_output_file(self, tmp_path):
        out_file = tmp_path / "custom" / "results.json"
        results = {"depth": {"rmse": 0.5}}
        config = {"name": "model", "output_file": str(out_file)}
        out = save_results(results, config)

        assert out == out_file
        assert out.exists()

    def test_fallback_to_cwd(self):
        results = {"depth": {"rmse": 0.5}}
        config = {"name": "model"}
        out = save_results(results, config)

        assert out == Path("eval.json")
        assert out.exists()
        out.unlink()  # cleanup

    def test_modalities_of_one_scoped_root_do_not_collide(self, tmp_path):
        root = _scoped_dir(tmp_path / "pred", "depth", "rgb")
        config = {
            "name": "model",
            "depth": {"path": f"{root}#scope=depth"},
            "rgb": {"path": f"{root}#scope=rgb"},
        }

        depth_out = save_results({"depth": 1}, config, modality="depth")
        rgb_out = save_results({"rgb": 2}, config, modality="rgb")

        assert depth_out == root / ".eval" / "depth" / "eval.json"
        assert rgb_out == root / ".eval" / "rgb" / "eval.json"
        assert json.loads(depth_out.read_text()) == {"depth": 1}
        assert json.loads(rgb_out.read_text()) == {"rgb": 2}

    def test_splits_of_one_root_do_not_collide(self, tmp_path):
        root = _scoped_dir(tmp_path / "pred", "depth")
        test_cfg = {"name": "model", "depth": {"path": f"{root}:test"}}
        val_cfg = {"name": "model", "depth": {"path": f"{root}:val"}}

        test_out = save_results({"split": "test"}, test_cfg, modality="depth")
        val_out = save_results({"split": "val"}, val_cfg, modality="depth")

        assert json.loads(test_out.read_text()) == {"split": "test"}
        assert json.loads(val_out.read_text()) == {"split": "val"}

    def test_eval_name_keeps_runs_side_by_side(self, tmp_path):
        root = _scoped_dir(tmp_path / "pred", "depth")
        config = {"name": "model", "depth": {"path": str(root)}}

        default = save_results({"run": 1}, config, modality="depth")
        named = save_results({"run": 2}, config, modality="depth", eval_name="no_sky")

        assert named == root / ".eval" / "depth" / "no_sky.json"
        assert json.loads(default.read_text()) == {"run": 1}
        assert json.loads(named.read_text()) == {"run": 2}

    def test_fallback_to_cwd_uses_the_eval_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = save_results({"x": 1}, {"name": "model"}, eval_name="no_sky")
        assert out == Path("no_sky.json")
        assert (tmp_path / "no_sky.json").is_file()

    def test_output_file_takes_precedence(self, tmp_path):
        out_file = tmp_path / "custom" / "results.json"
        root = _scoped_dir(tmp_path / "depth", "depth")
        config = {
            "name": "model",
            "depth": {"path": str(root)},
            "output_file": str(out_file),
        }

        out = save_results({"depth": 1}, config, modality="depth", eval_name="no_sky")

        assert out == out_file
        assert not (root / ".eval").exists()

    def test_does_not_create_a_ds_crawler_scope(self, tmp_path):
        """Results stay out of .ds_crawler/, so they never read as a scope."""
        from ds_crawler import list_metadata_scopes

        root = _unscoped_dir(tmp_path / "pred")
        save_results({"x": 1}, {"name": "model", "depth": {"path": str(root)}})

        assert list_metadata_scopes(root) == []


# ---------------------------------------------------------------------------
# save_results  (zip path)
# ---------------------------------------------------------------------------


class TestSaveResultsZip:
    def test_writes_eval_json_into_zip(self, tmp_path):
        zp = _zip(tmp_path / "dataset.zip")

        results = {"depth": {"rmse": 0.42}}
        config = {"name": "model", "depth": {"path": str(zp)}}
        out = save_results(results, config)

        assert out == zp / ".eval" / "eval.json"
        with zipfile.ZipFile(zp, "r") as zf:
            assert json.loads(zf.read(".eval/eval.json")) == results
            assert "eval.json" not in zf.namelist()

    def test_inline_split_and_scope_in_zip(self, tmp_path):
        zp = _zip(tmp_path / "dataset.zip", ".ds_crawler/rgb/index.json")

        results = {"rgb": {"psnr": 30.0}}
        config = {"name": "model", "rgb": {"path": f"{zp}:fog_day#scope=rgb"}}
        out = save_results(results, config, modality="rgb")

        assert out == zp / ".eval" / "rgb" / "split_fog_day" / "eval.json"
        with zipfile.ZipFile(zp, "r") as zf:
            stored = json.loads(zf.read(".eval/rgb/split_fog_day/eval.json"))
            assert stored == results

    def test_preserves_existing_zip_contents(self, tmp_path):
        zp = tmp_path / "dataset.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("image_001.png", b"\x89PNG")
            zf.writestr("image_002.png", b"\x89PNG")

        config = {"name": "model", "depth": {"path": str(zp)}}
        save_results({"depth": {}}, config)

        with zipfile.ZipFile(zp, "r") as zf:
            names = zf.namelist()
            assert "image_001.png" in names
            assert "image_002.png" in names
            assert ".eval/eval.json" in names

    def test_replaces_existing_results_in_zip(self, tmp_path):
        zp = _zip(tmp_path / "dataset.zip")
        config = {"name": "model", "depth": {"path": str(zp)}}

        save_results({"old": True}, config)
        save_results({"depth": {"rmse": 0.1}}, config)

        with zipfile.ZipFile(zp, "r") as zf:
            assert zf.namelist().count(".eval/eval.json") == 1
            assert json.loads(zf.read(".eval/eval.json")) == {"depth": {"rmse": 0.1}}

    def test_wrapper_folder_zip_keeps_its_single_root(self, tmp_path):
        """Zips that wrap everything in <stem>/ get results inside that folder."""
        from ds_crawler.zip_utils import get_zip_root_prefix

        zp = tmp_path / "model.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("model/.ds_crawler/depth/index.json", "{}")
            zf.writestr("model/frames/0001.npy", b"x")

        config = {"name": "model", "depth": {"path": str(zp)}}
        out = save_results({"depth": 1}, config, modality="depth")

        assert out == zp / ".eval" / "depth" / "eval.json"
        with zipfile.ZipFile(zp, "r") as zf:
            assert json.loads(zf.read("model/.eval/depth/eval.json")) == {"depth": 1}
        assert get_zip_root_prefix(zp) == "model/"

    def test_explicit_output_through_zip(self, tmp_path):
        """When output_file explicitly passes through a zip."""
        zp = tmp_path / "output.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "x")

        out_path = str(zp / "results" / "eval.json")
        results = {"depth": {"rmse": 0.3}}
        config = {"name": "model", "output_file": out_path}
        save_results(results, config)

        with zipfile.ZipFile(zp, "r") as zf:
            assert "results/eval.json" in zf.namelist()
            stored = json.loads(zf.read("results/eval.json"))
            assert stored == results


# ---------------------------------------------------------------------------
# --eval-name
# ---------------------------------------------------------------------------


class TestValidateEvalName:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("eval", "eval"),
            ("no_sky", "no_sky"),
            ("no_sky.json", "no_sky"),
            (" sky-depth.80 ", "sky-depth.80"),
        ],
    )
    def test_accepts(self, value, expected):
        assert validate_eval_name(value) == expected

    @pytest.mark.parametrize(
        "value", ["", "  ", ".json", ".hidden", "../eval", "evals/no_sky", "a b"]
    )
    def test_rejects(self, value):
        with pytest.raises(ValueError, match="not a valid evaluation name"):
            validate_eval_name(value)


class TestEvalNameCli:
    def _config(self, tmp_path, paths, **dataset_extra):
        dataset = {"name": "model", "depth": {"path": paths["pred"]}, **dataset_extra}
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"gt": {"depth": {"path": paths["gt"]}}, "datasets": [dataset]})
        )
        return config_path

    def _run(self, monkeypatch, config_path, *flags):
        monkeypatch.setattr(cli.sys, "argv", [
            "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
            "--no-sanity-check", "--depth-alignment", "none", *flags,
        ])
        cli.main()

    def test_named_run_sits_beside_the_default(
        self, monkeypatch, capsys, tmp_path, indexed_sky_prediction
    ):
        _patch_depth_metrics(monkeypatch)
        paths = indexed_sky_prediction
        config_path = self._config(tmp_path, paths)

        self._run(monkeypatch, config_path)
        self._run(monkeypatch, config_path, "--eval-name", "no_sky")
        out = capsys.readouterr().out

        # The fixture indexes the prediction under .ds_crawler/depth/.
        eval_dir = Path(paths["pred"]) / ".eval" / "depth"
        default = json.loads((eval_dir / "eval.json").read_text())
        named = json.loads((eval_dir / "no_sky.json").read_text())
        assert named["depth"] == default["depth"]
        assert f"results saved to: {eval_dir / 'no_sky.json'}" in out
        assert not (Path(paths["pred"]) / "eval.json").exists()

    def test_output_file_wins_over_eval_name(
        self, monkeypatch, tmp_path, indexed_sky_prediction
    ):
        _patch_depth_metrics(monkeypatch)
        paths = indexed_sky_prediction
        out_file = tmp_path / "results" / "model.json"
        config_path = self._config(tmp_path, paths, output_file=str(out_file))

        self._run(monkeypatch, config_path, "--eval-name", "no_sky")

        assert out_file.is_file()
        assert not (Path(paths["pred"]) / ".eval").exists()

    def test_invalid_name_is_a_usage_error(self, monkeypatch, capsys, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{}")
        monkeypatch.setattr(cli.sys, "argv", [
            "euler-eval", str(config_path), "--eval-name", "evals/no_sky",
        ])

        with pytest.raises(SystemExit) as exc:
            cli.main()

        assert exc.value.code == 2
        assert "--eval-name: 'evals/no_sky' is not a valid" in capsys.readouterr().err
