"""Regression tests for modality-atomic sparse pointcloud evaluations."""

import json
import sys
import zipfile
from unittest.mock import Mock

from euler_eval import cli


def _sparse_depth_results():
    return {
        "sparse_depth_native": None,
        "sparse_depth_metric": {
            "standard": {"image_mean": {"absrel": 0.1}},
        },
        "sparse_depth_benchmark": None,
        "dataset_info": {"num_pairs": 1},
        "space_info": {
            "input_space_detected": "metric",
            "metric_space_source": "native",
            "calibration_mode": "auto_affine",
            "calibration_applied": False,
            "emitted_spaces": ["metric"],
            "canonical_space": "metric",
        },
        "spatial_info": {},
        "per_file_metrics": {
            "files": [
                {
                    "id": "frame-1",
                    "metrics": {
                        "sparse_depth_metric": {
                            "standard": {"image_mean": {"absrel": 0.1}}
                        }
                    },
                }
            ]
        },
    }


def _sparse_points_3d_results():
    return {
        "points_3d_native": {
            "point_error": {"image_mean": {"mae3d": 0.2}},
        },
        "points_3d_metric": None,
        "dataset_info": {
            "num_pairs": 1,
            "fov_domain": "sfov",
            "threshold_deg": 15.0,
        },
        "space_info": {
            "input_space_detected": "metric",
            "metric_space_source": None,
            "calibration_mode": "auto_affine",
            "calibration_applied": False,
            "emitted_spaces": ["native"],
            "canonical_space": "native",
        },
        "spatial_info": {},
        "per_file_metrics": {
            "files": [
                {
                    "id": "frame-1",
                    "metrics": {
                        "points_3d_native": {
                            "point_error": {"image_mean": {"mae3d": 0.2}}
                        }
                    },
                }
            ]
        },
    }


def test_depth_modality_writes_only_sparse_depth_metrics(tmp_path, monkeypatch, capsys):
    """A depth prediction must not trigger an implicit points-3D evaluation."""
    gt_archive = tmp_path / "gt.zip"
    pred_archive = tmp_path / "depth.zip"
    for archive in (gt_archive, pred_archive):
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("placeholder", b"data")

    config = {
        "gt": {
            "sparse_depth": {"path": f"{gt_archive}#scope=sparse_depth"},
            "intrinsics": {"path": f"{gt_archive}#scope=intrinsics"},
            "camera_extrinsics": {"path": f"{gt_archive}#scope=camera_extrinsics"},
        },
        "datasets": [
            {
                "name": "model",
                "depth": {"path": str(pred_archive)},
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    dataset = [object()]
    monkeypatch.setattr(
        cli,
        "build_sparse_depth_eval_dataset",
        lambda **kwargs: dataset,
    )
    monkeypatch.setattr(
        cli,
        "get_sparse_depth_metadata",
        lambda _dataset: {"pred_radial_depth": False},
    )
    monkeypatch.setattr(
        cli,
        "evaluate_sparse_depth_samples",
        lambda **kwargs: _sparse_depth_results(),
    )
    points_3d_evaluator = Mock(
        side_effect=AssertionError("depth must not be evaluated as points_3d")
    )
    monkeypatch.setattr(cli, "evaluate_points_3d_sparse_samples", points_3d_evaluator)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "euler-eval",
            str(config_path),
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--no-sanity-check",
        ],
    )

    cli.main()

    with zipfile.ZipFile(pred_archive) as zf:
        names = zf.namelist()
        assert names.count("eval.json") == 1
        assert "points3d_eval.json" not in names
        payload = json.loads(zf.read("eval.json"))

    assert payload["metricSet"]["metricNamespace"] == "sparsedepth.eval"
    assert "additionalMetricSets" not in payload["metricSet"]
    assert "sparsedepth" in payload
    assert "points3d" not in payload
    file_metrics = payload["per_file_metrics"]["files"][0]["metrics"]
    assert set(file_metrics) == {"sparsedepth"}
    points_3d_evaluator.assert_not_called()

    output = capsys.readouterr().out
    assert f"Sparse depth results saved to: {pred_archive}/eval.json" in output
    assert "[POINTS_3D · SPARSE]" not in output
    assert "Points-3D (sparse GT) results saved" not in output


def test_points_3d_modality_writes_only_points_3d_metrics(
    tmp_path, monkeypatch, capsys
):
    """Sparse points-3D metrics require an explicit points_3d prediction."""
    gt_archive = tmp_path / "gt.zip"
    pred_archive = tmp_path / "points_3d.zip"
    for archive in (gt_archive, pred_archive):
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("placeholder", b"data")

    config = {
        "gt": {
            "sparse_depth": {"path": f"{gt_archive}#scope=sparse_depth"},
            "intrinsics": {"path": f"{gt_archive}#scope=intrinsics"},
            "camera_extrinsics": {"path": f"{gt_archive}#scope=camera_extrinsics"},
        },
        "datasets": [
            {
                "name": "model",
                "points_3d": {"path": str(pred_archive)},
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    dataset = [object()]
    monkeypatch.setattr(
        cli,
        "build_points_3d_sparse_eval_dataset",
        lambda **kwargs: dataset,
    )
    monkeypatch.setattr(
        cli,
        "get_points_3d_metadata",
        lambda _dataset: {"fov_domain": None},
    )
    monkeypatch.setattr(
        cli,
        "evaluate_points_3d_sparse_samples",
        lambda **kwargs: _sparse_points_3d_results(),
    )
    sparse_depth_evaluator = Mock(
        side_effect=AssertionError("points_3d must not be evaluated as depth")
    )
    monkeypatch.setattr(cli, "evaluate_sparse_depth_samples", sparse_depth_evaluator)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "euler-eval",
            str(config_path),
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--no-sanity-check",
        ],
    )

    cli.main()

    with zipfile.ZipFile(pred_archive) as zf:
        assert zf.namelist().count("eval.json") == 1
        payload = json.loads(zf.read("eval.json"))

    assert payload["metricSet"]["metricNamespace"] == "points3d.eval"
    assert "additionalMetricSets" not in payload["metricSet"]
    assert "points3d" in payload
    assert "sparsedepth" not in payload
    file_metrics = payload["per_file_metrics"]["files"][0]["metrics"]
    assert set(file_metrics) == {"points3d"}
    sparse_depth_evaluator.assert_not_called()

    output = capsys.readouterr().out
    assert "[SPARSE_DEPTH]" not in output
    assert "[POINTS_3D · SPARSE] Evaluating" in output
    assert f"Points-3D (sparse GT) results saved to: {pred_archive}/eval.json" in output
