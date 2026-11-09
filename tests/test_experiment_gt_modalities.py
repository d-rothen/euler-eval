"""Config ``experiments[].gt_modalities``: per-experiment selection of GT inputs."""

from __future__ import annotations

import pytest

from euler_eval import cli
from euler_eval.cli import _experiment_config, validate_experiments_config
from tests.test_depth_alignment_output import _patch_depth_metrics
from tests.test_experiments import _main, _results, _sky_config
from tests.test_sky_iou_cli import write_gt_segmentation


def _runs(config, *flags):
    parser = cli._build_parser()
    return cli._experiment_runs(
        parser, ["config.json", "--device", "cpu", *flags], config, selected=None
    )


def _dirs(tmp_path, *names):
    paths = {}
    for name in names:
        paths[name] = tmp_path / name
        paths[name].mkdir()
    return {name: str(path) for name, path in paths.items()}


class TestValidation:
    def test_accepts_gt_keys(self):
        validate_experiments_config(
            [
                {"name": "eval"},
                {"name": "no_ref", "gt_modalities": ["input_rgb"]},
                {"name": "seg", "gt_modalities": ["depth", "semantic_segmentation"]},
            ]
        )

    @pytest.mark.parametrize(
        ("gt_modalities", "message"),
        [
            ([], "must be a non-empty array of gt keys"),
            ("rgb", "must be a non-empty array of gt keys"),
            (["name"], "unknown gt input 'name'"),
            (["sky_mask"], "unknown gt input 'sky_mask'"),
            (["rgb", "rgb"], r"lists \['rgb'\] more than once"),
        ],
    )
    def test_rejects(self, gt_modalities, message):
        with pytest.raises(ValueError, match=message):
            validate_experiments_config(
                [{"name": "bad", "gt_modalities": gt_modalities}]
            )


def test_unlisted_gt_inputs_are_left_out():
    config = {
        "gt": {
            "name": "bench",
            "rgb": {"path": "/clear"},
            "input_rgb": {"path": "/hazy"},
            "semantic_segmentation": {"path": "/seg"},
        },
        "datasets": [],
    }

    reduced = _experiment_config(config, None, ["input_rgb", "segmentation"])

    assert reduced["gt"] == {
        "name": "bench",
        "input_rgb": {"path": "/hazy"},
        "semantic_segmentation": {"path": "/seg"},
    }
    assert config["gt"]["rgb"] == {"path": "/clear"}  # the base config is unchanged
    assert _experiment_config(config, None, None) is config


def test_both_rgb_references_and_the_input_alone(tmp_path):
    paths = _dirs(tmp_path, "clear", "hazy", "pred")
    config = {
        "gt": {"rgb": {"path": paths["clear"]}, "input_rgb": {"path": paths["hazy"]}},
        "datasets": [{"name": "dehazer", "rgb": {"path": paths["pred"]}}],
        "experiments": [
            {"name": "eval"},
            {"name": "no_reference", "gt_modalities": ["input_rgb"]},
        ],
    }

    full, input_only = _runs(config)

    assert set(full.config["gt"]) == {"rgb", "input_rgb"}
    assert set(input_only.config["gt"]) == {"input_rgb"}


def test_predictions_that_lose_their_gt_are_reported(tmp_path, capsys):
    paths = _dirs(tmp_path, "gt_rgb", "gt_depth", "pred_rgb", "pred_depth")
    config = {
        "gt": {"rgb": {"path": paths["gt_rgb"]}, "depth": {"path": paths["gt_depth"]}},
        "datasets": [
            {
                "name": "model",
                "rgb": {"path": paths["pred_rgb"]},
                "depth": {"path": paths["pred_depth"]},
            }
        ],
        "experiments": [{"name": "depth_only", "gt_modalities": ["depth", "rays"]}],
    }

    (run,) = _runs(config)

    err = capsys.readouterr().err
    assert "gt_modalities leave rgb predictions without a GT input" in err
    assert "gt_modalities ['rays'] are not configured in gt" in err
    assert set(run.config["gt"]) == {"depth"}


@pytest.mark.parametrize(
    ("gt_modalities", "args", "message"),
    [
        (["segmentation"], [], "'gt_modalities' leave an invalid gt section: gt must"),
        (["rgb"], [], "'gt_modalities' leave no prediction a GT input"),
        (["depth"], ["--mask-sky"], "--mask-sky requires gt.segmentation, which"),
        (["depth"], ["--sky-iou"], "--sky-iou requires gt.segmentation"),
    ],
)
def test_contradictions_stop_the_run_before_any_work(
    tmp_path, gt_modalities, args, message
):
    paths = _dirs(tmp_path, "gt_rgb", "gt_depth", "seg", "pred_depth")
    config = {
        "gt": {
            "rgb": {"path": paths["gt_rgb"]},
            "depth": {"path": paths["gt_depth"]},
            "segmentation": {"path": paths["seg"]},
        },
        "datasets": [{"name": "model", "depth": {"path": paths["pred_depth"]}}],
        "experiments": [{"name": "bad", "gt_modalities": gt_modalities, "args": args}],
    }

    with pytest.raises(ValueError, match=rf"experiments\[0\] \('bad'\): {message}"):
        _runs(config)


def test_the_protocol_records_only_the_selected_gt_inputs(
    monkeypatch, tmp_path, indexed_sky_prediction
):
    _patch_depth_metrics(monkeypatch)
    pred = indexed_sky_prediction["pred"]
    seg = write_gt_segmentation(tmp_path / "segmentation")
    _main(
        monkeypatch,
        _sky_config(
            tmp_path,
            indexed_sky_prediction,
            [
                {"name": "eval", "args": ["--mask-sky"]},
                {"name": "unmasked", "gt_modalities": ["depth"]},
            ],
            gt_extra={"segmentation": {"path": seg}},
        ),
    )

    masked = _results(pred, "eval")["protocol"]["parameters"]
    unmasked = _results(pred, "unmasked")["protocol"]["parameters"]
    assert masked["reference.segmentation.dataset"] == "sky_iou_test"
    assert unmasked["reference.segmentation.dataset"] is None
    assert unmasked["population.gt_sky_masking"] is False
