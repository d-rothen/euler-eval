"""Config ``experiments``: several named argument sets in one invocation."""

import json
import zipfile
from pathlib import Path

import pytest

from euler_eval import cli
from euler_eval.cli import default_eval_path, validate_experiments_config
from tests.test_cli_sparse_outputs import _sparse_depth_results
from tests.test_depth_alignment_output import _patch_depth_metrics
from tests.test_sky_iou_cli import write_gt_segmentation

EXPERIMENTS = [
    {"name": "eval"},
    {"name": "capped", "args": ["--sky-depth", 50]},
    {"name": "affine", "args": ["--depth-alignment", "affine"]},
]


def _config(tmp_path, paths, experiments=EXPERIMENTS, **extra):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {"depth": {"path": paths["gt"]}},
        "datasets": [{"name": "model", "depth": {"path": paths["pred"]}}],
        **({"experiments": experiments} if experiments is not None else {}),
        **extra,
    }))
    return config_path


def _main(monkeypatch, config_path, *flags, sanity=False):
    monkeypatch.setattr(cli.sys, "argv", [
        "euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0",
        "--depth-alignment", "none", *([] if sanity else ["--no-sanity-check"]),
        *flags,
    ])
    cli.main()


def _results(pred, name):
    path = default_eval_path(pred, modality_key="depth", eval_name=name)
    return json.loads(path.read_text()) if path.is_file() else None


def test_each_experiment_writes_its_named_results(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction
):
    _patch_depth_metrics(monkeypatch)
    pred = indexed_sky_prediction["pred"]
    _main(monkeypatch, _config(tmp_path, indexed_sky_prediction))
    out = capsys.readouterr().out

    saved = {name: _results(pred, name) for name in ("eval", "capped", "affine")}
    # The command line is the base; each experiment's args come after it.
    parameters = {name: s["protocol"]["parameters"] for name, s in saved.items()}
    assert parameters["eval"]["method.depth_alignment"] == "none"
    assert parameters["eval"]["method.sky_depth_m"] is None
    assert parameters["capped"]["method.sky_depth_m"] == 50.0
    assert parameters["capped"]["method.depth_alignment"] == "none"
    assert parameters["affine"]["method.depth_alignment"] == "affine"
    assert {name: s["meta"]["eval_name"] for name, s in saved.items()} == {
        "eval": "eval", "capped": "capped", "affine": "affine",
    }
    assert "Experiment 2/3: capped\n  args: --sky-depth 50" in out
    assert "Experiments: 3 of 3 succeeded" in out


def test_experiment_flag_selects_a_subset(monkeypatch, tmp_path, indexed_sky_prediction):
    _patch_depth_metrics(monkeypatch)
    pred = indexed_sky_prediction["pred"]
    _main(
        monkeypatch, _config(tmp_path, indexed_sky_prediction),
        "--experiment", "affine", "--experiment", "eval",
    )

    assert _results(pred, "affine") is not None
    assert _results(pred, "eval") is not None
    assert _results(pred, "capped") is None


def test_a_failing_experiment_does_not_stop_the_others(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction
):
    _patch_depth_metrics(monkeypatch)
    evaluate = cli.evaluate_depth_samples

    def flaky(**kwargs):
        if kwargs["alignment_mode"] == "affine":
            raise RuntimeError("boom")
        return evaluate(**kwargs)

    monkeypatch.setattr(cli, "evaluate_depth_samples", flaky)
    pred = indexed_sky_prediction["pred"]
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, _config(tmp_path, indexed_sky_prediction))

    assert exc.value.code == 1
    assert _results(pred, "eval") is not None
    assert _results(pred, "capped") is not None
    assert _results(pred, "affine") is None
    captured = capsys.readouterr()
    assert "Experiments: 2 of 3 succeeded" in captured.out
    assert "RuntimeError: boom" in captured.err
    assert "Failed: affine" in captured.err


def test_sanity_reports_are_named_per_experiment(
    monkeypatch, tmp_path, indexed_sky_prediction
):
    _patch_depth_metrics(monkeypatch)
    monkeypatch.chdir(tmp_path)
    experiments = [{"name": "eval"}, {"name": "affine", "args": ["--depth-alignment", "affine"]}]
    _main(
        monkeypatch, _config(tmp_path, indexed_sky_prediction, experiments),
        sanity=True,
    )

    assert (tmp_path / "sanity_check_report_eval.json").is_file()
    assert (tmp_path / "sanity_check_report_affine.json").is_file()
    assert not (tmp_path / "sanity_check_report.json").exists()


def test_euler_train_logs_one_evaluation_per_experiment(
    monkeypatch, tmp_path, indexed_sky_prediction
):
    _patch_depth_metrics(monkeypatch)
    calls = []

    class FakeRun:
        dir = tmp_path / "run"
        run_id = "run-1"

        def add_evaluation(self, key, **kwargs):
            calls.append(("add", key))

        def finish_evaluation(self, key, **kwargs):
            calls.append(("finish", key))

        def finish(self):
            calls.append(("run_finish", None))

    class FakeEulerTrain:
        @staticmethod
        def init(**kwargs):
            calls.append(("init", None))
            return FakeRun()

    monkeypatch.setattr(cli, "_euler_train", FakeEulerTrain)
    experiments = [{"name": "eval"}, {"name": "affine", "args": ["--depth-alignment", "affine"]}]
    config_path = _config(
        tmp_path, indexed_sky_prediction, experiments,
        euler_train={"dir": str(tmp_path / "runs")},
    )
    _main(monkeypatch, config_path)

    assert calls == [
        ("init", None),
        ("add", "model/eval"), ("add", "model/eval"), ("finish", "model/eval"),
        ("add", "model/affine"), ("add", "model/affine"), ("finish", "model/affine"),
        ("run_finish", None),
    ]


@pytest.mark.parametrize(
    ("experiments", "message"),
    [
        (
            [{"name": "ok"}, {"name": "bad", "args": ["--depth-alignment", "sideways"]}],
            "experiments[1] ('bad'): argument --depth-alignment: invalid choice",
        ),
        (
            [{"name": "bad", "args": ["--sky-depth", -1]}],
            "experiments[0] ('bad'): --sky-depth:",
        ),
        (
            [{"name": "bad", "args": ["--eval-name", "other"]}],
            "experiments[0] ('bad'): 'args' cannot set --eval-name or --experiment",
        ),
        (
            [{"name": "bad", "args": ["--experiment", "bad"]}],
            "experiments[0] ('bad'): 'args' cannot set --eval-name or --experiment",
        ),
        (
            [{"name": "bad", "args": ["stray"]}],
            "experiments[0] ('bad'): unrecognized arguments: stray",
        ),
    ],
)
def test_experiment_args_are_validated_before_any_work(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction, experiments, message
):
    monkeypatch.setattr(cli, "evaluate_depth_samples", pytest.fail)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, _config(tmp_path, indexed_sky_prediction, experiments))

    assert exc.value.code == 1
    assert f"Error loading config: {message}" in capsys.readouterr().err
    assert not (Path(indexed_sky_prediction["pred"]) / ".eval").exists()


@pytest.mark.parametrize(
    ("experiments", "flags", "message"),
    [
        (EXPERIMENTS, ("--eval-name", "x"), "--eval-name cannot be combined"),
        (EXPERIMENTS, ("--experiment", "nope"), "--experiment 'nope' is not in the"),
        (None, ("--experiment", "eval"), "which this config does not have"),
    ],
)
def test_command_line_conflicts_are_usage_errors(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction, experiments, flags, message
):
    with pytest.raises(SystemExit) as exc:
        _main(
            monkeypatch,
            _config(tmp_path, indexed_sky_prediction, experiments),
            *flags,
        )

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


class TestValidateExperimentsConfig:
    def test_accepts_names_and_token_lists(self):
        validate_experiments_config([
            {"name": "eval"},
            {"name": "no_sky", "args": ["--mask-sky", "--benchmark-depth-range", 1, 80.5]},
        ])

    @pytest.mark.parametrize(
        ("experiments", "message"),
        [
            ({}, "non-empty array"),
            ([], "non-empty array"),
            (["eval"], r"experiments\[0\] must be an object"),
            ([{"name": "a", "flags": []}], r"unknown keys \['flags'\]"),
            ([{"args": []}], r"experiments\[0\].name must be a string"),
            ([{"name": "a/b"}], r"experiments\[0\].name: 'a/b' is not a valid"),
            ([{"name": "a"}, {"name": "a.json"}], r"experiments\[1\].name 'a' is already used"),
            ([{"name": "a", "args": "--mask-sky"}], "array of command-line tokens"),
            ([{"name": "a", "args": [True]}], "array of command-line tokens"),
        ],
    )
    def test_rejects(self, experiments, message):
        with pytest.raises(ValueError, match=message):
            validate_experiments_config(experiments)


# ---------------------------------------------------------------------------
# modalities
# ---------------------------------------------------------------------------


def _sky_config(tmp_path, paths, experiments, *, gt_extra=None):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {"depth": {"path": paths["gt"]}, **(gt_extra or {})},
        "datasets": [{
            "name": "model",
            "depth": {"path": paths["pred"]},
            "sky_mask": {"path": paths["sky_mask"]},
        }],
        "experiments": experiments,
    }))
    return config_path


def test_modalities_decide_whether_the_sky_mask_is_used(
    monkeypatch, tmp_path, indexed_sky_prediction
):
    """--sky-depth fills predicted sky only when the sky_mask is loaded."""
    _patch_depth_metrics(monkeypatch)
    pred = indexed_sky_prediction["pred"]
    _main(monkeypatch, _sky_config(tmp_path, indexed_sky_prediction, [
        {"name": "filled", "args": ["--sky-depth", 50]},
        {"name": "capped", "args": ["--sky-depth", 50], "modalities": ["depth"]},
    ]))

    filled, capped = _results(pred, "filled"), _results(pred, "capped")
    p_filled = filled["protocol"]["parameters"]
    p_capped = capped["protocol"]["parameters"]
    assert p_filled["method.pred_sky_mask"] is True
    assert p_capped["method.pred_sky_mask"] is False
    assert {k for k in p_filled if p_filled[k] != p_capped[k]} == {"method.pred_sky_mask"}
    mae = "depth.eval.metric.standard.pixel_pool.mae".split(".")

    def pick(tree):
        for key in mae:
            tree = tree[key]
        return tree

    assert pick(filled) != pick(capped)
    assert capped["meta"]["pred"].get("sky_mask") is None


def test_modalities_keep_every_gt_input(monkeypatch, tmp_path, indexed_sky_prediction):
    """'modalities' selects predictions; GT segmentation stays for --mask-sky."""
    _patch_depth_metrics(monkeypatch)
    pred = indexed_sky_prediction["pred"]
    seg = write_gt_segmentation(tmp_path / "segmentation")
    _main(monkeypatch, _sky_config(
        tmp_path, indexed_sky_prediction,
        [{"name": "eval", "modalities": ["depth"], "args": ["--mask-sky"]}],
        gt_extra={"segmentation": {"path": seg}},
    ))

    parameters = _results(pred, "eval")["protocol"]["parameters"]
    assert parameters["population.gt_sky_masking"] is True
    assert parameters["reference.segmentation.dataset"] == "sky_iou_test"


def test_unconfigured_modalities_only_warn(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction
):
    _patch_depth_metrics(monkeypatch)
    _main(monkeypatch, _sky_config(tmp_path, indexed_sky_prediction, [
        {"name": "eval", "modalities": ["depth", "rays"]},
    ]))

    assert "modalities ['rays'] are not configured" in capsys.readouterr().err
    assert _results(indexed_sky_prediction["pred"], "eval") is not None


@pytest.mark.parametrize(
    ("experiment", "with_segmentation", "message"),
    [
        (
            {"name": "iou", "args": ["--sky-iou"], "modalities": ["depth"]},
            False,
            "experiments[0] ('iou'): --sky-iou requires gt.segmentation",
        ),
        (
            # A sky_mask is dropped with the depth prediction it belongs to.
            {"name": "masks", "modalities": ["sky_mask"]},
            True,
            "experiments[0] ('masks'): 'modalities' leave no dataset with a prediction",
        ),
        (
            {"name": "rgb", "modalities": ["rgb"]},
            True,
            "experiments[0] ('rgb'): 'modalities' leave no dataset with a prediction",
        ),
    ],
)
def test_modalities_are_checked_before_any_work(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction, experiment,
    with_segmentation, message,
):
    monkeypatch.setattr(cli, "evaluate_depth_samples", pytest.fail)
    gt_extra = {}
    if with_segmentation:
        gt_extra["segmentation"] = {
            "path": write_gt_segmentation(tmp_path / "segmentation")
        }
    config_path = _sky_config(
        tmp_path, indexed_sky_prediction, [experiment], gt_extra=gt_extra
    )
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, config_path)

    assert exc.value.code == 1
    assert f"Error loading config: {message}" in capsys.readouterr().err


def test_sparse_gt_config_with_prediction_modalities(monkeypatch, tmp_path):
    """The reported MUSES config: experiments choose predictions, GT stays whole."""
    paths = {}
    for name in ("lidar", "camera", "semantic", "depth", "dehazed", "points_3d",
                 "sky_mask"):
        paths[name] = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(paths[name], "w") as zf:
            zf.writestr("placeholder", b"data")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "experiments": [
            {"name": "eval", "modalities": ["depth"], "args": ["--mask-sky"]},
            {"name": "gt_sky_with_mask", "modalities": ["depth", "sky_mask"],
             "args": ["--mask-sky"]},
            {"name": "gt_and_pred_sky", "modalities": ["depth", "sky_mask"],
             "args": ["--mask-sky", "--mask-sky-prediction"]},
        ],
        "gt": {
            "sparse_depth": {"path": str(paths["lidar"])},
            "rgb": {"path": f"{paths['camera']}:fog_day"},
            "intrinsics": {"path": f"{paths['camera']}#scope=intrinsics"},
            "camera_extrinsics": {"path": f"{paths['camera']}#scope=camera_extrinsics"},
            "segmentation": {"path": str(paths["semantic"])},
        },
        "datasets": [{
            "name": "SD 2.1. Mari",
            "depth": {"path": str(paths["depth"])},
            "rgb": {"path": str(paths["dehazed"])},
            "points_3d": {"path": str(paths["points_3d"])},
            "sky_mask": {"path": str(paths["sky_mask"])},
        }],
        "distributions": {"n_bins": 30, "scale": "log", "min_error": 0.001,
                          "max_error": 240.0},
    }))

    built, evaluated = [], []
    monkeypatch.setattr(
        cli, "build_sparse_depth_eval_dataset",
        lambda **kwargs: built.append(kwargs) or [object()],
    )
    monkeypatch.setattr(
        cli, "get_sparse_depth_metadata", lambda _ds: {"pred_radial_depth": False}
    )
    monkeypatch.setattr(
        cli, "evaluate_sparse_depth_samples",
        lambda **kwargs: evaluated.append(kwargs) or _sparse_depth_results(),
    )
    for unused in ("evaluate_rgb_samples", "evaluate_points_3d_sparse_samples"):
        monkeypatch.setattr(cli, unused, pytest.fail)
    _main(monkeypatch, config_path)

    gt_masking = [kwargs["sky_mask_enabled"] for kwargs in evaluated]
    pred_masking = [kwargs["pred_sky_mask_enabled"] for kwargs in evaluated]
    assert gt_masking == [True, True, True]
    assert pred_masking == [False, False, True]
    assert [kwargs["segmentation_path"] for kwargs in built] == [str(paths["semantic"])] * 3
    # The sky_mask is only read by a flag that uses it (here --mask-sky-prediction).
    assert [kwargs["pred_sky_mask_path"] for kwargs in built] == [
        None, None, str(paths["sky_mask"]),
    ]
    with zipfile.ZipFile(paths["depth"]) as zf:
        saved = {
            name: json.loads(zf.read(f".eval/{name}.json"))
            for name in ("eval", "gt_sky_with_mask", "gt_and_pred_sky")
        }
    assert [s["protocol"]["parameters"]["method.pred_sky_mask"]
            for s in saved.values()] == [False, False, True]


def test_experiment_config_keeps_only_listed_predictions():
    config = {
        "gt": {
            "name": "GT",
            "depth": {"path": "/gt/depth"},
            "semantic_segmentation": {"path": "/gt/seg"},
            "calibration": {"path": "/gt/calib"},
        },
        "datasets": [
            {
                "name": "a", "output_file": "/out/a.json",
                "relative_depth": {"path": "/a/depth"}, "sky_mask": {"path": "/a/sky"},
                "rgb": {"path": "/a/rgb"},
            },
            {"name": "b", "rgb": {"path": "/b/rgb"}},
        ],
        "distributions": True,
    }

    reduced = cli._experiment_config(config, ["depth"])

    assert reduced["gt"] == config["gt"]
    assert reduced["datasets"] == [
        {"name": "a", "output_file": "/out/a.json", "relative_depth": {"path": "/a/depth"}},
    ]
    assert reduced["distributions"] is True
    assert cli._experiment_config(config, ["depth", "sky_mask"])["datasets"] == [{
        "name": "a", "output_file": "/out/a.json",
        "relative_depth": {"path": "/a/depth"}, "sky_mask": {"path": "/a/sky"},
    }]
    assert cli._experiment_config(config, ["rgb", "sky_mask"])["datasets"] == [
        {"name": "a", "output_file": "/out/a.json", "rgb": {"path": "/a/rgb"}},
        {"name": "b", "rgb": {"path": "/b/rgb"}},
    ]
    assert cli._experiment_config(config, None) is config


@pytest.mark.parametrize(
    ("modalities", "message"),
    [
        ([], "non-empty array of modality names"),
        ("depth", "non-empty array of modality names"),
        (["relative_depth"], r"unknown modality 'relative_depth' \(use 'depth'"),
        (["segmentation"], "'segmentation' is a GT input"),
        (["semantic_segmentation"], "'semantic_segmentation' is a GT input"),
        (["sparse_depth"], "'sparse_depth' is a GT input"),
        (["intrinsics"], "'intrinsics' is a GT input"),
        (["normals"], "unknown modality 'normals'"),
        (["depth", "depth"], r"lists \['depth'\] more than once"),
    ],
)
def test_modalities_validation(modalities, message):
    with pytest.raises(ValueError, match=message):
        validate_experiments_config([{"name": "a", "modalities": modalities}])
