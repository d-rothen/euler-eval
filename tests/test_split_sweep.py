"""Config ``splits``: one evaluation per ds-crawler split of the GT."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ds_crawler import build_dataset_head, create_mapped_dataset_splits
from ds_crawler.artifacts import save_output_artifacts

from euler_eval import cli
from euler_eval.cli import default_eval_path
from euler_eval.split_sweep import (
    GT_INPUT_KEYS,
    SplitSweep,
    ignore_all_split,
    list_gt_splits,
    natural_order,
    resolve_split_sweep,
    split_gt,
    validate_splits_config,
)
from tests.test_depth_alignment_output import _patch_depth_metrics
from tests.test_experiments import _main

FOG = {"gt_modality": "depth", "match": r"fog\d+"}


def _write_splits(root, splits, scope="depth"):
    """Write disjoint ds-crawler splits of file IDs (one call per overlap)."""
    create_mapped_dataset_splits(
        root,
        {"splits": {name: [[file_id] for file_id in ids] for name, ids in splits.items()}},
        metadata_scope=scope,
    )


@pytest.fixture
def fog_gt(indexed_sky_prediction):
    """GT depth with a split per fog level and an overlapping ``all`` split."""
    gt = indexed_sky_prediction["gt"]
    _write_splits(gt, {"fog25": ["first"], "fog100": ["second"]})
    _write_splits(gt, {"all": ["first", "second"]})
    return indexed_sky_prediction


def _config(tmp_path, paths, splits=FOG, **extra):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {"depth": {"path": paths["gt"]}},
        "datasets": [{"name": "model", "depth": {"path": paths["pred"]}}],
        **({"splits": splits} if splits is not None else {}),
        **extra,
    }))
    return config_path


def _results(pred, split, name="eval"):
    path = default_eval_path(pred, modality_key="depth", split=split, eval_name=name)
    return json.loads(path.read_text()) if path.is_file() else None


def test_each_split_writes_its_own_results(monkeypatch, capsys, tmp_path, fog_gt):
    _patch_depth_metrics(monkeypatch)
    pred = fog_gt["pred"]
    _main(monkeypatch, _config(tmp_path, fog_gt))
    out = capsys.readouterr().out

    # The prediction is not split: pairing by ID restricts it to the split.
    for split, frame in (("fog25", "first"), ("fog100", "second")):
        saved = _results(pred, split)
        assert saved["dataset_info"]["num_pairs"] == 1
        assert saved["meta"]["gt"]["split"] == split
        assert saved["protocol"]["parameters"]["reference.gt.split"] == split
        assert frame in json.dumps(saved["per_file_metrics"])
    assert _results(pred, "all") is None
    assert _results(pred, None) is None
    assert "Split sweep: 2 of gt.depth's splits matching 'fog\\\\d+': fog25, fog100" in out
    # Natural order: fog25 runs before fog100.
    assert out.index("Run 1/2: split_fog25") < out.index("Run 2/2: split_fog100")
    assert "Runs: 2 of 2 succeeded" in out


def test_without_match_every_split_is_evaluated(monkeypatch, tmp_path, fog_gt):
    _patch_depth_metrics(monkeypatch)
    _main(monkeypatch, _config(tmp_path, fog_gt, {"gt_modality": "depth"}))

    assert _results(fog_gt["pred"], "all")["dataset_info"]["num_pairs"] == 2
    assert _results(fog_gt["pred"], "fog25") is not None


def test_paths_selecting_all_are_swept_with_a_deprecation_warning(
    monkeypatch, tmp_path, fog_gt
):
    _patch_depth_metrics(monkeypatch)
    paths = {
        "gt": f"{fog_gt['gt']}:all",
        "pred": f"{fog_gt['pred']}:all#scope=depth",
    }
    with pytest.warns(FutureWarning, match=r"gt.depth, datasets\[0\].depth select "
                      "split 'all', which 'splits' ignores"):
        _main(monkeypatch, _config(tmp_path, paths))

    for split in ("fog25", "fog100"):
        saved = _results(fog_gt["pred"], split)
        assert saved["dataset_info"]["num_pairs"] == 1
        assert saved["meta"]["gt"]["split"] == split
    assert _results(fog_gt["pred"], "all") is None


def test_split_flag_selects_a_subset(monkeypatch, tmp_path, fog_gt):
    _patch_depth_metrics(monkeypatch)
    _main(monkeypatch, _config(tmp_path, fog_gt), "--split", "fog100")

    assert _results(fog_gt["pred"], "fog100") is not None
    assert _results(fog_gt["pred"], "fog25") is None


def test_experiments_run_once_per_split(monkeypatch, capsys, tmp_path, fog_gt):
    _patch_depth_metrics(monkeypatch)
    monkeypatch.chdir(tmp_path)
    calls = []

    class FakeRun:
        dir = tmp_path / "run"
        run_id = "run-1"

        def add_evaluation(self, key, **kwargs):
            calls.append(key)

        def finish_evaluation(self, key, **kwargs):
            pass

        def finish(self):
            pass

    class FakeEulerTrain:
        @staticmethod
        def init(**kwargs):
            return FakeRun()

    monkeypatch.setattr(cli, "_euler_train", FakeEulerTrain)
    experiments = [{"name": "eval"}, {"name": "affine", "args": ["--depth-alignment", "affine"]}]
    _main(
        monkeypatch,
        _config(
            tmp_path, fog_gt, experiments=experiments,
            euler_train={"dir": str(tmp_path / "runs")},
        ),
        sanity=True,
    )
    out = capsys.readouterr().out

    pred = fog_gt["pred"]
    for split in ("fog25", "fog100"):
        assert _results(pred, split, "eval")["protocol"]["parameters"][
            "method.depth_alignment"
        ] == "none"
        assert _results(pred, split, "affine")["protocol"]["parameters"][
            "method.depth_alignment"
        ] == "affine"
        for name in ("eval", "affine"):
            assert (tmp_path / f"sanity_check_report_split_{split}_{name}.json").is_file()
    # Split by split: every experiment of a split before the next split.
    assert list(dict.fromkeys(calls)) == [
        "model/split_fog25/eval", "model/split_fog25/affine",
        "model/split_fog100/eval", "model/split_fog100/affine",
    ]
    assert "Run 2/4: split_fog25/affine\n  split: fog25\n  args: --depth-alignment affine" in out
    assert "Runs: 4 of 4 succeeded" in out


def test_a_failing_split_does_not_stop_the_others(monkeypatch, capsys, tmp_path, fog_gt):
    _patch_depth_metrics(monkeypatch)
    evaluate = cli.evaluate_depth_samples
    calls = []

    def flaky(**kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return evaluate(**kwargs)

    monkeypatch.setattr(cli, "evaluate_depth_samples", flaky)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, _config(tmp_path, fog_gt))

    assert exc.value.code == 1
    assert _results(fog_gt["pred"], "fog25") is None
    assert _results(fog_gt["pred"], "fog100") is not None
    captured = capsys.readouterr()
    assert "Runs: 1 of 2 succeeded" in captured.out
    assert "Failed: split_fog25" in captured.err


def test_output_file_is_filled_per_split(monkeypatch, tmp_path, fog_gt):
    _patch_depth_metrics(monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {"depth": {"path": fog_gt["gt"]}},
        "datasets": [{
            "name": "model",
            "depth": {"path": fog_gt["pred"]},
            "output_file": str(tmp_path / "out" / "model_{split}.json"),
        }],
        "splits": FOG,
    }))
    _main(monkeypatch, config_path)

    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "model_fog100.json", "model_fog25.json",
    ]


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ({"gt_modality": "depth", "match": "rain\\d+"}, "splits.match 'rain\\\\d+' matches none of gt.depth's splits (all, fog25, fog100)"),
        # The pattern must match the whole name.
        ({"gt_modality": "depth", "match": "fog"}, "splits.match 'fog' matches none of gt.depth's splits (all, fog25, fog100); it must match a whole split name"),
    ],
)
def test_a_sweep_that_matches_nothing_stops_before_any_work(
    monkeypatch, capsys, tmp_path, fog_gt, splits, message
):
    monkeypatch.setattr(cli, "evaluate_depth_samples", pytest.fail)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, _config(tmp_path, fog_gt, splits))

    assert exc.value.code == 1
    assert f"Error loading config: {message}" in capsys.readouterr().err


def test_gt_without_splits_stops_before_any_work(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction
):
    monkeypatch.setattr(cli, "evaluate_depth_samples", pytest.fail)
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, _config(tmp_path, indexed_sky_prediction))

    assert exc.value.code == 1
    assert "gt.depth defines no ds-crawler splits" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("splits", "flags", "message"),
    [
        (FOG, ("--split", "fog30"), "--split 'fog30' is not among the splits the config's 'splits' sweep matches (fog25, fog100)"),
        (FOG, ("--split", "all"), "is not among the splits"),
        (None, ("--split", "fog25"), "which this config does not have"),
    ],
)
def test_split_flag_conflicts_are_usage_errors(
    monkeypatch, capsys, tmp_path, fog_gt, splits, flags, message
):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, _config(tmp_path, fog_gt, splits), *flags)

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------
# GT inputs without the split
# ---------------------------------------------------------------------------


def _sweep(**available):
    return SplitSweep(
        gt_modality="depth",
        match=None,
        splits=("fog25",),
        available={key: frozenset(names) for key, names in available.items()},
    )


def _run(gt, dataset, *flags):
    return cli._Run(
        name=None,
        args=cli._build_parser().parse_args(["config.json", *flags]),
        config={"gt": gt, "datasets": [dataset]},
        distribution_config=None,
        metric_sets=(),
    )


GT = {
    "name": "CEREMA",
    "depth": {"path": "/gt"},
    "rgb": {"path": "/gt#scope=rgb"},
    "intrinsics": {"path": "/gt#scope=intrinsics"},
}


def test_inputs_read_the_split_where_they_define_it():
    split = split_gt(GT, _sweep(depth={"fog25"}, rgb=set(), intrinsics=set()), "fog25")

    assert split == {**GT, "depth": {"path": "/gt", "split": "fog25"}}


def test_the_gt_a_prediction_is_scored_against_must_define_the_split():
    sweep = _sweep(depth={"fog25"}, rgb={"clear"}, intrinsics=set())
    depth_only = _run(GT, {"name": "m", "depth": {"path": "/p"}})
    with_rgb = _run(GT, {"name": "m", "depth": {"path": "/p"}, "rgb": {"path": "/p"}})

    # gt.rgb only joins the depth evaluation by ID: the split still holds.
    cli._check_split_coverage(depth_only, sweep, ("fog25",))
    # RGB that is not evaluated needs no split.
    cli._check_split_coverage(
        _run(GT, with_rgb.config["datasets"][0], "--skip-rgb"), sweep, ("fog25",)
    )
    with pytest.raises(
        ValueError,
        match=r"gt.rgb, which rgb predictions are scored against, has no split "
        r"fog25 \(its splits: clear\)",
    ):
        cli._check_split_coverage(with_rgb, sweep, ("fog25",))


def test_coverage_errors_name_the_experiment(monkeypatch, capsys, tmp_path, fog_gt):
    other = Path(fog_gt["sky_mask"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "gt": {"depth": {"path": fog_gt["gt"]}, "points_3d": {"path": str(other)}},
        "datasets": [{"name": "model", "points_3d": {"path": fog_gt["pred"]}}],
        "experiments": [{"name": "p3d"}],
        "splits": FOG,
    }))
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, config_path)

    assert exc.value.code == 1
    assert (
        "Error loading config: splits: experiment 'p3d': gt.points_3d, which "
        "points_3d predictions are scored against, has no split fog25, fog100"
    ) in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Listing and validation
# ---------------------------------------------------------------------------


def test_splits_are_listed_where_loading_reads_them(tmp_path):
    root = tmp_path / "flat"
    root.mkdir()
    (root / "a.npy").write_bytes(b"")
    head = build_dataset_head(
        dataset={"name": "flat"},
        modality={"key": "depth", "meta": {"radial_depth": False}},
        addons={"euler_loading": {
            "version": "1.0", "loader": "generic_dense_depth", "function": "depth",
        }},
    )
    save_output_artifacts(root, {
        "contract": {"kind": "dataset_index", "version": "1.0"},
        "head_file": "dataset-head.json", "head": head,
        "generator": {"name": "tests"}, "indexing": {}, "execution": {},
        "index": {"files": [{"path": "a.npy", "id": "a"}]},
    })
    _write_splits(root, {"fog5": ["a"]}, scope=None)

    # No .ds_crawler/depth/: loading falls back to the root-level metadata.
    assert list_gt_splits("depth", {"path": str(root)}) == ["fog5"]
    sweep = resolve_split_sweep({"gt_modality": "depth"}, {"depth": {"path": str(root)}})
    assert sweep.splits == ("fog5",)


def test_natural_order():
    assert natural_order(["fog100", "fog25", "clear", "fog5", "rain15"]) == [
        "clear", "fog5", "fog25", "fog100", "rain15",
    ]


def test_gt_input_keys_match_the_cli():
    assert set(GT_INPUT_KEYS) == set(cli._GT_INPUT_KEYS)


class TestValidateSplitsConfig:
    CONFIG = {
        "gt": {"depth": {"path": "/gt"}, "intrinsics": {"path": "/gt#scope=intrinsics"}},
        "datasets": [{"name": "m", "depth": {"path": "/p"}}],
    }

    def test_accepts(self):
        validate_splits_config(FOG, self.CONFIG)
        validate_splits_config({"gt_modality": "intrinsics"}, self.CONFIG)

    @pytest.mark.parametrize(
        ("splits", "message"),
        [
            ("fog", "'splits' must be an object"),
            ({"gt_modality": "depth", "pattern": "x"}, r"unknown keys \['pattern'\]"),
            ({"match": "fog"}, "splits.gt_modality must name the gt input"),
            ({"gt_modality": "lidar"}, "unknown gt input 'lidar'"),
            ({"gt_modality": "rgb"}, "gt.rgb is not configured"),
            ({"gt_modality": "depth", "match": ""}, "non-empty regular expression"),
            ({"gt_modality": "depth", "match": "fog("}, "not a valid regular expression"),
        ],
    )
    def test_rejects(self, splits, message):
        with pytest.raises(ValueError, match=message):
            validate_splits_config(splits, self.CONFIG)

    @pytest.mark.parametrize(
        ("gt", "dataset", "message"),
        [
            ({"depth": {"path": "/gt:test"}}, {}, r"gt.depth selects split 'test'"),
            (
                {"depth": {"path": "/gt"}, "intrinsics": {"path": "/gt", "split": "test"}},
                {},
                r"gt.intrinsics selects split 'test'",
            ),
            (
                {"depth": {"path": "/gt"}},
                {"depth": {"path": "/p:test#scope=depth"}},
                r"datasets\[0\].depth selects split 'test'",
            ),
            (
                {"depth": {"path": "/gt"}},
                {"output_file": "/out/model.json"},
                r"datasets\[0\].output_file names one file.*\{split\}",
            ),
        ],
    )
    def test_rejects_paths_that_fix_one_file_or_split(self, gt, dataset, message):
        config = {"gt": gt, "datasets": [{"name": "m", "depth": {"path": "/p"}, **dataset}]}
        with pytest.raises(ValueError, match=message):
            validate_splits_config(FOG, config)


class TestIgnoreAllSplit:
    def test_drops_all_and_keeps_the_scope(self):
        config = {
            "gt": {
                "depth": {"path": "/gt:all#scope=lidar"},
                "intrinsics": {"path": "/gt", "split": "all"},
                "calibration": {"path": "/cal:test"},
            },
            "datasets": [{"name": "m", "depth": {"path": "/p:all", "max_depth": 80}}],
            "splits": FOG,
        }
        with pytest.warns(FutureWarning) as caught:
            result = ignore_all_split(config)

        assert result["gt"] == {
            "depth": {"path": "/gt#scope=lidar"},
            "intrinsics": {"path": "/gt"},
            "calibration": {"path": "/cal:test"},
        }
        assert result["datasets"] == [
            {"name": "m", "depth": {"path": "/p", "max_depth": 80}}
        ]
        assert result["splits"] == FOG
        assert config["gt"]["depth"]["path"] == "/gt:all#scope=lidar"
        [warning] = caught
        assert str(warning.message) == (
            "gt.depth, gt.intrinsics, datasets[0].depth select split 'all', "
            "which 'splits' ignores: each run selects its own split. Selecting "
            "'all' alongside 'splits' is deprecated and will be an error in a "
            "future release; remove it from those paths."
        )
        # Any other split is still rejected.
        with pytest.raises(ValueError, match="gt.calibration selects split 'test'"):
            validate_splits_config(FOG, result)

    def test_leaves_other_configs_alone(self, recwarn):
        config = {
            "gt": {"depth": {"path": "/gt:fog25"}},
            "datasets": [{"name": "m", "depth": {"path": "/p"}}],
        }
        assert ignore_all_split(config) is config
        assert not recwarn.list

    def test_only_a_sweep_ignores_all(self, tmp_path, recwarn):
        (tmp_path / "gt").mkdir()
        (tmp_path / "pred").mkdir()
        gt = f"{tmp_path / 'gt'}:all"
        path = tmp_path / "config.json"
        path.write_text(json.dumps({
            "gt": {"depth": {"path": gt}},
            "datasets": [{"name": "m", "depth": {"path": str(tmp_path / "pred")}}],
        }))
        assert cli.load_config(str(path))["gt"]["depth"]["path"] == gt
        assert not recwarn.list
