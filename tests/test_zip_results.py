"""Results bound for zip archives are written once per archive."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import pytest

from euler_eval import zip_results
from euler_eval.cli import save_results
from euler_eval.zip_results import (
    ZipWriteError,
    batched_zip_writes,
    lock_path,
    save_json_to_zip,
)
from tests.test_depth_alignment_output import _patch_depth_metrics
from tests.test_experiments import _main
from tests.test_split_sweep import FOG, _write_splits


def _zip(path, *names):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, b"x")
    return path


def _entries(zp):
    with zipfile.ZipFile(zp) as zf:
        return {
            name: json.loads(zf.read(name)) if ".eval/" in name else None
            for name in zf.namelist()
        }


@pytest.fixture
def archive_writes(monkeypatch):
    """Record each archive write; spool results under a known directory."""
    calls = []
    write = zip_results.write_zip_entries

    def recording(zip_path, entries):
        calls.append((Path(zip_path).name, sorted(entries)))
        write(zip_path, entries)

    monkeypatch.setattr(zip_results, "write_zip_entries", recording)
    return calls


@pytest.fixture
def spool_root(monkeypatch, tmp_path):
    root = tmp_path / "local_tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def test_each_archive_is_written_once_at_the_end(tmp_path, archive_writes, spool_root):
    pred = _zip(tmp_path / "pred.zip", "frames/first.npy")
    wrapped = _zip(tmp_path / "model.zip", "model/.ds_crawler/depth/index.json")
    config = {"name": "m", "depth": {"path": str(pred)}}

    with batched_zip_writes():
        for split in ("fog25", "fog100"):
            save_results({"split": split}, config, split=split)
        save_results({"first": True}, {"name": "m", "depth": {"path": str(wrapped)}})
        save_results({"second": True}, {"name": "m", "depth": {"path": str(wrapped)}})
        assert ".eval/split_fog25/eval.json" not in _entries(pred)
        assert archive_writes == []

    assert archive_writes == [
        ("pred.zip", [".eval/split_fog100/eval.json", ".eval/split_fog25/eval.json"]),
        ("model.zip", ["model/.eval/eval.json"]),
    ]
    entries = _entries(pred)
    assert entries[".eval/split_fog25/eval.json"] == {"split": "fog25"}
    assert entries[".eval/split_fog100/eval.json"] == {"split": "fog100"}
    assert "frames/first.npy" in entries
    # A later result for the same entry replaces the earlier one.
    assert _entries(wrapped)["model/.eval/eval.json"] == {"second": True}
    assert list(spool_root.iterdir()) == []


def test_a_rerun_rebuilds_the_archive_once_and_keeps_its_mode(tmp_path, archive_writes):
    names = [f".eval/split_fog{level}/eval.json" for level in (25, 50, 100)]
    pred = _zip(tmp_path / "pred.zip", "frames/first.npy", *names)
    os.chmod(pred, 0o664)
    config = {"name": "m", "depth": {"path": str(pred)}}

    with batched_zip_writes():
        for level in (25, 50, 100):
            save_results({"new": level}, config, split=f"fog{level}")

    assert len(archive_writes) == 1
    with zipfile.ZipFile(pred) as zf:
        assert sorted(zf.namelist()) == sorted(["frames/first.npy", *names])
    assert _entries(pred)[names[0]] == {"new": 25}
    assert pred.stat().st_mode & 0o777 == 0o664
    # No temporary archive is left behind; the lock file stays.
    assert sorted(p.name for p in tmp_path.iterdir()) == [".pred.zip.lock", "pred.zip"]


def test_results_saved_before_an_error_are_written(tmp_path):
    pred = _zip(tmp_path / "pred.zip")
    with pytest.raises(RuntimeError, match="run failed"):
        with batched_zip_writes():
            save_results({"done": True}, {"name": "m", "depth": {"path": str(pred)}})
            raise RuntimeError("run failed")

    assert _entries(pred)[".eval/eval.json"] == {"done": True}


def test_results_an_archive_cannot_take_are_kept(tmp_path, spool_root):
    pred = _zip(tmp_path / "pred.zip")
    kept = _zip(tmp_path / "kept.zip")
    with pytest.raises(ZipWriteError, match="Could not write results into .*pred.zip") as raised:
        with batched_zip_writes():
            save_results({"lost": False}, {"name": "m", "depth": {"path": str(pred)}})
            save_results({"other": 1}, {"name": "m", "depth": {"path": str(kept)}})
            pred.write_bytes(b"not a zip")

    # The other archive is still written; the failed one's results stay.
    assert _entries(kept)[".eval/eval.json"] == {"other": 1}
    [spool] = spool_root.iterdir()
    assert f"kept in {spool}" in str(raised.value)
    assert json.loads((spool / "1.json").read_text()) == {"lost": False}


def test_without_a_batch_results_are_written_at_once(tmp_path):
    pred = _zip(tmp_path / "pred.zip")
    save_results({"now": True}, {"name": "m", "depth": {"path": str(pred)}})
    assert _entries(pred)[".eval/eval.json"] == {"now": True}


def _zipped(directory: Path) -> Path:
    archive = Path(shutil.make_archive(str(directory), "zip", directory))
    shutil.rmtree(directory)
    return archive


def test_a_sweep_writes_the_prediction_archive_once(
    monkeypatch, capsys, tmp_path, indexed_sky_prediction, archive_writes
):
    _patch_depth_metrics(monkeypatch)
    _write_splits(indexed_sky_prediction["gt"], {"fog25": ["first"], "fog100": ["second"]})
    pred = _zipped(Path(indexed_sky_prediction["pred"]))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "gt": {"depth": {"path": indexed_sky_prediction["gt"]}},
        "datasets": [{"name": "model", "depth": {"path": str(pred)}}],
        "splits": FOG,
    }))

    _main(monkeypatch, config)

    out = capsys.readouterr().out
    assert archive_writes == [(
        "pred.zip",
        [".eval/depth/split_fog100/eval.json", ".eval/depth/split_fog25/eval.json"],
    )]
    assert "Depth results queued for:" in out
    assert "Writing 2 result files into" in out
    entries = _entries(pred)
    for split in ("fog25", "fog100"):
        saved = entries[f".eval/depth/split_{split}/eval.json"]
        assert saved["meta"]["gt"]["split"] == split
        assert saved["dataset_info"]["num_pairs"] == 1


def test_a_rebuilt_archive_behind_a_symlink_stays_linked(tmp_path):
    (tmp_path / "store").mkdir()
    target = _zip(tmp_path / "store" / "pred.zip", ".eval/eval.json")
    link = tmp_path / "pred.zip"
    link.symlink_to(target)

    with batched_zip_writes():
        save_results({"new": True}, {"name": "m", "depth": {"path": str(link)}})

    assert link.is_symlink()
    assert _entries(target)[".eval/eval.json"] == {"new": True}


needs_flock = pytest.mark.skipif(zip_results.fcntl is None, reason="no fcntl")


@needs_flock
def test_concurrent_writers_take_turns(tmp_path):
    names = [f".eval/split_fog{i}/eval.json" for i in range(12)]
    # Half the entries exist, so some writers rebuild while others append.
    pred = _zip(tmp_path / "pred.zip", "frames/first.npy", *names[::2])
    start = threading.Barrier(len(names))
    errors = []

    def write(name):
        start.wait()
        try:
            save_json_to_zip(pred, name, {"name": name})
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=write, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    entries = _entries(pred)
    assert all(entries[name] == {"name": name} for name in names)
    with zipfile.ZipFile(pred) as zf:
        assert len(zf.namelist()) == len(set(zf.namelist())) == len(names) + 1
        assert zf.testzip() is None


@needs_flock
def test_a_held_lock_is_waited_for(tmp_path, capsys):
    pred = _zip(tmp_path / "pred.zip")
    fd = os.open(lock_path(pred), os.O_RDWR | os.O_CREAT)
    zip_results.fcntl.flock(fd, zip_results.fcntl.LOCK_EX)
    writer = threading.Thread(
        target=save_json_to_zip, args=(pred, "eval.json", {"waited": True})
    )
    try:
        writer.start()
        out = ""
        deadline = time.monotonic() + 10
        while "Waiting for another process" not in out and time.monotonic() < deadline:
            time.sleep(0.01)
            out += capsys.readouterr().out
        assert f"Waiting for another process writing into {pred}" in out
        assert "eval.json" not in _entries(pred)
    finally:
        os.close(fd)
        writer.join(timeout=10)
    with zipfile.ZipFile(pred) as zf:
        assert json.loads(zf.read("eval.json")) == {"waited": True}


@pytest.mark.parametrize("unavailable", ["no_flock", "no_fcntl"])
def test_without_a_lock_the_write_goes_ahead(monkeypatch, capsys, tmp_path, unavailable):
    if unavailable == "no_fcntl":
        monkeypatch.setattr(zip_results, "fcntl", None)
        reason = "no fcntl on this platform"
    else:
        if zip_results.fcntl is None:
            pytest.skip("no fcntl")

        def no_locks(fd, operation):
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(zip_results.fcntl, "flock", no_locks)
        reason = "No locks available"
    pred = _zip(tmp_path / "pred.zip")

    save_results({"unlocked": True}, {"name": "m", "depth": {"path": str(pred)}})

    assert _entries(pred)[".eval/eval.json"] == {"unlocked": True}
    err = capsys.readouterr().err
    assert f"Note: writing into {pred} without a lock" in err
    assert reason in err
