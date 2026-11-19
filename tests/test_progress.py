"""euler-view progress of an evaluation, through euler-connector: dataset j of m, one task per modality pass."""

import json
import sys
import zipfile

import pytest
from euler_connector import Connector, set_current
from euler_connector.progress import tqdm
from euler_connector.transports.memory import MemoryTransport

from euler_eval import cli
from tests.test_cli_sparse_outputs import _sparse_depth_results


@pytest.fixture
def records():
    """The records an euler-view launch would send, kept in memory."""
    memory = MemoryTransport()
    previous = set_current(Connector.from_environment({}, extra_transports=[memory]))
    yield memory
    set_current(previous)


def test_the_sparse_depth_pass_reports_its_dataset_and_task(tmp_path, monkeypatch, records):
    gt_archive = tmp_path / "gt.zip"
    pred_archive = tmp_path / "depth.zip"
    for archive in (gt_archive, pred_archive):
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("placeholder", b"data")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "gt": {
                    "sparse_depth": {"path": f"{gt_archive}#scope=sparse_depth"},
                    "intrinsics": {"path": f"{gt_archive}#scope=intrinsics"},
                    "camera_extrinsics": {"path": f"{gt_archive}#scope=camera_extrinsics"},
                },
                "datasets": [{"name": "model", "depth": {"path": str(pred_archive)}}],
            }
        )
    )

    def evaluate_sparse_depth_samples(**kwargs):
        # The evaluators' bars report (evaluate.py imports euler-connector's tqdm).
        for _ in tqdm(range(3), desc="Processing sparse depth pairs"):
            pass
        return _sparse_depth_results()

    monkeypatch.setattr(cli, "build_sparse_depth_eval_dataset", lambda **kwargs: [object()])
    monkeypatch.setattr(cli, "get_sparse_depth_metadata", lambda _dataset: {"pred_radial_depth": False})
    monkeypatch.setattr(cli, "evaluate_sparse_depth_samples", evaluate_sparse_depth_samples)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["euler-eval", str(config_path), "--device", "cpu", "--num-workers", "0", "--no-sanity-check"]
    )

    cli.main()

    sent = [envelope.data for envelope in records.by_channel()["progress"]]
    first, last = sent[0], sent[-1]
    assert (first["stage"], first["scope"], first["message"]) == ("eval", "task", "Processing sparse depth pairs")
    assert [(p["level"], p["index"], p["total"], p["name"]) for p in first["parents"]][-1] == ("dataset", 1, 1, "model")
    assert (first["task"]["index"], first["task"]["total"], first["task"]["name"]) == (1, None, "sparse_depth")
    assert (last["current"], last["total"], last["state"]) == (3, 3, "done")
