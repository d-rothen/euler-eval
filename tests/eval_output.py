"""Helpers for locating results written by a CLI run."""

import json
from pathlib import Path


def read_eval_json(pred_root) -> dict:
    """Read the one results file a run wrote under ``<pred_root>/.eval/``.

    Where inside ``.eval/`` it lands (scope and split folders) is covered by
    ``tests/test_save_results.py``; CLI tests only need its content.
    """
    found = sorted(Path(pred_root).glob(".eval/**/*.json"))
    assert len(found) == 1, f"expected one results file, found {found}"
    assert not (Path(pred_root) / "eval.json").exists()
    return json.loads(found[0].read_text())
