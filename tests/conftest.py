"""Shared fixtures for euler-eval tests."""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
from ds_crawler import build_dataset_head
from ds_crawler.artifacts import save_output_artifacts
from PIL import Image

MOCK_FILES = Path(__file__).parent / "mock_files"


@pytest.fixture
def depth_index_output():
    """Parsed depth output.json with radial_depth meta."""
    with open(MOCK_FILES / "depth_output.json") as f:
        return json.load(f)


@pytest.fixture
def rgb_index_output():
    """Parsed RGB output.json with rgb_range meta."""
    with open(MOCK_FILES / "rgb_output.json") as f:
        return json.load(f)


@pytest.fixture
def calibration_index_output():
    """Parsed calibration output.json."""
    with open(MOCK_FILES / "calibration_output.json") as f:
        return json.load(f)


@pytest.fixture
def segmentation_index_output():
    """Parsed segmentation output.json."""
    with open(MOCK_FILES / "segmentation_output.json") as f:
        return json.load(f)


@pytest.fixture
def sample_K():
    """A realistic 3x3 intrinsics matrix."""
    return np.array([
        [525.0, 0.0, 319.5],
        [0.0, 525.0, 239.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


@pytest.fixture
def mock_dataset():
    """Create a mock MultiModalDataset with configurable metadata.

    Returns a factory function that takes index_outputs dict and returns
    a MagicMock with get_modality_metadata and modality_paths wired up.
    """
    def _make(index_outputs: dict[str, dict], modality_names: Optional[list[str]] = None):
        ds = MagicMock()
        ds._index_outputs = index_outputs

        def get_modality_metadata(name):
            return index_outputs.get(name, {}).get("meta", {})

        ds.get_modality_metadata = get_modality_metadata

        names = modality_names or list(index_outputs.keys())
        ds.modality_paths.return_value = {n: f"/fake/{n}" for n in names}

        return ds

    return _make


@pytest.fixture
def indexed_sky_prediction(tmp_path):
    """Real generic loader metadata for depth and an RGB-encoded sky mask."""
    paths = {}
    for name in ("gt", "pred", "sky_mask"):
        root = tmp_path / name
        root.mkdir()
        paths[name] = str(root)
        is_mask = name == "sky_mask"
        modality = "sky_mask" if is_mask else "depth"
        meta = {"sky_mask": [12, 34, 56]} if is_mask else {"radial_depth": True}
        # Reverse mask order so pairing by position would score the wrong mask.
        ids = ["second", "first"] if is_mask else ["first", "second"]
        files = []
        for frame_id in ids:
            path = f"{frame_id}.png" if is_mask else f"{frame_id}.npy"
            files.append({"path": path, "id": frame_id})
            sky_index = (1, 1) if frame_id == "first" else (0, 0)
            if is_mask:
                rgb = np.zeros((2, 2, 3), dtype=np.uint8)
                rgb[sky_index] = meta["sky_mask"]
                Image.fromarray(rgb).save(root / path)
            else:
                depth = np.array(
                    [[10, 20], [30, 100]] if name == "gt" else [[10, 20], [80, 1]],
                    dtype=np.float32,
                )
                if frame_id == "second":
                    depth = np.flip(depth)
                np.save(root / path, depth)
        head = build_dataset_head(
            dataset={"name": "sky_test"},
            modality={"key": modality, "meta": meta},
            addons={"euler_loading": {
                "version": "1.0", "loader": "generic_dense_depth",
                "function": modality,
            }},
        )
        save_output_artifacts(root, {
            "contract": {"kind": "dataset_index", "version": "1.0"},
            "head_file": "dataset-head.json", "head": head,
            "generator": {"name": "tests"}, "indexing": {}, "execution": {},
            "index": {"files": files},
        }, metadata_scope=modality)
    return paths
