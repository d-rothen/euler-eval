"""requirements.txt mirrors [project.dependencies] of pyproject.toml."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _project_dependencies():
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["dependencies"]


def test_requirements_mirror_the_project_dependencies():
    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == _project_dependencies()
