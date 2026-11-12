"""The config's ``splits`` section: one evaluation per ds-crawler split.

A dataset with many named subsets, such as one split per fog level, is
evaluated split by split from a single config::

    "splits": {"gt_modality": "depth", "match": "fog\\d+"}

lists the splits of ``gt.depth`` whose whole name matches, in natural order
(``fog20`` before ``fog100``), and each run selects one of them on every GT
input that defines it. Predictions are used as configured: pairing by file ID
restricts them to the split's frames. Results land in each prediction's
``.eval/[<scope>/]split_<split>/``, where they would land had the prediction
itself been loaded with that split.

No path may select a split itself, except the deprecated ``all`` (see
:func:`ignore_all_split`).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ds_crawler import list_dataset_splits

from .config_paths import GT_SEGMENTATION_KEYS, gt_input_location, parse_modality_path

#: GT inputs paired with the evaluated frames by file ID.
PAIRED_GT_KEYS = (
    "rgb",
    "input_rgb",
    "depth",
    "sparse_depth",
    "rays",
    "points_3d",
    *GT_SEGMENTATION_KEYS,
)
#: GT inputs looked up by hierarchy position. A full index serves any subset
#: of frames, so they read a split only where they define it.
HIERARCHICAL_GT_KEYS = (
    "calibration",
    "intrinsics",
    "camera_extrinsics",
    "lidar_extrinsics",
)
GT_INPUT_KEYS = (*PAIRED_GT_KEYS, *HIERARCHICAL_GT_KEYS)

#: Placeholder a dataset's ``output_file`` needs in a sweep.
SPLIT_PLACEHOLDER = "{split}"

#: The one split a path may still select in a sweep, which ignores it: path
#: templates (euler-view's) write it for the whole dataset. Deprecated.
ALL_SPLIT = "all"

_EXAMPLE = '{"gt_modality": "depth", "match": "fog\\\\d+"}'


@dataclass(frozen=True)
class SplitSweep:
    """The resolved ``splits`` section.

    Attributes:
        gt_modality: The ``gt`` key whose splits are swept.
        match: The pattern split names must match in full (``None``: all).
        splits: The matching split names, in natural order.
        available: Every split each configured ``gt`` input defines.
    """

    gt_modality: str
    match: str | None
    splits: tuple[str, ...]
    available: Mapping[str, frozenset[str]]

    def defines(self, key: str, split: str) -> bool:
        return split in self.available.get(key, ())


def natural_order(names: Iterable[str]) -> list[str]:
    """Sort split names with their digit runs compared as numbers."""

    def key(name: str):
        parts = re.split(r"(\d+)", name)
        return [int(part) if i % 2 else part for i, part in enumerate(parts)]

    return sorted(names, key=key)


def _has_path(value: Any) -> bool:
    return isinstance(value, Mapping) and "path" in value


def _configured_inputs(gt: Mapping[str, Any]):
    for key in GT_INPUT_KEYS:
        entry = gt.get(key)
        if _has_path(entry):
            yield key, entry


def ignore_all_split(config: dict) -> dict:
    """Return *config* without its paths' selections of the ``all`` split.

    A sweep selects each run's split itself, so a path may not select one,
    but path templates write ``:all`` (or ``"split": "all"``) for the whole
    dataset. Until that is removed, such a selection is dropped with a
    :class:`FutureWarning`; the path's ``#scope=`` is kept. *config* is
    returned unchanged when no path selects ``all``.
    """
    ignored: list[str] = []

    def without_all(label: str, entry: Any) -> Any:
        if not _has_path(entry):
            return entry
        inline = parse_modality_path(entry["path"])
        if ALL_SPLIT not in (inline.split, entry.get("split")):
            return entry
        ignored.append(label)
        entry = {
            key: value
            for key, value in entry.items()
            if not (key == "split" and value == ALL_SPLIT)
        }
        if inline.split == ALL_SPLIT:
            entry["path"] = (
                inline.path
                if inline.metadata_scope is None
                else f"{inline.path}#scope={inline.metadata_scope}"
            )
        return entry

    gt = {
        key: without_all(f"gt.{key}", value) if key in GT_INPUT_KEYS else value
        for key, value in config["gt"].items()
    }
    datasets = [
        {
            key: without_all(f"datasets[{index}].{key}", value)
            for key, value in dataset.items()
        }
        for index, dataset in enumerate(config.get("datasets", []))
    ]
    if not ignored:
        return config
    warnings.warn(
        f"{', '.join(ignored)} select{'s' if len(ignored) == 1 else ''} split "
        f"{ALL_SPLIT!r}, which 'splits' ignores: each run selects its own "
        f"split. Selecting {ALL_SPLIT!r} alongside 'splits' is deprecated and "
        "will be an error in a future release; remove it from "
        f"{'that path' if len(ignored) == 1 else 'those paths'}.",
        FutureWarning,
        stacklevel=2,
    )
    return {**config, "gt": gt, "datasets": datasets}


def validate_splits_config(value: Any, config: Mapping[str, Any]) -> None:
    """Validate the ``splits`` section against the rest of *config*.

    Checks what needs no dataset access; :func:`resolve_split_sweep` lists
    the splits. Run :func:`ignore_all_split` first: a path selecting ``all``
    is rejected here like any other split.

    Raises:
        ValueError: Naming the offending field.
    """
    if not isinstance(value, dict):
        raise ValueError(f"'splits' must be an object such as {_EXAMPLE}")
    unknown = sorted(set(value) - {"gt_modality", "match"})
    if unknown:
        raise ValueError(
            f"splits has unknown keys {unknown}; only 'gt_modality' and "
            "'match' are supported"
        )
    gt_modality = value.get("gt_modality")
    if not isinstance(gt_modality, str):
        raise ValueError(
            "splits.gt_modality must name the gt input whose splits are "
            f'evaluated, e.g. "depth" ({_EXAMPLE})'
        )
    if gt_modality not in GT_INPUT_KEYS:
        raise ValueError(
            f"splits.gt_modality: unknown gt input {gt_modality!r}; expected "
            f"one of {', '.join(GT_INPUT_KEYS)}"
        )
    gt = config["gt"]
    if gt_modality not in dict(_configured_inputs(gt)):
        raise ValueError(
            f"splits.gt_modality is {gt_modality!r}, but gt.{gt_modality} is "
            "not configured"
        )
    if "match" in value:
        match = value["match"]
        if not isinstance(match, str) or not match:
            raise ValueError(
                "splits.match must be a non-empty regular expression, "
                'e.g. "fog\\\\d+"'
            )
        try:
            re.compile(match)
        except re.error as e:
            raise ValueError(
                f"splits.match {match!r} is not a valid regular expression: {e}"
            ) from e

    selected = [(f"gt.{key}", entry) for key, entry in _configured_inputs(gt)]
    for index, entry in enumerate(config.get("datasets", [])):
        selected += [
            (f"datasets[{index}].{key}", value)
            for key, value in entry.items()
            if _has_path(value)
        ]
    for label, entry in selected:
        own = parse_modality_path(entry["path"], split=entry.get("split")).split
        if own is not None:
            raise ValueError(
                f"{label} selects split {own!r}, but 'splits' selects one "
                f"split per run; remove it from {label}"
            )
    for index, entry in enumerate(config.get("datasets", [])):
        output_file = entry.get("output_file")
        if output_file is not None and SPLIT_PLACEHOLDER not in str(output_file):
            raise ValueError(
                f"datasets[{index}].output_file names one file, but 'splits' "
                f"writes results per split; put {SPLIT_PLACEHOLDER} in it, e.g. "
                f'"model_{SPLIT_PLACEHOLDER}.json", or remove it to use the '
                ".eval/ layout"
            )


def list_gt_splits(key: str, entry: Mapping[str, Any]) -> list[str]:
    """Return the splits one ``gt`` input defines, where loading reads them.

    ds-crawler stores splits per metadata scope, so this is the scope the
    input is loaded from, or the root-level metadata it falls back to.

    Raises:
        ValueError: When the input's metadata cannot be read.
    """
    root, scope, _ = gt_input_location(key, dict(entry, split=None))
    try:
        return list_dataset_splits(root, metadata_scope=scope)
    except (OSError, ValueError) as e:
        raise ValueError(f"gt.{key}: cannot list its splits: {e}") from e


def resolve_split_sweep(value: Mapping[str, Any], gt: Mapping[str, Any]) -> SplitSweep:
    """List the configured GT inputs' splits and select the swept ones.

    Raises:
        ValueError: When ``gt_modality`` defines no split ``match`` matches.
    """
    available = {
        key: frozenset(list_gt_splits(key, entry))
        for key, entry in _configured_inputs(gt)
    }
    gt_modality = value["gt_modality"]
    match = value.get("match")
    defined = natural_order(available[gt_modality])
    if not defined:
        raise ValueError(
            f"splits: gt.{gt_modality} defines no ds-crawler splits "
            "(no .ds_crawler/[<scope>/]split_<name>.json)"
        )
    pattern = re.compile(match) if match is not None else None
    splits = tuple(
        name for name in defined if pattern is None or pattern.fullmatch(name)
    )
    if not splits:
        raise ValueError(
            f"splits.match {match!r} matches none of gt.{gt_modality}'s splits "
            f"({', '.join(defined)}); it must match a whole split name"
        )
    return SplitSweep(
        gt_modality=gt_modality, match=match, splits=splits, available=available
    )


def split_gt(gt: Mapping[str, Any], sweep: SplitSweep, split: str) -> dict:
    """Return *gt* with *split* selected on every input that defines it."""
    return {
        key: (
            {**value, "split": split}
            if key in GT_INPUT_KEYS and sweep.defines(key, split)
            else value
        )
        for key, value in gt.items()
    }


def split_output_file(output_file: str | None, split: str | None) -> str | None:
    """Fill a dataset's ``output_file`` placeholder with the run's split."""
    if output_file is None or split is None:
        return output_file
    return str(output_file).replace(SPLIT_PLACEHOLDER, split)
