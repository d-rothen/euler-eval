"""Helpers for euler-loading modality path selectors in config files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl

from ds_crawler import list_metadata_scopes, validate_metadata_scope
from ds_crawler.config import CONFIG_FILENAME
from ds_crawler.zip_utils import (
    DATASET_HEAD_FILENAME,
    OUTPUT_FILENAME,
    read_metadata_json,
    validate_split_name,
)
from euler_loading import Modality

#: Config keys that name the GT segmentation; ``segmentation`` is canonical.
GT_SEGMENTATION_KEYS = ("segmentation", "semantic_segmentation")

#: GT config keys whose default metadata scope is not the key itself: the
#: degraded input is RGB data, read from its ``rgb`` scope like gt.rgb.
_GT_DEFAULT_SCOPES = {"input_rgb": "rgb"}


def gt_default_scope(key: str) -> str:
    """Return the metadata scope a ``gt`` config key reads without ``#scope=``."""
    return _GT_DEFAULT_SCOPES.get(key, key)


@dataclass(frozen=True)
class ModalityPath:
    path: str
    split: str | None = None
    metadata_scope: str | None = None


def _parse_inline_scope(path: str) -> tuple[str, str | None]:
    candidate_path, sep, fragment = path.rpartition("#")
    if not sep:
        return path, None

    params = parse_qsl(fragment, keep_blank_values=True)
    scope_values = [value for key, value in params if key == "scope"]
    if not scope_values:
        return path, None

    unsupported = [key for key, _ in params if key != "scope"]
    if unsupported:
        raise ValueError(
            "Unsupported modality path fragment parameter(s): "
            f"{', '.join(sorted(set(unsupported)))}. "
            "Only '#scope=<metadata_scope>' is supported."
        )
    if len(scope_values) > 1:
        raise ValueError("Modality path may contain at most one '#scope=' selector.")

    return candidate_path, validate_metadata_scope(scope_values[0])


def _parse_inline_split(path: str) -> tuple[str, str | None]:
    colon_pos = path.rfind(":")
    if colon_pos <= 1:
        return path, None

    candidate_split = path[colon_pos + 1 :]
    candidate_path = path[:colon_pos]

    try:
        split = validate_split_name(candidate_split)
    except ValueError:
        return path, None

    return candidate_path, split


def parse_modality_path(
    path: str | Path,
    *,
    split: str | None = None,
    metadata_scope: str | None = None,
) -> ModalityPath:
    """Parse euler-loading inline split/scope selectors from a config path."""
    raw_path = str(path)
    parsed_path, inline_scope = _parse_inline_scope(raw_path)
    parsed_path, inline_split = _parse_inline_split(parsed_path)

    if inline_split is not None and split is not None:
        raise ValueError(
            f"Modality path {raw_path!r} contains an inline split "
            f"({inline_split!r}) but an explicit split={split!r} was also "
            "provided. Use one or the other, not both."
        )

    normalized_scope = (
        validate_metadata_scope(metadata_scope)
        if metadata_scope is not None
        else None
    )
    if inline_scope is not None:
        if normalized_scope is not None and normalized_scope != inline_scope:
            raise ValueError(
                f"Modality path {raw_path!r} contains an inline metadata scope "
                f"({inline_scope!r}) but an explicit "
                f"metadata_scope={metadata_scope!r} was also provided. Use one "
                "metadata scope selector."
            )
        normalized_scope = inline_scope

    return ModalityPath(
        path=parsed_path,
        split=split if split is not None else inline_split,
        metadata_scope=normalized_scope,
    )


def normalize_modality_path(path: str | Path, *, split: str | None = None) -> Path:
    """Return the filesystem/archive root for a config modality path."""
    return Path(parse_modality_path(path, split=split).path)


def build_modality(
    *,
    path: str | Path,
    modality_key: str,
    split: str | None = None,
    metadata_scope: str | None = None,
    loader=None,
    used_as: str | None = None,
) -> Modality:
    """Create a Modality after parsing inline config selectors locally."""
    parsed = parse_modality_path(
        path,
        split=split,
        metadata_scope=metadata_scope,
    )
    return Modality(
        path=parsed.path,
        loader=loader,
        used_as=used_as,
        modality_type=modality_key,
        metadata_scope=parsed.metadata_scope or modality_key,
        split=parsed.split,
    )


def _read_metadata_safe(root: Path, filename: str, scope: str | None = None):
    try:
        return read_metadata_json(root, filename, metadata_scope=scope)
    except (FileNotFoundError, OSError, ValueError):
        return None


def has_scoped_metadata(root: Path, scope: str) -> bool:
    """Return whether *root* has ds-crawler metadata under ``.ds_crawler/<scope>/``.

    Mirrors euler-loading: a configured scope without its own metadata falls
    back to the root-level ``.ds_crawler/``, so only a scope that holds an
    index, crawler config or dataset head counts.
    """
    return any(
        _read_metadata_safe(root, filename, scope) is not None
        for filename in (OUTPUT_FILENAME, CONFIG_FILENAME, DATASET_HEAD_FILENAME)
    )


def resolve_dataset_location(
    path: str | Path, *, modality_key: str, split: str | None = None
) -> tuple[Path, str | None, str | None]:
    """Return ``(root, metadata scope, split)`` as euler-loading resolves them.

    The scope is the path's ``#scope=`` selector, else *modality_key*, and
    ``None`` when that scope holds no metadata of its own, because loading
    then falls back to the root-level ``.ds_crawler/``.
    """
    parsed = parse_modality_path(path, split=split)
    root = Path(parsed.path)
    scope = parsed.metadata_scope or modality_key
    return root, (scope if has_scoped_metadata(root, scope) else None), parsed.split


def dataset_identity(
    path: str | Path, *, modality_key: str, split: str | None = None
) -> dict[str, str | None]:
    """Identify a configured dataset independently of where it is stored.

    Returns the ``dataset.id`` of its ``dataset-head.json`` (its name when
    the head has no id, ``None`` without a head), the effective metadata
    scope and the split.
    """
    root, scope, split = resolve_dataset_location(
        path, modality_key=modality_key, split=split
    )
    head = _read_metadata_safe(root, DATASET_HEAD_FILENAME, scope)
    dataset = head.get("dataset") if isinstance(head, dict) else None
    dataset_id = None
    if isinstance(dataset, dict):
        dataset_id = next(
            (
                value
                for value in (dataset.get("id"), dataset.get("name"))
                if isinstance(value, str) and value
            ),
            None,
        )
    return {"dataset": dataset_id, "scope": scope, "split": split}


def _scope_modality_key(root: Path, scope: str) -> str | None:
    head = _read_metadata_safe(root, DATASET_HEAD_FILENAME, scope)
    modality = head.get("modality") if isinstance(head, dict) else None
    key = modality.get("key") if isinstance(modality, dict) else None
    return key if isinstance(key, str) and key else None


def resolve_gt_segmentation_path(path: str | Path, *, config_key: str) -> str:
    """Select the metadata scope of a GT segmentation path for its config key.

    Dataset builders use the config key as the metadata scope unless the path
    selects one with ``#scope=``. That misses archives whose segmentation
    scope carries the other segmentation name, e.g. a MUSES scope
    ``semantic_segmentation`` configured as ``gt.segmentation``. When the path
    has no ``#scope=``, no root-level metadata and no scope named
    *config_key*, this picks the one scope whose name or dataset-head
    modality key is a segmentation key (both names are candidates), or the
    only scope, and appends it as ``#scope=``. Any path that resolves without
    this help is returned unchanged.

    Raises:
        ValueError: When several scopes match, or none does among several.
    """
    raw = str(path)
    parsed = parse_modality_path(raw)
    if parsed.metadata_scope is not None:
        return raw
    root = Path(parsed.path)
    try:
        scopes = sorted(list_metadata_scopes(root))
    except (FileNotFoundError, OSError, ValueError):
        return raw
    if not scopes or config_key in scopes:
        return raw
    if any(
        _read_metadata_safe(root, filename) is not None
        for filename in (OUTPUT_FILENAME, CONFIG_FILENAME, DATASET_HEAD_FILENAME)
    ):
        return raw

    candidates = {config_key, *GT_SEGMENTATION_KEYS}
    matches = [
        scope
        for scope in scopes
        if scope in candidates or _scope_modality_key(root, scope) in candidates
    ]
    if len(matches) == 1:
        scope = matches[0]
    elif not matches and len(scopes) == 1:
        scope = scopes[0]
    else:
        listed = ", ".join(matches or scopes)
        raise ValueError(
            f"gt.{config_key}.path {raw!r} has several metadata scopes and none "
            f"is unambiguously the segmentation ({listed}). Append "
            "'#scope=<metadata_scope>' to the path."
        )
    return f"{raw}#scope={scope}"


def gt_input_location(key: str, entry: dict) -> tuple[Path, str | None, str | None]:
    """Return ``(root, metadata scope, split)`` of one configured ``gt`` input.

    Resolves the scope as loading does: GT segmentation through
    :func:`resolve_gt_segmentation_path` (its raw path when that is
    ambiguous, which only matters once the segmentation is read), every
    other key through its default scope.
    """
    path = entry["path"]
    if key in GT_SEGMENTATION_KEYS:
        try:
            path = resolve_gt_segmentation_path(path, config_key=key)
        except ValueError:
            pass
    return resolve_dataset_location(
        path, modality_key=gt_default_scope(key), split=entry.get("split")
    )
