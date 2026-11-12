"""The ``protocol`` block of ``eval.json``: what to compare between evaluations.

``protocol.parameters`` is a flat map of dotted keys. Its keys are the
comparison keys: two evaluations of the same modality were set up the same way
exactly when their maps are equal, and the keys whose values differ say where.
A key is in one of three sections, its first segment:

- ``reference``: the ground-truth inputs the evaluation draws on, identified by
  their ds-crawler dataset id, metadata scope and split, not by file path.
- ``population``: which samples and pixels were scored.
- ``method``: how they were scored.

Only settings that change the value of a metric both files contain belong
here. Settings that only add or remove metrics (``--domain``, ``--sky-iou``,
histogram bins, which are described by the distribution registry) and run
details (device, batch size, workers, paths) do not. Every key of a modality is
always written, ``None`` or ``False`` when a setting is off, so a missing key
means "not recorded", never "off".
"""

from __future__ import annotations

from typing import Any, Mapping

from .config_paths import (
    GT_SEGMENTATION_KEYS,
    dataset_identity,
    gt_default_scope,
    resolve_gt_segmentation_path,
)

#: Version of the ``protocol`` block. Renaming or removing a key is a major
#: bump; adding one is a minor bump.
SCHEMA_VERSION = "1.3"

#: Bump whenever a release changes metric values for identical inputs and
#: settings, and say so in the CHANGELOG. Evaluations with different revisions
#: are not comparable.
METRICS_REVISION = 3

#: GT config keys a modality may read as reference inputs.
REFERENCE_KEYS = (
    "depth",
    "sparse_depth",
    "rgb",
    "input_rgb",
    "rays",
    "points_3d",
    "calibration",
    "intrinsics",
    "camera_extrinsics",
    "lidar_extrinsics",
)

_NO_INPUT = {"dataset": None, "scope": None, "split": None}


def _segmentation_identity(gt_config: Mapping[str, Any]) -> dict[str, str | None]:
    key = next((k for k in GT_SEGMENTATION_KEYS if k in gt_config), None)
    entry = gt_config.get(key) if key is not None else None
    if not isinstance(entry, Mapping) or "path" not in entry:
        return dict(_NO_INPUT)
    try:
        path = resolve_gt_segmentation_path(entry["path"], config_key=key)
    except ValueError:
        path = entry["path"]
    return dataset_identity(path, modality_key=key, split=entry.get("split"))


def reference_identities(gt_config: Mapping[str, Any]) -> dict[str, dict]:
    """Identify every configured GT input once per run.

    Keys are GT config keys, plus ``segmentation`` for whichever of its
    aliases is configured. An input that is not configured has ``None`` for
    its dataset, scope and split.
    """
    identities = {}
    for key in REFERENCE_KEYS:
        entry = gt_config.get(key)
        if isinstance(entry, Mapping) and "path" in entry:
            identities[key] = dataset_identity(
                entry["path"],
                modality_key=gt_default_scope(key),
                split=entry.get("split"),
            )
        else:
            identities[key] = dict(_NO_INPUT)
    identities["segmentation"] = _segmentation_identity(gt_config)
    return identities


def evaluated_resolution(spatial_info: Mapping[str, Any] | None) -> list[int] | None:
    """Return ``[height, width]`` of the grid the metrics were computed on."""
    dims = (spatial_info or {}).get("evaluated_dimensions")
    if not isinstance(dims, Mapping):
        return None
    return [int(dims["height"]), int(dims["width"])]


def _check_value(key: str, value: Any) -> Any:
    if isinstance(value, tuple):
        value = list(value)
    items = value if isinstance(value, list) else [value]
    scalars = (bool, int, float, str)
    if not all(item is None or isinstance(item, scalars) for item in items):
        raise TypeError(
            f"protocol parameter {key!r} must be a JSON scalar or a list of "
            f"scalars, got {value!r}"
        )
    return value


def build_protocol(
    identities: Mapping[str, Mapping[str, str | None]],
    *,
    gt: str,
    inputs: tuple[str, ...],
    population: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the ``protocol`` block for one modality's ``eval.json``.

    Args:
        identities: :func:`reference_identities` of the run.
        gt: GT config key the prediction is scored against.
        inputs: Further GT config keys (or ``segmentation``) this modality's
            evaluation reads; each is recorded whether configured or not, so
            the key set depends only on the modality.
        population: ``population.*`` values.
        method: ``method.*`` values; ``metrics_revision`` is added.
    """
    reference = {"gt": {"modality": gt, **identities[gt]}}
    for key in inputs:
        reference[key] = identities[key]

    parameters: dict[str, Any] = {}
    for role, fields in reference.items():
        for field, value in fields.items():
            parameters[f"reference.{role}.{field}"] = value
    for key, value in population.items():
        parameters[f"population.{key}"] = value
    for key, value in {"metrics_revision": METRICS_REVISION, **method}.items():
        parameters[f"method.{key}"] = value
    return {
        "schemaVersion": SCHEMA_VERSION,
        "parameters": {
            key: _check_value(key, value) for key, value in parameters.items()
        },
    }
