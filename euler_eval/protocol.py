"""The ``protocol`` block of ``eval.json``: the evaluation-protocol contract v2.

euler-eval writes the contract of euler-metric-naming
(``euler_metric_naming.contract``, ``docs/contract.md`` there) into the
existing top-level ``protocol`` object, so consumers judge two evaluations
with one generic algorithm:

- ``references``: every GT input the evaluation actually read, each with a
  content ``identity`` (dataset head without volatile fields, plus the
  manifest of its sample ids) and, for a crop of a larger frame, the ``view``
  it covers;
- ``parameters``: flat scalars (settings and resolved choices), described by
  the ``descriptors`` copied from the vocabularies;
- ``observations``: measured properties of what was scored (evaluated region,
  resampling, depth-cap clipping, sky exclusion);
- ``population``: the manifest of the evaluated item ids, the ids that appear
  in ``per_file_metrics``;
- ``stratifications``, ``axes``, ``reducers`` and ``metrics``: how the metric
  names decompose and how aggregates are recomputed from per-item values;
- ``items``: per-item sufficient statistics (pooling weights).

Everything new lives inside ``protocol``: version-1 consumers read only
``schemaVersion`` and ``parameters`` and drop the rest.
"""

from __future__ import annotations

import copy
import math
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from euler_metric_naming import AxisDeclaration, decompose
from euler_metric_naming.contract import (
    SCHEMA_VERSION,
    ProtocolBuilder,
    flatten_metrics,
    flatten_per_file,
    hash_value,
    load_vocabulary,
    manifest_hash,
    sqrt_stratification,
)

from .config_paths import (
    GT_SEGMENTATION_KEYS,
    dataset_identity,
    gt_default_scope,
    head_dataset_id,
    read_dataset_head,
    resolve_gt_segmentation_path,
)

__all__ = [
    "ID_SCHEME",
    "ITEM_ID_SEPARATOR",
    "POLICIES",
    "RGB_DEPTH_BIN_EDGES_M",
    "SCHEMA_VERSION",
    "Reference",
    "ReferenceInput",
    "build_protocol",
    "canonical_head",
    "depth_gt_range",
    "domain_region",
    "item_ids",
    "nfc_per_file_metrics",
    "pairing_accounting",
    "read_references",
    "reference_identities",
    "rgb_depth_bins",
]

#: Item id scheme of ``population`` (per-file tree path joined with ``~``).
ID_SCHEME = "euler.item_id/1"
ITEM_ID_SEPARATOR = "~"

#: Policy parameters that replace v1's global ``method.metrics_revision``.
#: Each names one rule whose revision changes metric values; bump the value
#: (and say so in the CHANGELOG) when the rule changes.
POLICIES = {
    # Masks are resampled by nearest pixel centre (cv2.INTER_NEAREST_EXACT).
    "spatial.mask_resample": "nearest_exact",
    # Declared invalid sentinels are not scored (GT and prediction).
    "depth.invalid.policy": "exclude_sentinel",
    # v3: an empty fixed depth bin has no value (it used to score 0.0).
    "rgb.depth_bins.policy": "v3",
    # FoV domain and rho_A threshold from the GT intrinsics.
    "rays.fov_policy": "v2",
    # The degraded input follows the GT's alignment (a crop is cropped).
    "rgb.input.alignment_policy": "follow_gt",
}

#: ``[near, far]`` thresholds of the fixed depth bins of RGB's
#: ``depth_binned_photometric`` metrics (near ``(0, 1]``, mid ``(1, 5]``,
#: far ``> 5`` m).
RGB_DEPTH_BIN_EDGES_M = (1.0, 5.0)

#: Descriptors of keys euler-eval records that the vocabularies do not
#: (yet) define. They travel in every file that uses them.
EXTRA_DESCRIPTORS = {
    "spatial.uniform": {
        "label": "uniform evaluated grid",
        "role": "definition",
        "description": (
            "Whether every item was scored with the same spatial method on "
            "the same grid (same GT and prediction sizes, same crop origin). "
            "When false, spatial.method and the domain observations describe "
            "the first item only."
        ),
    },
    "depth.alignment.fit_max_gt_percentile": {
        "label": "depth alignment GT percentile",
        "role": "definition",
        "appliesWhen": {"param": "depth.alignment.applied", "op": "eq", "value": True},
        "affects": {"axes": {"space": ["metric"]}},
        "description": (
            "Upper GT-depth percentile of the pixels the scale/shift fit "
            "uses (95 under GT sky masking); null when every valid pixel is "
            "used."
        ),
    },
}

#: Vocabulary whose metric terms a namespace root's base metrics are.
_PRIMARY_VOCABULARY = {
    "depth": "euler.depth",
    "sparsedepth": "euler.depth",
    "rgb": "euler.rgb",
    "rays": "euler.rays",
    "points3d": "euler.points3d",
}

#: Keys of a dataset head that vary between copies of the same data.
_VOLATILE_HEAD_KEYS = frozenset(
    {
        "path",
        "paths",
        "root",
        "uri",
        "url",
        "created",
        "modified",
        "updated",
        "timestamp",
        "hostname",
        "host",
    }
)
_VOLATILE_HEAD_SUFFIXES = ("_path", "_paths", "_root", "_at", "_time", "_timestamp")


# ---------------------------------------------------------------------------
# Item ids
# ---------------------------------------------------------------------------


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def item_ids(per_file_metrics: Optional[Mapping[str, Any]]) -> list[str]:
    """Evaluated item ids (``euler.item_id/1``) of a ``per_file_metrics`` tree.

    The ``children`` keys from the root down, then the file entry's ``id``,
    joined with ``~`` and NFC-normalized; the order of the tree.
    """
    return [_nfc(item) for item in flatten_per_file(per_file_metrics or {})]


def nfc_per_file_metrics(tree: Any) -> Any:
    """A ``per_file_metrics`` tree with its hierarchy keys and file ids in NFC."""
    if not isinstance(tree, dict):
        return tree
    out = dict(tree)
    children = tree.get("children")
    if isinstance(children, dict):
        out["children"] = {
            (_nfc(key) if isinstance(key, str) else key): nfc_per_file_metrics(child)
            for key, child in children.items()
        }
    files = tree.get("files")
    if isinstance(files, list):
        out["files"] = [
            {**entry, "id": _nfc(entry["id"])}
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
            else entry
            for entry in files
        ]
    return out


def _sample_id_to_item(sample_id: str) -> str:
    """euler-loading's ``hierarchy/…/file_id`` lookup key as an item id."""
    return _nfc(ITEM_ID_SEPARATOR.join(part for part in sample_id.split("/")))


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceInput:
    """One GT input as the evaluated dataset was built from it.

    Attributes:
        role: Contract reference role (``gt``, ``calibration``,
            ``segmentation``, ``intrinsics``, ``camera_extrinsics``,
            ``lidar_extrinsics``, ``input``, ``depth``).
        key: The ``gt`` config key it came from (the reference's
            ``modality``).
        path: The path the dataset was built with.
        split: Its configured split.
        modality: Its name in the evaluated dataset (``gt``, ``calibration``,
            ``gt_depth``, …).
        scope: The modality key the dataset used as its default metadata
            scope (``depth``, ``intrinsics``, …).
    """

    role: str
    key: str
    path: Optional[str]
    split: Optional[str]
    modality: str
    scope: str


@dataclass
class Reference:
    """A resolved reference (``protocol.references.<role>``)."""

    role: str
    modality: str
    dataset: dict
    identity: str
    manifest: Optional[dict] = None
    view: Optional[dict] = None
    #: The head's crop box (top, left, height, width, output size) for a view.
    crop: Optional[dict] = None

    def builder_kwargs(self) -> dict:
        kwargs = {
            "identity": self.identity,
            "modality": self.modality,
            "dataset": self.dataset,
        }
        if self.manifest is not None:
            kwargs["manifest"] = self.manifest
        if self.view is not None:
            kwargs["view"] = self.view
        return kwargs


def canonical_head(head: Any) -> Any:
    """A dataset head without fields that vary between copies of the data.

    Paths and timestamps (keys such as ``path``, ``*_path``, ``created_at``,
    ``timestamp``) are dropped at every depth, so the same data stored
    elsewhere or re-indexed later keeps its identity.
    """
    if isinstance(head, dict):
        return {
            key: canonical_head(value)
            for key, value in head.items()
            if not (
                isinstance(key, str)
                and (
                    key in _VOLATILE_HEAD_KEYS
                    or key.endswith(_VOLATILE_HEAD_SUFFIXES)
                )
            )
        }
    if isinstance(head, list):
        return [canonical_head(value) for value in head]
    if isinstance(head, float) and not math.isfinite(head):
        return None
    return copy.deepcopy(head)


def _head_variant(head: Optional[dict]) -> Optional[str]:
    dataset = head.get("dataset") if isinstance(head, dict) else None
    if not isinstance(dataset, dict):
        return None
    attributes = dataset.get("attributes")
    for value in (
        dataset.get("variant"),
        attributes.get("variant") if isinstance(attributes, dict) else None,
    ):
        if isinstance(value, str) and value:
            return value
    return None


def _crop_box(head: Optional[dict]) -> Optional[dict]:
    """The single crop box an euler-preprocess crop recorded in *head*."""
    modality = head.get("modality") if isinstance(head, dict) else None
    meta = modality.get("meta") if isinstance(modality, dict) else None
    crop = meta.get("crop") if isinstance(meta, dict) else None
    boxes = crop.get("boxes") if isinstance(crop, dict) else None
    if not isinstance(boxes, list) or len(boxes) != 1 or not isinstance(boxes[0], dict):
        return None
    box = boxes[0]
    fields = ("top", "left", "height", "width", "source_height", "source_width")
    if not all(isinstance(box.get(f), (int, float)) for f in fields):
        return None
    if box["height"] <= 0 or box["width"] <= 0:
        return None
    out = {f: box[f] for f in fields}
    out["output_height"] = box.get("output_height") or box["height"]
    out["output_width"] = box.get("output_width") or box["width"]
    return out


def _view(head: Optional[dict], dataset_id: Optional[str]) -> tuple:
    box = _crop_box(head)
    if box is None or dataset_id is None:
        return None, None
    modality = head.get("modality") if isinstance(head, dict) else {}
    meta = modality.get("meta") if isinstance(modality.get("meta"), dict) else {}
    label = meta.get("camera") or modality.get("key") or "frame"
    source = (
        f"{dataset_id}/{label}@{int(box['source_height'])}x{int(box['source_width'])}"
    )
    region = [_number(box[f]) for f in ("top", "left", "height", "width")]
    return {"source": source, "region": region}, box


def _number(value: float) -> float | int:
    value = float(value)
    return int(value) if value.is_integer() else value


def _lookups(dataset: Any) -> tuple[Mapping, Mapping]:
    """euler-loading's per-modality id lookups (regular, hierarchical)."""
    source = getattr(dataset, "_with_input", None) or dataset
    regular = getattr(source, "_lookups", None)
    hierarchical = getattr(source, "_hierarchical_lookups", None)
    return (
        regular if isinstance(regular, Mapping) else {},
        hierarchical if isinstance(hierarchical, Mapping) else {},
    )


def _input_ids(dataset: Any, modality: str) -> Optional[list[str]]:
    """Item ids of the samples one dataset input provides, or ``None``."""
    regular, hierarchical = _lookups(dataset)
    if modality in regular:
        return [_sample_id_to_item(key) for key in regular[modality]]
    if modality in hierarchical:
        ids = []
        for level, entries in hierarchical[modality].items():
            for entry in entries:
                if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
                    ids.append(
                        _nfc(ITEM_ID_SEPARATOR.join((*tuple(level), entry["id"])))
                    )
        return ids
    return None


def _dataset_modalities(dataset: Any) -> Optional[set]:
    """Names of the inputs a dataset reads, ``None`` when it does not say."""
    names: set = set()
    known = False
    for method in ("modality_paths", "hierarchical_modality_paths"):
        lookup = getattr(dataset, method, None)
        if not callable(lookup):
            continue
        try:
            paths = lookup()
        except Exception:
            continue
        if isinstance(paths, Mapping):
            names.update(paths)
            known = True
    return names if known else None


def read_references(
    dataset: Any, inputs: Iterable[ReferenceInput]
) -> dict[str, Reference]:
    """Resolve the references of the GT inputs *dataset* actually read.

    An input that is not part of the dataset (not configured, or left out
    such as GT segmentation that only ``--sky-iou`` would have read) is
    skipped; ``gt`` is always recorded when it has a path.
    """
    available = _dataset_modalities(dataset)
    references: dict[str, Reference] = {}
    for item in inputs:
        if item.path is None:
            continue
        if item.role != "gt" and available is not None and item.modality not in available:
            continue
        references[item.role] = _reference(dataset, item)
    return references


def _reference(dataset: Any, item: ReferenceInput) -> Reference:
    scope, split, head = read_dataset_head(
        item.path, modality_key=item.scope, split=item.split
    )
    dataset_id = head_dataset_id(head)
    identity_dataset = {
        "id": dataset_id,
        "variant": _head_variant(head),
        "split": split,
        "scope": scope,
    }
    ids = _input_ids(dataset, item.modality)
    manifest = (
        {"hash": manifest_hash(ids), "count": len(set(ids))} if ids is not None else None
    )
    identity = hash_value(
        {"dataset": identity_dataset, "manifest": manifest, "head": canonical_head(head)}
    )
    view, box = _view(head, dataset_id)
    return Reference(
        role=item.role,
        modality=item.key,
        dataset=identity_dataset,
        identity=identity,
        manifest=manifest,
        view=view,
        crop=box,
    )


def domain_region(
    gt: Optional[Reference],
    evaluated: Optional[Sequence[int]],
    origin: Optional[Sequence[int]] = None,
) -> Optional[dict]:
    """The ``domain.region`` observation of a GT-grid evaluation.

    *evaluated* is the ``[height, width]`` of the scored grid, *origin* its
    ``(top, left)`` in the GT as loaded (a crop of the GT; ``(0, 0)``
    otherwise). With a GT view (a crop of a larger frame) the box is in that
    source frame's pixels and the frame is the view's ``source``; otherwise
    it is in the GT's own pixels and the frame is the GT's identity.
    """
    if gt is None or evaluated is None:
        return None
    height, width = (float(v) for v in evaluated[:2])
    top, left = (float(v) for v in (origin or (0, 0))[:2])
    if gt.view is not None and gt.crop is not None:
        box = gt.crop
        scale_y = box["height"] / box["output_height"]
        scale_x = box["width"] / box["output_width"]
        values = [
            box["top"] + top * scale_y,
            box["left"] + left * scale_x,
            height * scale_y,
            width * scale_x,
        ]
        return {"frame": gt.view["source"], "box": [_number(v) for v in values]}
    return {"frame": gt.identity, "box": [_number(v) for v in (top, left, height, width)]}


# ---------------------------------------------------------------------------
# Population accounting
# ---------------------------------------------------------------------------

#: Dataset inputs that are paired with the evaluated frames by id, so a frame
#: one of them lacks is not evaluated: dataset name -> accounting key.
_PAIRED_INPUTS = {
    "segmentation": "segmentation",
    "pred_sky_mask": "sky_mask",
    "input": "input",
    "gt_depth": "gt_depth",
}


def pairing_accounting(dataset: Any, *, reference: str = "gt") -> dict:
    """``population.accounting``: how the evaluated pairs came about.

    ``reference`` / ``prediction`` count the samples of the GT (*reference*)
    and the prediction, ``paired`` the pairs evaluated, and ``dropped`` how
    many pairs each id-paired optional input removed (inputs not read are
    absent). Frames kept without their own GT segmentation
    (``--missing-segmentation unmasked``) are not dropped.
    """
    source = getattr(dataset, "_complete", None) or dataset
    lookups = getattr(source, "_lookups", None)
    accounting: dict = {}
    if isinstance(lookups, Mapping):
        if reference in lookups:
            accounting["reference"] = len(lookups[reference])
        if "pred" in lookups:
            accounting["prediction"] = len(lookups["pred"])
    try:
        accounting["paired"] = len(dataset)
    except TypeError:
        pass
    if isinstance(lookups, Mapping):
        optional = tuple(name for name in _PAIRED_INPUTS if name in lookups)
        dropped = {}
        for name in optional:
            required = [
                set(ids)
                for other, ids in lookups.items()
                if other != name and other not in optional
            ]
            count = (
                len(set.intersection(*required) - set(lookups[name])) if required else 0
            )
            dropped[_PAIRED_INPUTS[name]] = count
        if dropped:
            accounting["dropped"] = dropped
    return accounting


# ---------------------------------------------------------------------------
# Stratifications
# ---------------------------------------------------------------------------


def depth_gt_range(
    boundaries: Optional[Mapping[str, Any]],
    benchmark_range: Optional[Sequence[float]],
    *,
    after_cap: bool = True,
) -> Optional[dict]:
    """The ``depth.gt_range`` stratification of the benchmark bins.

    Edges are the ones the evaluation used (*boundaries*, as
    ``get_benchmark_depth_bins`` computed them), else recomputed from
    *benchmark_range* with the square-root rule. *after_cap* records that the
    bins select GT after the depth cap (dense and sparse depth; RGB's GT depth
    is not capped).
    """
    after = ("depth.cap.max_m",) if after_cap else None
    if isinstance(boundaries, Mapping) and all(
        key in boundaries for key in ("range", "near", "mid")
    ):
        lo, hi = (float(v) for v in boundaries["range"])
        edges = [lo, float(boundaries["near"][1]), float(boundaries["mid"][1]), hi]
        return sqrt_stratification(lo, hi, edges=edges, after=after)
    if benchmark_range is not None:
        lo, hi = (float(v) for v in benchmark_range)
        return sqrt_stratification(lo, hi, after=after)
    return None


def rgb_depth_bins() -> dict:
    """The fixed GT-depth bins of RGB's ``depth_binned_photometric`` metrics.

    The far bin is open-ended (``> 5`` m), which stratification ``edges``
    cannot express, so the bins are categorical; their thresholds are the
    ``rgb.depth_bins.edges_m`` parameter. The bin is part of these metrics'
    base names (``mae.near``), not an axis.
    """
    near, far = RGB_DEPTH_BIN_EDGES_M
    return {
        "label": "GT depth (fixed photometric bins)",
        "variable": "gt_depth_m",
        "unit": "m",
        "categories": ["near", "mid", "far"],
        "total": "all",
        "closed": "right",
        "description": (
            f"near (0, {near:g}] m, mid ({near:g}, {far:g}] m, far > {far:g} m of "
            "valid GT depth on the prediction grid; an empty bin has no value."
        ),
    }


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def _declarations(metric_set: Mapping[str, Any]) -> dict:
    declarations = {}
    for name, axis in (metric_set.get("axes") or {}).items():
        declarations[name] = AxisDeclaration.from_dict(dict(axis))
    return declarations


def _fallback_definition(root: str, base: str, description: Any, per_item: bool) -> dict:
    """A metric definition for a base no vocabulary defines (yet).

    Built from the metricSet description; it carries no ``vocabulary`` tag,
    which is how such metrics are told apart. ``pool`` is ``none``, so
    pooled aggregates of it are not recomputed on a subset of files.
    """
    definition: dict = {"id": f"{root}.{base}", "kind": "score", "rev": 1}
    if isinstance(description, Mapping):
        if isinstance(description.get("displayName"), str):
            definition["displayName"] = description["displayName"]
        if isinstance(description.get("description"), str):
            definition["description"] = description["description"]
        higher = description.get("isHigherBetter")
        if isinstance(higher, bool):
            definition["direction"] = "higher" if higher else "lower"
        low, high = description.get("min"), description.get("max")
        if low is not None or high is not None:
            definition["bounds"] = [low, high]
        if isinstance(description.get("unit"), str):
            definition["unit"] = description["unit"]
    definition["perItem"] = per_item
    definition["pool"] = {"form": "none"}
    return definition


def _add_metrics(
    builder: ProtocolBuilder,
    *,
    namespace: str,
    metric_set: Mapping[str, Any],
    aggregate: Iterable[str],
    per_item: Iterable[str],
    reduction_axes: Sequence[str],
) -> None:
    declarations = _declarations(metric_set)
    root = namespace.split(".", 1)[0]
    primary = load_vocabulary(_PRIMARY_VOCABULARY.get(root, "euler.core"))
    core = load_vocabulary("euler.core")
    descriptions = metric_set.get("metricDescriptions") or {}
    per_item = set(per_item)
    bases: dict[str, bool] = {}
    reducers: set = set()
    for name in sorted(set(aggregate) | per_item):
        try:
            parsed = decompose(name, namespace, declarations)
        except (ValueError, TypeError):
            print(
                f"Warning: protocol: metric {name!r} does not decompose under "
                f"{namespace!r}; it gets no definition.",
                file=sys.stderr,
            )
            continue
        bases[parsed.metric] = bases.get(parsed.metric, False) or name in per_item
        for axis in reduction_axes:
            value = parsed.axes.get(axis)
            if value is not None:
                reducers.add(value)
    for base in sorted(bases):
        term = primary.metrics.get(base)
        if term is not None:
            builder.metric(base, {**copy.deepcopy(term), "vocabulary": primary.tag})
            continue
        builder.metric(
            base, _fallback_definition(root, base, descriptions.get(base), bases[base])
        )
    for value in sorted(reducers):
        term = core.reducers.get(value)
        if term is not None:
            builder.reducer(value, {**copy.deepcopy(term), "vocabulary": core.tag})
        else:
            builder.reducer(value, {"recompute": "none", "label": value})


def build_protocol(
    *,
    task: str,
    vocabularies: Sequence[str],
    eval_json: Mapping[str, Any],
    references: Mapping[str, Reference],
    parameters: Mapping[str, Any],
    observations: Optional[Mapping[str, Any]] = None,
    axis_roles: Optional[Mapping[str, Any]] = None,
    stratifications: Optional[Mapping[str, Optional[dict]]] = None,
    accounting: Optional[Mapping[str, Any]] = None,
    item_statistics: Optional[Mapping[str, Mapping[str, Any]]] = None,
    unit: str = "image",
) -> dict:
    """Return the v2 ``protocol`` block of one modality's ``eval.json``.

    Args:
        task: ``protocol.task`` (``depth.dense``, ``rgb.restoration``, …).
        vocabularies: Vocabulary ids whose terms the block may use by key.
        eval_json: The results file being written: its ``metricSet``, metric
            tree and ``per_file_metrics`` (ids and metric names are read
            from it).
        references: :func:`read_references` of the inputs read.
        parameters: Flat scalar parameters; keys are vocabulary terms or
            :data:`EXTRA_DESCRIPTORS`.
        observations: Observation values by key; ``None`` values are left out.
        axis_roles: ``metricSet`` axis name -> role, or ``(role,
            stratification id)`` for a stratum axis. Undeclared axes are
            ``family``.
        stratifications: Stratifications by id; ``None`` values are left out.
        accounting: ``population.accounting``.
        item_statistics: ``{item id: {per-item series name: value}}``; the
            series are aligned with the population ids (missing -> null).
        unit: ``population.unit``.

    Raises:
        euler_metric_naming.contract.ProtocolValidationError: When the block
            violates the contract.
    """
    metric_set = eval_json.get("metricSet") or {}
    namespace = metric_set.get("metricNamespace")
    builder = ProtocolBuilder(task, vocabularies)

    for role, reference in references.items():
        builder.reference(role, **reference.builder_kwargs())
    for key, value in parameters.items():
        builder.param(key, value, descriptor=EXTRA_DESCRIPTORS.get(key))
    for key, value in (observations or {}).items():
        if value is not None:
            builder.observe(key, value, descriptor=EXTRA_DESCRIPTORS.get(key))

    per_file = eval_json.get("per_file_metrics")
    ids = item_ids(per_file) if isinstance(per_file, Mapping) else None
    if ids is not None:
        builder.population(ids, unit=unit, accounting=accounting, id_scheme=ID_SCHEME)

    for sid, stratification in (stratifications or {}).items():
        if stratification is not None:
            builder.stratification(sid, stratification)

    reduction_axes = []
    for name in (metric_set.get("axes") or {}):
        role = (axis_roles or {}).get(name, "family")
        stratification = None
        if isinstance(role, tuple):
            role, stratification = role
        if role == "stratum" and (stratifications or {}).get(stratification) is None:
            role, stratification = "family", None
        builder.axis(name, role, stratification=stratification)
        if role == "reduction":
            reduction_axes.append(name)

    series: dict[str, list] = {}
    if ids is not None and item_statistics:
        stats = {_nfc(item): values for item, values in item_statistics.items()}
        names = sorted({name for values in stats.values() for name in values})
        for name in names:
            series[name] = [
                _series_value(stats.get(item, {}).get(name)) for item in ids
            ]

    if isinstance(namespace, str):
        per_item_names = {
            name
            for values in flatten_per_file(per_file or {}).values()
            for name in values
        }
        _add_metrics(
            builder,
            namespace=namespace,
            metric_set=metric_set,
            aggregate=flatten_metrics(eval_json),
            per_item=per_item_names | set(series),
            reduction_axes=reduction_axes,
        )
    for name, values in series.items():
        builder.series(name, values)
    return builder.build()


def _series_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


# ---------------------------------------------------------------------------
# Version-1 identities (kept for callers of the 2.x API)
# ---------------------------------------------------------------------------

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
    """Identify every configured GT input by dataset id, scope and split.

    The version-1 identities (``reference.<key>.{dataset,scope,split}``);
    version 2 identifies a reference by content (:func:`read_references`).
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
