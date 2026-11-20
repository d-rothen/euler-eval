"""Validation sets: in-training depth validation as contract-v2 evaluations.

A validation set is an evaluation of the live model on a fixed population
(``docs/training.md`` of euler-metric-naming). :class:`DepthValidationSet`
declares it once (:meth:`DepthValidationSet.declaration`: the metricSet and
protocol of the evaluation, as the CLI would write them for the same GT and
settings) and scores each validation pass in memory
(:meth:`DepthValidationSet.new_pass`), with the CLI's metric names, per-item
values and per-item pixel counts. A pass compared with the CLI's ``eval.json``
of the same predictions under the same settings reads ``same protocol``; a
pass on a subset of its items reads ``compared on N shared files``.

What the in-loop path does differently from the CLI is recorded, never
hidden: a GT validity range (``depth.gt.valid_range_m``) and an extra
validity mask (``depth.extra_mask``) make a pass ``not comparable`` with an
evaluation that did not apply them. Settings the in-loop path cannot apply
(GT or predicted sky masking) are refused.
"""

from __future__ import annotations

import copy
import importlib.metadata
import math
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Union

import numpy as np

from .calibration import (
    get_sample_intrinsics,
    get_sample_pointcloud_to_camera_extrinsics,
)
from .config_paths import parse_modality_path
from .data import (
    SourceGrid,
    _crop_origin,
    align_to_ground_truth,
    compose_sensor_to_camera_extrinsics,
    compute_scale_and_shift,
    declared_invalid_sentinel,
    mask_invalid_sentinel,
    process_depth,
    project_point_cloud_to_depth_map,
    to_numpy_depth,
    to_numpy_extrinsics,
    to_numpy_intrinsics,
    to_numpy_point_cloud,
)
from .evaluate import (
    _apply_sky_depth,
    _CapClipping,
    _changed,
    _extract_hierarchy,
    _item_id,
    _per_file_standard,
    _prediction_resample,
    _scored_depth_mask,
    _SpatialTracker,
)
from .metrics.depth_standard import (
    STANDARD_DEPTH_METRIC_KEYS,
    append_standard_depth_metrics,
    compute_standard_depth_metrics,
    init_standard_depth_store,
    summarize_standard_depth_store,
)
from .metrics.utils import get_benchmark_depth_bins
from .naming import (
    DEPTH_EVAL_DESCRIPTIONS,
    SPARSE_DEPTH_EVAL_DESCRIPTIONS,
    SPARSE_DEPTH_METRIC_ROOT,
    EvalNamespace,
    depth_eval_axes,
    sparse_depth_eval_axes,
)
from .protocol import (
    DEPTH_AXIS_ROLES,
    DEPTH_VOCABULARIES,
    POLICIES,
    depth_gt_range,
    depth_parameters,
    domain_region,
    protocol_block,
    read_references,
    reference_input,
    sky_gt_parameters,
    sparse_reference_inputs,
    spatial_parameters,
)

__all__ = [
    "PER_ITEM_DEFAULT_LIMIT",
    "VALIDATION_SET_ALIGNMENT_MODES",
    "DepthValidationPass",
    "DepthValidationSet",
]

#: Alignment modes of a validation set (the CLI's ``--depth-alignment``).
VALIDATION_SET_ALIGNMENT_MODES = ("none", "affine", "auto_affine")
#: Sets of at most this many items write per-item values by default.
PER_ITEM_DEFAULT_LIMIT = 1000

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_BINS = ("all", "near", "mid", "far")
#: Keys a pass records (resolved while scoring) and the set only describes.
_PASS_PARAMETERS = (
    "depth.alignment.applied",
    "depth.input_space",
    "spatial.method",
    "spatial.uniform",
    "depth.extra_mask",
)
_DENSE_OBSERVATIONS = ("domain.region", "prediction.resample", "depth.cap.clipped")
_SPARSE_OBSERVATIONS = ("domain.grid", "depth.cap.clipped")
#: Settings of an evaluation the in-loop path cannot apply.
_REFUSED = {
    "sky.gt.masking": (True, "GT sky masking"),
    "sky.pred.masking": (True, "predicted-sky masking"),
    "sky.pred.fill": (True, "predicted-sky filling"),
}


def _version() -> str:
    try:
        return importlib.metadata.version("euler-eval")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _split_of(path: Optional[str], split: Optional[str]) -> Optional[str]:
    """The split a path is read with: its inline ``:split``, else *split*."""
    if path is None:
        return None
    return parse_modality_path(path).split or split


class DepthValidationSet:
    """A depth validation set: the GT, the settings and the items, declared once.

    Args:
        key: The set's key (``[a-z0-9][a-z0-9_-]{0,63}``), e.g. ``"muses"``.
        depth_path: Dense GT depth (euler-loading path, inline
            ``:split#scope=`` selectors allowed). Exactly one of
            ``depth_path`` and ``sparse_depth_path``.
        sparse_depth_path: Sparse point-cloud GT; needs ``intrinsics_path``
            and ``camera_extrinsics_path``.
        intrinsics_path, camera_extrinsics_path, lidar_extrinsics_path:
            Hierarchical inputs of a sparse GT.
        split: Split for paths without an inline one.
        dataset: The GT-only dataset to score; default
            :func:`~euler_eval.validation.build_validation_gt_dataset` of the
            paths. Pass one built from the same paths (e.g. with loader
            transforms that keep the GT as stored).
        indices: Dataset indices the set evaluates (default: all), e.g. a
            fixed subset of a large set. Their item ids are the population.
        label: A human-readable label of the set document.
        alignment: ``none``, ``affine`` or ``auto_affine`` (the CLI's
            ``--depth-alignment``): ``auto_affine`` fits scale and shift only
            when the first prediction of a pass looks normalized.
        input_space_hint: ``relative`` or ``affine`` makes ``auto_affine`` fit
            whatever the value range (the CLI's prediction declaration).
        sky_depth: Depth cap in metres (the CLI's ``--sky-depth``).
        benchmark_depth_range: ``(min, max)`` metres of the benchmark bins
            (the CLI's ``--benchmark-depth-range``).
        spatial_alignment: ``auto``, ``crop``, ``resize`` or a prediction's
            recorded source grid (a mapping or
            :class:`~euler_eval.data.SourceGrid`): how a prediction at another
            resolution is put on the GT grid (dense GT).
        valid_depth_range: ``(min, max)`` GT depth in metres outside which
            pixels are not scored (either bound may be ``None``). In-loop
            only: recorded as ``depth.gt.valid_range_m``, so the set is not
            comparable with evaluations that did not bound GT validity.
        pred_is_radial: Sparse GT: whether predictions are radial already
            (planar z-depth predictions are converted before scoring).
        per_item: Write per-item values and pixel counts with every pass
            (default: when the set has at most
            :data:`PER_ITEM_DEFAULT_LIMIT` items).
        transforms: Loader transforms for the default dataset.
    """

    def __init__(
        self,
        key: str,
        *,
        depth_path: Optional[str] = None,
        sparse_depth_path: Optional[str] = None,
        intrinsics_path: Optional[str] = None,
        camera_extrinsics_path: Optional[str] = None,
        lidar_extrinsics_path: Optional[str] = None,
        split: Optional[str] = None,
        dataset: Any = None,
        indices: Optional[Sequence[int]] = None,
        label: Optional[str] = None,
        alignment: str = "none",
        input_space_hint: Optional[str] = None,
        sky_depth: Optional[float] = None,
        benchmark_depth_range: Optional[Sequence[float]] = None,
        spatial_alignment: Union[str, Mapping[str, Any], SourceGrid] = "auto",
        valid_depth_range: Optional[Sequence[Optional[float]]] = None,
        pred_is_radial: bool = False,
        per_item: Optional[bool] = None,
        transforms: Optional[list[Callable[[dict], dict]]] = None,
    ) -> None:
        if not (isinstance(key, str) and _KEY_RE.match(key)):
            raise ValueError("a validation set key must match [a-z0-9][a-z0-9_-]{0,63}")
        if (depth_path is None) == (sparse_depth_path is None):
            raise ValueError("give exactly one of depth_path and sparse_depth_path")
        if alignment not in VALIDATION_SET_ALIGNMENT_MODES:
            raise ValueError(f"alignment must be one of {VALIDATION_SET_ALIGNMENT_MODES}")
        if input_space_hint not in (None, "relative", "affine"):
            raise ValueError("input_space_hint must be None, 'relative' or 'affine'")
        if sky_depth is not None and not (math.isfinite(float(sky_depth)) and float(sky_depth) > 0):
            raise ValueError("sky_depth must be a finite positive number of metres")
        self.key = key
        self.label = label
        self.sparse = sparse_depth_path is not None
        self.alignment = alignment
        self.input_space_hint = input_space_hint
        self.sky_depth = None if sky_depth is None else float(sky_depth)
        self.benchmark_depth_range = (
            None if benchmark_depth_range is None
            else tuple(float(v) for v in benchmark_depth_range)
        )
        if isinstance(spatial_alignment, Mapping):
            spatial_alignment = SourceGrid.from_value(spatial_alignment)
        self.spatial_alignment = spatial_alignment
        self.valid_depth_range = (
            None if valid_depth_range is None
            else [_finite(valid_depth_range[0]), _finite(valid_depth_range[1])]
        )
        self.pred_is_radial = bool(pred_is_radial)
        self._paths = {
            "depth": depth_path,
            "sparse_depth": sparse_depth_path,
            "intrinsics": intrinsics_path,
            "camera_extrinsics": camera_extrinsics_path,
            "lidar_extrinsics": lidar_extrinsics_path,
        }
        self._split = split
        if dataset is None:
            from .validation import build_validation_gt_dataset

            dataset = build_validation_gt_dataset(
                depth_path=depth_path,
                sparse_depth_path=sparse_depth_path,
                intrinsics_path=intrinsics_path,
                camera_extrinsics_path=camera_extrinsics_path,
                lidar_extrinsics_path=lidar_extrinsics_path,
                split=split,
                transforms=transforms,
            )
        self.dataset = dataset
        self.gt_modality = "sparse_depth" if self.sparse else "depth"
        self.indices = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
        self.item_ids = [self._index_item_id(index) for index in self.indices]
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("the selected items have duplicate item ids")
        self.per_item = len(self.item_ids) <= PER_ITEM_DEFAULT_LIMIT if per_item is None else bool(per_item)
        self.gt_sentinel = declared_invalid_sentinel(dataset, self.gt_modality)
        if self.benchmark_depth_range is not None:
            self.benchmark_boundaries = get_benchmark_depth_bins(
                np.empty((0,), dtype=np.float32), *self.benchmark_depth_range
            )["boundaries"]
        else:
            self.benchmark_boundaries = None
        self.references = self._read_references()
        self.namespace = f"{SPARSE_DEPTH_METRIC_ROOT if self.sparse else 'depth'}.eval"
        self._document: Optional[dict] = None

    # -- construction from an evaluation --------------------------------------------------

    @classmethod
    def from_protocol(cls, key: str, evaluation: Any, **kwargs: Any) -> "DepthValidationSet":
        """A set with the settings of an existing depth evaluation.

        *evaluation* is an ``eval.json`` (a path or a parsed dict, schema 1 or
        2). Its alignment mode, depth cap, benchmark range and spatial policy
        are taken over; the GT paths and the rest come from *kwargs*. Settings
        the in-loop path cannot apply (GT or predicted sky masking) raise
        ``ValueError``; explicit *kwargs* win over the evaluation's.
        """
        from euler_metric_naming.contract import load_eval_file

        loaded = load_eval_file(evaluation if isinstance(evaluation, Mapping) else str(evaluation))
        parameters = loaded.contract.get("parameters") or {}
        refused = [
            label
            for parameter, (value, label) in _REFUSED.items()
            if parameters.get(parameter) == value
        ]
        if refused:
            raise ValueError(
                "the evaluation uses settings a validation set cannot apply: " + ", ".join(refused)
            )
        settings: dict[str, Any] = {}
        if isinstance(parameters.get("depth.alignment.mode"), str):
            settings["alignment"] = parameters["depth.alignment.mode"]
        if "depth.cap.max_m" in parameters:
            settings["sky_depth"] = parameters["depth.cap.max_m"]
        bench = parameters.get("depth.benchmark.range_m")
        if isinstance(bench, list) and len(bench) == 2:
            settings["benchmark_depth_range"] = tuple(bench)
        policy = parameters.get("spatial.policy")
        if isinstance(policy, str) and policy in ("auto", "crop", "resize"):
            settings["spatial_alignment"] = policy
        settings.update(kwargs)
        return cls(key, **settings)

    # -- items and references --------------------------------------------------------------

    def _index_item_id(self, index: int) -> str:
        lookup = getattr(self.dataset, "sample_full_id", None)
        if callable(lookup):
            full_id = lookup(index)
            parts = [part for part in str(full_id).strip("/").split("/") if part]
            if parts:
                return _nfc(_item_id(parts[:-1], parts[-1]))
        return self.sample_item_id(self.dataset[index])

    @staticmethod
    def sample_item_id(sample: Mapping[str, Any]) -> str:
        """The ``euler.item_id/1`` of a dataset sample (as the CLI writes it)."""
        hierarchy, entry_id = _extract_hierarchy(sample)
        return _nfc(_item_id(hierarchy, entry_id))

    def _read_references(self) -> dict:
        split = self._split
        if self.sparse:
            inputs = sparse_reference_inputs(
                gt=(self._paths["sparse_depth"], split),
                intrinsics=(self._paths["intrinsics"], split),
                camera_extrinsics=(self._paths["camera_extrinsics"], split),
                lidar_extrinsics=(self._paths["lidar_extrinsics"], split),
                segmentation=(None, None, None),
                gt_modality="sparse_depth",
            )
        else:
            inputs = [reference_input("gt", "depth", self._paths["depth"], split, "depth")]
        return read_references(self.dataset, inputs)

    # -- the set document --------------------------------------------------------------------

    def _axes(self) -> dict:
        benchmark = self.benchmark_depth_range is not None
        if self.sparse:
            return sparse_depth_eval_axes(benchmark=benchmark)
        return depth_eval_axes(benchmark=benchmark)

    def metric_set(self) -> dict:
        """The metricSet envelope of the set (namespace, axes, descriptions)."""
        root = SPARSE_DEPTH_METRIC_ROOT if self.sparse else "depth"
        namespace = EvalNamespace(
            producer="euler-eval",
            producer_version=_version(),
            modalities=(root,),
            axes=self._axes(),
            descriptions=SPARSE_DEPTH_EVAL_DESCRIPTIONS if self.sparse else DEPTH_EVAL_DESCRIPTIONS,
        )
        return namespace.metric_set_envelope(
            root,
            metadata={"calibration_mode": self.alignment, "validation_set": self.key},
        )

    def parameters(self) -> dict:
        """The configured parameters (the resolved ones are recorded per pass)."""
        parameters = {
            **depth_parameters(
                alignment_mode=self.alignment,
                alignment_applied=False,
                input_space=None,
                sky_depth=self.sky_depth,
                mask_sky=False,
                benchmark_depth_range=self.benchmark_depth_range,
                conversion=self._conversion(),
            ),
            **sky_gt_parameters(mask_sky=False, missing_policy=None),
            "sky.pred.masking": False,
            "sky.pred.fill": False,
            "sky.iou": False,
            **spatial_parameters(
                self._spatial_policy(),
                {},
                **(
                    {"spatial.mask_resample": POLICIES["spatial.mask_resample"]}
                    if self.sparse
                    else {
                        "spatial.resize_target": "gt_grid",
                        "spatial.mask_resample": POLICIES["spatial.mask_resample"],
                    }
                ),
            ),
            "depth.gt.valid_range_m": self.valid_depth_range,
        }
        for key in _PASS_PARAMETERS:
            parameters.pop(key, None)
        return parameters

    def _conversion(self) -> str:
        if self.sparse:
            return "none" if self.pred_is_radial else "planar_to_radial"
        return "none"

    def _spatial_policy(self) -> str:
        if self.sparse:
            return "auto"
        return "source_grid" if isinstance(self.spatial_alignment, SourceGrid) else str(self.spatial_alignment)

    def _names(self) -> tuple:
        """(aggregate names, per-item names) the set's passes can write (metric space)."""
        prefix = f"{self.namespace}.metric.standard"
        bins = _BINS if self.benchmark_depth_range is not None else ()
        aggregate, per_item = [], [f"{prefix}.count"]
        for reduction in ("image_mean", "image_median", "pixel_pool"):
            for metric in STANDARD_DEPTH_METRIC_KEYS:
                aggregate.append(f"{prefix}.{reduction}.{metric}")
                for bin_name in bins:
                    aggregate.append(f"{prefix}.{reduction}.{bin_name}.{metric}")
        for metric in STANDARD_DEPTH_METRIC_KEYS:
            per_item.append(f"{prefix}.{metric}")
        for bin_name in bins:
            per_item.append(f"{prefix}.{bin_name}.count")
            for metric in STANDARD_DEPTH_METRIC_KEYS:
                per_item.append(f"{prefix}.{bin_name}.{metric}")
        return aggregate, per_item

    def declaration(self) -> dict:
        """The set document (``validation/<key>/set.json``) of euler-train."""
        if self._document is None:
            aggregate, per_item = self._names()
            stratification = depth_gt_range(
                self.benchmark_boundaries, self.benchmark_depth_range
            )
            protocol = protocol_block(
                task="depth.sparse" if self.sparse else "depth.dense",
                vocabularies=DEPTH_VOCABULARIES,
                metric_set=self.metric_set(),
                references=self.references,
                parameters=self.parameters(),
                described=(
                    *_PASS_PARAMETERS,
                    *(_SPARSE_OBSERVATIONS if self.sparse else _DENSE_OBSERVATIONS),
                ),
                ids=self.item_ids,
                accounting={"reference": len(self.dataset), "paired": len(self.item_ids)},
                axis_roles=DEPTH_AXIS_ROLES,
                stratifications={"depth.gt_range": stratification},
                aggregate_names=aggregate,
                per_item_names=per_item,
            )
            document = {
                "kind": "euler.validation_set",
                "schemaVersion": "1.0",
                "key": self.key,
            }
            if self.label is not None:
                document["label"] = self.label
            document["metricSet"] = self.metric_set()
            document["protocol"] = protocol
            self._document = document
        return copy.deepcopy(self._document)

    def new_pass(self) -> "DepthValidationPass":
        """An empty pass: :meth:`DepthValidationPass.add` each sample, then ``record()``."""
        return DepthValidationPass(self)


class DepthValidationPass:
    """One validation pass of a :class:`DepthValidationSet`.

    :meth:`add` scores one dataset sample against an in-memory prediction the
    way the CLI scores the same pair; :meth:`record` returns the pass record
    for euler-train's ``run.log_validation``. For multi-process validation,
    exchange :meth:`state` and :meth:`merge` the others into one pass.
    """

    def __init__(self, validation_set: DepthValidationSet) -> None:
        self._set = validation_set
        self._items: dict[str, dict] = {}
        self._order: list[str] = []
        self._first: Optional[dict] = None
        self._uniform = True
        self._tracker = _SpatialTracker()
        self._cap = _CapClipping()
        self._applied = False
        self._extra_mask = False

    # -- scoring ---------------------------------------------------------------------------------

    def add(
        self,
        sample: Mapping[str, Any],
        prediction: Any,
        *,
        valid_mask: Any = None,
        intrinsics: Any = None,
        camera_extrinsics: Any = None,
        lidar_extrinsics: Any = None,
    ) -> dict:
        """Score *prediction* against the GT of dataset *sample*; return its per-item values.

        *sample* is an item of the set's dataset (its GT as loaded, its
        ``id`` / ``full_id``). *prediction* is the model's depth (metres,
        unless the set aligns), at any resolution for dense GT. A
        *valid_mask* (prediction or GT plane) removes more pixels and is
        recorded as ``depth.extra_mask``. Sparse GT: *intrinsics* (of the
        prediction plane) and *camera_extrinsics* / *lidar_extrinsics*
        default to the sample's own.
        """
        item = DepthValidationSet.sample_item_id(sample)
        if item in self._items:
            raise ValueError(f"item {item!r} was already scored in this pass")
        if valid_mask is not None:
            self._extra_mask = True
        if self._set.sparse:
            scored = self._score_sparse(
                sample, prediction, valid_mask, intrinsics, camera_extrinsics, lidar_extrinsics
            )
        else:
            scored = self._score_dense(sample, prediction, valid_mask)
        self._items[item] = scored
        self._order.append(item)
        return {name: value for name, value in scored["per_file"].items()}

    def _detect(self, prediction: np.ndarray) -> None:
        """The CLI's input-space detection on the first prediction of a pass."""
        if self._first is not None:
            return
        hint = self._set.input_space_hint
        if hint in ("relative", "affine"):
            normalized, space = True, hint
        else:
            low, high = float(np.nanmin(prediction)), float(np.nanmax(prediction))
            normalized = high <= 1.0 + 1e-3 and low >= -1.0 - 1e-3
            space = "normalized" if normalized else "metric"
        self._first = {"normalized": normalized, "input_space": space}

    def _fit(self, normalized: bool) -> bool:
        mode = self._set.alignment
        return mode == "affine" or (mode == "auto_affine" and normalized)

    def _valid(self, gt: np.ndarray, pred: np.ndarray, extra: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """The scored pixels: the CLI's default, narrowed by the in-loop settings."""
        bounds = self._set.valid_depth_range
        if extra is None and bounds is None:
            return None
        valid = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
        if bounds is not None:
            if bounds[0] is not None:
                valid &= gt >= bounds[0]
            if bounds[1] is not None:
                valid &= gt <= bounds[1]
        if extra is not None:
            valid &= extra
        return valid

    def _restrict(self, gt: np.ndarray, extra: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """In-loop masks that also bound the alignment fit and the bins."""
        bounds = self._set.valid_depth_range
        if extra is None and bounds is None:
            return None
        keep = np.ones(gt.shape, dtype=bool)
        if bounds is not None:
            if bounds[0] is not None:
                keep &= gt >= bounds[0]
            if bounds[1] is not None:
                keep &= gt <= bounds[1]
        if extra is not None:
            keep &= extra
        return keep

    def _score_dense(self, sample: Mapping[str, Any], prediction: Any, valid_mask: Any) -> dict:
        from .validation import _mask_on_gt_grid

        vset = self._set
        policy = vset.spatial_alignment
        depth_gt = mask_invalid_sentinel(
            to_numpy_depth(sample[vset.gt_modality]), vset.gt_sentinel, fill=0.0
        )
        depth_pred = to_numpy_depth(prediction)
        gt_shape, pred_shape = depth_gt.shape[:2], depth_pred.shape[:2]
        depth_pred, depth_gt, method = align_to_ground_truth(
            depth_pred, depth_gt, spatial_alignment=policy
        )
        origin = _crop_origin(method, policy, pred_shape)
        self._tracker.add(method, gt_shape, pred_shape, depth_gt.shape[:2], origin or (0, 0))
        first = self._first is None
        self._detect(depth_pred)
        if first:
            self._first.update(
                method=method,
                evaluated=list(depth_gt.shape[:2]),
                origin=list(origin) if origin else None,
                resample=_prediction_resample(method, policy, gt_shape, pred_shape),
            )
        extra = _mask_on_gt_grid(valid_mask, depth_gt.shape, gt_shape, method, policy)
        restrict = self._restrict(depth_gt, extra)
        normalized = self._first["normalized"]

        depth_gt = process_depth(depth_gt, True, None)
        raw = process_depth(depth_pred, True, None)
        aligned = raw
        if self._fit(normalized):
            fit_mask = (depth_gt > 0) & np.isfinite(depth_gt) & np.isfinite(depth_pred)
            if restrict is not None:
                fit_mask &= restrict
            aligned, _, _ = compute_scale_and_shift(depth_pred, depth_gt, fit_mask)
            self._applied = True
        uncapped = (depth_gt, raw, aligned)
        depth_gt, raw, aligned = _apply_sky_depth(depth_gt, raw, aligned, vset.sky_depth, None)
        return self._score_branches(depth_gt, raw, aligned, uncapped, restrict)

    def _score_sparse(
        self,
        sample: Mapping[str, Any],
        prediction: Any,
        valid_mask: Any,
        intrinsics: Any,
        camera_extrinsics: Any,
        lidar_extrinsics: Any,
    ) -> dict:
        from .validation import _as_optional_mask

        vset = self._set
        pred_raw = to_numpy_depth(prediction)
        K = to_numpy_intrinsics(intrinsics if intrinsics is not None else get_sample_intrinsics(sample))
        if camera_extrinsics is None:
            source_to_camera, _ = get_sample_pointcloud_to_camera_extrinsics(sample)
            source_to_camera = to_numpy_extrinsics(source_to_camera)
        else:
            source_to_camera = to_numpy_extrinsics(camera_extrinsics)
            if lidar_extrinsics is not None:
                source_to_camera = compose_sensor_to_camera_extrinsics(
                    to_numpy_extrinsics(lidar_extrinsics), source_to_camera
                )
        points = to_numpy_point_cloud(sample[vset.gt_modality])
        sparse_gt, sparse_mask, _ = project_point_cloud_to_depth_map(
            points, K, source_to_camera, pred_raw.shape[:2]
        )
        first = self._first is None
        self._detect(pred_raw)
        grid = list(pred_raw.shape[:2])
        self._tracker.add("pointcloud_projection", tuple(grid), tuple(grid), tuple(grid), (0, 0))
        if first:
            self._first.update(method="pointcloud_projection", evaluated=grid, origin=None, resample=None)
        normalized = self._first["normalized"]
        fit = self._fit(normalized)
        pred = pred_raw.astype(np.float32, copy=True) if fit else process_depth(
            pred_raw, vset.pred_is_radial, K
        )
        depth_gt = np.where(sparse_mask, sparse_gt, 0.0).astype(np.float32)
        extra = _as_optional_mask(valid_mask, pred)
        restrict = self._restrict(depth_gt, extra)
        aligned = pred
        if fit:
            fit_mask = sparse_mask & (depth_gt > 0) & np.isfinite(depth_gt) & np.isfinite(pred)
            if restrict is not None:
                fit_mask &= restrict
            aligned, _, _ = compute_scale_and_shift(pred, depth_gt, fit_mask)
            self._applied = True
        uncapped = (depth_gt, pred, aligned)
        depth_gt, raw, aligned = _apply_sky_depth(depth_gt, pred, aligned, vset.sky_depth, None)
        return self._score_branches(depth_gt, raw, aligned, uncapped, restrict)

    def _score_branches(self, depth_gt, raw, aligned, uncapped, restrict) -> dict:
        """Score the emitted branch (aligned when a fit ran, else raw), as the CLI does."""
        vset = self._set
        fitted = aligned is not raw
        branch = aligned if fitted else raw
        valid = self._valid(depth_gt, branch, restrict)
        metrics, pool = compute_standard_depth_metrics(branch, depth_gt, valid_mask=valid)
        if vset.sky_depth is not None:
            gt_capped = _changed(uncapped[0], depth_gt)
            raw_valid = self._valid(depth_gt, raw, restrict)
            self._cap.add(
                "raw",
                _scored_depth_mask(depth_gt, raw, raw_valid),
                gt_capped | _changed(uncapped[1], raw),
            )
            if fitted:
                self._cap.add(
                    "aligned",
                    _scored_depth_mask(depth_gt, aligned, valid),
                    gt_capped | _changed(uncapped[2], aligned),
                )
        per_file = {"standard": _per_file_standard(metrics)}
        scored = {
            "metrics": {key: float(value) for key, value in metrics.items()},
            "pool": {key: float(value) for key, value in pool.items()},
            "count": int(pool["count"]),
            "bins": {},
        }
        if vset.benchmark_depth_range is not None:
            bins = get_benchmark_depth_bins(depth_gt, *vset.benchmark_depth_range)
            for bin_name in _BINS:
                mask = bins[bin_name] & (branch > 0) & np.isfinite(branch)
                if restrict is not None:
                    mask &= restrict
                entry: dict = {"count": int(np.count_nonzero(mask))}
                if mask.any():
                    bin_metrics, bin_pool = compute_standard_depth_metrics(
                        branch, depth_gt, valid_mask=mask
                    )
                    entry["metrics"] = {k: float(v) for k, v in bin_metrics.items()}
                    entry["pool"] = {k: float(v) for k, v in bin_pool.items()}
                    per_file["standard"][bin_name] = _per_file_standard(bin_metrics)
                scored["bins"][bin_name] = entry
        scored["per_file"] = per_file
        return scored

    # -- multi-process --------------------------------------------------------------------------

    def state(self) -> dict:
        """A picklable / JSON-ready state to gather from other processes."""
        return copy.deepcopy(
            {
                "items": self._items,
                "order": self._order,
                "first": self._first,
                "uniform": self._tracker.uniform,
                "signature": self._tracker.first,
                "cap": self._cap.counts,
                "applied": self._applied,
                "extra_mask": self._extra_mask,
            }
        )

    def merge(self, state: Mapping[str, Any]) -> "DepthValidationPass":
        """Add another process's :meth:`state` of the same pass."""
        for item in state["order"]:
            if item in self._items:
                raise ValueError(f"item {item!r} was scored by two processes")
            self._items[item] = copy.deepcopy(state["items"][item])
            self._order.append(item)
        other = state.get("first")
        if other is not None:
            if self._first is None:
                self._first = copy.deepcopy(other)
            elif other.get("normalized") != self._first.get("normalized"):
                raise ValueError("processes resolved different input spaces for one pass")
        signature = state.get("signature")
        if signature is not None:
            signature = tuple(tuple(p) if isinstance(p, list) else p for p in signature)
            if self._tracker.first is None:
                self._tracker.first = signature
            elif signature != self._tracker.first:
                self._tracker.uniform = False
        if not state.get("uniform", True):
            self._tracker.uniform = False
        for branch, (changed, scored) in (state.get("cap") or {}).items():
            counts = self._cap.counts.setdefault(branch, [0, 0])
            counts[0] += int(changed)
            counts[1] += int(scored)
        self._applied = self._applied or bool(state.get("applied"))
        self._extra_mask = self._extra_mask or bool(state.get("extra_mask"))
        return self

    # -- the record --------------------------------------------------------------------------------

    def _space(self) -> str:
        normalized = bool(self._first and self._first.get("normalized"))
        return "metric" if (self._applied or not normalized) else "native"

    def record(
        self,
        *,
        step: int,
        epoch: Optional[int] = None,
        wall_time: Optional[float] = None,
        per_item: Optional[bool] = None,
    ) -> dict:
        """The pass record (``docs/training.md`` §3.3) for ``run.log_validation``."""
        if not self._items:
            raise ValueError("a pass needs at least one scored item")
        vset = self._set
        order = {item: index for index, item in enumerate(vset.item_ids)}
        added = {item: index for index, item in enumerate(self._order)}
        ids = sorted(self._items, key=lambda item: (order.get(item, len(order)), added[item]))
        space = self._space()
        prefix = f"{vset.namespace}.{space}.standard"

        store = init_standard_depth_store()
        bin_stores = {bin_name: init_standard_depth_store() for bin_name in _BINS}
        for item in ids:
            scored = self._items[item]
            append_standard_depth_metrics(store, scored["metrics"], scored["pool"])
            for bin_name, entry in scored["bins"].items():
                if "metrics" in entry:
                    append_standard_depth_metrics(bin_stores[bin_name], entry["metrics"], entry["pool"])
        metrics: dict = {}
        for reduction, values in summarize_standard_depth_store(store).items():
            for metric, value in values.items():
                if _finite(value) is not None:
                    metrics[f"{prefix}.{reduction}.{metric}"] = float(value)
        if vset.benchmark_depth_range is not None:
            for reduction in ("image_mean", "image_median", "pixel_pool"):
                for bin_name in _BINS:
                    summary = summarize_standard_depth_store(bin_stores[bin_name])
                    for metric, value in summary.get(reduction, {}).items():
                        if _finite(value) is not None:
                            metrics[f"{prefix}.{reduction}.{bin_name}.{metric}"] = float(value)

        record: dict = {
            "step": int(step),
            "epoch": None if epoch is None else int(epoch),
            "wall_time": time.time() if wall_time is None else float(wall_time),
            "metrics": dict(sorted(metrics.items())),
        }
        if record["epoch"] is None:
            del record["epoch"]
        first = self._first or {}
        record["parameters"] = {
            "depth.alignment.applied": self._applied,
            "depth.input_space": first.get("input_space"),
            "spatial.method": first.get("method", "none"),
            "spatial.uniform": self._tracker.uniform,
            "depth.extra_mask": self._extra_mask,
        }
        clipped = self._cap.fraction(["raw"] + (["aligned"] if self._applied else []))
        if vset.sparse:
            observations = {"domain.grid": first.get("evaluated"), "depth.cap.clipped": clipped}
        else:
            observations = {
                "domain.region": domain_region(
                    vset.references.get("gt"), first.get("evaluated"), first.get("origin")
                ),
                "prediction.resample": first.get("resample"),
                "depth.cap.clipped": clipped,
            }
        record["observations"] = {key: value for key, value in observations.items() if value is not None}
        if ids != vset.item_ids:
            record["population"] = {"ids": list(ids)}
        if self._set.per_item if per_item is None else per_item:
            record["per_item"], record["series"] = self._per_item(ids, prefix)
        return record

    def _per_item(self, ids: list, prefix: str) -> tuple:
        bins = _BINS if self._set.benchmark_depth_range is not None else ()
        per_item: dict = {}
        series: dict = {f"{prefix}.count": [self._items[item]["count"] for item in ids]}
        for metric in STANDARD_DEPTH_METRIC_KEYS:
            per_item[f"{prefix}.{metric}"] = [
                self._items[item]["per_file"]["standard"].get(metric) for item in ids
            ]
        for bin_name in bins:
            series[f"{prefix}.{bin_name}.count"] = [
                self._items[item]["bins"][bin_name]["count"] for item in ids
            ]
            for metric in STANDARD_DEPTH_METRIC_KEYS:
                per_item[f"{prefix}.{bin_name}.{metric}"] = [
                    (self._items[item]["per_file"]["standard"].get(bin_name) or {}).get(metric)
                    for item in ids
                ]
        return per_item, series
