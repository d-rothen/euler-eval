"""Programmatic (in-training) depth validation interface.

The CLI evaluators in :mod:`euler_eval.evaluate` pair on-disk ground truth
with an on-disk *prediction* dataset.  Training loops instead hold live,
in-memory predictions and only the ground truth on disk.  This module exposes
the same metric semantics for that use case:

* :func:`evaluate_dense_depth_sample` — score one dense prediction against a
  dense GT depth map.
* :func:`evaluate_sparse_depth_sample` — score one dense prediction against a
  sparse pointcloud GT (projected with intrinsics + source→camera extrinsics,
  exactly like ``evaluate_sparse_depth_samples``).
* :class:`DepthValidationAggregator` — accumulate per-sample results into
  dataset-level ``image_mean`` / ``image_median`` / ``pixel_pool`` summaries,
  with a fixed-key sufficient-statistics vector for multi-process reduction.
* :func:`build_validation_gt_dataset` — resolve a GT-only
  :class:`~euler_loading.MultiModalDataset` from euler-loading compatible
  paths (``.ds_crawler`` indexed roots, inline ``:split#scope=`` selectors).

All ``evaluate_*_sample`` inputs accept torch tensors or numpy arrays; every
computation runs on CPU numpy.  Predictions and GT are expected in metres
unless an affine ``alignment`` is requested.  Passing
``benchmark_depth_range=(min_metres, max_metres)`` adds the CLI-compatible
square-root-spaced ``all`` / ``near`` / ``mid`` / ``far`` results without
changing the regular metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from euler_loading import MultiModalDataset

from .calibration import (
    get_sample_intrinsics,
    get_sample_pointcloud_to_camera_extrinsics,
)
from .config_paths import parse_modality_path
from .data import (
    _modality,
    align_to_prediction,
    compose_sensor_to_camera_extrinsics,
    compute_scale_and_shift,
    process_depth,
    project_point_cloud_to_depth_map,
    to_numpy_depth,
    to_numpy_extrinsics,
    to_numpy_intrinsics,
    to_numpy_mask,
    to_numpy_point_cloud,
)
from .metrics.depth_standard import (
    STANDARD_DEPTH_METRIC_KEYS,
    append_standard_depth_metrics,
    compute_standard_depth_metrics,
    init_standard_depth_store,
    summarize_standard_depth_store,
)
from .metrics.utils import get_benchmark_depth_bins

__all__ = [
    "BENCHMARK_DEPTH_BIN_NAMES",
    "DepthBenchmarkEvaluation",
    "DepthSampleEvaluation",
    "DepthValidationAggregator",
    "VALIDATION_ALIGNMENT_MODES",
    "build_validation_gt_dataset",
    "evaluate_dense_depth_sample",
    "evaluate_sparse_depth_sample",
    "get_sample_intrinsics",
    "get_sample_pointcloud_to_camera_extrinsics",
    "summarize_reduced_state",
]

VALIDATION_ALIGNMENT_MODES = ("none", "affine")
BENCHMARK_DEPTH_BIN_NAMES = ("all", "near", "mid", "far")

_PROJECTION_STAT_KEYS = (
    "input_points",
    "finite_points",
    "in_front_points",
    "in_image_points",
    "projected_pixels",
)

_POOL_STAT_KEYS = (
    "count",
    "sum_absrel",
    "sum_sqrel",
    "sum_abs",
    "sum_sq",
    "sum_log10_abs",
    "sum_log_diff",
    "sum_log_diff_sq",
    "delta1_hits",
    "delta2_hits",
    "delta3_hits",
)


@dataclass(frozen=True)
class DepthBenchmarkEvaluation:
    """Additive depth metrics for one configured benchmark range.

    ``bins`` contains ``all`` plus square-root-scaled ``near`` / ``mid`` /
    ``far`` evaluations.  A bin is ``None`` when no otherwise-valid pixels
    fall inside it.
    """

    boundaries: dict[str, list[float]]
    bins: dict[str, Optional["DepthSampleEvaluation"]]


@dataclass(frozen=True)
class DepthSampleEvaluation:
    """Result of scoring one prediction/GT pair.

    Attributes:
        metrics: Per-image standard depth metrics
            (:data:`~euler_eval.metrics.depth_standard.STANDARD_DEPTH_METRIC_KEYS`).
        pool_stats: Pixel-pool sufficient statistics for dataset reducers.
        valid_pixels: Number of pixels that entered the metric computation.
        scale: Fitted affine scale when ``alignment="affine"`` was applied.
        shift: Fitted affine shift when ``alignment="affine"`` was applied.
        projection: Sparse projection statistics
            (:func:`~euler_eval.data.project_point_cloud_to_depth_map`
            metadata); ``None`` for dense evaluations.
        benchmark: Optional additive ``all`` / ``near`` / ``mid`` / ``far``
            metrics for the requested benchmark depth range.  Regular
            ``metrics`` remain unchanged.
    """

    metrics: dict[str, float]
    pool_stats: dict[str, float]
    valid_pixels: int
    scale: Optional[float] = None
    shift: Optional[float] = None
    projection: Optional[dict[str, int]] = None
    benchmark: Optional[DepthBenchmarkEvaluation] = None


def _validate_alignment(alignment: str) -> str:
    alignment = str(alignment).strip().lower()
    if alignment not in VALIDATION_ALIGNMENT_MODES:
        raise ValueError(
            f"alignment must be one of {VALIDATION_ALIGNMENT_MODES}, got {alignment!r}"
        )
    return alignment


def _as_optional_mask(valid_mask: Any, reference: np.ndarray) -> Optional[np.ndarray]:
    """Convert an optional mask to bool ``(H, W)`` aligned to ``reference``."""
    if valid_mask is None:
        return None
    mask = to_numpy_mask(valid_mask)
    if mask.shape != reference.shape:
        aligned = align_to_prediction(
            mask.astype(np.float32), reference
        )
        mask = aligned > 0.5
    return mask.astype(bool)


def _apply_depth_bounds(
    valid: np.ndarray,
    depth_gt: np.ndarray,
    min_depth: Optional[float],
    max_depth: Optional[float],
) -> np.ndarray:
    if min_depth is not None:
        valid = valid & (depth_gt >= float(min_depth))
    if max_depth is not None:
        valid = valid & (depth_gt <= float(max_depth))
    return valid


def _validate_benchmark_depth_range(
    benchmark_depth_range: Optional[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    if benchmark_depth_range is None:
        return None
    try:
        range_min, range_max = benchmark_depth_range
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "benchmark_depth_range must contain exactly two numeric bounds"
        ) from exc
    range_min = float(range_min)
    range_max = float(range_max)
    if not np.isfinite(range_min) or not np.isfinite(range_max):
        raise ValueError(
            "benchmark_depth_range bounds must be finite, got "
            f"[{range_min}, {range_max}]"
        )
    # Reuse the canonical CLI helper for validation as well as bin semantics.
    get_benchmark_depth_bins(np.empty((0,), dtype=np.float32), range_min, range_max)
    return range_min, range_max


def _finalize_pair(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid: np.ndarray,
    *,
    alignment: str,
    min_valid_pixels: int,
    projection: Optional[dict[str, int]] = None,
    benchmark_depth_range: Optional[tuple[float, float]] = None,
) -> Optional[DepthSampleEvaluation]:
    """Optionally align, re-mask, and score one prepared pred/GT pair."""
    scale: Optional[float] = None
    shift: Optional[float] = None
    if alignment == "affine":
        if int(valid.sum()) < max(int(min_valid_pixels), 2):
            return None
        depth_pred, fitted_scale, fitted_shift = compute_scale_and_shift(
            depth_pred, depth_gt, valid
        )
        scale, shift = float(fitted_scale), float(fitted_shift)
        # Alignment can push pixels non-positive; those cannot enter the
        # log-based metrics.
        valid = valid & (depth_pred > 0) & np.isfinite(depth_pred)

    if int(valid.sum()) < max(int(min_valid_pixels), 1):
        return None

    benchmark: Optional[DepthBenchmarkEvaluation] = None
    if benchmark_depth_range is not None:
        benchmark_bins = get_benchmark_depth_bins(
            depth_gt,
            benchmark_depth_range[0],
            benchmark_depth_range[1],
        )
        bin_evaluations: dict[str, Optional[DepthSampleEvaluation]] = {}
        for bin_name in BENCHMARK_DEPTH_BIN_NAMES:
            bin_valid = valid & benchmark_bins[bin_name]
            if not bin_valid.any():
                bin_evaluations[bin_name] = None
                continue
            bin_metrics, bin_pool_stats = compute_standard_depth_metrics(
                depth_pred, depth_gt, valid_mask=bin_valid
            )
            bin_evaluations[bin_name] = DepthSampleEvaluation(
                metrics=bin_metrics,
                pool_stats=bin_pool_stats,
                valid_pixels=int(bin_valid.sum()),
                scale=scale,
                shift=shift,
            )
        benchmark = DepthBenchmarkEvaluation(
            boundaries=benchmark_bins["boundaries"],
            bins=bin_evaluations,
        )

    metrics, pool_stats = compute_standard_depth_metrics(
        depth_pred, depth_gt, valid_mask=valid
    )
    return DepthSampleEvaluation(
        metrics=metrics,
        pool_stats=pool_stats,
        valid_pixels=int(valid.sum()),
        scale=scale,
        shift=shift,
        projection=projection,
        benchmark=benchmark,
    )


def evaluate_dense_depth_sample(
    depth_pred: Any,
    depth_gt: Any,
    valid_mask: Any = None,
    *,
    alignment: str = "none",
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    benchmark_depth_range: Optional[tuple[float, float]] = None,
    min_valid_pixels: int = 10,
) -> Optional[DepthSampleEvaluation]:
    """Score one dense depth prediction against a dense GT depth map.

    Args:
        depth_pred: Predicted depth, ``(H, W)`` / ``(1, H, W)`` / ``(H, W, 1)``
            torch tensor or numpy array.  Metres, unless ``alignment="affine"``
            calibrates an affine/relative prediction.
        depth_gt: GT depth in metres, same accepted layouts.  A GT at a
            different resolution is cropped/resized to the prediction plane
            (:func:`~euler_eval.data.align_to_prediction`).
        valid_mask: Optional extra validity mask (prediction- or GT-plane).
        alignment: ``"none"`` scores as-is; ``"affine"`` fits least-squares
            scale+shift on the valid pixels first.
        min_depth: Optional lower GT validity bound in metres.
        max_depth: Optional upper GT validity bound in metres.
        benchmark_depth_range: Optional ``(min, max)`` range in metres.  When
            set, the result additionally contains metrics for all pixels in
            the range and square-root-scaled near/mid/far bins.  The regular
            metrics are unchanged.
        min_valid_pixels: Minimum surviving pixels; fewer returns ``None``.

    Returns:
        A :class:`DepthSampleEvaluation`, or ``None`` when too few valid
        pixels remain.
    """
    alignment = _validate_alignment(alignment)
    benchmark_depth_range = _validate_benchmark_depth_range(benchmark_depth_range)
    pred = to_numpy_depth(depth_pred)
    gt = to_numpy_depth(depth_gt)
    if gt.shape != pred.shape:
        gt = align_to_prediction(gt, pred)

    valid = (
        (gt > 0)
        & (pred > 0)
        & np.isfinite(gt)
        & np.isfinite(pred)
    )
    if alignment == "affine":
        # A non-metric prediction may legitimately contain non-positive raw
        # values; alignment decides positivity afterwards.
        valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    valid = _apply_depth_bounds(valid, gt, min_depth, max_depth)

    extra_mask = _as_optional_mask(valid_mask, pred)
    if extra_mask is not None:
        valid = valid & extra_mask

    return _finalize_pair(
        pred,
        gt,
        valid,
        alignment=alignment,
        min_valid_pixels=min_valid_pixels,
        benchmark_depth_range=benchmark_depth_range,
    )


def evaluate_sparse_depth_sample(
    depth_pred: Any,
    point_cloud: Any,
    intrinsics: Any,
    camera_extrinsics: Any,
    *,
    lidar_extrinsics: Any = None,
    pred_is_radial: bool = False,
    valid_mask: Any = None,
    alignment: str = "none",
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    benchmark_depth_range: Optional[tuple[float, float]] = None,
    min_valid_pixels: int = 10,
) -> Optional[DepthSampleEvaluation]:
    """Score one dense depth prediction against a sparse pointcloud GT.

    Mirrors ``evaluate_sparse_depth_samples``: the cloud is transformed with
    the source→camera extrinsics, projected into the prediction plane with the
    intrinsics (radial GT depth at hit pixels, nearest-z wins), and metrics
    are computed only at projected pixels.

    Args:
        depth_pred: Predicted dense depth (metres unless ``alignment="affine"``).
        point_cloud: ``(N, C>=3)`` source-frame points; first three columns
            ``x, y, z`` in metres (e.g. MUSES lidar).
        intrinsics: ``(3, 3)`` camera matrix of the *prediction* plane.  When
            the model input was cropped/resized, pass the correspondingly
            adjusted matrix.
        camera_extrinsics: Direct source→camera transform (e.g. MUSES
            ``lidar2rgb``), or the camera sensor pose when
            ``lidar_extrinsics`` is provided.
        lidar_extrinsics: Optional source sensor pose in the same shared frame
            as ``camera_extrinsics``; both are composed via
            :func:`~euler_eval.data.compose_sensor_to_camera_extrinsics`.
        pred_is_radial: Whether the prediction is already radial (Euclidean)
            depth.  Planar z-depth predictions (the default) are converted to
            radial with the intrinsics before scoring, matching the projected
            GT convention.
        valid_mask: Optional extra validity mask in the prediction plane
            (e.g. a non-sky mask).
        alignment: ``"none"`` or ``"affine"`` (least-squares scale+shift fitted
            on the raw prediction at projected pixels, like the CLI's
            normalized-prediction path).
        min_depth: Optional lower bound on projected GT depth in metres.
        max_depth: Optional upper bound on projected GT depth in metres.
        benchmark_depth_range: Optional ``(min, max)`` range in metres.  When
            set, the result additionally contains metrics for all projected
            pixels in the range and square-root-scaled near/mid/far bins.
            The regular metrics are unchanged.
        min_valid_pixels: Minimum surviving projected pixels; fewer returns
            ``None``.

    Returns:
        A :class:`DepthSampleEvaluation` with ``projection`` statistics, or
        ``None`` when too few projected pixels remain.
    """
    alignment = _validate_alignment(alignment)
    benchmark_depth_range = _validate_benchmark_depth_range(benchmark_depth_range)
    pred_raw = to_numpy_depth(depth_pred)
    points = to_numpy_point_cloud(point_cloud)
    K = to_numpy_intrinsics(intrinsics)
    camera_T = to_numpy_extrinsics(camera_extrinsics)
    if lidar_extrinsics is not None:
        lidar_T = to_numpy_extrinsics(lidar_extrinsics)
        source_to_camera = compose_sensor_to_camera_extrinsics(lidar_T, camera_T)
    else:
        source_to_camera = camera_T

    sparse_gt, sparse_mask, projection_meta = project_point_cloud_to_depth_map(
        points, K, source_to_camera, pred_raw.shape[:2]
    )

    if alignment == "affine":
        # Fit scale/shift on the raw (possibly normalized) prediction, exactly
        # like the CLI does for declared/detected non-metric predictions.
        pred = pred_raw.astype(np.float32, copy=True)
    else:
        # Projected GT depth is radial; harmonize a planar prediction.
        pred = process_depth(pred_raw, 1.0, pred_is_radial, K)

    valid = (
        sparse_mask
        & (sparse_gt > 0)
        & np.isfinite(sparse_gt)
        & np.isfinite(pred)
    )
    if alignment == "none":
        valid = valid & (pred > 0)
    valid = _apply_depth_bounds(valid, sparse_gt, min_depth, max_depth)

    extra_mask = _as_optional_mask(valid_mask, pred)
    if extra_mask is not None:
        valid = valid & extra_mask

    return _finalize_pair(
        pred,
        sparse_gt,
        valid,
        alignment=alignment,
        min_valid_pixels=min_valid_pixels,
        projection=projection_meta,
        benchmark_depth_range=benchmark_depth_range,
    )


@dataclass
class DepthValidationAggregator:
    """Accumulate :class:`DepthSampleEvaluation` results over a validation set.

    ``summary()`` reports the same ``image_mean`` / ``image_median`` /
    ``pixel_pool`` reducers as the CLI evaluators.  For multi-process
    validation, exchange :meth:`reduced_state` vectors (element-wise sums)
    and rebuild mean-based summaries with :func:`summarize_reduced_state`
    (medians require the raw per-image values and are only available from the
    local ``summary()``).
    """

    _store: dict = field(default_factory=init_standard_depth_store)
    num_samples: int = 0
    num_skipped: int = 0
    valid_pixels: int = 0
    projection_totals: dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in _PROJECTION_STAT_KEYS}
    )

    def update(self, evaluation: Optional[DepthSampleEvaluation]) -> bool:
        """Record one sample evaluation.  ``None`` counts as skipped."""
        if evaluation is None:
            self.num_skipped += 1
            return False
        append_standard_depth_metrics(
            self._store, evaluation.metrics, evaluation.pool_stats
        )
        self.num_samples += 1
        self.valid_pixels += int(evaluation.valid_pixels)
        if evaluation.projection is not None:
            for key in _PROJECTION_STAT_KEYS:
                self.projection_totals[key] += int(evaluation.projection.get(key, 0))
        return True

    def summary(self) -> dict[str, Any]:
        """Full local summary (includes medians)."""
        result: dict[str, Any] = {
            "standard": summarize_standard_depth_store(self._store),
            "num_samples": self.num_samples,
            "num_skipped": self.num_skipped,
            "valid_pixels": self.valid_pixels,
        }
        if any(self.projection_totals.values()):
            result["projection"] = dict(self.projection_totals)
        return result

    # -- multi-process reduction support ---------------------------------

    @staticmethod
    def state_keys() -> tuple[str, ...]:
        """Fixed ordering of :meth:`reduced_state` entries."""
        keys: list[str] = ["num_samples", "num_skipped", "valid_pixels"]
        for metric in STANDARD_DEPTH_METRIC_KEYS:
            keys.append(f"image_sum_{metric}")
            keys.append(f"image_count_{metric}")
        keys.extend(f"pool_{key}" for key in _POOL_STAT_KEYS)
        keys.extend(f"projection_{key}" for key in _PROJECTION_STAT_KEYS)
        return tuple(keys)

    def reduced_state(self) -> dict[str, float]:
        """Sum-reducible sufficient statistics keyed by :meth:`state_keys`."""
        state: dict[str, float] = {
            "num_samples": float(self.num_samples),
            "num_skipped": float(self.num_skipped),
            "valid_pixels": float(self.valid_pixels),
        }
        for metric in STANDARD_DEPTH_METRIC_KEYS:
            values = self._store["per_image"][metric]
            state[f"image_sum_{metric}"] = float(np.sum(values)) if values else 0.0
            state[f"image_count_{metric}"] = float(len(values))
        pool = self._store["pool"]
        for key in _POOL_STAT_KEYS:
            state[f"pool_{key}"] = float(pool[key])
        for key in _PROJECTION_STAT_KEYS:
            state[f"projection_{key}"] = float(self.projection_totals[key])
        return state


def summarize_reduced_state(state: dict[str, float]) -> dict[str, Any]:
    """Rebuild mean-based summaries from (reduced) sufficient statistics.

    Args:
        state: A ``{key: summed value}`` mapping using
            :meth:`DepthValidationAggregator.state_keys` keys, typically after
            an across-process element-wise sum of per-process
            :meth:`DepthValidationAggregator.reduced_state` vectors.

    Returns:
        ``{"standard": {"image_mean": ..., "pixel_pool": ...},
        "num_samples": ..., "num_skipped": ..., "valid_pixels": ...,
        "projection": ...}`` — like
        :meth:`DepthValidationAggregator.summary` but without median-based
        reducers (they are not sum-reducible).
    """
    image_mean: dict[str, float] = {}
    for metric in STANDARD_DEPTH_METRIC_KEYS:
        count = float(state.get(f"image_count_{metric}", 0.0))
        total = float(state.get(f"image_sum_{metric}", 0.0))
        image_mean[metric] = total / count if count > 0 else float("nan")

    count = float(state.get("pool_count", 0.0))
    if count > 0:
        mean_log_diff = state.get("pool_sum_log_diff", 0.0) / count
        mean_log_diff_sq = state.get("pool_sum_log_diff_sq", 0.0) / count
        silog_sq = max(mean_log_diff_sq - mean_log_diff**2, 0.0)
        pixel_pool = {
            "absrel": state.get("pool_sum_absrel", 0.0) / count,
            "sqrel": state.get("pool_sum_sqrel", 0.0) / count,
            "mae": state.get("pool_sum_abs", 0.0) / count,
            "rmse": float(np.sqrt(state.get("pool_sum_sq", 0.0) / count)),
            "rmse_log": float(
                np.sqrt(state.get("pool_sum_log_diff_sq", 0.0) / count)
            ),
            "log10": state.get("pool_sum_log10_abs", 0.0) / count,
            "silog": float(np.sqrt(silog_sq)),
            "delta1": state.get("pool_delta1_hits", 0.0) / count,
            "delta2": state.get("pool_delta2_hits", 0.0) / count,
            "delta3": state.get("pool_delta3_hits", 0.0) / count,
        }
    else:
        pixel_pool = {key: float("nan") for key in STANDARD_DEPTH_METRIC_KEYS}

    result: dict[str, Any] = {
        "standard": {"image_mean": image_mean, "pixel_pool": pixel_pool},
        "num_samples": int(state.get("num_samples", 0.0)),
        "num_skipped": int(state.get("num_skipped", 0.0)),
        "valid_pixels": int(state.get("valid_pixels", 0.0)),
    }
    projection = {
        key: int(state.get(f"projection_{key}", 0.0))
        for key in _PROJECTION_STAT_KEYS
    }
    if any(projection.values()):
        result["projection"] = projection
    return result


def build_validation_gt_dataset(
    *,
    depth_path: Optional[str] = None,
    sparse_depth_path: Optional[str] = None,
    rgb_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    camera_extrinsics_path: Optional[str] = None,
    lidar_extrinsics_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
    split: Optional[str] = None,
    transforms: Optional[list[Callable[[dict], dict]]] = None,
) -> MultiModalDataset:
    """Build a GT-only euler-loading dataset for in-training validation.

    Exactly one of ``depth_path`` (dense GT) or ``sparse_depth_path`` (sparse
    pointcloud GT) is required.  Sparse GT additionally requires
    ``intrinsics_path`` and ``camera_extrinsics_path`` for projection.  All
    paths accept euler-loading inline selectors
    (``/data/muses.zip:val#scope=rgb``); ``split`` applies to paths without an
    inline split.

    The returned samples use modality-named keys: ``"depth"`` or
    ``"sparse_depth"``, optional ``"rgb"``, and hierarchical ``"intrinsics"``
    / ``"camera_extrinsics"`` / ``"lidar_extrinsics"`` / ``"segmentation"``
    entries — ready for :func:`get_sample_intrinsics`,
    :func:`get_sample_pointcloud_to_camera_extrinsics`, and the
    ``evaluate_*_sample`` functions.

    Loaders resolve automatically from each path's ds-crawler index metadata.
    """
    if (depth_path is None) == (sparse_depth_path is None):
        raise ValueError(
            "Provide exactly one of depth_path (dense GT) or "
            "sparse_depth_path (sparse pointcloud GT)."
        )
    if sparse_depth_path is not None and (
        intrinsics_path is None or camera_extrinsics_path is None
    ):
        raise ValueError(
            "sparse_depth_path requires intrinsics_path and "
            "camera_extrinsics_path for pointcloud projection."
        )

    def _fallback_split(path: str) -> Optional[str]:
        """Apply ``split`` only to paths without an inline ``:split`` selector."""
        if split is None:
            return None
        return None if parse_modality_path(path).split is not None else split

    modalities = {}
    if depth_path is not None:
        modalities["depth"] = _modality(
            path=depth_path, modality_key="depth", split=_fallback_split(depth_path)
        )
    else:
        modalities["sparse_depth"] = _modality(
            path=sparse_depth_path,
            modality_key="sparse_depth",
            split=_fallback_split(sparse_depth_path),
        )
    if rgb_path is not None:
        modalities["rgb"] = _modality(
            path=rgb_path, modality_key="rgb", split=_fallback_split(rgb_path)
        )

    hierarchical = {}
    if intrinsics_path is not None:
        hierarchical["intrinsics"] = _modality(
            path=intrinsics_path,
            modality_key="intrinsics",
            split=_fallback_split(intrinsics_path),
        )
    if camera_extrinsics_path is not None:
        hierarchical["camera_extrinsics"] = _modality(
            path=camera_extrinsics_path,
            modality_key="camera_extrinsics",
            split=_fallback_split(camera_extrinsics_path),
        )
    if lidar_extrinsics_path is not None:
        lidar_scope = (
            parse_modality_path(lidar_extrinsics_path).metadata_scope
            or "lidar_extrinsics"
        )
        hierarchical["lidar_extrinsics"] = _modality(
            path=lidar_extrinsics_path,
            modality_key="camera_extrinsics",
            metadata_scope=lidar_scope,
            split=_fallback_split(lidar_extrinsics_path),
        )
    if segmentation_path is not None:
        hierarchical["segmentation"] = _modality(
            path=segmentation_path,
            modality_key="segmentation",
            split=_fallback_split(segmentation_path),
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
        transforms=transforms,
    )
