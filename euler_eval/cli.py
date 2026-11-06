#!/usr/bin/env python3
"""Main entry point for depth, sparse depth, RGB, rays, and points_3d evaluation.

Parses config.json and runs evaluation using euler_loading datasets.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import re
import sys
import traceback
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import torch
from ds_crawler.zip_utils import get_zip_root_prefix

from .config_paths import (
    GT_SEGMENTATION_KEYS,
    normalize_modality_path,
    resolve_dataset_location,
    resolve_gt_segmentation_path,
)
from .data import (
    PartialInputDataset,
    build_depth_eval_dataset,
    build_points_3d_eval_dataset,
    build_points_3d_sparse_eval_dataset,
    build_rays_eval_dataset,
    build_rgb_eval_dataset,
    build_sparse_depth_eval_dataset,
    count_ids_dropped_by,
    get_depth_metadata,
    get_points_3d_metadata,
    get_rays_metadata,
    get_rgb_metadata,
    get_sparse_depth_metadata,
    validate_sky_depth,
)
from .distributions import DistributionConfig
from .evaluate import (
    evaluate_depth_samples,
    evaluate_points_3d_samples,
    evaluate_points_3d_sparse_samples,
    evaluate_rays_samples,
    evaluate_rgb_samples,
    evaluate_sparse_depth_samples,
)
from .metric_sets import (
    DOMAIN_METRIC_SETS,
    available_domains,
    resolve_metric_sets,
    selected_domain_categories,
)
from .protocol import build_protocol, evaluated_resolution, reference_identities
from .sanity_checker import SanityChecker

try:
    import euler_train as _euler_train
except ImportError:
    _euler_train = None

from euler_metric_naming import AxisDeclaration, MetricDescription, MetricNamespace

# ── Eval namespace ──────────────────────────────────────────────────────────
# MetricNamespace subclass for eval context with custom axis declarations,
# following the Normed Metric Namespacing convention (§3.6, §4, §5.2).


class _EvalNamespace(MetricNamespace):
    """MetricNamespace for eval context with caller-supplied axes."""

    def __init__(
        self,
        *,
        axes: dict[str, AxisDeclaration],
        **kwargs,
    ):
        self._eval_axes = dict(axes)
        super().__init__(context="eval", **kwargs)

    def _build_axes(self) -> dict[str, AxisDeclaration]:
        return dict(self._eval_axes)


# ── Axis declarations ───────────────────────────────────────────────────────

_DEPTH_SPACE_AXIS = AxisDeclaration(
    position=0,
    values=("native", "metric"),
    optional=False,
    description="Depth space semantics",
)

_DEPTH_CATEGORY_AXIS = AxisDeclaration(
    position=1,
    values=("image_quality", "standard", "depth_metrics", "geometric_metrics"),
    optional=True,
    description="Metric category",
)

_DEPTH_REDUCTION_AXIS = AxisDeclaration(
    position=2,
    values=("image_mean", "image_median", "pixel_pool"),
    optional=True,
    description="Dataset reduction mode",
)

_SPARSE_DEPTH_CATEGORY_AXIS = AxisDeclaration(
    position=1,
    values=("standard", "depth_metrics"),
    optional=True,
    description="Sparse depth metric category",
)

_BENCHMARK_BIN_AXIS = AxisDeclaration(
    position=3,
    values=("all", "near", "mid", "far"),
    optional=True,
    description="Depth benchmark bin",
)

_RGB_CORE_CATEGORIES = (
    "image_quality",
    "edge_f1",
    "tail_errors",
    "high_frequency",
    "depth_binned_photometric",
)


def _rgb_category_axis(domains=None) -> AxisDeclaration:
    return AxisDeclaration(
        position=0,
        values=(*_RGB_CORE_CATEGORIES, *selected_domain_categories(domains)),
        optional=True,
        description="Metric category",
    )


_RGB_CATEGORY_AXIS = _rgb_category_axis()

_RGB_BENCHMARK_BIN_AXIS = AxisDeclaration(
    position=1,
    values=("all", "near", "mid", "far"),
    optional=True,
    description="Depth benchmark bin",
)


def _distribution_category(axis: AxisDeclaration, enabled: bool) -> AxisDeclaration:
    if not enabled:
        return axis
    return AxisDeclaration(
        position=axis.position,
        values=(*axis.values, "distributions"),
        optional=axis.optional,
        description=axis.description,
    )


def _depth_eval_axes(
    *, benchmark: bool = False, distributions: bool = False
) -> dict[str, AxisDeclaration]:
    axes = {
        "space": _DEPTH_SPACE_AXIS,
        "category": _distribution_category(_DEPTH_CATEGORY_AXIS, distributions),
        "reduction": _DEPTH_REDUCTION_AXIS,
    }
    if benchmark:
        axes["bin"] = _BENCHMARK_BIN_AXIS
    return axes


def _sparse_depth_eval_axes(
    *, benchmark: bool = False, distributions: bool = False
) -> dict[str, AxisDeclaration]:
    axes = {
        "space": _DEPTH_SPACE_AXIS,
        "category": _distribution_category(_SPARSE_DEPTH_CATEGORY_AXIS, distributions),
        "reduction": _DEPTH_REDUCTION_AXIS,
    }
    if benchmark:
        axes["bin"] = _BENCHMARK_BIN_AXIS
    return axes


_SKY_SEGMENTATION_CATEGORY = "sky_segmentation"
_SKY_SEGMENTATION_VARIANTS = ("explicit", "implicit")


def _with_sky_segmentation_axes(
    axes: dict[str, AxisDeclaration],
) -> dict[str, AxisDeclaration]:
    """Declare the ``sky_segmentation.{variant}.{reduction}.{metric}`` subtree.

    Sky IoU does not depend on the evaluated depth space, so its keys carry no
    space segment: the space axis becomes optional, ``sky_segmentation`` joins
    the category values, and an optional ``variant`` axis follows the category.
    Existing keys decompose exactly as before. Only used when sky IoU keys are
    written, so other outputs keep their declarations.
    """
    category_position = axes["category"].position
    extended = {}
    for name, axis in axes.items():
        if name == "space":
            axis = replace(axis, optional=True)
        elif name == "category":
            axis = replace(axis, values=(*axis.values, _SKY_SEGMENTATION_CATEGORY))
        if axis.position > category_position:
            axis = replace(axis, position=axis.position + 1)
        extended[name] = axis
    extended["variant"] = AxisDeclaration(
        position=category_position + 1,
        values=_SKY_SEGMENTATION_VARIANTS,
        optional=True,
        description="Sky source: explicit sky_mask or implicit depth threshold",
    )
    return extended


def _rgb_eval_axes(
    *, benchmark: bool = False, domains=None
) -> dict[str, AxisDeclaration]:
    axes = {"category": _rgb_category_axis(domains)}
    if benchmark:
        axes["bin"] = _RGB_BENCHMARK_BIN_AXIS
    return axes


_RAYS_EVAL_AXES: dict[str, AxisDeclaration] = {}

_POINTS_3D_SPACE_AXIS = AxisDeclaration(
    position=0,
    values=("native", "metric"),
    optional=False,
    description="Point-map gauge space semantics",
)

_POINTS_3D_CATEGORY_AXIS = AxisDeclaration(
    position=1,
    values=(
        "point_error",
        "error_decomposition",
        "geometric",
        "cloud_distance",
    ),
    optional=True,
    description="Metric category",
)

_POINTS_3D_REDUCTION_AXIS = AxisDeclaration(
    position=2,
    values=("image_mean", "image_median", "pixel_pool"),
    optional=True,
    description="Dataset reduction mode (point_error only)",
)


def _points_3d_eval_axes(*, distributions: bool = False) -> dict[str, AxisDeclaration]:
    return {
        "space": _POINTS_3D_SPACE_AXIS,
        "category": _distribution_category(_POINTS_3D_CATEGORY_AXIS, distributions),
        "reduction": _POINTS_3D_REDUCTION_AXIS,
    }


_PRED_DEPTH_KEYS = ("depth", "relative_depth", "affine_depth")

#: Prediction modalities an experiment's ``modalities`` may list, and the
#: ``datasets[]`` keys each one covers. GT inputs are not selected this way:
#: they are used as configured, and flags such as ``--mask-sky`` decide
#: which of them an evaluation reads.
_EXPERIMENT_MODALITIES = {
    "depth": _PRED_DEPTH_KEYS,
    "rgb": ("rgb",),
    "rays": ("rays",),
    "points_3d": ("points_3d",),
    "sky_mask": ("sky_mask",),
}
_MODALITY_OF_CONFIG_KEY = {
    key: name for name, keys in _EXPERIMENT_MODALITIES.items() for key in keys
}
_GT_ONLY_KEYS = (
    "sparse_depth",
    *GT_SEGMENTATION_KEYS,
    "calibration",
    "intrinsics",
    "camera_extrinsics",
    "lidar_extrinsics",
)

# Downstream eval consumers validate metricSet.metricNamespace with a stricter
# first-segment rule than euler_metric_naming's modality validator. Keep the
# public Python sparse_depth result keys, but use a wire-compatible metric root
# in eval.json so flattened metric paths live under the declared namespace.
_SPARSE_DEPTH_METRIC_ROOT = "sparsedepth"
_SPARSE_DEPTH_METRIC_NAMESPACE = f"{_SPARSE_DEPTH_METRIC_ROOT}.eval"

# points_3d serializes under a wire-safe root (no underscore) so flattened
# metric paths satisfy the stricter first-segment namespace rule, mirroring
# the sparse-depth handling above.
_POINTS_3D_METRIC_ROOT = "points3d"
_POINTS_3D_METRIC_NAMESPACE = f"{_POINTS_3D_METRIC_ROOT}.eval"

# ── Metric descriptions ─────────────────────────────────────────────────────
# Keys are *base metric names* (after stripping namespace + axes).
# The same description applies across all axis combinations.

_DEPTH_EVAL_DESCRIPTIONS = {
    "psnr": MetricDescription(is_higher_better=True, unit="dB", display_name="PSNR"),
    "ssim": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="SSIM"
    ),
    "lpips": MetricDescription(is_higher_better=False, display_name="LPIPS"),
    "fid": MetricDescription(is_higher_better=False, display_name="FID"),
    "kid_mean": MetricDescription(is_higher_better=False, display_name="KID Mean"),
    "kid_std": MetricDescription(is_higher_better=False, display_name="KID Std"),
    "absrel": MetricDescription(
        is_higher_better=False, display_name="AbsRel"
    ),
    "sqrel": MetricDescription(
        is_higher_better=False, display_name="SqRel"
    ),
    "mae": MetricDescription(
        is_higher_better=False, unit="meters", display_name="MAE"
    ),
    "rmse": MetricDescription(
        is_higher_better=False, unit="meters", display_name="RMSE"
    ),
    "rmse_log": MetricDescription(
        is_higher_better=False, scale="log", display_name="RMSE Log"
    ),
    "log10": MetricDescription(
        is_higher_better=False, scale="log", display_name="Log10 Error"
    ),
    "silog": MetricDescription(
        is_higher_better=False, scale="log", display_name="SILog"
    ),
    "delta1": MetricDescription(
        is_higher_better=True,
        min_value=0.0,
        max_value=1.0,
        display_name="δ < 1.25",
    ),
    "delta2": MetricDescription(
        is_higher_better=True,
        min_value=0.0,
        max_value=1.0,
        display_name="δ < 1.25²",
    ),
    "delta3": MetricDescription(
        is_higher_better=True,
        min_value=0.0,
        max_value=1.0,
        display_name="δ < 1.25³",
    ),
    "absrel.median": MetricDescription(
        is_higher_better=False, display_name="AbsRel (Median)"
    ),
    "absrel.p90": MetricDescription(
        is_higher_better=False, display_name="AbsRel (P90)"
    ),
    "rmse.median": MetricDescription(
        is_higher_better=False, unit="meters", display_name="RMSE (Median)"
    ),
    "rmse.p90": MetricDescription(
        is_higher_better=False, unit="meters", display_name="RMSE (P90)"
    ),
    "silog.mean": MetricDescription(
        is_higher_better=False, scale="log", display_name="SILog (Mean)"
    ),
    "silog.median": MetricDescription(
        is_higher_better=False, scale="log", display_name="SILog (Median)"
    ),
    "silog.p90": MetricDescription(
        is_higher_better=False, scale="log", display_name="SILog (P90)"
    ),
    "normal_consistency.mean_angle": MetricDescription(
        is_higher_better=False, unit="degrees", display_name="Normal Mean Angle"
    ),
    "normal_consistency.median_angle": MetricDescription(
        is_higher_better=False, unit="degrees", display_name="Normal Median Angle"
    ),
    "normal_consistency.percent_below_11_25": MetricDescription(
        is_higher_better=True,
        scale="percentage",
        min_value=0.0,
        max_value=100.0,
        display_name="Normal < 11.25°",
    ),
    "normal_consistency.percent_below_22_5": MetricDescription(
        is_higher_better=True,
        scale="percentage",
        min_value=0.0,
        max_value=100.0,
        display_name="Normal < 22.5°",
    ),
    "normal_consistency.percent_below_30": MetricDescription(
        is_higher_better=True,
        scale="percentage",
        min_value=0.0,
        max_value=100.0,
        display_name="Normal < 30°",
    ),
    "depth_edge_f1.precision": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="Edge Precision"
    ),
    "depth_edge_f1.recall": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="Edge Recall"
    ),
    "depth_edge_f1.f1": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="Edge F1"
    ),
}

_SPARSE_DEPTH_EVAL_DESCRIPTIONS = {
    key: value
    for key, value in _DEPTH_EVAL_DESCRIPTIONS.items()
    if key
    in {
        "absrel",
        "sqrel",
        "mae",
        "rmse",
        "rmse_log",
        "log10",
        "silog",
        "delta1",
        "delta2",
        "delta3",
        "absrel.median",
        "absrel.p90",
        "rmse.median",
        "rmse.p90",
        "silog.mean",
        "silog.median",
        "silog.p90",
    }
}
_SPARSE_DEPTH_EVAL_DESCRIPTIONS["valid_pixel_count"] = MetricDescription(
    unit="pixels",
    display_name="Valid Pixel Count",
)

# Base names below ``sky_segmentation.{variant}.{reduction}`` (--sky-iou).
_SKY_SEGMENTATION_EVAL_DESCRIPTIONS = {
    "iou": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="Sky IoU"
    ),
    "precision": MetricDescription(
        is_higher_better=True,
        min_value=0.0,
        max_value=1.0,
        display_name="Sky Precision",
    ),
    "recall": MetricDescription(
        is_higher_better=True,
        min_value=0.0,
        max_value=1.0,
        display_name="Sky Recall",
    ),
}

_RGB_EVAL_DESCRIPTIONS = {
    "psnr": MetricDescription(is_higher_better=True, unit="dB", display_name="PSNR"),
    "ssim": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="SSIM"
    ),
    "sce": MetricDescription(is_higher_better=False, display_name="SCE"),
    "lpips": MetricDescription(is_higher_better=False, display_name="LPIPS"),
    "fid": MetricDescription(is_higher_better=False, display_name="FID"),
    "precision": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="Edge Precision"
    ),
    "recall": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="Edge Recall"
    ),
    "f1": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="Edge F1"
    ),
    "p95": MetricDescription(is_higher_better=False, display_name="Tail Error P95"),
    "p99": MetricDescription(is_higher_better=False, display_name="Tail Error P99"),
    "relative_diff": MetricDescription(
        is_higher_better=True, display_name="HF Relative Diff"
    ),
    "mae": MetricDescription(is_higher_better=False, display_name="MAE"),
    "mse": MetricDescription(is_higher_better=False, display_name="MSE"),
}
for _metric_set in DOMAIN_METRIC_SETS.values():
    for _metric in _metric_set.rgb_no_reference:
        _RGB_EVAL_DESCRIPTIONS[_metric.name] = MetricDescription(
            is_higher_better=_metric.is_higher_better,
            display_name=_metric.display_name,
        )

_RAYS_EVAL_DESCRIPTIONS = {
    "rho_a.mean": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="ρ_A (Mean)"
    ),
    "rho_a.median": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="ρ_A (Median)"
    ),
    "angular_error.mean_angle": MetricDescription(
        is_higher_better=False, unit="degrees", display_name="Mean Angular Error"
    ),
    "angular_error.median_angle": MetricDescription(
        is_higher_better=False, unit="degrees", display_name="Median Angular Error"
    ),
    "angular_error.percent_below_5": MetricDescription(
        is_higher_better=True,
        scale="percentage",
        min_value=0.0,
        max_value=100.0,
        display_name="< 5°",
    ),
    "angular_error.percent_below_10": MetricDescription(
        is_higher_better=True,
        scale="percentage",
        min_value=0.0,
        max_value=100.0,
        display_name="< 10°",
    ),
    "angular_error.percent_below_15": MetricDescription(
        is_higher_better=True,
        scale="percentage",
        min_value=0.0,
        max_value=100.0,
        display_name="< 15°",
    ),
    "angular_error.percent_below_20": MetricDescription(
        is_higher_better=True,
        scale="percentage",
        min_value=0.0,
        max_value=100.0,
        display_name="< 20°",
    ),
    "angular_error.percent_below_30": MetricDescription(
        is_higher_better=True,
        scale="percentage",
        min_value=0.0,
        max_value=100.0,
        display_name="< 30°",
    ),
}


def _points_3d_descriptions() -> dict:
    """Build the points_3d metric descriptions (base names after axes)."""
    descriptions = {
        # point_error
        "mae3d": MetricDescription(
            is_higher_better=False, unit="meters", display_name="3D EPE (MAE)"
        ),
        "rmse3d": MetricDescription(
            is_higher_better=False, unit="meters", display_name="3D RMSE"
        ),
        "median3d": MetricDescription(
            is_higher_better=False, unit="meters", display_name="3D Median Error"
        ),
        "p90": MetricDescription(
            is_higher_better=False, unit="meters", display_name="3D Error P90"
        ),
        "p95": MetricDescription(
            is_higher_better=False, unit="meters", display_name="3D Error P95"
        ),
        "rel_median": MetricDescription(
            is_higher_better=False, display_name="Relative 3D Error (Median)"
        ),
        "rel_p90": MetricDescription(
            is_higher_better=False, display_name="Relative 3D Error (P90)"
        ),
        # error_decomposition
        "radial_mae": MetricDescription(
            is_higher_better=False, unit="meters", display_name="Radial MAE (depth)"
        ),
        "radial_rmse": MetricDescription(
            is_higher_better=False, unit="meters", display_name="Radial RMSE (depth)"
        ),
        "lateral_mae": MetricDescription(
            is_higher_better=False, unit="meters", display_name="Lateral MAE (camera)"
        ),
        "lateral_rmse": MetricDescription(
            is_higher_better=False, unit="meters", display_name="Lateral RMSE (camera)"
        ),
        "lateral_fraction": MetricDescription(
            is_higher_better=False,
            min_value=0.0,
            max_value=1.0,
            display_name="Lateral Error Fraction",
        ),
        "angular_error.mean_angle": MetricDescription(
            is_higher_better=False, unit="degrees", display_name="Mean Ray Angle Error"
        ),
        "angular_error.median_angle": MetricDescription(
            is_higher_better=False, unit="degrees",
            display_name="Median Ray Angle Error",
        ),
        "rho_a.mean": MetricDescription(
            is_higher_better=True, min_value=0.0, max_value=1.0,
            display_name="ρ_A (Mean)",
        ),
        "rho_a.median": MetricDescription(
            is_higher_better=True, min_value=0.0, max_value=1.0,
            display_name="ρ_A (Median)",
        ),
        # geometric
        "normal_consistency.mean_angle": MetricDescription(
            is_higher_better=False, unit="degrees",
            display_name="Normal Mean Angle",
        ),
        "normal_consistency.median_angle": MetricDescription(
            is_higher_better=False, unit="degrees",
            display_name="Normal Median Angle",
        ),
        # geometric + cloud F-score share precision/recall/f1 leaf names
        "precision": MetricDescription(
            is_higher_better=True, min_value=0.0, max_value=1.0,
            display_name="Precision",
        ),
        "recall": MetricDescription(
            is_higher_better=True, min_value=0.0, max_value=1.0,
            display_name="Recall",
        ),
        "f1": MetricDescription(
            is_higher_better=True, min_value=0.0, max_value=1.0, display_name="F1"
        ),
        "f_a": MetricDescription(
            is_higher_better=True,
            scale="percentage",
            min_value=0.0,
            max_value=100.0,
            display_name="F-score AUC (F_A)",
        ),
        # cloud_distance
        "chamfer.accuracy": MetricDescription(
            is_higher_better=False, unit="meters", display_name="Chamfer Accuracy"
        ),
        "chamfer.completeness": MetricDescription(
            is_higher_better=False, unit="meters", display_name="Chamfer Completeness"
        ),
        "chamfer.distance": MetricDescription(
            is_higher_better=False, unit="meters", display_name="Chamfer Distance"
        ),
        "chamfer.median": MetricDescription(
            is_higher_better=False, unit="meters", display_name="Chamfer Median"
        ),
    }
    # δ-style absolute and relative accuracy thresholds (percentages).
    for tau in (0.05, 0.1, 0.25, 0.5, 1.0):
        key = "acc_" + ("%g" % tau).replace(".", "_")
        descriptions[key] = MetricDescription(
            is_higher_better=True, scale="percentage", min_value=0.0,
            max_value=100.0, display_name=f"3D Acc < {tau}m",
        )
    for tau in (0.05, 0.1, 0.25):
        key = "acc_rel_" + ("%g" % tau).replace(".", "_")
        descriptions[key] = MetricDescription(
            is_higher_better=True, scale="percentage", min_value=0.0,
            max_value=100.0, display_name=f"3D Rel Acc < {tau}",
        )
    for thr in (5, 10, 15, 20, 30):
        descriptions[f"angular_error.percent_below_{thr}"] = MetricDescription(
            is_higher_better=True, scale="percentage", min_value=0.0,
            max_value=100.0, display_name=f"Ray < {thr}°",
        )
    for thr in ("11_25", "22_5", "30"):
        descriptions[f"normal_consistency.percent_below_{thr}"] = MetricDescription(
            is_higher_better=True, scale="percentage", min_value=0.0,
            max_value=100.0, display_name=f"Normal < {thr.replace('_', '.')}°",
        )
    return descriptions


_POINTS_3D_EVAL_DESCRIPTIONS = _points_3d_descriptions()


def _distribution_descriptions(
    base: dict, config: DistributionConfig | None, metric: str
) -> dict:
    if config is None:
        return base
    return {
        **base,
        f"{metric}.distribution": MetricDescription(
            unit="pixels", min_value=0.0,
            display_name=f"{metric.upper()} error magnitude histogram",
        ),
    }


def _distribution_metadata(results: dict) -> dict:
    info = results.get("distribution_info")
    if info is None:
        return {}
    metadata = {"distributions": info}
    if "distribution_population" in results:
        metadata["distributionPopulation"] = results["distribution_population"]
    return metadata


def _resolve_distribution_config(value, args) -> DistributionConfig | None:
    settings = dict(value) if isinstance(value, dict) else {}
    overrides = {}
    if args.distribution_bins is not None:
        overrides["n_bins"] = args.distribution_bins
    if args.distribution_scale is not None:
        overrides["scale"] = args.distribution_scale
    if args.distribution_range is not None:
        overrides["min_error"], overrides["max_error"] = args.distribution_range
    enabled = args.distributions
    if enabled is None:
        enabled = value is True or isinstance(value, dict) or bool(overrides)
    settings.update(overrides)
    return DistributionConfig.from_config(settings if enabled else False)


def _get_version() -> str:
    """Return the installed euler-eval version, falling back to ``"0.0.0"``."""
    try:
        return importlib.metadata.version("euler-eval")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _package_metadata() -> dict:
    """Return stable package/runtime metadata for evaluation provenance."""
    return {
        "name": "euler-eval",
        "version": _get_version(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }


def _log_package_metadata() -> dict:
    """Emit and return the package metadata recorded at workflow startup."""
    metadata = _package_metadata()
    print(f"Evaluation package: {json.dumps(metadata, sort_keys=True)}")
    return metadata


def _sparse_depth_metric_set_envelope(
    namespace: _EvalNamespace,
    *,
    metadata: dict | None = None,
) -> dict:
    """Return the sparse-depth metricSet envelope with a compliant namespace."""
    envelope = namespace.metric_set_envelope("sparse_depth", metadata=metadata)
    envelope["metricNamespace"] = _SPARSE_DEPTH_METRIC_NAMESPACE
    return envelope


def _points_3d_metric_set_envelope(
    namespace: _EvalNamespace,
    *,
    metadata: dict | None = None,
) -> dict:
    """Return the points_3d metricSet envelope with a compliant namespace."""
    envelope = namespace.metric_set_envelope("points_3d", metadata=metadata)
    envelope["metricNamespace"] = _POINTS_3D_METRIC_NAMESPACE
    return envelope


def _clean_metric_tree(
    tree: dict,
    *,
    preserve_empty_dict_keys: set[str] | None = None,
) -> dict:
    """Recursively sanitize a metric dict for JSON schema compliance.

    - Removes entries where the value is ``None``
    - Removes entries where the value is a non-finite float (NaN, Inf)
    - Recursively processes nested dicts and list items
    - Prunes empty dicts after cleaning
    """
    preserve_empty_dict_keys = preserve_empty_dict_keys or set()
    cleaned = {}
    for key, value in tree.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, dict):
            sub = _clean_metric_tree(
                value,
                preserve_empty_dict_keys=preserve_empty_dict_keys,
            )
            if sub or key in preserve_empty_dict_keys:
                cleaned[key] = sub
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_metric_tree(
                    item,
                    preserve_empty_dict_keys=preserve_empty_dict_keys,
                )
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _empty_per_file_metric_ids(
    node: dict,
    hierarchy: tuple[str, ...] = (),
) -> list[str]:
    """Return file identifiers whose cleaned per-file metrics are empty."""
    ids = []
    for file_entry in node.get("files", []):
        if not isinstance(file_entry, dict):
            continue
        if file_entry.get("metrics") == {}:
            file_id = str(file_entry.get("id", "<missing-id>"))
            ids.append("/".join((*hierarchy, file_id)))

    children = node.get("children", {})
    if isinstance(children, dict):
        for child_name, child_node in children.items():
            if isinstance(child_node, dict):
                ids.extend(
                    _empty_per_file_metric_ids(
                        child_node,
                        (*hierarchy, str(child_name)),
                    )
                )
    return ids


def _clean_per_file_metrics(tree: dict, *, label: str = "per_file_metrics") -> dict:
    """Clean a per-file metric tree while preserving file-entry shape.

    Some per-file metric entries can legitimately clean down to no scalar
    metrics, for example when every metric for a file is ``None``/NaN after a
    failed computation or an invalid sample.  The generic cleaner prunes empty
    dicts, but per-file entries are schema-shaped as ``{"id", "metrics"}``;
    dropping the empty ``metrics`` object leaves an id-only file entry.
    """
    cleaned = _clean_metric_tree(tree, preserve_empty_dict_keys={"metrics"})
    empty_ids = _empty_per_file_metric_ids(cleaned)
    if empty_ids:
        sample = ", ".join(empty_ids[:10])
        suffix = "" if len(empty_ids) <= 10 else f", ... +{len(empty_ids) - 10} more"
        print(
            f"Warning: {label} has {len(empty_ids)} file entries with no finite "
            f"metrics after cleaning: {sample}{suffix}"
        )
    return cleaned


def _wrap_depth_space_pfm_metrics(
    metrics: dict,
    *,
    metric_root: str,
    canonical_key: str,
    canonical_space: str | None = "metric",
) -> dict:
    """Wrap per-file depth-like metrics under the serialized namespace path."""
    space_metrics = {
        space: metrics[f"{canonical_key}_{space}"]
        for space in ("native", "metric")
        if f"{canonical_key}_{space}" in metrics
    }
    if not space_metrics and canonical_key in metrics:
        space_metrics[canonical_space or "metric"] = metrics[canonical_key]
    if _SKY_SEGMENTATION_CATEGORY in metrics:
        space_metrics[_SKY_SEGMENTATION_CATEGORY] = metrics[_SKY_SEGMENTATION_CATEGORY]
    return {metric_root: {"eval": space_metrics}}


def _wrap_pfm_metrics(pfm: dict, wrapper_fn) -> dict:
    """Walk a ``perFileHierarchyNode`` tree, applying *wrapper_fn* to each
    file entry's ``metrics`` dict.  This restructures per-file metric keys
    to match the declared namespace path without changing ``evaluate.py``.
    """
    result = {}
    if "children" in pfm:
        result["children"] = {
            k: _wrap_pfm_metrics(v, wrapper_fn)
            for k, v in pfm["children"].items()
        }
    if "files" in pfm:
        result["files"] = [
            {"id": f["id"], "metrics": wrapper_fn(f["metrics"])}
            for f in pfm["files"]
        ]
    return result


def resolve_device(device: str) -> str:
    """Resolve runtime device with graceful CUDA fallback."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: --device cuda requested but CUDA is unavailable. "
            "Falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu"
    return device


def configure_torch_runtime(device: str) -> None:
    """Enable CUDA performance knobs when running on GPU."""
    if device != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def print_device_info(requested_device: str, resolved_device: str) -> None:
    """Print effective runtime device info."""
    if requested_device == resolved_device:
        print(f"Device: {resolved_device}")
    else:
        print(f"Device: {requested_device} -> {resolved_device}")

    if resolved_device == "cuda":
        try:
            idx = torch.cuda.current_device()
            print(f"GPU: {torch.cuda.get_device_name(idx)}")
        except Exception:
            pass


def _prediction_depth_entry(
    entry: dict, *, label: str = "Prediction dataset"
) -> tuple[str | None, dict | None]:
    """Return the configured dense-depth prediction entry, if any."""
    matches = [
        key
        for key in _PRED_DEPTH_KEYS
        if key in entry and "path" in entry.get(key, {})
    ]
    if not matches:
        return None, None
    if len(matches) > 1:
        raise ValueError(
            f"{label} contains multiple dense depth entries "
            f"{matches}. Use only one of {list(_PRED_DEPTH_KEYS)}."
        )
    key = matches[0]
    return key, entry[key]


def _prediction_depth_space_hint(key: str | None) -> str | None:
    if key == "relative_depth":
        return "relative"
    if key == "affine_depth":
        return "affine"
    return None


def _gt_segmentation_entry(gt: dict) -> tuple[str | None, dict | None]:
    """Return the configured GT segmentation ``(key, entry)``, if any.

    ``gt.semantic_segmentation`` is accepted as an alias of the canonical
    ``gt.segmentation``; configuring both is an error.
    """
    present = [key for key in GT_SEGMENTATION_KEYS if key in gt]
    if len(present) > 1:
        raise ValueError(
            "gt.segmentation and gt.semantic_segmentation are aliases for the "
            "same GT segmentation; configure only one (gt.segmentation is "
            "canonical)"
        )
    for key in present:
        if "path" in gt.get(key, {}):
            return key, gt[key]
    return None, None


def validate_gt_config(gt: dict) -> None:
    """Validate the ``gt`` section of the configuration.

    Raises:
        ValueError: If required fields are missing or paths do not exist.
    """
    has_rgb = "rgb" in gt and "path" in gt.get("rgb", {})
    has_depth = "depth" in gt and "path" in gt.get("depth", {})
    has_sparse_depth = "sparse_depth" in gt and "path" in gt.get("sparse_depth", {})
    has_rays = "rays" in gt and "path" in gt.get("rays", {})
    has_points_3d = "points_3d" in gt and "path" in gt.get("points_3d", {})
    has_intrinsics = "intrinsics" in gt and "path" in gt.get("intrinsics", {})
    has_camera_extrinsics = (
        "camera_extrinsics" in gt and "path" in gt.get("camera_extrinsics", {})
    )

    if (
        not has_rgb
        and not has_depth
        and not has_sparse_depth
        and not has_rays
        and not has_points_3d
    ):
        raise ValueError(
            "gt must have at least one of 'rgb.path', 'depth.path', "
            "'sparse_depth.path', 'rays.path', or 'points_3d.path'"
        )

    if has_sparse_depth and (not has_intrinsics or not has_camera_extrinsics):
        raise ValueError(
            "gt.sparse_depth requires gt.intrinsics.path and "
            "gt.camera_extrinsics.path for pointcloud projection"
        )

    _gt_segmentation_entry(gt)

    for modality in (
        "rgb",
        "depth",
        "sparse_depth",
        "rays",
        "points_3d",
        *GT_SEGMENTATION_KEYS,
        "calibration",
        "intrinsics",
        "camera_extrinsics",
        "lidar_extrinsics",
    ):
        if modality in gt and "path" in gt[modality]:
            p = normalize_modality_path(
                gt[modality]["path"],
                split=gt[modality].get("split"),
            )
            if not p.exists():
                raise ValueError(f"gt.{modality}.path does not exist: {p}")


def validate_dataset_entry(entry: dict, index: int) -> None:
    """Validate a single prediction dataset entry.

    Raises:
        ValueError: If the entry is malformed.
    """
    label = f"datasets[{index}]"
    if "name" not in entry:
        raise ValueError(f"{label} must have a 'name' field")

    has_rgb = "rgb" in entry and "path" in entry.get("rgb", {})
    depth_key, _ = _prediction_depth_entry(entry, label=label)
    has_depth = depth_key is not None
    has_rays = "rays" in entry and "path" in entry.get("rays", {})
    has_points_3d = "points_3d" in entry and "path" in entry.get("points_3d", {})
    if not has_rgb and not has_depth and not has_rays and not has_points_3d:
        raise ValueError(
            f"{label} must have at least 'rgb.path', 'depth.path', "
            "'relative_depth.path', 'affine_depth.path', 'rays.path', "
            "or 'points_3d.path'"
        )

    if "sky_mask" in entry:
        if not isinstance(entry["sky_mask"], dict) or not entry["sky_mask"].get("path"):
            raise ValueError(f"{label}.sky_mask must have a 'path' field")
        if not has_depth:
            raise ValueError(
                f"{label}.sky_mask requires a dense depth prediction "
                "('depth', 'relative_depth', or 'affine_depth')"
            )

    for modality in ("rgb", *_PRED_DEPTH_KEYS, "rays", "points_3d", "sky_mask"):
        if modality in entry and "path" in entry[modality]:
            p = normalize_modality_path(
                entry[modality]["path"],
                split=entry[modality].get("split"),
            )
            if not p.exists():
                raise ValueError(f"{label}.{modality}.path does not exist: {p}")


def validate_euler_train_config(et_config: dict) -> None:
    """Validate the optional ``euler_train`` config section.

    Raises:
        ValueError: If required fields are missing or euler_train is not installed.
    """
    if "dir" not in et_config:
        raise ValueError(
            "euler_train.dir is required when euler_train logging is enabled"
        )
    if _euler_train is None:
        raise ValueError(
            "euler_train logging is configured but the 'euler-train' package is not "
            "installed. Install it with: pip install euler-train"
        )
    unknown = set(et_config) - {"dir"}
    if unknown:
        raise ValueError(
            f"Unknown euler_train config keys: {unknown}. "
            f"Only 'dir' is supported — pass a project directory for a new run "
            f"or a run directory (containing meta.json) to resume."
        )


def validate_experiments_config(experiments) -> None:
    """Validate the optional ``experiments`` config section.

    Each experiment is ``{"name": ..., "args": [...]}``: a unique evaluation
    name and the command-line flags it adds. Flags are checked later, by the
    argument parser.

    Raises:
        ValueError: Naming the offending experiment.
    """
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(
            "'experiments' must be a non-empty array of "
            '{"name": ..., "args": [...]} objects'
        )
    seen = set()
    for index, entry in enumerate(experiments):
        label = f"experiments[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be an object with 'name' and 'args'")
        unknown = sorted(set(entry) - {"name", "args", "modalities"})
        if unknown:
            raise ValueError(
                f"{label} has unknown keys {unknown}; only 'name', 'args' and "
                "'modalities' are supported"
            )
        if not isinstance(entry.get("name"), str):
            raise ValueError(f"{label}.name must be a string")
        try:
            name = validate_eval_name(entry["name"])
        except ValueError as e:
            raise ValueError(f"{label}.name: {e}") from e
        if name in seen:
            raise ValueError(
                f"{label}.name {name!r} is already used; experiment names must "
                "be unique"
            )
        seen.add(name)
        args = entry.get("args", [])
        if not isinstance(args, list) or not all(
            isinstance(value, (str, int, float)) and not isinstance(value, bool)
            for value in args
        ):
            raise ValueError(
                f"{label}.args must be an array of command-line tokens (strings "
                'or numbers), e.g. ["--mask-sky", "--benchmark-depth-range", 1, 80]'
            )
        if "modalities" in entry:
            _validate_experiment_modalities(entry["modalities"], label)


def _validate_experiment_modalities(modalities, label: str) -> None:
    if (
        not isinstance(modalities, list)
        or not modalities
        or not all(isinstance(name, str) for name in modalities)
    ):
        raise ValueError(
            f"{label}.modalities must be a non-empty array of modality names, "
            'e.g. ["depth", "sky_mask"]'
        )
    for name in modalities:
        if name in _GT_ONLY_KEYS:
            raise ValueError(
                f"{label}.modalities: {name!r} is a GT input; 'modalities' selects "
                "prediction modalities, and GT inputs are used as configured "
                "(flags such as --mask-sky decide which ones are read)"
            )
        if name not in _EXPERIMENT_MODALITIES:
            covered_by = _MODALITY_OF_CONFIG_KEY.get(name)
            hint = f" (use {covered_by!r}, which covers it)" if covered_by else ""
            raise ValueError(
                f"{label}.modalities: unknown modality {name!r}{hint}; expected "
                f"one of {', '.join(_EXPERIMENT_MODALITIES)}"
            )
    duplicates = sorted({name for name in modalities if modalities.count(name) > 1})
    if duplicates:
        raise ValueError(f"{label}.modalities lists {duplicates} more than once")


def _has_prediction(entry: dict) -> bool:
    return _prediction_depth_entry(entry)[0] is not None or any(
        key in entry and "path" in entry[key] for key in ("rgb", "rays", "points_3d")
    )


def _configured_modalities(config: dict) -> set[str]:
    return {
        _MODALITY_OF_CONFIG_KEY[key]
        for entry in config["datasets"]
        for key in entry
        if key in _MODALITY_OF_CONFIG_KEY
    }


def _experiment_config(config: dict, modalities: list[str] | None) -> dict:
    """Return *config* with only the prediction modalities an experiment lists.

    Unlisted ``datasets[]`` entries are left out, as if they were not
    configured; the ``gt`` section is unchanged. A dataset left without a
    prediction is not evaluated, and a ``sky_mask`` is dropped with the depth
    prediction it belongs to.
    """
    if modalities is None:
        return config
    keep = set(modalities)

    def reduced(section: dict) -> dict:
        return {
            key: value
            for key, value in section.items()
            if key not in _MODALITY_OF_CONFIG_KEY
            or _MODALITY_OF_CONFIG_KEY[key] in keep
        }

    datasets = []
    for entry in config["datasets"]:
        entry = reduced(entry)
        if _prediction_depth_entry(entry)[0] is None:
            entry.pop("sky_mask", None)
        if _has_prediction(entry):
            datasets.append(entry)
    return {**config, "datasets": datasets}


def load_config(config_path: str) -> dict:
    """Load and validate configuration from JSON file.

    Args:
        config_path: Path to config.json file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ValueError: If configuration is invalid.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    if "gt" not in config:
        raise ValueError("Config must contain a 'gt' section")
    if "datasets" not in config or not config["datasets"]:
        raise ValueError("Config must contain a non-empty 'datasets' array")

    validate_gt_config(config["gt"])
    for i, entry in enumerate(config["datasets"]):
        validate_dataset_entry(entry, i)

    if "euler_train" in config:
        validate_euler_train_config(config["euler_train"])

    if "experiments" in config:
        validate_experiments_config(config["experiments"])

    DistributionConfig.from_config(config.get("distributions"))

    return config


def _find_zip_ancestor(path: Path):
    """Find a ``.zip`` file among the ancestors of *path*.

    Returns:
        ``(zip_path, internal_name)`` when a zip archive is detected in the
        path hierarchy, or ``(None, None)`` for plain filesystem paths.
    """
    parts = path.parts
    for i in range(1, len(parts) + 1):
        candidate = Path(*parts[:i])
        if candidate.suffix.lower() == ".zip" and candidate.is_file():
            remainder = "/".join(parts[i:])
            return candidate, remainder
    return None, None


def _save_json_to_zip(zip_path: Path, internal_name: str, data: dict) -> None:
    """Write JSON *data* into an existing zip archive at *internal_name*.

    If an entry with the same name already exists it is replaced.
    """
    json_bytes = json.dumps(data, indent=2).encode("utf-8")

    with zipfile.ZipFile(zip_path, "r") as zf:
        needs_replace = internal_name in zf.namelist()

    if needs_replace:
        tmp = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(zip_path, "r") as zf_in:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
                for item in zf_in.infolist():
                    if item.filename != internal_name:
                        zf_out.writestr(item, zf_in.read(item.filename))
                zf_out.writestr(internal_name, json_bytes)
        tmp.replace(zip_path)
    else:
        with zipfile.ZipFile(zip_path, "a") as zf:
            zf.writestr(internal_name, json_bytes)


EVAL_DIR = ".eval"
DEFAULT_EVAL_NAME = "eval"
_EVAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_eval_name(value: str) -> str:
    """Validate an evaluation name and return it without a ``.json`` suffix.

    The name becomes the results file name (``<name>.json``) below each
    prediction dataset's ``.eval/`` folder, so it is one portable path
    segment, like a ds-crawler split or scope name.

    Raises:
        ValueError: When *value* is not a plain name.
    """
    name = value.strip()
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if not _EVAL_NAME_PATTERN.match(name):
        raise ValueError(
            f"{value!r} is not a valid evaluation name: use letters, digits, "
            "'.', '-' or '_', starting with a letter or digit (e.g. 'no_sky'); "
            "'.json' is appended."
        )
    return name


def default_eval_path(
    modality_path: str | Path,
    *,
    modality_key: str,
    split: str | None = None,
    eval_name: str = DEFAULT_EVAL_NAME,
) -> Path:
    """Return where a prediction modality's results land by default.

    Results go to a dataset-level ``.eval/`` folder beside ``.ds_crawler/``:
    ``.eval/<scope>/`` when the modality's metadata scope (the ``#scope=``
    selector, else the config key) has its own ``.ds_crawler/<scope>/``
    metadata, otherwise ``.eval/`` itself, matching how euler-loading falls
    back to root-level metadata. A selected split adds ``split_<split>/``.
    """
    root, scope, split = resolve_dataset_location(
        modality_path, modality_key=modality_key, split=split
    )
    eval_dir = root / EVAL_DIR
    if scope is not None:
        eval_dir = eval_dir / scope
    if split is not None:
        eval_dir = eval_dir / f"split_{split}"
    return eval_dir / f"{eval_name}.json"


def _zip_internal_name(zip_path: Path, internal_name: str) -> str:
    """Place *internal_name* inside a zip's wrapper folder, if it has one.

    ds-crawler treats a single top-level folder named after the archive
    (``model.zip`` -> ``model/``) as the dataset root. Writing beside that
    folder would end the single-root layout and hide ``.ds_crawler/``.
    """
    prefix = get_zip_root_prefix(zip_path)
    if (
        prefix
        and prefix.rstrip("/") == zip_path.stem
        and not internal_name.startswith(prefix)
    ):
        return prefix + internal_name
    return internal_name


def save_results(
    results: dict,
    dataset_config: dict,
    modality: str | None = None,
    *,
    eval_name: str = DEFAULT_EVAL_NAME,
) -> Path:
    """Save results to output file.

    Handles both plain directories and zip archives.  When the resolved
    output path passes through a ``.zip`` file the results are written
    into the archive instead of onto the filesystem.

    Args:
        results: Results dictionary.
        dataset_config: Dataset configuration entry.
        modality: When set, save to this specific modality's path
            (e.g. ``"depth"`` or ``"rgb"``).  Falls back to the first
            available modality path when *None*.
        eval_name: Evaluation name (``--eval-name``); results are written
            as ``<eval_name>.json`` in the dataset's ``.eval/`` layout, see
            :func:`default_eval_path`.  A dataset's explicit ``output_file``
            takes precedence over the whole layout.

    Returns:
        Path where results were saved.
    """
    output_file = dataset_config.get("output_file")
    if output_file is None:
        selected_modality = modality
        if selected_modality == "depth" and "depth" not in dataset_config:
            selected_modality, _ = _prediction_depth_entry(dataset_config)
        if not (
            selected_modality is not None
            and selected_modality in dataset_config
            and "path" in dataset_config[selected_modality]
        ):
            # Default: save alongside first available modality path
            selected_modality = next(
                (
                    mod
                    for mod in (*_PRED_DEPTH_KEYS, "rgb", "rays")
                    if mod in dataset_config and "path" in dataset_config[mod]
                ),
                None,
            )
        if selected_modality is not None:
            entry = dataset_config[selected_modality]
            output_file = default_eval_path(
                entry["path"],
                modality_key=selected_modality,
                split=entry.get("split"),
                eval_name=eval_name,
            )
        else:
            output_file = Path(f"{eval_name}.json")
    else:
        output_file = Path(output_file)

    zip_path, internal_name = _find_zip_ancestor(output_file)
    if zip_path is not None and internal_name:
        _save_json_to_zip(
            zip_path, _zip_internal_name(zip_path, internal_name), results
        )
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return output_file


def print_results(results: dict, title: str) -> None:
    """Print results summary."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    def print_dict(d: dict, indent: int = 0) -> None:
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, float):
                print(f"{prefix}{key}: {value:.6f}")
            else:
                print(f"{prefix}{key}: {value}")

    print_dict(results)


# ID-paired optional inputs: dataset modality -> (builder kwargs, label).
_PAIRED_OPTIONAL_INPUTS = {
    "segmentation": (("segmentation_path", "segmentation_split"), "GT segmentation"),
    "pred_sky_mask": (("pred_sky_mask_path", "pred_sky_mask_split"), "sky_mask"),
}


def _build_eval_dataset(
    build,
    kwargs: dict,
    *,
    ds_name: str,
    iou_only: tuple[str, ...] = (),
    missing_segmentation: str = "drop",
) -> tuple:
    """Build an eval dataset and account for the frames its optional inputs drop.

    GT segmentation and a predicted ``sky_mask`` are paired with the evaluated
    frames by ID, so a frame without one is not evaluated; euler-loading only
    logs that, so it is reported here. An input named in *iou_only* is read
    only for ``--sky-iou``, which must not change the evaluated pairs: if it
    lacks any, the dataset is rebuilt without it.

    With *missing_segmentation* ``"unmasked"``, frames without their own GT
    segmentation are kept instead: they are read without it, so ``--mask-sky``
    leaves them unmasked and ``--sky-iou`` leaves them out of the IoU.

    Returns ``(dataset, skipped)`` with ``{input: reason}`` for every
    *iou_only* input that was left out.
    """
    optional = tuple(_PAIRED_OPTIONAL_INPUTS)
    dataset = build(**kwargs)
    partial_segmentation = (
        missing_segmentation == "unmasked"
        and kwargs.get("segmentation_path") is not None
        and bool(count_ids_dropped_by(dataset, "segmentation", optional=optional))
    )
    skipped = {}
    for name in iou_only:
        if name == "segmentation" and partial_segmentation:
            continue
        dropped = count_ids_dropped_by(dataset, name, optional=optional)
        if not dropped:
            continue
        keys, label = _PAIRED_OPTIONAL_INPUTS[name]
        print(
            f"Warning: --sky-iou: the {label} of '{ds_name}' lacks {dropped} "
            "evaluated pairs. It is read only for the sky IoU, which must not "
            "change the evaluated pairs, so it is not loaded."
        )
        kwargs = {**kwargs, **dict.fromkeys(keys)}
        skipped[name] = f"{label} does not cover every evaluated pair"
    if skipped:
        dataset = build(**kwargs)
    for name, (_, label) in _PAIRED_OPTIONAL_INPUTS.items():
        dropped = count_ids_dropped_by(dataset, name, optional=optional)
        if not dropped:
            continue
        if name == "segmentation" and partial_segmentation:
            print(
                f"Note: the {label} of '{ds_name}' lacks {dropped} pairs that the "
                "other inputs share; with --missing-segmentation unmasked they "
                "are evaluated without GT sky masking and left out of the sky IoU."
            )
        else:
            print(
                f"Warning: the {label} of '{ds_name}' lacks {dropped} pairs that "
                "the other inputs share; those pairs are not evaluated."
            )
    if partial_segmentation:
        keys, _ = _PAIRED_OPTIONAL_INPUTS["segmentation"]
        complete = build(**{**kwargs, **dict.fromkeys(keys)})
        dataset = PartialInputDataset(complete, dataset)
    return dataset, skipped


def _sky_segmentation_tree(results: dict) -> dict:
    """Return the cleaned ``sky_segmentation`` metric tree ({} when absent)."""
    return _clean_metric_tree(results.get(_SKY_SEGMENTATION_CATEGORY) or {})


def _sky_segmentation_meta(
    results: dict, gt_key: str | None, skipped_inputs: dict
) -> dict | None:
    """Return ``meta.sky_segmentation`` provenance, or ``None`` without --sky-iou.

    *skipped_inputs* names inputs left out by :func:`_build_eval_dataset`;
    they explain the variants that could not run.
    """
    info = results.get("sky_segmentation_info")
    if info is None:
        return None
    variants = {variant: dict(info[variant]) for variant in _SKY_SEGMENTATION_VARIANTS}
    for name, reason in skipped_inputs.items():
        affected = ("explicit",) if name == "pred_sky_mask" else variants
        for variant in affected:
            if not variants[variant]["available"]:
                variants[variant]["reason"] = reason
    rest = {k: v for k, v in info.items() if k not in _SKY_SEGMENTATION_VARIANTS}
    return {**variants, "gt_source": f"gt.{gt_key}", **rest}


def _sky_iou_summary_line(ds_name: str, results: dict) -> str | None:
    """Return the one-line pooled sky IoU summary of a prediction dataset."""
    info = results.get("sky_segmentation_info")
    if info is None:
        return None

    def _fmt(value) -> str:
        return f"{value:.4f}" if value is not None and math.isfinite(value) else "n/a"

    parts = []
    for variant in _SKY_SEGMENTATION_VARIANTS:
        pooled = (results.get(_SKY_SEGMENTATION_CATEGORY) or {}).get(variant)
        if pooled is None:
            continue
        pooled = pooled["pixel_pool"]
        label = (
            f"implicit (depth >= {info['threshold_m']:g} m)"
            if variant == "implicit"
            else variant
        )
        parts.append(
            f"{label} {_fmt(pooled['iou'])} (precision {_fmt(pooled['precision'])}, "
            f"recall {_fmt(pooled['recall'])})"
        )
    return f"Sky IoU, pixel_pool, '{ds_name}': " + (
        "; ".join(parts) if parts else "no variant computed"
    )


class _ArgumentParser(argparse.ArgumentParser):
    """An argument parser that can raise instead of exiting.

    Experiment ``args`` from the config are parsed with the same parser as the
    command line; their errors are reported as config errors.
    """

    raise_errors = False

    def error(self, message):
        if self.raise_errors:
            raise ValueError(message)
        super().error(message)


class _Run(NamedTuple):
    """One evaluation pass: parsed arguments plus what they resolve to."""

    name: str | None
    args: argparse.Namespace
    config: dict
    distribution_config: DistributionConfig | None
    metric_sets: tuple
    extra_args: tuple = ()


def _finish_args(args: argparse.Namespace) -> None:
    """Validate and normalize parsed arguments in place.

    Raises:
        ValueError: Prefixed with the offending flag.
    """
    try:
        validate_sky_depth(args.sky_depth)
    except ValueError as e:
        raise ValueError(f"--sky-depth: {e}") from e
    try:
        args.eval_name = validate_eval_name(args.eval_name)
    except ValueError as e:
        raise ValueError(f"--eval-name: {e}") from e


def _prepare_run(
    args: argparse.Namespace,
    config: dict,
    *,
    name: str | None = None,
    extra_args: tuple = (),
) -> _Run:
    """Resolve the settings of one evaluation pass, before any work starts."""
    return _Run(
        name=name,
        args=args,
        config=config,
        distribution_config=_resolve_distribution_config(
            config.get("distributions"), args
        ),
        metric_sets=tuple(resolve_metric_sets(args.domains)),
        extra_args=extra_args,
    )


def _experiment_runs(
    parser: _ArgumentParser,
    argv: list[str],
    config: dict,
    *,
    selected: list[str] | None,
) -> list[_Run]:
    """Parse every selected experiment as ``argv`` followed by its ``args``.

    Experiments run in config order. Every one is parsed and validated here,
    so a mistake in a later experiment stops the run before any work.

    Raises:
        ValueError: Naming the experiment and the problem.
    """
    base = parser.parse_args(argv)
    runs = []
    for index, entry in enumerate(config["experiments"]):
        name = validate_eval_name(entry["name"])
        if selected and name not in selected:
            continue
        extra = [str(value) for value in entry.get("args", [])]
        modalities = entry.get("modalities")
        run_config = _experiment_config(config, modalities)
        parser.raise_errors = True
        try:
            if modalities is not None:
                if not run_config["datasets"]:
                    raise ValueError(
                        "'modalities' leave no dataset with a prediction to evaluate"
                    )
                unconfigured = sorted(set(modalities) - _configured_modalities(config))
                if unconfigured:
                    print(
                        f"Warning: experiments[{index}] ({name!r}): modalities "
                        f"{unconfigured} are not configured, so listing them has "
                        "no effect",
                        file=sys.stderr,
                    )
            run_args = parser.parse_args([*argv, *extra])
            if (
                run_args.eval_name != base.eval_name
                or run_args.experiments != base.experiments
            ):
                raise ValueError(
                    "'args' cannot set --eval-name or --experiment; the "
                    "experiment's 'name' names its results"
                )
            run_args.eval_name = name
            _finish_args(run_args)
            if run_args.sky_iou and _gt_segmentation_entry(run_config["gt"])[1] is None:
                raise ValueError(
                    "--sky-iou requires gt.segmentation (or its alias "
                    "gt.semantic_segmentation) in config"
                )
            runs.append(
                _prepare_run(run_args, run_config, name=name, extra_args=tuple(extra))
            )
        except ValueError as e:
            raise ValueError(f"experiments[{index}] ({name!r}): {e}") from e
        finally:
            parser.raise_errors = False
    return runs


def _start_euler_train(config: dict):
    """Initialize euler-train logging if configured; once for all experiments."""
    et_config = config.get("euler_train")
    if et_config is None:
        return None, False
    et_dir = Path(et_config["dir"])
    et_resuming = (et_dir / "meta.json").is_file()
    et_run = _euler_train.init(dir=et_dir, mode="eval")
    print(f"euler_train logging enabled -> {et_run.dir}")
    return et_run, et_resuming


def _finish_euler_train(et_run, et_resuming: bool) -> None:
    if et_run is None:
        return
    if et_resuming:
        et_run.detach()
        print(f"\neuler_train run detached (run still active): {et_run.run_id}")
    else:
        et_run.finish()
        print(f"\neuler_train run finished: {et_run.run_id}")


def _evaluate_experiments(
    runs: list[_Run], *, package_metadata: dict, et_run=None
) -> list[str]:
    """Run each experiment in turn; a failing one does not stop the rest.

    Returns:
        Names of the experiments that failed.
    """
    failed = []
    for number, run in enumerate(runs, 1):
        print(f"\n{'#' * 60}\nExperiment {number}/{len(runs)}: {run.name}")
        print(f"  args: {' '.join(run.extra_args) or '(none)'}\n{'#' * 60}")
        try:
            _evaluate(
                run,
                package_metadata=package_metadata,
                et_run=et_run,
                experiment=run.name,
            )
        except SystemExit as e:
            if e.code not in (None, 0):
                failed.append(run.name)
        except Exception:
            traceback.print_exc()
            failed.append(run.name)
        if failed and failed[-1] == run.name:
            print(f"\nExperiment {run.name!r} failed.", file=sys.stderr)

    print(f"\nExperiments: {len(runs) - len(failed)} of {len(runs)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
    return failed


def _build_parser() -> argparse.ArgumentParser:
    """Return the ``euler-eval`` argument parser."""
    parser = _ArgumentParser(
        description="Evaluate depth, sparse depth, RGB, rays, and points_3d datasets using euler_loading"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config.json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for metrics that support batching (default: 16)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--domain",
        dest="domains",
        action="append",
        choices=available_domains(),
        default=[],
        help=(
            "Add a domain-specific metric set to the always-enabled core set. "
            "May be repeated (available: %(choices)s)."
        ),
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth evaluation",
    )
    parser.add_argument(
        "--skip-rgb",
        action="store_true",
        help="Skip RGB evaluation",
    )
    parser.add_argument(
        "--skip-rays",
        action="store_true",
        help="Skip rays (spherical direction map) evaluation",
    )
    parser.add_argument(
        "--skip-points-3d",
        action="store_true",
        help="Skip points_3d (per-pixel 3D point map) evaluation",
    )
    parser.add_argument(
        "--mask-sky",
        action="store_true",
        help="Mask sky regions from metrics using GT segmentation",
    )
    parser.add_argument(
        "--missing-segmentation",
        choices=["drop", "unmasked"],
        default="drop",
        help=(
            "Frames without their own GT segmentation: 'drop' leaves them out "
            "(default); 'unmasked' evaluates them without GT sky masking and "
            "leaves them out of the sky IoU"
        ),
    )
    parser.add_argument(
        "--mask-sky-prediction",
        action="store_true",
        help=(
            "Exclude pixels in a prediction's datasets[].sky_mask from dense and "
            "sparse depth metrics and the alignment fit; predictions without a "
            "sky_mask are scored in full"
        ),
    )
    parser.add_argument(
        "--sky-depth",
        type=float,
        default=None,
        metavar="METERS",
        help=(
            "Cap GT and predicted depths at this finite positive value after "
            "conversion/alignment; fill predicted sky_mask pixels with it "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--sky-iou",
        action="store_true",
        help=(
            "Score predicted sky against GT segmentation sky (IoU, precision, "
            "recall): explicit from datasets[].sky_mask, implicit from depth "
            ">= --sky-depth for metric predictions. Requires gt.segmentation; "
            "depth metrics are unchanged"
        ),
    )
    parser.add_argument(
        "--no-sanity-check",
        action="store_true",
        help="Disable sanity checking of metric configurations",
    )
    parser.add_argument(
        "--metrics-config",
        type=str,
        default=None,
        help="Path to metrics_config.json for sanity checking (default: auto-detect)",
    )
    parser.add_argument(
        "--depth-alignment",
        type=str,
        default="auto_affine",
        choices=["none", "auto_affine", "affine"],
        help=(
            "Depth calibration mode: none, auto_affine (default), or affine. "
            "Output is emitted in semantic native/metric spaces."
        ),
    )
    parser.add_argument(
        "--points-3d-alignment",
        type=str,
        default="auto",
        choices=["none", "scale", "similarity", "auto"],
        help=(
            "points_3d gauge alignment: none, scale, similarity (Umeyama), or "
            "auto (default). auto applies similarity only for declared-relative "
            "predictions, else none. Outputs native/metric spaces."
        ),
    )
    parser.add_argument(
        "--rgb-fid-backend",
        type=str,
        default="builtin",
        choices=["builtin", "clean-fid"],
        help="Backend for RGB FID computation: builtin (default) or clean-fid",
    )
    parser.add_argument(
        "--benchmark-depth-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help=(
            "Depth range [min, max] in meters for benchmark evaluation. "
            "Computes depth and RGB metrics only for pixels within this range, "
            "subdivided into square-root-scaled near/mid/far bins (additive to regular metrics)."
        ),
    )

    distributions_group = parser.add_mutually_exclusive_group()
    distributions_group.add_argument(
        "--distributions", action="store_true", default=None,
        help="Compute per-sample and pixel-pool RMSE histograms for depth and points_3d",
    )
    distributions_group.add_argument(
        "--no-distributions", dest="distributions", action="store_false",
        help="Disable histograms, including those enabled in config.json",
    )
    parser.add_argument(
        "--distribution-bins", type=int, default=None, metavar="N",
        help="Total histogram bins including tails (default: 50; minimum: 3)",
    )
    parser.add_argument(
        "--distribution-scale", choices=["log", "linear"], default=None,
        help="Histogram bin spacing (default: log)",
    )
    parser.add_argument(
        "--distribution-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Finite histogram range in evaluated space units (default: 0.001 100); tails are included",
    )
    parser.add_argument(
        "--eval-name",
        default=DEFAULT_EVAL_NAME,
        metavar="NAME",
        help=(
            "Name this evaluation: results are written as NAME.json in each "
            "prediction dataset's .eval/ folder, so differently named runs "
            "sit side by side (default: %(default)s). A dataset's "
            "output_file still takes precedence."
        ),
    )
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Run only this experiment from the config's 'experiments' section; "
            "repeatable (default: every experiment)"
        ),
    )
    return parser


def _evaluate(
    run: _Run,
    *,
    package_metadata: dict,
    et_run=None,
    experiment: str | None = None,
) -> None:
    """Evaluate every configured prediction dataset with one set of arguments.

    *experiment* is the experiment name when the config defines
    ``experiments``; it keys the euler-train evaluations and the sanity-check
    report so experiments do not overwrite each other.
    """
    args = run.args
    config = run.config
    distribution_config = run.distribution_config
    selected_metric_sets = run.metric_sets

    requested_device = args.device
    args.device = resolve_device(requested_device)
    configure_torch_runtime(args.device)
    print_device_info(requested_device, args.device)
    print(f"Depth alignment: {args.depth_alignment}")
    if distribution_config is not None:
        print(
            f"RMSE distributions: {distribution_config.n_bins} bins "
            f"({distribution_config.scale})"
        )
    if args.sky_depth is not None:
        print(f"Sky depth: {args.sky_depth} meters (depth cap and predicted sky fill)")
    print(f"RGB FID backend: {args.rgb_fid_backend}")
    print(
        "Metric sets: "
        + ", ".join(metric_set.name for metric_set in selected_metric_sets)
    )
    if args.benchmark_depth_range:
        bdr = args.benchmark_depth_range
        print(f"Benchmark depth range: [{bdr[0]}, {bdr[1]}] meters")
    print("-" * 60)

    # Check sky masking prerequisites
    segmentation_key, segmentation_entry = _gt_segmentation_entry(config["gt"])
    if args.mask_sky:
        if segmentation_entry is None:
            print(
                "Warning: --mask-sky requires gt.segmentation in config. "
                "Sky masking disabled.",
                file=sys.stderr,
            )
            args.mask_sky = False
        else:
            print(f"Sky masking enabled ({segmentation_key})")
    if args.mask_sky_prediction:
        print("Predicted sky masking enabled (datasets[].sky_mask)")
    if args.sky_iou:
        # Unlike --mask-sky, never disable silently: the user asked for a metric.
        if segmentation_entry is None:
            print(
                "Error: --sky-iou requires gt.segmentation (or its alias "
                "gt.semantic_segmentation) in config.",
                file=sys.stderr,
            )
            sys.exit(1)
        implicit_text = (
            f", implicit at depth >= {args.sky_depth:g} m"
            if args.sky_depth is not None
            else ""
        )
        print(
            f"Sky IoU enabled (GT: {segmentation_key}; explicit from "
            f"sky_mask{implicit_text})"
        )
    segmentation_selector = None
    if segmentation_entry is not None and (args.mask_sky or args.sky_iou):
        try:
            segmentation_selector = resolve_gt_segmentation_path(
                segmentation_entry["path"], config_key=segmentation_key
            )
        except ValueError as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    print("-" * 60)

    # Initialize sanity checker if not disabled
    sanity_checker = None
    if not args.no_sanity_check:
        config_path = Path(args.metrics_config) if args.metrics_config else None
        sanity_checker = SanityChecker(config_path)
        print("Sanity checking enabled")
    else:
        print("Sanity checking disabled")
    print("-" * 60)


    gt = config["gt"]
    protocol_identities = reference_identities(gt)
    protocol_benchmark_range = (
        list(args.benchmark_depth_range) if args.benchmark_depth_range else None
    )
    gt_depth_path = gt.get("depth", {}).get("path")
    gt_sparse_depth_path = gt.get("sparse_depth", {}).get("path")
    gt_rgb_path = gt.get("rgb", {}).get("path")
    gt_rays_path = gt.get("rays", {}).get("path")
    gt_points_3d_path = gt.get("points_3d", {}).get("path")
    calibration_path = gt.get("calibration", {}).get("path")
    intrinsics_path = gt.get("intrinsics", {}).get("path")
    camera_extrinsics_path = gt.get("camera_extrinsics", {}).get("path")
    lidar_extrinsics_path = gt.get("lidar_extrinsics", {}).get("path")
    segmentation_path = segmentation_selector if args.mask_sky else None
    gt_depth_split = gt.get("depth", {}).get("split")
    gt_sparse_depth_split = gt.get("sparse_depth", {}).get("split")
    gt_rgb_split = gt.get("rgb", {}).get("split")
    gt_rays_split = gt.get("rays", {}).get("split")
    gt_points_3d_split = gt.get("points_3d", {}).get("split")
    calibration_split = gt.get("calibration", {}).get("split")
    intrinsics_split = gt.get("intrinsics", {}).get("split")
    camera_extrinsics_split = gt.get("camera_extrinsics", {}).get("split")
    lidar_extrinsics_split = gt.get("lidar_extrinsics", {}).get("split")
    segmentation_split = (
        segmentation_entry.get("split") if segmentation_path is not None else None
    )
    # Depth and sparse depth also read GT segmentation for --sky-iou; it is
    # used only for sky masking when --mask-sky is set as well.
    depth_segmentation_path = segmentation_selector
    depth_segmentation_split = (
        segmentation_entry.get("split") if segmentation_selector is not None else None
    )

    # Evaluate each prediction dataset
    for dataset_config in config["datasets"]:
        ds_name = dataset_config["name"]
        et_key = ds_name if experiment is None else f"{ds_name}/{experiment}"
        pred_depth_key, pred_depth_config = _prediction_depth_entry(dataset_config)
        has_depth = pred_depth_config is not None
        pred_sky_mask_config = (
            dataset_config.get("sky_mask", {})
            if args.sky_depth is not None or args.sky_iou or args.mask_sky_prediction
            else {}
        )
        pred_sky_mask_path = pred_sky_mask_config.get("path")
        pred_sky_mask_split = pred_sky_mask_config.get("split")
        # ID-paired inputs that only --sky-iou reads must not drop depth pairs.
        # The sky_mask fill stays gated on --sky-depth, and masking on
        # --mask-sky (GT) and --mask-sky-prediction (sky_mask).
        sky_mask_scores_depth = (
            args.sky_depth is not None or args.mask_sky_prediction
        )
        iou_only_inputs = tuple(
            name
            for name, iou_only in (
                ("segmentation", depth_segmentation_path and not args.mask_sky),
                ("pred_sky_mask", pred_sky_mask_path and not sky_mask_scores_depth),
            )
            if iou_only
        )
        has_rgb = "rgb" in dataset_config and "path" in dataset_config["rgb"]
        has_rays = "rays" in dataset_config and "path" in dataset_config["rays"]
        has_points_3d = (
            "points_3d" in dataset_config
            and "path" in dataset_config["points_3d"]
        )

        all_results = {}
        depth_save = {}
        rgb_save = {}
        rays_save = {}
        points_3d_save = {}
        points_3d_sparse_save = {}
        et_eval_datasets = {}
        has_benchmark = args.benchmark_depth_range is not None

        # Register evaluation as running before work begins
        if et_run is not None:
            et_run.add_evaluation(
                et_key,
                name=et_key,
                status="running",
                metadata={"package": package_metadata},
            )

        # -- Depth evaluation --
        depth_dataset = None
        if (
            has_depth
            and gt_depth_path
            and not gt_sparse_depth_path
            and not args.skip_depth
        ):
            pred_depth_path = pred_depth_config["path"]
            pred_depth_split = pred_depth_config.get("split")
            print(f"\n[DEPTH] Evaluating: '{ds_name}'")
            print(f"  GT:   {gt_depth_path}")
            print(f"  Pred: {pred_depth_path}")
            if pred_depth_key != "depth":
                print(f"  Pred depth entry: {pred_depth_key}")

            depth_dataset, skipped_inputs = _build_eval_dataset(
                build_depth_eval_dataset,
                dict(
                    gt_depth_path=gt_depth_path,
                    pred_depth_path=pred_depth_path,
                    pred_sky_mask_path=pred_sky_mask_path,
                    pred_sky_mask_split=pred_sky_mask_split,
                    calibration_path=calibration_path,
                    segmentation_path=depth_segmentation_path,
                    pred_depth_metadata_scope=(
                        pred_depth_key if pred_depth_key != "depth" else None
                    ),
                    segmentation_modality_key=segmentation_key or "segmentation",
                    gt_depth_split=gt_depth_split,
                    pred_depth_split=pred_depth_split,
                    calibration_split=calibration_split,
                    segmentation_split=depth_segmentation_split,
                ),
                ds_name=ds_name,
                iou_only=iou_only_inputs,
                missing_segmentation=args.missing_segmentation,
            )
            et_eval_datasets["depth"] = depth_dataset

            depth_meta = get_depth_metadata(depth_dataset)
            print(f"  radial_depth: {depth_meta['radial_depth']}")
            print(f"  Matched pairs: {len(depth_dataset)}")

            depth_results = evaluate_depth_samples(
                distribution_config=distribution_config,
                dataset=depth_dataset,
                is_radial=depth_meta["radial_depth"],
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
                sky_mask_enabled=args.mask_sky,
                sky_depth=args.sky_depth,
                sky_iou=args.sky_iou,
                pred_sky_mask_enabled=args.mask_sky_prediction,
                alignment_mode=args.depth_alignment,
                benchmark_depth_range=(
                    tuple(args.benchmark_depth_range)
                    if args.benchmark_depth_range
                    else None
                ),
                input_space_hint=_prediction_depth_space_hint(pred_depth_key),
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, is_depth=True)

            # Build per-modality results for saving.
            # All metric names must be fully-qualified under the declared
            # metricNamespace. We nest semantic spaces under depth → eval so
            # every flattened path starts with "depth.eval.".
            space_info = depth_results.get("space_info", {})
            depth_dataset_info = depth_results.get("dataset_info", {})

            depth_spatial = depth_results.get("spatial_info", {})
            depth_axes = _depth_eval_axes(
                benchmark=has_benchmark, distributions=distribution_config is not None
            )
            depth_descriptions = _distribution_descriptions(
                _DEPTH_EVAL_DESCRIPTIONS, distribution_config, "rmse"
            )
            sky_tree = _sky_segmentation_tree(depth_results)
            if sky_tree:
                depth_axes = _with_sky_segmentation_axes(depth_axes)
                depth_descriptions = {
                    **depth_descriptions, **_SKY_SEGMENTATION_EVAL_DESCRIPTIONS
                }
            depth_ns = _EvalNamespace(
                producer="euler-eval",
                producer_version=_get_version(),
                modalities=("depth",),
                axes=depth_axes,
                descriptions=depth_descriptions,
            )
            depth_save = {
                "metricSet": depth_ns.metric_set_envelope(
                    "depth",
                    metadata={
                        **_distribution_metadata(depth_results),
                        "input_space_detected": space_info.get(
                            "input_space_detected", "unknown"
                        ),
                        "metric_space_source": space_info.get("metric_space_source"),
                        "calibration_mode": space_info.get(
                            "calibration_mode", "unknown"
                        ),
                        "calibration_applied": space_info.get(
                            "calibration_applied", False
                        ),
                        "emitted_spaces": space_info.get("emitted_spaces", []),
                        "canonical_space": space_info.get("canonical_space", "metric"),
                    },
                ),
                "protocol": build_protocol(
                    protocol_identities,
                    gt="depth",
                    inputs=("calibration", "segmentation"),
                    population={
                        "samples": depth_dataset_info.get("num_pairs"),
                        "gt_sky_masking": bool(args.mask_sky),
                        "missing_segmentation": (
                            args.missing_segmentation if args.mask_sky else None
                        ),
                        "pred_sky_masking": bool(args.mask_sky_prediction),
                        "evaluated_resolution": evaluated_resolution(depth_spatial),
                        "benchmark_depth_range_m": protocol_benchmark_range,
                    },
                    method={
                        "depth_alignment": args.depth_alignment,
                        "sky_depth_m": args.sky_depth,
                        "pred_sky_mask": bool(
                            pred_sky_mask_path and sky_mask_scores_depth
                        ),
                        "spatial_alignment": depth_spatial.get("method", "none"),
                    },
                ),
                "dataset_info": depth_dataset_info,
                "meta": _clean_metric_tree({
                    "version": _get_version(),
                    "eval_name": args.eval_name,
                    "modality": "depth",
                    "device": args.device,
                    "gt": {
                        "path": gt_depth_path,
                        "split": gt_depth_split,
                        "dimensions": depth_spatial.get("gt_dimensions"),
                    },
                    "pred": {
                        "path": pred_depth_path,
                        "split": pred_depth_split,
                        "entry": pred_depth_key,
                        "dimensions": depth_spatial.get("pred_dimensions"),
                        # A mask dropped for --sky-iou coverage was never read.
                        "sky_mask": (
                            None
                            if "pred_sky_mask" in skipped_inputs
                            else pred_sky_mask_config or None
                        ),
                    },
                    "spatial_alignment": {
                        "method": depth_spatial.get("method", "none"),
                        "evaluated_dimensions": depth_spatial.get(
                            "evaluated_dimensions"
                        ),
                    },
                    "modality_params": depth_meta,
                    "gt_sky_masking": depth_results.get("gt_sky_masking"),
                    "sky_segmentation": _sky_segmentation_meta(
                        depth_results, segmentation_key, skipped_inputs
                    ),
                    "pred_sky_masking": depth_results.get("pred_sky_masking"),
                    "eval_params": {
                        "sky_masking": args.mask_sky,
                        "depth_alignment_mode": args.depth_alignment,
                        "sky_depth": args.sky_depth,
                        "sky_iou": True if args.sky_iou else None,
                        "mask_sky_prediction": (
                            True if args.mask_sky_prediction else None
                        ),
                        "batch_size": args.batch_size,
                        "num_workers": args.num_workers,
                        "benchmark_depth_range": (
                            list(args.benchmark_depth_range)
                            if args.benchmark_depth_range
                            else None
                        ),
                    },
                }),
                "depth": {"eval": {}},
            }
            for space_name, result_key in (
                ("native", "depth_native"),
                ("metric", "depth_metric"),
            ):
                branch = depth_results.get(result_key)
                if branch is not None:
                    depth_save["depth"]["eval"][space_name] = _clean_metric_tree(branch)
            if sky_tree:
                depth_save["depth"]["eval"][_SKY_SEGMENTATION_CATEGORY] = sky_tree

            # Inject benchmark bin metrics under the existing category
            # keys so that the bin axis decomposes correctly:
            #   depth.eval.metric.standard.image_mean.{bin}.absrel
            #   depth.eval.metric.depth_metrics.{bin}.absrel.median
            #   depth.eval.metric.geometric_metrics.{bin}.normal_consistency.mean_angle
            depth_benchmark = depth_results.get("depth_benchmark")
            if depth_benchmark is not None:
                for space_name in ("native", "metric"):
                    if space_name not in depth_save["depth"]["eval"]:
                        continue
                    space_benchmark = depth_benchmark.get(space_name)
                    if space_benchmark is None:
                        continue
                    target = depth_save["depth"]["eval"][space_name]
                    for bn in ("all", "near", "mid", "far"):
                        bin_summary = space_benchmark.get(bn, {})
                        for category, metrics in bin_summary.items():
                            cleaned = _clean_metric_tree(metrics)
                            if cleaned:
                                if category in {"standard", "distributions"}:
                                    bucket = target.setdefault(category, {})
                                    for reduction, reduction_metrics in cleaned.items():
                                        bucket.setdefault(reduction, {})[bn] = reduction_metrics
                                else:
                                    target.setdefault(category, {})[bn] = cleaned
                depth_save["metricSet"]["metadata"]["benchmark"] = {
                    "depth_range": depth_benchmark["boundaries"]["range"],
                    "boundaries": depth_benchmark["boundaries"],
                }
            for depth_key in (
                "depth",
                "depth_native",
                "depth_metric",
                "depth_benchmark",
                _SKY_SEGMENTATION_CATEGORY,
            ):
                if depth_key in depth_results and depth_results[depth_key] is not None:
                    all_results[depth_key] = depth_results[depth_key]
            depth_pfm = depth_results.get("per_file_metrics", {})
            if depth_pfm:
                canonical_space = space_info.get("canonical_space", "metric")
                depth_save["per_file_metrics"] = _clean_per_file_metrics(
                    _wrap_pfm_metrics(
                        depth_pfm,
                        lambda m: _wrap_depth_space_pfm_metrics(
                            m,
                            metric_root="depth",
                            canonical_key="depth",
                            canonical_space=canonical_space,
                        ),
                    ),
                    label="depth per_file_metrics",
                )
            all_results.setdefault("per_file_metrics", {}).update(depth_pfm)

            print_results(
                {k: v for k, v in depth_results.items()
                 if k not in ("per_file_metrics", "spatial_info")},
                f"DEPTH: {ds_name}",
            )
            sky_iou_line = _sky_iou_summary_line(ds_name, depth_results)
            if sky_iou_line is not None:
                print(sky_iou_line)

        # -- Sparse pointcloud depth evaluation --
        sparse_depth_dataset = None
        if has_depth and gt_sparse_depth_path and not args.skip_depth:
            pred_depth_path = pred_depth_config["path"]
            pred_depth_split = pred_depth_config.get("split")
            print(f"\n[SPARSE_DEPTH] Evaluating: '{ds_name}'")
            print(f"  GT sparse pointcloud: {gt_sparse_depth_path}")
            print(f"  Pred dense depth:     {pred_depth_path}")
            if pred_depth_key != "depth":
                print(f"  Pred depth entry:     {pred_depth_key}")
            print(f"  Intrinsics:           {intrinsics_path}")
            print(f"  Camera extrinsics:    {camera_extrinsics_path}")
            if lidar_extrinsics_path:
                print(f"  Lidar extrinsics:     {lidar_extrinsics_path}")

            sparse_depth_dataset, skipped_inputs = _build_eval_dataset(
                build_sparse_depth_eval_dataset,
                dict(
                    gt_sparse_depth_path=gt_sparse_depth_path,
                    pred_depth_path=pred_depth_path,
                    pred_sky_mask_path=pred_sky_mask_path,
                    pred_sky_mask_split=pred_sky_mask_split,
                    intrinsics_path=intrinsics_path,
                    camera_extrinsics_path=camera_extrinsics_path,
                    lidar_extrinsics_path=lidar_extrinsics_path,
                    segmentation_path=depth_segmentation_path,
                    pred_depth_metadata_scope=(
                        pred_depth_key if pred_depth_key != "depth" else None
                    ),
                    segmentation_modality_key=segmentation_key or "segmentation",
                    gt_sparse_depth_split=gt_sparse_depth_split,
                    pred_depth_split=pred_depth_split,
                    intrinsics_split=intrinsics_split,
                    camera_extrinsics_split=camera_extrinsics_split,
                    lidar_extrinsics_split=lidar_extrinsics_split,
                    segmentation_split=depth_segmentation_split,
                ),
                ds_name=ds_name,
                iou_only=iou_only_inputs,
                missing_segmentation=args.missing_segmentation,
            )
            et_eval_datasets["sparse_depth"] = sparse_depth_dataset

            sparse_depth_meta = get_sparse_depth_metadata(sparse_depth_dataset)
            print(f"  pred_radial_depth: {sparse_depth_meta['pred_radial_depth']}")
            print(f"  Matched pairs: {len(sparse_depth_dataset)}")

            sparse_depth_results = evaluate_sparse_depth_samples(
                distribution_config=distribution_config,
                dataset=sparse_depth_dataset,
                pred_is_radial=sparse_depth_meta["pred_radial_depth"],
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
                sky_mask_enabled=args.mask_sky,
                sky_depth=args.sky_depth,
                sky_iou=args.sky_iou,
                pred_sky_mask_enabled=args.mask_sky_prediction,
                alignment_mode=args.depth_alignment,
                benchmark_depth_range=(
                    tuple(args.benchmark_depth_range)
                    if args.benchmark_depth_range
                    else None
                ),
                input_space_hint=_prediction_depth_space_hint(pred_depth_key),
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, is_depth=True)

            space_info = sparse_depth_results.get("space_info", {})
            sparse_depth_dataset_info = sparse_depth_results.get("dataset_info", {})
            sparse_depth_spatial = sparse_depth_results.get("spatial_info", {})
            sparse_depth_axes = _sparse_depth_eval_axes(
                benchmark=has_benchmark, distributions=distribution_config is not None
            )
            sparse_depth_descriptions = _distribution_descriptions(
                _SPARSE_DEPTH_EVAL_DESCRIPTIONS, distribution_config, "rmse"
            )
            sky_tree = _sky_segmentation_tree(sparse_depth_results)
            if sky_tree:
                sparse_depth_axes = _with_sky_segmentation_axes(sparse_depth_axes)
                sparse_depth_descriptions = {
                    **sparse_depth_descriptions, **_SKY_SEGMENTATION_EVAL_DESCRIPTIONS
                }
            sparse_depth_ns = _EvalNamespace(
                producer="euler-eval",
                producer_version=_get_version(),
                modalities=("sparse_depth",),
                axes=sparse_depth_axes,
                descriptions=sparse_depth_descriptions,
            )
            depth_save = {
                "metricSet": _sparse_depth_metric_set_envelope(
                    sparse_depth_ns,
                    metadata={
                        **_distribution_metadata(sparse_depth_results),
                        "input_space_detected": space_info.get(
                            "input_space_detected", "unknown"
                        ),
                        "metric_space_source": space_info.get("metric_space_source"),
                        "calibration_mode": space_info.get(
                            "calibration_mode", "unknown"
                        ),
                        "calibration_applied": space_info.get(
                            "calibration_applied", False
                        ),
                        "emitted_spaces": space_info.get("emitted_spaces", []),
                        "canonical_space": space_info.get("canonical_space", "metric"),
                    },
                ),
                "protocol": build_protocol(
                    protocol_identities,
                    gt="sparse_depth",
                    inputs=(
                        "intrinsics",
                        "camera_extrinsics",
                        "lidar_extrinsics",
                        "segmentation",
                    ),
                    population={
                        "samples": sparse_depth_dataset_info.get("num_pairs"),
                        "gt_sky_masking": bool(args.mask_sky),
                        "missing_segmentation": (
                            args.missing_segmentation if args.mask_sky else None
                        ),
                        "pred_sky_masking": bool(args.mask_sky_prediction),
                        "evaluated_resolution": evaluated_resolution(
                            sparse_depth_spatial
                        ),
                        "benchmark_depth_range_m": protocol_benchmark_range,
                    },
                    method={
                        "depth_alignment": args.depth_alignment,
                        "sky_depth_m": args.sky_depth,
                        "pred_sky_mask": bool(
                            pred_sky_mask_path and sky_mask_scores_depth
                        ),
                    },
                ),
                "dataset_info": sparse_depth_dataset_info,
                "meta": _clean_metric_tree({
                    "version": _get_version(),
                    "eval_name": args.eval_name,
                    "modality": "sparse_depth",
                    "device": args.device,
                    "gt": {
                        "path": gt_sparse_depth_path,
                        "split": gt_sparse_depth_split,
                        "representation": "point_cloud",
                    },
                    "pred": {
                        "path": pred_depth_path,
                        "split": pred_depth_split,
                        "entry": pred_depth_key,
                        "dimensions": sparse_depth_spatial.get("pred_dimensions"),
                        # A mask dropped for --sky-iou coverage was never read.
                        "sky_mask": (
                            None
                            if "pred_sky_mask" in skipped_inputs
                            else pred_sky_mask_config or None
                        ),
                    },
                    "calibration": {
                        "intrinsics_path": intrinsics_path,
                        "intrinsics_split": intrinsics_split,
                        "camera_extrinsics_path": camera_extrinsics_path,
                        "camera_extrinsics_split": camera_extrinsics_split,
                        "lidar_extrinsics_path": lidar_extrinsics_path,
                        "lidar_extrinsics_split": lidar_extrinsics_split,
                    },
                    "spatial_alignment": {
                        "method": sparse_depth_spatial.get(
                            "method", "pointcloud_projection"
                        ),
                        "evaluated_dimensions": sparse_depth_spatial.get(
                            "evaluated_dimensions"
                        ),
                    },
                    "modality_params": sparse_depth_meta,
                    "gt_sky_masking": sparse_depth_results.get("gt_sky_masking"),
                    "sky_segmentation": _sky_segmentation_meta(
                        sparse_depth_results, segmentation_key, skipped_inputs
                    ),
                    "pred_sky_masking": sparse_depth_results.get(
                        "pred_sky_masking"
                    ),
                    "eval_params": {
                        "sky_masking": args.mask_sky,
                        "depth_alignment_mode": args.depth_alignment,
                        "sky_depth": args.sky_depth,
                        "sky_iou": True if args.sky_iou else None,
                        "mask_sky_prediction": (
                            True if args.mask_sky_prediction else None
                        ),
                        "num_workers": args.num_workers,
                        "benchmark_depth_range": (
                            list(args.benchmark_depth_range)
                            if args.benchmark_depth_range
                            else None
                        ),
                    },
                }),
                _SPARSE_DEPTH_METRIC_ROOT: {"eval": {}},
            }
            for space_name, result_key in (
                ("native", "sparse_depth_native"),
                ("metric", "sparse_depth_metric"),
            ):
                branch = sparse_depth_results.get(result_key)
                if branch is not None:
                    depth_save[_SPARSE_DEPTH_METRIC_ROOT]["eval"][space_name] = (
                        _clean_metric_tree(branch)
                    )
            if sky_tree:
                depth_save[_SPARSE_DEPTH_METRIC_ROOT]["eval"][
                    _SKY_SEGMENTATION_CATEGORY
                ] = sky_tree

            sparse_depth_benchmark = sparse_depth_results.get("sparse_depth_benchmark")
            if sparse_depth_benchmark is not None:
                for space_name in ("native", "metric"):
                    if space_name not in depth_save[_SPARSE_DEPTH_METRIC_ROOT]["eval"]:
                        continue
                    space_benchmark = sparse_depth_benchmark.get(space_name)
                    if space_benchmark is None:
                        continue
                    target = depth_save[_SPARSE_DEPTH_METRIC_ROOT]["eval"][space_name]
                    for bn in ("all", "near", "mid", "far"):
                        bin_summary = space_benchmark.get(bn, {})
                        for category, metrics in bin_summary.items():
                            cleaned = _clean_metric_tree(metrics)
                            if cleaned:
                                if category in {"standard", "distributions"}:
                                    bucket = target.setdefault(category, {})
                                    for reduction, reduction_metrics in cleaned.items():
                                        bucket.setdefault(reduction, {})[bn] = (
                                            reduction_metrics
                                        )
                                else:
                                    target.setdefault(category, {})[bn] = cleaned
                depth_save["metricSet"]["metadata"]["benchmark"] = {
                    "depth_range": sparse_depth_benchmark["boundaries"]["range"],
                    "boundaries": sparse_depth_benchmark["boundaries"],
                }
            for depth_key in (
                "sparse_depth",
                "sparse_depth_native",
                "sparse_depth_metric",
                "sparse_depth_benchmark",
                _SKY_SEGMENTATION_CATEGORY,
            ):
                if (
                    depth_key in sparse_depth_results
                    and sparse_depth_results[depth_key] is not None
                ):
                    all_results[depth_key] = sparse_depth_results[depth_key]
            sparse_depth_pfm = sparse_depth_results.get("per_file_metrics", {})
            if sparse_depth_pfm:
                canonical_space = space_info.get("canonical_space", "metric")
                depth_save["per_file_metrics"] = _clean_per_file_metrics(
                    _wrap_pfm_metrics(
                        sparse_depth_pfm,
                        lambda m: _wrap_depth_space_pfm_metrics(
                            m,
                            metric_root=_SPARSE_DEPTH_METRIC_ROOT,
                            canonical_key="sparse_depth",
                            canonical_space=canonical_space,
                        ),
                    ),
                    label="sparse_depth per_file_metrics",
                )
            all_results.setdefault("per_file_metrics", {}).update(sparse_depth_pfm)

            print_results(
                {
                    k: v
                    for k, v in sparse_depth_results.items()
                    if k not in ("per_file_metrics", "spatial_info")
                },
                f"SPARSE_DEPTH: {ds_name}",
            )
            sky_iou_line = _sky_iou_summary_line(ds_name, sparse_depth_results)
            if sky_iou_line is not None:
                print(sky_iou_line)

        # -- RGB evaluation --
        rgb_dataset = None
        if has_rgb and not args.skip_rgb:
            pred_rgb_path = dataset_config["rgb"]["path"]
            pred_rgb_split = dataset_config["rgb"].get("split")
            print(f"\n[RGB] Evaluating: '{ds_name}'")
            print(f"  GT:   {gt_rgb_path}")
            print(f"  Pred: {pred_rgb_path}")

            rgb_dataset, _ = _build_eval_dataset(
                build_rgb_eval_dataset,
                dict(
                    gt_rgb_path=gt_rgb_path,
                    pred_rgb_path=pred_rgb_path,
                    gt_depth_path=gt_depth_path,
                    calibration_path=calibration_path,
                    segmentation_path=segmentation_path,
                    gt_rgb_split=gt_rgb_split,
                    pred_rgb_split=pred_rgb_split,
                    gt_depth_split=gt_depth_split,
                    calibration_split=calibration_split,
                    segmentation_split=segmentation_split,
                    segmentation_modality_key=segmentation_key or "segmentation",
                ),
                ds_name=ds_name,
                missing_segmentation=args.missing_segmentation,
            )
            et_eval_datasets["rgb"] = rgb_dataset

            rgb_meta = get_rgb_metadata(rgb_dataset)
            print(f"  rgb_range: {rgb_meta['rgb_range']}")
            print(f"  Matched pairs: {len(rgb_dataset)}")

            depth_meta = (
                get_depth_metadata(rgb_dataset)
                if "gt_depth" in rgb_dataset.modality_paths()
                else None
            )

            rgb_results = evaluate_rgb_samples(
                dataset=rgb_dataset,
                depth_meta=depth_meta,
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
                sky_mask_enabled=args.mask_sky,
                fid_backend=args.rgb_fid_backend,
                benchmark_depth_range=(
                    tuple(args.benchmark_depth_range)
                    if args.benchmark_depth_range
                    else None
                ),
                domains=tuple(args.domains),
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, is_depth=False)

            # Build per-modality results for saving.
            # All metric names must be fully-qualified under the declared
            # metricNamespace.  We nest metrics under rgb → eval so every
            # flattened path starts with "rgb.eval.".
            rgb_dataset_info = rgb_results.get("dataset_info", {})
            rgb_spatial = rgb_results.get("spatial_info", {})

            rgb_ns = _EvalNamespace(
                producer="euler-eval",
                producer_version=_get_version(),
                modalities=("rgb",),
                axes=_rgb_eval_axes(
                    benchmark=has_benchmark,
                    domains=args.domains,
                ),
                descriptions=_RGB_EVAL_DESCRIPTIONS,
            )
            rgb_save = {
                "metricSet": rgb_ns.metric_set_envelope(
                    "rgb",
                    metadata={
                        "metric_sets": [
                            metric_set.name for metric_set in selected_metric_sets
                        ]
                    },
                ),
                "protocol": build_protocol(
                    protocol_identities,
                    gt="rgb",
                    inputs=("depth", "calibration", "segmentation"),
                    population={
                        "samples": rgb_dataset_info.get("num_pairs"),
                        "gt_sky_masking": bool(args.mask_sky),
                        "missing_segmentation": (
                            args.missing_segmentation if args.mask_sky else None
                        ),
                        "evaluated_resolution": evaluated_resolution(rgb_spatial),
                        "benchmark_depth_range_m": protocol_benchmark_range,
                    },
                    method={
                        "fid_backend": args.rgb_fid_backend,
                        "spatial_alignment": rgb_spatial.get("method", "none"),
                    },
                ),
                "dataset_info": rgb_dataset_info,
                "meta": _clean_metric_tree({
                    "version": _get_version(),
                    "eval_name": args.eval_name,
                    "modality": "rgb",
                    "device": args.device,
                    "gt": {
                        "path": gt_rgb_path,
                        "split": gt_rgb_split,
                        "dimensions": rgb_spatial.get("gt_dimensions"),
                    },
                    "pred": {
                        "path": pred_rgb_path,
                        "split": pred_rgb_split,
                        "dimensions": rgb_spatial.get("pred_dimensions"),
                    },
                    "spatial_alignment": {
                        "method": rgb_spatial.get("method", "none"),
                        "evaluated_dimensions": rgb_spatial.get(
                            "evaluated_dimensions"
                        ),
                    },
                    "modality_params": rgb_meta,
                    "gt_sky_masking": rgb_results.get("gt_sky_masking"),
                    "eval_params": {
                        "sky_masking": args.mask_sky,
                        "fid_backend": args.rgb_fid_backend,
                        "metric_sets": [
                            metric_set.name for metric_set in selected_metric_sets
                        ],
                        "batch_size": args.batch_size,
                        "num_workers": args.num_workers,
                        "benchmark_depth_range": (
                            list(args.benchmark_depth_range)
                            if args.benchmark_depth_range
                            else None
                        ),
                    },
                }),
            }
            rgb_metrics = rgb_results.get("rgb", {})
            if rgb_metrics:
                rgb_save["rgb"] = {"eval": _clean_metric_tree(rgb_metrics)}
                all_results["rgb"] = rgb_metrics

            # Inject benchmark RGB bin metrics directly under rgb.eval
            # so that the bin axis decomposes correctly:
            #   rgb.eval.{bin}.mae
            #   rgb.eval.{bin}.mse
            rgb_benchmark = rgb_results.get("rgb_benchmark")
            if rgb_benchmark is not None:
                eval_dict = rgb_save.setdefault("rgb", {}).setdefault("eval", {})
                for bn in ("all", "near", "mid", "far"):
                    bin_data = _clean_metric_tree({
                        "mae": rgb_benchmark["mae"].get(bn),
                        "mse": rgb_benchmark["mse"].get(bn),
                    })
                    if bin_data:
                        eval_dict[bn] = bin_data
                rgb_save["metricSet"]["metadata"]["benchmark"] = {
                    "depth_range": rgb_benchmark["boundaries"]["range"],
                    "boundaries": rgb_benchmark["boundaries"],
                }
                all_results["rgb_benchmark"] = rgb_benchmark
            rgb_pfm = rgb_results.get("per_file_metrics", {})
            if rgb_pfm:
                rgb_save["per_file_metrics"] = _clean_per_file_metrics(
                    _wrap_pfm_metrics(
                        rgb_pfm,
                        lambda m: {"rgb": {"eval": m.get("rgb", {})}},
                    ),
                    label="rgb per_file_metrics",
                )
            all_results.setdefault("per_file_metrics", {}).update(rgb_pfm)

            print_results(
                {k: v for k, v in rgb_results.items()
                 if k not in ("per_file_metrics", "spatial_info")},
                f"RGB: {ds_name}",
            )

        # -- Rays (spherical direction map) evaluation --
        rays_dataset = None
        if has_rays and gt_rays_path and not args.skip_rays:
            pred_rays_path = dataset_config["rays"]["path"]
            pred_rays_split = dataset_config["rays"].get("split")
            print(f"\n[RAYS] Evaluating: '{ds_name}'")
            print(f"  GT:   {gt_rays_path}")
            print(f"  Pred: {pred_rays_path}")

            rays_dataset = build_rays_eval_dataset(
                gt_rays_path=gt_rays_path,
                pred_rays_path=pred_rays_path,
                calibration_path=calibration_path,
                gt_rays_split=gt_rays_split,
                pred_rays_split=pred_rays_split,
                calibration_split=calibration_split,
            )
            et_eval_datasets["rays"] = rays_dataset

            rays_meta = get_rays_metadata(rays_dataset)
            print(f"  fov_domain: {rays_meta['fov_domain'] or 'auto-detect'}")
            print(f"  Matched pairs: {len(rays_dataset)}")

            rays_results = evaluate_rays_samples(
                dataset=rays_dataset,
                fov_domain=rays_meta["fov_domain"],
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, modality="rays")

            rays_dataset_info = rays_results.get("dataset_info", {})
            rays_spatial = rays_results.get("spatial_info", {})

            rays_ns = _EvalNamespace(
                producer="euler-eval",
                producer_version=_get_version(),
                modalities=("rays",),
                axes=_RAYS_EVAL_AXES,
                descriptions=_RAYS_EVAL_DESCRIPTIONS,
            )
            rays_save = {
                "metricSet": rays_ns.metric_set_envelope(
                    "rays",
                    metadata={
                        "fov_domain": rays_dataset_info.get("fov_domain"),
                        "threshold_deg": rays_dataset_info.get("threshold_deg"),
                    },
                ),
                "protocol": build_protocol(
                    protocol_identities,
                    gt="rays",
                    inputs=("calibration",),
                    population={
                        "samples": rays_dataset_info.get("num_pairs"),
                        "evaluated_resolution": evaluated_resolution(rays_spatial),
                    },
                    method={
                        "spatial_alignment": rays_spatial.get("method", "none"),
                    },
                ),
                "dataset_info": rays_dataset_info,
                "meta": _clean_metric_tree({
                    "version": _get_version(),
                    "eval_name": args.eval_name,
                    "modality": "rays",
                    "device": args.device,
                    "gt": {
                        "path": gt_rays_path,
                        "split": gt_rays_split,
                        "dimensions": rays_spatial.get("gt_dimensions"),
                    },
                    "pred": {
                        "path": pred_rays_path,
                        "split": pred_rays_split,
                        "dimensions": rays_spatial.get("pred_dimensions"),
                    },
                    "spatial_alignment": {
                        "method": rays_spatial.get("method", "none"),
                        "evaluated_dimensions": rays_spatial.get(
                            "evaluated_dimensions"
                        ),
                    },
                    "modality_params": {
                        "fov_domain": rays_dataset_info.get("fov_domain"),
                        "threshold_deg": rays_dataset_info.get("threshold_deg"),
                    },
                    "eval_params": {},
                }),
            }
            rays_metrics = rays_results.get("rays", {})
            if rays_metrics:
                rays_save["rays"] = {"eval": _clean_metric_tree(rays_metrics)}
                all_results["rays"] = rays_metrics
            rays_pfm = rays_results.get("per_file_metrics", {})
            if rays_pfm:
                rays_save["per_file_metrics"] = _clean_per_file_metrics(
                    _wrap_pfm_metrics(
                        rays_pfm,
                        lambda m: {"rays": {"eval": m.get("rays", {})}},
                    ),
                    label="rays per_file_metrics",
                )
            all_results.setdefault("per_file_metrics", {}).update(rays_pfm)

            print_results(
                {k: v for k, v in rays_results.items()
                 if k not in ("per_file_metrics", "spatial_info")},
                f"RAYS: {ds_name}",
            )

        # -- Points-3D (per-pixel 3D point map) evaluation --
        # GT is either an explicit gt.points_3d map, or synthesized on the fly
        # by unprojecting gt.depth with intrinsics (gt.intrinsics or
        # gt.calibration) when no precomputed point map is available.
        p3d_intrinsics_path = intrinsics_path or calibration_path
        p3d_intrinsics_split = (
            intrinsics_split if intrinsics_path else calibration_split
        )
        p3d_gt_synth = (
            gt_points_3d_path is None
            and gt_depth_path is not None
            and p3d_intrinsics_path is not None
        )
        if (
            has_points_3d
            and not args.skip_points_3d
            and (gt_points_3d_path or p3d_gt_synth)
        ):
            pred_points_3d_path = dataset_config["points_3d"]["path"]
            pred_points_3d_split = dataset_config["points_3d"].get("split")
            p3d_gt_is_depth = gt_points_3d_path is None
            p3d_gt_path = gt_depth_path if p3d_gt_is_depth else gt_points_3d_path
            p3d_gt_split = gt_depth_split if p3d_gt_is_depth else gt_points_3d_split
            print(f"\n[POINTS_3D] Evaluating: '{ds_name}'")
            if p3d_gt_is_depth:
                print(f"  GT:   {gt_depth_path} (synthesized from depth + intrinsics)")
            else:
                print(f"  GT:   {gt_points_3d_path}")
            print(f"  Pred: {pred_points_3d_path}")

            points_3d_dataset, _ = _build_eval_dataset(
                build_points_3d_eval_dataset,
                dict(
                    pred_points_3d_path=pred_points_3d_path,
                    gt_points_3d_path=gt_points_3d_path,
                    gt_depth_path=gt_depth_path if p3d_gt_is_depth else None,
                    intrinsics_path=p3d_intrinsics_path,
                    calibration_path=calibration_path,
                    segmentation_path=segmentation_path,
                    pred_points_3d_split=pred_points_3d_split,
                    gt_points_3d_split=gt_points_3d_split,
                    gt_depth_split=gt_depth_split,
                    intrinsics_split=p3d_intrinsics_split,
                    calibration_split=calibration_split,
                    segmentation_split=segmentation_split,
                    segmentation_modality_key=segmentation_key or "segmentation",
                ),
                ds_name=ds_name,
                missing_segmentation=args.missing_segmentation,
            )
            et_eval_datasets["points_3d"] = points_3d_dataset

            points_3d_meta = get_points_3d_metadata(points_3d_dataset)
            p3d_gt_radial = True
            if p3d_gt_is_depth:
                p3d_gt_radial = get_depth_metadata(points_3d_dataset)["radial_depth"]
                print(f"  GT depth radial: {p3d_gt_radial}")
            print(f"  fov_domain: {points_3d_meta['fov_domain'] or 'auto-detect'}")
            print(f"  Matched pairs: {len(points_3d_dataset)}")

            points_3d_results = evaluate_points_3d_samples(
                distribution_config=distribution_config,
                dataset=points_3d_dataset,
                fov_domain=points_3d_meta["fov_domain"],
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
                sky_mask_enabled=args.mask_sky,
                alignment_mode=args.points_3d_alignment,
                gt_is_depth=p3d_gt_is_depth,
                gt_depth_is_radial=p3d_gt_radial,
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, modality="points_3d")

            p3_space_info = points_3d_results.get("space_info", {})
            p3_dataset_info = points_3d_results.get("dataset_info", {})
            p3_spatial = points_3d_results.get("spatial_info", {})

            points_3d_ns = _EvalNamespace(
                producer="euler-eval",
                producer_version=_get_version(),
                modalities=("points_3d",),
                axes=_points_3d_eval_axes(distributions=distribution_config is not None),
                descriptions=_distribution_descriptions(
                    _POINTS_3D_EVAL_DESCRIPTIONS, distribution_config, "rmse3d"
                ),
            )
            points_3d_save = {
                "metricSet": _points_3d_metric_set_envelope(
                    points_3d_ns,
                    metadata={
                        **_distribution_metadata(points_3d_results),
                        "input_space_detected": p3_space_info.get(
                            "input_space_detected", "unknown"
                        ),
                        "calibration_mode": p3_space_info.get(
                            "calibration_mode", "unknown"
                        ),
                        "calibration_applied": p3_space_info.get(
                            "calibration_applied", False
                        ),
                        "emitted_spaces": p3_space_info.get("emitted_spaces", []),
                        "canonical_space": p3_space_info.get(
                            "canonical_space", "native"
                        ),
                        "fov_domain": p3_dataset_info.get("fov_domain"),
                        "threshold_deg": p3_dataset_info.get("threshold_deg"),
                    },
                ),
                "protocol": build_protocol(
                    protocol_identities,
                    gt="depth" if p3d_gt_is_depth else "points_3d",
                    inputs=("intrinsics", "calibration", "segmentation"),
                    population={
                        "samples": p3_dataset_info.get("num_pairs"),
                        "gt_sky_masking": bool(args.mask_sky),
                        "missing_segmentation": (
                            args.missing_segmentation if args.mask_sky else None
                        ),
                        "evaluated_resolution": evaluated_resolution(p3_spatial),
                    },
                    method={
                        "points_3d_alignment": args.points_3d_alignment,
                        "spatial_alignment": p3_spatial.get("method", "none"),
                    },
                ),
                "dataset_info": p3_dataset_info,
                "meta": _clean_metric_tree({
                    "version": _get_version(),
                    "eval_name": args.eval_name,
                    "modality": "points_3d",
                    "device": args.device,
                    "gt": {
                        "path": p3d_gt_path,
                        "split": p3d_gt_split,
                        "synthesized_from_depth": p3d_gt_is_depth,
                        "dimensions": p3_spatial.get("gt_dimensions"),
                    },
                    "pred": {
                        "path": pred_points_3d_path,
                        "split": pred_points_3d_split,
                        "dimensions": p3_spatial.get("pred_dimensions"),
                    },
                    "spatial_alignment": {
                        "method": p3_spatial.get("method", "none"),
                        "evaluated_dimensions": p3_spatial.get(
                            "evaluated_dimensions"
                        ),
                    },
                    "modality_params": points_3d_meta,
                    "gt_sky_masking": points_3d_results.get("gt_sky_masking"),
                    "eval_params": {
                        "sky_masking": args.mask_sky,
                        "points_3d_alignment_mode": args.points_3d_alignment,
                        "num_workers": args.num_workers,
                    },
                }),
                "points3d": {"eval": {}},
            }
            for space_name, result_key in (
                ("native", "points_3d_native"),
                ("metric", "points_3d_metric"),
            ):
                branch = points_3d_results.get(result_key)
                if branch is not None:
                    points_3d_save["points3d"]["eval"][space_name] = (
                        _clean_metric_tree(branch)
                    )
            for p3_key in ("points_3d", "points_3d_native", "points_3d_metric"):
                if (
                    p3_key in points_3d_results
                    and points_3d_results[p3_key] is not None
                ):
                    all_results[p3_key] = points_3d_results[p3_key]
            points_3d_pfm = points_3d_results.get("per_file_metrics", {})
            if points_3d_pfm:
                canonical_space = p3_space_info.get("canonical_space", "native")
                points_3d_save["per_file_metrics"] = _clean_per_file_metrics(
                    _wrap_pfm_metrics(
                        points_3d_pfm,
                        lambda m: _wrap_depth_space_pfm_metrics(
                            m,
                            metric_root="points3d",
                            canonical_key="points_3d",
                            canonical_space=canonical_space,
                        ),
                    ),
                    label="points_3d per_file_metrics",
                )
            all_results.setdefault("per_file_metrics", {}).update(points_3d_pfm)

            print_results(
                {k: v for k, v in points_3d_results.items()
                 if k not in ("per_file_metrics", "spatial_info")},
                f"POINTS_3D: {ds_name}",
            )

        elif (
            has_points_3d
            and not args.skip_points_3d
            and not gt_sparse_depth_path
        ):
            # (When gt.sparse_depth is present the sparse points_3d branch below
            # evaluates the predicted point map against the LiDAR cloud instead.)
            print(
                f"\n[POINTS_3D] Skipping '{ds_name}': no usable GT. Provide "
                "gt.points_3d, or gt.depth with gt.intrinsics/gt.calibration to "
                "synthesize the GT point map."
            )

        # -- Points-3D against sparse pointcloud GT --
        # When the GT is a sparse LiDAR cloud (gt.sparse_depth), evaluate the
        # *3D* metrics (Chamfer/F-score completeness + per-correspondence 3D
        # error) against the sparse GT cloud. This is an atomic points_3d
        # evaluation and therefore requires an explicit predicted points_3d map;
        # a depth prediction is handled only by the sparse-depth branch above.
        if (
            gt_sparse_depth_path
            and has_points_3d
            and not args.skip_points_3d
        ):
            print(f"\n[POINTS_3D · SPARSE] Evaluating: '{ds_name}'")
            print(f"  GT sparse pointcloud: {gt_sparse_depth_path}")
            p3s_pred_path = dataset_config["points_3d"]["path"]
            p3s_pred_split = dataset_config["points_3d"].get("split")
            print(f"  Pred point map:       {p3s_pred_path}")
            points_3d_sparse_dataset, _ = _build_eval_dataset(
                build_points_3d_sparse_eval_dataset,
                dict(
                    gt_sparse_depth_path=gt_sparse_depth_path,
                    pred_points_3d_path=p3s_pred_path,
                    intrinsics_path=intrinsics_path,
                    camera_extrinsics_path=camera_extrinsics_path,
                    lidar_extrinsics_path=lidar_extrinsics_path,
                    segmentation_path=segmentation_path,
                    segmentation_modality_key=segmentation_key or "segmentation",
                    gt_sparse_depth_split=gt_sparse_depth_split,
                    pred_points_3d_split=p3s_pred_split,
                    intrinsics_split=intrinsics_split,
                    camera_extrinsics_split=camera_extrinsics_split,
                    lidar_extrinsics_split=lidar_extrinsics_split,
                    segmentation_split=segmentation_split,
                ),
                ds_name=ds_name,
                missing_segmentation=args.missing_segmentation,
            )
            p3ds_meta = get_points_3d_metadata(points_3d_sparse_dataset)
            p3s_alignment_mode = args.points_3d_alignment
            et_eval_datasets["points_3d_sparse"] = points_3d_sparse_dataset
            print(f"  Matched pairs: {len(points_3d_sparse_dataset)}")

            points_3d_sparse_results = evaluate_points_3d_sparse_samples(
                distribution_config=distribution_config,
                dataset=points_3d_sparse_dataset,
                pred_is_radial=True,
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
                sky_mask_enabled=args.mask_sky,
                alignment_mode=p3s_alignment_mode,
                input_space_hint=None,
                pred_is_depth=False,
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, modality="points_3d")

            p3s_space_info = points_3d_sparse_results.get("space_info", {})
            p3s_dataset_info = points_3d_sparse_results.get("dataset_info", {})
            p3s_spatial = points_3d_sparse_results.get("spatial_info", {})

            points_3d_sparse_ns = _EvalNamespace(
                producer="euler-eval",
                producer_version=_get_version(),
                modalities=("points_3d",),
                axes=_points_3d_eval_axes(distributions=distribution_config is not None),
                descriptions=_distribution_descriptions(
                    _POINTS_3D_EVAL_DESCRIPTIONS, distribution_config, "rmse3d"
                ),
            )
            points_3d_sparse_save = {
                "metricSet": _points_3d_metric_set_envelope(
                    points_3d_sparse_ns,
                    metadata={
                        **_distribution_metadata(points_3d_sparse_results),
                        "input_space_detected": p3s_space_info.get(
                            "input_space_detected", "unknown"
                        ),
                        "metric_space_source": p3s_space_info.get(
                            "metric_space_source"
                        ),
                        "calibration_mode": p3s_space_info.get(
                            "calibration_mode", "unknown"
                        ),
                        "calibration_applied": p3s_space_info.get(
                            "calibration_applied", False
                        ),
                        "emitted_spaces": p3s_space_info.get("emitted_spaces", []),
                        "canonical_space": p3s_space_info.get(
                            "canonical_space", "native"
                        ),
                        "gt_representation": "point_cloud",
                        "fov_domain": p3s_dataset_info.get("fov_domain"),
                        "threshold_deg": p3s_dataset_info.get("threshold_deg"),
                    },
                ),
                "protocol": build_protocol(
                    protocol_identities,
                    gt="sparse_depth",
                    inputs=(
                        "intrinsics",
                        "camera_extrinsics",
                        "lidar_extrinsics",
                        "segmentation",
                    ),
                    population={
                        "samples": p3s_dataset_info.get("num_pairs"),
                        "gt_sky_masking": bool(args.mask_sky),
                        "missing_segmentation": (
                            args.missing_segmentation if args.mask_sky else None
                        ),
                        "evaluated_resolution": evaluated_resolution(p3s_spatial),
                    },
                    method={
                        "points_3d_alignment": p3s_alignment_mode,
                    },
                ),
                "dataset_info": p3s_dataset_info,
                "meta": _clean_metric_tree({
                    "version": _get_version(),
                    "eval_name": args.eval_name,
                    "modality": "points_3d",
                    "device": args.device,
                    "gt": {
                        "path": gt_sparse_depth_path,
                        "split": gt_sparse_depth_split,
                        "representation": "point_cloud",
                    },
                    "pred": {
                        "path": p3s_pred_path,
                        "split": p3s_pred_split,
                        "entry": "points_3d",
                        "representation": "points_3d",
                        "dimensions": p3s_spatial.get("pred_dimensions"),
                    },
                    "calibration": {
                        "intrinsics_path": intrinsics_path,
                        "intrinsics_split": intrinsics_split,
                        "camera_extrinsics_path": camera_extrinsics_path,
                        "camera_extrinsics_split": camera_extrinsics_split,
                        "lidar_extrinsics_path": lidar_extrinsics_path,
                        "lidar_extrinsics_split": lidar_extrinsics_split,
                    },
                    "spatial_alignment": {
                        "method": p3s_spatial.get("method", "pointcloud_projection"),
                        "evaluated_dimensions": p3s_spatial.get(
                            "evaluated_dimensions"
                        ),
                    },
                    "modality_params": p3ds_meta,
                    "gt_sky_masking": points_3d_sparse_results.get("gt_sky_masking"),
                    "eval_params": {
                        "sky_masking": args.mask_sky,
                        "alignment_mode": p3s_alignment_mode,
                        "num_workers": args.num_workers,
                    },
                }),
                "points3d": {"eval": {}},
            }
            for space_name, result_key in (
                ("native", "points_3d_native"),
                ("metric", "points_3d_metric"),
            ):
                branch = points_3d_sparse_results.get(result_key)
                if branch is not None:
                    points_3d_sparse_save["points3d"]["eval"][space_name] = (
                        _clean_metric_tree(branch)
                    )
            p3s_pfm = points_3d_sparse_results.get("per_file_metrics", {})
            if p3s_pfm:
                canonical_space = p3s_space_info.get("canonical_space", "native")
                points_3d_sparse_save["per_file_metrics"] = _clean_per_file_metrics(
                    _wrap_pfm_metrics(
                        p3s_pfm,
                        lambda m: _wrap_depth_space_pfm_metrics(
                            m,
                            metric_root="points3d",
                            canonical_key="points_3d",
                            canonical_space=canonical_space,
                        ),
                    ),
                    label="points_3d sparse per_file_metrics",
                )
            all_results.setdefault("per_file_metrics", {}).update(p3s_pfm)

            print_results(
                {k: v for k, v in points_3d_sparse_results.items()
                 if k not in ("per_file_metrics", "spatial_info")},
                f"POINTS_3D · SPARSE: {ds_name}",
            )

        # Save per-modality results to respective dataset paths
        if depth_save:
            depth_out = save_results(
                depth_save,
                dataset_config,
                modality=pred_depth_key or "depth",
                eval_name=args.eval_name,
            )
            depth_label = (
                "Sparse depth"
                if _SPARSE_DEPTH_METRIC_ROOT in depth_save
                else "Depth"
            )
            print(f"\n  {depth_label} results saved to: {depth_out}")
        if rgb_save:
            rgb_out = save_results(
                rgb_save,
                dataset_config,
                modality="rgb",
                eval_name=args.eval_name,
            )
            print(f"\n  RGB results saved to: {rgb_out}")
        if rays_save:
            rays_out = save_results(
                rays_save,
                dataset_config,
                modality="rays",
                eval_name=args.eval_name,
            )
            print(f"\n  Rays results saved to: {rays_out}")
        if points_3d_save:
            points_3d_out = save_results(
                points_3d_save,
                dataset_config,
                modality="points_3d",
                eval_name=args.eval_name,
            )
            print(f"\n  Points-3D results saved to: {points_3d_out}")
        if points_3d_sparse_save:
            points_3d_sparse_out = save_results(
                points_3d_sparse_save,
                dataset_config,
                modality="points_3d",
                eval_name=args.eval_name,
            )
            print(
                f"\n  Points-3D (sparse GT) results saved to: {points_3d_sparse_out}"
            )

        # Log to euler_train
        if et_run is not None and et_eval_datasets:
            et_run.add_evaluation(
                et_key,
                datasets=et_eval_datasets,
                # name=ds_name,
                # status="completed",
                metadata={
                    "package": package_metadata,
                    "results": {
                        k: v for k, v in all_results.items() if k != "per_file_metrics"
                    }
                },
            )
            et_run.finish_evaluation(et_key)

    # Print sanity check report at the end
    if sanity_checker is not None:
        sanity_checker.print_report()

        report = sanity_checker.get_full_report()
        report_path = Path(
            "sanity_check_report.json"
            if experiment is None
            else f"sanity_check_report_{experiment}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSanity check report saved to: {report_path}")


def main():
    parser = _build_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    try:
        _finish_args(args)
    except ValueError as e:
        parser.error(str(e))

    # Emit producer provenance before configuration loading or evaluation work.
    package_metadata = _log_package_metadata()

    # Load and validate config
    try:
        config = load_config(args.config)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    experiments = config.get("experiments")
    if experiments is None:
        if args.experiments:
            parser.error(
                "--experiment selects from the config's 'experiments' section, "
                "which this config does not have"
            )
        try:
            runs = [_prepare_run(args, config)]
        except ValueError as e:
            parser.error(str(e))
    else:
        if args.eval_name != DEFAULT_EVAL_NAME:
            parser.error(
                "--eval-name cannot be combined with the config's 'experiments': "
                "each experiment's 'name' names its results"
            )
        names = [validate_eval_name(entry["name"]) for entry in experiments]
        unknown = [
            name for name in (args.experiments or []) if name not in names
        ]
        if unknown:
            parser.error(
                f"--experiment {', '.join(map(repr, unknown))} is not in the "
                f"config's experiments ({', '.join(names)})"
            )
        try:
            runs = _experiment_runs(parser, argv, config, selected=args.experiments)
        except ValueError as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

    et_run, et_resuming = _start_euler_train(config)
    failed = []
    if experiments is None:
        _evaluate(runs[0], package_metadata=package_metadata, et_run=et_run)
    else:
        failed = _evaluate_experiments(
            runs, package_metadata=package_metadata, et_run=et_run
        )
    _finish_euler_train(et_run, et_resuming)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
