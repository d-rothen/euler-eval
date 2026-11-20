"""Metric names of depth evaluations: the eval namespace, axes and descriptions.

Shared by the CLI (``eval.json``) and in-training validation sets
(:mod:`euler_eval.validation`), so a validation pass and the CLI's results
file name the same metric cell with the same string and the contract engine
compares them without any mapping.
"""

from __future__ import annotations

from euler_metric_naming import AxisDeclaration, MetricDescription, MetricNamespace

__all__ = [
    "BENCHMARK_BIN_AXIS",
    "DEPTH_EVAL_DESCRIPTIONS",
    "SPARSE_DEPTH_EVAL_DESCRIPTIONS",
    "SPARSE_DEPTH_METRIC_NAMESPACE",
    "SPARSE_DEPTH_METRIC_ROOT",
    "EvalNamespace",
    "depth_eval_axes",
    "sparse_depth_eval_axes",
]

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

# Downstream eval consumers validate metricSet.metricNamespace with a stricter
# first-segment rule than euler_metric_naming's modality validator. Keep the
# public Python sparse_depth result keys, but use a wire-compatible metric root
# in eval.json so flattened metric paths live under the declared namespace.
_SPARSE_DEPTH_METRIC_ROOT = "sparsedepth"
_SPARSE_DEPTH_METRIC_NAMESPACE = f"{_SPARSE_DEPTH_METRIC_ROOT}.eval"

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


# Public names.
EvalNamespace = _EvalNamespace
BENCHMARK_BIN_AXIS = _BENCHMARK_BIN_AXIS
depth_eval_axes = _depth_eval_axes
sparse_depth_eval_axes = _sparse_depth_eval_axes
SPARSE_DEPTH_METRIC_ROOT = _SPARSE_DEPTH_METRIC_ROOT
SPARSE_DEPTH_METRIC_NAMESPACE = _SPARSE_DEPTH_METRIC_NAMESPACE
DEPTH_EVAL_DESCRIPTIONS = _DEPTH_EVAL_DESCRIPTIONS
SPARSE_DEPTH_EVAL_DESCRIPTIONS = _SPARSE_DEPTH_EVAL_DESCRIPTIONS
