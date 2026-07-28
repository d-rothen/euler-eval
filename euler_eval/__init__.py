"""Depth evaluation package.

Public programmatic API (lazily imported so that ``import euler_eval`` stays
cheap; heavy metric backends only load on first use):

* :mod:`euler_eval.validation` — in-training/programmatic depth validation
  (``evaluate_dense_depth_sample``, ``evaluate_sparse_depth_sample``,
  ``DepthValidationAggregator``, ``build_validation_gt_dataset``).
* :mod:`euler_eval.calibration` — calibration extraction from euler-loading
  samples.
* :mod:`euler_eval.metrics` — pure-array metric functions.
* :mod:`euler_eval.data` — dataset builders, converters, projection, and
  alignment helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_VALIDATION_EXPORTS = (
    "BENCHMARK_DEPTH_BIN_NAMES",
    "DepthBenchmarkEvaluation",
    "DepthSampleEvaluation",
    "DepthValidationAggregator",
    "VALIDATION_ALIGNMENT_MODES",
    "build_validation_gt_dataset",
    "evaluate_dense_depth_sample",
    "evaluate_sparse_depth_sample",
    "summarize_reduced_state",
)
_CALIBRATION_EXPORTS = (
    "get_sample_intrinsics",
    "get_sample_pointcloud_to_camera_extrinsics",
)

__all__ = [*_VALIDATION_EXPORTS, *_CALIBRATION_EXPORTS]

if TYPE_CHECKING:  # pragma: no cover - static import surface for type checkers
    from .calibration import (  # noqa: F401
        get_sample_intrinsics,
        get_sample_pointcloud_to_camera_extrinsics,
    )
    from .validation import (  # noqa: F401
        BENCHMARK_DEPTH_BIN_NAMES,
        VALIDATION_ALIGNMENT_MODES,
        DepthBenchmarkEvaluation,
        DepthSampleEvaluation,
        DepthValidationAggregator,
        build_validation_gt_dataset,
        evaluate_dense_depth_sample,
        evaluate_sparse_depth_sample,
        summarize_reduced_state,
    )


def __getattr__(name: str):
    if name in _VALIDATION_EXPORTS:
        from . import validation

        return getattr(validation, name)
    if name in _CALIBRATION_EXPORTS:
        from . import calibration

        return getattr(calibration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
