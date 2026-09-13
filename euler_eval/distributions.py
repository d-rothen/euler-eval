"""Composable, streaming error histograms with shared, JSON-safe bin definitions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

# Counts must survive JSON.parse and arithmetic in JavaScript consumers exactly.
MAX_SAFE_COUNT = 2**53 - 1


@dataclass(frozen=True)
class DistributionConfig:
    """Bin settings shared by every sample and semantic space in an evaluation.

    ``n_bins`` includes the first bin containing zero and the final overflow
    bin. Positive ``min_error`` reserves a bin for ``[0, min_error)``; the
    remaining finite bins are spaced between ``min_error`` and ``max_error``.
    Linear spacing also accepts ``min_error=0``.
    """

    n_bins: int = 50
    scale: str = "log"
    min_error: float = 1e-3
    max_error: float = 100.0

    def __post_init__(self) -> None:
        if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, int):
            raise ValueError("distribution n_bins must be an integer >= 3")
        if self.n_bins < 3:
            raise ValueError("distribution n_bins must be an integer >= 3")
        if self.scale not in ("log", "linear"):
            raise ValueError("distribution scale must be 'log' or 'linear'")
        for name in ("min_error", "max_error"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"distribution {name} must be a finite number")
        if not 0 <= self.min_error < self.max_error:
            raise ValueError(
                "distribution bounds must satisfy 0 <= min_error < max_error"
            )
        if self.scale == "log" and self.min_error == 0:
            raise ValueError("log distribution min_error must be positive")
        if not np.all(np.diff(self.bin_edges()) > 0):
            raise ValueError("distribution bounds are too close for distinct bin edges")

    @classmethod
    def from_config(cls, value) -> DistributionConfig | None:
        """Resolve the optional JSON ``distributions`` section."""
        if value is False or value is None:
            return None
        if value is True:
            return cls()
        if not isinstance(value, dict):
            raise ValueError(
                "distributions must be a boolean or an object of bin settings"
            )
        unknown = value.keys() - {"n_bins", "scale", "min_error", "max_error"}
        if unknown:
            raise ValueError(
                f"Unknown distributions settings: {', '.join(sorted(unknown))}"
            )
        return cls(**value)

    def bin_edges(self) -> np.ndarray:
        """Return strictly increasing edges covering all nonnegative errors."""
        count = self.n_bins - int(self.min_error > 0)
        spacing = np.geomspace if self.scale == "log" else np.linspace
        edges = spacing(self.min_error, self.max_error, count, dtype=np.float64)
        if self.min_error > 0:
            edges = np.concatenate(([0.0], edges))
        return np.concatenate((edges, [np.inf]))

    def binning_description(self) -> dict:
        """Describe reusable numeric intervals, independently of metrics/units."""
        return {
            "binEdges": self.bin_edges()[:-1].tolist() + [None],
            "interval": "[left, right)",
            "spacing": self.scale,
        }


def bin_values(
    values: np.ndarray,
    bin_edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Count finite values, or sum weights, in left-closed intervals.

    Separating the binned values from optional weights lets a future caller
    bin GT depth and sum error magnitude using the same primitive. Values
    outside the edges and non-finite value/weight pairs are excluded.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be a strictly increasing 1D array")
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & (values >= edges[0]) & (values < edges[-1])
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape != values.shape:
            raise ValueError("weights and values must have the same size")
        valid &= np.isfinite(weights)
        weights = weights[valid]
    indices = np.searchsorted(edges, values[valid], side="right") - 1
    return np.bincount(indices, weights=weights, minlength=edges.size - 1)


class ErrorDistribution:
    """Accumulate counts in O(n_bins) memory while returning each sample's bins."""

    def __init__(self, config: DistributionConfig, metric: str = "rmse"):
        self.metric = metric
        self.edges = config.bin_edges()
        self.counts = np.zeros(config.n_bins, dtype=np.int64)

    def add(self, errors: np.ndarray) -> dict:
        counts = bin_values(errors, self.edges)
        if sum(map(int, self.counts)) + sum(map(int, counts)) > MAX_SAFE_COUNT:
            raise OverflowError(
                "distribution count exceeds the JSON safe integer limit"
            )
        self.counts += counts
        return {self.metric: {"distribution": counts.tolist()}}

    def summary(self) -> dict:
        return {"pixel_pool": {self.metric: {"distribution": self.counts.tolist()}}}


def depth_error_magnitudes(
    pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """Per-pixel RMSE magnitudes with the standard depth metric validity rules.

    For a single scalar pixel, sqrt(squared error) is absolute error. Taking
    the difference in float64 avoids overflow from squaring float32 errors.
    """
    if valid_mask is None:
        valid_mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
    return np.abs(
        pred[valid_mask].astype(np.float64) - gt[valid_mask].astype(np.float64)
    )


def point_error_magnitudes(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Euclidean errors of already filtered 3D correspondences, in float64.

    The scalar point metrics may downcast their intermediate errors to float32.
    Keep large finite errors in the histogram's overflow bin instead of losing
    them to that cast or to overflow when squaring coordinates.
    """
    delta = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    return np.hypot.reduce(delta, axis=-1)
