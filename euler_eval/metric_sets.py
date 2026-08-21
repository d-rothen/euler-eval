"""Composable core and domain-specific evaluation metric sets.

The evaluator's established metrics form the always-enabled ``core`` set.
Domains add narrowly scoped metrics without changing that baseline.  Keeping
selection here prevents the CLI and evaluation loop from accumulating
domain-specific conditionals as new domains are introduced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .metrics.fade import compute_fade
from .metrics.niqe import compute_niqe


@dataclass(frozen=True)
class NoReferenceRGBMetric:
    """A scalar RGB metric that scores a prediction without ground truth."""

    name: str
    display_name: str
    compute: Callable[[np.ndarray], float]
    is_higher_better: bool


@dataclass(frozen=True)
class MetricSet:
    """Metrics contributed by a named core or domain set."""

    name: str
    category: str | None = None
    rgb_no_reference: tuple[NoReferenceRGBMetric, ...] = ()


CORE_METRIC_SET = MetricSet(name="core")

DEHAZING_METRIC_SET = MetricSet(
    name="dehazing",
    category="dehazing",
    rgb_no_reference=(
        NoReferenceRGBMetric(
            name="niqe",
            display_name="NIQE",
            compute=compute_niqe,
            is_higher_better=False,
        ),
        NoReferenceRGBMetric(
            name="fade",
            display_name="FADE",
            compute=compute_fade,
            is_higher_better=False,
        ),
    ),
)

DOMAIN_METRIC_SETS: dict[str, MetricSet] = {
    DEHAZING_METRIC_SET.name: DEHAZING_METRIC_SET,
}


def available_domains() -> tuple[str, ...]:
    """Return domain names accepted by the CLI in stable order."""
    return tuple(sorted(DOMAIN_METRIC_SETS))


def resolve_metric_sets(
    domains: Iterable[str] | str | None = None,
) -> tuple[MetricSet, ...]:
    """Resolve additive metric sets, always placing ``core`` first.

    Repeated domains are de-duplicated while preserving their first occurrence.
    Unknown names raise a user-facing error rather than silently selecting no
    metrics.
    """
    if domains is None:
        requested: tuple[str, ...] = ()
    elif isinstance(domains, str):
        requested = (domains,)
    else:
        requested = tuple(domains)

    resolved = [CORE_METRIC_SET]
    seen: set[str] = set()
    for name in requested:
        if name in seen:
            continue
        try:
            metric_set = DOMAIN_METRIC_SETS[name]
        except KeyError as exc:
            choices = ", ".join(available_domains()) or "none"
            raise ValueError(
                f"Unknown evaluation domain {name!r}. Available domains: {choices}."
            ) from exc
        resolved.append(metric_set)
        seen.add(name)
    return tuple(resolved)


def selected_domain_categories(domains: Iterable[str] | str | None) -> tuple[str, ...]:
    """Return result categories contributed by the selected domains."""
    return tuple(
        metric_set.category
        for metric_set in resolve_metric_sets(domains)
        if metric_set.category is not None
    )
