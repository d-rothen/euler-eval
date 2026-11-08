"""Composable core and domain-specific evaluation metric sets.

The evaluator's established metrics form the always-enabled ``core`` set.
Domains add narrowly scoped metrics without changing that baseline.  Keeping
selection here prevents the CLI and evaluation loop from accumulating
domain-specific conditionals as new domains are introduced.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .metrics.fade import compute_fade
from .metrics.learned_iqa import load_clipiqa, load_musiq
from .metrics.niqe import compute_niqe


@dataclass(frozen=True)
class NoReferenceRGBMetric:
    """A scalar RGB metric that scores a prediction without ground truth.

    Give exactly one of ``compute``, a stateless per-image function, or
    ``load``, which is called once per evaluation with the torch device and
    returns the per-image function (a network with its weights on that
    device, say). ``requires`` lists optional packages, by a name that is both
    their distribution and their import name, that the metric needs, so a
    missing one is reported before evaluation starts. ``min_value`` and
    ``max_value`` are hard bounds of the score, when it has them.
    """

    name: str
    display_name: str
    is_higher_better: bool
    compute: Callable[[np.ndarray], float] | None = None
    load: Callable[[str], Callable[[np.ndarray], float]] | None = None
    requires: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self) -> None:
        if (self.compute is None) == (self.load is None):
            raise ValueError(
                f"Metric {self.name!r} needs exactly one of 'compute' or 'load'"
            )

    def bind(self, device: str) -> Callable[[np.ndarray], float]:
        """Return the per-image scorer for one evaluation on ``device``."""
        if self.load is not None:
            return self.load(device)
        return self.compute


@dataclass(frozen=True)
class MetricSet:
    """Metrics contributed by a named core or domain set.

    ``extra`` is the euler-eval pip extra that installs the optional packages
    its metrics require.
    """

    name: str
    category: str | None = None
    rgb_no_reference: tuple[NoReferenceRGBMetric, ...] = ()
    extra: str | None = None


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
        NoReferenceRGBMetric(
            name="musiq",
            display_name="MUSIQ",
            load=load_musiq,
            requires=("pyiqa",),
            is_higher_better=True,
        ),
        NoReferenceRGBMetric(
            name="clipiqa",
            display_name="CLIP-IQA",
            load=load_clipiqa,
            requires=("pyiqa",),
            is_higher_better=True,
            min_value=0.0,
            max_value=1.0,
        ),
    ),
    extra="dehazing",
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


def _required_packages(metric_sets: Iterable[MetricSet]) -> list[str]:
    return sorted(
        {
            package
            for metric_set in metric_sets
            for metric in metric_set.rgb_no_reference
            for package in metric.requires
        }
    )


def check_metric_set_requirements(metric_sets: Iterable[MetricSet]) -> None:
    """Fail before any work when a selected metric needs a missing package.

    Raises:
        ValueError: Naming the domain, its metrics, the missing package and
            how to install it.
    """
    problems = []
    for metric_set in metric_sets:
        missing: dict[str, list[str]] = {}
        for metric in metric_set.rgb_no_reference:
            for package in metric.requires:
                if importlib.util.find_spec(package) is None:
                    missing.setdefault(package, []).append(metric.display_name)
        for package, metric_names in missing.items():
            install = (
                f'pip install "euler-eval[{metric_set.extra}]"'
                if metric_set.extra
                else f"pip install {package}"
            )
            problems.append(
                f"--domain {metric_set.name} needs the '{package}' package for "
                f"{', '.join(metric_names)}, but it is not installed. "
                f"Install it with `{install}`."
            )
    if problems:
        raise ValueError(" ".join(problems))


def metric_set_package_versions(
    metric_sets: Iterable[MetricSet],
) -> dict[str, str | None]:
    """Return installed versions of the optional packages the sets use."""
    versions: dict[str, str | None] = {}
    for package in _required_packages(metric_sets):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions
