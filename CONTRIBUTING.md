# Contributing

Bug reports, new metrics and documentation fixes are all welcome.

## Development setup

```bash
git clone https://github.com/d-rothen/euler-eval.git
cd euler-eval
uv sync --extra dev          # or: pip install -e ".[dev]"
```

The released package depends on the PyPI releases of the companion packages
([euler-loading](https://github.com/d-rothen/euler-loading),
[ds-crawler](https://github.com/d-rothen/ds-crawler),
[euler-metric-naming](https://github.com/d-rothen/euler-metric-naming)). For
development, `[tool.uv.sources]` resolves them from GitHub instead, so a
`uv sync` tracks the ecosystem's main branches.

## Running tests

```bash
uv run pytest
```

The suite is fully mocked — no dataset, GPU or network access is required, and
it should stay that way. The root `conftest.py` stubs heavy optional
dependencies (`lpips`, `torchvision`) so the tests also run in a lean
environment where those are missing.

## Style

```bash
uvx ruff check .
```

Line length is 88 (`black`-compatible). The naming rules `N803`/`N806`/`N812`
are disabled on purpose: camera matrices and transforms keep their conventional
mathematical names (`K`, `R`, `T`, `P`), as does `torch.nn.functional as F`.

## Adding a metric

1. **Implement it** in `euler_eval/metrics/`, as a pure function over numpy
   arrays where possible — that keeps it usable from both the CLI evaluators and
   the [programmatic validation API](docs/validation.md).
2. **Export it** from `euler_eval/metrics/__init__.py`, adding it to `__all__`
   under the right group.
3. **Wire core metrics into the evaluator** in `euler_eval/evaluate.py`, and
   make sure the value lands under the correct metric key. Domain-specific,
   no-reference RGB metrics should instead be declared in
   `euler_eval/metric_sets.py`; the shared evaluator loop will compute and
   aggregate them.
4. **Declare it** in `euler_eval/cli.py`: add a `MetricDescription` (unit,
   direction, bounds) to the relevant `_*_EVAL_DESCRIPTIONS` mapping, and extend
   the axis declarations if the metric introduces a new category. This is what
   makes the metric self-describing to euler-train and euler-view — an undeclared
   key still serializes, but downstream tooling cannot interpret it.
5. **Document it** in [`docs/metrics.md`](docs/metrics.md) with the key it is
   written under.
6. **Add tests**, including the aggregate path and the empty/degenerate input
   case.

If the metric needs a sanity threshold, add it to
[`metrics_config.json`](metrics_config.json) with a `description` explaining
what the check catches, and a validator in `euler_eval/sanity_checker.py`.

New domains belong in `DOMAIN_METRIC_SETS`. Give each one a unique result
category and describe every contributed metric in its `MetricSet`; this makes
the repeatable `--domain` flag, namespace axis, metric descriptions, and output
aggregation extend together rather than through separate CLI conditionals.

## Documentation

The README is the shop window: what the package does, how it fits the Euler
ecosystem, and the shortest path to a first result. Reference material —
every flag, every metric key, every config field — belongs in
[`docs/`](docs/README.md), linked from the README's documentation table.

## Releasing

1. Bump `version` in `pyproject.toml` and refresh the lock (`uv lock`).
2. Add a [`CHANGELOG.md`](CHANGELOG.md) entry. Call out anything that moves
   metric values explicitly — results are compared across runs and releases.
3. Tag the commit `v<version>` and push the tag.

Pushing a `v*` tag runs
[`.github/workflows/workflow.yml`](.github/workflows/workflow.yml), which builds
with `uv build` and publishes to PyPI over OIDC — no API token required.
