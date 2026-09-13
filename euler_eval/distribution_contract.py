"""Versioned, self-describing distribution values for eval.json consumers.

Computations stay in :mod:`euler_eval.distributions`. This module binds their
arrays to metric/space definitions only after an evaluator resolves its spaces.
The packaged ``schemas/distribution-v1.schema.json`` specifies the JSON shapes;
``validate_distribution_contract`` also checks references and numeric invariants.
"""

from __future__ import annotations

from math import isfinite

from .distributions import MAX_SAFE_COUNT, DistributionConfig

DISTRIBUTION_SCHEMA_VERSION = 1


def _finite_number(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def _definition_id(metric: str, space: str) -> str:
    # Opaque, registry-local identifiers: consumers must resolve, never parse.
    return f"{metric}_{space}_error_count"


def distribution_metadata(
    config: DistributionConfig,
    metric: str,
    spaces: list[str],
    input_space: str,
) -> dict:
    """One set of edges, with separate semantics for each emitted space."""
    if metric not in ("rmse", "rmse3d"):
        raise ValueError(f"Unsupported error distribution metric: {metric}")
    definitions = {}
    for space in spaces:
        if space not in ("native", "metric"):
            raise ValueError(f"Unsupported error distribution space: {space}")
        definitions[_definition_id(metric, space)] = {
            "binSetId": "error_magnitude",
            "metric": metric,
            "space": space,
            "axes": {
                "x": {
                    "quantity": "absolute_error"
                    if metric == "rmse"
                    else "euclidean_error",
                    # Native uncalibrated predictions and metric GT are not
                    # commensurate. Do not claim those differences are meters.
                    "unit": "m"
                    if space == "metric" or input_space == "metric"
                    else "unspecified",
                },
                "y": {"quantity": "observations", "statistic": "count", "unit": "1"},
            },
            "observationUnit": "pixel",
            "merge": "sum",
        }
    return {
        "schemaVersion": DISTRIBUTION_SCHEMA_VERSION,
        "binSets": {"error_magnitude": config.binning_description()},
        "definitions": definitions,
    }


def finalize_distributions(
    result: dict, config: DistributionConfig | None, metric: str, branch: str
) -> dict:
    """Attach the registry and type aggregate/per-file leaves in place.

    Walk only the evaluator's distribution categories. Canonical aliases and
    benchmark slices resolve to the same definitions as their semantic space.
    No metric paths or scalar leaves change, and disabled evaluation is a no-op.
    """
    if config is None:
        return result
    info = result["space_info"]
    result["distribution_info"] = distribution_metadata(
        config, metric, info["emitted_spaces"], info["input_space_detected"]
    )

    def bind_counts(node: dict, definition_id: str) -> None:
        for key, value in node.items():
            if key == "distribution" and isinstance(value, list):
                node[key] = {
                    "type": "distribution",
                    "definitionId": definition_id,
                    "values": value,
                }
            elif isinstance(value, dict) and value.get("type") != "distribution":
                bind_counts(value, definition_id)

    def bind_branch(node: dict | None, space: str) -> None:
        if node is not None and "distributions" in node:
            bind_counts(node["distributions"], _definition_id(metric, space))

    def bind_spaces(node: dict) -> None:
        for space in info["emitted_spaces"]:
            bind_branch(node.get(f"{branch}_{space}"), space)
        bind_branch(node.get(branch), info["canonical_space"])

    bind_spaces(result)
    benchmark = result.get(f"{branch}_benchmark") or {}
    for space in info["emitted_spaces"]:
        for bin_summary in (benchmark.get(space) or {}).values():
            bind_branch(bin_summary, space)

    def bind_files(node: dict) -> None:
        for entry in node.get("files", []):
            bind_spaces(entry["metrics"])
        for child in node.get("children", {}).values():
            bind_files(child)

    bind_files(result.get("per_file_metrics", {}))
    return result


def validate_distribution_contract(metadata: dict, values: list[dict]) -> None:
    """Check semantic invariants beyond the companion JSON Schema.

    Callers must validate shapes with that schema first. This small reference
    validator is also useful for consumer conformance tests. Definition IDs are
    local references, not content hashes or evidence of cross-run compatibility.
    Both count and sum statistics are defined in v1; the CLI emits counts only.
    """
    if (
        type(metadata["schemaVersion"]) not in (int, float)
        or metadata["schemaVersion"] != DISTRIBUTION_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported distribution schemaVersion")
    bin_sets = metadata["binSets"]
    definitions = metadata["definitions"]
    for bin_set in bin_sets.values():
        edges = bin_set["binEdges"]
        finite_edges = edges[:-1] if edges and edges[-1] is None else edges
        if (
            len(edges) < 2
            or not finite_edges
            or any(not _finite_number(edge) for edge in finite_edges)
        ):
            raise ValueError(
                "Only the final bin edge may be null; other edges must be finite"
            )
        if any(right <= left for left, right in zip(finite_edges, finite_edges[1:])):
            raise ValueError("Distribution bin edges must be strictly increasing")
    for definition in definitions.values():
        if definition["binSetId"] not in bin_sets:
            raise ValueError("Unknown distribution binSetId")
        y = definition["axes"]["y"]
        if y["statistic"] == "count" and (
            y["unit"] != "1" or y["quantity"] != "observations"
        ):
            raise ValueError(
                "Count distributions must count observations with unit '1'"
            )
    for value in values:
        if value["definitionId"] not in definitions:
            raise ValueError("Unknown distribution definitionId")
        definition = definitions[value["definitionId"]]
        edges = bin_sets[definition["binSetId"]]["binEdges"]
        bins = value["values"]
        if len(bins) != len(edges) - 1:
            raise ValueError("Distribution values must match the bin count")
        if any(not _finite_number(number) for number in bins):
            raise ValueError("Distribution values must be finite numbers")
        if definition["axes"]["y"]["statistic"] == "count":
            if any(
                number < 0 or number > MAX_SAFE_COUNT or int(number) != number
                for number in bins
            ):
                raise ValueError(
                    "Distribution counts must be nonnegative safe integers"
                )
            if sum(bins) > MAX_SAFE_COUNT:
                raise ValueError(
                    "Distribution total count exceeds the JSON safe integer limit"
                )
