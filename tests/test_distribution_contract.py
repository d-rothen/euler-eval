"""Consumer conformance, reference integrity and precision of distributions."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import ValidationError

from euler_eval.distribution_contract import (
    distribution_metadata,
    finalize_distributions,
    validate_distribution_contract,
)
from euler_eval.distributions import (
    MAX_SAFE_COUNT,
    DistributionConfig,
    ErrorDistribution,
    bin_values,
    point_error_magnitudes,
)

from .distribution_contract_utils import (
    SCHEMA,
    assert_distribution_contract,
    distribution_values,
)

FIXTURES = Path(__file__).parent / "fixtures/distributions"


@pytest.mark.parametrize(
    "path", sorted(FIXTURES.glob("*.eval.json")), ids=lambda path: path.name
)
def test_portable_consumer_fixtures(path):
    document = json.loads(path.read_text())
    assert_distribution_contract(document)
    assert json.loads(json.dumps(document, allow_nan=False)) == document


def test_schema_is_valid():
    from jsonschema import Draft202012Validator

    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize(
    "input_space", ["metric", "relative", "affine", "normalized", "unknown"]
)
@pytest.mark.parametrize("metric", ["rmse", "rmse3d"])
def test_definitions_resolve_units_per_space_and_share_edges(input_space, metric):
    metadata = distribution_metadata(
        DistributionConfig(), metric, ["native", "metric"], input_space
    )
    definitions = list(metadata["definitions"].values())
    assert len(metadata["binSets"]) == 1
    for definition in definitions:
        assert definition["binSetId"] in metadata["binSets"]
        expected = (
            "m"
            if definition["space"] == "metric" or input_space == "metric"
            else "unspecified"
        )
        assert definition["axes"]["x"]["unit"] == expected
        assert definition["axes"]["y"] == {
            "quantity": "observations",
            "statistic": "count",
            "unit": "1",
        }


def test_empty_and_canonical_aliases_have_the_correct_reference():
    config = DistributionConfig(n_bins=3)
    native = {"distributions": ErrorDistribution(config).summary()}
    metric = copy.deepcopy(native)
    result = {
        "depth_native": native,
        "depth_metric": metric,
        "depth": metric,
        "depth_benchmark": {
            "native": {"all": copy.deepcopy(native)},
            "metric": {"all": copy.deepcopy(metric)},
        },
        "space_info": {
            "emitted_spaces": ["native", "metric"],
            "canonical_space": "metric",
            "input_space_detected": "relative",
        },
        "per_file_metrics": {
            "children": {
                "scene": {
                    "files": [
                        {
                            "id": "empty",
                            "metrics": {
                                "depth_native": copy.deepcopy(native),
                                "depth_metric": copy.deepcopy(metric),
                                "depth": copy.deepcopy(metric),
                            },
                        }
                    ]
                }
            }
        },
    }
    finalized = finalize_distributions(result, config, "rmse", "depth")
    assert_distribution_contract(finalized)
    for key in ("depth_native", "depth_metric", "depth"):
        space = "native" if key.endswith("native") else "metric"
        for leaf in distribution_values(finalized[key]):
            assert (
                finalized["distribution_info"]["definitions"][leaf["definitionId"]][
                    "space"
                ]
                == space
            )
    assert finalize_distributions(finalized, config, "rmse", "depth") == finalized


@pytest.mark.parametrize(
    "target,replacement,message",
    [
        (("schemaVersion",), 2, "schemaVersion"),
        (("binSets", "error_magnitude", "binEdges"), [0, 1, 1, 3, None], "increasing"),
        (
            ("binSets", "error_magnitude", "binEdges"),
            [0, None, 2, 3, None],
            "final bin edge",
        ),
        (
            ("binSets", "error_magnitude", "binEdges"),
            [0, 1, 2, float("inf"), None],
            "finite",
        ),
        (("definitions", "rmse_metric_error_count", "binSetId"), "missing", "binSetId"),
        (
            ("definitions", "rmse_metric_error_count", "axes", "y", "unit"),
            "pixels",
            "unit",
        ),
    ],
)
def test_reject_invalid_registry_semantics(target, replacement, message):
    metadata = distribution_metadata(
        DistributionConfig(n_bins=4), "rmse", ["metric"], "metric"
    )
    parent = metadata
    for key in target[:-1]:
        parent = parent[key]
    parent[target[-1]] = replacement
    with pytest.raises(ValueError, match=message):
        validate_distribution_contract(metadata, [])


@pytest.mark.parametrize(
    "change,message",
    [
        ({"definitionId": "missing"}, "definitionId"),
        ({"values": [1, 2]}, "bin count"),
        ({"values": [0, -1, 0, 0]}, "nonnegative"),
        ({"values": [0, 0.5, 0, 0]}, "integers"),
        ({"values": [MAX_SAFE_COUNT + 1, 0, 0, 0]}, "safe integers"),
        ({"values": [MAX_SAFE_COUNT, 1, 0, 0]}, "total count"),
        ({"values": [0, True, 0, 0]}, "finite numbers"),
        ({"values": [0, None, 0, 0]}, "finite numbers"),
        ({"values": [0, float("nan"), 0, 0]}, "finite numbers"),
        ({"values": [0, float("inf"), 0, 0]}, "finite numbers"),
        ({"values": [0, 10**400, 0, 0]}, "finite numbers"),
    ],
)
def test_reject_invalid_values(change, message):
    metadata = distribution_metadata(
        DistributionConfig(n_bins=4), "rmse", ["metric"], "metric"
    )
    value = {
        "type": "distribution",
        "definitionId": "rmse_metric_error_count",
        "values": [0] * 4,
        **change,
    }
    with pytest.raises(ValueError, match=message):
        validate_distribution_contract(metadata, [value])


def test_v1_allows_fractional_sums_and_finite_upper_edge():
    metadata = distribution_metadata(
        DistributionConfig(n_bins=3), "rmse", ["metric"], "metric"
    )
    metadata["binSets"]["error_magnitude"]["binEdges"] = [0, 1, 2, 3]
    definition = metadata["definitions"]["rmse_metric_error_count"]
    definition["axes"]["y"] = {
        "quantity": "absolute_error",
        "statistic": "sum",
        "unit": "m",
    }
    document = {
        "distribution_info": metadata,
        "example": {
            "type": "distribution",
            "definitionId": "rmse_metric_error_count",
            "values": [0.1, 0.5, 1.25],
        },
    }
    assert_distribution_contract(document)


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_nonadditive_statistics_require_a_new_contract(statistic):
    document = json.loads((FIXTURES / "counts-depth.eval.json").read_text())
    document["metricSet"]["metadata"]["distributions"]["definitions"][
        "rmse_metric_error_count"
    ]["axes"]["y"]["statistic"] = statistic
    with pytest.raises(ValidationError):
        assert_distribution_contract(document)


def test_streaming_counts_fail_before_losing_javascript_integer_precision():
    store = ErrorDistribution(DistributionConfig(n_bins=3))
    store.counts[0] = MAX_SAFE_COUNT
    with pytest.raises(OverflowError, match="safe integer"):
        store.add([0])
    assert store.counts.tolist() == [MAX_SAFE_COUNT, 0, 0]


def test_extreme_finite_point_errors_reach_the_overflow_bin():
    pred = np.array([[3e38, 3e38, 3e38], [1, 1, 1]], dtype=np.float32)
    gt = np.ones_like(pred)
    errors = point_error_magnitudes(pred, gt)
    assert errors.dtype == np.float64 and np.isfinite(errors).all()
    assert bin_values(errors, [0, 1, 3, np.inf]).tolist() == [1, 0, 1]
