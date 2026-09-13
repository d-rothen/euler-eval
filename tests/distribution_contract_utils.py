"""Checks shared by producer integration and portable fixture tests."""

import json
from importlib.resources import files

from jsonschema import Draft202012Validator

from euler_eval.distribution_contract import validate_distribution_contract

SCHEMA = json.loads(
    files("euler_eval").joinpath("schemas/distribution-v1.schema.json").read_text()
)
REGISTRY_VALIDATOR = Draft202012Validator(SCHEMA)
VALUE_VALIDATOR = Draft202012Validator({**SCHEMA, "$ref": "#/$defs/value"})


def distribution_values(node):
    if isinstance(node, dict):
        if node.get("type") == "distribution":
            yield node
        else:
            for value in node.values():
                yield from distribution_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from distribution_values(value)


def assert_distribution_contract(document):
    values = list(distribution_values(document))
    metadata = document.get("distribution_info") or document["metricSet"].get(
        "metadata", {}
    ).get("distributions")
    if metadata is None:
        assert not values, "Typed distribution leaves require a registry"
        return
    REGISTRY_VALIDATOR.validate(metadata)
    for value in values:
        VALUE_VALIDATOR.validate(value)
    validate_distribution_contract(metadata, values)
