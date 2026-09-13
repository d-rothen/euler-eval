"""Regenerate small, synthetic eval.json documents for consumer conformance.

Run from the repository root: python scripts/generate_distribution_fixtures.py
These are contract examples, not results from a trained model. Evaluator/CLI
integration tests separately verify real output against the same schema.
"""

import copy
import json
from pathlib import Path

from euler_eval import cli
from euler_eval.distribution_contract import distribution_metadata
from euler_eval.distributions import DistributionConfig

DESTINATION = Path(__file__).resolve().parents[1] / "tests/fixtures/distributions"


def count_document(root):
    points = root == "points3d"
    metric = "rmse3d" if points else "rmse"
    axes = (
        cli._points_3d_eval_axes
        if points
        else cli._sparse_depth_eval_axes
        if root == "sparsedepth"
        else cli._depth_eval_axes
    )
    namespace = cli._EvalNamespace(
        producer="euler-eval",
        producer_version="contract-fixture-v1",
        modalities=("points_3d" if points else root,),
        axes=axes(distributions=True),
        descriptions=cli._distribution_descriptions({}, DistributionConfig(), metric),
    )
    envelope = namespace.metric_set_envelope("points_3d" if points else root)
    envelope["metricNamespace"] = f"{root}.eval"
    envelope["metadata"] = {
        "input_space_detected": "relative",
        "calibration_applied": True,
        "calibration_mode": "scale" if points else "affine",
        "emitted_spaces": ["native", "metric"],
        "canonical_space": "metric",
        "distributions": distribution_metadata(
            DistributionConfig(n_bins=4, scale="linear", min_error=0, max_error=3),
            metric,
            ["native", "metric"],
            "relative",
        ),
    }

    def metrics(space, counts, pooled=False):
        histogram = {
            metric: {
                "distribution": {
                    "type": "distribution",
                    "definitionId": f"{metric}_{space}_error_count",
                    "values": counts,
                }
            }
        }
        return {
            "distributions": {"pixel_pool": histogram} if pooled else histogram,
            "point_error" if points else "standard" if pooled else "depth_metrics": (
                {"pixel_pool": {metric: 2.0}} if pooled else {metric: 2.0}
            ),
        }

    def file_entry(identifier, native, metric_counts):
        return {
            "id": identifier,
            "metrics": {
                root: {
                    "eval": {
                        "native": metrics("native", native),
                        "metric": metrics("metric", metric_counts),
                    }
                }
            },
        }

    return {
        "metricSet": envelope,
        "dataset_info": {
            "num_pairs": 3,
            "gt_name": "synthetic",
            "pred_name": "contract-fixture",
        },
        root: {
            "eval": {
                "native": metrics("native", [0, 0, 0, 6], pooled=True),
                "metric": metrics("metric", [2, 2, 1, 1], pooled=True),
            }
        },
        "per_file_metrics": {
            "children": {
                "scene": {
                    "children": {
                        "left": {
                            "files": [file_entry("frame_0", [0, 0, 0, 4], [2, 1, 0, 1])]
                        },
                        "right": {
                            "files": [
                                file_entry("frame_0", [0, 0, 0, 2], [0, 1, 1, 0]),
                                file_entry("empty", [0, 0, 0, 0], [0, 0, 0, 0]),
                            ]
                        },
                    }
                }
            }
        },
    }


def write(name, document):
    (DESTINATION / name).write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n"
    )


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for root in ("depth", "sparsedepth", "points3d"):
        write(f"counts-{root}.eval.json", count_document(root))

    document = count_document("depth")
    document["depth"]["eval"] = {
        space: {"distributions": metrics["distributions"]}
        for space, metrics in document["depth"]["eval"].items()
    }
    document.pop("per_file_metrics")
    write("histogram-only.eval.json", document)
    document = count_document("depth")
    document.pop("depth")
    write("per-file-only.eval.json", document)

    document = count_document("depth")
    document["metricSet"]["metadata"].pop("distributions")
    document["depth"]["eval"] = {"metric": {"standard": {"pixel_pool": {"rmse": 2.0}}}}
    document.pop("per_file_metrics")
    write("scalar-only.eval.json", document)

    # A second view of the same metric: supported by v1, not emitted by the CLI.
    document = count_document("depth")
    registry = document["metricSet"]["metadata"]["distributions"]
    registry["binSets"]["gt_depth"] = {
        "binEdges": [0, 2, 5, None],
        "interval": "[left, right)",
        "spacing": "explicit",
    }
    definition_id = "rmse_metric_depth_error_sum"
    definition = copy.deepcopy(registry["definitions"]["rmse_metric_error_count"])
    definition.update(
        binSetId="gt_depth",
        axes={
            "x": {"quantity": "gt_depth", "unit": "m"},
            "y": {"quantity": "absolute_error", "statistic": "sum", "unit": "m"},
        },
    )
    registry["definitions"][definition_id] = definition
    document["metricSet"]["metricDescriptions"]["rmse.distribution_by_depth"] = {
        "unit": "m",
        "displayName": "Summed absolute error by GT depth",
    }

    def value(bins):
        return {"type": "distribution", "definitionId": definition_id, "values": bins}

    document["depth"]["eval"]["metric"]["distributions"]["pixel_pool"]["rmse"][
        "distribution_by_depth"
    ] = value([0.75, 2.5, 10.0])
    children = document["per_file_metrics"]["children"]["scene"]["children"]
    entries = children["left"]["files"] + children["right"]["files"]
    for entry, bins in zip(
        entries, ([0.25, 0.5, 10.0], [0.5, 2.0, 0.0], [0.0, 0.0, 0.0])
    ):
        entry["metrics"]["depth"]["eval"]["metric"]["distributions"]["rmse"][
            "distribution_by_depth"
        ] = value(bins)
    write("future-sum-by-depth.eval.json", document)


if __name__ == "__main__":
    main()
