# Error distributions

Enable error histograms for every evaluated dataset and sample:

```bash
euler-eval config.json --distributions --distribution-bins 50
```

Depth and sparse-depth evaluation emit `rmse.distribution`; dense and sparse
point-map evaluation emit `rmse3d.distribution`. RGB and ray metrics currently
have no distribution output. The option is off by default and adds a
`distributions` category without changing existing scalar metric paths or values.

## What is counted

The x-axis is **per-pixel error magnitude**. For scalar depth, the RMSE
contribution of one pixel is `sqrt((pred - gt)^2) = abs(pred - gt)`. For 3D
points, it is the Euclidean distance `norm(pred - gt, 2)`. This differs from a
histogram of per-image RMSE scores or of squared errors.

The y-axis is **count of valid pixels**. A dataset histogram is the elementwise
sum of its per-sample histograms in the same semantic space. Larger images and
images with more valid observations contribute more counts; there is no
image-mean or image-median histogram reduction.

Histograms use the same spatial alignment, depth conversion, validity masks,
sky masking, depth caps and calibration/gauge spaces as the corresponding
metrics. Sparse evaluation counts projected, evaluated pixel correspondences,
after projection visibility filtering, rather than every source cloud point.
An image with no valid observations emits an array of zeros. Non-finite error
values are excluded.

## Bin settings

Settings can live in a top-level `distributions` section of `config.json`:

```json
{
  "distributions": {
    "n_bins": 50,
    "scale": "log",
    "min_error": 0.001,
    "max_error": 100.0
  }
}
```

This is a config fragment; the usual `gt` and `datasets` sections are still
required. `"distributions": true` or `{}` enables the defaults; `false` or an
omitted section disables histograms.

| Setting | Default | Meaning |
|---|---|---|
| `n_bins` | `50` | Total array length, including the zero/small-error and overflow bins; integer ≥ 3 |
| `scale` | `"log"` | `"log"` or `"linear"` spacing of finite bins |
| `min_error` | `0.001` | First positive edge; must be positive for log spacing and nonnegative for linear spacing |
| `max_error` | `100.0` | Start of the overflow bin; finite and greater than `min_error` |

All samples, semantic spaces and datasets in a run share these numeric edges.
Each space has its own definition: metric errors use `m` under the evaluator's
meter-based GT convention; native errors also use `m` when the input was
resolved as metric. Uncalibrated native comparisons use `unspecified` because
relative/normalized predictions and metric GT are not commensurate. Their
numerical differences must not be presented as meters or merged across runs.

With the defaults, bin 0 covers `[0, 0.001)`, 48 logarithmically spaced bins
cover `[0.001, 100)`, and the last bin covers `[100, infinity)`. Zero errors are
retained despite logarithmic spacing. All bins are left-inclusive and
right-exclusive, so a value exactly on an interior edge goes into the bin to
its right. For linear spacing with `min_error=0`, the finite bins start at zero
directly, followed by the overflow bin.

CLI settings override the corresponding config fields and also enable the
feature:

```bash
euler-eval config.json --distribution-bins 32 --distribution-range 0.001 80
euler-eval config.json --distribution-scale linear --distribution-range 0 10
euler-eval config.json --no-distributions
```

`--no-distributions` disables histograms even if the config or other histogram
flags enable them. Fixed edges keep counts comparable across runs and allow
exact streaming aggregation without storing all errors or rescanning a dataset.

## JSON paths and definitions

| Scope | Example metric path |
|---|---|
| Dataset, depth | `depth.eval.metric.distributions.pixel_pool.rmse.distribution` |
| Per-file, depth | `depth.eval.metric.distributions.rmse.distribution` |
| Dataset, sparse depth | `sparsedepth.eval.metric.distributions.pixel_pool.rmse.distribution` |
| Dataset, point maps | `points3d.eval.native.distributions.pixel_pool.rmse3d.distribution` |
| Per-file, point maps | `points3d.eval.native.distributions.rmse3d.distribution` |

Per-file paths are inside each file entry's `metrics` object. The emitted spaces
follow the usual [alignment rules](alignment.md). With benchmark depth ranges,
the dataset also emits histograms for each benchmark bin, for example
`depth.eval.metric.distributions.pixel_pool.near.rmse.distribution`.

Each leaf is a typed distribution value. For example, the value at
`depth.eval.metric.distributions.pixel_pool.rmse.distribution` is:

```json
{
  "type": "distribution",
  "definitionId": "rmse_metric_error_count",
  "values": [5, 2, 1, 1]
}
```

The referenced registry lives at `metricSet.metadata.distributions`:

```json
{
  "schemaVersion": 1,
  "binSets": {
    "error_magnitude": {
      "binEdges": [0.0, 1.0, 2.0, 3.0, null],
      "interval": "[left, right)",
      "spacing": "linear"
    }
  },
  "definitions": {
    "rmse_metric_error_count": {
      "binSetId": "error_magnitude",
      "metric": "rmse",
      "space": "metric",
      "axes": {
        "x": {"quantity": "absolute_error", "unit": "m"},
        "y": {"quantity": "observations", "statistic": "count", "unit": "1"}
      },
      "observationUnit": "pixel",
      "merge": "sum"
    }
  }
}
```

This means five pixels with error below 1, two in `[1, 2)`, one in `[2, 3)`,
and one at or above 3. There are always `values.length + 1` edges. Only the
final edge may be `null`, meaning positive infinity. Bin values are never null.
`spacing` describes how the finite interior edges were constructed; it does
not request a logarithmic display axis. The zero and overflow bins need special
treatment in a future chart.

Multiple definitions can reference one bin set, so edges appear once even when
both native and metric spaces are emitted. Dataset, per-file and benchmark
histograms in the same space share a definition. `rmse3d` definitions use
`axes.x.quantity: "euclidean_error"`. `observationUnit: "pixel"` also covers
the projected pixel correspondences in sparse evaluation. `axes.y.unit: "1"`
means a dimensionless count; it is separate from the observation unit.

Definition and bin-set IDs are **opaque references local to this metric set**.
They are neither global IDs nor content hashes. Resolve each reference; do not
infer semantics from the ID or assume equal IDs imply equal edges across runs.
The existing category axis and `metricDescriptions` still describe metric paths;
the distribution registry is authoritative for the histogram's axes and units.

## Consumer contract and validation

The packaged [v1 JSON Schema](../euler_eval/schemas/distribution-v1.schema.json)
validates the registry at its root and the typed leaf at `#/$defs/value`.
This is an extension to the metric tree; it does not replace the surrounding
`eval.json` envelope. `schemaVersion` versions this extension independently of
`producerVersion`. A reader must reject unsupported versions before storing any
part of the import. Scalar-only output has no registry or typed leaves and
retains its existing contract.

A reader must recognize `type: "distribution"` **before** recursing into a
metric object. Keep the full metric path as its identity and validate its
namespace exactly as for a scalar. Never flatten `values` into separate metrics
or treat `type` and `definitionId` as metric names. Stop at the typed leaf.

After shape validation, check these cross-field rules:

- Every `definitionId` resolves in this registry, and every `binSetId` resolves
  in `binSets`. The definition's metric/space must match the enclosing metric
  path according to the envelope's axis declarations.
- Finite edges strictly increase. Only the last edge may be null. Values have
  exactly one entry per interval. All numbers are finite; booleans are invalid.
- With `statistic: "count"`, values are nonnegative integers, each value and
  their total are at most `2^53 - 1`. This preserves exact counts in JavaScript.
  The producer fails before its pooled counts exceed this limit. Count axes
  have `quantity: "observations"` and `unit: "1"`.
- With `statistic: "sum"`, values may be fractional finite numbers. Do not
  round them or apply integer-count validation.
- An all-zero histogram is valid. Missing distributions mean the feature or
  metric is absent, which is different from evaluating zero observations.

`validate_distribution_contract(metadata, values)` in
`euler_eval.distribution_contract` implements the reference/edge/value checks
after schema validation. The enclosing importer is responsible for validating
metric paths, duplicate identities, namespaces and payload size limits.

Small, synthetic [consumer fixtures](../tests/fixtures/distributions/README.md)
cover all wire namespaces, empty samples, multiple spaces, repeated file IDs in
different hierarchy nodes, scalar-only and distribution-only imports, and a
second summed-error view. They use the same schema as the CLI integration tests.
The schema ships in the wheel; the source archive also includes these docs,
fixtures, tests and their regeneration script.

The earlier unversioned array-only prototype is superseded by v1. Consumers
should support v1 explicitly; any migration of saved prototype files needs an
explicit adapter with known units and space mapping. There is no implicit
fallback from a malformed v1 payload to a legacy format.

## Python and future axes

Pass `distribution_config=DistributionConfig(...)` to any of
`evaluate_depth_samples`, `evaluate_sparse_depth_samples`,
`evaluate_points_3d_samples` or `evaluate_points_3d_sparse_samples` in
`euler_eval.evaluate`. Import `DistributionConfig` from
`euler_eval.distributions`. The result includes `distribution_info` with the
registry shown above and typed leaves in aggregate and per-file branches.
The CLI moves that registry into `metricSet.metadata.distributions` once.

The histogram primitive separates the values being binned from optional
weights. A later depth-axis view can sum errors within depth intervals with
`bin_values(gt_depths, depth_edges, weights=error_magnitudes)`. This primitive
is available now; that alternate axis is not yet exposed in CLI output.

V1 already supports such additive views: add a bin set for GT depth and a
definition with `x.quantity: "gt_depth"`, `y.quantity: "absolute_error"`,
`y.statistic: "sum"`, both axes in `m`, and `merge: "sum"`. Store it at a
distinct metric path such as `rmse.distribution_by_depth`; the existing error
histogram can coexist. The `future-sum-by-depth.eval.json` fixture demonstrates
this contract without enabling a new evaluator mode.

For a later **RMSE by depth** view, store both summed squared errors (`m^2`)
and counts in compatible depth bins; derive `sqrt(sum_squared_error / count)`.
Summed absolute errors or binned error counts cannot reconstruct exact RMSE.
Do not average per-image RMSE values to create pooled RMSE.

`merge: "sum"` permits elementwise addition only for matching definitions and
edges and compatible evaluation populations. Check the metric, space, quantities,
units, observation unit, statistic and evaluation provenance; never merge
histograms just because they have the same length or reference ID. Benchmark
`all`/`near`/`mid`/`far` paths describe populations, not additional array axes,
and overlapping populations must not be summed together. Cross-run merging of
`unspecified` units requires extra calibration evidence; per-file pooling within
the same run remains valid.
