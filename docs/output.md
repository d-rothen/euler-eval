# Results & output

## Where results land

Results are written **per modality, per prediction dataset**, into a `.eval/`
folder at the prediction dataset's root, beside its `.ds_crawler/` metadata:

```text
/data/model_a/                     # a datasets[].<modality>.path
  .ds_crawler/
    depth/  rgb/                   # ds-crawler metadata scopes
  .eval/
    depth/
      eval.json                    # depth results
      no_sky.json                  # depth results of --eval-name no_sky
      split_test/eval.json         # depth results for "/data/model_a:test"
    rgb/
      eval.json                    # rgb results
```

The folder below `.eval/` mirrors how the prediction was loaded:

| Prediction | Results file |
|---|---|
| Metadata scope with its own `.ds_crawler/<scope>/` | `.eval/<scope>/<name>.json` |
| Root-level `.ds_crawler/` metadata (no scope) | `.eval/<name>.json` |
| A split is selected (`:test` or `split`), or evaluated by a [`splits`](configuration.md#splits-section-optional) sweep | `split_<split>/` is added before the file name |

The scope is the path's `#scope=` selector, otherwise the config key (`depth`,
`relative_depth`, `rgb`, `points_3d`, …). As in euler-loading, a scope without
metadata of its own falls back to the root-level `.ds_crawler/`, and so do its
results. This way modalities and splits that share one dataset root or archive
each get their own file. If the root is a `.zip`, `.eval/` is written *into* the
archive (inside its top-level folder when the archive wraps everything in one
named after it), so zip-backed datasets stay self-contained.

Results bound for an archive are written when the whole invocation ends, once
per archive, not once per result: adding an entry rewrites the archive's
central directory, and replacing one, as a re-run does, rewrites the whole
archive. Until then they wait in local temporary files (under `$TMPDIR`), and
the run reports them as `queued for` that path. Results of runs that finished
are written even when a later run fails. If an archive cannot be written, the
run exits with `1` and names the temporary folder holding its results. A job
killed before the end (a time limit, `scancel`) writes none of its archive
results.

Jobs writing into the same archive at once, such as a job array of `--split`
runs, take turns: each write holds an `flock` lock on a hidden
`.<archive>.lock` file beside the archive, which is left in place. A job that
finds it held prints `Waiting for another process writing into …` and waits
for that write, which is the only time it waits. A job that dies releases its
lock. Where the filesystem offers no `flock` (a Lustre mount without `flock`,
NFS mounted `nolock`, Windows) or the lock file cannot be created, the run
prints a note and writes unlocked, so overlapping writes can then corrupt the
archive or lose results. A Lustre `localflock` mount locks only within one
node, so jobs on different nodes are not kept apart there.

`<name>` is `eval` unless the run is named with `--eval-name`:
`--mask-sky --eval-name no_sky` writes `no_sky.json` next to an earlier
`eval.json` instead of replacing it. Names use letters, digits, `.`, `-` and
`_`; `.json` is appended.

`output_file` in a dataset entry overrides all of this. It is one path for the
whole dataset entry: every modality evaluated for that entry is written to it,
and the last one written replaces the others, so use it for single-modality
entries.

Earlier releases wrote `eval.json` directly at the dataset root (inside the
archive for zips). Readers such as euler-view look in `.eval/<scope>/`, then
`.eval/`, then fall back to that legacy location.

Each `eval.json` is atomic and declares one metric namespace. With
`gt.sparse_depth`, a depth-family prediction writes only `sparsedepth.eval`; an
explicit `points_3d` prediction writes only `points3d.eval` to the point
map's own results file. A depth prediction does not implicitly produce a second points-3D metric
set.

## Anatomy of `eval.json`

```json
{
  "metricSet": {
    "metricNamespace": "depth.eval",
    "producerKey": "euler-eval",
    "producerVersion": "2.25.0",
    "sourceKind": "computed",
    "metadata": {
      "input_space_detected": "metric",
      "calibration_mode": "auto_affine",
      "calibration_applied": false,
      "emitted_spaces": ["metric"],
      "canonical_space": "metric"
    },
    "axes": { "space": { "…": "…" }, "category": { "…": "…" } },
    "metricDescriptions": { "absrel": { "…": "…" } }
  },
  "protocol": {
    "schemaVersion": "1.3",
    "parameters": { "reference.gt.dataset": "muses", "method.depth_alignment": "auto_affine", "…": "…" }
  },
  "dataset_info": { "num_pairs": 500, "gt_name": "GT", "pred_name": "model_a" },
  "meta": {
    "version": "2.25.0",
    "modality": "depth",
    "device": "cuda",
    "gt":   { "path": "/data/gt/depth",      "dimensions": { "height": 1080, "width": 1920 } },
    "pred": { "path": "/data/model_a/depth", "entry": "depth" },
    "spatial_alignment": { "mode": "auto", "method": "resize_prediction", "evaluated_dimensions": { "height": 1080, "width": 1920 } },
    "modality_params": { "radial_depth": false },
    "eval_params": { "sky_masking": false, "depth_alignment_mode": "auto_affine" }
  },
  "depth": { "eval": { "native": { "…": "…" }, "metric": { "…": "…" } } },
  "per_file_metrics": { "children": { "…": "…" } }
}
```

| Block | What it is for |
|---|---|
| `metricSet` | The euler-metric-naming envelope: producer, namespace, axis declarations and per-metric descriptions (unit, direction, bounds), plus how the spaces were resolved |
| `protocol` | The comparison keys: every setting that changes metric values, as a flat map ([Comparing evaluations](#comparing-evaluations)) |
| `dataset_info` | How many pairs were actually matched, and the display names of both sides |
| `meta` | Full provenance: package version, device, both source paths and their resolved dimensions, modality metadata, and the effective evaluation flags |
| *metric tree* | The metrics themselves, under the modality root (`depth`, `rgb`, `rays`, `sparsedepth`, `points3d`) |
| `per_file_metrics` | The same metrics per image, in dataset-hierarchy order |

With `--distributions`, the metric tree and per-file entries also contain
typed histogram leaves under the `distributions` category, each containing
`type: "distribution"`, a `definitionId` and a `values` array. Their versioned
registry at `metricSet.metadata.distributions` shares bin boundaries across
metric/space definitions. See [Error distributions](distributions.md) for the
schema, paths, units and consumer fixtures.
For depth, `metricSet.metadata.distributionPopulation` also records the GT
range, sky-depth cap and selection used by the default dataset/per-file
histograms. With a benchmark range they represent its `all` population.

`meta` is what makes a result reproducible after the fact: it records the
alignment mode that ran, whether sky masking was on, the benchmark range, and
the exact paths and versions involved. `spatial_alignment.method` reports how a
GT/prediction resolution mismatch was resolved — `none` when they already match,
`vae_crop` when the GT was cropped to the top-left region the prediction covers,
`crop` when it was cropped at the origin the prediction's recorded
`source_grid` gives, `resize_prediction` when dense depth or point maps were
resized onto the GT grid, `resize` when RGB or rays GT was resized onto the
prediction grid, `warp_prediction` / `warp` when a recorded mapping that is
neither (e.g. a resize, then a crop) resampled the prediction or the GT, and
`pointcloud_projection` when GT was a sparse cloud. `spatial_alignment.mode` is
the [`--spatial-alignment`](alignment.md#resolution-alignment) setting that
chose between crop and resize, or `source_grid` when the prediction's
recorded grid did.

### Sky segmentation provenance

With `--sky-iou`, dense and sparse depth results record
`meta.eval_params.sky_iou: true` and a `meta.sky_segmentation` block. Without
the flag, neither field exists. Values below are illustrative:

```json
"sky_segmentation": {
  "explicit": {
    "available": true, "source": "pred.sky_mask", "samples": 500,
    "samples_undefined_iou": 3, "pred_sky_fraction": 0.19
  },
  "implicit": { "available": false, "reason": "prediction is not metric (relative)" },
  "gt_source": "gt.segmentation",
  "domain": "prediction_grid",
  "samples": 500,
  "samples_without_gt": 0,
  "gt_sky_fraction": 0.21
}
```

| Field | Meaning |
|---|---|
| `explicit`, `implicit` | Whether each variant was computed. A computed variant gives its `source`, how many `samples` it scored, how many had an undefined IoU (no sky in prediction or GT), and its pooled predicted sky fraction; the implicit one adds `rule: "depth >= threshold_m"`. A skipped variant gives the `reason` instead |
| `gt_source` | The config key the GT segmentation came from |
| `threshold_m` | The `--sky-depth` threshold of the implicit variant; present only when that variant was computed |
| `domain` | Pixels scored: `prediction_grid`, the full prediction grid |
| `samples` | Depth pairs evaluated |
| `samples_without_gt` | Pairs left out because no GT segmentation of their own frame was found |
| `gt_sky_fraction` | Pooled GT sky fraction over the scored samples |

### Predicted-sky masking

With `--mask-sky-prediction`, dense and sparse depth results record
`meta.eval_params.mask_sky_prediction: true` and how much the prediction's own
sky removed, since that differs from model to model:

```json
"pred_sky_masking": {
  "source": "pred.sky_mask",
  "samples": 500,
  "samples_with_sky_mask": 500,
  "gt_valid_pixels": 1048576000,
  "excluded_gt_pixels": 220200960,
  "excluded_gt_fraction": 0.21
}
```

`gt_valid_pixels` counts the GT pixels (projected lidar points for sparse
depth) that would be scored without the flag, after `--mask-sky` if set;
`excluded_gt_pixels` is how many of them fell in predicted sky.

### GT sky masking coverage

With `--mask-sky`, every modality it applies to records how many frames had
their own GT segmentation:

```json
"gt_sky_masking": { "samples": 500, "samples_without_segmentation": 100, "samples_with_sky": 396 }
```

`samples_without_segmentation` is `0` unless `--missing-segmentation unmasked`
kept frames without one; those frames were scored without sky masking.
`samples_with_sky` counts the frames whose mask removed any pixel.

When sky keys are written, the metric-set axes also declare them:
`sky_segmentation` joins the `category` values, an optional `variant` axis
(`explicit`, `implicit`) follows `category`, and `space` becomes optional
because these keys have no depth space. `metricDescriptions` gains `iou`,
`precision` and `recall` (higher is better, bounds `[0, 1]`). Every other key
decomposes exactly as before.

## Comparing evaluations

`protocol.parameters` says how an evaluation was set up, as one flat map of
dotted keys. Its keys are the comparison keys: two evaluations of the same
modality were set up the same way when their maps are equal, and the keys whose
values differ say where. Nothing else in the file needs to be compared for that;
`meta` stays the full record of the run, paths and device included.

```json
"protocol": {
  "schemaVersion": "1.3",
  "parameters": {
    "reference.gt.modality": "sparse_depth",
    "reference.gt.dataset": "seeingthroughfog",
    "reference.gt.scope": "sparse_depth",
    "reference.gt.split": "test",
    "reference.intrinsics.dataset": "seeingthroughfog",
    "reference.intrinsics.scope": "intrinsics",
    "reference.intrinsics.split": "test",
    "reference.camera_extrinsics.dataset": "seeingthroughfog",
    "reference.camera_extrinsics.scope": "camera_extrinsics",
    "reference.camera_extrinsics.split": "test",
    "reference.lidar_extrinsics.dataset": null,
    "reference.lidar_extrinsics.scope": null,
    "reference.lidar_extrinsics.split": null,
    "reference.segmentation.dataset": null,
    "reference.segmentation.scope": null,
    "reference.segmentation.split": null,
    "population.samples": 943,
    "population.gt_sky_masking": false,
    "population.missing_segmentation": null,
    "population.pred_sky_masking": false,
    "population.evaluated_resolution": [1024, 1920],
    "population.benchmark_depth_range_m": [0.01, 80.0],
    "method.metrics_revision": 3,
    "method.depth_alignment": "auto_affine",
    "method.sky_depth_m": null,
    "method.pred_sky_mask": false
  }
}
```

The first segment of a key is its section:

| Section | Holds |
|---|---|
| `reference` | The GT the prediction is scored against (`gt`, with its config key as `modality`) and every other GT input the modality reads, each identified by `dataset` (the `dataset.id` of its `dataset-head.json`), metadata `scope` and `split` — not by file path, so the same data on another machine matches |
| `population` | Which samples and pixels were scored: the matched `samples`, sky masking, the grid the metrics were computed on and benchmark depth bins |
| `method` | How they were scored: alignment modes, the sky-depth cap and whether the prediction's `sky_mask` was used, how GT and prediction resolutions were reconciled, the FID backend and `metrics_revision` |

What is recorded, per modality:

| Modality | `reference` inputs besides `gt` | `population` | `method` |
|---|---|---|---|
| `depth` | `calibration`, `segmentation` | `samples`, `gt_sky_masking`, `missing_segmentation`, `pred_sky_masking`, `evaluated_resolution`, `benchmark_depth_range_m` | `depth_alignment`, `sky_depth_m`, `pred_sky_mask`, `spatial_alignment` |
| `sparsedepth` | `intrinsics`, `camera_extrinsics`, `lidar_extrinsics`, `segmentation` | as `depth` | `depth_alignment`, `sky_depth_m`, `pred_sky_mask` |
| `rgb` | `input_rgb`, `depth`, `calibration`, `segmentation` | `samples`, `gt_sky_masking`, `missing_segmentation`, `evaluated_resolution`, `benchmark_depth_range_m` | `fid_backend`, `spatial_alignment`, `input_alignment` |
| `rays` | `calibration` | `samples`, `evaluated_resolution` | `spatial_alignment` |
| `points3d` (dense) | `intrinsics`, `calibration`, `segmentation` | `samples`, `gt_sky_masking`, `missing_segmentation`, `evaluated_resolution` | `points_3d_alignment`, `spatial_alignment` |
| `points3d` (sparse GT) | `intrinsics`, `camera_extrinsics`, `lidar_extrinsics`, `segmentation` | `samples`, `gt_sky_masking`, `missing_segmentation`, `evaluated_resolution` | `points_3d_alignment` |

Every modality also has `method.metrics_revision`.

RGB evaluated without `gt.rgb`, against [`gt.input_rgb`](configuration.md#rgb-references)
only, computes no full-reference metric, so the settings that act on them are
recorded as off: `reference.gt`, `reference.depth`, `reference.calibration` and
`reference.segmentation` are `null`, as are `population.missing_segmentation`,
`population.benchmark_depth_range_m`, `method.fid_backend` and
`method.spatial_alignment`, and `population.gt_sky_masking` is `false`.
`method.input_alignment` is `none`, `upscale_input` or `upscale_prediction`
when `gt.input_rgb` is configured, else `null`.

Rules that keep comparisons a plain map comparison:

- Only settings that change the value of a metric both files contain are
  parameters. `--domain`, `--sky-iou` and histogram settings only add metrics
  (histograms describe their bins in the distribution registry); device, batch
  size, workers, paths and `--eval-name` do not change values.
- Every key of a modality is always written, `null` or `false` when off, so a
  key missing from one side means "not recorded", never "off". Compare the union
  of both key sets.
- Values are scalars or lists of scalars, never objects.
- `method.metrics_revision` changes only when a release changes metric values
  for identical inputs and settings, which the [changelog](../CHANGELOG.md)
  then says. Different revisions are not comparable even when every other
  key matches.
- Settings are recorded, not what they caused. `population.pred_sky_masking`
  is the `--mask-sky-prediction` flag, and `method.pred_sky_mask` says whether
  the prediction's `sky_mask` was loaded and scored with (filled at
  `--sky-depth`, or excluded by `--mask-sky-prediction`). It is `false` for a
  prediction without one, or an [experiment](configuration.md#experiments-section-optional)
  that leaves it out. How much each model's own sky removed is in
  `meta.pred_sky_masking`, and `metricSet.metadata` records whether an
  alignment was actually applied.
- Renaming or removing a key bumps the major `schemaVersion`; adding one bumps
  the minor version.

## Spaces and canonical aliases

Depth-like and points-3d results are nested by space:

```text
depth.eval.native.…      the prediction as emitted
depth.eval.metric.…      after scale/shift (or gauge) fitting
```

The `metric` space appears only when a comparison in that space was possible —
for points-3d, only when a gauge alignment actually ran. The in-memory result
dicts (and the euler-train payload) additionally expose a canonical `depth` /
`points_3d` key aliasing the `metric` branch when it exists, else `native`, so a
consumer can read one key without knowing how the run was configured. See
[Spaces & alignment](alignment.md).

## Wire-safe metric roots

Serialized sparse-depth and points-3d metric paths are rooted at
`sparsedepth.eval` and `points3d.eval` — no underscore — because downstream
namespace validation applies a stricter rule to the first segment of a flattened
metric path. The internal Python result dicts keep the `sparse_depth_*` /
`points_3d_*` keys.

## Per-file metrics

`per_file_metrics` mirrors the dataset hierarchy, so per-image values stay
attributable to the scene or sequence they came from:

```json
{
  "children": {
    "scene_01": {
      "files": [
        { "id": "0001", "metrics": { "depth": { "eval": { "metric": { "…": "…" } } } } }
      ]
    }
  }
}
```

Non-finite values (`NaN`, `Inf`) and `None` are stripped from every metric tree
before writing, so the JSON stays schema-valid. A file whose metrics all fail to
compute keeps its entry with an empty `metrics` object, and the run prints a
warning naming the affected files rather than dropping them silently.

## Sanity-check report

Sanity checking is on by default. It validates inputs and results against the
thresholds in [`metrics_config.json`](../metrics_config.json) — implausible
depth ranges, degenerate value ranges that make SSIM or LPIPS meaningless,
AbsRel above 1.0, SILog indicating a scale mismatch, suspicious edge densities —
and reports them grouped by metric and warning type.

Warnings are printed per prediction dataset as the run proceeds, and the full
report is written to `sanity_check_report.json` in the **current working
directory** at the end of the run. Disable the checks with `--no-sanity-check`,
or point them at a different threshold file with `--metrics-config`.

Sanity warnings never fail a run — they exist to catch a misconfigured
comparison (wrong units, wrong space, wrong dataset) before its numbers are
taken at face value.

## RGB FID backends

| Backend | Behaviour |
|---|---|
| `builtin` *(default)* | In-process Inception v3 features, no extra dependency |
| `clean-fid` | Delegates to [clean-fid](https://github.com/GaParmar/clean-fid); requires the `[fid]` extra and gives scores closer to standard published FID |

With `--rgb-fid-backend clean-fid`, euler-eval honors `CLEANFID_CACHE_DIR` for
staging or downloading the Inception checkpoint — see
[offline cache warmup](cli.md#offline-cache-warmup).

## euler-train logging

When the config carries a [`euler_train`](configuration.md#euler_train-section-optional)
section, each prediction dataset is registered as an evaluation in the run —
first as `running`, then completed with its aggregate metrics and package
provenance attached. Per-file metrics are not sent; they stay in `eval.json`.
