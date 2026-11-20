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
    "schemaVersion": "2.0",
    "task": "depth.dense",
    "parameters": { "depth.alignment.mode": "auto_affine", "spatial.method": "resize_prediction", "…": "…" },
    "references": { "gt": { "dataset": { "id": "muses", "…": "…" }, "identity": "sha256:…" } },
    "population": { "count": 500, "manifest": "sha256:…" },
    "…": "…"
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
| `protocol` | The evaluation-protocol contract: references, settings, the evaluated items and how to compare and recompute every metric ([Comparing evaluations](#comparing-evaluations)) |
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

`protocol` describes how an evaluation was set up and what it scored, so a
consumer can tell whether two evaluations can be compared, and how. Since
euler-eval 3.0 it is the **evaluation-protocol contract, schema 2**, defined by
[euler-metric-naming](https://github.com/d-rothen/euler-metric-naming)
(`docs/contract.md` there; Python API `euler_metric_naming.contract`). A
consumer judges every modality with one generic algorithm: everything it needs
— what each key means, how it is compared, how metric names decompose and how
aggregates are recomputed — travels in the file. `meta` stays the full record
of the run, paths and device included; nothing else needs to be compared.

```python
from euler_metric_naming.contract import compare_files, load_eval_file

a, b = load_eval_file("a/eval.json"), load_eval_file("b/eval.json")
verdict = compare_files(a, b, "depth.eval.metric.standard.image_mean.absrel")
verdict.label   # "same protocol", "compared on 680 shared files",
                # "comparable · domain differs by 3.7 %", "not comparable · GT differs"
```

A verdict is computed per metric: two evaluations can be comparable for
`native` metrics and not for `metric` ones (a different alignment), or paired
for `image_mean` but only partially comparable for FID.

### The block

A dense depth evaluation of a cropped GT (trimmed):

```json
"protocol": {
  "schemaVersion": "2.0",
  "task": "depth.dense",
  "vocabularies": { "euler.core": "1.0", "euler.depth": "1.0", "euler.sky": "1.0" },
  "parameters": {
    "depth.alignment.mode": "none",
    "depth.alignment.applied": false,
    "depth.cap.max_m": 60.0,
    "depth.benchmark.range_m": [1.0, 80.0],
    "depth.invalid.policy": "exclude_sentinel",
    "sky.gt.masking": false,
    "sky.pred.masking": true,
    "spatial.policy": "auto",
    "spatial.method": "vae_crop",
    "spatial.uniform": true,
    "spatial.resize_target": "gt_grid",
    "spatial.mask_resample": "nearest_exact",
    "…": "…"
  },
  "descriptors": { "depth.cap.max_m": { "role": "definition", "inertIf": { "…": "…" } }, "…": "…" },
  "references": {
    "gt": {
      "modality": "depth",
      "dataset": { "id": "cerema", "variant": "full_crop", "split": null, "scope": "depth" },
      "identity": "sha256:4fbc81…",
      "manifest": { "hash": "sha256:2df47b…", "count": 1600 },
      "view": { "source": "cerema/rgb_left@1024x1920", "region": [270, 170, 707, 1730] }
    }
  },
  "observations": {
    "domain.region": { "frame": "cerema/rgb_left@1024x1920", "box": [270, 170, 704, 1728] },
    "prediction.resample": [1, 1],
    "depth.cap.clipped": 0.0123,
    "sky.pred.excluded": 0.08
  },
  "population": {
    "unit": "image", "idScheme": "euler.item_id/1", "count": 680,
    "manifest": "sha256:69922b…",
    "accounting": { "reference": 1600, "prediction": 680, "paired": 680, "dropped": { "sky_mask": 0 } }
  },
  "stratifications": {
    "depth.gt_range": {
      "label": "GT depth", "variable": "gt_depth_m", "unit": "m",
      "edges": [1.0, 13.3086, 39.6419, 80.0], "labels": ["near", "mid", "far"],
      "closed": "left", "lastClosed": true, "total": "all",
      "scheme": "sqrt_equal_width", "after": ["depth.cap.max_m"]
    }
  },
  "axes": {
    "space": { "role": "space" }, "category": { "role": "family" },
    "reduction": { "role": "reduction" },
    "bin": { "role": "stratum", "stratification": "depth.gt_range" }
  },
  "reducers": { "image_mean": { "recompute": "mean" }, "image_median": { "recompute": "median" }, "pixel_pool": { "recompute": "pool" } },
  "metrics": { "absrel": { "id": "depth.absrel", "rev": 1, "pool": { "form": "mean", "weight": "count" }, "…": "…" }, "…": "…" },
  "items": {
    "ids": ["1~day~fog100~0001", "…"],
    "series": {
      "depth.eval.metric.standard.count": [1000, "…"],
      "depth.eval.metric.standard.near.count": [312, "…"]
    }
  }
}
```

| Property | What it records |
|---|---|
| `task`, `vocabularies` | The task (`depth.dense`, `depth.sparse`, `rgb.restoration`, `rays.directions`, `points3d.dense`, `points3d.sparse`) and the vocabulary versions the terms come from |
| `parameters` | Settings and resolved choices, as flat scalars or lists of scalars (the v1 limits) |
| `descriptors` | For every parameter, observation and reference: its role (`definition`, `provenance`, `domain`, …), how a difference is judged, which metrics it affects and when it is inert — copied from the vocabularies, so consumers never need them |
| `references` | Every GT input the evaluation **read**, by content (below) |
| `observations` | Measured properties of what was scored: the scored region or grid, resampling, depth-cap clipping, excluded predicted sky, frames without segmentation |
| `population` | The evaluated items: the manifest hash of their ids and how the pairs came about |
| `stratifications`, `axes`, `reducers`, `metrics` | How metric names decompose (which axis is the space, the reduction, a stratum), the bins of a stratum, how each reduction is recomputed and each metric's definition (`rev`, direction, pool form and weight) |
| `items` | Per-item sufficient statistics: the pixel/point counts that weight pooled metrics |

**References.** A reference's `identity` is the hash of its dataset (`id` and
`variant` of its `dataset-head.json`, metadata `scope`, `split`), the
**manifest** of its sample ids as euler-loading paired them, and its dataset
head without volatile fields (paths, timestamps). Two crops of one dataset
(`cerema` `cropped` and `full_crop`) therefore differ, and the same data
stored elsewhere does not. A head that records an euler-preprocess crop
(`modality.meta.crop.boxes`) also gives a `view`: the `region`
`[top, left, height, width]` in its `source` frame, named
`<dataset id>/<camera or modality>@<height>x<width>`. GT inputs that were
configured but not read are absent: GT segmentation without `--mask-sky` or
`--sky-iou`, GT depth and calibration of an RGB evaluation without `gt.rgb`.

| Modality | References (when read) |
|---|---|
| `depth` | `gt`, `calibration`, `segmentation` |
| `sparsedepth` | `gt` (`sparse_depth`), `intrinsics`, `camera_extrinsics`, `lidar_extrinsics`, `segmentation` |
| `rgb` | `gt` (`rgb`), `input` (`input_rgb`), `depth`, `calibration`, `segmentation` |
| `rays` | `gt`, `calibration` |
| `points3d` (dense) | `gt` (`points_3d`, or `depth` it is synthesized from), `intrinsics`, `calibration`, `segmentation` |
| `points3d` (sparse GT) | as `sparsedepth` |

**Population.** The item ids are the per-file tree paths of
`per_file_metrics` — the `children` keys from the root down, then the file
`id`, joined with `~` (`euler.item_id/1`), in Unicode NFC (euler-eval writes
the per-file tree in NFC). `manifest` is the hash of exactly the ids that
appear there. `accounting` records the GT and prediction sample counts, the
pairs evaluated and how many pairs each id-paired optional input dropped
(`segmentation`, `sky_mask`, `input`, `gt_depth`; inputs not read are
absent).

**Parameters**, per modality (`spatial.*`, `sky.*` and the policy keys below
are shared):

| Modality | Parameters |
|---|---|
| `depth`, `sparsedepth` | `depth.alignment.mode` (configured) and `depth.alignment.applied` (whether a fit ran), `depth.alignment.fit`, `fit_trim`, `fit_max_gt_percentile`, `depth.input_space`, `depth.cap.max_m` (`--sky-depth`), `depth.benchmark.range_m`, `depth.invalid.policy`, `depth.conversion`, `sky.gt.masking`, `sky.gt.missing_policy`, `sky.pred.masking`, `sky.pred.fill`, `sky.iou`, `spatial.policy`, `spatial.method`, `spatial.uniform`, `spatial.mask_resample`; dense depth also `spatial.resize_target` |
| `rgb` | `sky.gt.masking`, `sky.gt.missing_policy`, `spatial.policy`, `spatial.method`, `spatial.uniform`, `spatial.resize_target`, `spatial.mask_resample`, `spatial.gt_resample` (with `gt.rgb`), `rgb.fid.backend`, `rgb.input.alignment`, `rgb.input.alignment_policy`, `rgb.depth_bins.policy`, `rgb.depth_bins.edges_m`, `depth.benchmark.range_m`, `depth.invalid.policy` |
| `rays` | `spatial.policy`, `spatial.method`, `spatial.uniform`, `spatial.resize_target`, `spatial.gt_resample`, `rays.fov_policy`, `rays.rho_a.threshold_deg` (resolved) |
| `points3d` | `points3d.alignment.mode` and `points3d.alignment.applied` (`auto` resolves to no alignment from the CLI), `points3d.f_a.threshold_m`, `rays.fov_policy`, `rays.rho_a.threshold_deg`, `sky.gt.masking`, `sky.gt.missing_policy`, `spatial.policy`, `spatial.method`, `spatial.uniform`, `spatial.mask_resample`; dense also `spatial.resize_target` and, for a GT synthesized from depth, `depth.invalid.policy` |

Settings that only add metrics (`--domain`, histogram settings) or only affect
the run (device, batch size, workers, paths, `--eval-name`) are not
parameters; `--sky-iou` is recorded as `sky.iou` with role `provenance`, which
never makes evaluations incomparable.

**Policies instead of `metrics_revision`.** Version 1 had one global
`method.metrics_revision`. Version 2 records each rule whose revision changed
values as its own parameter, so a change makes only the metrics it affects
incomparable: `spatial.resize_target` (`gt_grid` for dense depth and
`points_3d`, `prediction_grid` for RGB and rays), `spatial.mask_resample`
(`nearest_exact`), `depth.invalid.policy` (`exclude_sentinel`),
`rgb.input.alignment_policy` (`follow_gt`), `rays.fov_policy` (`v2`) and
`rgb.depth_bins.policy` (`v3`: an empty depth bin has no value). A
definition change of one metric bumps its `rev` in `metrics` instead.

**Observations.** Dense depth and `points_3d` record `domain.region`: the box
of the scored grid in the GT view's source frame when the GT has a view, else
in the GT's own pixels with the GT's identity as frame — a crop of the GT
(`vae_crop`, a recorded crop) starts at its origin. Other modalities record
`domain.grid` (`[height, width]`). `prediction.resample` is the factor by
which `resize_prediction`/`warp_prediction` resampled the prediction
(`[1, 1]` when it was not). `depth.cap.clipped` is the largest fraction, over
the emitted spaces, of scored pixels whose GT or prediction the depth cap
clipped or filled; a cap that changed nothing is inert. `sky.pred.excluded` and
`sky.gt.unsegmented` come from `meta.pred_sky_masking` and
`meta.gt_sky_masking`.

**First sample, `spatial.uniform`.** `spatial.method` and the domain
observations describe the first sample, as `meta.spatial_alignment` does.
`spatial.uniform` is `true` when every sample was scored with the same method
on the same grid (same GT and prediction sizes, same crop origin); `false`
means those values describe the first sample only, and such an evaluation is
not comparable with a uniform one.

**Strata and per-item values.** The benchmark bins
(`--benchmark-depth-range`) are the stratification `depth.gt_range` (edges as
computed, square-root rule, after the depth cap for depth; RGB's GT depth is
not capped). Dense and sparse depth write the `standard` metrics of every bin
per file, `depth.eval.<space>.standard.<bin>.<metric>` (the aggregate name
without its reduction), next to the unbinned per-file values, and the number
of scored pixels per file and bin goes to `items.series`
(`depth.eval.<space>.standard.count`, `….standard.<bin>.count`). `points_3d`
writes the scored points per file (`points3d.eval.<space>.point_error.count`).
With them a consumer recomputes `image_mean`, `image_median` and `pixel_pool`
(for metrics with a pool form) on any subset of files, which is what
"compared on N shared files" does. RGB writes its benchmark bins per file too
(`rgb.eval.<bin>.mae`); RGB metrics have no reduction axis, so they are not
recomputed. RGB's fixed depth bins (`depth_binned_photometric`, near ≤ 1 m,
mid ≤ 5 m, far beyond) are recorded as the categorical stratification
`rgb.depth_bins`, since the far bin has no upper edge.

Everything new lives inside `protocol`: no top-level block and no per-file
entry property was added, and the metric paths of 2.x are unchanged, so
readers of schema 1 (which read only `schemaVersion` and `parameters`) keep
working.

### Schema 1.x

euler-eval 2.x wrote `schemaVersion` `1.x`: a flat map of
`reference.<input>.{dataset,scope,split}`, `population.*` and `method.*` keys,
compared by strict equality, with `method.metrics_revision`. Such files are
still compared: euler-metric-naming normalizes them through its frozen legacy
table (`contract/legacy/protocol-1.json`), which maps each key to its v2
counterpart. Their limits remain: a v1 GT is identified only by dataset id,
scope and split (two crops of one dataset look like the same GT),
`population.samples` is a count rather than the evaluated files, and without
per-item weights a pooled metric cannot be compared on shared files.

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

Hierarchy keys and file ids are written in Unicode NFC; joined with `~` they
are the item ids of `protocol.population`. Besides the metric tree's per-image
values, dense and sparse depth write the `standard` metrics of every benchmark
bin per file (`depth.eval.metric.standard.near.absrel`), and RGB its benchmark
bins (`rgb.eval.near.mae`); a bin without pixels in an image has no value
there. The per-file pixel counts that weight pooled metrics are in
`protocol.items`, not in the metric tree.

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
