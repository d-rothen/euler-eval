# Configuration

A run is described by one JSON file: the ground truth, the prediction datasets
to score against it, and optional experiment logging.

```json
{
  "euler_train": { "dir": "runs/my_project" },
  "gt": {
    "rgb":          { "path": "/data/gt/rgb" },
    "depth":        { "path": "/data/gt/depth" },
    "rays":         { "path": "/data/gt/rays" },
    "segmentation": { "path": "/data/gt/segmentation" },
    "calibration":  { "path": "/data/gt/calibration" }
  },
  "datasets": [
    {
      "name": "model_a",
      "rgb":   { "path": "/data/model_a/rgb" },
      "depth": { "path": "/data/model_a/depth" },
      "rays":  { "path": "/data/model_a/rays" },
      "output_file": "/path/to/output/model_a_eval.json"
    },
    { "name": "model_b_depth_only", "depth": { "path": "/data/model_b/depth" } }
  ]
}
```

Runnable starting points live in the repository root:
[`example_config.json`](../example_config.json),
[`example_sparse_depth_config.json`](../example_sparse_depth_config.json), and
[`example_points_3d_config.json`](../example_points_3d_config.json).

The config is validated before any work starts: a missing `gt` section, an empty
`datasets` array, a dataset without a `name` or without a single prediction
modality, and any path that does not exist all fail immediately with a message
naming the offending field.

The optional top-level [`distributions`](distributions.md#bin-settings) section
enables per-sample and pooled RMSE histograms. Use `true` for defaults or an
object with `n_bins`, `scale`, `min_error` and `max_error`; CLI histogram settings
override the corresponding fields. Omitted or `false` leaves histograms disabled.

## Paths and splits

Every modality entry is an object with a `path`, and optionally a `split`:

```json
{ "path": "/data/gt/depth", "split": "test" }
```

euler-loading inline selectors work directly in `path` as well, which is often
shorter:

| Form | Meaning |
|---|---|
| `/data/muses` | Directory root, canonical index |
| `/data/muses.zip` | Zip archive, read without extraction |
| `/data/muses.zip:test` | The `test` split of that archive |
| `/data/muses.zip:test#scope=rgb` | The `test` split, reading `.ds_crawler/rgb/` metadata |

## `gt` section

At least one of `rgb`, `depth`, `sparse_depth`, `rays`, or `points_3d` is
required.

| Field | Required | Description |
|---|---|---|
| `gt.rgb.path` | no\* | GT RGB dataset |
| `gt.depth.path` | no\* | GT dense depth dataset |
| `gt.sparse_depth.path` | no\* | Sparse pointcloud GT, `(N,C)` points whose first columns are `x,y,z` in metres |
| `gt.rays.path` | no\* | GT ray direction map dataset |
| `gt.points_3d.path` | no\* | GT per-pixel 3D point map `(H,W,3)` in camera-frame metres |
| `gt.segmentation.path` | no | GT segmentation (needed for `--mask-sky` and `--sky-iou`); see [GT segmentation](#gt-segmentation) |
| `gt.semantic_segmentation.path` | no | Alias of `gt.segmentation`; configure only one of the two |
| `gt.calibration.path` | no | Calibration data (camera intrinsics); enables true camera-frame [surface normals](metrics.md#surface-normals-and-calibration) |
| `gt.intrinsics.path` | with `gt.sparse_depth` | Camera intrinsics for pointcloud projection |
| `gt.camera_extrinsics.path` | with `gt.sparse_depth` | Source-to-camera extrinsics (e.g. MUSES `lidar2rgb`) |
| `gt.lidar_extrinsics.path` | no | Optional lidar pose in a shared frame; composed with `gt.camera_extrinsics` |
| `gt.name` | no | Display name for ground truth (default `"GT"`) |

\* At least one of the five modality entries must be present.

### GT segmentation

`gt.segmentation` is the canonical key. `gt.semantic_segmentation` is accepted
as an alias, for archives whose segmentation scope carries that name (MUSES,
for example); configuring both keys is an error. The segmentation is read only
when `--mask-sky` or `--sky-iou` needs it. `--mask-sky` without it warns and
disables itself, while `--sky-iou` without it exits with an error.

The dataset head's loader module must expose a `sky_mask` function, and the
head must declare the sky label as `skyclass` (euler-loading 2.29 or later).
euler-eval contains no label IDs.

Segmentation is per frame, so it is paired with the evaluated frames by
hierarchy path and file ID, like depth: its index must use the same folder
structure and IDs as the frames it masks. A frame without its own segmentation
file is not evaluated, and the run prints how many pairs the segmentation
dropped. When segmentation is read only for `--sky-iou`, it must not change
the evaluated pairs, so a segmentation that lacks frames is not loaded; the run
warns and the sky IoU is skipped. A segmentation whose IDs match no frame
stops the run with euler-loading's "No common IDs" error and per-modality file
counts.

The config key is also the default metadata scope. If the path has no
`#scope=` selector, no root-level metadata, and no scope of that name, the
scope is chosen among the archive's scopes: the one whose name or
`dataset-head.json` modality key is `segmentation` or `semantic_segmentation`
(both names are candidates under either key), or the only scope. If several
scopes match, or none among several, the run stops and asks for `#scope=`.

## `datasets` section

Each entry needs a `name` and at least one prediction modality. Use only **one**
dense depth-like entry (`depth`, `relative_depth`, or `affine_depth`) per
dataset — configuring two is an error, because the space the prediction lives in
would be ambiguous.

| Field | Required | Description |
|---|---|---|
| `name` | yes | Display name for this prediction dataset |
| `rgb.path` | no\* | Predicted RGB dataset |
| `depth.path` | no\* | Predicted dense metric depth; also used against sparse pointcloud GT |
| `relative_depth.path` | no\* | Predicted dense relative depth (scale/shift alignment supported) |
| `affine_depth.path` | no\* | Predicted dense affine depth (scale/shift alignment supported) |
| `sky_mask.path` | no | Auxiliary predicted sky mask for a dense depth entry; used with `--sky-depth`, `--mask-sky-prediction` and `--sky-iou` |
| `rays.path` | no\* | Predicted ray direction map dataset |
| `points_3d.path` | no\* | Predicted per-pixel 3D point map `(H,W,3)` |
| `output_file` | no | Custom results path (default: `eval.json` in the first modality path) |

\* At least one prediction modality must be present.

Which entry you use is also a *declaration*: `relative_depth` and `affine_depth`
tell the evaluator the prediction is not metric, which is what
`--depth-alignment auto_affine` and `--points-3d-alignment auto` key off. See
[Spaces & alignment](alignment.md).

### Predicted sky depth

Add a `sky_mask` entry alongside `depth`, `relative_depth`, or `affine_depth`
and run with `--sky-depth 100` to evaluate predicted sky at 100 meters:

```json
{
  "name": "model_with_sky",
  "depth": { "path": "/data/model/depth" },
  "sky_mask": { "path": "/data/model/sky_mask", "split": "test" }
}
```

The sky mask is paired with GT and predicted depth by sample ID and hierarchy.
It supports the same archive, split, and `#scope=` selectors as depth, defaulting
to the `sky_mask` metadata scope. Its loader must return a boolean mask with
**True meaning sky**. For example, `generic_dense_depth.sky_mask` uses the RGB
sky color in the mask dataset's `meta.skyclass` (`[R, G, B]`; the older
`meta.sky_mask` key is deprecated) and returns a `(1, H, W)` tensor. Masks at another resolution follow the depth grid's crop
rule or use nearest-neighbor resizing.

`--sky-depth` applies to dense and sparse depth evaluation. GT and both native
and aligned prediction branches are capped after depth conversion and affine
fitting; predicted sky pixels are then set to the cap. Predicted sky pixels are
excluded from the affine fit. Benchmark bins use the capped GT depth.

The cap also works without a predicted mask. Positive infinity is capped;
NaNs and nonpositive depths remain invalid, except that masked prediction
pixels are filled even if their original depth is invalid. Sparse GT still
contributes only at projected points. Predicted sky stays in the scores where
GT is valid; `--mask-sky` separately excludes pixels using GT segmentation and
can be combined with `--sky-depth`.

Without `--sky-depth`, `--mask-sky-prediction` or `--sky-iou`, the predicted
mask is not loaded and depth evaluation uses the usual uncapped values.
`--sky-iou` loads the mask
even without `--sky-depth`, but only to score it against GT sky
([Sky segmentation](metrics.md#sky-segmentation)); the depth fill and the
affine-fit exclusion stay tied to `--sky-depth` (and `--mask-sky-prediction`),
so depth values do not change.
Because the mask is paired by sample ID, a mask that lacks some depth frames
also drops them from the depth metrics; the run prints how many. When the mask
is loaded only for `--sky-iou`, such a mask is not loaded at all: the run
warns, keeps every depth pair and skips the explicit variant. The effective cap and mask
source are recorded in `eval.json` under `meta.eval_params.sky_depth` and
`meta.pred.sky_mask`.

### Masking predicted sky

`--mask-sky-prediction` excludes the pixels of a prediction's `sky_mask` from
dense and sparse depth evaluation, as `--mask-sky` does for GT sky: from every
metric, benchmark bin and histogram, and from the scale/shift fit. For sparse
depth, lidar points that project into predicted sky are not scored. A dataset
without a `sky_mask` is scored in full and the run warns. The flag loads the
mask without `--sky-depth`, combines with `--mask-sky` (the union of both skies
is excluded) and with `--sky-depth` (the cap still applies to the remaining
pixels). It does not change `--sky-iou`, which scores the masks themselves.

The excluded pixels come from the evaluated model, so two models are scored on
different pixel sets, and a model can lower its error by calling hard pixels
sky. Each result records how much it excluded under `meta.pred_sky_masking`
([Results & output](output.md#predicted-sky-masking)); compare models with
this flag only alongside that fraction, or prefer `--mask-sky`, whose GT sky is
the same for every model.

## Sparse depth (pointcloud GT)

Use `gt.sparse_depth` instead of `gt.depth` to evaluate a dense depth prediction
against a sparse pointcloud, e.g. raw LiDAR returns.

The evaluator projects the sparse GT cloud into the prediction plane using
`gt.intrinsics` and `gt.camera_extrinsics`, then computes pointwise depth
metrics only at projected valid pixels. The resulting depth `eval.json` is an
atomic `sparsedepth.eval` metric set; a depth prediction is not also unprojected
and evaluated as `points_3d`.

An explicit `datasets[].points_3d` prediction can be evaluated separately
against the same sparse cloud. That evaluation uses the points-3D similarity
gauge, writes an atomic `points3d.eval` metric set beside the point-map
prediction, and is controlled by `--skip-points-3d`. If a dataset provides both
depth and `points_3d`, each prediction modality gets its own `eval.json` at its
own path.

Sparse depth does not require segmentation GT. `gt.segmentation` is loaded only
when `--mask-sky` or `--sky-iou` is set. `--mask-sky` excludes sky pixels from
projected-point metrics and from scale/shift fitting; `--sky-iou` scores the
dense prediction's sky against it without changing any depth metric.

```json
{
  "gt": {
    "name": "MUSES sparse GT",
    "sparse_depth":      { "path": "/data/muses.zip:test#scope=lidar" },
    "intrinsics":        { "path": "/data/muses.zip:test#scope=intrinsics" },
    "camera_extrinsics": { "path": "/data/muses.zip:test#scope=extrinsics" }
  },
  "datasets": [
    { "name": "model_a", "depth": { "path": "/data/model_a/dense_depth:test" } }
  ]
}
```

## Points-3D ground truth sources

A `points_3d` prediction is compared against a GT point map from one of two
sources:

1. **Explicit** — `gt.points_3d.path` (a stored `(H,W,3)` map).
2. **Synthesized from depth** — if `gt.points_3d` is absent but `gt.depth.path`
   and intrinsics (`gt.intrinsics` or `gt.calibration`) are present, GT depth is
   unprojected on the fly. Whether the GT depth is radial or planar is read from
   its metadata (`radial_depth`).

## `euler_train` section (optional)

When present, results are logged to an
[euler-train](https://github.com/d-rothen/euler-train) run, which requires the
`[logging]` extra. `euler_train.dir` is either a project directory (a new run is
created and finished on completion) or a full path to an existing run directory
(the run is resumed and detached afterwards, leaving it active). `dir` is the
only supported key.

Each prediction dataset is registered as an evaluation named after its config
`name`, first as `running` and then with its metrics attached, alongside package
provenance (euler-eval version, Python, torch, CUDA).

## Loader resolution & dataset metadata

Loaders are resolved automatically by euler-loading from each dataset's
ds-crawler index metadata — no manual loader selection is needed. Each dataset
directory declares its loader via the `euler_loading.loader` and
`euler_loading.function` fields of its index. Modality metadata
(`radial_depth`, `rgb_range`, sparse point columns, coordinate units) is read
automatically; depth and point-cloud coordinates are assumed to already be in
metres.

Each dataset root or archive must contain ds-crawler metadata under
`.ds_crawler/` (`dataset-head.json`, `ds-crawler.json`, `index.json`). When one
physical root holds several logical modalities, the artifacts may be scoped
under `.ds_crawler/<modality>/` with a `.ds_crawler/scopes.json` manifest,
letting a path like `/data/muses.zip:test` load `.ds_crawler/rgb/index.json`,
`.ds_crawler/depth/index.json`, and so on from the same archive. GT and
prediction datasets are matched by hierarchy path and file ID.

### Multiple modalities from one archive

Give every modality the same archive root and let the scope select the metadata.
The scope defaults to the configuration key name, so the common case is compact:

```json
{
  "gt": {
    "rgb":               { "path": "/data/capture.zip:test" },
    "depth":             { "path": "/data/capture.zip:test" },
    "intrinsics":        { "path": "/data/capture.zip:test" },
    "camera_extrinsics": { "path": "/data/capture.zip:test" }
  }
}
```

Use `#scope=<scope>` when an artifact scope is not named exactly like the config
key, e.g. `"sparse_depth": { "path": "/data/capture.zip:test#scope=lidar" }`.
GT segmentation is the exception: under either of its keys it also finds a
scope named after the other key, as described in [GT segmentation](#gt-segmentation).

Regular modalities (`rgb`, `depth`, `rays`, `points_3d`, `sparse_depth`, GT
segmentation and a predicted `sky_mask`) participate in file-ID intersection;
calibration and pose data (`intrinsics`, `calibration`, `camera_extrinsics`,
`lidar_extrinsics`) are loaded hierarchically and do not. A hierarchical file
applies to every frame at or below its folder, which suits one calibration per
scene but not per-frame data.

## `metrics_config.json`

Controls the thresholds used by sanity checking. When `--metrics-config` is not
given, the tool auto-detects `metrics_config.json` at the project root and falls
back to built-in defaults if there is none.

The file is grouped by modality, and each metric block carries its thresholds
plus a `description` explaining what the check is for:

```json
{
  "depth": {
    "expected_range": { "min": 0.0, "max": 100.0 },
    "absrel": { "warn_if_median_exceeds": 1.0, "warn_if_p90_exceeds": 2.0 },
    "silog":  { "warn_if_exceeds": 0.5 }
  }
}
```

See [`metrics_config.json`](../metrics_config.json) for the full set of options,
and [Results & output](output.md#sanity-check-report) for what the checks
produce.
