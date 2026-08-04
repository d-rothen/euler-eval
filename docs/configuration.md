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
| `gt.segmentation.path` | no | GT segmentation (needed for `--mask-sky`) |
| `gt.calibration.path` | no | Calibration data (camera intrinsics) |
| `gt.intrinsics.path` | with `gt.sparse_depth` | Camera intrinsics for pointcloud projection |
| `gt.camera_extrinsics.path` | with `gt.sparse_depth` | Source-to-camera extrinsics (e.g. MUSES `lidar2rgb`) |
| `gt.lidar_extrinsics.path` | no | Optional lidar pose in a shared frame; composed with `gt.camera_extrinsics` |
| `gt.name` | no | Display name for ground truth (default `"GT"`) |

\* At least one of the five modality entries must be present.

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
| `rays.path` | no\* | Predicted ray direction map dataset |
| `points_3d.path` | no\* | Predicted per-pixel 3D point map `(H,W,3)` |
| `output_file` | no | Custom results path (default: `eval.json` in the first modality path) |

\* At least one prediction modality must be present.

Which entry you use is also a *declaration*: `relative_depth` and `affine_depth`
tell the evaluator the prediction is not metric, which is what
`--depth-alignment auto_affine` and `--points-3d-alignment auto` key off. See
[Spaces & alignment](alignment.md).

## Sparse depth (pointcloud GT)

Use `gt.sparse_depth` instead of `gt.depth` to evaluate a dense depth prediction
against a sparse pointcloud, e.g. raw LiDAR returns.

The evaluator projects the sparse GT cloud into the prediction plane using
`gt.intrinsics` and `gt.camera_extrinsics`, then computes pointwise depth
metrics only at projected valid pixels. It also produces 3D (`points_3d`)
metrics unless `--skip-points-3d` is set — the scored prediction is either a
predicted `points_3d` map (similarity gauge) or a dense depth map unprojected
with the GT intrinsics (affine gauge). If a dataset provides both, the
`points_3d` map is preferred.

Sparse depth does not require segmentation GT. `gt.segmentation` is loaded only
when `--mask-sky` is set, and then excludes sky pixels from projected-point
metrics and from scale/shift fitting.

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

Regular modalities (`rgb`, `depth`, `rays`, `points_3d`, `sparse_depth`)
participate in file-ID intersection; calibration and pose data (`intrinsics`,
`calibration`, `camera_extrinsics`, `lidar_extrinsics`, segmentation) are loaded
hierarchically and do not.

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
