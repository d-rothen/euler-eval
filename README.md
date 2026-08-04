# euler-eval

A comprehensive evaluation toolkit for comparing predicted **depth maps**, **RGB images**, **camera ray direction maps**, and **per-pixel 3D point maps** against ground truth — powered by [euler-loading](https://github.com/d-rothen/euler-loading) for flexible, metadata-driven dataset loading.

## Features

- **Depth** — PSNR, SSIM, LPIPS, FID, KID, AbsRel, RMSE, Scale-Invariant Log Error, the standard monocular-depth metric set (`absrel`…`delta3`), surface-normal consistency, and depth-edge F1.
- **Sparse depth** — evaluate a dense depth prediction against a sparse pointcloud GT (e.g. LiDAR) via reprojection, plus directed 3D Chamfer completeness / recall.
- **RGB** — PSNR, SSIM, LPIPS, FID, SCE (Structural Chromatic Error), edge F1, tail errors (p95/p99), high-frequency energy ratio, and depth-binned photometric error.
- **Rays** — ρ_A (AUC of the angular-accuracy curve), angular-error statistics, and threshold percentages.
- **Points-3D** — for models that predict a per-pixel 3D point map *and their own camera*: 3D EPE/RMSE/δ-accuracy, radial-vs-lateral error decomposition with ρ_A, true-3D surface normals, 3D edge F1, and correspondence-free Chamfer / F-score, in native and similarity-gauge-aligned (Umeyama) spaces.
- **Benchmark depth binning** — optionally subdivide metrics into square-root-scaled near/mid/far depth bins.
- **Sanity checking** — validate metric results against configurable thresholds with detailed warning reports.
- **Sky masking** — optionally exclude sky regions using GT segmentation.
- **Per-file + aggregate results** — per-image metrics and dataset-level aggregates written to JSON, per modality.
- **euler-train integration** — optional experiment logging via [euler-train](https://github.com/d-rothen/euler-train).

## Installation

Requires **Python 3.9+**.

```bash
# core
pip install "euler-eval @ git+https://github.com/d-rothen/euler-eval.git"

# with euler-train logging support
pip install "euler-eval[logging] @ git+https://github.com/d-rothen/euler-eval.git"

# with the clean-fid RGB FID backend
pip install "euler-eval[fid] @ git+https://github.com/d-rothen/euler-eval.git"
```

Or install in editable mode for development:

```bash
git clone https://github.com/d-rothen/euler-eval.git
cd euler-eval
pip install -e ".[dev]"
```

### Dependencies

Core: `numpy`, `scipy`, `Pillow`, `torch`, `torchvision`, `opencv-python-headless`, `lpips`, `torchmetrics`, `tqdm`, and the companion packages [euler-loading](https://github.com/d-rothen/euler-loading), [ds-crawler](https://github.com/d-rothen/ds-crawler), and [euler-metric-naming](https://github.com/d-rothen/euler-metric-naming).

Optional: [euler-train](https://github.com/d-rothen/euler-train) (`[logging]` extra), [clean-fid](https://github.com/GaParmar/clean-fid) (`[fid]` extra).

> **Note:** the `euler-loading`, `ds-crawler`, `euler-train`, and `euler-metric-naming` packages are not on PyPI. When installing with `uv`, the `[tool.uv.sources]` entries in `pyproject.toml` resolve them from GitHub automatically. With plain `pip`, install them from their GitHub URLs first if resolution fails.

## Quick start

The package provides an `euler-eval` console script:

```bash
euler-eval <config> [options]
```

A minimal run against a config that declares GT and one prediction dataset:

```bash
euler-eval config.json --batch-size 32
```

Results are written as `eval.json` per modality (see [Output](#output)).

### Offline cache warmup

Compute nodes without network access need the model weights pre-fetched. Run the cache warmup helper on a machine with network access:

```bash
HF_HOME=/shared/cache/hf \
TORCH_HOME=/shared/cache/hf/torch \
CLEANFID_CACHE_DIR=/shared/cache/clean-fid \
euler-eval.init
```

This pre-downloads the torchvision AlexNet and Inception v3 weights, the LPIPS AlexNet weights, and the clean-fid Inception checkpoint (if `clean-fid` is installed).

## CLI reference

### Positional arguments

| Argument | Description |
|---|---|
| `config` | Path to a JSON configuration file (see [Configuration](#configuration)) |

### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--device` | `{auto,cuda,cpu}` | `auto` | Compute device (`auto` prefers CUDA when available) |
| `--batch-size` | `int` | `16` | Batch size for metrics that support batching |
| `--num-workers` | `int` | `4` | Number of data-loading workers |
| `--verbose`, `-v` | flag | off | Enable verbose output |
| `--skip-depth` | flag | off | Skip depth evaluation |
| `--skip-rgb` | flag | off | Skip RGB evaluation |
| `--skip-rays` | flag | off | Skip rays (spherical direction map) evaluation |
| `--skip-points-3d` | flag | off | Skip points_3d (per-pixel 3D point map) evaluation |
| `--mask-sky` | flag | off | Mask sky regions from metrics using GT segmentation |
| `--no-sanity-check` | flag | off | Disable sanity checking of metric configurations |
| `--metrics-config` | `str` | auto-detect | Path to `metrics_config.json` for sanity checking |
| `--depth-alignment` | `{none,auto_affine,affine}` | `auto_affine` | Depth calibration mode (see [Depth alignment](#depth-alignment)) |
| `--points-3d-alignment` | `{none,scale,similarity,auto}` | `auto` | points_3d gauge alignment (see [Points-3D gauge alignment](#points-3d-gauge-alignment)) |
| `--rgb-fid-backend` | `{builtin,clean-fid}` | `builtin` | RGB FID backend; `clean-fid` requires the `[fid]` extra |
| `--benchmark-depth-range` | `float float` | none | Depth range `[MIN, MAX]` (m) for near/mid/far benchmark bins (additive) |

### Examples

```bash
# Default settings (auto-selects CUDA when available)
euler-eval config.json --batch-size 32

# Sky masking (requires gt.segmentation)
euler-eval config.json --mask-sky -v

# Only depth evaluation
euler-eval config.json --skip-rgb --skip-rays --skip-points-3d

# Force affine scale+shift alignment on all depth predictions
euler-eval config.json --depth-alignment affine

# Use clean-fid for RGB FID
euler-eval config.json --rgb-fid-backend clean-fid

# Benchmark depth/RGB metrics within a depth range (near/mid/far bins)
euler-eval config.json --benchmark-depth-range 0.01 80.0

# Dense depth against sparse pointcloud GT (pointwise + 3D metrics)
euler-eval example_sparse_depth_config.json --skip-rgb --skip-rays

# Per-pixel 3D point maps (predict-your-own-camera models)
euler-eval example_points_3d_config.json --skip-depth --skip-rgb --skip-rays

# Align relative point maps to GT with a similarity (Umeyama) gauge
euler-eval example_points_3d_config.json --points-3d-alignment similarity
```

## Configuration

The config defines the GT modalities, the prediction datasets to evaluate, and optional euler-train logging. See [example_config.json](example_config.json), [example_sparse_depth_config.json](example_sparse_depth_config.json), and [example_points_3d_config.json](example_points_3d_config.json).

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

Each modality entry may include a `split` field (e.g. `{ "path": "/data/gt/depth", "split": "test" }`). euler-loading inline selectors are also accepted directly in `path`, such as `/data/muses.zip:test` or `/data/muses.zip:test#scope=rgb`.

### `gt` section

At least one of `rgb`, `depth`, `sparse_depth`, `rays`, or `points_3d` is required.

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

### `datasets` section

Each entry needs a `name` and at least one prediction modality. Use only **one** dense depth-like entry (`depth`, `relative_depth`, or `affine_depth`) per dataset.

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

### Sparse depth (pointcloud GT)

Use `gt.sparse_depth` instead of `gt.depth` to evaluate against a sparse pointcloud. The evaluator projects the sparse GT cloud into the prediction plane using `gt.intrinsics` and `gt.camera_extrinsics`, then computes pointwise depth metrics only at projected valid pixels. It also produces 3D (`points_3d`) metrics unless `--skip-points-3d` is set — the scored prediction is either a predicted `points_3d` map (similarity gauge) or a dense depth map unprojected with the GT intrinsics (affine gauge). If a dataset provides both, the `points_3d` map is preferred.

Sparse depth does not require segmentation GT. `gt.segmentation` is loaded only when `--mask-sky` is set, and then excludes sky pixels from projected-point metrics and from scale/shift fitting.

### Points-3D ground truth sources

A `points_3d` prediction is compared against a GT point map from one of two sources:

1. **Explicit** — `gt.points_3d.path` (a stored `(H,W,3)` map).
2. **Synthesized from depth** — if `gt.points_3d` is absent but `gt.depth.path` and intrinsics (`gt.intrinsics` or `gt.calibration`) are present, GT depth is unprojected on the fly. Whether the GT depth is radial or planar is read from its metadata (`radial_depth`).

### `euler_train` section (optional)

When present, results are logged to an [euler-train](https://github.com/d-rothen/euler-train) run (requires the `[logging]` extra). `euler_train.dir` is either a project directory (creates a new run, finished on completion) or a full path to an existing run directory (resumed and detached).

### Loader resolution & dataset metadata

Loaders are resolved automatically by euler-loading from each dataset's ds-crawler index metadata — no manual loader selection is needed. Each dataset directory declares its loader via the `euler_loading.loader` and `euler_loading.function` fields of its index. Modality metadata (`radial_depth`, `rgb_range`, sparse point columns, coordinate units) is read automatically; depth and point-cloud coordinates are assumed to already be in metres.

Each dataset root or archive must contain ds-crawler metadata under `.ds_crawler/` (`dataset-head.json`, `ds-crawler.json`, `index.json`). When one physical root holds several logical modalities, the artifacts may be scoped under `.ds_crawler/<modality>/` with a `.ds_crawler/scopes.json` manifest, letting a path like `/data/muses.zip:test` load `.ds_crawler/rgb/index.json`, `.ds_crawler/depth/index.json`, and so on from the same archive. GT and prediction datasets are matched by hierarchy path and file ID.

#### Multiple modalities from one archive

Give every modality the same archive root and let the scope select the metadata. The scope defaults to the configuration key name, so the common case is compact:

```json
{
  "gt": {
    "rgb":              { "path": "/data/capture.zip:test" },
    "depth":            { "path": "/data/capture.zip:test" },
    "intrinsics":       { "path": "/data/capture.zip:test" },
    "camera_extrinsics": { "path": "/data/capture.zip:test" }
  }
}
```

Use `#scope=<scope>` when an artifact scope is not named exactly like the config key, e.g. `"sparse_depth": { "path": "/data/capture.zip:test#scope=lidar" }`. Regular modalities (`rgb`, `depth`, `rays`, `points_3d`, `sparse_depth`) participate in file-ID intersection; calibration and pose data (`intrinsics`, `calibration`, `camera_extrinsics`, `lidar_extrinsics`, segmentation) are loaded hierarchically and do not.

### `metrics_config.json`

Controls sanity-check thresholds. When `--metrics-config` is not given, the tool auto-detects `metrics_config.json` at the project root, falling back to built-in defaults. See [metrics_config.json](metrics_config.json) for all options.

## Depth alignment

Relative depth predictions are aligned to metric GT by fitting a global affine map `d_aligned = s · d_pred + t` (least squares over valid pixels). `--depth-alignment` controls when the fit runs:

- `none` — never; score the prediction as-is (already-metric models).
- `auto_affine` (default) — fit only when the first prediction looks normalized (roughly `[-1, 1]`).
- `affine` — always fit.

The fit uses pixels where GT and prediction are finite and GT > 0. With `--mask-sky`, sky pixels are excluded and the fit is trimmed to `gt ≤ P95(gt)` so residual sky outliers do not dominate. Results are emitted in semantic `native` / `metric` spaces, with `depth` aliasing the canonical branch.

## Points-3D gauge alignment

Point maps from different model families live in different frames, so `points_3d` resolves an unknown **similarity gauge** (the 3D generalization of depth's scale-and-shift) before the `metric`-space comparison. `--points-3d-alignment` selects the gauge:

- `none` — score raw points as-is (metric, predict-your-own-camera models).
- `scale` — fit a single global scalar.
- `similarity` — fit a 7-DoF Umeyama transform `s·R·p + t` over the known per-pixel correspondences (relative / free-frame models).
- `auto` (default) — `similarity` for declared-relative predictions, else `none`.

The angular `error_decomposition` ray metrics and `rho_a` are computed on the **native** point-map directions (the camera-faithful frame); radial/lateral and Euclidean metrics are reported per space.

## Benchmark depth bins

`--benchmark-depth-range MIN MAX` filters valid GT depth pixels to `[MIN, MAX]`, then splits them into three equal-width intervals in **square-root** depth space:

```text
sqrt_min = sqrt(MIN);  sqrt_max = sqrt(MAX);  step = (sqrt_max - sqrt_min) / 3
near = [MIN, (sqrt_min + step)^2)
mid  = [(sqrt_min + step)^2, (sqrt_min + 2·step)^2)
far  = [(sqrt_min + 2·step)^2, MAX]
```

For `--benchmark-depth-range 0.01 80.0` this yields `near=[0.01, 9.29)`, `mid=[9.29, 35.95)`, `far=[35.95, 80.0]`, plus an `all` bin over the full range.

## Metrics

### Depth

| Metric | Key | Description |
|---|---|---|
| PSNR | `depth.image_quality.psnr` | Peak Signal-to-Noise Ratio (dB), max depth as dynamic range |
| SSIM | `depth.image_quality.ssim` | Structural Similarity Index |
| LPIPS | `depth.image_quality.lpips` | Learned Perceptual Image Patch Similarity |
| FID | `depth.image_quality.fid` | Fréchet Inception Distance (dataset-level) |
| KID | `depth.image_quality.kid_mean`, `kid_std` | Kernel Inception Distance |
| Standard | `depth.standard.{image_mean,image_median,pixel_pool}.*` | `absrel`, `sqrel`, `mae`, `rmse`, `rmse_log`, `log10`, `silog`, `delta1-3` |
| AbsRel / RMSE / SILog | `depth.depth_metrics.*` | Median/p90 (SILog also mean) error statistics |
| Normal consistency | `depth.geometric_metrics.normal_consistency` | Surface-normal angular error (deg) + % below 11.25°/22.5°/30° |
| Depth edge F1 | `depth.geometric_metrics.depth_edge_f1` | Edge precision/recall/F1 for depth discontinuities |

### Sparse depth

Only metrics meaningful at isolated projected points are reported (dense image-quality and geometric metrics are skipped). Serialized under the `sparsedepth.eval` root:

| Metric | Key |
|---|---|
| Standard | `sparsedepth.eval.{native,metric}.standard.{image_mean,image_median,pixel_pool}.*` |
| AbsRel / RMSE / SILog | `sparsedepth.eval.{native,metric}.depth_metrics.*` |

### RGB

| Metric | Key | Description |
|---|---|---|
| PSNR | `rgb.image_quality.psnr` | Peak Signal-to-Noise Ratio (dB) |
| SSIM | `rgb.image_quality.ssim` | Structural Similarity Index |
| SCE | `rgb.image_quality.sce` | Structural Chromatic Error |
| LPIPS | `rgb.image_quality.lpips` | Learned Perceptual Image Patch Similarity |
| FID | `rgb.image_quality.fid` | Fréchet Inception Distance (dataset-level) |
| Edge F1 | `rgb.edge_f1` | Edge preservation precision/recall/F1 |
| Tail errors | `rgb.tail_errors` | 95th / 99th percentile per-pixel errors |
| High-frequency energy | `rgb.high_frequency` | HF energy preservation ratio and relative difference |
| Depth-binned photometric | `rgb.depth_binned_photometric` | MAE/MSE in near/mid/far depth bins (needs GT depth) |

### Rays

| Metric | Key | Description |
|---|---|---|
| ρ_A | `rays.rho_a.mean`, `rho_a.median` | AUC of the angular-accuracy curve, integrated to a FoV-dependent threshold (S.FoV 15°, L.FoV 20°, Pano 30°) |
| Angular error | `rays.angular_error.mean_angle`, `median_angle` | Per-pixel angular error (deg) |
| Thresholds | `rays.angular_error.percent_below_*` | % of pixels below 5°/10°/15°/20°/30° |

### Points-3D

Keyed as `points3d.eval.{native,metric}.{category}…`, with a `points_3d` canonical alias.

| Category | Key (per space) | Description |
|---|---|---|
| Euclidean 3D agreement | `point_error.{image_mean,image_median,pixel_pool}.*` | 3D EPE `mae3d`/`rmse3d`/`median3d`/`p90`/`p95`, relative error, δ-accuracy `acc_<τ>` / `acc_rel_<τ>` |
| Error decomposition | `error_decomposition.{radial_*,lateral_*,lateral_fraction}` + `angular_error.*` + `rho_a.*` | Radial (≈ depth) vs lateral (≈ camera-model) split; `lateral_fraction ∈ [0,1]` attributes blame |
| Geometric | `geometric.normal_consistency.*`, `geometric.point_edge_f1.*` | True-3D surface normals and 3D discontinuity F1 |
| Cloud distance | `cloud_distance.chamfer.*`, `cloud_distance.fscore.tau_<τ>.*`, `cloud_distance.f_a` | Correspondence-free Chamfer, F-score at fixed thresholds, and F-score AUC |

### Sparse points-3D

With a sparse pointcloud GT, only the categories that stay meaningful are emitted (dense-neighbourhood `geometric` metrics are skipped), and `cloud_distance` reports only the directed `gt→pred` side (completeness/recall), since a correct dense prediction has many legitimate points far from any sparse GT return. Results serialize under `points3d.eval.{native,metric}`.

## Output

Results are saved as JSON per modality per prediction dataset (default `eval.json` inside each modality's dataset path, overridable with `output_file`).

For RGB FID, two backends are available: `builtin` (in-process, this repo) and `clean-fid` (delegates to [clean-fid](https://github.com/GaParmar/clean-fid); requires the `[fid]` extra and gives scores closer to standard published FID). With `--rgb-fid-backend clean-fid`, `euler-eval` honors `CLEANFID_CACHE_DIR` for staging/downloading the Inception checkpoint.

### Output structure

```json
{
  "depth_native": { "...": "native model depth space, if diagnostically distinct" },
  "depth_metric": { "...": "metric depth space, if available" },
  "depth":        { "...": "canonical alias of depth_metric, else depth_native" },
  "sparsedepth":  { "eval": { "...": "sparse-depth metrics when gt.sparse_depth is used" } },
  "points3d":     { "eval": { "native": { "..." }, "metric": { "..." } } },
  "rgb":          { "...": "..." },
  "per_file_metrics": { "children": { "...": "per-image metrics keyed by depth/depth_native/depth_metric/sparsedepth/rgb" } }
}
```

For depth outputs, `standard` carries the monocular-depth metrics under three reducers: `image_mean`, `image_median`, and `pixel_pool`. The `metric` space for points_3d is emitted only when a gauge alignment ran. Serialized sparse-depth and points-3d metric paths are rooted at `sparsedepth.eval` and `points3d.eval` (no underscore) to satisfy external namespace validation, while the internal Python result dicts keep the `sparse_depth_*` / `points_3d_*` keys.

### Sanity check report

When sanity checking is enabled (the default), a `sanity_check_report.json` is saved to the current working directory with warnings grouped by metric type.

## Programmatic use (in-training validation)

Besides the file-based CLI, `euler_eval.validation` exposes the same depth-metric semantics for **in-memory predictions** — e.g. scoring a training run's validation pass against dense *or* sparse GT without writing predictions to disk:

```python
from euler_eval import (
    DepthValidationAggregator,
    build_validation_gt_dataset,
    evaluate_sparse_depth_sample,
    get_sample_intrinsics,
    get_sample_pointcloud_to_camera_extrinsics,
)

# GT-only dataset from euler-loading compatible paths (inline :split#scope= selectors supported).
dataset = build_validation_gt_dataset(
    sparse_depth_path="/data/muses.zip:val#scope=lidar",
    rgb_path="/data/muses.zip:val#scope=rgb",
    intrinsics_path="/data/muses.zip:val#scope=intrinsics",
    camera_extrinsics_path="/data/muses.zip:val#scope=extrinsics",
)

aggregator = DepthValidationAggregator()
for i in range(len(dataset)):
    sample = dataset[i]
    depth_pred = my_model(sample["rgb"])            # (H, W) metres, planar z
    K = get_sample_intrinsics(sample)
    lidar2cam, _ = get_sample_pointcloud_to_camera_extrinsics(sample)
    result = evaluate_sparse_depth_sample(
        depth_pred,
        sample["sparse_depth"],                     # (N, C>=3) lidar points
        K,
        lidar2cam,
        alignment="none",                           # "affine" for relative depth
        benchmark_depth_range=(0.0, 80.0),          # optional, additive
    )
    aggregator.update(result)

summary = aggregator.summary()
print(summary["standard"]["image_mean"]["absrel"])
```

Key pieces:

- `evaluate_dense_depth_sample(pred, gt, valid_mask=None, alignment=..., min_depth=..., max_depth=..., benchmark_depth_range=...)` — one dense prediction vs a dense GT map. GT at another resolution is aligned to the prediction plane; returns the standard metric set plus pooled pixel statistics, or `None` when too few valid pixels remain.
- `evaluate_sparse_depth_sample(pred, point_cloud, intrinsics, camera_extrinsics, lidar_extrinsics=None, pred_is_radial=False, ...)` — projects the sparse GT cloud into the prediction plane (radial depth, nearest-z occlusion handling), converts a planar prediction to radial with the same intrinsics, and scores only the projected pixels — matching the CLI's `gt.sparse_depth` pipeline. Pass the intrinsics of the *actual* prediction plane (adjusted for any crop/resize of the model input).
- `alignment="none"` scores metric predictions as-is; `"affine"` fits least-squares scale+shift first (relative/affine depth models).
- `benchmark_depth_range=(MIN, MAX)` adds `result.benchmark` without changing the regular result; its `bins` mapping holds `all`, `near`, `mid`, `far` values using the CLI's square-root-spaced boundaries (empty bins are `None`).
- `DepthValidationAggregator` accumulates per-sample results into the CLI's `image_mean` / `image_median` / `pixel_pool` reducers. For multi-process validation, sum `aggregator.reduced_state()` vectors across ranks (fixed key order via `DepthValidationAggregator.state_keys()`) and rebuild mean-based summaries with `summarize_reduced_state(...)`.
- Inputs accept torch tensors or numpy arrays; all math runs on CPU numpy.

## License

[MIT](LICENSE)
