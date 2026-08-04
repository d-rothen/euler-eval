# CLI reference

```bash
euler-eval <config> [options]
```

The config is the only positional argument — a JSON file describing the ground
truth and the prediction datasets to score against it, documented in
[Configuration](configuration.md). Invalid JSON or a config that fails
validation exits with status `1` before any dataset is opened.

## Options

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
| `--depth-alignment` | `{none,auto_affine,affine}` | `auto_affine` | Depth calibration mode ([details](alignment.md#depth-alignment)) |
| `--points-3d-alignment` | `{none,scale,similarity,auto}` | `auto` | points_3d gauge alignment ([details](alignment.md#points-3d-gauge-alignment)) |
| `--rgb-fid-backend` | `{builtin,clean-fid}` | `builtin` | RGB FID backend; `clean-fid` requires the `[fid]` extra |
| `--benchmark-depth-range` | `float float` | none | Depth range `[MIN, MAX]` (m) for near/mid/far benchmark bins ([details](alignment.md#benchmark-depth-bins)) |

## Examples

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

## What a run prints

Before any dataset is opened, the run prints its own provenance — package
version, Python, torch and CUDA versions — which is also what gets recorded in
`meta` and, when configured, in the euler-train run. It then echoes the resolved
device, alignment mode, FID backend and benchmark range, followed by one block
per prediction dataset and modality with the GT and prediction paths, the
resolved modality metadata (such as `radial_depth`), and the number of matched
pairs.

A matched-pair count that is much lower than expected is the first sign that GT
and predictions are not indexed consistently — see
[Loader resolution & dataset metadata](configuration.md#loader-resolution--dataset-metadata).

## Device selection

`--device auto` resolves to CUDA when `torch.cuda.is_available()`, otherwise
CPU. Requesting `--device cuda` on a machine without CUDA prints a warning and
falls back to CPU rather than failing.

On CUDA, the run enables cuDNN benchmarking and TF32 matmuls — throughput knobs
that do not change which metrics are computed.

`--batch-size` applies to the batched image-quality metrics (PSNR, SSIM, LPIPS,
FID/KID); `--num-workers` is passed to the underlying dataloaders. Sparse-depth
and points-3d evaluation is CPU/NumPy work, so `--num-workers` matters more than
`--batch-size` there.

## Offline cache warmup

Compute nodes without network access need the model weights pre-fetched. Run the
cache warmup helper on a machine that does have network access, pointing the
cache environment variables at a shared location:

```bash
HF_HOME=/shared/cache/hf \
TORCH_HOME=/shared/cache/hf/torch \
CLEANFID_CACHE_DIR=/shared/cache/clean-fid \
euler-eval.init
```

This pre-downloads the torchvision AlexNet and Inception v3 weights (used by
LPIPS and the builtin FID/KID), the LPIPS AlexNet weights, and the clean-fid
Inception checkpoint if `clean-fid` is installed. `TORCH_HOME` is derived from
`HF_HOME` when it is not set explicitly, so pointing `HF_HOME` at the shared
cache is usually enough.

Set the same variables for the evaluation run itself so the cached weights are
found.
