# Metrics

Every metric below is written under a structured key built with
[euler-metric-naming](https://github.com/d-rothen/euler-metric-naming). The keys
in this page are the paths inside `eval.json`; see
[Results & output](output.md) for the surrounding envelope.

## Reducers

Depth-like metrics are reported under three reducers, because they answer
different questions:

| Reducer | How it aggregates | Reads as |
|---|---|---|
| `image_mean` | Per-image value, averaged over the dataset | Every image counts equally |
| `image_median` | Per-image value, median over the dataset | Robust to a few catastrophic frames |
| `pixel_pool` | All valid pixels pooled, reduced once | Every valid pixel counts equally |

A dataset with a few very large or very dense images will separate
`pixel_pool` from `image_mean`; a dataset with a handful of failures will
separate `image_mean` from `image_median`.

## Depth

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

The `standard` block is the conventional monocular-depth metric set — the one
most papers report — and `delta1-3` are the `δ < 1.25ⁿ` accuracy thresholds.

## Sparse depth

With a sparse pointcloud GT, only metrics that stay meaningful at isolated
projected points are reported; dense image-quality and geometric metrics are
skipped. Results serialize under the `sparsedepth.eval` root:

| Metric | Key |
|---|---|
| Standard | `sparsedepth.eval.{native,metric}.standard.{image_mean,image_median,pixel_pool}.*` |
| AbsRel / RMSE / SILog | `sparsedepth.eval.{native,metric}.depth_metrics.*` |

## RGB

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

## Rays

| Metric | Key | Description |
|---|---|---|
| ρ_A | `rays.rho_a.mean`, `rho_a.median` | AUC of the angular-accuracy curve, integrated to a FoV-dependent threshold (S.FoV 15°, L.FoV 20°, Pano 30°) |
| Angular error | `rays.angular_error.mean_angle`, `median_angle` | Per-pixel angular error (deg) |
| Thresholds | `rays.angular_error.percent_below_*` | % of pixels below 5°/10°/15°/20°/30° |

## Points-3D

For models that predict a per-pixel 3D point map — and, implicitly, their own
camera. Keyed as `points3d.eval.{native,metric}.{category}…`, with a `points_3d`
canonical alias.

| Category | Key (per space) | Description |
|---|---|---|
| Euclidean 3D agreement | `point_error.{image_mean,image_median,pixel_pool}.*` | 3D EPE `mae3d`/`rmse3d`/`median3d`/`p90`/`p95`, relative error, δ-accuracy `acc_<τ>` / `acc_rel_<τ>` |
| Error decomposition | `error_decomposition.{radial_*,lateral_*,lateral_fraction}` + `angular_error.*` + `rho_a.*` | Radial (≈ depth) vs lateral (≈ camera-model) split; `lateral_fraction ∈ [0,1]` attributes blame |
| Geometric | `geometric.normal_consistency.*`, `geometric.point_edge_f1.*` | True-3D surface normals and 3D discontinuity F1 |
| Cloud distance | `cloud_distance.chamfer.*`, `cloud_distance.fscore.tau_<τ>.*`, `cloud_distance.f_a` | Correspondence-free Chamfer, F-score at fixed thresholds, and F-score AUC |

The error decomposition is the reason to evaluate point maps rather than depth
alone: it separates error *along* the viewing ray (a depth mistake) from error
*across* it (a camera-model mistake), and `lateral_fraction` says which of the
two dominates.

## Sparse points-3D

With a sparse pointcloud GT, only the categories that stay meaningful are
emitted — dense-neighbourhood `geometric` metrics are skipped — and
`cloud_distance` reports only the directed `gt→pred` side
(completeness/recall), since a correct dense prediction legitimately contains
many points far from any sparse GT return. Results serialize under
`points3d.eval.{native,metric}`.

## Benchmark bins

When `--benchmark-depth-range` is set, depth and RGB metrics are additionally
computed per depth bin under a `bin` axis (`all`, `near`, `mid`, `far`). See
[Benchmark depth bins](alignment.md#benchmark-depth-bins).
