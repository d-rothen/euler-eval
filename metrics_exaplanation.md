# Metrics Explanation

This document describes the metrics as they are actually computed in this repository. It is based on the current implementations in `euler_eval/evaluate.py`, `euler_eval/data.py`, and `euler_eval/metrics/*`.

## Shared evaluation conventions

- Ground truth is spatially aligned to the prediction before metrics are computed.
  - If the mismatch looks like a common VAE crop, GT is cropped from the top-left.
  - Otherwise GT is resized to the prediction resolution.
- Depth is evaluated in metres.
  - If a depth modality is marked as planar and camera intrinsics are available, it is converted to radial depth before evaluation.
- Most depth metrics only use valid pixels where both prediction and GT are finite and strictly positive.
- If sky masking is enabled:
  - Depth metrics exclude sky pixels from the valid mask.
  - RGB metrics zero out sky pixels in both GT and prediction before computing the metric.
- Depth evaluation has two branches:
  - `depth_raw`: metric on the processed prediction as-is.
  - `depth_aligned`: metric after optional affine alignment `s * pred + t`.
- In `auto_affine`, affine alignment is only applied if the first prediction looks normalized (roughly within `[-1, 1]`).
- RGB metrics assume images are already in `[0, 1]`.

## Benchmark depth-range bins

When `--benchmark-depth-range MIN MAX` is provided, benchmark summaries use only
valid GT depth pixels where `MIN <= depth <= MAX`. The range is subdivided into
`near`, `mid`, and `far` by taking three equal-width intervals in square-root
depth space and squaring the split points back to metres:

```text
sqrt_min = sqrt(MIN)
sqrt_max = sqrt(MAX)
step = (sqrt_max - sqrt_min) / 3
near_max = (sqrt_min + step)^2
mid_max = (sqrt_min + 2 * step)^2
```

The masks are `near=[MIN, near_max)`, `mid=[near_max, mid_max)`, and
`far=[mid_max, MAX]`; the `all` mask is the full inclusive range.

For `MIN=0.01` and `MAX=80.0`, the computed bins are:

| Bin | Depth interval |
|---|---|
| `all` | `[0.01, 80.0]` |
| `near` | `[0.01, 9.290856529)` |
| `mid` | `[9.290856529, 35.954189863)` |
| `far` | `[35.954189863, 80.0]` |

## Depth metrics

### PSNR

- Computation:
  - Over valid pixels, compute `MSE = mean((pred - gt)^2)`.
  - Compute `PSNR = 10 * log10(max_val^2 / MSE)`.
  - If `max_val` is not supplied, the implementation uses the maximum valid GT depth for that image.
  - If `MSE` is essentially zero, it returns `inf`.
- Reported as:
  - Per file: the image PSNR.
  - Dataset: the mean of per-image PSNR values.
- Intuition:
  - A standard image-fidelity score on depth values in metres.
  - It rewards small absolute errors and penalizes large errors quadratically through MSE.
- Output / range:
  - Unbounded above; can be negative in bad cases.
  - Higher is better.

### SSIM

- Computation:
  - Valid pixels are found first.
  - Prediction and GT are jointly normalized to `[0, 1]` using the shared min/max over valid pixels in that pair.
  - Invalid pixels are set to `0`.
  - A Gaussian-window SSIM map is computed with `window_size=11`, `k1=0.01`, `k2=0.03`.
  - The final score is the mean SSIM over a blurred-validity mask thresholded at `> 0.5`.
- Reported as:
  - Per file: the image SSIM.
  - Dataset: the mean of per-image SSIM values.
- Intuition:
  - Measures whether local depth structure is preserved, not just whether the absolute values are close.
  - Because the pair is normalized before SSIM, this is more about relative structure than raw metric scale.
- Output / range:
  - Intended to be around `[0, 1]`.
  - Higher is better.

### LPIPS

- Computation:
  - Each depth map is normalized independently to `[0, 1]` using its own valid min/max.
  - It is then clipped, rescaled to `[-1, 1]`, replicated to 3 channels, and passed to LPIPS (`alex` backbone by default).
- Reported as:
  - Per file: the image LPIPS.
  - Dataset: the mean of per-image LPIPS values.
- Intuition:
  - Measures perceptual or structural similarity after turning depth into a pseudo-image.
  - Because each map is normalized independently, this score largely ignores absolute metric scale and focuses on shape/detail similarity.
- Output / range:
  - Non-negative, with `0` meaning identical after preprocessing.
  - No fixed upper bound.
  - Lower is better.

### FID

- Computation:
  - This is dataset-level only.
  - Each depth map is normalized independently to `[0, 1]`, converted to a 3-channel image, resized to `299 x 299`, normalized with ImageNet statistics, and embedded with Inception v3.
  - FID is then computed from the feature means and covariances of GT and prediction sets.
- Reported as:
  - Dataset: a single FID value.
- Intuition:
  - Measures whether the distribution of predicted depth maps looks like the GT distribution in Inception feature space.
  - It is a set-level realism/distribution score, not a per-pixel reconstruction score.
- Output / range:
  - Ideally `0`.
  - No fixed upper bound.
  - Lower is better.

### KID

- Computation:
  - Uses the same Inception features as FID.
  - Computes an MMD-style distance with the polynomial kernel `((x^T y) / d + 1)^3`.
  - The implementation samples `100` random subsets and reports the mean and standard deviation across subsets.
- Reported as:
  - Dataset: `kid_mean` and `kid_std`.
- Intuition:
  - Another set-level distribution metric, similar in purpose to FID but based on kernel two-sample testing.
- Output / range:
  - Ideally close to `0`.
  - `kid_mean` can be slightly negative because the estimator is unbiased.
  - Lower is better in magnitude.

### AbsRel

- Computation:
  - For each valid pixel, compute `abs(pred - gt) / gt`.
- Reported as:
  - Per file: the mean AbsRel over valid pixels in that image.
  - Dataset: the median and 90th percentile over all valid pixels from all images pooled together.
- Intuition:
  - Measures relative depth error rather than absolute metre error.
  - A `10%` error at `10 m` and at `1 m` are treated the same.
- Output / range:
  - `[0, inf)`.
  - `0` is perfect.
  - Lower is better.

### RMSE

- Computation:
  - The repository stores per-pixel squared errors `(pred - gt)^2`.
  - Per-file RMSE is `sqrt(mean(squared_error))`.
  - Dataset `median` and `p90` are computed from pooled per-pixel absolute errors `sqrt((pred - gt)^2) = |pred - gt|`, not from per-image RMSE values.
- Reported as:
  - Per file: standard RMSE in metres.
  - Dataset: median and 90th percentile of pooled per-pixel absolute error, in metres.
- Intuition:
  - Captures absolute depth error in metric units.
  - Large mistakes hurt more strongly in the per-file RMSE because they are squared before averaging.
- Output / range:
  - `[0, inf)`.
  - `0` is perfect.
  - Lower is better.

### SILog

- Computation:
  - For valid pixels, define `d_i = log(pred_i) - log(gt_i)`.
  - The scalar SILog value is:
    - `sqrt(mean(d_i^2) - mean(d_i)^2)`
  - This uses `lambda_weight = 1.0`.
  - The evaluator also stores per-pixel `|log(pred) - log(gt)|` values for percentile summaries.
- Reported as:
  - Per file: the scalar SILog value above.
  - Dataset:
    - `mean`: mean of per-image scalar SILog values.
    - `median` and `p90`: median and 90th percentile of pooled per-pixel absolute log differences.
- Intuition:
  - Measures multiplicative or relative depth consistency.
  - Global scale shifts matter less than in plain RMSE.
- Output / range:
  - `[0, inf)`.
  - `0` is perfect.
  - Lower is better.

### Normal consistency

- Computation:
  - A normal is estimated from each depth map using Sobel depth gradients.
  - The normal formula is the simple image-space form `n = (-dz/dx, -dz/dy, 1)`, then normalized.
  - The valid mask is eroded with a `3 x 3` kernel before comparing normals to avoid border artifacts.
  - Angular error is `acos(dot(n_pred, n_gt))` in degrees.
  - Although the normal code accepts a focal length, the evaluator currently uses the default `1.0`, so this is effectively an image-space relative normal comparison.
- Reported as:
  - Per file: mean angular normal error.
  - Dataset:
    - mean angle
    - median angle
    - percent below `11.25°`
    - percent below `22.5°`
    - percent below `30°`
  - These dataset summaries are computed from pooled per-pixel angles.
- Intuition:
  - Tests whether local surface orientation is correct, which is often more geometric than raw depth difference.
  - Good for assessing shape quality and boundary orientation.
- Output / range:
  - Angular error is in `[0°, 180°]`.
  - Lower mean/median angle is better.
  - Higher threshold percentages are better.

### Depth Edge F1

- Computation:
  - Edges are detected independently in GT and prediction.
  - The default method is `relative`:
    - compute horizontal and vertical 4-neighbor depth differences
    - keep the max of those differences
    - mark an edge where `max_diff > 0.1 * local_depth`
  - Matching uses a `1`-pixel tolerance via binary dilation.
  - Precision, recall, and F1 are then computed on the edge maps.
- Reported as:
  - Per file: precision, recall, F1.
  - Dataset: mean of per-image precision, recall, and F1 over images that contain at least one predicted or GT edge.
- Intuition:
  - Tests whether depth discontinuities are preserved in the right places.
  - Useful for evaluating object boundaries and depth jumps.
- Output / range:
  - Precision, recall, and F1 are in `[0, 1]`.
  - Higher is better.

## RGB metrics

### PSNR

- Computation:
  - Compute `MSE = mean((pred - gt)^2)` over all RGB values.
  - Use `max_val = 1.0`.
  - If `MSE` is essentially zero, return `inf`.
  - If sky masking is enabled, both images are zeroed in masked regions before the metric is computed.
- Reported as:
  - Per file: image PSNR.
  - Dataset: mean of per-image PSNR values.
- Intuition:
  - Standard fidelity score for normalized RGB reconstruction.
- Output / range:
  - Unbounded above; can be negative in bad cases.
  - Higher is better.

### SSIM

- Computation:
  - Standard Gaussian-window SSIM is computed independently on each RGB channel and the 3 channel scores are averaged.
  - Uses `window_size=11`, `k1=0.01`, `k2=0.03`.
  - If sky masking is enabled, it operates on the zeroed images.
- Reported as:
  - Per file: image SSIM.
  - Dataset: mean of per-image SSIM values.
- Intuition:
  - Measures structural similarity, not just per-pixel color error.
  - Useful for blur, contrast, and local texture preservation.
- Output / range:
  - Intended to be around `[0, 1]`.
  - Higher is better.

### SCE

- Computation:
  - Convert each pixel to chromaticity by dividing RGB by `R + G + B + eps`.
  - Compute per-pixel chromaticity error as the mean absolute difference across channels.
  - Compute grayscale gradients for GT and prediction with a Scharr filter.
  - Build a symmetric structural weight `0.5 * (grad_pred + grad_gt)`.
  - Return the weighted mean chromaticity error divided by the mean structural weight.
- Reported as:
  - Per file: image SCE.
  - Dataset: mean of per-image SCE values.
- Intuition:
  - Focuses on color proportions rather than brightness alone.
  - The structural weighting emphasizes errors on edges and textured regions more than errors in flat regions.
- Output / range:
  - Non-negative.
  - `0` is perfect.
  - Lower is better.

### LPIPS

- Computation:
  - RGB images are rescaled from `[0, 1]` to `[-1, 1]` and passed to LPIPS (`alex` by default).
- Reported as:
  - Per file: image LPIPS.
  - Dataset: mean of per-image LPIPS values.
- Intuition:
  - Measures perceptual similarity using deep features rather than raw pixel differences.
- Output / range:
  - Non-negative, with `0` meaning identical after preprocessing.
  - No fixed upper bound.
  - Lower is better.

### FID

- Computation:
  - This is dataset-level only.
  - The evaluator supports two backends.
  - `builtin`:
    - RGB images are clipped to `[0, 1]`, resized so that the shorter side becomes `299` while preserving aspect ratio, then symmetrically padded within each batch to a common size.
    - The padded tensors are normalized with ImageNet statistics and embedded with Inception v3.
  - `clean-fid`:
    - RGB images are written to temporary PNG files and delegated to the official `clean-fid` implementation, which applies its own image loading, quantization, and resizing pipeline.
  - If sky masking is enabled, FID is computed on the masked images after sky pixels are zeroed in both GT and prediction.
  - FID is then computed from the feature means and covariances of the GT and prediction sets.
- Reported as:
  - Dataset: a single FID value.
- Intuition:
  - Measures whether the distribution of predicted RGB images matches the GT distribution in Inception feature space.
  - It is a set-level realism/distribution score, not a per-image reconstruction score.
- Output / range:
  - Ideally `0`.
  - No fixed upper bound.
  - Lower is better.

### Edge F1

- Computation:
  - Convert RGB to grayscale.
  - Detect edges with Sobel gradient magnitude and threshold `0.1`.
  - Match predicted and GT edges with a `1`-pixel dilation tolerance.
  - Compute precision, recall, and F1.
- Reported as:
  - Per file: precision, recall, F1.
  - Dataset: mean of per-image precision, recall, and F1 over images that contain at least one predicted or GT edge.
- Intuition:
  - Measures whether visible contours appear in the right places.
  - Useful for blur, edge loss, and misplaced boundaries.
- Output / range:
  - Precision, recall, and F1 are in `[0, 1]`.
  - Higher is better.

### Tail errors (`p95`, `p99`)

- Computation:
  - For each pixel, compute mean absolute RGB error:
    - `abs_error = mean(abs(pred - gt), axis=channel)`
  - Per-file `p95` and `p99` are percentiles of that per-pixel error map.
  - Dataset `p95` and `p99` are computed after concatenating all per-pixel error values from all images.
- Reported as:
  - Per file: `p95`, `p99`.
  - Dataset: pooled `p95`, pooled `p99`.
- Intuition:
  - Designed to expose rare but ugly localized failures that average metrics can hide.
  - A model can have good mean quality and still have bad `p99`.
- Output / range:
  - With normalized RGB, these values lie in `[0, 1]`.
  - `0` is perfect.
  - Lower is better.

### High-frequency energy

- Computation:
  - Convert the image to grayscale.
  - Compute the 2D FFT and power spectrum.
  - Mark as "high frequency" the outer radial band where normalized frequency distance is greater than `0.75` (`cutoff_ratio=0.25`).
  - Compute:
    - `pred_hf_ratio = high_freq_energy / total_energy`
    - `gt_hf_ratio = high_freq_energy / total_energy`
    - `relative_diff = (pred_hf_ratio - gt_hf_ratio) / gt_hf_ratio`
      when `gt_hf_ratio > 0`
- Reported as:
  - Per file: `pred_hf_ratio`, `gt_hf_ratio`, `relative_diff`.
  - Dataset: mean of those per-image quantities.
- Intuition:
  - Detects oversmoothing, detail loss, or artificial sharpening/noise.
  - If `relative_diff < 0`, the prediction has less high-frequency energy than GT.
  - If `relative_diff > 0`, it has more.
- Output / range:
  - `pred_hf_ratio` and `gt_hf_ratio` are in `[0, 1]`.
  - `relative_diff` is ideally `0`, can be negative, positive, or `inf` in degenerate cases.
  - For fidelity to GT, closer to `0` is better for `relative_diff`.

### Depth-binned photometric error

- Computation:
  - Requires GT depth.
  - GT depth is converted to metres/radial depth first, then aligned to the RGB prediction size if needed.
  - Pixels are split into depth bins:
    - `near`: `depth <= 1.0`
    - `mid`: `1.0 < depth <= 5.0`
    - `far`: `depth > 5.0`
  - For each pixel, compute:
    - `MAE = mean(abs(pred - gt), axis=channel)`
    - `MSE = mean((pred - gt)^2, axis=channel)`
  - Report the mean MAE and mean MSE inside each bin, plus `all`.
  - If a bin is empty in an image, the implementation records `0.0` for that image.
- Reported as:
  - Per file: nested `mae` and `mse` dictionaries with `near`, `mid`, `far`, `all`.
  - Dataset: mean of the per-image bin values.
- Intuition:
  - Tells you whether RGB quality degrades differently at close, medium, or far depth.
  - Useful when models behave differently on foreground vs background.
- Output / range:
  - With normalized RGB, MAE and MSE are in `[0, 1]`.
  - `0` is perfect.
  - Lower is better.

## Directional ray metrics

These metrics operate on per-pixel direction maps with shape `(H, W, 3)`. Vectors are normalized inside the metric before comparison, so the metric uses direction, not vector magnitude.

### Angular error

- Computation:
  - Valid pixels are those where GT and prediction both have finite, non-zero vector norm.
  - Normalize both vectors to unit length.
  - Compute `angle = acos(clamp(dot(pred, gt), -1, 1))` in degrees.
- Reported as:
  - Per file: mean and median angular error.
  - Dataset:
    - mean angle
    - approximate median angle
    - percent below `5°`, `10°`, `15°`, `20°`, `30°`
  - The dataset aggregate is pooled over all valid pixels.
- Intuition:
  - Directly measures how wrong each predicted ray direction is.
  - This is the most literal geometric error for direction maps.
- Output / range:
  - `[0°, 180°]`.
  - `0°` is perfect.
  - Lower is better.

### `rho_a`

- Computation:
  - Build the angular accuracy curve:
    - for thresholds `theta` between `0` and `T`, compute the fraction of pixels with angular error `<= theta`
  - Compute the normalized AUC of that curve with trapezoidal integration.
  - The threshold `T` depends on FoV domain:
    - `sfov`: `15°`
    - `lfov`: `20°`
    - `pano`: `30°`
  - If FoV domain is not supplied, the evaluator auto-detects it from the first sample's intrinsics using diagonal FoV:
    - `<= 90°` -> `sfov`
    - `>= 160°` -> `pano`
    - otherwise `lfov`
  - If there are no intrinsics, it defaults to `lfov`.
- Reported as:
  - Per file: one `rho_a` score.
  - Dataset: mean, median, and standard deviation over per-image `rho_a` values.
- Intuition:
  - Summarizes the whole angular error distribution into one bounded score.
  - It rewards having many pixels below small angular error, not just having a good mean.
  - The FoV-dependent threshold makes the score less unfair across camera types.
- Output / range:
  - `[0, 1]`.
  - `1` is perfect.
  - Higher is better.

## Points-3D metrics

These metrics operate on per-pixel 3D point maps with shape `(H, W, 3)` storing
camera-frame `(X, Y, Z)` coordinates in metres, as produced by models that predict
a point map together with their own camera model. Ground truth is either an
explicit per-pixel point map (`gt.points_3d`) or, when no point map is configured,
synthesized on the fly by unprojecting GT depth (`gt.depth`) with GT intrinsics
(`gt.intrinsics` or `gt.calibration`) — radial vs planar GT depth is read from the
depth dataset metadata.

### Shared conventions

- GT is spatially aligned to the prediction grid exactly as for the other modalities.
- A pixel is valid when both GT and prediction are finite and the GT point has
  non-zero norm (intersected with the non-sky mask when `--mask-sky` is set).
- A **gauge alignment** is resolved before the `metric`-space comparison, the 3D
  analog of depth scale-and-shift:
  - `none`: compare raw points (metric models).
  - `scale`: fit a single global scalar `s` minimizing `Σ‖s·p − g‖²`.
  - `similarity`: fit a 7-DoF Umeyama transform `s·R·p + t` over the known per-pixel
    correspondences (relative models).
  - `auto`: `similarity` for declared-relative predictions, else `none`.
- Two spaces are emitted: `native` (raw) and, when alignment runs, `metric`
  (aligned), with `points_3d` aliasing the canonical (metric when present, else
  native). The similarity fit is per image.
- Let `d = pred − gt` per valid pixel, `e = ‖d‖`, and `r = gt / ‖gt‖` the GT ray.

### Euclidean 3D agreement (`point_error`)

- `mae3d` / `rmse3d`: mean and root-mean-square of `e` (3D end-point error), metres.
- `median3d`, `p90`, `p95`: pooled percentiles of `e`.
- `rel_median`, `rel_p90`: percentiles of `e / ‖gt‖` (scale-invariant, AbsRel analog).
- `acc_<τ>`: percentage of points with `e < τ` metres (3D δ-accuracy), for
  `τ ∈ {0.05, 0.1, 0.25, 0.5, 1.0}`; `acc_rel_<τ>` uses the relative error for
  `τ ∈ {0.05, 0.1, 0.25}`.
- Reported with three reducers: `image_mean` and `image_median` (per-image scalars
  averaged / medianed across images) and `pixel_pool` (pooled over all valid points).
- Lower error / higher accuracy is better.

### Error decomposition (`error_decomposition`)

- The error vector is split relative to the GT ray:
  - radial `a = d · r` (≈ depth error): `radial_mae = mean|a|`, `radial_rmse`.
  - lateral `l = ‖d − a·r‖` (≈ camera-model error): `lateral_mae`, `lateral_rmse`.
- `lateral_fraction = mean(l) / (mean|a| + mean(l)) ∈ [0, 1]`: near 0 the error is
  depth-dominated, near 1 it is camera-model-dominated.
- `angular_error` and `rho_a`: the ray angular error and ρ_A (identical to the rays
  modality) computed on the **native** point-map directions — the camera-faithful
  frame — and attached to every emitted space.
- Intuition: tells you whether a model should fix its depth head or its camera head.

### Geometric (`geometric`)

- `normal_consistency`: surface normals computed **directly from the point map** via
  the cross product of central-difference tangents (no assumed focal length, unlike
  the depth `normal_consistency`). Reports `mean_angle`, `median_angle`, and
  `percent_below_{11.25, 22.5, 30}°` over an eroded valid mask.
- `point_edge_f1`: precision/recall/F1 of 3D discontinuities, where an edge is a
  pixel whose maximum 4-neighbour 3D distance exceeds `0.1·‖P‖`, matched with a
  1-pixel dilation tolerance.

### Cloud distance (`cloud_distance`)

- Correspondence-free set agreement, robust to camera mis-parameterization that
  merely slides points along the surface (which `point_error` penalizes hard).
- `chamfer`: `accuracy` (mean nearest-neighbour distance pred→gt), `completeness`
  (gt→pred), `distance` (their mean), and `median` of the pooled distances, metres.
- `fscore.tau_<τ>`: precision/recall/F1 of points within `τ` metres of the other
  cloud, for `τ ∈ {0.05, 0.1, 0.25, 0.5}`. F-score is the bounded, outlier-robust
  headline; raw Chamfer is the diagnostic.
- Clouds are deterministically subsampled (default cap 50k points/cloud) to bound
  the KD-tree cost.
