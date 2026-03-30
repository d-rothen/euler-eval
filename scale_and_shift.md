# Scale-and-Shift Depth Alignment

This note describes how relative depth is aligned to metric ground-truth depth in this repository.

## Why alignment is needed

Many depth models predict a relative depth map rather than depth in metres. Typical outputs are:

- normalized to `[0, 1]`
- normalized to `[-1, 1]`
- inverted in sign or ordering
- correct up to an unknown global scale and offset

For evaluation, we therefore fit an affine mapping from prediction to GT depth:

`d_aligned = s * d_pred + t`

where:

- `d_pred` is the predicted depth
- `d_aligned` is the aligned prediction in metric units
- `s` is a scalar scale
- `t` is a scalar shift

## Optimization problem

Given a set of valid pixels `V`, we solve:

`min_{s,t} sum_{i in V} (s * p_i + t - g_i)^2`

where:

- `p_i` is the prediction at pixel `i`
- `g_i` is the GT depth at pixel `i`

This is a 2-parameter least-squares problem. In matrix form:

`A = [[p_1, 1], [p_2, 1], ..., [p_n, 1]]`

`x = [s, t]^T`

`b = [g_1, g_2, ..., g_n]^T`

and we solve:

`min_x ||A x - b||_2^2`

The implementation uses `numpy.linalg.lstsq`, which is equivalent to solving the normal equations but is more numerically robust.

## Closed-form solution

If we write:

- `S_p = sum p_i`
- `S_g = sum g_i`
- `S_pp = sum p_i^2`
- `S_pg = sum p_i g_i`
- `n = |V|`

then, when the system is not degenerate, the solution is:

`D = n * S_pp - S_p^2`

`s = (n * S_pg - S_p * S_g) / D`

`t = (S_pp * S_g - S_p * S_pg) / D`

This is the affine fit that best matches GT in the least-squares sense.

## Which pixels are used

The fit is not done over every pixel blindly.

### Default valid mask

If no explicit mask is given, the fit uses pixels where:

- `gt > 0`
- `gt` is finite
- `pred` is finite

This excludes zero-depth GT, `NaN`, and `inf`.

### Sky masking

When `--mask-sky` is enabled, the fit mask is additionally intersected with the non-sky mask derived from GT segmentation.

### P95 trimming under sky masking

Even with sky masking, some datasets still contain residual sky pixels because segmentation is imperfect. Those pixels often have extremely large GT depth and can dominate the least-squares fit.

To reduce that failure mode, this repository now trims the fit to:

`gt <= P95(gt_valid)`

but only for the alignment fit, and only when sky masking is enabled.

Important:

- this does not change the metric definitions themselves
- it only changes which pixels influence `s` and `t`
- trimming is only applied if at least 2 valid pixels remain

## Where alignment happens in the pipeline

For depth evaluation, the sequence is:

1. Spatially align GT to prediction resolution.
2. Convert planar depth to radial depth if needed.
3. Decide whether affine alignment should run:
   - `none`: never
   - `auto_affine`: only if the first prediction looks normalized, roughly in `[-1, 1]`
   - `affine`: always
4. Build the fit mask.
5. Fit `s` and `t`.
6. Apply `d_aligned = s * d_pred + t`.
7. Evaluate metrics on both:
   - `depth_raw`
   - `depth_aligned`

One subtlety:

- if predictions are detected as normalized in `auto_affine`, the fit is done on the original normalized prediction values
- otherwise the fit is done on the processed prediction (for example after planar-to-radial conversion if applicable)

## Interpretation of the fitted parameters

### Positive scale

`s > 0` means prediction ordering agrees with GT ordering. This is the common case.

### Negative scale

`s < 0` means the prediction is effectively inverted relative to GT. This can happen if a model predicts "far is small" while GT uses "far is large", or vice versa. The least-squares fit can still recover a good aligned map by flipping the sign.

### Shift

`t` corrects the global offset after scaling. This matters when the model output is not centered on the GT range.

## Edge cases

### Too few valid pixels

If fewer than 2 valid pixels remain, the repository does not try to fit an affine mapping. It returns:

- aligned prediction = unchanged prediction
- `s = 1`
- `t = 0`

This avoids fitting nonsense from underdetermined data.

### Constant or near-constant prediction

If `d_pred` is constant, the scale-and-shift problem is rank-deficient because the `pred` column and the constant column are linearly dependent. In that case:

- `s` and `t` are not uniquely identifiable
- the least-squares solution still gives the best constant aligned output
- in practice, the aligned map collapses to the mean GT depth over valid pixels

So the aligned image is meaningful, even though the individual parameters are not uniquely interpretable.

### Outliers in GT depth

Least-squares is sensitive to large residuals because errors are squared. A small number of extreme GT values can move the solution substantially, especially if those values are much larger than the rest of the scene. This is the reason for the P95 trimming in sky-mask mode.

### Nonlinear relative depth

This method assumes the mapping from prediction to GT is well-approximated by a global affine transform. If a model output is related to metric depth by a strongly nonlinear function, affine alignment will only be an approximation.

### Perfect affine match

If the prediction is already related to GT by a true affine mapping, the fit should recover that mapping exactly up to floating-point precision.

### Already metric prediction

If prediction is already in metric depth, affine fitting is often unnecessary. That is why `auto_affine` first checks whether predictions look normalized; if they do not, alignment is skipped by default.

## What this alignment does not do

This method does not:

- enforce positivity after alignment
- preserve monotonicity if the fitted scale is negative
- perform per-region or per-object fitting
- fit nonlinear transforms such as inverse-depth or log-depth models directly

It is a simple global affine calibration step.

## Practical takeaway

The repository treats relative depth alignment as:

`find the single global scale and shift that best map prediction to GT over a trustworthy set of pixels`

The quality of that trustworthy set matters as much as the least-squares solver itself. That is why masking and the new P95 trimming are important for robust evaluation in scenes with sky-depth outliers.
