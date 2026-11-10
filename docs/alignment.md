# Spaces & alignment

Not every prediction is directly comparable to metric ground truth. A monocular
depth model may output values that are only correct up to a scale and a shift; a
point-map model may reconstruct the scene in its own frame entirely. Scoring
such a prediction as-is measures the missing calibration, not the geometry.

euler-eval handles this by fitting away exactly the unknown degrees of freedom
and reporting **both** results:

| Space | Meaning |
|---|---|
| `native` | The prediction as the model emitted it, compared to GT directly |
| `metric` | The prediction after the unknown gauge has been fitted to GT |

`native` diagnoses whether the model is metric at all; `metric` isolates
geometry quality. A canonical alias (`depth`, `points_3d`) points at `metric`
when it exists, else `native`, so downstream tooling can read one key without
knowing how the run was configured — see [Results & output](output.md).

## Depth alignment

Relative depth predictions are aligned to metric GT by fitting a global affine
map `d_aligned = s · d_pred + t` by least squares over valid pixels.
`--depth-alignment` controls when the fit runs:

| Mode | Behaviour |
|---|---|
| `none` | Never fit; score the prediction as-is (already-metric models) |
| `auto_affine` *(default)* | Fit only when the first prediction looks normalized (roughly `[-1, 1]`) |
| `affine` | Always fit |

The fit uses pixels where GT and prediction are valid: finite, GT > 0, and
neither equal to a declared `invalid.sentinel` ([details](configuration.md#loader-resolution--dataset-metadata)). With
`--mask-sky`, sky pixels are excluded and the fit is trimmed to `gt ≤ P95(gt)`
so residual sky outliers do not dominate. Pixels in a prediction's `sky_mask`
are excluded from the fit with `--sky-depth` or `--mask-sky-prediction`.

Declaring the prediction with the `relative_depth` or `affine_depth` config key
(instead of `depth`) tells the evaluator the model is not metric, so
`auto_affine` fits even when the values happen to fall outside the normalized
range — see [Configuration](configuration.md#datasets-section).

## Points-3D gauge alignment

Point maps from different model families live in different frames, so
`points_3d` resolves an unknown **similarity gauge** — the 3D generalization of
depth's scale-and-shift — before the `metric`-space comparison.
`--points-3d-alignment` selects it:

| Mode | Fits | Typical model |
|---|---|---|
| `none` | nothing | Metric, predict-your-own-camera models |
| `scale` | a single global scalar | Scale-ambiguous but frame-correct predictions |
| `similarity` | a 7-DoF Umeyama transform `s·R·p + t` over known per-pixel correspondences | Relative / free-frame models |
| `auto` *(default)* | `similarity` for declared-relative predictions, else nothing | Mixed comparisons |

The angular `error_decomposition` ray metrics and `rho_a` are computed on the
**native** point-map directions — the camera-faithful frame — because rotating
the cloud into GT would erase exactly the camera error they measure.
Radial/lateral and Euclidean metrics are reported per space.

## Resolution alignment

A prediction often comes at another resolution than its GT: the model ran on a
resized input, or cropped it to a multiple of 8. Before scoring, both go onto
one pixel grid.

**Which grid.** Dense depth and `points_3d` are scored on the **GT grid**. The
GT stays as measured and only the prediction is resampled (area average when
shrinking, bilinear when enlarging, at pixel centres), so:

- invalid GT pixels (`0` or non-finite) never blend into valid ones. Resizing
  sparse, lidar-derived GT would average a 10 m measurement with three empty
  neighbours into 2.5 m, which still counts as valid;
- GT values are never interpolated across object edges;
- the GT's intrinsics describe the evaluated grid unchanged, for the
  planar-to-radial conversion and surface normals.

RGB and rays are scored on the prediction grid as before, with the GT resized
onto it. GT and predicted masks are resized by nearest pixel centre
(`INTER_NEAREST_EXACT`), the convention of the image resize and of the
intrinsics update. GT depth used beside RGB (depth-binned metrics) is resized
without mixing in invalid pixels: valid pixels are area-averaged among
themselves, or, when its dataset head declares `"sparse": true`, each
measurement moves to its nearest target pixel and the nearest surface wins
(euler-loading's `resize_raster(..., interpolation="sparse")`).

**Crop or resize.** The shapes alone cannot tell whether a smaller prediction
is a crop of the input or a resized copy of it, and reading one as the other
misplaces pixels by up to the size difference.

**Recorded: `source_grid`.** A prediction dataset can record exactly how its
grid derives from the input image, as euler-dataset-contract's `source_grid`
(the source `size` and an affine pixel mapping). euler-inference writes it for
outputs that declare `outputs[].source_grid` (`resize`, `crop_top_left`,
`crop_center`, `identity`): in each file's attributes, and in the dataset
head when it is the same for every sample. Under `auto` it decides, per
sample (the file's value over the head's):

- a crop at any origin trims the GT to exactly the pixels the prediction
  covers, centre crops included; nothing is resampled;
- a whole-image resize goes through the same path as `resize` below;
- anything else, such as a resize followed by a crop, resamples through the
  exact mapping, onto the GT grid for dense depth and `points_3d` (GT pixels
  the prediction does not cover drop out) and onto the prediction grid for
  RGB, rays and GT masks (sparse GT depth moves sample by sample, nearest
  surface first);
- sparse point-cloud GT is projected with the intrinsics moved onto the
  prediction grid (`T @ K`), as is the prediction's planar-to-radial
  conversion.

The run records `spatial_alignment.mode: "source_grid"`. A grid recorded for
another source size than the GT's cannot place the prediction; the shapes are
guessed from instead, with a warning.

**Without a record**, `--spatial-alignment`, or a `datasets[]` entry's
`spatial_alignment`, says which:

| Mode | Behaviour |
|---|---|
| `auto` *(default)* | The recorded `source_grid` when there is one; else a crop when the prediction's sides are multiples of 8 and at most 7 px shorter than the GT's (latent models that crop), else a resize |
| `crop` | Always a top-left crop: the GT is trimmed to the region the prediction covers. A prediction larger than the GT is an error |
| `resize` | Always a resize, even for the multiple-of-8 pattern |

An explicit `crop` or `resize` overrides a recorded grid. Set one whenever a
model resizes its input to a multiple of 8 without recording it: a 879 px wide
DrivingStereo image resized to 872 px would otherwise be scored as a crop,
misplacing pixels at the right edge by up to 7 px. The method that ran is
recorded as `spatial_alignment.method` (`none`, `vae_crop`, `crop`,
`resize_prediction`, `resize`, `warp_prediction` or `warp`) and the evaluated
grid as `population.evaluated_resolution`; see [Results & output](output.md).

## Benchmark depth bins

`--benchmark-depth-range MIN MAX` filters valid GT depth pixels to `[MIN, MAX]`,
then splits them into three equal-width intervals in **square-root** depth
space, which keeps the near field from being compressed into a single bin:

```text
sqrt_min = sqrt(MIN);  sqrt_max = sqrt(MAX);  step = (sqrt_max - sqrt_min) / 3
near = [MIN, (sqrt_min + step)^2)
mid  = [(sqrt_min + step)^2, (sqrt_min + 2·step)^2)
far  = [(sqrt_min + 2·step)^2, MAX]
```

For `--benchmark-depth-range 0.01 80.0` this yields `near=[0.01, 9.29)`,
`mid=[9.29, 35.95)`, `far=[35.95, 80.0]`, plus an `all` bin over the full range.

Benchmark bins are **additive**: the regular metrics are still computed and
written exactly as before, with the binned values added under a `bin` axis
(`…standard.image_mean.near.absrel`) and the resolved boundaries recorded in the
`metricSet` metadata. RGB metrics are binned too, using GT depth to decide which
pixels fall in which bin.

The same binning is available programmatically through
`benchmark_depth_range=(MIN, MAX)` — see
[In-training validation](validation.md).

With `--distributions`, default depth dataset/per-file histograms use the
benchmark `all` population, with optional sky-depth capping applied before GT
range selection. Their counts match the explicit `all` histogram, while the
unbinned scalar summaries retain their existing population. See
[Error distributions](distributions.md#benchmark-depth-range-and-sky-depth).
