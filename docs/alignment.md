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

The fit uses pixels where GT and prediction are finite and GT > 0. With
`--mask-sky`, sky pixels are excluded and the fit is trimmed to `gt ≤ P95(gt)`
so residual sky outliers do not dominate.

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
