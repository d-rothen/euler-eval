# Proposal: a `points_3d` modality and its metrics

This is a design proposal for a new evaluation modality,
`points_3d`: the **3D-space representation of the `depth` modality**, produced by
models that predict a **per-pixel 3D point map together with their own camera
model**. It is written against the current code in `euler_eval/evaluate.py`,
`euler_eval/data.py`, `euler_eval/cli.py`, and `euler_eval/metrics/*`.

> **Implementation status (landed).** Categories A–D (`point_error`,
> `error_decomposition`, `geometric`, `cloud_distance`), the
> `none`/`scale`/`similarity`/`auto` gauge alignment with `native`/`metric`
> spaces, **three GT sources** — the explicit `gt.points_3d ↔
> datasets[].points_3d` path, on-the-fly GT synthesis from `gt.depth` +
> intrinsics (`gt.intrinsics`/`gt.calibration`), **and the sparse-LiDAR
> `gt.sparse_depth` path** (§4-D): a dense depth prediction is unprojected with
> the GT intrinsics and scored in 3D against the sparse cloud via directed
> (completeness/recall) `cloud_distance` plus per-correspondence `point_error` /
> `error_decomposition` (`evaluate_points_3d_sparse_samples`, depth-affine
> `native`/`metric` gauge) — full CLI + output (`points3d.eval`) + sanity + docs
> + tests are implemented (`tests/test_points_3d.py`,
> `tests/test_points_3d_sparse.py`). **Deferred follow-ups:** category E
> (`camera_model` recovered-intrinsics fidelity) and benchmark depth-range bins
> for points_3d — noted inline below.

---

## 1. How the package loads modalities and compares results (recap)

I traced the existing pipeline so that the new modality slots into the same
structure rather than bolting on a parallel one. The package today supports four
modalities — `depth`, `sparse_depth`, `rgb`, `rays` — and every one follows the
same five stages.

1. **Config → dataset builder** (`euler_eval/data.py`).
   `build_depth_eval_dataset` / `build_sparse_depth_eval_dataset` /
   `build_rgb_eval_dataset` / `build_rays_eval_dataset` each construct a
   `MultiModalDataset` (from `euler_loading`) with `gt` and `pred` *regular*
   modalities plus *hierarchical* modalities (`calibration`, `segmentation`,
   `intrinsics`, `camera_extrinsics`, `lidar_extrinsics`). Loaders are resolved
   automatically from each dataset's ds-crawler index. Inline `:split` and
   `#scope=` selectors are parsed in `config_paths.py`.

2. **Tensor → numpy conversion** (`data.py`). `to_numpy_depth` → `(H,W)`,
   `to_numpy_rgb`/`to_numpy_directions` → `(H,W,3)`, `to_numpy_point_cloud` →
   `(N,C)`, `to_numpy_intrinsics` → `(3,3)`, `to_numpy_extrinsics` → `(4,4)`.

3. **Spatial + value normalization.** GT is aligned to the prediction grid by
   `align_to_prediction` (VAE multiple-of-8 top-left crop, else resize). Depth is
   put in metres and converted planar→radial via `process_depth` /
   `convert_planar_to_radial` when intrinsics are available. Non-metric depth is
   calibrated to GT by `compute_scale_and_shift` (least-squares affine
   `s·pred + t`; see `scale_and_shift.md`).

4. **Per-pair metric loop with streaming aggregation** (`evaluate.py`).
   `evaluate_*_samples` iterate `_prefetched_iter(dataset, num_workers)`, compute
   per-image metrics, and feed streaming stores (`_StreamingValueStore`,
   `init_standard_depth_store`, histogram accumulators) so percentiles/means are
   produced without holding every pixel in memory. Depth and sparse-depth emit
   **two semantic spaces** — `native` (raw prediction) and `metric` (after
   calibration) — with a canonical alias; `rays` aggregates angular error and
   `rho_a` (AUC of the angular-accuracy curve up to a FoV-dependent threshold).

5. **Envelope, namespace, save** (`cli.py`). Each modality builds an
   `_EvalNamespace` (from `euler_metric_naming`) with **axis declarations**
   (`space`, `category`, `reduction`, `bin`) and per-metric `MetricDescription`s
   (`is_higher_better`, `unit`, `min/max`, `display_name`). Results are written as
   `{modality}.eval.…` plus a hierarchical `per_file_metrics` tree
   (`set_value(... {"id", "metrics"})`), saved one JSON per modality by
   `save_results`. `SanityChecker` validates against `metrics_config.json`
   thresholds keyed by modality.

**Takeaway for the new modality:** to add `points_3d` cleanly I reuse (a) the
`gt`/`pred` + hierarchical dataset builder pattern, (b) the `native`/`metric`
two-space output with a canonical alias, (c) the streaming aggregation loop, (d)
the axis/namespace/description envelope, and (e) the sanity-config pattern. The
*only genuinely new* machinery is 3D point handling and **3D gauge alignment**
(the generalization of scale-and-shift to similarity transforms).

---

## 2. What `points_3d` is, and why it needs its own metrics

A `points_3d` prediction is a **point map**: a per-pixel 3D point
`P(u,v) = (X, Y, Z)` in the camera frame, shape `(H, W, 3)` — the same layout the
loaders already produce for `rays`/`rgb`. It is what self-calibrating /
"predict-your-own-camera" models emit directly (DUSt3R / MASt3R / VGGT / MoGe /
UniDepth / Metric3D-v2 style outputs). Such a model jointly decides **both** the
depth along each ray **and** the ray geometry (focal length, principal point,
field of view, possibly distortion). The point map is the product of those two
decisions.

Ground truth is the per-pixel point map obtained by unprojecting GT depth with GT
intrinsics, `P_gt(u,v) = D_gt(u,v) · K_gt⁻¹ · [u, v, 1]ᵀ` — i.e. it can be
synthesized from the `gt.depth` + `gt.intrinsics` the repo already loads — or
supplied directly as a `points_3d` GT, or (sparsely) as a LiDAR cloud via the
existing `gt.sparse_depth` path.

**Why not just reuse `depth` + `rays`?**

- `depth` eval **needs GT intrinsics to unproject** and is blind to the model's
  *own* camera. A model can get depth-along-ray right but place points laterally
  wrong because its focal/principal point is off; depth-only metrics miss that
  entirely.
- `rays` eval measures direction but is blind to scale and translation.
- The two errors **couple**: a wrong focal length compensated by a wrong depth can
  look acceptable in *each* projection yet be wrong in metric 3D. Only a joint 3D
  comparison exposes it.
- Evaluating the model's *actual output* (the point map) is more faithful than
  decomposing it into two channels the model may not separately produce.

So `points_3d` is the **joint depth-⊗-camera evaluation in metric 3D space**, with
a built-in **error decomposition** that attributes each error to the *depth*
component (along the ray) or the *camera-model* component (perpendicular to the
ray). That attribution is the unique diagnostic value of the modality.

---

## 3. The central design decision: 3D gauge alignment

`depth` resolves an unknown 1D affine gauge with `compute_scale_and_shift`
(`s·pred + t`). The 3D analog is a **similarity gauge** (Umeyama / Procrustes),
because point maps from different model families live in different frames:

| Model class | Unknown gauge | Alignment to apply |
|---|---|---|
| **Metric** self-calibrating (absolute scale + metric camera) | none | `none` |
| **Up-to-scale** | global scale `s` | `scale` |
| **Relative / free-frame** | similarity `(s, R, t)` (7-DoF) | `similarity` |

Closed-form fit over the valid corresponding pixels (correspondence is **known** —
both maps are on the same pixel grid after spatial alignment, so no ICP is
needed):

- `scale`: `s* = Σ(P_pred·P_gt) / Σ‖P_pred‖²` (or `median(‖P_gt‖/‖P_pred‖)`).
- `similarity`: Umeyama (1991) — center both clouds, SVD of the cross-covariance
  for `R`, `s` from singular values / variance, `t` from the centroids.

This mirrors `--depth-alignment {none, auto_affine, affine}` exactly:

```
--points-3d-alignment {none, scale, similarity, auto}
```

`auto` chooses `none` for declared-metric predictions and `similarity` for
declared-relative predictions (the analog of `auto_affine`'s normalized-range
sniff and the `relative_depth` / `affine_depth` scope hints).

As with depth, emit **two spaces** and a canonical alias:

- `points_3d_native` — metrics on the raw predicted points (the honest score for a
  model that *claims* metric scale and camera).
- `points_3d_metric` — metrics after the chosen gauge alignment (the fair score
  for relative models; isolates *shape* error from *gauge* error).
- `points_3d` — canonical alias (`metric` when alignment ran, else `native`),
  matching the depth convention.

> **Subtlety to record in the implementation:** the *angular / ray* metrics
> (Section 4-B) describe the **predicted camera** and are therefore most
> meaningful on the **native** directions (translation- and scale-invariant, but
> they must be read *before* a `similarity` alignment rotates them away). Recommend
> computing ray-angle / `rho_a` from native-space directions, while the Euclidean
> and radial/lateral metrics are reported per branch.

---

## 4. Proposed metric set

Organized to fit the existing axis system —
`space ∈ {native, metric}` × `category` × `reduction ∈ {image_mean, image_median,
pixel_pool}`. Notation: per valid pixel `i`, `d_i = P̂_i − P_gt,i` (where `P̂` is
the gauge-aligned prediction), `e_i = ‖d_i‖₂` in metres, and GT ray
`r_i = P_gt,i / ‖P_gt,i‖`. Valid mask = both points finite and `‖P_gt,i‖ > 0`
(intersected with the sky mask when `--mask-sky` is set, reusing
`_get_sky_mask`).

### A. `point_error` — Euclidean 3D agreement (the workhorse) — **P0**

The direct 3D generalization of depth's AbsRel/RMSE/δ.

| Metric | Definition | Unit | Better |
|---|---|---|---|
| `epe3d` / `mae3d` | `mean(e_i)` (3D end-point error) | m | lower |
| `rmse3d` | `sqrt(mean(e_i²))` | m | lower |
| `median3d`, `p90`, `p95` | pooled percentiles of `e_i` | m | lower |
| `rel_pt_dist` | `e_i / ‖P_gt,i‖`; report `median`, `p90` (scale-invariant, AbsRel analog) | – | lower |
| `acc@τ` (3D δ) | `mean(e_i < τ)·100`, `τ ∈ {0.05, 0.1, 0.25, 0.5, 1.0} m` | % | higher |
| `acc_rel@τ` | `mean(e_i/‖P_gt,i‖ < τ)·100`, `τ ∈ {0.05, 0.1, 0.25}` | % | higher |

### B. `error_decomposition` — depth vs camera attribution (the differentiator) — **P0**

Project the error vector onto / off the GT ray:

- **radial** (≈ depth error): `a_i = d_i · r_i` → `radial_mae = mean|a_i|`,
  `radial_rmse = sqrt(mean a_i²)` [m].
- **lateral** (≈ camera-model error): `l_i = ‖d_i − a_i r_i‖` →
  `lateral_mae = mean(l_i)`, `lateral_rmse = sqrt(mean l_i²)` [m].
- **`lateral_fraction`** `= mean(l_i) / (mean|a_i| + mean(l_i)) ∈ [0,1]`: a single
  number — near 0 ⇒ error is depth-dominated, near 1 ⇒ camera-model-dominated.
- **ray angular error** `θ_i = acos(clip(P̂_i·P_gt,i / (‖P̂_i‖‖P_gt,i‖), −1, 1))`
  [deg]: `mean`, `median`, `percent_below {5,10,15,20,30}°`.
- **`rho_a`** — AUC of the angular-accuracy curve up to the FoV-dependent
  threshold.

> **Maximal reuse:** ray angular error and `rho_a` are exactly the `rays` metrics.
> `euler_eval/metrics/rho_a.py` (`compute_angular_errors`, `compute_rho_a`,
> `classify_fov_domain`, `get_threshold_for_domain`, `aggregate_rho_a`) can be
> called as-is on the point-map directions, with the FoV domain auto-detected from
> GT intrinsics just like `evaluate_rays_samples`.

This category is what makes `points_3d` worth having: it tells you *whether a model
should fix its depth head or its camera head*.

### C. `geometric` — surface/structure quality in true 3D — **P1**

- **3D normal consistency.** Compute normals **directly from the point map** via
  the cross product of tangents,
  `n(u,v) = normalize( (P(u+1,v) − P(u−1,v)) × (P(u,v+1) − P(u,v−1)) )`, and
  angular error vs GT normals: `mean`/`median`, `percent_below
  {11.25, 22.5, 30}°`. This is strictly more correct than the current
  `normal_consistency.py`, which assumes `focal_length = 1.0` and works in image
  space (a known limitation noted in `metrics_exaplanation.md`). The aggregation
  (`aggregate_normal_consistency`) is reused unchanged.
- **`point_edge_f1`.** Discontinuity F1, analogous to `depth_edge_f1`, but on 3D
  neighbour distance: mark an edge where `‖P(u,v) − P(neighbour)‖ > k·‖P(u,v)‖`;
  precision/recall/F1 with 1-pixel dilation tolerance.

### D. `cloud_distance` — set-level surface agreement (Chamfer / F-score) — **P1**

A correspondence-free lens that **complements, not duplicates, A**. Because both
maps share the pixel grid, A already has *exact* per-pixel correspondence — so
Chamfer here is not a stricter Euclidean error, it answers a *different* question:
*does the predicted surface match the GT surface as a set of points, regardless of
which pixel each point landed on?* That makes it robust to precisely the failure A
punishes hardest — a wrong camera that slides points along an otherwise-correct
surface — so reporting both **separates "the surface is wrong" from "the surface is
right but mis-parameterized."** It is also the field-standard expectation for any
3D point output (DUSt3R/MASt3R report Chamfer; DTU / Tanks-and-Temples report
F-score), and the *only* sensible geometry metric on the sparse-LiDAR-GT path.

Build clouds from the valid (non-sky, finite) points of each map; two
`scipy.spatial.cKDTree`s (one per cloud).

| Metric | Definition | Unit | Better |
|---|---|---|---|
| `chamfer.accuracy` | `mean_{p∈pred} min_{q∈gt} ‖p−q‖` (floaters / precision side) | m | lower |
| `chamfer.completeness` | `mean_{q∈gt} min_{p∈pred} ‖q−p‖` (holes / recall side) | m | lower |
| `chamfer.distance` | `½(accuracy + completeness)` (CD-L1) | m | lower |
| `chamfer.median` | median of pooled NN distances (outlier-robust) | m | lower |
| `fscore@τ` | `precision@τ` / `recall@τ` / `f1@τ`: fraction within `τ` of the other cloud, `τ ∈ {0.05, 0.1, 0.25, 0.5} m` | – | higher |

Design notes (the parts that make Chamfer behave):

- **F-score is the headline; raw Chamfer is the diagnostic.** Mean Chamfer is
  dominated by a handful of floaters, which is exactly why Tanks-and-Temples / DTU
  / DUSt3R lead with the bounded, thresholded `fscore@τ`. Always report `accuracy`
  and `completeness` **separately** (they localize floaters vs. holes) and add
  `median` / `p90`, never a lone mean.
- **Compute per gauge branch.** Chamfer is metric (metres), so report it on
  `native` and on the gauge-aligned `metric` points (Section 3). Chamfer *after*
  the known-correspondence Umeyama fit is residual shape error under the best
  similarity — a clean "shape-only" score, decoupled from the gauge.
- **Cost & streaming.** `N = H·W` reaches ~10⁶; voxel-downsample (more uniform than
  random) to ~50–100k points/cloud and `log()` the rate (the repo's no-silent-caps
  habit). Per-image NN distances still feed `_StreamingValueStore` for pooled
  percentiles; only one image's two clouds are resident at a time.
- **Sparse LiDAR GT** (`gt.sparse_depth`) — *landed* via
  `evaluate_points_3d_sparse_samples` + `compute_sparse_cloud_distance_metrics`.
  Here there is no dense correspondence, so cloud_distance is the **primary**
  geometry metric, not a complement. The implementation leads with
  **completeness / recall** (is every LiDAR return covered?) and **omits** the
  misleading `accuracy` / `precision` / `f1` side, because a correct *dense*
  prediction has many legitimate points far from any *sparse* GT point. The
  dense depth prediction is unprojected with the GT intrinsics into a point map;
  the sparse GT cloud is projected into that camera frame, yielding the visible
  GT cloud (for completeness) and per-pixel correspondences (for `point_error` /
  `error_decomposition`). The `native`/`metric` gauge is the depth affine
  scale-and-shift (`--depth-alignment`), since the prediction is a depth map.
  Dense-neighbourhood `geometric` metrics (normals, edge F1) are omitted.

### E. `camera_model` — recovered-camera fidelity — **P2 (optional)**

The most direct test of "did the model get *its own* camera right." If the model
emits intrinsics alongside the point map, compare directly; otherwise least-squares
fit a pinhole `(fx, fy, cx, cy)` to the predicted point map
(`u ≈ fx·X/Z + cx`, `v ≈ fy·Y/Z + cy`) and compare to GT `K`:

- `focal_rel_error` `= |f_pred − f_gt| / f_gt` (fx, fy),
- `principal_point_error_px` `= ‖(cx,cy)_pred − (cx,cy)_gt‖` (also normalized by
  image size),
- `fov_error_deg` `= |FoV_pred − FoV_gt|` (reuse the diagonal-FoV computation in
  `classify_fov_domain`).

### Reducers and benchmark bins

Per-image reducers `image_mean` / `image_median` / `pixel_pool` apply to A–C
exactly as for depth's `standard` category. The existing
`--benchmark-depth-range` near/mid/far bins (`get_benchmark_depth_bins`, square-root
splits) extend naturally: bin by GT range `‖P_gt,i‖` and report A/B per bin —
e.g. "lateral error grows in the far bin" cleanly localizes camera error to large
depths.

---

## 5. Output structure

Mirror the depth/sparse-depth convention. The Python result dict uses
`points_3d_native` / `points_3d_metric` / `points_3d`; the serialized `eval.json`
roots metric paths at a wire-safe namespace `points3d.eval.…` (the same
underscore-stripping trick already used for `sparsedepth`, see
`_SPARSE_DEPTH_METRIC_ROOT` in `cli.py`, so flattened names satisfy
`metricSet.metricNamespace`).

```jsonc
{
  "points_3d_native": {
    "point_error":         { "image_mean": { "mae3d": 0.41, "rmse3d": 0.93,
                                             "acc@0.1": 38.2, "acc@0.25": 71.0 },
                             "image_median": { "...": "..." },
                             "pixel_pool":   { "median3d": 0.22, "p90": 0.85 } },
    "error_decomposition": { "radial_rmse": 0.55, "lateral_rmse": 0.74,
                             "lateral_fraction": 0.57,
                             "angular_error": { "mean": 4.8, "median": 2.9,
                                               "percent_below_5": 63.0 },
                             "rho_a": 0.81 },
    "geometric":           { "normal_consistency": { "mean_angle": 14.2,
                                                     "percent_below_22_5": 79.1 },
                             "point_edge_f1": { "precision": 0.7, "recall": 0.66,
                                               "f1": 0.68 } }
  },
  "points_3d_metric": { "...": "same shape, after gauge alignment" },
  "points_3d":        { "...": "canonical alias of points_3d_metric|native" },
  "per_file_metrics": {
    "children": { "scene_01": { "...": { "files": [
      { "id": "frame_0001",
        "metrics": { "points3d": { "eval": {
          "native": { "...": "..." }, "metric": { "...": "..." } } } } }
    ] } } }
  }
}
```

`metricSet.metadata` records `alignment_mode`, `emitted_spaces`,
`canonical_space`, the recovered/declared `fov_domain` + `threshold_deg`, and
(when fit) the recovered camera — paralleling the depth `space_info` block.

---

## 6. Integration plan (where each piece lands)

| Layer | Change |
|---|---|
| **Config** | GT via `gt.points_3d.path`, **or** synthesize from `gt.depth` + `gt.intrinsics`, **or** sparse `gt.sparse_depth` (cloud_distance only). Prediction via `datasets[].points_3d.path`. Extend `validate_gt_config` / `validate_dataset_entry`. |
| **`data.py`** | `to_numpy_points_3d(data) → (H,W,3)` (reuse `to_numpy_directions` layout logic, no re-normalization); `build_points_3d_eval_dataset(...)`; helpers `unproject_depth_to_points(depth, K)`, `umeyama_alignment(pred, gt, mask, with_scale, with_rotation)`, `decompose_point_errors(P_pred, P_gt, mask)`; `get_points_3d_metadata(...)`. `align_to_prediction` already handles `(H,W,3)`. |
| **`metrics/`** | New `points3d_distance.py` (A), `points3d_decomposition.py` (B), `points3d_geometry.py` (C, partly reusing normal aggregation), `points3d_cloud.py` (D), `points3d_camera.py` (E). **Reuse** `rho_a.py` and the `normal_consistency` aggregators. Export via `metrics/__init__.py`. |
| **`evaluate.py`** | `evaluate_points_3d_samples(...)` following the `evaluate_rays_samples` / `evaluate_sparse_depth_samples` streaming pattern: per-pair convert → spatial align → fit gauge (first-sample sniff for `auto`) → compute native + aligned metrics → stream into stores → build `native`/`metric`/canonical summaries + `per_file_metrics`. |
| **`cli.py`** | `_points_3d_eval_axes()` (`space`, `category ∈ {point_error, error_decomposition, geometric, cloud_distance, camera_model}`, `reduction`, `bin`); `_POINTS_3D_EVAL_DESCRIPTIONS`; `_EvalNamespace(modalities=("points_3d",))`; `_POINTS_3D_METRIC_ROOT="points3d"`; `has_points_3d` dispatch in `main`; `--skip-points-3d` and `--points-3d-alignment {none,scale,similarity,auto}`; save wiring + `save_results(..., modality="points_3d")`. |
| **`metrics_config.json` / `sanity_checker.py`** | A `points_3d` block: e.g. `warn_if_mae3d_exceeds`, `warn_if_lateral_fraction_above` (camera-dominated error flag), `warn_if_rho_a_below`, `warn_if_valid_points_below`, reusing the `rays`/`depth` threshold style. |
| **Docs** | Extend `README.md` (Features, Configuration, Metrics, Output) and `metrics_exaplanation.md`; add a "Similarity (Umeyama) alignment" note alongside `scale_and_shift.md`. |

---

## 7. Recommended phasing

- **P0 — core, ship first.** `point_error` (A) + `error_decomposition` (B, reusing
  `rho_a`); `native`/`metric`/canonical spaces; `none`/`scale`/`similarity`/`auto`
  alignment; GT-from-`depth`+`intrinsics`; full CLI + output + sanity wiring. This
  alone delivers the modality's headline value: joint metric-3D error **and** the
  depth-vs-camera attribution.
- **P1 — geometry & set agreement.** `geometric` (C: true-3D normals, 3D edge F1)
  and `cloud_distance` (D: Chamfer + F-score). D is **primary, not optional**, on
  the sparse-LiDAR-GT path, and the standard companion to A elsewhere.
- **P2 — optional/advanced.** `camera_model` (E, recovered-intrinsics fidelity) and
  any distribution-level (FID-style) scores.

### Open questions for the maintainer

1. **GT source priority** — prefer synthesizing the GT point map from
   `gt.depth` + `gt.intrinsics` (dense, exact pixel correspondence), or require an
   explicit `gt.points_3d`? (Recommendation: support both; synthesize when only
   depth+intrinsics are present.)
2. **Default alignment for `auto`** — key off a declared input space (a
   `relative_points_3d` scope hint, mirroring `relative_depth`/`affine_depth`) or
   off a runtime metric-scale sniff? (Recommendation: prefer the declared hint,
   fall back to the sniff.)
3. **`rho_a` thresholds** — reuse the `rays` FoV thresholds (15/20/30°) verbatim,
   or widen them since point-map directions also absorb depth-induced parallax?
4. **Distortion** — do target models emit non-pinhole cameras (fisheye/pano)? If so
   the `camera_model` fit and any planar↔radial assumptions need a distortion term.
