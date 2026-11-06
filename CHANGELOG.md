# Changelog

Notable changes to euler-eval. Releases before 2.25.0 predate this file; see
the git history for those.

## Unreleased

### Changed

- **Moves results out of the dataset root (breaking for readers).** Results are
  now written to a `.eval/` folder beside the prediction dataset's
  `.ds_crawler/`: `.eval/<scope>/eval.json` when the prediction is loaded from
  its own `.ds_crawler/<scope>/` metadata, `.eval/eval.json` for root-level
  metadata, with `split_<split>/` added when a split is selected. Modalities and
  splits that share one dataset root or archive no longer overwrite each
  other's `eval.json`. Zips get `.eval/` inside the archive, within its
  top-level folder when the archive wraps everything in one named after it.
  Readers of the old `<dataset>/eval.json` must look in `.eval/` first; existing
  root-level files are left untouched. `output_file` is unchanged.

### Fixed

- **Moves `--mask-sky` metric values.** GT segmentation is now paired with each
  frame by hierarchy path and file ID, like depth, in every evaluation that
  uses it (dense and sparse depth, RGB, dense and sparse points-3D). It was a
  hierarchical modality, which gives each sample every mask in its folder, and
  `--mask-sky` used the first one: with per-frame segmentation in one folder,
  every frame was masked with that folder's first frame's sky, so the other
  frames' real sky stayed in the metrics and the scale/shift fit, unreported.
  Results with `--mask-sky` on such layouts change and are not comparable with
  earlier releases. A frame without its own segmentation file is now not
  evaluated, and the run prints how many pairs the segmentation (or a
  `sky_mask`) dropped; segmentation whose IDs match no frame stops the run.
  Each sample also decodes only its own mask instead of caching its folder's.
  `--mask-sky` warns about samples without segmentation of their own frame.
- Default dataset and per-file depth distributions now honor the benchmark GT
  range after sky-depth capping, matching the named benchmark `all` histogram
  and metrics. The selected population is recorded in metric-set metadata.
- Sparse pointcloud evaluation now follows the configured prediction modality:
  depth predictions emit only an atomic `sparsedepth.eval`, while sparse
  points-3D metrics require an explicit `points_3d` prediction. This prevents an
  implicitly derived `points3d.eval` from overwriting the depth `eval.json`.

### Added

- `--eval-name NAME` names an evaluation: its results are written as
  `NAME.json` (default `eval`) in the `.eval/` layout, so e.g.
  `--mask-sky --eval-name no_sky` sits beside an earlier `eval.json` instead of
  replacing it. A dataset's `output_file` still takes precedence.
- `--mask-sky-prediction` excludes the pixels of a prediction's
  `datasets[].sky_mask` from dense and sparse depth metrics, benchmark bins,
  histograms and the scale/shift fit, as `--mask-sky` does for GT sky; lidar
  points projecting into predicted sky are not scored. Predictions without a
  `sky_mask` are scored in full, with a warning. Because the excluded pixels
  depend on the model, `meta.pred_sky_masking` records the share of evaluable
  GT pixels each prediction excluded.
- Opt-in `--sky-iou` scores predicted sky against GT segmentation sky for
  dense and sparse depth: IoU, precision and recall under
  `{depth,sparsedepth}.eval.sky_segmentation.{explicit,implicit}.{pixel_pool,image_mean,image_median}`,
  with per-file values and `meta.sky_segmentation` provenance. `explicit`
  compares a `datasets[].sky_mask`; `implicit` compares `depth >= --sky-depth`
  on metric predictions before calibration and capping. The two are never
  merged, and implicit IoU depends on the threshold as much as on the model
  (see `docs/metrics.md`). The flag requires GT segmentation and loads it, and
  a `sky_mask`, without `--mask-sky` or `--sky-depth`; it changes no depth
  value, and without it outputs are unchanged.
- `gt.semantic_segmentation` is accepted as an alias of `gt.segmentation`;
  configuring both is an error. Under either key, a GT segmentation path
  without `#scope=` also finds an archive scope named after the other key, or
  whose head declares it, where it previously failed to load.

- `--mask-sky` warns when the GT segmentation masks no sky in any evaluated
  sample of a modality, as a wrong `skyclass` in the segmentation's dataset
  head or segmentation that matches no sample does; metrics then include sky.
- Optional per-sample and pooled RMSE error histograms for dense/sparse depth
  and point maps, enabled with `--distributions` or the config's `distributions`
  section. Configurable logarithmic or linear bins share JSON-safe boundaries
  and axis definitions; existing scalar metrics keep their paths and values.
  Typed histogram leaves reference a versioned registry with explicit units,
  reusable bin sets, JavaScript-safe counts, a packaged JSON Schema and consumer
  fixtures. The contract also supports future additive error sums by other axes.
- Predicted `sky_mask` inputs alongside dense depth and optional `--sky-depth`
  capping/inpainting for dense and sparse depth evaluation. Sky pixels are
  filled after calibration and remain evaluated unless excluded by GT validity
  or `--mask-sky`.
- Additive evaluation metric sets selected with repeatable `--domain` flags;
  the existing metrics remain the always-enabled `core` set.
- `--domain dehazing` adds per-image and dataset-mean NIQE and FADE scores for
  predicted RGB images. The official LIVE natural-scene model parameters are
  bundled for deterministic offline evaluation and recorded in the output
  metadata.

### Changed

- Require `euler-loading>=2.29`, the first release whose `sky_mask` loaders
  read the contract's `skyclass`.

## 2.25.0

### Fixed

- **Depth normal consistency is now computed from real 3D geometry.** The
  previous estimator took normals from image-space depth gradients
  (`n = (-dz/dx, -dz/dy, 1)`), which ignores the perspective divide and never
  used the `focal_length` argument it accepted: a ground plane receding to the
  horizon was reported as almost fronto-parallel. Depth maps are now
  unprojected with the camera intrinsics — planar and radial depth both handled
  exactly — and normals are the cross product of central-difference tangents,
  the same estimator the `points_3d` modality uses.

  `gt.calibration` / `gt.intrinsics` supply the camera; when absent, a pinhole
  camera with a centred principal point and `fx = fy = width` is assumed and
  reported as `intrinsics_source: "assumed"` in the per-image metadata.
  Intrinsics are rescaled when ground truth is resized onto the prediction
  plane.

  **`normal_consistency` values are not comparable with earlier releases.** The
  metric is now invariant to a global depth scale and markedly more sensitive to
  per-pixel depth noise.

- Normal angles use a numerically stable half-angle formula instead of
  `arccos(a · b)`, which reported ~0.02° for a perfect prediction. Applies to
  both the depth and `points_3d` normal metrics.

- `euler_eval.cli` and `euler_eval.metrics.fid_kid` used PEP 604 (`X | None`)
  annotations in runtime-evaluated positions without
  `from __future__ import annotations`, so importing them failed on Python 3.9
  despite the declared `requires-python = ">=3.9"`.

### Removed

- `compute_normal_consistency()` — unused and unexported; use
  `compute_normal_angles()` with `aggregate_normal_consistency()`.
- `depth_to_normals()` / `compute_normal_angles()` no longer take a
  `focal_length` argument. Pass `intrinsics=` instead, which accepts a `(3, 3)`
  camera matrix, an `{fx, fy, cx, cy}` mapping, or a scalar focal length in
  pixels. Arguments after `valid_mask` are keyword-only, so a stale positional
  call fails loudly rather than being silently misread.
- The unused `depth.normal_consistency.expected_focal_length` key in
  `metrics_config.json`.

### Changed

- `points_to_normals()` and `depth_to_points()` take an explicit `dtype`.
  The depth path defaults to float32, halving the time and memory of the
  normal metric at 1080p; stored point maps keep float64.
- Packaging metadata: project URLs, author, dependency floors, and a
  description covering every evaluated modality.

### Documentation

- The README is now an overview with an ecosystem diagram; the reference
  material moved to [`docs/`](docs/README.md).
- Added `CONTRIBUTING.md` and a CI workflow (tests, lint, distribution build).
