# Changelog

Notable changes to euler-eval. Releases before 2.25.0 predate this file; see
the git history for those.

## Unreleased

### Added

- Additive evaluation metric sets selected with repeatable `--domain` flags;
  the existing metrics remain the always-enabled `core` set.
- `--domain dehazing` adds per-image and dataset-mean NIQE and FADE scores for
  predicted RGB images. The official LIVE natural-scene model parameters are
  bundled for deterministic offline evaluation and recorded in the output
  metadata.

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
