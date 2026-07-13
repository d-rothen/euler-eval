# Evaluating multiple modalities from one inline dataset

Use one physical dataset root or ZIP archive when its existing ds-crawler
metadata is already scoped per logical modality. Do not extract or duplicate
the data merely to evaluate `rgb`, `depth`, `points_3d`, calibration, and
similar data independently.

## Required layout

Each logical modality needs a complete ds-crawler artifact set in its own
scope. Keep the raw files at their natural locations; the scope selects
metadata, loaders, and file IDs, not a subdirectory of the raw data.

```text
capture.zip
  frames/...
  lidar/...
  calibration/...
  .ds_crawler/
    scopes.json
    rgb/
      dataset-head.json
      ds-crawler.json
      index.json
    depth/
      dataset-head.json
      ds-crawler.json
      index.json
    points_3d/
      dataset-head.json
      ds-crawler.json
      index.json
    intrinsics/
      dataset-head.json
      ds-crawler.json
      index.json
    camera_extrinsics/
      dataset-head.json
      ds-crawler.json
      index.json
```

The regular-modality indexes must describe the same hierarchy and sample IDs
up to their modality-specific file location. `euler-loading` intersects those
IDs, so inconsistent IDs silently remove samples and an empty intersection
fails evaluation. Index calibration and pose data as hierarchical modalities:
their files should be ancestors of the samples they apply to, rather than
copied once per frame.

## Configuration pattern

Give every modality the same archive root. `euler-eval` selects the scope
whose name matches the configuration key by default, so the common case is
compact:

```json
{
  "gt": {
    "rgb": {"path": "/data/capture.zip:test"},
    "depth": {"path": "/data/capture.zip:test"},
    "intrinsics": {"path": "/data/capture.zip:test"},
    "camera_extrinsics": {"path": "/data/capture.zip:test"}
  },
  "datasets": [{
    "name": "model-a",
    "rgb": {"path": "/data/model-a.zip:test"},
    "depth": {"path": "/data/model-a.zip:test"}
  }]
}
```

Here `:test` loads the inline ds-crawler split
`.ds_crawler/split_test.json`; it does not change the physical archive path.
Use the same split on modalities that are intended to be paired. Evaluation
writes its result back to the selected prediction root as `eval.json` unless
`output_file` is explicitly configured.

## Explicit scopes

Use `#scope=<scope>` whenever an artifact scope is not named exactly like the
configuration key. Put it after the optional split selector:

```json
{
  "gt": {
    "sparse_depth": {"path": "/data/capture.zip:test#scope=lidar"},
    "intrinsics": {"path": "/data/capture.zip:test#scope=camera_intrinsics"},
    "camera_extrinsics": {"path": "/data/capture.zip:test#scope=lidar_to_camera"}
  },
  "datasets": [{
    "name": "point-model",
    "points_3d": {"path": "/data/model.zip:test#scope=point_maps"}
  }]
}
```

The selector is part of the config path only. The archive passed to
`euler-loading` remains `/data/capture.zip` or `/data/model.zip`, while the
matching `.ds_crawler/<scope>/index.json` supplies the correct loader and
metadata. Use one scope selector per path; do not also encode a competing
scope elsewhere.

## Modality checklist

- Use `rgb`, `depth`, `rays`, `points_3d`, and `sparse_depth` scopes for
  sampled data that must participate in file-ID intersection.
- Use `intrinsics`, `calibration`, `camera_extrinsics`,
  `lidar_extrinsics`, and segmentation scopes for per-scene or per-sequence
  data. These are loaded hierarchically and do not participate in the regular
  modality intersection.
- For `relative_depth` and `affine_depth` predictions, retain distinct scopes
  with those names; the evaluator passes the prediction type through so the
  appropriate ds-crawler metadata is selected.
- Keep a legacy root-level `.ds_crawler/index.json` only when it is genuinely
  the unambiguous default. With multiple scopes, prefer explicit names or
  `#scope=` selectors so changing the archive later cannot change resolution.
- Verify each scoped `index.json` declares the appropriate
  `euler_loading.loader` and `euler_loading.function`; these select the loader
  without evaluator-specific code.

This arrangement keeps one portable archive self-describing while letting
future evaluations add modalities by adding a scoped ds-crawler artifact set
and a matching config entry.
