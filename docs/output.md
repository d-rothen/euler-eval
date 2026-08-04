# Results & output

## Where results land

One `eval.json` is written **per modality, per prediction dataset**, next to the
prediction it scores — `/data/model_a/depth/eval.json` for the depth results of
the dataset whose depth path is `/data/model_a/depth`, and so on. If that path
passes through a `.zip`, the JSON is written *into* the archive rather than
beside it, so zip-backed datasets stay self-contained.

`output_file` in a dataset entry overrides the destination. It applies to the
dataset as a whole, so when several modalities are evaluated for the same
dataset the per-modality default locations are usually what you want.

## Anatomy of `eval.json`

```json
{
  "metricSet": {
    "metricNamespace": "depth.eval",
    "producerKey": "euler-eval",
    "producerVersion": "2.24.0",
    "sourceKind": "computed",
    "metadata": {
      "input_space_detected": "metric",
      "calibration_mode": "auto_affine",
      "calibration_applied": false,
      "emitted_spaces": ["metric"],
      "canonical_space": "metric"
    },
    "axes": { "space": { "…": "…" }, "category": { "…": "…" } },
    "metricDescriptions": { "absrel": { "…": "…" } }
  },
  "dataset_info": { "num_pairs": 500, "gt_name": "GT", "pred_name": "model_a" },
  "meta": {
    "version": "2.24.0",
    "modality": "depth",
    "device": "cuda",
    "gt":   { "path": "/data/gt/depth",      "dimensions": { "height": 1080, "width": 1920 } },
    "pred": { "path": "/data/model_a/depth", "entry": "depth" },
    "spatial_alignment": { "method": "resize", "evaluated_dimensions": { "height": 512, "width": 960 } },
    "modality_params": { "radial_depth": false },
    "eval_params": { "sky_masking": false, "depth_alignment_mode": "auto_affine" }
  },
  "depth": { "eval": { "native": { "…": "…" }, "metric": { "…": "…" } } },
  "per_file_metrics": { "children": { "…": "…" } }
}
```

| Block | What it is for |
|---|---|
| `metricSet` | The euler-metric-naming envelope: producer, namespace, axis declarations and per-metric descriptions (unit, direction, bounds), plus how the spaces were resolved |
| `dataset_info` | How many pairs were actually matched, and the display names of both sides |
| `meta` | Full provenance: package version, device, both source paths and their resolved dimensions, modality metadata, and the effective evaluation flags |
| *metric tree* | The metrics themselves, under the modality root (`depth`, `rgb`, `rays`, `sparsedepth`, `points3d`) |
| `per_file_metrics` | The same metrics per image, in dataset-hierarchy order |

`meta` is what makes a result reproducible after the fact: it records the
alignment mode that ran, whether sky masking was on, the benchmark range, and
the exact paths and versions involved. `spatial_alignment.method` reports how a
GT/prediction resolution mismatch was resolved — `none` when they already match,
`vae_crop` for the multiple-of-8 crop that latent models produce, `resize`
otherwise, and `pointcloud_projection` when GT was a sparse cloud.

## Spaces and canonical aliases

Depth-like and points-3d results are nested by space:

```text
depth.eval.native.…      the prediction as emitted
depth.eval.metric.…      after scale/shift (or gauge) fitting
```

The `metric` space appears only when a comparison in that space was possible —
for points-3d, only when a gauge alignment actually ran. The in-memory result
dicts (and the euler-train payload) additionally expose a canonical `depth` /
`points_3d` key aliasing the `metric` branch when it exists, else `native`, so a
consumer can read one key without knowing how the run was configured. See
[Spaces & alignment](alignment.md).

## Wire-safe metric roots

Serialized sparse-depth and points-3d metric paths are rooted at
`sparsedepth.eval` and `points3d.eval` — no underscore — because downstream
namespace validation applies a stricter rule to the first segment of a flattened
metric path. The internal Python result dicts keep the `sparse_depth_*` /
`points_3d_*` keys.

## Per-file metrics

`per_file_metrics` mirrors the dataset hierarchy, so per-image values stay
attributable to the scene or sequence they came from:

```json
{
  "children": {
    "scene_01": {
      "files": [
        { "id": "0001", "metrics": { "depth": { "eval": { "metric": { "…": "…" } } } } }
      ]
    }
  }
}
```

Non-finite values (`NaN`, `Inf`) and `None` are stripped from every metric tree
before writing, so the JSON stays schema-valid. A file whose metrics all fail to
compute keeps its entry with an empty `metrics` object, and the run prints a
warning naming the affected files rather than dropping them silently.

## Sanity-check report

Sanity checking is on by default. It validates inputs and results against the
thresholds in [`metrics_config.json`](../metrics_config.json) — implausible
depth ranges, degenerate value ranges that make SSIM or LPIPS meaningless,
AbsRel above 1.0, SILog indicating a scale mismatch, suspicious edge densities —
and reports them grouped by metric and warning type.

Warnings are printed per prediction dataset as the run proceeds, and the full
report is written to `sanity_check_report.json` in the **current working
directory** at the end of the run. Disable the checks with `--no-sanity-check`,
or point them at a different threshold file with `--metrics-config`.

Sanity warnings never fail a run — they exist to catch a misconfigured
comparison (wrong units, wrong space, wrong dataset) before its numbers are
taken at face value.

## RGB FID backends

| Backend | Behaviour |
|---|---|
| `builtin` *(default)* | In-process Inception v3 features, no extra dependency |
| `clean-fid` | Delegates to [clean-fid](https://github.com/GaParmar/clean-fid); requires the `[fid]` extra and gives scores closer to standard published FID |

With `--rgb-fid-backend clean-fid`, euler-eval honors `CLEANFID_CACHE_DIR` for
staging or downloading the Inception checkpoint — see
[offline cache warmup](cli.md#offline-cache-warmup).

## euler-train logging

When the config carries a [`euler_train`](configuration.md#euler_train-section-optional)
section, each prediction dataset is registered as an evaluation in the run —
first as `running`, then completed with its aggregate metrics and package
provenance attached. Per-file metrics are not sent; they stay in `eval.json`.
