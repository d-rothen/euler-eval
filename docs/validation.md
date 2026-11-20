# In-training validation

The CLI pairs an on-disk ground truth with an on-disk *prediction* dataset. A
training loop has neither: it holds live, in-memory predictions and only the
ground truth on disk. `euler_eval.validation` exposes the same depth-metric
semantics for that case, so validation numbers during training are comparable to
the ones the CLI reports afterwards — against dense *or* sparse GT, without ever
writing predictions to disk.

```python
from euler_eval import (
    DepthValidationAggregator,
    build_validation_gt_dataset,
    evaluate_sparse_depth_sample,
    get_sample_intrinsics,
    get_sample_pointcloud_to_camera_extrinsics,
)

# GT-only dataset from euler-loading compatible paths
# (inline :split#scope= selectors supported).
dataset = build_validation_gt_dataset(
    sparse_depth_path="/data/muses.zip:val#scope=lidar",
    rgb_path="/data/muses.zip:val#scope=rgb",
    intrinsics_path="/data/muses.zip:val#scope=intrinsics",
    camera_extrinsics_path="/data/muses.zip:val#scope=extrinsics",
)

aggregator = DepthValidationAggregator()
for i in range(len(dataset)):
    sample = dataset[i]
    depth_pred = my_model(sample["rgb"])            # (H, W) metres, planar z
    K = get_sample_intrinsics(sample)
    lidar2cam, _ = get_sample_pointcloud_to_camera_extrinsics(sample)
    result = evaluate_sparse_depth_sample(
        depth_pred,
        sample["sparse_depth"],                     # (N, C>=3) lidar points
        K,
        lidar2cam,
        alignment="none",                           # "affine" for relative depth
        benchmark_depth_range=(0.0, 80.0),          # optional, additive
    )
    aggregator.update(result)

summary = aggregator.summary()
print(summary["standard"]["image_mean"]["absrel"])
```

## The pieces

**`evaluate_dense_depth_sample(pred, gt, valid_mask=None, alignment=...,
min_depth=..., max_depth=..., benchmark_depth_range=...,
spatial_alignment="auto")`** — one dense prediction against a dense GT map,
scored on the GT grid like the CLI: a GT at another resolution is cropped to
the region the prediction covers, or the prediction is resized onto it
([Resolution alignment](alignment.md#resolution-alignment)). `valid_mask` may
be on either plane. Returns the standard metric set plus pooled pixel
statistics, or `None` when too few valid pixels remain.

**`evaluate_sparse_depth_sample(pred, point_cloud, intrinsics,
camera_extrinsics, lidar_extrinsics=None, pred_is_radial=False, ...)`** —
projects the sparse GT cloud into the prediction plane (radial depth, nearest-z
occlusion handling), converts a planar prediction to radial with the same
intrinsics, and scores only the projected pixels, matching the CLI's
`gt.sparse_depth` pipeline. Pass the intrinsics of the *actual* prediction
plane, adjusted for any crop or resize of the model input.

**`alignment=`** — `"none"` scores metric predictions as-is; `"affine"` fits a
least-squares scale+shift first, for relative or affine depth models. These are
the two modes in `VALIDATION_ALIGNMENT_MODES`.

**`benchmark_depth_range=(MIN, MAX)`** — adds `result.benchmark` without
changing the regular result. Its `bins` mapping holds `all`, `near`, `mid` and
`far` values using the CLI's square-root-spaced boundaries; a bin with no valid
pixels is `None`. See [Benchmark depth bins](alignment.md#benchmark-depth-bins).

**`DepthValidationAggregator`** — accumulates per-sample results into the same
`image_mean` / `image_median` / `pixel_pool` reducers the CLI uses.

**`build_validation_gt_dataset(...)`** — resolves a GT-only euler-loading
`MultiModalDataset` from the same path syntax the config file accepts, so
validation reads the dataset exactly the way evaluation will.

Inputs accept torch tensors or numpy arrays; all math runs on CPU numpy.
Predictions and GT are expected in metres unless an affine `alignment` is
requested.

## Multi-process validation

For distributed validation, reduce sufficient statistics rather than metrics —
averaging per-rank averages is only correct when every rank saw the same number
of pixels, which it generally did not.

```python
import torch
from euler_eval import DepthValidationAggregator, summarize_reduced_state

keys = DepthValidationAggregator.state_keys()   # fixed order, all ranks agree
local = aggregator.reduced_state()              # {key: summable float}

vector = torch.tensor([local[k] for k in keys], dtype=torch.float64, device="cuda")
torch.distributed.all_reduce(vector)            # element-wise sum across ranks

summary = summarize_reduced_state(dict(zip(keys, vector.tolist())))
print(summary["standard"]["image_mean"]["absrel"])
```

`reduced_state()` returns sufficient statistics — sums and counts — so summing
them across ranks and dividing once gives exactly the single-process answer.
`state_keys()` fixes the ordering so the vector stays stable across ranks and
versions. Only the mean-based reducers can be rebuilt this way; a global median
is not recoverable from summed statistics, so `summarize_reduced_state` omits
`image_median`.

## Validation sets: validation curves that compare with the CLI

The functions above give numbers; a **validation set** also says what they
mean. It is an evaluation of the live model on a fixed population, in the
terms of euler-metric-naming's evaluation-protocol contract
([training.md](https://github.com/d-rothen/euler-metric-naming/blob/main/docs/training.md)):
declared once with the protocol the CLI would write for the same GT and
settings, and recorded per validation pass with the CLI's own metric names,
per-item values and per-item pixel counts. A consumer (euler-view, or
`euler_metric_naming.contract`) then judges a pass against any CLI
evaluation with the same engine it uses for two evaluations:

- the same predictions under the same settings read **`same protocol`**,
  with equal values;
- a pass on some of an evaluation's items reads **`compared on N shared
  files`**, and both aggregates are recomputed on the shared items;
- a difference in settings is named: `not comparable · depth cap differs`,
  `comparable · domain differs by 3.7 %`, and so on.

```python
import euler_train
from euler_eval.validation import DepthValidationSet

muses = DepthValidationSet(
    "muses",
    sparse_depth_path="/data/muses.zip:val#scope=lidar",
    intrinsics_path="/data/muses.zip:val#scope=intrinsics",
    camera_extrinsics_path="/data/muses.zip:val#scope=extrinsics",
    indices=range(64),                       # a fixed subset; its ids are the population
    sky_depth=240, benchmark_depth_range=(0.01, 200),
)
run = euler_train.init(...)
run.declare_validation_set(muses.key, muses.declaration())   # once

for epoch in range(epochs):
    ...
    validation = muses.new_pass()
    for index in muses.indices:
        sample = muses.dataset[index]
        validation.add(sample, model(sample), intrinsics=K_of_the_prediction_plane)
    run.log_validation(muses.key, validation.record(step=step, epoch=epoch))
```

| Setting | Meaning | CLI counterpart |
|---|---|---|
| `depth_path` / `sparse_depth_path` (+ `intrinsics_path`, `camera_extrinsics_path`, `lidar_extrinsics_path`), `split` | The GT; the references are read from these paths as the CLI reads its `gt` config | `gt.depth` / `gt.sparse_depth`, … |
| `indices` | The items the set evaluates (default: all); their item ids are the population | the paired files |
| `alignment` | `none`, `affine`, `auto_affine` | `--depth-alignment` |
| `input_space_hint` | `relative` / `affine` forces `auto_affine` to fit | the prediction's declared space |
| `sky_depth` | depth cap in metres | `--sky-depth` |
| `benchmark_depth_range` | `(min, max)` of the benchmark bins | `--benchmark-depth-range` |
| `spatial_alignment` | `auto`, `crop`, `resize`, or a recorded source grid | `--spatial-alignment` |
| `valid_depth_range` | GT validity bounds; **in-loop only**, recorded as `depth.gt.valid_range_m` | none: such a pass is not comparable with a CLI evaluation |
| `per_item` | write per-item values and counts (default: sets of at most 1000 items) | per-file metrics |

`add(sample, prediction, valid_mask=None, intrinsics=None, …)` scores one
dataset sample (its GT as loaded, its `id` / `full_id`) against an in-memory
prediction exactly as the CLI scores the same pair: the prediction is put on
the GT grid (dense), alignment, cap, benchmark bins and the standard depth
metrics follow the CLI's code. A `valid_mask` removes more pixels and is
recorded (`depth.extra_mask`), so it never passes for an unmasked evaluation.
`record(step=..., epoch=...)` returns the pass record: the aggregates
(`depth.eval.metric.standard.image_mean.absrel`, …), the parameters and
observations resolved in this pass (spatial method, whether a fit ran, the
scored region, how much the cap clipped), the ids when the pass scored other
items than the set lists, and the per-item values.

`DepthValidationSet.from_protocol(key, "pred/.eval/depth/eval.json",
depth_path=...)` takes the alignment, cap, benchmark range and spatial policy
of an existing evaluation, so a validation set matches the evaluation it is
meant to predict by construction. Settings an in-loop pass cannot apply (GT
or predicted sky masking) raise `ValueError`.

For multi-process validation, every process scores its share into its own
pass; gather `validation.state()` (JSON-ready) and `merge` them into one
before `record()`.

What is checked: `tests/test_validation_set.py` runs the CLI and a validation
set over the same synthetic GT and predictions and requires `same protocol`
with equal values for every cell a pass writes (no alignment with benchmark
bins, affine alignment, a depth cap), and `compared on N shared files` with
equal recomputed values for a pass on a subset. Sparse GT is built from the
same helpers and tested on its own; an end-to-end comparison with the CLI on a
sparse GT fixture is still to be added.
