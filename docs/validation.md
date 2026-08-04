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
min_depth=..., max_depth=..., benchmark_depth_range=...)`** — one dense
prediction against a dense GT map. GT at another resolution is aligned to the
prediction plane. Returns the standard metric set plus pooled pixel statistics,
or `None` when too few valid pixels remain.

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
