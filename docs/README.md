# euler-eval documentation

| Guide | Covers |
|---|---|
| [Configuration](configuration.md) | The config file, `gt` and `datasets` sections, paths and selectors, sparse-depth and points-3d ground truth, euler-train logging |
| [CLI reference](cli.md) | Every flag, worked examples, device selection, offline cache warmup |
| [Spaces & alignment](alignment.md) | Why `native` and `metric` exist, depth affine fitting, points-3d gauge alignment, benchmark depth bins |
| [Metrics](metrics.md) | The full metric inventory per modality, with the key each one is written under |
| [Results & output](output.md) | Where files land, the anatomy of `eval.json`, per-file metrics, the sanity-check report |
| [In-training validation](validation.md) | The programmatic API for scoring in-memory predictions during training |

## Concepts in one page

**A dataset is an indexed root, not a folder of files.** Every path in the
config points at a root — a directory or a `.zip` — carrying
[ds-crawler](https://github.com/d-rothen/ds-crawler) metadata under
`.ds_crawler/`. That metadata is what makes a path self-describing: it names the
loader to use, whether depth is radial or planar, what the value range is, and
which columns of a point cloud are `x,y,z`.

**Ground truth and predictions are paired by ID, not by order.**
[euler-loading](https://github.com/d-rothen/euler-loading) intersects file IDs
across modalities, so sample *i* holds the GT depth, the predicted depth and the
RGB frame that genuinely belong together. Files present on one side only are
left out of the pairing rather than silently misaligned. Calibration and pose
data are matched hierarchically instead — one intrinsics file can serve every
sample beneath it in the tree.

**Predictions are scored in a space, and there can be two.** A metric model is
compared to GT directly. A relative or free-frame model first has its unknown
degrees of freedom fitted away — scale and shift for depth, a similarity
transform for point maps. euler-eval reports the raw comparison as `native` and
the fitted one as `metric`, so a scale-recovery failure never masquerades as a
geometry failure. See [Spaces & alignment](alignment.md).

**Per-image scores are reduced three ways.** `image_mean` and `image_median`
average per-image values across the dataset; `pixel_pool` pools every valid
pixel first and reduces once. They answer different questions — a few huge
images dominate `pixel_pool` but not `image_mean` — so all three are reported.

**Metric names are structured, not free-form.** Keys are built with
[euler-metric-naming](https://github.com/d-rothen/euler-metric-naming), so each
`eval.json` carries a `metricSet` envelope declaring its namespace and axes, and
consumers such as euler-train and euler-view can decompose a key like
`depth.eval.metric.standard.image_mean.absrel` structurally rather than by
string-splitting.

**Nothing is computed twice, and nothing is required.** Each modality is
evaluated only when both GT and prediction sides are configured (and not
skipped), and results are written per modality — so a depth-only prediction is a
perfectly valid run.
