# Distribution consumer fixtures

These small, synthetic `eval.json` documents exercise the v1 distribution
extension. `producerVersion: "contract-fixture-v1"` marks example data, not a
package release. They are intended for both Python and TypeScript importer tests.

| File | Expected acceptance and behavior |
|---|---|
| `counts-depth.eval.json` | Scalar + histogram depth metrics, native/metric definitions, shared edges, three per-file entries |
| `counts-sparsedepth.eval.json` | Same contract under `sparsedepth.eval` |
| `counts-points3d.eval.json` | `rmse3d` and Euclidean error under `points3d.eval`; both dense and sparse point-map evaluators use this wire namespace |
| `histogram-only.eval.json` | No scalar rows; import must still create a metric set, store distributions and make the split discoverable |
| `per-file-only.eval.json` | No aggregate metrics; retain hierarchical file IDs and make the split discoverable |
| `scalar-only.eval.json` | No distribution registry/leaves; existing scalar import continues to work |
| `future-sum-by-depth.eval.json` | Two simultaneous views of RMSE, two bin sets; fractional sums must remain fractional. This future view is a contract example, not a current CLI mode |

`scene/left/frame_0` and `scene/right/frame_0` deliberately reuse the basename
`frame_0`. They must stay distinct using the consumer's existing hierarchy-ID
encoding. The third file, `scene/right/empty`, has all-zero counts and is valid.
Per-file counts add to aggregate counts in each space. Native units are
`unspecified`; calibrated metric units are `m`.

Validate the registry with
[`distribution-v1.schema.json`](../../../euler_eval/schemas/distribution-v1.schema.json)
and leaves with that schema's `#/$defs/value`, then apply the cross-field rules
in [`docs/distributions.md`](../../../docs/distributions.md#consumer-contract-and-validation).
Consumer negative tests should mutate these documents to include an unsupported
version/type/statistic, missing registry/reference, unordered/interior-null
edges, length mismatch, non-finite/fractional/negative/unsafe counts, duplicate
file+metric identities, and foreign namespaces. Every rejected reload must leave
previously imported data intact. A native/metric definition mismatch must also
fail. Retrying an accepted document must not create duplicate rows.

Regenerate from the repository root with:

```bash
python scripts/generate_distribution_fixtures.py
```

Producer checks live in `tests/test_distribution_contract.py` and
`tests/test_distributions.py`; the latter runs actual CLI serialization for all
four evaluators, including ZIP output, benchmark slices and empty samples.
