# Dense normal-reference AB snapshot (2026-08-31)

This is an **unversioned** research snapshot copied from H100 host source
`/path/to/workspace/dense_normal_reference_ab_20260831`.
It is path-isolated under `experiments/` and is not part of the canonical
package, default CLI, or default CI quality claim.

- `quality_claims=false`
- Canonical dependency: repository commit `4edbc34` (`fix: seal actual evaluation contract`).
- This snapshot is retained for provenance and inspection only. It must not be
  presented as a validated model, evaluation, or production policy.

## Exact copied inventory

Only these seven H100 source files are included:

1. `src/dense_ref/__init__.py`
2. `src/dense_ref/cohort.py`
3. `src/dense_ref/core.py`
4. `src/dense_ref/evaluate.py`
5. `src/dense_ref/runner.py`
6. `scripts/run_full_2gpu.sh`
7. `scripts/run_smoke_2gpu.sh`

No cohorts, outputs, logs, receipts, caches, runtime arrays, or other H100
artifacts are included.

`SOURCE_SHA256SUMS` records the seven captured files in sorted
`<sha256><two spaces><snapshot-relative path>` form. It fixes the bytes of this
preserved snapshot; it does not claim the mutable H100 source will remain
unchanged after the 2026-09-01 comparison.
