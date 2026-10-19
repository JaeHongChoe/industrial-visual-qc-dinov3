# Experimental post-processing methods

This directory documents development diagnostics for investigating score adjustments from frozen actual-evaluation outputs and patch-level primary, auxiliary, mask, recurrence, normal-prior, and structural evidence.

The modules and configurations added with this experiment are research-only. They are not a default CLI path, production PASS/FAIL policy, or an operational threshold. Any operational decision must continue to use its separately approved production model, policy, and thresholds.

## Known limitations / 실행 제한

- `src/patch_qc/normal_prior_postprocess.py::_sift_similarity_registration()` unconditionally destructures each `cv2.BFMatcher().knnMatch(..., k=2)` row as `(first, second)`. A sparse one-neighbor row can therefore raise `ValueError` while `register_reference_thumbnail(..., allow_feature_fallback=True)` is attempting feature fallback. Until that path is fixed, covered by a regression test, and assigned a new frozen runtime-code digest, it must not be described as production-safe or fail-open.
- `src/patch_qc/normal_prior_spike.py::export_comparisons()` accepts reference-shard directories but does not accept or enforce the frozen `inputs.reference_shard_receipt_sha256` values recorded by `configs/normal_peak_preservation_v1.json`. `normal_peak_preservation_v1` is legacy, superseded, and non-runnable for new claims: use it only as provenance for reproducing existing results, never for a new execution, new quality claim, or threshold decision.
- `src/patch_qc/structural_suppression_eval.py` and `src/patch_qc/dual_evidence_rescue_eval.py` write receipt metadata with the constant `code.dirty: true`. This is a conservative known provenance limitation, not a measured report of the Git worktree state; do not use those receipts to assert actual workspace cleanliness.
