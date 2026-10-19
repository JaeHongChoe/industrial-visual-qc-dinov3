# Visual QC 실험 매트릭스

| 분류 | 방법/계보 | 코드 위치 | 기본 운영 정책 | 상태 |
| --- | --- | --- | --- | --- |
| Canonical | 256 patch Binary + DINOv3 | `src/patch_qc` | 기준 patch 분류 | 유지 |
| Canonical | ROI 합성, Aux+Mask 멀티태스크 | `src/patch_qc` | 학습·분석 흐름 | 유지 |
| Canonical | sealed actual evaluation, LeGrad, review/refinement | `src/patch_qc` | frozen policy 뒤 평가·검토만 | 유지 |
| Supported experimental | normal-prior / recurrence spike | `src/patch_qc/` | 기본 threshold 변경 금지 | 테스트와 함께 보존; [known limitations](../experiments/README.md#known-limitations--실행-제한) 적용 |
| Legacy v1 provenance | `normal_peak_preservation_v1` | `configs/normal_peak_preservation_v1.json`, `src/patch_qc/normal_prior_spike.py` | 신규 실행·품질 주장·threshold 결정 금지 | superseded/non-runnable; [known limitations](../experiments/README.md#known-limitations--실행-제한) 적용 |
| Supported experimental | structural suppression / dual-evidence rescue | `src/patch_qc` | 기본 PASS/FAIL 변경 금지 | 테스트와 함께 보존 |
| Preserved isolated | dense normal reference A/B (2026-08-31) | `experiments/dense_normal_reference_ab_20260831/` | 품질·운영 주장 금지 | 성공 smoke `quality_claims=false`; 7파일 보존본 |

이 표의 experimental 항목은 기본 코드에 승격되었다는 뜻이 아니다. 비교 결과는 실제 NG를 통한 별도, 사전 승인된 sealed 평가와 인간 검토 없이는 성능 개선 또는 운영 정책 변경으로 해석하지 않는다.

