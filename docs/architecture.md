# Visual QC 모델 코드 아키텍처

## 범위와 경계

이 저장소는 다품종 산업 부품 외관 QC의 모델 연구 코드만 다룬다. 운영 대시보드 UI, 고객사 배포, 데이터 수집 서비스는 포함하지 않으며 별도 대시보드 저장소도 수정하지 않는다.

기준 코드는 canonical `src/patch_qc`다. 기존 import 경로의 검증 범위를 보존하기 위해 flat package 구조를 유지한다.

## Canonical 처리 흐름

```text
외부 OK 이미지/manifest
  -> capture 및 split
  -> 256×256 patch + ROI 기반 합성 결함 배치
  -> DINOv3 backbone + Binary head
  -> Aux + Mask 멀티태스크 학습 또는 추론
  -> 후보 설명(LeGrad)·review export·label-free refinement
  -> frozen policy 뒤의 sealed actual evaluation
```

Binary 분류는 patch 단위 정상/결함 점수를 낸다. Aux와 Mask는 별도 보조/위치 학습 출력이며, ROI는 합성 위치와 전처리 보조에만 사용한다. 실제 추론 결과를 ROI로 강제 마스킹하지 않는다.

## 코드 계층

| 계층 | 위치 | 책임 |
| --- | --- | --- |
| Canonical | `src/patch_qc/` | 재현 가능한 기본 모델, 학습, 추론, sealed evaluation |
| Configuration | `configs/` | 실행 파라미터와 실험 계약 |
| Scheduler | `slurm/` | 단일 scheduler-visible GPU 단위의 H100 launcher |
| Verification | `tests/`, `scripts/check_repository_safety.py` | 단위/CLI 검증과 code-only 안전 게이트 |
| Documentation | `docs/` | 데이터·평가 계약과 출처 |
| Preserved isolated snapshot | `experiments/dense_normal_reference_ab_20260831/` | H100 Git 미추적 dense-normal 7파일 보존본; canonical import 금지 |

## 실험 계층의 원칙

normal-prior, recurrence, structural suppression, dual-evidence는 지원 실험 코드다. 테스트와 명확한 입력·출력 계약은 보존하지만 기본 CLI, 기본 threshold, PASS/FAIL 정책을 바꾸지 않는다. normal-prior와 v1 재현 경로의 실행 제약은 [`experiments/README.md`](../experiments/README.md#known-limitations--실행-제한)에 고정한다. `filtered_patch_view`/`leakage_guard`는 clean-retrain 계보의 일부이므로 canonical root에 overlay하지 않고 독립 스냅샷에서 보존한다.


`dense_normal_reference_ab_20260831`은 `experiments/dense_normal_reference_ab_20260831/`에 체크인한 별도 untracked 7파일 계열의 **preserved isolated snapshot**이다. 성공 smoke의 `quality_claims=false` 때문에 품질 개선 또는 운영 가능성을 주장할 근거가 없으며 candidate나 canonical 승격 대상으로 표시하지 않는다.

clean retrain은 `dataset.py`, `model.py`, `train_probe.py` 계보가 canonical과 다르므로 동일 모듈을 덮어쓰지 않는다. 필요한 allowlist만 독립 snapshot에 보관하고, manifest·cache·checkpoint·log·receipt는 외부에 둔다.
