# 산업 부품 외관 결함 검사: 원본 해상도 패치 분류 (DINOv3)

고해상도 산업 부품 이미지를 축소하지 않고 `256×256` 패치로 잘라, 사전학습된 DINOv3 모델 위에 가벼운 분류기를 얹어 각 패치가 정상인지 결함인지 판별하는 연구·PoC 코드입니다.

## 한눈에 보기

- **문제**: 여러 제품(다품종)의 부품 사진에서 백점·흑점·균열·chipping 같은 아주 작은 외관 결함을 찾아야 합니다. 제품마다 모양·촬영 각도·초점이 달라 "정상"의 모습이 매우 다양합니다.
- **접근**: 원본 해상도에서 패치를 잘라 정상/결함 2분류를 합니다. Backbone(frozen DINOv3 ViT)은 고정하고 경량 Binary head만 학습하며, 위치 mask를 함께 배우는 선택형 Auxiliary·Mask 멀티태스크 branch가 있습니다.
- **학습 데이터**: 실제 불량(NG) 이미지는 학습에 쓰지 않습니다. 정상(OK) 이미지의 유효 영역(ROI) 안에 결함을 **합성**해 학습 신호를 만듭니다.
- **출력**: 패치별 결함 점수, 전체 이미지 heatmap, CAM·LeGrad 설명 overlay, 사람 검토(`REVIEW`)용 후보 목록과 실행 증빙(run receipt).
- **기술 스택**: Python 3.10+, PyTorch, timm, Transformers, OpenCV, NumPy, pytest, Slurm(GPU 실행 template).
- **포함 / 미포함**: 모델 연구 코드, 설정, 테스트, 안전 검사만 포함합니다. 원본 이미지, 학습 데이터, backbone weight, checkpoint, 운영 대시보드, 배포 코드는 포함하지 않습니다.

> 이 저장소는 최종 운영 모델이 아니며 성능 수치를 주장하지 않습니다. 합성 결함으로 얻은 결과는 실제 불량 검출률을 대신 증명하지 않습니다.

## 왜 패치 분류인가

수천만 화소 원본을 모델 입력 크기로 한 번에 축소하면 몇 픽셀짜리 미세 결함이 사라집니다.
전체 이미지를 하나의 정상 분포로 보는 이상탐지는 홀·림·핀·각인·반사 같은 **정상 구조**에 과하게 반응했습니다.
그래서 원본 크기 그대로 작은 패치를 잘라, 다양한 정상 구조는 정상으로 학습시키고 미세 결함은 해상도 손실 없이 보도록 했습니다.
전체 이미지는 패치를 겹쳐 훑은 뒤 점수를 원래 좌표로 다시 모아 heatmap으로 보여줍니다.

## 동작 방식

```text
OK 원본 이미지 (저장소 밖)
  -> Manifest 생성 + 제품·개체 단위 split (train / calibration / holdout)
  -> 256×256 패치 샘플링 + ROI 내부 합성 결함 생성
  -> frozen DINOv3 backbone + Binary head 학습   (선택: Aux + Mask branch)
  -> 전체 이미지 sliding window 추론 (256×256, stride 128)
  -> 패치 점수 heatmap / CAM / LeGrad overlay
  -> 사람 검토 (REVIEW)

실제 NG 이미지: 정책을 고정한 뒤 sealed evaluation에서만 사용
```

- **Binary head**: DINOv3의 CLS token과 patch token 평균을 이어 붙여 작은 MLP로 정상/결함 logit을 냅니다. Backbone은 학습하지 않으므로 checkpoint 파일이 작습니다.
- **합성 결함**: `spot`, `discolor`, `scratch`, `pollution`, `fiber`, `chipping` 여섯 종류를 `subtle`·`medium`·`strong` 세 강도로 만듭니다. ROI는 합성 결함을 놓을 위치를 제한할 뿐, 실제 추론 결과를 지우는 데 쓰지 않습니다.
- **설명 도구**: heatmap, gradient×token CAM, attention-gradient 기반 LeGrad는 사람이 결함 후보 위치를 확인하기 위한 도구이며 정확한 segmentation을 보장하지 않습니다.
- **Label-free refinement**: 수동 NG label 없이 OK·합성 신호·normal memory·TTA 안정성을 조합해 검토할 이미지를 순위화합니다. 결과는 자동 PASS/FAIL이 아니라 `REVIEW` 상태입니다.

## 평가 원칙

학습이나 threshold 조정에 평가 데이터가 섞이면 점수가 실제보다 좋아 보입니다(데이터 누수). 이를 막기 위해 다음 원칙을 따릅니다.

- **Group-safe split**: 같은 물리 개체(`product::object_id`)의 여러 각도·초점 이미지는 항상 같은 split에 둡니다.
- **Frozen policy**: 실제 NG를 열기 전에 설정·checkpoint·정책의 hash를 먼저 고정합니다.
- **Sealed evaluation**: 고정된 정책을 실제 NG에 한 번만 적용합니다. 결과를 보고 정책이나 threshold를 다시 조정하지 않으며, 조정이 필요하면 새 미사용 평가셋을 따로 마련합니다.
- **역할 분리**: OK는 정상 학습, 합성 결함은 학습 신호, 실제 NG·LabelMe는 sealed evaluation과 사람 검토에만 씁니다.
- **실행 증빙**: run receipt에 manifest·config·checkpoint digest와 실행 상태를 남겨 무엇으로 평가했는지 추적합니다.

자세한 계약은 [데이터 계약](docs/data_contract.md)과 [기술 가이드 6절](docs/technical_guide.md#6-데이터와-평가-계약)을 참고하세요.

## 빠른 시작

설치와 로컬 검증 (GPU 불필요):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m pytest -q
python scripts/check_repository_safety.py --root .
```

학습·추론에는 NVIDIA GPU 1장, 실행 환경에 미리 캐시된 DINOv3 backbone, 저장소 밖의 OK 데이터가 필요합니다. 아래 경로는 모두 플레이스홀더입니다.

```bash
# 1) OK 이미지 manifest + group-safe split
patch-qc build-manifest \
  --root /path/to/QC_Data/OK \
  --output-dir /path/to/runs/manifest \
  --seed 17 --ratios 0.70 0.15 0.15

# 2) (권장) 합성 결함 미리보기로 크기·대비·위치 확인
patch-qc make-synthetic-preview \
  --image /path/to/QC_Data/OK/sample.jpg \
  --output-dir /path/to/runs/synthetic_preview

# 3) Binary probe 학습 (config는 저장소 밖에 복사해 경로 수정)
cp configs/poc.yaml /path/to/run-configs/poc.yaml
CUDA_VISIBLE_DEVICES=0 patch-qc-train \
  --config /path/to/run-configs/poc.yaml \
  --manifest /path/to/runs/manifest/manifest.jsonl \
  --output-dir /path/to/runs/binary_probe_001
```

추론: 범용 inference CLI는 없습니다. 전체 이미지 sliding-window 추론은 라이브러리 함수 `patch_qc.inference.predict_image`(기본 `256×256`, stride `128`)로 수행하며, 고정 정책의 실제 OK/NG 평가는 `patch-qc-eval-actual-multitask`를 사용합니다. 멀티태스크 학습(`patch-qc-train-multitask`), refinement(`patch-qc-refine`), Slurm template(`slurm/`)까지 포함한 전체 절차는 [기술 가이드 10~15절](docs/technical_guide.md#10-설치)에 있습니다.

## 폴더 구조

```text
industrial-visual-qc-dinov3/
├── src/patch_qc/   # 패키지: manifest, 합성 결함, 모델, 학습, 추론, 설명, refinement
├── configs/        # 기준 설정 (poc.yaml, 멀티태스크·refinement 설정 등)
├── tests/          # 단위·CLI·계약 테스트
├── scripts/        # 저장소 안전 검사 (check_repository_safety.py)
├── slurm/          # GPU 클러스터 실행 template
├── experiments/    # 기준 코드와 분리해 보존한 실험 snapshot
└── docs/           # 기술 가이드, 아키텍처, 데이터 계약, 실험 매트릭스
```

## 더 읽기

| 문서 | 내용 |
| --- | --- |
| [기술 가이드](docs/technical_guide.md) | 전체 설계, 실행 명령, 출력·재현성 규칙, 알려진 제한, 용어 정리 |
| [아키텍처](docs/architecture.md) | 코드 계층과 기준 코드/분리 실험 경계 |
| [데이터 계약](docs/data_contract.md) | OK·합성·실제 NG·ROI 사용 원칙 |
| [실험 매트릭스](docs/experiment_matrix.md) | 방법론별 상태와 사용 범위 |
| [실험 실행 제한](experiments/README.md) | 실험용 후처리 모듈의 known limitation |

## 검증 상태

- 로컬 테스트: `python -m pytest -q` 기준 `270 passed` (GPU 학습 없이 실행되는 단위·CLI·계약 테스트).
- 저장소 안전 검사: 금지 경로·확장자, secret처럼 보이는 파일명, 대용량 파일을 검사합니다.
- 이는 코드와 계약이 의도대로 동작한다는 의미이며, 실제 GPU 학습 결과나 모델 품질을 증명하지 않습니다.

## 한계

- Backbone weight와 checkpoint가 저장소에 없어 clone만으로 학습·추론을 재현할 수 없습니다.
- 범용 production inference CLI가 없고, Binary holdout은 합성 결함 기반이라 실제 NG 성능이 아닙니다.
- 학습 이미지와 평가 이미지의 hash 교집합 검사가 코드에서 자동으로 강제되지 않아, sealed evaluation 전에 별도 preflight가 필요합니다([기술 가이드 6.4절](docs/technical_guide.md#64-학습평가-이미지-hash-교집합-preflight)).
- Canonical Binary 학습은 resolved config·run receipt를 자동 기록하지 않아 증빙을 외부에서 보강해야 합니다.
- 과거 실제 NG 비교에는 정책 선택에 반복 사용된 개발 진단 결과가 포함되어 있어 unseen sealed test로 볼 수 없습니다.
