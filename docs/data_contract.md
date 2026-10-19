# Visual QC 모델 데이터 계약

## 저장소에 두지 않는 것

다음은 Git 및 이 작업 디렉터리에 포함하지 않는다.

- OK/NG 원본 이미지, LabelMe, manifest, patch/tar cache
- checkpoint, DINOv3 backbone, ONNX/engine 등 모델 바이너리
- 실행 output, artifact, log, result, run, receipt
- `.env`와 secret, credential, token 파일

실행자는 외부 경로를 명시적으로 전달하고, 데이터와 모델의 접근 권한은 저장소와 별도로 관리한다. `scripts/check_repository_safety.py`는 금지 경로·확장자와 10 MiB 초과 파일을 fail-closed로 검사한다.

## 학습과 평가 경계

| 구분 | 허용 용도 | 금지 용도 |
| --- | --- | --- |
| 정상 OK | split, 정상 patch 학습, 합성 결함의 바탕 이미지 | 실제 NG로 간주하거나 false-positive 분모라고 주장 |
| 합성 결함 | Binary/Aux/Mask supervised 학습 | 실제 NG 성능의 대체 주장 |
| 실제 NG·LabelMe | frozen policy 뒤 sealed evaluation과 사람 검토 | 학습, calibration, threshold tuning, fixture, policy 재선정 |
| ROI | 합성 결함 배치와 전처리 보조 | 실제 추론 PASS/FAIL 강제 마스크 |

`product::object_id`가 같은 capture는 같은 split에 머문다. actual evaluation은 frozen policy, 외부 요청, training hash disjointness를 검증한 뒤 한 번만 실행한다. 학습 데이터와 평가 이미지/JSON의 hash 충돌 또는 frozen digest 변경은 평가를 거부해야 한다.

## 입력·출력 재현성

기본 patch는 `256×256`이며 native full-image inference는 stride `128`과 오른쪽/아래 tail window를 명시한다. 설정, source commit, 입력 manifest digest, checkpoint digest, runtime receipt는 외부 실행 결과로 남기되 저장소에 생성물 자체를 넣지 않는다.
