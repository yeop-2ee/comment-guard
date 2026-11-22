# 새 폴더 재실행 기록

과제 "마. 재실행" 항목의 기록입니다. 깃허브의 공개 저장소를 새 폴더에 내려받아 **README의 명령만** 순서대로 실행했습니다. 계획은 `docs/PROJECT_SPEC.md` 9장에 있습니다.

## 1. 환경

| 항목 | 값 |
|---|---|
| 일시 | 2026-11-22 12:05~13:11 KST |
| 폴더 | 새 임시 폴더 `fresh-20261122` (기존 작업 폴더와 별개, 모델·가상환경 없음). 기록 안의 절대 경로는 `<임시 폴더>`로 줄였습니다 |
| 받은 코드 | `git clone https://github.com/yeop-2ee/comment-guard.git` → 커밋 `eeba8ff` |
| OS / CPU | macOS 26.6.2, arm64 (Apple Silicon, M2 학습은 MPS) |
| Python | 3.11.15 |
| 설치된 패키지 | 75개, `pip freeze` sha256 앞 16자리 `c7e7f3bab4a605dd` |
| 환경 변수 | 셸 환경 변수 없이 실행(`env -i`로 HOME, PATH, LANG만 넘김) |

## 2. 실행한 명령과 결과

| 순서 | README 명령 | 결과 | 걸린 시간 | 기록 |
|---|---|---|---|---|
| 1 | `git clone ...`, `cd comment-guard` | 성공 | — | — |
| 2 | `make setup` | 성공. 의존성 75개 설치 (torch 포함) | 47분 18초 (대부분 다운로드) | — |
| 3 | `make all-local` | 성공. 데이터 해시 검증 → 검사·분할 → M1 학습 → M1 평가 → pytest **75개 통과, 2개 건너뜀** | 1분 22초 | `runs/20261122-125331-data`, `-125404-validate`, `-125408-train`, `-125428-evaluate` |
| 4 | `python -m commentguard.m2`, `python -m commentguard.evaluate models/$(cat models/LATEST_M2)` | 성공. 학습률 2e-5, 에폭 4 선택 (원래와 같음) | 학습 13분 7초 | `runs/20261122-125754-train-m2`, `-131107-evaluate-m2` |
| 5 | `MODEL_URI=models/$(cat models/LATEST) make serve` + `make smoke-local` | 로컬 스모크 **13/13 통과** (새로 학습한 M1 `m1-20261122-eeba8ff`) | — | `runs/20261122-130552-smoke-local` |
| 6 | 같은 스모크를 배포 주소에 (`scripts/smoke_cloud.py --url https://comment-guard-jade.vercel.app --target cloud`) | 클라우드 스모크 **13/13 통과** | — | `runs/20261122-130554-smoke-cloud-rerun` |

- 건너뛴 테스트 2개는 `models/`에 ONNX 파일이 있어야 하는 테스트입니다(`make export-onnx` 전에는 건너뜀).
- 모든 기록이 `git_dirty: false`, git 커밋 `eeba8ff`입니다.

## 3. 원래 실행과 비교

| 항목 | 원래 기록 | 재실행 | 같은가 |
|---|---|---|---|
| 가공 데이터 sha256 (train_fit / valid / eval) | `b6a752ef7bfc` / `aec190ccee10` / `6904b087a99e` | 같음 | 같음 |
| M1 eval macro-F1 [95% 구간] | 0.5694 [0.5239, 0.6129] (`runs/20261109-232715-evaluate`) | 0.5694 [0.5239, 0.6129] | 같음 |
| B0 eval macro-F1 | 0.169 | 0.169 | 같음 |
| M2 valid macro-F1, 선택한 학습률·에폭 | 0.7132, 2e-5, 에폭 4 (`runs/20261110-003741-train-m2`) | 0.7132, 2e-5, 에폭 4 | 같음 |
| M2 심각도 기준값 | 주의 39, 심각 70 | 주의 39, 심각 70 | 같음 |
| M2 eval macro-F1 [95% 구간] | 0.6936 [0.649, 0.7332] (`runs/20261110-010215-evaluate-m2`) | 0.6936 [0.649, 0.7332] | 같음 |
| M2 medium 정상 오차단율 / 혐오 차단율 | 5.0% / 66.4% | 5.0% / 66.4% | 같음 |
| M2 놓친 혐오 (hate → none) | 1건 | 1건 | 같음 |

같은 seed와 같은 데이터로 M1과 M2 모두 원래 결과를 그대로 재현했습니다. 모델 버전은 만든 날짜와 커밋이 달라 `m1-20261122-eeba8ff`, `m2-20261122-eeba8ff`로 새로 붙었습니다.

## 4. 막힌 부분과 처리

| 문제 | 내용 | 처리 |
|---|---|---|
| 로컬 포트 충돌 | 첫 시도에서 `PORT=8090`을 다른 프로그램이 쓰고 있어 우리 서버가 뜨지 않았고, 스모크가 그 프로그램에 요청해 실패함 | 비어 있는 포트(8765)로 다시 실행. README 명령의 문제는 아님 (기본 포트 8080은 `PORT=`로 바꿀 수 있음) |
| 클라우드 503 | 첫 시도의 클라우드 스모크에서 모든 요청이 503. Vercel Blob 저장소가 무료 한도 초과로 정지돼 있었음 | 모델을 GitHub Release로 옮겨 다시 배포한 뒤 재실행해 통과 (`docs/CLOUD_RUNS.md` 8장) |

- 두 실패는 새 폴더의 `runs/index.csv`에 `failed`로 남아 있고 이 저장소의 `runs/index.csv`에도 옮겼습니다(`20261122-125505-smoke-local`, `-125506-smoke-cloud-rerun`). 다만 **다시 실행하기 전에 그 두 실행의 기록 폴더를 지워서**, 폴더(`run.json`, 스모크 결과)는 남아 있지 않습니다.
- README 명령 자체를 고쳐야 할 부분은 없었습니다.

## 5. 재실행하지 않은 것

- 클라우드 배포 단계(`make vercel-setup`, `upload-model`, `deploy`)는 새 폴더에서 다시 하지 않았습니다. 같은 프로젝트에 다시 배포하면 서비스 중인 배포를 바꾸게 되기 때문입니다. 배포 단계는 원래 폴더에서 여러 번 실행한 기록이 있습니다(`docs/CLOUD_RUNS.md`).
- M3, 앙상블, 행동 검사는 다시 실행하지 않았습니다(서빙 경로가 아님).
