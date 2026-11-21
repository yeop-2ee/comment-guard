# 클라우드 실행 기록 (Vercel)

과제 "나. 클라우드 실행" 항목의 기록입니다. 구성의 근거는 `docs/PROJECT_SPEC.md` 6장, 실행 기록 원본은 `runs/`에 있습니다. Blob 저장소 ID는 모든 기록에서 `*`로 가렸습니다.

## 1. 구성

| 항목 | 값 | 선택 이유 |
|---|---|---|
| 서비스 | Vercel Functions, Python 3.12 런타임, FastAPI (`app.py` → `commentguard.api:app`) | 요청이 있을 때만 실행되고 요청이 없으면 비용이 없음. 로컬 `uvicorn`과 같은 앱을 그대로 실행. 화면과 API를 같은 주소로 제공 |
| 플랜 | Hobby (무료, 결제 수단 없음) | 한도를 넘으면 과금 없이 중지되므로 비용이 생기지 않음 |
| 리전 | 함수 `icn1`(서울), Blob 저장소 `icn1` | 사용자가 한국에 있음. 모델을 받는 거리도 짧게 |
| 실행 설정 | 최대 실행 시간 60초(`vercel.json`), 메모리는 Hobby 기본값 2GB(vCPU 1) | M2 최대 메모리 약 1.3GB(로컬 측정)에 여유 |
| 모델 저장 | Vercel Blob **비공개** 저장소 `models/m2-20261110-314b2b2/` (`model.onnx` 434MB, 토크나이저, 모델 카드, `serving.json`) | CLI 배포 업로드 한도(Hobby 100MB)를 넘어서 배포에 넣을 수 없음. 코드와 모델을 분리해 모델만 교체 가능 |
| 모델 실행 형식 | ONNX (onnxruntime 1.32.0) | PyTorch는 Python 함수 크기 한도(500MB)를 넘음. 변환 결과는 valid 790건에서 PyTorch와 같음(아래 2장) |
| 함수 크기 | 59.06MB (빌드 출력 `λ fastapi (59.06MB) [icn1]`) | 의존성만 들어가고 모델·데이터·기록은 `vercel.json`의 `excludeFiles`와 `.vercelignore`로 뺌 |
| 연결 | 브라우저·curl → 프로덕션 도메인 `https://comment-guard-jade.vercel.app` → 함수 → 첫 요청에서 Blob의 모델 로드 | 프로덕션 도메인만 공개. 개별 배포 주소는 Deployment Protection으로 Vercel 로그인 필요 |
| 배포 방식 | `vercel deploy --prod --skip-domain --env MODEL_URI=...` → 새 배포의 `/health` 200 확인 → `vercel promote` | 새 배포가 실패하면 승격하지 않아 이전 배포가 계속 서비스함 |

## 2. 모델 변환 (ONNX)

`make export-onnx` → `runs/20261122-031030-export-onnx/`

| 확인 항목 | 기준 | 결과 |
|---|---|---|
| 토큰 ID (valid 790건, 배치 13개) | 완전 일치 | 불일치 배치 0 |
| 클래스 확률 최대 차이 | ≤ 1e-4 | 1.5e-5 |
| hate·bias 예측 일치율 | 100% | 100% / 100% |
| 파일 크기 | — | 434.2MB |

같은 가중치를 형식만 바꾼 것이라 모델 버전(`m2-20261110-314b2b2`)은 그대로입니다. 클라우드와 같은 ONNX 경로로 로컬 서버를 띄워 스모크 13개도 통과했습니다(`runs/20261122-031113-smoke-local-onnx/`).

## 3. 준비와 배포

| 순서 | 명령 | 결과 | 기록 |
|---|---|---|---|
| 1 | `make vercel-setup` | 첫 실행 실패: 저장소는 만들었지만 `.env.local`에서 저장소 ID를 찾지 못함 → 저장소 목록에서 읽도록 스크립트 수정 | `runs/20261122-031935-vercel-setup` (failed) |
| 1 | `make vercel-setup` (재실행) | 성공. 프로젝트 `comment-guard`, Blob 저장소 `comment-guard-models`(비공개, icn1). 실행 중 Vercel CLI가 `.gitignore`에 줄을 추가해서 `git_dirty: true`로 기록됨(스크립트 코드는 커밋된 상태, 추가된 줄은 되돌림) | `runs/20261122-032003-vercel-setup` |
| 2 | `make upload-model` | 두 번 실패: Blob CLI가 단기 OIDC 토큰과 저장소 ID를 함께 요구 → 둘 다 환경 변수로 넘기도록 수정 | `runs/20261122-032021-upload-model`, `-032032-` (failed) |
| 2 | `make upload-model` (재실행) | 성공. 파일 5개 업로드, `serving.json`은 마지막에 올림 | `runs/20261122-032044-upload-model` |
| 3 | `make deploy` | 실패: 함수에 라이브러리가 하나도 설치되지 않음(`ModuleNotFoundError: No module named 'fastapi'`). Vercel이 `requirements.txt` 대신 `pyproject.toml`(의존성 목록 없음)을 읽음 → 고정 버전을 `pyproject.toml`에도 적고 일치 검사 테스트 추가. 새 배포 `/health` 500이라 **승격하지 않음** | `runs/20261122-032205-deploy-main` (failed, `runtime_logs_fetched_later.jsonl`) |
| 3 | `make deploy` | 실패: 모델 다운로드 중 `OSError: [Errno 28] No space left on device`. 함수의 `/tmp`가 모델(434MB)보다 작음 → `model.onnx`는 메모리로 받아 바로 로드하도록 수정. 새 배포 `/health` 503이라 **승격하지 않음** | `runs/20261122-032351-deploy-main` (failed, `runtime_logs_fetched_later.jsonl`) |
| 3 | `make deploy` | `/health` 200 확인 후 승격은 됐지만, 그 뒤 프로덕션 주소를 찾는 부분이 실패 → 별칭 목록에서 찾도록 수정 | `runs/20261122-032644-deploy-main` (failed) |
| 3 | `make deploy` (재실행) | 성공. 배포 `dpl_HaxxJVMbd2TuYfYjVbLzDZzqS3gn`, 새 배포 `/health` 200 확인 후 승격 | `runs/20261122-032913-deploy-main` |

세 번의 배포 실패 모두 배포 스크립트가 새 배포의 `/health`를 먼저 확인했고, 실패한 배포는 프로덕션 도메인으로 승격되지 않았습니다. 실패한 두 배포의 런타임 로그는 Hobby 보존 시간(1시간) 안에 받아서 해당 기록 폴더에 저장했습니다.

## 4. 클라우드 스모크 (실제 입력 → 결과)

`make smoke` → `runs/20261122-033033-smoke-cloud-main/cloud_smoke.json` (git 커밋 `6c25474`, 2026-11-22 03:30 KST)

- 대상: `https://comment-guard-jade.vercel.app`, 배포 `dpl_HaxxJVMbd2TuYfYjVbLzDZzqS3gn`, `MODEL_URI=blob://models/m2-20261110-314b2b2/`
- 응답의 `model_version`: `m2-20261110-314b2b2` (모든 케이스에서 `/health`와 같음)
- **14/14 통과**

| 케이스 | 내용 | 상태 | 지연(ms) |
|---|---|---|---|
| T11 | `/health` | 200 | 288 |
| T1 | 정상 댓글 1건 | 200 | 109 |
| T2 | 100건 배치, 순서 유지 | 200 | 9,623 |
| T3 | 이모지·영어·초성 섞인 댓글 | 200 | 355 |
| T4~T7 | 빈 문자열, 1,001자, 101건, 잘못된 민감도 → 422 | 422 | 66~300 |
| F | 필터 medium 3건 | 200 | 282 |
| T8~T10 | 증거 리포트(경고, 해시 재현, CSV) | 200 | 344~555 |
| META | `/v1/meta` | 200 | 61 |
| LOG | 런타임 로그에 요청 기록이 있고 보낸 댓글 원문 120개 중 어느 것도 없음 | — | — |

예시(T1): 입력 `{"texts": ["둘다 넘 좋다~행복하세요"]}` (공개 데이터 댓글) → `label: none`, `probs: {none: 0.9856, offensive: 0.0111, hate: 0.0033}`, `severity_score: 1`, `grade: normal`.

지연은 한국에서 보낸 요청의 왕복 시간입니다. 100건 배치(T2)가 9.6초로 로컬 맥 단일 스레드 측정값(p50 879ms, `EVALUATION.md` 8.5장)보다 느린 것은 함수가 vCPU 1개이기 때문입니다.

## 5. Cold start

새 인스턴스의 첫 요청은 Blob에서 모델 434MB를 받아 ONNX 세션을 만드는 시간을 포함합니다. 성공한 배포의 런타임 로그(`runs/20261122-033033-smoke-cloud-main/runtime_logs.jsonl`)에서 첫 요청(`/`)의 처리 시간은 **25.1초**였고, 그 뒤 요청은 위 표처럼 수십~수백 ms입니다. 같은 배포에서 인스턴스가 두 개 떠서 모델을 두 번 받았습니다(`model_loaded` 2회).

- 측정은 한 번뿐이라 cold start의 분포는 모릅니다.
- 인스턴스가 새로 뜰 때마다 Blob 전송량(Hobby 월 10GB)을 약 434MB씩 씁니다. 이 날의 업로드·배포 시도로 쓴 양은 Vercel 대시보드의 사용량으로 확인할 수 있고, 기록에는 남기지 않았습니다.

## 6. 확인하지 못한 것

- 동시 요청이 많을 때의 동작(인스턴스 수, 지연)은 재지 않았습니다.
- 메모리 사용량은 Vercel이 함수별로 보여 주는 값을 따로 기록하지 않았습니다.
- 프로덕션 도메인은 공개라서, 남용 방지는 입력 크기 제한과 Hobby 한도에 의존합니다. 요청 속도 제한(rate limit)은 설정하지 않았습니다.
