# 클라우드 실행 기록 (Vercel)

과제 "나. 클라우드 실행" 항목의 기록입니다.

> **현재 상태 (2026-11-22 13:05 KST):** 프로덕션 `https://comment-guard-jade.vercel.app`, 배포 `dpl_CqKhq1Pou85PAvirQK7xroX2QcZx`, 서빙 모델 M2 `m2-20261110-314b2b2`. 모델은 이 저장소의 GitHub Release [`model-m2-20261110-314b2b2`](https://github.com/yeop-2ee/comment-guard/releases/tag/model-m2-20261110-314b2b2)에서 받습니다(8장). 마지막 클라우드 스모크 **14/14 통과** (`runs/20261122-130351-smoke-cloud-release/`).
 구성의 근거는 `docs/PROJECT_SPEC.md` 6장, 실행 기록 원본은 `runs/`에 있습니다. Blob 저장소 ID는 모든 기록에서 `*`로 가렸습니다.

## 1. 구성

| 항목 | 값 | 선택 이유 |
|---|---|---|
| 서비스 | Vercel Functions, Python 3.12 런타임, FastAPI (`app.py` → `commentguard.api:app`) | 요청이 있을 때만 실행되고 요청이 없으면 비용이 없음. 로컬 `uvicorn`과 같은 앱을 그대로 실행. 화면과 API를 같은 주소로 제공 |
| 플랜 | Hobby (무료, 결제 수단 없음) | 한도를 넘으면 과금 없이 중지되므로 비용이 생기지 않음 |
| 리전 | 함수 `icn1`(서울), Blob 저장소 `icn1` | 사용자가 한국에 있음. 모델을 받는 거리도 짧게 |
| 실행 설정 | 최대 실행 시간 60초(`vercel.json`), 메모리는 Hobby 기본값 2GB(vCPU 1) | ONNX로 M2를 받아 올리는 동안의 최대 메모리 1.65GB(로컬 측정)가 들어감 |
| 모델 저장 | 처음: Vercel Blob **비공개** 저장소 `models/m2-20261110-314b2b2/`. 지금: 이 저장소의 GitHub Release `model-m2-20261110-314b2b2` (`model.onnx` 434MB, 토크나이저, 모델 카드, `serving.json`) | CLI 배포 업로드 한도(Hobby 100MB)를 넘어서 배포에 넣을 수 없음. 코드와 모델을 분리해 모델만 교체 가능. Blob이 한도 초과로 정지돼서 옮김(8장) |
| 모델 실행 형식 | ONNX (onnxruntime 1.32.0) | PyTorch는 Python 함수 크기 한도(500MB)를 넘음. 변환 결과는 valid 790건에서 PyTorch와 같음(아래 2장) |
| 함수 크기와 의존성 | 빌드 출력 `λ fastapi (59.06MB) [icn1]`. 의존성은 `pyproject.toml`의 서빙용 32개만(FastAPI, onnxruntime, tokenizers, vercel-blob 등) | 런타임 로그에서 함수가 처음 뜰 때 의존성을 `/tmp`에 설치하는 것을 확인함(`Installing runtime dependencies...`). 모델을 받을 공간이 남도록 scikit-learn·scipy·pandas는 뺌(7장) |
| 연결 | 브라우저·curl → 프로덕션 도메인 `https://comment-guard-jade.vercel.app` → 함수 → 첫 요청에서 Blob의 모델 로드 | 프로덕션 도메인만 공개. 개별 배포 주소는 Deployment Protection으로 Vercel 로그인 필요 |
| 배포 방식 | 배포 전 `MODEL_URI` 검사(M5) → `vercel deploy --prod --skip-domain --env MODEL_URI=...` → 새 배포의 `/health` 200과 **판정 1건** 확인 → `vercel promote` | 새 배포가 실패하면 승격하지 않아 이전 배포가 계속 서비스함. 판정 확인은 7장의 장애 뒤에 추가 |

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

새 인스턴스의 첫 요청은 Blob에서 모델 434MB를 받아 ONNX 세션을 만드는 시간을 포함합니다. 런타임 로그에서 첫 요청의 처리 시간은 세 번 측정했습니다.

| 기록 | 첫 요청 처리 시간 |
|---|---|
| `runs/20261122-033033-smoke-cloud-main/runtime_logs.jsonl` | 25.1초 |
| `runs/20261122-041302-smoke-cloud-restore/runtime_logs.jsonl` | 29.3초 |
| `runs/20261122-041509-smoke-cloud-main/runtime_logs.jsonl` | 29.8초 |

그 뒤 요청은 4장 표처럼 수십~수백 ms입니다. 같은 배포에서 인스턴스가 여러 개 뜨면 각각 모델을 받습니다(`model_loaded`가 여러 번). 인스턴스가 새로 뜰 때마다 Blob 전송량(Hobby 월 10GB)을 약 434MB씩 씁니다.

## 6. 화면에서 모델 고르기 시도 (M1·M3)

배포한 화면에서도 모델을 고를 수 있게 M1과 M3를 함께 올려 봤습니다(명세서 4.6장). 큰 모델은 한 번에 하나만 메모리에 두고 바꿔 끼우는 방식입니다.

| 순서 | 내용 | 기록 |
|---|---|---|
| 1 | M3 ONNX 변환, valid에서 PyTorch와 같음(확률 차이 최대 1.4e-5, 예측 100% 일치) | `runs/20261122-034620-export-onnx` |
| 2 | M1·M3 업로드, 함께 배포 | `runs/20261122-035010-upload-model`, `-035028-upload-model`, `-035155-deploy-models` |
| 3 | 스모크: 기본 14개와 M1 전환·M2 복귀는 통과, **M3 전환은 500** | `runs/20261122-035330-smoke-cloud-models` (failed) |
| 원인 | 런타임 로그: `Vercel Runtime Error: instance was killed because it ran out of available memory`. 로컬 측정으로 M3를 받아 올리는 동안 최대 2.07GB(M2는 1.65GB)라 함수 메모리 2GB를 넘음 | 같은 폴더 `runtime_logs.jsonl` |

메모리를 줄이는 형식(`.ort`, 받은 바이트를 복사 없이 사용)을 만들고 확인했지만(`runs/20261122-035903-export-onnx`, `-035949-export-onnx`), **사용자 결정으로 클라우드는 M2만 유지**했습니다. M1·M3 파일은 저장소에서 지웠습니다(`runs/20261122-040210-cleanup-extra-models`). 모델 비교는 로컬 서버에서 합니다.

## 7. 운영 중 장애: 판정 실패 (2026-11-22 04:03~04:13 KST, 약 10분)

M2만 남기면서 M2를 `.ort` 형식으로 다시 올리고 배포한 뒤, 프로덕션의 판정 요청이 실패했습니다. 계획한 오류 재현(`docs/INCIDENT.md`)과 별개로 실제로 생긴 장애입니다.

| 시각 (KST) | 내용 | 기록 |
|---|---|---|
| 04:01~04:03 | M2를 `.ort` 형식으로 다시 올리고(이전 `model.onnx` 삭제) 배포. 새 배포의 `/health` 200이라 승격됨 | `runs/20261122-040109-upload-model`, `runs/20261122-040255-deploy-m2only` |
| 04:04 | 스모크 2/14: `/health`, `/v1/meta` 외 판정 요청이 500·503. 런타임 로그: `No usable temporary directory found in ['/tmp', ...]` | `runs/20261122-040428-smoke-cloud-m2only` (failed), `runs/20261122-040255-deploy-m2only/runtime_logs_fetched_later.jsonl` |
| 원인 1 | 함수가 처음 뜰 때 의존성 약 350MB를 `/tmp`에 설치하는데, 서빙에 쓰지 않는 scikit-learn·scipy·pandas까지 들어가 `/tmp`가 가득 참 | 위 로그의 `Installing runtime dependencies...` |
| 04:06 | 수정 1: Vercel 의존성(`pyproject.toml`)을 서빙용 32개로 줄임 → 재배포 | `runs/20261122-040641-deploy-m2only` |
| 04:07 | 스모크 8/14: 모델은 매번 불러오지만 판정 요청에서 500. 판정 중 프로세스가 종료되고 새 인스턴스가 다시 모델을 받는 것이 반복됨 | `runs/20261122-040751-smoke-cloud-m2only` (failed), `runs/20261122-040641-deploy-m2only/runtime_logs_fetched_later.jsonl` |
| 원인 2 | `.ort` 형식(바이트를 가중치로 직접 쓰는 설정)이 로컬 맥에서는 동작했지만 Vercel 함수에서는 판정 중 종료됨. 정확한 원인은 확인하지 못함 | — |
| 04:11 | 수정 2: 검증된 `model.onnx` 방식으로 되돌려 다시 올리고 재배포 | `runs/20261122-041030-upload-model`, `runs/20261122-041150-deploy-restore` |
| 04:13 | 스모크 14/14, 복구 | `runs/20261122-041302-smoke-cloud-restore` |
| 04:14 | 재발 방지: 배포 스크립트가 `/health`만이 아니라 새 배포에서 **판정 1건**이 되는지 확인한 뒤 승격 → 재배포, 스모크 14/14 | `runs/20261122-041403-deploy-main`, `runs/20261122-041509-smoke-cloud-main` |

- **영향:** 약 10분 동안 프로덕션 도메인의 판정 요청(분류, 필터, 리포트)이 실패했습니다. `/health`는 계속 200이었습니다.
- **왜 막지 못했나:** 배포 전 검사(경로·파일)와 `/health`(모델 로드)는 통과했고, 판정을 실제로 해 보는 단계가 없었습니다. 이전 정상 배포로 되돌리는 것도 불가능했습니다. 이전 배포가 쓰던 `model.onnx`를 저장소에서 먼저 지웠기 때문입니다.
- **교훈:** 승격 전에 실제 판정을 확인합니다(반영함). 모델 파일 형식을 바꿀 때는 이전 파일을 지우기 전에 새 배포를 확인합니다.

## 8. Blob 저장소 정지와 모델 저장 위치 변경 (2026-11-22 12:55 KST 확인)

새 폴더 재실행 중 클라우드 스모크가 실패해서 확인해 보니 프로덕션이 모든 요청에 503을 반환하고 있었습니다.

| 항목 | 내용 | 기록 |
|---|---|---|
| 증상 | `/health`와 판정 요청 모두 `{"error": "model_load_failed"}` 503. 런타임 로그: `serving.json 읽기 실패: BlobAccessError: Vercel Blob: Blob request failed with HTTP 403` | 정리 기록의 저장소 상태 |
| 원인 | Blob 저장소가 **Suspended(정지)**. Hobby는 한도를 넘으면 과금 없이 Blob을 30일 동안 쓸 수 없게 함. 이날 배포 시도, M3 전환, 7장 장애 중 재시작으로 함수 인스턴스가 새로 뜰 때마다 모델(434~509MB)을 받아 전송 한도(월 10GB)를 넘은 것으로 봄. 정확한 사용량은 대시보드에서만 볼 수 있어 기록하지 못함 | `runs/20261122-130523-cleanup-keep/cleanup_status.txt` (`● Suspended`) |
| 결정 | 모델을 이 저장소의 **GitHub Release**에 올리고 함수가 공개 HTTPS 주소에서 받게 함(사용자 결정). 전송량 한도와 과금이 없고 토큰도 필요 없음. 가중치가 공개되므로 출처와 CC BY-SA 4.0 조건을 Release 설명에 적음 | 명세서 6.2장 |
| 코드 | `MODEL_URI=https://.../releases/download/model-<version>/`를 지원(`_HttpStore`), 배포 전 검사·해시 확인·메모리로 받기는 Blob과 같은 코드 경로. 테스트 추가(로컬 HTTP 서버) | `tests/test_cloud.py::test_https_store_loads_and_checks` |
| 복구 | Release 생성 → 배포 전 검사 통과 → 새 배포 `/health`·판정 확인 → 승격 → 스모크 **14/14** | `runs/20261122-130049-upload-model`, `-130212-deploy-release`, `-130351-smoke-cloud-release` |
| 정리 | 정지된 Blob 저장소는 쓰지 않음. 삭제는 Vercel CLI가 사람의 확인을 요구해서 자동 정리에서는 건너뜀(첫 시도는 실패로 기록) | `runs/20261122-130444-cleanup-keep` (failed), `-130523-cleanup-keep` |

- **멈춘 시간:** 정지된 시점은 기록이 없어 정확히 모릅니다. 13:03에 복구했습니다.
- **교훈:** 함수가 뜰 때마다 큰 모델을 받는 구조는 무료 저장소의 전송 한도를 빨리 씁니다. 전송 한도가 없는 곳(GitHub Release)에 두거나, 재시작이 반복되는 상황(7장)을 먼저 막아야 합니다.

## 9. 확인하지 못한 것

- 동시 요청이 많을 때의 동작(인스턴스 수, 지연)은 재지 않았습니다.
- 메모리 사용량은 Vercel이 함수별로 보여 주는 값을 따로 기록하지 않았습니다.
- 프로덕션 도메인은 공개라서, 남용 방지는 입력 크기 제한과 Hobby 한도에 의존합니다. 요청 속도 제한(rate limit)은 설정하지 않았습니다.
