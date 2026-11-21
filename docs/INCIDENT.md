# 오류 재현과 복구 기록

과제 "라. 실행 기록과 오류 복구" 항목의 기록입니다. 계획은 `docs/PROJECT_SPEC.md` 8.1장에 있고, 원본 기록은 `runs/`에 있습니다. Blob 저장소 ID는 `*`로 가렸습니다.

## 오류 1: 잘못된 모델 경로로 배포

| 항목 | 내용 |
|---|---|
| 발생 시각 | 2026-11-22 03:33:57 KST |
| 환경 | Vercel (Hobby), 리전 `icn1`, 실패한 배포 `dpl_27nomzKHcWhPaW8xjdvh3h4JDnpq`, git 커밋 `e554cbd` |
| 재현 방법 | 존재하지 않는 버전 경로를 `MODEL_URI`로 지정해 배포 (재현을 위해 M4 배포 스크립트에는 배포 전 경로 검사를 일부러 넣지 않았음) |
| 실패한 명령 | `RUN_SUFFIX=incident MODEL_URI=blob://models/m2-20261110-WRONG/ make deploy` |
| 기록 | `runs/20261122-033357-deploy-incident/` (`status: failed`, `output.txt`, `runtime_logs.jsonl`) |

### 오류 메시지

배포 스크립트 (새 배포의 `/health` 응답, 3회 모두 같음):
```
[deploy] /health 시도 1: HTTP 503 {"status":"unavailable","error":"model_load_failed","model_version":null}
[deploy] 새 배포의 /health 실패 → 승격하지 않음. 프로덕션 도메인은 이전 배포를 계속 서비스합니다
[runlog] deploy failed: command exited with code 1
```

함수 런타임 로그 (`model_load_failed` 10건):
```
{"severity": "ERROR", "event": "model_load_failed", "model_uri": "blob://models/m2-20261110-WRONG/",
 "error": "blob://models/m2-20261110-WRONG/serving.json 읽기 실패: BlobNotFoundError: Vercel Blob: Blob request failed with HTTP 404. 경로를 확인하세요"}
```

### 영향

- **안 된 것:** 새 배포는 모델을 불러오지 못해 모든 요청에 503을 반환했습니다. 배포 명령은 0이 아닌 코드로 끝났고, `make`가 그 자리에서 멈췄습니다.
- **계속 된 것:** 새 배포를 프로덕션으로 승격하지 않았으므로 프로덕션 도메인은 이전 정상 배포가 계속 서비스했습니다. 오류 직후 프로덕션 도메인에 스모크를 실행해 **14/14 통과**를 확인했습니다(`runs/20261122-033520-smoke-cloud-incident-during/`).
- **남은 문제:** 망가진 배포가 Vercel에 하나 만들어졌고, 빌드 시간과 배포 1개를 썼습니다. 문제는 배포를 만든 **뒤**에야 드러났습니다.

### 원인

- 런타임 로그에서 `serving.json`을 읽을 때 Blob이 404를 반환한 것을 확인했습니다.
- 저장소 목록을 조회해 `models/` 아래에는 `m2-20261110-314b2b2/`만 있고 `m2-20261110-WRONG/`은 없음을 확인했습니다(`runs/20261122-033614-investigate-incident/`). 첫 조회 명령은 CLI가 목록을 표준 오류로 출력해서 검사가 실패했고, 그대로 실패 기록으로 남겼습니다(`runs/20261122-033557-investigate-incident/`).
- 결론: 배포 스크립트가 `MODEL_URI`가 실제로 있는지 확인하지 않고 배포를 만들었습니다. 함수는 첫 요청에서야 모델을 읽으므로, 경로 오류가 배포 뒤에 드러났습니다.

### 수정 (커밋 `976ab1e`)

- `model_store.check_blob_uri()`를 추가했습니다. 배포 전에 `serving.json`을 읽고, 거기 적힌 파일(`model_card.json`, `model.onnx`, 토크나이저)이 저장소에 모두 있고 크기가 맞는지 메타데이터만 조회해서 확인합니다.
- `scripts/deploy_vercel.sh`가 이 검사를 통과해야만 `vercel deploy`를 실행합니다. 실패하면 새 배포를 만들지 않고 0이 아닌 코드로 끝납니다.
- 테스트를 추가했습니다: 잘못된 경로, 파일 누락을 검사가 막는지(`tests/test_cloud.py::test_predeploy_check_blocks_wrong_path_and_missing_file`). 전체 75개 통과.

### 재검증

| 확인 | 명령 | 결과 | 기록 |
|---|---|---|---|
| 같은 잘못된 경로로 다시 시도 | `RUN_SUFFIX=incident-recheck MODEL_URI=blob://models/m2-20261110-WRONG/ make deploy` | 배포 전 검사에서 `serving.json 없음: BlobNotFoundError ... HTTP 404`로 멈춤, **새 배포를 만들지 않음** | `runs/20261122-033719-deploy-incident-recheck/` (failed) |
| 새 배포가 안 생겼는지 | `vercel ls comment-guard` | 배포 수 10개, 검사 전과 같음 | `runs/20261122-033730-investigate-incident-recheck/` |
| 올바른 경로로 다시 배포 | `RUN_SUFFIX=incident-fixed make deploy` | 배포 전 검사 통과 → 새 배포 `dpl_9zhhA5M93NrU93zVXN7GTaQ7Mz77` `/health` 200 → 프로덕션 승격 | `runs/20261122-033754-deploy-incident-fixed/` |
| 정상 입력 재실행 | `RUN_SUFFIX=incident-fixed make smoke` | **14/14 통과** (T1~T11, 필터, 메타, 로그 원문 검사) | `runs/20261122-033916-smoke-cloud-incident-fixed/` |

### 재발 방지

- 배포 전 검사로 경로 오류는 배포가 만들어지기 전에 막힙니다.
- 검사를 통과해도 새 배포의 `/health`가 200이어야만 승격하는 단계는 그대로 둡니다(모델 파일이 손상된 경우 등은 해시 확인과 `/health`가 막음).
- 실패한 실행은 모두 `status: failed`로 남기고, 실패하면 다음 단계(승격, 스모크)로 넘어가지 않습니다.

## 참고: 배포 준비 중 만난 오류 (계획한 재현과 별개)

M4 배포 과정에서 실제로 세 번 실패했고, 모두 새 배포의 `/health` 확인 단계에서 걸려 프로덕션에는 영향이 없었습니다. 원인과 수정은 `docs/CLOUD_RUNS.md` 3장에 있습니다: 의존성 미설치(Vercel이 `pyproject.toml`을 읽음), 함수 `/tmp` 공간 부족(모델을 메모리로 받도록 수정), 프로덕션 주소 탐색 실패.
