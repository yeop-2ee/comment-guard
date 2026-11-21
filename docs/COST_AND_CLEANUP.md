# 권한, 비용, 자원 정리

과제 "마. 재실행과 자원 정리" 중 권한·비용·정리 항목의 기록입니다. 계획은 `docs/PROJECT_SPEC.md` 10장에 있습니다.

## 1. 접근 권한과 대상

| 주체 | 권한 | 대상 | 비고 |
|---|---|---|---|
| 본인 계정 (`vercel login`, 브라우저 인증) | 배포·승격, 환경 변수, Blob 저장소 생성·업로드·삭제, 로그 조회 | Hobby 팀 `MLOps`의 프로젝트 `comment-guard`와 저장소 `comment-guard-models` | 장기 토큰을 직접 만들지 않음. 공용 자원과 조직 보안 설정은 바꾸지 않음 |
| 함수 (런타임) | Blob 저장소 읽기·쓰기가 가능한 토큰 `BLOB_READ_WRITE_TOKEN` | 이 저장소 하나 | 저장소를 프로젝트에 연결할 때 Vercel이 자동으로 만든 환경 변수. **코드는 읽기만 하지만 토큰 자체는 쓰기도 가능합니다.** Hobby에서 읽기 전용 토큰을 따로 두는 방법은 쓰지 않았고, 한계로 남깁니다 |
| 로컬 스크립트 | 단기 OIDC 토큰(약 12시간)과 저장소 ID로 Blob 업로드·조회 | 이 저장소 하나 | `vercel env pull`이 `.env.local`(Git 제외)에 받음. 부분 정리 때 `.env.local`을 지움 |
| 배포 스크립트의 `vercel curl` | 보호된 새 배포 주소의 `/health` 요청 | 이 프로젝트 | 처음 실행할 때 프로젝트에 "Protection Bypass for Automation" 비밀값이 생김. Vercel 프로젝트 설정에만 있고 저장소·기록에는 없음 |
| 호출자 | 없음 (공개) | 프로덕션 도메인 `https://comment-guard-jade.vercel.app` | 개별 배포 주소는 Deployment Protection으로 Vercel 로그인 필요 |

- 토큰 값은 코드, 기록, 커밋 어디에도 없습니다. 기록에서는 Blob 저장소 ID와 Vercel 사용자 이름을 `*`로 가렸습니다.
- `configs/deploy.env`, `.env.local`, `.vercel/`, `.state/`는 Git에서 뺍니다(`.gitignore`).

## 2. 비용이 생기는 자원과 제한 방법

모두 **Hobby(무료) 플랜**이고 결제 수단을 등록하지 않았습니다. Hobby는 한도를 넘으면 과금 없이 기능이 멈춥니다. 그래서 이 과제에서 실제로 청구될 수 있는 비용은 없습니다.

| 자원 | 비용 요인 | Hobby 한도와 제한 방법 |
|---|---|---|
| Vercel Functions | 실행 시간(Active CPU), 메모리 시간, 호출 수 | 요청이 없으면 0. 최대 실행 시간 60초(`vercel.json`). 입력은 한 번에 100건, 댓글당 1,000자로 제한 |
| Vercel Blob | 저장 용량, 전송량, 작업 수 | 저장 1GB(사용 416.64MB, M2 한 버전만 보관. M1·M3를 함께 올렸을 때 약 950MB), 전송 월 10GB. 함수 인스턴스가 새로 뜰 때마다 모델 약 434MB를 받으므로 cold start가 잦으면 전송 한도를 먼저 씀 |
| 빌드 | 빌드 시간 | 배포 1번에 빌드 약 40초. 이 과제에서 배포 11개를 만듦(시작·판정에 실패한 배포 5개 포함, 배포 전 검사로 막힌 1번은 배포가 만들어지지 않음). 남은 것은 1개 |
| 런타임 로그 | — | 1시간 보존, 추가 비용 없음. 필요한 로그는 받은 즉시 `runs/`에 저장 |

## 3. 정리 실행과 정리 후 상태

부분 정리(`make cleanup-keep`)를 두 번 실행했습니다. 첫 번째(`runs/20261122-034232-cleanup-keep/`)는 M5까지의 지난 배포 5개를 지웠고, 이 문서의 상태는 마지막 실행 기준입니다.

`make cleanup-keep` → `runs/20261122-041711-cleanup-keep/` (`cleanup_status.txt`)

| 대상 | 처리 |
|---|---|
| 첫 정리 뒤 만든 지난 배포 5개 (M1·M3 시도, 판정이 실패한 배포 2개, 복구 배포 포함) | **삭제** |
| M1·M3 모델 파일 8개 (화면 모델 선택 시도용) | **삭제** (`runs/20261122-040210-cleanup-extra-models/`) |
| 프로덕션 배포 `dpl_9MuBmdTdMVgmmGBQLgyT7UaEPhwF` | 남김 |
| Blob 저장소 `comment-guard-models` (M2 한 버전, 파일 5개, 416.64MB) | 남김 |
| Vercel 프로젝트 `comment-guard` | 남김 |
| 로컬 `.env.local` (개발용 토큰) | **삭제** |

정리 후 다시 조회한 상태 (`cleanup_status.txt` 요약):

- `vercel ls comment-guard`: 배포 1개(프로덕션, Ready)만 남음
- `vercel blob list-stores`: `comment-guard-models` Active, icn1, 416.64MB, 파일 5개
- `vercel blob list`: `models/m2-20261110-314b2b2/` 아래 5개 파일만 있음 (`model.onnx`, `model_card.json`, `serving.json`, 토크나이저 2개)
- 프로덕션 도메인 `/health`: `{"status":"ok","model_version":"m2-20261110-314b2b2"}` HTTP 200

## 4. 남긴 자원과 정리 계획

| 남긴 자원 | 이유 | 정리 계획 |
|---|---|---|
| 프로덕션 배포, 프로젝트 | 채점 기간 동안 화면과 API를 직접 열어 확인할 수 있게 하려고 남김 | 채점이 끝나면 `make cleanup`(`scripts/cleanup.sh --all`)을 실행해 프로젝트를 지우고 상태를 다시 조회해 기록 |
| Blob 저장소와 모델 | 프로덕션 배포가 이 모델을 읽음 | 위와 같이 `make cleanup`에서 모델 삭제 후 저장소 삭제 |
| 실행 기록 (`runs/`), 로컬 모델 (`models/`) | 과제 증빙과 재현용. 로컬 모델은 Git에 올리지 않음 | 보관 |

- 전체 삭제 모드(`--all`)는 아직 실행하지 않았습니다. 실행한 것은 부분 정리(`--keep-production`)입니다.
- Hobby는 과금이 없으므로 남겨 둔 동안 생기는 위험은 한도 초과로 서비스가 멈추는 것뿐입니다.
