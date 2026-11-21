# Comment Guard

> 한국어 댓글의 악플 정도를 판정하는 필터링·증거 정리 API (클라우드 MLOps 대체 실습과제)

한국어 댓글을 **정상 / 공격 / 혐오**로 분류하고 0~100점 **심각도 점수**를 매깁니다. 이 점수로 보기 싫은 댓글을 민감도에 맞춰 가려 주고, 악플 피해자가 직접 모은 댓글을 심각도 순 리포트로 정리해 줍니다. 공개 데이터로 직접 학습한 모델(KcELECTRA 미세조정)을 FastAPI로 만들어 **Vercel**에서 실행합니다.

**배포 주소: https://comment-guard-jade.vercel.app** (웹 화면, API 문서는 `/docs`)

> 채점 기간 동안 켜 둡니다. 채점이 끝나면 `make cleanup`으로 지웁니다([권한, 비용, 자원 정리](#권한-비용-자원-정리)). 오래 쓰지 않은 뒤 첫 요청은 모델을 받느라 25~30초 걸립니다.

---

## 과제 항목별 진행 상태

| 과제 항목 | 상태 | 근거 |
|------|------|------|
| **가. 데이터와 모델 준비** | 실행함 | [데이터](#데이터), [모델](#모델), `docs/DATA.md` |
| **나. 클라우드 실행** | 실행함 (Vercel, 서울 리전, 클라우드 스모크 14/14) | [클라우드 실행 기록](#클라우드-실행-기록), `docs/CLOUD_RUNS.md` |
| **다. 결과 평가** | 실행함 (eval 471건, 로컬·클라우드 정상 동작 검사) | [평가 결과](#평가-결과), `docs/EVALUATION.md` |
| **라. 실행 기록** | 실행함 (모든 단계 `runs/`에 기록, 실패도 `failed`로 구분) | [실행 기록과 버전 관리](#실행-기록과-버전-관리) |
| **라. 오류 재현과 복구** | 실행함 (잘못된 모델 경로 → 배포 전 검사 추가 → 재검증) | [오류와 복구](#오류와-복구), `docs/INCIDENT.md` |
| **마. 재실행** | 실행함 (새 폴더에서 README 명령만으로) | [재실행 결과](#재실행-결과), `docs/RERUN.md` |
| **마. 권한·비용·자원 정리** | 실행함 (부분 정리 후 상태 재조회, 프로덕션은 채점 기간 동안 유지) | [권한, 비용, 자원 정리](#권한-비용-자원-정리), `docs/COST_AND_CLEANUP.md` |

---

## 활용할 수 있는 곳

댓글이나 채팅을 받는 서비스가 API를 불러 욕설과 악플을 가리는 용도로 붙여 쓸 수 있습니다.

| 활용처 | 쓰는 방법 | 확인이 필요한 점 |
|------|------|------|
| **뉴스·기사 댓글** | 댓글 목록을 `/v1/filter`로 보내 민감도별로 숨기거나 흐리게 표시 | 학습 데이터(연예 뉴스 댓글)와 가장 비슷한 영역 |
| **유튜브·커뮤니티 댓글** | 운영자가 댓글을 모아 `/v1/classify`로 점수를 받고, 심각 등급만 검토 대기열로 보냄 | 댓글 수집은 각 플랫폼의 공식 API와 약관 안에서 운영자가 직접 해야 함 (이 서비스는 수집하지 않음) |
| **게임·라이브 채팅** | 채팅 메시지를 보내기 전에 점수를 받아 심각 등급을 가림 | 채팅은 짧고 신조어·은어가 많아 학습 데이터와 말투가 다름. 붙이기 전에 그 서비스의 채팅으로 따로 평가해야 함 |
| **악플 피해자 지원** | 피해자가 직접 모은 댓글을 `/v1/evidence/report`로 심각도 순 정리 | 법률 판단이 아님 |

- 한 번에 최대 100건, 댓글당 1,000자까지 받습니다. 클라우드에서 1건은 수백 ms, 100건은 약 10~14초 걸렸습니다(`docs/CLOUD_RUNS.md` 4장).
- 어느 활용처든 자동 판정만으로 지우기보다는, 사람이 검토하거나 이용자가 민감도를 고르는 방식을 권장합니다.

---

## 시스템 구성도

```
[브라우저 / 클라이언트] ── HTTPS ─▶ [Vercel 프로덕션 도메인] ─▶ [현재 배포: Python 함수 (FastAPI), icn1]
                                                                 ├─ 첫 요청에서 로드 ─▶ [Vercel Blob (비공개): models/<version>/]
                                                                 └─ 요청 로그(원문 제외) ─▶ [Vercel 런타임 로그]
[로컬] 데이터 → 검증 → 학습 → 평가 → ONNX 변환 → 테스트 → 모델 업로드
       → 배포 전 경로 검사 → 배포 → /health·판정 확인 → 프로덕션 승격 → 스모크
        └─ 모든 단계 기록 ─▶ runs/<run_id>/run.json
```

| 항목 | 선택 | 이유 |
|------|------|------|
| **실행** | Vercel Functions (Python 3.12, Hobby 플랜) | 요청이 없으면 비용이 없음. 같은 FastAPI 앱을 로컬 `uvicorn`과 Vercel에서 그대로 실행하고, 화면과 API를 같은 주소로 제공 |
| **리전** | `icn1` (서울), 모델 저장소도 같은 리전 | 사용자가 한국에 있음 |
| **모델 저장** | Vercel Blob 비공개 저장소, 파일 목록과 sha256을 적은 `serving.json` | 모델(434MB)이 CLI 배포 업로드 한도(100MB)보다 커서 코드와 분리. 받은 파일은 해시로 확인 |
| **모델 실행 형식** | ONNX (onnxruntime) | PyTorch는 Python 함수 크기 한도(500MB)를 넘음. 변환 결과가 PyTorch와 같은지 valid 790건으로 확인 |
| **버전·롤백** | 배포 전 경로 검사 → 새 배포의 `/health`와 판정 1건 확인 → `vercel promote` | 새 배포가 실패하면 승격하지 않아 이전 배포가 계속 서비스 |

선택 이유의 전체 표는 `docs/PROJECT_SPEC.md` 6장에 있습니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **댓글 분류** | 정상 / 공격 / 혐오 라벨과 클래스별 확률, 편향 표시 (`POST /v1/classify`, 최대 100건) |
| **심각도 점수와 등급** | `round(100 × (P(혐오) + 0.5 × P(공격)))`, 일반 / 주의 / 심각 3등급 |
| **민감도별 필터링** | 낮음·보통·높음에 따라 보이기 / 흐리게 / 숨기기. 삭제가 아니라 숨김 (`POST /v1/filter`) |
| **증거 정리 리포트** | 직접 모은 댓글을 심각도 순으로 정리, 원문·리포트 SHA-256, 누락 정보 경고, JSON·CSV (`POST /v1/evidence/report`, 최대 500건) |
| **웹 화면** | 댓글 필터, 증거 리포트, 판정 상세, 모델 성능 탭. 외부 CDN 없이 파일 하나 (`GET /`) |
| **모델 선택 비교** | 로컬 서버에서 모델(M1, M2, M3, 앙상블)을 골라 같은 댓글의 판정을 비교 (`GET /v1/models`, `?model=`). 클라우드는 메모리 한도 때문에 M2만 |
| **원문을 남기지 않는 로그** | 건수, 해시, 처리 시간, 모델 버전만 기록. 테스트와 클라우드 런타임 로그 검사로 확인 |
| **실행 기록** | 모든 단계가 시각, 성공 여부, git 커밋, 모델 버전, 입력 해시를 `runs/`에 기록 |

하지 않는 것: 법률 판단, 댓글 크롤링, 작성자 신원 추적, 원본 증거 대체. 기능과 입력 규칙의 전체 명세는 `docs/PROJECT_SPEC.md` 2장에 있습니다.

---

## 서비스 흐름

```
댓글 입력 (1건 ~ 100건, 리포트는 500건)
        ↓
전처리 (NFC 정규화·공백 정리, 학습과 서빙이 같은 함수 사용)
        ↓
모델 판정 (서빙 모델 M2: KcELECTRA-base 미세조정, 클라우드는 ONNX)
  ├─ 정상 / 공격 / 혐오 확률
  └─ 편향 유형 (없음 / 집단 / 성별)
        ↓
심각도 점수 (0~100) → 등급 (일반 / 주의 / 심각)
        ↓
기능별 결과
  ├─ 필터링: 민감도에 따라 보이기 / 흐리게 / 숨기기
  └─ 증거 리포트: 심각도 순 정렬 + 해시 + 체크리스트 + 고지 문구 (JSON · CSV)
```

### 파이프라인

```
make data          원본 다운로드 + 해시 검증
make validate      잘못된 값 검사 + 학습/검증/평가 분할
make train         B0·M1 학습 (M2는 python -m commentguard.m2)
make evaluate      eval 471건으로 평가
make test          pytest 77개
make export-onnx   M2 → ONNX, valid로 PyTorch와 같은지 확인
make vercel-setup → upload-model → deploy → smoke → cleanup-keep / cleanup
```

---

## 기술 스택

[![Python](https://img.shields.io/badge/Python-3.11_|_3.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.32-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Vercel](https://img.shields.io/badge/Vercel-Python_Functions-000000?logo=vercel&logoColor=white)](https://vercel.com/docs/functions/runtimes/python)

| 기술 | 버전 | 용도 |
|------|------|------|
| [FastAPI](https://fastapi.tiangolo.com) | 0.162.2 | API 서버 + 웹 화면 서빙 |
| [Uvicorn](https://www.uvicorn.org) | 0.54.0 | 로컬 ASGI 서버 |
| [scikit-learn](https://scikit-learn.org) | 1.9.1 | M1 (TF-IDF + 로지스틱 회귀), 지표 계산 |
| [PyTorch](https://pytorch.org) / [Transformers](https://huggingface.co/docs/transformers) | 2.15.0 / 5.22.2 | M2·M3 미세조정 (로컬) |
| [ONNX Runtime](https://onnxruntime.ai) / [tokenizers](https://github.com/huggingface/tokenizers) | 1.32.0 / 0.23.3 | 클라우드 추론 (torch 없이) |
| [onnx](https://onnx.ai) | 1.23.2 | M2 → ONNX 변환 (로컬) |
| [Vercel Python 런타임](https://vercel.com/docs/functions/runtimes/python) | Python 3.12 | FastAPI 앱을 Vercel 함수로 실행 |
| [Vercel Blob](https://vercel.com/docs/vercel-blob) / vercel-blob | — / 0.6.2 | 모델 저장(비공개)과 읽기 |
| [Vercel CLI](https://vercel.com/docs/cli) | 66.6.0 (`npx`로 고정) | 배포, 업로드, 로그, 정리 |
| [pytest](https://docs.pytest.org) | 9.2.0 | 정상 동작 검사 |

모든 Python 의존성은 `requirements.txt`(공용·서빙), `requirements-dev.txt`(테스트), `requirements-m2.txt`(학습·ONNX 변환)에 버전을 고정했습니다. Vercel은 `pyproject.toml`의 서빙용 목록(같은 버전)으로 설치합니다.

---

## 프로젝트 구조

```
comment-guard/
│
├── app.py                   # Vercel 진입점 (commentguard.api:app)
├── vercel.json              # Vercel 함수 설정 (리전, 최대 실행 시간, 함수에서 뺄 폴더)
├── .vercelignore            # 배포 업로드에서 뺄 파일
├── src/commentguard/
│   ├── text.py              # normalize() — 학습·서빙 공용 전처리
│   ├── data.py              # 잘못된 값 검사, 층화 분할
│   ├── train.py             # B0, M1, M1-bias 학습
│   ├── m2.py                # 사전학습 모델 미세조정과 로더 (M2, M3)
│   ├── onnx_export.py       # M2 → ONNX 변환, PyTorch와 동등성 확인
│   ├── onnx_model.py        # ONNX 서빙판 (torch 없이)
│   ├── ensemble.py          # 확률 평균 앙상블
│   ├── evaluate.py          # 지표, 부트스트랩 구간, 혼동행렬, 필터 지표 (eval을 읽는 유일한 모듈)
│   ├── compare.py           # 모델 비교, 서빙 모델 선택 규칙
│   ├── probe.py             # 행동 검사 세트 평가
│   ├── severity.py          # 심각도 점수, 등급, 민감도 규칙, 기준값 조정
│   ├── report.py            # 증거 리포트 (JSON·CSV, 해시)
│   ├── model_store.py       # 로컬·Blob 모델 로드, 배포 전 경로 검사
│   ├── api.py               # FastAPI 앱
│   ├── runlog.py            # 실행 기록 공통 함수
│   └── static/index.html    # 웹 화면
│
├── scripts/
│   ├── download_data.sh         # 데이터 다운로드 + 해시 검증
│   ├── vercel_env.sh            # deploy.env 읽기, 로그인 확인
│   ├── vercel_setup.sh          # 프로젝트 연결·Blob 저장소 준비
│   ├── upload_model.sh          # 모델 업로드 (dirty 모델, 동등성 확인 안 된 모델은 거부)
│   ├── deploy_vercel.sh         # 배포 전 경로 검사, 배포, /health·판정 확인 후 승격
│   ├── smoke_cloud.py           # 로컬·클라우드 공용 스모크, 런타임 로그 원문 검사
│   ├── cleanup.sh               # 자원 정리와 상태 재조회
│   └── build_perf_snapshot.py   # 웹 화면 성능 탭 수치를 runs/ 기록에서 생성
│
├── configs/                 # train.yaml, severity.yaml, serving.yaml, deploy.env.example
├── data/                    # raw/ (원본 + MANIFEST.sha256), processed/, probes/
├── tests/                   # pytest 77개
├── runs/                    # 실행 기록 (run.json, 지표, 모델 카드, 스모크, 정리 상태)
├── models/                  # 로컬 모델 (Git 제외)
├── docs/
│   ├── PROJECT_SPEC.md      # 구현 기준 문서
│   ├── DATA.md              # 데이터 설명
│   ├── EVALUATION.md        # 평가 결과
│   ├── CLOUD_RUNS.md        # 클라우드 실행 기록
│   ├── INCIDENT.md          # 오류 재현과 복구
│   ├── RERUN.md             # 새 폴더 재실행
│   └── COST_AND_CLEANUP.md  # 권한, 비용, 정리 후 상태
│
├── Makefile                 # 파이프라인 진입점
├── pyproject.toml           # 패키지 정보, Vercel 함수 의존성
└── requirements*.txt        # 버전 고정
```

---

## 시작하기

### 사전 요구사항

| 항목 | 비고 |
|------|------|
| Python | 3.11 (로컬). Vercel 함수는 3.12 |
| make, git | - |
| Apple Silicon(MPS) 또는 GPU | 선택 — M2 학습이 빨라짐 (맥 MPS에서 약 24분) |
| Node.js 20+ | 클라우드 단계에만 필요 (`npx`로 Vercel CLI 66.6.0 실행) |
| Vercel 계정 (Hobby) | 클라우드 단계에만 필요. 결제 수단 없이 가능 |

### 1. 저장소 클론

```bash
git clone https://github.com/yeop-2ee/comment-guard.git
cd comment-guard
```

### 2. 의존성 설치

```bash
make setup          # 가상환경(.venv) 생성과 의존성 설치
```

### 3. 데이터 검증부터 M1 평가까지

```bash
make all-local      # 데이터 해시 검증 → 검사·분할 → M1 학습 → M1 평가 → pytest
```

### 4. 서빙 모델(M2) 학습과 ONNX 변환

모델 가중치는 저장소에 없습니다. 서빙 모델이 M2라서(`configs/serving.yaml`) 서버를 띄우기 전에 학습해야 합니다.

```bash
export PYTHONPATH=src
.venv/bin/python -m commentguard.m2                                        # → models/LATEST_M2
.venv/bin/python -m commentguard.evaluate models/$(cat models/LATEST_M2)   # eval로 평가
make export-onnx                                                           # 클라우드용 ONNX + 동등성 확인
```

M2를 학습하지 않고 M1으로 띄우려면 5번에서 `MODEL_URI=models/$(cat models/LATEST) make serve`를 씁니다.

### 5. 로컬 서버

```bash
make serve          # uvicorn, 포트 8080 (models/LATEST*의 모델을 모두 고를 수 있음)
make serve-onnx     # 클라우드와 같은 ONNX 경로로 실행 (둘 중 하나만)
make smoke-local    # 다른 터미널에서. 13개 케이스 → runs/*/local_smoke.json
```

- 웹 화면: http://127.0.0.1:8080
- API 문서 (Swagger): http://127.0.0.1:8080/docs

### 6. 클라우드 (Vercel)

```bash
npx -y vercel@66.6.0 login                        # 브라우저 인증 (토큰을 만들지 않음)
cp configs/deploy.env.example configs/deploy.env  # 프로젝트·저장소 이름 (Git 제외)
make vercel-setup   # 프로젝트 연결, Blob 비공개 저장소 생성 (저장소 ID는 deploy.env에만)
make upload-model   # 서빙 모델을 Blob 저장소에 업로드
make deploy         # 테스트 → 배포 전 경로 검사 → 배포 → /health·판정 확인 → 프로덕션 승격
make smoke          # 프로덕션 도메인 스모크 + 런타임 로그 원문 검사 → runs/*/cloud_smoke.json
make cleanup-keep   # 지난 배포만 삭제하고 상태 재조회 (프로덕션 유지)
make cleanup        # Blob 저장소와 프로젝트까지 모두 삭제하고 상태 재조회
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 웹 화면 |
| GET | `/health` | 상태와 모델 버전. 모델 로드 실패 시 503 |
| GET | `/v1/meta` | 모델 버전, 학습 데이터 해시, 기준값, 평가 요약 |
| GET | `/v1/models` | 고를 수 있는 모델 목록 |
| POST | `/v1/classify` | 댓글 분류 (최대 100건) |
| POST | `/v1/filter` | 민감도별 필터링 |
| POST | `/v1/evidence/report` | 증거 정리 리포트 (`?format=json\|csv`) |

`/v1/classify`, `/v1/filter`, `/v1/evidence/report`, `/v1/meta`는 `?model=<모델 ID 또는 버전>`을 받습니다. 빈 문자열, 1,000자 초과, 건수 초과는 422로 거부하고 입력값을 응답에 되돌려주지 않습니다.

```bash
curl -s -X POST https://comment-guard-jade.vercel.app/v1/filter -H 'content-type: application/json' \
  -d '{"sensitivity":"medium","comments":[{"id":"c1","text":"오늘 무대 좋았어요"}]}'
```

---

## 데이터

| 항목 | 내용 |
|------|------|
| **데이터** | Korean HateSpeech Dataset (BEEP!), Moon et al., 2020. 한국 연예 뉴스 댓글에 사람이 라벨을 붙인 데이터 |
| **출처** | https://github.com/kocohub/korean-hate-speech (커밋 `f8d05dc` 고정) |
| **사용 조건** | CC BY-SA 4.0. 출처를 표시하고 같은 라이선스로 재배포할 수 있어 원본과 가공본을 저장소에 포함 (`data/LICENSE-DATA.md`) |
| **한 건의 의미** | 뉴스 기사에 달린 댓글 1개와 그 라벨. 입력은 댓글 텍스트 하나뿐이고 물리 단위는 없음 |
| **정답** | `hate`(none / offensive / hate)가 주 정답, `bias`와 `contain_gender_bias`는 보조 정답. 정답 컬럼은 입력에 넣지 않음 |
| **잘못된 값 처리** | 빈 댓글, 허용값 밖 라벨, 중복, 라벨 충돌, train·dev 겹침을 검사해 제거. 이 버전의 원본에서는 제거 건수가 모두 0건 |
| **분할** | `train_fit` 7,106건(학습) / `valid` 790건(조정) / `eval` 471건(최종 평가 전용). train.tsv를 `hate` 비율로 층화 분할(`random_state=42`), eval은 제작자의 공식 dev 분할 |
| **확보** | `make data`가 고정 커밋에서 받고 `data/raw/MANIFEST.sha256`과 대조. 해시가 다르면 중단 |

eval은 학습, 어휘 사전, 하이퍼파라미터, 심각도 기준값 조정, 서빙 모델 선택, ONNX 동등성 확인 어디에도 쓰지 않았습니다. 자세한 내용과 누수 방지 체크리스트는 `docs/DATA.md`에 있습니다.

---

## 모델

| ID | 모델 | 역할 |
|------|------|------|
| **B0** | 최빈값 분류기 | 단순 기준 |
| **M1** | 글자 2~5-gram TF-IDF + 로지스틱 회귀 | 직접 학습한 기준·대체 모델 |
| **M2** | `beomi/KcELECTRA-base` 미세조정 (MIT, 리비전 `681201a` 고정) | **서빙 모델** (클라우드는 ONNX로 변환) |
| **M3** | `beomi/KcELECTRA-base-v2022` 미세조정 (MIT, 리비전 `8728557` 고정) | 비교용 (로컬) |
| **M23X6** | M2·M3 각각 시드 3개, 여섯 모델의 확률 평균 | 비교용 (로컬) |

- **버전 규칙:** `<모델ID>-<YYYYMMDD>-<git 커밋 7자리>` (예: `m2-20261110-314b2b2`). 커밋 안 된 변경이 있는 상태로 만든 모델은 업로드하지 않습니다.
- **모델 카드:** 모델마다 `model_card.json`에 버전, 커밋, 학습 데이터 해시, 하이퍼파라미터, 기준값, 지표를 기록합니다. 가중치는 저장소에 올리지 않고 카드만 `runs/<학습 run_id>/`에 둡니다.
- **ONNX 변환:** 같은 가중치를 형식만 바꾸므로 버전은 그대로입니다. valid 790건에서 토큰 ID 완전 일치, 확률 차이 최대 1.5e-5, 예측 100% 일치를 확인했습니다(`runs/*-export-onnx/onnx_check.json`).
- **서빙 모델 선택:** eval이 아니라 valid 지표와 운영 지표로만 고르는 규칙을 결과를 보기 전에 정했습니다. 후보(M3, KLUE-RoBERTa-large, 앙상블 4종)를 비교했지만 valid에서 M2를 넘지 못해 M2를 유지했습니다(`docs/EVALUATION.md` 9장, 10장).

---

## 평가 결과

평가 자료는 eval 471건(none 160 / offensive 189 / hate 122)이고, 주 지표는 3클래스 macro-F1입니다. 클래스 비율이 고르지 않아 정확도는 보조로만 씁니다.

| 항목 | B0 | M1 | **M2 (서빙)** |
|------|------|------|------|
| **eval macro-F1 [95% 구간]** | 0.169 [0.152, 0.184] | 0.569 [0.524, 0.613] | **0.694 [0.649, 0.733]** |
| **medium 정상 오차단율** (목표 ≤ 15%) | — | 8.1% | **5.0%** |
| **medium 혐오 차단율** (목표 ≥ 60%) | — | 59.8% (미달) | **66.4%** |
| **놓친 혐오** (hate → none) | — | 15건 | 1건 |

| 항목 | 내용 |
|------|------|
| **판단 기준** | 학습 전에 정함: 기준 모델보다 확실히 높을 것, medium 민감도에서 정상 오차단율 ≤ 15%, 혐오 차단율 ≥ 60%. M2는 세 가지를 모두 통과 |
| **정상 동작 검사** | pytest 77개(입력 검증, 리포트 재현성, 로그 원문 차단, Blob 로드, 배포 전 검사 등), 로컬 스모크 13개, **클라우드 스모크 14개**(13개 + 런타임 로그 원문 검사) 통과 |
| **잘못된 결과 분석** | eval에서 놓친 혐오 1건(초성 비속어가 어휘 사전에 없음), 사용 중 발견한 오판 1건(신조어 칭찬을 혐오로 판정) |
| **가장 큰 약점** | 노골적인 단어가 없는 성희롱과 비꼼 (직접 만든 행동 검사 36문장에서 성희롱 1/6, 비꼼 0/6) |
| **한계** | 연예 뉴스 댓글에 한정, eval이 471건이라 구간이 넓음, 클라우드 지연·cold start는 몇 번만 측정, 동시 요청 부하는 미측정, 확률 보정·리포트의 법적 유용성은 미측정 |

계산 방법, 혼동행렬, 오류 분석, 후보 비교의 전체 내용은 `docs/EVALUATION.md`에 있습니다.

---

## 클라우드 실행 기록

| 항목 | 내용 |
|------|------|
| **서비스·리전** | Vercel Functions (Hobby), 함수와 Blob 저장소 모두 `icn1`(서울) |
| **설정** | Python 3.12, 메모리 2GB(기본값), 최대 실행 시간 60초, 의존성은 서빙용 32개 |
| **데이터·결과 위치** | 모델 `blob://models/m2-20261110-314b2b2/`(비공개), 결과는 `runs/*-smoke-cloud-*/cloud_smoke.json`과 `runtime_logs.jsonl` |
| **실제 입력 → 결과** | 공개 데이터 댓글로 T1~T11·필터·메타 요청, 응답의 `model_version`이 서빙 모델과 같음. 마지막 스모크 14/14 (`runs/20261122-041509-smoke-cloud-main/`) |
| **지연** | 1건 수백 ms, 100건 약 10~14초, cold start 25~30초(3회 측정) |
| **배포 중 문제** | 의존성 미설치, `/tmp` 부족, 주소 탐색 실패를 고친 뒤 배포. 운영 중 판정 실패 약 10분이 있었고 복구한 뒤 승격 전 판정 확인을 추가(`docs/CLOUD_RUNS.md` 3장, 7장) |
| **모델 선택 시도** | M1·M3도 클라우드 화면에서 고르게 해 봤지만 M3 전환 때 메모리 부족 → M2만 유지 (6장) |

---

## 실행 기록과 버전 관리

| 항목 | 내용 |
|------|------|
| **단계별 실행** | `Makefile`이 진입점. 각 단계는 실패하면 0이 아닌 코드로 끝나고 Make가 그 자리에서 멈춤. `make deploy`는 `make test`가 통과해야 진행 |
| **실행 기록** | `runs/<run_id>/run.json`에 시각, 성공 여부, git 커밋, dirty 여부, 모델 버전, 입력 해시, 설정, 산출물 위치. 셸 단계는 출력도 `output.txt`로. 요약은 `runs/index.csv` |
| **실패 처리** | 실패한 실행은 `status: "failed"`로 구분하고 다음 단계로 넘어가지 않음 (예: 배포 실패 시 승격·스모크 없음) |
| **버전 연결** | 배포마다 `MODEL_URI`(모델 버전 경로)를 `--env`로 지정하고, 스모크 기록에 배포 ID·모델 경로·응답의 모델 버전을 함께 남김 |
| **외부 모델** | HuggingFace 리비전 해시로 고정 |
| **민감 정보** | Blob 저장소 ID와 Vercel 사용자 이름은 기록에서 `*`로 가림. 토큰은 기록·커밋에 없음 |

---

## 오류와 복구

재현한 오류는 **배포할 때 잘못된 모델 경로(`MODEL_URI`)를 지정**한 경우입니다(`docs/INCIDENT.md`).

| 단계 | 결과 | 기록 |
|------|------|------|
| 재현 | `MODEL_URI=blob://models/m2-20261110-WRONG/ make deploy` → 새 배포 `/health` 503, 로그 `serving.json 읽기 실패: BlobNotFoundError ... HTTP 404`, 승격 안 됨 | `runs/20261122-033357-deploy-incident` (failed) |
| 영향 확인 | 프로덕션 도메인 스모크 14/14 (기존 배포 계속 서비스) | `runs/20261122-033520-smoke-cloud-incident-during` |
| 원인 | 저장소 목록에 그 경로가 없음. 배포 스크립트가 경로를 확인하지 않고 배포를 만듦 | `runs/20261122-033614-investigate-incident` |
| 수정 | 배포 전에 `serving.json`과 모델 파일이 모두 있는지 검사 (`check_blob_uri`, 커밋 `976ab1e`) | `tests/test_cloud.py` |
| 재검증 | 같은 잘못된 경로 → 배포 전에 차단(새 배포 없음) / 올바른 경로로 재배포 → 성공 / 스모크 14/14 | `runs/20261122-033719-*`, `-033754-*`, `-033916-*` |

---

## 재실행 결과

`docs/RERUN.md`에 새 폴더 재실행 결과를 적었습니다.

---

## 권한, 비용, 자원 정리

| 항목 | 내용 |
|------|------|
| **권한** | 배포자는 `vercel login`(브라우저 인증)으로 이 프로젝트와 저장소만 다룸. 함수는 저장소 연결 때 Vercel이 만든 토큰으로 이 저장소 하나를 읽음(토큰 자체는 쓰기도 가능, 한계로 기록). 프로덕션 도메인만 공개, 다른 배포 주소는 로그인 필요 |
| **비용** | Hobby(무료) 플랜, 결제 수단 없음. 한도를 넘으면 과금 없이 멈춤. 요청이 없으면 0, 최대 실행 시간 60초, Blob 1GB 중 416.64MB 사용 |
| **정리** | `make cleanup-keep`로 지난 배포를 지우고 상태를 다시 조회: 배포 1개(프로덕션), 저장소 1개(M2 파일 5개), `/health` 200 (`runs/20261122-041711-cleanup-keep/cleanup_status.txt`) |
| **남긴 자원과 계획** | 채점 기간 동안 프로덕션·모델·프로젝트를 남김. 채점 뒤 `make cleanup`으로 모두 지우고 상태를 다시 조회 |

자세한 내용은 `docs/COST_AND_CLEANUP.md`에 있습니다.

---

## 한계와 윤리적 고려

| 항목 | 내용 |
|------|------|
| **데이터의 민감성** | 학습 데이터에 욕설, 혐오 표현, 연예인 실명이 포함. 문서의 예시는 실명과 심한 욕설 일부를 `*`로 가렸고, 웹 화면 예시는 실명이 없고 수위가 지나치지 않은 댓글만 사용 |
| **로그** | 댓글 원문과 작성자 정보를 남기지 않음. 건수, 해시, 처리 시간, 모델 버전만 남기고, 클라우드 런타임 로그에도 원문이 없는지 스모크마다 검사 |
| **리포트** | 법률 자문이 아니라는 고지를 항상 포함하고 서버에 저장하지 않음 |
| **오판** | "일반" 판정이 피해가 없다는 뜻은 아님. 맥락에 의존하는 괴롭힘은 놓칠 수 있음 |
| **편향** | 편향이 들어간 댓글에서 성능이 더 낮음 (`docs/EVALUATION.md` 4.3장) |
| **공개 엔드포인트** | 요청 속도 제한이 없음. 남용은 입력 크기 제한과 Hobby 한도로만 막음 |

---

## 결과 파일 위치

| 내용 | 위치 |
|------|------|
| 구현 기준 문서 (기능, 규칙, 결정 사항, 진행 상태) | `docs/PROJECT_SPEC.md` |
| 데이터 설명 | `docs/DATA.md`, `data/README.md`, `data/LICENSE-DATA.md` |
| 평가 결과 | `docs/EVALUATION.md` |
| 실행 기록 | `runs/index.csv`, `runs/<run_id>/run.json`, `output.txt` |
| 모델 카드 | `runs/*-train*/model_card.json` |
| 평가 지표와 혼동행렬 | `runs/*-evaluate*/metrics.json`, `confusion_matrix_*.png` |
| 모델 비교, 행동 검사 | `runs/*-compare*/comparison.json`, `runs/*-probe*/probe_metrics.json` |
| ONNX 동등성 확인 | `runs/*-export-onnx/onnx_check.json` |
| 로컬·클라우드 스모크 | `runs/*-smoke-local*/local_smoke.json`, `runs/*-smoke-cloud-*/cloud_smoke.json`, `runtime_logs.jsonl` |
| 클라우드 실행 기록 | `docs/CLOUD_RUNS.md`, `runs/*-vercel-setup`, `*-upload-model`, `*-deploy-*` |
| 오류 재현과 복구 | `docs/INCIDENT.md`, `runs/*-incident*` |
| 재실행 | `docs/RERUN.md` |
| 권한·비용·정리 후 상태 | `docs/COST_AND_CLEANUP.md`, `runs/*-cleanup-*/cleanup_status.txt` |

---

## 참고 자료와 본인 구현 구분

### 외부에서 가져온 것

| 항목 | 내용 |
|------|------|
| **데이터** | Korean HateSpeech Dataset (BEEP!), CC BY-SA 4.0. Moon, J., Cho, W. I., & Lee, J. (2020). *BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection.* https://arxiv.org/abs/2005.12503 |
| **사전학습 모델** | `beomi/KcELECTRA-base`, `beomi/KcELECTRA-base-v2022` (MIT). 가중치는 저장소에 포함하지 않음 |
| **라이브러리** | FastAPI, scikit-learn, PyTorch, Transformers, ONNX Runtime, tokenizers, vercel-blob 등 (`requirements*.txt`) |
| **클라우드 문서** | Vercel Python 런타임, 함수 한도, Blob, Deployment Protection 문서 (`docs/PROJECT_SPEC.md` 16장) |
| **참고 문서** | 증거 체크리스트와 신고 절차 문구의 참고 자료는 `docs/PROJECT_SPEC.md` 16장 |

### 이 과제에서 만든 것

- `src/commentguard/`의 전처리, 검증·분할, 학습, 평가, 심각도 점수와 기준값 조정, 리포트, API, ONNX 변환·서빙, Blob 모델 로드와 배포 전 검사, 실행 기록, 웹 화면
- `scripts/`의 배포·업로드·스모크·정리 스크립트, `tests/`, `Makefile`, `app.py`, `vercel.json`, `docs/`
- 행동 검사 세트(`data/probes/`)의 문장

코드 라이선스는 MIT(`LICENSE`)이고, 데이터에는 적용되지 않습니다.
