# Comment Guard

> 한국어 댓글의 악플 정도를 판정하는 필터링·증거 정리 API (클라우드 MLOps 대체 실습과제)

한국어 댓글을 **정상 / 공격 / 혐오**로 분류하고 0~100점 **심각도 점수**를 매깁니다. 이 점수로 보기 싫은 댓글을 민감도에 맞춰 가려 주고, 악플 피해자가 직접 모은 댓글을 심각도 순 리포트로 정리해 줍니다. 공개 데이터로 직접 학습한 모델을 FastAPI로 서빙하며, Vercel 배포를 목표로 합니다.

> **진행 상태 (2026-11-16 기준).** 데이터, 모델 학습과 평가, API, 로컬 검증까지 실행했습니다. **클라우드 배포와 그 뒤 단계(오류 재현, 새 폴더 재실행, 자원 정리)는 아직 실행하지 않았습니다.** 실행하지 않은 부분은 이 문서에 "미실행"이라고 적었습니다.

---

## 과제 항목별 진행 상태

| 과제 항목 | 상태 | 근거 |
|------|------|------|
| **가. 데이터와 모델 준비** | 실행함 | [데이터](#데이터), [모델](#모델), `docs/DATA.md` |
| **나. 클라우드 실행** | **미실행** (클라우드 구성만 결정) | [클라우드 실행 기록](#클라우드-실행-기록) |
| **다. 결과 평가** | 실행함 (로컬) | [평가 결과](#평가-결과), `docs/EVALUATION.md` |
| **라. 실행 기록** | 실행함 (`runs/`에 32건) | [실행 기록과 버전 관리](#실행-기록과-버전-관리) |
| **라. 오류 재현과 복구** | **미실행** (시나리오만 정함) | [오류와 복구](#오류와-복구) |
| **마. 재실행** | **미실행** | [재실행 결과](#재실행-결과) |
| **마. 권한·비용·자원 정리** | **미실행** (계획만 있음) | [권한, 비용, 자원 정리](#권한-비용-자원-정리) |

---

## 시스템 구성도

```
[브라우저 / 클라이언트] ── HTTPS ─▶ [Vercel 프로덕션 도메인] ─▶ [현재 배포: Python 함수 (FastAPI)]
                                                                 ├─ 시작 시 로드 ─▶ [Vercel Blob: models/<version>/]
                                                                 └─ 요청 로그(원문 제외) ─▶ [Vercel 런타임 로그]
[로컬] 데이터 → 검증 → 학습 → 평가 → 테스트 → 모델 업로드 → 배포 → /health 확인 → 프로덕션 승격 → 스모크
        └─ 모든 단계 기록 ─▶ runs/<run_id>/run.json
```

| 항목 | 선택 | 이유 |
|------|------|------|
| **실행** | Vercel Functions (Python 런타임) | 요청이 없으면 비용이 없음. 같은 FastAPI 앱을 로컬 `uvicorn`과 Vercel에서 그대로 실행 |
| **진입점** | Vercel 프로덕션 도메인 (HTTPS) | 화면과 API를 같은 주소로 제공 |
| **버전·롤백** | 배포마다 새 배포 생성, 새 배포의 `/health`가 통과해야 프로덕션 승격(`vercel promote`) | 새 배포가 시작에 실패하면 이전 배포가 계속 서비스 |
| **모델 저장** | Vercel Blob (비공개 저장소) | 코드와 모델을 분리해 모델만 교체 가능 |
| **리전** | `icn1` (서울) | 사용자가 한국에 있음 |

이 구성은 설계와 스크립트까지 준비했고, **실제 배포는 아직 하지 않았습니다.** 선택 이유의 전체 표는 `docs/PROJECT_SPEC.md` 6장에 있습니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **댓글 분류** | 정상 / 공격 / 혐오 라벨과 클래스별 확률, 편향 표시 (`POST /v1/classify`, 최대 100건) |
| **심각도 점수와 등급** | `round(100 × (P(혐오) + 0.5 × P(공격)))`, 일반 / 주의 / 심각 3등급 |
| **민감도별 필터링** | 낮음·보통·높음에 따라 보이기 / 흐리게 / 숨기기. 삭제가 아니라 숨김 (`POST /v1/filter`) |
| **증거 정리 리포트** | 직접 모은 댓글을 심각도 순으로 정리, 원문·리포트 SHA-256, 누락 정보 경고, JSON·CSV (`POST /v1/evidence/report`, 최대 500건) |
| **웹 화면** | 댓글 필터, 증거 리포트, 판정 상세, 모델 성능 탭. 외부 CDN 없이 파일 하나 (`GET /`) |
| **모델 선택 비교** | 로컬 서버에서 모델을 골라 같은 댓글의 판정을 비교 (`GET /v1/models`, `?model=`) |
| **원문을 남기지 않는 로그** | 건수, 해시, 처리 시간, 모델 버전만 기록하고 테스트로 확인 |
| **실행 기록** | 모든 단계가 시각, 성공 여부, git 커밋, 모델 버전, 입력 해시를 `runs/`에 기록 |

하지 않는 것: 법률 판단, 댓글 크롤링, 작성자 신원 추적, 원본 증거 대체. 기능과 입력 규칙의 전체 명세는 `docs/PROJECT_SPEC.md` 2장에 있습니다.

---

## 서비스 흐름

```
댓글 입력 (1건 ~ 100건, 리포트는 500건)
        ↓
전처리 (NFC 정규화·공백 정리, 학습과 서빙이 같은 함수 사용)
        ↓
모델 판정 (서빙 모델 M2: KcELECTRA-base 미세조정)
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
make data      원본 다운로드 + 해시 검증
    ↓
make validate  잘못된 값 검사 + 학습/검증/평가 분할
    ↓
make train     B0·M1 학습 (M2는 python -m commentguard.m2)
    ↓
make evaluate  eval 471건으로 평가
    ↓
make test      pytest 69개
    ↓
upload-model → deploy → smoke → cleanup   (클라우드 단계, 미실행)
```

---

## 기술 스택

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Vercel](https://img.shields.io/badge/Vercel-Python_Functions-000000?logo=vercel&logoColor=white)](https://vercel.com/docs/functions/runtimes/python)

| 기술 | 버전 | 용도 |
|------|------|------|
| [FastAPI](https://fastapi.tiangolo.com) | 0.162.2 | API 서버 + 웹 화면 서빙 |
| [Uvicorn](https://www.uvicorn.org) | 0.54.0 | ASGI 서버 |
| [scikit-learn](https://scikit-learn.org) | 1.9.1 | M1 (TF-IDF + 로지스틱 회귀), 지표 계산 |
| [PyTorch](https://pytorch.org) | 2.15.0 | M2·M3 미세조정과 추론 |
| [Transformers](https://huggingface.co/docs/transformers) | 5.22.2 | 사전학습 모델 로드 (KcELECTRA) |
| [pandas](https://pandas.pydata.org) | 3.1.1 | 데이터 검증·분할 |
| [pytest](https://docs.pytest.org) | 9.2.0 | 정상 동작 검사 |
| [Vercel Blob](https://vercel.com/docs/vercel-blob) | — | 모델 파일 저장 (비공개 저장소) |
| [Vercel Python 런타임](https://vercel.com/docs/functions/runtimes/python) | Python 3.12 | FastAPI 앱을 Vercel 함수로 실행 |

모든 의존성은 `requirements.txt`, `requirements-dev.txt`, `requirements-m2.txt`에 버전을 고정했습니다.

---

## 프로젝트 구조

```
comment-guard/
│
├── src/commentguard/
│   ├── text.py              # normalize() — 학습·서빙 공용 전처리
│   ├── data.py              # 잘못된 값 검사, 층화 분할
│   ├── train.py             # B0, M1, M1-bias 학습
│   ├── m2.py                # 사전학습 모델 미세조정과 로더 (M2, M3)
│   ├── ensemble.py          # 확률 평균 앙상블
│   ├── evaluate.py          # 지표, 부트스트랩 구간, 혼동행렬, 필터 지표 (eval을 읽는 유일한 모듈)
│   ├── compare.py           # 모델 비교, 서빙 모델 선택 규칙
│   ├── probe.py             # 행동 검사 세트 평가
│   ├── severity.py          # 심각도 점수, 등급, 민감도 규칙, 기준값 조정
│   ├── report.py            # 증거 리포트 (JSON·CSV, 해시)
│   ├── model_store.py       # 로컬·Blob 모델 로드, MODEL_URI 처리
│   ├── api.py               # FastAPI 앱
│   ├── runlog.py            # 실행 기록 공통 함수
│   └── static/index.html    # 웹 화면
│
├── scripts/
│   ├── download_data.sh         # 데이터 다운로드 + 해시 검증
│   ├── vercel_setup.sh          # 프로젝트 연결·Blob 저장소 준비
│   ├── upload_model.sh          # 모델 업로드
│   ├── deploy_vercel.sh         # 배포, /health 확인 후 프로덕션 승격
│   ├── smoke_cloud.py           # 로컬·클라우드 공용 스모크 테스트
│   └── build_perf_snapshot.py   # 웹 화면 성능 탭 수치를 runs/ 기록에서 생성
│
├── configs/                 # train.yaml, severity.yaml, serving.yaml, deploy.env.example
├── data/
│   ├── raw/                 # 원본 3개 파일 + MANIFEST.sha256
│   ├── processed/           # train_fit, valid, eval
│   └── probes/              # 행동 검사 세트 (직접 만든 문장)
├── tests/                   # pytest 69개
├── runs/                    # 실행 기록 (run.json, 지표, 모델 카드)
├── models/                  # 로컬 모델 (Git 제외)
├── docs/
│   ├── PROJECT_SPEC.md      # 구현 기준 문서
│   ├── DATA.md              # 데이터 설명
│   └── EVALUATION.md        # 평가 결과
│
├── Makefile                 # 파이프라인 진입점
├── vercel.json              # Vercel 함수 설정
└── requirements*.txt        # 버전 고정
```

---

## 시작하기 (로컬 개발)

### 사전 요구사항

| 항목 | 비고 |
|------|------|
| Python | 3.11 |
| make, git | - |
| Apple Silicon(MPS) 또는 GPU | 선택 — M2 학습이 빨라짐 (맥 MPS에서 약 24분) |
| Node.js 20+, Vercel CLI | 클라우드 단계에만 필요 |

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

### 4. 서빙 모델(M2) 학습

모델 가중치는 저장소에 없습니다. 서빙 모델이 M2라서(`configs/serving.yaml`) 서버를 띄우기 전에 학습해야 합니다.

```bash
export PYTHONPATH=src
.venv/bin/python -m commentguard.m2                                        # → models/LATEST_M2
.venv/bin/python -m commentguard.evaluate models/$(cat models/LATEST_M2)   # eval로 평가
```

M2를 학습하지 않고 M1으로 띄우려면 5번에서 `MODEL_URI=models/$(cat models/LATEST) make serve`를 씁니다.

### 5. 서버 실행

```bash
make serve          # uvicorn, 포트 8080
make smoke-local    # 다른 터미널에서. 13개 케이스 → runs/*/local_smoke.json
```

- 웹 화면: http://127.0.0.1:8080
- API 문서 (Swagger): http://127.0.0.1:8080/docs

### 6. 클라우드 단계 (미실행)

```bash
make vercel-setup   # Vercel 프로젝트 연결·Blob 저장소 1회 준비
make upload-model   # 서빙 모델을 Blob 저장소에 업로드
make deploy         # 테스트 통과 후 배포, /health 확인 후 프로덕션 승격
make smoke          # 클라우드 스모크 → runs/*/cloud_smoke.json
```

명령은 준비되어 있지만 **아직 실행해 검증하지 않았습니다.** `make cleanup`은 아직 구현하지 않았습니다.

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
curl -s -X POST http://127.0.0.1:8080/v1/filter -H 'content-type: application/json' \
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

eval은 학습, 어휘 사전, 하이퍼파라미터, 심각도 기준값 조정, 서빙 모델 선택 어디에도 쓰지 않았습니다. 자세한 내용과 누수 방지 체크리스트는 `docs/DATA.md`에 있습니다.

---

## 모델

| ID | 모델 | 역할 |
|------|------|------|
| **B0** | 최빈값 분류기 | 단순 기준 |
| **M1** | 글자 2~5-gram TF-IDF + 로지스틱 회귀 | 직접 학습한 기준·대체 모델 |
| **M2** | `beomi/KcELECTRA-base` 미세조정 (MIT, 리비전 `681201a` 고정) | **서빙 모델** |
| **M3** | `beomi/KcELECTRA-base-v2022` 미세조정 (MIT, 리비전 `8728557` 고정) | 비교용 |
| **M23X6** | M2·M3 각각 시드 3개, 여섯 모델의 확률 평균 | 비교용 |

- **버전 규칙:** `<모델ID>-<YYYYMMDD>-<git 커밋 7자리>` (예: `m2-20261110-314b2b2`). 커밋 안 된 변경이 있는 상태로 만든 모델은 제출용으로 쓰지 않습니다.
- **모델 카드:** 모델마다 `model_card.json`에 버전, 커밋, 학습 데이터 해시, 하이퍼파라미터, 기준값, 지표를 기록합니다. 가중치는 저장소에 올리지 않고 카드만 `runs/<학습 run_id>/`에 둡니다.
- **서빙 모델 선택:** eval이 아니라 valid 지표와 운영 지표로만 고르는 규칙을 결과를 보기 전에 정했습니다. 더 높은 점수를 얻으려고 후보(M3, KLUE-RoBERTa-large, 앙상블 4종)를 만들어 비교했지만 valid에서 M2를 넘지 못해 M2를 유지했습니다. 지운 후보의 결과도 `docs/EVALUATION.md` 9장, 10장에 남겼습니다.

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
| **정상 동작 검사** | pytest 69개(입력 검증, 리포트 재현성, 로그에 원문이 남지 않는지 등), 로컬 HTTP 스모크 13개 케이스 통과 |
| **잘못된 결과 분석** | eval에서 놓친 혐오 1건(초성 비속어가 어휘 사전에 없음), 사용 중 발견한 오판 1건(신조어 칭찬을 혐오로 판정) |
| **가장 큰 약점** | 노골적인 단어가 없는 성희롱과 비꼼 (직접 만든 행동 검사 36문장에서 성희롱 1/6, 비꼼 0/6) |
| **한계** | 연예 뉴스 댓글에 한정, eval이 471건이라 구간이 넓음, 지연과 메모리는 로컬에서만 측정, 확률 보정·사용자 만족도·리포트의 법적 유용성은 미측정 |

계산 방법, 혼동행렬, 오류 분석, 후보 비교의 전체 내용은 `docs/EVALUATION.md`에 있습니다.

---

## 클라우드 실행 기록

**미실행입니다.** 준비된 것과 남은 것은 다음과 같습니다.

| 준비된 것 | 파일 |
|------|------|
| Vercel 함수 설정 (리전, 메모리, 최대 실행 시간) | `vercel.json` |
| 프로젝트 연결·Blob 저장소 준비 | `scripts/vercel_setup.sh` |
| 모델 업로드 (dirty 모델은 거부) | `scripts/upload_model.sh` |
| 배포 (`/health` 확인 후 프로덕션 승격) | `scripts/deploy_vercel.sh` |
| 실제 요청을 보내고 응답·지연·버전 정보를 기록하는 스모크 | `scripts/smoke_cloud.py` |

남은 것: Vercel 프로젝트·Blob 저장소 생성, 모델 업로드, 배포, 클라우드 스모크(`runs/*/cloud_smoke.json`), cold start·클라우드 지연 측정, `docs/CLOUD_RUNS.md` 작성. 같은 스모크 스크립트를 로컬 서버에는 실행했고 13/13 통과했습니다(`runs/20261110-010553-smoke-local/`).

---

## 실행 기록과 버전 관리

| 항목 | 내용 |
|------|------|
| **단계별 실행** | `Makefile`이 진입점. 각 단계는 실패하면 0이 아닌 코드로 끝나고 Make가 그 자리에서 멈춤. `make deploy`는 `make test`가 통과해야 진행 |
| **실행 기록** | `runs/<run_id>/run.json`에 시각, 성공 여부, git 커밋, dirty 여부, 모델 버전, 입력 해시, 설정, 산출물 위치. 요약은 `runs/index.csv` |
| **실패 처리** | 실패한 실행은 `status: "failed"`로 구분하고 다음 단계로 넘어가지 않음 |
| **외부 모델** | HuggingFace 리비전 해시로 고정 |

---

## 오류와 복구

**미실행입니다.** 재현할 오류는 "배포할 때 잘못된 모델 경로(`MODEL_URI`)를 지정"으로 정해 두었습니다. 새 배포가 시작에 실패하고 프로덕션으로 승격되지 않아 이전 배포가 계속 서비스하는지 확인한 뒤, 배포 전에 경로를 검사하도록 고치고 다시 검증할 계획입니다(`docs/PROJECT_SPEC.md` 8장).

로컬에서 확인한 범위: 잘못된 경로로 서버를 띄우면 빈 모델로 뜨지 않고 시작이 실패합니다(`tests/test_api.py::test_startup_fails_on_bad_model_uri`).

---

## 재실행 결과

**미실행입니다.** 새 폴더에 저장소를 내려받아 [시작하기](#시작하기-로컬-개발)의 명령만으로 다시 실행하고, 환경과 결과 차이를 `docs/RERUN.md`에 기록할 계획입니다.

---

## 권한, 비용, 자원 정리

**미실행입니다.** 자원을 만든 적이 없어 지금 비용이 드는 자원은 없습니다. 계획은 다음과 같습니다.

| 항목 | 계획 |
|------|------|
| **권한** | 함수는 프로젝트에 연결한 Blob 저장소만 읽음. 배포는 `vercel login`으로 하고 장기 토큰을 만들지 않음 |
| **비용 제한** | Hobby(무료) 플랜 한도 안에서만 동작(초과 시 과금 없이 중지), 요청이 없으면 0, 최대 실행 시간 60초 |
| **정리** | Blob 저장소와 Vercel 프로젝트를 지우고 상태를 다시 조회해 기록 |

프로젝트·저장소 이름은 `configs/deploy.env`(Git 제외)에만 두고, 저장소에는 예시 파일만 있습니다.

---

## 한계와 윤리적 고려

| 항목 | 내용 |
|------|------|
| **데이터의 민감성** | 학습 데이터에 욕설, 혐오 표현, 연예인 실명이 포함. 문서의 예시는 실명과 심한 욕설 일부를 `*`로 가렸고, 웹 화면 예시는 실명이 없고 수위가 지나치지 않은 댓글만 사용 |
| **로그** | 댓글 원문과 작성자 정보를 남기지 않음. 건수, 해시, 처리 시간, 모델 버전만 남기고 테스트로 확인 |
| **리포트** | 법률 자문이 아니라는 고지를 항상 포함하고 서버에 저장하지 않음 |
| **오판** | "일반" 판정이 피해가 없다는 뜻은 아님. 맥락에 의존하는 괴롭힘은 놓칠 수 있음 |
| **편향** | 편향이 들어간 댓글에서 성능이 더 낮음 (`docs/EVALUATION.md` 4.3장) |

---

## 결과 파일 위치

| 내용 | 위치 |
|------|------|
| 구현 기준 문서 (기능, 규칙, 결정 사항, 진행 상태) | `docs/PROJECT_SPEC.md` |
| 데이터 설명 | `docs/DATA.md`, `data/README.md`, `data/LICENSE-DATA.md` |
| 평가 결과 | `docs/EVALUATION.md` |
| 실행 기록 | `runs/index.csv`, `runs/<run_id>/run.json` |
| 모델 카드 | `runs/*-train*/model_card.json` |
| 평가 지표와 혼동행렬 | `runs/*-evaluate*/metrics.json`, `confusion_matrix_*.png` |
| 모델 비교 | `runs/*-compare*/comparison.json` |
| 행동 검사 | `data/probes/`, `runs/*-probe*/probe_metrics.json` |
| 로컬 스모크 | `runs/*-smoke-local/local_smoke.json` |
| 클라우드 실행 기록, 오류 보고서, 재실행, 자원 정리 | 아직 없음 |

---

## 참고 자료와 본인 구현 구분

### 외부에서 가져온 것

| 항목 | 내용 |
|------|------|
| **데이터** | Korean HateSpeech Dataset (BEEP!), CC BY-SA 4.0. Moon, J., Cho, W. I., & Lee, J. (2020). *BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection.* https://arxiv.org/abs/2005.12503 |
| **사전학습 모델** | `beomi/KcELECTRA-base`, `beomi/KcELECTRA-base-v2022` (MIT). 가중치는 저장소에 포함하지 않음 |
| **라이브러리** | FastAPI, scikit-learn, PyTorch, Transformers 등 (`requirements*.txt`) |
| **참고 문서** | 증거 체크리스트와 신고 절차 문구의 참고 자료는 `docs/PROJECT_SPEC.md` 16장 |

### 이 과제에서 만든 것

- `src/commentguard/`의 전처리, 검증·분할, 학습, 평가, 심각도 점수와 기준값 조정, 리포트, API, 실행 기록, 웹 화면
- `scripts/`의 배포·스모크 스크립트, `tests/`, `Makefile`, `vercel.json`, `docs/`
- 행동 검사 세트(`data/probes/`)의 문장


코드 라이선스는 MIT(`LICENSE`)이고, 데이터에는 적용되지 않습니다.
