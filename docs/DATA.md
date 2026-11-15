# 데이터

근거 파일: `runs/20261109-231905-data/run.json`, `runs/20261109-231906-validate/data_report.json`. 이 문서의 건수와 분포는 2026-11-15에 이 기록과 원본 파일로 다시 확인했습니다.

## 1. 출처와 사용 조건

| 항목 | 내용 |
|---|---|
| 데이터 | Korean HateSpeech Dataset (BEEP!), Moon et al., 2020 |
| 원본 | https://github.com/kocohub/korean-hate-speech, 커밋 `f8d05dce2b22007bb149e5139c0060c68ad8f94b`로 고정 |
| 라이선스 | CC BY-SA 4.0. 출처를 표시하면 재배포할 수 있고, 가공물도 같은 라이선스로 배포해야 합니다 |
| 저장소 포함 | 원본과 가공본을 모두 포함합니다(결정 D3). 고지는 `data/LICENSE-DATA.md`에 있습니다 |

## 2. 데이터 한 건의 의미

- **한 건 = 한국 연예 뉴스 기사에 달린 댓글 1개**와, 사람이 그 댓글에 붙인 라벨입니다.
- 입력은 텍스트라 물리 단위가 없습니다. 모델 출력 확률은 0~1, 심각도 점수는 0~100 정수(점)입니다.

| 컬럼 | 값 | 역할 |
|---|---|---|
| `comments` | 댓글 원문 (4~137자, 평균 약 39자) | **입력 (유일)** |
| `hate` | `none` / `offensive` / `hate` | 주 정답 |
| `bias` | `none` / `others` / `gender` | 보조 정답 |
| `contain_gender_bias` | `True` / `False` | 보조 정답. 모든 행에서 `bias == "gender"`와 같아서 따로 학습하지 않고 `bias` 예측에서 도출합니다 |

`bias`와 `contain_gender_bias`는 **정답**이라 입력에 넣지 않습니다. 서비스는 댓글 텍스트만 받습니다.

## 3. 잘못된 값 처리 (`src/commentguard/data.py`)

| 검사 | 처리 | train 제거 건수 | dev 제거 건수 |
|---|---|---|---|
| 빈 댓글·공백만 | 제거 | 0 | 0 |
| 허용값 밖 라벨 | 제거 | 0 | 0 |
| `contain_gender_bias`와 `bias` 불일치 | 제거 | 0 | 0 |
| train 안의 완전 중복 | 1건만 남김 | 0 | — |
| train 안에서 같은 텍스트, 다른 라벨 | 모두 제거 | 0 | — |
| train과 dev에 같은 텍스트 | train에서 제거 | 0 | — |
| 정규화 (NFC, 공백 정리) | 학습·서빙 공용 `normalize()` | 바뀐 행 0 | 바뀐 행 0 |

- 중복과 겹침은 정규화한 텍스트 기준으로 검사합니다.
- 이 버전의 원본에서는 모든 검사의 제거 건수가 0건이었습니다. 검사 자체가 동작하는지는 `tests/test_data.py`에서 가짜 예시로 확인합니다.
- 숫자는 `runs/*-validate/data_report.json`에 실행마다 기록됩니다.
- TSV에 큰따옴표가 들어간 댓글이 있어서(train 10건, dev 2건) 따옴표 처리 없이 읽습니다. 기본 설정으로 읽으면 여러 행이 하나로 합쳐질 수 있습니다.

## 4. 분할 기준

| 분할 | 출처 | 건수 | `hate`: none / offensive / hate | `bias`: none / others / gender | 용도 |
|---|---|---|---|---|---|
| `train_fit` | train.tsv 90% | 7,106 | 3,137 / 2,249 / 1,720 | 4,631 / 1,378 / 1,097 | 학습 |
| `valid` | train.tsv 10% | 790 | 349 / 250 / 191 | 517 / 138 / 135 | 하이퍼파라미터, 심각도 기준값 조정 |
| `eval` | dev.tsv 전체 | 471 | 160 / 189 / 122 | 342 / 62 / 67 | **최종 평가만** |

- `train_fit`/`valid`는 `hate` 라벨 비율을 유지하는 층화 분할이고 `random_state=42`입니다(`configs/train.yaml`). `bias`는 층화 기준이 아닙니다.
- 모델은 `train_fit`으로 학습한 것을 그대로 씁니다. 기준값을 `valid` 확률로 조정했기 때문에 `train_fit + valid`로 다시 학습하지 않습니다(명세서 4.2장).
- `eval`은 데이터 제작자가 나눈 공식 분할을 그대로 씁니다. train과 비교하면 `none` 비율이 낮고(34% vs 44%) `offensive` 비율이 높습니다(40% vs 32%). 이 차이는 평가 결과를 해석할 때 고려합니다.
- `test.no_label.tsv`(974건)는 정답이 없어서 평가에 쓰지 않고, 클라우드 스모크 테스트 입력으로만 씁니다.

## 5. 누수 방지 체크리스트

- [x] 입력은 `comments` 하나뿐 (`data.py`가 다른 컬럼을 입력으로 내보내지 않음)
- [x] train과 dev의 중복 텍스트 제거 (`validate()`가 분할 후 겹침이 남아 있으면 중단)
- [x] eval을 학습, 어휘 사전 구축, 기준값 조정에 쓰지 않음 (`train.py`는 `train_fit`·`valid`만 읽고, eval은 `evaluate.py`만 읽음)
- [x] 클래스 가중치를 `train_fit` 기준으로 계산 (M1의 `class_weight="balanced"`와 M2의 loss 가중치 모두 `train_fit` 라벨로 계산됨)
- [x] M2도 같은 규칙을 따름 (`m2.py`는 `train_fit`·`valid`만 읽고, 학습률·에폭·기준값을 `valid`로 선택)
- [x] 서빙 모델을 eval로 고르지 않음 (valid 지표와 운영 지표만 사용, `EVALUATION.md` 8.6장)

## 6. 행동 검사 세트 (`data/probes/`)

공개 데이터가 아니라 직접 만든 가짜 예시입니다. 학습, 어휘 사전, 하이퍼파라미터, 기준값 조정에 쓰지 않고 측정에만 씁니다(명세서 5.6장).

| 파일 | 문장 수 | 기대값 | 해시 고정 |
|---|---|---|---|
| `behavior.tsv` (v1) | 42 (7유형 × 6) | `ok` / `toxic` (medium에서 보임 / 가림) | `MANIFEST.sha256` |
| `behavior_v2.tsv` (v2) | 36 (6유형 × 6) | 등급 `severe` / `caution` | `MANIFEST_v2.sha256` |

- 두 세트 모두 모델 결과를 보기 전에 문장을 작성하고 해시를 고정했습니다. 결과를 본 뒤 문장을 바꾸지 않습니다.
- 결과는 `EVALUATION.md` 8.4장, 8.4.1장에 있습니다.

## 7. 다시 확보하는 방법

```bash
make data       # 없는 파일만 고정 커밋에서 받고 MANIFEST.sha256과 대조. 다르면 중단
make validate   # 검사와 분할 → data/processed/, runs/*-validate/data_report.json
```
