# data/

| 경로 | 내용 | 만드는 명령 |
|---|---|---|
| `raw/train.tsv`, `raw/dev.tsv`, `raw/test.no_label.tsv` | 원본 (커밋 `f8d05dc` 고정) | `make data` |
| `raw/MANIFEST.sha256` | 원본 SHA-256. `make data`가 매번 대조하고, 다르면 중단 | 직접 작성 |
| `processed/train_fit.tsv` | train.tsv의 90%, 학습용 | `make validate` |
| `processed/valid.tsv` | train.tsv의 10%, 하이퍼파라미터·기준값 조정용 | `make validate` |
| `processed/eval.tsv` | dev.tsv 전체, **최종 평가 전용** | `make validate` |

- 데이터는 저장소에 포함되어 있습니다. `make data`는 없는 파일만 내려받고 해시를 검사합니다. 다시 받으려면 `FORCE=1 make data`를 실행하세요.
- TSV는 따옴표 처리 없이 저장되어 있습니다. pandas로 읽을 때는 `quoting=csv.QUOTE_NONE`을 쓰거나 `commentguard.data.read_tsv()`를 사용하세요.
- 처리 기준과 분포는 `docs/DATA.md`, 라이선스는 `LICENSE-DATA.md`에 있습니다.
