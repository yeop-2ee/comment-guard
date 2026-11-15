# 데이터 라이선스와 출처

`data/raw/`와 `data/processed/`의 데이터는 아래 데이터셋을 사용하거나 가공한 것입니다.

- 데이터셋: Korean HateSpeech Dataset (BEEP!)
- 출처: https://github.com/kocohub/korean-hate-speech (커밋 `f8d05dce2b22007bb149e5139c0060c68ad8f94b`)
- 라이선스: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
- 저작자: Jihyung Moon, Won Ik Cho, Junbum Lee

## 인용

```
@inproceedings{moon-etal-2020-beep,
  title     = "{BEEP}! {K}orean Corpus of Online News Comments for Toxic Speech Detection",
  author    = "Moon, Jihyung and Cho, Won Ik and Lee, Junbum",
  booktitle = "Proceedings of the Eighth International Workshop on Natural Language Processing for Social Media",
  year      = "2020",
  publisher = "Association for Computational Linguistics",
  pages     = "25--31",
}
```

## 변경 사항

- `data/raw/`: 원본 파일을 그대로 둡니다 (`labeled/train.tsv`, `labeled/dev.tsv`, `test.no_label.tsv`).
- `data/processed/`: `src/commentguard/data.py`로 텍스트 정규화(NFC, 공백 정리), 검증, 층화 분할을 한 결과입니다.
  가공한 데이터도 **CC BY-SA 4.0**으로 배포합니다.
- 웹 화면(`src/commentguard/static/index.html`)의 예시 댓글: `test.no_label.tsv`에서 실명이 없는 댓글을 골라 원문 그대로 실었습니다. 화면에 출처와 라이선스를 표시하며, 이 발췌에도 **CC BY-SA 4.0**이 적용됩니다.

## 주의

데이터에는 욕설, 혐오 표현, 연예인 실명이 들어 있습니다. 연구·교육 목적의 악성 댓글 탐지에만 사용하세요.

이 저장소의 코드 라이선스(MIT, `LICENSE`)는 데이터에 적용되지 않습니다.
