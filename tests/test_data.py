import pandas as pd
import pytest

from commentguard.data import DataValidationError, clean, dedupe_train


def df(rows):
    return pd.DataFrame(rows, columns=["comments", "contain_gender_bias", "bias", "hate"])


def test_clean_removes_bad_rows():
    out, rep = clean(df([
        ["좋은 기사네요", "False", "none", "none"],
        ["   ", "False", "none", "none"],                 # 빈 댓글
        ["라벨 이상", "False", "none", "toxic"],            # 허용값 밖
        ["편향 불일치", "True", "none", "offensive"],        # gender 아닌데 True
    ]))
    assert list(out["comments"]) == ["좋은 기사네요"]
    assert rep["removed_empty"] == 1
    assert rep["removed_invalid_hate"] == 1
    assert rep["removed_gender_bias_inconsistent"] == 1


def test_clean_requires_columns():
    with pytest.raises(DataValidationError):
        clean(pd.DataFrame({"comments": ["a"]}))


def test_dedupe_uses_normalized_text():
    train, _ = clean(df([
        ["같은  댓글", "False", "none", "none"],
        ["같은 댓글 ", "False", "none", "none"],          # 정규화하면 위와 완전 중복
        ["충돌 댓글", "False", "none", "none"],
        ["충돌 댓글", "False", "none", "hate"],           # 같은 텍스트, 다른 라벨
        ["평가에도 있음", "False", "none", "none"],
    ]))
    eval_df, _ = clean(df([["평가에도  있음", "False", "none", "none"]]))
    out, rep = dedupe_train(train, eval_df)
    assert list(out["comments"]) == ["같은 댓글"]
    assert rep["removed_exact_duplicates"] == 1
    assert rep["removed_conflicting_label_rows"] == 2
    assert rep["removed_train_eval_overlap"] == 1
