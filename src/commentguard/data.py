"""데이터 검증과 분할 (명세서 3.4~3.6장).

python -m commentguard.data
    data/raw/{train,dev}.tsv → data/processed/{train_fit,valid,eval}.tsv
    처리 결과 숫자는 runs/<run_id>/data_report.json 에 남깁니다.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from commentguard.runlog import REPO_ROOT, run_step, sha256_file
from commentguard.text import normalize

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TEXT = "comments"
LABELS: dict[str, tuple[str, ...]] = {
    "hate": ("none", "offensive", "hate"),
    "bias": ("none", "others", "gender"),
    "contain_gender_bias": ("False", "True"),
}
COLUMNS = [TEXT, *LABELS]


class DataValidationError(RuntimeError):
    pass


def read_tsv(path: str | Path) -> pd.DataFrame:
    # 댓글 안에 큰따옴표가 있어서 따옴표 처리를 끔 (명세서 3.2장)
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False, encoding="utf-8")


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, sep="\t", quoting=csv.QUOTE_NONE, index=False, encoding="utf-8",
              lineterminator="\n")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """정규화, 빈 댓글·허용값 밖 라벨·편향 라벨 불일치 제거."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"필수 컬럼 없음: {missing}")
    report = {"rows_in": len(df)}
    df = df[COLUMNS].copy()
    df[TEXT] = df[TEXT].map(normalize)

    empty = df[TEXT] == ""
    report["removed_empty"] = int(empty.sum())
    df = df[~empty]

    bad = pd.Series(False, index=df.index)
    for col, allowed in LABELS.items():
        col_bad = ~df[col].isin(allowed)
        report[f"removed_invalid_{col}"] = int((col_bad & ~bad).sum())
        bad |= col_bad
    df = df[~bad]

    inconsistent = (df["contain_gender_bias"] == "True") != (df["bias"] == "gender")
    report["removed_gender_bias_inconsistent"] = int(inconsistent.sum())
    df = df[~inconsistent]

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def dedupe_train(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """train 안 중복·라벨 충돌 제거, eval과 겹치는 텍스트를 train에서 제거. 모두 정규화 텍스트 기준."""
    report: dict[str, int] = {}
    before = len(train)
    train = train.drop_duplicates(subset=COLUMNS)
    report["removed_exact_duplicates"] = before - len(train)

    n_label_sets = train.groupby(TEXT)[list(LABELS)].nunique().max(axis=1)
    conflict_texts = set(n_label_sets[n_label_sets > 1].index)
    conflict = train[TEXT].isin(conflict_texts)
    report["conflicting_label_texts"] = len(conflict_texts)
    report["removed_conflicting_label_rows"] = int(conflict.sum())
    train = train[~conflict]

    overlap = train[TEXT].isin(set(eval_df[TEXT]))
    report["removed_train_eval_overlap"] = int(overlap.sum())
    train = train[~overlap]
    return train.reset_index(drop=True), report


def distribution(df: pd.DataFrame) -> dict[str, Any]:
    return {"rows": len(df), **{col: dict(Counter(df[col])) for col in LABELS}}


def validate(raw_dir: Path, out_dir: Path, valid_size: float, seed: int,
             stratify_on: str = "hate") -> dict[str, Any]:
    train_raw, eval_raw = read_tsv(raw_dir / "train.tsv"), read_tsv(raw_dir / "dev.tsv")
    train, train_clean = clean(train_raw)
    eval_df, eval_clean = clean(eval_raw)
    train, dedupe = dedupe_train(train, eval_df)

    train_fit, valid = train_test_split(train, test_size=valid_size, random_state=seed,
                                        stratify=train[stratify_on])

    # 누수 방지 최종 확인: eval 텍스트가 학습 쪽에 하나라도 있으면 중단
    leaked = set(eval_df[TEXT]) & (set(train_fit[TEXT]) | set(valid[TEXT]))
    if leaked:
        raise DataValidationError(f"eval 텍스트 {len(leaked)}건이 학습 데이터에 남아 있음")
    if min(len(train_fit), len(valid), len(eval_df)) == 0:
        raise DataValidationError("빈 분할이 있음")

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train_fit": train_fit, "valid": valid, "eval": eval_df}
    outputs = {}
    for name, df in splits.items():
        path = out_dir / f"{name}.tsv"
        write_tsv(df, path)
        outputs[str(path.relative_to(REPO_ROOT))] = sha256_file(path)

    return {
        "train_clean": train_clean,
        "eval_clean": eval_clean,
        "train_dedupe": dedupe,
        "eval_duplicate_texts": int(eval_df[TEXT].duplicated().sum()),  # 공식 분할이라 제거하지 않고 기록만
        "split": {"valid_size": valid_size, "random_state": seed, "stratify_on": stratify_on},
        "distribution": {name: distribution(df) for name, df in splits.items()},
        "outputs": outputs,
    }


def main() -> int:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "train.yaml", encoding="utf-8"))
    split_cfg = {"valid_size": cfg["split"]["valid_size"], "seed": cfg["seed"],
                 "stratify_on": cfg["split"]["stratify_on"]}
    try:
        with run_step("validate", config=split_cfg,
                      inputs=[RAW_DIR / "train.tsv", RAW_DIR / "dev.tsv"]) as run:
            report = validate(RAW_DIR, PROCESSED_DIR, **split_cfg)
            report_path = run.dir / "data_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            run.outputs += [*report["outputs"], str(report_path.relative_to(REPO_ROOT))]
            run.extra["rows"] = {k: v["rows"] for k, v in report["distribution"].items()}
    except Exception as e:
        print(f"[validate] 실패: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report["distribution"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
