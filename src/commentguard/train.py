"""B0(최빈값), M1(글자 n-gram TF-IDF + 로지스틱 회귀), M1-bias 학습 (명세서 4장).

python -m commentguard.train
    train_fit으로 학습, valid로 C와 심각도 기준값 선택. eval은 읽지 않습니다.
    → models/<model_version>/, models/LATEST, runs/<run_id>/model_card.json
"""

from __future__ import annotations

import platform
import shutil
import sys
from typing import Any

import joblib
import numpy as np
import sklearn
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from commentguard import model_store
from commentguard.data import PROCESSED_DIR, read_tsv
from commentguard.runlog import (REPO_ROOT, make_model_version, now_kst, rel, run_step,
                                 sha256_file)
from commentguard.severity import (CLASSES, SeverityConfig, filter_rates, grade_macro_f1,
                                   scores_from_probs, tune_thresholds)

TRAIN_CFG = REPO_ROOT / "configs" / "train.yaml"
SEVERITY_CFG = REPO_ROOT / "configs" / "severity.yaml"


def macro_f1(y_true, y_pred, labels) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def select_C(X_fit, y_fit, X_valid, y_valid, labels, grid, logreg_cfg, seed) -> tuple[Any, list[dict]]:
    """C 후보마다 train_fit으로 학습하고 valid macro-F1이 가장 높은 모델을 고릅니다."""
    results, best, best_f1 = [], None, -1.0
    for C in grid:
        clf = LogisticRegression(C=C, random_state=seed, **logreg_cfg).fit(X_fit, y_fit)
        f1 = macro_f1(y_valid, clf.predict(X_valid), labels)
        results.append({"C": C, "valid_macro_f1": round(f1, 4)})
        if f1 > best_f1:
            best, best_f1 = clf, f1
    return best, results


def main() -> int:
    cfg = yaml.safe_load(open(TRAIN_CFG, encoding="utf-8"))
    sev_cfg = yaml.safe_load(open(SEVERITY_CFG, encoding="utf-8"))
    m1 = cfg["m1"]
    seed = cfg["seed"]
    fit_path, valid_path = PROCESSED_DIR / "train_fit.tsv", PROCESSED_DIR / "valid.tsv"

    try:
        with run_step("train", config={"m1": m1, "seed": seed, "severity_tuning": sev_cfg["tuning"]},
                      inputs=[fit_path, valid_path, TRAIN_CFG, SEVERITY_CFG]) as run:
            fit, valid = read_tsv(fit_path), read_tsv(valid_path)
            version = make_model_version("m1")
            run.model_version = version
            if run.git_dirty:
                print("[train] 경고: 커밋 안 된 변경이 있습니다. 이 모델은 제출용으로 쓰지 않습니다.",
                      file=sys.stderr)

            # 어휘 사전과 IDF는 train_fit으로만 만듦
            tfidf_cfg = {**m1["tfidf"], "ngram_range": tuple(m1["tfidf"]["ngram_range"])}
            vectorizer = TfidfVectorizer(**tfidf_cfg)
            X_fit = vectorizer.fit_transform(fit["comments"])
            X_valid = vectorizer.transform(valid["comments"])

            b0 = DummyClassifier(strategy="most_frequent").fit(X_fit, fit["hate"])
            b0_bias = DummyClassifier(strategy="most_frequent").fit(X_fit, fit["bias"])
            b0_valid_f1 = macro_f1(valid["hate"], b0.predict(X_valid), CLASSES)

            hate_clf, hate_grid = select_C(X_fit, fit["hate"], X_valid, valid["hate"], CLASSES,
                                           m1["C_grid"], m1["logreg"], seed)
            bias_clf, bias_grid = select_C(X_fit, fit["bias"], X_valid, valid["bias"],
                                           model_store.BIAS_CLASSES, m1["C_grid"], m1["logreg"], seed)

            # 심각도 기준값은 valid 확률로만 조정
            model = model_store.CommentModel(vectorizer, hate_clf, bias_clf, {"model_version": version})
            valid_hate_p, _ = model.predict_proba(valid["comments"])
            valid_scores = scores_from_probs(valid_hate_p)
            tuning = tune_thresholds(valid_scores, valid["hate"], sev_cfg["sensitivity"],
                                     max_false_block=sev_cfg["tuning"]["max_none_false_block_medium"],
                                     min_hate_block=sev_cfg["tuning"]["min_hate_block_medium"])
            start = SeverityConfig.from_dict(sev_cfg)
            severity = SeverityConfig(tuning["caution"], tuning["severe"], sev_cfg["sensitivity"])
            start_rates = {"caution": start.caution, "severe": start.severe,
                           "grade_macro_f1": grade_macro_f1(valid_scores, valid["hate"],
                                                            start.caution, start.severe),
                           **filter_rates(valid_scores, valid["hate"], start)["medium"]}

            card = {
                "model_version": version,
                "model_id": "m1",
                "created_at": now_kst().isoformat(timespec="seconds"),
                "git_commit": run.git_commit,
                "git_dirty": run.git_dirty,
                "data": {"train_fit": sha256_file(fit_path), "valid": sha256_file(valid_path),
                         "rows": {"train_fit": len(fit), "valid": len(valid)}},
                "features": {"type": "TfidfVectorizer", **m1["tfidf"],
                             "vocab_size": len(vectorizer.vocabulary_)},
                "hate_model": {"type": "LogisticRegression", **m1["logreg"], "C": hate_clf.C,
                               "C_search": hate_grid},
                "bias_model": {"type": "LogisticRegression", **m1["logreg"], "C": bias_clf.C,
                               "C_search": bias_grid,
                               "contain_gender_bias": "bias == 'gender' 로 도출"},
                "severity": severity.to_dict(),
                "severity_tuning": {"selected": tuning, "start_30_60": start_rates},
                "valid_metrics": {
                    "b0_hate_macro_f1": round(b0_valid_f1, 4),
                    "m1_hate_macro_f1": max(r["valid_macro_f1"] for r in hate_grid),
                    "m1_bias_macro_f1": max(r["valid_macro_f1"] for r in bias_grid),
                },
                "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                              "numpy": np.__version__},
            }

            model_dir = model_store.MODELS_DIR / version
            model_store.save(model_dir, vectorizer, hate_clf, bias_clf, card)
            joblib.dump({"hate": b0, "bias": b0_bias}, model_dir / "b0.joblib")
            model_store.LATEST_FILE.write_text(version + "\n")
            size = (model_dir / "model.joblib").stat().st_size
            card["artifact_bytes"] = size
            model_store.write_card(model_dir, card)
            shutil.copy(model_dir / "model_card.json", run.dir / "model_card.json")

            run.outputs += [f"models/{version}/", rel(run.dir / "model_card.json")]
            run.extra["valid_metrics"] = card["valid_metrics"]
            run.extra["severity"] = card["severity"]["thresholds"]
    except Exception as e:
        print(f"[train] 실패: {e}", file=sys.stderr)
        return 1

    print(f"model_version={version}  artifact={size / 1e6:.1f}MB")
    print("valid:", card["valid_metrics"])
    print("C hate:", hate_grid)
    print("C bias:", bias_grid)
    print("severity:", tuning)
    print("start 30/60:", start_rates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
