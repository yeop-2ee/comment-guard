"""최종 평가 (명세서 5.1~5.2, 5.4장). eval(dev.tsv)을 읽는 유일한 모듈입니다.

python -m commentguard.evaluate
    models/LATEST 모델과 B0를 같은 eval로 비교
    → runs/<run_id>/metrics.json, confusion_matrix.png, error_candidates.json
"""

from __future__ import annotations

import json
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from commentguard import model_store
from commentguard.data import PROCESSED_DIR, read_tsv
from commentguard.runlog import REPO_ROOT, rel, run_step, sha256_file
from commentguard.severity import CLASSES, filter_rates, scores_from_probs

TRAIN_CFG = REPO_ROOT / "configs" / "train.yaml"
ORDINAL = {"none": 0, "offensive": 1, "hate": 2}


def macro_f1(y_true, y_pred, labels) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def classification_metrics(y_true, y_pred, labels: Sequence[str]) -> dict[str, Any]:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return {
        "macro_f1": round(macro_f1(y_true, y_pred, labels), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class": {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4), "support": int(n[i])}
                      for i, lab in enumerate(labels)},
        "confusion_matrix": {"labels": list(labels),
                             "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()},
    }


def bootstrap_macro_f1(y_true, preds: dict[str, np.ndarray], labels, iters: int, seed: int) -> dict[str, Any]:
    """같은 재표본 인덱스로 모든 모델의 macro-F1 95% 구간과, 짝지은 차이(M1 - B0)의 구간을 계산."""
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in preds}
    for _ in range(iters):
        idx = rng.integers(0, len(y), len(y))
        for name, pred in preds.items():
            samples[name].append(macro_f1(y[idx], pred[idx], labels))
    out = {name: {"ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
           for name, v in samples.items()}
    diff = np.array(samples["m1"]) - np.array(samples["b0"])
    out["m1_minus_b0"] = {"ci95": [round(float(np.percentile(diff, 2.5)), 4),
                                   round(float(np.percentile(diff, 97.5)), 4)],
                          "share_le_0": float(np.mean(diff <= 0))}
    out["iters"], out["seed"] = iters, seed
    return out


def wilson_ci(rate: float, n: int, z: float = 1.96) -> list[float]:
    """비율의 95% Wilson 구간. 표본이 작아도(혐오 122건) 0~1을 벗어나지 않음."""
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(center - half), 4), round(float(center + half), 4)]


def top_ngrams(model: model_store.CommentModel, text: str, k: int = 8) -> list[dict[str, Any]]:
    """한 댓글에서 hate 클래스 쪽으로 가장 크게 기여한 n-gram (tfidf 값 × 계수, none 대비)."""
    X = model._features([text])
    classes = list(model.hate_clf.classes_)
    w = model.hate_clf.coef_[classes.index("hate")] - model.hate_clf.coef_[classes.index("none")]
    vocab = np.array(model.vectorizer.get_feature_names_out())
    cols, vals = X.indices, X.data
    contrib = vals * w[cols]
    order = np.argsort(contrib)
    pick = list(order[-k:][::-1]) + list(order[:k])  # hate 쪽 상위 k, none 쪽 상위 k
    return [{"ngram": str(vocab[cols[i]]), "toward_hate": round(float(contrib[i]), 4)} for i in pick]


def save_confusion_png(cm: list[list[int]], labels: Sequence[str], title: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    vmax = max(max(r) for r in cm)
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    cfg = yaml.safe_load(open(TRAIN_CFG, encoding="utf-8"))
    ev = cfg["evaluation"]
    eval_path = PROCESSED_DIR / "eval.tsv"
    try:
        model_dir = model_store.latest_dir()
        model = model_store.load_local(model_dir)
        with run_step("evaluate", config={"bootstrap": ev, "model_dir": rel(model_dir)},
                      inputs=[eval_path, model_dir / "model.joblib", model_dir / "model_card.json"],
                      model_version=model.version) as run:
            df = read_tsv(eval_path)
            y, y_bias = df["hate"].to_numpy(), df["bias"].to_numpy()

            hate_p, bias_p = model.predict_proba(df["comments"])
            m1_pred = np.array(CLASSES)[hate_p.argmax(1)]
            bias_pred = np.array(model_store.BIAS_CLASSES)[bias_p.argmax(1)]
            b0 = joblib.load(model_dir / "b0.joblib")  # 최빈값 분류기는 입력을 보지 않음
            b0_pred = np.asarray(b0["hate"].predict(np.zeros((len(df), 1))))
            b0_bias_pred = np.asarray(b0["bias"].predict(np.zeros((len(df), 1))))
            scores = scores_from_probs(hate_p)
            severity = model.severity

            m1_metrics = classification_metrics(y, m1_pred, CLASSES)
            b0_metrics = classification_metrics(y, b0_pred, CLASSES)
            boot = bootstrap_macro_f1(y, {"b0": b0_pred, "m1": m1_pred}, CLASSES,
                                      ev["bootstrap_iters"], ev["bootstrap_seed"])
            b0_metrics["bootstrap"], m1_metrics["bootstrap"] = boot["b0"], boot["m1"]

            gender_true, gender_pred = y_bias == "gender", bias_pred == "gender"
            bias_metrics = classification_metrics(y_bias, bias_pred, model_store.BIAS_CLASSES)
            bias_metrics["contain_gender_bias_f1"] = round(float(f1_score(gender_true, gender_pred)), 4)
            bias_metrics["b0_macro_f1"] = round(macro_f1(y_bias, b0_bias_pred, model_store.BIAS_CLASSES), 4)
            bias_metrics["b0_predicts"] = str(b0_bias_pred[0])

            rates = filter_rates(scores, y, severity)
            n_hate, n_none = int((y == "hate").sum()), int((y == "none").sum())
            for r in rates.values():
                r["hate_block_ci95"] = wilson_ci(r["hate_block_rate"], n_hate)
                r["none_false_block_ci95"] = wilson_ci(r["none_false_block_rate"], n_none)
                r["n_hate"], r["n_none"] = n_hate, n_none
            rho = spearmanr([ORDINAL[v] for v in y], scores)
            # 편향 라벨별 hate 분류 성능 (명세서 11.3장)
            by_bias = {g: {"n": int((y_bias == g).sum()),
                           "hate_macro_f1": round(macro_f1(y[y_bias == g], m1_pred[y_bias == g], CLASSES), 4),
                           "accuracy": round(float(np.mean(y[y_bias == g] == m1_pred[y_bias == g])), 4)}
                       for g in model_store.BIAS_CLASSES}

            medium = rates["medium"]
            fitness = {
                "m1_ci_above_b0_ci": m1_metrics["bootstrap"]["ci95"][0] > b0_metrics["bootstrap"]["ci95"][1],
                "medium_none_false_block_le_15pct": medium["none_false_block_rate"] <= 0.15,
                "medium_hate_block_ge_60pct": medium["hate_block_rate"] >= 0.60,
            }

            metrics = {
                "model_version": model.version,
                "eval": {"path": rel(eval_path), "sha256": sha256_file(eval_path), "rows": len(df),
                         "label_counts": {c: int((y == c).sum()) for c in CLASSES}},
                "hate": {"b0": b0_metrics, "m1": m1_metrics, "m1_minus_b0": boot["m1_minus_b0"],
                         "bootstrap_iters": boot["iters"], "bootstrap_seed": boot["seed"]},
                "bias": bias_metrics,
                "severity": {"thresholds": {"caution": severity.caution, "severe": severity.severe},
                             "filter_rates": rates,
                             "spearman_label_vs_score": {"rho": round(float(rho.statistic), 4),
                                                         "p_value": float(rho.pvalue)},
                             "grade_counts": {g: int(sum(severity.grade(int(s)) == g for s in scores))
                                              for g in ("normal", "caution", "severe")}},
                "hate_by_bias_group": by_bias,
                "fitness_criteria": fitness,
                "valid_metrics_from_card": model.card.get("valid_metrics"),
            }
            (run.dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
            save_confusion_png(m1_metrics["confusion_matrix"]["rows_true_cols_pred"], CLASSES,
                               f"M1 on eval (n={len(df)})", run.dir / "confusion_matrix_m1.png")

            # 오류 분석 후보: 놓친 혐오(정답 hate, 예측 none)를 P(none)이 높은 순으로
            missed = np.where((y == "hate") & (m1_pred == "none"))[0]
            missed = missed[np.argsort(-hate_p[missed, 0])]
            candidates = [{"eval_row": int(i), "true": y[i], "pred": m1_pred[i],
                           "probs": dict(zip(CLASSES, np.round(hate_p[i], 4).tolist())),
                           "severity_score": int(scores[i]), "bias_true": y_bias[i],
                           "top_ngrams": top_ngrams(model, df["comments"].iloc[i])}
                          for i in missed[:5]]
            (run.dir / "error_candidates.json").write_text(
                json.dumps({"missed_hate_total": int(len(missed)), "candidates": candidates},
                           ensure_ascii=False, indent=2), encoding="utf-8")

            run.outputs += [rel(run.dir / n) for n in
                            ("metrics.json", "confusion_matrix_m1.png", "error_candidates.json")]
            run.extra["summary"] = {"b0_macro_f1": b0_metrics["macro_f1"], "m1_macro_f1": m1_metrics["macro_f1"],
                                    "m1_ci95": m1_metrics["bootstrap"]["ci95"], "fitness": fitness}
    except Exception as e:
        print(f"[evaluate] 실패: {e}", file=sys.stderr)
        return 1

    print(json.dumps({"model_version": model.version, **run.extra["summary"],
                      "medium": medium, "bias_macro_f1": bias_metrics["macro_f1"],
                      "gender_f1": bias_metrics["contain_gender_bias_f1"],
                      "spearman": metrics["severity"]["spearman_label_vs_score"]["rho"],
                      "by_bias": by_bias, "missed_hate": int(len(missed))},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
