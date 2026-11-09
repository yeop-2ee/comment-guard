"""심각도 점수, 등급, 민감도별 처리 (명세서 2.1, 2.3장)와 기준값 조정."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

GRADES = ("normal", "caution", "severe")
SENSITIVITIES = ("low", "medium", "high")
ACTIONS = ("show", "blur", "hide")
CLASSES = ("none", "offensive", "hate")  # 모델 확률의 열 순서


@dataclass(frozen=True)
class SeverityConfig:
    caution: int
    severe: int
    rules: dict[str, dict[str, str]]

    @classmethod
    def load(cls, path: str | Path) -> "SeverityConfig":
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SeverityConfig":
        t = cfg["thresholds"]
        conf = cls(caution=int(t["caution"]), severe=int(t["severe"]), rules=cfg["sensitivity"])
        if not 0 < conf.caution < conf.severe <= 100:
            raise ValueError(f"기준값은 0 < caution < severe <= 100 이어야 함: {t}")
        return conf

    def to_dict(self) -> dict[str, Any]:
        return {"thresholds": {"caution": self.caution, "severe": self.severe},
                "sensitivity": self.rules}

    def grade(self, score: int) -> str:
        if score >= self.severe:
            return "severe"
        if score >= self.caution:
            return "caution"
        return "normal"

    def action(self, grade: str, sensitivity: str) -> str:
        return self.rules[sensitivity][grade]


def scores_from_probs(probs: np.ndarray) -> np.ndarray:
    """probs: (n, 3), 열 순서 CLASSES. severity = round(100 × (P(hate) + 0.5 × P(offensive)))."""
    probs = np.asarray(probs, dtype=float)
    return np.rint(100 * (probs[:, 2] + 0.5 * probs[:, 1])).astype(int)


def filter_rates(scores: np.ndarray, y_true: Sequence[str], conf: SeverityConfig) -> dict[str, dict[str, float]]:
    """민감도별 혐오 차단율(hate 중 hide)과 정상 오차단율(none 중 blur 또는 hide)."""
    y = np.asarray(y_true)
    grades = np.array([conf.grade(int(s)) for s in scores])
    out = {}
    for sens in SENSITIVITIES:
        actions = np.array([conf.action(g, sens) for g in grades])
        out[sens] = {
            "hate_block_rate": float(np.mean(actions[y == "hate"] == "hide")),
            "none_false_block_rate": float(np.mean(actions[y == "none"] != "show")),
        }
    return out


def grade_macro_f1(scores: np.ndarray, y_true: Sequence[str], caution: int, severe: int) -> float:
    """등급을 정답 라벨(normal↔none, caution↔offensive, severe↔hate)과 비교한 macro-F1."""
    from sklearn.metrics import f1_score

    s = np.asarray(scores)
    pred = np.where(s >= severe, "hate", np.where(s >= caution, "offensive", "none"))
    return float(f1_score(y_true, pred, labels=list(CLASSES), average="macro", zero_division=0))


def tune_thresholds(scores: np.ndarray, y_true: Sequence[str], rules: dict[str, dict[str, str]],
                    max_false_block: float = 0.15, min_hate_block: float = 0.60,
                    grid: range = range(5, 96)) -> dict[str, Any]:
    """valid 데이터로 기준값을 고릅니다 (명세서 2.1장의 조정 규칙).

    우선순위 (사전식 비교):
    1) medium 정상 오차단율 <= max_false_block  (명세서 5.2장 목표)
    2) medium 혐오 차단율 >= min_hate_block     (명세서 5.2장 목표)
    3) 등급-라벨 macro-F1 최대 (혐오 차단율만 올리면 주의 등급이 사라지는 퇴화를 막음)
    4) medium 혐오 차단율 최대
    1)을 만족하는 조합이 하나도 없으면 오차단율이 가장 낮은 조합을 고릅니다.
    """
    best, best_key = None, None
    for c in grid:
        for s in grid:
            if s <= c:
                continue
            rates = filter_rates(scores, y_true, SeverityConfig(c, s, rules))["medium"]
            f1 = grade_macro_f1(scores, y_true, c, s)
            fpr_ok = rates["none_false_block_rate"] <= max_false_block
            hate_ok = rates["hate_block_rate"] >= min_hate_block
            key = (fpr_ok, hate_ok if fpr_ok else False,
                   f1 if fpr_ok else -rates["none_false_block_rate"], rates["hate_block_rate"])
            if best_key is None or key > best_key:
                best, best_key = {"caution": c, "severe": s, "grade_macro_f1": f1, **rates,
                                  "false_block_target_met": fpr_ok, "hate_block_target_met": hate_ok}, key
    return {**best, "max_false_block": max_false_block, "min_hate_block": min_hate_block}
