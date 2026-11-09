import numpy as np
import pytest

from commentguard.runlog import REPO_ROOT
from commentguard.severity import (SeverityConfig, filter_rates, scores_from_probs,
                                   tune_thresholds)

CONF = SeverityConfig.load(REPO_ROOT / "configs" / "severity.yaml")


def test_score_formula():
    probs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.21, 0.64, 0.15]])
    assert list(scores_from_probs(probs)) == [0, 50, 100, 47]


@pytest.mark.parametrize("score,grade", [(0, "normal"), (29, "normal"), (30, "caution"),
                                         (59, "caution"), (60, "severe"), (100, "severe")])
def test_grade_boundaries(score, grade):
    assert CONF.grade(score) == grade


def test_sensitivity_rules():
    assert CONF.action("caution", "low") == "show"
    assert CONF.action("caution", "medium") == "blur"
    assert CONF.action("caution", "high") == "hide"
    assert all(CONF.action("severe", s) == "hide" for s in ("low", "medium", "high"))


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        SeverityConfig.from_dict({"thresholds": {"caution": 60, "severe": 30}, "sensitivity": {}})


def test_filter_rates():
    scores = np.array([10, 40, 70, 80])
    y = ["none", "none", "hate", "hate"]
    r = filter_rates(scores, y, CONF)
    assert r["medium"] == {"hate_block_rate": 1.0, "none_false_block_rate": 0.5}
    assert r["low"]["none_false_block_rate"] == 0.0


def test_tune_respects_constraint_and_keeps_caution_band():
    rng = np.random.default_rng(0)
    y = np.array(["none"] * 100 + ["offensive"] * 100 + ["hate"] * 100)
    scores = np.concatenate([rng.integers(0, 35, 100), rng.integers(30, 70, 100),
                             rng.integers(60, 100, 100)])
    t = tune_thresholds(scores, y, CONF.rules, max_false_block=0.15, min_hate_block=0.6)
    assert t["false_block_target_met"] and t["hate_block_target_met"]
    assert t["hate_block_rate"] >= 0.6
    assert t["none_false_block_rate"] <= 0.15
    assert t["severe"] - t["caution"] > 10  # 주의 등급이 사라지지 않음


def test_tune_falls_back_when_targets_impossible():
    # 정상과 혐오 점수가 완전히 겹치면 목표를 못 맞춤 → 오차단율 우선
    scores = np.array([50] * 10 + [50] * 10)
    y = ["none"] * 10 + ["hate"] * 10
    t = tune_thresholds(scores, y, CONF.rules, max_false_block=0.15, min_hate_block=0.6)
    assert t["none_false_block_rate"] == 0.0
    assert not t["hate_block_target_met"]
