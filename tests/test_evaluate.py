import numpy as np

from commentguard.evaluate import bootstrap_macro_f1, wilson_ci

LABELS = ("none", "offensive", "hate")


def test_bootstrap_is_deterministic_and_orders_models():
    rng = np.random.default_rng(1)
    y = rng.choice(LABELS, 300)
    good = np.where(rng.random(300) < 0.8, y, rng.choice(LABELS, 300))
    preds = {"b0": np.full(300, "none"), "m1": good}
    a = bootstrap_macro_f1(y, preds, LABELS, iters=200, seed=42)
    b = bootstrap_macro_f1(y, preds, LABELS, iters=200, seed=42)
    assert a == b
    assert a["m1"]["ci95"][0] > a["b0"]["ci95"][1]
    assert a["m1_minus_b0"]["ci95"][0] > 0


def test_wilson_ci_contains_rate_and_stays_in_range():
    lo, hi = wilson_ci(0.598, 122)
    assert lo < 0.598 < hi
    assert wilson_ci(0.0, 10)[0] == 0.0 and wilson_ci(1.0, 10)[1] <= 1.0
