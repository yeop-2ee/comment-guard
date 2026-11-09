"""행동 검사 세트 평가 (명세서 5.6장). 학습·조정에는 쓰지 않고 측정에만 씁니다.

python -m commentguard.probe [--set behavior_v2] [모델 폴더 ...]
    폴더를 주지 않으면 models/LATEST. 여러 개를 주면 같은 세트로 나란히 비교합니다.
    세트: behavior(v1, 기대값 ok/toxic = medium에서 보임/가림), behavior_v2(기대값 = 등급 caution/severe)
    → runs/<run_id>/probe_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from commentguard import model_store
from commentguard.data import read_tsv
from commentguard.runlog import REPO_ROOT, rel, run_step, sha256_file
from commentguard.severity import CLASSES, scores_from_probs

PROBE_DIR = REPO_ROOT / "data" / "probes"
SETS = {"behavior": (PROBE_DIR / "behavior.tsv", PROBE_DIR / "MANIFEST.sha256"),
        "behavior_v2": (PROBE_DIR / "behavior_v2.tsv", PROBE_DIR / "MANIFEST_v2.sha256")}
PROBE_PATH = SETS["behavior"][0]


def check_manifest(name: str = "behavior") -> str:
    """결과를 본 뒤 문장을 바꾸지 않았는지 확인 (해시가 다르면 중단)."""
    path, manifest = SETS[name]
    expected = manifest.read_text().split()[0]
    actual = sha256_file(path).removeprefix("sha256:")
    if actual != expected:
        raise RuntimeError(f"검사 세트 해시가 매니페스트와 다릅니다: {actual} != {expected}")
    return actual


def run_probe(model: Any, df) -> dict[str, Any]:
    """model은 predict_proba(texts) -> (hate_probs, bias_probs)와 severity를 가진 객체."""
    hate_p, _ = model.predict_proba(list(df["text"]))
    scores = scores_from_probs(hate_p)
    sev = model.severity
    grades = np.array([sev.grade(int(s)) for s in scores])
    actions = np.array([sev.action(g, "medium") for g in grades])
    expected = df["expected"].to_numpy()
    if set(expected) <= {"ok", "toxic"}:   # v1: medium에서 보여야/가려야 하는지
        correct = np.where(expected == "ok", actions == "show", actions != "show")
    else:                                   # v2: 등급이 기대와 같은지
        correct = grades == expected
    rows = [{"category": c, "expected": e, "text": t, "probs": dict(zip(CLASSES, np.round(p, 4).tolist())),
             "severity_score": int(s), "grade": g, "action_medium": a, "correct": bool(k)}
            for c, e, t, p, s, g, a, k in zip(df["category"], expected, df["text"], hate_p, scores, grades, actions, correct)]
    by_cat = {}
    for cat in dict.fromkeys(df["category"]):  # 파일 순서 유지
        m = df["category"].to_numpy() == cat
        by_cat[cat] = {"expected": expected[m][0], "n": int(m.sum()), "correct": int(correct[m].sum()),
                       "accuracy": round(float(correct[m].mean()), 4)}
    out = {"accuracy": round(float(correct.mean()), 4), "by_category": by_cat, "rows": rows}
    ok, tox = expected == "ok", expected == "toxic"
    if ok.any():
        out["ok_false_block_rate"] = round(float((actions[ok] != "show").mean()), 4)   # 멀쩡한 댓글을 가림
    if tox.any():
        out["toxic_miss_rate"] = round(float((actions[tox] == "show").mean()), 4)      # 악플을 놓침
    sev_m = expected == "severe"
    if sev_m.any():
        out["severe_recall"] = round(float((grades[sev_m] == "severe").mean()), 4)     # 심각해야 할 댓글을 심각으로
        out["severe_shown_rate"] = round(float((actions[sev_m] == "show").mean()), 4)  # 심각해야 할 댓글이 그대로 보임
        out["severe_mean_score"] = round(float(scores[sev_m].mean()), 1)
    cau_m = expected == "caution"
    if cau_m.any():
        out["caution_overcall_rate"] = round(float((grades[cau_m] == "severe").mean()), 4)  # 무례한 정도를 심각으로
    return out


def main(argv: list[str]) -> int:
    name = "behavior"
    if argv[:1] == ["--set"]:
        name, argv = argv[1], argv[2:]
    path = SETS[name][0]
    dirs = [Path(a) for a in argv] or [model_store.latest_dir()]
    try:
        with run_step("probe", suffix=None if name == "behavior" else name, config={"set": name, "models": [rel(d) for d in dirs]},
                      inputs=[path, *[d / "model_card.json" for d in dirs]]) as run:
            probe_hash = check_manifest(name)
            df = read_tsv(path)
            results = {}
            for d in dirs:
                model = model_store.load_any(d)
                results[model.version] = run_probe(model, df)
            out = {"set": name, "probe_sha256": probe_hash, "rows": len(df), "models": results}
            path = run.dir / "probe_metrics.json"
            path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            run.outputs.append(rel(path))
            run.model_version = ",".join(results)
            run.extra["summary"] = {v: {k: x for k, x in r.items() if k not in ("by_category", "rows")}
                                    for v, r in results.items()}
    except Exception as e:
        print(f"[probe] 실패: {e}", file=sys.stderr)
        return 1
    for v, r in results.items():
        print(f"== {v}: " + ", ".join(f"{k} {x}" for k, x in r.items() if k not in ("by_category", "rows")))
        for cat, c in r["by_category"].items():
            print(f"   {cat:<20} ({c['expected']:<5}) {c['correct']}/{c['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
