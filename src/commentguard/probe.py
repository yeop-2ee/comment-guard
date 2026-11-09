"""행동 검사 세트 평가 (명세서 5.6장). 학습·조정에는 쓰지 않고 측정에만 씁니다.

python -m commentguard.probe [모델 폴더 ...]
    폴더를 주지 않으면 models/LATEST. 여러 개를 주면 같은 세트로 나란히 비교합니다.
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

PROBE_PATH = REPO_ROOT / "data" / "probes" / "behavior.tsv"
MANIFEST = REPO_ROOT / "data" / "probes" / "MANIFEST.sha256"


def check_manifest() -> str:
    """결과를 본 뒤 문장을 바꾸지 않았는지 확인 (해시가 다르면 중단)."""
    expected = MANIFEST.read_text().split()[0]
    actual = sha256_file(PROBE_PATH).removeprefix("sha256:")
    if actual != expected:
        raise RuntimeError(f"검사 세트 해시가 매니페스트와 다릅니다: {actual} != {expected}")
    return actual


def run_probe(model: Any, df) -> dict[str, Any]:
    """model은 predict_proba(texts) -> (hate_probs, bias_probs)와 severity를 가진 객체."""
    hate_p, _ = model.predict_proba(list(df["text"]))
    scores = scores_from_probs(hate_p)
    sev = model.severity
    actions = np.array([sev.action(sev.grade(int(s)), "medium") for s in scores])
    expected = df["expected"].to_numpy()
    correct = np.where(expected == "ok", actions == "show", actions != "show")
    rows = [{"category": c, "expected": e, "text": t, "probs": dict(zip(CLASSES, np.round(p, 4).tolist())),
             "severity_score": int(s), "action_medium": a, "correct": bool(k)}
            for c, e, t, p, s, a, k in zip(df["category"], expected, df["text"], hate_p, scores, actions, correct)]
    by_cat = {}
    for cat in dict.fromkeys(df["category"]):  # 파일 순서 유지
        m = df["category"].to_numpy() == cat
        by_cat[cat] = {"expected": expected[m][0], "n": int(m.sum()), "correct": int(correct[m].sum()),
                       "accuracy": round(float(correct[m].mean()), 4)}
    ok, tox = expected == "ok", expected == "toxic"
    return {
        "accuracy": round(float(correct.mean()), 4),
        "ok_false_block_rate": round(float((actions[ok] != "show").mean()), 4),   # 멀쩡한 댓글을 가림
        "toxic_miss_rate": round(float((actions[tox] == "show").mean()), 4),      # 악플을 놓침
        "by_category": by_cat,
        "rows": rows,
    }


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv] or [model_store.latest_dir()]
    try:
        with run_step("probe", config={"models": [rel(d) for d in dirs]},
                      inputs=[PROBE_PATH, *[d / "model_card.json" for d in dirs]]) as run:
            probe_hash = check_manifest()
            df = read_tsv(PROBE_PATH)
            results = {}
            for d in dirs:
                model = model_store.load_any(d)
                results[model.version] = run_probe(model, df)
            out = {"probe_sha256": probe_hash, "rows": len(df), "models": results}
            path = run.dir / "probe_metrics.json"
            path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            run.outputs.append(rel(path))
            run.model_version = ",".join(results)
            run.extra["summary"] = {v: {k: r[k] for k in ("accuracy", "ok_false_block_rate", "toxic_miss_rate")}
                                    for v, r in results.items()}
    except Exception as e:
        print(f"[probe] 실패: {e}", file=sys.stderr)
        return 1
    for v, r in results.items():
        print(f"== {v}: 정답률 {r['accuracy']:.0%}, 멀쩡한 댓글 가림 {r['ok_false_block_rate']:.0%}, "
              f"악플 놓침 {r['toxic_miss_rate']:.0%}")
        for cat, c in r["by_category"].items():
            print(f"   {cat:<20} ({c['expected']:<5}) {c['correct']}/{c['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
