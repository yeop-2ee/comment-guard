"""웹 화면 '모델 성능' 탭의 수치 스냅샷을 평가 기록(runs/)에서 다시 만듭니다 (명세서 2.6장).

python scripts/build_perf_snapshot.py
    - 모델별 가장 최근 evaluate 기록(metrics.json), compare-all 기록(comparison.json),
      행동 검사 v2 기록(probe_metrics.json)을 읽어
    - src/commentguard/static/index.html의 /*PERF-BEGIN*/ ... /*PERF-END*/ 사이를 바꿉니다.
손으로 숫자를 옮기지 않기 위한 스크립트입니다. 모델을 돌리지 않고 기록 파일만 읽습니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
HTML = ROOT / "src" / "commentguard" / "static" / "index.html"
PROBE1 = [("intensifier_praise", "‘개’ 강조 칭찬", "보여야 함"), ("intensifier_insult", "‘개’ 비하", "가려야 함"),
          ("chosung_praise", "초성 칭찬", "보여야 함"), ("chosung_insult", "초성 욕설", "가려야 함"),
          ("emoji_praise", "이모지 칭찬", "보여야 함"), ("plain_criticism", "정상 비판", "보여야 함"),
          ("plain_insult", "일반 욕설", "가려야 함")]
PROBE2 = [("group_hate_gender", "성별 혐오", "심각"), ("group_hate_other", "기타 집단 혐오", "심각"),
          ("strong_insult", "강한 모욕", "심각"), ("sexual_harassment", "성희롱", "심각"),
          ("rude_only", "단순 무례", "주의"), ("sarcasm", "비꼼", "주의")]


def latest(pattern: str, pick) -> tuple[str, dict] | None:
    """runs/ 에서 조건에 맞는 가장 최근 기록 (run_id가 시각순)."""
    for p in sorted(RUNS.glob(pattern), reverse=True):
        d = json.loads(p.read_text(encoding="utf-8"))
        if pick(d):
            return p.parent.name, d
    return None


def pct(x: float) -> float:
    return round(x * 100, 1)


def main() -> int:
    found = latest("*-compare-all/comparison.json", lambda d: True)
    if not found:
        print("compare-all 기록이 없습니다. 먼저 python -m commentguard.compare --all ... 을 실행하세요", file=sys.stderr)
        return 1
    cmp_id, cmp = found
    sources = [cmp_id]
    models, b0 = {}, None
    for key, c in cmp["models"].items():  # 비교할 때 준 순서
        if not (ROOT / "models" / c["model_version"] / "model_card.json").is_file():
            continue  # 비교 뒤 models/에서 지운 후보는 화면에 올리지 않음 (기록은 runs/와 EVALUATION.md에 있음)
        ev = latest("*-evaluate*/metrics.json", lambda d, v=c["model_version"]: d["model_version"] == v)
        card = latest("*-train*/model_card.json", lambda d, v=c["model_version"]: d["model_version"] == v)
        if not ev or not card:
            print(f"{key}: evaluate 또는 train 기록 없음", file=sys.stderr)
            return 1
        (ev_id, m), (card_id, cd) = ev, card
        sources += [ev_id, card_id]
        h, sev, rt = m["hate"][key], m["severity"], c["runtime_cpu"]
        if cd.get("members"):
            name, desc = key.upper(), cd.get("name") or " + ".join(x["model_id"].upper() for x in cd["members"]) + " 확률 평균"
        else:
            name, desc = key.upper(), (cd.get("base_model") or {}).get("name", "글자 n-gram TF-IDF + 로지스틱 회귀").split("/")[-1]
        cmx = h["confusion_matrix"]["rows_true_cols_pred"]
        diff = (cmp.get("minus_m2_macro_f1") or {}).get(key)
        models[key] = {
            "name": name, "desc": desc, "version": c["model_version"],
            "valid": c["valid_hate_macro_f1"], "f1": h["macro_f1"], "ci": h["bootstrap"]["ci95"], "acc": h["accuracy"],
            "rho": sev["spearman_label_vs_score"]["rho"], "missed": cmx[2][0],
            "bias": m["bias"]["macro_f1"], "gender": m["bias"]["contain_gender_bias_f1"],
            "th": [sev["thresholds"]["caution"], sev["thresholds"]["severe"]],
            "cls": {k: [v["precision"], v["recall"], v["f1"]] for k, v in h["per_class"].items()},
            "cm": cmx,
            "filter": {s: [pct(r["hate_block_rate"]), pct(r["hate_block_ci95"][0]), pct(r["hate_block_ci95"][1]),
                           pct(r["none_false_block_rate"]), pct(r["none_false_block_ci95"][0]), pct(r["none_false_block_ci95"][1])]
                       for s, r in sev["filter_rates"].items()},
            "grades": [sev["grade_counts"][g] for g in ("normal", "caution", "severe")],
            "run": {"load": rt["load_seconds"], "p50": rt["latency_ms_single_p50"], "p90": rt["latency_ms_single_p90"],
                    "batch": rt["latency_ms_batch100_p50"], "rss": rt["peak_rss_mb"], "size": rt["artifact_mb"]},
            "probe1": [c["probe"]["by_category"][k]["correct"] for k, _, _ in PROBE1],
            "diffM2": {"mean": diff["mean"], "ci": diff["ci95"]} if diff else None,
        }
        if b0 is None:
            hb = m["hate"]["b0"]
            b0 = {"name": "B0", "desc": "최빈값 기준 모델", "f1": hb["macro_f1"], "ci": hb["bootstrap"]["ci95"], "acc": hb["accuracy"],
                  "valid": (m.get("valid_metrics_from_card") or {}).get("b0_hate_macro_f1")}
            counts, rows = m["eval"]["label_counts"], sum(m["eval"]["label_counts"].values())
    p2 = latest("*-probe-behavior_v2/probe_metrics.json",
                lambda d: all(v["version"] in d["models"] for v in models.values()))
    if p2:
        sources.append(p2[0])
        for v in models.values():
            v["probe2"] = [p2[1]["models"][v["version"]]["by_category"][k]["correct"] for k, _, _ in PROBE2]
    perf = {"evalRows": rows, "evalCounts": counts, "serving": cmp["selection"]["serving_model"],
            "order": ["b0", *models], "models": {"b0": b0, **models},
            "probe1": [[label, exp] for _, label, exp in PROBE1], "probe2": [[label, exp] for _, label, exp in PROBE2],
            "sources": sorted(set(sources))}
    html = HTML.read_text(encoding="utf-8")
    a, b = html.index("/*PERF-BEGIN*/"), html.index("/*PERF-END*/")
    HTML.write_text(html[:a] + "/*PERF-BEGIN*/" + json.dumps(perf, ensure_ascii=False) + html[b:], encoding="utf-8")
    print(f"serving={perf['serving']} models={list(models)} sources={len(perf['sources'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
