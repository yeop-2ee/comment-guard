"""M1과 M2를 같은 자료로 비교해서 서빙 모델을 고르는 근거를 만듭니다 (명세서 4.3, M6).

python -m commentguard.compare <M1 폴더> <M2 폴더>
    → runs/<run_id>/comparison.json
    - eval macro-F1과 짝지은 부트스트랩 차이(M2 - M1), 서비스 지표, 행동 검사
    - CPU 지연시간·메모리·로드 시간·산출물 크기 (모델마다 별도 프로세스에서 측정)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from commentguard import model_store
from commentguard.data import PROCESSED_DIR, read_tsv
from commentguard.evaluate import macro_f1
from commentguard.probe import PROBE_PATH, check_manifest, run_probe
from commentguard.runlog import REPO_ROOT, rel, run_step
from commentguard.severity import CLASSES, filter_rates, scores_from_probs

USER_REPORTED = ["ㅇㅈ 이번 곡 개좋음 👍👍"]  # 웹 화면 사용 중 발견한 오판 (EVALUATION.md 5.1)


def measure(model_dir: str) -> dict[str, Any]:
    """이 프로세스에서 모델 하나만 올리고 CPU 단일 스레드 기준 지연과 메모리를 잽니다 (서빙 함수 2GB = vCPU 1개 기준)."""
    import resource
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    t0 = time.perf_counter()
    model = model_store.load_any(model_dir)
    load_s = time.perf_counter() - t0
    texts = list(read_tsv(REPO_ROOT / "data" / "raw" / "test.no_label.tsv")["comments"][:100])  # 공개 데이터, 정답 없음
    model.predict_proba(texts[:8])  # 워밍업
    single = []
    for t in texts[:30]:
        s = time.perf_counter(); model.predict_proba([t]); single.append((time.perf_counter() - s) * 1000)
    batch = []
    for _ in range(3):
        s = time.perf_counter(); model.predict_proba(texts); batch.append((time.perf_counter() - s) * 1000)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024  # macOS는 바이트, 리눅스는 KB
    size = sum(p.stat().st_size for p in Path(model_dir).iterdir() if p.is_file())
    return {"load_seconds": round(load_s, 2), "latency_ms_single_p50": round(float(np.median(single)), 1),
            "latency_ms_single_p90": round(float(np.percentile(single, 90)), 1),
            "latency_ms_batch100_p50": round(float(np.median(batch)), 1),
            "peak_rss_mb": round(rss_mb, 1), "artifact_mb": round(size / 1e6, 1), "threads": 1}


def paired_bootstrap(y, a, b, iters: int, seed: int) -> dict[str, Any]:
    """같은 재표본에서 macro-F1(b) - macro-F1(a)."""
    y, a, b = map(np.asarray, (y, a, b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iters):
        i = rng.integers(0, len(y), len(y))
        diffs.append(macro_f1(y[i], b[i], CLASSES) - macro_f1(y[i], a[i], CLASSES))
    d = np.array(diffs)
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "share_le_0": round(float((d <= 0).mean()), 4), "iters": iters, "seed": seed}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("사용법: python -m commentguard.compare <M1 폴더> <M2 폴더>", file=sys.stderr)
        return 2
    d1, d2 = Path(argv[0]), Path(argv[1])
    ev_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "train.yaml", encoding="utf-8"))["evaluation"]
    eval_path = PROCESSED_DIR / "eval.tsv"
    try:
        with run_step("compare", config={"m1": rel(d1), "m2": rel(d2), "bootstrap": ev_cfg},
                      inputs=[eval_path, PROBE_PATH, d1 / "model_card.json", d2 / "model_card.json"]) as run:
            check_manifest()
            df, probe_df = read_tsv(eval_path), read_tsv(PROBE_PATH)
            y = df["hate"].to_numpy()
            out: dict[str, Any] = {"eval_rows": len(df), "models": {}}
            preds = {}
            for key, d in (("m1", d1), ("m2", d2)):
                model = model_store.load_any(d)
                hp, _ = model.predict_proba(list(df["comments"]))
                preds[key] = np.array(CLASSES)[hp.argmax(1)]
                scores = scores_from_probs(hp)
                rep_p, _ = model.predict_proba(USER_REPORTED)
                rep_s = scores_from_probs(rep_p)
                probe = run_probe(model, probe_df)
                out["models"][key] = {
                    "model_version": model.version,
                    "valid_hate_macro_f1": (model.card.get("valid_metrics") or {}).get(f"{key}_hate_macro_f1"),
                    "eval_hate_macro_f1": round(macro_f1(y, preds[key], CLASSES), 4),
                    "eval_accuracy": round(float((preds[key] == y).mean()), 4),
                    "thresholds": {"caution": model.severity.caution, "severe": model.severity.severe},
                    "eval_filter_medium": filter_rates(scores, y, model.severity)["medium"],
                    "probe": {k: probe[k] for k in ("accuracy", "ok_false_block_rate", "toxic_miss_rate", "by_category")},
                    "user_reported": [{"text": t, "probs": dict(zip(CLASSES, np.round(p, 4).tolist())), "severity_score": int(s),
                                       "grade": model.severity.grade(int(s))} for t, p, s in zip(USER_REPORTED, rep_p, rep_s)],
                }
                del model
            v1, v2 = out["models"]["m1"]["valid_hate_macro_f1"], out["models"]["m2"]["valid_hate_macro_f1"]
            out["selection_rule"] = "명세서 4.3장: valid 차이 >= 0.02, 단건 p50 <= 500ms, 메모리 <= 1500MB 모두 만족하면 M2"
            out["m2_minus_m1_macro_f1"] = paired_bootstrap(y, preds["m1"], preds["m2"], ev_cfg["bootstrap_iters"], ev_cfg["bootstrap_seed"])
            for key, d in (("m1", d1), ("m2", d2)):  # 메모리를 섞지 않도록 별도 프로세스
                res = subprocess.run([sys.executable, "-m", "commentguard.compare", "--measure", str(d)],
                                     capture_output=True, text=True, check=True, cwd=REPO_ROOT)
                out["models"][key]["runtime_cpu"] = json.loads(res.stdout.strip().splitlines()[-1])
            r2 = out["models"]["m2"]["runtime_cpu"]
            checks = {"valid_gain_ge_0.02": (v2 - v1) >= 0.02, "single_p50_le_500ms": r2["latency_ms_single_p50"] <= 500,
                      "peak_rss_le_1500mb": r2["peak_rss_mb"] <= 1500}
            out["selection"] = {"valid_gain": round(v2 - v1, 4), "checks": checks, "serving_model": "m2" if all(checks.values()) else "m1"}
            path = run.dir / "comparison.json"
            path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            run.outputs.append(rel(path))
            run.model_version = f"{out['models']['m1']['model_version']},{out['models']['m2']['model_version']}"
    except Exception as e:
        print(f"[compare] 실패: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps({k: {kk: v[kk] for kk in ("eval_hate_macro_f1", "eval_filter_medium", "runtime_cpu")} | {"probe_acc": v["probe"]["accuracy"], "user_reported": v["user_reported"][0]["grade"]}
                      for k, v in out["models"].items()} | {"m2_minus_m1": out["m2_minus_m1_macro_f1"], "selection": out["selection"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--measure":
        print(json.dumps(measure(sys.argv[2])))
        sys.exit(0)
    sys.exit(main(sys.argv[1:]))
