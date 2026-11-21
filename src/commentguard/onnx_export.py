"""M2(트랜스포머) → ONNX 변환과 결과 동등성 확인 (명세서 6.2장). torch가 필요합니다 (requirements-m2.txt).

python -m commentguard.onnx_export [모델 폴더]
    폴더를 주지 않으면 configs/serving.yaml이 가리키는 모델.
    → models/<version>/model.onnx, model.ort, onnx_check.json, runs/<id>-export-onnx/

Vercel 함수 크기 한도(500MB) 안에 PyTorch가 들어가지 않아서 서빙은 onnxruntime으로 합니다.
같은 가중치를 형식만 바꾸므로 모델 버전은 그대로 두고, valid 790건으로 PyTorch 결과와 같은지 확인합니다.
클라우드에는 model.ort(ONNX Runtime 전용 형식, 기본 최적화)를 올립니다. 받은 바이트를 복사 없이 가중치로 써서
메모리를 아끼려는 것이고(M3 기준 최대 2.07GB → 1.38GB, 로컬 측정), .ort도 같은 기준으로 확인합니다.
기준: 토큰 ID 완전 일치, 확률 최대 차이 ≤ 1e-4, hate·bias 예측 100% 일치. 하나라도 어기면 실패로 기록합니다.
eval은 읽지 않습니다.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from commentguard import model_store
from commentguard.data import PROCESSED_DIR, read_tsv
from commentguard.runlog import rel, run_step, sha256_file

MAX_ABS_DIFF = 1e-4
OPSET = 17


def export(model_dir: Path, out: Path) -> None:
    import torch
    from torch import nn

    from commentguard.m2 import load_m2

    m = load_m2(model_dir)

    class Probs(nn.Module):  # 서빙에 필요한 softmax까지 그래프에 넣음
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, input_ids, attention_mask):
            h, b = self.net(input_ids, attention_mask)
            return torch.softmax(h, -1), torch.softmax(b, -1)

    wrapper = Probs(m.net).eval()
    ids = torch.ones((2, 8), dtype=torch.long)
    mask = torch.ones((2, 8), dtype=torch.long)
    axes = {0: "batch", 1: "seq"}
    torch.onnx.export(wrapper, (ids, mask), str(out), input_names=["input_ids", "attention_mask"],
                      output_names=["hate", "bias"], opset_version=OPSET, dynamo=False,
                      dynamic_axes={"input_ids": axes, "attention_mask": axes, "hate": {0: "batch"}, "bias": {0: "batch"}})


def to_ort(model_dir: Path) -> Path:
    import onnxruntime as ort

    out = model_dir / "model.ort"
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(out)
    opts.add_session_config_entry("session.save_model_format", "ORT")
    ort.InferenceSession(str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    return out


def check(model_dir: Path, texts: Sequence[str], model_file: str = "model.onnx") -> dict:
    import torch

    from commentguard.m2 import encode, load_m2
    from commentguard.onnx_model import load_onnx

    ref, onx = load_m2(model_dir), load_onnx(model_dir, model_file=model_file)
    max_length = ref.card["hyperparameters"]["max_length"]
    token_mismatch = 0
    for i in range(0, len(texts), 64):
        batch = texts[i: i + 64]
        t = encode(ref.tok, batch, max_length)
        ids, mask = onx.encode(batch)
        if not (np.array_equal(t["input_ids"].numpy(), ids) and np.array_equal(t["attention_mask"].numpy(), mask)):
            token_mismatch += 1
    torch.set_num_threads(1)
    start = time.perf_counter()
    h_ref, b_ref = ref.predict_proba(texts)
    t_ref = time.perf_counter() - start
    start = time.perf_counter()
    h_onx, b_onx = onx.predict_proba(texts)
    t_onx = time.perf_counter() - start
    diff = float(max(np.abs(h_ref - h_onx).max(), np.abs(b_ref - b_onx).max()))
    hate_agree = float((h_ref.argmax(1) == h_onx.argmax(1)).mean())
    bias_agree = float((b_ref.argmax(1) == b_onx.argmax(1)).mean())
    return {
        "rows": len(texts), "token_batches_mismatch": token_mismatch,
        "max_abs_prob_diff": diff, "hate_pred_agreement": hate_agree, "bias_pred_agreement": bias_agree,
        "seconds_torch": round(t_ref, 2), "seconds_onnx": round(t_onx, 2),
        "criteria": {"max_abs_prob_diff": MAX_ABS_DIFF, "pred_agreement": 1.0, "token_batches_mismatch": 0},
        "passed": token_mismatch == 0 and diff <= MAX_ABS_DIFF and hate_agree == 1.0 and bias_agree == 1.0,
    }


def main(argv: Sequence[str] = ()) -> int:
    argv = list(argv)
    try:
        model_dir = Path(argv[0]) if argv else model_store.serving_dir()
        card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
        valid_path = PROCESSED_DIR / "valid.tsv"
        with run_step("export-onnx", config={"model_dir": rel(model_dir), "opset": OPSET, "max_abs_diff": MAX_ABS_DIFF},
                      inputs=[valid_path, model_dir / "model.safetensors", model_dir / "tokenizer.json"]) as run:
            run.model_version = card["model_version"]
            out = model_dir / "model.onnx"
            texts = list(read_tsv(valid_path)["comments"])
            export(model_dir, out)
            result = check(model_dir, texts)
            ort_out = to_ort(model_dir)
            ort_result = check(model_dir, texts, model_file="model.ort")
            result.update(model_version=card["model_version"], onnx_sha256=sha256_file(out),
                          onnx_mb=round(out.stat().st_size / 1e6, 1),
                          ort={**ort_result, "sha256": sha256_file(ort_out), "mb": round(ort_out.stat().st_size / 1e6, 1)})
            result["passed"] = result["passed"] and ort_result["passed"]
            text = json.dumps(result, ensure_ascii=False, indent=2)
            (model_dir / "onnx_check.json").write_text(text, encoding="utf-8")
            shutil.copy(model_dir / "onnx_check.json", run.dir / "onnx_check.json")
            run.outputs += [f"models/{model_dir.name}/model.onnx", f"models/{model_dir.name}/model.ort", rel(run.dir / "onnx_check.json")]
            run.extra.update({k: result[k] for k in ("max_abs_prob_diff", "hate_pred_agreement", "bias_pred_agreement",
                                                     "token_batches_mismatch", "onnx_mb", "passed")},
                             ort_max_abs_prob_diff=result["ort"]["max_abs_prob_diff"],
                             ort_pred_agreement=[result["ort"]["hate_pred_agreement"], result["ort"]["bias_pred_agreement"]])
            if not result["passed"]:
                out.unlink(missing_ok=True)  # 기준을 못 넘은 파일은 업로드되지 않게 지움
                ort_out.unlink(missing_ok=True)
                raise RuntimeError(f"PyTorch와 결과가 다릅니다: {result}")
    except Exception as e:
        print(f"[export-onnx] 실패: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[export-onnx] {card['model_version']} max_diff={result['max_abs_prob_diff']:.2e} "
          f"agree={result['hate_pred_agreement']:.3f}/{result['bias_pred_agreement']:.3f} {result['onnx_mb']}MB, "
          f"ort max_diff={result['ort']['max_abs_prob_diff']:.2e} {result['ort']['mb']}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
