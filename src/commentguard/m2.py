"""M2: KcELECTRA 미세조정 (명세서 4.3장). torch와 transformers가 필요합니다 (requirements-m2.txt).
같은 코드로 M3·M4(명세서 4.5장)도 학습합니다. 설정은 configs/train.yaml의 m2 / m3 / m4.

python -m commentguard.m2 [--model m3|m4] [--seed N]
    --seed를 주면 시드 변형(예: m2s43, 명세서 4.7장)을 만듭니다. 학습률은 같은 모델의 기존 카드에서 고른 값으로 고정합니다.
    train_fit으로 학습, valid로 학습률·에폭(조기 종료)과 심각도 기준값 선택. eval은 읽지 않습니다.
    → models/<id>-<날짜>-<커밋>/ (model.safetensors, config.json, tokenizer, model_card.json), models/LATEST_<ID>

토크나이저 주의: M2에 고정한 리비전의 tokenizer.json에는 post-processor가 없어서 [CLS]/[SEP]이 자동으로 붙지 않습니다.
모델마다 달라지지 않도록 encode()에서 항상 직접 붙입니다.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import transformers
import yaml
from safetensors.torch import load_file, save_file
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from commentguard.data import PROCESSED_DIR, read_tsv
from commentguard.model_store import BIAS_CLASSES, MODELS_DIR
from commentguard.runlog import (REPO_ROOT, make_model_version, now_kst, rel, run_step, sha256_file)
from commentguard.severity import (CLASSES, SeverityConfig, filter_rates, grade_macro_f1,
                                   scores_from_probs, tune_thresholds)
from commentguard.text import normalize

TRAIN_CFG = REPO_ROOT / "configs" / "train.yaml"
SEVERITY_CFG = REPO_ROOT / "configs" / "severity.yaml"
LATEST_M2 = MODELS_DIR / "LATEST_M2"
M2_NOTE = ("main 리비전은 v2023 가중치 + byte-level BPE 토크나이저(5deca41). "
           "tokenizer.json에 post-processor가 없어 [CLS]/[SEP]을 코드에서 직접 부착. "
           "토크나이저 파일은 같은 리비전으로 가정")


class ElectraMultiTask(nn.Module):
    """인코더(ELECTRA, RoBERTa 등) 하나에 hate(3)·bias(3) 분류 머리 두 개. [CLS] 위치의 은닉 상태를 씀."""

    def __init__(self, backbone: nn.Module, dropout: float):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.hate_head = nn.Linear(hidden, len(CLASSES))
        self.bias_head = nn.Linear(hidden, len(BIAS_CLASSES))

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.drop(h)
        return self.hate_head(h), self.bias_head(h)


def encode(tok, texts: Sequence[str], max_length: int) -> dict[str, torch.Tensor]:
    """normalize() → 토큰화(특수 토큰 없이) → [CLS] ... [SEP] 직접 부착 → 배치 안 최대 길이로 패딩."""
    # 세 모델 모두 [CLS]/[SEP]/[PAD]를 쓰지만, 토크나이저가 이름을 알려주면 그 값을 따름
    names = [getattr(tok, a, None) or d for a, d in (("cls_token", "[CLS]"), ("sep_token", "[SEP]"), ("pad_token", "[PAD]"))]
    cls, sep, pad = tok.convert_tokens_to_ids(names)
    seqs = [[cls] + tok(normalize(t), add_special_tokens=False)["input_ids"][: max_length - 2] + [sep] for t in texts]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return {"input_ids": ids, "attention_mask": mask}


@dataclass
class M2Model:
    """서빙·평가용. M1의 CommentModel과 같은 인터페이스 (predict_proba, version, severity, card)."""
    net: ElectraMultiTask
    tok: Any
    card: dict[str, Any]
    device: str = "cpu"
    batch_size: int = 64

    @property
    def version(self) -> str:
        return self.card["model_version"]

    @property
    def severity(self) -> SeverityConfig:
        return SeverityConfig.from_dict(self.card["severity"])

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        hate, bias = [], []
        for i in range(0, len(texts), self.batch_size):
            batch = {k: v.to(self.device) for k, v in encode(self.tok, texts[i: i + self.batch_size],
                                                             self.card["hyperparameters"]["max_length"]).items()}
            h, b = self.net(**batch)
            hate.append(torch.softmax(h, -1).float().cpu().numpy())
            bias.append(torch.softmax(b, -1).float().cpu().numpy())
        return np.concatenate(hate), np.concatenate(bias)


def load_m2(model_dir: str | Path, device: str = "cpu") -> M2Model:
    """허브에 접속하지 않고 폴더 안 파일만으로 로드 (서빙 시 외부 의존 없음)."""
    model_dir = Path(model_dir)
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    backbone = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
    net = ElectraMultiTask(backbone, card["hyperparameters"]["dropout"])
    net.load_state_dict(load_file(model_dir / "model.safetensors"))
    net.to(device)
    tok = AutoTokenizer.from_pretrained(model_dir)
    return M2Model(net, tok, card, device)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def macro_f1(y_true, y_pred, labels) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def train_one(cfg: dict, lr: float, fit, valid, tok, device: str, seed: int, log) -> tuple[dict, list[dict]]:
    """한 학습률로 학습. 에폭마다 valid hate macro-F1을 재고 가장 좋은 상태를 돌려줌."""
    torch.manual_seed(seed); np.random.seed(seed)
    backbone = AutoModel.from_pretrained(cfg["base_model"], revision=cfg["revision"])
    net = ElectraMultiTask(backbone, cfg["dropout"]).to(device)

    y_hate = np.array([CLASSES.index(v) for v in fit["hate"]])
    y_bias = np.array([BIAS_CLASSES.index(v) for v in fit["bias"]])
    # 클래스 가중치는 train_fit 기준 (명세서 3.6장)
    w_hate = torch.tensor(compute_class_weight("balanced", classes=np.arange(3), y=y_hate), dtype=torch.float, device=device)
    w_bias = torch.tensor(compute_class_weight("balanced", classes=np.arange(3), y=y_bias), dtype=torch.float, device=device)
    ce_hate, ce_bias = nn.CrossEntropyLoss(weight=w_hate), nn.CrossEntropyLoss(weight=w_bias)

    texts = list(fit["comments"])
    steps_per_epoch = math.ceil(len(texts) / cfg["batch_size"])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps_per_epoch * cfg["max_epochs"]),
                                            steps_per_epoch * cfg["max_epochs"])
    eval_model = M2Model(net, tok, {"hyperparameters": cfg}, device)
    history, best, best_f1, bad = [], None, -1.0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, cfg["max_epochs"] + 1):
        net.train(); t0 = time.time(); total = 0.0
        order = rng.permutation(len(texts))
        for s in range(steps_per_epoch):
            idx = order[s * cfg["batch_size"]: (s + 1) * cfg["batch_size"]]
            batch = {k: v.to(device) for k, v in encode(tok, [texts[i] for i in idx], cfg["max_length"]).items()}
            h, b = net(**batch)
            loss = ce_hate(h, torch.tensor(y_hate[idx], device=device)) + \
                cfg["bias_loss_weight"] * ce_bias(b, torch.tensor(y_bias[idx], device=device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); total += loss.item()
        hp, bp = eval_model.predict_proba(list(valid["comments"]))
        f1_h = macro_f1(valid["hate"], np.array(CLASSES)[hp.argmax(1)], CLASSES)
        f1_b = macro_f1(valid["bias"], np.array(BIAS_CLASSES)[bp.argmax(1)], BIAS_CLASSES)
        row = {"lr": lr, "epoch": epoch, "train_loss": round(total / steps_per_epoch, 4),
               "valid_hate_macro_f1": round(f1_h, 4), "valid_bias_macro_f1": round(f1_b, 4),
               "seconds": round(time.time() - t0, 1)}
        history.append(row); log(row)
        if f1_h > best_f1:
            best_f1, bad = f1_h, 0
            best = {"state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    "epoch": epoch, "valid_hate_macro_f1": f1_h, "valid_bias_macro_f1": f1_b, "config": backbone.config}
        else:
            bad += 1
            if bad > cfg["patience"]:
                break
    del net, opt
    if device == "mps":
        torch.mps.empty_cache()
    return best, history


def main(argv: Sequence[str] = ()) -> int:
    argv, key, seed_arg = list(argv), "m2", None
    while argv:
        if argv[0] == "--model":
            key = argv[1]
        elif argv[0] == "--seed":
            seed_arg = int(argv[1])
        else:
            print(f"[m2] 모르는 인자: {argv[0]}", file=sys.stderr)
            return 2
        argv = argv[2:]
    cfg_all = yaml.safe_load(open(TRAIN_CFG, encoding="utf-8"))
    sev_cfg = yaml.safe_load(open(SEVERITY_CFG, encoding="utf-8"))
    if key not in ("m2", "m3", "m4"):
        print(f"[m2] 모르는 모델: {key} (m2, m3, m4 중 하나)", file=sys.stderr)
        return 2
    cfg, seed = dict(cfg_all[key]), cfg_all["seed"]
    base_license, base_note = cfg.pop("license", "MIT"), cfg.pop("note", M2_NOTE)
    base_key = key
    if seed_arg is not None:  # 시드 변형: 기존 모델이 valid로 고른 학습률만 사용
        base_card = json.loads((MODELS_DIR / (MODELS_DIR / f"LATEST_{key.upper()}").read_text().strip()
                                / "model_card.json").read_text(encoding="utf-8"))
        cfg["lr_grid"] = [base_card["hyperparameters"]["selected_lr"]]
        seed, key = seed_arg, f"{key}s{seed_arg}"
    fit_path, valid_path = PROCESSED_DIR / "train_fit.tsv", PROCESSED_DIR / "valid.tsv"
    device = pick_device()
    try:
        with run_step("train", suffix=key, config={key: cfg, "seed": seed, "device": device,
                                                    "severity_tuning": sev_cfg["tuning"]},
                      inputs=[fit_path, valid_path, TRAIN_CFG, SEVERITY_CFG]) as run:
            version = make_model_version(key)
            run.model_version = version
            if run.git_dirty:
                print(f"[{key}] 경고: 커밋 안 된 변경이 있습니다. 이 모델은 제출용으로 쓰지 않습니다.", file=sys.stderr)
            fit, valid = read_tsv(fit_path), read_tsv(valid_path)
            tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["revision"])

            history, best = [], None
            for lr in cfg["lr_grid"]:
                b, h = train_one(cfg, lr, fit, valid, tok, device, seed, lambda r: print("  ", r, flush=True))
                history += h
                if best is None or b["valid_hate_macro_f1"] > best["valid_hate_macro_f1"]:
                    best = {**b, "lr": lr}

            backbone = AutoModel.from_config(best["config"])
            net = ElectraMultiTask(backbone, cfg["dropout"]); net.load_state_dict(best["state"])
            model = M2Model(net.to(device), tok, {"model_version": version, "hyperparameters": cfg}, device)
            hp, _ = model.predict_proba(list(valid["comments"]))
            scores = scores_from_probs(hp)
            tuning = tune_thresholds(scores, valid["hate"], sev_cfg["sensitivity"],
                                     max_false_block=sev_cfg["tuning"]["max_none_false_block_medium"],
                                     min_hate_block=sev_cfg["tuning"]["min_hate_block_medium"])
            severity = SeverityConfig(tuning["caution"], tuning["severe"], sev_cfg["sensitivity"])
            start = SeverityConfig.from_dict(sev_cfg)

            model_dir = MODELS_DIR / version
            model_dir.mkdir(parents=True, exist_ok=True)
            save_file({k: v.contiguous() for k, v in best["state"].items()}, model_dir / "model.safetensors")
            best["config"].save_pretrained(model_dir)
            tok.save_pretrained(model_dir)
            card = {
                "model_version": version, "model_id": key, "seed": seed,
                **({"variant_of": base_key} if key != base_key else {}),
                "created_at": now_kst().isoformat(timespec="seconds"),
                "git_commit": run.git_commit, "git_dirty": run.git_dirty,
                "base_model": {"name": cfg["base_model"], "revision": cfg["revision"], "license": base_license,
                               "note": base_note},
                "data": {"train_fit": sha256_file(fit_path), "valid": sha256_file(valid_path),
                         "rows": {"train_fit": len(fit), "valid": len(valid)}},
                "hyperparameters": {**cfg, "selected_lr": best["lr"], "selected_epoch": best["epoch"]},
                "search": history,
                "severity": severity.to_dict(),
                "severity_tuning": {"selected": tuning, "start_30_60": {
                    "caution": start.caution, "severe": start.severe,
                    "grade_macro_f1": grade_macro_f1(scores, valid["hate"], start.caution, start.severe),
                    **filter_rates(scores, valid["hate"], start)["medium"]}},
                "valid_metrics": {f"{key}_hate_macro_f1": round(best["valid_hate_macro_f1"], 4),
                                  f"{key}_bias_macro_f1": round(best["valid_bias_macro_f1"], 4)},
                "training_device": device,
                "libraries": {"python": platform.python_version(), "torch": torch.__version__,
                              "transformers": transformers.__version__},
                "artifact_bytes": (model_dir / "model.safetensors").stat().st_size,
            }
            (model_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
            if key == base_key:  # 시드 변형은 앙상블 구성 요소로만 쓰므로 LATEST를 만들지 않음
                (MODELS_DIR / f"LATEST_{key.upper()}").write_text(version + "\n")
            shutil.copy(model_dir / "model_card.json", run.dir / "model_card.json")
            run.outputs += [f"models/{version}/", rel(run.dir / "model_card.json")]
            run.extra.update(valid_metrics=card["valid_metrics"], selected={"lr": best["lr"], "epoch": best["epoch"]},
                             severity=card["severity"]["thresholds"])
    except Exception as e:
        print(f"[{key}] 실패: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"model_version={version} lr={best['lr']} epoch={best['epoch']} valid={card['valid_metrics']} severity={tuning}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
