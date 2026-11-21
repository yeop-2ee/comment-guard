"""M2(트랜스포머) 모델의 ONNX 서빙판 (명세서 6.2장). torch 없이 onnxruntime과 tokenizers만 씁니다.

models/<version>/model.onnx, model.ort 는 python -m commentguard.onnx_export 로 만듭니다.
입력 처리는 m2.encode()와 같습니다: normalize() → 특수 토큰 없이 토큰화 → [CLS] ... [SEP] → 배치 안 최대 길이로 패딩.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from commentguard.severity import SeverityConfig
from commentguard.text import normalize

# 클라우드에 올리는 파일. model.ort(바이트를 복사 없이 쓰는 형식)는 Vercel 함수에서 판정 중 프로세스가 종료돼서
# 쓰지 않고, 검증된 model.onnx를 올립니다 (docs/CLOUD_RUNS.md 6장)
SERVING_FILES = ("model_card.json", "model.onnx", "tokenizer.json", "tokenizer_config.json")
MODEL_FILES = ("model.onnx", "model.ort")


def _special_ids(tok, model_dir: Path) -> tuple[int, int, int]:
    """m2.encode()와 같은 규칙: tokenizer_config.json이 이름을 알려주면 그 값, 아니면 [CLS]/[SEP]/[PAD]."""
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    ids = []
    for key, default in (("cls_token", "[CLS]"), ("sep_token", "[SEP]"), ("pad_token", "[PAD]")):
        name = cfg.get(key) or default
        if isinstance(name, dict):
            name = name.get("content") or default
        token_id = tok.token_to_id(name)
        if token_id is None:
            raise ValueError(f"토크나이저에 {name} 토큰이 없습니다")
        ids.append(token_id)
    return ids[0], ids[1], ids[2]


@dataclass
class OnnxModel:
    """M1의 CommentModel, M2Model과 같은 인터페이스 (predict_proba, version, severity, card)."""
    session: Any
    tok: Any
    special: tuple[int, int, int]
    card: dict[str, Any]
    batch_size: int = 64
    buffer: bytes | None = None  # model.ort 바이트를 가중치로 직접 쓰는 동안 살아 있어야 함

    @property
    def version(self) -> str:
        return self.card["model_version"]

    @property
    def severity(self) -> SeverityConfig:
        return SeverityConfig.from_dict(self.card["severity"])

    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        cls, sep, pad = self.special
        max_length = self.card["hyperparameters"]["max_length"]
        seqs = [[cls] + self.tok.encode(normalize(t), add_special_tokens=False).ids[: max_length - 2] + [sep]
                for t in texts]
        width = max(len(s) for s in seqs)
        ids = np.full((len(seqs), width), pad, dtype=np.int64)
        mask = np.zeros((len(seqs), width), dtype=np.int64)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        return ids, mask

    def predict_proba(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        texts = list(texts)
        hate, bias = [], []
        for i in range(0, len(texts), self.batch_size):
            ids, mask = self.encode(texts[i: i + self.batch_size])
            h, b = self.session.run(["hate", "bias"], {"input_ids": ids, "attention_mask": mask})
            hate.append(h)
            bias.append(b)
        return np.concatenate(hate).astype(float), np.concatenate(bias).astype(float)


def session_options(ort_bytes: bool = False):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if ort_bytes:  # .ort 바이트를 그대로 그래프와 가중치로 씀 (복사 없음, 로컬 비교용)
        opts.enable_cpu_mem_arena = False
        opts.add_session_config_entry("session.disable_prepacking", "1")
        opts.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
        opts.add_session_config_entry("session.use_ort_model_bytes_for_initializers", "1")
    return opts


def load_onnx(model_dir: str | Path, model_bytes: bytes | None = None, model_file: str | None = None) -> OnnxModel:
    """model.onnx가 있으면 그것을, 없으면 model.ort를 읽습니다.
    model_bytes를 주면 디스크 대신 메모리에서 읽습니다 (Vercel 함수의 /tmp가 모델보다 작음). model_file은 그 바이트의 형식."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(model_dir)
    for name in ("model_card.json", "tokenizer.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"모델 파일 없음: {model_dir / name}")
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    if model_bytes is None:
        model_file = model_file or next((f for f in MODEL_FILES if (model_dir / f).is_file()), None)
        if model_file is None:
            raise FileNotFoundError(f"모델 파일 없음: {model_dir}/model.ort 또는 model.onnx")
        if model_file == "model.ort":
            model_bytes = (model_dir / model_file).read_bytes()
    use_bytes = model_bytes is not None
    is_ort = model_file == "model.ort"
    source = model_bytes if use_bytes else str(model_dir / model_file)
    session = ort.InferenceSession(source, session_options(ort_bytes=use_bytes and is_ort), providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return OnnxModel(session, tok, _special_ids(tok, model_dir), card, buffer=model_bytes if is_ort else None)
