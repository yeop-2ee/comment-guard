"""모델 산출물 저장·로드. 학습, 평가, 서빙이 같은 CommentModel을 씁니다.

models/<model_version>/
    model.joblib      {"vectorizer", "hate_clf", "bias_clf"} (sklearn 객체만 저장)
    model_card.json   버전, 커밋, 데이터 해시, 하이퍼파라미터, 기준값, 지표
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from commentguard.runlog import REPO_ROOT
from commentguard.severity import CLASSES, SeverityConfig
from commentguard.text import normalize

MODELS_DIR = REPO_ROOT / "models"
LATEST_FILE = MODELS_DIR / "LATEST"
BIAS_CLASSES = ("none", "others", "gender")


@dataclass
class CommentModel:
    vectorizer: Any
    hate_clf: Any
    bias_clf: Any
    card: dict[str, Any]

    @property
    def version(self) -> str:
        return self.card["model_version"]

    @property
    def severity(self) -> SeverityConfig:
        return SeverityConfig.from_dict(self.card["severity"])

    def _features(self, texts: Sequence[str]):
        return self.vectorizer.transform([normalize(t) for t in texts])

    @staticmethod
    def _ordered(clf, proba: np.ndarray, classes: Sequence[str]) -> np.ndarray:
        # 분류기 내부 클래스 순서(알파벳)를 고정 순서로 맞춤
        idx = [list(clf.classes_).index(c) for c in classes]
        return proba[:, idx]

    def predict_proba(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(hate 확률 (n,3) 순서 CLASSES, bias 확률 (n,3) 순서 BIAS_CLASSES)."""
        X = self._features(texts)
        hate = self._ordered(self.hate_clf, self.hate_clf.predict_proba(X), CLASSES)
        bias = self._ordered(self.bias_clf, self.bias_clf.predict_proba(X), BIAS_CLASSES)
        return hate, bias


def save(model_dir: Path, vectorizer, hate_clf, bias_clf, card: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "hate_clf": hate_clf, "bias_clf": bias_clf},
                model_dir / "model.joblib", compress=3)
    write_card(model_dir, card)


def write_card(model_dir: Path, card: dict[str, Any]) -> None:
    (model_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2),
                                               encoding="utf-8")


def load_local(model_dir: str | Path) -> CommentModel:
    model_dir = Path(model_dir)
    card_path, weights = model_dir / "model_card.json", model_dir / "model.joblib"
    for p in (card_path, weights):
        if not p.is_file():
            raise FileNotFoundError(f"모델 파일 없음: {p}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    obj = joblib.load(weights)
    return CommentModel(obj["vectorizer"], obj["hate_clf"], obj["bias_clf"], card)


def load_any(model_dir: str | Path):
    """model_card.json의 model_id로 M1/M2 로더를 고릅니다. M2는 torch가 필요해서 그때만 import."""
    model_dir = Path(model_dir)
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    if card.get("model_id") == "m2":
        from commentguard.m2 import load_m2
        return load_m2(model_dir)
    return load_local(model_dir)


def latest_dir() -> Path:
    if not LATEST_FILE.is_file():
        raise FileNotFoundError("models/LATEST 없음. 먼저 make train을 실행하세요")
    return MODELS_DIR / LATEST_FILE.read_text().strip()


class ModelLoadError(RuntimeError):
    pass


def load_from_uri(uri: str | None) -> CommentModel:
    """MODEL_URI 해석. 없으면 models/LATEST, 있으면 그 로컬 경로."""
    try:
        if not uri:
            return load_local(latest_dir())
        return load_local(uri)
    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError(f"모델 로드 실패 (MODEL_URI={uri!r}): {type(e).__name__}: {e}") from e
