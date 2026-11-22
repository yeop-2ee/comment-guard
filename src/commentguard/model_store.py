"""모델 산출물 저장·로드. 학습, 평가, 서빙이 같은 CommentModel을 씁니다.

models/<model_version>/
    model.joblib      {"vectorizer", "hate_clf", "bias_clf"} (sklearn 객체만 저장)
    model_card.json   버전, 커밋, 데이터 해시, 하이퍼파라미터, 기준값, 지표
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from commentguard.runlog import REPO_ROOT
from commentguard.severity import CLASSES, SeverityConfig
from commentguard.text import normalize

MODELS_DIR = REPO_ROOT / "models"
LATEST_FILE = MODELS_DIR / "LATEST"
SERVING_CFG = REPO_ROOT / "configs" / "serving.yaml"
BIAS_CLASSES = ("none", "others", "gender")
TRANSFORMER_IDS = ("m2", "m3", "m4")  # m2.py로 학습·로드하는 모델


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
    import joblib  # 서빙 함수(Vercel)에는 scikit-learn·joblib을 설치하지 않아서 M1을 다룰 때만 import
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
    import joblib  # M1 파일을 읽을 때만 필요

    card = json.loads(card_path.read_text(encoding="utf-8"))
    obj = joblib.load(weights)
    return CommentModel(obj["vectorizer"], obj["hate_clf"], obj["bias_clf"], card)


def load_any(model_dir: str | Path):
    """model_card.json의 model_id로 로더를 고릅니다. M2~M4는 torch가 필요해서 그때만 import.
    가중치(model.safetensors) 없이 model.ort나 model.onnx만 있으면 ONNX Runtime으로 로드합니다."""
    model_dir = Path(model_dir)
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    if card.get("members"):
        from commentguard.ensemble import load_ensemble
        return load_ensemble(model_dir)
    if str(card.get("model_id", "m1"))[:2] in TRANSFORMER_IDS:  # m2s43 같은 시드 변형 포함
        if not (model_dir / "model.safetensors").is_file() and any((model_dir / f).is_file() for f in ("model.ort", "model.onnx")):
            from commentguard.onnx_model import load_onnx  # 클라우드: torch 없이 ONNX로 (명세서 6.2장)
            return load_onnx(model_dir)
        from commentguard.m2 import load_m2
        return load_m2(model_dir)
    return load_local(model_dir)


_CACHE: dict[str, Any] = {}


def load_cached(model_dir: str | Path):
    """같은 폴더는 한 번만 메모리에 올립니다 (여러 모델 서빙, 앙상블이 구성 모델을 같이 씀)."""
    key = str(Path(model_dir).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_any(model_dir)
    return _CACHE[key]


def local_candidates() -> list[Path]:
    """models/LATEST* 가 가리키는 로컬 모델 폴더 (화면의 모델 선택용, 명세서 4.6장)."""
    out = []
    for pointer in sorted(MODELS_DIR.glob("LATEST*")):
        d = MODELS_DIR / pointer.read_text().strip()
        if (d / "model_card.json").is_file() and d not in out:
            out.append(d)
    return out


def latest_dir(model_id: str = "m1") -> Path:
    pointer = LATEST_FILE if model_id == "m1" else MODELS_DIR / f"LATEST_{model_id.upper()}"
    if not pointer.is_file():
        raise FileNotFoundError(f"{rel_name(pointer)} 없음. 먼저 해당 모델을 학습하세요")
    return MODELS_DIR / pointer.read_text().strip()


def rel_name(p: Path) -> str:
    return f"models/{p.name}"


def serving_model_id() -> str:
    import yaml
    return yaml.safe_load(open(SERVING_CFG, encoding="utf-8"))["model_id"]


def cloud_extra_dirs() -> list[Path]:
    """configs/serving.yaml의 cloud_extra가 가리키는 로컬 모델 폴더 (업로드·배포 대상)."""
    import yaml
    ids = yaml.safe_load(open(SERVING_CFG, encoding="utf-8")).get("cloud_extra") or []
    return [latest_dir(i) for i in ids]


def serving_dir() -> Path:
    """configs/serving.yaml이 가리키는 로컬 모델 폴더."""
    return latest_dir(serving_model_id())


class ModelLoadError(RuntimeError):
    pass


BLOB_SCHEME = "blob://"
HTTPS_SCHEME = "https://"
REMOTE_SCHEMES = (BLOB_SCHEME, HTTPS_SCHEME)
MANIFEST = "serving.json"


def parse_blob_uri(uri: str) -> str:
    """blob://models/v/ → "models/v" (Vercel Blob 저장소 안의 경로)"""
    prefix = uri[len(BLOB_SCHEME):].strip("/")
    if not prefix or "//" in prefix:
        raise ModelLoadError(f"MODEL_URI 형식 오류 (blob://<경로>/ 이어야 함): {uri!r}")
    return prefix


def _blob_api():
    import vercel.blob.sync as blob  # 로컬 학습에는 필요 없어서 여기서만 import
    return blob


class _BlobStore:
    """blob://<경로>/ : Vercel Blob 비공개 저장소."""

    def __init__(self, uri: str):
        self.prefix, self.api = parse_blob_uri(uri), _blob_api()

    def label(self, name: str) -> str:
        return f"{BLOB_SCHEME}{self.prefix}/{name}"

    def get(self, name: str) -> bytes:
        return self.api.get_blob(f"{self.prefix}/{name}", access="private").body

    def stream(self, name: str):
        return self.api.stream_blob(f"{self.prefix}/{name}", access="private")

    def size(self, name: str) -> int:
        return self.api.head_blob(f"{self.prefix}/{name}").size


class _HttpStore:
    """https://<주소>/ : 공개 HTTPS 주소 (GitHub Release 파일). 파일 이름을 주소 뒤에 붙여 받음."""

    def __init__(self, uri: str):
        if not uri.endswith("/") or len(uri) <= len(HTTPS_SCHEME) + 1:
            raise ModelLoadError(f"MODEL_URI 형식 오류 (https://<주소>/ 처럼 /로 끝나야 함): {uri!r}")
        self.base = uri

    def label(self, name: str) -> str:
        return self.base + name

    def _open(self, name: str, method: str = "GET"):
        import urllib.request
        req = urllib.request.Request(self.base + name, method=method, headers={"User-Agent": "comment-guard"})
        return urllib.request.urlopen(req, timeout=60)  # GitHub Release는 저장소 주소로 리디렉션됨

    def get(self, name: str) -> bytes:
        with self._open(name) as r:
            return r.read()

    def stream(self, name: str):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            with self._open(name) as r:
                yield iter(lambda: r.read(1 << 20), b"")
        return ctx()

    def size(self, name: str) -> int:
        with self._open(name, "HEAD") as r:
            return int(r.headers["Content-Length"])


def _store(uri: str):
    if uri.startswith(BLOB_SCHEME):
        return _BlobStore(uri)
    if uri.startswith(HTTPS_SCHEME):
        return _HttpStore(uri)
    raise ModelLoadError(f"원격 MODEL_URI가 아님: {uri!r}")


def read_model_card(uri: str) -> dict[str, Any]:
    """모델 카드만 읽음 (클라우드 화면의 모델 목록용, 가중치는 고를 때 받음)."""
    store = _store(uri)
    try:
        return json.loads(store.get("model_card.json"))
    except Exception as e:
        raise ModelLoadError(f"{store.label('model_card.json')} 읽기 실패: {type(e).__name__}: {e}") from e


def check_model_uri(uri: str) -> dict[str, Any]:
    """배포 전 검사 (M5 수정, 명세서 8.1장): serving.json과 거기 적힌 파일이 모두 있고 크기가 맞는지 확인.
    파일을 내려받지 않고 메타데이터만 조회합니다. 문제가 있으면 ModelLoadError."""
    store = _store(uri)
    try:
        manifest = json.loads(store.get(MANIFEST))
    except Exception as e:
        raise ModelLoadError(f"{store.label(MANIFEST)} 없음: {type(e).__name__}: {e}") from e
    if not {"model_card.json"} <= set(manifest.get("files", {})):
        raise ModelLoadError(f"{MANIFEST}에 model_card.json이 없습니다")
    for name, meta in manifest["files"].items():
        try:
            size = store.size(name)
        except Exception as e:
            raise ModelLoadError(f"{store.label(name)} 없음: {type(e).__name__}: {e}") from e
        if size != meta["bytes"]:
            raise ModelLoadError(f"{store.label(name)} 크기 불일치 ({size} != {meta['bytes']})")
    return manifest


IN_MEMORY = {"model.onnx", "model.ort"}  # Vercel 함수의 /tmp(약 500MB)보다 커서 디스크에 쓰지 않고 메모리로 받는 파일


def _download(uri: str, dest: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """경로의 serving.json(파일 목록과 sha256)을 읽고, 적힌 파일을 받아 해시를 확인합니다.
    IN_MEMORY 파일은 메모리로, 나머지는 dest에 받습니다."""
    import hashlib

    store = _store(uri)
    try:
        manifest = json.loads(store.get(MANIFEST))
    except Exception as e:  # 경로 없음(404), 권한 없음(403) 등
        raise ModelLoadError(f"{store.label(MANIFEST)} 읽기 실패: {type(e).__name__}: {e}. 경로를 확인하세요") from e
    memory: dict[str, bytes] = {}
    for name, meta in manifest["files"].items():
        h, size = hashlib.sha256(), 0
        # 큰 파일은 크기만큼 bytes를 먼저 만들고 그 자리에 채움. 조각을 모았다가 합치면 모델 크기만큼 메모리를 한 번 더 써서
        # Vercel 함수(2GB)에서 M3(509MB)를 받다가 메모리가 부족했음. onnxruntime은 bytes만 받아서 bytearray는 쓸 수 없음
        buf = bytes(meta["bytes"]) if name in IN_MEMORY else None
        addr = ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p).value if buf is not None else None
        try:
            with store.stream(name) as stream:
                f = None if buf is not None else open(dest / name, "wb")
                try:
                    for chunk in stream:
                        if buf is not None:
                            if size + len(chunk) > len(buf):
                                raise ModelLoadError(f"{store.label(name)} 크기가 serving.json보다 큼")
                            ctypes.memmove(addr + size, chunk, len(chunk))
                        else:
                            f.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
                finally:
                    if f is not None:
                        f.close()
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"{store.label(name)} 다운로드 실패: {type(e).__name__}: {e}") from e
        if h.hexdigest() != meta["sha256"] or size != meta["bytes"]:
            raise ModelLoadError(f"{store.label(name)} 해시 불일치 (받은 {size}바이트)")
        if buf is not None:
            memory[name] = buf
    return manifest, memory


def load_from_uri(uri: str | None) -> CommentModel:
    """MODEL_URI 해석. 없으면 configs/serving.yaml의 로컬 모델, https://(GitHub Release)나 blob://(Vercel Blob)면
    serving.json에 적힌 파일을 받아서, 그 외에는 로컬 경로."""
    try:
        if not uri:
            return load_any(serving_dir())
        if uri.startswith(REMOTE_SCHEMES):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:  # Vercel 함수에서는 /tmp만 쓰기 가능
                manifest, memory = _download(uri, Path(tmp))
                if memory:
                    from commentguard.onnx_model import load_onnx
                    name, data = memory.popitem()
                    model = load_onnx(tmp, model_bytes=data, model_file=name)
                else:
                    model = load_any(tmp)  # 가중치는 메모리로 올라가므로 임시 폴더는 지워도 됨
                if model.version != manifest["model_version"]:
                    raise ModelLoadError(f"serving.json({manifest['model_version']})과 모델 카드({model.version})의 버전이 다릅니다")
                return model
        return load_any(uri)
    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError(f"모델 로드 실패 (MODEL_URI={uri!r}): {type(e).__name__}: {e}") from e
