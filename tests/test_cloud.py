"""클라우드(Vercel) 경로 검사: Blob 모델 로드(명세서 6.3장), 모델 로드 실패 시 503, ONNX 서빙판 동등성."""

import hashlib
import json
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from commentguard import model_store
from commentguard.api import create_app
from commentguard.model_store import ModelLoadError
from conftest import TEST_VERSION

M2_DIR = model_store.MODELS_DIR / "m2-20261110-314b2b2"


def _fake_blob(monkeypatch, objects: dict[str, bytes]):
    """get_blob / stream_blob 만 흉내 내는 가짜 Blob 저장소."""
    def get_blob(path, access):
        if path not in objects:
            raise FileNotFoundError(f"BlobNotFoundError: {path}")
        return types.SimpleNamespace(body=objects[path])

    @contextmanager
    def stream_blob(path, access):
        if path not in objects:
            raise FileNotFoundError(f"BlobNotFoundError: {path}")
        data = objects[path]
        yield iter([data[:5], data[5:]])

    def head_blob(path):
        if path not in objects:
            raise FileNotFoundError(f"BlobNotFoundError: {path}")
        return types.SimpleNamespace(size=len(objects[path]))

    monkeypatch.setattr(model_store, "_blob_api",
                        lambda: types.SimpleNamespace(get_blob=get_blob, stream_blob=stream_blob, head_blob=head_blob))


def _blob_objects(tiny_model, tmp_path: Path, prefix: str = "models/v1") -> dict[str, bytes]:
    model_store.save(tmp_path, tiny_model.vectorizer, tiny_model.hate_clf, tiny_model.bias_clf, tiny_model.card)
    objects, files = {}, {}
    for name in ("model_card.json", "model.joblib"):
        data = (tmp_path / name).read_bytes()
        objects[f"{prefix}/{name}"] = data
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    objects[f"{prefix}/serving.json"] = json.dumps({"model_version": TEST_VERSION, "files": files}).encode()
    return objects


@pytest.mark.parametrize("uri", ["blob://", "blob:///", "blob://models//v1/"])
def test_bad_blob_uri_format(uri):
    with pytest.raises(ModelLoadError, match="형식 오류"):
        model_store.parse_blob_uri(uri)


def test_blob_load_checks_manifest_and_hash(monkeypatch, tmp_path, tiny_model):
    _fake_blob(monkeypatch, _blob_objects(tiny_model, tmp_path))
    m = model_store.load_from_uri("blob://models/v1/")
    assert m.version == TEST_VERSION


def test_blob_wrong_path_gives_clear_error(monkeypatch, tmp_path, tiny_model):
    _fake_blob(monkeypatch, _blob_objects(tiny_model, tmp_path))
    with pytest.raises(ModelLoadError, match="serving.json 읽기 실패"):
        model_store.load_from_uri("blob://models/m2-20261110-WRONG/")


def test_blob_hash_mismatch_is_rejected(monkeypatch, tmp_path, tiny_model):
    objects = _blob_objects(tiny_model, tmp_path)
    objects["models/v1/model.joblib"] += b"x"
    _fake_blob(monkeypatch, objects)
    with pytest.raises(ModelLoadError, match="해시 불일치"):
        model_store.load_from_uri("blob://models/v1/")


def test_blob_model_loads_on_first_request(monkeypatch, tmp_path, tiny_model):
    """blob:// 모델은 첫 요청에서 불러오고, 그 뒤로는 같은 모델을 씀."""
    _fake_blob(monkeypatch, _blob_objects(tiny_model, tmp_path))
    monkeypatch.setenv("MODEL_URI", "blob://models/v1/")
    with TestClient(create_app()) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["model_version"] == TEST_VERSION
        assert [m["model_version"] for m in c.get("/v1/models").json()["models"]] == [TEST_VERSION]


def test_blob_load_failure_is_503_not_empty_model(monkeypatch, tmp_path, tiny_model, caplog):
    """잘못된 경로면 빈 모델로 응답하지 않고 503 + model_load_failed 로그 (명세서 8.1장)."""
    _fake_blob(monkeypatch, _blob_objects(tiny_model, tmp_path))
    monkeypatch.setenv("MODEL_URI", "blob://models/m2-20261110-WRONG/")
    with TestClient(create_app()) as c:
        for path in ("/health", "/v1/meta"):
            r = c.get(path)
            assert r.status_code == 503 and r.json()["error"] == "model_load_failed"
    assert any('"event": "model_load_failed"' in rec.message for rec in caplog.records)


@pytest.mark.skipif(not (M2_DIR / "model.onnx").is_file(), reason="models/에 M2 ONNX가 없음 (make export-onnx)")
def test_onnx_matches_torch_on_fake_examples():
    """ONNX 서빙판이 PyTorch판과 같은 확률을 내는지 (전체 확인은 onnx_check.json, valid 790건)."""
    pytest.importorskip("torch")
    import numpy as np

    from commentguard.m2 import load_m2
    from commentguard.onnx_model import load_onnx
    from conftest import FAKE

    texts = [t for t, _, _ in FAKE] + ["😂👍 lol ㅋㅋㅋ ㅎㅇㅌ", "가" * 400]
    h1, b1 = load_m2(M2_DIR).predict_proba(texts)
    h2, b2 = load_onnx(M2_DIR).predict_proba(texts)
    assert np.abs(h1 - h2).max() < 1e-4 and np.abs(b1 - b2).max() < 1e-4


@pytest.mark.skipif(not (M2_DIR / "model.onnx").is_file(), reason="models/에 M2 ONNX가 없음 (make export-onnx)")
def test_blob_onnx_loads_from_memory(monkeypatch):
    """model.onnx는 /tmp에 쓰지 않고 메모리에서 불러옴 (Vercel /tmp가 모델보다 작음)."""
    from commentguard.onnx_model import SERVING_FILES

    objects, files = {}, {}
    for name in SERVING_FILES:
        data = (M2_DIR / name).read_bytes()
        objects[f"models/m2/{name}"] = data
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    objects["models/m2/serving.json"] = json.dumps({"model_version": M2_DIR.name, "files": files}).encode()
    _fake_blob(monkeypatch, objects)
    written = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda f, *a, **k: (written.append(str(f)), real_open(f, *a, **k))[1])
    m = model_store.load_from_uri("blob://models/m2/")
    assert m.version == M2_DIR.name and not any(w.endswith("model.onnx") for w in written)
    hate, _ = m.predict_proba(["오늘 무대 정말 좋았어요"])
    assert abs(hate.sum() - 1) < 1e-4


def test_predeploy_check_blocks_wrong_path_and_missing_file(monkeypatch, tmp_path, tiny_model):
    """M5 수정: 배포 전에 MODEL_URI 경로와 파일을 검사해서 망가진 배포를 만들지 않음."""
    objects = _blob_objects(tiny_model, tmp_path)
    _fake_blob(monkeypatch, objects)
    assert model_store.check_blob_uri("blob://models/v1/")["model_version"] == TEST_VERSION
    with pytest.raises(ModelLoadError, match="serving.json 없음"):
        model_store.check_blob_uri("blob://models/m2-20261110-WRONG/")
    del objects["models/v1/model.joblib"]
    with pytest.raises(ModelLoadError, match="model.joblib 없음"):
        model_store.check_blob_uri("blob://models/v1/")
