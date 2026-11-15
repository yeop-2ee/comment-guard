"""명세서 5.3장 정상 동작 검사 T1~T11. 같은 케이스를 scripts/smoke_cloud.py가 클라우드에서도 검사합니다."""

import csv
import io

import pytest
from fastapi.testclient import TestClient

from commentguard.api import create_app
from commentguard.model_store import ModelLoadError
from conftest import TEST_VERSION


def _detail_indexes(resp):
    return [d["index"] for d in resp.json()["detail"]]


def test_t1_single_normal_comment(client):
    r = client.post("/v1/classify", json={"texts": ["오늘 무대 정말 좋았어요"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == TEST_VERSION
    (res,) = body["results"]
    assert set(res) == {"index", "label", "probs", "severity_score", "grade", "bias"}
    assert abs(sum(res["probs"].values()) - 1) < 1e-3
    assert 0 <= res["severity_score"] <= 100
    assert res["grade"] in ("normal", "caution", "severe")
    assert set(res["bias"]) == {"type", "contain_gender_bias"}


def test_t2_batch_100_keeps_order(client, tiny_model):
    texts = [f"댓글 {i}번 {'바보' if i % 3 == 0 else '좋아요'}" for i in range(100)]
    r = client.post("/v1/classify", json={"texts": texts})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["index"] for x in results] == list(range(100))
    # 순서 유지: 한 건씩 보낸 결과와 같은 위치에서 같은 점수
    for i in (0, 1, 57, 99):
        single = client.post("/v1/classify", json={"texts": [texts[i]]}).json()["results"][0]
        assert single["severity_score"] == results[i]["severity_score"]


def test_t3_emoji_english_chosung(client):
    r = client.post("/v1/classify", json={"texts": ["ㅋㅋㅋ 대박😂 lol ㅅㄱ", "Good job!! 👍", "ㅎㅇㅌ"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_t4_blank_text_rejected_with_index(client, bad):
    r = client.post("/v1/classify", json={"texts": ["정상 댓글", bad]})
    assert r.status_code == 422
    assert 1 in _detail_indexes(r)


def test_t4_missing_or_empty_texts(client):
    assert client.post("/v1/classify", json={}).status_code == 422
    assert client.post("/v1/classify", json={"texts": []}).status_code == 422


def test_t5_too_long_text(client):
    r = client.post("/v1/classify", json={"texts": ["가" * 1001]})
    assert r.status_code == 422
    assert _detail_indexes(r) == [0]
    assert client.post("/v1/classify", json={"texts": ["가" * 1000]}).status_code == 200


def test_t6_batch_101_rejected(client):
    assert client.post("/v1/classify", json={"texts": ["좋아요"] * 101}).status_code == 422


def test_t7_invalid_sensitivity(client):
    r = client.post("/v1/filter", json={"sensitivity": "extreme",
                                        "comments": [{"id": "c1", "text": "좋아요"}]})
    assert r.status_code == 422


def test_filter_actions_follow_rules(client):
    comments = [{"id": f"c{i}", "text": t} for i, t in
                enumerate(["응원합니다 힘내세요", "진짜 바보 같네", "여자들은 다 똑같아 멍청해"])]
    out = {}
    for sens in ("low", "medium", "high"):
        r = client.post("/v1/filter", json={"sensitivity": sens, "comments": comments})
        assert r.status_code == 200
        body = r.json()
        assert [x["id"] for x in body["results"]] == ["c0", "c1", "c2"]
        assert sum(body["summary"].values()) == 3
        out[sens] = body["results"]
    rule = {"low": {"normal": "show", "caution": "show", "severe": "hide"},
            "medium": {"normal": "show", "caution": "blur", "severe": "hide"},
            "high": {"normal": "show", "caution": "hide", "severe": "hide"}}
    for sens, results in out.items():
        for x in results:
            assert x["action"] == rule[sens][x["grade"]]
    # 기본 민감도는 medium
    r = client.post("/v1/filter", json={"comments": comments})
    assert r.json()["sensitivity"] == "medium"


def test_t8_report_text_only_warns_missing_fields(client):
    r = client.post("/v1/evidence/report", json={"items": [{"text": "진짜 바보 같네"}]})
    assert r.status_code == 200
    body = r.json()
    item = body["items"][0]
    assert set(item["missing_fields"]) == {"url", "posted_at", "author_handle", "captured_at",
                                           "screenshot_sha256"}
    assert body["warnings"]
    assert any("법률 자문이 아닙니다" in n for n in body["notices"])


def test_t9_report_is_reproducible(client):
    payload = {"case_title": "테스트 사건", "items": [
        {"text": "진짜 바보 같네", "url": "https://example.com/p/1#c1",
         "posted_at": "2026-11-01T21:14:00+09:00", "author_handle": "가짜닉네임1"},
        {"text": "여자들은 다 똑같아 멍청해", "posted_at": "2026-11-01T10:00:00Z"},
    ]}
    a = client.post("/v1/evidence/report", json=payload).json()
    b = client.post("/v1/evidence/report", json=payload).json()
    assert [i["text_sha256"] for i in a["items"]] == [i["text_sha256"] for i in b["items"]]
    assert a["report_sha256"] == b["report_sha256"]
    assert a["report_id"] != b["report_id"]
    scores = [i["severity_score"] for i in a["items"]]
    assert scores == sorted(scores, reverse=True)
    assert [i["rank"] for i in a["items"]] == [1, 2]


def test_t10_report_csv(client):
    items = [{"text": "오늘 무대 정말 좋았어요"}, {"text": "진짜 바보 같네"}, {"text": "멍청한 소리 그만해"}]
    r = client.post("/v1/evidence/report?format=csv", json={"case_title": "한글 제목", "items": items})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    raw = r.content
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "한글 제목" in text
    data_lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(data_lines))))
    assert len(rows) == len(items)
    assert {row["text"] for row in rows} == {it["text"] for it in items}


def test_t11_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": TEST_VERSION}


def test_meta(client):
    r = client.get("/v1/meta")
    assert r.status_code == 200
    assert r.json()["severity"]["thresholds"] == {"caution": 30, "severe": 60}


def test_health_503_without_model():
    with TestClient(create_app(model=None, load_on_startup=False)) as c:
        assert c.get("/health").status_code == 503


def test_startup_fails_on_bad_model_uri(monkeypatch, tmp_path):
    """잘못된 MODEL_URI면 빈 모델로 뜨지 않고 시작이 실패해야 함 (명세서 6.3장)."""
    monkeypatch.setenv("MODEL_URI", str(tmp_path / "models" / "m1-20261115-WRONG"))
    with pytest.raises(ModelLoadError):
        with TestClient(create_app()):
            pass


def test_report_rejects_bad_screenshot_hash_and_naive_datetime(client):
    r = client.post("/v1/evidence/report", json={"items": [{"text": "좋아요", "screenshot_sha256": "abc"}]})
    assert r.status_code == 422
    r = client.post("/v1/evidence/report", json={"items": [{"text": "좋아요", "posted_at": "2026-11-01T10:00:00"}]})
    assert r.status_code == 422


def test_web_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "댓글가드" in r.text
    # 화면은 API만 호출하고, 외부 스크립트를 불러오지 않음
    assert "/v1/filter" in r.text and "/v1/evidence/report" in r.text
    assert "<script src=" not in r.text


@pytest.fixture()
def two_clients(tiny_model):
    import copy
    other = copy.copy(tiny_model)
    # 기준값만 다른 두 번째 모델: 같은 입력이 다른 등급을 받을 수 있음
    other.card = {**tiny_model.card, "model_version": "other-20261115-0000000", "model_id": "m9",
                  "severity": {**tiny_model.card["severity"], "thresholds": {"caution": 1, "severe": 2}}}
    with TestClient(create_app(model=tiny_model, extra_models=[other])) as c:
        yield c


def test_models_lists_default_and_extra(two_clients):
    d = two_clients.get("/v1/models").json()
    assert d["default"] == TEST_VERSION
    assert {m["model_version"] for m in d["models"]} == {TEST_VERSION, "other-20261115-0000000"}
    assert [m["default"] for m in d["models"] if m["model_version"] == TEST_VERSION] == [True]


def test_model_param_selects_model(two_clients):
    body = {"texts": ["오늘 무대 정말 좋았어요"]}
    base = two_clients.post("/v1/classify", json=body).json()
    by_version = two_clients.post("/v1/classify?model=other-20261115-0000000", json=body).json()
    by_id = two_clients.post("/v1/classify?model=m9", json=body).json()
    assert base["model_version"] == TEST_VERSION
    assert by_version["model_version"] == by_id["model_version"] == "other-20261115-0000000"
    assert by_version["results"][0]["grade"] == "severe" and base["results"][0]["grade"] != "severe"
    f = two_clients.post("/v1/filter?model=m9", json={"comments": [{"id": "a", "text": "응원합니다 힘내세요"}]}).json()
    assert f["model_version"] == "other-20261115-0000000" and f["results"][0]["action"] == "hide"
    r = two_clients.post("/v1/evidence/report?model=m9", json={"items": [{"text": "응원합니다 힘내세요"}]}).json()
    assert r["model_version"] == "other-20261115-0000000"
    assert two_clients.get("/v1/meta?model=m9").json()["severity"]["thresholds"] == {"caution": 1, "severe": 2}


def test_unknown_model_is_404(two_clients):
    r = two_clients.post("/v1/classify?model=nope", json={"texts": ["응원합니다"]})
    assert r.status_code == 404 and r.json()["error"] == "unknown_model"


def test_selected_model_version_is_logged(two_clients, caplog):
    import json as _json, logging
    with caplog.at_level(logging.INFO, logger="commentguard"):
        two_clients.post("/v1/classify?model=m9", json={"texts": ["응원합니다"]})
    rec = [_json.loads(r.message) for r in caplog.records if '"request"' in r.message][-1]
    assert rec["model_version"] == "other-20261115-0000000"


def test_ensemble_averages_member_probabilities(tiny_model):
    import numpy as np
    from commentguard.ensemble import EnsembleModel
    ens = EnsembleModel([tiny_model, tiny_model], {"model_version": "ens-test", "severity": tiny_model.card["severity"]})
    h, b = ens.predict_proba(["진짜 바보 같네", "응원합니다"])
    h0, b0 = tiny_model.predict_proba(["진짜 바보 같네", "응원합니다"])
    assert np.allclose(h, h0) and np.allclose(b, b0) and ens.severity.caution == 30


def test_select_serving_rule():
    from commentguard.compare import select_serving
    rt = lambda mb, p50=10, batch=100, rss=1000: {"latency_ms_single_p50": p50, "latency_ms_batch100_p50": batch,
                                                  "peak_rss_mb": rss, "artifact_mb": mb}
    m = lambda f1, **k: {"valid_hate_macro_f1": f1, "runtime_cpu": rt(**k)}
    # 가장 높은 후보, 0.005 미만 차이는 작은 모델, 운영 조건을 못 맞추면 제외
    assert select_serving({"m2": m(0.713, mb=437), "m3": m(0.730, mb=500), "ens": m(0.750, mb=2200)})["serving_model"] == "ens"
    assert select_serving({"m2": m(0.713, mb=437), "m3": m(0.716, mb=500)})["serving_model"] == "m2"
    assert select_serving({"m2": m(0.713, mb=437), "m4": m(0.760, mb=1300, rss=9000)})["serving_model"] == "m2"
