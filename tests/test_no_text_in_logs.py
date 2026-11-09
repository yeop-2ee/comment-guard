"""로그에 댓글 원문이 남지 않는지 (명세서 11.1장). 정상 요청과 422 요청 모두 확인."""

import logging

MARKER = "원문표식XQ7"  # 로그에 나타나면 안 되는 고유 문자열
AUTHOR = "작성자표식ZK3"


def test_no_comment_text_in_logs(client, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    text = f"{MARKER} 진짜 바보 같네"
    calls = [
        ("/v1/classify", {"texts": [text]}),
        ("/v1/filter", {"comments": [{"id": "c1", "text": text}]}),
        ("/v1/evidence/report", {"items": [{"text": text, "author_handle": AUTHOR}]}),
        ("/v1/evidence/report?format=csv", {"items": [{"text": text, "author_handle": AUTHOR}]}),
        ("/v1/classify", {"texts": [text, "   "]}),            # 422
        ("/v1/classify", {"texts": [MARKER * 200]}),           # 422 (1,000자 초과)
    ]
    for path, payload in calls:
        r = client.post(path, json=payload)
        assert r.status_code in (200, 422)

    logged = "\n".join(rec.getMessage() for rec in caplog.records)
    captured = capsys.readouterr()
    assert '"event": "request"' in logged  # 요청 로그 자체는 남아야 함
    for secret in (MARKER, AUTHOR):
        assert secret not in logged
        assert secret not in captured.out
        assert secret not in captured.err


def test_validation_error_response_does_not_echo_input(client):
    r = client.post("/v1/classify", json={"texts": [MARKER * 200]})
    assert r.status_code == 422
    assert MARKER not in r.text
