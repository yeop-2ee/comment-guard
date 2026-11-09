from datetime import datetime, timedelta, timezone

from commentguard.report import build_report, canonical_sha256, report_to_csv, sha256_text

KST = timezone(timedelta(hours=9))


def pred(score, grade):
    return {"grade": grade, "severity_score": score,
            "probs": {"none": 0.2, "offensive": 0.3, "hate": 0.5},
            "bias": {"type": "none", "contain_gender_bias": False}}


def make(inputs, preds):
    return build_report("사건", inputs, preds, "m1-test", {"caution": 30, "severe": 60})


def test_sorted_by_severity_then_input_order():
    r = make([{"text": "a"}, {"text": "b"}, {"text": "c"}],
             [pred(10, "normal"), pred(80, "severe"), pred(80, "severe")])
    assert [i["text"] for i in r["items"]] == ["b", "c", "a"]
    assert [i["rank"] for i in r["items"]] == [1, 2, 3]
    assert r["summary"]["grade_counts"] == {"severe": 2, "caution": 0, "normal": 1}


def test_hashes():
    r = make([{"text": "가나다"}], [pred(10, "normal")])
    assert r["items"][0]["text_sha256"] == sha256_text("가나다")
    assert r["report_sha256"] == canonical_sha256(r["items"])
    # 키 순서가 달라도 같은 해시
    assert canonical_sha256({"b": 1, "a": "한"}) == canonical_sha256({"a": "한", "b": 1})


def test_period_compares_datetimes_across_timezones():
    early = datetime(2026, 11, 1, 23, 0, tzinfo=KST)          # = 14:00 UTC
    late = datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc)  # 문자열로는 더 앞
    r = make([{"text": "a", "posted_at": late}, {"text": "b", "posted_at": early}],
             [pred(1, "normal"), pred(2, "normal")])
    assert r["summary"]["period"] == {"earliest_posted_at": early.isoformat(),
                                      "latest_posted_at": late.isoformat()}


def test_authors_and_checklist():
    full = {"url": "https://example.com/1", "posted_at": datetime(2026, 11, 1, tzinfo=KST),
            "author_handle": "가짜1", "captured_at": datetime(2026, 11, 2, tzinfo=KST),
            "screenshot_sha256": "a" * 64}
    r = make([{"text": "x", **full}, {"text": "y", **full}], [pred(70, "severe"), pred(40, "caution")])
    assert r["summary"]["authors"] == [{"author_handle": "가짜1", "count": 2, "max_severity_score": 70}]
    status = {c["key"]: c["status"] for c in r["evidence_checklist"]}
    assert status["url_capture"] == "met" and status["author_and_time"] == "met"
    assert status["context"] == "user_check"
    assert r["warnings"] == []

    r2 = make([{"text": "x"}], [pred(70, "severe")])
    assert {c["key"]: c["status"] for c in r2["evidence_checklist"]}["url_capture"] == "not_met"


def test_csv_has_bom_comment_header_and_rows():
    r = make([{"text": "쉼표, 와 \"따옴표\""}], [pred(70, "severe")])
    out = report_to_csv(r)
    assert out.startswith("﻿# report_id:")
    assert '"쉼표, 와 ""따옴표"""' in out
