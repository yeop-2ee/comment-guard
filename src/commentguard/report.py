"""증거 정리 리포트 (명세서 2.4장). 서버에 저장하지 않고 응답으로만 돌려줍니다 (D5)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Sequence

from commentguard.runlog import KST

OPTIONAL_FIELDS = ("url", "posted_at", "author_handle", "captured_at", "screenshot_sha256")

DISCLAIMER = (
    "이 리포트는 기계학습 모델의 자동 판정 결과를 정리한 참고 자료이며 법률 자문이 아닙니다. "
    "모욕·명예훼손 등의 해당 여부는 피해자를 알아볼 수 있는지(특정성), 공개된 장소인지(공연성), "
    "사실 적시 여부 같은 법적 요건에 따라 달라지며, 이 모델은 이를 판단하지 않습니다. "
    "실제 신고·고소 시에는 원본 캡처와 URL 등 원 증거를 함께 제출하고, 필요하면 법률 전문가와 상담하세요. "
    "온라인 신고 창구로 경찰청 사이버범죄 신고시스템(ECRM)이 있으며, "
    "온라인 접수 후 경찰서 방문 진술이 필요할 수 있습니다."
)
NORMAL_NOTICE = (
    "모델이 '일반'으로 판정한 댓글도 피해가 없다는 뜻이 아닙니다. "
    "맥락에 따라 의미가 달라지는 괴롭힘, 새로운 은어와 변형 표기는 놓칠 수 있습니다."
)
CHECKLIST = (
    ("url_capture", "브라우저 주소창(URL)이 보이게 원본 화면을 캡처했는지"),
    ("author_and_time", "작성자 ID/닉네임과 작성 일시가 함께 보이는지"),
    ("context", "앞뒤 맥락(원글, 다른 댓글)까지 보존했는지"),
    ("screen_recording", "화면 녹화로 해당 글을 찾아 들어가는 과정을 남겼는지"),
    ("original_kept", "원본 파일을 편집하지 않고 따로 보관했는지"),
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _iso(v: datetime | None) -> str | None:
    return v.isoformat() if v is not None else None


def _checklist(items: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """입력 상태로 알 수 있는 항목만 met/not_met, 나머지는 user_check(사용자가 직접 확인)."""
    def all_have(*fields: str) -> str:
        return "met" if all(all(it.get(f) for f in fields) for it in items) else "not_met"

    status = {
        "url_capture": all_have("url", "screenshot_sha256"),
        "author_and_time": all_have("author_handle", "posted_at"),
    }
    return [{"key": k, "item": text, "status": status.get(k, "user_check")} for k, text in CHECKLIST]


def build_report(case_title: str | None, inputs: Sequence[dict[str, Any]],
                 predictions: Sequence[dict[str, Any]], model_version: str,
                 thresholds: dict[str, int]) -> dict[str, Any]:
    """inputs: 요청 항목(dict), predictions: 같은 순서의 판정 결과(grade, severity_score, probs, bias)."""
    now = datetime.now(timezone.utc)
    items = []
    for idx, (inp, pred) in enumerate(zip(inputs, predictions)):
        missing = [f for f in OPTIONAL_FIELDS if not inp.get(f)]
        items.append({
            "input_index": idx,
            "text": inp["text"],
            "text_sha256": sha256_text(inp["text"]),
            "grade": pred["grade"],
            "severity_score": pred["severity_score"],
            "probs": pred["probs"],
            "bias": pred["bias"],
            "url": inp.get("url"),
            "posted_at": _iso(inp.get("posted_at")),
            "author_handle": inp.get("author_handle"),
            "captured_at": _iso(inp.get("captured_at")),
            "screenshot_sha256": inp.get("screenshot_sha256"),
            "missing_fields": missing,
        })
    # 심각도 내림차순, 같으면 입력 순서 (정렬이 안정적이라 같은 입력이면 같은 결과)
    items.sort(key=lambda it: (-it["severity_score"], it["input_index"]))
    for rank, it in enumerate(items, start=1):
        it["rank"] = rank

    # 시간대가 섞여 있을 수 있어서 문자열이 아니라 datetime으로 비교 (API는 시간대 있는 값만 받음)
    posted = [_iso(d) for d in sorted(inp["posted_at"] for inp in inputs if inp.get("posted_at"))]
    by_author: dict[str, dict[str, Any]] = {}
    for it in items:
        if it["author_handle"]:
            a = by_author.setdefault(it["author_handle"], {"author_handle": it["author_handle"],
                                                          "count": 0, "max_severity_score": 0})
            a["count"] += 1
            a["max_severity_score"] = max(a["max_severity_score"], it["severity_score"])
    grade_counts = Counter(it["grade"] for it in items)
    missing_counts = Counter(f for it in items for f in it["missing_fields"])

    return {
        "report_id": str(uuid.uuid4()),
        "case_title": case_title,
        "generated_at": {"kst": now.astimezone(KST).isoformat(timespec="seconds"),
                         "utc": now.isoformat(timespec="seconds")},
        "model_version": model_version,
        "severity_thresholds": thresholds,
        "summary": {
            "total": len(items),
            "grade_counts": {g: grade_counts.get(g, 0) for g in ("severe", "caution", "normal")},
            "period": {"earliest_posted_at": posted[0] if posted else None,
                       "latest_posted_at": posted[-1] if posted else None},
            "authors": sorted(by_author.values(), key=lambda a: (-a["count"], -a["max_severity_score"])),
        },
        "warnings": [f"{n}건에 {f} 값이 없습니다" for f, n in sorted(missing_counts.items())],
        "items": items,
        "report_sha256": canonical_sha256(items),
        "evidence_checklist": _checklist(items),
        "notices": [DISCLAIMER, NORMAL_NOTICE],
    }


CSV_COLUMNS = ("rank", "grade", "severity_score", "text", "prob_none", "prob_offensive", "prob_hate",
               "bias_type", "contain_gender_bias", "url", "posted_at", "author_handle", "captured_at",
               "screenshot_sha256", "text_sha256", "missing_fields")


def report_to_csv(report: dict[str, Any]) -> str:
    """UTF-8 BOM + '#' 주석 행(헤더·요약·고지) + 항목 1건당 1행."""
    buf = io.StringIO()
    buf.write("﻿")
    s = report["summary"]
    comments = [
        f"report_id: {report['report_id']}",
        f"case_title: {report['case_title'] or ''}",
        f"generated_at: {report['generated_at']['kst']} (KST) / {report['generated_at']['utc']} (UTC)",
        f"model_version: {report['model_version']}",
        f"severity_thresholds: caution>={report['severity_thresholds']['caution']}, "
        f"severe>={report['severity_thresholds']['severe']}",
        f"total: {s['total']} / " + ", ".join(f"{g} {n}" for g, n in s["grade_counts"].items()),
        f"period: {s['period']['earliest_posted_at'] or '-'} ~ {s['period']['latest_posted_at'] or '-'}",
        f"report_sha256: {report['report_sha256']}",
        *[f"notice: {n}" for n in report["notices"]],
    ]
    for line in comments:
        buf.write("# " + line.replace("\n", " ") + "\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for it in report["items"]:
        w.writerow([it["rank"], it["grade"], it["severity_score"], it["text"],
                    it["probs"]["none"], it["probs"]["offensive"], it["probs"]["hate"],
                    it["bias"]["type"], it["bias"]["contain_gender_bias"], it["url"] or "",
                    it["posted_at"] or "", it["author_handle"] or "", it["captured_at"] or "",
                    it["screenshot_sha256"] or "", it["text_sha256"], ";".join(it["missing_fields"])])
    return buf.getvalue()
