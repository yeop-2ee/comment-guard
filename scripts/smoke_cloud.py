"""배포된(또는 로컬) API에 T1~T11을 실제 HTTP로 보내고 결과를 기록합니다 (명세서 5.3, 6.4장).

    # 로컬
    python scripts/smoke_cloud.py --url http://127.0.0.1:8080 --target local
    # 클라우드 (배포한 주소, 배포 정보는 기록용)
    python scripts/smoke_cloud.py --url https://<배포 주소> --target cloud \
        --region <리전> --revision <배포 ID> --model-uri <모델 경로>

입력 댓글은 공개 데이터(data/raw/test.no_label.tsv)에서만 가져옵니다.
결과: runs/<run_id>/<target>_smoke.json. 하나라도 실패하면 종료 코드 1.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from commentguard.data import read_tsv  # noqa: E402
from commentguard.runlog import mask, rel, run_step  # noqa: E402


def public_comments(n: int) -> list[str]:
    df = read_tsv(ROOT / "data" / "raw" / "test.no_label.tsv")
    return [t for t in df["comments"] if 5 <= len(t) <= 100][:n]


class Smoke:
    def __init__(self, base: str, timeout: float):
        self.base, self.timeout = base.rstrip("/"), timeout
        self.cases: list[dict[str, Any]] = []

    def call(self, method: str, path: str, body: Any = None) -> tuple[requests.Response, float]:
        start = time.perf_counter()
        r = requests.request(method, self.base + path, json=body, timeout=self.timeout)
        return r, round((time.perf_counter() - start) * 1000, 1)

    def case(self, cid: str, desc: str, method: str, path: str, body: Any,
             check: Callable[[requests.Response], str | None], record_body: bool = True) -> requests.Response | None:
        try:
            r, ms = self.call(method, path, body)
            problem = check(r)
            try:
                resp: Any = r.json()
            except ValueError:
                resp = r.text[:2000]
            entry = {"id": cid, "desc": desc, "method": method, "path": path,
                     "request": body if record_body else "(생략: 크기가 큼, 공개 데이터 댓글)",
                     "status": r.status_code, "latency_ms": ms, "response": resp,
                     "passed": problem is None, "problem": problem}
        except Exception as e:  # 연결 실패 등
            r = None
            entry = {"id": cid, "desc": desc, "method": method, "path": path, "status": None,
                     "passed": False, "problem": f"{type(e).__name__}: {e}"}
        self.cases.append(entry)
        print(f"{'PASS' if entry['passed'] else 'FAIL'} {cid} {desc} "
              f"[{entry.get('status')}, {entry.get('latency_ms')}ms] {entry['problem'] or ''}")
        return r


def expect(status: int, extra: Callable[[requests.Response], str | None] | None = None):
    def check(r: requests.Response) -> str | None:
        if r.status_code != status:
            return f"status {r.status_code} != {status}"
        return extra(r) if extra else None
    return check


def run_cases(s: Smoke, comments: list[str]) -> dict[str, Any]:
    health = s.case("T11", "/health 200, model_version 확인", "GET", "/health", None,
                    expect(200, lambda r: None if r.json().get("status") == "ok" else "status != ok"))
    version = health.json().get("model_version") if health is not None and health.ok else None

    def t1(r):
        res = r.json()["results"][0]
        if r.json()["model_version"] != version:
            return "model_version이 /health와 다름"
        if abs(sum(res["probs"].values()) - 1) > 1e-3:
            return "확률 합이 1이 아님"
        missing = {"label", "probs", "severity_score", "grade", "bias"} - set(res)
        return f"필드 없음: {missing}" if missing else None
    s.case("T1", "정상 댓글 1건", "POST", "/v1/classify", {"texts": [comments[1]]}, expect(200, t1))

    batch = comments[:100]
    def t2(r):
        res = r.json()["results"]
        if len(res) != 100 or [x["index"] for x in res] != list(range(100)):
            return "결과 수 또는 순서가 다름"
        return None
    s.case("T2", "100건 배치, 순서 유지", "POST", "/v1/classify", {"texts": batch}, expect(200, t2),
           record_body=False)

    s.case("T3", "이모지·영어·초성 섞인 댓글", "POST", "/v1/classify",
           {"texts": [comments[2] + " 😂👍 lol ㅋㅋㅋ ㅎㅇㅌ"]}, expect(200))

    def has_index(i):
        return lambda r: None if i in [d.get("index") for d in r.json().get("detail", [])] else f"index {i} 없음"
    s.case("T4", "빈 문자열·공백만 → 422, index 포함", "POST", "/v1/classify",
           {"texts": [comments[0], "   "]}, expect(422, has_index(1)))
    s.case("T5", "1,001자 → 422", "POST", "/v1/classify", {"texts": ["가" * 1001]}, expect(422, has_index(0)),
           record_body=False)
    s.case("T6", "101건 → 422", "POST", "/v1/classify", {"texts": [comments[0]] * 101}, expect(422),
           record_body=False)
    s.case("T7", "잘못된 민감도 → 422", "POST", "/v1/filter",
           {"sensitivity": "extreme", "comments": [{"id": "c1", "text": comments[0]}]}, expect(422))

    s.case("F", "필터 medium 3건", "POST", "/v1/filter",
           {"sensitivity": "medium", "comments": [{"id": f"c{i}", "text": t} for i, t in enumerate(comments[3:6])]},
           expect(200, lambda r: None if sum(r.json()["summary"].values()) == 3 else "summary 합계 오류"))

    report_body = {"case_title": "스모크 테스트 (공개 데이터)", "items": [{"text": t} for t in comments[6:9]]}
    first = s.case("T8", "리포트 text만 → missing_fields 경고", "POST", "/v1/evidence/report", report_body,
                   expect(200, lambda r: None if all(i["missing_fields"] for i in r.json()["items"])
                          and r.json()["warnings"] else "missing_fields/warnings 없음"))

    def t9(r):
        if first is None or not first.ok:
            return "T8 응답 없음"
        a, b = first.json(), r.json()
        same = sorted(i["text_sha256"] for i in a["items"]) == sorted(i["text_sha256"] for i in b["items"])
        if not same or a["report_sha256"] != b["report_sha256"]:
            return "해시가 다름"
        return None
    s.case("T9", "리포트 같은 입력 2회 → 해시 동일", "POST", "/v1/evidence/report", report_body, expect(200, t9))

    def t10(r):
        if not r.content.startswith(b"\xef\xbb\xbf"):
            return "UTF-8 BOM 없음"
        lines = [ln for ln in r.content.decode("utf-8-sig").splitlines() if not ln.startswith("#")]
        rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
        return None if len(rows) == len(report_body["items"]) else f"행 수 {len(rows)}"
    s.case("T10", "리포트 CSV (BOM, 행 수)", "POST", "/v1/evidence/report?format=csv", report_body,
           expect(200, t10))

    s.case("META", "/v1/meta", "GET", "/v1/meta", None,
           expect(200, lambda r: None if r.json()["model_version"] == version else "버전 불일치"))
    return {"model_version": version}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--target", choices=["local", "cloud"], required=True)
    ap.add_argument("--region")
    ap.add_argument("--revision")
    ap.add_argument("--model-uri")
    ap.add_argument("--suffix", help="run_id 꼬리표 (예: incident-before)")
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()

    deploy = {"url": args.url, "region": args.region, "revision": args.revision,
              "model_uri": args.model_uri}
    s = Smoke(args.url, args.timeout)
    suffix = "-".join(x for x in (args.target, args.suffix) if x)
    try:
        with run_step("smoke", config={"target": args.target, **deploy}, suffix=suffix) as run:
            info = run_cases(s, public_comments(120))
            run.model_version = info["model_version"]
            passed = sum(c["passed"] for c in s.cases)
            result = {"target": args.target, **deploy, "model_version": info["model_version"],
                      "run_id": run.run_id, "git_commit": run.git_commit, "started_at": run.started_at,
                      "passed": passed, "total": len(s.cases), "cases": s.cases}
            out = run.dir / f"{args.target}_smoke.json"
            out.write_text(mask(json.dumps(result, ensure_ascii=False, indent=2)), encoding="utf-8")
            run.outputs.append(rel(out))
            run.extra.update(passed=passed, total=len(s.cases), **deploy)
            if passed != len(s.cases):
                raise RuntimeError(f"스모크 실패: {len(s.cases) - passed}/{len(s.cases)}건")
    except Exception as e:
        print(f"[smoke] 실패: {e}", file=sys.stderr)
        return 1
    print(f"[smoke] {passed}/{len(s.cases)} 통과 → {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
