"""댓글 가드 API (명세서 2장).

로컬:  MODEL_URI 없이 실행하면 configs/serving.yaml이 가리키는 모델을 씁니다.
    uvicorn commentguard.api:app --port 8080
클라우드(Vercel): MODEL_URI=blob://models/<version>/ (Vercel Blob 비공개 저장소, 명세서 6.3장)
    Blob은 Vercel이 요청 헤더로 주는 OIDC 토큰으로 읽어서, blob:// 모델은 첫 요청에서 불러옵니다.

로그에는 댓글 원문을 남기지 않습니다. 요청 ID, 경로, 상태, 건수, 등급별 건수, 처리 시간,
모델 버전, report_sha256만 JSON 한 줄로 남깁니다 (런타임 로그에서 JSON 필드로 조회 가능).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import AfterValidator, AwareDatetime, BaseModel, Field, StringConstraints

from commentguard import model_store
from commentguard.report import build_report, report_to_csv
from commentguard.severity import CLASSES, scores_from_probs

MAX_TEXT_CHARS = 1000
MAX_BATCH = 100
MAX_REPORT_ITEMS = 500
INDEX_HTML = Path(__file__).parent / "static" / "index.html"

log = logging.getLogger("commentguard")


def _setup_logging() -> None:
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = True  # 테스트(caplog)에서도 잡히게


def log_event(event: str, severity: str = "INFO", **fields: Any) -> None:
    # 원문이 들어갈 수 있는 필드는 여기에 넘기지 않습니다 (호출하는 쪽 규칙 + tests/test_no_text_in_logs.py)
    record = {"severity": severity, "event": event, **fields}
    log.log(logging.ERROR if severity == "ERROR" else logging.INFO, json.dumps(record, ensure_ascii=False))


def _check_text(v: str) -> str:
    if not v.strip():
        raise ValueError("공백만 있는 댓글은 판정할 수 없습니다")
    if len(v) > MAX_TEXT_CHARS:
        # 잘라서 판정하면 결과가 왜곡될 수 있어서 거부 (명세서 2.2장)
        raise ValueError(f"댓글은 {MAX_TEXT_CHARS}자 이하여야 합니다 (현재 {len(v)}자)")
    return v


Text = Annotated[str, AfterValidator(_check_text)]
ModelParam = Annotated[str | None, Query(max_length=100, description="모델 ID 또는 버전. 없으면 기본 모델")]


class ClassifyRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH)


class FilterComment(BaseModel):
    id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    text: Text


class FilterRequest(BaseModel):
    sensitivity: Literal["low", "medium", "high"] = "medium"
    comments: list[FilterComment] = Field(min_length=1, max_length=MAX_BATCH)


class ReportItem(BaseModel):
    text: Text
    url: Annotated[str, StringConstraints(max_length=2000)] | None = None
    posted_at: AwareDatetime | None = None
    author_handle: Annotated[str, StringConstraints(max_length=200)] | None = None
    captured_at: AwareDatetime | None = None
    screenshot_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-fA-F]{64}$")] | None = None


class ReportRequest(BaseModel):
    case_title: Annotated[str, StringConstraints(max_length=200)] | None = None
    items: list[ReportItem] = Field(min_length=1, max_length=MAX_REPORT_ITEMS)


def predict(model: model_store.CommentModel, texts: list[str]) -> list[dict[str, Any]]:
    hate_p, bias_p = model.predict_proba(texts)
    sev = model.severity
    scores = scores_from_probs(hate_p)
    out = []
    for i in range(len(texts)):
        bias_type = model_store.BIAS_CLASSES[int(bias_p[i].argmax())]
        out.append({
            "index": i,
            "label": CLASSES[int(hate_p[i].argmax())],
            "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, hate_p[i])},
            "severity_score": int(scores[i]),
            "grade": sev.grade(int(scores[i])),
            "bias": {"type": bias_type, "contain_gender_bias": bias_type == "gender"},
        })
    return out


def _display_name(card: dict[str, Any]) -> str:
    mid = card.get("model_id", "m1")
    if card.get("name"):
        return f"{mid.upper()} · {card['name']}"
    if card.get("members"):
        return f"{mid.upper()} · " + " + ".join(m["model_id"].upper() for m in card["members"]) + " 앙상블"
    base = (card.get("base_model") or {}).get("name", "").split("/")[-1]
    return f"{mid.upper()} · {base or 'TF-IDF + 로지스틱 회귀'}"


class ModelRegistry:
    """고를 수 있는 모델 (명세서 4.6장). 기본 모델은 바로 쓰고, 나머지는 처음 선택될 때 불러옵니다."""

    def __init__(self, default: Any, extra: list[Any] | None = None, dirs: list[Path] | None = None):
        self.default = default
        self._loaded = {m.version: m for m in [default, *(extra or [])]}
        self._dirs: dict[str, Path] = {}
        self._cards = {m.version: m.card for m in self._loaded.values()}
        self._lock = threading.Lock()
        for d in dirs or []:
            card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
            if card["model_version"] not in self._cards:
                self._cards[card["model_version"]] = card
                self._dirs[card["model_version"]] = d

    def list(self) -> list[dict[str, Any]]:
        rows = [{"model_id": c.get("model_id", "m1"), "model_version": v, "name": _display_name(c),
                 "default": v == self.default.version, "loaded": v in self._loaded,
                 "thresholds": (c.get("severity") or {}).get("thresholds")} for v, c in self._cards.items()]
        return sorted(rows, key=lambda r: r["model_id"])

    def get(self, name: str | None) -> Any:
        """name은 모델 버전 또는 모델 ID. 없으면 기본 모델. 모르는 이름이면 KeyError."""
        if not name:
            return self.default
        version = name if name in self._cards else next(
            (v for v, c in self._cards.items() if c.get("model_id", "m1") == name), None)
        if version is None:
            raise KeyError(name)
        with self._lock:  # 같은 모델을 동시에 두 번 올리지 않게
            if version not in self._loaded:
                self._loaded[version] = model_store.load_cached(self._dirs[version])
                log_event("model_loaded", model_version=version, on_demand=True)
        return self._loaded[version]


class UnknownModel(Exception):
    pass


def create_app(model: model_store.CommentModel | None = None, load_on_startup: bool = True,
               extra_models: list[Any] | None = None) -> FastAPI:
    """model을 넘기면 그 모델을 쓰고(테스트), 아니면 시작할 때 MODEL_URI에서 로드합니다."""
    _setup_logging()

    load_lock = threading.Lock()

    def load(uri: str | None) -> None:
        try:
            app.state.model = model_store.load_from_uri(uri)
        except model_store.ModelLoadError as e:
            # 빈 모델로 조용히 뜨지 않음 → 로컬은 시작 실패, 클라우드는 /health 503으로 드러남
            log_event("model_load_failed", severity="ERROR", model_uri=uri, error=str(e))
            raise
        log_event("model_loaded", model_uri=uri, model_version=app.state.model.version)
        if not uri:  # 로컬: models/LATEST* 의 다른 모델도 고를 수 있게 (클라우드는 MODEL_URI 하나만)
            app.state.registry = ModelRegistry(app.state.model, dirs=model_store.local_candidates())
        else:
            app.state.registry = ModelRegistry(app.state.model)

    def ensure_model(request: Request) -> None:
        """blob:// 모델은 첫 요청에서 불러옵니다. 실패하면 다음 요청에서 다시 시도합니다."""
        if app.state.model is not None or not app.state.lazy_uri:
            return
        with load_lock:
            if app.state.model is None:
                from vercel.headers import headers_from_asgi_scope, set_headers
                set_headers(headers_from_asgi_scope(request.scope))  # OIDC 토큰(x-vercel-oidc-token)을 SDK에 넘김
                load(app.state.lazy_uri)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.model is None and load_on_startup:
            uri = os.environ.get("MODEL_URI")
            if uri and uri.startswith(model_store.BLOB_SCHEME):
                app.state.lazy_uri = uri
            else:
                load(uri)
        if app.state.model is not None and app.state.registry is None:
            app.state.registry = ModelRegistry(app.state.model, extra=extra_models)
        yield

    app = FastAPI(title="Comment Guard", version="0.1.0", lifespan=lifespan)
    app.state.model = model
    app.state.registry = None
    app.state.lazy_uri = None

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.log = {}
        start = time.perf_counter()
        status = 500
        try:
            try:
                ensure_model(request)
            except model_store.ModelLoadError:
                status = 503
                return JSONResponse(status_code=503, content={"status": "unavailable", "error": "model_load_failed",
                                                              "model_version": None}, headers={"x-request-id": request_id})
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            m = app.state.model
            fields = {"model_version": m.version if m else None, **request.state.log}  # 고른 모델이 있으면 그 버전
            log_event("request", request_id=request_id, method=request.method, path=request.url.path,
                      status=status, latency_ms=round((time.perf_counter() - start) * 1000, 1), **fields)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # 입력값(원문)을 응답에 되돌려주지 않고, 문제 항목의 index만 알려줌
        errors = []
        for e in exc.errors():
            loc = [p for p in e["loc"] if p != "body"]
            index = next((p for p in loc if isinstance(p, int)), None)
            errors.append({"loc": loc, "index": index, "msg": e["msg"], "type": e["type"]})
        request.state.log["validation_errors"] = len(errors)
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": errors})

    @app.exception_handler(UnknownModel)
    async def unknown_model(request: Request, exc: UnknownModel):
        return JSONResponse(status_code=404, content={
            "error": "unknown_model", "available": [m["model_version"] for m in request.app.state.registry.list()]})

    def _model(request: Request, name: str | None = None) -> model_store.CommentModel:
        if not name:
            return request.app.state.model
        try:
            m = request.app.state.registry.get(name)
        except KeyError:
            raise UnknownModel(name) from None
        request.state.log["model_version"] = m.version
        return m

    @app.get("/", include_in_schema=False)
    def index():
        # 웹 화면 (명세서 2.6장). 판정은 화면이 /v1/* 를 호출해서 받음
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")

    @app.get("/health")
    def health(request: Request):
        m = _model(request)
        if m is None:
            return JSONResponse(status_code=503, content={"status": "unavailable", "model_version": None})
        return {"status": "ok", "model_version": m.version}

    @app.get("/v1/models")
    def models(request: Request):
        return {"default": request.app.state.model.version, "models": request.app.state.registry.list()}

    @app.get("/v1/meta")
    def meta(request: Request, model: ModelParam = None):
        card = _model(request, model).card
        return {
            "model_version": card["model_version"],
            "model_id": card.get("model_id", "m1"),
            "name": _display_name(card),
            "git_commit": card.get("git_commit"),
            "git_dirty": card.get("git_dirty"),
            "data": card.get("data"),
            "severity": card.get("severity"),
            "valid_metrics": card.get("valid_metrics"),
            "eval_metrics": card.get("eval_metrics"),
            "limits": {"max_batch": MAX_BATCH, "max_text_chars": MAX_TEXT_CHARS,
                       "max_report_items": MAX_REPORT_ITEMS},
        }

    @app.post("/v1/classify")
    def classify(req: ClassifyRequest, request: Request, model: ModelParam = None):
        m = _model(request, model)
        results = predict(m, req.texts)
        request.state.log.update(n_items=len(results), grade_counts=dict(Counter(r["grade"] for r in results)))
        return {"model_version": m.version, "results": results}

    @app.post("/v1/filter")
    def filter_comments(req: FilterRequest, request: Request, model: ModelParam = None):
        m = _model(request, model)
        preds = predict(m, [c.text for c in req.comments])
        sev = m.severity
        results = [{"id": c.id, "action": sev.action(p["grade"], req.sensitivity), "grade": p["grade"],
                    "severity_score": p["severity_score"]} for c, p in zip(req.comments, preds)]
        summary = {a: sum(r["action"] == a for r in results) for a in ("show", "blur", "hide")}
        request.state.log.update(n_items=len(results), sensitivity=req.sensitivity, actions=summary)
        return {"model_version": m.version, "sensitivity": req.sensitivity, "results": results,
                "summary": summary}

    @app.post("/v1/evidence/report")
    def evidence_report(req: ReportRequest, request: Request,
                        format: Literal["json", "csv"] = Query("json"), model: ModelParam = None):
        m = _model(request, model)
        inputs = [it.model_dump() for it in req.items]
        preds = predict(m, [it["text"] for it in inputs])
        sev = m.severity
        report = build_report(req.case_title, inputs, preds, m.version,
                              {"caution": sev.caution, "severe": sev.severe})
        request.state.log.update(n_items=len(inputs), grade_counts=report["summary"]["grade_counts"],
                                 report_sha256=report["report_sha256"], format=format)
        if format == "csv":
            return Response(content=report_to_csv(report).encode("utf-8"),
                            media_type="text/csv; charset=utf-8",
                            headers={"content-disposition": f'attachment; filename="report-{report["report_id"]}.csv"'})
        return report

    return app


app = create_app()
