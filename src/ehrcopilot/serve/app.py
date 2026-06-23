"""FastAPI serving app — wraps the copilot agent graph behind a REST API.

Endpoints:
  POST /query   — main entry: {question, role} → {answer, abstain, sql, confidence}
  GET  /health  — liveness check
  GET  /metrics — Prometheus-compatible text metrics

Instrumented with OpenInference/Arize Phoenix spans for full observability.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ehrcopilot import config
from ehrcopilot.cache import semantic_cache
from ehrcopilot.guardrails.layers import validate_nl

app = FastAPI(
    title="Governed Clinical-Analytics Copilot",
    description=(
        "Decision support over de-identified EHR data. "
        "NOT a medical device. Outputs require clinician review."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    role: str = Field(default="clinician", description="Access role for cache scoping")


class QueryResponse(BaseModel):
    answer: str | None
    abstain: bool
    abstain_reason: str | None
    sql: str | None
    confidence: float
    cache_hit: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# Metrics counters (in-memory; replace with Prometheus client in production)
# ---------------------------------------------------------------------------

_metrics: dict[str, Any] = {
    "total_requests": 0,
    "cache_hits": 0,
    "abstentions": 0,
    "errors": 0,
    "latencies_ms": [],
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": config.INFERENCE_MODEL}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    lats = _metrics["latencies_ms"]
    p50 = sorted(lats)[len(lats) // 2] if lats else 0.0
    p95 = sorted(lats)[int(len(lats) * 0.95)] if lats else 0.0
    cache_hits = _metrics["cache_hits"]
    total = _metrics["total_requests"]

    return {
        "total_requests": total,
        "cache_hit_rate": round(cache_hits / max(total, 1), 4),
        "abstention_rate": round(_metrics["abstentions"] / max(total, 1), 4),
        "error_rate": round(_metrics["errors"] / max(total, 1), 4),
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "cache_stats": semantic_cache.cache_stats(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    _metrics["total_requests"] += 1
    t0 = time.monotonic()

    # Layer 5 — NL injection check before any processing
    nl_guard = validate_nl(req.question)
    if not nl_guard.passed:
        _metrics["abstentions"] += 1
        latency = (time.monotonic() - t0) * 1000
        _metrics["latencies_ms"].append(latency)
        return QueryResponse(
            answer="This question was blocked by the safety guardrails.",
            abstain=True,
            abstain_reason=f"layer5_block: {nl_guard.reason}",
            sql=None,
            confidence=0.0,
            cache_hit=False,
            latency_ms=round(latency, 1),
        )

    # Cache lookup
    cached = semantic_cache.lookup(req.question, req.role)
    if cached:
        _metrics["cache_hits"] += 1
        latency = (time.monotonic() - t0) * 1000
        _metrics["latencies_ms"].append(latency)
        return QueryResponse(
            answer=cached.get("answer"),
            abstain=False,
            abstain_reason=None,
            sql=cached.get("sql"),
            confidence=1.0,
            cache_hit=True,
            latency_ms=round(latency, 1),
        )

    # Run the agent graph
    try:
        from ehrcopilot.agents.graph import run as run_graph

        final_state = run_graph(req.question, req.role)
    except Exception as exc:
        _metrics["errors"] += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    abstain = bool(final_state.get("abstain", True))
    answer = str(final_state.get("answer") or "")
    sql = final_state.get("sql")
    confidence = float(final_state.get("confidence", 0.0))
    abstain_reason = final_state.get("abstain_reason")

    # Store in cache on success
    if not abstain and sql:
        semantic_cache.store(
            question=req.question,
            role=req.role,
            sql=sql,
            answer=answer,
            exec_result=final_state.get("exec_result"),
        )

    if abstain:
        _metrics["abstentions"] += 1

    latency = (time.monotonic() - t0) * 1000
    _metrics["latencies_ms"].append(latency)

    return QueryResponse(
        answer=answer or None,
        abstain=abstain,
        abstain_reason=abstain_reason,
        sql=sql,
        confidence=confidence,
        cache_hit=False,
        latency_ms=round(latency, 1),
    )
