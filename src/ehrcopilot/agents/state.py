"""Typed state for the CopilotState LangGraph graph."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from ehrcopilot.guardrails.layers import GuardResult


class CopilotState(TypedDict, total=False):
    # Input
    question: str
    role: str                          # access role for cache scoping (e.g. "clinician", "researcher")

    # Planner output
    intent: str                        # decomposed intent
    answerable: bool                   # can the schema answer this question?

    # Schema linker output
    linked_schema: dict[str, list[str]]  # table -> [col, ...] subset

    # SQL writer output
    sql: str | None

    # Guardrail output
    guard_result: GuardResult

    # Executor output
    exec_result: list[dict[str, Any]] | None
    exec_error: str | None

    # Repair tracking
    repair_count: int                  # max config.MAX_REPAIR_ATTEMPTS

    # Reliability gate output
    confidence: float                  # 0.0 – 1.0

    # Final output
    answer: str | None
    abstain: bool
    abstain_reason: str | None

    # Cache hit flag (set by semantic cache before entering the graph)
    cache_hit: bool
