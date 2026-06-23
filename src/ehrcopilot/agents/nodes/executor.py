"""Executor node: runs validated SQL against the read-only SQLite DB.

Also applies Layer 4 (small-cell suppression) on the returned rows.
"""

from __future__ import annotations

from ehrcopilot.agents.state import CopilotState
from ehrcopilot.db.connection import QueryTimeoutError, RowCapExceededError, execute_query
from ehrcopilot.guardrails.layers import GuardResult, validate_result


def executor_node(state: CopilotState) -> CopilotState:
    sql = state.get("sql")

    if not sql:
        return {**state, "exec_result": None, "exec_error": "No SQL to execute"}

    try:
        rows = execute_query(sql)
    except (QueryTimeoutError, RowCapExceededError) as exc:
        return {**state, "exec_result": None, "exec_error": str(exc)}
    except Exception as exc:
        return {**state, "exec_result": None, "exec_error": f"SQL execution error: {exc}"}

    # Layer 4 — small-cell suppression
    l4_result = validate_result(rows, sql)
    if not l4_result.passed:
        return {
            **state,
            "exec_result": None,
            "exec_error": None,
            "guard_result": l4_result,
            "abstain": True,
            "abstain_reason": l4_result.reason,
            "confidence": 0.0,
        }

    return {**state, "exec_result": rows, "exec_error": None}
