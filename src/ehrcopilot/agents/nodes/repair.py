"""Repair node: re-prompts SQL writer with error context and increments repair_count."""

from __future__ import annotations

from ehrcopilot import config
from ehrcopilot.agents.state import CopilotState


def repair_node(state: CopilotState) -> CopilotState:
    """Increment repair_count and clear SQL so sql_writer re-runs with error context."""
    repair_count = state.get("repair_count", 0) + 1

    guard_result = state.get("guard_result")
    exec_error = state.get("exec_error")

    error_context = (
        exec_error
        or (guard_result.reason if guard_result and not guard_result.passed else None)
        or "unknown error"
    )

    return {
        **state,
        "repair_count": repair_count,
        "exec_error": error_context,
        "sql": None,
        "exec_result": None,
    }
