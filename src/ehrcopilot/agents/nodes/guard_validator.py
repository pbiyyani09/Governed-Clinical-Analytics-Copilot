"""Guard validator node: runs Layers 1, 2, 3 on the generated SQL."""

from __future__ import annotations

from ehrcopilot.agents.state import CopilotState
from ehrcopilot.guardrails.layers import GuardResult, validate_sql


def guard_validator_node(state: CopilotState) -> CopilotState:
    sql = state.get("sql")

    if not sql:
        # Model already abstained — propagate the abstain state
        return {
            **state,
            "guard_result": GuardResult.ok(),
        }

    result = validate_sql(sql)
    return {**state, "guard_result": result}
