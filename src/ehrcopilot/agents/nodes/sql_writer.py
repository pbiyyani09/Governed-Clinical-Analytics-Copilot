"""SQL writer node: generates SQL (or [ABSTAIN]) from intent + linked schema."""

from __future__ import annotations

from ehrcopilot import config
from ehrcopilot.agents.state import CopilotState
from ehrcopilot.agents.nodes._llm import call_llm

_SYSTEM = """\
You are a clinical analytics SQL expert. Convert the user's intent into a valid SQLite SELECT query.

Rules:
1. Output ONLY the SQL query — no explanation, no markdown, no backticks.
2. If the question cannot be answered with the available data, output exactly: [ABSTAIN]
3. Use only the tables and columns listed in the schema below.
4. Do not reference PHI columns (name, dob, ssn, address, phone, email, etc.).
5. Always add a LIMIT clause (max 1000 rows) for non-aggregate queries.
6. Use standard SQLite syntax (no window functions that SQLite <3.25 doesn't support).

{schema}
"""

_REPAIR_SUFFIX = "\n\nThe previous attempt produced an error. Reason: {error}\nPlease fix the SQL."


def sql_writer_node(state: CopilotState) -> CopilotState:
    intent = state.get("intent", state.get("question", ""))
    linked_schema = state.get("linked_schema") or {}
    repair_count = state.get("repair_count", 0)
    exec_error = state.get("exec_error")
    guard_result = state.get("guard_result")

    schema_text = config.schema_to_prompt(linked_schema)
    user_msg = intent

    if repair_count > 0:
        error_context = exec_error or (guard_result.reason if guard_result else "unknown error")
        prev_sql = state.get("sql") or ""
        user_msg = (
            f"Previous SQL attempt:\n{prev_sql}\n"
            + _REPAIR_SUFFIX.format(error=error_context)
            + f"\n\nOriginal intent: {intent}"
        )

    raw = call_llm(
        system=_SYSTEM.format(schema=schema_text),
        user=user_msg,
        max_new_tokens=512,
        temperature=0.0,
    )

    sql = raw.strip()
    abstain = sql == "[ABSTAIN]" or not sql

    return {
        **state,
        "sql": None if abstain else sql,
        "abstain": abstain,
        "abstain_reason": "model_abstained" if abstain else None,
        "exec_error": None,  # clear previous error before next guard check
    }
