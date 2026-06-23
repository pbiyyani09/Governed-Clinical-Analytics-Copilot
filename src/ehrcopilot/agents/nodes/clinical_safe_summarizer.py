"""Clinical-safe summarizer node: converts SQL rows into plain-English answers.

Never invents clinical interpretations. Always appends the de-identified-data disclaimer.
On abstain: returns a transparent refusal with the reason code.
"""

from __future__ import annotations

from ehrcopilot.agents.state import CopilotState
from ehrcopilot.agents.nodes._llm import call_llm

_DISCLAIMER = (
    "\n\n⚠ Decision support only: this figure reflects de-identified demo data. "
    "Verify with a clinician before acting on it. Not a medical device."
)

_ANSWER_SYSTEM = """\
You are a clinical analytics assistant. Convert the SQL result rows into a clear,
plain-English answer to the user's question.

Rules:
1. Be factual and precise — report numbers exactly as returned.
2. Do NOT invent clinical interpretations or causal claims.
3. Do NOT mention SQL, tables, or column names in the answer.
4. Keep the answer concise (2–4 sentences max).
5. Use clinical terminology appropriate for a healthcare professional.
"""

_ABSTAIN_MESSAGES: dict[str, str] = {
    "model_abstained": (
        "I cannot reliably answer this question with the available data. "
        "The question may reference clinical concepts not captured in the MIMIC-IV-Demo dataset, "
        "or may be too ambiguous to formulate a specific query."
    ),
    "small_cell_suppression": (
        "This query returned a very small number of results (fewer than the minimum cohort size "
        "required for privacy protection). The result has been suppressed to protect patient privacy."
    ),
    "execution_failed": (
        "The generated SQL query could not be executed successfully. "
        "Please rephrase your question or contact the administrator."
    ),
    "confidence_below_threshold": (
        "I'm not confident enough in this answer to report it. "
        "The question may be ambiguous or the available data insufficient."
    ),
    "upstream_abstain": (
        "I am unable to answer this question at this time."
    ),
}


def _format_rows(rows: list[dict], max_rows: int = 20) -> str:
    if not rows:
        return "No results returned."
    display = rows[:max_rows]
    lines = []
    for row in display:
        lines.append(", ".join(f"{k}: {v}" for k, v in row.items()))
    suffix = f"\n[... {len(rows) - max_rows} more rows not shown]" if len(rows) > max_rows else ""
    return "\n".join(lines) + suffix


def clinical_safe_summarizer_node(state: CopilotState) -> CopilotState:
    question = state.get("question", "")
    abstain = state.get("abstain", False)
    abstain_reason = state.get("abstain_reason") or "upstream_abstain"
    exec_result = state.get("exec_result")

    if abstain:
        # Match the reason prefix to a canned message
        message = _ABSTAIN_MESSAGES.get("upstream_abstain")
        for key, msg in _ABSTAIN_MESSAGES.items():
            if abstain_reason.startswith(key):
                message = msg
                break
        return {**state, "answer": message, "abstain": True}

    if not exec_result:
        return {
            **state,
            "answer": _ABSTAIN_MESSAGES["execution_failed"],
            "abstain": True,
        }

    rows_text = _format_rows(exec_result)
    user_prompt = f"Question: {question}\n\nData:\n{rows_text}"

    answer = call_llm(
        system=_ANSWER_SYSTEM,
        user=user_prompt,
        max_new_tokens=256,
        temperature=0.0,
    )

    return {**state, "answer": answer + _DISCLAIMER, "abstain": False}
