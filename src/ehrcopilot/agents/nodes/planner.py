"""Planner node: intent decomposition + schema-feasibility check."""

from __future__ import annotations

from ehrcopilot import config
from ehrcopilot.agents.state import CopilotState
from ehrcopilot.agents.nodes._llm import call_llm

_SYSTEM = """\
You are a clinical analytics assistant. Your job is to assess whether a natural-language
question can be answered using the MIMIC-IV-Demo database and to decompose its intent.

{schema}

Respond with EXACTLY this JSON (no prose, no markdown fences):
{{
  "intent": "<one-sentence decomposition of what the question is asking>",
  "answerable": true | false,
  "reason": "<why it is or isn't answerable — only needed when answerable=false>"
}}

A question is NOT answerable if:
- It references clinical concepts not captured in the schema (e.g. radiology images, genomics).
- It requires real-time data or data not in the demo subset.
- It is too vague to formulate a specific SQL query.
"""


def planner_node(state: CopilotState) -> CopilotState:
    import json

    question = state.get("question", "")
    schema_text = config.schema_to_prompt()

    raw = call_llm(
        system=_SYSTEM.format(schema=schema_text),
        user=question,
        max_new_tokens=256,
        temperature=0.0,
    )

    try:
        parsed = json.loads(raw)
        intent = str(parsed.get("intent", question))
        answerable = bool(parsed.get("answerable", True))
    except (json.JSONDecodeError, KeyError):
        # Degrade gracefully: treat as answerable, use raw text as intent
        intent = raw.strip()[:500]
        answerable = True

    return {**state, "intent": intent, "answerable": answerable}
