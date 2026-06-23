"""Schema linker node: BM25 retrieval to select the minimal relevant schema subset.

Keeps prompts short enough to fit within the 1536-token budget on 16 GB VRAM.
"""

from __future__ import annotations

import re

from ehrcopilot import config
from ehrcopilot.agents.state import CopilotState

# Table descriptions for BM25 scoring — concise semantic labels
_TABLE_DESCRIPTIONS: dict[str, str] = {
    "patients": "demographics gender age year of death anchor",
    "admissions": "hospital admission discharge insurance race marital language expire",
    "diagnoses_icd": "diagnosis ICD code version sequence",
    "d_icd_diagnoses": "ICD diagnosis code description long title",
    "procedures_icd": "procedure ICD code version chart date",
    "d_icd_procedures": "ICD procedure code description long title",
    "labevents": "laboratory lab test result value unit reference flag priority",
    "d_labitems": "lab item label fluid category",
    "prescriptions": "medication drug prescription dose route strength pharmacy",
    "icustays": "ICU intensive care unit stay length of stay careunit",
    "chartevents": "chart vital signs measurement value warning",
    "d_items": "item label abbreviation category unit parameter",
    "microbiologyevents": "microbiology culture specimen organism antibiotic sensitivity",
    "pharmacy": "pharmacy medication dispensation fill",
    "transfers": "transfer careunit event type movement",
    "edstays": "emergency department ED stay disposition transport",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 10.0,
) -> float:
    """Simplified BM25 score for a query against a document."""
    dl = len(doc_tokens)
    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    score = 0.0
    for term in set(query_tokens):
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        idf = 1.0  # simplified: no corpus-level IDF needed for a 16-table schema
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf * numerator / denominator

    return score


def link_schema(intent: str, top_k: int = 6) -> dict[str, list[str]]:
    """Return the top-k most relevant tables and their allowed columns for the given intent."""
    query_tokens = _tokenize(intent)

    scores: list[tuple[float, str]] = []
    for table, description in _TABLE_DESCRIPTIONS.items():
        doc_tokens = _tokenize(description)
        s = _bm25_score(query_tokens, doc_tokens)
        scores.append((s, table))

    scores.sort(reverse=True)
    selected_tables = [t for _, t in scores[:top_k] if _ > 0]

    # Always include the core join tables if any table was selected
    if selected_tables and "patients" not in selected_tables:
        selected_tables.append("patients")
    if selected_tables and "admissions" not in selected_tables:
        selected_tables.append("admissions")

    # Fallback: if nothing scored, return all tables
    if not selected_tables:
        return dict(config.SCHEMA_ALLOWLIST)

    return {t: config.SCHEMA_ALLOWLIST[t] for t in selected_tables if t in config.SCHEMA_ALLOWLIST}


def schema_linker_node(state: CopilotState) -> CopilotState:
    intent = state.get("intent", state.get("question", ""))
    linked = link_schema(intent)
    return {**state, "linked_schema": linked}
