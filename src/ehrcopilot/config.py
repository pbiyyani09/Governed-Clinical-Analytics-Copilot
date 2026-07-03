"""Central configuration: schema allowlist, PHI denylist, and runtime thresholds.

All guardrail layers read from this module so thresholds can be tuned in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SQLITE_DB_PATH = DATA_DIR / "ehrsql2024/mimic_iv/mimic_iv.sqlite"
EHRSQL_DATA_DIR = DATA_DIR / "ehrsql2024/mimic_iv"

# ---------------------------------------------------------------------------
# Schema allowlist — only these tables/columns are permitted through Layer 2.
# Derived from the EHRSQL 2024 MIMIC-IV SQLite schema (mimic_iv.sqlite).
#
# This is a custom MIMIC-IV schema designed for the EHRSQL 2024 shared task.
# It uses MIMIC-IV table/column naming (stay_id, icd_code, starttime, careunit)
# and includes tables absent from MIMIC-IV-Demo: cost, inputevents, outputevents.
# ---------------------------------------------------------------------------

SCHEMA_ALLOWLIST: dict[str, list[str]] = {
    "patients": [
        "row_id",
        "subject_id",
        "gender",
        "dob",
        "dod",
    ],
    "admissions": [
        "row_id",
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "age",
    ],
    "d_icd_diagnoses": [
        "row_id",
        "icd_code",
        "long_title",
    ],
    "d_icd_procedures": [
        "row_id",
        "icd_code",
        "long_title",
    ],
    "d_labitems": [
        "row_id",
        "itemid",
        "label",
    ],
    "d_items": [
        "row_id",
        "itemid",
        "label",
        "abbreviation",
        "linksto",
    ],
    "diagnoses_icd": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icd_code",
        "charttime",
    ],
    "procedures_icd": [
        "row_id",
        "subject_id",
        "hadm_id",
        "icd_code",
        "charttime",
    ],
    "labevents": [
        "row_id",
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ],
    "chartevents": [
        "row_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ],
    "prescriptions": [
        "row_id",
        "subject_id",
        "hadm_id",
        "starttime",
        "stoptime",
        "drug",
        "dose_val_rx",
        "dose_unit_rx",
        "route",
    ],
    "icustays": [
        "row_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
    ],
    "transfers": [
        "row_id",
        "subject_id",
        "hadm_id",
        "transfer_id",
        "eventtype",
        "careunit",
        "intime",
        "outtime",
    ],
    "microbiologyevents": [
        "row_id",
        "subject_id",
        "hadm_id",
        "charttime",
        "spec_type_desc",
        "test_name",
        "org_name",
    ],
    "inputevents": [
        "row_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "starttime",
        "itemid",
        "totalamount",
        "totalamountuom",
    ],
    "outputevents": [
        "row_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "value",
        "valueuom",
    ],
    "cost": [
        "row_id",
        "subject_id",
        "hadm_id",
        "event_type",
        "event_id",
        "chargetime",
        "cost",
    ],
}

# All valid table names — derived from allowlist
ALLOWED_TABLES: frozenset[str] = frozenset(SCHEMA_ALLOWLIST.keys())

# ---------------------------------------------------------------------------
# PHI column denylist — hard block in Layer 3, regardless of table.
# Based on HIPAA Safe Harbor identifiers applicable to the MIMIC schema.
# dob is in the DB but is blocked — queries should use admissions.age instead.
# ---------------------------------------------------------------------------

PHI_COLUMNS: frozenset[str] = frozenset(
    {
        "dob",
        "date_of_birth",
        "name",
        "first_name",
        "last_name",
        "ssn",
        "social_security",
        "phone",
        "fax",
        "email",
        "address",
        "street",
        "city",
        "state",
        "zip",
        "zipcode",
        "geo",
        "account_number",
        "mrn",
        "medical_record_number",
        "certificate",
        "license",
        "vehicle",
        "device",
        "web_url",
        "url",
        "ip_address",
        "biometric",
        "photo",
        "note",
        "text",
    }
)

# ---------------------------------------------------------------------------
# k-Anonymity threshold (Layer 4 — small-cell suppression)
# k=11 follows UK NHS standard.
# k=5 follows US HIPAA Safe Harbor rough guide.
# Set via env var EHRCOPILOT_K_THRESHOLD to override at runtime.
# ---------------------------------------------------------------------------

K_THRESHOLD: int = int(os.getenv("EHRCOPILOT_K_THRESHOLD", "11"))

# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

CACHE_COSINE_THRESHOLD: float = float(os.getenv("EHRCOPILOT_CACHE_THRESHOLD", "0.90"))
CACHE_TTL_SECONDS: int = int(os.getenv("EHRCOPILOT_CACHE_TTL", str(24 * 3600)))  # 24h
CACHE_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

# Bump this when schema changes to invalidate all cache entries.
DB_VERSION: int = 2

# ---------------------------------------------------------------------------
# Agent / inference
# ---------------------------------------------------------------------------

MAX_REPAIR_ATTEMPTS: int = 3
QUERY_TIMEOUT_SECONDS: int = int(os.getenv("EHRCOPILOT_QUERY_TIMEOUT", "10"))
MAX_ROWS: int = int(os.getenv("EHRCOPILOT_MAX_ROWS", "1000"))
MAX_SEQ_LENGTH: int = 1536

# Model for inference (overridden in serve config for AWQ path)
INFERENCE_MODEL: str = os.getenv(
    "EHRCOPILOT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
)

# Reliability gate confidence threshold — tune on dev set
CONFIDENCE_THRESHOLD: float = float(os.getenv("EHRCOPILOT_CONF_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Schema prompt helper
# ---------------------------------------------------------------------------

# Foreign key relationships — shown in prompts to help the model find join paths.
_FK_HINTS: str = (
    "Foreign keys: "
    "admissions.subject_id→patients.subject_id | "
    "icustays.hadm_id→admissions.hadm_id | "
    "icustays.subject_id→patients.subject_id | "
    "diagnoses_icd.hadm_id→admissions.hadm_id | "
    "diagnoses_icd.icd_code=d_icd_diagnoses.icd_code | "
    "procedures_icd.hadm_id→admissions.hadm_id | "
    "procedures_icd.icd_code=d_icd_procedures.icd_code | "
    "labevents.hadm_id→admissions.hadm_id | "
    "labevents.itemid→d_labitems.itemid | "
    "prescriptions.hadm_id→admissions.hadm_id | "
    "chartevents.stay_id→icustays.stay_id | "
    "chartevents.itemid→d_items.itemid | "
    "microbiologyevents.hadm_id→admissions.hadm_id | "
    "inputevents.stay_id→icustays.stay_id | "
    "inputevents.itemid→d_items.itemid | "
    "outputevents.stay_id→icustays.stay_id | "
    "outputevents.itemid→d_items.itemid | "
    "cost.hadm_id→admissions.hadm_id | "
    "transfers.hadm_id→admissions.hadm_id"
)

# Schema shown to the model excludes PHI columns (dob is blocked by Layer 3).
_PROMPT_SCHEMA: dict[str, list[str]] = {
    tbl: [c for c in cols if c not in PHI_COLUMNS]
    for tbl, cols in SCHEMA_ALLOWLIST.items()
}


def schema_to_prompt(linked_schema: dict[str, list[str]] | None = None) -> str:
    """Serialize schema (or a linked subset) into a compact prompt string with FK hints."""
    source = linked_schema if linked_schema else _PROMPT_SCHEMA
    lines: list[str] = ["Database schema (EHRSQL-MIMIC-IV):"]
    for table, cols in source.items():
        lines.append(f"  {table}({', '.join(cols)})")
    lines.append(_FK_HINTS)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unified system prompt — used identically in training (build_pairs.py) and
# eval (harness.py). Must be kept in sync so the model sees the same context
# it was trained on.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a clinical analytics assistant. Convert the user's question into "
    "a valid SQLite SELECT query over the EHRSQL-MIMIC-IV database.\n\n"
    "Output exactly [ABSTAIN] — nothing else — when the question asks about "
    "information NOT derivable from the schema below. This includes: "
    "doctor or provider identities, family history, drug side effects or "
    "pharmacology, future or scheduled hospital visits, patient contact "
    "information, or any concept not represented by the tables and columns "
    "listed below.\n\n"
    "Otherwise output only the SQL query with no explanation.\n\n"
    + schema_to_prompt()
)
