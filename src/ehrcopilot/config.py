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

# EHRSQL-2024 (ClinicalNLP@NAACL 2024) — MIMIC-IV Demo with the EHRSQL-preprocessed
# schema. This is the apples-to-apples benchmark target (leaderboard RS(10)=0.8132).
# The DB and the question/SQL splits both live under data/ehrsql2024/.
EHRSQL2024_DIR = DATA_DIR / "ehrsql2024" / "mimic_iv"
SQLITE_DB_PATH = EHRSQL2024_DIR / "mimic_iv.sqlite"
# Legacy EHRSQL-2022 (MIMIC-III + eICU) tree — retained for reference, NOT the target.
EHRSQL_DATA_DIR = DATA_DIR / "ehrsql"

# ---------------------------------------------------------------------------
# Schema allowlist — only these tables/columns are permitted through Layer 2.
# EHRSQL-2024 schema (MIMIC-IV Demo, EHRSQL-preprocessed). Column names match
# data/ehrsql2024/mimic_iv/mimic_iv.sqlite EXACTLY. NB: this is a flattened,
# de-identified schema — patients.dob is a synthesized year (anchor_year-anchor_age),
# there is no race/ethnicity column, ICU events use stay_id, ICD uses icd_code (no
# version) — distinct from the raw MIMIC-IV-Demo schema.
# ---------------------------------------------------------------------------

SCHEMA_ALLOWLIST: dict[str, list[str]] = {
    "patients": ["row_id", "subject_id", "gender", "dob", "dod"],
    "admissions": [
        "row_id", "subject_id", "hadm_id", "admittime", "dischtime",
        "admission_type", "admission_location", "discharge_location",
        "insurance", "language", "marital_status", "age",
    ],
    "d_icd_diagnoses": ["row_id", "icd_code", "long_title"],
    "d_icd_procedures": ["row_id", "icd_code", "long_title"],
    "d_labitems": ["row_id", "itemid", "label"],
    "d_items": ["row_id", "itemid", "label", "abbreviation", "linksto"],
    "diagnoses_icd": ["row_id", "subject_id", "hadm_id", "icd_code", "charttime"],
    "procedures_icd": ["row_id", "subject_id", "hadm_id", "icd_code", "charttime"],
    "labevents": [
        "row_id", "subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom",
    ],
    "prescriptions": [
        "row_id", "subject_id", "hadm_id", "starttime", "stoptime",
        "drug", "dose_val_rx", "dose_unit_rx", "route",
    ],
    "cost": ["row_id", "subject_id", "hadm_id", "event_type", "event_id", "chargetime", "cost"],
    "chartevents": [
        "row_id", "subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum", "valueuom",
    ],
    "inputevents": [
        "row_id", "subject_id", "hadm_id", "stay_id", "starttime", "itemid",
        "totalamount", "totalamountuom",
    ],
    "outputevents": [
        "row_id", "subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valueuom",
    ],
    "microbiologyevents": [
        "row_id", "subject_id", "hadm_id", "charttime", "spec_type_desc", "test_name", "org_name",
    ],
    "icustays": [
        "row_id", "subject_id", "hadm_id", "stay_id",
        "first_careunit", "last_careunit", "intime", "outtime",
    ],
    "transfers": [
        "row_id", "subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime",
    ],
}

# All valid table names — derived from allowlist
ALLOWED_TABLES: frozenset[str] = frozenset(SCHEMA_ALLOWLIST.keys())

# ---------------------------------------------------------------------------
# PHI column denylist — hard block in Layer 3, regardless of table.
# Based on HIPAA Safe Harbor identifiers applicable to the MIMIC schema.
# NB: `dob` is intentionally NOT blocked here — in the EHRSQL-2024 schema it is a
# synthesized de-identified year (anchor_year - anchor_age), required by many gold
# queries; it is not a real date of birth.
# ---------------------------------------------------------------------------

PHI_COLUMNS: frozenset[str] = frozenset(
    {
        "name",
        "first_name",
        "last_name",
        "date_of_birth",
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


# Foreign key relationships between EHRSQL-2024 (MIMIC-IV) tables.
# Included in schema prompts to help the model find join paths without guessing.
_FK_HINTS: str = (
    "Foreign keys: "
    "admissions.subject_id→patients.subject_id | "
    "diagnoses_icd.hadm_id→admissions.hadm_id | "
    "diagnoses_icd.icd_code→d_icd_diagnoses.icd_code | "
    "procedures_icd.hadm_id→admissions.hadm_id | "
    "procedures_icd.icd_code→d_icd_procedures.icd_code | "
    "labevents.hadm_id→admissions.hadm_id | "
    "labevents.itemid→d_labitems.itemid | "
    "prescriptions.hadm_id→admissions.hadm_id | "
    "cost.hadm_id→admissions.hadm_id | "
    "icustays.hadm_id→admissions.hadm_id | "
    "icustays.subject_id→patients.subject_id | "
    "chartevents.stay_id→icustays.stay_id | "
    "chartevents.itemid→d_items.itemid | "
    "inputevents.stay_id→icustays.stay_id | "
    "inputevents.itemid→d_items.itemid | "
    "outputevents.stay_id→icustays.stay_id | "
    "outputevents.itemid→d_items.itemid | "
    "microbiologyevents.hadm_id→admissions.hadm_id | "
    "transfers.hadm_id→admissions.hadm_id"
)


def schema_to_prompt(linked_schema: dict[str, list[str]] | None = None) -> str:
    """Serialize schema (or a linked subset) into a compact prompt string with FK hints."""
    source = linked_schema if linked_schema else SCHEMA_ALLOWLIST
    lines: list[str] = ["Database schema (MIMIC-IV, EHRSQL-2024):"]
    for table, cols in source.items():
        lines.append(f"  {table}({', '.join(cols)})")
    lines.append(_FK_HINTS)
    return "\n".join(lines)


# Canonical abstention token — must match harness + SFT data + serving.
ABSTAIN_TOKEN: str = "[ABSTAIN]"


def system_prompt() -> str:
    """Canonical system prompt used for BOTH SFT data prep and eval, so the trained
    model sees the same instruction at train and inference time (train/inference parity)."""
    return (
        "You are a clinical analytics assistant. Convert the user's question into a valid "
        "SQLite SELECT query over the MIMIC-IV (EHRSQL) database. "
        f"If the question cannot be answered with the available schema, output exactly: {ABSTAIN_TOKEN}\n\n"
        + schema_to_prompt()
    )
