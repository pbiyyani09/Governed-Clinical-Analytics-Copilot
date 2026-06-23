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
SQLITE_DB_PATH = DATA_DIR / "mimic_iv_demo.db"
EHRSQL_DATA_DIR = DATA_DIR / "ehrsql"

# ---------------------------------------------------------------------------
# Schema allowlist — only these tables/columns are permitted through Layer 2.
# Derived from MIMIC-IV-Demo v2.2 schema.
# ---------------------------------------------------------------------------

SCHEMA_ALLOWLIST: dict[str, list[str]] = {
    "patients": [
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
        "dod",
    ],
    "admissions": [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "edregtime",
        "edouttime",
        "hospital_expire_flag",
    ],
    "diagnoses_icd": [
        "subject_id",
        "hadm_id",
        "seq_num",
        "icd_code",
        "icd_version",
    ],
    "d_icd_diagnoses": [
        "icd_code",
        "icd_version",
        "long_title",
    ],
    "procedures_icd": [
        "subject_id",
        "hadm_id",
        "seq_num",
        "chartdate",
        "icd_code",
        "icd_version",
    ],
    "d_icd_procedures": [
        "icd_code",
        "icd_version",
        "long_title",
    ],
    "labevents": [
        "labevent_id",
        "subject_id",
        "hadm_id",
        "specimen_id",
        "itemid",
        "charttime",
        "storetime",
        "value",
        "valuenum",
        "valueuom",
        "ref_range_lower",
        "ref_range_upper",
        "flag",
        "priority",
        "comments",
    ],
    "d_labitems": [
        "itemid",
        "label",
        "fluid",
        "category",
    ],
    "prescriptions": [
        "subject_id",
        "hadm_id",
        "pharmacy_id",
        "poe_id",
        "poe_seq",
        "starttime",
        "stoptime",
        "drug_type",
        "drug",
        "formulary_drug_cd",
        "gsn",
        "ndc",
        "prod_strength",
        "form_rx",
        "dose_val_rx",
        "dose_unit_rx",
        "form_val_disp",
        "form_unit_disp",
        "doses_per_24_hrs",
        "route",
    ],
    "icustays": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
    ],
    "chartevents": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "storetime",
        "value",
        "valuenum",
        "valueuom",
        "warning",
    ],
    "d_items": [
        "itemid",
        "label",
        "abbreviation",
        "linksto",
        "category",
        "unitname",
        "param_type",
        "lownormalvalue",
        "highnormalvalue",
    ],
    "microbiologyevents": [
        "microevent_id",
        "subject_id",
        "hadm_id",
        "micro_specimen_id",
        "chartdate",
        "charttime",
        "spec_itemid",
        "spec_type_desc",
        "test_seq",
        "storedate",
        "storetime",
        "test_itemid",
        "test_name",
        "org_itemid",
        "org_name",
        "isolate_num",
        "quantity",
        "ab_itemid",
        "ab_name",
        "dilution_text",
        "dilution_comparison",
        "dilution_value",
        "interpretation",
        "comments",
    ],
    "pharmacy": [
        "subject_id",
        "hadm_id",
        "pharmacy_id",
        "poe_id",
        "starttime",
        "stoptime",
        "medication",
        "proc_type",
        "status",
        "entertime",
        "verifiedtime",
        "route",
        "frequency",
        "disp_sched",
        "infusion_type",
        "sliding_scale",
        "lockout_interval",
        "basal_rate",
        "one_hr_max",
        "doses_per_24_hrs",
        "duration",
        "duration_interval",
        "expiration_value",
        "expiration_unit",
        "expirationdate",
        "dispensation",
        "fill_quantity",
    ],
    "transfers": [
        "subject_id",
        "hadm_id",
        "transfer_id",
        "eventtype",
        "careunit",
        "intime",
        "outtime",
    ],
    # edstays is not included in MIMIC-IV-Demo v2.2 (only in full MIMIC-IV)
}

# All valid table names — derived from allowlist
ALLOWED_TABLES: frozenset[str] = frozenset(SCHEMA_ALLOWLIST.keys())

# ---------------------------------------------------------------------------
# PHI column denylist — hard block in Layer 3, regardless of table.
# Based on HIPAA Safe Harbor identifiers applicable to the MIMIC schema.
# ---------------------------------------------------------------------------

PHI_COLUMNS: frozenset[str] = frozenset(
    {
        "name",
        "first_name",
        "last_name",
        "dob",
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
DB_VERSION: int = 1

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


def schema_to_prompt(linked_schema: dict[str, list[str]] | None = None) -> str:
    """Serialize schema (or a linked subset) into a compact prompt string."""
    source = linked_schema if linked_schema else SCHEMA_ALLOWLIST
    lines: list[str] = ["Database schema (MIMIC-IV-Demo):"]
    for table, cols in source.items():
        lines.append(f"  {table}({', '.join(cols)})")
    return "\n".join(lines)
