"""Convert MIMIC-IV-Demo CSVs into a single SQLite database.

Usage:
    python -m ehrcopilot.db.build_sqlite [CSV_DIR] [OUTPUT_DB]
    # or directly:
    python src/ehrcopilot/db/build_sqlite.py [CSV_DIR] [OUTPUT_DB]

Defaults:
    CSV_DIR   = data/mimic-iv-demo
    OUTPUT_DB = data/mimic_iv_demo.db
"""

from __future__ import annotations

import csv
import gzip
import os
import sqlite3
import sys
from pathlib import Path

DDL: dict[str, str] = {
    "patients": """
        CREATE TABLE patients (
            subject_id        INTEGER PRIMARY KEY,
            gender            TEXT,
            anchor_age        INTEGER,
            anchor_year       INTEGER,
            anchor_year_group TEXT,
            dod               TEXT
        )
    """,
    "admissions": """
        CREATE TABLE admissions (
            subject_id           INTEGER,
            hadm_id              INTEGER PRIMARY KEY,
            admittime            TEXT,
            dischtime            TEXT,
            deathtime            TEXT,
            admission_type       TEXT,
            admit_provider_id    TEXT,
            admission_location   TEXT,
            discharge_location   TEXT,
            insurance            TEXT,
            language             TEXT,
            marital_status       TEXT,
            race                 TEXT,
            edregtime            TEXT,
            edouttime            TEXT,
            hospital_expire_flag INTEGER
        )
    """,
    "diagnoses_icd": """
        CREATE TABLE diagnoses_icd (
            subject_id  INTEGER,
            hadm_id     INTEGER,
            seq_num     INTEGER,
            icd_code    TEXT,
            icd_version INTEGER
        )
    """,
    "d_icd_diagnoses": """
        CREATE TABLE d_icd_diagnoses (
            icd_code    TEXT,
            icd_version INTEGER,
            long_title  TEXT,
            PRIMARY KEY (icd_code, icd_version)
        )
    """,
    "procedures_icd": """
        CREATE TABLE procedures_icd (
            subject_id  INTEGER,
            hadm_id     INTEGER,
            seq_num     INTEGER,
            chartdate   TEXT,
            icd_code    TEXT,
            icd_version INTEGER
        )
    """,
    "d_icd_procedures": """
        CREATE TABLE d_icd_procedures (
            icd_code    TEXT,
            icd_version INTEGER,
            long_title  TEXT,
            PRIMARY KEY (icd_code, icd_version)
        )
    """,
    "labevents": """
        CREATE TABLE labevents (
            labevent_id       INTEGER PRIMARY KEY,
            subject_id        INTEGER,
            hadm_id           INTEGER,
            specimen_id       INTEGER,
            itemid            INTEGER,
            order_provider_id TEXT,
            charttime         TEXT,
            storetime         TEXT,
            value             TEXT,
            valuenum          REAL,
            valueuom          TEXT,
            ref_range_lower   REAL,
            ref_range_upper   REAL,
            flag              TEXT,
            priority          TEXT,
            comments          TEXT
        )
    """,
    "d_labitems": """
        CREATE TABLE d_labitems (
            itemid   INTEGER PRIMARY KEY,
            label    TEXT,
            fluid    TEXT,
            category TEXT
        )
    """,
    "prescriptions": """
        CREATE TABLE prescriptions (
            subject_id        INTEGER,
            hadm_id           INTEGER,
            pharmacy_id       INTEGER,
            poe_id            TEXT,
            poe_seq           INTEGER,
            order_provider_id TEXT,
            starttime         TEXT,
            stoptime          TEXT,
            drug_type         TEXT,
            drug              TEXT,
            formulary_drug_cd TEXT,
            gsn               TEXT,
            ndc               TEXT,
            prod_strength     TEXT,
            form_rx           TEXT,
            dose_val_rx       TEXT,
            dose_unit_rx      TEXT,
            form_val_disp     TEXT,
            form_unit_disp    TEXT,
            doses_per_24_hrs  REAL,
            route             TEXT
        )
    """,
    "icustays": """
        CREATE TABLE icustays (
            subject_id     INTEGER,
            hadm_id        INTEGER,
            stay_id        INTEGER PRIMARY KEY,
            first_careunit TEXT,
            last_careunit  TEXT,
            intime         TEXT,
            outtime        TEXT,
            los            REAL
        )
    """,
    "chartevents": """
        CREATE TABLE chartevents (
            subject_id   INTEGER,
            hadm_id      INTEGER,
            stay_id      INTEGER,
            caregiver_id INTEGER,
            charttime    TEXT,
            storetime    TEXT,
            itemid       INTEGER,
            value        TEXT,
            valuenum     REAL,
            valueuom     TEXT,
            warning      INTEGER
        )
    """,
    "d_items": """
        CREATE TABLE d_items (
            itemid          INTEGER PRIMARY KEY,
            label           TEXT,
            abbreviation    TEXT,
            linksto         TEXT,
            category        TEXT,
            unitname        TEXT,
            param_type      TEXT,
            lownormalvalue  REAL,
            highnormalvalue REAL
        )
    """,
    "microbiologyevents": """
        CREATE TABLE microbiologyevents (
            microevent_id       INTEGER PRIMARY KEY,
            subject_id          INTEGER,
            hadm_id             INTEGER,
            micro_specimen_id   INTEGER,
            order_provider_id   TEXT,
            chartdate           TEXT,
            charttime           TEXT,
            spec_itemid         INTEGER,
            spec_type_desc      TEXT,
            test_seq            INTEGER,
            storedate           TEXT,
            storetime           TEXT,
            test_itemid         INTEGER,
            test_name           TEXT,
            org_itemid          INTEGER,
            org_name            TEXT,
            isolate_num         INTEGER,
            quantity            TEXT,
            ab_itemid           INTEGER,
            ab_name             TEXT,
            dilution_text       TEXT,
            dilution_comparison TEXT,
            dilution_value      REAL,
            interpretation      TEXT,
            comments            TEXT
        )
    """,
    "pharmacy": """
        CREATE TABLE pharmacy (
            subject_id         INTEGER,
            hadm_id            INTEGER,
            pharmacy_id        INTEGER PRIMARY KEY,
            poe_id             TEXT,
            starttime          TEXT,
            stoptime           TEXT,
            medication         TEXT,
            proc_type          TEXT,
            status             TEXT,
            entertime          TEXT,
            verifiedtime       TEXT,
            route              TEXT,
            frequency          TEXT,
            disp_sched         TEXT,
            infusion_type      TEXT,
            sliding_scale      TEXT,
            lockout_interval   TEXT,
            basal_rate         REAL,
            one_hr_max         REAL,
            doses_per_24_hrs   REAL,
            duration           REAL,
            duration_interval  TEXT,
            expiration_value   REAL,
            expiration_unit    TEXT,
            expirationdate     TEXT,
            dispensation       TEXT,
            fill_quantity      TEXT
        )
    """,
    "transfers": """
        CREATE TABLE transfers (
            subject_id  INTEGER,
            hadm_id     INTEGER,
            transfer_id INTEGER PRIMARY KEY,
            eventtype   TEXT,
            careunit    TEXT,
            intime      TEXT,
            outtime     TEXT
        )
    """,
    "edstays": """
        CREATE TABLE edstays (
            subject_id        INTEGER,
            hadm_id           INTEGER,
            stay_id           INTEGER PRIMARY KEY,
            intime            TEXT,
            outtime           TEXT,
            gender            TEXT,
            race              TEXT,
            arrival_transport TEXT,
            disposition       TEXT
        )
    """,
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_admissions_subject    ON admissions(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_diagnoses_hadm        ON diagnoses_icd(hadm_id)",
    "CREATE INDEX IF NOT EXISTS idx_diagnoses_subject     ON diagnoses_icd(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_labevents_subject     ON labevents(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_labevents_hadm        ON labevents(hadm_id)",
    "CREATE INDEX IF NOT EXISTS idx_prescriptions_hadm    ON prescriptions(hadm_id)",
    "CREATE INDEX IF NOT EXISTS idx_icustays_hadm         ON icustays(hadm_id)",
    "CREATE INDEX IF NOT EXISTS idx_chartevents_stay      ON chartevents(stay_id)",
    "CREATE INDEX IF NOT EXISTS idx_transfers_hadm        ON transfers(hadm_id)",
    "CREATE INDEX IF NOT EXISTS idx_microevents_subject   ON microbiologyevents(subject_id)",
]


def find_csv(table: str, base_dir: Path) -> Path | None:
    """Walk base_dir recursively to find <table>.csv or <table>.csv.gz."""
    for root, _, files in os.walk(base_dir):
        for fname in files:
            lower = fname.lower()
            if lower in (f"{table}.csv", f"{table}.csv.gz"):
                return Path(root) / fname
    return None


def load_table(conn: sqlite3.Connection, table: str, csv_path: Path) -> int:
    """Load all rows from csv_path into table. Returns row count."""
    opener = gzip.open if str(csv_path).endswith(".gz") else open

    with opener(csv_path, "rt", encoding="utf-8", newline="") as f:  # type: ignore[call-overload]
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return 0

    cols = list(rows[0].keys())
    quoted_cols = ",".join(f'"{c}"' for c in cols)
    placeholders = ",".join("?" for _ in cols)
    insert_sql = f"INSERT OR IGNORE INTO {table} ({quoted_cols}) VALUES ({placeholders})"

    batch = [[r.get(c) or None for c in cols] for r in rows]
    conn.executemany(insert_sql, batch)
    conn.commit()
    return len(rows)


def build(csv_dir: Path, output_db: Path) -> dict[str, int]:
    """Build the SQLite DB. Returns {table: row_count} for smoke-check."""
    output_db.parent.mkdir(parents=True, exist_ok=True)
    output_db.unlink(missing_ok=True)

    conn = sqlite3.connect(str(output_db))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")

    print("Creating schema...")
    for ddl in DDL.values():
        conn.execute(ddl)
    conn.commit()

    print("Loading CSV files...")
    counts: dict[str, int] = {}
    missing: list[str] = []

    for table in DDL:
        csv_path = find_csv(table, csv_dir)
        if csv_path is None:
            print(f"  MISSING  {table}.csv — skipping")
            missing.append(table)
            counts[table] = 0
            continue
        n = load_table(conn, table, csv_path)
        counts[table] = n
        print(f"  OK  {table}: {n:,} rows")

    print("Creating indexes...")
    for idx in INDEXES:
        conn.execute(idx)
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    print(f"\nDatabase written to {output_db}")
    if missing:
        print(f"WARNING: {len(missing)} tables had no source CSV: {missing}")

    return counts


if __name__ == "__main__":
    csv_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/mimic-iv-demo")
    output_db = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/mimic_iv_demo.db")

    if not csv_dir.exists():
        print(f"ERROR: CSV directory '{csv_dir}' not found.", file=sys.stderr)
        print(
            "Download MIMIC-IV-Demo from https://physionet.org/content/mimic-iv-demo/2.2/\n"
            f"and unzip into {csv_dir}/",
            file=sys.stderr,
        )
        sys.exit(1)

    build(csv_dir, output_db)
