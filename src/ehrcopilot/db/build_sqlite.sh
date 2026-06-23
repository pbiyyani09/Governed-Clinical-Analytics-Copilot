#!/usr/bin/env bash
# build_sqlite.sh — Convert MIMIC-IV-Demo CSVs into a single SQLite database.
#
# Usage:
#   bash src/ehrcopilot/db/build_sqlite.sh [CSV_DIR] [OUTPUT_DB]
#
# Download MIMIC-IV-Demo (PhysioNet — free registration required):
#   wget -r -N -c -np --user <physionet-user> --ask-password \
#        https://physionet.org/files/mimic-iv-demo/2.2/
#   mv physionet.org/files/mimic-iv-demo/2.2 data/mimic-iv-demo

set -euo pipefail

CSV_DIR="${1:-data/mimic-iv-demo}"
OUTPUT_DB="${2:-data/mimic_iv_demo.db}"

if [[ ! -d "$CSV_DIR" ]]; then
  echo "ERROR: CSV directory '$CSV_DIR' not found."
  echo "Download MIMIC-IV-Demo from https://physionet.org/content/mimic-iv-demo/2.2/"
  echo "and place the unzipped CSVs in $CSV_DIR/"
  exit 1
fi

python3 src/ehrcopilot/db/build_sqlite.py "$CSV_DIR" "$OUTPUT_DB"
echo "Smoke-checking row counts..."
python3 -c "
import sys; sys.path.insert(0, 'src')
from ehrcopilot.db.connection import verify_db
from pathlib import Path
counts = verify_db(Path('$OUTPUT_DB'))
for t, n in counts.items():
    status = 'OK' if n > 0 else 'EMPTY' if n == 0 else 'MISSING'
    print(f'  {status:7s}  {t}: {n}')
"
