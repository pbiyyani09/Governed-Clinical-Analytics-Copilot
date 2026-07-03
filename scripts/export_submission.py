#!/usr/bin/env python3
"""Convert --save-predictions JSONL to official EHRSQL 2024 prediction.json format.

The official scorer (scoring_program/scoring.py) expects:
    {id: sql_string}
where unanswerable predictions must be the string "null" (not JSON null).

Usage:
    python3 scripts/export_submission.py \\
        --predictions tests/evalgen/orpo_v4_test_corrected_preds.jsonl \\
        --output prediction.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ABSTAIN_TOKEN = "[ABSTAIN]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="JSONL from --save-predictions")
    parser.add_argument("--output", default="prediction.json", help="Output file path")
    args = parser.parse_args()

    pred_dict: dict[str, str] = {}
    with open(args.predictions) as f:
        for line in f:
            rec = json.loads(line.strip())
            id_ = rec["id"]
            sql = rec.get("predicted_sql") or ""
            # Map [ABSTAIN] and empty/null predictions to official "null" string
            if not sql or sql == ABSTAIN_TOKEN or sql.strip().lower() in ("null", "none"):
                pred_dict[id_] = "null"
            else:
                pred_dict[id_] = sql

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(pred_dict, f, indent=2)

    n_null = sum(1 for v in pred_dict.values() if v == "null")
    print(f"Exported {len(pred_dict)} predictions → {out}")
    print(f"  SQL predictions: {len(pred_dict) - n_null}")
    print(f"  Abstentions (null): {n_null}")


if __name__ == "__main__":
    main()
