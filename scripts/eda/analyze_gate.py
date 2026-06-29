"""Offline RS(10) what-if analysis for an existing EHRSQL-2024 prediction.json.

Simulates the top-team abstention gates on predictions ALREADY generated (no GPU, no
model re-run): re-executes each predicted SQL against the sqlite and recomputes the
official Reliability Score under several abstention policies. Also breaks down the
wrong-answerable errors into "errors/empty (a gate catches these)" vs "valid-but-wrong
result (needs self-consistency/confidence)".

Usage:
  PYTHONPATH=src python scripts/eda/analyze_gate.py tests/evalgen/g4_2024_sft.prediction.json
  # default test dir + db from config; override with --split / --db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ehrcopilot import config
from ehrcopilot.eval.harness import load_ehrsql2024_split
from ehrcopilot.eval.ehrsql2024_scoring import process_answer, score_predictions

NULL = "null"


def exec_status(sql, db):
    try:
        con = sqlite3.connect(db); con.text_factory = lambda b: b.decode(errors="ignore")
        rows = con.execute(sql).fetchall(); con.close()
        nonempty = bool(rows) and any(any(v not in (None, 0, "0", "") for v in r) for r in rows)
        return True, nonempty
    except Exception:
        return False, False


def gate(pred_dict, db, mode):
    out = {}
    for k, v in pred_dict.items():
        if v == NULL or mode == "off":
            out[k] = v; continue
        ok, nonempty = exec_status(v, db)
        if (not ok and mode in ("error", "both")) or (ok and not nonempty and mode in ("empty", "both")):
            out[k] = NULL
        else:
            out[k] = v
    return out


def line(tag, m):
    return (f"{tag:18s} EX={m['EX']:.4f}  RS(0)={m['RS(0)']:7.2f}  RS(5)={m['RS(5)']:7.2f}  "
            f"RS(10)={m['RS(10)']:8.2f}  | correct={m['correct_answers']} "
            f"wrong_ans={m['wrong_answers_on_answerable']} wrong_unans={m['wrong_answers_on_unanswerable']} "
            f"corr_abst={m['correct_abstentions']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prediction", help="prediction.json (id -> sql|null)")
    p.add_argument("--split", default="data/ehrsql2024/mimic_iv/test")
    p.add_argument("--db", default=str(config.SQLITE_DB_PATH))
    args = p.parse_args()

    pred = json.load(open(args.prediction))
    ex = load_ehrsql2024_split(Path(args.split))
    real = {e.id: (e.gold_sql if e.is_answerable else NULL) for e in ex}
    # align keys
    pred = {e.id: pred.get(e.id, NULL) for e in ex}

    print(f"# {args.prediction}  ({len(real)} examples, db={args.db})\n")
    print(line("baseline (as-run)", score_predictions(real, pred, args.db)))
    for mode in ("error", "empty", "both"):
        print(line(f"gate={mode}", score_predictions(real, gate(pred, args.db, mode), args.db)))

    # Error decomposition: of the currently-wrong answerable predictions, how many error/empty
    # (a gate catches) vs execute to a valid-but-wrong result (needs self-consistency/confidence)?
    err = empty = valwrong = 0
    for e in ex:
        if not e.is_answerable:
            continue
        pv = pred[e.id]
        if pv == NULL:
            continue
        if process_answer(_safe_exec(pv, args.db)) == process_answer(_safe_exec(e.gold_sql, args.db)):
            continue  # correct
        ok, nonempty = exec_status(pv, args.db)
        if not ok:
            err += 1
        elif not nonempty:
            empty += 1
        else:
            valwrong += 1
    total_wrong = err + empty + valwrong
    print(f"\nwrong-answerable breakdown (n={total_wrong}):")
    print(f"  execution error : {err:4d}  -> caught by gate=error/both")
    print(f"  empty result    : {empty:4d}  -> caught by gate=empty/both")
    print(f"  valid-but-wrong : {valwrong:4d}  -> NOT caught by exec gate (needs self-consistency / confidence)")


def _safe_exec(sql, db):
    try:
        con = sqlite3.connect(db); con.text_factory = lambda b: b.decode(errors="ignore")
        r = con.execute(sql).fetchall(); con.close(); return r
    except Exception:
        return "error"


if __name__ == "__main__":
    main()
