"""CPU/offline calibration of the confidence abstention gate (runs locally, no GPU).

Consumes the prediction.json + features.json that a Colab GPU eval2024 --dump-features run
produced for the VALID and TEST splits. Fits a single-signal abstention threshold on VALID to
maximize official RS(10) (the LG/KAIST = max-entropy and ProbGate = bottom-k-logprob recipe),
then applies it to TEST and reports the gain. Threshold search is arithmetic (correctness is
computed once per example), so trying many thresholds is instant.

Signals (each example, on top of the already-applied execution gate):
  entropy : abstain if max_entropy   > tau   (high entropy = uncertain)
  logprob : abstain if bottom10_logprob < tau (low log-prob = uncertain)

Usage:
  PYTHONPATH=src python scripts/eda/calibrate_gate.py \
    --valid-pred valid.prediction.json --valid-feat valid.features.json \
    --test-pred  test.prediction.json  --test-feat  test.features.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ehrcopilot import config
from ehrcopilot.eval.harness import load_ehrsql2024_split
from ehrcopilot.eval.ehrsql2024_scoring import process_answer

NULL = "null"


def _result(sql, db):
    if sql == NULL:
        return NULL
    try:
        con = sqlite3.connect(db); con.text_factory = lambda b: b.decode(errors="ignore")
        r = con.execute(sql).fetchall(); con.close()
        return process_answer(r)
    except Exception:
        return "error_pred"


def build_records(split_dir, pred_path, feat_path, db):
    ex = load_ehrsql2024_split(Path(split_dir))
    pred = json.load(open(pred_path))
    feat = json.load(open(feat_path)) if feat_path and Path(feat_path).exists() else {}
    recs = []
    for e in ex:
        p = pred.get(e.id, NULL)
        gold_null = not e.is_answerable
        pred_null = (p == NULL)
        correct = None
        if not gold_null and not pred_null:
            correct = (_result(p, db) == process_answer(_result_gold(e.gold_sql, db)))
        f = feat.get(e.id, {}) or {}
        recs.append({
            "gold_null": gold_null, "pred_null": pred_null, "correct": correct,
            "ent": f.get("max_entropy"), "blp": f.get("bottom10_logprob"),
        })
    return recs


def _result_gold(sql, db):
    try:
        con = sqlite3.connect(db); con.text_factory = lambda b: b.decode(errors="ignore")
        r = con.execute(sql).fetchall(); con.close()
        return r
    except Exception:
        return "error_real"


def rs(recs, c, gate):
    ca = cab = wa = wu = 0
    n = len(recs)
    for r in recs:
        gated = gate(r) if gate else False
        answered = (not r["pred_null"]) and (not gated)
        if r["gold_null"]:
            if answered:
                wu += 1
            else:
                cab += 1
        else:
            if not answered:
                pass  # abstain on answerable => 0
            elif r["correct"]:
                ca += 1
            else:
                wa += 1
    score = 100 * (ca + cab - c * (wa + wu)) / n if n else 0.0
    return score, {"correct": ca, "correct_abstain": cab, "wrong_ans": wa, "wrong_unans": wu}


def gate_fn(signal, tau):
    if signal == "entropy":
        return lambda r: r["ent"] is not None and r["ent"] > tau
    return lambda r: r["blp"] is not None and r["blp"] < tau


def calibrate(valid, c=10):
    best = (None, None, -1e9)
    for signal in ("entropy", "logprob"):
        vals = sorted({(r["ent"] if signal == "entropy" else r["blp"])
                       for r in valid if not r["pred_null"]
                       and (r["ent"] if signal == "entropy" else r["blp"]) is not None})
        if not vals:
            continue
        # candidate thresholds = midpoints + extremes (so "gate off" is included)
        cands = [vals[0] - 1] + vals + [vals[-1] + 1]
        for tau in cands:
            score, _ = rs(valid, c, gate_fn(signal, tau))
            if score > best[2]:
                best = (signal, tau, score)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid-pred", required=True)
    p.add_argument("--valid-feat", required=True)
    p.add_argument("--valid-split", default="data/ehrsql2024/mimic_iv/valid")
    p.add_argument("--test-pred", required=True)
    p.add_argument("--test-feat", required=True)
    p.add_argument("--test-split", default="data/ehrsql2024/mimic_iv/test")
    p.add_argument("--db", default=str(config.SQLITE_DB_PATH))
    p.add_argument("--c", type=int, default=10)
    args = p.parse_args()

    print("Building valid records...")
    valid = build_records(args.valid_split, args.valid_pred, args.valid_feat, args.db)
    print("Building test records...")
    test = build_records(args.test_split, args.test_pred, args.test_feat, args.db)

    signal, tau, vbest = calibrate(valid, args.c)
    print(f"\nCalibrated on VALID: signal={signal}  tau={tau:.4f}  ->  valid RS({args.c})={vbest:.2f}")

    for name, recs in (("VALID", valid), ("TEST", test)):
        base0, _ = rs(recs, 0, None)
        base, bd0 = rs(recs, args.c, None)
        gated, bd = rs(recs, args.c, gate_fn(signal, tau))
        gated0, _ = rs(recs, 0, gate_fn(signal, tau))
        print(f"\n[{name}]  baseline: RS(0)={base0:.2f} RS({args.c})={base:.2f}  {bd0}")
        print(f"[{name}]  +conf-gate: RS(0)={gated0:.2f} RS({args.c})={gated:.2f}  {bd}  (target 81.32)")


if __name__ == "__main__":
    main()
