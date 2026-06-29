"""EHRSQL-2024 evaluation with official Reliability-Score parity.

Runs a model over a 2024 split directory (data.json [+ label.json]) and scores its
predictions with the *vendored official scorer* (ehrsql2024_scoring) so numbers are
directly comparable to the shared-task leaderboard (RS(10)=0.8132). Writes a
prediction.json (id -> sql | "null", the Codabench submission format) and a metrics JSON.

Usage:
  python -m ehrcopilot.eval.eval2024 data/ehrsql2024/mimic_iv/test \
      --model checkpoints/orpo_gemma4_2024/adapter_final \
      --few-shot data/ehrsql2024/mimic_iv/train \
      --retrieval-mode embed --repair \
      --output tests/evalgen/gemma4_2024_test.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from ehrcopilot import config
from ehrcopilot.eval.ehrsql2024_scoring import score_predictions
from ehrcopilot.eval.harness import (
    ABSTAIN_TOKEN,
    _extract_sql,
    build_few_shot_retriever,
    load_ehrsql2024_split,
    run_hf_baseline,
)


def _to_pred(sql_text: str | None) -> str:
    """Map a model completion to the official skip indicator or a one-line SQL string."""
    if not sql_text:
        return "null"
    s = sql_text.strip()
    if s.upper() == ABSTAIN_TOKEN or ABSTAIN_TOKEN.lower() in s.lower():
        return "null"
    return " ".join(s.split())


def _exec_error(sql: str, db_path: str) -> str | None:
    try:
        con = sqlite3.connect(db_path)
        con.text_factory = lambda b: b.decode(errors="ignore")
        con.execute(sql).fetchall()
        con.close()
        return None
    except Exception as exc:
        return str(exc)


def _exec_status(sql: str, db_path: str) -> tuple[bool, bool]:
    """Return (executes_ok, nonempty) for the execution-based abstention gate."""
    try:
        con = sqlite3.connect(db_path)
        con.text_factory = lambda b: b.decode(errors="ignore")
        rows = con.execute(sql).fetchall()
        con.close()
        nonempty = bool(rows) and any(any(v not in (None, 0, "0", "") for v in r) for r in rows)
        return True, nonempty
    except Exception:
        return False, False


def main() -> None:
    p = argparse.ArgumentParser(description="EHRSQL-2024 eval (official RS parity)")
    p.add_argument("split", help="2024 split dir (contains data.json [+ label.json])")
    p.add_argument("--model", default=config.INFERENCE_MODEL)
    p.add_argument("--few-shot", default=None, help="2024 train dir for few-shot retrieval")
    p.add_argument("--retrieval-mode", default="embed", choices=["bm25", "embed", "hybrid"])
    p.add_argument("--repair", action="store_true", help="execution-guided repair loop")
    p.add_argument(
        "--exec-gate", default="error", choices=["off", "error", "empty", "both"],
        help="Execution-based abstention (the top-team RS(10) lever): after repair, abstain on "
             "SQL that errors ('error'), returns empty ('empty'), or either ('both'). Default: error.",
    )
    p.add_argument("--db", default=str(config.SQLITE_DB_PATH))
    p.add_argument("--output", default=None, help="metrics JSON path")
    p.add_argument("--pred-output", default=None, help="prediction.json path (Codabench format)")
    p.add_argument(
        "--dump-features", action="store_true",
        help="Harvest per-prediction confidence features (max-entropy, bottom-k log-prob, "
             "was_repaired, is_empty) for offline conformal calibration of the confidence gate. "
             "Run this on Colab (GPU) for valid AND test; calibrate locally (CPU).",
    )
    p.add_argument("--features-output", default=None, help="features JSON path (id -> feature dict)")
    p.add_argument("--limit", type=int, default=0, help="debug: cap #examples")
    args = p.parse_args()

    examples = load_ehrsql2024_split(Path(args.split))
    if args.limit:
        examples = examples[: args.limit]
    n_ans = sum(1 for e in examples if e.is_answerable)
    print(f"{len(examples)} examples ({n_ans} answerable / {len(examples)-n_ans} unanswerable) from {args.split}")
    real_dict = {e.id: (e.gold_sql if e.is_answerable else "null") for e in examples}

    retriever = None
    if args.few_shot:
        print(f"Building {args.retrieval_mode} few-shot retriever from {args.few_shot}")
        retriever = build_few_shot_retriever(Path(args.few_shot), mode=args.retrieval_mode)
        print("  retriever built.")

    print(f"Loading model: {args.model}")
    model_fn = run_hf_baseline(args.model, few_shot_retriever=retriever)
    has_repair = args.repair and hasattr(model_fn, "repair")
    print(f"Repair loop: {'enabled' if has_repair else 'off'}")

    pred_dict: dict[str, str] = {}
    features: dict[str, dict] | None = {} if args.dump_features else None
    if args.dump_features:
        try:
            model_fn.capture_features = True  # harness _UnslothPredictor flag
        except Exception:
            print("  (features requested but model_fn doesn't support capture — HF path?)")
    t0 = time.time()
    for i, e in enumerate(examples):
        pred = _to_pred(model_fn(e.question) or "")
        feat = dict(getattr(model_fn, "last_features", None) or {})
        was_repaired = 0
        if has_repair and pred != "null":
            for _ in range(config.MAX_REPAIR_ATTEMPTS):
                err = _exec_error(pred, args.db)
                if not err:
                    break
                rpred = _to_pred(model_fn.repair(e.question, pred, err) or "")
                was_repaired = 1
                if rpred == "null":
                    pred = "null"
                    break
                pred = rpred
            f2 = getattr(model_fn, "last_features", None)  # features of the FINAL decode
            if f2:
                feat = dict(f2)
        # Execution-based abstention gate — the highest-ROI RS(10) lever (LG/KAIST: +155 on dev).
        # After repair, abstain on SQL that still errors (and optionally returns empty) instead of
        # submitting a broken query and eating the -10 penalty.
        is_empty = 0
        if pred != "null":
            ok, nonempty = _exec_status(pred, args.db)
            is_empty = 1 if (ok and not nonempty) else 0
            if (not ok and args.exec_gate in ("error", "both")) or (
                ok and not nonempty and args.exec_gate in ("empty", "both")
            ):
                pred = "null"
        pred_dict[e.id] = pred
        if features is not None:
            feat.update({"was_repaired": was_repaired, "is_empty": is_empty})
            features[e.id] = feat
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(examples)}  ({(time.time()-t0)/(i+1):.2f}s/ex)", flush=True)

    metrics = score_predictions(real_dict, pred_dict, args.db)
    metrics["model"] = args.model
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    print("\n=== EHRSQL-2024 results ===")
    for k, v in metrics.items():
        print(f"  {k:34s}: {v}")

    pred_out = args.pred_output
    if not pred_out and args.output:
        pred_out = str(Path(args.output).with_suffix(".prediction.json"))
    if pred_out:
        Path(pred_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(pred_dict, open(pred_out, "w"))
        print(f"predictions -> {pred_out}")
    if features is not None:
        feat_out = args.features_output or (
            str(Path(args.output).with_suffix(".features.json")) if args.output else "features.json"
        )
        Path(feat_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(features, open(feat_out, "w"))
        print(f"features -> {feat_out}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(args.output, "w"), indent=2)
        print(f"metrics -> {args.output}")


if __name__ == "__main__":
    main()
