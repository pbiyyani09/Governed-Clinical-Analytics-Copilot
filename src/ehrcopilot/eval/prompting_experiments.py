"""Exhaust inference-time techniques on BASE gemma-3-12b before fine-tuning.

Each technique is layered cumulatively and scored (EX + RS) on the same subset,
so we can see exactly how far prompting/context-engineering/CoT/agentic loops get
us — and where they plateau relative to the EHRSQL target. The generation model is
loaded ONCE; all configs reuse it.

Configs (cumulative):
  base      fusion-top-5 few-shot, direct SQL (the current pipeline)
  ctx       + context engineering: enriched schema (SQLite dialect, MIMIC join/
            value hints, explicit schema-use + abstain rule)
  cot       + chain-of-thought decomposition (reason -> final ```sql```)
  agentic   + execution-guided loop (error feedback AND empty-result feedback,
            up to 3 rounds, with reasoning)
  abstain   + explicit answerability verification pass -> [ABSTAIN] (targets RS)
  selfcons  cot + self-consistency (N samples, execution-based majority vote)

Usage:
  python -m ehrcopilot.eval.prompting_experiments \
     --split data/ehrsql/ehrsql/mimic_iii/test_cmp75.json \
     --train data/ehrsql/ehrsql/mimic_iii/train.json \
     --model unsloth/gemma-3-12b-it \
     --configs base ctx cot agentic abstain \
     --out tests/evalgen/prompting_experiments.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from ehrcopilot import config
from ehrcopilot.eval.harness import (
    _exec_safe, results_match, _canonicalize_gold_sql, _extract_sql, ABSTAIN_TOKEN,
)
from ehrcopilot.eval.template_retriever import build_classifier_retriever


def _load(p):
    r = json.load(open(p)); return r["data"] if isinstance(r, dict) and "data" in r else r


def _is_answerable(e: dict) -> bool:
    if str(e.get("is_impossible", False)).lower() in ("true", "1"):
        return False
    return (e.get("query") or e.get("sql") or "").strip().lower() not in ("", "null", "none", "n/a")


# --- prompts -----------------------------------------------------------------

_BASE_SYS = (
    "You are a clinical analytics assistant. Convert the user's question into a "
    "valid SQLite SELECT query over the MIMIC-IV-Demo database. "
    f"If the question cannot be answered with the available data, output exactly: {ABSTAIN_TOKEN}\n\n"
    + config.schema_to_prompt()
)

# Context-engineered system prompt: dialect + MIMIC join/value hints + rules.
_CTX_SYS = (
    "You are an expert clinical analytics SQL assistant for the MIMIC-IV-Demo "
    "database (SQLite dialect).\n"
    "RULES:\n"
    "- Use ONLY the tables and columns listed in the schema below. If the question "
    f"needs data not present in this schema, output exactly: {ABSTAIN_TOKEN}\n"
    "- SQLite only: use strftime()/julianday()/date() for dates; there is no "
    "DATEDIFF/TO_DATE/EXTRACT. Length-of-stay is in icustays.los (fractional days).\n"
    "- Item names are values, not columns: to filter a lab by name join "
    "labevents.itemid = d_labitems.itemid and filter d_labitems.label; to filter a "
    "chart item join chartevents.itemid = d_items.itemid on d_items.label; for a "
    "diagnosis title join diagnoses_icd.icd_code = d_icd_diagnoses.icd_code and "
    "filter d_icd_diagnoses.long_title; same pattern for procedures.\n"
    "- A drug name filters prescriptions.drug; a medication filters pharmacy.medication.\n"
    "- Output ONLY the SQL query, no explanation.\n\n"
    + config.schema_to_prompt()
)

_COT_SUFFIX = (
    "\nThink step by step: (1) which tables are needed, (2) the join path, "
    "(3) the filters/aggregation. Then give the final query inside a ```sql ... ``` "
    f"block. If unanswerable with this schema, output {ABSTAIN_TOKEN} as the final line."
)


def _extract_final_sql(text: str) -> str:
    """For CoT output: take the LAST fenced SQL block (the final answer)."""
    if ABSTAIN_TOKEN.lower() in text.lower().split("```")[-1:][0:1] and "select" not in text.lower():
        return ABSTAIN_TOKEN
    blocks = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return _extract_sql("```sql\n" + blocks[-1].strip() + "\n```")
    return _extract_sql(text)


# --- experiment driver -------------------------------------------------------

def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", default="unsloth/gemma-3-12b-it")
    ap.add_argument("--configs", nargs="+",
                    default=["base", "ctx", "cot", "agentic", "abstain"])
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--out", default="tests/evalgen/prompting_experiments.json")
    args = ap.parse_args()

    examples = [e for e in _load(args.split)]
    print(f"{len(examples)} eval examples", flush=True)

    retriever = build_classifier_retriever(Path(args.train), top_k=5, method="fusion")

    from unsloth import FastModel
    model, tok = FastModel.from_pretrained(model_name=args.model,
                                           max_seq_length=config.MAX_SEQ_LENGTH,
                                           dtype=torch.bfloat16, load_in_4bit=True)
    FastModel.for_inference(model)

    def gen(messages, max_new=256, temperature=0.0):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=do_sample,
                                 temperature=temperature if do_sample else None,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def fewshot(q):
        return f"{retriever(q)}\n\nQuestion: {q}"

    # --- per-config prediction functions ---
    def predict_base(q):
        return _extract_sql(gen([{"role": "system", "content": _BASE_SYS},
                                 {"role": "user", "content": fewshot(q)}]))

    def predict_ctx(q):
        return _extract_sql(gen([{"role": "system", "content": _CTX_SYS},
                                 {"role": "user", "content": fewshot(q)}]))

    def predict_cot(q):
        txt = gen([{"role": "system", "content": _CTX_SYS},
                   {"role": "user", "content": fewshot(q) + _COT_SUFFIX}], max_new=512)
        return _extract_final_sql(txt)

    def _agentic(q, start_sql=None):
        sql = start_sql if start_sql is not None else predict_cot(q)
        msgs = [{"role": "system", "content": _CTX_SYS},
                {"role": "user", "content": fewshot(q) + _COT_SUFFIX}]
        for _ in range(args.max_rounds):
            if sql == ABSTAIN_TOKEN or not sql:
                return sql
            rows, err = _exec_safe(sql)
            if err is None and rows:
                return sql
            if err is not None:
                fb = f"That SQL failed with SQLite error: {err}\nFix it using only the schema. Output the corrected query in a ```sql``` block."
            else:
                fb = "That SQL executed but returned no rows. Reconsider the table/column/value choices (item names need a join to d_items/d_labitems/d_icd_*). Output a corrected query in a ```sql``` block."
            msgs += [{"role": "assistant", "content": f"```sql\n{sql}\n```"},
                     {"role": "user", "content": fb}]
            sql = _extract_final_sql(gen(msgs, max_new=512))
        return sql

    def predict_agentic(q):
        return _agentic(q)

    def _answerable_check(q, sql):
        """Verification pass: is the question answerable & SQL schema-grounded?"""
        if sql == ABSTAIN_TOKEN or not sql:
            return False
        v = gen([{"role": "system", "content":
                  "You verify clinical NL->SQL. Given the schema, the question, and a "
                  "proposed SQL, answer with exactly one word: ANSWERABLE if the question "
                  "can be answered with the listed tables/columns AND the SQL references only "
                  "existing tables/columns; otherwise UNANSWERABLE.\n\n" + config.schema_to_prompt()},
                 {"role": "user", "content": f"Question: {q}\nProposed SQL: {sql}\nVerdict:"}],
                max_new=8)
        return "unanswerable" not in v.lower()

    def predict_abstain(q):
        sql = _agentic(q)
        return sql if _answerable_check(q, sql) else ABSTAIN_TOKEN

    def predict_selfcons(q):
        preds = [_extract_final_sql(gen([{"role": "system", "content": _CTX_SYS},
                 {"role": "user", "content": fewshot(q) + _COT_SUFFIX}], max_new=512,
                 temperature=0.7)) for _ in range(args.n_samples)]
        # execution-based majority vote on result set
        buckets = {}
        for p in preds:
            if p == ABSTAIN_TOKEN or not p:
                buckets.setdefault(("ABSTAIN",), ["ABSTAIN", 0])[1] += 1; continue
            rows, err = _exec_safe(p)
            if err is not None:
                continue
            key = frozenset(tuple(sorted(r.items())) for r in rows[:50]) if rows else frozenset()
            buckets.setdefault(key, [p, 0])[1] += 1
        if not buckets:
            return preds[0] if preds else ABSTAIN_TOKEN
        return max(buckets.values(), key=lambda v: v[1])[0]

    PREDICT = {"base": predict_base, "ctx": predict_ctx, "cot": predict_cot,
               "agentic": predict_agentic, "abstain": predict_abstain, "selfcons": predict_selfcons}

    results = {}
    for cfg in args.configs:
      try:
        fn = PREDICT[cfg]; t0 = time.time()
        corr = wrong_ans = wrong_unans = corr_abst = wrong_abst = 0
        n_ans = n_unans = 0
        for i, e in enumerate(examples):
            q = e["question"]; ans = _is_answerable(e)
            pred = fn(q)
            abstained = pred == ABSTAIN_TOKEN or not pred
            if ans:
                n_ans += 1
                if abstained:
                    wrong_abst += 1
                else:
                    pr, pe = _exec_safe(pred)
                    gr, _ = _exec_safe(_canonicalize_gold_sql(e.get("query") or ""))
                    if pe is None and results_match(pr, gr):
                        corr += 1
                    else:
                        wrong_ans += 1
            else:
                n_unans += 1
                if abstained:
                    corr_abst += 1
                else:
                    wrong_unans += 1
            if (i + 1) % 15 == 0:
                print(f"  [{cfg}] {i+1}/{len(examples)} corr={corr} wrong_unans={wrong_unans}", flush=True)
        total = len(examples)
        ex = corr / n_ans if n_ans else 0.0
        rs10 = (corr + corr_abst - 10 * wrong_unans) / total
        rs0 = (corr + corr_abst) / total
        results[cfg] = {"EX": ex, "RS(0)": rs0, "RS(10)": rs10, "correct": corr,
                        "wrong_answers": wrong_ans, "wrong_abstentions": wrong_abst,
                        "wrong_on_unanswerable": wrong_unans, "correct_abstentions": corr_abst,
                        "n_answerable": n_ans, "n_unanswerable": n_unans,
                        "minutes": round((time.time() - t0) / 60, 1)}
        print(f"== {cfg}: EX={ex:.4f} RS(10)={rs10:+.4f} corr={corr}/{n_ans} "
              f"wrong_unans={wrong_unans}/{n_unans} ({results[cfg]['minutes']}min)", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
      except Exception as exc:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"== {cfg}: FAILED {type(exc).__name__}: {exc}", flush=True)
        results[cfg] = {"error": f"{type(exc).__name__}: {exc}"}

    print("\n=== SUMMARY ===")
    print(f"{'config':10s} | {'EX':>6} | {'RS(10)':>8} | corr | wrong_unans")
    for c, r in results.items():
        print(f"{c:10s} | {r['EX']:.4f} | {r['RS(10)']:+.4f} | {r['correct']:>4} | {r['wrong_on_unanswerable']}")


if __name__ == "__main__":
    main()
