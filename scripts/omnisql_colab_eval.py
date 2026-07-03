"""OmniSQL-7B evaluation on EHRSQL 2024 MIMIC-IV — self-contained Colab script.

Run on A100 (80 GB) with bfloat16, no quantization.
K=10 hybrid retrieval (BM25 + bge-large-en-v1.5).

Files to upload to Colab (or mount from Drive):
  data/
    mimic_iv.sqlite               (36 MB)
    train/data.json               (~2 MB)
    train/label.json              (~1 MB)
    train_aug/data.json           (~13 MB)
    train_aug/label.json          (~9 MB)
    test/data.json                (~0.5 MB)
    test/label.json               (~0.1 MB)
    train_combined_embeddings_bge_large.npy  (164 MB — 40030 × 1024 float32)

Total upload: ~190 MB (or ~26 MB if you skip the .npy and let Colab recompute it)

Usage:
  !python omnisql_colab_eval.py
  !python omnisql_colab_eval.py --no-few-shot     # ablation without retrieval
  !python omnisql_colab_eval.py --repair          # enable execution-guided repair
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# Uncomment and run first in Colab:
# !pip install -q rank_bm25 sentence-transformers transformers accelerate

# ============================================================
# CELL 2 — Imports and configuration
# ============================================================
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration — edit these paths if your layout differs
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")               # folder containing the files listed above
DB_PATH  = DATA_DIR / "mimic_iv.sqlite"
TRAIN_DIR    = DATA_DIR / "train"
TRAIN_AUG_DIR = DATA_DIR / "train_aug"
TEST_DIR     = DATA_DIR / "test"
EMBED_CACHE  = DATA_DIR / "train_combined_embeddings_bge_large.npy"

MODEL_ID     = "seeklhy/OmniSQL-7B"
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
FEW_SHOT_K   = 10
MAX_NEW_TOKENS = 1024
OUTPUT_FILE  = "omnisql_7b_results.json"
PREDS_FILE   = "omnisql_7b_preds.jsonl"

# EHRSQL 2024 synthetic "current time" — all MIMIC-IV dates are in year 2100
_CURRENT_TIME      = "2100-12-31 23:59:00"
_CURRENT_DATE      = "2100-12-31"
_CURRENT_TIME_ONLY = "23:59:00"

SYSTEM_PROMPT = (
    "You are a clinical analytics assistant. Convert the user's question into "
    "a valid SQLite SELECT query over the EHRSQL-MIMIC-IV database.\n\n"
    "Output exactly [ABSTAIN] — nothing else — when the question asks about "
    "information NOT derivable from the schema below. This includes: "
    "doctor or provider identities, family history, drug side effects or "
    "pharmacology, future or scheduled hospital visits, patient contact "
    "information, or any concept not represented by the tables and columns "
    "listed below.\n\n"
    "Otherwise output only the SQL query with no explanation.\n\n"
    "Database schema (EHRSQL-MIMIC-IV):\n"
    "  patients(row_id, subject_id, gender, dod)\n"
    "  admissions(row_id, subject_id, hadm_id, admittime, dischtime, "
    "admission_type, admission_location, discharge_location, insurance, "
    "language, marital_status, age)\n"
    "  d_icd_diagnoses(row_id, icd_code, long_title)\n"
    "  d_icd_procedures(row_id, icd_code, long_title)\n"
    "  d_labitems(row_id, itemid, label)\n"
    "  d_items(row_id, itemid, label, abbreviation, linksto)\n"
    "  diagnoses_icd(row_id, subject_id, hadm_id, icd_code, charttime)\n"
    "  procedures_icd(row_id, subject_id, hadm_id, icd_code, charttime)\n"
    "  labevents(row_id, subject_id, hadm_id, itemid, charttime, valuenum, valueuom)\n"
    "  chartevents(row_id, subject_id, hadm_id, stay_id, itemid, charttime, valuenum, valueuom)\n"
    "  prescriptions(row_id, subject_id, hadm_id, starttime, stoptime, drug, "
    "dose_val_rx, dose_unit_rx, route)\n"
    "  icustays(row_id, subject_id, hadm_id, stay_id, first_careunit, "
    "last_careunit, intime, outtime)\n"
    "  transfers(row_id, subject_id, hadm_id, transfer_id, eventtype, "
    "careunit, intime, outtime)\n"
    "  microbiologyevents(row_id, subject_id, hadm_id, charttime, "
    "spec_type_desc, test_name, org_name)\n"
    "  inputevents(row_id, subject_id, hadm_id, stay_id, starttime, "
    "itemid, totalamount, totalamountuom)\n"
    "  outputevents(row_id, subject_id, hadm_id, stay_id, charttime, "
    "itemid, value, valueuom)\n"
    "  cost(row_id, subject_id, hadm_id, event_type, event_id, chargetime, cost)\n"
    "Foreign keys: admissions.subject_id→patients.subject_id | "
    "icustays.hadm_id→admissions.hadm_id | icustays.subject_id→patients.subject_id | "
    "diagnoses_icd.hadm_id→admissions.hadm_id | "
    "diagnoses_icd.icd_code=d_icd_diagnoses.icd_code | "
    "procedures_icd.hadm_id→admissions.hadm_id | "
    "procedures_icd.icd_code=d_icd_procedures.icd_code | "
    "labevents.hadm_id→admissions.hadm_id | labevents.itemid→d_labitems.itemid | "
    "prescriptions.hadm_id→admissions.hadm_id | "
    "chartevents.stay_id→icustays.stay_id | chartevents.itemid→d_items.itemid | "
    "microbiologyevents.hadm_id→admissions.hadm_id | "
    "inputevents.stay_id→icustays.stay_id | inputevents.itemid→d_items.itemid | "
    "outputevents.stay_id→icustays.stay_id | outputevents.itemid→d_items.itemid | "
    "cost.hadm_id→admissions.hadm_id | transfers.hadm_id→admissions.hadm_id"
)

ABSTAIN = "[ABSTAIN]"

# ============================================================
# CELL 3 — Data loading
# ============================================================

def load_split(split_dir: Path) -> list[dict]:
    """Load EHRSQL 2024 directory split → list of {id, question, sql, is_answerable}."""
    data   = json.load(open(split_dir / "data.json"))["data"]
    labels = json.load(open(split_dir / "label.json"))
    examples = []
    for ex in data:
        sql = labels.get(ex["id"], "null")
        is_answerable = sql.strip().lower() not in ("null", "none", "n/a", "")
        examples.append({
            "id": ex["id"],
            "question": ex["question"],
            "sql": sql if is_answerable else "",
            "is_answerable": is_answerable,
        })
    return examples

# ============================================================
# CELL 4 — SQL post-processing (mirrors official EHRSQL scorer)
# ============================================================

_TIME_PATTERN = re.compile(
    r"(DATE_SUB|DATE_ADD)\((\w+\(\)|'[^']+')[, ]+INTERVAL (\d+) (MONTH|YEAR|DAY)\)",
    re.IGNORECASE,
)
_VITAL_RANGES = {
    "temperature":  (35.5,  38.1),
    "sao2":         (95.0,  100.0),
    "heart rate":   (60.0,  100.0),
    "respiration":  (12.0,   18.0),
    "systolic bp":  (90.0,  120.0),
    "diastolic bp": (60.0,   90.0),
    "mean bp":      (60.0,  110.0),
}
_VITAL_LOWER_RE = re.compile(r"[ \n]+([a-zA-Z0-9_]+_lower)")
_VITAL_UPPER_RE = re.compile(r"[ \n]+([a-zA-Z0-9_]+_upper)")


def _date_fn_to_sqlite(m: re.Match) -> str:
    fn, date, n, unit = m.group(1).upper(), m.group(2), m.group(3), m.group(4).lower()
    unit = unit.rstrip("s") if n == "1" else (unit if unit.endswith("s") else unit + "s")
    sign = "-" if fn == "DATE_SUB" else "+"
    return f"datetime({date}, '{sign}{n} {unit}')"


def post_process_sql(sql: str) -> str:
    sql = re.sub(r"[ ]+", " ", sql.replace("\n", " ")).strip()
    sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    sql = _TIME_PATTERN.sub(_date_fn_to_sqlite, sql)
    for placeholder, value in [
        ("current_time", f"'{_CURRENT_TIME}'"),
        ("current_date", f"'{_CURRENT_DATE}'"),
        ("'now'",        f"'{_CURRENT_TIME}'"),
        ("NOW()",        f"'{_CURRENT_TIME}'"),
        ("CURDATE()",    f"'{_CURRENT_DATE}'"),
        ("CURTIME()",    f"'{_CURRENT_TIME_ONLY}'"),
    ]:
        if placeholder in sql:
            sql = sql.replace(placeholder, value)
    lower_m = _VITAL_LOWER_RE.search(sql)
    upper_m = _VITAL_UPPER_RE.search(sql)
    if lower_m and upper_m:
        lower_expr = lower_m.group(1)
        upper_expr = upper_m.group(1)
        vital_names = list(set(
            re.findall(r"([a-zA-Z0-9_]+)_lower", lower_expr) +
            re.findall(r"([a-zA-Z0-9_]+)_upper", upper_expr)
        ))
        if len(vital_names) == 1:
            vital_key = vital_names[0].replace("_", " ")
            if vital_key in _VITAL_RANGES:
                lo, hi = _VITAL_RANGES[vital_key]
                sql = sql.replace(lower_expr, str(lo)).replace(upper_expr, str(hi))
    return sql.replace("%y", "%Y").replace("%j", "%J")


def _normalize_item(v) -> str:
    try:
        return str(round(float(v), 3))
    except (TypeError, ValueError):
        return str(v)


def _normalize_result(rows) -> str:
    if not rows:
        return "[]"
    return str(sorted([[_normalize_item(v) for v in row.values()] for row in rows])[:100])


def exec_sql(sql: str) -> tuple[list | None, str | None]:
    """Execute SQL against the MIMIC-IV SQLite DB. Returns (rows, error)."""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchmany(100)]
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)

# ============================================================
# CELL 5 — Hybrid retriever (BM25 + bge-large RRF, K=10)
# ============================================================

def build_hybrid_retriever(
    train_examples: list[dict],
    embed_cache: Path,
    k: int = 10,
) -> "callable":
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    # --- BM25 ---
    tokenized = [ex["question"].lower().split() for ex in train_examples]
    bm25 = BM25Okapi(tokenized)
    n = len(train_examples)

    # --- Embeddings ---
    def _sql_skeleton(sql: str) -> str:
        s = sql.lower().strip()
        s = re.sub(r"'[^']*'", "'X'", s)
        s = re.sub(r'\b\d+(?:\.\d+)?\b', 'N', s)
        return re.sub(r'\s+', ' ', s).strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device} ...")
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)

    if embed_cache.exists():
        train_embeds = np.load(str(embed_cache)).astype(np.float32)
        if train_embeds.shape[0] == n:
            print(f"  Loaded cached embeddings: {train_embeds.shape}")
        else:
            print(f"  Cache shape mismatch ({train_embeds.shape[0]} vs {n}) — rebuilding")
            train_embeds = None
    else:
        train_embeds = None

    if train_embeds is None:
        print(f"  Computing {n} embeddings (this takes ~10 min on GPU) ...")
        index_texts = [ex["question"] + " " + _sql_skeleton(ex["sql"]) for ex in train_examples]
        train_embeds = embed_model.encode(
            index_texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True,
        ).astype(np.float32)
        np.save(str(embed_cache), train_embeds)
        print(f"  Saved to {embed_cache}")

    train_embeds_t = torch.from_numpy(train_embeds).to(device)

    def retrieve(question: str) -> list[int]:
        # BM25 ranks
        bm25_scores = bm25.get_scores(question.lower().split())
        bm25_order  = (-bm25_scores).argsort()
        bm25_ranks  = np.empty(n, dtype=np.float32)
        bm25_ranks[bm25_order] = np.arange(1, n + 1, dtype=np.float32)

        # Embedding ranks
        q_vec = embed_model.encode([question], normalize_embeddings=True)
        q_t   = torch.from_numpy(q_vec.astype(np.float32)).to(device)
        cos   = (train_embeds_t @ q_t.T).squeeze().cpu().numpy()
        embed_order  = (-cos).argsort()
        embed_ranks  = np.empty(n, dtype=np.float32)
        embed_ranks[embed_order] = np.arange(1, n + 1, dtype=np.float32)

        # RRF fusion
        rrf = 1.0 / (60 + bm25_ranks) + 1.0 / (60 + embed_ranks)
        top = list(map(int, (-rrf).argsort()[:k]))
        return top

    def format_examples(question: str) -> str:
        indices = retrieve(question)
        lines = ["Similar examples:"]
        for i in indices:
            ex = train_examples[i]
            lines.append(f"Q: {ex['question']}")
            lines.append(f"SQL: {ex['sql']}")
        return "\n".join(lines)

    return format_examples

# ============================================================
# CELL 6 — OmniSQL-7B model loading (bfloat16, no quantization)
# ============================================================

def load_omnisql(model_id: str = MODEL_ID):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {model_id} in bfloat16 ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded. Devices: {set(str(p.device) for p in model.parameters())}")
    return model, tokenizer


def _strip_fences(text: str) -> str:
    """Extract SQL from markdown code fences if present."""
    m = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return _strip_fences(text)

# ============================================================
# CELL 7 — Evaluation loop
# ============================================================

def evaluate(
    test_examples: list[dict],
    model,
    tokenizer,
    retriever=None,
    enable_repair: bool = False,
    save_preds_path: Path | None = None,
) -> dict:
    total = len(test_examples)
    answerable_total    = sum(1 for e in test_examples if e["is_answerable"])
    unanswerable_total  = total - answerable_total

    correct_answers     = 0
    wrong_abstentions   = 0
    wrong_sql_ans       = 0   # wrong SQL on answerable
    wrong_on_unans      = 0   # any SQL on unanswerable
    correct_abstentions = 0
    gold_exec_valid     = 0
    gold_exec_empty     = 0
    gold_exec_error     = 0
    repair_attempts     = 0
    repair_successes    = 0
    latencies: list[float] = []
    preds_log: list[dict]  = []

    preds_fh = open(save_preds_path, "w") if save_preds_path else None

    for idx, ex in enumerate(test_examples):
        t0 = time.time()
        q  = ex["question"]

        # Build user content (with or without few-shot)
        user_content = q
        if retriever is not None:
            examples_block = retriever(q)
            user_content = f"{examples_block}\n\nQuestion: {q}"

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        predicted = generate(model, tokenizer, msgs)
        abstained = (predicted == ABSTAIN or not predicted)

        # Execution-guided repair for SQL errors
        if not abstained and enable_repair:
            _, err = exec_sql(post_process_sql(predicted))
            if err:
                for _ in range(3):
                    repair_attempts += 1
                    repair_msgs = msgs + [
                        {"role": "assistant", "content": predicted},
                        {"role": "user", "content": (
                            f"SQLite error: {err}\n"
                            "Fix the SQL using only tables and columns in the schema above. "
                            "Output only the corrected SQL (no explanation)."
                        )},
                    ]
                    repaired = generate(model, tokenizer, repair_msgs)
                    _, new_err = exec_sql(post_process_sql(repaired))
                    if new_err is None:
                        predicted = repaired
                        repair_successes += 1
                        break
                    err = new_err

        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)

        outcome = "unknown"

        if ex["is_answerable"]:
            gold_rows, gold_err = exec_sql(post_process_sql(ex["sql"]))
            if gold_err:
                gold_exec_error += 1
            elif not gold_rows:
                gold_exec_empty += 1
            else:
                gold_exec_valid += 1

            if abstained:
                wrong_abstentions += 1
                outcome = "wrong_abstention"
            else:
                pred_rows, pred_err = exec_sql(post_process_sql(predicted))
                if pred_err is None and _normalize_result(pred_rows) == _normalize_result(gold_rows):
                    correct_answers += 1
                    outcome = "correct"
                else:
                    wrong_sql_ans += 1
                    outcome = "wrong_sql"
        else:
            if abstained:
                correct_abstentions += 1
                outcome = "correct_abstention"
            else:
                wrong_on_unans += 1
                outcome = "hallucination"

        row = {
            "id": ex["id"],
            "question": q,
            "gold_sql": ex["sql"],
            "predicted_sql": predicted,
            "outcome": outcome,
            "latency_ms": round(elapsed_ms, 1),
        }
        preds_log.append(row)
        if preds_fh:
            preds_fh.write(json.dumps(row) + "\n")
            preds_fh.flush()

        # Progress report every 25 questions
        if (idx + 1) % 25 == 0 or idx == total - 1:
            done = idx + 1
            ex_so_far = correct_answers / max(1, sum(
                1 for r in preds_log if test_examples[preds_log.index(r)]["is_answerable"]
                if r["outcome"] in ("correct", "wrong_sql", "wrong_abstention")
            )) if any(r["outcome"] in ("correct", "wrong_sql", "wrong_abstention")
                      for r in preds_log) else 0.0
            rs10 = (correct_answers + correct_abstentions - 10 * wrong_on_unans) / done
            eta_s = (elapsed_ms / 1000) * (total - done)
            print(f"  [{done:4d}/{total}] "
                  f"correct={correct_answers} hall={wrong_on_unans} "
                  f"RS10_so_far={rs10:.3f}  "
                  f"lat={elapsed_ms:.0f}ms  ETA≈{eta_s/60:.1f}min")

    if preds_fh:
        preds_fh.close()

    lat_sorted = sorted(latencies)
    results = {
        "model": MODEL_ID,
        "few_shot_k": FEW_SHOT_K if retriever else 0,
        "total": total,
        "answerable": answerable_total,
        "unanswerable": unanswerable_total,
        "EX":    round(correct_answers / max(1, answerable_total), 4),
        "RS(0)": round((correct_answers + correct_abstentions) / total, 4),
        "RS(5)": round((correct_answers + correct_abstentions - 5  * wrong_on_unans) / total, 4),
        "RS(10)":round((correct_answers + correct_abstentions - 10 * wrong_on_unans) / total, 4),
        "correct_answers":             correct_answers,
        "wrong_abstentions":           wrong_abstentions,
        "wrong_sql_answerable":        wrong_sql_ans,
        "wrong_answers_on_unanswerable": wrong_on_unans,
        "correct_abstentions":         correct_abstentions,
        "gold_exec_valid":             gold_exec_valid,
        "gold_exec_empty":             gold_exec_empty,
        "gold_exec_error":             gold_exec_error,
        "p50_latency_ms": round(lat_sorted[len(lat_sorted) // 2], 1),
        "p95_latency_ms": round(lat_sorted[int(len(lat_sorted) * 0.95)], 1),
    }
    if enable_repair:
        results["repair_attempts"]  = repair_attempts
        results["repair_successes"] = repair_successes
    return results

# ============================================================
# CELL 8 — Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--repair",      action="store_true")
    parser.add_argument("--output",      default=OUTPUT_FILE)
    parser.add_argument("--preds",       default=PREDS_FILE)
    parser.add_argument("--k",           type=int, default=FEW_SHOT_K)
    args = parser.parse_args()

    print("=" * 60)
    print(f" OmniSQL-7B Evaluation — EHRSQL 2024 MIMIC-IV")
    print(f" Few-shot K={args.k if not args.no_few_shot else 0}  repair={args.repair}")
    print("=" * 60)

    # Load data
    print("\nLoading data splits ...")
    train_all = [e for e in load_split(TRAIN_DIR) if e["is_answerable"]]
    aug_all   = [e for e in load_split(TRAIN_AUG_DIR) if e["is_answerable"]]
    corpus    = train_all + aug_all
    test_all  = load_split(TEST_DIR)
    print(f"  Train+aug corpus: {len(corpus):,}  |  Test: {len(test_all):,}")

    # Build retriever
    retriever = None
    if not args.no_few_shot:
        print(f"\nBuilding Hybrid retriever (K={args.k}) ...")
        retriever = build_hybrid_retriever(corpus, EMBED_CACHE, k=args.k)

    # Load model
    print(f"\nLoading {MODEL_ID} ...")
    model, tokenizer = load_omnisql(MODEL_ID)

    # Run eval
    print(f"\nEvaluating on {len(test_all)} test questions ...")
    results = evaluate(
        test_all, model, tokenizer,
        retriever=retriever,
        enable_repair=args.repair,
        save_preds_path=Path(args.preds),
    )

    # Save + print
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    for k in ["EX", "RS(0)", "RS(5)", "RS(10)"]:
        print(f"  {k:8s}: {results[k]:.4f}")
    print()
    for k in ["correct_answers", "wrong_abstentions", "wrong_sql_answerable",
              "wrong_answers_on_unanswerable", "correct_abstentions"]:
        print(f"  {k}: {results[k]}")
    print(f"\n  Latency p50={results['p50_latency_ms']}ms  p95={results['p95_latency_ms']}ms")
    print(f"\n  Results → {args.output}")
    print(f"  Predictions → {args.preds}")


if __name__ == "__main__":
    main()
