"""Aggressive template re-instantiation augmenter for EHRSQL-2024.

EHRSQL questions are templated: every train example in `annotated.json` carries its
question `template` and a `val_dict` of the concrete values that were slotted in. The
dataset itself was *built* by instantiating ~100 templates with values — so we can grow
it the same way: for each answerable example, swap each value placeholder for other real
values **sampled from the actual mimic_iv.sqlite**, substituting into BOTH the question
and the SQL, and keeping only variants whose SQL executes. Because values come from the
column being filtered, the WHERE clause matches real rows (mostly non-empty results).

We also synthesize extra *unanswerable* questions (asking for attributes absent from the
schema — address, phone, physician, manufacturer, future predictions) to strengthen the
abstention signal that drives RS.

Output mirrors the official split layout so prepare_sft / build_pairs read it identically:
  <out>/data.json       {"data": [{"id","question"}, ...]}
  <out>/label.json      {id: sql | "null"}
  <out>/annotated.json  [{id,question,query,template,val_dict,is_orig}, ...]

Usage:
  python -m ehrcopilot.finetune.augment_ehrsql2024 \
      --annotated data/ehrsql2024/mimic_iv/train/annotated.json \
      --db data/ehrsql2024/mimic_iv/mimic_iv.sqlite \
      --out data/ehrsql2024/mimic_iv/train_aug \
      --target-answerable 40000 --synthetic-unanswerable 3000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

NULL = "null"


# ---------------------------------------------------------------------------
# Value pools from the live DB
# ---------------------------------------------------------------------------
class ValuePools:
    def __init__(self, db_path: str):
        self.con = sqlite3.connect(db_path)
        self.con.text_factory = lambda b: b.decode(errors="ignore")
        self._cache: dict[tuple[str, str], list] = {}

    def pool(self, table: str, col: str, limit: int = 1500) -> list:
        key = (table.lower(), col.lower())
        if key not in self._cache:
            try:
                rows = self.con.execute(
                    f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {limit}'
                ).fetchall()
                self._cache[key] = [r[0] for r in rows if r[0] is not None and str(r[0]) != ""]
            except Exception:
                self._cache[key] = []
        return self._cache[key]

    def executes(self, sql: str) -> tuple[bool, bool]:
        """Return (ok, nonempty)."""
        try:
            rows = self.con.execute(sql).fetchall()
            nonempty = bool(rows) and any(
                any(v not in (None, 0, "0", "") for v in r) for r in rows
            )
            return True, nonempty
        except Exception:
            return False, False


# ---------------------------------------------------------------------------
# Locate the table.column a value is filtered on, inside the SQL
# ---------------------------------------------------------------------------
def _locate_column(query: str, value) -> tuple[str, str] | None:
    """Find `table.col <op> <value>` in the SQL and return (table, col)."""
    q = query
    sval = str(value)
    if isinstance(value, str):
        # string literal: table.col = 'value'  /  IN ('value'  / LIKE 'value'
        m = re.search(
            r"(\w+)\.(\w+)\s*(?:=|<>|!=|like|in\s*\()\s*'" + re.escape(sval) + r"'",
            q, flags=re.IGNORECASE,
        )
    else:
        # numeric literal: table.col = 123 (word-bounded so 10 != 100)
        m = re.search(
            r"(\w+)\.(\w+)\s*(?:=|<>|!=|<=|>=|<|>|in\s*\()\s*" + re.escape(sval) + r"(?![0-9.])",
            q, flags=re.IGNORECASE,
        )
    if m:
        return m.group(1), m.group(2)
    return None


def _sub_value(text: str, old, new, in_sql: bool) -> str | None:
    """Replace old value with new in question (case-insensitive) or SQL (quoted/bounded).
    Returns None if the old value isn't present (so we never desync question vs SQL)."""
    so, sn = str(old), str(new)
    if isinstance(old, str):
        if in_sql:
            needle = "'" + so + "'"
            if needle not in text:
                return None
            return text.replace(needle, "'" + sn + "'")
        # question: case-insensitive
        if not re.search(re.escape(so), text, flags=re.IGNORECASE):
            return None
        return re.sub(re.escape(so), sn, text, flags=re.IGNORECASE)
    # numeric: word-bounded
    pat = r"(?<![0-9.])" + re.escape(so) + r"(?![0-9.])"
    if not re.search(pat, text):
        return None
    return re.sub(pat, sn, text)


# ---------------------------------------------------------------------------
# Synthetic unanswerable templates (attributes absent from the EHRSQL-2024 schema)
# ---------------------------------------------------------------------------
_UNANS_PATIENT = [
    "What is the home address of patient {pid}?",
    "What is the phone number of patient {pid}?",
    "What is the email address of patient {pid}?",
    "Who is the attending physician for patient {pid}?",
    "What is patient {pid}'s social security number?",
    "What is the insurance premium paid by patient {pid}?",
    "What is the next of kin for patient {pid}?",
    "Predict the next diagnosis for patient {pid}.",
    "What will patient {pid}'s blood pressure be tomorrow?",
    "What is the religion of patient {pid}?",
    "What is the occupation of patient {pid}?",
    "How tall is patient {pid} in centimeters?",
]
_UNANS_DRUG = [
    "Who is the manufacturer of {drug}?",
    "What is the retail price of {drug} at the pharmacy?",
    "What is the patent expiry date of {drug}?",
    "What are the known side effects of {drug}?",
    "Which country produces {drug}?",
]


def build_synthetic_unanswerable(pools: ValuePools, n: int, seed: int) -> list[dict]:
    rnd = random.Random(seed + 7)
    pids = pools.pool("patients", "subject_id")
    drugs = pools.pool("prescriptions", "drug")
    out: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        if drugs and rnd.random() < 0.35:
            q = rnd.choice(_UNANS_DRUG).format(drug=rnd.choice(drugs))
        elif pids:
            q = rnd.choice(_UNANS_PATIENT).format(pid=rnd.choice(pids))
        else:
            break
        if q in seen:
            continue
        seen.add(q)
        out.append({"question": q, "query": NULL, "template": "-synthetic-unans-", "val_dict": {}, "is_orig": False})
    return out


# ---------------------------------------------------------------------------
# Augment
# ---------------------------------------------------------------------------
def augment(
    annotated_path: Path,
    db_path: Path,
    out_dir: Path,
    target_answerable: int,
    synthetic_unanswerable: int,
    seed: int = 42,
) -> dict:
    rnd = random.Random(seed)
    pools = ValuePools(str(db_path))
    records = json.load(open(annotated_path))
    if isinstance(records, dict) and "data" in records:
        records = records["data"]

    answerable = [r for r in records if str(r.get("query", "")).strip().lower() != NULL and r.get("query")]
    unanswerable = [r for r in records if str(r.get("query", "")).strip().lower() == NULL or not r.get("query")]

    # Group answerable by template for balanced quotas
    by_tmpl: dict[str, list] = defaultdict(list)
    for r in answerable:
        by_tmpl[r.get("template", "-")].append(r)
    n_tmpl = max(1, len(by_tmpl))
    quota = max(1, target_answerable // n_tmpl)

    out_records: list[dict] = []
    seen: set[str] = set()

    def add(question: str, query: str, template: str, val_dict: dict, is_orig: bool) -> bool:
        k = question.strip().lower() + " ||| " + " ".join(query.lower().split())
        if k in seen:
            return False
        seen.add(k)
        out_records.append({
            "question": question, "query": query, "template": template,
            "val_dict": val_dict, "is_orig": is_orig,
        })
        return True

    stats = {"templates": n_tmpl, "quota_per_template": quota,
             "orig_answerable": len(answerable), "orig_unanswerable": len(unanswerable),
             "gen_attempts": 0, "gen_exec_fail": 0, "gen_dupe": 0,
             "gen_nonempty": 0, "gen_empty_ok": 0}

    for tmpl, group in by_tmpl.items():
        # always keep originals
        for r in group:
            add(r["question"], r["query"], tmpl, r.get("val_dict", {}), True)
        made = 0
        gi = 0
        guard = 0
        max_guard = quota * 12 + 50
        while made < quota - len(group) and guard < max_guard:
            guard += 1
            base = group[gi % len(group)]
            gi += 1
            vd = (base.get("val_dict") or {}).get("val_placeholder") or {}
            if not vd:
                continue
            new_q, new_sql = base["question"], base["query"]
            varied = False
            for ph, val in vd.items():
                loc = _locate_column(base["query"], val)
                if not loc:
                    continue
                pool = pools.pool(*loc)
                if len(pool) < 2:
                    continue
                newval = rnd.choice(pool)
                if str(newval) == str(val):
                    continue
                q2 = _sub_value(new_q, val, newval, in_sql=False)
                s2 = _sub_value(new_sql, val, newval, in_sql=True)
                if q2 is None or s2 is None:
                    continue
                new_q, new_sql = q2, s2
                varied = True
            if not varied:
                continue
            stats["gen_attempts"] += 1
            ok, nonempty = pools.executes(new_sql)
            if not ok:
                stats["gen_exec_fail"] += 1
                continue
            if not add(new_q, new_sql, tmpl, {}, False):
                stats["gen_dupe"] += 1
                continue
            made += 1
            stats["gen_nonempty" if nonempty else "gen_empty_ok"] += 1

    # originals + synthetic unanswerables
    for r in unanswerable:
        add(r["question"], NULL, r.get("template", "-"), {}, True)
    synth = build_synthetic_unanswerable(pools, synthetic_unanswerable, seed)
    for r in synth:
        add(r["question"], NULL, r["template"], {}, False)

    rnd.shuffle(out_records)

    # assign deterministic ids and write official layout
    out_dir.mkdir(parents=True, exist_ok=True)
    data, label, annotated = [], {}, []
    for i, r in enumerate(out_records):
        rid = hashlib.md5(f"{i}|{r['question']}|{r['query']}".encode()).hexdigest()[:24]
        data.append({"id": rid, "question": r["question"]})
        label[rid] = r["query"]
        annotated.append({"id": rid, **r})
    json.dump({"version": "ehrsql2024-aug", "data": data}, open(out_dir / "data.json", "w"))
    json.dump(label, open(out_dir / "label.json", "w"))
    json.dump(annotated, open(out_dir / "annotated.json", "w"))

    n_ans = sum(1 for v in label.values() if str(v).lower() != NULL)
    n_unans = len(label) - n_ans
    stats.update({"final_total": len(label), "final_answerable": n_ans, "final_unanswerable": n_unans,
                  "synthetic_unanswerable": len(synth)})
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotated", default="data/ehrsql2024/mimic_iv/train/annotated.json")
    p.add_argument("--db", default="data/ehrsql2024/mimic_iv/mimic_iv.sqlite")
    p.add_argument("--out", default="data/ehrsql2024/mimic_iv/train_aug")
    p.add_argument("--target-answerable", type=int, default=40000)
    p.add_argument("--synthetic-unanswerable", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    stats = augment(Path(args.annotated), Path(args.db), Path(args.out),
                    args.target_answerable, args.synthetic_unanswerable, args.seed)
    print("Augmentation complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
