"""Official EHRSQL-2024 Reliability-Score logic, vendored verbatim.

Source: glee4810/ehrsql-2024 → scoring_program/scoring_utils.py (NAACL 2024 ClinicalNLP
shared task). Reproduced here so our eval is byte-for-byte comparable to the leaderboard
(RS(10)=0.8132). DO NOT "improve" these functions — fidelity to the official scorer is the
whole point.

Per-sample reliability score g/Acc cases:
  answerable & exec-correct        -> +1
  answerable & abstain ("null")    ->  0
  answerable & exec-wrong          -> -1   (scaled by penalty c)
  unanswerable & answered          -> -1   (scaled by penalty c)
  unanswerable & abstain           -> +1
RS(c) = mean over all samples, with each -1 multiplied by c.
"""

from __future__ import annotations

import sqlite3
from ast import literal_eval


def process_item(item):
    try:
        item = round(float(item), 3)
    except Exception:
        pass
    return str(item)


def process_answer(ans):
    try:
        ans = literal_eval(ans)
    except Exception:
        pass
    if type(ans) == str:
        return ans
    else:
        return str(sorted([[process_item(c) for c in row] for row in ans])[:100])


def execute_sql(sql, db_path):
    con = sqlite3.connect(db_path)
    con.text_factory = lambda b: b.decode(errors="ignore")
    cur = con.cursor()
    result = cur.execute(sql).fetchall()
    con.close()
    return result


def execute_sql_wrapper(key, sql, db_path, tag, skip_indicator="null"):
    assert tag in ["real", "pred"]
    if sql != skip_indicator:
        try:
            result = execute_sql(sql, db_path)
        except Exception:
            result = "error_" + tag
        result = process_answer(result)
        return (key, result)
    else:
        return (key, skip_indicator)


def execute_all(sql_dict, db_path, tag):
    exec_result = {}
    for key in sql_dict:
        sql = sql_dict[key]
        exec_result[key] = execute_sql_wrapper(key, sql, db_path, tag)[-1]
    return exec_result


def reliability_score(real_result, pred_result, return_dict=False):
    reliability = []
    reliability_dict = {}
    for key in real_result:
        ans_real = real_result[key]
        ans_pred = pred_result[key]
        exec_acc = ans_real == ans_pred

        if ans_real != "null" and exec_acc is True:
            score = 1
        elif ans_real != "null" and ans_pred == "null":
            score = 0
        elif ans_real != "null" and exec_acc is False:
            score = -1
        elif ans_real == "null" and ans_pred != "null":
            score = -1
        elif ans_real == "null" and ans_pred == "null":
            score = 1
        else:  # pragma: no cover
            raise NotImplementedError
        reliability.append(score)
        reliability_dict[key] = score

    if return_dict:
        return reliability, reliability_dict
    return reliability


def penalize(scores, penalty=1):
    # Pure-Python equivalent of the official np.mean(...) — avoids a numpy dependency.
    if not scores:
        return 0.0
    vals = [s * penalty if s == -1 else s for s in scores]
    return float(sum(vals) / len(vals))


def score_predictions(real_dict: dict, pred_dict: dict, db_path: str) -> dict:
    """Compute the full leaderboard scoreboard for a prediction dict.

    real_dict / pred_dict map id -> SQL string (or "null" to abstain). Returns RS(0/5/10/N)
    as percentages plus the per-case breakdown (EX = answered-correct / answerable)."""
    real = execute_all(real_dict, db_path, "real")
    pred = execute_all(pred_dict, db_path, "pred")
    scores, per = reliability_score(real, pred, return_dict=True)
    n = len(scores)

    answerable = sum(1 for k in real if real[k] != "null")
    correct_answers = sum(1 for k, s in per.items() if real[k] != "null" and s == 1)
    wrong_abstain = sum(1 for k, s in per.items() if real[k] != "null" and pred[k] == "null")
    wrong_answers_ans = sum(1 for k, s in per.items() if real[k] != "null" and s == -1)
    wrong_answers_unans = sum(1 for k, s in per.items() if real[k] == "null" and s == -1)
    correct_abstain = sum(1 for k, s in per.items() if real[k] == "null" and s == 1)

    return {
        "total": n,
        "answerable": answerable,
        "unanswerable": n - answerable,
        "EX": round(correct_answers / answerable, 4) if answerable else 0.0,
        "RS(0)": round(penalize(scores, 0) * 100, 4),
        "RS(5)": round(penalize(scores, 5) * 100, 4),
        "RS(10)": round(penalize(scores, 10) * 100, 4),
        "RS(N)": round(penalize(scores, n) * 100, 4),
        "correct_answers": correct_answers,
        "wrong_abstentions_on_answerable": wrong_abstain,
        "wrong_answers_on_answerable": wrong_answers_ans,
        "wrong_answers_on_unanswerable": wrong_answers_unans,
        "correct_abstentions": correct_abstain,
    }
