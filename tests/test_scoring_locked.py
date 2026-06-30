"""LOCKED: EHRSQL 2024 official scoring formula — do not modify.

These tests pin the exact scoring behaviour described in the EHRSQL 2024 shared
task paper (arXiv 2405.06673) and verified against the official postprocessing.py
(scoring_program/postprocessing.py from the competition starter kit).

DO NOT change any assertion in this file without:
  1. Quoting the specific line of the official EHRSQL 2024 scoring code that
     justifies the change, and
  2. Re-running all three known baseline numbers at the bottom of this file
     against a fresh eval run to confirm they still hold.

The three immutable reference points:
  baseline (no filters)    RS(10) = 12.77%   (correct=378, abs=191, wu=42, total=1167)
  entropy-only             RS(10) = 40.87%   (correct=332, abs=225, wu=8,  total=1167)
  corrected (all fixes)    RS(10) = 53.47%   (correct=479, abs=225, wu=8,  total=1167)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Resolve imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ehrcopilot.eval.harness import EvalMetrics, _normalize_result, post_process_sql, results_match


# ===========================================================================
# Section 1 — RS(N) formula
# ===========================================================================

class TestRSFormula:
    """
    Official formula (arXiv 2405.06673, §3.2):

        score(q) = +1  if correct answer (pred SQL result == gold SQL result)
                        OR correct abstention (pred='null', gold='null')
                    0  if wrong abstention (pred='null', gold≠'null')
                        OR wrong SQL on answerable (pred SQL result ≠ gold result)
                   -1  if hallucination on unanswerable (pred SQL ≠ 'null', gold='null')

        RS(N) = mean([s * N  if s == -1  else  s   for s in scores])

    ONLY the -1 scores (hallucinations on unanswerable) are multiplied by N.
    Wrong SQL on answerable stays at 0 — it is NOT penalised at rate N.
    """

    def _metrics(
        self,
        *,
        total: int,
        answerable: int,
        correct_answers: int,
        wrong_abstentions: int,
        correct_abstentions: int,
        wrong_answers: int,   # hallucinations on unanswerable
    ) -> EvalMetrics:
        m = EvalMetrics()
        m.total = total
        m.answerable = answerable
        m.unanswerable = total - answerable
        m.correct_answers = correct_answers
        m.wrong_abstentions = wrong_abstentions
        m.correct_abstentions = correct_abstentions
        m.wrong_answers = wrong_answers
        return m

    # ── RS(N=0) is just fraction of all-correct questions ──────────────────

    def test_rs0_equals_fraction_correct(self):
        m = self._metrics(
            total=10, answerable=7,
            correct_answers=4, wrong_abstentions=2,
            correct_abstentions=3, wrong_answers=1,
        )
        # RS(0): N=0 so -1 × 0 = 0; only +1 questions contribute
        # correct_answers=4, correct_abstentions=3 → 7/10 = 0.7
        assert m.rs(0) == pytest.approx(0.7, abs=1e-9)

    # ── Wrong SQL on answerable scores 0, not −N ───────────────────────────

    def test_wrong_sql_answerable_scores_zero(self):
        """Changing wrong SQL on answerable from 0 to -1 would break the baseline."""
        # 5 answerable: 2 correct, 2 wrong SQL, 1 wrong abstention
        # 0 unanswerable hallucinations
        m = self._metrics(
            total=5, answerable=5,
            correct_answers=2, wrong_abstentions=1,
            correct_abstentions=0, wrong_answers=0,
        )
        # RS(10) = (2 + 0 - 0*10) / 5 = 0.4  (wrong_sql_answerable=2 contributes 0)
        assert m.rs(10) == pytest.approx(0.4, abs=1e-9)
        assert m.wrong_sql_answerable == 2

    def test_wrong_sql_answerable_does_not_scale_with_n(self):
        m = self._metrics(
            total=4, answerable=4,
            correct_answers=1, wrong_abstentions=1,
            correct_abstentions=0, wrong_answers=0,
        )
        # wrong_sql_answerable = 4-1-1 = 2; score=0 regardless of N
        assert m.rs(0)  == pytest.approx(0.25, abs=1e-9)
        assert m.rs(5)  == pytest.approx(0.25, abs=1e-9)
        assert m.rs(10) == pytest.approx(0.25, abs=1e-9)

    # ── Hallucinations on unanswerable are penalised at rate N ─────────────

    def test_hallucination_penalised_at_rate_n(self):
        # 2 questions: 1 unanswerable correctly abstained (+1), 1 hallucination (-1 → -N)
        m = self._metrics(
            total=2, answerable=0,
            correct_answers=0, wrong_abstentions=0,
            correct_abstentions=1, wrong_answers=1,
        )
        assert m.rs(0)  == pytest.approx((1 + 0)   / 2, abs=1e-9)
        assert m.rs(5)  == pytest.approx((1 - 5)   / 2, abs=1e-9)
        assert m.rs(10) == pytest.approx((1 - 10)  / 2, abs=1e-9)

    # ── Wrong abstention scores 0 ───────────────────────────────────────────

    def test_wrong_abstention_scores_zero(self):
        # 2 answerable: 1 correct, 1 wrong abstention (model said [ABSTAIN] on answerable)
        m = self._metrics(
            total=2, answerable=2,
            correct_answers=1, wrong_abstentions=1,
            correct_abstentions=0, wrong_answers=0,
        )
        assert m.rs(0)  == pytest.approx(0.5, abs=1e-9)
        assert m.rs(10) == pytest.approx(0.5, abs=1e-9)

    # ── Correct abstention scores +1 ───────────────────────────────────────

    def test_correct_abstention_scores_plus_one(self):
        m = self._metrics(
            total=2, answerable=0,
            correct_answers=0, wrong_abstentions=0,
            correct_abstentions=2, wrong_answers=0,
        )
        assert m.rs(0)  == pytest.approx(1.0, abs=1e-9)
        assert m.rs(10) == pytest.approx(1.0, abs=1e-9)

    # ── RS(0) denominator is total questions, not answerable ────────────────

    def test_rs_denominator_is_total_not_answerable(self):
        # 4 total, 2 answerable, 2 unanswerable
        # 1 correct SQL, 1 wrong SQL, 2 correct abstentions
        m = self._metrics(
            total=4, answerable=2,
            correct_answers=1, wrong_abstentions=0,
            correct_abstentions=2, wrong_answers=0,
        )
        # RS(0) = (1 + 2) / 4 = 0.75
        assert m.rs(0) == pytest.approx(0.75, abs=1e-9)

    # ── Zero-division guard ─────────────────────────────────────────────────

    def test_rs_zero_total_returns_zero(self):
        m = EvalMetrics()
        assert m.rs(10) == 0.0


# ===========================================================================
# Section 2 — Baseline numerical regression tests
#
# These numbers come from real eval runs on the EHRSQL 2024 test split and
# must never drift.  If they do, it means the rs() formula was changed.
# ===========================================================================

class TestBaselineRegression:
    """Pin the three known RS(10) reference points."""

    def _metrics_from_counts(self, *, correct_answers, correct_abstentions,
                              wrong_abstentions, wrong_answers, total=1167, answerable=934):
        m = EvalMetrics()
        m.total = total
        m.answerable = answerable
        m.unanswerable = total - answerable
        m.correct_answers = correct_answers
        m.wrong_abstentions = wrong_abstentions
        m.correct_abstentions = correct_abstentions
        m.wrong_answers = wrong_answers
        return m

    def test_baseline_rs0(self):
        """Baseline (no filters): RS(0) = 48.76%"""
        m = self._metrics_from_counts(
            correct_answers=378, wrong_abstentions=25,
            correct_abstentions=191, wrong_answers=42,
        )
        assert round(m.rs(0), 4) == 0.4876

    def test_baseline_rs5(self):
        """Baseline (no filters): RS(5) = 30.76%"""
        m = self._metrics_from_counts(
            correct_answers=378, wrong_abstentions=25,
            correct_abstentions=191, wrong_answers=42,
        )
        assert round(m.rs(5), 4) == 0.3076

    def test_baseline_rs10(self):
        """Baseline (no filters): RS(10) = 12.77%"""
        m = self._metrics_from_counts(
            correct_answers=378, wrong_abstentions=25,
            correct_abstentions=191, wrong_answers=42,
        )
        assert round(m.rs(10), 4) == 0.1277

    def test_entropy_only_rs10(self):
        """Entropy-only run: RS(10) = 40.87%"""
        m = self._metrics_from_counts(
            correct_answers=332, wrong_abstentions=278,
            correct_abstentions=225, wrong_answers=8,
        )
        assert round(m.rs(10), 4) == 0.4087

    def test_corrected_all_fixes_rs10(self):
        """Corrected run (all fixes + entropy + abstain-on-error): RS(10) = 53.47%"""
        m = self._metrics_from_counts(
            correct_answers=479, wrong_abstentions=374,
            correct_abstentions=225, wrong_answers=8,
        )
        assert round(m.rs(10), 4) == 0.5347

    def test_rs_formula_cross_check(self):
        """RS(0)/RS(5)/RS(10) relationship must hold for any counts."""
        m = self._metrics_from_counts(
            correct_answers=378, wrong_abstentions=25,
            correct_abstentions=191, wrong_answers=42,
        )
        # RS(N) decreases as N increases (more wrong-on-unanswerable penalty)
        assert m.rs(0) > m.rs(5) > m.rs(10)
        # RS(0) == (correct_answers + correct_abstentions) / total (N=0 cancels penalty)
        assert m.rs(0) == pytest.approx((378 + 191) / 1167, abs=1e-9)


# ===========================================================================
# Section 3 — post_process_sql locked against official postprocessing.py
# ===========================================================================

OFFICIAL_CURRENT_DATE = "2100-12-31"
OFFICIAL_CURRENT_TIME = "23:59:00"
OFFICIAL_NOW = f"{OFFICIAL_CURRENT_DATE} {OFFICIAL_CURRENT_TIME}"


class TestPostProcessSQL:
    """
    Locked against scoring_program/postprocessing.py from the EHRSQL 2024
    competition starter kit (captured 2026-06-28).
    """

    def test_current_time_replacement(self):
        sql = "SELECT * FROM t WHERE charttime > current_time"
        result = post_process_sql(sql)
        assert f"'{OFFICIAL_NOW}'" in result
        assert "current_time" not in result

    def test_current_date_replacement(self):
        sql = "SELECT * FROM t WHERE date = current_date"
        result = post_process_sql(sql)
        assert f"'{OFFICIAL_CURRENT_DATE}'" in result
        assert "current_date" not in result

    def test_now_string_replacement(self):
        sql = "SELECT * FROM t WHERE charttime < 'now'"
        result = post_process_sql(sql)
        assert f"'{OFFICIAL_NOW}'" in result
        assert "'now'" not in result

    def test_now_function_replacement(self):
        sql = "SELECT * FROM t WHERE charttime < NOW()"
        result = post_process_sql(sql)
        assert f"'{OFFICIAL_NOW}'" in result
        assert "NOW()" not in result

    def test_curdate_replacement(self):
        sql = "SELECT * FROM t WHERE date = CURDATE()"
        result = post_process_sql(sql)
        assert f"'{OFFICIAL_CURRENT_DATE}'" in result
        assert "CURDATE()" not in result

    def test_curtime_replacement(self):
        sql = "SELECT * FROM t WHERE time = CURTIME()"
        result = post_process_sql(sql)
        assert f"'{OFFICIAL_CURRENT_TIME}'" in result
        assert "CURTIME()" not in result

    def test_date_sub_year_conversion(self):
        # DATE_SUB/ADD regex matches NOW() (function call) or quoted strings, NOT bare identifiers.
        # current_time is a bare identifier so DATE_SUB(current_time, ...) is NOT converted;
        # instead current_time is first replaced with the quoted NOW string, leaving
        # DATE_SUB('2100-12-31 23:59:00', INTERVAL N UNIT) unconverted.
        # Use NOW() to exercise the conversion path.
        sql = "SELECT * FROM t WHERE charttime > DATE_SUB(NOW(), INTERVAL 1 YEAR)"
        result = post_process_sql(sql)
        assert "datetime(" in result
        assert "'-1 year'" in result
        assert "DATE_SUB" not in result

    def test_date_add_month_conversion(self):
        sql = "SELECT * FROM t WHERE d > DATE_ADD('2100-01-01', INTERVAL 3 MONTH)"
        result = post_process_sql(sql)
        assert "datetime(" in result
        assert "'+3 months'" in result
        assert "DATE_ADD" not in result

    def test_date_sub_singular_unit(self):
        """Singular unit when number is 1 (e.g. '1 year' not '1 years')."""
        sql = "SELECT * FROM t WHERE d > DATE_SUB(NOW(), INTERVAL 1 YEAR)"
        result = post_process_sql(sql)
        assert "'-1 year'" in result  # singular, not 'years'

    def test_date_sub_plural_unit(self):
        """Plural unit when number > 1."""
        sql = "SELECT * FROM t WHERE d > DATE_SUB(NOW(), INTERVAL 2 YEAR)"
        result = post_process_sql(sql)
        assert "'-2 years'" in result  # plural

    def test_current_time_in_date_sub_not_converted_to_datetime(self):
        """current_time is a bare identifier — DATE_SUB regex fires before current_time
        replacement, so the regex does NOT match, and the result is DATE_SUB with the
        quoted NOW string (not converted to SQLite datetime()). Mirrors official scorer."""
        sql = "SELECT * FROM t WHERE d > DATE_SUB(current_time, INTERVAL 1 YEAR)"
        result = post_process_sql(sql)
        # current_time IS replaced with the NOW string
        assert OFFICIAL_NOW in result
        # but DATE_SUB is NOT converted (bare identifier doesn't match regex)
        assert "DATE_SUB" in result
        assert "datetime(" not in result

    def test_vital_lower_upper_temperature(self):
        sql = "SELECT * FROM t WHERE val BETWEEN temperature_lower AND temperature_upper"
        result = post_process_sql(sql)
        assert "35.5" in result
        assert "38.1" in result
        assert "temperature_lower" not in result
        assert "temperature_upper" not in result

    def test_vital_lower_upper_heart_rate(self):
        sql = "SELECT * FROM t WHERE val BETWEEN heart_rate_lower AND heart_rate_upper"
        result = post_process_sql(sql)
        assert "60.0" in result
        assert "100.0" in result

    def test_vital_ranges_exact_values(self):
        """Pin exact vital sign range values — these come from the official PRECOMPUTED_DICT."""
        EXPECTED = {
            "temperature": (35.5, 38.1),
            "sao2": (95.0, 100.0),
            "heart_rate": (60.0, 100.0),
            "respiration": (12.0, 18.0),
            "systolic_bp": (90.0, 120.0),
            "diastolic_bp": (60.0, 90.0),
            "mean_bp": (60.0, 110.0),
        }
        for key, (lo, hi) in EXPECTED.items():
            sql = f"SELECT * FROM t WHERE val > {key}_lower AND val < {key}_upper"
            result = post_process_sql(sql)
            assert str(lo) in result, f"{key}_lower should become {lo}"
            assert str(hi) in result, f"{key}_upper should become {hi}"

    def test_percent_y_case_fix(self):
        sql = "SELECT strftime('%y', charttime)"
        result = post_process_sql(sql)
        assert "%Y" in result
        assert "%y" not in result

    def test_percent_j_case_fix(self):
        sql = "SELECT strftime('%j', charttime)"
        result = post_process_sql(sql)
        assert "%J" in result
        assert "%j" not in result

    def test_whitespace_normalisation(self):
        sql = "SELECT  *  FROM   t"
        result = post_process_sql(sql)
        assert "  " not in result

    def test_newline_removal(self):
        sql = "SELECT *\nFROM t\nWHERE x = 1"
        result = post_process_sql(sql)
        assert "\n" not in result

    def test_operator_spacing_normalisation(self):
        sql = "SELECT * FROM t WHERE x > = 1 AND y < = 2 AND z ! = 3"
        result = post_process_sql(sql)
        assert ">=" in result
        assert "<=" in result
        assert "!=" in result
        assert "> =" not in result
        assert "< =" not in result
        assert "! =" not in result

    def test_no_mutation_on_plain_sql(self):
        """SQL with no special tokens should pass through unchanged."""
        sql = "SELECT COUNT(*) FROM patients WHERE age > 65"
        result = post_process_sql(sql)
        assert result == sql


# ===========================================================================
# Section 4 — _normalize_result locked against official process_answer
# ===========================================================================

class TestNormalizeResult:
    """
    Locked against scoring_program/scoring.py::process_answer:
    - Values only (column names ignored)
    - Floats rounded to 3 dp
    - Sorted rows
    - First 100 rows only
    """

    def test_empty_returns_bracket_bracket(self):
        assert _normalize_result([]) == "[]"
        assert _normalize_result(None) == "[]"

    def test_column_names_ignored(self):
        rows_a = [{"col_a": 1, "col_b": 2}]
        rows_b = [{"alias_x": 1, "alias_y": 2}]
        assert _normalize_result(rows_a) == _normalize_result(rows_b)

    def test_row_order_independent(self):
        rows_a = [{"v": 1}, {"v": 2}]
        rows_b = [{"v": 2}, {"v": 1}]
        assert _normalize_result(rows_a) == _normalize_result(rows_b)

    def test_float_rounded_to_3dp(self):
        rows = [{"v": 1.23456789}]
        result = _normalize_result(rows)
        assert "1.235" in result

    def test_float_1dp_preserved(self):
        rows = [{"v": 1.5}]
        result = _normalize_result(rows)
        assert "1.5" in result

    def test_100_row_cap(self):
        # _normalize_result sorts ALL rows first, then caps at 100.
        # So adding a row that sorts AFTER the 100th sorted row has no effect.
        # 9999.0 sorts after all values 0–99 (lexicographic: '9999.0' > '99.0')
        rows_100 = [{"v": i} for i in range(100)]
        rows_101 = rows_100 + [{"v": 9999}]  # 9999 sorts last, outside cap
        assert _normalize_result(rows_100) == _normalize_result(rows_101)

        # A value that sorts BEFORE the cap boundary changes the result
        rows_101_front = rows_100 + [{"v": -1}]  # -1.0 sorts first, pushes out the last
        assert _normalize_result(rows_100) != _normalize_result(rows_101_front)


# ===========================================================================
# Section 5 — results_match gold-error guard
# ===========================================================================

class TestResultsMatch:
    """Gold-error guard: if gold SQL failed, never return True."""

    def test_gold_error_always_false(self):
        rows = [{"v": 1}]
        assert results_match(rows, rows, gold_err="table not found") is False

    def test_matching_rows_true(self):
        rows = [{"v": 1}]
        assert results_match(rows, rows, gold_err=None) is True

    def test_mismatching_rows_false(self):
        assert results_match([{"v": 1}], [{"v": 2}], gold_err=None) is False

    def test_pred_empty_gold_nonempty_false(self):
        assert results_match([], [{"v": 1}], gold_err=None) is False

    def test_both_empty_is_true(self):
        assert results_match([], [], gold_err=None) is True

    def test_none_pred_gold_nonempty_false(self):
        assert results_match(None, [{"v": 1}], gold_err=None) is False
