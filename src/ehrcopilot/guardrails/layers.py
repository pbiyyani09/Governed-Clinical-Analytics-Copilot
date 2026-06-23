"""5-layer AST guardrail stack for governed EHR SQL access.

Layer 1 — Read-only / single-statement enforcement (AST pre-execution)
Layer 2 — Table + column allowlist enforcement (AST pre-execution)
Layer 3 — PHI-column hard block (AST pre-execution)
Layer 4 — Small-cell suppression / k-anonymity (post-execution)
Layer 5 — Prompt-injection detection (pre-SQL-generation, on NL input)

All pre-execution layers run before any SQL touches the database.
Layer 4 runs after execution on the returned result set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sqlglot
import sqlglot.expressions as exp

from ehrcopilot import config


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    layer: int       # 0 = no layer checked yet; 1-5 = the layer that blocked
    reason: str      # human-readable explanation; empty string if passed=True
    detail: str = "" # extra detail (e.g. offending column name)

    @classmethod
    def ok(cls) -> "GuardResult":
        return cls(passed=True, layer=0, reason="")

    @classmethod
    def block(cls, layer: int, reason: str, detail: str = "") -> "GuardResult":
        return cls(passed=False, layer=layer, reason=reason, detail=detail)


# ---------------------------------------------------------------------------
# Layer 1 — Read-only / single-statement
# ---------------------------------------------------------------------------

_WRITE_NODE_TYPES = (
    exp.Drop,
    exp.Create,
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,  # catches EXEC, xp_, etc.
)


def layer1_readonly(sql: str) -> GuardResult:
    """Reject DDL, DML, multi-statement, or non-SELECT queries."""
    # Fast pre-parse: multi-statement via semicolon (after stripping comments)
    stripped = re.sub(r"--[^\n]*", "", sql)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    if stripped.count(";") > 1 or (stripped.count(";") == 1 and not stripped.strip().endswith(";")):
        return GuardResult.block(1, "Multi-statement SQL is not permitted", sql[:120])

    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
    except Exception as exc:
        return GuardResult.block(1, f"SQL parse error: {exc}", sql[:120])

    if len(statements) == 0:
        return GuardResult.block(1, "Empty SQL statement", "")

    if len(statements) > 1:
        return GuardResult.block(1, "Multi-statement SQL is not permitted", "")

    stmt = statements[0]

    # Must be a SELECT at the root
    if not isinstance(stmt, exp.Select):
        kind = type(stmt).__name__
        return GuardResult.block(1, f"Only SELECT statements are permitted; got {kind}", kind)

    # Walk the full AST for any write/DDL nodes (could be inside a subquery)
    for node in stmt.walk():
        if isinstance(node, _WRITE_NODE_TYPES):
            return GuardResult.block(
                1,
                f"Write/DDL operation detected: {type(node).__name__}",
                type(node).__name__,
            )

    return GuardResult.ok()


# ---------------------------------------------------------------------------
# Layer 2 — Table + column allowlist
# ---------------------------------------------------------------------------


def _collect_cte_names(stmt: exp.Select) -> frozenset[str]:
    """Return names of all CTEs defined in WITH clauses (these are virtual tables)."""
    names: set[str] = set()
    # sqlglot stores the WITH clause under 'with_' key (note the underscore)
    with_clause = stmt.args.get("with_")
    if with_clause:
        for cte in with_clause.expressions:
            names.add(cte.alias.lower())
    return frozenset(names)


def _collect_column_aliases(stmt: exp.Select) -> frozenset[str]:
    """Return all column alias names in the SELECT list (e.g. COUNT(*) AS cnt → 'cnt')."""
    aliases: set[str] = set()
    for expr in stmt.expressions:
        if isinstance(expr, exp.Alias):
            aliases.add(expr.alias.lower())
    return frozenset(aliases)


def _resolve_aliases(stmt: exp.Select) -> dict[str, str]:
    """Build alias → canonical_table_name map from FROM / JOIN clauses."""
    alias_map: dict[str, str] = {}
    for table_node in stmt.find_all(exp.Table):
        tname = table_node.name.lower()
        alias = table_node.alias
        if alias:
            alias_map[alias.lower()] = tname
    return alias_map


def layer2_allowlist(sql: str) -> GuardResult:
    """Reject queries referencing tables or columns not in the schema allowlist."""
    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
    except Exception as exc:
        return GuardResult.block(2, f"SQL parse error: {exc}")

    stmt = statements[0] if statements else None
    if not isinstance(stmt, exp.Select):
        return GuardResult.block(2, "Statement is not a SELECT")

    cte_names = _collect_cte_names(stmt)
    col_aliases = _collect_column_aliases(stmt)
    alias_map = _resolve_aliases(stmt)

    # Check tables — skip CTE virtual names
    for table_node in stmt.find_all(exp.Table):
        tname = table_node.name.lower()
        if not tname or tname in cte_names:
            continue
        if tname not in config.ALLOWED_TABLES:
            return GuardResult.block(2, f"Table '{tname}' is not in the schema allowlist", tname)

    # Check columns — skip star, column aliases, and CTE-derived names
    all_allowed = {c.lower() for cols in config.SCHEMA_ALLOWLIST.values() for c in cols}

    for col_node in stmt.find_all(exp.Column):
        col_name = col_node.name.lower()
        if not col_name or col_name == "*":
            continue
        # Skip aliases defined in the same SELECT list (e.g. ORDER BY cnt)
        if col_name in col_aliases:
            continue

        # Determine which table this column belongs to
        table_ref = col_node.table
        if table_ref:
            table_ref_lower = table_ref.lower()
            # Skip columns that directly reference a CTE virtual table
            if table_ref_lower in cte_names:
                continue
            # Resolve alias to canonical table name
            canonical = alias_map.get(table_ref_lower, table_ref_lower)
            # Skip if the alias resolves to a CTE name (e.g. JOIN cte AS i → i.col)
            if canonical in cte_names:
                continue
            allowed_cols = config.SCHEMA_ALLOWLIST.get(canonical, [])
            if col_name not in [c.lower() for c in allowed_cols]:
                return GuardResult.block(
                    2,
                    f"Column '{col_name}' is not in the allowlist for table '{canonical}'",
                    f"{canonical}.{col_name}",
                )
        else:
            # Unqualified column — check across all allowlisted tables
            if col_name not in all_allowed:
                return GuardResult.block(
                    2,
                    f"Column '{col_name}' is not in any allowlisted table",
                    col_name,
                )

    return GuardResult.ok()


# ---------------------------------------------------------------------------
# Layer 3 — PHI-column hard block
# ---------------------------------------------------------------------------


def layer3_phi_block(sql: str) -> GuardResult:
    """Hard-block any query referencing PHI columns, regardless of table context."""
    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
    except Exception as exc:
        return GuardResult.block(3, f"SQL parse error: {exc}")

    stmt = statements[0] if statements else None
    if not isinstance(stmt, exp.Select):
        return GuardResult.block(3, "Statement is not a SELECT")

    for col_node in stmt.find_all(exp.Column):
        col_name = col_node.name.lower()
        if col_name in config.PHI_COLUMNS:
            return GuardResult.block(
                3,
                f"PHI column '{col_name}' is blocked by policy",
                col_name,
            )

    return GuardResult.ok()


# ---------------------------------------------------------------------------
# Layer 4 — Small-cell suppression (k-anonymity) — POST execution
# ---------------------------------------------------------------------------


def layer4_small_cell(
    rows: list[dict[str, Any]],
    sql: str,
) -> GuardResult:
    """Suppress results where a count or group is below the k-anonymity threshold.

    Must be called AFTER execution with the returned rows.
    """
    k = config.K_THRESHOLD
    n_rows = len(rows)

    if n_rows == 0:
        return GuardResult.ok()

    # Aggregate / COUNT(*) — single numeric result: check the value, not the row count
    if n_rows == 1 and len(rows[0]) == 1:
        value = next(iter(rows[0].values()))
        try:
            count = int(value)  # type: ignore[arg-type]
            if 0 < count < k:
                return GuardResult.block(
                    4,
                    f"Small-cell suppression: count={count} is below k={k} threshold",
                    f"count={count}",
                )
            # count >= k or count == 0: aggregate result is safe
            return GuardResult.ok()
        except (TypeError, ValueError):
            # Non-integer single-column result: fall through to row-count check
            pass

    # Non-aggregate multi-row (or single-row non-integer) — apply row-count check
    if 0 < n_rows < k:
        return GuardResult.block(
            4,
            f"Small-cell suppression: {n_rows} rows is below k={k} threshold",
            f"rows={n_rows}",
        )

    return GuardResult.ok()


# ---------------------------------------------------------------------------
# Layer 5 — Prompt-injection detection (on NL input, before SQL generation)
# ---------------------------------------------------------------------------

# SQL keywords that are suspicious in a natural-language context
# Note: XP_ and SP_ are prefixes (no trailing \b) since underscore is a word char
_SQL_KEYWORD_PATTERN = re.compile(
    r"\b(DROP|TRUNCATE|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC|EXECUTE|"
    r"UNION\s+ALL|UNION|INFORMATION_SCHEMA|SYS\.)|(?:XP_|SP_)",
    re.IGNORECASE,
)

# SQL comment / escape patterns
_SQL_COMMENT_PATTERN = re.compile(r"(--|;|/\*|\*/|0x[0-9a-fA-F]+)")

# Classic prompt injection phrases
_PROMPT_INJECTION_PATTERN = re.compile(
    r"(ignore\s+(?:previous|all|prior)\s+instructions?|"
    r"you\s+are\s+now|pretend\s+(?:you\s+are|to\s+be)|"
    r"act\s+as\s+(?:a\s+)?(?:root|admin|superuser|dba)|"
    r"disregard\s+(?:the\s+)?(?:above|previous|prior)|"
    r"forget\s+(?:the\s+)?(?:above|previous|prior)|"
    r"new\s+(?:role|persona|instruction|task):|"
    r"system\s*:\s*you|"
    r"jailbreak|"
    r"dan\s+mode)",
    re.IGNORECASE,
)

# Attempts to exfiltrate all data
_EXFILTRATION_PATTERN = re.compile(
    r"(return\s+all\s+(?:rows|data|records|patients)|"
    r"without\s+(?:any\s+)?(?:where|filter|limit)|"
    r"show\s+(?:me\s+)?everything|"
    r"dump\s+(?:the\s+)?(?:database|table|data)|"
    r"list\s+all\s+patients)",
    re.IGNORECASE,
)


def layer5_injection(nl_question: str) -> GuardResult:
    """Detect prompt injection and SQL injection attempts in the NL input."""
    text = nl_question.strip()

    m = _SQL_KEYWORD_PATTERN.search(text)
    if m:
        return GuardResult.block(
            5,
            f"Potential SQL injection: SQL keyword '{m.group()}' in natural language input",
            m.group(),
        )

    m = _SQL_COMMENT_PATTERN.search(text)
    if m:
        return GuardResult.block(
            5,
            f"Potential SQL injection: comment/escape pattern '{m.group()}' in input",
            m.group(),
        )

    m = _PROMPT_INJECTION_PATTERN.search(text)
    if m:
        return GuardResult.block(
            5,
            f"Prompt injection attempt detected: '{m.group()}'",
            m.group(),
        )

    m = _EXFILTRATION_PATTERN.search(text)
    if m:
        return GuardResult.block(
            5,
            f"Data exfiltration attempt detected: '{m.group()}'",
            m.group(),
        )

    return GuardResult.ok()


# ---------------------------------------------------------------------------
# Combined pre-execution validation (Layers 1, 2, 3)
# ---------------------------------------------------------------------------


def validate_sql(sql: str) -> GuardResult:
    """Run Layers 1→2→3 in sequence. Returns the first blocking result or ok()."""
    for check in (layer1_readonly, layer2_allowlist, layer3_phi_block):
        result = check(sql)
        if not result.passed:
            return result
    return GuardResult.ok()


def validate_nl(nl_question: str) -> GuardResult:
    """Run Layer 5 on the natural-language input."""
    return layer5_injection(nl_question)


def validate_result(rows: list[dict[str, Any]], sql: str) -> GuardResult:
    """Run Layer 4 on the execution result."""
    return layer4_small_cell(rows, sql)
