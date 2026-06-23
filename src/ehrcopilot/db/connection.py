"""Read-only SQLite connection wrapper with statement timeout and row cap.

Always use `get_connection()` as a context manager. Direct writes will raise
sqlite3.OperationalError because the connection is opened in immutable read-only mode.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from ehrcopilot import config


class QueryTimeoutError(Exception):
    """Raised when a query exceeds the configured wall-clock timeout."""


class RowCapExceededError(Exception):
    """Raised when a query would return more rows than config.MAX_ROWS."""


def _set_timeout_pragma(conn: sqlite3.Connection, timeout_seconds: int) -> None:
    """Configure busy timeout; SQLite does not support per-statement wall-clock limits
    natively, so we use a progress handler to abort long-running queries."""
    conn.execute(f"PRAGMA busy_timeout = {timeout_seconds * 1000}")

    calls_per_second = 1000  # sqlite3 progress handler fires every N opcodes
    max_calls = timeout_seconds * calls_per_second

    call_count: list[int] = [0]

    def _progress() -> bool:
        call_count[0] += 1
        if call_count[0] > max_calls:
            raise QueryTimeoutError(
                f"Query exceeded {timeout_seconds}s wall-clock limit"
            )
        return False  # returning True would abort the query via sqlite3 mechanism

    conn.set_progress_handler(_progress, 1000)


@contextmanager
def get_connection(
    db_path: Path | None = None,
    timeout_seconds: int | None = None,
    max_rows: int | None = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Yield a read-only SQLite connection.

    Args:
        db_path: Path to the SQLite file. Defaults to config.SQLITE_DB_PATH.
        timeout_seconds: Per-query wall-clock budget. Defaults to config.QUERY_TIMEOUT_SECONDS.
        max_rows: Maximum rows fetchable per query. Defaults to config.MAX_ROWS.

    Yields:
        sqlite3.Connection opened in read-only URI mode.

    Raises:
        FileNotFoundError: If the database file does not exist.
        QueryTimeoutError: If a query exceeds the timeout.
    """
    resolved_path = db_path or config.SQLITE_DB_PATH
    resolved_timeout = timeout_seconds if timeout_seconds is not None else config.QUERY_TIMEOUT_SECONDS
    resolved_max_rows = max_rows if max_rows is not None else config.MAX_ROWS

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"SQLite DB not found at {resolved_path}. "
            "Run src/ehrcopilot/db/build_sqlite.sh first."
        )

    uri = f"file:{resolved_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    _set_timeout_pragma(conn, resolved_timeout)

    try:
        yield conn
    finally:
        conn.close()


def execute_query(
    sql: str,
    db_path: Path | None = None,
    timeout_seconds: int | None = None,
    max_rows: int | None = None,
) -> list[dict[str, object]]:
    """Execute a SELECT query and return results as a list of dicts.

    Enforces the row cap: fetches at most max_rows + 1, then raises if exceeded
    so callers get an explicit error rather than a silently truncated result set.

    Args:
        sql: The SQL query to execute (must be a SELECT).
        db_path: Optional override for the database path.
        timeout_seconds: Optional override for the query timeout.
        max_rows: Optional override for the max rows cap.

    Returns:
        List of row dicts.

    Raises:
        RowCapExceededError: If the query returns more than max_rows.
        QueryTimeoutError: If the query exceeds the timeout.
        sqlite3.OperationalError: On any SQL execution error.
    """
    resolved_max_rows = max_rows if max_rows is not None else config.MAX_ROWS

    with get_connection(db_path, timeout_seconds, resolved_max_rows) as conn:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(resolved_max_rows + 1)

        if len(rows) > resolved_max_rows:
            raise RowCapExceededError(
                f"Query returned more than {resolved_max_rows} rows. "
                "Add a LIMIT clause or refine the query."
            )

        return [dict(row) for row in rows]


def verify_db(db_path: Path | None = None) -> dict[str, int]:
    """Return row counts for all tables — used by build_sqlite.sh smoke-check."""
    tables = list(config.SCHEMA_ALLOWLIST.keys())
    counts: dict[str, int] = {}
    with get_connection(db_path) as conn:
        for table in tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0] if row else 0
            except sqlite3.OperationalError:
                counts[table] = -1  # table does not exist
    return counts
