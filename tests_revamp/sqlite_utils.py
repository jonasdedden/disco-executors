from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def ensure_counter_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS atomic_counter (
                counter_key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
            """
        )


def register_counter(db_path: Path, counter_key: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO atomic_counter (counter_key, value)
            VALUES (?, 0)
            """,
            (counter_key,),
        )


def raise_counter(db_path: Path, counter_key: str, *_: Any) -> int:
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        row = conn.execute(
            """
            UPDATE atomic_counter
            SET value = value + 1
            WHERE counter_key = ?
            RETURNING value
            """,
            (counter_key,),
        ).fetchone()

    if row is None:
        raise RuntimeError(f"Missing retry_state row for failure_key={counter_key!r}")

    counter_value = int(row[0])
    assert isinstance(counter_value, int)
    assert counter_value >= 0

    return counter_value
