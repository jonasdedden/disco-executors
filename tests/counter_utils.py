from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def register_counter(root: Path, counter_key: str) -> None:
    (root / counter_key).mkdir()


def raise_counter(root: Path, counter_key: str, *_: Any) -> int:
    """Atomically increment the counter and return its new value, safe across threads and processes.

    Each counter is a directory; an increment claims the next free file name `1`, `2`, ... via `O_CREAT | O_EXCL`, which
    the filesystem guarantees only one caller can win. There is no shared lock, so unrelated counters never contend.
    """
    counter_dir = root / counter_key
    # Every existing entry is a value that has already been handed out, so this is a lower bound for the next one.
    value = sum(1 for _ in counter_dir.iterdir()) + 1
    while True:
        try:
            os.close(os.open(counter_dir / str(value), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            value += 1
        else:
            return value
