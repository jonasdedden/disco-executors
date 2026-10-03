from __future__ import annotations

import os

# Ray's `uv run` integration tries to rebuild this package inside every worker's runtime env, which fails (e.g. because
# `uv-dynamic-versioning` can't find git there). Ray reads this flag at import time, so it must be set before any
# `import ray` — this package `__init__` runs before `conftest.py`.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
