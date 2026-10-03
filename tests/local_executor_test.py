from __future__ import annotations

import pytest

from .utils import foo
from disco.executors.local import LocalExecutor


def test_invalid_retry_config_raises_type_error() -> None:
    # `retry_config` comes from the public API, so untyped callers can pass anything.
    with pytest.raises(TypeError, match="must be an int or RetryConfig: '3'"):
        LocalExecutor().submit(foo, retry_config="3")(1, 2)  # type: ignore[call-overload]
