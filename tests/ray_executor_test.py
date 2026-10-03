from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import ray.exceptions

from .utils import foo
from disco.executors import ResultConfig
from disco.executors.ray import RayExecutor, RayKwargs

if TYPE_CHECKING:
    from disco.executors.base import Executor


@pytest.mark.parametrize("executor", ["ray"], indirect=True)
class TestRayExecutor:
    class TestRayFuture:
        @pytest.mark.parametrize(
            "error",
            [
                ray.exceptions.WorkerCrashedError(),
                ray.exceptions.NodeDiedError("node died"),  # type: ignore[no-untyped-call]
                ray.exceptions.RaySystemError(RuntimeError("system error")),  # type: ignore[no-untyped-call]
            ],
        )
        def test_exception_returns_system_error(self, executor: Executor, error: ray.exceptions.RayError) -> None:
            bar = baz = biz = 1
            future = executor.submit(foo, result_config=ResultConfig.FUTURE_PENDING)(bar, baz, biz=biz)
            with patch("ray.get", side_effect=error):
                assert future.exception() is error

    class TestRayKwargs:
        def test_submit(self, executor: Executor) -> None:
            bar = baz = biz = 1
            result: int = executor.submit(
                foo,
                executor_kwargs={
                    RayExecutor: RayKwargs(
                        func_remote_kwargs={"name": "foo_different"},
                        func_options_kwargs={"name": "foo_very_different"},
                        get_timeout=2,
                    )
                },
            )(bar, baz, biz=biz)
            expected_result = sum((bar, baz, biz))
            assert isinstance(result, int)
            assert result == expected_result

        def test_map(self, executor: Executor) -> None:
            bar = baz = biz = 1
            [result] = executor.map(
                foo,
                executor_kwargs={
                    RayExecutor: RayKwargs(
                        func_remote_kwargs={"name": "foo_different"},
                        func_options_kwargs={"name": "foo_very_different"},
                        get_timeout=2,
                        wait_poll_interval=1,
                    )
                },
            )([bar], baz, biz=biz)
            expected_result = sum((bar, baz, biz))
            assert isinstance(result, int)
            assert result == expected_result
