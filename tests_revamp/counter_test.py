from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

import pytest

from .utils import CUSTOM_ERROR_MSG, CustomError, FailNTimesInput, foo_fail_n_times

if TYPE_CHECKING:
    from collections.abc import Callable


def test_counter_basic(atomic_counter: Callable[..., int]) -> None:
    assert atomic_counter() == 1
    assert atomic_counter() == 2


@pytest.mark.parametrize("concurrency_pool", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_counter_concurrent(
    atomic_counter: Callable[..., int], concurrency_pool: type[ThreadPoolExecutor | ProcessPoolExecutor]
) -> None:
    tasks = 20

    with concurrency_pool() as pool:
        futures = [pool.submit(atomic_counter) for _ in range(tasks)]

    assert sorted(f.result() for f in futures) == list(range(1, tasks + 1))


@pytest.mark.parametrize("concurrency_pool", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_foo_exp_failures(
    atomic_counter: Callable[..., int], concurrency_pool: type[ThreadPoolExecutor | ProcessPoolExecutor]
) -> None:
    required_executions = 5
    inp = FailNTimesInput(counter_callable=atomic_counter, num=42)
    foo: Callable[[], int] = partial(foo_fail_n_times, inp, required_executions=required_executions)

    tasks = 10

    with concurrency_pool() as pool:
        futures = [pool.submit(foo) for _ in range(tasks)]

    results: list[int] = []
    exceptions: list[Exception] = []

    for future in futures:
        exc = future.exception()
        if isinstance(exc, Exception):
            exceptions.append(exc)
        else:
            results.append(future.result())

    assert results == [inp.num]

    with pytest.RaisesGroup(
        *[
            pytest.RaisesExc(CustomError, match=CUSTOM_ERROR_MSG + rf" \({i}/{required_executions}, num={inp.num}\)")
            for i in list(range(1, required_executions)) + list(range(required_executions + 1, tasks + 1))
        ]
    ):
        raise ExceptionGroup("exc", exceptions)
