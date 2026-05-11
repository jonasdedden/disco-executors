from __future__ import annotations

import itertools
import os
import random
from collections.abc import Collection, Iterable, Sequence
from typing import Final

import dask.distributed
import pytest
import ray

from disco.executors.base import ErrorRaiseMode, Executor, Future
from disco.executors.dask import DaskExecutor
from disco.executors.local import LocalExecutor, LocalFuture, LocalFutureCancelledError, LocalFutureState
from disco.executors.ray import RayExecutor

RANDOM_INTEGER_SPAN: Final[tuple[int, int]] = (0, 100)
RANDOM_N_NUMS: Final[int] = 100
RANDOM_SEED: Final[int] = 12345

random.seed(RANDOM_SEED)


def foo(bar: int, baz: int, biz: int = 5) -> int:
    return bar + baz + biz


def foo_without_kwargs(bar: int, baz: int) -> int:
    return bar + baz


def foo_exc(inp: int) -> int:
    if inp % 10 == 0:
        raise ValueError("Throwing tantrum!")

    return inp


def foo_rnd_exc(inp: int) -> int:
    if random.random() >= 0.5:
        raise ValueError
    return inp


def foo_colliding_kwargs(inp: int, key: int = 1, workers: int = 1) -> int:
    # This is a function that has kwargs that are shared with dask.distributed.Client.submit kwargs
    return inp + key + workers


def foo_list_of_ints(inp: Iterable[int], num_1: int, num_2: int = 1) -> int:
    return sum(inp) + num_1 + num_2


def generate_num() -> int:
    return random.randint(*RANDOM_INTEGER_SPAN)


def generate_num_tuple() -> tuple[int, int, int]:
    bar, baz, biz = (random.randint(*RANDOM_INTEGER_SPAN) for _ in range(3))
    return bar, baz, biz


def generate_nums(n: int = RANDOM_N_NUMS) -> list[int]:
    return sorted([random.randint(*RANDOM_INTEGER_SPAN) for _ in range(n)])


class TestLocalFuture:
    def test_local_future_no_error(self) -> None:
        bar, baz, biz = generate_num_tuple()
        future = LocalFuture(foo, bar, baz, biz=biz)

        assert future.status is LocalFutureState.PENDING
        assert future.cancelled() is False
        assert future.done() is False
        assert future.result() == sum((bar, baz, biz))

        # trick to reset assumptions mypy has about future instance
        # https://github.com/python/mypy/issues/9005
        future = future

        assert future.status is LocalFutureState.FINISHED
        assert future.cancelled() is False
        assert future.done() is True
        assert future.exception() is None

    def test_local_future_cancel(self) -> None:
        bar, baz, biz = generate_num_tuple()
        future = LocalFuture(foo, bar, baz, biz=biz)

        future.cancel()

        assert future.status is LocalFutureState.CANCELLED
        assert future.cancelled() is True
        assert future.done() is True
        with pytest.raises(LocalFutureCancelledError):
            future.result()
        with pytest.raises(LocalFutureCancelledError):
            future.exception()

    def test_local_future_error_raise(self) -> None:
        future = LocalFuture(foo_exc, 10)

        with pytest.raises(ValueError):
            future.result()

    def test_local_future_error_capture(self) -> None:
        future = LocalFuture(foo_exc, 10)

        assert isinstance(future.exception(), ValueError)
        assert future.status is LocalFutureState.ERROR
        assert future.done() is True

        with pytest.raises(ValueError):
            future.result()


@pytest.fixture(scope="session")
def executors() -> dict[str, Executor]:
    num_cpus = int(os.environ.get("KUBERNETES_CPU_REQUEST", "8"))
    cluster = dask.distributed.LocalCluster(n_workers=num_cpus)  # type: ignore[no-untyped-call]
    client = dask.distributed.Client(cluster)  # type: ignore[no-untyped-call]
    ray.init(num_cpus=num_cpus)
    return {"dask": DaskExecutor(client=client), "local": LocalExecutor(), "ray": RayExecutor()}


@pytest.mark.parametrize("_executor", ("dask", "local", "ray"))
class TestExecutors:
    class TestSingle:
        def test_single_submit(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            bar, baz, biz = generate_num_tuple()
            submit_future: Future[int] = executor.submit(foo, bar, baz, biz=biz)
            assert submit_future.result() == sum((bar, baz, biz))

        def test_single_exec(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            bar, baz, biz = generate_num_tuple()
            exec_result: int = executor.exec(foo, bar, baz, biz=biz)
            assert exec_result == sum((bar, baz, biz))

        def test_single_submit_colliding_kwargs(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            bar, baz, biz = generate_num_tuple()
            submit_future: Future[int] = executor.submit(foo_colliding_kwargs, bar, key=baz, workers=biz)
            assert submit_future.result() == sum((bar, baz, biz))

    class TestMap:
        def test_map(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # expects iterable of results in right order
            nums = generate_nums()
            nums2 = generate_nums()
            biz = generate_num()
            map_result: Iterable[int] = executor.map(foo, nums, nums2, biz=biz)
            assert list(map_result) == [num + num2 + biz for num, num2 in zip(nums, nums2)]

        def test_exec_map(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # expects iterable of results in right order
            nums = generate_nums()
            nums2 = generate_nums()
            biz = generate_num()
            map_result: Sequence[int] = executor.exec_map(foo, nums, nums2, biz=biz)
            assert map_result == [num + num2 + biz for num, num2 in zip(nums, nums2)]

        def test_map_unequal_length_collections(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map with collections of unequal length; shall give error
            nums = generate_nums()
            nums2 = generate_nums()[:50]
            biz = 3
            with pytest.raises(ValueError):
                list(executor.map(foo, nums, nums2, biz=biz))

        def test_map_unequal_length_iterables(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map with iterables of unequal length; shall not give error but have "incomplete" results
            nums_it = (i for i in range(100))
            nums2_it = (i for i in range(50))
            biz = 3
            map_result: Iterable[int] = executor.map(foo, nums_it, nums2_it, biz=biz)
            assert list(map_result) == [2 * i + 3 for i in range(50)]

        def test_map_iterable(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map with iterable; expects iterable of results in right order
            nums_it = (i for i in range(100))
            nums2_it = (i for i in range(100))
            biz = 3
            map_result: Iterable[int] = executor.map(foo, nums_it, nums2_it, biz=biz)
            assert list(map_result) == [2 * i + 3 for i in range(100)]

        def test_map_colliding_kwargs(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map with colliding kwargs with Dask; expects iterable of results in right order
            nums = generate_nums()
            baz = generate_num()
            biz = generate_num()
            map_result: Iterable[int] = executor.map(foo_colliding_kwargs, nums, key=baz, workers=biz)
            assert list(map_result) == [num + baz + biz for num in nums]

    class TestMapArgs:
        def test_map_args(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args; expects iterable of futures in right order
            nums = generate_nums()
            baz = generate_num()
            biz = generate_num()
            map_args_futures: Iterable[Future[int]] = executor.map_args(foo, nums, baz, biz=biz)
            assert [future.result() for future in map_args_futures] == [num + baz + biz for num in nums]

        def test_exec_map_args(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test exec_map_args; expects iterable of futures in right order
            nums = generate_nums()
            baz = generate_num()
            biz = generate_num()
            exec_map_args_result: Iterable[int] = executor.exec_map_args(foo, nums, baz, biz=biz)
            assert exec_map_args_result == [num + baz + biz for num in nums]

        def test_map_args_iterable(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args with iterable; expects iterable of results in right order
            nums_gen, nums_copy_gen = itertools.tee((i for i in range(100)), 2)
            baz = generate_num()
            biz = generate_num()
            assert not isinstance(nums_gen, Collection)
            map_args_futures: Iterable[Future[int]] = executor.map_args(foo, nums_gen, baz, biz=biz)
            assert [future.result() for future in map_args_futures] == [i + baz + biz for i in nums_copy_gen]

        def test_map_args_colliding_kwargs(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args with colliding kwargs with Dask; expects iterable of futures in right order
            nums = generate_nums()
            baz = generate_num()
            biz = generate_num()
            map_args_futures: Iterable[Future[int]] = executor.map_args(
                foo_colliding_kwargs, nums, key=baz, workers=biz
            )
            assert [future.result() for future in map_args_futures] == [num + baz + biz for num in nums]

    class TestGatherAsCompleted:
        def test_map_args_as_completed(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + as_completed; expects iterable of futures not necessarily in right order
            nums = generate_nums()
            baz = generate_num()
            biz = generate_num()
            map_args_futures: Iterable[Future[int]] = executor.map_args(foo, nums, baz, biz=biz)
            assert sorted(future.result() for future in executor.as_completed(map_args_futures)) == (
                [num + baz + biz for num in nums]
            )

        def test_map_args_gather(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + gather; expects list of results in right order
            nums = generate_nums()
            baz = generate_num()
            biz = generate_num()
            map_args_futures: Iterable[Future[int]] = executor.map_args(foo, nums, baz, biz=biz)
            assert executor.gather(map_args_futures) == [num + baz + biz for num in nums]

    class TestSmartMapArgs:
        def test_smart_map_args(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test smart_map_args; expects iterable of results not necessarily in right order
            nums = generate_nums()
            baz = generate_num()
            biz = generate_num()
            exec_map_args_result: Iterable[int] = executor.smart_map_args(foo, nums, baz, biz=biz)
            assert sorted(exec_map_args_result) == [num + baz + biz for num in nums]

    class TestError:
        def test_map_args_gather_skip(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + gather on deterministically raising function; error mode = skip => missing result shall be None
            #  Expects list of results in right order
            nums = generate_nums()
            assert any(num % 10 == 0 for num in nums)
            map_args_error_futures: Iterable[Future[int]] = executor.map_args(foo_exc, nums)
            assert executor.gather(map_args_error_futures, errors=ErrorRaiseMode.SKIP) == [
                num if num % 10 != 0 else None for num in nums
            ]

        def test_map_args_gather_raise(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + gather on deterministically raising function; error mode = raise => shall raise
            nums = generate_nums()
            assert any(num % 10 == 0 for num in nums)
            map_args_error_futures: Iterable[Future[int]] = executor.map_args(foo_exc, nums)
            with pytest.raises(ValueError):
                executor.gather(map_args_error_futures, errors=ErrorRaiseMode.RAISE)

    class TestRetries:
        def test_map_args_gather_retries_0(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + gather with retries on randomly raising function
            #  retries = 0 => shall rise
            nums = generate_nums()
            map_args_rnd_error_futures: Iterable[Future[int]] = executor.map_args(foo_rnd_exc, nums)
            with pytest.raises(ValueError):
                executor.gather(map_args_rnd_error_futures, retries=0)

        def test_map_args_gather_retries_1(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + gather with retries on randomly raising function
            #  retries = 1 => shall rise
            nums = generate_nums()
            map_args_rnd_error_futures: Iterable[Future[int]] = executor.map_args(foo_rnd_exc, nums)
            with pytest.raises(ValueError):
                executor.gather(map_args_rnd_error_futures, retries=1)

        def test_map_args_gather_retries_15(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + gather with retries on randomly raising function
            #  retries = 15 => shall reproduce correct result
            nums = generate_nums()
            map_args_rnd_error_futures: Iterable[Future[int]] = executor.map_args(foo_rnd_exc, nums)
            assert executor.gather(map_args_rnd_error_futures, retries=15) == nums

        def test_map_args_as_completed_retries_0(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + as_completed with retries on randomly raising function
            #  retries = 0 => shall rise
            nums = generate_nums()
            map_args_rnd_error_futures: Iterable[Future[int]] = executor.map_args(foo_rnd_exc, nums)
            with pytest.raises(ValueError):
                [future.result() for future in executor.as_completed(map_args_rnd_error_futures, retries=0)]

        def test_map_args_as_completed_retries_1(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + as_completed with retries on randomly raising function
            #  retries = 1 => shall rise
            nums = generate_nums()
            map_args_rnd_error_futures: Iterable[Future[int]] = executor.map_args(foo_rnd_exc, nums)
            with pytest.raises(ValueError):
                [future.result() for future in executor.as_completed(map_args_rnd_error_futures, retries=1)]

        def test_map_args_as_completed_retries_15(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map_args + as_completed with retries on randomly raising function
            #  retries = 15 => shall reproduce correct result
            nums = generate_nums()
            map_args_rnd_error_futures: Iterable[Future[int]] = executor.map_args(foo_rnd_exc, nums)
            assert (
                sorted(future.result() for future in executor.as_completed(map_args_rnd_error_futures, retries=15))
                == nums
            )

    class TestMapReduce:
        def test_map_reduce(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map reduce approach
            list_of_nums = [generate_nums() for _ in range(10)]
            num = generate_num()
            num_1 = generate_num()
            num_2 = generate_num()
            futures: list[Sequence[Future[int]]] = [
                executor.map_args(foo_without_kwargs, nums, num) for nums in list_of_nums
            ]
            assert executor.gather(executor.reduce(foo_list_of_ints, futures, num_1, num_2=num_2)) == [
                sum(nums) + num * len(nums) + num_1 + num_2 for nums in list_of_nums
            ]

        def test_map_reduce_retries(self, executors: dict[str, Executor], _executor: str) -> None:
            executor = executors[_executor]
            # Test map reduce approach with failing futures
            list_of_nums = [generate_nums() for _ in range(10)]
            num_1 = generate_num()
            num_2 = generate_num()
            random.seed(RANDOM_SEED)  # test if this makes test more reproducible in CI
            futures: list[Sequence[Future[int]]] = [executor.map_args(foo_rnd_exc, nums) for nums in list_of_nums]
            assert executor.gather(executor.reduce_with_retries(foo_list_of_ints, 15, futures, num_1, num_2=num_2)) == [
                sum(nums) + num_1 + num_2 for nums in list_of_nums
            ]
