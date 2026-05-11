from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given, settings

from .utils import (
    CUSTOM_ERROR_MSG,
    EXPECTED_EXCEPTION_FAIL_INPUT,
    EXPECTED_EXCEPTION_INPUTS_LISTS,
    INT_LISTS,
    INT_LISTS_SMALL,
    INTS,
    CustomError,
    ExpectedExceptionInput,
    FailNTimesInput,
    foo,
    foo_exc,
    foo_fail_n_times,
)
from disco.executors.revamp.base import ExceptionConfig, Executor, Future, ResultConfig, RetryConfig, mwrap, wrap

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class DifferentCustomError(Exception):
    pass


@pytest.mark.parametrize("executor", ["local", "ray", "thread", "process", "dask"], indirect=True)
class TestExecutors:
    class TestSubmit:
        """Tests executing a single task"""

        @settings(deadline=400)  # This usually is the first test, which can take a time to let Ray warm up
        @given(bar=INTS, baz=INTS, biz=INTS)
        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        def test_submit(self, executor: Executor, result_config: ResultConfig, bar: int, baz: int, biz: int) -> None:
            result: Future[int] | int = executor.submit(wrap(foo, bar, baz, biz=biz), result_config=result_config)
            expected_result = sum((bar, baz, biz))
            match result_config:
                case ResultConfig.RESULT:
                    assert isinstance(result, int)
                    assert result == expected_result
                case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                    assert isinstance(result, Future)
                    assert result.result() == expected_result
                case _:
                    raise ValueError(f"Unexpected result config: {result_config}")

    class TestSubmitException:
        """Tests executing a task that raises an exception"""

        @given(inp=EXPECTED_EXCEPTION_FAIL_INPUT)
        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        def test_submit_exception(
            self, executor: Executor, result_config: ResultConfig, inp: ExpectedExceptionInput
        ) -> None:
            with pytest.raises(CustomError, match=CUSTOM_ERROR_MSG + rf" \({inp.num}\)"):
                executor.submit(wrap(foo_exc, inp), result_config=result_config)

        @given(inp=EXPECTED_EXCEPTION_FAIL_INPUT)
        def test_submit_exception_pending_future(self, executor: Executor, inp: ExpectedExceptionInput) -> None:
            future = executor.submit(wrap(foo_exc, inp), result_config=ResultConfig.FUTURE_PENDING)

            exception = future.exception()
            assert isinstance(exception, CustomError)
            assert exception.args == (CUSTOM_ERROR_MSG + f" ({inp.num})",)

            with pytest.raises(CustomError, match=CUSTOM_ERROR_MSG + rf" \({inp.num}\)"):
                future.result()

    @pytest.mark.parametrize("required_executions", [2, 3, 5, 7])
    class TestSubmitRetries:
        """Tests submitting a task that fails a precisely specified number of times"""

        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        def test_submit_exception_retries(
            self,
            executor: Executor,
            result_config: ResultConfig,
            atomic_counter: Callable[..., int],
            required_executions: int,
        ) -> None:
            inp = FailNTimesInput(counter_callable=atomic_counter, num=42)
            result: Future[int] | int = executor.submit(
                wrap(foo_fail_n_times, inp, required_executions=required_executions),
                result_config=result_config,
                retry_config=required_executions - 1,
            )
            match result_config:
                case ResultConfig.RESULT:
                    assert isinstance(result, int)
                    assert result == inp.num
                case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                    assert isinstance(result, Future)
                    assert result.result() == inp.num
                case _:
                    raise ValueError(f"Unexpected result config: {result_config}")

        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        def test_submit_exception_too_few_retries(
            self,
            executor: Executor,
            result_config: ResultConfig,
            atomic_counter: Callable[..., int],
            required_executions: int,
        ) -> None:
            inp = FailNTimesInput(counter_callable=atomic_counter, num=42)
            with pytest.raises(
                CustomError,
                match=CUSTOM_ERROR_MSG + rf" \({required_executions - 1}/{required_executions}, num={inp.num}\)",
            ):
                executor.submit(
                    wrap(foo_fail_n_times, inp, required_executions=required_executions),
                    result_config=result_config,
                    retry_config=required_executions - 2,
                )

        def test_submit_exception_too_few_retries_pending_future(
            self,
            executor: Executor,
            atomic_counter: Callable[..., int],
            required_executions: int,
        ) -> None:
            inp = FailNTimesInput(counter_callable=atomic_counter, num=42)
            future: Future[int] = executor.submit(
                wrap(foo_fail_n_times, inp, required_executions=required_executions),
                result_config=ResultConfig.FUTURE_PENDING,
                retry_config=required_executions - 2,
            )

            exc = future.exception()
            assert isinstance(exc, CustomError)
            assert exc.args == (
                CUSTOM_ERROR_MSG + f" ({required_executions - 1}/{required_executions}, num={inp.num})",
            )

            with pytest.raises(
                CustomError,
                match=CUSTOM_ERROR_MSG + rf" \({required_executions - 1}/{required_executions}, num={inp.num}\)",
            ):
                future.result()

    @pytest.mark.parametrize("required_executions", [2, 3])
    class TestSubmitRetrySpecificException:
        """Tests submitting a task that fails a precisely specified number of times, but we only retry on a specific exception type"""

        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        def test_submit_exception_retry_correct_exception_type(
            self,
            executor: Executor,
            result_config: ResultConfig,
            atomic_counter: Callable[..., int],
            required_executions: int,
        ) -> None:
            inp = FailNTimesInput(counter_callable=atomic_counter, num=42)
            result: Future[int] | int = executor.submit(
                wrap(foo_fail_n_times, inp, required_executions=required_executions),
                result_config=result_config,
                retry_config=RetryConfig(retries=required_executions - 1, exceptions=[CustomError]),
            )
            match result_config:
                case ResultConfig.RESULT:
                    assert isinstance(result, int)
                    assert result == inp.num
                case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                    assert isinstance(result, Future)
                    assert result.result() == inp.num
                case _:
                    raise ValueError(f"Unexpected result config: {result_config}")

        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        def test_submit_exception_retry_wrong_exception_type(
            self,
            executor: Executor,
            result_config: ResultConfig,
            atomic_counter: Callable[..., int],
            required_executions: int,
        ) -> None:
            inp = FailNTimesInput(counter_callable=atomic_counter, num=42)
            with pytest.raises(CustomError, match=CUSTOM_ERROR_MSG + rf" \(1/{required_executions}, num={inp.num}\)"):
                executor.submit(
                    wrap(foo_fail_n_times, inp, required_executions=required_executions),
                    result_config=result_config,
                    retry_config=RetryConfig(retries=required_executions - 1, exceptions=[DifferentCustomError]),
                )

        def test_submit_exception_retry_wrong_exception_type_pending_future(
            self,
            executor: Executor,
            atomic_counter: Callable[..., int],
            required_executions: int,
        ) -> None:
            inp = FailNTimesInput(counter_callable=atomic_counter, num=42)
            future = executor.submit(
                wrap(foo_fail_n_times, inp, required_executions=required_executions),
                result_config=ResultConfig.FUTURE_PENDING,
                retry_config=RetryConfig(retries=required_executions - 2, exceptions=[DifferentCustomError]),
            )

            exc = future.exception()
            assert isinstance(exc, CustomError)
            assert exc.args == (CUSTOM_ERROR_MSG + f" (1/{required_executions}, num={inp.num})",)

            with pytest.raises(CustomError, match=CUSTOM_ERROR_MSG + rf" \(1/{required_executions}, num={inp.num}\)"):
                future.result()

    @pytest.mark.parametrize("max_pending_ratio", [0.1, 0.25, 0.5, 1.0, 2.0, None])
    class TestMap:
        @settings(deadline=1000)  # 1000ms deadline per test, as this can take a while with Ray
        @given(nums=INT_LISTS, baz=INTS, biz=INTS)
        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        def test_map(
            self,
            executor: Executor,
            result_config: ResultConfig,
            max_pending_ratio: float | None,
            nums: Sequence[int],
            baz: int,
            biz: int,
        ) -> None:
            results: Sequence[Future[int]] | Sequence[int] = executor.map(
                mwrap(foo, nums, baz, biz=biz),
                result_config=result_config,
                max_pending_tasks=None if max_pending_ratio is None else max(1, int(len(nums) * max_pending_ratio)),
            )
            assert isinstance(results, Sequence)
            expected_result = [num + baz + biz for num in nums]
            match result_config:
                case ResultConfig.RESULT:
                    assert all(isinstance(result, int) for result in results)
                    assert results == expected_result
                case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                    assert all(isinstance(result, Future) for result in results)
                    assert [future.result() for future in results] == expected_result  # type: ignore[union-attr]
                case _:
                    raise ValueError(f"Unexpected result config: {result_config}")

        @pytest.mark.parametrize("exception_config", [ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RAISE_GROUPED])
        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        @given(expected_exception_input=EXPECTED_EXCEPTION_INPUTS_LISTS)
        def test_map_expected_excs(
            self,
            executor: Executor,
            result_config: ResultConfig,
            max_pending_ratio: float | None,
            exception_config: ExceptionConfig,
            expected_exception_input: Sequence[ExpectedExceptionInput],
        ) -> None:
            fail_inputs = [inp for inp in expected_exception_input if inp.should_fail]

            test_harness: AbstractContextManager[Any]
            match exception_config:
                case ExceptionConfig.RAISE_EAGERLY:
                    test_harness = pytest.raises(CustomError, match=CUSTOM_ERROR_MSG)
                case ExceptionConfig.RAISE_GROUPED:
                    test_harness = pytest.RaisesGroup(
                        *[
                            pytest.RaisesExc(CustomError, match=CUSTOM_ERROR_MSG + rf" \({inp.num}\)")
                            for inp in fail_inputs
                        ]
                    )
                case _:
                    raise ValueError(f"Invalid exception_config: {exception_config}")

            with test_harness:
                executor.map(
                    mwrap(foo_exc, expected_exception_input),
                    result_config=result_config,
                    exception_config=exception_config,
                    max_pending_tasks=None
                    if max_pending_ratio is None
                    else max(1, int(len(expected_exception_input) * max_pending_ratio)),
                )

        @given(expected_exception_input=EXPECTED_EXCEPTION_INPUTS_LISTS)
        def test_map_expected_excs_pending_futures(
            self,
            executor: Executor,
            max_pending_ratio: float | None,
            expected_exception_input: Sequence[ExpectedExceptionInput],
        ) -> None:
            success_inputs = [inp for inp in expected_exception_input if not inp.should_fail]
            fail_inputs = [inp for inp in expected_exception_input if inp.should_fail]

            futures: Sequence[Future[int]] = executor.map(
                mwrap(foo_exc, expected_exception_input),
                result_config=ResultConfig.FUTURE_PENDING,
                max_pending_tasks=None
                if max_pending_ratio is None
                else max(1, int(len(expected_exception_input) * max_pending_ratio)),
            )

            results: list[int] = []
            exceptions: list[Exception] = []

            for future in futures:
                if exc := future.exception():
                    exceptions.append(exc)
                else:
                    results.append(future.result())

            assert results == [inp.num for inp in success_inputs]

            with pytest.RaisesGroup(
                *[pytest.RaisesExc(CustomError, match=CUSTOM_ERROR_MSG + rf" \({inp.num}\)") for inp in fail_inputs]
            ):
                raise ExceptionGroup("exc", exceptions)

        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        @given(expected_exception_input=EXPECTED_EXCEPTION_INPUTS_LISTS)
        def test_map_expected_excs_return(
            self,
            executor: Executor,
            result_config: ResultConfig,
            max_pending_ratio: float | None,
            expected_exception_input: Sequence[ExpectedExceptionInput],
        ) -> None:
            """With `ExceptionConfig.RETURN`, exceptions are returned inline at their
            input position — a [PASS, FAIL, FAIL, PASS] input must yield
            [result, Exception, Exception, result] in the same order."""
            results: Sequence[int | Exception] | Sequence[Future[int] | Exception] = executor.map(
                mwrap(foo_exc, expected_exception_input),
                result_config=result_config,
                exception_config=ExceptionConfig.RETURN,
                max_pending_tasks=None
                if max_pending_ratio is None
                else max(1, int(len(expected_exception_input) * max_pending_ratio)),
            )
            assert len(results) == len(expected_exception_input)

            for result, inp in zip(results, expected_exception_input, strict=True):
                if inp.should_fail:
                    assert isinstance(result, CustomError)
                    assert result.args == (CUSTOM_ERROR_MSG + f" ({inp.num})",)
                else:
                    match result_config:
                        case ResultConfig.RESULT:
                            assert result == inp.num
                        case ResultConfig.FUTURE_COMPLETED:
                            assert isinstance(result, Future)
                            assert result.result() == inp.num
                        case _:
                            raise ValueError(f"Unexpected result_config: {result_config}")

    @pytest.mark.parametrize("required_executions", [2, 3])
    class TestMapRetries:
        """Tests mapping tasks that fail a precisely specified number of times"""

        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        @given(nums=INT_LISTS_SMALL)
        def test_map_retries(
            self,
            executor: Executor,
            result_config: ResultConfig,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]
            results: Sequence[Future[int]] | Sequence[int] = executor.map(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=result_config,
                retry_config=required_executions - 1,
            )
            match result_config:
                case ResultConfig.RESULT:
                    assert all(isinstance(r, int) for r in results)
                    assert list(results) == list(nums)
                case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                    assert all(isinstance(r, Future) for r in results)
                    assert [f.result() for f in results] == list(nums)  # type: ignore[union-attr]
                case _:
                    raise ValueError(f"Unexpected result config: {result_config}")

        @pytest.mark.parametrize("exception_config", [ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RAISE_GROUPED])
        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        @given(nums=INT_LISTS_SMALL)
        def test_map_too_few_retries(
            self,
            executor: Executor,
            result_config: ResultConfig,
            exception_config: ExceptionConfig,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]

            test_harness: AbstractContextManager[Any]
            match exception_config:
                case ExceptionConfig.RAISE_EAGERLY:
                    test_harness = pytest.raises(
                        CustomError,
                        match=CUSTOM_ERROR_MSG + rf" \({required_executions - 1}/{required_executions}, num=\d+\)",
                    )
                case ExceptionConfig.RAISE_GROUPED:
                    test_harness = pytest.RaisesGroup(
                        *[
                            pytest.RaisesExc(
                                CustomError,
                                match=CUSTOM_ERROR_MSG
                                + rf" \({required_executions - 1}/{required_executions}, num={num}\)",
                            )
                            for num in nums
                        ]
                    )
                case _:
                    raise ValueError(f"Invalid exception_config: {exception_config}")

            with test_harness:
                executor.map(
                    mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                    result_config=result_config,
                    exception_config=exception_config,
                    retry_config=required_executions - 2,
                )

        @given(nums=INT_LISTS_SMALL)
        def test_map_too_few_retries_pending_futures(
            self,
            executor: Executor,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]

            futures: Sequence[Future[int]] = executor.map(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=ResultConfig.FUTURE_PENDING,
                retry_config=required_executions - 2,
            )

            for future, inp in zip(futures, inputs, strict=True):
                exc = future.exception()
                assert isinstance(exc, CustomError)
                assert exc.args == (
                    CUSTOM_ERROR_MSG + f" ({required_executions - 1}/{required_executions}, num={inp.num})",
                )

                with pytest.raises(
                    CustomError,
                    match=CUSTOM_ERROR_MSG + rf" \({required_executions - 1}/{required_executions}, num={inp.num}\)",
                ):
                    future.result()

    @pytest.mark.parametrize("required_executions", [2, 3])
    class TestMapRetrySpecificException:
        """Tests mapping tasks that fail N times, but we only retry on a specific exception type"""

        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        @given(nums=INT_LISTS_SMALL)
        def test_map_retry_correct_exception_type(
            self,
            executor: Executor,
            result_config: ResultConfig,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]
            results: Sequence[Future[int]] | Sequence[int] = executor.map(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=result_config,
                retry_config=RetryConfig(retries=required_executions - 1, exceptions=[CustomError]),
            )
            match result_config:
                case ResultConfig.RESULT:
                    assert all(isinstance(r, int) for r in results)
                    assert list(results) == list(nums)
                case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                    assert all(isinstance(r, Future) for r in results)
                    assert [f.result() for f in results] == list(nums)  # type: ignore[union-attr]
                case _:
                    raise ValueError(f"Unexpected result config: {result_config}")

        @pytest.mark.parametrize("exception_config", [ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RAISE_GROUPED])
        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        @given(nums=INT_LISTS_SMALL)
        def test_map_retry_wrong_exception_type(
            self,
            executor: Executor,
            result_config: ResultConfig,
            exception_config: ExceptionConfig,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]

            test_harness: AbstractContextManager[Any]
            match exception_config:
                case ExceptionConfig.RAISE_EAGERLY:
                    test_harness = pytest.raises(
                        CustomError,
                        match=CUSTOM_ERROR_MSG + rf" \(1/{required_executions}, num=\d+\)",
                    )
                case ExceptionConfig.RAISE_GROUPED:
                    test_harness = pytest.RaisesGroup(
                        *[
                            pytest.RaisesExc(
                                CustomError,
                                match=CUSTOM_ERROR_MSG + rf" \(1/{required_executions}, num={num}\)",
                            )
                            for num in nums
                        ]
                    )
                case _:
                    raise ValueError(f"Invalid exception_config: {exception_config}")

            with test_harness:
                executor.map(
                    mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                    result_config=result_config,
                    exception_config=exception_config,
                    retry_config=RetryConfig(retries=required_executions - 1, exceptions=[DifferentCustomError]),
                )

        @given(nums=INT_LISTS_SMALL)
        def test_map_retry_wrong_exception_type_pending_futures(
            self,
            executor: Executor,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]

            futures: Sequence[Future[int]] = executor.map(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=ResultConfig.FUTURE_PENDING,
                retry_config=RetryConfig(retries=required_executions - 2, exceptions=[DifferentCustomError]),
            )

            for future, inp in zip(futures, inputs, strict=True):
                exc = future.exception()
                assert isinstance(exc, CustomError)
                assert exc.args == (CUSTOM_ERROR_MSG + f" (1/{required_executions}, num={inp.num})",)

                with pytest.raises(
                    CustomError, match=CUSTOM_ERROR_MSG + rf" \(1/{required_executions}, num={inp.num}\)"
                ):
                    future.result()

    @pytest.mark.parametrize("max_pending_ratio", [0.1, 0.5, 1.0, None])
    @pytest.mark.parametrize("ordered", [True, False])
    class TestMapLazy:
        @settings(deadline=400)
        @given(nums=INT_LISTS, baz=INTS, biz=INTS)
        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        def test_map_lazy(
            self,
            executor: Executor,
            result_config: ResultConfig,
            ordered: bool,
            max_pending_ratio: float | None,
            nums: Sequence[int],
            baz: int,
            biz: int,
        ) -> None:
            it = executor.map_lazy(
                mwrap(foo, nums, baz, biz=biz),
                result_config=result_config,
                ordered=ordered,
                max_pending_tasks=None if max_pending_ratio is None else max(1, int(len(nums) * max_pending_ratio)),
            )
            expected = [num + baz + biz for num in nums]
            if ordered:
                # Consume one-by-one and verify each position
                for i, result in enumerate(it):
                    match result_config:
                        case ResultConfig.RESULT:
                            assert result == expected[i]
                        case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                            assert isinstance(result, Future)
                            assert result.result() == expected[i]
            else:
                # Unordered: collect all and compare as a set
                results = list(it)
                match result_config:
                    case ResultConfig.RESULT:
                        assert sorted(results) == sorted(expected)  # type: ignore[type-var]
                    case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                        assert sorted(r.result() for r in results) == sorted(expected)  # type: ignore[attr-defined]

        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        @given(expected_exception_input=EXPECTED_EXCEPTION_INPUTS_LISTS)
        def test_map_lazy_expected_excs(
            self,
            executor: Executor,
            result_config: ResultConfig,
            ordered: bool,
            max_pending_ratio: float | None,
            expected_exception_input: Sequence[ExpectedExceptionInput],
        ) -> None:
            # With batched ray.wait, a failing task may be in the same batch as preceding
            # successful tasks, causing the exception before those successes are yielded.
            # We just verify the exception surfaces during iteration; the order of successful
            # results is already covered by test_map_lazy.
            it = executor.map_lazy(
                mwrap(foo_exc, expected_exception_input),
                result_config=result_config,
                ordered=ordered,
                max_pending_tasks=None
                if max_pending_ratio is None
                else max(1, int(len(expected_exception_input) * max_pending_ratio)),
            )
            with pytest.raises(CustomError, match=CUSTOM_ERROR_MSG):
                for _ in it:
                    pass

        @given(expected_exception_input=EXPECTED_EXCEPTION_INPUTS_LISTS)
        def test_map_lazy_expected_excs_pending_futures(
            self,
            executor: Executor,
            ordered: bool,
            max_pending_ratio: float | None,
            expected_exception_input: Sequence[ExpectedExceptionInput],
        ) -> None:
            it = executor.map_lazy(
                mwrap(foo_exc, expected_exception_input),
                result_config=ResultConfig.FUTURE_PENDING,
                ordered=ordered,
                max_pending_tasks=None
                if max_pending_ratio is None
                else max(1, int(len(expected_exception_input) * max_pending_ratio)),
            )
            if ordered:
                # Futures arrive in input order — check each against its input
                for future, inp in zip(it, expected_exception_input, strict=True):
                    if inp.should_fail:
                        exc = future.exception()
                        assert isinstance(exc, CustomError)
                        assert exc.args == (CUSTOM_ERROR_MSG + f" ({inp.num})",)
                    else:
                        assert future.result() == inp.num
            else:
                # Unordered: verify counts and values match
                success_results: list[int] = []
                fail_count = 0
                for future in it:
                    if future.exception():
                        fail_count += 1
                    else:
                        success_results.append(future.result())
                assert sorted(success_results) == sorted(
                    inp.num for inp in expected_exception_input if not inp.should_fail
                )
                assert fail_count == sum(1 for inp in expected_exception_input if inp.should_fail)

        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        @given(expected_exception_input=EXPECTED_EXCEPTION_INPUTS_LISTS)
        def test_map_lazy_expected_excs_return(
            self,
            executor: Executor,
            result_config: ResultConfig,
            ordered: bool,
            max_pending_ratio: float | None,
            expected_exception_input: Sequence[ExpectedExceptionInput],
        ) -> None:
            """With `ExceptionConfig.RETURN`, exceptions are yielded inline; when
            `ordered=True`, they appear at their exact input position (e.g. a
            [PASS, FAIL, FAIL, PASS] input yields [result, Exception, Exception, result])."""
            it = executor.map_lazy(
                mwrap(foo_exc, expected_exception_input),
                result_config=result_config,
                exception_config=ExceptionConfig.RETURN,
                ordered=ordered,
                max_pending_tasks=None
                if max_pending_ratio is None
                else max(1, int(len(expected_exception_input) * max_pending_ratio)),
            )
            if ordered:
                for result, inp in zip(it, expected_exception_input, strict=True):
                    if inp.should_fail:
                        assert isinstance(result, CustomError)
                        assert result.args == (CUSTOM_ERROR_MSG + f" ({inp.num})",)
                    else:
                        match result_config:
                            case ResultConfig.RESULT:
                                assert result == inp.num
                            case ResultConfig.FUTURE_COMPLETED:
                                assert isinstance(result, Future)
                                assert result.result() == inp.num
                            case _:
                                raise ValueError(f"Unexpected result_config: {result_config}")
            else:
                # Unordered: verify the multiset of successes and the multiset of exception
                # messages both match what was requested. Duplicates in the input are tolerated
                # because we compare sorted sequences (not sets).
                success_values: list[int] = []
                failure_msgs: list[str] = []
                for item in it:
                    if isinstance(item, CustomError):
                        failure_msgs.append(item.args[0])
                    elif isinstance(item, Future):
                        success_values.append(item.result())
                    else:
                        assert isinstance(item, int)
                        success_values.append(item)
                assert sorted(success_values) == sorted(
                    inp.num for inp in expected_exception_input if not inp.should_fail
                )
                assert sorted(failure_msgs) == sorted(
                    CUSTOM_ERROR_MSG + f" ({inp.num})" for inp in expected_exception_input if inp.should_fail
                )

    @pytest.mark.parametrize("required_executions", [2, 3])
    @pytest.mark.parametrize("ordered", [True, False])
    class TestMapLazyRetries:
        """Tests lazily mapping tasks that fail a precisely specified number of times"""

        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        @given(nums=INT_LISTS_SMALL)
        def test_map_lazy_retries(
            self,
            executor: Executor,
            result_config: ResultConfig,
            ordered: bool,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]
            it = executor.map_lazy(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=result_config,
                ordered=ordered,
                retry_config=required_executions - 1,
            )
            if ordered:
                for result, expected_num in zip(it, nums, strict=True):
                    match result_config:
                        case ResultConfig.RESULT:
                            assert result == expected_num
                        case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                            assert isinstance(result, Future)
                            assert result.result() == expected_num
            else:
                results = list(it)
                match result_config:
                    case ResultConfig.RESULT:
                        assert sorted(results) == sorted(nums)  # type: ignore[type-var]
                    case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                        assert sorted(r.result() for r in results) == sorted(nums)  # type: ignore[attr-defined]

        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        @given(nums=INT_LISTS_SMALL)
        def test_map_lazy_too_few_retries(
            self,
            executor: Executor,
            result_config: ResultConfig,
            ordered: bool,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]
            it = executor.map_lazy(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=result_config,
                ordered=ordered,
                retry_config=required_executions - 2,
            )
            # All tasks fail — the very first yielded result raises
            with pytest.raises(
                CustomError,
                match=CUSTOM_ERROR_MSG + rf" \({required_executions - 1}/{required_executions}, num=\d+\)",
            ):
                next(it)

        @given(nums=INT_LISTS_SMALL)
        def test_map_lazy_too_few_retries_pending_futures(
            self,
            executor: Executor,
            ordered: bool,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]
            it = executor.map_lazy(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=ResultConfig.FUTURE_PENDING,
                ordered=ordered,
                retry_config=required_executions - 2,
            )
            if ordered:
                for future, inp in zip(it, inputs, strict=True):
                    exc = future.exception()
                    assert isinstance(exc, CustomError)
                    assert exc.args == (
                        CUSTOM_ERROR_MSG + f" ({required_executions - 1}/{required_executions}, num={inp.num})",
                    )
            else:
                for future in it:
                    assert isinstance(future.exception(), CustomError)

    @pytest.mark.parametrize("required_executions", [2, 3])
    @pytest.mark.parametrize("ordered", [True, False])
    class TestMapLazyRetrySpecificException:
        """Tests lazily mapping tasks that fail N times, but we only retry on a specific exception type"""

        @pytest.mark.parametrize(
            "result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED]
        )
        @given(nums=INT_LISTS_SMALL)
        def test_map_lazy_retry_correct_exception_type(
            self,
            executor: Executor,
            result_config: ResultConfig,
            ordered: bool,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]
            it = executor.map_lazy(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=result_config,
                ordered=ordered,
                retry_config=RetryConfig(retries=required_executions - 1, exceptions=[CustomError]),
            )
            if ordered:
                for result, expected_num in zip(it, nums, strict=True):
                    match result_config:
                        case ResultConfig.RESULT:
                            assert result == expected_num
                        case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                            assert isinstance(result, Future)
                            assert result.result() == expected_num
            else:
                results = list(it)
                match result_config:
                    case ResultConfig.RESULT:
                        assert sorted(results) == sorted(nums)  # type: ignore[type-var]
                    case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                        assert sorted(r.result() for r in results) == sorted(nums)  # type: ignore[attr-defined]

        @pytest.mark.parametrize("result_config", [ResultConfig.RESULT, ResultConfig.FUTURE_COMPLETED])
        @given(nums=INT_LISTS_SMALL)
        def test_map_lazy_retry_wrong_exception_type(
            self,
            executor: Executor,
            result_config: ResultConfig,
            ordered: bool,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]
            it = executor.map_lazy(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=result_config,
                ordered=ordered,
                retry_config=RetryConfig(retries=required_executions - 1, exceptions=[DifferentCustomError]),
            )
            # All tasks fail on first attempt — first yielded result raises
            with pytest.raises(
                CustomError,
                match=CUSTOM_ERROR_MSG + rf" \(1/{required_executions}, num=\d+\)",
            ):
                next(it)

        @given(nums=INT_LISTS_SMALL)
        def test_map_lazy_retry_wrong_exception_type_pending_futures(
            self,
            executor: Executor,
            ordered: bool,
            atomic_counter_factory: Callable[[], Callable[..., int]],
            required_executions: int,
            nums: Sequence[int],
        ) -> None:
            inputs = [FailNTimesInput(counter_callable=atomic_counter_factory(), num=num) for num in nums]
            it = executor.map_lazy(
                mwrap(foo_fail_n_times, inputs, required_executions=required_executions),
                result_config=ResultConfig.FUTURE_PENDING,
                ordered=ordered,
                retry_config=RetryConfig(retries=required_executions - 2, exceptions=[DifferentCustomError]),
            )
            if ordered:
                for future, inp in zip(it, inputs, strict=True):
                    exc = future.exception()
                    assert isinstance(exc, CustomError)
                    assert exc.args == (CUSTOM_ERROR_MSG + f" (1/{required_executions}, num={inp.num})",)
            else:
                for future in it:
                    assert isinstance(future.exception(), CustomError)
