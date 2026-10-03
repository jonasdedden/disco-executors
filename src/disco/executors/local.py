from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, assert_never

from .base import (
    ExceptionConfig,
    Executor,
    Future,
    MultipleWrap,
    ResultConfig,
    RetryConfig,
    SingleWrap,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence


class LocalFutureState(enum.StrEnum):
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class LocalFutureCancelledError(RuntimeError):
    pass


class _PendingSentinel:
    pass


_PENDING = _PendingSentinel()


class _CancelledSentinel:
    pass


_CANCELLED = _CancelledSentinel()


class _DoneWrapper[T](NamedTuple):
    result: T


class _ErrorWrapper(NamedTuple):
    exception: Exception


_FUTURE_ALREADY_CANCELLED_MSG = "Future already cancelled!"


class LocalFuture[R](Future[R]):
    def __init__[**P](self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result: _DoneWrapper[R] | _ErrorWrapper | _PendingSentinel | _CancelledSentinel = _PENDING

    @property
    def status(self) -> LocalFutureState:
        match self._result:
            case _DoneWrapper(result=_):
                return LocalFutureState.FINISHED
            case _ErrorWrapper(exception=_):
                return LocalFutureState.ERROR
            case _CancelledSentinel():
                return LocalFutureState.CANCELLED
            case _PendingSentinel():
                return LocalFutureState.PENDING
            case _:
                assert_never(self._result)

    def cancel(self) -> bool:
        if self.status in (LocalFutureState.PENDING, LocalFutureState.CANCELLED):
            self._result = _CANCELLED
            return True
        return False

    def reset(self) -> None:
        self._result = _PENDING

    def _execute(self) -> _DoneWrapper[R] | _ErrorWrapper:
        try:
            res = self.func(*self.args, **self.kwargs)
        except Exception as exc:
            self._result = _ErrorWrapper(exception=exc)
        else:
            self._result = _DoneWrapper(result=res)
        return self._result

    def result(self, timeout: float | None = None) -> R:
        if self.status == LocalFutureState.CANCELLED:
            raise LocalFutureCancelledError(_FUTURE_ALREADY_CANCELLED_MSG)

        # `mypy` issues here are because of https://github.com/python/mypy/issues/21050
        match self._result:
            case _DoneWrapper(result=res):
                return res  # type: ignore[no-any-return]
            case _ErrorWrapper(exception=exc):
                raise exc

        match self._execute():
            case _DoneWrapper(result=res):
                return res  # type: ignore[no-any-return]
            case _ErrorWrapper(exception=exc):
                raise exc
            case _:
                raise RuntimeError("unreachable")

    def exception(self, timeout: float | None = None) -> Exception | None:
        if self.status == LocalFutureState.CANCELLED:
            raise LocalFutureCancelledError(_FUTURE_ALREADY_CANCELLED_MSG)

        match self._result:
            case _DoneWrapper(result=_):
                return None
            case _ErrorWrapper(exception=exc):
                return exc

        match self._execute():
            case _DoneWrapper(result=_):
                return None
            case _ErrorWrapper(exception=exc):
                return exc
            case _:
                raise RuntimeError("unreachable")

    def __repr__(self) -> str:
        # Stolen from concurrent.futures._base.Future
        match self._result:
            case _DoneWrapper(result=res):
                return (
                    f"<{self.__class__.__name__} at {id(self):#x} state={self.status} raised {res.__class__.__name__}>"
                )
            case _ErrorWrapper(exception=exc):
                return f"<{self.__class__.__name__} at {id(self):#x} state={self.status} returned {exc.__class__.__name__}>"
            case _:
                return f"<{self.__class__.__name__} at {id(self):#x} state={self.status}>"


def _retry_local_future[R](
    fut: LocalFuture[R],
    result_config: ResultConfig,
    retry_config: int | RetryConfig,
) -> _DoneWrapper[LocalFuture[R] | R] | _ErrorWrapper:
    num_retries: int
    if isinstance(retry_config, RetryConfig):
        num_retries = retry_config.retries
    elif isinstance(retry_config, int):
        num_retries = retry_config
    else:
        raise TypeError(f"`retry_config` must be an int or RetryConfig: {retry_config:?}")

    last_error: _ErrorWrapper | None = None

    if result_config is ResultConfig.FUTURE_PENDING:
        # The final execution in FUTURE_PENDING mode shall happen on user-side
        num_retries -= 1

    for _ in range(num_retries + 1):
        value: _DoneWrapper[R] | _ErrorWrapper = fut._execute()
        match value:
            case _DoneWrapper(result=_):
                match result_config:
                    case ResultConfig.RESULT:
                        return value
                    case ResultConfig.FUTURE_PENDING | ResultConfig.FUTURE_COMPLETED:
                        return _DoneWrapper(fut)
                    case _:
                        assert_never(result_config)
            case _ErrorWrapper(exception=exc):
                last_error = value
                if isinstance(retry_config, RetryConfig) and not any(
                    isinstance(exc, allowed_exc) for allowed_exc in retry_config.exceptions
                ):
                    return value
                fut.reset()
            case _:
                assert_never(value)
    assert last_error is not None
    return last_error


class LocalExecutor(Executor):
    def __init__(self) -> None:
        pass

    def _submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: ResultConfig,
        retry_config: int | RetryConfig | None,
        executor_kwargs: Mapping[type[Executor], Any] | None,
    ) -> Future[R] | R:
        if retry_config:
            fut = LocalFuture(w.func, *w.args, **w.kwargs)
            res = _retry_local_future(fut, result_config, retry_config)
            match res:
                case _DoneWrapper(result=res):
                    return res  # type: ignore[no-any-return]
                case _ErrorWrapper(exception=exc):
                    if result_config is ResultConfig.FUTURE_PENDING:
                        # The future will have been tried `retries - 1` implicitly, return the future such that an
                        # eventual exception on the last retry will happen at user-side.
                        return fut
                    raise exc
        else:
            match result_config:
                case ResultConfig.RESULT:
                    return w.func(*w.args, **w.kwargs)
                case ResultConfig.FUTURE_PENDING:
                    return LocalFuture(w.func, *w.args, **w.kwargs)
                case ResultConfig.FUTURE_COMPLETED:
                    fut = LocalFuture(w.func, *w.args, **w.kwargs)
                    fut.result()
                    return fut

    def _map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig,
        exception_config: ExceptionConfig,
        retry_config: int | RetryConfig | None,
        max_pending_tasks: int | None,
        executor_kwargs: Mapping[type[Executor], Any] | None,
    ) -> Sequence[Future[R]] | Sequence[Future[R] | Exception] | Sequence[R] | Sequence[R | Exception]:
        match exception_config:
            case ExceptionConfig.RAISE_EAGERLY:
                return [  # type: ignore[return-value] # all items are the same variant, but mypy can't narrow
                    self._submit(
                        SingleWrap(w.func, arg, *w.args, **w.kwargs),
                        result_config=result_config,
                        retry_config=retry_config,
                        executor_kwargs=executor_kwargs,
                    )
                    for arg in w.first_args
                ]
            case ExceptionConfig.RAISE_GROUPED:
                if result_config == ResultConfig.FUTURE_PENDING:
                    raise ValueError(
                        "Cannot use `ExceptionConfig.RAISE_GROUPED` with `ResultConfig.FUTURE_PENDING`",
                    )

                exceptions: list[Exception] = []
                futures = [LocalFuture(w.func, arg, *w.args, **w.kwargs) for arg in w.first_args]

                for future in futures:
                    if retry_config:
                        result = _retry_local_future(future, result_config, retry_config)
                        if isinstance(result, _ErrorWrapper):
                            exceptions.append(result.exception)
                    else:
                        exc = future.exception()
                        if exc is not None:
                            exceptions.append(exc)

                if exceptions:
                    raise ExceptionGroup("LocalExecutor.map", exceptions)

                match result_config:
                    case ResultConfig.RESULT:
                        return [future.result() for future in futures]
                    case ResultConfig.FUTURE_COMPLETED:
                        return futures
                    case ResultConfig.FUTURE_PENDING:
                        raise AssertionError("FUTURE_PENDING is rejected above for RAISE_GROUPED")
                    case _:
                        assert_never(result_config)
            case ExceptionConfig.RETURN:
                # Exceptions from each submit are captured inline. For FUTURE_PENDING the
                # submit doesn't execute the task yet, so the list is guaranteed to hold only
                # LocalFuture instances in that mode.
                results: list[R | Future[R] | Exception] = []
                for arg in w.first_args:
                    try:
                        results.append(
                            self._submit(
                                SingleWrap(w.func, arg, *w.args, **w.kwargs),
                                result_config=result_config,
                                retry_config=retry_config,
                                executor_kwargs=executor_kwargs,
                            ),
                        )
                    except Exception as exc:
                        results.append(exc)
                return results  # type: ignore[return-value] # narrowed by the overload on exception_config
            case _:
                assert_never(exception_config)

    def _map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RETURN],
        retry_config: int | RetryConfig | None,
        ordered: bool,
        max_pending_tasks: int | None,
        executor_kwargs: Mapping[type[Executor], Any] | None,
    ) -> Iterator[Future[R]] | Iterator[Future[R] | Exception] | Iterator[R] | Iterator[R | Exception]:
        # LocalExecutor is sequential, so ordered/max_pending_tasks have no effect.
        for arg in w.first_args:
            match exception_config:
                case ExceptionConfig.RAISE_EAGERLY:
                    yield self._submit(
                        SingleWrap(w.func, arg, *w.args, **w.kwargs),
                        result_config=result_config,
                        retry_config=retry_config,
                        executor_kwargs=executor_kwargs,
                    )
                case ExceptionConfig.RETURN:
                    try:
                        yield self._submit(
                            SingleWrap(w.func, arg, *w.args, **w.kwargs),
                            result_config=result_config,
                            retry_config=retry_config,
                            executor_kwargs=executor_kwargs,
                        )
                    except Exception as exc:
                        yield exc
                case _:
                    assert_never(exception_config)
