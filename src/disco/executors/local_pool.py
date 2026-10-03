from __future__ import annotations

import concurrent.futures
from collections import deque
from typing import TYPE_CHECKING, Any, Literal, assert_never, overload

from .base import (
    DEFAULT_EXCEPTION_CONFIG,
    DEFAULT_MAX_PENDING_TASKS,
    DEFAULT_RESULT_CONFIG,
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


class LocalPoolFuture[R](Future[R]):
    """Thin wrapper around `concurrent.futures.Future` that matches the `Future` protocol.

    The only substantive adjustment over the wrapped future is `exception()`: the stdlib
    returns `BaseException | None`, whereas our protocol promises `Exception | None`.
    BaseExceptions (`KeyboardInterrupt`, `SystemExit`, `CancelledError`, ...) are re-raised
    rather than silently handed back as values.
    """

    def __init__(self, fut: concurrent.futures.Future[R]) -> None:
        self._fut = fut

    def cancel(self) -> bool:
        return self._fut.cancel()

    def result(self, timeout: float | None = None) -> R:
        return self._fut.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> Exception | None:
        exc = self._fut.exception(timeout=timeout)
        if exc is None or isinstance(exc, Exception):
            return exc
        raise exc


class _RetryCallable[**P, R]:
    """Pickleable callable that runs `func` with up-to-N retries on matching exceptions.

    A module-level class (rather than a closure) is used so that instances are picklable —
    `ProcessPoolExecutor` requires anything it receives via `submit` to pass through `pickle`.
    """

    __slots__ = ("_allowed_excs", "_func", "_num_retries")

    def __init__(
        self,
        func: Callable[P, R],
        num_retries: int,
        allowed_excs: tuple[type[BaseException], ...],
    ) -> None:
        self._func = func
        self._num_retries = num_retries
        self._allowed_excs = allowed_excs

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        last_exc: BaseException | None = None
        for _ in range(self._num_retries + 1):
            try:
                return self._func(*args, **kwargs)
            except Exception as exc:
                if not isinstance(exc, self._allowed_excs):
                    raise
                last_exc = exc
        assert last_exc is not None
        raise last_exc


def _prepare_func[**P, R](
    func: Callable[P, R],
    retry_config: int | RetryConfig | None,
) -> Callable[P, R]:
    """Wrap `func` in a retry loop when `retry_config` asks for it; otherwise return as-is."""
    if not retry_config:
        return func
    if isinstance(retry_config, RetryConfig):
        # An empty `exceptions` tuple means "don't retry anything" — preserved faithfully.
        allowed_excs: tuple[type[BaseException], ...] = tuple(retry_config.exceptions)
        return _RetryCallable(func, retry_config.retries, allowed_excs)
    return _RetryCallable(func, retry_config, (Exception,))


class LocalPoolExecutor(Executor):
    """Revamp executor backed by a `concurrent.futures.Executor` (thread or process pool).

    The same implementation serves both `ThreadPoolExecutor` and `ProcessPoolExecutor` because
    they share the `concurrent.futures.Executor` interface. The caller owns the pool's
    lifecycle — we don't shut it down.
    """

    def __init__(self, pool: concurrent.futures.Executor) -> None:
        self._pool = pool

    @overload
    def submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: Literal[ResultConfig.RESULT] = ...,
        retry_config: int | RetryConfig | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> R: ...

    @overload
    def submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: Literal[ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED] = ...,
        retry_config: int | RetryConfig | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> LocalPoolFuture[R]: ...

    @overload
    def submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: ResultConfig = ...,
        retry_config: int | RetryConfig | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> LocalPoolFuture[R] | R: ...

    def submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: ResultConfig = DEFAULT_RESULT_CONFIG,
        retry_config: int | RetryConfig | None = None,
        executor_kwargs: Mapping[type[Executor], Any] | None = None,
    ) -> LocalPoolFuture[R] | R:
        del executor_kwargs  # unused; the pool's configuration lives on the pool instance itself
        func = _prepare_func(w.func, retry_config)
        cf_future = self._pool.submit(func, *w.args, **w.kwargs)
        match result_config:
            case ResultConfig.RESULT:
                return cf_future.result()
            case ResultConfig.FUTURE_PENDING:
                return LocalPoolFuture(cf_future)
            case ResultConfig.FUTURE_COMPLETED:
                fut = LocalPoolFuture(cf_future)
                if exc := fut.exception():
                    raise exc
                return fut

    @overload
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.RESULT] = ...,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RAISE_GROUPED] = ...,
        retry_config: int | RetryConfig | None = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Sequence[R]: ...

    @overload
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.RESULT] = ...,
        exception_config: Literal[ExceptionConfig.RETURN] = ...,
        retry_config: int | RetryConfig | None = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Sequence[R | Exception]: ...

    @overload
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED] = ...,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RAISE_GROUPED] = ...,
        retry_config: int | RetryConfig | None = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Sequence[LocalPoolFuture[R]]: ...

    @overload
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED] = ...,
        exception_config: Literal[ExceptionConfig.RETURN] = ...,
        retry_config: int | RetryConfig | None = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Sequence[LocalPoolFuture[R] | Exception]: ...

    @overload
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = ...,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RAISE_GROUPED] = ...,
        retry_config: int | RetryConfig | None = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Sequence[LocalPoolFuture[R]] | Sequence[R]: ...

    @overload
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = ...,
        exception_config: Literal[ExceptionConfig.RETURN] = ...,
        retry_config: int | RetryConfig | None = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Sequence[R | Exception] | Sequence[LocalPoolFuture[R] | Exception]: ...

    @overload
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = ...,
        exception_config: ExceptionConfig = ...,
        retry_config: int | RetryConfig | None = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> (
        Sequence[LocalPoolFuture[R]] | Sequence[LocalPoolFuture[R] | Exception] | Sequence[R] | Sequence[R | Exception]
    ): ...

    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = DEFAULT_RESULT_CONFIG,
        exception_config: ExceptionConfig = DEFAULT_EXCEPTION_CONFIG,
        retry_config: int | RetryConfig | None = None,
        max_pending_tasks: int | None = DEFAULT_MAX_PENDING_TASKS,
        executor_kwargs: Mapping[type[Executor], Any] | None = None,
    ) -> (
        Sequence[LocalPoolFuture[R]] | Sequence[LocalPoolFuture[R] | Exception] | Sequence[R] | Sequence[R | Exception]
    ):
        del executor_kwargs
        func, first_args, args, kwargs = w.func, w.first_args, w.args, w.kwargs
        del w
        # Force to iterator so any large input Sequence can be GC'd gradually.
        first_args = iter(first_args)
        func = _prepare_func(func, retry_config)

        # Submit with backpressure: at most `max_pending_tasks` tasks in flight at any time.
        all_futures: list[concurrent.futures.Future[R]] = []
        pending: set[concurrent.futures.Future[R]] = set()
        for arg in first_args:
            if isinstance(max_pending_tasks, int) and len(pending) >= max_pending_tasks:
                _, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            fut = self._pool.submit(func, arg, *args, **kwargs)
            all_futures.append(fut)
            pending.add(fut)
        del func, first_args, args, kwargs, pending

        if result_config == ResultConfig.FUTURE_PENDING:
            # Under FUTURE_PENDING the caller drives completion — exceptions surface via the
            # individual futures, so `exception_config` is deliberately ignored here.
            return [LocalPoolFuture(f) for f in all_futures]

        # RESULT / FUTURE_COMPLETED: block on every task before shaping the output.
        concurrent.futures.wait(all_futures, return_when=concurrent.futures.ALL_COMPLETED)

        exceptions_by_idx: dict[int, Exception] = {}
        for idx, fut in enumerate(all_futures):
            exc = fut.exception()
            if exc is None:
                continue
            if not isinstance(exc, Exception):
                # BaseException (KeyboardInterrupt etc.) — always propagate eagerly.
                raise exc
            match exception_config:
                case ExceptionConfig.RAISE_EAGERLY:
                    raise exc
                case ExceptionConfig.RAISE_GROUPED | ExceptionConfig.RETURN:
                    exceptions_by_idx[idx] = exc
                case _:
                    assert_never(exception_config)

        if exception_config == ExceptionConfig.RAISE_GROUPED and exceptions_by_idx:
            raise ExceptionGroup("LocalPoolExecutor.map", list(exceptions_by_idx.values()))

        # Under RETURN, failed tasks appear inline as Exception objects at their submission slot;
        # under RAISE_EAGERLY / RAISE_GROUPED, `exceptions_by_idx` is empty and every slot is a success.
        match result_config:
            case ResultConfig.RESULT:
                return [
                    exceptions_by_idx[i] if i in exceptions_by_idx else all_futures[i].result()
                    for i in range(len(all_futures))
                ]
            case ResultConfig.FUTURE_COMPLETED:
                return [
                    exceptions_by_idx[i] if i in exceptions_by_idx else LocalPoolFuture(all_futures[i])
                    for i in range(len(all_futures))
                ]
            case _:
                raise AssertionError("FUTURE_PENDING is handled by the early return above")

    @overload
    def map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.RESULT] = ...,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY] = ...,
        retry_config: int | RetryConfig | None = ...,
        ordered: bool = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Iterator[R]: ...

    @overload
    def map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.RESULT] = ...,
        exception_config: Literal[ExceptionConfig.RETURN] = ...,
        retry_config: int | RetryConfig | None = ...,
        ordered: bool = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Iterator[R | Exception]: ...

    @overload
    def map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED] = ...,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY] = ...,
        retry_config: int | RetryConfig | None = ...,
        ordered: bool = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Iterator[LocalPoolFuture[R]]: ...

    @overload
    def map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED] = ...,
        exception_config: Literal[ExceptionConfig.RETURN] = ...,
        retry_config: int | RetryConfig | None = ...,
        ordered: bool = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Iterator[LocalPoolFuture[R] | Exception]: ...

    @overload
    def map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = ...,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY] = ...,
        retry_config: int | RetryConfig | None = ...,
        ordered: bool = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Iterator[LocalPoolFuture[R]] | Iterator[R]: ...

    @overload
    def map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = ...,
        exception_config: Literal[ExceptionConfig.RETURN] = ...,
        retry_config: int | RetryConfig | None = ...,
        ordered: bool = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Iterator[R | Exception] | Iterator[LocalPoolFuture[R] | Exception]: ...

    def map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = DEFAULT_RESULT_CONFIG,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RETURN] = DEFAULT_EXCEPTION_CONFIG,
        retry_config: int | RetryConfig | None = None,
        ordered: bool = True,
        max_pending_tasks: int | None = DEFAULT_MAX_PENDING_TASKS,
        executor_kwargs: Mapping[type[Executor], Any] | None = None,
    ) -> Iterator[Future[R]] | Iterator[Future[R] | Exception] | Iterator[R] | Iterator[R | Exception]:
        del executor_kwargs
        func, first_args, args, kwargs = w.func, w.first_args, w.args, w.kwargs
        del w
        first_args = iter(first_args)
        func = _prepare_func(func, retry_config)

        # FUTURE_PENDING: futures are yielded without waiting for completion, so `ordered` and
        # `exception_config` are moot (exceptions surface via the returned futures). We yield in
        # submission order, batching up futures between backpressure pauses.
        if result_config == ResultConfig.FUTURE_PENDING:
            inflight: set[concurrent.futures.Future[R]] = set()
            unyielded: deque[concurrent.futures.Future[R]] = deque()
            for arg in first_args:
                if isinstance(max_pending_tasks, int) and len(inflight) >= max_pending_tasks:
                    _, inflight = concurrent.futures.wait(inflight, return_when=concurrent.futures.FIRST_COMPLETED)
                    while unyielded:
                        yield LocalPoolFuture(unyielded.popleft())
                fut = self._pool.submit(func, arg, *args, **kwargs)
                inflight.add(fut)
                unyielded.append(fut)
            while unyielded:
                yield LocalPoolFuture(unyielded.popleft())
            return

        # RESULT / FUTURE_COMPLETED: yield results/futures as tasks complete. The submission loop
        # and the yield-on-completion loop share one `pending` set; when backpressure kicks in we
        # drain completed futures (respecting `ordered`) before submitting more.
        pending: set[concurrent.futures.Future[R]] = set()
        fut_to_idx: dict[concurrent.futures.Future[R], int] = {}
        next_to_yield = 0
        buffered: dict[int, R | LocalPoolFuture[R] | Exception] = {}

        def _resolve(completed: set[concurrent.futures.Future[R]]) -> list[R | LocalPoolFuture[R] | Exception]:
            """Resolve a batch of completed futures into items respecting `ordered` and the configs."""
            nonlocal next_to_yield
            to_yield: list[R | LocalPoolFuture[R] | Exception] = []
            # Sort completions by submission idx so that `ordered=True` can yield contiguously
            # from `next_to_yield` without needing additional dict scans.
            for cf_future in sorted(completed, key=fut_to_idx.__getitem__):
                idx = fut_to_idx.pop(cf_future)
                exc = cf_future.exception()
                resolved: R | LocalPoolFuture[R] | Exception
                if exc is not None:
                    if not isinstance(exc, Exception):
                        raise exc  # BaseException — always propagate
                    match exception_config:
                        case ExceptionConfig.RAISE_EAGERLY:
                            raise exc
                        case ExceptionConfig.RETURN:
                            resolved = exc
                        case _:
                            assert_never(exception_config)
                else:
                    match result_config:
                        case ResultConfig.RESULT:
                            resolved = cf_future.result()
                        case ResultConfig.FUTURE_COMPLETED:
                            resolved = LocalPoolFuture(cf_future)
                        case ResultConfig.FUTURE_PENDING:
                            raise AssertionError("FUTURE_PENDING never reaches _resolve")
                        case _:
                            assert_never(result_config)
                if ordered:
                    buffered[idx] = resolved
                else:
                    to_yield.append(resolved)
            if ordered:
                while next_to_yield in buffered:
                    to_yield.append(buffered.pop(next_to_yield))
                    next_to_yield += 1
            return to_yield

        for total_submitted, arg in enumerate(first_args):
            if isinstance(max_pending_tasks, int) and len(pending) >= max_pending_tasks:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                yield from _resolve(done)
            fut = self._pool.submit(func, arg, *args, **kwargs)
            fut_to_idx[fut] = total_submitted
            pending.add(fut)
        del func, first_args, args, kwargs

        # Drain remaining completions in batches.
        while pending:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            yield from _resolve(done)
