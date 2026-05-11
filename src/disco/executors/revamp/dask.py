from __future__ import annotations

import functools
from collections import deque
from itertools import repeat
from typing import TYPE_CHECKING, Any, Literal, Self, assert_never, overload

import dask.base
import dask.distributed
import dask.utils

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

    from disco.executors.base import Executor as LegacyExecutor


_MAX_VALUE_LENGTH = 150


def _format_value(value: object, max_value_length: int = _MAX_VALUE_LENGTH) -> str:
    """Short, human-readable rendering of `value` for use in a Dask task key.

    Long reprs get truncated to a head/tail pair so the key stays bounded while still being
    distinguishable by eye (e.g. `[1, 2, 3, ..., 99, 100]`).
    """
    full = repr(value)
    if max_value_length and len(full) > max_value_length:
        return f"{full[: max_value_length // 2]}[...]{full[-max_value_length // 2 :]}"
    return full


def _generate_key(func_name: str, base_hash: str, arg: object, idx: int) -> str:
    """Deterministic, unique Dask task key.

    Giving Dask an explicit key avoids its default tokenization of the full arg tuple, which is
    very expensive when args are large. The trailing `-{idx}` guarantees per-submission
    uniqueness even when two fan-out args happen to hash identically — Dask silently dedups
    duplicate keys inside a single `client.map` call, which would collapse distinct output slots.
    """
    return f"{func_name[:50]}-{base_hash[:8]}-{_format_value(arg)}-{dask.base.tokenize(arg)[:8]}-{idx}"


class DaskFuture[R](Future[R]):
    """Wrapper around `dask.distributed.Future` matching the revamp `Future` protocol.

    `dask.distributed.Future.exception()` returns `BaseException | None`; our protocol promises
    `Exception | None`. BaseExceptions (`KeyboardInterrupt`, `SystemExit`, `CancelledError`,
    ...) are re-raised rather than silently handed back as values.

    Important: once all references to the underlying `dask.distributed.Future` drop, Dask's
    `__del__` signals the scheduler to release the task's result in the object store. Retaining
    a `DaskFuture` keeps exactly one extra reference; callers who stash futures in containers
    should be aware they are also extending the scheduler-side result's lifetime.
    """

    __slots__ = ("_fut",)

    def __init__(self, fut: dask.distributed.Future[R]) -> None:
        self._fut = fut

    def cancel(self) -> bool:
        # `dask.distributed.Future.cancel()` returns None; normalize to bool.
        self._fut.cancel()  # type: ignore[no-untyped-call]
        return True

    def result(self, timeout: float | None = None) -> R:
        return self._fut.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> Exception | None:
        exc = self._fut.exception(timeout=timeout)  # type: ignore[no-untyped-call]
        if exc is None or isinstance(exc, Exception):
            return exc
        raise exc


class _RetryCallable[**P, R]:
    """Pickleable callable that runs `func` with up-to-N retries on matching exceptions.

    Used when `retry_config` is a `RetryConfig` with a specific exception allow-list — Dask's
    built-in `retries=` is blanket and cannot filter by exception type, so we implement the
    filter in a per-worker wrapper. For plain `int` retry configs we skip the wrapper and let
    Dask's native mechanism handle retries (which has the benefit of being able to retry on a
    different worker, covering infrastructure failures).

    A module-level class with `__slots__` (rather than a closure) is used so instances pickle
    cleanly to Dask workers.
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


def _prepare_func_and_retries[**P, R](
    func: Callable[P, R],
    retry_config: int | RetryConfig | None,
) -> tuple[Callable[P, R], int]:
    """Decide between Dask's native `retries=` and an in-worker retry wrapper.

    - `None` or zero retries: no wrapping, `dask_retries = 0`.
    - plain `int`: leave `func` alone and pass-through to Dask. Dask retries any exception and
      may choose a different worker for each attempt, which is generally stronger.
    - `RetryConfig(retries=N, exceptions=[...])`: Dask cannot filter exceptions, so we wrap the
      function and run the retry loop on the worker ourselves; `dask_retries = 0`.
    """
    if not retry_config:
        return func, 0
    if isinstance(retry_config, RetryConfig):
        allowed_excs: tuple[type[BaseException], ...] = tuple(retry_config.exceptions)
        return _RetryCallable(func, retry_config.retries, allowed_excs), 0
    return func, retry_config


class DaskExecutor(Executor):
    """Revamp executor backed by a `dask.distributed.Client`.

    Submissions go through `client.map` whenever there is more than one task, because Dask's
    scheduler is a single-threaded Python process that is very easily overloaded by many
    individual `client.submit` calls. Task fan-in/fan-out uses `dask.distributed.as_completed`
    in batches so we talk to the scheduler as little as possible.

    Every internal bookkeeping structure keys on `Future.key` (a small string) rather than on
    the futures themselves, so our code path does not pin `dask.distributed.Future` instances
    in memory longer than strictly necessary — dropping a future's last reference is the
    signal Dask uses to release the task's result on the worker.
    """

    def __init__(self, client: dask.distributed.Client) -> None:
        self._client = client

    @classmethod
    def from_legacy_executor(cls, legacy_executor: LegacyExecutor) -> Self:
        from disco.executors.dask import DaskExecutor as LegacyDaskExecutor  # noqa: PLC0415

        if not isinstance(legacy_executor, LegacyDaskExecutor):
            raise TypeError(f"Expected a DaskExecutor, got: {legacy_executor!r}")
        return cls(client=legacy_executor.client)

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
    ) -> DaskFuture[R]: ...

    @overload
    def submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: ResultConfig = ...,
        retry_config: int | RetryConfig | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> DaskFuture[R] | R: ...

    def submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: ResultConfig = DEFAULT_RESULT_CONFIG,
        retry_config: int | RetryConfig | None = None,
        executor_kwargs: Mapping[type[Executor], Any] | None = None,
    ) -> DaskFuture[R] | R:
        del executor_kwargs
        original_func = w.func
        # `w.args` / `w.kwargs` are typed via `ParamSpec`; mypy treats those as special forms
        # that do not support indexing or iteration. Re-bind to plain `tuple` / `dict` so the
        # helper calls below can work with them uniformly.
        # `w.args` / `w.kwargs` are ParamSpec-typed; rebind via star-unpack so mypy sees
        # plain `tuple[Any, ...]` / `dict[str, Any]` that support indexing and iteration.
        # mypy can't statically recognise `**P.kwargs` as satisfying `SupportsKeysAndGetItem`
        # even though it always does at runtime, so the dict unpack takes a narrow ignore.
        shared_args: tuple[Any, ...] = (*w.args,)
        shared_kwargs: dict[str, Any] = {**w.kwargs}  # type: ignore[dict-item]
        del w

        wrapped_func, dask_retries = _prepare_func_and_retries(original_func, retry_config)
        # Primary arg is the first positional — the key bakes its repr so tasks are visually
        # distinguishable in the Dask dashboard (mirrors the legacy behaviour).
        primary = shared_args[0] if shared_args else None
        non_primary_args = shared_args[1:] if shared_args else ()
        base_hash = dask.base.tokenize(wrapped_func, *non_primary_args, **shared_kwargs)
        func_name = dask.utils.funcname(original_func)
        key = _generate_key(func_name, base_hash, primary, idx=0)

        submit_func: Callable[..., R] = (
            functools.partial(wrapped_func, **shared_kwargs) if shared_kwargs else wrapped_func
        )
        cf_future: dask.distributed.Future[R] = self._client.submit(
            submit_func, *shared_args, key=key, retries=dask_retries
        )
        match result_config:
            case ResultConfig.RESULT:
                return cf_future.result()
            case ResultConfig.FUTURE_PENDING:
                return DaskFuture(cf_future)
            case ResultConfig.FUTURE_COMPLETED:
                fut = DaskFuture(cf_future)
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
    ) -> Sequence[DaskFuture[R]]: ...

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
    ) -> Sequence[DaskFuture[R] | Exception]: ...

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
    ) -> Sequence[DaskFuture[R]] | Sequence[R]: ...

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
    ) -> Sequence[R | Exception] | Sequence[DaskFuture[R] | Exception]: ...

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
    ) -> Sequence[DaskFuture[R]] | Sequence[DaskFuture[R] | Exception] | Sequence[R] | Sequence[R | Exception]: ...

    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = DEFAULT_RESULT_CONFIG,
        exception_config: ExceptionConfig = DEFAULT_EXCEPTION_CONFIG,
        retry_config: int | RetryConfig | None = None,
        max_pending_tasks: int | None = DEFAULT_MAX_PENDING_TASKS,
        executor_kwargs: Mapping[type[Executor], Any] | None = None,
    ) -> Sequence[DaskFuture[R]] | Sequence[DaskFuture[R] | Exception] | Sequence[R] | Sequence[R | Exception]:
        del executor_kwargs
        original_func = w.func
        first_args: list[Any] = list(w.first_args)
        # Rebind ParamSpec-typed attributes to plain containers so helpers can work with them.
        # `w.args` / `w.kwargs` are ParamSpec-typed; rebind via star-unpack so mypy sees
        # plain `tuple[Any, ...]` / `dict[str, Any]` that support indexing and iteration.
        # mypy can't statically recognise `**P.kwargs` as satisfying `SupportsKeysAndGetItem`
        # even though it always does at runtime, so the dict unpack takes a narrow ignore.
        shared_args: tuple[Any, ...] = (*w.args,)
        shared_kwargs: dict[str, Any] = {**w.kwargs}  # type: ignore[dict-item]
        del w
        total = len(first_args)
        if total == 0:
            return []

        submit_func, dask_retries, keys = self._prepare_map_submission(
            original_func, first_args, shared_args, shared_kwargs, retry_config
        )
        key_to_idx: dict[str, int] = {k: i for i, k in enumerate(keys)}

        # Accumulators for the three possible kinds of output slot. Only the dicts matching the
        # active config get populated; the rest stay empty. We key by submission-index (an int)
        # rather than by future, so the bookkeeping never pins a future in memory.
        results_by_idx: dict[int, R] = {}
        futures_by_idx: dict[int, DaskFuture[R]] = {}
        exceptions_by_idx: dict[int, Exception] = {}

        def _ingest(fut: dask.distributed.Future[R]) -> None:
            """Process a single completed future from `as_completed` into the right accumulator.

            Raises eagerly under `RAISE_EAGERLY`; otherwise stashes the exception for later.
            `fut` is expected to drop out of scope at the caller's iteration boundary so Dask can
            release its result on the worker.
            """
            assert isinstance(fut.key, str)  # we only submit futures with string keys
            idx = key_to_idx[fut.key]
            exc = fut.exception()  # type: ignore[no-untyped-call]
            if exc is not None:
                if not isinstance(exc, Exception):
                    raise exc  # BaseException — always propagate
                match exception_config:
                    case ExceptionConfig.RAISE_EAGERLY:
                        raise exc
                    case ExceptionConfig.RAISE_GROUPED | ExceptionConfig.RETURN:
                        exceptions_by_idx[idx] = exc
                    case _:
                        assert_never(exception_config)
                return
            match result_config:
                case ResultConfig.RESULT:
                    results_by_idx[idx] = fut.result()
                case ResultConfig.FUTURE_COMPLETED:
                    futures_by_idx[idx] = DaskFuture(fut)
                case ResultConfig.FUTURE_PENDING:
                    raise AssertionError("FUTURE_PENDING never reaches _ingest")
                case _:
                    assert_never(result_config)

        # --- FUTURE_PENDING: submit and return without draining results ---
        # `exception_config` is intentionally ignored — exceptions surface via the returned
        # futures when the caller inspects them individually.
        if result_config == ResultConfig.FUTURE_PENDING:
            return self._submit_future_pending(
                submit_func, first_args, shared_args, keys, dask_retries, max_pending_tasks, total
            )

        # --- RESULT / FUTURE_COMPLETED: submit and drain via as_completed ---
        self._submit_and_drain(
            submit_func,
            first_args,
            shared_args,
            keys,
            dask_retries,
            max_pending_tasks,
            total,
            _ingest,
        )

        if exception_config == ExceptionConfig.RAISE_GROUPED and exceptions_by_idx:
            raise ExceptionGroup("DaskExecutor.map", list(exceptions_by_idx.values()))

        # Under RETURN, failed tasks appear inline as `Exception` at their submission slot;
        # under RAISE_EAGERLY / RAISE_GROUPED `exceptions_by_idx` is empty and every slot is
        # a success.
        match result_config:
            case ResultConfig.RESULT:
                return [exceptions_by_idx[i] if i in exceptions_by_idx else results_by_idx[i] for i in range(total)]
            case ResultConfig.FUTURE_COMPLETED:
                return [exceptions_by_idx[i] if i in exceptions_by_idx else futures_by_idx[i] for i in range(total)]
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
    ) -> Iterator[DaskFuture[R]]: ...

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
    ) -> Iterator[DaskFuture[R] | Exception]: ...

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
    ) -> Iterator[DaskFuture[R]] | Iterator[R]: ...

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
    ) -> Iterator[R | Exception] | Iterator[DaskFuture[R] | Exception]: ...

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
    ) -> Iterator[Future[R] | Exception] | Iterator[Future[R]] | Iterator[R] | Iterator[R | Exception]:
        del executor_kwargs
        original_func = w.func
        first_args: list[Any] = list(w.first_args)
        # `w.args` / `w.kwargs` are ParamSpec-typed; rebind via star-unpack so mypy sees
        # plain `tuple[Any, ...]` / `dict[str, Any]` that support indexing and iteration.
        # mypy can't statically recognise `**P.kwargs` as satisfying `SupportsKeysAndGetItem`
        # even though it always does at runtime, so the dict unpack takes a narrow ignore.
        shared_args: tuple[Any, ...] = (*w.args,)
        shared_kwargs: dict[str, Any] = {**w.kwargs}  # type: ignore[dict-item]
        del w
        total = len(first_args)
        if total == 0:
            return

        submit_func, dask_retries, keys = self._prepare_map_submission(
            original_func, first_args, shared_args, shared_kwargs, retry_config
        )
        key_to_idx: dict[str, int] = {k: i for i, k in enumerate(keys)}

        # --- FUTURE_PENDING: yield wrapped futures in submission order, throttled by
        # `max_pending_tasks`. The drain calls in the throttle loop are only to keep the
        # in-flight count under the cap; we never fetch results here.
        if result_config == ResultConfig.FUTURE_PENDING:
            yield from self._map_lazy_future_pending(
                submit_func, first_args, shared_args, keys, dask_retries, max_pending_tasks, total
            )
            return

        # --- RESULT / FUTURE_COMPLETED: stream results/futures via as_completed ---
        next_to_yield = 0
        buffered: dict[int, R | DaskFuture[R] | Exception] = {}

        def _resolve(fut: dask.distributed.Future[R]) -> R | DaskFuture[R] | Exception:
            """Turn one completed future into the item we want to yield."""
            exc = fut.exception()  # type: ignore[no-untyped-call]
            if exc is not None:
                if not isinstance(exc, Exception):
                    raise exc  # BaseException — always propagate
                match exception_config:
                    case ExceptionConfig.RAISE_EAGERLY:
                        raise exc
                    case ExceptionConfig.RETURN:
                        return exc
                    case _:
                        assert_never(exception_config)
            match result_config:
                case ResultConfig.RESULT:
                    return fut.result()
                case ResultConfig.FUTURE_COMPLETED:
                    return DaskFuture(fut)
                case ResultConfig.FUTURE_PENDING:
                    raise AssertionError("FUTURE_PENDING never reaches _resolve")
                case _:
                    assert_never(result_config)

        # Interleaved submit/drain loop driven by `as_completed.next_batch(block=True)`.
        ac = dask.distributed.as_completed()  # type: ignore[no-untyped-call]
        next_to_submit = 0
        in_flight = 0
        chunk_size = max_pending_tasks if isinstance(max_pending_tasks, int) else total

        def _top_up(cap: int) -> None:
            """Submit up to `cap` fresh tasks (or fewer if we are out of args)."""
            nonlocal next_to_submit, in_flight
            want = min(cap, total - next_to_submit)
            if want <= 0:
                return
            start, end = next_to_submit, next_to_submit + want
            chunk_futs = self._client.map(
                submit_func,
                first_args[start:end],
                *[list(repeat(a, want)) for a in shared_args],
                key=keys[start:end],
                retries=dask_retries,
            )
            ac.update(chunk_futs)  # type: ignore[no-untyped-call]
            in_flight += want
            next_to_submit = end
            del chunk_futs  # only `ac` holds refs now

        _top_up(chunk_size)

        def _key_idx(f: dask.distributed.Future[R]) -> int:
            assert isinstance(f.key, str)  # we only submit futures with string keys
            return key_to_idx[f.key]

        while in_flight > 0:
            batch = ac.next_batch(block=True)  # type: ignore[no-untyped-call]
            to_yield: list[R | DaskFuture[R] | Exception] = []
            # Sort by submission index so `ordered=True` can drain `buffered` contiguously.
            batch.sort(key=_key_idx)
            for fut in batch:
                idx = _key_idx(fut)
                resolved = _resolve(fut)
                if ordered:
                    buffered[idx] = resolved
                else:
                    to_yield.append(resolved)
                in_flight -= 1
                del fut  # drop worker-side reference as soon as possible
            if ordered:
                while next_to_yield in buffered:
                    to_yield.append(buffered.pop(next_to_yield))
                    next_to_yield += 1
            yield from to_yield
            if isinstance(max_pending_tasks, int):
                _top_up(max_pending_tasks - in_flight)

    # --------------------------- internal helpers ---------------------------

    def _prepare_map_submission[**P, R](
        self,
        original_func: Callable[..., R],
        first_args: list[Any],
        shared_args: tuple[Any, ...],
        shared_kwargs: Mapping[str, Any],
        retry_config: int | RetryConfig | None,
    ) -> tuple[Callable[..., R], int, list[str]]:
        """Shared setup for `map` and `map_lazy`: apply the retry wrapper, bake kwargs, and
        build a list of deterministic Dask task keys (one per fan-out arg).

        `dask.utils.funcname` is called on the *original* function so the task key stays
        readable even when `_RetryCallable` / `functools.partial` wrapping kicks in.
        """
        wrapped_func, dask_retries = _prepare_func_and_retries(original_func, retry_config)
        submit_func: Callable[..., R] = (
            functools.partial(wrapped_func, **shared_kwargs) if shared_kwargs else wrapped_func
        )
        func_name = dask.utils.funcname(original_func)
        base_hash = dask.base.tokenize(wrapped_func, *shared_args, **shared_kwargs)
        keys = [_generate_key(func_name, base_hash, arg, idx) for idx, arg in enumerate(first_args)]
        return submit_func, dask_retries, keys

    def _submit_and_drain[R](
        self,
        submit_func: Callable[..., R],
        first_args: list[Any],
        shared_args: tuple[Any, ...],
        keys: list[str],
        dask_retries: int,
        max_pending_tasks: int | None,
        total: int,
        on_complete: Callable[[dask.distributed.Future[R]], None],
    ) -> None:
        """Eager submit+drain used by `map` under RESULT / FUTURE_COMPLETED.

        Keeps at most `max_pending_tasks` tasks in flight; calls `on_complete` exactly once per
        completed future. `on_complete` is expected not to retain the future past its return.
        """
        ac = dask.distributed.as_completed()  # type: ignore[no-untyped-call]
        next_to_submit = 0
        in_flight = 0
        chunk_size = max_pending_tasks if isinstance(max_pending_tasks, int) else total

        def _top_up(cap: int) -> None:
            nonlocal next_to_submit, in_flight
            want = min(cap, total - next_to_submit)
            if want <= 0:
                return
            start, end = next_to_submit, next_to_submit + want
            chunk_futs = self._client.map(
                submit_func,
                first_args[start:end],
                *[list(repeat(a, want)) for a in shared_args],
                key=keys[start:end],
                retries=dask_retries,
            )
            ac.update(chunk_futs)  # type: ignore[no-untyped-call]
            in_flight += want
            next_to_submit = end
            del chunk_futs

        _top_up(chunk_size)
        while in_flight > 0:
            batch = ac.next_batch(block=True)  # type: ignore[no-untyped-call]
            for fut in batch:
                on_complete(fut)
                in_flight -= 1
                del fut
            if isinstance(max_pending_tasks, int):
                _top_up(max_pending_tasks - in_flight)

    def _submit_future_pending[R](
        self,
        submit_func: Callable[..., R],
        first_args: list[Any],
        shared_args: tuple[Any, ...],
        keys: list[str],
        dask_retries: int,
        max_pending_tasks: int | None,
        total: int,
    ) -> list[DaskFuture[R]]:
        """Submission path for `map(..., result_config=FUTURE_PENDING)`.

        Submits in one shot when unthrottled; otherwise chunks and drains `as_completed` just
        enough between chunks to keep the scheduler's in-flight count under the cap. Results
        are never fetched — completed futures only flow through `as_completed` for counting.
        """
        if max_pending_tasks is None or total <= max_pending_tasks:
            chunk_futs = self._client.map(
                submit_func,
                first_args,
                *[list(repeat(a, total)) for a in shared_args],
                key=keys,
                retries=dask_retries,
            )
            wrapped = [DaskFuture(f) for f in chunk_futs]
            del chunk_futs
            return wrapped

        ac = dask.distributed.as_completed()  # type: ignore[no-untyped-call]
        wrapped_all: list[DaskFuture[R]] = []
        in_flight = 0

        for start in range(0, total, max_pending_tasks):
            # Make room for the next chunk before submitting it.
            while in_flight + max_pending_tasks > total - start + in_flight and in_flight > 0:
                # This clause is just the guard `in_flight > 0` — it's only here to keep the
                # loop exited when there is no need to drain.
                break
            while in_flight > 0 and in_flight >= max_pending_tasks:
                batch = ac.next_batch(block=True)  # type: ignore[no-untyped-call]
                in_flight -= len(batch)
                del batch  # drop the internal refs
            end = min(start + max_pending_tasks, total)
            want = end - start
            chunk_futs = self._client.map(
                submit_func,
                first_args[start:end],
                *[list(repeat(a, want)) for a in shared_args],
                key=keys[start:end],
                retries=dask_retries,
            )
            ac.update(chunk_futs)  # type: ignore[no-untyped-call]
            in_flight += want
            wrapped_all.extend(DaskFuture(f) for f in chunk_futs)
            del chunk_futs  # only `ac` + the wrapper in `wrapped_all` hold refs now
        return wrapped_all

    def _map_lazy_future_pending[R](
        self,
        submit_func: Callable[..., R],
        first_args: list[Any],
        shared_args: tuple[Any, ...],
        keys: list[str],
        dask_retries: int,
        max_pending_tasks: int | None,
        total: int,
    ) -> Iterator[DaskFuture[R]]:
        """Streaming version of `_submit_future_pending` — yields wrapped futures in submission
        order, with a chunked submit/drain cadence when `max_pending_tasks` is set."""
        if max_pending_tasks is None or total <= max_pending_tasks:
            chunk_futs = self._client.map(
                submit_func,
                first_args,
                *[list(repeat(a, total)) for a in shared_args],
                key=keys,
                retries=dask_retries,
            )
            for f in chunk_futs:
                yield DaskFuture(f)
            return

        ac = dask.distributed.as_completed()  # type: ignore[no-untyped-call]
        unyielded: deque[DaskFuture[R]] = deque()
        in_flight = 0
        for start in range(0, total, max_pending_tasks):
            while in_flight >= max_pending_tasks:
                batch = ac.next_batch(block=True)  # type: ignore[no-untyped-call]
                in_flight -= len(batch)
                del batch
                # Hand the previous chunk to the caller before submitting the next one.
                while unyielded:
                    yield unyielded.popleft()
            end = min(start + max_pending_tasks, total)
            want = end - start
            chunk_futs = self._client.map(
                submit_func,
                first_args[start:end],
                *[list(repeat(a, want)) for a in shared_args],
                key=keys[start:end],
                retries=dask_retries,
            )
            ac.update(chunk_futs)  # type: ignore[no-untyped-call]
            in_flight += want
            for f in chunk_futs:
                unyielded.append(DaskFuture(f))
            del chunk_futs
        while unyielded:
            yield unyielded.popleft()
