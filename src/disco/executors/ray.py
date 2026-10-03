# Ray's / Dask's APIs are only partially annotated, so values derived from them are `Unknown` / `Any` to pyright.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportUnknownLambdaType=false, reportAny=false

from __future__ import annotations

import functools
import logging
import os
import socket
from collections import deque
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, assert_never, cast, overload, override

import ray
import ray.exceptions

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


@overload
def _get_object_refs[T](
    object_refs: Sequence[ray.ObjectRef[T]],
    *,
    raise_eagerly: Literal[True],
    timeout: float | None = ...,
) -> Sequence[T]: ...


@overload
def _get_object_refs[T](
    object_refs: Sequence[ray.ObjectRef[T]],
    *,
    raise_eagerly: bool = ...,
    timeout: float | None = ...,
) -> Sequence[T | Exception]: ...


def _get_object_refs[T](
    object_refs: Sequence[ray.ObjectRef[T]],
    *,
    raise_eagerly: bool = False,
    timeout: float | None = None,
) -> Sequence[T] | Sequence[T | Exception]:
    """Resolve `object_refs` and return values in the same order as the input.

    In Ray Client mode, a batched `ray.get([...])` is translated to a single gRPC
    request which fails as a whole as soon as any ref raises — there is no way to
    retrieve partial successes. Retrieving refs one-by-one is the only portable way
    to collect successes around failures, at the cost of an extra round-trip per ref.

    With `raise_eagerly=True`, the batched `ray.get` is used instead: faster, but
    the first exception aborts retrieval. Use this when the caller would re-raise
    the first exception anyway.

    TODO: Right now, the `timeout` parameter is having different effects depending on `raise_eagerly`.
    """
    if raise_eagerly:
        return ray.get(object_refs, timeout=timeout)
    results: list[T | Exception] = []
    for ref in object_refs:
        try:
            results.append(ray.get(ref, timeout=timeout))
        except ray.exceptions.RayTaskError as exc:
            # `ray.get` raises `exc.as_instanceof_cause()` — a dynamic subclass that wraps the
            # original user exception. Unwrapping to `exc.cause` restores the original exception
            # (same semantics as `RayFuture.exception()`), so that callers see plain exception
            # objects rather than the ray-internal wrappers.
            results.append(exc.cause if exc.cause is not None else exc)
        except Exception as exc:
            results.append(exc)
    return results


class _RaySuccessSentinel:
    """Singleton placeholder that `setup_func`'s wrapper returns alongside the real result.

    Retrieving this ref via `ray.get` is the inexpensive way to check whether the underlying task
    succeeded or failed — the full (possibly large) return value stays in the object store.
    """


_RAY_SUCCESS = _RaySuccessSentinel()


class _SentinelRemoteFunction[R](Protocol):
    """The remote function built by `RayExecutor._setup_func`.

    Its `num_returns=2` makes `.remote(...)` return `[sentinel_ref, result_ref]`. Ray leaves `.remote` unannotated, so
    this spells out the shape; it's typed as a tuple so that unpacking keeps both element types.
    """

    def options(self, **task_options: object) -> _SentinelRemoteFunction[R]: ...
    def remote(
        self, *args: object, **kwargs: object
    ) -> tuple[ray.ObjectRef[_RaySuccessSentinel], ray.ObjectRef[R]]: ...


class RayFuture[R](Future[R]):
    result_ref: ray.ObjectRef[R]
    sentinel_ref: ray.ObjectRef[_RaySuccessSentinel]

    def __init__(
        self,
        result_ref: ray.ObjectRef[R],
        sentinel_ref: ray.ObjectRef[_RaySuccessSentinel],
    ) -> None:
        self.result_ref = result_ref
        self.sentinel_ref = sentinel_ref

    @override
    def cancel(self) -> bool:
        # Both refs originate from the same underlying task (split via `num_returns=2`),
        # so cancelling either one cancels the task.
        ray.cancel(self.result_ref)
        return True

    @override
    def result(self, timeout: float | None = None) -> R:
        return ray.get(self.result_ref, timeout=timeout)

    @override
    def exception(self, timeout: float | None = None) -> Exception | None:
        try:
            sentinel = ray.get(self.sentinel_ref, timeout=timeout)
        except ray.exceptions.RayTaskError as e:
            return e.cause  # type: ignore[no-any-return]
        except _RAY_SYSTEM_ERRORS as e:
            # These are not wrapped in RayTaskError but raised directly by ray.get()
            return e
        else:
            assert isinstance(sentinel, _RaySuccessSentinel), (
                f"Expected `_RaySuccessSentinel` from the `setup_func` wrapper, got {sentinel!r}"
            )
            return None


EMPTY_DICT: Mapping[str, object] = MappingProxyType({})


class RayKwargs(NamedTuple):
    # Additional `kwargs` for `remote_func = ray.remote([func], **kwargs)`
    func_remote_kwargs: Mapping[str, object] = EMPTY_DICT
    # Additional `kwargs` for `obj_refs = remote_func.options(**kwargs).remote([args])`
    func_options_kwargs: Mapping[str, object] = EMPTY_DICT
    # Timeout for `results = ray.get(obj_refs, timeout=timeout)`
    get_timeout: float | None = None
    # Timeout for `done, pending = ray.wait(obj_refs, timeout=timeout)`
    wait_poll_interval: float | None = 10


# Errors from Ray that always should be retried
_RAY_SYSTEM_ERRORS: tuple[type[ray.exceptions.RayError], ...] = (
    ray.exceptions.RaySystemError,
    ray.exceptions.WorkerCrashedError,
    ray.exceptions.NodeDiedError,
)

LOGGER = logging.getLogger(__name__)


def _wrap_with_exception_logging[**P, R](func: Callable[P, R]) -> Callable[P, tuple[_RaySuccessSentinel, R]]:
    """
    Wrapper that catches exceptions, logs their stack trace, and re-raises them.

    Additionally, it wraps the return type in a tuple with a success sentinel for
    allowing retrieving the task success state without deserializing the full return value.

    This is necessary because Ray's error handling serializes exceptions that occur
    in remote tasks and transfers them back to the caller. During this process, Ray
    captures the exception information but does not log it to the worker's stdout/stderr.
    Instead, Ray stores the exception in the object store and only raises it when
    .result() or .get() is called on the future/ObjectRef.

    This means that if a Ray task fails:
    1. The exception traceback is NOT printed to the worker logs by default
    2. The exception only becomes visible when explicitly retrieved by the caller
    3. If the caller doesn't check the result or handles it in a way that suppresses
       the traceback (e.g., broad exception handling), the stack trace is effectively lost

    By logging the exception here before Ray serializes it, we ensure the full stack
    trace is captured in the worker logs where the exception actually occurred, making
    debugging significantly easier.
    """

    @functools.wraps(func)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> tuple[_RaySuccessSentinel, R]:
        try:
            return _RAY_SUCCESS, func(*args, **kwargs)
        except Exception:
            log_lines: list[str] = [f"Saw exception on call of {func.__name__}:"]

            try:
                log_lines.append(f"[args={args!r}, kwargs={kwargs!r}]")
            except Exception as e:
                log_lines.append(f"[args=<error: {e!r}>, kwargs=<error: {e!r}>]")

            log_lines.append("Ray Execution Environment:")
            try:
                runtime_context = ray.get_runtime_context()
                log_lines.append(f"  Task ID: {runtime_context.get_task_id()}")
                log_lines.append(f"  Job ID: {runtime_context.get_job_id()}")
                log_lines.append(f"  Node ID: {runtime_context.get_node_id()}")
                log_lines.append(f"  Worker ID: {runtime_context.get_worker_id()}")
                log_lines.append(f"  Actor ID: {runtime_context.get_actor_id()}")
                log_lines.append(f"  Namespace: {runtime_context.namespace}")
            except Exception as e:
                log_lines.append(f"  <error getting Ray runtime context: {e!r}>")

            try:
                log_lines.append(f"  Worker Hostname: {socket.gethostname()}")
            except Exception as e:
                log_lines.append(f"  Worker Hostname: <error: {e!r}>")

            try:
                log_lines.append(f"  Worker PID: {os.getpid()}")
            except Exception as e:
                log_lines.append(f"  Worker PID: <error: {e!r}>")

            # Log all the execution environment information followed by the stack trace
            LOGGER.exception("\n".join(log_lines))
            raise

    return _wrapped


class RayExecutor(Executor):
    def __init__(self) -> None:
        pass

    @staticmethod
    def _setup_func[**P, R](
        func: Callable[P, R],
        retries: int | RetryConfig | None = None,
        func_options_args: Mapping[str, object] | None = None,
    ) -> _SentinelRemoteFunction[R]:
        # Ray accepts more than its annotated `ray.remote(...)` signature declares (e.g. `name`, or `retry_exceptions` as a
        # list), so these options are passed through untyped.
        options: dict[str, Any] = {}  # pyright: ignore[reportExplicitAny]

        if retries:
            if isinstance(retries, RetryConfig):
                retry_exceptions = list(retries.exceptions) + [
                    exc for exc in _RAY_SYSTEM_ERRORS if exc not in retries.exceptions
                ]
                options["max_retries"] = retries.retries
                options["retry_exceptions"] = retry_exceptions
            else:
                options["max_retries"] = retries
                options["retry_exceptions"] = True
        else:
            options["retry_exceptions"] = _RAY_SYSTEM_ERRORS

        options["name"] = func.__name__

        # Split each task's output into (sentinel, result) so that success/failure can be checked
        # cheaply via the sentinel ref without materializing the full return value in the driver.
        options["num_returns"] = 2

        if func_options_args:
            options |= func_options_args

        # Ray's annotations pick the remote-function type by the function's arity and can't express `num_returns=2`.
        remote_func = cast("object", ray.remote(**options)(_wrap_with_exception_logging(func)))
        return cast("_SentinelRemoteFunction[R]", remote_func)

    @override
    def _submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: ResultConfig,
        retry_config: int | RetryConfig | None,
        executor_kwargs: Mapping[type[Executor], object] | None,
    ) -> RayFuture[R] | R:
        ray_executor_kwargs = (executor_kwargs or {}).get(type(self), RayKwargs())
        if not isinstance(ray_executor_kwargs, RayKwargs):
            raise TypeError(f"`submission_args` must be RayKwargs, got {ray_executor_kwargs!r}")

        func, args, kwargs = w.func, w.args, w.kwargs
        del w

        remote_func = self._setup_func(func, retry_config, func_options_args=ray_executor_kwargs.func_remote_kwargs)
        del func

        if ray_executor_kwargs.func_options_kwargs:
            remote_func = remote_func.options(**ray_executor_kwargs.func_options_kwargs)
        # `num_returns=2` in `setup_func` makes `.remote(...)` return [sentinel_ref, result_ref].
        sentinel_ref, result_ref = remote_func.remote(*args, **kwargs)
        del remote_func, args, kwargs

        match result_config:
            case ResultConfig.RESULT:
                return ray.get(result_ref, timeout=ray_executor_kwargs.get_timeout)
            case ResultConfig.FUTURE_PENDING:
                return RayFuture(result_ref=result_ref, sentinel_ref=sentinel_ref)
            case ResultConfig.FUTURE_COMPLETED:
                fut = RayFuture(result_ref=result_ref, sentinel_ref=sentinel_ref)
                # `fut.exception()` reads the sentinel ref; the full result payload is never fetched.
                if exc := fut.exception(timeout=ray_executor_kwargs.get_timeout):
                    raise exc
                return fut

    @override
    def _map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig,
        exception_config: ExceptionConfig,
        retry_config: int | RetryConfig | None,
        max_pending_tasks: int | None,
        executor_kwargs: Mapping[type[Executor], object] | None,
    ) -> Sequence[R | RayFuture[R] | Exception]:
        ray_executor_kwargs = (executor_kwargs or {}).get(type(self), RayKwargs())
        if not isinstance(ray_executor_kwargs, RayKwargs):
            raise TypeError(f"`submission_args` must be RayKwargs, got {ray_executor_kwargs!r}")

        func, first_args, args, kwargs = w.func, w.first_args, w.args, w.kwargs
        del w
        # Force making `first_args` an iterator such that it can gradually be GC'd if it was a Sequence
        first_args = iter(first_args)

        remote_func = self._setup_func(func, retry_config, func_options_args=ray_executor_kwargs.func_remote_kwargs)
        if ray_executor_kwargs.func_options_kwargs:
            remote_func = remote_func.options(**ray_executor_kwargs.func_options_kwargs)
        del func

        # Every task emits a pair `(sentinel_ref, result_ref)` via `num_returns=2`. Waiting on /
        # fetching the sentinel ref is inexpensive and tells us whether the task succeeded (returns the
        # `_RaySuccessSentinel`) or failed (raises) without pulling the potentially large result
        # payload into the driver.
        #
        # `pending` holds exactly one kind of ref per invocation, determined by `result_config`:
        #   - RESULT: result refs — we must fetch the values anyway, so no sentinel-trick gain.
        #     Values land in `collected[idx]`; once processed, refs drop out of `pending` and
        #     `ref_to_idx` and can be GC'd. `futures` stays empty.
        #   - FUTURE_COMPLETED: sentinel refs — inexpensive success/failure check; result refs are never
        #     fetched in the executor, only kept inside `futures` for the returned RayFutures.
        #   - FUTURE_PENDING: sentinel refs purely for backpressure; nothing is fetched.
        pending: list[ray.ObjectRef[R] | ray.ObjectRef[_RaySuccessSentinel]] = []
        ref_to_idx: dict[ray.ObjectRef[R] | ray.ObjectRef[_RaySuccessSentinel], int] = {}
        # Built at submission time in FUTURE_* modes; indexed by submission_idx for the output.
        futures: list[RayFuture[R]] = []
        collected: dict[int, R] = {}
        # Exceptions are always tracked by submission_idx. RAISE_GROUPED iterates the values
        # into an ExceptionGroup; RETURN looks them up by idx to place them inline in the output.
        exceptions_by_idx: dict[int, Exception] = {}

        def _record_exception(submission_idx: int, exc: Exception) -> None:
            match exception_config:
                case ExceptionConfig.RAISE_GROUPED | ExceptionConfig.RETURN:
                    exceptions_by_idx[submission_idx] = exc
                case ExceptionConfig.RAISE_EAGERLY:
                    raise AssertionError("RAISE_EAGERLY is surfaced by _get_object_refs")
                case _:
                    assert_never(exception_config)

        def _process_completed(completed: list[ray.ObjectRef[R] | ray.ObjectRef[_RaySuccessSentinel]]) -> None:
            """Resolve a batch of completed refs, distributing values and exceptions to the outer state.

            `completed` is monomorphic per invocation (all result refs under RESULT, all sentinels
            under FUTURE_COMPLETED). The per-element dispatch below relies on the value type, not
            the ref type: the sentinel branch cannot fire under RESULT (we wait on result refs), and
            the value branch cannot fire under FUTURE_COMPLETED (we wait on sentinels).
            """
            # `ray.ObjectRef` is invariant, so mypy/pyright cannot distribute the union
            # `list[ObjectRef[R] | ObjectRef[_RaySuccessSentinel]]` over `_get_object_refs`'s
            # single `T`. At runtime the list is monomorphic (all result refs under RESULT, all
            # sentinels under FUTURE_COMPLETED), so the call is safe; the value-type isinstance
            # checks below pick up whichever element type was actually returned.
            batch_results: Sequence[R | _RaySuccessSentinel | Exception] = _get_object_refs(
                completed,  # type: ignore[arg-type]
                raise_eagerly=(exception_config == ExceptionConfig.RAISE_EAGERLY),
                timeout=ray_executor_kwargs.get_timeout,
            )
            for completed_ref, value in zip(completed, batch_results, strict=True):
                submission_idx = ref_to_idx.pop(completed_ref)
                if isinstance(value, Exception):
                    _record_exception(submission_idx, value)
                elif isinstance(value, _RaySuccessSentinel):
                    # FUTURE_COMPLETED: success confirmed without materializing the result payload.
                    pass
                else:
                    collected[submission_idx] = value

        # Tight submission-and-retrieval loop — at any point, at most max_pending_tasks are pending.
        # For unlimited max_pending_tasks, all tasks are submitted first, then drained below.
        total_submitted = 0

        for arg in first_args:
            if isinstance(max_pending_tasks, int) and len(pending) >= max_pending_tasks:
                done, pending = ray.wait(
                    pending,
                    timeout=ray_executor_kwargs.wait_poll_interval,
                    num_returns=max(1, max_pending_tasks // 4),
                    fetch_local=False,
                )
                if result_config != ResultConfig.FUTURE_PENDING:
                    _process_completed(done)
            sentinel_ref, result_ref = remote_func.remote(arg, *args, **kwargs)
            wait_ref: ray.ObjectRef[R] | ray.ObjectRef[_RaySuccessSentinel]
            match result_config:
                case ResultConfig.RESULT:
                    wait_ref = result_ref
                    ref_to_idx[wait_ref] = total_submitted
                case ResultConfig.FUTURE_COMPLETED:
                    wait_ref = sentinel_ref
                    ref_to_idx[wait_ref] = total_submitted
                    futures.append(RayFuture(result_ref=result_ref, sentinel_ref=sentinel_ref))
                case ResultConfig.FUTURE_PENDING:
                    wait_ref = sentinel_ref
                    futures.append(RayFuture(result_ref=result_ref, sentinel_ref=sentinel_ref))
                case _:
                    assert_never(result_config)
            pending.append(wait_ref)
            total_submitted += 1
        del remote_func, first_args, args, kwargs

        if result_config == ResultConfig.FUTURE_PENDING:
            return futures

        # Drain remaining pending tasks through the same retrieval path
        while pending:
            done, pending = ray.wait(
                pending,
                timeout=ray_executor_kwargs.wait_poll_interval,
                num_returns=min(max(2, total_submitted // 4), len(pending)),
                fetch_local=False,
            )
            _process_completed(done)

        if exception_config == ExceptionConfig.RAISE_GROUPED and exceptions_by_idx:
            raise ExceptionGroup("RayExecutor.map", list(exceptions_by_idx.values()))

        # Under RETURN, failed tasks appear inline as Exception objects at their submission slot;
        # successful tasks come from `collected` (RESULT) or a fresh RayFuture (FUTURE_COMPLETED).
        # For RAISE_EAGERLY/RAISE_GROUPED, `exceptions_by_idx` is empty and every slot is a success.
        match result_config:
            case ResultConfig.RESULT:
                return [
                    exceptions_by_idx[i] if i in exceptions_by_idx else collected[i] for i in range(total_submitted)
                ]
            case ResultConfig.FUTURE_COMPLETED:
                return [exceptions_by_idx[i] if i in exceptions_by_idx else futures[i] for i in range(total_submitted)]
            case _:
                assert_never(result_config)

    @override
    def _map_lazy[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig,
        exception_config: Literal[ExceptionConfig.RAISE_EAGERLY, ExceptionConfig.RETURN],
        retry_config: int | RetryConfig | None,
        ordered: bool,
        max_pending_tasks: int | None,
        executor_kwargs: Mapping[type[Executor], object] | None,
    ) -> Iterator[R | RayFuture[R] | Exception]:
        ray_executor_kwargs = (executor_kwargs or {}).get(type(self), RayKwargs())
        if not isinstance(ray_executor_kwargs, RayKwargs):
            raise TypeError(f"`submission_args` must be RayKwargs, got {ray_executor_kwargs!r}")

        func, first_args, args, kwargs = w.func, w.first_args, w.args, w.kwargs
        del w
        first_args = iter(first_args)

        remote_func = self._setup_func(func, retry_config, func_options_args=ray_executor_kwargs.func_remote_kwargs)
        if ray_executor_kwargs.func_options_kwargs:
            remote_func = remote_func.options(**ray_executor_kwargs.func_options_kwargs)
        del func

        # FUTURE_PENDING: submit in bulk up to max_pending_tasks, yield the batch, repeat.
        # exception_config is irrelevant here — futures are returned without waiting, so exceptions
        # only surface when the caller inspects each future individually.
        # Wait on sentinel refs purely for backpressure; the result refs are never fetched here.
        if result_config == ResultConfig.FUTURE_PENDING:
            # `inflight` tracks the sentinel refs for `ray.wait`-based backpressure; `unyielded`
            # holds the already-constructed RayFutures waiting to be yielded.
            inflight: list[ray.ObjectRef[_RaySuccessSentinel]] = []
            unyielded: deque[RayFuture[R]] = deque()
            for arg in first_args:
                if isinstance(max_pending_tasks, int) and len(inflight) >= max_pending_tasks:
                    _, inflight = ray.wait(
                        inflight,
                        timeout=ray_executor_kwargs.wait_poll_interval,
                        num_returns=max(1, max_pending_tasks // 4),
                        fetch_local=False,
                    )
                    # Yield all submitted-but-not-yet-yielded futures before submitting more
                    while unyielded:
                        yield unyielded.popleft()
                sentinel_ref, result_ref = remote_func.remote(arg, *args, **kwargs)
                inflight.append(sentinel_ref)
                unyielded.append(RayFuture(result_ref=result_ref, sentinel_ref=sentinel_ref))
            while unyielded:
                yield unyielded.popleft()
            return

        # RESULT / FUTURE_COMPLETED: same `pending`-union structure as `map()`. Under RESULT the
        # ref we wait on is the result ref (we fetch the value anyway); under FUTURE_COMPLETED it's
        # the sentinel ref (inexpensive success/failure check, no fetch of the large payload).
        pending: list[ray.ObjectRef[R] | ray.ObjectRef[_RaySuccessSentinel]] = []
        ref_to_idx: dict[ray.ObjectRef[R] | ray.ObjectRef[_RaySuccessSentinel], int] = {}
        # FUTURE_COMPLETED only: idx → the pre-built RayFuture for that task, yielded once the
        # sentinel resolves. Keyed by idx (not by ref) to keep the value-side types strong without
        # having to narrow the union-typed pending ref.
        future_by_idx: dict[int, RayFuture[R]] = {}
        next_to_yield = 0
        buffered: dict[int, R | RayFuture[R] | Exception] = {}

        def _buffer_or_yield(idx: int, resolved: R | RayFuture[R] | Exception) -> list[R | RayFuture[R] | Exception]:
            """Route a resolved item through the ordering buffer, returning anything ready to yield."""
            nonlocal next_to_yield
            if not ordered:
                return [resolved]
            buffered[idx] = resolved
            to_yield: list[R | RayFuture[R] | Exception] = []
            while next_to_yield in buffered:
                to_yield.append(buffered.pop(next_to_yield))
                next_to_yield += 1
            return to_yield

        def _resolve_completed(
            completed: list[ray.ObjectRef[R] | ray.ObjectRef[_RaySuccessSentinel]],
        ) -> list[R | RayFuture[R] | Exception]:
            """Resolve a batch of completed refs into values/futures, respecting ordering.

            Per invocation, `completed` is monomorphic: result refs under RESULT, sentinels under
            FUTURE_COMPLETED. The per-element dispatch discriminates on the value type — the
            sentinel branch can only fire under FUTURE_COMPLETED, the value branch only under
            RESULT — so the single loop covers both modes. The `# type: ignore[arg-type]` is
            needed because `ObjectRef` is invariant and mypy cannot distribute the union over
            `_get_object_refs`'s single `T`; the call is safe because the list is homogeneous
            at runtime.
            """
            batch_results: Sequence[R | _RaySuccessSentinel | Exception] = _get_object_refs(
                completed,  # type: ignore[arg-type]
                raise_eagerly=(exception_config == ExceptionConfig.RAISE_EAGERLY),
                timeout=ray_executor_kwargs.get_timeout,
            )
            to_yield: list[R | RayFuture[R] | Exception] = []
            for completed_ref, value in zip(completed, batch_results, strict=True):
                idx = ref_to_idx.pop(completed_ref)
                resolved: R | RayFuture[R] | Exception
                if isinstance(value, Exception):
                    # Only reached under RETURN — RAISE_EAGERLY would have been raised from _get_object_refs
                    resolved = value
                elif isinstance(value, _RaySuccessSentinel):
                    # FUTURE_COMPLETED: the RayFuture was pre-built at submission time.
                    resolved = future_by_idx.pop(idx)
                else:
                    resolved = value
                to_yield.extend(_buffer_or_yield(idx, resolved))
            return to_yield

        # Submission-and-retrieval loop — same batching pattern as map()
        total_submitted = 0

        for arg in first_args:
            if isinstance(max_pending_tasks, int) and len(pending) >= max_pending_tasks:
                done, pending = ray.wait(
                    pending,
                    timeout=ray_executor_kwargs.wait_poll_interval,
                    num_returns=max(1, max_pending_tasks // 4),
                    fetch_local=False,
                )
                yield from _resolve_completed(done)
            sentinel_ref, result_ref = remote_func.remote(arg, *args, **kwargs)
            wait_ref: ray.ObjectRef[R] | ray.ObjectRef[_RaySuccessSentinel]
            match result_config:
                case ResultConfig.RESULT:
                    wait_ref = result_ref
                case ResultConfig.FUTURE_COMPLETED:
                    wait_ref = sentinel_ref
                    future_by_idx[total_submitted] = RayFuture(result_ref=result_ref, sentinel_ref=sentinel_ref)
                case _:
                    assert_never(result_config)
            ref_to_idx[wait_ref] = total_submitted
            pending.append(wait_ref)
            total_submitted += 1
        del remote_func, first_args, args, kwargs

        # Drain remaining pending tasks
        while pending:
            done, pending = ray.wait(
                pending,
                timeout=ray_executor_kwargs.wait_poll_interval,
                num_returns=min(max(2, total_submitted // 4), len(pending)),
                fetch_local=False,
            )
            yield from _resolve_completed(done)
