from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    overload,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from disco.executors.base import Executor as LegacyExecutor


@runtime_checkable
class Future[R](Protocol):
    def cancel(self) -> bool: ...
    def result(self, timeout: float | None = None) -> R: ...
    def exception(self, timeout: float | None = None) -> Exception | None: ...


# The classes `SingleWrap` and `MultipleWrap` are needed because as soon as a method has `ParamSpec` `*args` or
# `**kwargs`, it can't have any other `kwargs` anymore, as `*args: P.args`, `**kwargs: P.kwargs` by design always have
# to be directly besides each other. This means the `Executor` implementations consume the function and its arguments
# through a typed wrapper, but all other arguments (such as `result_config`) can be normal `kwargs`.

# These are bare Python classes as it's impossible to annotate the `args` or `kwargs` types in a dataclass or NamedTuple
# https://github.com/python/typing/issues/1252


class SingleWrap[**P, R]:
    __slots__ = ("args", "func", "kwargs")

    def __init__(self, func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs


class MultipleWrap[T, **P, R]:
    __slots__ = ("func", "kwargs", "args", "first_args")

    def __init__(
        self,
        func: Callable[Concatenate[T, P], R],
        first_args: Iterable[T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        self.func = func
        self.first_args = first_args
        self.args = args
        self.kwargs = kwargs


def wrap[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> SingleWrap[P, R]:
    return SingleWrap(func, *args, **kwargs)


def mwrap[T, **P, R](
    func: Callable[Concatenate[T, P], R],
    first_args: Iterable[T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> MultipleWrap[T, P, R]:
    return MultipleWrap(func, first_args, *args, **kwargs)


class RetryConfig(NamedTuple):
    retries: int
    exceptions: Sequence[type[BaseException]]

    def __bool__(self) -> bool:
        return self.retries > 0


class ResultConfig(enum.StrEnum):
    RESULT = "RESULT"  # Return the result of the function
    FUTURE_PENDING = "FUTURE_PENDING"  # Return a Future object that isn't necessarily completed yet
    FUTURE_COMPLETED = "FUTURE_COMPLETED"  # Return a Future object that is completed


DEFAULT_RESULT_CONFIG = ResultConfig.RESULT


class ExceptionConfig(enum.StrEnum):
    RAISE_EAGERLY = "RAISE_EAGERLY"  # Raise exceptions immediately
    RAISE_GROUPED = "RAISE_GROUPED"  # Raise all exceptions that occurred in a exception group
    RETURN = "RETURN"  # Return exceptions as values inline in the result sequence instead of raising


DEFAULT_EXCEPTION_CONFIG: Literal[ExceptionConfig.RAISE_EAGERLY] = ExceptionConfig.RAISE_EAGERLY

# Orchestrators such as Ray anyway have problems displaying more than 10k tasks. This should hopefully be a generally
# usable default.
DEFAULT_MAX_PENDING_TASKS = 2500


class Executor(ABC):
    @classmethod
    @abstractmethod
    def from_legacy_executor(cls, legacy_executor: LegacyExecutor) -> Self: ...

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
    ) -> Future[R]: ...

    @overload
    def submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: ResultConfig = ...,
        retry_config: int | RetryConfig | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Future[R] | R: ...

    @abstractmethod
    def submit[**P, R](
        self,
        w: SingleWrap[P, R],
        *,
        result_config: ResultConfig = DEFAULT_RESULT_CONFIG,
        retry_config: int | RetryConfig | None = None,
        executor_kwargs: Mapping[type[Executor], Any] | None = None,
    ) -> Future[R] | R:
        """Submit a single task for execution.

        Executes the function wrapped in `w` and returns either the computed result
        directly or a `Future` representing the (possibly still-running) computation,
        depending on `result_config`.

        Args:
            w: A `SingleWrap` bundling the function to execute together with its
                positional and keyword arguments.  Created via `wrap(func, *args, **kwargs)`.
            result_config: Controls what is returned to the caller.
                * `ResultConfig.RESULT` (default) — blocks until the function completes
                  and returns the value of type `R` directly.
                * `ResultConfig.FUTURE_PENDING` — returns a `Future[R]` immediately.
                  The task may or may not have completed by the time the future is
                  returned. If `retry_config` is set, retries happen implicitly, such that
                  `future.result()` will include up to your specified number of retries.
                * `ResultConfig.FUTURE_COMPLETED` — returns a `Future[R]` that is
                  guaranteed to have completed successfully. If the task raised an
                  exception (even after retries), the exception is raised from this method
                  rather than deferred to the future.
            retry_config: Configures automatic retries for failed tasks.
                * `None` (default) — no retries; a failure raises immediately (or is
                  surfaced through the future, depending on `result_config`).
                * `int` — retry up to this many times on *any* exception.
                * `RetryConfig(retries=N, exceptions=[...])` — retry up to `N` times,
                  but only when the raised exception is an instance of one of the listed
                  exception types. Other exceptions propagate immediately.
                  Note that platform-sided exceptions (such as Ray workers dying) are always
                  part of the retried exceptions, even when you supply an empty list
                  in `RetryConfig.exceptions`.
            executor_kwargs: Optional, implementation-specific configuration keyed by
                executor type.  Each executor implementation looks up its own class in this
                mapping and interprets the value however it sees fit (e.g. resource
                requirements, scheduling options).  Ignored by implementations whose type
                is not present in the mapping.  `None` (default) passes no extra options.

        Returns:
            The function's return value `R` when `result_config` is `RESULT`, or a
            `Future[R]` when `result_config` is `FUTURE_PENDING` or
            `FUTURE_COMPLETED`.

        Raises:
            Exception: Any exception raised by the wrapped function, after exhausting
                retries (if configured). The exact delivery depends on `result_config`:
                for `RESULT` and `FUTURE_COMPLETED` the exception is raised from this
                method; for `FUTURE_PENDING` it is deferred to `future.result()` /
                `future.exception()`.
        """
        ...

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
    ) -> Sequence[Future[R]]: ...

    @overload
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: Literal[ResultConfig.FUTURE_PENDING, ResultConfig.FUTURE_COMPLETED],
        exception_config: Literal[ExceptionConfig.RETURN],
        retry_config: int | RetryConfig | None = ...,
        max_pending_tasks: int | None = ...,
        executor_kwargs: Mapping[type[Executor], Any] | None = ...,
    ) -> Sequence[Future[R] | Exception]: ...

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
    ) -> Sequence[Future[R]] | Sequence[R]: ...

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
    ) -> Sequence[R | Exception] | Sequence[Future[R] | Exception]: ...

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
    ) -> Sequence[Future[R]] | Sequence[R] | Sequence[R | Exception] | Sequence[Future[R] | Exception]: ...

    @abstractmethod
    def map[T, **P, R](
        self,
        w: MultipleWrap[T, P, R],
        *,
        result_config: ResultConfig = DEFAULT_RESULT_CONFIG,
        exception_config: ExceptionConfig = DEFAULT_EXCEPTION_CONFIG,
        retry_config: int | RetryConfig | None = None,
        max_pending_tasks: int | None = DEFAULT_MAX_PENDING_TASKS,
        executor_kwargs: Mapping[type[Executor], Any] | None = None,
    ) -> Sequence[Future[R]] | Sequence[R] | Sequence[R | Exception] | Sequence[Future[R] | Exception]:
        """Map a function over an iterable of first arguments, executing tasks in parallel.

        For each value `t` in `w.first_args`, calls `w.func(t, *w.args, **w.kwargs)`
        as a separate task.  Results are always returned in the same order as the input
        iterable, regardless of the order in which tasks complete.

        Submissions are throttled by `max_pending_tasks` to avoid overwhelming the
        executor's scheduler.  Results are retrieved inline as tasks complete, keeping
        memory pressure low — in `RESULT` mode, object references are released as soon
        as their value has been collected.

        Args:
            w: A `MultipleWrap` bundling the function, the iterable of first arguments
                to fan out over, and any additional shared positional/keyword arguments.
                Created via `mwrap(func, first_args, *args, **kwargs)`.
            result_config: Controls what is returned for each task.
                * `ResultConfig.RESULT` (default) — blocks until all tasks complete and
                  returns a `Sequence[R]` of computed values in input order.
                * `ResultConfig.FUTURE_PENDING` — returns a `Sequence[Future[R]]`
                  immediately after all tasks have been submitted.  Individual futures may
                  or may not have completed yet. If `retry_config` is set, retries happen
                  implicitly when `future.result()` is called.
                * `ResultConfig.FUTURE_COMPLETED` — returns a `Sequence[Future[R]]`
                  where every future is guaranteed to have completed successfully.  If any
                  task failed (even after retries), an exception is raised from this method
                  rather than deferred to the futures.
            exception_config: Controls how task exceptions are surfaced.  Only relevant
                when `result_config` is `RESULT` or `FUTURE_COMPLETED` (for
                `FUTURE_PENDING`, exceptions are always deferred to the individual
                futures).
                * `ExceptionConfig.RAISE_EAGERLY` (default) — raises the first exception
                  encountered, in approximate completion order. Remaining tasks may still
                  be running when the exception propagates. This gives the fastest
                  possible error detection.
                * `ExceptionConfig.RAISE_GROUPED` — waits for all tasks to finish, then
                  raises an `ExceptionGroup` containing every exception that occurred.
                  This is useful when the caller needs to inspect or handle all failures
                  rather than just the first one.
                * `ExceptionConfig.RETURN` — waits for all tasks to finish and returns a
                  sequence in which each failed task appears as an `Exception` instance
                  inline with the successful results (or their futures). Nothing is
                  raised from the method itself. Useful when the caller wants to decide
                  per-item how to handle failures.
            retry_config: Configures automatic retries for failed tasks.
                * `None` (default) — no retries; a failure is surfaced according to
                  `exception_config`.
                * `int` — retry each failed task up to this many times on *any* exception.
                * `RetryConfig(retries=N, exceptions=[...])` — retry up to `N` times,
                  but only when the raised exception is an instance of one of the listed
                  exception types. Other exceptions propagate immediately.
                  Note that platform-sided exceptions (such as Ray workers dying) are always
                  part of the retried exceptions, even when you supply an empty list
                  in `RetryConfig.exceptions`.
            max_pending_tasks: Maximum number of tasks that may be submitted but not yet
                completed at any point in time.  When this limit is reached, the method
                waits for some tasks to complete before submitting more.  This prevents
                overwhelming the executor's scheduler when mapping over very large
                iterables (tens or hundreds of thousands of items).
                * `int` (default `DEFAULT_MAX_PENDING_TASKS`) —
                  throttle to at most this many concurrent pending tasks.
                * `None` — no throttling; all tasks are submitted immediately and results
                  are collected afterwards.
            executor_kwargs: Optional, implementation-specific configuration keyed by
                executor type.  Each executor implementation looks up its own class in this
                mapping and interprets the value however it sees fit (e.g. resource
                requirements, scheduling options).  Ignored by implementations whose type
                is not present in the mapping.  `None` (default) passes no extra options.

        Returns:
            A `Sequence[R]` when `result_config` is `RESULT`, or a
            `Sequence[Future[R]]` when `result_config` is `FUTURE_PENDING` or
            `FUTURE_COMPLETED`.  In all cases the sequence preserves the order of the
            input iterable.
            When `exception_config` is`ExceptionConfig.RETURN`, the
            yielded type is additionally widened with `| Exception`
            so that failed tasks appear inline with successes.

        Raises:
            ExceptionGroup: When `exception_config` is `RAISE_GROUPED` and one or more
                tasks failed.  The group contains every exception that occurred.
            Exception: When `exception_config` is `RAISE_EAGERLY`, the first exception
                from a completed task is raised directly.
        """
        ...

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
    ) -> Iterator[Future[R]]: ...

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
    ) -> Iterator[Future[R] | Exception]: ...

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
    ) -> Iterator[Future[R]] | Iterator[R]: ...

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
    ) -> Iterator[R | Exception] | Iterator[Future[R] | Exception]: ...

    @abstractmethod
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
        """Lazily map a function over an iterable, yielding each result as it becomes available.

        Like `map`, but returns an `Iterator` instead of a `Sequence`. Results are yielded
        individually as tasks complete, rather than waiting for all tasks to finish.  This
        is useful for pipelines where downstream processing can begin before all tasks are
        done.

        Unlike `map`, `exception_config` does not support `RAISE_GROUPED` here — grouping
        exceptions would require draining all tasks, which defeats the purpose of a lazy
        iterator.

        Args:
            w: A `MultipleWrap` bundling the function, the iterable of first arguments
                to fan out over, and any additional shared positional/keyword arguments.
                Created via `mwrap(func, first_args, *args, **kwargs)`.
            result_config: Controls what is yielded for each task.
                * `ResultConfig.RESULT` (default) — yields computed values of type `R` as
                  tasks complete.
                * `ResultConfig.FUTURE_PENDING` — yields `Future[R]` objects immediately
                  after submission.  The futures may or may not have completed yet.
                * `ResultConfig.FUTURE_COMPLETED` — yields `Future[R]` objects that are
                  guaranteed to have completed successfully.  If a task raised an exception
                  (even after retries), the exception is raised from the iterator.
            exception_config: Controls how task exceptions are surfaced. Only relevant
                when `result_config` is `RESULT` or `FUTURE_COMPLETED` (for
                `FUTURE_PENDING`, exceptions are always deferred to the individual
                futures).
                * `ExceptionConfig.RAISE_EAGERLY` (default) — raises the first exception
                  encountered directly from the iterator.
                * `ExceptionConfig.RETURN` — yields each failed task as an `Exception`
                  instance inline with the successful results (or their futures) instead
                  of raising.
                * `ExceptionConfig.RAISE_GROUPED` — **not supported** for `map_lazy`;
                  passing it is a type error.
            retry_config: Configures automatic retries for failed tasks.
                * `None` (default) — no retries; a failure raises immediately from the
                  iterator (or is deferred to the future for `FUTURE_PENDING`).
                * `int` — retry each failed task up to this many times on *any* exception.
                * `RetryConfig(retries=N, exceptions=[...])` — retry up to `N` times,
                  but only when the raised exception is an instance of one of the listed
                  exception types. Other exceptions propagate immediately.
                  Note that platform-sided exceptions (such as Ray workers dying) are always
                  part of the retried exceptions, even when you supply an empty list
                  in `RetryConfig.exceptions`.
            ordered: Controls the order in which results are yielded.
                * `True` (default) — results are yielded in the same order as the input
                  iterable.  Internally, out-of-order completions are buffered until all
                  preceding results have been yielded.
                * `False` — results are yielded in completion order, i.e. whichever task
                  finishes first is yielded first.  This gives the lowest possible latency
                  per result.
            max_pending_tasks: Maximum number of tasks that may be submitted but not yet
                completed at any point in time.  When this limit is reached, the method
                waits for some tasks to complete before submitting more.  This prevents
                overwhelming the executor's scheduler when mapping over very large
                iterables (tens or hundreds of thousands of items).
                * `int` (default `DEFAULT_MAX_PENDING_TASKS`) —
                  throttle to at most this many concurrent pending tasks.
                * `None` — no throttling; all tasks are submitted immediately.
            executor_kwargs: Optional, implementation-specific configuration keyed by
                executor type.  Each executor implementation looks up its own class in this
                mapping and interprets the value however it sees fit (e.g. resource
                requirements, scheduling options).  Ignored by implementations whose type
                is not present in the mapping.  `None` (default) passes no extra options.

        Yields:
            `R` when `result_config` is `RESULT`, or `Future[R]` when `result_config` is
            `FUTURE_PENDING` or `FUTURE_COMPLETED`. When `exception_config` is
            `ExceptionConfig.RETURN`, the yielded type is additionally widened with
            `| Exception` so that failed tasks appear inline with successes.

        Raises:
            Exception: For `RESULT` and `FUTURE_COMPLETED` under
                `ExceptionConfig.RAISE_EAGERLY`, any task exception is raised directly
                from the iterator upon encountering the failed task's result.
        """
        ...
