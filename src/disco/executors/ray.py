from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import itertools
import logging
import os
import socket
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Concatenate, Self

import ray
from ray import ObjectRef
from ray.remote_function import RemoteFunction

from .base import Executor, Future, P, R, T
from .revamp.ray import RayExecutor as RevampRayExecutor

LOGGER = logging.getLogger(__name__)


def _wrap_with_exception_logging(func: Callable[P, R]) -> Callable[P, R]:
    """
    Wrapper that catches exceptions, logs their stack trace, and re-raises them.

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
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            log_lines = [f"Saw exception on call of {func.__name__} [args={args}, kwargs={kwargs}]"]

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
                log_lines.append(f"  Worker IP: {socket.gethostbyname(socket.gethostname())}")
            except Exception as e:
                log_lines.append(f"  Worker Hostname/IP: <error: {e!r}>")

            try:
                log_lines.append(f"  Worker PID: {os.getpid()}")
            except Exception as e:
                log_lines.append(f"  Worker PID: <error: {e!r}>")

            # Log all the execution environment information followed by the stack trace
            LOGGER.error("\n".join(log_lines), exc_info=exc)
            raise

    return wrapper


class RayFuture(Future[R]):
    def __init__(
        self,
        future: concurrent.futures.Future[R],
        func: ray.remote_function.RemoteFunction,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.future = future
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.retries = 0

    def cancel(self) -> bool | None:
        return self.future.cancel()

    def cancelled(self) -> bool:
        return self.future.cancelled()

    def done(self) -> bool:
        return self.future.done()

    def add_done_callback(self, fn: Callable[[concurrent.futures.Future[R]], Any]) -> None:  # type: ignore[override]
        return self.future.add_done_callback(fn)

    def result(self, timeout: float | None = None) -> R:
        return self.future.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self.future.exception(timeout=timeout)

    def retry(self) -> None:
        self.future = self.func.remote(*self.args, **self.kwargs).future()
        self.retries += 1

    @property
    def object_ref(self) -> ObjectRef[R]:
        return self.future.object_ref  # type: ignore[attr-defined, no-any-return]

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.future, attr)


class RayExecutor(Executor):
    def __init__(self, address: str | None = None) -> None:
        self.address = address

    def to_revamp_executor(self) -> RevampRayExecutor:
        return RevampRayExecutor()

    @classmethod
    @contextlib.contextmanager
    def worker_executor(cls, **kwargs: Any) -> Iterator[Self]:
        yield cls()

    def submit(self, func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        wrapped_func = _wrap_with_exception_logging(func)
        func_wrapper: RemoteFunction = ray.remote(wrapped_func).options(name=func.__name__)  # type: ignore[assignment]
        future = func_wrapper.remote(*args, **kwargs).future()
        return RayFuture(future, func_wrapper, *args, **kwargs)

    def _submit_from_remote(self, remote_func: RemoteFunction, /, *args: Any, **kwargs: Any) -> RayFuture[R]:
        future = remote_func.remote(*args, **kwargs).future()
        return RayFuture(future, remote_func, *args, **kwargs)

    def map(
        self,
        func: Callable[..., R],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> Iterator[R]:
        wrapped_func = _wrap_with_exception_logging(func)
        func_wrapper = ray.remote(wrapped_func).options(name=func.__name__)

        if self.is_all_collection(iterables):
            self.check_collections_same_length(iterables)

        yield from (  # type: ignore[var-annotated]
            future.result(timeout=timeout)
            for future in (self._submit_from_remote(func_wrapper, *iterable, **kwargs) for iterable in zip(*iterables))
        )

    def map_args(
        self,
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[Future[R]]:
        wrapped_func = _wrap_with_exception_logging(func)
        func_wrapper = ray.remote(wrapped_func).options(name=func.__name__)

        return [self._submit_from_remote(func_wrapper, arg, *add_args, **kwargs) for arg in args]  # type: ignore[arg-type]

    def get_workers(self) -> int:
        return ray.available_resources().get("CPU", 100)  # type: ignore[no-any-return]

    def smart_map_args(
        self,
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[R]:
        wrapped_func = _wrap_with_exception_logging(func)
        func_wrapper = ray.remote(wrapped_func).options(name=func.__name__)
        args_iter = iter(args)
        del args
        result_refs: list[ObjectRef[R]] = []
        chunk_size = self.get_workers()
        for i in itertools.count():
            if i % 10 == 0:
                chunk_size = self.get_workers()
            if len(result_refs) > 3 * chunk_size:
                # update result_refs to only
                # track the remaining tasks.
                ready_refs, result_refs = ray.wait(result_refs)
                yield from ray.get(ready_refs)
            try:
                next_arg = next(args_iter)
            except StopIteration:
                break
            result_refs.append(func_wrapper.remote(next_arg, *add_args, **kwargs))

        yield from ray.get(result_refs)

    def as_completed(
        self,
        futures: Iterable[RayFuture[R]],  # type: ignore[override]
        timeout: float | None = None,
        retries: int | None = None,
        **kwargs: Any,
    ) -> Iterator[RayFuture[R]]:
        unfinished = list(futures)
        del futures
        if not retries:
            while unfinished:
                finished, unfinished_objs_refs = ray.wait([future.object_ref for future in unfinished])
                unfinished = [object_ref.future() for object_ref in unfinished_objs_refs]

                yield from [object_ref.future() for object_ref in finished]
        else:
            obj_refs_to_futures: dict[ObjectRef[R], RayFuture[R]] = {future.object_ref: future for future in unfinished}
            del unfinished

            while obj_refs_to_futures:
                finished_obj_refs, _ = ray.wait([object_ref for object_ref in obj_refs_to_futures])

                for object_ref in finished_obj_refs:
                    future = obj_refs_to_futures[object_ref]
                    if exc := future.exception():
                        if future.retries >= retries:
                            raise exc
                        future.retry()
                        # After retry, future will have a different object_ref
                        obj_refs_to_futures.pop(object_ref)
                        obj_refs_to_futures[future.object_ref] = future
                        continue
                    obj_refs_to_futures.pop(object_ref)
                    yield future

    def reduce(
        self,
        func: Callable[Concatenate[Sequence[T], P], R],
        futures: Iterable[Sequence[Future[T]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[RayFuture[R]]:
        wrapped_func = _wrap_with_exception_logging(func)
        func_wrapper = ray.remote(wrapped_func).options(name=func.__name__)
        return [
            self._submit_from_remote(
                func_wrapper,  # type: ignore[arg-type]
                [fut.result() for fut in future_batch],
                *args,
                **kwargs,
            )
            for future_batch in futures
        ]

    def reduce_with_retries(
        self,
        func: Callable[Concatenate[Sequence[T], P], R],
        retries: int,
        futures: Iterable[Iterable[RayFuture[T]]],  # type: ignore[override]
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[RayFuture[R]]:
        wrapped_func = _wrap_with_exception_logging(func)
        func_wrapper = ray.remote(wrapped_func).options(name=func.__name__)
        yield from (
            self._submit_from_remote(
                func_wrapper,  # type: ignore[arg-type]
                [future.result() for future in self.as_completed(future_batch, retries=retries)],
                *args,
                **kwargs,
            )
            for future_batch in futures
        )
