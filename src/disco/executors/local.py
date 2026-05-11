from __future__ import annotations

import contextlib
import enum
import functools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Concatenate, Self, cast

from .base import Executor, Future, P, R, T
from .revamp.local import LocalExecutor as RevampLocalExecutor


class LocalFutureState(enum.StrEnum):
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class LocalFutureCancelledError(RuntimeError):
    pass


class LocalFuture(Future[R]):
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result: R | None = None
        self._exception: BaseException | None = None
        self._status: LocalFutureState = LocalFutureState.PENDING
        self._done_callback: list[Callable[[Future[R]], Any]] = []

    @property
    def status(self) -> LocalFutureState:
        return self._status

    def cancel(self) -> bool | None:
        if self._status in (LocalFutureState.PENDING, LocalFutureState.CANCELLED):
            self._status = LocalFutureState.CANCELLED
            return True
        else:
            return False

    def cancelled(self) -> bool:
        return self._status == LocalFutureState.CANCELLED

    def done(self) -> bool:
        return self._status in (LocalFutureState.FINISHED, LocalFutureState.ERROR, LocalFutureState.CANCELLED)

    def add_done_callback(self, fn: Callable[[Future[R]], Any]) -> None:
        self._done_callback.append(fn)
        return

    def _execute(self) -> None:
        self._result = self.func(*self.args, **self.kwargs)
        self._status = LocalFutureState.FINISHED
        for callback in self._done_callback:
            callback(self)

    def result(self, timeout: float | None = None) -> R:
        if self._status == LocalFutureState.CANCELLED:
            raise LocalFutureCancelledError("Future already cancelled!")

        if self._status == LocalFutureState.ERROR:
            assert isinstance(self._exception, BaseException)
            raise self._exception

        if self._status == LocalFutureState.FINISHED:
            # Can not 'assert self._result' here since the result could be None after all
            return cast(R, self._result)

        self._execute()

        return cast(R, self._result)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if self._status == LocalFutureState.CANCELLED:
            raise LocalFutureCancelledError("Future already cancelled!")

        if self._status == LocalFutureState.FINISHED:
            return None

        if self._status == LocalFutureState.ERROR:
            return self._exception

        try:
            self._execute()
        except BaseException as e:
            self._status = LocalFutureState.ERROR
            self._exception = e

        return self._exception

    def retry(self) -> None:
        if self._status == LocalFutureState.FINISHED:
            return

        try:
            self._execute()
        except BaseException as e:
            self._status = LocalFutureState.ERROR
            self._exception = e

    def __repr__(self) -> str:
        # Stolen from concurrent.futures._base.Future
        if self.status is LocalFutureState.ERROR:
            return f"<{self.__class__.__name__} at {id(self):#x} state={self.status} raised {self._exception.__class__.__name__}>"
        elif self.status is LocalFutureState.FINISHED:
            return f"<{self.__class__.__name__} at {id(self):#x} state={self.status} returned {self._result.__class__.__name__}>"
        else:
            return f"<{self.__class__.__name__} at {id(self):#x} state={self.status}>"


class LocalExecutor(Executor):
    exec_args: dict[str, Any]

    def __init__(self, exec_args: dict[str, Any] | None = None) -> None:
        self.exec_args = exec_args or {}

    def to_revamp_executor(self) -> RevampLocalExecutor:
        return RevampLocalExecutor()

    @staticmethod
    def submit(func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> LocalFuture[R]:
        return LocalFuture(func, *args, **kwargs)

    @staticmethod
    def map(
        func: Callable[..., R],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> Iterator[R]:
        # If iterables are collections, check if they have the same size
        # Otherwise, behave like builtin 'map', e.g. stop after first iterable is exhausted
        if LocalExecutor.is_all_collection(iterables):
            LocalExecutor.check_collections_same_length(iterables)
        func = functools.partial(func, **kwargs)
        return map(func, *iterables)

    @staticmethod
    def map_args(
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[LocalFuture[R]]:
        return [LocalFuture(func, k, *add_args, **kwargs) for k in args]

    @staticmethod
    def smart_map_args(
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[R]:
        yield from (func(arg, *add_args, **kwargs) for arg in args)

    @classmethod
    def as_completed(
        cls, futures: Iterable[Future[R]], timeout: float | None = None, retries: int | None = None, **kwargs: Any
    ) -> Iterator[Future[R]]:
        if retries:
            yield from cls._lazy_iterate_with_retries(futures, retries)
        else:
            yield from futures

    @classmethod
    @contextlib.contextmanager
    def worker_executor(cls, **kwargs: Any) -> Iterator[Self]:
        yield cls()

    def reduce(
        self,
        func: Callable[Concatenate[Sequence[T], P], R],
        futures: Iterable[Sequence[Future[T]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[LocalFuture[R]]:
        return [
            LocalFuture(func, [future.result() for future in future_batch], *args, **kwargs) for future_batch in futures
        ]

    def reduce_with_retries(
        self,
        func: Callable[Concatenate[Sequence[T], P], R],
        retries: int,
        futures: Iterable[Iterable[Future[T]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[LocalFuture[R]]:
        yield from [
            LocalFuture(
                func, [future.result() for future in self.as_completed(future_batch, retries=retries)], *args, **kwargs
            )
            for future_batch in futures
        ]
