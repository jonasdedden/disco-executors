from __future__ import annotations

import collections
import contextlib
import enum
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import Any, Concatenate, Literal, Self, TypeGuard, TypeVar, overload

from typing_extensions import ParamSpec

from .revamp import Executor as RevampExecutor

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


class Future[R](ABC):
    @abstractmethod
    def cancel(self) -> bool | None: ...

    @abstractmethod
    def cancelled(self) -> bool: ...

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def add_done_callback(self, fn: Callable[[Future[R]], Any]) -> None: ...

    @abstractmethod
    def result(self, timeout: float | None = None) -> R: ...

    @abstractmethod
    def exception(self, timeout: float | None = None) -> BaseException | None: ...

    @abstractmethod
    def retry(self) -> None: ...


class ErrorRaiseMode(enum.StrEnum):
    RAISE = "raise"
    SKIP = "skip"


class Executor(ABC):
    @abstractmethod
    def to_revamp_executor(self) -> RevampExecutor: ...

    @abstractmethod
    def submit(self, func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Future[R]: ...

    def exec(self, func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.submit(func, *args, **kwargs).result()

    @abstractmethod
    def map(
        self,
        func: Callable[..., R],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> Iterator[R]: ...

    def exec_map(
        self,
        func: Callable[..., R],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> Sequence[R]:
        return list(self.map(func, *iterables, timeout=timeout, chunksize=chunksize, **kwargs))

    @abstractmethod
    def map_args(
        self,
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[Future[R]]: ...

    def exec_map_args(
        self,
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[R]:
        return [future.result() for future in self.map_args(func, args, *add_args, **kwargs)]

    @abstractmethod
    def smart_map_args(
        self,
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[R]: ...

    @abstractmethod
    def as_completed(
        self, futures: Iterable[Future[R]], timeout: float | None = None, retries: int | None = None, **kwargs: Any
    ) -> Iterator[Future[R]]: ...

    @classmethod
    @abstractmethod
    @contextlib.contextmanager
    def worker_executor(cls, **kwargs: Any) -> Iterator[Self]: ...

    @staticmethod
    def _strict_iterate_with_retries(futures: Iterable[Future[R]], retries: int) -> Iterator[Future[R]]:
        for future in futures:
            tries = 0
            while exc := future.exception():
                if tries >= retries:
                    raise exc
                future.retry()
                tries += 1
            yield future

    @staticmethod
    def _lazy_iterate_with_retries(futures: Iterable[Future[R]], retries: int) -> Iterator[Future[R]]:
        # This function is as lazy as possible; uncertain whether this is really needed
        retries_counter: dict[Future[R], int] = collections.Counter()
        retried_futures: list[Future[R]] = []

        for future in itertools.chain(futures, retried_futures):
            if exc := future.exception():
                if retries_counter[future] >= retries:
                    raise exc
                future.retry()
                retries_counter[future] += 1
                retried_futures.append(future)
                continue
            yield future

    @overload
    def gather(
        self,
        futures: Iterable[Future[R]],
        *,
        errors: Literal[ErrorRaiseMode.SKIP],
        retries: int | None,
    ) -> Sequence[R | None]: ...

    @overload
    def gather(
        self,
        futures: Iterable[Future[R]],
        *,
        errors: Literal[ErrorRaiseMode.RAISE],
        retries: int | None,
    ) -> Sequence[R]: ...

    @overload
    def gather(self, futures: Iterable[Future[R]], *, errors: ErrorRaiseMode) -> list[R | None] | Sequence[R]: ...

    @overload
    def gather(self, futures: Iterable[Future[R]], *, retries: int | None) -> Sequence[R]: ...

    @overload
    def gather(self, futures: Iterable[Future[R]]) -> Sequence[R]: ...

    def gather(
        self,
        futures: Iterable[Future[R]],
        *,
        errors: ErrorRaiseMode = ErrorRaiseMode.RAISE,
        retries: int | None = None,
    ) -> Sequence[R | None] | Sequence[R]:
        if errors is ErrorRaiseMode.RAISE:
            if not retries:
                return [future.result() for future in futures]
            return [future.result() for future in self._strict_iterate_with_retries(futures, retries)]
        else:
            return [future.result() if not future.exception() else None for future in futures]

    @abstractmethod
    def reduce(
        self,
        func: Callable[Concatenate[Sequence[T], P], R],
        futures: Iterable[Sequence[Future[T]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[Future[R]]: ...

    @abstractmethod
    def reduce_with_retries(
        self,
        func: Callable[Concatenate[Sequence[T], P], R],
        retries: int,
        futures: Iterable[Iterable[Future[T]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[Future[R]]: ...

    @staticmethod
    def is_all_collection(inp: Sequence[Iterable[T]]) -> TypeGuard[Sequence[Collection[T]]]:
        return all(isinstance(item, Collection) for item in inp)

    @staticmethod
    def check_collections_same_length(items: Sequence[Collection[T]]) -> None:
        collection_length = len(items[0])
        if not all(len(collection) == collection_length for collection in items):
            raise ValueError(
                f"Make sure that all iterables have the same length! Expected was length '{collection_length}'."
            )
