from __future__ import annotations

import collections
import contextlib
import functools
import logging
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from itertools import chain, count, islice, repeat
from typing import Any, Concatenate, Self, cast

import dask.base
import dask.distributed
import dask.utils
import distributed.client

from .base import Executor, Future, P, R, T
from .revamp import Executor as RevampExecutor

LOGGER = logging.getLogger(__name__)
_MAP_WARNING_MESSAGE = (
    "`.map` was called with an Iterable (instead of a Collection), which will lead to falling back "
    "to submitting each task individually instead of calling Dask's `.map` function. "
    "This can be slow and can lead to scheduler overloads or other problems."
)


def check_future_for_cancellation(future: dask.distributed.Future[R]) -> None:
    if future.cancelled():  # type: ignore[no-untyped-call]
        # This should raise
        future.result()
        # Raising explicitly if previous statement didn't raise
        assert isinstance(future.key, str)
        raise distributed.client.FutureCancelledError(future.key, reason=None)


class DaskFuture(Future[R], dask.distributed.Future[R]):  # type: ignore[misc]
    ...


class DaskExecutor(Executor):
    def __init__(self, client: dask.distributed.Client):
        self.client = client

    def to_revamp_executor(self) -> RevampExecutor:
        raise NotImplementedError()

    @staticmethod
    def _generate_key(func: Callable[..., Any], /, base_hash: str, arg: Any) -> str:
        return (
            f"{dask.utils.funcname(func)[:50]}-{base_hash[:8]}-"
            f"{DaskExecutor.format_value(arg)}-{dask.base.tokenize(arg)[:8]}"
        )

    @staticmethod
    def format_value(_value: Any, max_value_length: int = 150) -> str:
        full_repr = repr(_value)
        if max_value_length and len(full_repr) > max_value_length:
            return f"{full_repr[: max_value_length // 2]}[...]{full_repr[-max_value_length // 2 :]}"
        return full_repr

    @staticmethod
    def base_tokenize(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        return dask.base.tokenize(func, *args, **kwargs)

    def submit(self, func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> DaskFuture[R]:
        # Similar to .map_args, expect the first arg to be "primary" which explicitly shall be part of the Future key
        # All others only occur in the form of a hash
        base_hash = self.base_tokenize(func, *args[1:], **kwargs)
        func = functools.partial(func, **kwargs)
        future = self.client.submit(
            func,
            *args,
            key=self._generate_key(func, base_hash, args[0] if args else None),
        )
        return cast(DaskFuture[R], future)

    def map(
        self,
        func: Callable[..., R],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int | None = None,
        **kwargs: Any,
    ) -> Iterator[R]:
        # Not submitting with custom keys here since there is no "primary" argument, all are equally iterable
        func = functools.partial(func, **kwargs)

        if self.is_all_collection(iterables):
            self.check_collections_same_length(iterables)
            futures = self.client.map(func, *iterables, batch_size=chunksize)
        else:
            # This branch is slow, logging a warning message
            LOGGER.warning(_MAP_WARNING_MESSAGE)
            futures = [self.client.submit(func, *iterable) for iterable in zip(*iterables)]
        yield from (future.result() for future in futures)

    def map_args(
        self,
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[DaskFuture[R]]:
        # 'add_args' & 'kwargs' are constant across all submitted futures, calculate single hash for that first
        # 'repr(args)' is explicitly used for the individual future keys
        base_hash = self.base_tokenize(func, *add_args, **kwargs)
        func = functools.partial(func, **kwargs)
        if isinstance(args, Collection):
            # 'args' implements __len__() -> able to give to Dask directly
            args_len = len(args)
            return self.client.map(  # type: ignore[return-value]
                func,
                args,
                *[list(repeat(arg, args_len)) for arg in add_args],
                key=[self._generate_key(func, base_hash, arg) for arg in args],
            )
        elif isinstance(args, Iterable):
            # This branch is slow, logging a warning message
            LOGGER.warning(_MAP_WARNING_MESSAGE)
            # 'args' does not implement __len__() -> it is a pure Iterable or Iterator
            # This means Dask does not support it natively, and it's suggested using .submit() in a for loop
            # Yielding from a list comprehension here to consume args as quickly as possible
            return [
                cast(
                    DaskFuture[R],
                    self.client.submit(
                        func,
                        arg,
                        *add_args,
                        key=self._generate_key(func, base_hash, arg),
                    ),
                )
                for arg in args
            ]
        else:
            raise TypeError(f"Args ({args}) does neither seem to be a Collection nor a Iterable.")

    def get_workers(self) -> int:
        return sum(self.client.nthreads().values())  # type: ignore[no-untyped-call]

    def smart_map_args(
        self,
        func: Callable[Concatenate[T, P], R],
        args: Iterable[T],
        *add_args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[R]:
        base_hash = self.base_tokenize(func, *add_args, **kwargs)
        func = functools.partial(func, **kwargs)

        as_completed = dask.distributed.as_completed()  # type: ignore[no-untyped-call]

        args_iter = iter(args)
        del args

        chunk_size = self.get_workers()
        for i in count():
            if i % 10 == 0:
                chunk_size = self.get_workers()
            while as_completed.count() > 3 * chunk_size:  # type: ignore[no-untyped-call]
                yield from (fut.result() for fut in as_completed.next_batch(block=True))  # type: ignore[no-untyped-call]
            chunk = tuple(islice(args_iter, chunk_size))
            if not chunk:
                break
            as_completed.update(  # type: ignore[no-untyped-call]
                self.client.map(
                    func,
                    chunk,
                    *[list(repeat(arg, chunk_size)) for arg in add_args],
                    key=[self._generate_key(func, base_hash, arg) for arg in chunk],
                )
            )
        for batch in as_completed.batches():  # type: ignore[no-untyped-call]
            yield from (fut.result() for fut in batch)

    def as_completed(
        self,
        futures: Iterable[Future[R]],
        timeout: float | None = None,
        retries: int | None = None,
        **kwargs: Any,
    ) -> Iterator[Future[R]]:
        """
        This function basically wraps 'dask.distributed.as_completed', but does some additional logic ontop.

        - Since we potentially see some problems with Dask's `as_completed` "forgetting" futures, we track them with
          some additional functionality, to make sure that all futures are eventually yielded.
        - We also provide functionality to retry futures when they have failed.
        """
        as_completed = dask.distributed.as_completed(list(futures), timeout=timeout, **kwargs)  # type: ignore[no-untyped-call]

        class FutureTracker(set[Future[R]]):
            def remove_future(self, fut: Future[R]) -> None:
                try:
                    self.remove(fut)
                except KeyError:
                    LOGGER.warning(f"Tried to remove future '{fut}', which apparently was yielded already.")

            def remaining_futures(self) -> set[Future[R]]:
                if self:
                    LOGGER.error(
                        "There were futures that 'dask.distributed.as_completed' for some reason didn't track anymore "
                        "and would have been lost if we didn't additionally track them ourselves. "
                        f"Manually & sequentially gathering futures: {self}"
                    )
                return self

        # additionally track futures in our own container to ensure `dask.distributed.as_completed` didn't "forget" any
        tracked_futures = FutureTracker(futures)

        # to allow Dask to release futures as early as possible, remove unneeded reference to futures which are
        # now handled by both 'dask.distributed.as_completed' and our FutureTracker
        del futures

        if not retries:
            for future in as_completed:
                check_future_for_cancellation(future)
                yield future
                tracked_futures.remove_future(future)

            yield from tracked_futures.remaining_futures()
            return

        retries_counter: dict[DaskFuture[R], int] = collections.Counter()

        for future in chain.from_iterable(as_completed.batches()):  # type: ignore[no-untyped-call]
            if exc := future.exception():
                if retries_counter[future] >= retries:
                    raise exc
                future.retry()
                retries_counter[future] += 1
                as_completed.add(future)  # type: ignore[no-untyped-call]
                continue
            check_future_for_cancellation(future)
            yield future
            tracked_futures.remove_future(future)

        yield from self._lazy_iterate_with_retries(iter(tracked_futures.remaining_futures()), retries=retries)

    @classmethod
    @contextlib.contextmanager
    def worker_executor(cls, **kwargs: Any) -> Iterator[Self]:
        with dask.distributed.worker_client(**kwargs) as client:
            yield cls(client=client)

    def reduce(
        self,
        func: Callable[Concatenate[Sequence[T], P], R],
        futures: Iterable[Sequence[Future[T]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[DaskFuture[R]]:
        return [self.client.submit(func, future_batch, *args, **kwargs) for future_batch in futures]  # type: ignore[misc]

    def reduce_with_retries(
        self,
        func: Callable[Concatenate[Sequence[T], P], R],
        retries: int,
        futures: Iterable[Iterable[Future[T]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[DaskFuture[R]]:
        yield from (  # type: ignore[misc]
            self.client.submit(
                func,
                list(self.as_completed(future_batch, retries=retries)),
                *args,
                **kwargs,
            )
            for future_batch in futures
        )
