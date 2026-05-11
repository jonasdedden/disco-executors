from __future__ import annotations

import pytest
from hypothesis import given

from .utils import (
    CUSTOM_ERROR_MSG,
    EXPECTED_EXCEPTION_FAIL_INPUT,
    INTS,
    CustomError,
    ExpectedExceptionInput,
    foo,
    foo_exc,
)
from disco.executors.revamp.local import LocalFuture, LocalFutureCancelledError, LocalFutureState


class TestLocalFuture:
    @given(bar=INTS, baz=INTS, biz=INTS)
    def test_local_future_no_error(self, bar: int, baz: int, biz: int) -> None:
        future = LocalFuture(foo, bar, baz, biz=biz)

        assert future.status is LocalFutureState.PENDING
        assert future.result() == bar + baz + biz

        # trick to reset assumptions mypy has about future instance
        # https://github.com/python/mypy/issues/9005
        future = future

        assert future.status is LocalFutureState.FINISHED
        assert future.exception() is None

    @given(bar=INTS, baz=INTS, biz=INTS)
    def test_local_future_cancel(self, bar: int, baz: int, biz: int) -> None:
        future = LocalFuture(foo, bar, baz, biz=biz)

        future.cancel()

        assert future.status is LocalFutureState.CANCELLED
        with pytest.raises(LocalFutureCancelledError):
            future.result()
        with pytest.raises(LocalFutureCancelledError):
            future.exception()

    @given(inp=EXPECTED_EXCEPTION_FAIL_INPUT)
    def test_local_future_error_raise(self, inp: ExpectedExceptionInput) -> None:
        future = LocalFuture(foo_exc, inp)

        with pytest.raises(CustomError, match=CUSTOM_ERROR_MSG):
            future.result()

    @given(inp=EXPECTED_EXCEPTION_FAIL_INPUT)
    def test_local_future_error_capture(self, inp: ExpectedExceptionInput) -> None:
        future = LocalFuture(foo_exc, inp)

        assert isinstance(future.exception(), CustomError)
        assert future.status is LocalFutureState.ERROR

        with pytest.raises(CustomError, match=CUSTOM_ERROR_MSG):
            future.result()
