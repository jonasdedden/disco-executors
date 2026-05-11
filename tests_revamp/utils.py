from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Callable


class CustomError(Exception):
    pass


CUSTOM_ERROR_MSG = "Throwing tantrum!"


class ExpectedExceptionInput(NamedTuple):
    num: int
    should_fail: bool


INTS = st.integers(min_value=0, max_value=100)
INT_LISTS = st.lists(INTS, min_size=1, max_size=100)
INT_LISTS_SMALL = st.lists(INTS, min_size=1, max_size=10)

EXPECTED_EXCEPTION_INPUTS = st.builds(
    ExpectedExceptionInput,
    num=INTS,
    should_fail=st.booleans(),
)
EXPECTED_EXCEPTION_FAIL_INPUT = st.builds(
    ExpectedExceptionInput,
    num=INTS,
    should_fail=st.just(True),
)
EXPECTED_EXCEPTION_INPUTS_LISTS = st.lists(
    EXPECTED_EXCEPTION_INPUTS,
    min_size=1,
    max_size=100,
).filter(lambda xs: any(x.should_fail for x in xs))


def foo_exc(inp: ExpectedExceptionInput) -> int:
    if inp.should_fail:
        raise CustomError(CUSTOM_ERROR_MSG + f" ({inp.num})")

    return inp.num


def foo(bar: int, baz: int, biz: int = 5) -> int:
    return bar + baz + biz


class FailNTimesInput(NamedTuple):
    counter_callable: Callable[..., int]
    num: int


def foo_fail_n_times(inp: FailNTimesInput, required_executions: int) -> int:
    """This is a function that will fail `required_executions - 1` times before it returns the expected result"""
    counter_value = inp.counter_callable()

    if counter_value != required_executions:
        raise CustomError(CUSTOM_ERROR_MSG + f" ({counter_value}/{required_executions}, num={inp.num})")

    return inp.num
