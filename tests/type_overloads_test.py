"""Static type-overload validation via `typing.assert_type`.

The overload matrix for `Executor.submit` / `map` / `map_lazy` is large enough that a
regression in one of the return-type arms would be easy to miss. These assertions pin
each overload to the type documented in the README's "Return-type matrix" section; at
runtime they are no-ops (the bodies live under `if TYPE_CHECKING:`), so the real gate
is `mypy --strict`.

We drive the checks through the abstract `Executor` (holding a `LocalExecutor` instance),
which owns the public overloads; backends only implement the private hooks. No
`RayExecutor` / `LocalPoolExecutor` is instantiated.

Note on runtime: the bodies intentionally live inside `if TYPE_CHECKING:`. This way
`mypy` analyzes every `assert_type` call while pytest still collects the test functions
(they run as trivial no-ops). We avoid actually calling `_EX.map(...)(...)` etc., which is
important because some statically-legal combinations (e.g. `FUTURE_PENDING +
RAISE_GROUPED`) are rejected at runtime by `LocalExecutor`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

from disco.executors.base import ExceptionConfig, Executor, Future, ResultConfig
from disco.executors.local import LocalExecutor

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


# `_foo`'s return type `bytes` is deliberately disjoint from every parameter type
# (`int`, `str`, `float`) so the `assert_type` checks below are only satisfied by the
# correct overload — a mistaken overload that returned e.g. a parameter type would fail the
# check instead of silently aliasing.
def _foo(a: int, b: str, *, scale: float = 1.0) -> bytes:
    return f"{b}:{a * scale}".encode()


_EX: Executor = LocalExecutor()

# Runtime-chosen (non-literal) configs mirror what callers hold when the mode is only known
# dynamically — mypy cannot narrow these to a single literal and must dispatch to the widest
# matching overload.
_RC_DYNAMIC: ResultConfig = ResultConfig.RESULT
_EC_DYNAMIC: ExceptionConfig = ExceptionConfig.RAISE_EAGERLY


def test_submit_overloads() -> None:
    if TYPE_CHECKING:
        # Overload 1: result_config = Literal[RESULT]  ->  R
        assert_type(_EX.submit(_foo)(1, "hello", scale=2.0), bytes)
        assert_type(_EX.submit(_foo, result_config=ResultConfig.RESULT)(1, "hello", scale=2.0), bytes)

        # Overload 2: result_config = Literal[FUTURE_PENDING | FUTURE_COMPLETED]  ->  Future[R]
        assert_type(_EX.submit(_foo, result_config=ResultConfig.FUTURE_PENDING)(1, "hello", scale=2.0), Future[bytes])
        assert_type(_EX.submit(_foo, result_config=ResultConfig.FUTURE_COMPLETED)(1, "hello", scale=2.0), Future[bytes])

        # Overload 3: result_config = ResultConfig (dynamic)  ->  Future[R] | R
        assert_type(_EX.submit(_foo, result_config=_RC_DYNAMIC)(1, "hello", scale=2.0), Future[bytes] | bytes)


def test_map_overloads() -> None:
    if TYPE_CHECKING:
        # Overload 1: (RESULT) + (RAISE_EAGERLY | RAISE_GROUPED)  ->  Sequence[R]
        assert_type(_EX.map(_foo)([1, 2, 3], "hello", scale=2.0), Sequence[bytes])  # defaults: RESULT + RAISE_EAGERLY
        assert_type(
            _EX.map(_foo, exception_config=ExceptionConfig.RAISE_EAGERLY)([1, 2, 3], "hello", scale=2.0),
            Sequence[bytes],
        )
        assert_type(
            _EX.map(_foo, exception_config=ExceptionConfig.RAISE_GROUPED)([1, 2, 3], "hello", scale=2.0),
            Sequence[bytes],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.RESULT, exception_config=ExceptionConfig.RAISE_EAGERLY)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[bytes],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.RESULT, exception_config=ExceptionConfig.RAISE_GROUPED)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[bytes],
        )

        # Overload 2: (RESULT) + (RETURN)  ->  Sequence[R | Exception]
        assert_type(
            _EX.map(_foo, exception_config=ExceptionConfig.RETURN)([1, 2, 3], "hello", scale=2.0),
            Sequence[bytes | Exception],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.RESULT, exception_config=ExceptionConfig.RETURN)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[bytes | Exception],
        )

        # Overload 3: (FUTURE_PENDING | FUTURE_COMPLETED) + (RAISE_EAGERLY | RAISE_GROUPED)  ->  Sequence[Future[R]]
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.FUTURE_PENDING)([1, 2, 3], "hello", scale=2.0),
            Sequence[Future[bytes]],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.FUTURE_COMPLETED)([1, 2, 3], "hello", scale=2.0),
            Sequence[Future[bytes]],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.FUTURE_PENDING, exception_config=ExceptionConfig.RAISE_EAGERLY)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[Future[bytes]],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.FUTURE_PENDING, exception_config=ExceptionConfig.RAISE_GROUPED)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[Future[bytes]],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.FUTURE_COMPLETED, exception_config=ExceptionConfig.RAISE_EAGERLY)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[Future[bytes]],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.FUTURE_COMPLETED, exception_config=ExceptionConfig.RAISE_GROUPED)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[Future[bytes]],
        )

        # Overload 4: (FUTURE_PENDING | FUTURE_COMPLETED) + (RETURN)  ->  Sequence[Future[R] | Exception]
        # Both arguments are required (no defaults) for this overload.
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.FUTURE_PENDING, exception_config=ExceptionConfig.RETURN)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[Future[bytes] | Exception],
        )
        assert_type(
            _EX.map(_foo, result_config=ResultConfig.FUTURE_COMPLETED, exception_config=ExceptionConfig.RETURN)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[Future[bytes] | Exception],
        )

        # Overload 5: (ResultConfig dynamic) + (RAISE_EAGERLY | RAISE_GROUPED)  ->  Sequence[Future[R]] | Sequence[R]
        assert_type(
            _EX.map(_foo, result_config=_RC_DYNAMIC)([1, 2, 3], "hello", scale=2.0),
            Sequence[Future[bytes]] | Sequence[bytes],
        )
        assert_type(
            _EX.map(_foo, result_config=_RC_DYNAMIC, exception_config=ExceptionConfig.RAISE_EAGERLY)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[Future[bytes]] | Sequence[bytes],
        )
        assert_type(
            _EX.map(_foo, result_config=_RC_DYNAMIC, exception_config=ExceptionConfig.RAISE_GROUPED)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[Future[bytes]] | Sequence[bytes],
        )

        # Overload 6: (ResultConfig dynamic) + (RETURN)  ->  Sequence[R | Exception] | Sequence[Future[R] | Exception]
        assert_type(
            _EX.map(_foo, result_config=_RC_DYNAMIC, exception_config=ExceptionConfig.RETURN)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Sequence[bytes | Exception] | Sequence[Future[bytes] | Exception],
        )

        # Overload 7: (ResultConfig dynamic) + (ExceptionConfig dynamic)  ->  full 4-way union
        assert_type(
            _EX.map(_foo, result_config=_RC_DYNAMIC, exception_config=_EC_DYNAMIC)([1, 2, 3], "hello", scale=2.0),
            Sequence[Future[bytes]]
            | Sequence[bytes]
            | Sequence[bytes | Exception]
            | Sequence[Future[bytes] | Exception],
        )


def test_map_lazy_overloads() -> None:
    # `map_lazy` restricts `exception_config` to `Literal[RAISE_EAGERLY, RETURN]` — `RAISE_GROUPED`
    # is a type error here — so there are only six overloads (vs. `map`'s seven).
    if TYPE_CHECKING:
        # Overload 1: (RESULT) + (RAISE_EAGERLY)  ->  Iterator[R]
        assert_type(
            _EX.map_lazy(_foo)([1, 2, 3], "hello", scale=2.0), Iterator[bytes]
        )  # defaults: RESULT + RAISE_EAGERLY
        assert_type(
            _EX.map_lazy(_foo, exception_config=ExceptionConfig.RAISE_EAGERLY)([1, 2, 3], "hello", scale=2.0),
            Iterator[bytes],
        )
        assert_type(
            _EX.map_lazy(_foo, result_config=ResultConfig.RESULT)([1, 2, 3], "hello", scale=2.0), Iterator[bytes]
        )
        assert_type(
            _EX.map_lazy(_foo, result_config=ResultConfig.RESULT, exception_config=ExceptionConfig.RAISE_EAGERLY)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Iterator[bytes],
        )

        # Overload 2: (RESULT) + (RETURN)  ->  Iterator[R | Exception]
        assert_type(
            _EX.map_lazy(_foo, exception_config=ExceptionConfig.RETURN)([1, 2, 3], "hello", scale=2.0),
            Iterator[bytes | Exception],
        )
        assert_type(
            _EX.map_lazy(_foo, result_config=ResultConfig.RESULT, exception_config=ExceptionConfig.RETURN)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Iterator[bytes | Exception],
        )

        # Overload 3: (FUTURE_PENDING | FUTURE_COMPLETED) + (RAISE_EAGERLY)  ->  Iterator[Future[R]]
        assert_type(
            _EX.map_lazy(_foo, result_config=ResultConfig.FUTURE_PENDING)([1, 2, 3], "hello", scale=2.0),
            Iterator[Future[bytes]],
        )
        assert_type(
            _EX.map_lazy(_foo, result_config=ResultConfig.FUTURE_COMPLETED)([1, 2, 3], "hello", scale=2.0),
            Iterator[Future[bytes]],
        )
        assert_type(
            _EX.map_lazy(
                _foo, result_config=ResultConfig.FUTURE_PENDING, exception_config=ExceptionConfig.RAISE_EAGERLY
            )([1, 2, 3], "hello", scale=2.0),
            Iterator[Future[bytes]],
        )
        assert_type(
            _EX.map_lazy(
                _foo, result_config=ResultConfig.FUTURE_COMPLETED, exception_config=ExceptionConfig.RAISE_EAGERLY
            )([1, 2, 3], "hello", scale=2.0),
            Iterator[Future[bytes]],
        )

        # Overload 4: (FUTURE_PENDING | FUTURE_COMPLETED) + (RETURN)  ->  Iterator[Future[R] | Exception]
        assert_type(
            _EX.map_lazy(_foo, result_config=ResultConfig.FUTURE_PENDING, exception_config=ExceptionConfig.RETURN)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Iterator[Future[bytes] | Exception],
        )
        assert_type(
            _EX.map_lazy(_foo, result_config=ResultConfig.FUTURE_COMPLETED, exception_config=ExceptionConfig.RETURN)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Iterator[Future[bytes] | Exception],
        )

        # Overload 5: (ResultConfig dynamic) + (RAISE_EAGERLY)  ->  Iterator[Future[R]] | Iterator[R]
        assert_type(
            _EX.map_lazy(_foo, result_config=_RC_DYNAMIC)([1, 2, 3], "hello", scale=2.0),
            Iterator[Future[bytes]] | Iterator[bytes],
        )
        assert_type(
            _EX.map_lazy(_foo, result_config=_RC_DYNAMIC, exception_config=ExceptionConfig.RAISE_EAGERLY)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Iterator[Future[bytes]] | Iterator[bytes],
        )

        # Overload 6: (ResultConfig dynamic) + (RETURN)  ->  Iterator[R | Exception] | Iterator[Future[R] | Exception]
        assert_type(
            _EX.map_lazy(_foo, result_config=_RC_DYNAMIC, exception_config=ExceptionConfig.RETURN)(
                [1, 2, 3], "hello", scale=2.0
            ),
            Iterator[bytes | Exception] | Iterator[Future[bytes] | Exception],
        )


def _clashing(x: int, *, result_config: str = "mine", flag: bool = False) -> bytes:
    """A function with its own keyword argument named like one of the executor options."""
    return f"{x}:{result_config}:{flag}".encode()


def test_call_signature() -> None:
    # Executor options go to `submit` / `map` / `map_lazy`, the function's own arguments to the returned callable, so
    # equally named keywords never clash and both are typed. `mypy --strict` enables `warn_unused_ignores`, so every
    # `type: ignore[...]` below also asserts that mypy rejects that call.
    if TYPE_CHECKING:
        assert_type(
            _EX.submit(_clashing, result_config=ResultConfig.FUTURE_PENDING)(1, result_config="theirs", flag=True),
            Future[bytes],
        )
        assert_type(_EX.map(_clashing)([1, 2], result_config="theirs", flag=True), Sequence[bytes])
        assert_type(_EX.map_lazy(_clashing)(iter([1, 2]), flag=True), Iterator[bytes])

        _EX.submit(_clashing)("1")  # type: ignore[arg-type]
        _EX.submit(_clashing)(1, result_config=5, flag=True)  # type: ignore[arg-type]
        _EX.submit(_clashing, result_config="RESULT")  # type: ignore[call-overload]
        _EX.map(_foo)([1], "hello", scale="2")  # type: ignore[arg-type]
        _EX.map(_foo)(["1"], "hello")  # type: ignore[list-item]
        _EX.map(_foo)([1])  # type: ignore[call-arg]
        _EX.map_lazy(_foo, exception_config=ExceptionConfig.RAISE_GROUPED)  # type: ignore[call-overload]
