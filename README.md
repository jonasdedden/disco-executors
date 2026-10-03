# disco-executors

A small, typed façade over pluggable task-execution backends (local, Ray) with
explicit control over how results, exceptions, retries and back-pressure are surfaced.

---

## Quick start

```python
from disco.executors import ResultConfig
from disco.executors.local import LocalExecutor


def add(a: int, b: int, *, scale: int = 1) -> int:
    return (a + b) * scale


executor = LocalExecutor()

# Single task — executor options first, then call with the function's own arguments
total = executor.submit(add)(2, 3, scale=10)
assert total == 50

# Parallel map — the first argument fans out, remaining positional/keyword args are shared
sums = executor.map(add)([1, 2, 3], 10, scale=2)
assert sums == [22, 24, 26]

# Lazy map — yields as tasks complete
for s in executor.map_lazy(add, ordered=False)(range(1_000_000), 10, scale=2):
    process(s)
```

---

## Core concepts

### `Executor`

`Executor` is the abstract interface implemented by each backend. Each of its three
entry points takes the function plus the executor's options, and returns a callable
that takes the function's own arguments:

| method                      | the returned callable takes           | and returns                                                        |
|-----------------------------|---------------------------------------|--------------------------------------------------------------------|
| `submit(func, **options)`   | `func`'s arguments                    | one result / future                                                |
| `map(func, **options)`      | an iterable + `func`'s remaining args | `Sequence` of results / futures (fully materialized before return) |
| `map_lazy(func, **options)` | an iterable + `func`'s remaining args | `Iterator` of results / futures (yields as tasks complete)         |

### Two calls: options, then arguments

```python
executor.submit(func, result_config=..., retry_config=...)(*args, **kwargs)
executor.map(func, result_config=..., max_pending_tasks=...)(first_args, *args, **kwargs)
```

Splitting the call keeps both sides fully typed. A function's `*args, **kwargs`
can't share a signature with further keyword options, so the executor options
(`result_config`, `exception_config`, `retry_config`, …) go to the first call and the
function's arguments to the second. The type checker checks the second call against
`func`'s real signature, keyword-only parameters included. If `func` has a parameter
with the same name as an executor option, both are passed separately and typed
separately:

```python
def g(x: int, *, result_config: str = "mine") -> float: ...

executor.submit(g, result_config=ResultConfig.FUTURE_PENDING)(1, result_config="theirs")  # -> Future[float]
```

For `map` / `map_lazy`, the first argument of the returned callable is the iterable
that fans out one task per element (into `func`'s first parameter); all other
arguments are shared across tasks.

Nothing runs until the returned callable is called, and it can be reused:

```python
square_all = executor.map(square, retry_config=3)
first = square_all(batch_1)
second = square_all(batch_2)
```

---

## `ResultConfig` — what gets returned

`ResultConfig` controls whether the caller receives the computed value directly
or a `Future` wrapping it.

| `ResultConfig`       | you receive   | blocks on completion? |
|----------------------|---------------|-----------------------|
| `RESULT` *(default)* | the value `R` | yes                   |
| `FUTURE_PENDING`     | `Future[R]`   | no  ⚠                 |
| `FUTURE_COMPLETED`   | `Future[R]`   | yes                   |

- **`RESULT`** — simplest; use when you want the computed values.
- **`FUTURE_PENDING`** — the method returns as soon as every task has been
  *submitted* to the scheduler. Useful when you want to pipeline other work
  while tasks run, or when you don't need the values at all (fire-and-forget)
  and only care about side effects. Exceptions are deferred to
  `future.result()` / `future.exception()`.
- **`FUTURE_COMPLETED`** — tasks have all finished successfully by the time
  the method returns. Useful when you don't want result values but also want
  to fail fast if something went wrong. If any task raised (even after retries),
  the behavior is determined by `exception_config`.

> **Tip.** If you're not interested in the result values at all — only in the
> side effects — prefer either `FUTURE_*` mode over `RESULT`: there is a great
> optimization at play that lets you entirely avoid materializing the results
> on the orchestrator.

> **⚠ `FUTURE_PENDING` — can be at least slightly blocking**.
> Although it in principle ideally is non-blocking, it still
> can have blocking behavior if otherwise `max_pending_tasks` would not be respected.
> The executor will wait for the cluster to finish already submitting tasks before
> submitting additional ones and returning futures to the caller.

> **⚠ `FUTURE_PENDING` — the caller must drive futures to completion.** The
> method returns while tasks may still be in flight, so nothing else waits on
> them. If the process exits before every future has been
> awaited via `future.result()` / `future.exception()` in-flight tasks can be killed
> when the cluster/driver tears down. When you don't need the values, at least
> drain the futures (e.g. `[f.result() for f in futures]` in a trailing step)
> before shutdown.

---

## `ExceptionConfig` — how failures are surfaced

`ExceptionConfig` controls what happens when a task raises.

| `ExceptionConfig`           | on failure                                                 | affects return type         |
|-----------------------------|------------------------------------------------------------|-----------------------------|
| `RAISE_EAGERLY` *(default)* | raise the first exception observed                         | no                          |
| `RAISE_GROUPED`             | gather all exceptions, raise `ExceptionGroup` at the end ⚠ | no                          |
| `RETURN`                    | each failed task becomes an `Exception` value inline       | widened with `\| Exception` |

> **Order of inlined exceptions is preserved under `RETURN`:**
> `[PASSING_INPUT, FAILING_INPUT, FAILING_INPUT, PASSING_INPUT]`
> yields exactly `[result, Exception, Exception, result]` (for `map`,
> and for `map_lazy` with `ordered=True`).

> ⚠ **`map_lazy` caveat:**
> it does **not** accept `RAISE_GROUPED` — grouping exceptions requires
> draining every task, which defeats the point of a lazy iterator.

---

## Return-type matrix depending on `ResultConfig` and `ExceptionConfig`

### `map(...)` returns `Sequence[...]`

| `ResultConfig` \ `ExceptionConfig` | `RAISE_EAGERLY` / `RAISE_GROUPED` | `RETURN`                             |
|------------------------------------|-----------------------------------|--------------------------------------|
| `RESULT`                           | `Sequence[R]`                     | `Sequence[R \| Exception]`           |
| `FUTURE_PENDING`                   | `Sequence[Future[R]]`             | `Sequence[Future[R] \| Exception]` ⚠ |
| `FUTURE_COMPLETED`                 | `Sequence[Future[R]]`             | `Sequence[Future[R] \| Exception]`   |

### `map_lazy(...)` yields via `Iterator[...]`

| `ResultConfig` \ `ExceptionConfig` | `RAISE_EAGERLY`       | `RETURN`                             |
|------------------------------------|-----------------------|--------------------------------------|
| `RESULT`                           | `Iterator[R]`         | `Iterator[R \| Exception]`           |
| `FUTURE_PENDING`                   | `Iterator[Future[R]]` | `Iterator[Future[R] \| Exception]` ⚠ |
| `FUTURE_COMPLETED`                 | `Iterator[Future[R]]` | `Iterator[Future[R] \| Exception]`   |

> **⚠ Note on `FUTURE_PENDING` + `RETURN`.** Under `FUTURE_PENDING`,
> `ExceptionConfig` is effectively ignored at runtime because futures are
> returned before tasks complete — exceptions surface through
> `future.exception()` / `future.result()` on the individual futures. The
> `| Exception` in the type signature is kept for symmetry with
> `FUTURE_COMPLETED`, but no `Exception` values will actually appear inline in
> the returned sequence/iterator.

---

### Why `map` is preferable to `map_lazy` when you can afford the memory

`map_lazy` yields control back to the caller after every completed task.
During that time, the generator is *suspended*: no new tasks get submitted,
no poll for finished tasks happens. If the consumer does heavy per-item work
(or blocks on IO) between iterations, **workers can starve** — you'll see idle
CPUs while the executor waits to resume submission.

`map` drains tasks internally in a tight loop, so submission and retrieval
always run at full pace. The trade-off is that `map` materializes the full
result sequence before returning.

Rule of thumb:

- **Always prefer `map`** when the result sequence fits in memory, and you
  don't need streaming.
- **Use `map_lazy`** when the iterable is too large to hold all results, or
  when downstream processing should begin before every task is done. If
  downstream per-item work is expensive, consider processing in batches or
  offloading it back into the executor.

---

## Retries

All methods accept a `retry_config` parameter:

| `retry_config` value                       | behaviour                                               |
|--------------------------------------------|---------------------------------------------------------|
| `None` *(default)*                         | no retries                                              |
| `int` (e.g. `3`)                           | retry up to 3 times on **any** `Exception`              |
| `RetryConfig(retries=N, exceptions=[...])` | retry up to N times, only on the listed exception types |

Platform-sided exceptions (e.g., Ray workers dying) are **always** retried,
even if you pass `RetryConfig(exceptions=[])`. The retry happens inside the
backend — under `FUTURE_PENDING` you only learn about retries when the future
resolves.

```python
from disco.executors import RetryConfig

executor.map(
    flaky_api_call,
    retry_config=RetryConfig(retries=5, exceptions=[TimeoutError, ConnectionError]),
)(urls)
```

---

## Back-pressure (`max_pending_tasks`)

`map` and `map_lazy` both accept `max_pending_tasks`.
It caps how many tasks can be "submitted but not yet completed" at any moment:

| value  | effect                                                              |
|--------|---------------------------------------------------------------------|
| `int`  | throttle submission; new tasks only go in after some have completed |
| `None` | no throttling; all tasks submitted up-front                         |

The default exists because orchestrators like Ray or Dask struggle when there is
an extreme number of tasks in a pending state.
Raise or disable only if you've measured and know you don't hit scheduler limits.

---

## Executor backends

### `LocalExecutor`

```python
from disco.executors.local import LocalExecutor
```

Runs tasks sequentially in the calling process. Useful for development and
for unit tests. Notes:

- `max_pending_tasks` and `ordered` are no-ops (always sequential, always
  in order).
- Tasks under `FUTURE_PENDING` execute lazily when `future.result()` /
  `future.exception()` is called.

### `RayExecutor`

```python
from disco.executors.ray import RayExecutor, RayKwargs
```

Runs tasks on a Ray cluster. Must be called after `ray.init()`.

Per-call Ray-specific options go through `executor_kwargs`:

```python
executor.map(
    my_task,
    executor_kwargs={
        RayExecutor: RayKwargs(
            func_remote_kwargs={"num_cpus": 2, "memory": 512 * 1024 ** 2},
            get_timeout=30.0,
            wait_poll_interval=10.0,
        ),
    },
)(inputs)
```

| `RayKwargs` field    | feeds into                                         |
|----------------------|----------------------------------------------------|
| `func_remote_kwargs` | `ray.remote(**kwargs)(func)` — static task options |
| `get_timeout`        | timeout on each `ray.get(...)`                     |
| `wait_poll_interval` | timeout on each `ray.wait(...)` poll               |

### `LocalPoolExecutor`

```python
import concurrent.futures
from disco.executors.local_pool import LocalPoolExecutor
```

Runs tasks on a `concurrent.futures.Executor` — a `ThreadPoolExecutor` or a
`ProcessPoolExecutor`. One implementation serves both pool types; the caller
constructs the pool and owns its lifecycle. Useful when you want parallelism on
a single machine without spawning a distributed execution cluster.

```python
with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
    executor = LocalPoolExecutor(pool)
    results = executor.map(my_task)(inputs)
```

### `DaskExecutor`

> ⚠ **Experimental.** This backend was essentially vibe-coded against the abstract
> `Executor` contract — it has not been battle-tested on a real cluster, and edge
> cases around scheduler pressure, future lifetime, and retry semantics may bite. Use with suspicion, prefer
> `RayExecutor` / `LocalPoolExecutor` where feasible, and please report oddities.

```python
import dask.distributed
from disco.executors.dask import DaskExecutor
```

Wraps a caller-supplied `dask.distributed.Client` (the executor does not own the
client's lifecycle).

```python
client = dask.distributed.Client(...)
executor = DaskExecutor(client)
results = executor.map(my_task)(inputs)
```

Behavioural notes specific to Dask:

- **Batched submissions.** `map` / `map_lazy` always fan out through `client.map`
  (even under `max_pending_tasks` throttling, where they chunk into a few large
  `client.map` calls rather than looping `client.submit`). Dask's scheduler is a
  single-threaded Python process and is easy to overload with many individual
  `submit` calls.
- **Drains in batches.** Completion is consumed via
  `dask.distributed.as_completed().next_batch(block=True)` so each round-trip to the
  scheduler fetches as many completions as possible.
- **Explicit task keys.** Keys are generated as
  `funcname[:50]-base_hash[:8]-repr(arg)-tokenize(arg)[:8]-idx` to bypass Dask's
  (slow) default hashing of argument tuples.
- **Retries.** `retry_config: int` forwards to Dask's native `retries=` (which can
  retry across workers, covering infra failures). `RetryConfig(retries=N,
  exceptions=[...])` uses a per-worker retry wrapper because Dask's native retry
  cannot filter by exception type.
- **Future lifetime.** Dropping the last reference to a `dask.distributed.Future` is
  what signals the scheduler to release the task's result on its worker. The
  executor's internal bookkeeping keys on `Future.key` (a small string) rather than
  on future objects so it never keeps futures pinned longer than a drain iteration.
  Callers should, however, be aware that storing `DaskFuture` instances in
  long-lived containers keeps the worker-side result alive too.

---

## Choosing configuration: practical recipes

| Goal                                                                                | `func`     | `ResultConfig`     | `ExceptionConfig` |
|-------------------------------------------------------------------------------------|------------|--------------------|-------------------|
| I want the values; fail fast on first error                                         | `map`      | `RESULT`           | `RAISE_EAGERLY`   |
| I want the values; report **all** failures together                                 | `map`      | `RESULT`           | `RAISE_GROUPED`   |
| I want the values; handle each failure per-item                                     | `map`      | `RESULT`           | `RETURN`          |
| I don't want values; fail fast                                                      | `map`      | `FUTURE_COMPLETED` | `RAISE_EAGERLY`   |
| I don't want values; report all failures                                            | `map`      | `FUTURE_COMPLETED` | `RAISE_GROUPED`   |
| I don't want values; handle each failure per-item                                   | `map`      | `FUTURE_COMPLETED` | `RETURN`          |
| I want to stream results as they come, fail fast                                    | `map_lazy` | `RESULT`           | `RAISE_EAGERLY`   |
| I want to stream results, handle failures per-item                                  | `map_lazy` | `RESULT`           | `RETURN`          |
| Fire-and-forget (potentially slightly blocking through `max_pending_tasks`)         | `map`      | `FUTURE_PENDING`   | any (ignored)     |
| Streaming fire-and-forget (move control fully over to user, no waiting for results) | `map_lazy` | `FUTURE_PENDING`   | any (ignored)     |

