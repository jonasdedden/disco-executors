# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Changed

- **Breaking:** The executors from `disco.executors.revamp` now live directly in `disco.executors`
  (e.g. `disco.executors.ray.RayExecutor`); the `disco.executors.revamp` submodule is gone.
- **Breaking:** `submit` / `map` / `map_lazy` take the function plus the executor options and return a callable that
  takes the function's own arguments: `executor.map(func, retry_config=3)(items, *args, **kwargs)` instead of
  `executor.map(mwrap(func, items, *args, **kwargs), retry_config=3)`. Both calls are fully typed, and the function's
  keywords can no longer clash with executor options.
- Backends implement the private hooks `_submit` / `_map` / `_map_lazy`; the public overloads live once on `Executor`.

### Added

- `Future` is exported from `disco.executors`.
- `LocalFuture.execute()` (previously the private `_execute()`), the counterpart of `LocalFuture.reset()`.

### Fixed

- `LocalExecutor` raised `ValueError: Unknown format code '?'` instead of the intended `TypeError` for an invalid
  `retry_config`.
- `DaskExecutor` ignored keyword arguments named like `dask.base.tokenize`'s options (`ensure_deterministic`) when
  generating task keys, so maps differing only in such an argument could get identical keys and reuse each other's
  results.

### Removed

- The legacy executors (previously `disco.executors.[base/local/ray/dask]`) including `ErrorRaiseMode`,
  `to_revamp_executor` and `from_legacy_executor`.
- `wrap` and `mwrap`; see the new call style above.

## [0.2.7] - 2026-04-21

### Added

- `LocalPoolExecutor` for executing tasks locally through `concurrent.futures.[ThreadPoolExecutor/ProcessPoolExecutor]`.

## [0.2.6] - 2026-04-21

### Changed

- `RayExecutor` now internally uses *two* different `ObjectRef`'s for retrieving task results; one for finding out
  whether the task has finished without overhead and one for retrieving the actual result.
- `RayExecutor` now wraps the called function with additional logging infrastructure on failures, similar to what the
  old executors did.

## [0.2.5] - 2026-04-20

### Added

- `RayExecutor` of `disco.executors.revamp` now supports `ExceptionConfig` type `RETURN` which will return exceptions
  inline instead of raising them.

### Fixed

- `RayExecutor` of `disco.executors.revamp` now works in Ray Client contexts (by simplifying and fixing
  `_get_object_refs`).
- `RayFuture` doesn't raise anymore on `.exception()` if the task failed because of a Ray system error.

## [0.2.4] - 2026-04-10

### Fixed

- `map_lazy` in `FUTURES_PENDING` now should be dramatically faster

## [0.2.3] - 2026-04-10

### Added

- Revamp executors in `disco.executors.revamp` now have a `map_lazy` method

## [0.2.2] - 2026-04-10

### Fixed

- Actually make `executor_kwargs` work for Ray

## [0.2.1] - 2026-04-10

### Added

- The "legacy"/current executors in `disco.executors` got a `to_revamp_executor`, spawning an executor from
  `disco.executors.revamp`.

## [0.2.0] - 2026-04-10

### Added

- The `disco.executors.revamp` submodule providing a total rewrite of the executor framework.

## [0.2.0a7] - 2025-12-01

### Changed

- `RayExecutor` now logs full stack traces when tasks fail. Exceptions are logged at the worker where they occur before
  Ray serializes them.

## [0.2.0a4] - 2025-07-10

### Fixed

- The `get_workers()` method in the `DaskExecutor` class now uses `self.client.nthreads()` instead of
  `self.client.scheduler_info()["workers"]` since `scheduler_info` defaults to returning info for only 5 workers and
  includes a lot of other unrelated information.
