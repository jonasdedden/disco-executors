from __future__ import annotations

import concurrent.futures
import contextlib
import os
from functools import partial
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import dask.distributed
import pytest
import ray
from hypothesis import settings

from .counter_utils import raise_counter, register_counter
from disco.executors.dask import DaskExecutor
from disco.executors.local import LocalExecutor
from disco.executors.local_pool import LocalPoolExecutor
from disco.executors.ray import RayExecutor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from disco.executors.base import Executor

EXECUTOR_NAMES: Final[tuple[str, ...]] = ("local", "dask", "ray", "thread", "process")

# Shared CI runners have noisy timings, and the default 200ms per-example deadline mostly measures scheduler latency.
# Tests with an explicit `@settings(deadline=...)` keep theirs.
settings.register_profile("ci", deadline=None, print_blob=True)
if os.environ.get("CI"):
    settings.load_profile("ci")


def pytest_configure(config: pytest.Config) -> None:
    for name in EXECUTOR_NAMES:
        config.addinivalue_line("markers", f"{name}: runs on the `{name}` executor (set automatically)")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Mark each executor-parametrized test with its executor, so e.g. `-m ray` selects exactly the Ray tests.
    for item in items:
        # `callspec` only exists on parametrized test functions.
        if isinstance(item, pytest.Function) and hasattr(item, "callspec") and "executor" in item.callspec.params:
            executor_name = item.callspec.params["executor"]
            assert isinstance(executor_name, str)
            item.add_marker(executor_name)


def _num_cpus() -> int:
    return int(os.environ.get("KUBERNETES_CPU_REQUEST", "8"))


@contextlib.contextmanager
def make_local_executor() -> Iterator[Executor]:
    yield LocalExecutor()


@contextlib.contextmanager
def make_dask_executor() -> Iterator[Executor]:
    cluster = dask.distributed.LocalCluster(n_workers=_num_cpus(), processes=False, silence_logs=100)  # type: ignore[no-untyped-call]
    client = dask.distributed.Client(cluster)  # type: ignore[no-untyped-call]
    try:
        yield DaskExecutor(client=client)
    finally:
        client.close()  # type: ignore[no-untyped-call]
        cluster.close()


@contextlib.contextmanager
def make_ray_executor() -> Iterator[Executor]:
    # Shorter "internal heartbeat" such that tasks are retried faster
    ray.init(num_cpus=_num_cpus(), _system_config={"core_worker_internal_heartbeat_ms": 10})
    try:
        yield RayExecutor()
    finally:
        ray.shutdown()


@contextlib.contextmanager
def make_thread_pool_executor() -> Iterator[Executor]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=_num_cpus()) as pool:
        yield LocalPoolExecutor(pool)


@contextlib.contextmanager
def make_process_pool_executor() -> Iterator[Executor]:
    with concurrent.futures.ProcessPoolExecutor(max_workers=_num_cpus()) as pool:
        yield LocalPoolExecutor(pool)


@pytest.fixture(scope="session")
def executor(request: pytest.FixtureRequest) -> Iterator[Executor]:
    factory = EXECUTOR_FACTORIES[request.param]
    with factory() as executor:
        yield executor


EXECUTOR_FACTORIES = {
    "local": make_local_executor,
    "dask": make_dask_executor,
    "ray": make_ray_executor,
    "thread": make_thread_pool_executor,
    "process": make_process_pool_executor,
}


@pytest.fixture(scope="session")
def counter_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("counters")


@pytest.fixture(scope="session")
def atomic_counter_factory(counter_root: Path) -> Callable[[], Callable[..., int]]:
    def factory() -> Callable[..., int]:
        key = uuid4().hex
        register_counter(counter_root, key)
        return partial(raise_counter, counter_root, key)

    return factory


@pytest.fixture
def atomic_counter(atomic_counter_factory: Callable[[], Callable[..., int]]) -> Callable[..., int]:
    return atomic_counter_factory()
