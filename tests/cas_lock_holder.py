"""Spawn target that avoids importing the engine and worker test modules."""

from multiprocessing.synchronize import Event
from pathlib import Path

from dinkster_caches import DiskCacheStore
from dinkster_values import TypeRegistry, register_core_types


def hold_disk_cache_process_lock(root: str, ready: Event, release: Event) -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    store = DiskCacheStore(Path(root), registry)

    def hold() -> None:
        ready.set()
        if not release.wait(30):
            raise TimeoutError("cache lock test was not released")

    store._with_lock(hold)
