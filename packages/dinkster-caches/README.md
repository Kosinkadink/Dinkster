# dinkster-caches

`dinkster-caches` provides `CacheStore` implementations for memory, persistent
disk, and fastest-first layered caching, plus the shared cache-entry wire
format and content-addressed blob store. It depends on values, the
`dinkster-protocol` boundary package, memory, and assets - not on the engine;
the engine and server consume its stores.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root, install the complete workspace:

```sh
uv sync --all-packages
```

It does not install a console script.

## Use

`MemoryLRUCache` is the smallest `CacheStore` implementation:

```python
from dinkster_caches import MemoryLRUCache

cache = MemoryLRUCache(max_entries=256)
await cache.put(cache_key, outputs)
outputs_or_none = await cache.get(cache_key)
```

Compose stores fastest-first. A slower hit is promoted into every faster
layer, and writes pass through all layers:

```python
from dinkster_caches import DiskCacheStore, LayeredCache, MemoryLRUCache

disk = DiskCacheStore(cache_root, registry, max_bytes=20 * 1024**3)
cache = LayeredCache(MemoryLRUCache(), disk)
```

Disk mutations and garbage collection use a root-wide thread and process
lock, so engine instances on one machine may safely share a directory. The
disk budget evicts manifests by last hit and removes payload blobs only after
their final reference disappears. Startup removes interrupted manifests and
orphaned blobs. Entries containing live resource references or values without
persistable bytes are refused by disk and remain usable in memory.

`LayeredCache.hits_by_layer` counts hit sources, and the engine reports the
source as `cacheLayer` on `node_cached` events.

The package also exports `DiskCAS` and the versioned entry encoding helpers
for disk and trusted peer-cache transports.

## Learn more

See DESIGN 3.4 for the cache contract and persistent CAS, DESIGN 3.2 for
the cache wire trust boundary, and DESIGN 3.13 for value-tree accounting.
Related invariants are in `docs/hazards.md`, especially H4. Focused coverage
is in `tests/test_engine.py`, `tests/test_cache_share.py`, `tests/test_cas.py`,
and `tests/test_memory.py`.
