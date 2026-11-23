# dinkster-caches

Memory, persistent disk and layered cache stores for Dinkster execution results.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_cache_share.py tests/test_cas.py tests/test_memory.py
```

See [contributor setup](../../README.md) and [testing](../../docs/testing.md).

## Reference and operations

- [Cache operations and accounting cautions](../../docs/cache-operations.md)
- [Public API](src/dinkster_caches/__init__.py)
- [Memory](src/dinkster_caches/memory.py), [disk](src/dinkster_caches/disk.py) and [layering](src/dinkster_caches/layered.py)
- [Entry wire format](src/dinkster_caches/wire.py) and [blob store](src/dinkster_caches/cas.py)
- [Architecture](../../DESIGN.md) and [hazards](../../docs/hazards.md)
