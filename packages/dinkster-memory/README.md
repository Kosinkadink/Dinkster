# dinkster-memory

Process-usable system-memory measurements and memory governance for Dinkster.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_system_memory.py tests/test_memory.py tests/test_leases.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Public API](src/dinkster_memory/__init__.py)
- [System memory](src/dinkster_memory/system.py) and [governor](src/dinkster_memory/governor.py)
- [Resident loading and caller obligations](src/dinkster_memory/resident.py)
- [Architecture](../../DESIGN.md) and [hazards](../../docs/hazards.md)
