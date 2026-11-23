# dinkster-engine

Dinkster's location-agnostic graph scheduler and synchronous pack-testing helper.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_engine.py tests/test_parallel.py tests/test_reporting.py
```

See [contributor setup](../../README.md) and [testing](../../docs/testing.md).
For testing an independently installed pack, use [the pack testing guide](../../docs/pack-testing.md).

## Reference

- [Public API](src/dinkster_engine/__init__.py)
- [Scheduler](src/dinkster_engine/engine.py) and [graph admission](src/dinkster_engine/compile.py)
- [Pack testing API](src/dinkster_engine/testing.py)
- [Architecture](../../DESIGN.md) and [hazards](../../docs/hazards.md)
