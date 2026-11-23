# Testing an installed pack

Install the engine, pytest, your pack and its runtime dependencies into a test
environment. Pure Python packs need no Torch:

```sh
python -m pip install dinkster-engine pytest
```

For tensor packs, CPU-only Torch is sufficient for ordinary CI:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pytest -m 'not accelerator'
```

Keep accelerator-only cases separately marked, independent of vendor. See
[pack authoring](pack-authoring.md#running-and-testing) for the import boundary,
marker registration and doctor step. Doctor and registry checks are not a
replacement for execution tests.

Use `from dinkster_engine.testing import load_pack` in a module- or
session-scoped fixture in `tests/`. Do not call this synchronous helper inside a
running asyncio event loop. Start from the
[complete executable example](../tests/fixtures/pack-testing/tests/example.py)
for a graph, a pack-owned codec and cache assertions. Its
[manifest](../tests/fixtures/pack-testing/dinkster-pack.toml) shows the installable
example pack.

The [testing API source](../packages/dinkster-engine/src/dinkster_engine/testing.py)
is the reference for worker lifetime, execution modes and result objects.
Release maintainers can run the same example against built API/engine wheels
outside the checkout in both execution modes:

```sh
python scripts/verify_pack_testing.py dist
```
