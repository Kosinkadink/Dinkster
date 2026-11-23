# dinkster-graph

Dinkster's pure graph document model, structural validation and execution planning.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_graph.py tests/test_graph_wire.py tests/test_regions.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Public API](src/dinkster_graph/__init__.py)
- [Validation](src/dinkster_graph/validate.py) and [migration](src/dinkster_graph/migrate.py)
- [Architecture](../../DESIGN.md), [loop authoring](../../docs/loops.md) and [hazards](../../docs/hazards.md)
