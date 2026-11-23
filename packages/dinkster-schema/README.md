# dinkster-schema

The shared typed node interface model and schema elaboration, solving and
replacement tools used by Dinkster's graph and execution layers.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_schema_current_contracts.py tests/test_output_descriptors.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Schema declarations](src/dinkster_schema/model.py) and [wire API](src/dinkster_schema/wire.py)
- [Node authoring API](src/dinkster_schema/node.py)
- [Pack authoring and preview integration](../../docs/pack-authoring.md)
- [Extension integration cautions](../../docs/extension-integration.md)
- [Architecture](../../DESIGN.md) and [wire compatibility policy](../../docs/hazards.md#h7-pre-10-wire-formats-are-unstable-and-only-the-current-version-decodes)
