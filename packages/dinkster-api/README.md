# dinkster-api

The versioned extension entry point for Dinkster pack authors.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_api_v1.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Versioned API and compatibility policy](src/dinkster_api/v1.py)
- [Pack authoring](../../docs/pack-authoring.md)
- [Extension integration cautions](../../docs/extension-integration.md)
- [Pack template](https://github.com/Kosinkadink/dinkster-pack-template)
- [Architecture](../../DESIGN.md#36-extension-api-predictable-no-hacks-required)
