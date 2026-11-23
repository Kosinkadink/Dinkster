# dinkster-values

Typed value envelopes, codecs, transports and resource ownership primitives for
Dinkster's execution boundaries.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_values.py tests/test_stream_values.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Value API](src/dinkster_values/__init__.py) and [type registry](src/dinkster_values/registry.py)
- [Media consumer cleanup](../../docs/media-consumption.md)
- [Produced document ownership](../../docs/image-document-operations.md)
- [Pack authoring](../../docs/pack-authoring.md#value-types-and-codecs)
- [Architecture](../../DESIGN.md) and [hazards](../../docs/hazards.md)
