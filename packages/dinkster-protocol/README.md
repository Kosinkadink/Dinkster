# dinkster-protocol

Execution-boundary contracts shared by the Dinkster engine, workers and caches.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_attention_route_token.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Public contracts](src/dinkster_protocol/__init__.py)
- [Attention route contracts](src/dinkster_protocol/attention.py)
- [Architecture and package boundaries](../../DESIGN.md)
- [Hazards](../../docs/hazards.md)
