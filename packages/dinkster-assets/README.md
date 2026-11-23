# Dinkster assets

Content-digest asset identity, catalogs, mounts, verified vaults, acquisition and
save targets for the engine, workers and server.

## Develop locally

From the repository root, use the contributor runtime:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_assets.py tests/test_asset_writer.py tests/test_mounts.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Public exports](src/dinkster_assets/__init__.py) and [asset records](src/dinkster_assets/records.py)
- [Mounts and grants](src/dinkster_assets/mounts.py), [writers](src/dinkster_assets/writer.py) and [resolver subscriptions](src/dinkster_assets/resolver_subscription.py)
- [Library trust and retention operations](../../docs/library-operations.md)
- [Catalog migration and resolver operating notes](../../docs/asset-operations.md)
- [Asset authority and node-authoring hazards](../../docs/hazards.md) and [architecture](../../DESIGN.md)
