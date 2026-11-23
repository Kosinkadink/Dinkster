# Asset catalog and resolver operations

## Catalog migrations

Back up a legacy SQLite catalog before upgrading. To inspect its Alembic state
from the repository root, use the contributor runtime and the same catalog
argument for both commands:

```sh
python scripts/develop.py run -- alembic -c packages/dinkster-assets/alembic.ini -x catalog=/absolute/path/library.sqlite current
python scripts/develop.py run -- alembic -c packages/dinkster-assets/alembic.ini -x catalog=/absolute/path/library.sqlite check
```

For a schema change, update
[schema.py](../packages/dinkster-assets/src/dinkster_assets/schema.py), then run
`alembic revision --autogenerate -m "Describe the schema change"` through that
runtime with the same configuration and catalog arguments. Review the generated
operations before applying them. Migration resources belong in the package wheel.

Read-only inspection still needs SQLite WAL coordination access; follow the
[shared journal connection notes](../packages/dinkster-library/src/dinkster_library/_journal.py)
rather than bypassing recovery or retrying with write access.

## Helper-backed consumers

See [library operations](library-operations.md) for borrowed-client lifetime,
engine authorization, byte verification, raw extensions, strict parser projections
and collection duties.

Consumers needing managed resolver leads must use helper-backed provenance too.
The helper-backed resolver publishes its managed layers transactionally; do not
expect it to write those leads through a separate legacy provenance store.

## Resolver selection

Configure the official export URL and its stable provider ID explicitly. Use the
export's exact `name` as provider identity, not a guessed/default value. See the
[serve CLI options](serve-cli.md#--official-resolver-url--official-resolver-provider-id).
Resolver indexes are download leads, not permission to trust fetched bytes or to
read a host path. Acquisitions must still verify the requested content identity.

## Reference

- [Catalog schema](../packages/dinkster-assets/src/dinkster_assets/schema.py) and [migrations](../packages/dinkster-assets/src/dinkster_assets/migrations)
- [Resolver index parser](../packages/dinkster-assets/src/dinkster_assets/resolver_index.py) and [subscription tests](../tests/test_resolver_indexes.py)
- [Receipt verification](../packages/dinkster-assets/src/dinkster_assets/public_acquisition.py) and [receipt tests](../tests/test_public_acquisition_receipts.py)
