# Dinkster library

Shared library records and account-wide file indexing over a private
child-process pipe, with synchronous and asynchronous engine clients.

## Develop locally

From the repository root, use the contributor runtime:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m dinkster_library --describe
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_library_helper.py tests/test_library_migration.py
```

See [contributor setup](../../README.md) for prerequisites and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Public clients](src/dinkster_library/__init__.py) and [pipe entry point](src/dinkster_library/__main__.py)
- [Record operations](src/dinkster_library/_records.py) and [contract tests](../../tests/test_library_helper.py)
- [API versions and removal procedure](../../docs/library-api.md)
- [Client, trust and retention operating notes](../../docs/library-operations.md)
- [Architecture](../../DESIGN.md)
