# dinkster-collab

Optional server-ordered collaborative document sessions, separate from the
execution engine and job queue.

## Develop locally

From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_collab.py tests/test_collab_durability.py
```

See [contributor setup](../../README.md) to run the composed server and
[testing](../../docs/testing.md) for validation commands.

## Reference

- [Session service](src/dinkster_collab/sessions.py)
- [HTTP and WebSocket routes](src/dinkster_collab/routes.py)
- [Host, client and agent integration](../../docs/collaboration-integration.md)
- [Authentication](../../docs/auth.md)
- [Package architecture](../../DESIGN.md#package-layout)
