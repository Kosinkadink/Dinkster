# dinkster-server

Dinkster's aiohttp HTTP/WebSocket server, queue policy, event fanout and host
composition interfaces. The umbrella package supplies `dinkster-serve`.

## Run locally

From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- dinkster-serve
```

See [contributor setup](../../README.md) for prerequisites, isolated account
storage and server options, and [testing](../../docs/testing.md) for checks.

## Reference

- [Server exports](src/dinkster_server/__init__.py)
- [Host integration and result publications](../../docs/server-integration.md)
- [Authentication](../../docs/auth.md)
- [Library helper operations](../../docs/library-operations.md)
- [Native protocol architecture](../../DESIGN.md#35-server-and-native-protocol)
