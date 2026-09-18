# dinkster-protocol

The execution-boundary contracts that the engine, its workers, and its
caches all plug into - and nothing else. This is the lowest layer of the
execution stack: it depends only on `dinkster-schema` and `dinkster-values`,
so a worker interpreter (in-process, another venv, or another machine)
can import the invocation contract without pulling in the scheduler.

Two protocols live here (hazard H3 - the engine never calls node code):

- **`Worker`** - receives an `Invocation`, returns an `InvocationResult`,
  streams `InvocationEvent`s while running. In-process, isolated-venv,
  and remote are three implementations of this one protocol.
- **`CacheStore`** - `get`/`put` keyed by `CacheKey`; the layered/disk/
  memory caches in `dinkster-caches` implement it.

Plus the frozen data they exchange: `Invocation`, `InvocationEvent`,
`OnInvocationEvent`, `NodeError`, `InvocationResult`.

Attention route tokens bind job policy to worker capability evidence.
Versions 1 and 2 encode supported global and per-role policies. Version 3
encodes requests with at least one unavailable policy routed to SDPA,
preserving the requested policy and any supported routes. All workers
derive and compare the full token against their own capabilities; a portable
fallback does not authorize forged device, provider, or kernel identities.

## Why it is its own package

These contracts used to live in `dinkster-engine`, so `dinkster-workers` and
`dinkster-caches` both depended on the whole engine just to name an
`Invocation` or a `CacheKey`. That meant a worker child interpreter
transitively installed the scheduler it never runs. Splitting the
contracts out lets workers and caches depend on this leaf instead, and
`dinkster-engine` re-exports every name for backward compatibility
(`from dinkster_engine import Invocation` still works).
