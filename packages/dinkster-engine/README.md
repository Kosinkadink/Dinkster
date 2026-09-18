# dinkster-engine

`dinkster-engine` is Dinkster's location-agnostic scheduler. It consumes the
`Worker` and `CacheStore` protocols, which live one layer down in
`dinkster-protocol` (and are re-exported here as part of the engine's API).
It depends on the schema, value, graph, and protocol packages; worker and
cache implementations plug into the protocol boundary, while the server
uses the engine to execute graphs.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root, install the complete workspace:

```sh
uv sync --all-packages
```

It does not install a console script.

## Use

The public surface exports `Engine`, its structured events and run results,
execution errors, and the `Worker`, `CacheStore`, and invocation contracts.
Construct an engine from concrete schemas, a type registry, and protocol
implementations, then run graph targets:

```python
engine = Engine(
    schemas=schemas,
    registry=registry,
    worker=worker,
    cache=cache,
    max_concurrency=4,
)
result = await engine.run(graph, ["preview"])
```

`admit_parent_graph` is the pure parent-side boundary for a bounded graph
delta. It accepts graph wire plus complete origin records and explicit virtual
joins, then returns an immutable complete candidate or `GraphAdmissionError`.
It does not schedule or execute the candidate.

An event listener can observe progress without changing execution:

```python
engine = Engine(
    schemas=schemas, registry=registry, worker=worker, cache=cache,
    on_event=lambda event: print(event.kind, event.node_id),
)
```

## Learn more

See DESIGN 3.3 for boundary-first execution, DESIGN 3.4 for caching, and
DESIGN 3.8 for structured events. The invariants behind the protocols and
cache keys are documented in `docs/hazards.md`, especially H3, H4, and H12.
Focused coverage is in `tests/test_engine.py`, `tests/test_parallel.py`,
`tests/test_regions.py`, and `tests/test_reporting.py`.
