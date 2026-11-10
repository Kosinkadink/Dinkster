# dinkster-engine

`dinkster-engine` is Dinkster's location-agnostic scheduler. It consumes the
`Worker` and `CacheStore` protocols, which live one layer down in
`dinkster-protocol` (and are re-exported here as part of the engine's API).
It depends on the schema, value, graph, and protocol packages; worker and
cache implementations plug into the protocol boundary, while the server
uses the engine to execute graphs.

## Setup

Install the engine and its exact-version API companion without Torch:

```sh
python -m pip install dinkster-engine pytest
```

The wheel includes `dinkster_engine` and `dinkster_graph`. It does not install
a console script or the server. Install a pack and its dependencies separately.

## Tests from pytest

Use ordinary synchronous tests and a module- or session-scoped fixture:

```python
from pathlib import Path

import pytest
from dinkster_engine.testing import load_pack


@pytest.fixture(scope="module")
def pack():
    with load_pack(Path(__file__).parents[1] / "dinkster-pack.toml") as loaded:
        yield loaded


def test_graph(pack):
    graph = {
        "shout": ("my-pack.shout", {"text": "hello"}),
        "tally": ("my-pack.tally", {"text": pack.link("shout", "text")}),
    }
    result = pack.run(graph, targets=["tally"])
    assert result.value("tally", "count") == 5
    assert result.executed == {"shout", "tally"}
    again = pack.run(graph, targets=["tally"])
    assert again.cached == {"shout", "tally"}
```

`pack.run_node(type_name, inputs)` returns the same result shape under node
id `"node"`. `outputs` contains decoded Python values keyed by node and output;
`executed` and `cached` are sets of node ids reported by the engine.

The fixture shares one real worker process, engine and cache across its tests.
Closing the context closes the worker. `load_pack(path, in_process=True)` opts
out of process isolation without changing test calls. Both modes use the real
engine; the parent also loads pack codecs to decode custom output types.
The helper loads one pack, not a serving composition or HTTP application.
It is synchronous and must not be called inside a running asyncio event loop.

The [installable example pack](../../tests/fixtures/pack-testing) includes
`example.py`, a complete pytest file testing a two-node graph, a pack-owned
codec, caching, and shared worker startup. Release CI runs this same file
against built API/engine wheels with
`python scripts/verify_pack_testing.py dist`, in both execution modes, from
outside the checkout. Its test environment contains no Torch.

Pure Python packs need no Torch. Tensor packs can install CPU-only Torch:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
Keep accelerator-specific tests separately marked (for example
`@pytest.mark.accelerator`) and run the ordinary suite with
`pytest -m 'not accelerator'` on CPU CI. Doctor and registry checks validate
pack declarations; neither is a replacement for these execution tests.

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

Expanding nodes build children with `ExpansionBuilder`. Children retain their
`output_members` and `slot_variants` through worker handoff and cache replay.
`ExpansionBuilder(cache_policy="rerun")` disables cache hits and single-flight
coalescing for its children. The default `inherit` policy carries an enclosing
expansion's policy, or uses ordinary content-key reuse at the root. An explicit
`reuse` overrides inherited reruns for that fragment and its descendants.
Admitted loop pair metadata is available in the run context's immutable graph
view even after an end node's body-linked inputs are removed.

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
