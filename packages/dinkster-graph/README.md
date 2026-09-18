# dinkster-graph

`dinkster-graph` provides Dinkster's pure graph document model, wire format,
elaboration-aware structural validation, dependency analysis, and deterministic
execution planning. It depends on `dinkster-schema` and performs no IO or
execution; `dinkster-engine` and server-facing graph handling build on it.

## Setup

This package is a uv workspace member. From the repository root, install the
whole workspace with:

```sh
uv sync --all-packages
```

The package is not published separately yet and provides no console script.

## Use

The public surface includes `Graph`, `GraphNode`, `Link`, repetition-region
types, `validate`, `elaborate_graph`, `plan`, dependency helpers, graph wire
conversion, and `snapshot_graph`. Build links with real output IDs, then plan
only the nodes needed by the requested target:

```python
from dinkster_graph import Graph, GraphNode, Link, plan

graph = Graph(
    nodes={
        "source": GraphNode("example.constant", inputs={"value": 4}),
        "double": GraphNode(
            "example.double",
            inputs={"value": Link("source", "value")},
        ),
    }
)

assert plan(graph, ["double"]) == ["source", "double"]
```

Given a mapping of node type to `NodeSchema`, `validate(graph, schemas,
targets)` returns structured `Diagnostic` objects. Structural errors are
authoritative; type mismatches are advisory warnings except for the scoped
`core.combo` boundary, which is hard so strings and combos require explicit
converter nodes.

## Learn more

See the package layout in DESIGN 3, DESIGN 3.1 for document elaboration, DESIGN
3.10 for dependency-DAG scheduling, and DESIGN 3.13 for repetition regions.
See hazards H10, H12, and H20 in `docs/hazards.md`.

Focused tests include `tests/test_graph.py`, `tests/test_graph_wire.py`,
`tests/test_regions.py`, `tests/test_dynamic.py`, and `tests/test_compose.py`.
