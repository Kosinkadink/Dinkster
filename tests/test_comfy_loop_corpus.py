"""The recorded Generic Loops corpus runs through native loop expansion only."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dinkster_graph import GraphNode, Link, RegionNode
from dinkster_nodes_dev.loop_corpus import Observations

from tools.run_comfy_loop_corpus import compile_prompt, make_engine

# Recorded issue408 prompts and observations; no sibling evidence checkout is required.
PROMPTS = json.loads(
    (Path(__file__).parent / "fixtures/loop-corpus-prompts.json").read_text(encoding="utf-8")
)


def test_recorded_prompts_compile_to_native_loop_nodes() -> None:
    graph = compile_prompt(PROMPTS["nested-carry"]["api_prompt"])
    outer = graph.nodes["outer"]
    inner_close = graph.nodes["inner_close"]
    assert isinstance(outer, GraphNode)
    assert isinstance(inner_close, GraphNode)
    assert outer.node_type == "StartLoop"
    assert inner_close.node_type == "EndLoop"
    assert isinstance(graph.nodes["capture"], RegionNode)
    assert any(
        isinstance(node, GraphNode) and node.node_type == "std.list.element"
        for node in graph.nodes.values()
    )


def test_list_outputs_are_explicitly_flattened_before_map_capture() -> None:
    graph = compile_prompt(PROMPTS["accumulate-list-output"]["api_prompt"])
    capture = graph.nodes["capture"]
    assert isinstance(capture, RegionNode)
    capture_input = capture.inputs["item"]
    assert isinstance(capture_input, Link)
    flatten = graph.nodes[capture_input.node_id]
    assert isinstance(flatten, RegionNode)
    assert flatten.kind == "map"
    assert flatten.outputs["list"].mode == "flatten"


def test_native_loop_compiler_runs_carried_value_fixtures() -> None:
    async def scenario() -> None:
        for name in ("carry-only", "heterogeneous-carry", "list-backed-carry", "for-state"):
            issue408 = PROMPTS[name]
            Observations.reset()
            result = await make_engine().run(
                compile_prompt(issue408["api_prompt"]),
                issue408["outputs"],
                cache_enabled=False,
            )
            assert not [item for item in result.diagnostics if item.severity == "error"]
            assert {key: Observations.values[key] for key in issue408["observed"]} == issue408[
                "observed"
            ]

    asyncio.run(scenario())


def test_corpus_includes_the_runtime_expansion_prompt() -> None:
    graph = compile_prompt(PROMPTS["runtime-expanded-descendant"]["api_prompt"])
    expanded = graph.nodes["expand"]
    assert isinstance(expanded, GraphNode)
    assert expanded.node_type == "Issue408ExpandIncrement"
