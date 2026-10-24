"""The recorded Generic Loops corpus runs through native loop expansion only."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dinkster_graph import GraphNode
from dinkster_nodes_dev.loop_corpus import Observations

from tools.run_comfy_loop_corpus import compile_prompt, make_engine

CORPUS = Path("/home/kosin/comfy-vibe-station/dinkster-evidence/comfyui-generic-loops")


def test_recorded_prompts_compile_to_native_loop_nodes() -> None:
    workflow = json.loads((CORPUS / "workflows/nested-carry.json").read_text(encoding="utf-8"))
    graph = compile_prompt(workflow["extra"]["issue408"]["api_prompt"])
    outer = graph.nodes["outer"]
    inner_close = graph.nodes["inner_close"]
    assert isinstance(outer, GraphNode)
    assert isinstance(inner_close, GraphNode)
    assert outer.node_type == "StartLoop"
    assert inner_close.node_type == "EndLoop"
    assert all(isinstance(node, GraphNode) for node in graph.nodes.values())


def test_native_loop_compiler_runs_carried_value_fixtures() -> None:
    async def scenario() -> None:
        for name in ("carry-only", "heterogeneous-carry", "list-backed-carry"):
            workflow = json.loads((CORPUS / f"workflows/{name}.json").read_text(encoding="utf-8"))
            issue408 = workflow["extra"]["issue408"]
            Observations.reset()
            result = await make_engine().run(
                compile_prompt(issue408["api_prompt"]),
                issue408["outputs"],
                cache_enabled=False,
            )
            assert not [item for item in result.diagnostics if item.severity == "error"]
            assert {
                key: Observations.values[key] for key in issue408["recorded_cpu_result"]["observed"]
            } == issue408["recorded_cpu_result"]["observed"]

    asyncio.run(scenario())


def test_corpus_includes_the_runtime_expansion_prompt() -> None:
    workflow = json.loads(
        (CORPUS / "workflows/runtime-expanded-descendant.json").read_text(encoding="utf-8")
    )
    graph = compile_prompt(workflow["extra"]["issue408"]["api_prompt"])
    expanded = graph.nodes["expand"]
    assert isinstance(expanded, GraphNode)
    assert expanded.node_type == "Issue408ExpandIncrement"
