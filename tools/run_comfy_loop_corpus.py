"""Run recorded Generic Loops prompts through native StartLoop and EndLoop nodes."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_dev.loop_corpus import LOOP_CORPUS_NODES, Observations
from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

EXPECTED_DIVERGENCES = {"nested-cache-enabled": {"outer_probe_calls": [1, 2]}}
OUTPUTS = {
    "StartLoop": ("iteration_index", "is_first", "is_last", "list_item", "current_iteration_value"),
    "EndLoop": ("outputs",),
    "Issue408LazySwitch": ("output",),
}
TYPE_NAMES = {"Issue408LazySwitch": "ComfySwitchNode"}
RUN_TIMEOUT_SECONDS = 10


def _link(value: object) -> object:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(part, (str, int)) for part in value)
    ):
        return value
    node, output = value
    if not isinstance(node, str) or not isinstance(output, int):
        return value
    node_type = str(_active_prompt[node]["class_type"])
    return Link(node, OUTPUTS.get(node_type, ("output_0",))[output])


_active_prompt: dict[str, dict[str, object]]


def compile_prompt(prompt: MappingPrompt) -> Graph:
    """Compile the recorded prompt shape; this is test tooling, not an importer."""
    global _active_prompt
    _active_prompt = prompt
    nodes: dict[str, GraphNode] = {}
    for node_id, spec in prompt.items():
        source_type = str(spec["class_type"])
        inputs = {
            name: _link(value) for name, value in cast(dict[str, object], spec["inputs"]).items()
        }
        if source_type == "EndLoop":
            inputs = {
                f"terminations.{name.removeprefix('termination')}"
                if name.startswith("termination")
                else name: value
                for name, value in inputs.items()
            }
        node_type = TYPE_NAMES.get(source_type, source_type)
        variants: dict[str, str] = {}
        if source_type == "StartLoop":
            variants["mode"] = str(inputs.pop("mode", "simple"))
        elif source_type == "Issue408LazySwitch":
            inputs["switch"] = inputs.pop("switch")
        nodes[node_id] = GraphNode(node_type, inputs, slot_variants=variants)
    return Graph(nodes)


MappingPrompt = dict[str, dict[str, object]]


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    registry.register("core.value")
    nodes = (*COMFY_NODES, *FOUNDATION_NODES, *LOOP_CORPUS_NODES)
    return Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
        max_concurrency=1,
    )


async def run_corpus(root: Path) -> list[str]:
    outcomes = await collect_corpus(root)
    failures = [outcome for outcome in outcomes if outcome.status == "FAIL"]
    if failures:
        details = "\n".join(f"{outcome.name}: {outcome.detail}" for outcome in failures)
        raise AssertionError(details)
    return [outcome.name for outcome in outcomes if outcome.status == "EXACT"]


@dataclass(frozen=True)
class CorpusOutcome:
    name: str
    status: str
    detail: str


async def collect_corpus(root: Path) -> list[CorpusOutcome]:
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))["cases"]
    outcomes: list[CorpusOutcome] = []
    for case in results:
        workflow = json.loads((root / case["workflow"]).read_text(encoding="utf-8"))
        issue408 = workflow["extra"]["issue408"]
        name = case["case"]
        Observations.reset()
        engine = make_engine()
        graph = compile_prompt(cast(MappingPrompt, issue408["api_prompt"]))
        targets = cast(list[str], issue408["outputs"])
        try:
            for run in range(case["runs"]):
                result = await asyncio.wait_for(
                    engine.run(
                        graph,
                        targets,
                        cache_enabled=name != "cache-disabled",
                        run_id=f"{name}-{run}",
                    ),
                    timeout=RUN_TIMEOUT_SECONDS,
                )
                errors = [
                    diagnostic
                    for diagnostic in result.diagnostics
                    if diagnostic.severity == "error"
                ]
                if errors:
                    raise AssertionError(repr(errors))
            expected = dict(case["observed"])
            observed = {key: Observations.values[key] for key in expected}
            divergence = EXPECTED_DIVERGENCES.get(name)
            if divergence is not None:
                if observed != {**expected, **divergence}:
                    raise AssertionError(f"expected {expected!r}, got {observed!r}")
                outcomes.append(CorpusOutcome(name, "DIVERGENCE", repr(observed)))
            elif observed == expected:
                outcomes.append(CorpusOutcome(name, "EXACT", repr(observed)))
            else:
                outcomes.append(
                    CorpusOutcome(name, "FAIL", f"expected {expected!r}, got {observed!r}")
                )
        except Exception as exc:
            outcomes.append(CorpusOutcome(name, "FAIL", f"{type(exc).__name__}: {exc}"))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    outcomes = asyncio.run(collect_corpus(args.corpus))
    for outcome in outcomes:
        print(f"{outcome.status} {outcome.name}: {outcome.detail}")
    exact = sum(outcome.status == "EXACT" for outcome in outcomes)
    divergent = sum(outcome.status == "DIVERGENCE" for outcome in outcomes)
    failed = sum(outcome.status == "FAIL" for outcome in outcomes)
    print(f"SUMMARY exact={exact} divergence={divergent} failed={failed} total={len(outcomes)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
