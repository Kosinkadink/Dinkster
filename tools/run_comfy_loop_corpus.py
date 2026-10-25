"""Run recorded Generic Loops prompts through native StartLoop and EndLoop nodes."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinkster_api.v1 import CORE_INT
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode, RegionOutput
from dinkster_nodes_comfy.loops import LOOP_NODES
from dinkster_nodes_comfy.switch import ComfySwitchNode
from dinkster_nodes_dev.loop_corpus import LOOP_CORPUS_NODES, VALUE_TYPE, Observations
from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_schema import TypeExpr, build_node_types, build_schemas
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
INT = TypeExpr.concrete(CORE_INT)
VALUE = TypeExpr.concrete(VALUE_TYPE)


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
    type_cache: dict[str, TypeExpr] = {}
    nodes: dict[str, GraphNode | RegionNode] = {}
    generated = 0

    def reference_type(value: object) -> TypeExpr:
        if not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str):
            raise ValueError(f"corpus compiler expected a prompt link, got {value!r}")
        return output_type(value[0])

    def output_type(node_id: str) -> TypeExpr:
        cached = type_cache.get(node_id)
        if cached is not None:
            return cached
        source_type = str(prompt[node_id]["class_type"])
        if source_type in {
            "Issue408Constant",
            "Issue408Increment",
            "Issue408OuterProbe",
            "Issue408ExpandIncrement",
            "Issue408IndexIndependentProbe",
            "Issue408CapturePassthrough",
            "Issue408CapturePassthroughFirst",
            "Issue408CapturePassthroughSecond",
            "Issue408FalseBranch",
            "Issue408TrueBranch",
            "Issue408CaptureLoopState",
            "Issue408LazySwitch",
        }:
            result = INT
        elif source_type == "Issue408Pair":
            result = TypeExpr.list_of(VALUE)
        elif source_type in {
            "Issue408ListBackedScalar",
            "Issue408RecordCarried",
            "Issue408AppendIndex",
        }:
            result = VALUE
        elif source_type == "Issue408EmptyList":
            result = TypeExpr.list_of(VALUE)
        elif source_type == "Issue408IntegerList":
            result = TypeExpr.list_of(INT)
        elif source_type == "StartLoop":
            result = INT
        elif source_type == "EndLoop":
            source_inputs = cast(dict[str, object], prompt[node_id]["inputs"])
            output = source_inputs.get("output_value")
            if output is None:
                result = TypeExpr.list_of(VALUE)
            else:
                payload = reference_type(output)
                result = TypeExpr.list_of(payload)
        else:
            raise ValueError(f"corpus compiler does not know output type for {source_type}")
        type_cache[node_id] = result
        return result

    def generated_id(prefix: str) -> str:
        nonlocal generated
        generated += 1
        return f"{prefix}_{generated}"

    def flatten(link: Link, source_type: TypeExpr) -> Link:
        assert source_type.element is not None
        node_id = generated_id("flatten")
        nodes[node_id] = RegionNode(
            kind="map",
            body=Graph({}),
            ports={"item": source_type.element},
            inputs={"item": link},
            element_ports=("item",),
            outputs={"list": RegionOutput(Link(PORTS_NODE_ID, "item"), mode="flatten")},
        )
        return Link(node_id, "list")

    def last_element(link: Link) -> Link:
        node_id = generated_id("last")
        nodes[node_id] = GraphNode("std.list.element", {"list": link, "index": -1})
        return Link(node_id, "item")

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
            for name in ("output_value", "next_iteration_value"):
                original = cast(dict[str, object], spec["inputs"]).get(name)
                value = inputs.get(name)
                if not isinstance(value, Link) or original is None:
                    continue
                if (
                    name == "next_iteration_value"
                    and isinstance(original, list)
                    and str(prompt[original[0]]["class_type"]) == "EndLoop"
                ):
                    inputs[name] = last_element(value)
        node_type = TYPE_NAMES.get(source_type, source_type)
        variants: dict[str, str] = {}
        if source_type == "StartLoop":
            variants["mode"] = str(inputs.pop("mode", "simple"))
        elif source_type == "Issue408LazySwitch":
            inputs["switch"] = inputs.pop("switch")
        if source_type == "Issue408Capture" and isinstance(inputs.get("value"), Link):
            value = cast(Link, inputs["value"])
            value_type = reference_type(cast(dict[str, object], spec["inputs"])["value"])
            while (
                value_type.kind == "list"
                and value_type.element is not None
                and value_type.element.kind == "list"
            ):
                value = flatten(value, value_type)
                value_type = value_type.element
            if value_type.kind == "list" and value_type.element is not None:
                item_type = value_type.element
                nodes[node_id] = RegionNode(
                    kind="map",
                    body=Graph(
                        {"capture": GraphNode(node_type, {"value": Link(PORTS_NODE_ID, "item")})}
                    ),
                    ports={"item": item_type},
                    inputs={"item": value},
                    element_ports=("item",),
                    outputs={"outputs": RegionOutput(Link("capture", "output_0"))},
                )
                continue
        nodes[node_id] = GraphNode(node_type, inputs, slot_variants=variants)
    return Graph(nodes)


MappingPrompt = dict[str, dict[str, object]]


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    registry.register(VALUE_TYPE)
    nodes = (*LOOP_NODES, ComfySwitchNode, *FOUNDATION_NODES, *LOOP_CORPUS_NODES)
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
