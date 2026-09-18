"""Canonical real-engine preview and partial-execution conformance proofs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from dinkster_graph import Graph, GraphNode, GraphWireError, graph_from_wire

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "tests" / "goldens" / "preview_partial_execution.json"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_preview_partial_execution_golden",
        REPO_ROOT / "scripts" / "generate_preview_partial_execution_golden.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preview_partial_execution_golden_is_real_engine_authored() -> None:
    generator = _generator()
    on_disk = GOLDEN.read_bytes()
    assert on_disk == generator.render_golden()
    contract = json.loads(on_disk)
    assert contract["formatVersion"] == 1
    assert [case["id"] for case in contract["cases"]] == [
        "selected-target-closure",
        "effectful-rerun",
        "absence-policies",
        "region-list-inspection",
        "preview-and-terminal-paths",
    ]
    assert all(case["targetNodeIds"] for case in contract["cases"])
    assert all(case["graphDigest"].startswith("sha256:") for case in contract["cases"])


def test_every_case_phase_graph_digest_target_and_inspection_is_self_consistent() -> None:
    contract = json.loads(GOLDEN.read_bytes())
    for case in contract["cases"]:
        for phase in case["phases"].values():
            graph_wire = phase.get("graph", case["graph"])
            digest = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        graph_wire,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest()
            )
            assert digest == phase.get("graphDigest", case["graphDigest"])
            graph = graph_from_wire(graph_wire)
            targets = phase.get("targetNodeIds", case["targetNodeIds"])
            assert targets and all(target in graph.nodes for target in targets)
            selections = phase.get("inspectSelections", case["inspectSelections"])
            assert all(selection["nodeId"] in graph.nodes for selection in selections)


def test_selected_internal_sinks_use_only_node_targets_and_execute_shared_ancestor_once() -> None:
    contract = json.loads(GOLDEN.read_bytes())
    case = next(case for case in contract["cases"] if case["id"] == "selected-target-closure")
    assert case["targetNodeIds"] == ["split", "left", "right"]
    assert case["inspectSelections"] == [
        {"nodeId": "split", "outputId": "left"},
        {"nodeId": "split", "outputId": "right"},
        {"nodeId": "left", "outputId": "value"},
        {"nodeId": "right", "outputId": "value"},
    ]
    cold = case["phases"]["cold"]
    assert cold["expectedExecuted"] == ["left", "right", "source", "split"]
    assert len(cold["expectedExecuted"]) == len(set(cold["expectedExecuted"]))
    split_inspections = [
        selection for selection in cold["postRunInspection"] if selection["nodeId"] == "split"
    ]
    assert all(selection["descriptor"] is not None for selection in split_inspections)
    assert "downstream" not in cold["expectedExecuted"]
    assert "unselectedEffect" not in cold["expectedExecuted"]
    assert case["phases"]["warm"]["expectedExecuted"] == []
    assert case["phases"]["warm"]["expectedCached"] == [
        "left",
        "right",
        "source",
        "split",
    ]
    assert case["phases"]["inputMutated"]["expectedExecuted"] == [
        "left",
        "right",
        "source",
        "split",
    ]


def test_effect_absence_region_preview_failure_and_cancellation_contract() -> None:
    cases = {case["id"]: case for case in json.loads(GOLDEN.read_bytes())["cases"]}
    effect = cases["effectful-rerun"]["phases"]
    assert "effect" in effect["first"]["expectedExecuted"]
    assert effect["second"]["expectedExecuted"] == ["effect"]
    assert effect["second"]["expectedCached"] == ["source"]

    absence = cases["absence-policies"]["phases"]
    assert absence["defaultSkipAndOmit"]["expectedSkipped"] == ["skip"]
    assert absence["defaultSkipAndOmit"]["outputs"]["omit"]["value"]["value"] == 11
    assert absence["fail"]["terminalState"] == "failed"

    region = cases["region-list-inspection"]["phases"]["run"]
    assert region["exactElementInspection"]["value"] == 5
    assert cases["region-list-inspection"]["targetNodeIds"] == ["mapped"]

    terminal = cases["preview-and-terminal-paths"]["phases"]
    assert [frame["nodeId"] for frame in terminal["preview"]["runtimePreviews"]] == [
        "preview",
        "preview",
    ]
    assert terminal["preview"]["renditions"] == [
        {"default": True, "kind": "png", "mime": "image/png"}
    ]
    assert terminal["preview"]["fallbackRenditions"] == []
    assert terminal["stableFailure"]["terminalState"] == "failed"
    assert terminal["cancelled"]["resourceInUseBeforeCancel"] == 1
    assert terminal["cancelled"]["resourceInUseAfterCancel"] == 0


def test_malformed_graph_refuses_before_any_engine_submission() -> None:
    malformed = {"nodes": {"bad": {"inputs": {}}}}
    try:
        graph_from_wire(malformed)
    except GraphWireError as exc:
        assert "nodeType" in str(exc)
    else:
        raise AssertionError("malformed graph wire decoded")

    # Output focus is not representable in the graph or target list. The
    # native submission remains a graph plus top-level node ids.
    graph = Graph({"n": GraphNode("dev.conformance.source", {"value": 1})})
    assert graph.nodes["n"].inputs == {"value": 1}
