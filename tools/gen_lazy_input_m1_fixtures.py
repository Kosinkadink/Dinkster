"""Generate the lazy-input contract fixtures.

The cases are a data oracle for the reviewed Dinkster contract, grounded in the
exact ComfyUI source files named below. The generator verifies those source
bytes from the pinned git object before writing; it does not import ComfyUI or
torch.

Run from the Dinkster repository root:

    .venv/bin/python tools/gen_lazy_input_m1_fixtures.py
    .venv/bin/python tools/gen_lazy_input_m1_fixtures.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "lazy_input_m1.json"
REFERENCE_COMMIT = "947c2749dd04c51ef0e21b069544d8b0b4f9b411"
REFERENCE_FILES = {
    "comfy_execution/graph.py": "772f5b0d300c71aa0036447c8cdf87f2efa2aa1a13acae0af889cd5c00d206fe",
    "execution.py": "1d43412cb45a779ae3859cf5eea59484ef0aeb671b7a8406a8bd0736740c3cab",
    "tests/execution/testing_nodes/testing-pack/specific_tests.py": (
        "6ac74cc1721a51de4b1f24f77392004dafdd8e110dfe1c54d8400bfdd2e60ed7"
    ),
    "tests/execution/test_execution.py": (
        "a53324f2b7430570fe3c42abc755f1247a6e7dd746c4ed2a42e64c85baed7457"
    ),
    "comfy_extras/nodes_logic.py": (
        "2cb1ce149ce616c0b12a67962ed7b799eb398fbcdeb20dcec4c5210eb6af6df1"
    ),
}
WHOLE_LIST_REFERENCE_COMMIT = "f4b99bc62389af315013dda85f24f2bbd262b686"
WHOLE_LIST_REFERENCE_FILES = {
    "execution.py": "b89508de4be872c4a0d44349bc18d53865faa66f63f5ee186de0640721c65cf9",
    "comfy_api/latest/_io.py": ("9ab1aaab9344d5cc86723ff02e0371f96aeefd0c444e9ad0616678cf9ac1d571"),
}


def _input(
    input_id: str,
    binding: str,
    *,
    lazy: bool,
    value: object = None,
    producer: str | None = None,
    producer_state: str = "cold",
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": input_id,
        "binding": binding,
        "lazy": lazy,
    }
    if binding != "omitted":
        result["value"] = value
    if producer is not None:
        result["producer"] = producer
        result["producerState"] = producer_state
    return result


def _case(
    case_id: str,
    covers: Sequence[str],
    inputs: Sequence[Mapping[str, object]],
    hook_rounds: Sequence[object],
    *,
    targets: Sequence[str] = ("lazy",),
    static_cycle: Sequence[str] = (),
    schedule: str = "serial",
    cache: str = "cold",
) -> dict[str, object]:
    return {
        "id": case_id,
        "covers": list(covers),
        "targets": list(targets),
        "schedule": schedule,
        "cache": cache,
        "inputs": [dict(item) for item in inputs],
        "hookRounds": list(hook_rounds),
        "staticCycle": list(static_cycle),
    }


def _cases() -> list[dict[str, object]]:
    selector = _input("selector", "literal", lazy=False, value=False)
    left = _input("left", "link", lazy=True, value="LEFT", producer="left_source")
    right = _input("right", "link", lazy=True, value="RIGHT", producer="right_source")
    return [
        _case("demand-none", ("none", "termination"), (selector, left, right), ((),)),
        _case(
            "demand-some",
            ("some", "termination"),
            (selector, left, right),
            (("left",), ()),
        ),
        _case(
            "demand-all",
            ("all", "termination"),
            (selector, left, right),
            (("left", "right"), ()),
        ),
        _case(
            "demand-multi-round",
            ("multi-round", "fixpoint"),
            (selector, left, right),
            (("left",), ("right",), ()),
        ),
        _case(
            "demand-repeated-satisfied",
            ("repeated", "satisfied", "termination"),
            (selector, left),
            (("left",), ("left",)),
        ),
        _case(
            "demand-duplicate-one-round",
            ("repeated", "normalization"),
            (selector, left),
            (("left", "left"), ()),
        ),
        _case(
            "request-invalid-item",
            ("invalid", "hook-error-attribution"),
            (selector, left),
            ((7,),),
        ),
        _case(
            "request-unknown-input",
            ("unknown", "hook-error-attribution"),
            (selector, left),
            (("ghost",),),
        ),
        _case(
            "request-literal-input",
            ("literal", "hook-error-attribution"),
            (selector, _input("literal_lazy", "literal", lazy=True, value=3)),
            (("literal_lazy",),),
        ),
        _case(
            "request-unconnected-input",
            ("unconnected", "hook-error-attribution"),
            (selector, _input("optional_lazy", "omitted", lazy=True)),
            (("optional_lazy",),),
        ),
        _case(
            "request-non-lazy-input",
            ("non-lazy", "hook-error-attribution"),
            (
                selector,
                _input("ordinary", "link", lazy=False, value="READY", producer="ordinary_source"),
            ),
            (("ordinary",),),
        ),
        _case(
            "computed-selector",
            ("linked-selector", "computed-selector"),
            (
                _input("selector", "link", lazy=False, value=True, producer="selector_source"),
                left,
                right,
            ),
            (("right",), ()),
        ),
        _case(
            "argument-tristate",
            ("connected-undemanded", "omitted", "default", "tri-state"),
            (
                left,
                _input("optional", "omitted", lazy=True),
                _input("defaulted", "default", lazy=True, value="DEFAULT"),
                _input("ordinary", "literal", lazy=False, value="ORDINARY"),
            ),
            ((),),
        ),
        _case(
            "static-cycle-undemanded",
            ("static-cycle", "validation"),
            (selector, left),
            ((),),
            static_cycle=("lazy", "left_source", "lazy"),
        ),
        _case(
            "hook-error",
            ("hook-error", "hook-error-attribution"),
            (selector, left),
            ({"error": "hook exploded"},),
        ),
        _case(
            "demanded-producer-error",
            ("producer-error", "producer-attribution"),
            (
                selector,
                _input(
                    "left",
                    "link",
                    lazy=True,
                    value="LEFT",
                    producer="left_source",
                    producer_state="error",
                ),
            ),
            (("left",),),
        ),
        _case(
            "cancellation-during-demand",
            ("cancellation", "demand"),
            (
                selector,
                _input(
                    "left",
                    "link",
                    lazy=True,
                    value="LEFT",
                    producer="left_source",
                    producer_state="cancel",
                ),
            ),
            (("left",),),
        ),
        _case(
            "partial-target-excludes-lazy",
            ("partial-target", "excluded"),
            (selector, left),
            ((),),
            targets=("other",),
        ),
        _case(
            "partial-target-includes-lazy",
            ("partial-target", "included"),
            (selector, left),
            (("left",), ()),
            targets=("lazy",),
        ),
        _case(
            "partial-target-mixed",
            ("partial-target", "multiple"),
            (selector, left),
            (("left",), ()),
            targets=("other", "lazy"),
        ),
        _case(
            "determinism-serial-cold",
            ("deterministic-visibility", "serial", "cold-cache"),
            (selector, left),
            (("left",), ()),
            schedule="serial",
            cache="cold",
        ),
        _case(
            "determinism-parallel-cold",
            ("deterministic-visibility", "parallel", "cold-cache"),
            (
                selector,
                _input(
                    "left",
                    "link",
                    lazy=True,
                    value="LEFT",
                    producer="left_source",
                    producer_state="incidental",
                ),
            ),
            (("left",), ()),
            schedule="parallel",
            cache="cold",
        ),
        _case(
            "determinism-serial-warm",
            ("deterministic-visibility", "serial", "warm-cache"),
            (
                selector,
                _input(
                    "left",
                    "link",
                    lazy=True,
                    value="LEFT",
                    producer="left_source",
                    producer_state="warm",
                ),
            ),
            (("left",), ()),
            schedule="serial",
            cache="warm",
        ),
        _case(
            "determinism-parallel-warm",
            ("deterministic-visibility", "parallel", "warm-cache"),
            (
                selector,
                _input(
                    "left",
                    "link",
                    lazy=True,
                    value="LEFT",
                    producer="left_source",
                    producer_state="warm",
                ),
            ),
            (("left",), ()),
            schedule="parallel",
            cache="warm",
        ),
    ]


def _visible_inputs(
    inputs: Sequence[Mapping[str, object]], demanded: set[str]
) -> dict[str, object]:
    visible: dict[str, object] = {}
    for item in inputs:
        input_id = cast(str, item["id"])
        binding = item["binding"]
        if binding == "omitted":
            continue
        if item["lazy"] and binding == "link" and input_id not in demanded:
            visible[input_id] = None
        else:
            visible[input_id] = item.get("value")
    return visible


def _normalize_requested(
    raw: Sequence[object], inputs: Sequence[Mapping[str, object]]
) -> tuple[list[str], tuple[str, str] | None]:
    by_id = {cast(str, item["id"]): item for item in inputs}
    for item in raw:
        if not isinstance(item, str):
            return [], ("lazy-request-invalid", "lazy")
        if item not in by_id:
            return [], ("lazy-request-unknown-input", "lazy")
        spec = by_id[item]
        if not spec["lazy"]:
            return [], ("lazy-request-non-lazy", "lazy")
        if spec["binding"] == "literal":
            return [], ("lazy-request-literal", "lazy")
        if spec["binding"] in {"omitted", "default"}:
            return [], ("lazy-request-unconnected", "lazy")
    raw_set = set(cast("Sequence[str]", raw))
    normalized = [cast(str, item["id"]) for item in inputs if item["id"] in raw_set]
    return normalized, None


def _oracle(case: Mapping[str, object]) -> dict[str, object]:
    targets = cast("list[str]", case["targets"])
    if "lazy" not in targets:
        return {
            "outcome": "not-reached",
            "demandedInputs": [],
            "executedProducers": [],
            "reusedProducers": [],
            "hookViews": [],
            "demandEvents": [],
        }
    static_cycle = cast("list[str]", case["staticCycle"])
    if static_cycle:
        return {
            "outcome": "validation-error",
            "errorCode": "cycle",
            "errorNode": None,
            "demandedInputs": [],
            "executedProducers": [],
            "reusedProducers": [],
            "hookViews": [],
            "demandEvents": [],
        }

    inputs = cast("list[dict[str, object]]", case["inputs"])
    by_id = {cast(str, item["id"]): item for item in inputs}
    demanded: set[str] = set()
    executed: list[str] = []
    reused: list[str] = []
    views: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    outcome = "execute"
    error_code: str | None = None
    error_node: str | None = None

    for round_index, round_result in enumerate(cast("list[object]", case["hookRounds"]), 1):
        views.append(_visible_inputs(inputs, demanded))
        if isinstance(round_result, Mapping):
            outcome = "error"
            error_code = "lazy-hook-failed"
            error_node = "lazy"
            break
        requested, problem = _normalize_requested(cast("Sequence[object]", round_result), inputs)
        if problem is not None:
            outcome = "error"
            error_code, error_node = problem
            break
        new = [input_id for input_id in requested if input_id not in demanded]
        demanded.update(new)
        producer_nodes = sorted(
            {
                cast(str, by_id[input_id]["producer"])
                for input_id in new
                if "producer" in by_id[input_id]
            }
        )
        events.append(
            {
                "name": "lazy_demand",
                "data": {
                    "round": round_index,
                    "status": "waiting" if new else "ready",
                    "requestedInputs": requested,
                    "newInputs": new,
                    "demandedInputs": [
                        cast(str, item["id"]) for item in inputs if item["id"] in demanded
                    ],
                    "producerNodes": producer_nodes,
                },
            }
        )
        if not new:
            break
        for input_id in new:
            item = by_id[input_id]
            producer = cast(str, item["producer"])
            state = cast(str, item.get("producerState", "cold"))
            if state == "error":
                outcome = "error"
                error_code = "producer-failed"
                error_node = producer
                break
            if state == "cancel":
                outcome = "cancelled"
                break
            if state in {"warm", "incidental"}:
                if producer not in reused:
                    reused.append(producer)
            elif producer not in executed:
                executed.append(producer)
        if outcome != "execute":
            break

    result: dict[str, object] = {
        "outcome": outcome,
        "demandedInputs": [cast(str, item["id"]) for item in inputs if item["id"] in demanded],
        "executedProducers": executed,
        "reusedProducers": reused,
        "hookViews": views,
        "demandEvents": events,
    }
    if error_code is not None:
        result["errorCode"] = error_code
        result["errorNode"] = error_node
    return result


def build_fixture() -> dict[str, object]:
    cases = _cases()
    for case in cases:
        case["expected"] = _oracle(case)
    return {
        "_meta": {
            "format": 1,
            "generator": "tools/gen_lazy_input_m1_fixtures.py",
            "contract": "docs/lazy-input-m1-contract.md",
            "referenceCommit": REFERENCE_COMMIT,
            "referenceFiles": REFERENCE_FILES,
            "wholeListReferenceCommit": WHOLE_LIST_REFERENCE_COMMIT,
            "wholeListReferenceFiles": WHOLE_LIST_REFERENCE_FILES,
        },
        "currentRuntime": {
            "generalLazyDisposition": "supported scalar and declared whole-list lazy semantics",
            "computedSelectorProblem": None,
            "engineImplementation": True,
            "wholeListProjection": {
                "connectedUndemanded": [None],
                "demanded": ["FIRST", "SECOND"],
                "unconnected": "omitted",
            },
        },
        "cases": cases,
        "deferred": [
            {"id": "async-hook", "milestone": "M1-follow-up"},
            {"id": "list-mapping", "milestone": "M3"},
            {"id": "blocker-in-list", "milestone": "M3"},
            {"id": "is-changed", "milestone": "M2"},
            {"id": "raw-link", "milestone": "M3"},
            {"id": "accept-all", "milestone": "M3"},
            {"id": "dynamic-expansion", "milestone": "M4"},
            {"id": "native-control-flow-prediction-speculation", "milestone": "M5"},
        ],
    }


def render_fixture() -> bytes:
    return (json.dumps(build_fixture(), separators=(",", ":")) + "\n").encode("ascii")


def _git_bytes(comfy_root: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(comfy_root), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def _verify_reference(comfy_root: Path, commit: str, files: Mapping[str, str]) -> None:
    resolved = subprocess.run(
        ["git", "-C", str(comfy_root), "rev-parse", commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != commit:
        raise SystemExit(f"reference commit resolved to {resolved}, expected {commit}")
    for path, expected in files.items():
        actual = hashlib.sha256(_git_bytes(comfy_root, commit, path)).hexdigest()
        if actual != expected:
            raise SystemExit(f"{path} sha256 {actual}, expected {expected}")


def verify_reference(comfy_root: Path) -> None:
    _verify_reference(comfy_root, REFERENCE_COMMIT, REFERENCE_FILES)
    _verify_reference(comfy_root, WHOLE_LIST_REFERENCE_COMMIT, WHOLE_LIST_REFERENCE_FILES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, default=REPO.parent / "ComfyUI")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_reference(args.comfyui.resolve())
    rendered = render_fixture()
    if args.check:
        if not OUT.exists() or OUT.read_bytes() != rendered:
            raise SystemExit(f"{OUT} does not match the deterministic generator")
        print(f"verified {OUT}")
        return
    OUT.write_bytes(rendered)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
