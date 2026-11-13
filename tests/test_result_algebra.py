from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any, cast

import pytest
from dinkster_protocol import (
    BlockedOutput,
    CurrentNodeRef,
    CurrentOutputRef,
    DirectReturn,
    ExpandedReturn,
    ExportSnapshot,
    Invocation,
    InvocationFailure,
    InvocationResult,
    InvocationReturn,
    JsonObject,
    LiteralInput,
    LocalExpansion,
    LocalNode,
    LocalNodeRef,
    LocalOutputRef,
    NodeError,
    PresentOutput,
    ReturnBatch,
)
from dinkster_schema import NodeSchema
from dinkster_values import ListPayload, TypeRegistry, register_core_types
from dinkster_workers import boundary
from dinkster_workers.boundary import (
    BoundaryError,
    ValueCodec,
    decode_invocation,
    decode_invocation_outcome,
    encode_invocation_outcome,
    encode_result,
)
from dinkster_workers.session import BoundarySession

from tests.async_scenarios import run_scenario


def registry() -> TypeRegistry:
    result = TypeRegistry()
    register_core_types(result)
    return result


def direct_outcome() -> InvocationReturn:
    value = registry().wrap("STRING", "payload")
    return InvocationReturn(
        ReturnBatch(
            "scalar",
            (
                DirectReturn(
                    (
                        ("a", PresentOutput(value)),
                        ("b", BlockedOutput(None)),
                        ("c", CurrentOutputRef("a")),
                    )
                ),
            ),
        )
    )


def expanded_outcome() -> InvocationReturn:
    value = registry().wrap("INT", 7)
    expansion = LocalExpansion(
        (
            LocalNode(
                "child",
                "Comfy.Node",
                (
                    (
                        "literal",
                        LiteralInput(JsonObject((("a", 1), ("b", (True, None))))),
                    ),
                    ("source", CurrentOutputRef("seed")),
                ),
                parent=CurrentNodeRef(),
                display=LocalNodeRef("display"),
            ),
            LocalNode("display", "Comfy.Display"),
        )
    )
    return InvocationReturn(
        ReturnBatch(
            "mapped",
            (
                ExpandedReturn(
                    expansion,
                    (
                        ("result", LocalOutputRef("child", "out")),
                        ("seed", PresentOutput(value)),
                    ),
                ),
            ),
        )
    )


def encode(outcome: InvocationReturn | InvocationFailure) -> tuple[dict[str, object], list[bytes]]:
    header, blobs, segments = encode_invocation_outcome(
        ValueCodec(registry(), use_shm=False),
        outcome,
        "inv-1",
        1.5,
    )
    assert segments == []
    return header, blobs


def decode(header: dict[str, object], blobs: list[bytes]) -> object:
    return decode_invocation_outcome(
        ValueCodec(registry(), use_shm=False),
        header,
        blobs,
        [],
    )[0]


def test_positional_tuple_result_roundtrip_and_envelope_validation() -> None:
    reg = registry()
    value = reg.wrap("tuple<INT,STRING>", (7, "label"))
    outcome = InvocationReturn(
        ReturnBatch("scalar", (DirectReturn((("out", PresentOutput(value)),)),))
    )
    header, blobs = encode(outcome)
    restored = decode(header, blobs)
    assert isinstance(restored, InvocationReturn)
    returned = restored.batch.units[0]
    assert isinstance(returned, DirectReturn)
    present = dict(returned.bindings)["out"]
    assert isinstance(present, PresentOutput)
    assert present.value.type_id == "tuple<INT,STRING>"
    assert present.value.resolve() == (7, "label")
    for type_id in ("tuple<STRING,INT>", "tuple<INT>"):
        changed = copy.deepcopy(header)
        algebra(changed)["values"][0]["typeId"] = type_id
        with pytest.raises(BoundaryError):
            decode(changed, blobs)


def test_heterogeneous_list_result_roundtrip_and_envelope_validation() -> None:
    reg = registry()
    value = reg.wrap(
        "list<*>",
        [reg.wrap("STRING", "seven"), reg.wrap("INT", 29), reg.wrap("list<INT>", [3, 8])],
    )
    outcome = InvocationReturn(
        ReturnBatch("scalar", (DirectReturn((("out", PresentOutput(value)),)),))
    )
    header, blobs = encode(outcome)
    restored = decode(header, blobs)
    assert isinstance(restored, InvocationReturn)
    returned = restored.batch.units[0]
    assert isinstance(returned, DirectReturn)
    present = dict(returned.bindings)["out"]
    assert isinstance(present, PresentOutput)
    assert present.value.type_id == "list<*>"
    assert present.value.resolve() == ["seven", 29, [3, 8]]
    assert isinstance(present.value.payload, ListPayload)
    assert [child.type_id for child in present.value.payload.children] == [
        "STRING",
        "INT",
        "list<INT>",
    ]
    changed = copy.deepcopy(header)
    algebra(changed)["values"][0]["typeId"] = "list<STRING>"
    with pytest.raises(BoundaryError, match="child type"):
        decode(changed, blobs)
    changed = copy.deepcopy(header)
    algebra(changed)["values"][0]["elements"][2]["elements"][0]["typeId"] = "STRING"
    with pytest.raises(BoundaryError, match="child type"):
        decode(changed, blobs)
    changed = copy.deepcopy(header)
    algebra(changed)["values"][0]["elements"][0]["unexpected"] = True
    with pytest.raises(BoundaryError, match="descriptor fields"):
        decode(changed, blobs)


def algebra(header: dict[str, object]) -> dict[str, Any]:
    return cast("dict[str, Any]", header["resultAlgebra"])


def replace_document(header: dict[str, object], document: object) -> dict[str, object]:
    changed = copy.deepcopy(header)
    algebra(changed)["document"] = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return changed


@pytest.mark.parametrize(
    "literal",
    [
        (),
        JsonObject(()),
        (("a", "b"),),
        JsonObject((("array", (JsonObject(()), (), (1, 2))),)),
        JsonObject((("$typed", JsonObject((("type", "list<INT>"), ("value", ())))),)),
    ],
)
def test_json_array_and_object_literals_round_trip_without_ambiguity(literal: object) -> None:
    outcome = InvocationReturn(
        ReturnBatch(
            "scalar",
            (
                ExpandedReturn(
                    LocalExpansion(
                        (
                            LocalNode(
                                "child", "Node", (("value", LiteralInput(cast(Any, literal))),)
                            ),
                        )
                    ),
                    (),
                ),
            ),
        )
    )
    header, blobs = encode(outcome)
    decoded = cast(InvocationReturn, decode(header, blobs))
    unit = cast(ExpandedReturn, decoded.batch.units[0])
    assert cast(LiteralInput, unit.expansion.nodes[0].inputs[0][1]).value == literal


def test_expanded_and_error_variants_round_trip() -> None:
    header, blobs = encode(expanded_outcome())
    decoded = cast(InvocationReturn, decode(header, blobs))
    unit = cast(ExpandedReturn, decoded.batch.units[0])
    assert unit.expansion.nodes[0].parent == CurrentNodeRef()
    assert unit.expansion.nodes[0].display == LocalNodeRef("display")
    assert cast(PresentOutput, unit.bindings[1][1]).value.resolve() == 7

    failure = InvocationFailure(NodeError("node", "Type", "failed"))
    error_header, error_blobs = encode(failure)
    assert error_blobs == []
    assert decode(error_header, error_blobs) == failure


def test_non_strict_invocation_header_is_a_node_error_without_session_death() -> None:
    reg = registry()
    session = BoundarySession(
        reg,
        role="test worker",
        pack="test",
        codec=ValueCodec(reg, use_shm=False),
    )
    session._alive = True  # noqa: SLF001 - exercise an adopted session without a peer
    session._writer = cast(Any, object())  # noqa: SLF001 - serialization fails first
    invocation = Invocation(
        invocation_id="invalid-json",
        node_id="node",
        node_type="Node",
        inputs={},
        effective_schema=NodeSchema("Node"),
        export_snapshot=ExportSnapshot(prompt={"value": float("nan")}),
    )

    async def scenario() -> object:
        result = await session.invoke(invocation)
        assert session.alive
        return result

    result = cast(InvocationResult, run_scenario(scenario()))
    assert result.error is not None
    assert result.error.message == "frame header is not strict JSON"


def test_failed_invocation_write_rolls_back_cas_for_retry() -> None:
    reg = registry()
    codec = ValueCodec(reg, use_shm=False, cas_threshold=1)
    codec.enable_cas()
    session = BoundarySession(reg, role="test worker", pack="test", codec=codec)
    session._alive = True  # noqa: SLF001 - exercise an adopted session without a peer
    session._writer = cast(Any, object())  # noqa: SLF001 - serialization fails first
    value = reg.wrap("STRING", "retained")
    invalid = Invocation(
        "invalid-json",
        "node",
        "Node",
        {"value": value},
        NodeSchema("Node"),
        export_snapshot=ExportSnapshot(prompt={"value": float("nan")}),
    )

    result = run_scenario(session.invoke(invalid))
    assert result.error is not None
    retry = Invocation("retry", "node", "Node", {"value": value}, NodeSchema("Node"))
    header, blobs, _, _ = boundary.encode_invocation(codec, retry)
    wire = cast(dict[str, Any], cast(dict[str, object], header["inputs"])["value"])
    assert "blob" in cast(dict[str, object], wire["payload"])
    peer = ValueCodec(reg, use_shm=False, cas_threshold=1)
    peer.enable_cas()
    decoded = decode_invocation(peer, header, blobs, [])
    assert decoded.inputs["value"].resolve() == "retained"


def test_expansion_output_replacement_is_preserved_on_the_wire() -> None:
    unit = expanded_outcome().batch.units[0]
    assert isinstance(unit, ExpandedReturn)
    result = InvocationResult(expansion=unit)
    updated = {"seed": registry().wrap("INT", 23)}
    result = replace(result, outputs=updated)
    updated["seed"] = registry().wrap("INT", 99)
    header, blobs, _ = encode_result(
        ValueCodec(registry(), use_shm=False),
        result,
        "expanded",
        0.0,
    )
    decoded, _ = boundary.decode_result(
        ValueCodec(registry(), use_shm=False),
        header,
        blobs,
        [],
    )
    assert decoded.outputs is not None
    assert decoded.outputs["seed"].resolve() == 23
    assert decoded.expansion is not None
    seed = dict(decoded.expansion.bindings)["seed"]
    assert isinstance(seed, PresentOutput)
    assert seed.value.resolve() == 23
