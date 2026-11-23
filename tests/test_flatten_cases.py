"""The flattening cases the editor checks too (Dinkster-Frontend
``packages/core/test/fixtures/flatten_v2.json``), so the engine and the editor
give one node the same flattened ids."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_graph.model import Link, is_link, literal_value
from dinkster_schema.automatic import work_out_automatic
from dinkster_schema.elaborate import elaborate
from dinkster_schema.model import NodeSchema
from dinkster_schema.wire import SCHEMA_WIRE_VERSION, body_row_from_wire

CASES = Path(__file__).parent / "fixtures" / f"flatten_v{SCHEMA_WIRE_VERSION}.json"
CASES_SHA256 = "b864f2d8adfb2cddf9a050f097945319cb4414d0f399b781fd3e7d241b7d7d23"


def _cases() -> list[Mapping[str, Any]]:
    return cast("list[Mapping[str, Any]]", json.loads(CASES.read_text("ascii"))["cases"])


def test_case_file_is_the_one_both_sides_check() -> None:
    assert hashlib.sha256(CASES.read_bytes()).hexdigest() == CASES_SHA256


def test_case_file_matches_its_generator() -> None:
    from tools.generate_flatten_cases import render

    assert CASES.read_text("ascii") == render()


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["id"]))
def test_engine_flattens_each_case_as_written(case: Mapping[str, Any]) -> None:
    body = tuple(body_row_from_wire(row, SCHEMA_WIRE_VERSION) for row in case["body"])
    if case["outcome"] == "refused-at-load":
        with pytest.raises(ValueError, match=_literal(case["message"])):
            NodeSchema("cases.node", body=body)  # type: ignore[arg-type]
        return
    schema = NodeSchema("cases.node", body=body)  # type: ignore[arg-type]
    stored = {
        key: Link("source", "out") if value == {"link": True} else value
        for key, value in cast("list[tuple[str, object]]", case["stored"])
    }
    members = {path: tuple(keys) for path, keys in case["members"].items()}

    def run() -> NodeSchema:
        return elaborate(
            schema, stored, members, dict(case["choices"]), linked=is_link, literal=literal_value
        )

    if case["outcome"] == "refused":
        with pytest.raises(ValueError, match=_literal(case["message"])):
            run()
        return
    flat = run()
    assert [
        [spec.id, spec.type.runtime_type_id(), spec.required, spec.default] for spec in flat.inputs
    ] == case["inputs"]
    assert [spec.id for spec in flat.outputs] == case["outputs"]
    assert dict(flat.slot_choices) == case["chose"]
    if "automatic" in case:
        outcomes = work_out_automatic(
            schema.body,
            stored,
            members,
            dict(case["choices"]),
            tables=schema.tables,
            derived=schema.derived,
            linked=is_link,
            literal=literal_value,
        )
        assert {
            flat_id: [outcome.value, "worked-out" if outcome.worked_out else "default"]
            for flat_id, outcome in outcomes.items()
        } == case["automatic"]


def _literal(text: str) -> str:
    import re

    return re.escape(text)
