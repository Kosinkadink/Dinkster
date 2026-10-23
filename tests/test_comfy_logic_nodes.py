"""Native eager logic preserves Python truthiness and ordered short circuit."""

from __future__ import annotations

import pytest
from dinkster_nodes_comfy.logic import ComfyAndNode, ComfyNotNode, ComfyOrNode


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (0, True), ([], True), ({}, True), ("False", False), (-3, False), ([0], False)],
)
def test_not_uses_truthiness(value: object, expected: bool) -> None:
    assert ComfyNotNode.execute(value=value) == {"BOOLEAN": expected}


@pytest.mark.parametrize(
    ("values", "conjunction", "disjunction"),
    [
        ({}, True, False),
        ({"value0": "False", "value1": -3}, True, True),
        ({"value0": [0], "value1": []}, False, True),
        ({"value0": {}, "value1": 0}, False, False),
        ({"value0": False, "value1": True, "value2": False}, False, True),
    ],
)
def test_mapping_values_determine_truth(
    values: dict[str, object], conjunction: bool, disjunction: bool
) -> None:
    assert ComfyAndNode.execute(values=values) == {"BOOLEAN": conjunction}
    assert ComfyOrNode.execute(values=values) == {"BOOLEAN": disjunction}


class InvalidTruth:
    def __bool__(self) -> bool:
        raise ValueError("truth requested")


def test_mapping_iteration_short_circuits_in_insertion_order() -> None:
    invalid = InvalidTruth()
    assert ComfyAndNode.execute(values={"value9": 0, "value0": invalid}) == {"BOOLEAN": False}
    assert ComfyOrNode.execute(values={"value9": [0], "value0": invalid}) == {"BOOLEAN": True}
    with pytest.raises(ValueError, match="truth requested"):
        ComfyAndNode.execute(values={"value9": invalid, "value0": 0})
    with pytest.raises(ValueError, match="truth requested"):
        ComfyOrNode.execute(values={"value9": invalid, "value0": [0]})
    with pytest.raises(ValueError, match="truth requested"):
        ComfyNotNode.execute(value=invalid)
