"""Conditioning transforms preserve ComfyUI metadata and tensor behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.conditioning import (
    CONDITIONING_NODES,
    ConditioningAverage,
    ConditioningCombine,
    ConditioningConcat,
    ConditioningMultiply,
    ConditioningSetArea,
    ConditioningSetAreaPercentage,
    ConditioningSetAreaStrength,
    ConditioningSetMask,
    ConditioningSetTimestepRange,
    ConditioningZeroOut,
)


@pytest.fixture
def inference_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def set_mask(conditioning: object, mask: Any, area: str, strength: float) -> object:
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, 0)
        return [
            [
                text,
                {
                    **metadata,
                    "mask": mask,
                    "set_area_to_bounds": area != "default",
                    "mask_strength": strength,
                },
            ]
            for text, metadata in cast(list[list[Any]], conditioning)
        ]

    def set_timesteps(conditioning: object, timestep_range: tuple[float, float]) -> object:
        return [
            [
                text,
                {
                    **metadata,
                    "start_percent": timestep_range[0],
                    "end_percent": timestep_range[1],
                },
            ]
            for text, metadata in cast(list[list[Any]], conditioning)
        ]

    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.hooks",
        SimpleNamespace(
            set_mask_for_conditioning=set_mask, set_timesteps_for_conditioning=set_timesteps
        ),
    )


def test_registered_classes_have_exact_ids_atoms_and_ordered_interfaces() -> None:
    expected = (
        (
            "ConditioningAverage",
            "Conditioning (Average)",
            ("conditioning_to", "conditioning_from", "conditioning_to_strength"),
        ),
        ("ConditioningCombine", "Conditioning (Combine)", ("conditioning_1", "conditioning_2")),
        ("ConditioningConcat", "Conditioning (Concat)", ("conditioning_to", "conditioning_from")),
        ("ConditioningMultiply", "Conditioning (Multiply)", ("conditioning", "multiplier")),
        (
            "ConditioningSetArea",
            "Conditioning (Set Area)",
            ("conditioning", "width", "height", "x", "y", "strength"),
        ),
        (
            "ConditioningSetAreaPercentage",
            "Conditioning (Set Area with Percentage)",
            ("conditioning", "width", "height", "x", "y", "strength"),
        ),
        (
            "ConditioningSetAreaStrength",
            "Conditioning (Set Area Strength)",
            ("conditioning", "strength"),
        ),
        (
            "ConditioningSetMask",
            "Conditioning (Set Mask)",
            ("conditioning", "mask", "strength", "set_cond_area"),
        ),
        ("ConditioningZeroOut", "Conditioning Zero Out", ("conditioning",)),
        (
            "ConditioningSetTimestepRange",
            "ConditioningSetTimestepRange",
            ("conditioning", "start", "end"),
        ),
    )
    assert len(CONDITIONING_NODES) == 10
    for node, (node_type, display_name, inputs) in zip(CONDITIONING_NODES, expected, strict=True):
        schema = node.schema()
        assert schema.node_type == node_type
        assert (schema.display_name or schema.node_type) == display_name
        assert schema.category == "model/conditioning/transform"
        assert tuple(input_spec.id for input_spec in schema.inputs) == inputs
        assert tuple(output.type.types for output in schema.outputs) == (("CONDITIONING",),)
    mask_wire = schema_to_wire(ConditioningSetMask.schema())
    interface = cast(list[dict[str, object]], mask_wire["interface"])
    mask_input = next(item for item in interface if item["id"] == "set_cond_area")
    assert mask_input["widget"] == {"type": "COMBO", "options": ["default", "mask bounds"]}


def test_combine_appends_records_without_rewriting_either_side() -> None:
    left = [[np.array([[[1.0]]]), {"side": "left"}]]
    right = [[np.array([[[9.0]]]), {"side": "right"}]]
    combined = ConditioningCombine.execute(conditioning_1=left, conditioning_2=right)[
        "CONDITIONING"
    ]
    assert cast(list[object], combined)[0] is left[0]
    assert cast(list[object], combined)[1] is right[0]


def test_average_concat_and_multiply_preserve_metadata_without_mutating_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cat=lambda values, dim: np.concatenate(values, axis=dim),
            zeros=lambda shape: np.zeros(shape),
            mul=lambda value, strength: value * strength,
        ),
    )
    to = [[np.array([[[1.0], [3.0], [5.0]]]), {"pooled_output": np.array([[2.0]]), "keep": "to"}]]
    source = [[np.array([[[10.0]]]), {"pooled_output": np.array([[8.0]]), "keep": "from"}]]
    average = cast(
        list[list[Any]],
        ConditioningAverage.execute(
            conditioning_to=to, conditioning_from=source, conditioning_to_strength=0.25
        )["CONDITIONING"],
    )
    concat = cast(
        list[list[Any]],
        ConditioningConcat.execute(conditioning_to=to, conditioning_from=source)["CONDITIONING"],
    )
    multiplied = cast(
        list[list[Any]],
        ConditioningMultiply.execute(conditioning=to, multiplier=-2.0)["CONDITIONING"],
    )
    np.testing.assert_array_equal(average[0][0], np.array([[[7.75], [0.75], [1.25]]]))
    np.testing.assert_array_equal(average[0][1]["pooled_output"], np.array([[6.5]]))
    np.testing.assert_array_equal(concat[0][0], np.array([[[1.0], [3.0], [5.0], [10.0]]]))
    np.testing.assert_array_equal(multiplied[0][0], np.array([[[-2.0], [-6.0], [-10.0]]]))
    np.testing.assert_array_equal(multiplied[0][1]["pooled_output"], np.array([[-4.0]]))
    assert average[0][1]["keep"] == "to"
    np.testing.assert_array_equal(to[0][1]["pooled_output"], np.array([[2.0]]))
    np.testing.assert_array_equal(source[0][1]["pooled_output"], np.array([[8.0]]))


def test_area_strength_mask_and_timestep_transforms_copy_metadata_at_boundaries(
    inference_helpers: None,
) -> None:
    original = [[np.array([[[1.0]]]), {"preserve": {"nested": True}, "strength": 0.5}]]
    area = cast(
        list[list[Any]],
        ConditioningSetArea.execute(
            conditioning=original, width=79, height=66, x=17, y=31, strength=2.5
        )["CONDITIONING"],
    )
    percentage = cast(
        list[list[Any]],
        ConditioningSetAreaPercentage.execute(
            conditioning=original, width=0.75, height=0.25, x=0.5, y=0.0, strength=0.6
        )["CONDITIONING"],
    )
    strength = cast(
        list[list[Any]],
        ConditioningSetAreaStrength.execute(conditioning=original, strength=3.0)["CONDITIONING"],
    )
    mask = cast(
        list[list[Any]],
        ConditioningSetMask.execute(
            conditioning=original,
            mask=np.ones((2, 3)),
            strength=0.7,
            set_cond_area="mask bounds",
        )["CONDITIONING"],
    )
    timestep = cast(
        list[list[Any]],
        ConditioningSetTimestepRange.execute(conditioning=original, start=0.125, end=0.875)[
            "CONDITIONING"
        ],
    )
    assert area[0][1]["area"] == (8, 9, 3, 2)
    assert area[0][1]["set_area_to_bounds"] is False
    assert percentage[0][1]["area"] == ("percentage", 0.25, 0.75, 0.0, 0.5)
    assert strength[0][1]["strength"] == 3.0
    assert mask[0][1]["set_area_to_bounds"] is True
    assert mask[0][1]["mask_strength"] == 0.7
    assert mask[0][1]["mask"].shape == (1, 2, 3)
    assert (timestep[0][1]["start_percent"], timestep[0][1]["end_percent"]) == (0.125, 0.875)
    assert original[0][1] == {"preserve": {"nested": True}, "strength": 0.5}
    assert area[0][1]["preserve"] is original[0][1]["preserve"]


def test_zero_out_zeros_all_supported_payloads_and_leaves_other_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros_like=np.zeros_like))
    source = [
        [
            np.array([[[3.0, -4.0]]]),
            {
                "pooled_output": np.array([[2.0]]),
                "conditioning_lyrics": np.array([[1.0]]),
                "conditioning_scale": np.array([4.0]),
                "direct_context": np.array([5.0]),
                "keep": "metadata",
            },
        ]
    ]
    result = cast(list[list[Any]], ConditioningZeroOut.execute(conditioning=source)["CONDITIONING"])
    np.testing.assert_array_equal(result[0][0], np.zeros((1, 1, 2)))
    for key in ("pooled_output", "conditioning_lyrics", "conditioning_scale", "direct_context"):
        np.testing.assert_array_equal(result[0][1][key], np.zeros_like(source[0][1][key]))
    assert result[0][1]["keep"] == "metadata"
    np.testing.assert_array_equal(source[0][0], np.array([[[3.0, -4.0]]]))
