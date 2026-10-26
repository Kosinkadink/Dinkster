"""Conditioning transforms preserve ComfyUI metadata and tensor behavior."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr, schema_to_wire
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.conditioning import (
    CONDITIONING_NODES,
    ConditioningAverage,
    ConditioningCombine,
    ConditioningConcat,
    ConditioningMultiply,
    ConditioningSetArea,
    ConditioningSetAreaPercentage,
    ConditioningSetAreaPercentageVideo,
    ConditioningSetAreaStrength,
    ConditioningSetMask,
    ConditioningSetTimestepRange,
    ConditioningStableAudio,
    ConditioningZeroOut,
    conditioning_set_values,
    conditioning_set_values_with_timestep_range,
)

from dinkster.compose import PackSpec, ServingComposer


def test_conditioning_metadata_append_preserves_sources_and_default_overwrite() -> None:
    text = object()
    existing = [object()]
    incoming = [object(), object()]
    source = [[text, {"refs": existing, "unset": None, "count": 3, "keep": "unchanged"}]]
    appended = conditioning_set_values(
        source, {"refs": incoming, "unset": incoming, "count": 7}, append=True
    )
    assert appended[0][0] is text
    assert appended[0][1] == {
        "refs": existing + incoming,
        "unset": incoming,
        "count": 10,
        "keep": "unchanged",
    }
    assert source[0][1]["refs"] is existing
    assert len(existing) == 1
    assert source[0][1]["unset"] is None
    assert source[0][1]["count"] == 3
    replaced = conditioning_set_values(source, {"refs": incoming})
    assert replaced[0][1]["refs"] is incoming
    assert conditioning_set_values([[text, {"refs": (1,)}]], {"refs": (2, 3)}, append=True)[0][1][
        "refs"
    ] == (1, 2, 3)


def test_conditioning_timestep_values_intersect_and_preserve_outside_intervals() -> None:
    text = object()
    source = [[text, {"start_percent": 0.1, "end_percent": 0.9, "keep": "source"}]]
    result = conditioning_set_values_with_timestep_range(source, {"control": "map"}, 0.25, 0.6)
    assert all(entry[0] is text for entry in result)
    assert [entry[1] for entry in result] == [
        {"start_percent": 0.1, "end_percent": 0.25 - 1e-5, "keep": "source"},
        {"start_percent": 0.25, "end_percent": 0.6, "keep": "source", "control": "map"},
        {"start_percent": 0.6 + 1e-5, "end_percent": 0.9, "keep": "source"},
    ]
    assert source[0][1] == {"start_percent": 0.1, "end_percent": 0.9, "keep": "source"}
    clipped = conditioning_set_values_with_timestep_range(source, {"control": "map"}, 0.0, 0.4)
    assert clipped[0][1] == {
        "start_percent": 0.1,
        "end_percent": 0.4,
        "keep": "source",
        "control": "map",
    }
    for start, end in ((0.9, 1.0), (0.4, 0.4)):
        assert (
            conditioning_set_values_with_timestep_range(source, {"control": "map"}, start, end)[0]
            is source[0]
        )
    assert conditioning_set_values_with_timestep_range(source, {}, 0.7, 0.2) is source


class MaskTensor:
    def __init__(self, value: object) -> None:
        self.array = np.asarray(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def unsqueeze(self, dim: int) -> MaskTensor:
        return MaskTensor(np.expand_dims(self.array, dim))

    def detach(self) -> MaskTensor:
        return self


@pytest.fixture
def inference_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(as_tensor=MaskTensor))

    def set_mask(conditioning: object, mask: Any, area: str, strength: float) -> object:
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
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
            "ConditioningSetAreaPercentageVideo",
            "Conditioning (Set Area with Percentage for Video)",
            ("conditioning", "width", "height", "temporal", "x", "y", "z", "strength"),
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
        (
            "ConditioningStableAudio",
            "ConditioningStableAudio",
            ("positive", "negative", "seconds_start", "seconds_total"),
        ),
    )
    assert len(CONDITIONING_NODES) == 12
    for node, (node_type, display_name, inputs) in zip(CONDITIONING_NODES, expected, strict=True):
        schema = node.schema()
        assert schema.node_type == node_type
        assert (schema.display_name or schema.node_type) == display_name
        assert schema.category == (
            "model/conditioning/stable audio"
            if node_type == "ConditioningStableAudio"
            else "model/conditioning/transform"
        )
        assert tuple(input_spec.id for input_spec in schema.inputs) == inputs
        assert all(output.type.types == ("CONDITIONING",) for output in schema.outputs)
        assert tuple(output.id for output in schema.outputs) == (
            ("positive", "negative")
            if node_type == "ConditioningStableAudio"
            else ("CONDITIONING",)
        )
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


def test_audio_timing_and_video_percentage_area_preserve_sources() -> None:
    positive = [[np.array([[[2.0, -7.0]]]), {"tag": "positive", "seconds_total": 999.0}]]
    negative = [[np.array([[[5.0, -3.0]]]), {"tag": "negative", "seconds_start": 100.0}]]
    timed = cast(
        dict[str, list[list[Any]]],
        ConditioningStableAudio.execute(
            positive=positive, negative=negative, seconds_start=2.25, seconds_total=37.5
        ),
    )
    for name, source in (("positive", positive), ("negative", negative)):
        result = timed[name]
        assert result[0][0] is source[0][0]
        assert result[0][1] == {"tag": name, "seconds_start": 2.25, "seconds_total": 37.5}
        assert result[0][1] is not source[0][1]
    area = cast(
        list[list[Any]],
        ConditioningSetAreaPercentageVideo.execute(
            conditioning=timed["positive"],
            width=0.75,
            height=0.5,
            temporal=0.25,
            x=0.125,
            y=0.375,
            z=0.625,
            strength=2.5,
        )["CONDITIONING"],
    )
    assert area[0][0] is positive[0][0]
    assert area[0][1] == {
        "tag": "positive",
        "seconds_start": 2.25,
        "seconds_total": 37.5,
        "area": ("percentage", 0.25, 0.5, 0.75, 0.625, 0.375, 0.125),
        "strength": 2.5,
        "set_area_to_bounds": False,
    }
    assert "area" not in timed["positive"][0][1]
    assert positive[0][1] == {"tag": "positive", "seconds_total": 999.0}
    assert negative[0][1] == {"tag": "negative", "seconds_start": 100.0}
    assert ConditioningStableAudio.execute(
        positive=[], negative=[], seconds_start=0.0, seconds_total=0.0
    ) == {"positive": [], "negative": []}


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


class ConditioningScalarSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="conditioning-fixture.source",
            outputs=(OutputSpec("conditioning", TypeExpr.concrete("CONDITIONING")),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return {
            "conditioning": [
                [
                    np.array([[3.0, -5.0]], dtype=np.float32),
                    {"strength": 0.25, "tag": "source"},
                ]
            ]
        }


class ConditioningScalarInspector(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="conditioning-fixture.inspect",
            inputs=(InputSpec("conditioning", TypeExpr.concrete("CONDITIONING")),),
            outputs=(
                OutputSpec("payload", TypeExpr.concrete("FLOAT")),
                OutputSpec("strength", TypeExpr.concrete("FLOAT")),
            ),
        )

    @classmethod
    def execute(cls, conditioning: object) -> dict[str, object]:
        record = cast(list[list[Any]], conditioning)[0]
        return {"payload": float(record[0][0][1]), "strength": float(record[1]["strength"])}


@pytest.mark.parametrize("in_process", (True, False))
def test_manifest_links_raw_conditioning_through_the_resident_codec_without_host_types(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = (
                Path(__file__).resolve().parents[1]
                / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
            )
            (tmp_path / "conditioning_fixture_sources.py").write_text(
                "from test_comfy_conditioning_nodes import ConditioningScalarInspector\n"
                "from test_comfy_conditioning_nodes import ConditioningScalarSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ConditioningScalarSource, ConditioningScalarInspector)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["conditioning-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "conditioning_fixture_sources:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "conditioning_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            nodes = {
                "source": GraphNode("conditioning-fixture.source"),
                "source_inspect": GraphNode(
                    "conditioning-fixture.inspect",
                    {"conditioning": Link("source", "conditioning")},
                ),
                "strength": GraphNode(
                    "ConditioningSetAreaStrength",
                    {
                        "conditioning": Link("source", "conditioning"),
                        "strength": 3.5,
                    },
                ),
                "strength_inspect": GraphNode(
                    "conditioning-fixture.inspect",
                    {"conditioning": Link("strength", "CONDITIONING")},
                ),
            }
            outputs = ["source_inspect", "strength_inspect"]
            if in_process:
                outputs.append("source")
            else:
                nodes["low_strength"] = GraphNode(
                    "ConditioningSetAreaStrength",
                    {
                        "conditioning": Link("source", "conditioning"),
                        "strength": 0.125,
                    },
                )
                nodes["low_strength_inspect"] = GraphNode(
                    "conditioning-fixture.inspect",
                    {"conditioning": Link("low_strength", "CONDITIONING")},
                )
                outputs.append("low_strength_inspect")
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(nodes=nodes),
                outputs,
            )
            strength_output = result.outputs["strength_inspect"]
            assert strength_output["payload"].resolve() == -5.0
            assert strength_output["strength"].resolve() == 3.5
            source_output = result.outputs["source_inspect"]
            assert source_output["payload"].resolve() == -5.0
            assert source_output["strength"].resolve() == 0.25
            if in_process:
                source = cast(list[list[Any]], result.outputs["source"]["conditioning"].resolve())
                np.testing.assert_array_equal(
                    source[0][0], np.array([[3.0, -5.0]], dtype=np.float32)
                )
                assert source[0][1] == {"strength": 0.25, "tag": "source"}
            else:
                low_strength_output = result.outputs["low_strength_inspect"]
                assert low_strength_output["payload"].resolve() == -5.0
                assert low_strength_output["strength"].resolve() == 0.125
        finally:
            await composer.close()

    asyncio.run(scenario())
