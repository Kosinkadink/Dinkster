"""Layer document values, sparse fields, and serialized native schema."""

from __future__ import annotations

import asyncio
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    LAYERS_TYPE,
    MASK_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
    schema_to_wire,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.image_layers import AddLayer

from dinkster.compose import PackSpec, ServingComposer


def test_add_layer_appends_without_reordering_or_changing_media() -> None:
    image = np.arange(24, dtype=np.float16).reshape(2, 1, 3, 4)
    mask = np.array([[[0.0, 0.25, 1.0]]], dtype=np.float32)
    previous_item = {"image": image, "z_index": 9, "custom": ["retained"]}
    previous = {"version": 7, "layers": [previous_item], "canvas": [13.9, 7.2], "extra": 42}
    result = AddLayer.execute(
        image=image,
        mask=mask,
        layers=previous,
        name="Upper",
        x=-3,
        y=5,
        opacity=0.375,
        blend_mode="multiply",
        rotation=-90,
        width=17,
        height=11,
        z_index=-4,
        flip_h=True,
    )["LAYERS"]
    assert result is not previous
    assert result.keys() == {"version", "layers", "canvas"}
    assert result["version"] == 1 and result["canvas"] == (13, 7)
    assert result["layers"][0] is previous_item
    item = result["layers"][1]
    assert item["image"] is image and item["mask"] is mask
    assert {key: value for key, value in item.items() if key not in ("image", "mask")} == {
        "type": "raster",
        "name": "Upper",
        "x": -3,
        "y": 5,
        "z_index": -4,
        "opacity": 0.375,
        "blend_mode": "multiply",
        "rotation": -math.pi / 2,
        "w": 17,
        "h": 11,
        "flip_h": True,
    }
    assert len(previous["layers"]) == 1 and previous["canvas"] == [13.9, 7.2]
    np.testing.assert_array_equal(image, np.arange(24, dtype=np.float16).reshape(2, 1, 3, 4))
    np.testing.assert_array_equal(mask, [[[0, 0.25, 1]]])


@pytest.mark.parametrize("canvas", ([True, 7], ["13", 7], [13, 0], [13, 7, 2]))
def test_add_layer_omits_invalid_canvas_and_default_fields(canvas: object) -> None:
    image = np.array([[[[0.125, 0.25, 0.75]]]], dtype=np.float32)
    result = AddLayer.execute(image=image, layers={"layers": None, "canvas": canvas})["LAYERS"]
    assert result.keys() == {"version", "layers"}
    assert len(result["layers"]) == 1
    item = result["layers"][0]
    assert item.keys() == {"image", "type", "x", "y", "z_index"}
    assert item["image"] is image
    assert (item["x"], item["y"], item["z_index"], item["type"]) == (0, 0, 0, "raster")


def test_add_layer_serialized_schema_preserves_optional_controls() -> None:
    wire = schema_to_wire(AddLayer.schema())
    inputs = [item for item in wire["interface"] if item["role"] == "input"]
    assert [item["id"] for item in inputs] == [
        "layers",
        "image",
        "mask",
        "name",
        "x",
        "y",
        "opacity",
        "blend_mode",
        "rotation",
        "width",
        "height",
        "z_index",
        "flip_h",
        "flip_v",
    ]
    assert [item["id"] for item in inputs if item["required"]] == ["image"]
    assert inputs[0]["type"] == {"kind": "concrete", "types": ["LAYERS"]}
    assert inputs[3]["widget"] == {"type": "STRING", "multiline": False}
    assert inputs[4]["widget"]["min"] == -16_384
    assert inputs[8]["widget"]["step"] == 1.0
    assert inputs[7]["default"] == "normal"
    assert len(inputs[7]["widget"]["options"]) == 26
    assert all(item["doc"] for item in inputs)
    assert [item["id"] for item in wire["interface"] if item["role"] == "output"] == ["LAYERS"]


class _LayerHostSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "layer-host-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("image", IMAGE_TYPE),
                    ("mask", MASK_TYPE),
                    ("layers", LAYERS_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls):
        image = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4) / 64
        mask = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 16
        return cls.outputs(
            image=image,
            mask=mask,
            layers={
                "version": 1,
                "canvas": (23, 11),
                "layers": [{"image": image + 1, "mask": 1 - mask, "z_index": 7}],
            },
        )


class _LayerHostInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "layer-host-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(LAYERS_TYPE)) for name in ("original", "appended")
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, original, appended):
        original, appended = materialize_latent(original), materialize_latent(appended)
        assert original["version"] == appended["version"] == 1
        assert original["canvas"] == appended["canvas"] == (23, 11)
        assert len(original["layers"]) == 1 and len(appended["layers"]) == 2
        image = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4) / 64
        mask = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 16
        for document in (original, appended):
            previous = document["layers"][0]
            assert previous.keys() == {"image", "mask", "z_index"}
            assert previous["z_index"] == 7
            np.testing.assert_array_equal(np.asarray(previous["image"]), image + 1)
            np.testing.assert_array_equal(np.asarray(previous["mask"]), 1 - mask)
        added = appended["layers"][1]
        np.testing.assert_array_equal(np.asarray(added["image"]), image)
        np.testing.assert_array_equal(np.asarray(added["mask"]), mask)
        assert {key: value for key, value in added.items() if key not in ("image", "mask")} == {
            "type": "raster",
            "x": -3,
            "y": 5,
            "z_index": -4,
            "name": "upper",
            "opacity": 0.375,
            "blend_mode": "multiply",
            "rotation": -math.pi / 2,
            "w": 17,
            "h": 9,
            "flip_v": True,
        }
        return cls.outputs(summary="two layers retain media and placement")


@pytest.mark.parametrize("in_process", (True, False))
def test_add_layer_uses_native_manifest_hosts_and_persistent_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario():
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"layer_host_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_image_layers import _LayerHostSource, _LayerHostInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (_LayerHostSource, _LayerHostInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="layer-host-fixture"\npublisher="dinkster"\n'
            'namespaces=["layer-host-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        graph = Graph(
            nodes={
                "source": GraphNode("layer-host-fixture.source"),
                "append": GraphNode(
                    "AddLayer",
                    {
                        "layers": Link("source", "layers"),
                        "image": Link("source", "image"),
                        "mask": Link("source", "mask"),
                        "name": "upper",
                        "x": -3,
                        "y": 5,
                        "z_index": -4,
                        "opacity": 0.375,
                        "blend_mode": "multiply",
                        "rotation": -90.0,
                        "width": 17,
                        "height": 9,
                        "flip_v": True,
                    },
                ),
                "inspect": GraphNode(
                    "layer-host-fixture.inspect",
                    {"original": Link("source", "layers"), "appended": Link("append", "LAYERS")},
                ),
            }
        )
        paths = (str(tmp_path), str(Path(__file__).parent), *(p for p in sys.path if p))
        for restart in range(2):
            composer = ServingComposer(
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
                worker_env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "DINKSTER_ACCELERATOR": "cpu",
                    "PYTHONPATH": os.pathsep.join(paths),
                },
            )
            try:
                manifests = (native, fixture)
                for manifest in manifests:
                    await composer.add_pack(
                        PackSpec(
                            manifest=manifest,
                            in_process=in_process,
                            worker_group=None if in_process else "layer-host-fixture",
                            group_manifests=() if in_process else manifests,
                        )
                    )
                events = []
                engine = composer.composition.make_engine(events.append)
                for repeat in range(2):
                    events.clear()
                    result = await engine.run(graph, ["inspect"])
                    assert result.outputs["inspect"]["summary"].resolve() == (
                        "two layers retain media and placement"
                    )
                    if restart or repeat:
                        assert set(result.cached) == {"source", "append"}
                        assert set(result.executed) == {"inspect"}
                        assert {
                            event.detail["cacheLayer"]
                            for event in events
                            if event.kind == "node_cached"
                        } == {"disk" if restart and not repeat else "memory"}
                    else:
                        assert set(result.executed) == {"source", "append", "inspect"}
            finally:
                await composer.close()

    asyncio.run(scenario())
