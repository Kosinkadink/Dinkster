"""Public tensor behavior of native image padding."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.image_transform import PadImageNode

from dinkster.compose import PackSpec, ServingComposer


def test_mirror_repeats_without_edge_duplication_and_preserves_batches() -> None:
    image = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3, 1)
    actual = PadImageNode.execute(
        image=image,
        padding_type={
            "padding_type": "pad dimensions",
            "left": 4,
            "top": 1,
            "right": 1,
            "bottom": 2,
        },
        padding_style={"padding_style": "mirror image"},
    )["image"]
    expected = image[:, [1, 0, 1, 0, 1], :, :][:, :, [0, 1, 2, 1, 0, 1, 2, 1], :]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "selected,values",
    [
        ("pad by multiplier", {"padding_type.multiplier": 1.5}),
        ("pad to multiple", {"padding_type.multiple": 4}),
    ],
)
def test_odd_padding_places_extra_pixel_right_and_bottom(selected: str, values: dict) -> None:
    image = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 20
    actual = PadImageNode.execute(
        image=image,
        padding_type=selected,
        padding_style="color",
        **values,
        **{"padding_style.color": "#20408080", "padding_style.feathering": 0},
    )["image"]
    height = 3 if selected == "pad by multiplier" else 4
    expected = torch.tensor([32 / 255, 64 / 255, 128 / 255]).expand(1, height, 4, 3).clone()
    top = 0 if selected == "pad by multiplier" else 1
    expected[:, top : top + 2, :3] = image
    assert torch.equal(actual, expected)
    assert torch.equal(image, torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 20)


def test_feathering_only_uses_padded_sides_and_ignores_color_alpha() -> None:
    image = torch.zeros((1, 3, 4, 4), dtype=torch.float16)
    actual = PadImageNode.execute(
        image=image,
        padding_type={
            "padding_type": "pad dimensions",
            "left": 1,
            "top": 0,
            "right": 0,
            "bottom": 0,
        },
        padding_style={"padding_style": "color", "color": "#80402000", "feathering": 2},
    )["image"]
    fill = torch.tensor([128 / 255, 64 / 255, 32 / 255, 1], dtype=torch.float16)
    expected = torch.zeros((1, 3, 5, 4), dtype=torch.float16)
    expected[:, :, 0] = fill
    expected[:, :, 1] = fill * torch.tensor(4 / 9, dtype=torch.float16)
    expected[:, :, 2] = fill * torch.tensor(1 / 9, dtype=torch.float16)
    assert torch.equal(actual, expected)
    assert torch.equal(image, torch.zeros_like(image))


def test_single_pixel_mirror_and_zero_padding_bypass_style() -> None:
    image = torch.tensor([[[[0.2, 0.4, 0.8]]]])
    actual = PadImageNode.execute(
        image=image,
        padding_type={
            "padding_type": "pad dimensions",
            "left": 3,
            "top": 2,
            "right": 4,
            "bottom": 1,
        },
        padding_style={"padding_style": "mirror image"},
    )["image"]
    assert torch.equal(actual, image.expand(1, 4, 8, 3))
    unchanged = PadImageNode.execute(
        image=image,
        padding_type={"padding_type": "pad to multiple", "multiple": 1},
        padding_style={"padding_style": "unused"},
    )["image"]
    assert unchanged is image


@pytest.mark.parametrize("color", ["oops", "#gg0000"])
def test_invalid_color_uses_reference_white_fallback(color: str) -> None:
    image = torch.zeros((1, 1, 2, 3))
    actual = PadImageNode.execute(
        image=image,
        padding_type={
            "padding_type": "pad dimensions",
            "left": 1,
            "top": 0,
            "right": 0,
            "bottom": 0,
        },
        padding_style={"padding_style": "color", "color": color, "feathering": 0},
    )["image"]
    assert torch.equal(actual[:, :, :1], torch.ones((1, 1, 1, 3)))
    assert torch.equal(actual[:, :, 1:], image)


class PaddingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "padding-fixture.source", outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),)
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(image=torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 20)


class PaddingInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "padding-fixture.inspect",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Any:
        actual = materialize_image(image)
        expected = torch.tensor([32 / 255, 64 / 255, 128 / 255]).expand(1, 4, 4, 3).clone()
        expected[:, 1:3, :3] = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 20
        assert torch.equal(actual, expected)
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_padding_crosses_ordinary_manifest_hosts_and_caches(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "padding_fixture.py").write_text(
            "from test_image_padding_torch import PaddingSource, PaddingInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (PaddingSource, PaddingInspect)\n",
            encoding="ascii",
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "padding-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["padding-fixture"]\n\n[pack.entry]\n'
            'nodes = "padding_fixture:NODES"\n'
            'types = "padding_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(native_manifest.parent / "src"),
                        str(Path.cwd() / "src"),
                    )
                ),
            }
        )
        try:
            for pack in (native_manifest, manifest):
                await composer.add_pack(PackSpec(pack, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("padding-fixture.source"),
                    "pad": GraphNode(
                        "PadImageNode",
                        {
                            "image": Link("source", "image"),
                            "padding_type.multiple": 4,
                            "padding_style.color": "#20408080",
                            "padding_style.feathering": 0,
                        },
                        slot_variants={"padding_type": "pad to multiple", "padding_style": "color"},
                    ),
                    "inspect": GraphNode(
                        "padding-fixture.inspect", {"image": Link("pad", "image")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["inspect"])
            assert first.outputs["inspect"]["valid"].resolve() is True
            second = await engine.run(graph, ["inspect"])
            assert second.outputs["inspect"]["valid"].resolve() is True
            assert set(second.cached) == {
                "source",
                "pad",
                "inspect",
            }
        finally:
            await composer.close()

    asyncio.run(scenario())
