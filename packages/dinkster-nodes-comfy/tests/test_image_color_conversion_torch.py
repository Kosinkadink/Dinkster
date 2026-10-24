"""CPU behavior and schema coverage for image color conversion nodes."""

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
from dinkster_nodes_comfy.image_color_conversion import ImageColorSpace, ImageSDRToHDR

from dinkster.compose import PackSpec, ServingComposer


def _image() -> Any:
    return torch.tensor(
        [
            [
                [[0.002, 0.04045, 0.73, 0.13], [1.4, 0.2, 0.0031308, 0.79]],
                [[0.8, 0.1, 0.5, 0.29], [0.01, 0.9, 0.05, 0.67]],
            ],
            [
                [[0.11, 0.62, 0.27, 0.47], [0.95, 1.1, 0.31, 0.83]],
                [[0.0, 0.001, 0.04, 0.19], [0.44, 0.78, 0.21, 0.59]],
            ],
        ],
        dtype=torch.float64,
    )


def test_color_conversion_schemas_preserve_node_ids_and_dynamic_algorithm_slot() -> None:
    hdr = ImageSDRToHDR.define_schema()
    color = ImageColorSpace.define_schema()

    assert hdr.node_type == "ImageSDRToHDR"
    assert hdr.combos[0].id == "algorithm"
    assert hdr.combos[0].options[0].key == "lottes"
    assert hdr.combos[0].options[0].inputs[0].id == "target_luminance"
    assert color.node_type == "ImageColorSpace"
    assert tuple(option.value for option in color.inputs[1].widget.options) == (
        "sRGB",
        "HDR",
        "HDR ACEScct",
        "HDR LogC3",
        "HDR PQ",
        "linear",
        "linear ACEScg",
    )


def test_color_conversion_preserves_straight_alpha_and_linear_extended_values() -> None:
    image = torch.tensor([[[[0.25, 0.5, 1.5, 0.3]]]], dtype=torch.float32)
    linear = ImageColorSpace.execute(image=image, source="sRGB", destination="linear")["IMAGE"]
    restored = ImageColorSpace.execute(image=linear, source="linear", destination="sRGB")["IMAGE"]
    hdr = ImageSDRToHDR.execute(
        image=image, algorithm="lottes", **{"algorithm.target_luminance": 650.0}
    )["IMAGE"]

    assert linear[0, 0, 0, 2] > 1.0
    torch.testing.assert_close(restored[..., 3], image[..., 3])
    torch.testing.assert_close(hdr[..., 3], image[..., 3])
    assert torch.isfinite(hdr).all()


def test_color_conversion_returns_intermediate_float32_for_identity_and_hdr() -> None:
    image = _image()
    before = image.clone()
    identity = ImageColorSpace.execute(image=image, source="sRGB", destination="sRGB")["IMAGE"]
    linear = ImageColorSpace.execute(image=image, source="sRGB", destination="linear")["IMAGE"]
    hdr = ImageSDRToHDR.execute(
        image=image.half(), algorithm="lottes", **{"algorithm.target_luminance": 1400.0}
    )["IMAGE"]
    for result in (identity, linear, hdr):
        assert result.dtype == torch.float32
        assert result.device == torch.device("cpu")
    assert torch.equal(identity, image.float())
    torch.testing.assert_close(
        linear[:, 0, 0],
        torch.tensor(
            [
                [
                    0.00015480443835258484,
                    0.0031308052130043507,
                    0.4919051229953766,
                    0.12999999523162842,
                ],
                [0.01164543628692627, 0.34239163994789124, 0.0592535175383091, 0.4699999988079071],
            ]
        ),
        rtol=0,
        atol=0,
    )
    _assert_hdr(hdr)
    assert torch.equal(image, before)


def _assert_hdr(result: Any) -> None:
    assert result.dtype == torch.float32
    assert result.device == torch.device("cpu")
    torch.testing.assert_close(
        result[:, 0, 0],
        torch.tensor(
            [
                [0.17928335070610046, 0.11021348834037781, 0.7042750120162964, 0.1300048828125],
                [0.3791734576225281, 0.5923642516136169, 0.3126797378063202, 0.469970703125],
            ]
        ),
        rtol=0,
        atol=0,
    )
    assert torch.equal(result[..., 3], _image().half().float()[..., 3])


_IMAGE = TypeExpr.concrete(IMAGE_TYPE)


class ColorSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "color-fixture.source",
            outputs=(OutputSpec("image", _IMAGE), OutputSpec("half", _IMAGE)),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(image=_image().numpy(), half=_image().half().numpy())


class ColorInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "color-fixture.inspect",
            inputs=(InputSpec("identity", _IMAGE), InputSpec("hdr", _IMAGE)),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, identity: Any, hdr: Any) -> Any:
        identity = materialize_image(identity)
        assert identity.dtype == torch.float32
        assert identity.device == torch.device("cpu")
        assert torch.equal(identity, _image().float())
        _assert_hdr(materialize_image(hdr))
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_color_conversion_ordinary_manifest_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd() / "src")))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "color_fixture.py").write_text(
                "from test_image_color_conversion_torch import ColorSource, ColorInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ColorSource, ColorInspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["color-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "color_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "color_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("color-fixture.source"),
                    "identity": GraphNode(
                        "ImageColorSpace",
                        {"image": Link("source", "image"), "source": "sRGB", "destination": "sRGB"},
                    ),
                    "hdr": GraphNode(
                        "ImageSDRToHDR",
                        {
                            "image": Link("source", "half"),
                            "algorithm.target_luminance": 1400.0,
                        },
                        slot_variants={"algorithm": "lottes"},
                    ),
                    "inspect": GraphNode(
                        "color-fixture.inspect",
                        {"identity": Link("identity", "IMAGE"), "hdr": Link("hdr", "IMAGE")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
