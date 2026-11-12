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
from dinkster_image.ported.image_color_conversion import ImageColorSpace, ImageSDRToHDR

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
    # Direct sRGB decoding of float32 inputs, evaluated with 80-digit Decimal,
    # independently of the implementation's Rec.2020 primaries round trip.
    expected = torch.tensor(
        [
            [
                [
                    [0.00015479876896245376, 0.003130804897277348, 0.4919050695416527],
                    [2.163043551516219, 0.033104767499448746, 0.00024232198844660916],
                ],
                [
                    [0.6038273590607631, 0.01002282580612334, 0.21404114048223244],
                    [0.0007739937907494258, 0.7874122422164633, 0.003935939571117615],
                ],
            ],
            [
                [
                    [0.011645430804880955, 0.34239164539758316, 0.059253525811831645],
                    [0.8900053816569532, 1.2427703420684366, 0.07828832086604431],
                ],
                [
                    [0.0, 7.739938448122688e-05, 0.0030959751629977034],
                    [0.16264719178799114, 0.5704815485625002, 0.036306481962325006],
                ],
            ],
        ],
        dtype=torch.float64,
    )
    # AVX2/AVX512 pow differs by up to 2 ULP before the float32 matrix round trip.
    # Whole-fixture error is <= 3.47 unit roundoffs times the pixel maximum;
    # 16 allows accumulated rounding without dividing by near-zero channels.
    error = (linear[..., :3].double() - expected).abs()
    bound = 8 * torch.finfo(torch.float32).eps * expected.amax(dim=-1, keepdim=True)
    assert torch.all(error <= bound), error / bound
    assert torch.equal(linear[..., 3], image.float()[..., 3])
    _assert_hdr(hdr)
    assert torch.equal(image, before)


def _assert_hdr(result: Any) -> None:
    assert result.dtype == torch.float32
    assert result.device == torch.device("cpu")
    expected = torch.tensor(
        [
            [0.17928335070610046, 0.11021348834037781, 0.7042750120162964, 0.1300048828125],
            [0.3791734576225281, 0.5923642516136169, 0.3126797378063202, 0.469970703125],
        ]
    )
    # Lottes/HLG CUDA compute with CPU output differs by up to 2 ULP from CPU
    # compute. Four float32 epsilons give 3x the measured relative-error margin.
    torch.testing.assert_close(
        result[:, 0, 0, :3],
        expected[..., :3],
        rtol=4 * torch.finfo(torch.float32).eps,
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
                "from dinkster_image import NODES as COMFY_NODES\n"
                "from dinkster_image import register_types as register_inference_types\n"
                "NODES = (*COMFY_NODES, ColorSource, ColorInspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["color-fixture",')
                .replace("dinkster_image:NODES", "color_fixture:NODES")
                .replace(
                    "dinkster_image.ported.device_routing:device_choices",
                    "color_fixture:device_choices",
                )
                .replace(
                    "dinkster_image:register_types",
                    "color_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(
                PackSpec(trust_reserved=True, manifest=fixture_manifest, in_process=in_process)
            )
            graph = Graph(
                nodes={
                    "source": GraphNode("color-fixture.source"),
                    "identity": GraphNode(
                        "dinkster.image.image_color_space",
                        {"image": Link("source", "image"), "source": "sRGB", "destination": "sRGB"},
                    ),
                    "hdr": GraphNode(
                        "dinkster.image.image_sdr_to_hdr",
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
