"""CPU tensor contracts for core VAE nodes."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.vae import VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled

from dinkster.compose import PackSpec, ServingComposer

torch: Any = importlib.import_module("torch")


class ImageVAE:
    def encode(self, pixels: Any) -> Any:
        return pixels.movedim(-1, 1) * 3.0 - 2.0

    def decode(self, samples: Any) -> Any:
        return ((samples + 2.0) / 3.0).movedim(1, -1)

    def spacial_compression_decode(self) -> int:
        return 8

    def temporal_compression_decode(self) -> int | None:
        return None

    def encode_tiled(
        self,
        pixels: Any,
        *,
        tile_x: int,
        tile_y: int,
        overlap: int,
        tile_t: int,
        overlap_t: int,
    ) -> Any:
        return self.encode(pixels) + tile_x + 2 * tile_y + 3 * overlap + 5 * tile_t + 7 * overlap_t

    def decode_tiled(
        self,
        samples: Any,
        *,
        tile_x: int,
        tile_y: int,
        overlap: int,
        tile_t: int | None,
        overlap_t: int | None,
    ) -> Any:
        return (
            self.decode(samples)
            + tile_x
            + 2 * tile_y
            + 3 * overlap
            + 5 * (tile_t or 0)
            + 7 * (overlap_t or 0)
        )


class VideoVAE(ImageVAE):
    def temporal_compression_decode(self) -> int:
        return 4

    def decode(self, samples: Any) -> Any:
        return samples.movedim(1, -1)


def test_encode_decode_preserve_numpy_transport_and_source() -> None:
    pixels = torch.arange(2 * 7 * 11 * 3, dtype=torch.float32).reshape(2, 7, 11, 3) / 512
    original = pixels.clone()
    encoded: Any = VAEEncode.execute(vae=ImageVAE(), pixels=pixels.numpy())["LATENT"]
    assert torch.equal(encoded["samples"], pixels.movedim(-1, 1) * 3 - 2)
    decoded: Any = VAEDecode.execute(
        vae=ImageVAE(), samples={"samples": encoded["samples"].numpy()}
    )["IMAGE"]
    torch.testing.assert_close(decoded, pixels, rtol=0, atol=3e-8)
    assert torch.equal(pixels, original)


@pytest.mark.parametrize("video", (False, True))
def test_tiled_spatial_and_temporal_units(video: bool) -> None:
    pixels = torch.full((1, 7, 11, 3), 0.25)
    encoded: Any = VAEEncodeTiled.execute(
        vae=ImageVAE(),
        pixels=pixels.numpy(),
        tile_size=192,
        overlap=32,
        temporal_size=12,
        temporal_overlap=4,
    )["LATENT"]
    assert torch.equal(encoded["samples"], torch.full((1, 3, 7, 11), 758.75))
    samples = torch.full((2, 3, 5, 7, 11) if video else (2, 3, 7, 11), 0.25)
    result: Any = VAEDecodeTiled.execute(
        vae=VideoVAE() if video else ImageVAE(),
        samples={"samples": samples},
        tile_size=64,
        overlap=48,
        temporal_size=8,
        temporal_overlap=12,
    )["IMAGE"]
    # Spatial overlap becomes 16 pixels, then 2 latent cells. Temporal overlap
    # becomes 6 frames, then 1 cell after video compression.
    expected = 47.25 if video else 30.75
    assert torch.equal(result, torch.full((10, 7, 11, 3) if video else (2, 7, 11, 3), expected))


def test_decode_nested_latent_uses_first_stream_and_flattens_video() -> None:
    first = torch.arange(2 * 3 * 5 * 7 * 11, dtype=torch.float32).reshape(2, 3, 5, 7, 11)
    second = torch.zeros(1, 4, 3, 2, 2)
    nested = torch.nested.nested_tensor([first, second])
    result: Any = VAEDecode.execute(vae=VideoVAE(), samples={"samples": nested})["IMAGE"]
    np.testing.assert_array_equal(
        result.numpy(), first.movedim(1, -1).reshape(10, 7, 11, 3).numpy()
    )


class VAESource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vae-fixture.source",
            outputs=(
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                OutputSpec("pixels", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(vae=ImageVAE(), pixels=np.full((2, 7, 11, 3), 0.25, dtype=np.float32))


class VAEInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vae-fixture.inspect",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("pixel", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, image: Any) -> dict[str, object]:
        return cls.outputs(pixel=float(materialize_image(image)[1, 6, 10, 2]))


@pytest.mark.parametrize("in_process", (True,))
def test_manifest_executes_vae_encode_decode(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "src"))
                )
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "vae_fixture.py").write_text(
                "from test_vae_torch import VAESource, VAEInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, VAESource, VAEInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["vae-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "vae_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "vae_fixture:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "vae_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("vae-fixture.source"),
                    "encode": GraphNode(
                        "VAEEncode",
                        {"vae": Link("source", "vae"), "pixels": Link("source", "pixels")},
                    ),
                    "decode": GraphNode(
                        "VAEDecode",
                        {"vae": Link("source", "vae"), "samples": Link("encode", "LATENT")},
                    ),
                    "inspect": GraphNode("vae-fixture.inspect", {"image": Link("decode", "IMAGE")}),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["pixel"].resolve() == 0.25
        finally:
            await composer.close()

    asyncio.run(scenario())
