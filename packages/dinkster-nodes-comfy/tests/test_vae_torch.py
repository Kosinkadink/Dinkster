"""CPU tensor contracts for core VAE nodes."""

from __future__ import annotations

import asyncio
import importlib
import inspect
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
    inference_nodes,
    initialize_inference_process,
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


def test_tiled_image_vae_runtime_decodes_on_cpu() -> None:
    initialize_inference_process()
    runtime = importlib.import_module("dinkster_inference.sd")
    vae = runtime.VAE(
        sd={},
        device=torch.device("cpu"),
        dtype=torch.float32,
        config={
            "params": {
                "embed_dim": 4,
                "ddconfig": {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 8,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 32,
                    "ch_mult": [1],
                    "num_res_blocks": 1,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                    "conv3d": True,
                    "time_compress": 1,
                },
            }
        },
    )
    # A tiny image/video decoder with known RGB output exercises the real
    # tiled scaler and in-place VAE process_output, without model downloads.
    with torch.no_grad():
        for parameter in vae.first_stage_model.parameters():
            parameter.zero_()
        vae.first_stage_model.decoder.conv_out.conv.bias.copy_(torch.tensor([-2.0, 0.0, 2.0]))
    vae.upscale_ratio = (lambda size: size, 1, 1)
    vae.downscale_ratio = (lambda size: size, 1, 1)
    vae.latent_dim = 3
    samples = torch.linspace(-1, 1, 4 * 3 * 5 * 7).reshape(1, 4, 3, 5, 7)
    assert not torch.is_inference_mode_enabled()
    result = VAEDecodeTiled.execute(
        vae=vae,
        samples={"samples": samples},
        tile_size=4,
        overlap=1,
        temporal_size=2,
        temporal_overlap=1,
    )["IMAGE"]
    assert not torch.is_inference_mode_enabled()
    expected = torch.tensor([0.0, 0.5, 1.0]).expand(3, 5, 7, 3)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fail", (False, True))
def test_ported_execution_restores_mode_and_preserves_ordinary_gradients(fail: bool) -> None:
    class ModeVAE(ImageVAE):
        def decode(self, samples: Any) -> Any:
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            if fail:
                raise ValueError("decode failed")
            return super().decode(samples)

    class GradientNode(Node):
        @classmethod
        def execute(cls, *, value: Any) -> Any:
            assert not torch.is_inference_mode_enabled()
            return value.square().sum()

    assert not torch.is_inference_mode_enabled()
    samples = torch.ones(1, 3, 2, 5)
    if fail:
        with pytest.raises(ValueError, match="decode failed"):
            VAEDecode.execute(vae=ModeVAE(), samples={"samples": samples})
    else:
        result = VAEDecode.execute(vae=ModeVAE(), samples={"samples": samples})["IMAGE"]
        torch.testing.assert_close(result, torch.ones(1, 2, 5, 3), rtol=0, atol=0)
    assert not torch.is_inference_mode_enabled()
    value = torch.tensor([-3.0, 2.0], requires_grad=True)
    GradientNode.execute(value=value).backward()
    torch.testing.assert_close(value.grad, torch.tensor([-6.0, 4.0]), rtol=0, atol=0)


def test_inference_nodes_preserve_class_binding_and_async_isolation() -> None:
    class InferenceBody(Node):
        @classmethod
        def execute(cls, *, value: Any) -> Any:
            assert torch.is_inference_mode_enabled()
            return cls, value + 2

    class AsyncBody(Node):
        @classmethod
        async def execute(cls, *, value: Any) -> Any:
            assert not torch.is_inference_mode_enabled()
            await asyncio.sleep(0)
            assert not torch.is_inference_mode_enabled()
            return value * 3

    signature = inspect.signature(InferenceBody.execute)
    nodes = inference_nodes((InferenceBody, AsyncBody))
    assert inference_nodes(nodes) == nodes
    assert inspect.signature(InferenceBody.execute) == signature

    class Child(InferenceBody):
        pass

    cls, value = Child.execute(value=torch.tensor(5.0))
    assert cls is Child
    assert value.item() == 7.0
    assert not torch.is_inference_mode_enabled()
    value = asyncio.run(AsyncBody.execute(value=torch.tensor(5.0, requires_grad=True)))
    value.backward()
    assert not torch.is_inference_mode_enabled()


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
