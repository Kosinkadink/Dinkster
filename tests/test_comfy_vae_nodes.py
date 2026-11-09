"""VAE interfaces and candidate engine/pack inference execution contracts."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from typing import Any

import pytest
from dinkster_api.v1 import Node, inference_nodes, initialize_inference_process
from dinkster_latent import NODES
from dinkster_latent.vae import VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled


def test_vae_core_interfaces() -> None:
    nodes = (VAEDecode, VAEEncode, VAEDecodeTiled, VAEEncodeTiled)
    assert [node.schema().node_type for node in nodes] == [
        "dinkster.latent.vae_decode",
        "dinkster.latent.vae_encode",
        "dinkster.latent.vae_decode_tiled",
        "dinkster.latent.vae_encode_tiled",
    ]
    for node in nodes:
        assert node in NODES
        schema = node.schema()
        encode = "_encode" in schema.node_type
        tiled = schema.node_type.endswith("_tiled")
        assert [field.id for field in schema.inputs] == [
            "pixels" if encode else "samples",
            "vae",
            *(["tile_size", "overlap", "temporal_size", "temporal_overlap"] if tiled else []),
        ]
        assert [field.id for field in schema.outputs] == ["LATENT" if encode else "IMAGE"]
        assert schema.category == "model/latent"


def test_tiled_image_vae_runtime_decodes_on_cpu() -> None:
    torch = pytest.importorskip("torch")
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
    torch = pytest.importorskip("torch")

    class ModeVAE:
        def decode(self, samples: Any) -> Any:
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            if fail:
                raise ValueError("decode failed")
            return samples.movedim(1, -1)

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
    torch = pytest.importorskip("torch")

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
