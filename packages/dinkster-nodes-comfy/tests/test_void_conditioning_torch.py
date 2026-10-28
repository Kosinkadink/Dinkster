"""CPU contracts for native VOID conditioning and warped noise.

Hand-derived from nodes_void.py at ComfyUI
bc5f880dbdd43e426077f42ccec53bb577ffee9f. Threshold bands quantize
to 0/63/127/255 before inversion; lengths 1/9 become 5, while 13 stays
13. Controlled encoders select channel zero every four frames/eight pixels,
then repeat the last encoded frame or truncate to match the target time.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    NOISE_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.void_conditioning import (
    VOIDInpaintConditioning,
    VOIDQuadmaskPreprocess,
    VOIDWarpedNoiseSource,
)

from dinkster.compose import PackSpec, ServingComposer


class _VAE:
    def __init__(self, temporal: int | None = None) -> None:
        self.temporal = temporal
        self.images: list[torch.Tensor] = []

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.images.append(images.clone())
        values = images[::4, ::8, ::8, 0]
        if self.temporal is not None:
            values = values[:1].repeat(self.temporal, 1, 1)
            values = values + torch.arange(self.temporal).reshape(-1, 1, 1)
        return values.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _conditioning() -> list[list[Any]]:
    return [[torch.tensor([[3.0]]), {"label": "retained"}]]


def test_quadmask_threshold_edges_and_normalized_input_preserve_source() -> None:
    source = torch.tensor([0, 31, 32, 95, 96, 191, 192, 255], dtype=torch.float32)
    expected = (255 - torch.tensor([0, 0, 63, 63, 127, 127, 255, 255])) / 255
    original = source.clone()
    for mask in (source.reshape(1, 2, 4), (source / 255).reshape(1, 2, 4)):
        result = VOIDQuadmaskPreprocess.execute(mask=mask)["quadmask"]
        torch.testing.assert_close(result.flatten(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)


def test_quadmask_dilation_is_spatial_and_keeps_other_frames() -> None:
    mask = torch.full((2, 4, 5), 255.0)
    mask[0, 1, 2] = 63
    expected = torch.zeros_like(mask)
    expected[0, :3, 1:4] = 1
    torch.testing.assert_close(
        VOIDQuadmaskPreprocess.execute(mask=mask, dilate_width=1)["quadmask"],
        expected,
        rtol=0,
        atol=0,
    )
    assert mask[0, 1, 2] == 63 and torch.count_nonzero(mask == 63) == 1


@pytest.mark.parametrize(("length", "frames", "latent_t"), ((1, 5, 2), (9, 5, 2), (13, 13, 4)))
@pytest.mark.parametrize("encoded_t", (1, 6))
def test_void_conditioning_rounds_length_and_matches_encoder_time(
    length: int, frames: int, latent_t: int, encoded_t: int
) -> None:
    video = torch.arange(13 * 16 * 24 * 4, dtype=torch.float32).reshape(13, 16, 24, 4) / 20000
    mask = torch.full((13, 16, 24), 0.25)
    original = video.clone()
    positive, negative = _conditioning(), _conditioning()
    vae = _VAE(encoded_t)
    result = VOIDInpaintConditioning.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        video=video,
        quadmask=mask,
        width=24,
        height=16,
        length=length,
        batch_size=2,
    )
    torch.testing.assert_close(vae.images[0], torch.full((frames, 16, 24, 3), 0.75))
    torch.testing.assert_close(vae.images[1], video[:frames, ..., :3] * 0.75)
    steps = torch.arange(latent_t).clamp(max=encoded_t - 1).reshape(1, 1, -1, 1, 1)
    expected_mask = (0.75 + steps).expand(1, 16, latent_t, 2, 3)
    expected_video = video[0, ::8, ::8, 0].reshape(1, 1, 1, 2, 3) * 0.75 + steps
    expected = torch.cat((expected_mask, expected_video.expand(1, 16, latent_t, 2, 3)), dim=1)
    for label in ("positive", "negative"):
        assert result[label][0][1]["label"] == "retained"
        torch.testing.assert_close(result[label][0][1]["concat_latent_image"], expected)
    assert result["latent"]["samples"].shape == (2, 16, latent_t, 2, 3)
    assert torch.count_nonzero(result["latent"]["samples"]) == 0
    assert positive[0][1] == negative[0][1] == {"label": "retained"}
    torch.testing.assert_close(video, original, rtol=0, atol=0)
    assert torch.all(mask == 0.25)


def test_warped_noise_source_returns_independent_cpu_noise() -> None:
    source = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 2, 4)
    noise = VOIDWarpedNoiseSource.execute(warped_noise={"samples": source})["NOISE"]
    first = noise.generate_noise({"samples": torch.ones_like(source)})
    assert noise.seed == 0 and first.device.type == "cpu"
    torch.testing.assert_close(first, source, rtol=0, atol=0)
    first.zero_()
    torch.testing.assert_close(noise.generate_noise(None), source, rtol=0, atol=0)
    assert source.flatten()[-1] == 47


def test_void_discovery_is_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.void_conditioning import "
            "VOIDQuadmaskPreprocess, VOIDInpaintConditioning, VOIDWarpedNoiseSource; "
            "[node.schema() for node in (VOIDQuadmaskPreprocess, VOIDInpaintConditioning, "
            "VOIDWarpedNoiseSource)]; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stderr


_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


class VoidSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="void-conditioning-fixture.source",
            outputs=(
                OutputSpec("video", _IMAGE),
                OutputSpec("mask", _MASK),
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                OutputSpec("warped", _LATENT),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        mask = torch.tensor([[0.0, 63.0, 127.0], [255.0, 32.0, 192.0]])
        mask = mask.repeat_interleave(8, dim=0).repeat_interleave(8, dim=1)
        return cls.outputs(
            video=torch.arange(13 * 16 * 24 * 3, dtype=torch.float32).reshape(13, 16, 24, 3)
            / 10000,
            mask=mask.unsqueeze(0).repeat(13, 1, 1),
            positive=_conditioning(),
            negative=_conditioning(),
            vae=_VAE(1),
            warped={"samples": torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 2, 4)},
        )


class VoidConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="void-conditioning-fixture.consumer",
            inputs=(
                InputSpec("conditioning", _CONDITIONING),
                InputSpec("latent", _LATENT),
                InputSpec("noise", TypeExpr.concrete(NOISE_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any, latent: object, noise: Any) -> dict[str, object]:
        encoded = conditioning[0][1]["concat_latent_image"]
        assert conditioning[0][1]["label"] == "retained"
        assert encoded.shape == (1, 32, 2, 2, 3)
        assert materialize_latent(latent)["samples"].shape == (2, 16, 2, 2, 3)
        expected_mask = torch.tensor([[0.0, 63.0, 127.0], [255.0, 63.0, 255.0]])
        expected_mask = 1.0 - (255.0 - expected_mask) / 255.0
        first_video = torch.tensor([[0.0, 24.0, 48.0], [576.0, 600.0, 624.0]]) / 10000
        expected = torch.cat(
            (
                expected_mask.reshape(1, 1, 1, 2, 3).expand(1, 16, 2, 2, 3),
                (first_video * expected_mask).reshape(1, 1, 1, 2, 3).expand(1, 16, 2, 2, 3),
            ),
            dim=1,
        )
        torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
        first = noise.generate_noise(None)
        assert noise.seed == 0 and first.device.type == "cpu"
        first.zero_()
        assert noise.generate_noise(None).flatten()[-1] == 47
        return cls.outputs(summary="void-conditioning")


HOST_NODES = (VoidSource, VoidConsumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_void_nodes_run_twice_through_native_manifest_and_shared_worker(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        import dinkster_nodes_comfy.void_conditioning as void_conditioning

        assert Path(void_conditioning.__file__).resolve().is_relative_to(package / "src")
        fixture_entry = tmp_path / "void_conditioning_fixture.py"
        fixture_entry.write_text(
            "from test_void_conditioning_torch import HOST_NODES\n"
            "from dinkster_nodes_comfy import register_inference_types\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            '[pack]\nname = "void-conditioning-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["void-conditioning-fixture"]\n\n[pack.entry]\n'
            'nodes = "void_conditioning_fixture:HOST_NODES"\n'
            'types = "void_conditioning_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        native_manifest = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(package / "src"),
                        *sys.path,
                    )
                ),
            }
        )
        try:
            manifests = (native_manifest, fixture_manifest)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "void-conditioning-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("void-conditioning-fixture.source"),
                    "quadmask": GraphNode(
                        "VOIDQuadmaskPreprocess", {"mask": Link("source", "mask")}
                    ),
                    "inpaint": GraphNode(
                        "VOIDInpaintConditioning",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "vae": Link("source", "vae"),
                            "video": Link("source", "video"),
                            "quadmask": Link("quadmask", "quadmask"),
                            "width": 24,
                            "height": 16,
                            "length": 9,
                            "batch_size": 2,
                        },
                    ),
                    "noise": GraphNode(
                        "VOIDWarpedNoiseSource", {"warped_noise": Link("source", "warped")}
                    ),
                    "consumer": GraphNode(
                        "void-conditioning-fixture.consumer",
                        {
                            "conditioning": Link("inpaint", "positive"),
                            "latent": Link("inpaint", "latent"),
                            "noise": Link("noise", "NOISE"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "void-conditioning"
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
