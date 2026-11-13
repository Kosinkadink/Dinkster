"""CPU video latent creation, transforms, and ordinary pack execution."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_cosmos.video_latent import EmptyCosmosLatentVideo
from dinkster_graph import Graph, GraphNode, Link
from dinkster_hunyuan.video_latent import (
    EmptyHunyuanLatentVideo,
    EmptyHunyuanVideo15Latent,
)
from dinkster_latent.video_latent_ops import (
    NormalizeVideoLatentStart,
    TrimVideoLatent,
)
from dinkster_ltx.video_latent import EmptyLTXVLatentVideo
from dinkster_ltx.video_latent_ops import LTXVFreezeLatent
from dinkster_misc_models.video_latent import EmptyARVideoLatent

from dinkster.compose import PackSpec, ServingComposer, _installed_pack_spec

VIDEO_LATENT_NODES = (
    EmptyHunyuanLatentVideo,
    EmptyHunyuanVideo15Latent,
    EmptyLTXVLatentVideo,
    EmptyCosmosLatentVideo,
    EmptyARVideoLatent,
)


class VideoLatentInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-latent-fixture.inspect",
            inputs=tuple(
                InputSpec(node.schema().node_type, TypeExpr.concrete(LATENT_TYPE))
                for node in VIDEO_LATENT_NODES
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **latents: object) -> dict[str, object]:
        expected = (
            ((2, 16, 2, 11, 18), {"downscale_ratio_spacial": 8}),
            ((2, 32, 3, 6, 9), {"downscale_ratio_spacial": 16}),
            ((2, 128, 2, 3, 2), {"downscale_ratio_spacial": 32}),
            ((2, 16, 2, 8, 9), {}),
            ((2, 16, 2, 8, 9), {}),
        )
        shapes = []
        for node, (shape, metadata) in zip(VIDEO_LATENT_NODES, expected, strict=True):
            latent = materialize_latent(latents[node.schema().node_type])
            samples = latent["samples"]
            assert tuple(samples.shape) == shape
            assert samples.device.type == "cpu"
            assert samples.dtype == torch.float32
            assert not samples.count_nonzero()
            assert {key: value for key, value in latent.items() if key != "samples"} == metadata
            shapes.append(str(shape))
        return cls.outputs(summary=";".join(shapes))


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_executes_video_latents_twice(in_process: bool, tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            }
        )
        try:
            (tmp_path / "video_latent_fixture.py").write_text(
                "from test_video_latent_torch import VideoLatentInspect\n"
                "from dinkster_api.v1 import register_inference_types\n"
                "NODES = (VideoLatentInspect,)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                '[pack]\nname = "video-latent-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["video-latent-fixture"]\n[pack.entry]\n'
                'nodes = "video_latent_fixture:NODES"\n'
                'types = "video_latent_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for identity, module in (
                ("dinkster-latent", "dinkster_latent"),
                ("dinkster-conditioning", "dinkster_conditioning"),
                ("dinkster-video", "dinkster_video"),
                ("dinkster-models", "dinkster_models"),
                ("dinkster-llm", "dinkster_llm"),
                ("dinkster-qwen", "dinkster_qwen"),
                ("dinkster-hunyuan", "dinkster_hunyuan"),
                ("dinkster-ltx", "dinkster_ltx"),
                ("dinkster-cosmos", "dinkster_cosmos"),
                ("dinkster-misc-models", "dinkster_misc_models"),
            ):
                await composer.add_pack(
                    _installed_pack_spec(identity, module, in_process=in_process)
                )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            inputs = (
                {"width": 151, "height": 95, "length": 8, "batch_size": 2},
                {"width": 159, "height": 111, "length": 9, "batch_size": 2},
                {"width": 95, "height": 127, "length": 16, "batch_size": 2},
                {"width": 79, "height": 71, "length": 16, "batch_size": 2},
                {"width": 79, "height": 71, "length": 8, "batch_size": 2},
            )
            graph = Graph(
                nodes={
                    **{
                        node.schema().node_type: GraphNode(node.schema().node_type, inputs[index])
                        for index, node in enumerate(VIDEO_LATENT_NODES)
                    },
                    "inspect": GraphNode(
                        "video-latent-fixture.inspect",
                        {
                            node.schema().node_type: Link(node.schema().node_type, "LATENT")
                            for node in VIDEO_LATENT_NODES
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve()
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("trim_amount", (0, 2, 5, 99))
def test_trim_video_latent_slices_only_the_temporal_axis_and_preserves_metadata(
    trim_amount: int,
) -> None:
    samples = torch.arange(2 * 3 * 5 * 7 * 11, dtype=torch.float64).reshape(2, 3, 5, 7, 11)
    noise_mask = torch.full((2, 1, 5, 1, 1), 0.75, dtype=torch.float32)
    latent = {"samples": samples, "noise_mask": noise_mask, "metadata": {"keep": [1, 2]}}
    samples_before, noise_mask_before = samples.clone(), noise_mask.clone()

    result = TrimVideoLatent.execute(samples=latent, trim_amount=trim_amount)["LATENT"]

    assert result is not latent
    assert torch.equal(result["samples"], samples[:, :, trim_amount:])
    assert result["noise_mask"] is noise_mask
    assert result["metadata"] is latent["metadata"]
    assert torch.equal(samples, samples_before)
    assert torch.equal(noise_mask, noise_mask_before)


@pytest.mark.parametrize(
    ("shape", "mask_shape"),
    (
        ((2, 3, 5, 7, 11), (2, 1, 5, 1, 1)),
        ((2, 3, 5, 11), (2, 1, 5, 1)),
    ),
)
def test_freeze_latent_replaces_the_mask_without_mutating_the_source(
    shape: tuple[int, ...], mask_shape: tuple[int, ...]
) -> None:
    samples = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    original_mask = torch.full(mask_shape, 0.75, dtype=torch.float64)
    latent = {"samples": samples, "noise_mask": original_mask, "metadata": {"keep": True}}
    samples_before, mask_before = samples.clone(), original_mask.clone()

    result = LTXVFreezeLatent.execute(latent=latent)["latent"]

    assert result is not latent
    assert result["samples"] is samples
    assert result["metadata"] is latent["metadata"]
    assert result["noise_mask"].shape == mask_shape
    assert result["noise_mask"].dtype == torch.float32
    assert result["noise_mask"].device == samples.device
    assert not torch.count_nonzero(result["noise_mask"])
    assert torch.equal(samples, samples_before)
    assert torch.equal(original_mask, mask_before)


def test_normalize_video_latent_start_uses_nondefault_counts() -> None:
    samples = torch.tensor([[-1, 0, 1, 2, 3, 7], [1, 2, 3, 4, 5, 9]], dtype=torch.float64).reshape(
        1, 2, 6, 1, 1
    )
    result = NormalizeVideoLatentStart.execute(
        latent={"samples": samples}, start_frame_count=4, reference_frame_count=1
    )["latent"]
    deviation = math.sqrt(2) / (math.sqrt(2) + 1e-8)
    expected = torch.tensor(
        [
            [0.35 - deviation, 1.35 - deviation, 2.35 - deviation, 3.35 - deviation, 3, 7],
            [0.35 + deviation, 1.35 + deviation, 2.35 + deviation, 3.35 + deviation, 5, 9],
        ],
        dtype=torch.float64,
    ).reshape(1, 2, 6, 1, 1)
    torch.testing.assert_close(result["samples"], expected)


def test_normalize_video_latent_start_returns_single_frame_mapping_unchanged() -> None:
    latent = {
        "samples": torch.arange(2 * 2 * 1 * 3 * 5, dtype=torch.float64).reshape(2, 2, 1, 3, 5)
    }
    result = NormalizeVideoLatentStart.execute(
        latent=latent, start_frame_count=4, reference_frame_count=5
    )["latent"]
    assert result is latent
