"""CPU video latent creation, transforms, and ordinary pack execution."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from collections.abc import Mapping
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
from dinkster_graph import Graph, GraphNode, Link
from dinkster_inference_wire.latents import MultiStreamLatent
from dinkster_nodes_comfy.video_latent import VIDEO_LATENT_NODES
from dinkster_nodes_comfy.video_latent_ops import (
    LTXVFreezeLatent,
    NormalizeVideoLatentStart,
    TrimVideoLatent,
)

from dinkster.compose import PackSpec, ServingComposer


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
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "video_latent_fixture.py").write_text(
                "from test_video_latent_torch import VideoLatentInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, VideoLatentInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["video-latent-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "video_latent_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "video_latent_fixture:register_inference_types",
                ),
                encoding="ascii",
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


@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 3, 5, 7, 11, 13)))
def test_freeze_latent_rejects_unsupported_tensor_ranks(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="Expected a 4D audio or 5D video latent"):
        LTXVFreezeLatent.execute(latent={"samples": torch.zeros(shape)})


def test_freeze_latent_rejects_concatenated_av_latents() -> None:
    multistream = MultiStreamLatent[object].from_pairs(
        (("video", {"samples": torch.zeros((1, 1, 1, 1, 1))}),)
    )
    with pytest.raises(ValueError, match="plain tensor"):
        LTXVFreezeLatent.execute(latent={"samples": multistream})


def test_normalize_video_latent_start_matches_tensor_contract_and_preserves_source() -> None:
    samples = torch.arange(2 * 2 * 6 * 3 * 5, dtype=torch.float64).reshape(2, 2, 6, 3, 5)
    samples[1] = samples[1].flip(2) * -0.7 + 5.0
    latent = {
        "samples": samples,
        "noise_mask": torch.full((2, 1, 6, 1, 1), 0.25),
        "metadata": {"keep": ["value"]},
    }
    source_before = samples.clone()

    result = NormalizeVideoLatentStart.execute(
        latent=latent, start_frame_count=2, reference_frame_count=3
    )["latent"]

    assert result is not latent
    assert hashlib.sha256(result["samples"].numpy().tobytes()).hexdigest() == (
        "5f9a1a558a7d96853766b4d26f5c80e6d29f46c661c0206af975c50af0ea1b1c"
    )
    assert torch.equal(result["samples"][:, :, 2:], samples[:, :, 2:])
    assert torch.equal(samples, source_before)
    assert result["metadata"] is latent["metadata"]
    assert result["noise_mask"] is latent["noise_mask"]


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


def test_normalize_video_latent_start_keeps_empty_reference_nan_semantics() -> None:
    samples = torch.arange(2 * 2 * 3 * 3 * 5, dtype=torch.float64).reshape(2, 2, 3, 3, 5)
    result = NormalizeVideoLatentStart.execute(
        latent={"samples": samples}, start_frame_count=9, reference_frame_count=2
    )["latent"]
    assert torch.isnan(result["samples"]).all()
    assert not torch.isnan(samples).any()


class VideoLatentTransformSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-latent-transform-fixture.source",
            outputs=(
                OutputSpec("video", TypeExpr.concrete(LATENT_TYPE)),
                OutputSpec("audio", TypeExpr.concrete(LATENT_TYPE)),
                OutputSpec("normalization", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            video={
                "samples": torch.arange(2 * 3 * 5 * 7 * 11, dtype=torch.float64).reshape(
                    2, 3, 5, 7, 11
                ),
                "noise_mask": torch.full((2, 1, 5, 1, 1), 0.75),
                "metadata": "video",
            },
            audio={
                "samples": torch.arange(2 * 3 * 5 * 11, dtype=torch.float64).reshape(2, 3, 5, 11),
                "noise_mask": torch.full((2, 1, 5, 1), 0.75),
                "metadata": "audio",
            },
            normalization={
                "samples": torch.cat(
                    (
                        torch.arange(2 * 2 * 6 * 3 * 5, dtype=torch.float64).reshape(2, 2, 6, 3, 5)[
                            :1
                        ],
                        torch.arange(2 * 2 * 6 * 3 * 5, dtype=torch.float64)
                        .reshape(2, 2, 6, 3, 5)[1:]
                        .flip(3)
                        .mul(-0.7)
                        .add(5.0),
                    )
                ),
                "noise_mask": torch.full((2, 1, 6, 1, 1), 0.25),
                "metadata": {"keep": ["value"]},
            },
        )


class VideoLatentTransformInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        latent = TypeExpr.concrete(LATENT_TYPE)
        return NodeSchema(
            node_type="video-latent-transform-fixture.inspect",
            inputs=tuple(
                InputSpec(name, latent)
                for name in ("video", "trimmed", "audio", "frozen", "normalized")
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **latents: object) -> Mapping[str, object]:
        video = materialize_latent(latents["video"])
        trimmed = materialize_latent(latents["trimmed"])
        audio = materialize_latent(latents["audio"])
        frozen = materialize_latent(latents["frozen"])
        normalized = materialize_latent(latents["normalized"])
        assert video["samples"].shape == (2, 3, 5, 7, 11)
        assert torch.equal(video["noise_mask"], torch.full((2, 1, 5, 1, 1), 0.75))
        assert trimmed["samples"].shape == (2, 3, 3, 7, 11)
        assert torch.equal(trimmed["noise_mask"], video["noise_mask"])
        assert trimmed["metadata"] == "video"
        assert audio["samples"].shape == (2, 3, 5, 11)
        assert torch.equal(audio["noise_mask"], torch.full((2, 1, 5, 1), 0.75))
        assert frozen["samples"].shape == (2, 3, 5, 11)
        assert frozen["noise_mask"].shape == (2, 1, 5, 1)
        assert frozen["noise_mask"].dtype == torch.float32
        assert not torch.count_nonzero(frozen["noise_mask"])
        assert frozen["metadata"] == "audio"
        assert normalized["samples"].shape == (2, 2, 6, 3, 5)
        assert hashlib.sha256(normalized["samples"].numpy().tobytes()).hexdigest() == (
            "5f9a1a558a7d96853766b4d26f5c80e6d29f46c661c0206af975c50af0ea1b1c"
        )
        assert normalized["metadata"] == {"keep": ["value"]}
        assert torch.equal(normalized["noise_mask"], torch.full((2, 1, 6, 1, 1), 0.25))
        return cls.outputs(summary="video-latent-transforms")


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_executes_video_latent_transforms_twice(
    in_process: bool, tmp_path: Path
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "video_latent_transform_fixture.py").write_text(
                "from test_video_latent_torch import (\n"
                "    VideoLatentTransformInspect, VideoLatentTransformSource,\n"
                ")\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, VideoLatentTransformSource, VideoLatentTransformInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["video-latent-transform-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "video_latent_transform_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "video_latent_transform_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("video-latent-transform-fixture.source", {}),
                    "trim": GraphNode(
                        "TrimVideoLatent", {"samples": Link("source", "video"), "trim_amount": 2}
                    ),
                    "freeze": GraphNode("LTXVFreezeLatent", {"latent": Link("source", "audio")}),
                    "normalize": GraphNode(
                        "NormalizeVideoLatentStart",
                        {
                            "latent": Link("source", "normalization"),
                            "start_frame_count": 2,
                            "reference_frame_count": 3,
                        },
                    ),
                    "inspect": GraphNode(
                        "video-latent-transform-fixture.inspect",
                        {
                            "video": Link("source", "video"),
                            "trimmed": Link("trim", "LATENT"),
                            "audio": Link("source", "audio"),
                            "frozen": Link("freeze", "latent"),
                            "normalized": Link("normalize", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "video-latent-transforms"
        finally:
            await composer.close()

    asyncio.run(scenario())
