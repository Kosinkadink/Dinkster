"""MiniMax guide media values through unchanged native pack hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
    register_audio_value_type,
    register_inference_types,
    resolver_from_env,
)
from dinkster_graph import Graph, GraphNode, Link
from minimax_pack_fixtures import conditioning_providers

from dinkster.compose import PackSpec, ServingComposer


def test_isolated_minimax_manifest_installs_only_consumed_codecs() -> None:
    from dinkster_minimax import register_types
    from dinkster_workers import load_manifest
    from dinkster_workers.host import load_pack

    manifest = load_manifest(Path(__file__).resolve().parents[1] / "dinkster-pack.toml")
    worker, registry, _, _ = load_pack(manifest)
    required = {
        "AUDIO",
        "CLIP",
        "CONDITIONING",
        "IMAGE",
        "LATENT",
        "MASK",
        "MODEL",
        "MODEL_PATCH",
        "VAE",
    }
    for type_id in required:
        assert registry.spec(type_id).declared_codec, type_id
        assert type_id in worker.registered_types
        assert registry.owner_of(type_id) != manifest.name
    for type_id in (
        "IC_LORA_PARAMETERS",
        "WAN_CAMERA_EMBEDDING",
        "TRACKS",
        "OPTICAL_FLOW",
        "SAMPLER",
    ):
        assert type_id not in registry, type_id
    register_types(registry)
    latent = {"samples": torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)}
    codec = registry.spec("LATENT")
    decoded = materialize_latent(codec.decode(codec.encode(latent)))
    torch.testing.assert_close(decoded["samples"], latent["samples"], rtol=0, atol=0)
    for type_id in ("CLIP", "MODEL", "MODEL_PATCH", "VAE"):
        resident = object()
        codec = registry.spec(type_id)
        assert codec.decode(codec.encode(resident)) is resident


class GuideVideoEncoder:
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.clone()


class GuideAudioEncoder:
    audio_sample_rate = 32000

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.movedim(-1, 1).unsqueeze(2).clone()


class GuideSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="minimax-guide-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("positive", CONDITIONING_TYPE),
                    ("vae", VAE_TYPE),
                    ("audio_vae", VAE_TYPE),
                    ("image", IMAGE_TYPE),
                    ("audio", AUDIO_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        image = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 2, 3, 4) / 5
        return cls.outputs(
            positive=[[torch.ones(1, 2, 3), {"minimax_keyframes": [{"resolved_frame_index": 2}]}]],
            vae=GuideVideoEncoder(),
            audio_vae=GuideAudioEncoder(),
            image=image,
            audio={
                "waveform": torch.arange(48, dtype=torch.float32).reshape(2, 2, 12) / 64,
                "sample_rate": 32000,
            },
        )


class GuideConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="minimax-guide-fixture.consumer",
            inputs=(InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any) -> dict[str, object]:
        keyframes = positive[0][1]["minimax_keyframes"]
        assert len(keyframes) == 3
        assert keyframes[0] == {"resolved_frame_index": 2}
        assert keyframes[1]["resolved_frame_index"] == 0
        assert keyframes[2]["resolved_frame_index"] == 17
        expected_image = (
            torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 32, 32, 3) / 5
        )
        torch.testing.assert_close(keyframes[1]["latent"], expected_image, rtol=0, atol=0)
        expected_audio = torch.arange(24, dtype=torch.float32).reshape(1, 2, 1, 12)[..., :8] / 64
        torch.testing.assert_close(keyframes[2]["audio_latent"], expected_audio, rtol=0, atol=0)
        assert keyframes[2]["audio_latent"].dtype == torch.float32
        return cls.outputs(summary="guide-image0-audio17-existing2")


HOST_NODES = (GuideSource, GuideConsumer)


def register_host_types(registry: Any) -> None:
    register_inference_types(registry)
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())


@pytest.mark.parametrize("in_process", (True, False))
def test_guides_run_twice_through_both_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "minimax_guide_fixture.py").write_text(
            "from test_minimax_h3_guide_hosts_torch import HOST_NODES, register_host_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="minimax-guide-fixture"\npublisher="dinkster"\n'
            'namespaces=["minimax-guide-fixture"]\n[pack.entry]\n'
            'nodes="minimax_guide_fixture:HOST_NODES"\n'
            'types="minimax_guide_fixture:register_host_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        providers = conditioning_providers(in_process)
        manifests = (*providers, native, fixture)
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        trust_reserved=True,
                        manifest=manifest,
                        packs=providers[manifest].packs if manifest in providers else None,
                        in_process=in_process,
                        worker_group=None if in_process else "minimax-guide-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("minimax-guide-fixture.source"),
                    "latent": GraphNode(
                        ("dinkster.minimax.empty_mini_max_h3_latent_av"),
                        {"width": 32, "height": 32, "length": 22},
                    ),
                    "image": GraphNode(
                        "dinkster.minimax.mini_max_h3_add_guide",
                        {
                            "positive": Link("source", "positive"),
                            "latent": Link("latent", "LATENT"),
                            "vae": Link("source", "vae"),
                            "image": Link("source", "image"),
                            "frame_idx": 0,
                        },
                    ),
                    "audio": GraphNode(
                        "dinkster.minimax.mini_max_h3_add_guide",
                        {
                            "positive": Link("image", "positive"),
                            "latent": Link("latent", "LATENT"),
                            "audio_vae": Link("source", "audio_vae"),
                            "audio": Link("source", "audio"),
                            "frame_idx": -5,
                        },
                    ),
                    "consumer": GraphNode(
                        "minimax-guide-fixture.consumer", {"positive": Link("audio", "positive")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert (
                    result.outputs["consumer"]["summary"].resolve()
                    == "guide-image0-audio17-existing2"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
