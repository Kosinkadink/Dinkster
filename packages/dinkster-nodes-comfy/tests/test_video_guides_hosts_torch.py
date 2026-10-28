"""Native video guides through ordinary in-process and isolated pack hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    GUIDER_TYPE,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    initialize_inference_process,
    materialize_latent,
    register_audio_value_type,
    resolver_from_env,
)
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


def register_fixture_types(registry: TypeRegistry) -> None:
    if AUDIO_TYPE not in registry:
        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    register_inference_types(registry)


class VideoVae:
    downscale_index_formula = (8, 32, 32)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels[::8, ::32, ::32, 0][None, None].repeat(1, 128, 1, 1, 1)


class AudioVae:
    audio_sample_rate = 16000

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform[:, :6].reshape(1, 2, 3, 2)


class Clip:
    def tokenize(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return {"prompt": prompt, **kwargs}

    def encode_from_tokens_scheduled(self, tokens: dict[str, Any]) -> list[Any]:
        return [[torch.ones(1, 2), {"tokens": tokens}]]


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-guides-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("latent", LATENT_TYPE),
                    ("guide", LATENT_TYPE),
                    ("vae", VAE_TYPE),
                    ("audio_vae", VAE_TYPE),
                    ("audio", AUDIO_TYPE),
                    ("image", IMAGE_TYPE),
                    ("model", MODEL_TYPE),
                    ("model_patch", MODEL_PATCH_TYPE),
                    ("clip", CLIP_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference.model_patcher import ModelPatcher

        model = torch.nn.Linear(1, 1)
        model.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 10 * (1 - percent))
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        return cls.outputs(
            positive=[[torch.ones(1, 2), {"keep": "positive"}]],
            negative=[[torch.zeros(1, 2), {"keep": "negative"}]],
            latent={"samples": torch.zeros(1, 128, 7, 2, 3), "keep": 17},
            guide={"samples": torch.full((1, 128, 1, 2, 3), 0.25)},
            vae=VideoVae(),
            audio_vae=AudioVae(),
            audio={"waveform": torch.arange(12.0).reshape(1, 2, 6), "sample_rate": 16000},
            image=torch.full((1, 64, 96, 3), 0.5),
            model=patcher,
            model_patch=SimpleNamespace(model=SimpleNamespace(injection_layers=[2, 5])),
            clip=Clip(),
        )


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-guides-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("latent", LATENT_TYPE),
                    ("keyframes", LATENT_TYPE),
                    ("guider", GUIDER_TYPE),
                    ("reference", CONDITIONING_TYPE),
                    ("reference_latent", LATENT_TYPE),
                    ("control", MODEL_TYPE),
                    ("source_model", MODEL_TYPE),
                    ("source_latent", LATENT_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **values: Any) -> dict[str, object]:
        samples = materialize_latent(values["latent"])
        assert samples["samples"].shape == (1, 128, 10, 2, 3)
        assert torch.equal(samples["samples"][:, :, -1:], torch.full((1, 128, 1, 2, 3), 0.25))
        assert samples["keep"] == 17
        keyframes = materialize_latent(values["keyframes"])
        assert keyframes["generated_keyframe_indices"] == [8, 16]
        assert keyframes["samples"].shape == (1, 128, 2, 2, 3)
        for name in ("positive", "negative"):
            metadata = values[name][0][1]
            assert metadata["keep"] == name
            assert metadata["keyframe_idxs"].shape == (1, 3, 18, 2)
            assert len(metadata["guide_attention_entries"]) == 3
        guider = values["guider"]
        assert len(guider.model_options["sampler_post_cfg_function"]) == 3
        ref = guider.original_conds["positive"][0]["ref_audio"]["tokens"]
        assert torch.equal(
            ref,
            torch.tensor([[[0.0, 6.0, 3.0, 9.0], [1.0, 7.0, 4.0, 10.0], [2.0, 8.0, 5.0, 11.0]]]),
        )
        guider.inner_model, guider.conds = None, guider.original_conds
        output = guider.predict_noise(
            torch.ones(1, 1, 2),
            torch.tensor([11.0]),
            {
                "sampler_calc_cond_batch_function": lambda _: (
                    torch.full((1, 1, 2), 4.0),
                    torch.ones(1, 1, 2),
                )
            },
        )
        assert torch.equal(output, torch.full((1, 1, 2), 10.0))
        reference = values["reference"][0][1]["tokens"]
        assert reference["prompt"] == "host references"
        assert len(reference["minimax_ref_items"]) == 1
        assert reference["minimax_ref_items"][0]["data"].shape == (1, 32, 64, 3)
        av = materialize_latent(values["reference_latent"])["samples"]
        assert isinstance(av, MultiStreamLatent)
        assert av.by_role("video").shape == (1, 24, 2, 2, 4)
        assert values["control"] is not values["source_model"]
        assert values["control"].model_options["transformer_options"]["patches_replace"][
            "dit"
        ].keys() == {("double_block", 2), ("double_block", 5)}
        assert values["source_model"].model_options == {"transformer_options": {}}
        assert torch.count_nonzero(materialize_latent(values["source_latent"])["samples"]) == 0
        return cls.outputs(summary="native-video-guides")


HOST_NODES = (Source, Consumer)


@pytest.mark.parametrize("in_process", [True, False])
def test_video_guides_run_twice_in_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "video_guides_entry.py").write_text(
            "from test_video_guides_hosts_torch import HOST_NODES, register_fixture_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="video-guides-fixture"\npublisher="dinkster"\n'
            'namespaces=["video-guides-fixture"]\n[pack.entry]\n'
            'nodes="video_guides_entry:HOST_NODES"\n'
            'types="video_guides_entry:register_fixture_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
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
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "video-guides-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            conds = {"positive": Link("source", "positive"), "negative": Link("source", "negative")}
            graph = Graph(
                nodes={
                    "source": GraphNode("video-guides-fixture.source"),
                    "generated": GraphNode(
                        "LTXVAddGeneratedKeyframes",
                        {
                            **conds,
                            "vae": Link("source", "vae"),
                            "latent": Link("source", "latent"),
                            "frame_indices": "8,16",
                        },
                    ),
                    "separated": GraphNode(
                        "LTXVSeparateGeneratedKeyframes",
                        {
                            "positive": Link("generated", "positive"),
                            "negative": Link("generated", "negative"),
                            "latent": Link("generated", "latent"),
                        },
                    ),
                    "converted": GraphNode(
                        "LTXVGeneratedKeyframesToGuides",
                        {
                            "positive": Link("separated", "positive"),
                            "negative": Link("separated", "negative"),
                            "latent": Link("separated", "latent"),
                            "vae": Link("source", "vae"),
                            "keyframes": Link("separated", "keyframes"),
                        },
                    ),
                    "attached": GraphNode(
                        "LTXVAddLatentGuide",
                        {
                            "positive": Link("converted", "positive"),
                            "negative": Link("converted", "negative"),
                            "latent": Link("converted", "latent"),
                            "vae": Link("source", "vae"),
                            "guiding_latent": Link("source", "guide"),
                            "latent_idx": 1,
                            "strength": 0.5,
                        },
                    ),
                    "stg": GraphNode(
                        "LTXVSpatioTemporalGuidance",
                        {"model": Link("source", "model"), "scale": 2.0, "blocks": "2,7"},
                    ),
                    "modality": GraphNode(
                        "LTXVModalityGuidance",
                        {"model": Link("stg", "MODEL"), "modality_scale": 3.0},
                    ),
                    "audio": GraphNode(
                        "LTXVReferenceAudio",
                        {
                            **conds,
                            "model": Link("modality", "MODEL"),
                            "reference_audio": Link("source", "audio"),
                            "audio_vae": Link("source", "audio_vae"),
                            "identity_guidance_scale": 2.0,
                        },
                    ),
                    "guider": GraphNode(
                        "LTXVDualCFGGuider",
                        {
                            "model": Link("audio", "MODEL"),
                            "positive": Link("audio", "positive"),
                            "negative": Link("audio", "negative"),
                        },
                    ),
                    "reference": GraphNode(
                        "MiniMaxH3ReferenceToVideo",
                        {
                            "clip": Link("source", "clip"),
                            "prompt": "host references",
                            "width": 64,
                            "height": 32,
                            "length": 5,
                            "ref_images.ref_image_1": Link("source", "image"),
                        },
                    ),
                    "control": GraphNode(
                        "MiniMaxH3FunControlNetApply",
                        {
                            "model": Link("source", "model"),
                            "model_patch": Link("source", "model_patch"),
                            "vae": Link("source", "vae"),
                            "control_video": Link("source", "image"),
                        },
                    ),
                    "consumer": GraphNode(
                        "video-guides-fixture.consumer",
                        {
                            "positive": Link("attached", "positive"),
                            "negative": Link("attached", "negative"),
                            "latent": Link("attached", "latent"),
                            "keyframes": Link("separated", "keyframes"),
                            "guider": Link("guider", "GUIDER"),
                            "reference": Link("reference", "positive"),
                            "reference_latent": Link("reference", "LATENT"),
                            "control": Link("control", "MODEL"),
                            "source_model": Link("source", "model"),
                            "source_latent": Link("source", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "native-video-guides"
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
