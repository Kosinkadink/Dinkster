"""Advanced Wan public values through unchanged native pack hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    SAM3_TRACK_DATA_TYPE,
    TRACKS_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


class VideoEncoder:
    latent_channels = 16

    def spacial_compression_encode(self) -> int:
        return 8

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image[::4, ::8, ::8, :1].movedim(-1, 0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


class AudioProjection(torch.nn.Module):
    seq_len = 5

    def forward(self, first: torch.Tensor, latter: torch.Tensor) -> torch.Tensor:
        return torch.cat((first.mean((2, 3)), latter.mean((2, 3))), dim=1).unsqueeze(2)


class TalkModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_proj = AudioProjection()


def _packed_masks(masks: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, dtype=torch.uint8)
    return (masks.to(torch.uint8).view(*masks.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan-advanced-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("vae", VAE_TYPE),
                    ("image", IMAGE_TYPE),
                    ("audio", AUDIO_ENCODER_OUTPUT_TYPE),
                    ("latent", LATENT_TYPE),
                    ("model", MODEL_TYPE),
                    ("model_patch", MODEL_PATCH_TYPE),
                    ("track_data", SAM3_TRACK_DATA_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        from dinkster_inference.model_patcher import CoreModelPatcher

        image = torch.stack([torch.full((16, 32, 3), value) for value in (0.25, 0.5, 0.75, 1.0)])
        audio = torch.arange(3 * 24, dtype=torch.float32).reshape(1, 24, 3)
        masks = torch.zeros((2, 1, 16, 32), dtype=torch.bool)
        masks[0, 0, 2:4, 3:5] = True
        masks[1, 0, 4:8, 8:12] = True
        return cls.outputs(
            positive=[[torch.ones(1, 2), {"source": "positive"}]],
            negative=[[torch.full((1, 2), 2.0), {"source": "negative"}]],
            vae=VideoEncoder(),
            image=image,
            audio={
                "encoded_audio": audio,
                "encoded_audio_all_layers": tuple(audio + 10 * layer for layer in range(33)),
                "audio_samples": 24 * 640,
            },
            latent={"samples": torch.full((1, 16, 3, 2, 4), 0.125)},
            model=CoreModelPatcher(TalkModel(), torch.device("cpu"), torch.device("cpu")),
            model_patch=SimpleNamespace(model=SimpleNamespace(audio_proj=AudioProjection())),
            track_data={
                "packed_masks": _packed_masks(masks),
                "orig_size": (16, 32),
                "n_frames": 2,
                "scores": [1.0],
            },
        )


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan-advanced-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("camera", CONDITIONING_TYPE),
                    ("move", CONDITIONING_TYPE),
                    ("sound", CONDITIONING_TYPE),
                    ("extend", CONDITIONING_TYPE),
                    ("humo", CONDITIONING_TYPE),
                    ("talk", MODEL_TYPE),
                    ("tracks", TRACKS_TYPE),
                    ("visualized", IMAGE_TYPE),
                    ("pose_mask", IMAGE_TYPE),
                    ("reference_mask", IMAGE_TYPE),
                    ("source_image", IMAGE_TYPE),
                    ("source_latent", LATENT_TYPE),
                    ("source_model", MODEL_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        camera: Any,
        move: Any,
        sound: Any,
        extend: Any,
        humo: Any,
        talk: Any,
        tracks: Any,
        visualized: Any,
        pose_mask: Any,
        reference_mask: Any,
        source_image: Any,
        source_latent: Any,
        source_model: Any,
    ) -> dict[str, object]:
        source = materialize_image(source_image)
        assert torch.equal(source[:, 0, 0, 0], torch.tensor([0.25, 0.5, 0.75, 1.0]))
        assert torch.equal(
            materialize_latent(source_latent)["samples"], torch.full((1, 16, 3, 2, 4), 0.125)
        )
        for conditioning in (camera, move, sound, extend, humo):
            assert conditioning[0][1]["source"] == "positive"
        assert camera[0][1]["camera_conditions"].shape == (1, 24, 2, 16, 32)
        assert move[0][1]["concat_latent_image"].shape == (1, 16, 2, 2, 4)
        assert sound[0][1]["audio_embed"].shape[-1] == 8
        assert extend[0][1]["reference_motion"].shape == (1, 16, 3, 2, 4)
        assert humo[0][1]["audio_embed"].shape[-2:] == (5, 3)
        assert "audio_embeds" in talk.model_options["transformer_options"]
        assert "audio_embeds" not in source_model.model_options["transformer_options"]
        track_values = materialize_latent(tracks)
        assert track_values["track_path"].shape == (2, 2, 2)
        assert track_values["track_visibility"].shape == (2, 2)
        assert materialize_image(visualized).shape == (2, 16, 32, 3)
        assert materialize_image(pose_mask).shape == (2, 16, 32, 3)
        assert materialize_image(reference_mask).shape == (1, 16, 32, 3)
        return cls.outputs(summary="advanced-wan-public-values")


HOST_NODES = (Source, Consumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_advanced_wan_nodes_run_twice_in_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "wan_advanced_host_entry.py").write_text(
            "from test_wan_advanced_hosts_torch import HOST_NODES, register_inference_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="wan-advanced-fixture"\npublisher="dinkster"\n'
            'namespaces=["wan-advanced-fixture"]\n[pack.entry]\n'
            'nodes="wan_advanced_host_entry:HOST_NODES"\n'
            'types="wan_advanced_host_entry:register_inference_types"\n',
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
                        worker_group=None if in_process else "wan-advanced-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            shared = {
                "positive": Link("source", "positive"),
                "negative": Link("source", "negative"),
                "vae": Link("source", "vae"),
                "width": 32,
                "height": 16,
                "length": 5,
                "batch_size": 1,
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("wan-advanced-fixture.source"),
                    "camera_embedding": GraphNode(
                        "WanCameraEmbedding",
                        {
                            "width": 32,
                            "height": 16,
                            "length": 5,
                            "camera_pose": "Static",
                            "fx": 0.5,
                            "fy": 0.5,
                            "cx": 0.5,
                            "cy": 0.5,
                        },
                    ),
                    "camera": GraphNode(
                        "WanCameraImageToVideo",
                        {
                            **shared,
                            "start_image": Link("source", "image"),
                            "camera_conditions": Link("camera_embedding", "camera_embedding"),
                        },
                    ),
                    "coords": GraphNode(
                        "WanMoveTracksFromCoords",
                        {"track_coords": "[{'x': 2, 'y': 3}, {'x': 8, 'y': 5}]"},
                    ),
                    "generated": GraphNode(
                        "GenerateTracks",
                        {
                            "width": 32,
                            "height": 16,
                            "start_x": 0.0,
                            "start_y": 0.0,
                            "end_x": 1.0,
                            "end_y": 1.0,
                            "num_frames": 2,
                            "num_tracks": 1,
                            "track_spread": 0.0,
                            "bezier": False,
                            "mid_x": 0.5,
                            "mid_y": 0.5,
                            "interpolation": "linear",
                        },
                    ),
                    "tracks": GraphNode(
                        "WanMoveConcatTrack",
                        {
                            "tracks_1": Link("coords", "TRACKS"),
                            "tracks_2": Link("generated", "TRACKS"),
                        },
                    ),
                    "move": GraphNode(
                        "WanMoveTrackToVideo",
                        {
                            **shared,
                            "tracks": Link("tracks", "TRACKS"),
                            "strength": 1.0,
                            "start_image": Link("source", "image"),
                        },
                    ),
                    "visualize": GraphNode(
                        "WanMoveVisualizeTracks",
                        {
                            "images": Link("masks", "pose_video_mask"),
                            "tracks": Link("tracks", "TRACKS"),
                            "line_resolution": 2,
                            "circle_size": 1,
                            "opacity": 1.0,
                            "line_width": 1,
                        },
                    ),
                    "sound": GraphNode(
                        "WanSoundImageToVideo",
                        {
                            **shared,
                            "audio_encoder_output": Link("source", "audio"),
                            "ref_image": Link("source", "image"),
                        },
                    ),
                    "extend": GraphNode(
                        "WanSoundImageToVideoExtend",
                        {
                            "positive": Link("source", "positive"),
                            "negative": Link("source", "negative"),
                            "vae": Link("source", "vae"),
                            "length": 5,
                            "video_latent": Link("source", "latent"),
                            "audio_encoder_output": Link("source", "audio"),
                        },
                    ),
                    "humo": GraphNode(
                        "WanHuMoImageToVideo",
                        {
                            **shared,
                            "audio_encoder_output": Link("source", "audio"),
                            "ref_image": Link("source", "image"),
                        },
                    ),
                    "talk": GraphNode(
                        "WanInfiniteTalkToVideo",
                        {
                            "model": Link("source", "model"),
                            "model_patch": Link("source", "model_patch"),
                            **{key: value for key, value in shared.items() if key != "batch_size"},
                            "start_image": Link("source", "image"),
                            "audio_encoder_output_1": Link("source", "audio"),
                            "motion_frame_count": 1,
                            "audio_scale": 1.0,
                        },
                        slot_variants={"mode": "single_speaker"},
                    ),
                    "masks": GraphNode(
                        "SCAIL2ColoredMask",
                        {
                            "driving_track_data": Link("source", "track_data"),
                            "object_indices": "",
                            "sort_by": "left_to_right",
                            "replacement_mode": False,
                        },
                    ),
                    "consumer": GraphNode(
                        "wan-advanced-fixture.consumer",
                        {
                            "camera": Link("camera", "positive"),
                            "move": Link("move", "positive"),
                            "sound": Link("sound", "positive"),
                            "extend": Link("extend", "positive"),
                            "humo": Link("humo", "positive"),
                            "talk": Link("talk", "model"),
                            "tracks": Link("tracks", "TRACKS"),
                            "visualized": Link("visualize", "IMAGE"),
                            "pose_mask": Link("masks", "pose_video_mask"),
                            "reference_mask": Link("masks", "reference_image_mask"),
                            "source_image": Link("source", "image"),
                            "source_latent": Link("source", "latent"),
                            "source_model": Link("source", "model"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert (
                    result.outputs["consumer"]["summary"].resolve() == "advanced-wan-public-values"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
