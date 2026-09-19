"""The ComfyUI benchmark driver's graph construction, phase extraction,
and report assembly, exercised without a ComfyUI server."""

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dinkster_workers.backend_env import validate_benchmark_report

from tests.test_backend_env import complete_benchmark_report, complete_report

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_comfyui.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_comfyui", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_comfyui = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_comfyui)

LOADER = benchmark_comfyui.LOADER_NODE
LORA = benchmark_comfyui.LORA_NODE
POSITIVE = benchmark_comfyui.POSITIVE_NODE
NEGATIVE = benchmark_comfyui.NEGATIVE_NODE
LATENT = benchmark_comfyui.LATENT_NODE
SAMPLER = benchmark_comfyui.SAMPLER_NODE
DECODE = benchmark_comfyui.DECODE_NODE
SINK = benchmark_comfyui.SINK_NODE
CLIP_LOADER = benchmark_comfyui.CLIP_LOADER_NODE
VAE_LOADER = benchmark_comfyui.VAE_LOADER_NODE
MODEL_SAMPLING = benchmark_comfyui.MODEL_SAMPLING_NODE
MODEL_PATCH = benchmark_comfyui.MODEL_PATCH_NODE
AUDIO_ENCODER = benchmark_comfyui.AUDIO_ENCODER_NODE
AUDIO_INPUT_1 = benchmark_comfyui.AUDIO_INPUT_1_NODE
AUDIO_INPUT_2 = benchmark_comfyui.AUDIO_INPUT_2_NODE
AUDIO_ENCODE_1 = benchmark_comfyui.AUDIO_ENCODE_1_NODE
AUDIO_ENCODE_2 = benchmark_comfyui.AUDIO_ENCODE_2_NODE
AUDIO_INPUT = benchmark_comfyui.AUDIO_INPUT_NODE
AUDIO_ENCODE = benchmark_comfyui.AUDIO_ENCODE_NODE
IMAGE_INPUT = benchmark_comfyui.IMAGE_INPUT_NODE
MASK_BASE = benchmark_comfyui.MASK_BASE_NODE
MASK_HALF = benchmark_comfyui.MASK_HALF_NODE
MASK_1 = benchmark_comfyui.MASK_1_NODE
MASK_2 = benchmark_comfyui.MASK_2_NODE
CONDITIONING = benchmark_comfyui.CONDITIONING_NODE
NOISE = benchmark_comfyui.NOISE_NODE
GUIDER = benchmark_comfyui.GUIDER_NODE
SAMPLER_SELECT = benchmark_comfyui.SAMPLER_SELECT_NODE
SCHEDULER = benchmark_comfyui.SCHEDULER_NODE
CLIP_VISION_LOADER = benchmark_comfyui.CLIP_VISION_LOADER_NODE
CLIP_VISION_ENCODE = benchmark_comfyui.CLIP_VISION_ENCODE_NODE
AUDIO_VAE_LOADER = benchmark_comfyui.AUDIO_VAE_LOADER_NODE
AUDIO_DECODE = benchmark_comfyui.AUDIO_DECODE_NODE
FLUX_GUIDANCE = benchmark_comfyui.FLUX_GUIDANCE_NODE
T5_TOKENIZER = benchmark_comfyui.T5_TOKENIZER_NODE


def build_graph(family: str, **overrides: Any) -> dict[str, Any]:
    keywords: dict[str, Any] = {
        "ckpt_name": "model.safetensors",
        "prompt": "an astronaut",
        "negative_prompt": "",
        "seed": 667,
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg": 7.0,
        "sampler": "euler",
        "scheduler": "simple",
    }
    keywords.update(overrides)
    return benchmark_comfyui.build_graph(family, **keywords)


def start(t: float, node: str, prompt_id: str = "p1") -> dict[str, Any]:
    return {"t": t, "event": "node_start", "node": node, "prompt_id": prompt_id}


def finish(t: float, node: str, prompt_id: str = "p1") -> dict[str, Any]:
    return {"t": t, "event": "node_finish", "node": node, "prompt_id": prompt_id}


def graph_edges(graph: dict[str, Any]) -> set[tuple[str, str, str, int]]:
    edges: set[tuple[str, str, str, int]] = set()

    def visit(target: str, name: str, value: object) -> None:
        if (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
            and value[0] in graph
            and isinstance(value[1], int)
        ):
            edges.add((target, name, value[0], value[1]))
        elif isinstance(value, dict):
            for child_name, child in value.items():
                visit(target, f"{name}.{child_name}", child)

    for target, node in graph.items():
        for name, value in node["inputs"].items():
            visit(target, name, value)
    return edges


def cold_events(prompt_id: str = "p1") -> list[dict[str, Any]]:
    return [
        start(0.0, LOADER, prompt_id),
        finish(10.0, LOADER, prompt_id),
        start(10.0, POSITIVE, prompt_id),
        finish(10.2, POSITIVE, prompt_id),
        start(10.2, NEGATIVE, prompt_id),
        finish(10.4, NEGATIVE, prompt_id),
        start(10.4, LATENT, prompt_id),
        finish(10.5, LATENT, prompt_id),
        start(10.5, SAMPLER, prompt_id),
        finish(15.5, SAMPLER, prompt_id),
        start(15.5, DECODE, prompt_id),
        finish(16.0, DECODE, prompt_id),
        start(16.0, SINK, prompt_id),
        finish(16.1, SINK, prompt_id),
    ]


class TestBuildGraph:
    def test_sd15_graph_wires_the_canonical_path(self) -> None:
        graph = build_graph("sd15")
        assert LORA not in graph
        assert graph[LOADER]["class_type"] == "CheckpointLoaderSimple"
        assert graph[SAMPLER]["inputs"]["model"] == [LOADER, 0]
        assert graph[POSITIVE]["inputs"]["clip"] == [LOADER, 1]
        assert graph[DECODE]["inputs"]["vae"] == [LOADER, 2]
        assert graph[DECODE]["inputs"]["samples"] == [SAMPLER, 0]
        assert graph[SINK]["class_type"] == "DinksterBenchmarkSink"
        assert graph[SINK]["inputs"]["images"] == [DECODE, 0]
        assert graph[SAMPLER]["inputs"]["denoise"] == 1.0

    def test_lora_graph_routes_model_and_clip_through_the_lora(self) -> None:
        graph = build_graph(
            "lora",
            lora_name="offset.safetensors",
            lora_strength_model=0.8,
            lora_strength_clip=0.6,
        )
        assert graph[LORA]["class_type"] == "LoraLoader"
        assert graph[LORA]["inputs"]["model"] == [LOADER, 0]
        assert graph[LORA]["inputs"]["strength_model"] == 0.8
        assert graph[LORA]["inputs"]["strength_clip"] == 0.6
        assert graph[SAMPLER]["inputs"]["model"] == [LORA, 0]
        assert graph[POSITIVE]["inputs"]["clip"] == [LORA, 1]
        # The VAE never routes through the LoRA.
        assert graph[DECODE]["inputs"]["vae"] == [LOADER, 2]

    def test_lora_family_requires_a_lora_name(self) -> None:
        with pytest.raises(ValueError, match="lora_name"):
            build_graph("lora")

    def test_workload_parameters_land_on_their_nodes(self) -> None:
        graph = build_graph("sd15", seed=9, steps=30, width=1024, height=768, cfg=5.5)
        assert graph[SAMPLER]["inputs"]["seed"] == 9
        assert graph[SAMPLER]["inputs"]["steps"] == 30
        assert graph[SAMPLER]["inputs"]["cfg"] == 5.5
        assert graph[LATENT]["inputs"] == {"width": 1024, "height": 768, "batch_size": 1}

    def test_zimage_graph_wires_the_split_loaders(self) -> None:
        graph = build_graph(
            "zimage",
            ckpt_name=None,
            diffusion_name="z_image_turbo_bf16.safetensors",
            text_encoder_name="qwen_3_4b.safetensors",
            vae_name="ae.safetensors",
        )
        assert graph[LOADER]["class_type"] == "UNETLoader"
        assert graph[LOADER]["inputs"] == {
            "unet_name": "z_image_turbo_bf16.safetensors",
            "weight_dtype": "default",
        }
        assert graph[CLIP_LOADER]["class_type"] == "CLIPLoader"
        assert graph[CLIP_LOADER]["inputs"] == {
            "clip_name": "qwen_3_4b.safetensors",
            "type": "lumina2",
        }
        assert graph[VAE_LOADER]["class_type"] == "VAELoader"
        assert graph[VAE_LOADER]["inputs"] == {"vae_name": "ae.safetensors"}
        assert graph[MODEL_SAMPLING]["class_type"] == "ModelSamplingAuraFlow"
        assert graph[MODEL_SAMPLING]["inputs"] == {"model": [LOADER, 0], "shift": 3.0}
        assert graph[SAMPLER]["inputs"]["model"] == [MODEL_SAMPLING, 0]
        assert graph[POSITIVE]["inputs"]["clip"] == [CLIP_LOADER, 0]
        assert graph[NEGATIVE]["class_type"] == "ConditioningZeroOut"
        assert graph[NEGATIVE]["inputs"] == {"conditioning": [POSITIVE, 0]}
        assert graph[DECODE]["inputs"]["vae"] == [VAE_LOADER, 0]
        assert graph[LATENT]["class_type"] == "EmptySD3LatentImage"
        assert graph[LATENT]["inputs"] == {"width": 512, "height": 512, "batch_size": 1}

    def test_wan21_graph_builds_a_video_latent(self) -> None:
        graph = build_graph(
            "wan21",
            ckpt_name=None,
            diffusion_name="wan2.1_t2v_1.3B_fp16.safetensors",
            text_encoder_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            vae_name="wan_2.1_vae.safetensors",
            length=33,
        )
        assert graph[CLIP_LOADER]["inputs"]["type"] == "wan"
        assert graph[MODEL_SAMPLING]["class_type"] == "ModelSamplingSD3"
        assert graph[MODEL_SAMPLING]["inputs"]["shift"] == 8.0
        assert graph[NEGATIVE]["class_type"] == "CLIPTextEncode"
        assert graph[LATENT]["class_type"] == "EmptyHunyuanLatentVideo"
        assert graph[LATENT]["inputs"] == {
            "width": 512,
            "height": 512,
            "batch_size": 1,
            "length": 33,
        }

    def test_anima_graph_matches_the_official_pinned_template_path(self) -> None:
        graph = build_graph(
            "anima",
            ckpt_name=None,
            diffusion_name="anima-base-v1.0.safetensors",
            text_encoder_name="qwen_3_06b_base.safetensors",
            vae_name="qwen_image_vae.safetensors",
            prompt=benchmark_comfyui.BENCHMARK_ANIMA_PROMPT,
            seed=875817230929465,
            steps=30,
            width=1024,
            height=1024,
            cfg=4.0,
            sampler="er_sde",
            scheduler="simple",
        )

        assert {node: entry["class_type"] for node, entry in graph.items()} == {
            LOADER: "UNETLoader",
            CLIP_LOADER: "CLIPLoader",
            VAE_LOADER: "VAELoader",
            LATENT: "EmptyLatentImage",
            POSITIVE: "CLIPTextEncode",
            NEGATIVE: "CLIPTextEncode",
            SAMPLER: "KSampler",
            DECODE: "VAEDecode",
            SINK: "DinksterBenchmarkSink",
        }
        assert graph[LOADER]["inputs"] == {
            "unet_name": "anima-base-v1.0.safetensors",
            "weight_dtype": "default",
        }
        assert graph[CLIP_LOADER]["inputs"] == {
            "clip_name": "qwen_3_06b_base.safetensors",
            "type": "stable_diffusion",
            "device": "default",
        }
        assert graph[VAE_LOADER]["inputs"] == {"vae_name": "qwen_image_vae.safetensors"}
        assert graph[LATENT]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
        assert graph[SAMPLER]["inputs"] == {
            "model": [LOADER, 0],
            "seed": 875817230929465,
            "steps": 30,
            "cfg": 4.0,
            "sampler_name": "er_sde",
            "scheduler": "simple",
            "positive": [POSITIVE, 0],
            "negative": [NEGATIVE, 0],
            "latent_image": [LATENT, 0],
            "denoise": 1.0,
        }
        assert graph_edges(graph) == {
            (POSITIVE, "clip", CLIP_LOADER, 0),
            (NEGATIVE, "clip", CLIP_LOADER, 0),
            (SAMPLER, "model", LOADER, 0),
            (SAMPLER, "positive", POSITIVE, 0),
            (SAMPLER, "negative", NEGATIVE, 0),
            (SAMPLER, "latent_image", LATENT, 0),
            (DECODE, "samples", SAMPLER, 0),
            (DECODE, "vae", VAE_LOADER, 0),
            (SINK, "images", DECODE, 0),
        }

    def test_infinitetalk_graph_matches_the_official_base_generation_path(self) -> None:
        graph = build_graph(
            "wan21_infinitetalk",
            ckpt_name=None,
            diffusion_name="wan_i2v.safetensors",
            text_encoder_name="umt5.safetensors",
            vae_name="wan_vae.safetensors",
            lora_name="lightx2v.safetensors",
            model_patch_name="multitalk.safetensors",
            audio_encoder_name="wav2vec2.safetensors",
            clip_vision_name="clip_vision_h.safetensors",
            input_image_name="two_character_talking.png",
            input_audio_1_name="speaker1.mp3",
            input_audio_2_name="speaker2.mp3",
            prompt="The camera zooms in. Two characters are talking.",
            seed=0,
            steps=6,
            width=832,
            height=480,
            length=81,
            cfg=1.0,
            sampler="euler",
            scheduler="normal",
            lora_strength_model=1.0,
            motion_frame_count=9,
            audio_scale=1.0,
        )
        expected_classes = {
            LOADER: "UNETLoader",
            LORA: "LoraLoaderModelOnly",
            POSITIVE: "CLIPTextEncode",
            NEGATIVE: "ConditioningZeroOut",
            SAMPLER: "SamplerCustomAdvanced",
            DECODE: "VAEDecode",
            SINK: "DinksterBenchmarkSink",
            CLIP_LOADER: "CLIPLoader",
            VAE_LOADER: "VAELoader",
            MODEL_PATCH: "ModelPatchLoader",
            AUDIO_ENCODER: "AudioEncoderLoader",
            CLIP_VISION_LOADER: "CLIPVisionLoader",
            CLIP_VISION_ENCODE: "CLIPVisionEncode",
            AUDIO_INPUT_1: "LoadAudio",
            AUDIO_INPUT_2: "LoadAudio",
            AUDIO_ENCODE_1: "AudioEncoderEncode",
            AUDIO_ENCODE_2: "AudioEncoderEncode",
            IMAGE_INPUT: "LoadImage",
            MASK_BASE: "SolidMask",
            MASK_HALF: "SolidMask",
            MASK_1: "MaskComposite",
            MASK_2: "MaskComposite",
            CONDITIONING: "WanInfiniteTalkToVideo",
            NOISE: "RandomNoise",
            GUIDER: "CFGGuider",
            SAMPLER_SELECT: "KSamplerSelect",
            SCHEDULER: "BasicScheduler",
        }
        assert {node: entry["class_type"] for node, entry in graph.items()} == expected_classes
        assert graph[LOADER]["inputs"] == {
            "unet_name": "wan_i2v.safetensors",
            "weight_dtype": "default",
        }
        assert graph[LORA]["inputs"]["strength_model"] == 1.0
        assert graph[CLIP_LOADER]["inputs"] == {
            "clip_name": "umt5.safetensors",
            "type": "wan",
            "device": "default",
        }
        assert graph[MODEL_PATCH]["inputs"] == {"name": "multitalk.safetensors"}
        assert graph[AUDIO_ENCODER]["inputs"] == {"audio_encoder_name": "wav2vec2.safetensors"}
        assert graph[CLIP_VISION_LOADER]["inputs"] == {"clip_name": "clip_vision_h.safetensors"}
        assert graph[CLIP_VISION_ENCODE]["inputs"]["crop"] == "center"
        assert graph[MASK_BASE]["inputs"] == {"value": 0.0, "width": 832, "height": 480}
        assert graph[MASK_HALF]["inputs"] == {"value": 1.0, "width": 416, "height": 480}
        assert graph[MASK_1]["inputs"]["x"] == 0
        assert graph[MASK_2]["inputs"]["x"] == 416
        assert graph[CONDITIONING]["inputs"]["mode"] == "two_speakers"
        assert graph[CONDITIONING]["inputs"]["motion_frame_count"] == 9
        assert graph[CONDITIONING]["inputs"]["audio_scale"] == 1.0
        assert graph[NOISE]["inputs"] == {"noise_seed": 0}
        assert graph[GUIDER]["inputs"]["cfg"] == 1.0
        assert graph[SAMPLER_SELECT]["inputs"] == {"sampler_name": "euler"}
        assert graph[SCHEDULER]["inputs"]["scheduler"] == "normal"
        assert graph[SCHEDULER]["inputs"]["steps"] == 6
        assert graph_edges(graph) == {
            (LORA, "model", LOADER, 0),
            (POSITIVE, "clip", CLIP_LOADER, 0),
            (NEGATIVE, "conditioning", POSITIVE, 0),
            (AUDIO_ENCODE_1, "audio_encoder", AUDIO_ENCODER, 0),
            (AUDIO_ENCODE_1, "audio", AUDIO_INPUT_1, 0),
            (AUDIO_ENCODE_2, "audio_encoder", AUDIO_ENCODER, 0),
            (AUDIO_ENCODE_2, "audio", AUDIO_INPUT_2, 0),
            (MASK_1, "destination", MASK_BASE, 0),
            (MASK_1, "source", MASK_HALF, 0),
            (MASK_2, "destination", MASK_BASE, 0),
            (MASK_2, "source", MASK_HALF, 0),
            (CONDITIONING, "mode.audio_encoder_output_2", AUDIO_ENCODE_2, 0),
            (CONDITIONING, "mode.mask_1", MASK_1, 0),
            (CONDITIONING, "mode.mask_2", MASK_2, 0),
            (CONDITIONING, "model", LORA, 0),
            (CONDITIONING, "model_patch", MODEL_PATCH, 0),
            (CONDITIONING, "positive", POSITIVE, 0),
            (CONDITIONING, "negative", NEGATIVE, 0),
            (CONDITIONING, "vae", VAE_LOADER, 0),
            (CONDITIONING, "clip_vision_output", CLIP_VISION_ENCODE, 0),
            (CONDITIONING, "audio_encoder_output_1", AUDIO_ENCODE_1, 0),
            (CONDITIONING, "start_image", IMAGE_INPUT, 0),
            (CLIP_VISION_ENCODE, "clip_vision", CLIP_VISION_LOADER, 0),
            (CLIP_VISION_ENCODE, "image", IMAGE_INPUT, 0),
            (GUIDER, "model", CONDITIONING, 0),
            (GUIDER, "positive", CONDITIONING, 1),
            (GUIDER, "negative", CONDITIONING, 2),
            (SCHEDULER, "model", CONDITIONING, 0),
            (SAMPLER, "noise", NOISE, 0),
            (SAMPLER, "guider", GUIDER, 0),
            (SAMPLER, "sampler", SAMPLER_SELECT, 0),
            (SAMPLER, "sigmas", SCHEDULER, 0),
            (SAMPLER, "latent_image", CONDITIONING, 3),
            (DECODE, "samples", SAMPLER, 0),
            (DECODE, "vae", VAE_LOADER, 0),
            (SINK, "images", DECODE, 0),
        }

    def test_infinitetalk_speaker_masks_are_exact_non_overlapping_halves(self) -> None:
        masks = np.zeros((2, 480, 832), dtype=np.float32)
        for mask, (x, y, width, height) in zip(
            masks, benchmark_comfyui.speaker_mask_regions(832, 480), strict=True
        ):
            mask[y : y + height, x : x + width] = 1.0
        assert np.all(masks[0, :, :416] == 1.0)
        assert np.all(masks[0, :, 416:] == 0.0)
        assert np.all(masks[1, :, :416] == 0.0)
        assert np.all(masks[1, :, 416:] == 1.0)
        assert np.all(masks.sum(axis=0) == 1.0)

    def test_humo_graph_matches_the_pinned_template_generation_path(self) -> None:
        graph = build_graph(
            "wan21_humo",
            ckpt_name=None,
            diffusion_name="humo_17B_fp8_e4m3fn.safetensors",
            text_encoder_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            vae_name="wan_2.1_vae.safetensors",
            lora_name="lightx2v_rank64.safetensors",
            audio_encoder_name="whisper_large_v3_fp16.safetensors",
            input_image_name="video_humo_reference_image.png",
            input_audio_name="video_humo_input_audio.wav",
            prompt=(
                "A young boy in sci-fi style clothing is talking to the camera in an alien desert."
            ),
            negative_prompt="negative",
            seed=0,
            steps=6,
            width=640,
            height=640,
            length=97,
            cfg=1.0,
            sampler="uni_pc",
            scheduler="simple",
        )
        assert {node: value["class_type"] for node, value in graph.items()} == {
            LOADER: "UNETLoader",
            LORA: "LoraLoaderModelOnly",
            POSITIVE: "CLIPTextEncode",
            NEGATIVE: "CLIPTextEncode",
            LATENT: "WanHuMoImageToVideo",
            SAMPLER: "KSampler",
            DECODE: "VAEDecode",
            SINK: "DinksterBenchmarkSink",
            CLIP_LOADER: "CLIPLoader",
            VAE_LOADER: "VAELoader",
            MODEL_SAMPLING: "ModelSamplingSD3",
            AUDIO_ENCODER: "AudioEncoderLoader",
            AUDIO_INPUT: "LoadAudio",
            AUDIO_ENCODE: "AudioEncoderEncode",
            IMAGE_INPUT: "LoadImage",
        }
        assert graph_edges(graph) == {
            (LORA, "model", LOADER, 0),
            (MODEL_SAMPLING, "model", LORA, 0),
            (POSITIVE, "clip", CLIP_LOADER, 0),
            (NEGATIVE, "clip", CLIP_LOADER, 0),
            (AUDIO_ENCODE, "audio_encoder", AUDIO_ENCODER, 0),
            (AUDIO_ENCODE, "audio", AUDIO_INPUT, 0),
            (LATENT, "positive", POSITIVE, 0),
            (LATENT, "negative", NEGATIVE, 0),
            (LATENT, "vae", VAE_LOADER, 0),
            (LATENT, "audio_encoder_output", AUDIO_ENCODE, 0),
            (LATENT, "ref_image", IMAGE_INPUT, 0),
            (SAMPLER, "model", MODEL_SAMPLING, 0),
            (SAMPLER, "positive", LATENT, 0),
            (SAMPLER, "negative", LATENT, 1),
            (SAMPLER, "latent_image", LATENT, 2),
            (DECODE, "samples", SAMPLER, 0),
            (DECODE, "vae", VAE_LOADER, 0),
            (SINK, "images", DECODE, 0),
        }
        assert graph[CLIP_LOADER]["inputs"]["device"] == "default"
        assert graph[LORA]["inputs"]["strength_model"] == 1.0
        assert graph[MODEL_SAMPLING]["inputs"]["shift"] == 8.0
        assert graph[LATENT]["inputs"]["width"] == 640
        assert graph[LATENT]["inputs"]["height"] == 640
        assert graph[LATENT]["inputs"]["length"] == 97
        assert graph[LATENT]["inputs"]["batch_size"] == 1
        assert graph[SAMPLER]["inputs"]["seed"] == 0
        assert graph[SAMPLER]["inputs"]["steps"] == 6
        assert graph[SAMPLER]["inputs"]["cfg"] == 1.0
        assert graph[SAMPLER]["inputs"]["sampler_name"] == "uni_pc"
        assert graph[SAMPLER]["inputs"]["scheduler"] == "simple"
        assert graph[SAMPLER]["inputs"]["denoise"] == 1.0

    def test_minimax_h3_graph_matches_the_official_non_turbo_path(self) -> None:
        graph = build_graph(
            "minimax_h3",
            ckpt_name=None,
            diffusion_name="minimax_h3_fl2va_int8_convrot.safetensors",
            text_encoder_name="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            vae_name="minimax_h3_video_vae_fp16.safetensors",
            audio_vae_name="minimax_h3_audio_vae_fp32.safetensors",
            prompt="A red square centered on a black background.",
            negative_prompt="",
            seed=20260813,
            steps=20,
            width=1344,
            height=768,
            length=124,
            cfg=1.0,
            sampler="res_multistep",
            scheduler="simple",
        )
        assert {node: value["class_type"] for node, value in graph.items()} == {
            LOADER: "UNETLoader",
            CLIP_LOADER: "CLIPLoader",
            VAE_LOADER: "VAELoader",
            AUDIO_VAE_LOADER: "VAELoader",
            CONDITIONING: "MiniMaxH3ImageToVideo",
            NOISE: "RandomNoise",
            GUIDER: "BasicGuider",
            SAMPLER_SELECT: "KSamplerSelect",
            SCHEDULER: "BasicScheduler",
            SAMPLER: "SamplerCustomAdvanced",
            DECODE: "VAEDecode",
            AUDIO_DECODE: "VAEDecodeAudio",
            SINK: "DinksterBenchmarkSink",
        }
        assert graph[LOADER]["inputs"] == {
            "unet_name": "minimax_h3_fl2va_int8_convrot.safetensors",
            "weight_dtype": "default",
        }
        assert graph[CLIP_LOADER]["inputs"] == {
            "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "type": "minimax",
            "device": "default",
        }
        assert graph[CONDITIONING]["inputs"] == {
            "clip": [CLIP_LOADER, 0],
            "vae": [VAE_LOADER, 0],
            "prompt": "A red square centered on a black background.",
            "width": 1344,
            "height": 768,
            "length": 124,
        }
        assert "first_frame" not in graph[CONDITIONING]["inputs"]
        assert "last_frame" not in graph[CONDITIONING]["inputs"]
        assert graph[NOISE]["inputs"] == {"noise_seed": 20260813}
        assert graph[SAMPLER_SELECT]["inputs"] == {"sampler_name": "res_multistep"}
        assert graph[SCHEDULER]["inputs"]["scheduler"] == "simple"
        assert graph[SCHEDULER]["inputs"]["steps"] == 20
        assert graph[SCHEDULER]["inputs"]["denoise"] == 1.0
        assert graph_edges(graph) == {
            (CONDITIONING, "clip", CLIP_LOADER, 0),
            (CONDITIONING, "vae", VAE_LOADER, 0),
            (GUIDER, "model", LOADER, 0),
            (GUIDER, "conditioning", CONDITIONING, 0),
            (SCHEDULER, "model", LOADER, 0),
            (SAMPLER, "noise", NOISE, 0),
            (SAMPLER, "guider", GUIDER, 0),
            (SAMPLER, "sampler", SAMPLER_SELECT, 0),
            (SAMPLER, "sigmas", SCHEDULER, 0),
            (SAMPLER, "latent_image", CONDITIONING, 1),
            (DECODE, "samples", SAMPLER, 0),
            (DECODE, "vae", VAE_LOADER, 0),
            (AUDIO_DECODE, "samples", SAMPLER, 0),
            (AUDIO_DECODE, "vae", AUDIO_VAE_LOADER, 0),
            (SINK, "images", DECODE, 0),
            (SINK, "audio", AUDIO_DECODE, 0),
        }

    def test_flux_graph_routes_the_positive_through_flux_guidance(self) -> None:
        graph = build_graph(
            "flux",
            ckpt_name=None,
            diffusion_name="flux1-dev.safetensors",
            clip_l_name="clip_l.safetensors",
            text_encoder_name="t5xxl_fp16.safetensors",
            vae_name="ae.safetensors",
            prompt="a photograph of an astronaut riding a horse",
            width=1024,
            height=1024,
            cfg=1.0,
            guidance=3.5,
        )
        assert graph[LOADER]["class_type"] == "UNETLoader"
        assert graph[LOADER]["inputs"] == {
            "unet_name": "flux1-dev.safetensors",
            "weight_dtype": "default",
        }
        assert graph[CLIP_LOADER]["class_type"] == "DualCLIPLoader"
        assert graph[CLIP_LOADER]["inputs"] == {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5xxl_fp16.safetensors",
            "type": "flux",
        }
        assert graph[VAE_LOADER]["inputs"] == {"vae_name": "ae.safetensors"}
        # Flux samples on its flat built-in schedule: no ModelSampling patch,
        # matching the Dinkster runner's family default.
        assert MODEL_SAMPLING not in graph
        assert graph[LATENT]["class_type"] == "EmptySD3LatentImage"
        assert graph[LATENT]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
        assert graph[POSITIVE]["inputs"]["clip"] == [CLIP_LOADER, 0]
        # The negative prompt is encoded, not zeroed, on both systems.
        assert graph[NEGATIVE]["class_type"] == "CLIPTextEncode"
        assert graph[NEGATIVE]["inputs"] == {"text": "", "clip": [CLIP_LOADER, 0]}
        assert graph[FLUX_GUIDANCE]["class_type"] == "FluxGuidance"
        assert graph[FLUX_GUIDANCE]["inputs"] == {
            "conditioning": [POSITIVE, 0],
            "guidance": 3.5,
        }
        assert graph[SAMPLER]["inputs"]["model"] == [LOADER, 0]
        assert graph[SAMPLER]["inputs"]["positive"] == [FLUX_GUIDANCE, 0]
        assert graph[SAMPLER]["inputs"]["negative"] == [NEGATIVE, 0]
        assert graph[SAMPLER]["inputs"]["cfg"] == 1.0
        assert graph[DECODE]["inputs"]["vae"] == [VAE_LOADER, 0]

    def test_flux_family_requires_its_model_names_and_guidance(self) -> None:
        names = {
            "ckpt_name": None,
            "diffusion_name": "flux1-dev.safetensors",
            "clip_l_name": "clip_l.safetensors",
            "text_encoder_name": "t5xxl_fp16.safetensors",
            "vae_name": "ae.safetensors",
        }
        with pytest.raises(ValueError, match="requires diffusion, clip_l"):
            build_graph("flux", **{**names, "clip_l_name": None}, guidance=3.5)
        with pytest.raises(ValueError, match="requires a guidance value"):
            build_graph("flux", **names, guidance=None)

    def test_chroma_graph_routes_both_encodes_through_tokenizer_options(self) -> None:
        graph = build_graph(
            "chroma",
            ckpt_name=None,
            diffusion_name="Chroma1-HD-fp8mixed.safetensors",
            text_encoder_name="t5xxl_fp8_e4m3fn_scaled.safetensors",
            vae_name="ae.safetensors",
            prompt="a photograph of an astronaut riding a horse",
            seed=667,
            steps=26,
            width=1024,
            height=1024,
            cfg=3.5,
            sampler="euler",
            scheduler="beta",
        )
        assert graph[LOADER]["class_type"] == "UNETLoader"
        assert graph[LOADER]["inputs"] == {
            "unet_name": "Chroma1-HD-fp8mixed.safetensors",
            "weight_dtype": "default",
        }
        assert graph[CLIP_LOADER]["class_type"] == "CLIPLoader"
        assert graph[CLIP_LOADER]["inputs"] == {
            "clip_name": "t5xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "chroma",
        }
        assert graph[VAE_LOADER]["inputs"] == {"vae_name": "ae.safetensors"}
        assert graph[MODEL_SAMPLING]["class_type"] == "ModelSamplingAuraFlow"
        assert graph[MODEL_SAMPLING]["inputs"] == {"model": [LOADER, 0], "shift": 1.0}
        assert graph[T5_TOKENIZER]["class_type"] == "T5TokenizerOptions"
        assert graph[T5_TOKENIZER]["inputs"] == {
            "clip": [CLIP_LOADER, 0],
            "min_padding": 0,
            "min_length": 0,
        }
        assert graph[POSITIVE]["inputs"]["clip"] == [T5_TOKENIZER, 0]
        # The negative prompt is encoded, not zeroed, on both systems.
        assert graph[NEGATIVE]["class_type"] == "CLIPTextEncode"
        assert graph[NEGATIVE]["inputs"] == {"text": "", "clip": [T5_TOKENIZER, 0]}
        assert graph[LATENT]["class_type"] == "EmptySD3LatentImage"
        assert graph[LATENT]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
        assert graph[SAMPLER]["inputs"]["model"] == [MODEL_SAMPLING, 0]
        assert graph[SAMPLER]["inputs"]["positive"] == [POSITIVE, 0]
        assert graph[SAMPLER]["inputs"]["negative"] == [NEGATIVE, 0]
        assert graph[SAMPLER]["inputs"]["seed"] == 667
        assert graph[SAMPLER]["inputs"]["steps"] == 26
        assert graph[SAMPLER]["inputs"]["cfg"] == 3.5
        assert graph[SAMPLER]["inputs"]["sampler_name"] == "euler"
        assert graph[SAMPLER]["inputs"]["scheduler"] == "beta"
        assert FLUX_GUIDANCE not in graph
        assert graph[DECODE]["inputs"]["vae"] == [VAE_LOADER, 0]

    def test_split_family_requires_all_three_model_names(self) -> None:
        with pytest.raises(ValueError, match="diffusion, text encoder, and VAE"):
            build_graph("zimage", ckpt_name=None, diffusion_name="d.safetensors")

    def test_wan21_family_requires_a_length(self) -> None:
        with pytest.raises(ValueError, match="length"):
            build_graph(
                "wan21",
                ckpt_name=None,
                diffusion_name="d.safetensors",
                text_encoder_name="t.safetensors",
                vae_name="v.safetensors",
            )


class TestNodeIntervals:
    def test_intervals_span_start_to_finish(self) -> None:
        intervals = benchmark_comfyui.node_intervals(cold_events(), "p1")
        assert intervals[LOADER] == pytest.approx(10.0)
        assert intervals[POSITIVE] == pytest.approx(0.2)
        assert intervals[NEGATIVE] == pytest.approx(0.2)
        assert intervals[SAMPLER] == pytest.approx(5.0)
        assert intervals[DECODE] == pytest.approx(0.5)
        assert intervals[SINK] == pytest.approx(0.1)

    def test_other_prompts_events_are_ignored(self) -> None:
        events = cold_events() + cold_events("p2")
        intervals = benchmark_comfyui.node_intervals(events, "p1")
        assert intervals[LOADER] == pytest.approx(10.0)

    def test_cached_node_finish_without_start_gets_no_interval(self) -> None:
        events = [
            finish(0.1, LOADER),
            start(0.2, SAMPLER),
            finish(4.2, SAMPLER),
        ]
        intervals = benchmark_comfyui.node_intervals(events, "p1")
        assert LOADER not in intervals
        assert intervals[SAMPLER] == pytest.approx(4.0)

    def test_started_but_unfinished_node_raises(self) -> None:
        events = [start(0.0, LOADER), finish(1.0, LOADER), start(1.0, SAMPLER)]
        with pytest.raises(RuntimeError, match="never finished"):
            benchmark_comfyui.node_intervals(events, "p1")


class TestPhases:
    def test_cold_phases_map_nodes_onto_report_names(self) -> None:
        intervals = benchmark_comfyui.node_intervals(cold_events(), "p1")
        phases = benchmark_comfyui.cold_phases(intervals, "sd15")
        assert phases["load_s"] == pytest.approx(10.0)
        assert phases["encode_s"] == pytest.approx(0.4)
        assert phases["sample_s"] == pytest.approx(5.0)
        assert phases["decode_s"] == pytest.approx(0.5)
        assert phases["total_s"] == pytest.approx(15.9)
        assert "lora_s" not in phases

    def test_cold_phases_include_the_lora_interval_for_the_lora_family(self) -> None:
        intervals = {
            LOADER: 10.0,
            LORA: 1.5,
            POSITIVE: 0.2,
            NEGATIVE: 0.2,
            SAMPLER: 5.0,
            DECODE: 0.5,
        }
        phases = benchmark_comfyui.cold_phases(intervals, "lora")
        assert phases["lora_s"] == pytest.approx(1.5)
        assert phases["total_s"] == pytest.approx(17.4)

    def test_cold_phases_missing_node_raises(self) -> None:
        with pytest.raises(RuntimeError, match="sample_s"):
            benchmark_comfyui.cold_phases({LOADER: 1.0, POSITIVE: 0.1, NEGATIVE: 0.1}, "sd15")

    def test_warm_phases_cover_sample_and_decode(self) -> None:
        entry = benchmark_comfyui.warm_phases({SAMPLER: 4.0, DECODE: 0.5, SINK: 0.1})
        assert entry == {"sample_s": 4.0, "decode_s": 0.5, "total_s": 4.5}

    def test_warm_phases_reject_reexecuted_cold_nodes(self) -> None:
        with pytest.raises(RuntimeError, match="cold-only"):
            benchmark_comfyui.warm_phases({LOADER: 1.0, SAMPLER: 4.0, DECODE: 0.5})

    def test_cold_phases_sum_the_split_loaders_into_load(self) -> None:
        intervals = {
            LOADER: 6.0,
            CLIP_LOADER: 2.0,
            VAE_LOADER: 1.0,
            MODEL_SAMPLING: 0.5,
            POSITIVE: 0.2,
            NEGATIVE: 0.2,
            SAMPLER: 5.0,
            DECODE: 0.5,
        }
        phases = benchmark_comfyui.cold_phases(intervals, "zimage")
        assert phases["load_s"] == pytest.approx(9.5)
        assert phases["total_s"] == pytest.approx(15.4)
        assert "lora_s" not in phases

    def test_cold_phases_missing_split_loader_raises(self) -> None:
        intervals = {
            LOADER: 6.0,
            POSITIVE: 0.2,
            NEGATIVE: 0.2,
            SAMPLER: 5.0,
            DECODE: 0.5,
        }
        with pytest.raises(RuntimeError, match="load_s"):
            benchmark_comfyui.cold_phases(intervals, "wan21")

    def test_anima_cold_load_uses_only_the_three_loader_intervals(self) -> None:
        intervals = {
            LOADER: 6.0,
            CLIP_LOADER: 2.0,
            VAE_LOADER: 1.0,
            POSITIVE: 0.2,
            NEGATIVE: 0.2,
            SAMPLER: 5.0,
            DECODE: 0.5,
        }
        phases = benchmark_comfyui.cold_phases(intervals, "anima")
        assert phases["load_s"] == pytest.approx(9.0)
        assert phases["total_s"] == pytest.approx(14.9)

        with pytest.raises(RuntimeError, match="load_s"):
            benchmark_comfyui.cold_phases(
                {node: seconds for node, seconds in intervals.items() if node != CLIP_LOADER},
                "anima",
            )

    def test_warm_phases_reject_reexecuted_split_loaders(self) -> None:
        with pytest.raises(RuntimeError, match="cold-only"):
            benchmark_comfyui.warm_phases({CLIP_LOADER: 1.0, SAMPLER: 4.0, DECODE: 0.5})

    def test_flux_cold_phases_count_the_guidance_node_as_encode(self) -> None:
        intervals = {
            LOADER: 6.0,
            CLIP_LOADER: 2.0,
            VAE_LOADER: 1.0,
            POSITIVE: 0.2,
            NEGATIVE: 0.2,
            FLUX_GUIDANCE: 0.1,
            SAMPLER: 5.0,
            DECODE: 0.5,
        }
        phases = benchmark_comfyui.cold_phases(intervals, "flux")
        assert phases["load_s"] == pytest.approx(9.0)
        assert phases["encode_s"] == pytest.approx(0.5)
        assert phases["total_s"] == pytest.approx(15.0)

        with pytest.raises(RuntimeError, match="encode_s"):
            benchmark_comfyui.cold_phases(
                {node: seconds for node, seconds in intervals.items() if node != FLUX_GUIDANCE},
                "flux",
            )

    def test_warm_phases_reject_a_reexecuted_flux_guidance_node(self) -> None:
        with pytest.raises(RuntimeError, match="cold-only"):
            benchmark_comfyui.warm_phases({FLUX_GUIDANCE: 0.1, SAMPLER: 4.0, DECODE: 0.5})

    def test_chroma_cold_phases_count_the_tokenizer_options_node_as_encode(self) -> None:
        intervals = {
            LOADER: 6.0,
            CLIP_LOADER: 2.0,
            VAE_LOADER: 1.0,
            MODEL_SAMPLING: 0.5,
            T5_TOKENIZER: 0.1,
            POSITIVE: 0.2,
            NEGATIVE: 0.2,
            SAMPLER: 5.0,
            DECODE: 0.5,
        }
        phases = benchmark_comfyui.cold_phases(intervals, "chroma")
        assert phases["load_s"] == pytest.approx(9.5)
        assert phases["encode_s"] == pytest.approx(0.5)
        assert phases["total_s"] == pytest.approx(15.5)

        with pytest.raises(RuntimeError, match="encode_s"):
            benchmark_comfyui.cold_phases(
                {node: seconds for node, seconds in intervals.items() if node != T5_TOKENIZER},
                "chroma",
            )

    def test_warm_phases_reject_a_reexecuted_tokenizer_options_node(self) -> None:
        with pytest.raises(RuntimeError, match="cold-only"):
            benchmark_comfyui.warm_phases({T5_TOKENIZER: 0.1, SAMPLER: 4.0, DECODE: 0.5})

    def test_infinitetalk_cold_phases_include_audio_conditioning(self) -> None:
        load_nodes = (
            LOADER,
            LORA,
            CLIP_LOADER,
            VAE_LOADER,
            MODEL_PATCH,
            AUDIO_ENCODER,
            CLIP_VISION_LOADER,
        )
        audio_nodes = (
            AUDIO_INPUT_1,
            AUDIO_INPUT_2,
            AUDIO_ENCODE_1,
            AUDIO_ENCODE_2,
            IMAGE_INPUT,
            CLIP_VISION_ENCODE,
            MASK_BASE,
            MASK_HALF,
            MASK_1,
            MASK_2,
            CONDITIONING,
        )
        sample_nodes = (GUIDER, SAMPLER_SELECT, SCHEDULER, NOISE, SAMPLER)
        intervals = {
            **{node: 1.0 for node in load_nodes},
            POSITIVE: 0.2,
            NEGATIVE: 0.1,
            **{node: 0.5 for node in audio_nodes},
            **{node: 0.4 for node in sample_nodes},
            DECODE: 0.7,
        }
        phases = benchmark_comfyui.cold_phases(intervals, "wan21_infinitetalk")
        assert phases == {
            "load_s": 7.0,
            "encode_s": 0.3,
            "audio_encode_s": 5.5,
            "sample_s": 2.0,
            "decode_s": 0.7,
            "total_s": 15.5,
        }

    def test_infinitetalk_warm_phase_includes_fresh_noise(self) -> None:
        entry = benchmark_comfyui.warm_phases(
            {NOISE: 0.1, SAMPLER: 4.0, DECODE: 0.5, SINK: 0.1},
            "wan21_infinitetalk",
        )
        assert entry == {"sample_s": 4.1, "decode_s": 0.5, "total_s": 4.6}

    def test_humo_cold_phases_include_audio_conditioning(self) -> None:
        intervals = {
            **{
                node: 1.0
                for node in (
                    LOADER,
                    LORA,
                    CLIP_LOADER,
                    VAE_LOADER,
                    MODEL_SAMPLING,
                    AUDIO_ENCODER,
                )
            },
            POSITIVE: 0.2,
            NEGATIVE: 0.1,
            AUDIO_INPUT: 0.3,
            AUDIO_ENCODE: 0.8,
            IMAGE_INPUT: 0.2,
            LATENT: 1.2,
            SAMPLER: 4.0,
            DECODE: 0.7,
        }
        assert benchmark_comfyui.cold_phases(intervals, "wan21_humo") == {
            "load_s": 6.0,
            "encode_s": 0.3,
            "audio_encode_s": 2.5,
            "sample_s": 4.0,
            "decode_s": 0.7,
            "total_s": 13.5,
        }

    def test_minimax_h3_phases_cover_both_decoders_and_warm_noise(self) -> None:
        intervals = {
            LOADER: 4.0,
            CLIP_LOADER: 3.0,
            VAE_LOADER: 2.0,
            AUDIO_VAE_LOADER: 1.0,
            CONDITIONING: 0.5,
            NOISE: 0.1,
            GUIDER: 0.1,
            SAMPLER_SELECT: 0.1,
            SCHEDULER: 0.2,
            SAMPLER: 5.0,
            DECODE: 0.7,
            AUDIO_DECODE: 0.3,
        }
        assert benchmark_comfyui.cold_phases(intervals, "minimax_h3") == {
            "load_s": 10.0,
            "encode_s": 0.5,
            "sample_s": 5.5,
            "decode_s": 1.0,
            "total_s": 17.0,
        }
        assert benchmark_comfyui.warm_phases(
            {NOISE: 0.1, SAMPLER: 4.0, DECODE: 0.5, AUDIO_DECODE: 0.25},
            "minimax_h3",
        ) == {"sample_s": 4.1, "decode_s": 0.75, "total_s": 4.85}


class TestSamplerStepWall:
    def progress(
        self, t: float, value: float, node: str = SAMPLER, max_value: float = 2.0
    ) -> dict[str, Any]:
        return {
            "t": t,
            "event": "progress",
            "node": node,
            "prompt_id": "p1",
            "value": value,
            "max": max_value,
        }

    def test_one_boundary_per_step(self) -> None:
        events = [
            start(10.0, SAMPLER),
            self.progress(10.25, 1.0, max_value=3.0),
            self.progress(10.5, 2.0, max_value=3.0),
            self.progress(10.75, 3.0, max_value=3.0),
        ]
        wall = benchmark_comfyui.sampler_step_wall_ms(events, "p1", 3)
        assert wall == [250.0, 250.0, 250.0]

    def test_zero_and_non_increasing_values_are_skipped(self) -> None:
        events = [
            start(10.0, SAMPLER),
            self.progress(10.0, 0.0),
            self.progress(10.25, 1.0),
            self.progress(10.3, 1.0),
            self.progress(10.5, 2.0),
        ]
        assert benchmark_comfyui.sampler_step_wall_ms(events, "p1", 2) == [250.0, 250.0]

    def test_count_mismatch_returns_none(self) -> None:
        events = [start(10.0, SAMPLER), self.progress(10.25, 1.0)]
        assert benchmark_comfyui.sampler_step_wall_ms(events, "p1", 2) is None

    def test_other_nodes_progress_is_ignored(self) -> None:
        events = [
            start(10.0, SAMPLER),
            self.progress(10.25, 1.0),
            self.progress(10.4, 1.0, node=DECODE),
            self.progress(10.5, 2.0),
        ]
        assert benchmark_comfyui.sampler_step_wall_ms(events, "p1", 2) == [250.0, 250.0]

    def test_other_max_progress_is_ignored(self) -> None:
        # Model weight loading reports progress on the sampler node with
        # its own max; only ticks whose max equals the step count are
        # sampling steps.
        events = [
            start(10.0, SAMPLER),
            self.progress(10.02, 100.0, max_value=218.0),
            self.progress(10.05, 218.0, max_value=218.0),
            self.progress(10.25, 1.0),
            self.progress(10.5, 2.0),
        ]
        assert benchmark_comfyui.sampler_step_wall_ms(events, "p1", 2) == [250.0, 250.0]

    def test_missing_max_returns_none(self) -> None:
        events = [
            start(10.0, SAMPLER),
            {"t": 10.25, "event": "progress", "node": SAMPLER, "prompt_id": "p1", "value": 1.0},
            {"t": 10.5, "event": "progress", "node": SAMPLER, "prompt_id": "p1", "value": 2.0},
        ]
        assert benchmark_comfyui.sampler_step_wall_ms(events, "p1", 2) is None


class TestMiniMaxH3SinkEvidence:
    def cell(self) -> Any:
        return benchmark_comfyui.ComfyCell(
            SimpleNamespace(family="minimax_h3", length=124, height=768, width=1344),
            None,
        )

    def observation(self) -> dict[str, Any]:
        return {
            "finite": True,
            "shape": [124, 768, 1344, 3],
            "audio_finite": True,
            "audio_shape": [1, 2, 165_333],
            "audio_sample_rate": 32_000,
        }

    def test_exact_video_and_stereo_audio_observation_passes(self) -> None:
        cell = self.cell()
        cell.observations = [self.observation()]
        assert "all values finite" in cell.finite_output()

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("shape", [123, 768, 1344, 3], "video shape"),
            ("audio_shape", [1, 1, 165_333], "audio shape"),
            ("audio_sample_rate", 44_100, "sample rate"),
            ("audio_finite", False, "non-finite"),
        ],
    )
    def test_invalid_av_observation_is_rejected(
        self, field: str, value: object, expected: str
    ) -> None:
        cell = self.cell()
        observation = self.observation()
        observation[field] = value
        cell.observations = [observation]
        with pytest.raises(RuntimeError, match=expected):
            cell.finite_output()

    def test_quality_capture_uses_a_fresh_post_measurement_seed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capture = {
            "version": 1,
            "seed": 20260817,
            "image": {"path": str(tmp_path / "capture_image.npy")},
            "audio": {"path": str(tmp_path / "capture_audio.npy")},
            "audio_sample_rate": 32_000,
        }
        observation = {**self.observation(), "quality_capture": capture}
        calls: list[object] = []

        class Server:
            def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
                calls.append((path, payload))
                return {"ok": True}

            def get(self, path: str) -> dict[str, Any]:
                calls.append(path)
                return {
                    "events": [],
                    "observations": [self_observation.copy() for _ in range(4)] + [observation],
                }

        self_observation = self.observation()
        arguments = SimpleNamespace(
            family="minimax_h3",
            length=124,
            height=768,
            width=1344,
            seed=20260813,
            warm_runs=3,
            cell_timeout=10.0,
            quality_output_dir=tmp_path,
        )
        cell = benchmark_comfyui.ComfyCell(arguments, Server())
        monkeypatch.setattr(cell, "_graph", lambda seed: {"seed": seed})
        monkeypatch.setattr(
            benchmark_comfyui,
            "run_prompt",
            lambda server, graph, timeout: calls.append(("prompt", graph, timeout)),
        )

        assert "seed 20260817" in cell.capture_quality()
        assert cell.quality_capture == capture
        assert calls[0] == ("/dinkster_benchmark/arm_quality", {"seed": 20260817})
        assert calls[1] == ("prompt", {"seed": 20260817}, 10.0)


class TestArguments:
    def infinitetalk_arguments(self, tmp_path: Path) -> list[str]:
        comfyui_root = tmp_path / "ComfyUI"
        comfyui_root.mkdir()
        (comfyui_root / "main.py").write_text("")
        arguments = [
            "--backend",
            "cuda",
            "--family",
            "wan21_infinitetalk",
            "--comfyui-root",
            str(comfyui_root),
            "--comfyui-python",
            str(tmp_path / "python"),
        ]
        for option in (
            "diffusion",
            "text-encoder",
            "vae",
            "lora",
            "model-patch",
            "audio-encoder",
            "clip-vision",
            "input-image",
            "input-audio-1",
            "input-audio-2",
        ):
            arguments.extend((f"--{option}", str(tmp_path / f"{option}.bin")))
        return arguments

    def humo_arguments(self, tmp_path: Path) -> list[str]:
        comfyui_root = tmp_path / "ComfyUI"
        comfyui_root.mkdir(exist_ok=True)
        (comfyui_root / "main.py").write_text("")
        arguments = [
            "--backend",
            "cuda",
            "--family",
            "wan21_humo",
            "--comfyui-root",
            str(comfyui_root),
            "--comfyui-python",
            str(tmp_path / "python"),
        ]
        for option in (
            "diffusion",
            "text-encoder",
            "vae",
            "lora",
            "audio-encoder",
            "input-image",
            "input-audio",
        ):
            arguments.extend((f"--{option}", str(tmp_path / f"{option}.bin")))
        return arguments

    def anima_arguments(self, tmp_path: Path) -> list[str]:
        comfyui_root = tmp_path / "ComfyUI"
        comfyui_root.mkdir(exist_ok=True)
        (comfyui_root / "main.py").write_text("")
        arguments = [
            "--backend",
            "cuda",
            "--family",
            "anima",
            "--comfyui-root",
            str(comfyui_root),
            "--comfyui-python",
            str(tmp_path / "python"),
        ]
        for option in ("diffusion", "text-encoder", "vae"):
            arguments.extend((f"--{option}", str(tmp_path / f"{option}.safetensors")))
        return arguments

    def minimax_h3_arguments(self, tmp_path: Path) -> list[str]:
        comfyui_root = tmp_path / "ComfyUI"
        comfyui_root.mkdir(exist_ok=True)
        (comfyui_root / "main.py").write_text("")
        arguments = [
            "--backend",
            "cuda",
            "--family",
            "minimax_h3",
            "--comfyui-root",
            str(comfyui_root),
            "--comfyui-python",
            str(tmp_path / "python"),
        ]
        for option in ("diffusion", "text-encoder", "vae", "audio-vae"):
            arguments.extend((f"--{option}", str(tmp_path / f"{option}.bin")))
        return arguments

    def flux_arguments(self, tmp_path: Path) -> list[str]:
        comfyui_root = tmp_path / "ComfyUI"
        comfyui_root.mkdir(exist_ok=True)
        (comfyui_root / "main.py").write_text("")
        arguments = [
            "--backend",
            "cuda",
            "--family",
            "flux",
            "--comfyui-root",
            str(comfyui_root),
            "--comfyui-python",
            str(tmp_path / "python"),
        ]
        for option in ("diffusion", "clip-l", "text-encoder", "vae"):
            arguments.extend((f"--{option}", str(tmp_path / f"{option}.safetensors")))
        return arguments

    def chroma_arguments(self, tmp_path: Path) -> list[str]:
        comfyui_root = tmp_path / "ComfyUI"
        comfyui_root.mkdir(exist_ok=True)
        (comfyui_root / "main.py").write_text("")
        arguments = [
            "--backend",
            "cuda",
            "--family",
            "chroma",
            "--comfyui-root",
            str(comfyui_root),
            "--comfyui-python",
            str(tmp_path / "python"),
        ]
        for option in ("diffusion", "text-encoder", "vae"):
            arguments.extend((f"--{option}", str(tmp_path / f"{option}.safetensors")))
        return arguments

    def test_infinitetalk_cli_defaults_are_the_pinned_workload(self, tmp_path: Path) -> None:
        arguments = benchmark_comfyui._parse_arguments(self.infinitetalk_arguments(tmp_path))
        assert arguments.prompt == "The camera zooms in. Two characters are talking."
        assert arguments.negative_prompt == ""
        assert arguments.seed == 0
        assert arguments.steps == 6
        assert arguments.width == 832
        assert arguments.height == 480
        assert arguments.length == 81
        assert arguments.cfg == 1.0
        assert arguments.sampler == "euler"
        assert arguments.scheduler == "normal"
        assert arguments.warm_runs == 3
        assert arguments.motion_frame_count == 9
        assert arguments.audio_scale == 1.0
        assert arguments.lora_strength_model == 1.0

    def test_infinitetalk_cli_rejects_a_missing_required_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        arguments = self.infinitetalk_arguments(tmp_path)
        index = arguments.index("--clip-vision")
        del arguments[index : index + 2]
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(arguments)
        assert "requires --clip-vision" in capsys.readouterr().err

    def test_infinitetalk_cli_rejects_non_4k_plus_1_length(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        arguments = [*self.infinitetalk_arguments(tmp_path), "--length", "82"]
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(arguments)
        assert "4k+1" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("option", "value"),
        [("--sampler", "uni_pc"), ("--lora-strength-model", "0.5")],
    )
    def test_infinitetalk_cli_rejects_non_pinned_settings(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        option: str,
        value: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.infinitetalk_arguments(tmp_path), option, value]
            )
        assert "for wan21_infinitetalk" in capsys.readouterr().err

    def test_humo_cli_defaults_are_the_pinned_workload(self, tmp_path: Path) -> None:
        arguments = benchmark_comfyui._parse_arguments(self.humo_arguments(tmp_path))
        assert arguments.prompt == (
            "A young boy in sci-fi style clothing is talking to the camera in an alien desert."
        )
        assert arguments.negative_prompt.startswith("\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd")
        assert arguments.negative_prompt.endswith(
            "\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70"
        )
        assert arguments.seed == 0
        assert arguments.steps == 6
        assert arguments.width == 640
        assert arguments.height == 640
        assert arguments.length == 97
        assert arguments.cfg == 1.0
        assert arguments.sampler == "uni_pc"
        assert arguments.scheduler == "simple"
        assert arguments.warm_runs == 3
        assert arguments.lora_strength_model == 1.0
        assert arguments.require_commit == benchmark_comfyui.COMFYUI_PIN

    @pytest.mark.parametrize(
        "override",
        ["", "b78cec87", "f" * 40],
    )
    def test_humo_cli_rejects_a_commit_override(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        override: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.humo_arguments(tmp_path), "--require-commit", override]
            )
        assert "--require-commit must be" in capsys.readouterr().err

    def test_humo_cli_rejects_a_missing_required_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        arguments = self.humo_arguments(tmp_path)
        index = arguments.index("--audio-encoder")
        del arguments[index : index + 2]
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(arguments)
        assert "requires --audio-encoder" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("option", "value"),
        [("--sampler", "euler"), ("--length", "93"), ("--lora-strength-model", "0.5")],
    )
    def test_humo_cli_rejects_non_pinned_settings(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        option: str,
        value: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments([*self.humo_arguments(tmp_path), option, value])
        assert "for wan21_humo" in capsys.readouterr().err

    def test_anima_cli_defaults_are_the_pinned_primary_workload(self, tmp_path: Path) -> None:
        arguments = benchmark_comfyui._parse_arguments(self.anima_arguments(tmp_path))
        assert arguments.prompt == benchmark_comfyui.BENCHMARK_ANIMA_PROMPT
        assert arguments.negative_prompt == ""
        assert arguments.seed == 875817230929465
        assert arguments.steps == 30
        assert (arguments.width, arguments.height) == (1024, 1024)
        assert arguments.cfg == 4.0
        assert arguments.sampler == "er_sde"
        assert arguments.scheduler == "simple"
        assert arguments.warm_runs == 5
        assert arguments.require_commit == benchmark_comfyui.COMFYUI_PIN
        assert arguments.fallback_768 is False

    @pytest.mark.parametrize("override", ["", "b78cec87", "f" * 40])
    def test_anima_cli_rejects_a_commit_override(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        override: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.anima_arguments(tmp_path), "--require-commit", override]
            )
        assert "--require-commit must be" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (("--sampler", "euler"), "for anima"),
            (("--seed", "0"), "for anima"),
            (("--width", "768"), "Anima geometry"),
        ],
    )
    def test_anima_cli_rejects_noncanonical_workload_settings(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        extra: tuple[str, str],
        message: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments([*self.anima_arguments(tmp_path), *extra])
        assert message.lower() in capsys.readouterr().err.lower()

    def test_anima_cli_rejects_the_unlabeled_symmetric_fallback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.anima_arguments(tmp_path), "--width", "768", "--height", "768"]
            )
        assert "requires --fallback-768" in capsys.readouterr().err

    def test_anima_cli_accepts_the_explicit_fallback(self, tmp_path: Path) -> None:
        arguments = benchmark_comfyui._parse_arguments(
            [*self.anima_arguments(tmp_path), "--fallback-768"]
        )
        assert (arguments.width, arguments.height) == (768, 768)
        assert arguments.fallback_768 is True

    def test_anima_cli_rejects_a_missing_required_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        arguments = self.anima_arguments(tmp_path)
        index = arguments.index("--text-encoder")
        del arguments[index : index + 2]
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(arguments)
        assert "requires --text-encoder" in capsys.readouterr().err

    def test_minimax_h3_cli_defaults_are_the_pinned_workload(self, tmp_path: Path) -> None:
        arguments = benchmark_comfyui._parse_arguments(self.minimax_h3_arguments(tmp_path))
        assert arguments.prompt == "A red square centered on a black background."
        assert arguments.negative_prompt == ""
        assert arguments.seed == 20260813
        assert arguments.steps == 20
        assert arguments.width == 1344
        assert arguments.height == 768
        assert arguments.length == 124
        assert arguments.cfg == 1.0
        assert arguments.sampler == "res_multistep"
        assert arguments.scheduler == "simple"
        assert arguments.warm_runs == 3
        assert arguments.require_commit == benchmark_comfyui.COMFYUI_PIN
        assert arguments.attention_policy == "auto"
        assert arguments.quality_output_dir is None

    @pytest.mark.parametrize("policy", ["sdpa", "dinkster_kitchen_int8", "sage"])
    def test_minimax_h3_cli_accepts_explicit_attention_evidence(
        self, tmp_path: Path, policy: str
    ) -> None:
        output = tmp_path / "quality"
        arguments = benchmark_comfyui._parse_arguments(
            [
                *self.minimax_h3_arguments(tmp_path),
                "--attention-policy",
                policy,
                "--quality-output-dir",
                str(output),
            ]
        )
        assert arguments.attention_policy == policy
        assert arguments.quality_output_dir == output

    def test_attention_evidence_options_are_rejected_for_other_families(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.flux_arguments(tmp_path), "--attention-policy", "sage"]
            )
        assert "only with MiniMax H3" in capsys.readouterr().err

    @pytest.mark.parametrize("override", ["", "b78cec87", "f" * 40])
    def test_minimax_h3_cli_rejects_a_commit_override(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        override: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.minimax_h3_arguments(tmp_path), "--require-commit", override]
            )
        assert "--require-commit must be" in capsys.readouterr().err

    def test_minimax_h3_cli_requires_the_audio_vae(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        arguments = self.minimax_h3_arguments(tmp_path)
        index = arguments.index("--audio-vae")
        del arguments[index : index + 2]
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(arguments)
        assert "requires --audio-vae" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("option", "value"),
        [("--seed", "0"), ("--length", "107"), ("--sampler", "euler")],
    )
    def test_minimax_h3_cli_rejects_non_pinned_settings(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        option: str,
        value: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.minimax_h3_arguments(tmp_path), option, value]
            )
        assert "for minimax_h3" in capsys.readouterr().err

    def test_flux_cli_defaults_are_the_pinned_workload(self, tmp_path: Path) -> None:
        arguments = benchmark_comfyui._parse_arguments(self.flux_arguments(tmp_path))
        assert arguments.prompt == "a photograph of an astronaut riding a horse"
        assert arguments.negative_prompt == ""
        assert arguments.seed == 667
        assert arguments.steps == 20
        assert arguments.width == 1024
        assert arguments.height == 1024
        assert arguments.cfg == 1.0
        assert arguments.guidance == 3.5
        assert arguments.sampler == "euler"
        assert arguments.scheduler == "simple"
        assert arguments.warm_runs == 5
        assert arguments.require_commit == benchmark_comfyui.COMFYUI_PIN

    @pytest.mark.parametrize("override", ["", "b78cec87", "f" * 40])
    def test_flux_cli_rejects_a_commit_override(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        override: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.flux_arguments(tmp_path), "--require-commit", override]
            )
        assert "--require-commit must be" in capsys.readouterr().err

    def test_flux_cli_requires_the_clip_l_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        arguments = self.flux_arguments(tmp_path)
        index = arguments.index("--clip-l")
        del arguments[index : index + 2]
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(arguments)
        assert "requires --clip-l" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("option", "value"),
        [("--seed", "0"), ("--guidance", "4.0"), ("--sampler", "uni_pc")],
    )
    def test_flux_cli_rejects_non_pinned_settings(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        option: str,
        value: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments([*self.flux_arguments(tmp_path), option, value])
        assert "for flux" in capsys.readouterr().err

    def test_chroma_cli_defaults_are_the_pinned_workload(self, tmp_path: Path) -> None:
        arguments = benchmark_comfyui._parse_arguments(self.chroma_arguments(tmp_path))
        assert arguments.prompt == "a photograph of an astronaut riding a horse"
        assert arguments.negative_prompt == ""
        assert arguments.seed == 667
        assert arguments.steps == 26
        assert arguments.width == 1024
        assert arguments.height == 1024
        assert arguments.cfg == 3.5
        assert arguments.guidance is None
        assert arguments.sampler == "euler"
        assert arguments.scheduler == "beta"
        assert arguments.warm_runs == 5
        assert arguments.require_commit == benchmark_comfyui.COMFYUI_PIN

    @pytest.mark.parametrize("override", ["", "b78cec87", "f" * 40])
    def test_chroma_cli_rejects_a_commit_override(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        override: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments(
                [*self.chroma_arguments(tmp_path), "--require-commit", override]
            )
        assert "--require-commit must be" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("option", "value"),
        [("--seed", "0"), ("--steps", "20"), ("--cfg", "1.0"), ("--sampler", "uni_pc")],
    )
    def test_chroma_cli_rejects_non_pinned_settings(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        option: str,
        value: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments([*self.chroma_arguments(tmp_path), option, value])
        assert "for chroma" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (("--clip-l", "clip_l.safetensors"), "--clip-l is only meaningful"),
            (("--guidance", "3.5"), "--guidance is only meaningful"),
        ],
    )
    def test_flux_only_options_are_rejected_elsewhere(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        extra: tuple[str, str],
        message: str,
    ) -> None:
        with pytest.raises(SystemExit):
            benchmark_comfyui._parse_arguments([*self.anima_arguments(tmp_path), *extra])
        assert message in capsys.readouterr().err


class TestArtifactPins:
    def test_pin_rejects_the_wrong_bytes_and_digest(self, tmp_path: Path) -> None:
        artifact = tmp_path / "input.bin"
        artifact.write_bytes(b"input")
        pin = (6, "0" * 64, "https://example.invalid/immutable/input.bin")
        with pytest.raises(ValueError, match="does not match its pin"):
            benchmark_comfyui._artifact_entry("input_image", artifact, pin)

    def test_infinitetalk_pins_cover_every_report_role(self) -> None:
        assert set(benchmark_comfyui._INFINITETALK_ARTIFACT_PINS) == {
            "diffusion",
            "text_encoder",
            "vae",
            "lora",
            "model_patch",
            "audio_encoder",
            "clip_vision",
            "input_image",
            "input_audio_1",
            "input_audio_2",
        }
        for size, digest, url in benchmark_comfyui._INFINITETALK_ARTIFACT_PINS.values():
            assert size > 0
            assert len(digest) == 64
            assert "resolve/main/" not in url

    def test_humo_pins_cover_every_report_role(self) -> None:
        assert set(benchmark_comfyui._HUMO_ARTIFACT_PINS) == {
            "diffusion",
            "text_encoder",
            "vae",
            "lora",
            "audio_encoder",
            "input_image",
            "input_audio",
        }
        for size, digest, url in benchmark_comfyui._HUMO_ARTIFACT_PINS.values():
            assert size > 0
            assert len(digest) == 64
            assert "resolve/main/" not in url

    def test_anima_pins_cover_every_report_role(self) -> None:
        assert set(benchmark_comfyui._ANIMA_ARTIFACT_PINS) == {
            "diffusion",
            "text_encoder",
            "vae",
        }
        for size, digest, url in benchmark_comfyui._ANIMA_ARTIFACT_PINS.values():
            assert size > 0
            assert len(digest) == 64
            assert "resolve/main/" not in url

    def test_minimax_h3_pins_cover_every_report_role(self) -> None:
        assert set(benchmark_comfyui._MINIMAX_H3_ARTIFACT_PINS) == {
            "diffusion",
            "text_encoder",
            "video_vae",
            "audio_vae",
        }
        for size, digest, url in benchmark_comfyui._MINIMAX_H3_ARTIFACT_PINS.values():
            assert size > 0
            assert len(digest) == 64
            assert "resolve/main/" not in url

    def test_flux_pins_cover_every_report_role(self) -> None:
        assert set(benchmark_comfyui._FLUX_ARTIFACT_PINS) == {
            "diffusion",
            "clip_l",
            "text_encoder",
            "vae",
        }
        for size, digest, url in benchmark_comfyui._FLUX_ARTIFACT_PINS.values():
            assert size > 0
            assert len(digest) == 64
            assert "resolve/main/" not in url

    def test_chroma_pins_cover_every_report_role(self) -> None:
        assert set(benchmark_comfyui._CHROMA_ARTIFACT_PINS) == {
            "diffusion",
            "text_encoder",
            "vae",
        }
        for size, digest, url in benchmark_comfyui._CHROMA_ARTIFACT_PINS.values():
            assert size > 0
            assert len(digest) == 64
            assert "resolve/main/" not in url

    def test_entrypoint_refuses_unpinned_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arguments = TestArguments().infinitetalk_arguments(tmp_path)
        for value in arguments:
            path = Path(value)
            if path.suffix == ".bin":
                path.write_bytes(b"wrong")
        monkeypatch.setattr(benchmark_comfyui.sys, "argv", ["benchmark_comfyui.py", *arguments])
        monkeypatch.setattr(benchmark_comfyui, "enforce_checkout", lambda root, commit: "b78cec87")

        with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
            benchmark_comfyui.main()

    def test_humo_entrypoint_refuses_unpinned_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arguments = TestArguments().humo_arguments(tmp_path)
        for value in arguments:
            path = Path(value)
            if path.suffix == ".bin":
                path.write_bytes(b"wrong")
        monkeypatch.setattr(benchmark_comfyui.sys, "argv", ["benchmark_comfyui.py", *arguments])
        monkeypatch.setattr(benchmark_comfyui, "enforce_checkout", lambda root, commit: commit)
        with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
            benchmark_comfyui.main()

    def test_anima_entrypoint_refuses_unpinned_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arguments = TestArguments().anima_arguments(tmp_path)
        for value in arguments:
            path = Path(value)
            if path.suffix == ".safetensors":
                path.write_bytes(b"wrong")
        monkeypatch.setattr(benchmark_comfyui.sys, "argv", ["benchmark_comfyui.py", *arguments])
        monkeypatch.setattr(benchmark_comfyui, "enforce_checkout", lambda root, commit: commit)
        with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
            benchmark_comfyui.main()

    def test_flux_entrypoint_refuses_unpinned_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arguments = TestArguments().flux_arguments(tmp_path)
        for value in arguments:
            path = Path(value)
            if path.suffix == ".safetensors":
                path.write_bytes(b"wrong")
        monkeypatch.setattr(benchmark_comfyui.sys, "argv", ["benchmark_comfyui.py", *arguments])
        monkeypatch.setattr(benchmark_comfyui, "enforce_checkout", lambda root, commit: commit)
        with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
            benchmark_comfyui.main()

    def test_chroma_entrypoint_refuses_unpinned_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arguments = TestArguments().chroma_arguments(tmp_path)
        for value in arguments:
            path = Path(value)
            if path.suffix == ".safetensors":
                path.write_bytes(b"wrong")
        monkeypatch.setattr(benchmark_comfyui.sys, "argv", ["benchmark_comfyui.py", *arguments])
        monkeypatch.setattr(benchmark_comfyui, "enforce_checkout", lambda root, commit: commit)
        with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
            benchmark_comfyui.main()

    def test_minimax_h3_entrypoint_refuses_unpinned_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arguments = TestArguments().minimax_h3_arguments(tmp_path)
        for value in arguments:
            path = Path(value)
            if path.suffix == ".bin":
                path.write_bytes(b"wrong")
        monkeypatch.setattr(benchmark_comfyui.sys, "argv", ["benchmark_comfyui.py", *arguments])
        monkeypatch.setattr(benchmark_comfyui, "enforce_checkout", lambda root, commit: commit)
        with pytest.raises(ValueError, match="diffusion artifact does not match its pin"):
            benchmark_comfyui.main()


class TestEnforceCheckout:
    PIN = benchmark_comfyui.COMFYUI_PIN

    def _git_stub(self, commit: str | None, status: str | None) -> Any:
        def fake_git(root: Path, *args: str) -> str | None:
            if args == ("rev-parse", "HEAD"):
                return commit
            if args == ("status", "--porcelain"):
                return status
            raise AssertionError(f"unexpected git call {args}")

        return fake_git

    def test_clean_checkout_at_the_pin_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(benchmark_comfyui, "_git", self._git_stub(self.PIN, ""))
        assert benchmark_comfyui.enforce_checkout(Path("/checkout"), self.PIN) == self.PIN

    def test_wrong_commit_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(benchmark_comfyui, "_git", self._git_stub("f" * 40, ""))
        with pytest.raises(SystemExit, match="does not start with required commit"):
            benchmark_comfyui.enforce_checkout(Path("/checkout"), self.PIN)

    def test_unanswerable_commit_lookup_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(benchmark_comfyui, "_git", self._git_stub(None, ""))
        with pytest.raises(SystemExit, match="does not start with required commit"):
            benchmark_comfyui.enforce_checkout(Path("/checkout"), self.PIN)

    def test_dirty_tree_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(benchmark_comfyui, "_git", self._git_stub(self.PIN, " M nodes.py"))
        with pytest.raises(SystemExit, match="local modifications"):
            benchmark_comfyui.enforce_checkout(Path("/checkout"), self.PIN)

    def test_status_command_failure_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(benchmark_comfyui, "_git", self._git_stub(self.PIN, None))
        with pytest.raises(SystemExit, match="refusing to record"):
            benchmark_comfyui.enforce_checkout(Path("/checkout"), self.PIN)

    def test_empty_require_commit_skips_enforcement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_on_status(root: Path, *args: str) -> str | None:
            assert args == ("rev-parse", "HEAD")
            return "f" * 40

        monkeypatch.setattr(benchmark_comfyui, "_git", fail_on_status)
        assert benchmark_comfyui.enforce_checkout(Path("/checkout"), "") == "f" * 40

    def test_git_failure_returns_none(self, tmp_path: Path) -> None:
        assert benchmark_comfyui._git(tmp_path / "missing", "rev-parse", "HEAD") is None


class TestExtraModelPathsConfig:
    def test_config_points_at_artifacts_and_the_shim(self) -> None:
        checkpoints = Path("/models/checkpoints")
        loras = Path("/models/loras")
        text = benchmark_comfyui.extra_model_paths_config(checkpoints, loras)
        assert text.startswith("dinkster_benchmark:\n")
        # str(Path) is separator-native, so the expectations must be too.
        assert f"  checkpoints: '{checkpoints}'\n" in text
        assert f"  loras: '{loras}'\n" in text
        assert "comfyui_benchmark_nodes" in text

    def test_lora_directory_is_optional(self) -> None:
        assert "loras" not in benchmark_comfyui.extra_model_paths_config(Path("/models"))

    def test_split_directories_replace_the_checkpoint_entry(self) -> None:
        diffusion = Path("/models/diffusion_models")
        text_encoders = Path("/models/text_encoders")
        vae = Path("/models/vae")
        text = benchmark_comfyui.extra_model_paths_config(
            diffusion_directory=diffusion,
            text_encoder_directory=text_encoders,
            vae_directory=vae,
        )
        assert "checkpoints" not in text
        assert f"  diffusion_models: '{diffusion}'\n" in text
        assert f"  text_encoders: '{text_encoders}'\n" in text
        assert f"  vae: '{vae}'\n" in text
        assert "comfyui_benchmark_nodes" in text

    def test_infinitetalk_directories_use_the_pin_native_model_folders(self) -> None:
        model_patches = Path("/models/model_patches")
        audio_encoders = Path("/models/audio_encoders")
        clip_vision = Path("/models/clip_vision")
        text = benchmark_comfyui.extra_model_paths_config(
            model_patch_directory=model_patches,
            audio_encoder_directory=audio_encoders,
            clip_vision_directory=clip_vision,
        )
        # str(Path) is separator-native, so the expectations must be too.
        assert f"  model_patches: '{model_patches}'\n" in text
        assert f"  audio_encoders: '{audio_encoders}'\n" in text
        assert f"  clip_vision: '{clip_vision}'\n" in text

    def test_humo_audio_encoder_uses_the_pin_native_model_folder(self) -> None:
        audio_encoders = Path("/models/audio_encoders")
        text = benchmark_comfyui.extra_model_paths_config(audio_encoder_directory=audio_encoders)
        assert f"  audio_encoders: '{audio_encoders}'\n" in text

    def test_flux_clip_l_directory_merges_into_text_encoders(self) -> None:
        text_encoders = Path("/models/text_encoders")
        clip_l = Path("/models/clip")
        text = benchmark_comfyui.extra_model_paths_config(
            text_encoder_directory=text_encoders,
            clip_l_directory=clip_l,
        )
        assert "  text_encoders: |\n" in text
        assert f"    {text_encoders}\n" in text
        assert f"    {clip_l}\n" in text

        shared = benchmark_comfyui.extra_model_paths_config(
            text_encoder_directory=text_encoders,
            clip_l_directory=text_encoders,
        )
        assert f"  text_encoders: '{text_encoders}'\n" in shared
        assert "text_encoders: |" not in shared

    def test_minimax_h3_vae_directories_share_the_vae_search_path(self) -> None:
        video_vae = Path("/models/video_vae")
        audio_vae = Path("/models/audio_vae")
        text = benchmark_comfyui.extra_model_paths_config(
            vae_directory=video_vae,
            audio_vae_directory=audio_vae,
        )
        assert "  vae: |\n" in text
        assert f"    {video_vae}\n" in text
        assert f"    {audio_vae}\n" in text


class TestAssembleReport:
    def assemble(self, family: str = "sd15") -> dict[str, Any]:
        smoke = complete_report("rocm")
        identity = {
            "host": smoke["host"],
            "driver": smoke["driver"],
            "torch": smoke["torch"],
            "devices": smoke["devices"],
            "comfyui_version": "0.3.75",
        }
        audio_family = family == "wan21_infinitetalk"
        steps = 6 if audio_family else 20
        warm_runs = 3
        workload = {
            "prompt": (
                "The camera zooms in. Two characters are talking."
                if audio_family
                else "an astronaut"
            ),
            "negative_prompt": "",
            "sampler_id": "euler",
            "scheduler_id": "normal" if audio_family else "simple",
            "seed": 0 if audio_family else 667,
            "steps": steps,
            "width": 832 if audio_family else 512,
            "height": 480 if audio_family else 512,
            "length": 81 if audio_family else (33 if family == "wan21" else None),
            "cfg": 1.0 if audio_family else 7.0,
            "warm_runs": warm_runs,
            "lora_strength_model": 1.0 if family in ("lora", "wan21_infinitetalk") else None,
            "lora_strength_clip": 1.0 if family == "lora" else None,
            "motion_frame_count": 9 if audio_family else None,
            "audio_scale": 1.0 if audio_family else None,
            "speaker_mask_layout": "left_right_half" if audio_family else None,
        }
        artifact_roles = {
            "lora": ("checkpoint", "lora"),
            "zimage": ("diffusion", "text_encoder", "vae"),
            "wan21": ("diffusion", "text_encoder", "vae"),
            "wan21_infinitetalk": (
                "diffusion",
                "text_encoder",
                "vae",
                "lora",
                "model_patch",
                "audio_encoder",
                "clip_vision",
                "input_image",
                "input_audio_1",
                "input_audio_2",
            ),
        }
        input_digests = {
            "input_image": "88a9d7bd3832304a5b66626c442886f0b82ddbce176089e504b8aeaf4cc3333e",
            "input_audio_1": "d008494976e34b05108f181942a6d4363e2bf1176ebabc10ecb69d2e61245afb",
            "input_audio_2": "632aecb453a9a58d37f9f9e70d07f6748ab604af59a564b84eb76031440d3545",
        }
        artifacts = [
            {
                "role": role,
                "path": f"/models/{role}.safetensors",
                "sha256": input_digests.get(role, "c" * 64),
                "bytes": 2_132_696_762,
            }
            for role in artifact_roles.get(family, ("checkpoint",))
        ]
        cold = {
            "load_s": 10.0,
            "encode_s": 0.4,
            "sample_s": 5.0,
            "decode_s": 0.5,
            "total_s": 15.9,
            **({"lora_s": 1.5} if family == "lora" else {}),
            **({"audio_encode_s": 2.4} if audio_family else {}),
        }
        family_checks = ("load", "lora_apply") if family == "lora" else ("load",)
        if audio_family:
            family_checks += ("encode_text", "encode_audio")
        else:
            family_checks += ("encode_text",)
        checks = {
            name: {"ok": True, "detail": "ok"}
            for name in family_checks + ("cold_run", "finite_output", "warm_runs", "unload")
        }
        return benchmark_comfyui.assemble_report(
            backend="rocm",
            family=family,
            identity=identity,
            comfyui_commit="b78cec87" + "0" * 32,
            workload=workload,
            artifacts=artifacts,
            import_s=21.7,
            cold=cold,
            warm_entries=[
                {"sample_s": 4.0, "decode_s": 0.5, "total_s": 4.5} for _ in range(warm_runs)
            ],
            memory={
                "peak_allocated_bytes": 5_335_341_056,
                "peak_reserved_bytes": 5_536_481_280,
                "peak_rss_bytes": 17_179_869_184,
            },
            residual_allocated=0,
            checks=checks,
        )

    def test_assembled_report_passes_validation(self) -> None:
        report = self.assemble()
        assert validate_benchmark_report(report, accelerator="rocm") == ()
        assert report["system"] == "comfyui"
        assert report["mode"] == "eager"
        assert report["placement"] == "comfyui_model_management"
        assert report["family_id"] == "comfyui.sd15"
        assert report["comfyui"] == {
            "version": "0.3.75",
            "commit": "b78cec87" + "0" * 32,
            "clean": True,
        }
        assert report["timings"]["warm"]["median_sample_s"] == 4.0

    def test_assembled_lora_report_passes_validation(self) -> None:
        report = self.assemble("lora")
        assert validate_benchmark_report(report, accelerator="rocm") == ()

    def test_assembled_infinitetalk_report_passes_validation(self) -> None:
        report = self.assemble("wan21_infinitetalk")
        assert validate_benchmark_report(report, accelerator="rocm") == ()
        assert report["timings"]["cold"]["audio_encode_s"] == 2.4
        assert report["workload"]["length"] == 81

    def test_assembled_humo_report_passes_validation(self) -> None:
        expected = complete_benchmark_report("rocm", family="wan21_humo", system="comfyui")
        report = benchmark_comfyui.assemble_report(
            backend="rocm",
            family="wan21_humo",
            identity={
                "host": expected["host"],
                "driver": expected["driver"],
                "torch": expected["torch"],
                "devices": expected["devices"],
                "comfyui_version": "0.3.75",
            },
            comfyui_commit=benchmark_comfyui.COMFYUI_PIN,
            workload=expected["workload"],
            artifacts=expected["artifacts"],
            import_s=expected["timings"]["import_s"],
            cold=expected["timings"]["cold"],
            warm_entries=expected["timings"]["warm"]["runs"],
            memory=expected["memory"],
            residual_allocated=0,
            checks=expected["checks"],
        )
        assert validate_benchmark_report(report, accelerator="rocm") == ()
        assert report["placement"] == "comfyui_model_management"
        assert report["timings"]["cold"]["audio_encode_s"] == 1.7
        assert report["workload"]["length"] == 97

    def test_assembled_anima_report_carries_the_primary_variant(self) -> None:
        expected = complete_benchmark_report("rocm", family="anima", system="comfyui")
        report = benchmark_comfyui.assemble_report(
            backend="rocm",
            family="anima",
            identity={
                "host": expected["host"],
                "driver": expected["driver"],
                "torch": expected["torch"],
                "devices": expected["devices"],
                "comfyui_version": "0.3.75",
            },
            comfyui_commit=benchmark_comfyui.COMFYUI_PIN,
            workload=expected["workload"],
            artifacts=expected["artifacts"],
            import_s=expected["timings"]["import_s"],
            cold=expected["timings"]["cold"],
            warm_entries=expected["timings"]["warm"]["runs"],
            memory=expected["memory"],
            residual_allocated=0,
            checks=expected["checks"],
            variant=benchmark_comfyui.BENCHMARK_PRIMARY_VARIANT,
        )

        assert validate_benchmark_report(report, accelerator="rocm") == ()
        assert report["variant"] == "primary"

    def test_assembled_minimax_h3_report_passes_validation(self) -> None:
        expected = complete_benchmark_report("rocm", family="minimax_h3", system="comfyui")
        report = benchmark_comfyui.assemble_report(
            backend="rocm",
            family="minimax_h3",
            identity={
                "host": expected["host"],
                "driver": expected["driver"],
                "torch": expected["torch"],
                "devices": expected["devices"],
                "comfyui_version": "0.3.75",
            },
            comfyui_commit=benchmark_comfyui.COMFYUI_PIN,
            workload=expected["workload"],
            artifacts=expected["artifacts"],
            import_s=expected["timings"]["import_s"],
            cold=expected["timings"]["cold"],
            warm_entries=expected["timings"]["warm"]["runs"],
            memory=expected["memory"],
            residual_allocated=0,
            checks=expected["checks"],
            execution_path="sampler_custom_advanced",
        )
        assert validate_benchmark_report(report, accelerator="rocm") == ()
        assert report["placement"] == "comfyui_model_management"
        assert report["execution_path"] == "sampler_custom_advanced"
        assert report["workload"]["length"] == 124
        assert "audio_encode_s" not in report["timings"]["cold"]

    @pytest.mark.parametrize("family", ["zimage", "wan21"])
    def test_assembled_split_family_reports_pass_validation(self, family: str) -> None:
        report = self.assemble(family)
        assert validate_benchmark_report(report, accelerator="rocm") == ()
        assert report["family_id"] == f"comfyui.{family}"
        assert report["workload"]["length"] == (33 if family == "wan21" else None)

    def test_all_ok_reflects_check_failures(self) -> None:
        report = self.assemble()
        assert report["all_ok"] is True
        failing = dict(report["checks"])
        failing["unload"] = {"ok": False, "detail": "residual"}
        rebuilt = benchmark_comfyui.assemble_report(
            backend="rocm",
            family="sd15",
            identity={
                "host": report["host"],
                "driver": report["driver"],
                "torch": report["torch"],
                "devices": report["devices"],
                "comfyui_version": "0.3.75",
            },
            comfyui_commit="b78cec87",
            workload=report["workload"],
            artifacts=report["artifacts"],
            import_s=21.7,
            cold=report["timings"]["cold"],
            warm_entries=report["timings"]["warm"]["runs"],
            memory=report["memory"],
            residual_allocated=0,
            checks=failing,
        )
        assert rebuilt["all_ok"] is False


_SHIM_PATH = (
    _MODULE_PATH.parent / "comfyui_benchmark_nodes" / "dinkster_benchmark_shim" / "__init__.py"
)


def _load_shim_with_stubs(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Import the shim against stubs of everything it touches at import.

    The stubs reproduce the ComfyUI surface the shim wraps (PromptQueue,
    PromptExecutor, soft_empty_cache, ProgressRegistry, ProgressBar, the
    server routes), so its tracking wrappers install onto the stub classes
    and its HTTP handlers register into a plain dict.
    """
    events: list[str] = []

    class _PromptQueue:
        def __init__(self) -> None:
            self.mutex = threading.Lock()
            self.flags: dict[str, Any] = {}

        def set_flag(self, name: str, data: Any) -> None:
            with self.mutex:
                self.flags[name] = data

        def get_flags(self, reset: bool = True) -> dict[str, Any]:
            with self.mutex:
                if reset:
                    flags = self.flags
                    self.flags = {}
                    return flags
                return self.flags.copy()

    class _PromptExecutor:
        def reset(self) -> None:
            events.append("executor_reset")

    class _ProgressBar:
        def update_absolute(self, value: Any, total: Any = None, preview: Any = None) -> None:
            return None

    class _ProgressRegistry:
        def start_progress(self, node_id: Any) -> None:
            return None

        def finish_progress(self, node_id: Any) -> None:
            return None

    def _soft_empty_cache(*args: Any, **kwargs: Any) -> None:
        events.append("soft_empty_cache")

    handlers: dict[tuple[str, str], Any] = {}

    def _route(method: str) -> Any:
        def register(path: str) -> Any:
            def decorator(handler: Any) -> Any:
                handlers[(method, path)] = handler
                return handler

            return decorator

        return register

    queue = _PromptQueue()

    def _module(name: str, **attrs: Any) -> Any:
        mod: Any = ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    comfy_mod = _module("comfy")
    model_management = _module(
        "comfy.model_management",
        get_torch_device=lambda: SimpleNamespace(type="cuda"),
        soft_empty_cache=_soft_empty_cache,
        args=SimpleNamespace(
            use_pytorch_cross_attention=False,
            use_ck_attention=False,
            use_sage_attention=False,
        ),
    )
    comfy_mod.model_management = model_management
    comfy_mod.utils = _module("comfy.utils", ProgressBar=_ProgressBar)
    comfy_mod.ops = _module(
        "comfy.ops",
        scaled_dot_product_attention=lambda *args, **kwargs: "sdpa",
    )
    comfy_ldm = _module("comfy.ldm")
    comfy_ldm_modules = _module("comfy.ldm.modules")
    attention_module: Any = _module("comfy.ldm.modules.attention")
    kitchen = _module(
        "comfy_kitchen",
        int8_attention=lambda *args, **kwargs: "int8",
        int8_attention_from_prequantized=lambda *args, **kwargs: "int8_prequantized",
    )

    def attention_pytorch(*args: Any, **kwargs: Any) -> str:
        return comfy_mod.ops.scaled_dot_product_attention(*args, **kwargs)

    def sage_body(*args: Any, fallback: bool = False, **kwargs: Any) -> str:
        if fallback:
            return attention_module.attention_pytorch(*args, **kwargs)
        return attention_module.sageattn(*args, **kwargs)

    def attention_sage(*args: Any, **kwargs: Any) -> str:
        return sage_body(*args, **kwargs)

    attention_sage.__wrapped__ = sage_body  # type: ignore[attr-defined]
    attention_module.comfy = comfy_mod
    attention_module.comfy_kitchen = kitchen
    attention_module.attention_pytorch = attention_pytorch
    attention_module.attention_sage = attention_sage
    attention_module.sageattn = lambda *args, **kwargs: "sage"
    comfy_ldm.modules = comfy_ldm_modules
    comfy_ldm_modules.attention = attention_module
    _module("execution", PromptQueue=_PromptQueue, PromptExecutor=_PromptExecutor)
    _module(
        "torch",
        __version__="test-torch",
        cuda=SimpleNamespace(
            memory_allocated=lambda device: 0,
            empty_cache=lambda: None,
            synchronize=lambda device: None,
        ),
        xpu=SimpleNamespace(),
        _C=SimpleNamespace(),
    )
    _module(
        "aiohttp",
        web=SimpleNamespace(
            json_response=lambda payload, status=200: SimpleNamespace(
                payload=payload, status=status
            )
        ),
    )
    comfy_execution = _module("comfy_execution")
    comfy_execution.progress = _module(
        "comfy_execution.progress", ProgressRegistry=_ProgressRegistry
    )
    comfy_execution.utils = _module("comfy_execution.utils", get_executing_context=lambda: None)
    _module(
        "server",
        PromptServer=SimpleNamespace(
            instance=SimpleNamespace(
                routes=SimpleNamespace(get=_route("GET"), post=_route("POST")),
                prompt_queue=queue,
            )
        ),
    )

    spec = importlib.util.spec_from_file_location("dinkster_benchmark_shim_under_test", _SHIM_PATH)
    assert spec is not None and spec.loader is not None
    shim: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)

    return SimpleNamespace(
        shim=shim,
        handlers=handlers,
        queue=queue,
        executor=_PromptExecutor(),
        model_management=model_management,
        attention=attention_module,
        kitchen=kitchen,
        events=events,
    )


def test_shim_cuda_device_identity_uses_compute_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _load_shim_with_stubs(monkeypatch)
    properties = SimpleNamespace(
        name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        gcnArchName="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        major=12,
        minor=0,
        total_memory=102_642_761_728,
    )
    loaded.shim.torch.cuda.device_count = lambda: 1
    loaded.shim.torch.cuda.get_device_properties = lambda index: properties

    assert loaded.shim._device_entries("cuda") == [
        {
            "index": 0,
            "name": properties.name,
            "architecture": "sm_120",
            "total_memory": properties.total_memory,
        }
    ]


class TestAttentionLaunchEvidence:
    def test_server_launch_maps_provider_to_exact_flag_and_capture_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launched: dict[str, Any] = {}

        class Process:
            pass

        def popen(command: list[str], **kwargs: Any) -> Process:
            launched["command"] = command
            launched["kwargs"] = kwargs
            return Process()

        monkeypatch.setattr(benchmark_comfyui.subprocess, "Popen", popen)
        quality = tmp_path / "quality"
        server = benchmark_comfyui.ComfyServer(
            root=tmp_path,
            python=tmp_path / "python",
            port=8299,
            config_path=tmp_path / "paths.yaml",
            log_path=tmp_path / "server.log",
            boot_nonce="nonce",
            attention_policy="sage",
            quality_output_dir=quality,
            quality_spatial_stride=4,
        )

        server.launch()
        server._log_file.close()

        assert launched["command"][-1] == "--use-sage-attention"
        environment = launched["kwargs"]["env"]
        assert environment["DINKSTER_BENCHMARK_ATTENTION_POLICY"] == "sage"
        assert environment["DINKSTER_BENCHMARK_QUALITY_OUTPUT_DIR"] == str(quality)
        assert environment["DINKSTER_BENCHMARK_QUALITY_SPATIAL_STRIDE"] == "4"

    def test_shim_reports_the_selected_flag_and_managed_sage_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DINKSTER_BENCHMARK_ATTENTION_POLICY", "sage")
        harness = _load_shim_with_stubs(monkeypatch)
        harness.model_management.args.use_sage_attention = True
        monkeypatch.setattr(
            harness.shim,
            "_sage_module_identity",
            lambda: {
                "module": "sageattention",
                "path": "/managed/sageattention/__init__.py",
                "distribution": "dinkster-kitchen",
                "version": "2.2.0.post1",
                "authenticated": True,
            },
        )

        identity = harness.shim._attention_identity()

        assert identity["requested_policy"] == "sage"
        assert identity["selected_policy"] == "sage"
        assert identity["provider_versions"] == [
            ["torch", str(harness.shim.torch.__version__)],
            ["dinkster-kitchen", "2.2.0.post1"],
        ]
        assert identity["provider_module"] == {
            "module": "sageattention",
            "path": "/managed/sageattention/__init__.py",
            "distribution": "dinkster-kitchen",
            "version": "2.2.0.post1",
            "authenticated": True,
        }
        assert identity["fallback"] == "sdpa"
        assert "provider_exception" in identity["fallback_conditions"]
        assert benchmark_comfyui.attention_identity_problem(identity, "sage") is None

        identity["provider_versions"][1][1] = None
        assert (
            benchmark_comfyui.attention_identity_problem(identity, "sage")
            == "a provider version is incomplete"
        )

    def test_sage_module_identity_binds_the_imported_file_to_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = _load_shim_with_stubs(monkeypatch)
        module_path = tmp_path / "sageattention" / "__init__.py"
        module_path.parent.mkdir()
        module_path.write_text("")
        module = ModuleType("sageattention")
        module.__file__ = str(module_path)
        module.__distribution__ = "dinkster-kitchen"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sageattention", module)

        distribution = SimpleNamespace(
            files=(Path("sageattention/__init__.py"),),
            version="2.2.0.post1",
            locate_file=lambda file: tmp_path / file,
        )
        monkeypatch.setattr(
            harness.shim.importlib.metadata,
            "distribution",
            lambda name: distribution,
        )

        assert harness.shim._sage_module_identity() == {
            "module": "sageattention",
            "path": str(module_path),
            "distribution": "dinkster-kitchen",
            "version": "2.2.0.post1",
            "authenticated": True,
        }

    def test_shim_records_sage_provider_success_and_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DINKSTER_BENCHMARK_ATTENTION_POLICY", "sage")
        harness = _load_shim_with_stubs(monkeypatch)

        assert harness.attention.attention_sage() == "sage"
        assert harness.attention.attention_sage(fallback=True) == "sdpa"

        execution = harness.shim._attention_execution()
        assert execution == {
            "policy": "sage",
            "selected_calls": 2,
            "provider_attempts": 1,
            "provider_successes": 1,
            "provider_exceptions": 0,
            "fallback_calls": 1,
        }
        assert benchmark_comfyui.attention_execution_problem(execution, "sage") is None

    def test_execution_proof_refuses_a_provider_that_never_ran(self) -> None:
        execution = {
            "policy": "sage",
            "selected_calls": 3,
            "provider_attempts": 0,
            "provider_successes": 0,
            "provider_exceptions": 0,
            "fallback_calls": 3,
        }

        assert (
            benchmark_comfyui.attention_execution_problem(execution, "sage")
            == "the requested attention provider did not execute successfully"
        )


class TestShimUnloadEndpoint:
    """The unload endpoint waits for the whole free path, not merely flag
    consumption, even when allocation already sits below the drain target
    (the race window a real GPU run cannot reach when the family's
    retained tensors start above the ceiling)."""

    def test_waits_for_free_path_when_allocation_already_below_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = _load_shim_with_stubs(monkeypatch)
        handler = harness.handlers[("POST", "/dinkster_benchmark/unload")]

        async def scenario() -> Any:
            async def prompt_worker() -> None:
                # The worker consumes the flags first (clearing them)...
                while not harness.queue.get_flags(reset=False):
                    await asyncio.sleep(0.01)
                flags = harness.queue.get_flags()
                assert flags.get("free_memory") is True
                # ...then unload_all_models(), whose internal
                # soft_empty_cache() must not count as completion...
                harness.model_management.soft_empty_cache()
                await asyncio.sleep(0.2)
                # ...then the executor reset and the trailing
                # gc + soft_empty_cache that end the free path.
                harness.executor.reset()
                harness.model_management.soft_empty_cache()
                harness.events.append("free_path_finished")

            async def request_json() -> dict[str, int]:
                return {"drain_target_bytes": 1_048_576}

            worker = asyncio.create_task(prompt_worker())
            response = await handler(SimpleNamespace(json=request_json))
            harness.events.append("handler_returned")
            await worker
            return response

        response = asyncio.run(scenario())
        assert response.status == 200
        assert response.payload == {"residual_allocated_bytes": 0}
        assert harness.events.index("handler_returned") > harness.events.index("free_path_finished")
