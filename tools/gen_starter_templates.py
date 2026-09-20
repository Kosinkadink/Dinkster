#!/usr/bin/env python3
"""Generate the native model-family starter workflows and thumbnails."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "# BEGIN GENERATED STARTER TEMPLATES"
END = "# END GENERATED STARTER TEMPLATES"

TARGETS = {
    "generation": (
        ROOT / "packages/dinkster-nodes-generation",
        "dinkster_nodes_generation",
    ),
    "qwen-image": (
        ROOT / "packages/dinkster-model-qwen-image",
        "dinkster_model_qwen_image",
    ),
    "wan": (ROOT / "packages/dinkster-model-wan", "dinkster_model_wan"),
    "triposplat": (
        ROOT / "packages/dinkster-model-triposplat",
        "dinkster_model_triposplat",
    ),
}

OWNER_BY_SLUG = {
    "qwen-image": "qwen-image",
    "wan21": "wan",
    "wan22": "wan",
    "triposplat": "triposplat",
}

FAMILIES = (
    (
        "sd15",
        "dinkster.sd15",
        "Stable Diffusion 1.5",
        ("v1-5-pruned-emaonly-fp16.safetensors",),
        "checkpoint-image",
    ),
    (
        "sdxl",
        "dinkster.sdxl",
        "Stable Diffusion XL",
        ("sd_xl_base_1.0.safetensors",),
        "checkpoint-image",
    ),
    (
        "sdxl-refiner",
        "dinkster.sdxl_refiner",
        "Stable Diffusion XL Refiner",
        ("sd_xl_base_1.0.safetensors", "sd_xl_refiner_1.0.safetensors"),
        "refiner-image",
    ),
    (
        "chroma",
        "dinkster.chroma",
        "Chroma",
        (
            "Chroma1-HD-fp8mixed.safetensors",
            "t5xxl_fp8_e4m3fn_scaled.safetensors",
            "ae.safetensors",
        ),
        "sd3-image",
    ),
    (
        "chroma-radiance",
        "dinkster.chroma_radiance",
        "Chroma Radiance",
        ("chroma-radiance-x0.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors"),
        "pixel-image",
    ),
    (
        "flux-dev",
        "dinkster.flux_dev",
        "Flux Dev",
        ("flux1-dev.safetensors", "t5xxl_fp16.safetensors", "ae.safetensors"),
        "sd3-image",
    ),
    (
        "flux-schnell",
        "dinkster.flux_schnell",
        "Flux Schnell",
        ("flux1-schnell-fp8.safetensors",),
        "checkpoint-image",
    ),
    (
        "flux2-dev",
        "dinkster.flux2_dev",
        "Flux 2 Dev",
        (
            "flux2_dev_fp8mixed.safetensors",
            "mistral_3_small_flux2_bf16.safetensors",
            "full_encoder_small_decoder.safetensors",
        ),
        "sd3-image",
    ),
    (
        "flux2-klein-9b",
        "dinkster.flux2_klein_9b",
        "Flux 2 Klein 9B",
        (
            "flux-2-klein-base-9b-fp8.safetensors",
            "qwen_3_8b_fp8mixed.safetensors",
            "full_encoder_small_decoder.safetensors",
        ),
        "sd3-image",
    ),
    (
        "flux2-klein-4b",
        "dinkster.flux2_klein_4b",
        "Flux 2 Klein 4B",
        (
            "flux-2-klein-base-4b.safetensors",
            "qwen_3_4b.safetensors",
            "flux2-vae.safetensors",
        ),
        "sd3-image",
    ),
    (
        "wan21",
        "dinkster.wan21",
        "Wan 2.1",
        (
            "wan2.1_t2v_14B_fp8_scaled.safetensors",
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "wan_2.1_vae.safetensors",
        ),
        "wan-video",
    ),
    (
        "wan22",
        "dinkster.wan22",
        "Wan 2.2",
        (
            "wan2.2_ti2v_5B_fp16.safetensors",
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "wan_2.1_vae.safetensors",
        ),
        "wan-video",
    ),
    (
        "ltxv",
        "dinkster.ltxv",
        "LTX-Video",
        ("ltx-video-2b-v0.9.safetensors", "t5xxl_fp16.safetensors"),
        "ltx-video",
    ),
    (
        "ltxav",
        "dinkster.ltxav",
        "LTX Audio/Video",
        ("ltx-2-19b-dev-fp8.safetensors", "gemma_3_12B_it_fp4_mixed.safetensors"),
        "ltxav-video",
    ),
    (
        "qwen-image",
        "dinkster.qwen_image",
        "Qwen Image",
        (
            "qwen_image_fp8_e4m3fn.safetensors",
            "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "qwen_image_vae.safetensors",
        ),
        "sd3-image",
    ),
    (
        "z-image",
        "dinkster.z_image",
        "Z-Image",
        ("z_image_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"),
        "sd3-image",
    ),
    (
        "z-image-pixel",
        "dinkster.z_image_pixel_space",
        "Z-Image Pixel Space",
        ("zeta-chroma.safetensors", "qwen_3_4b.safetensors"),
        "pixel-image",
    ),
    (
        "minimax-h3",
        "dinkster.minimax_h3",
        "MiniMax H3",
        (
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors",
        ),
        "h3-video",
    ),
    (
        "minimax-music3",
        "dinkster.minimax_music3",
        "MiniMax Music 3",
        (
            "minimax_music3_dit_fp16.safetensors",
            "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            "minimax_music3_dav.safetensors",
        ),
        "music-audio",
    ),
    (
        "krea2",
        "dinkster.krea2",
        "Krea 2",
        (
            "krea2_turbo_fp8_scaled.safetensors",
            "qwen3vl_4b_fp8_scaled.safetensors",
            "qwen_image_vae.safetensors",
        ),
        "image",
    ),
    (
        "ideogram4",
        "dinkster.ideogram4",
        "Ideogram 4",
        (
            "ideogram4_fp8_scaled.safetensors",
            "ideogram4_unconditional_fp8_scaled.safetensors",
            "qwen3vl_8b_fp8_scaled.safetensors",
            "flux2-vae.safetensors",
        ),
        "sd3-image",
    ),
    (
        "seedvr2",
        "dinkster.seedvr2",
        "SeedVR2",
        ("seedvr2_3b_int8_convrot.safetensors", "seedvr2_ema_vae_fp16.safetensors"),
        "seedvr-image",
    ),
    (
        "anima",
        "dinkster.anima",
        "Anima",
        (
            "anima-base-v1.0.safetensors",
            "qwen_3_06b_base.safetensors",
            "qwen_image_vae.safetensors",
        ),
        "image",
    ),
    (
        "lumina2",
        "dinkster.lumina2",
        "Lumina Image 2.0",
        ("NetaYumev35_pretrained_all_in_one.safetensors",),
        "checkpoint-image",
    ),
    (
        "triposplat",
        "dinkster.triposplat",
        "TripoSplat",
        (
            "triposplat_fp16.safetensors",
            "dino_v3_vit_h.safetensors",
            "flux2-vae.safetensors",
            "triposplat_vae_decoder_fp16.safetensors",
        ),
        "triposplat-3d",
    ),
    (
        "trellis2",
        "dinkster.trellis2",
        "TRELLIS.2 / Pixal3D",
        (
            "trellis_2_int8_convrot.safetensors",
            "dino_v3_L_naf_fp32.safetensors",
            "trellis_2_shape_vae_bf16.safetensors",
            "trellis_2_texture_vae_bf16.safetensors",
        ),
        "trellis-3d",
    ),
)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def thumbnail(index: int) -> bytes:
    base = (
        (37 * index + 42) % 180 + 32,
        (67 * index + 68) % 160 + 48,
        (97 * index + 96) % 140 + 64,
    )
    rows = []
    for y in range(64):
        row = bytearray((0,))
        for x in range(64):
            lift = (x + y) // 8
            stripe = 24 if (x - y) % 31 < 6 else 0
            row.extend(min(255, channel + lift // 3 + stripe) for channel in base)
        rows.append(bytes(row))
    raw = b"".join(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, object]] = {}
        self.links: dict[str, dict[str, object]] = {}
        self.positions: dict[str, dict[str, object]] = {}

    def node(
        self,
        node_id: str,
        node_type: str,
        values: dict[str, object] | None = None,
        *,
        column: int,
        row: int = 0,
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, "values": values or {}}
        self.positions[node_id] = {"position": {"x": 40 + column * 340, "y": 80 + row * 300}}

    def link(self, source: str, output: str, target: str, input_id: str) -> None:
        link_id = f"l{len(self.links) + 1}"
        self.links[link_id] = {
            "id": link_id,
            "from": {"node": source, "port": output},
            "to": {"node": target, "port": input_id},
        }


def _sampler_values(*, cfg: float = 7.0, steps: int = 20) -> dict[str, object]:
    return {
        "seed": 91,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "dinkster.euler",
        "scheduler": "dinkster.simple",
        "denoise": 1.0,
        "conditioning_batching": "auto",
        "max_fused_lanes": 2,
    }


def _text_conditioning(graph: Graph, clip_node: str, *, column: int) -> None:
    graph.node(
        "positive",
        "dinkster.clip_text_encode",
        {"text": "a cinematic landscape, detailed lighting"},
        column=column,
    )
    graph.node(
        "negative",
        "dinkster.clip_text_encode",
        {"text": "blurry, low quality"},
        column=column,
        row=1,
    )
    graph.link(clip_node, "clip", "positive", "clip")
    graph.link(clip_node, "clip", "negative", "clip")


def _image_output(graph: Graph, slug: str, vae_node: str, *, column: int) -> None:
    graph.node("decode", "dinkster.vae_decode", column=column)
    graph.node(
        "save",
        "dinkster.save_image",
        {"target": {"mount": "output", "prefix": slug}},
        column=column + 1,
    )
    graph.link("sampler", "latent", "decode", "samples")
    graph.link(vae_node, "vae", "decode", "vae")
    graph.link("decode", "image", "save", "images")


def _checkpoint_image(graph: Graph, slug: str, models: tuple[str, ...]) -> None:
    graph.node(
        "checkpoint",
        "dinkster.load_checkpoint",
        {"checkpoint": models[0]},
        column=0,
        row=1,
    )
    _text_conditioning(graph, "checkpoint", column=1)
    graph.node(
        "latent",
        "dinkster.empty_latent_image",
        {"width": 512, "height": 512, "batch_size": 1},
        column=1,
        row=2,
    )
    graph.node("sampler", "dinkster.ksampler", _sampler_values(cfg=8.0), column=2, row=1)
    graph.link("checkpoint", "model", "sampler", "model")
    graph.link("positive", "conditioning", "sampler", "positive")
    graph.link("negative", "conditioning", "sampler", "negative")
    graph.link("latent", "latent", "sampler", "latent_image")
    _image_output(graph, slug, "checkpoint", column=3)


def _refiner_image(graph: Graph, slug: str, models: tuple[str, ...]) -> None:
    graph.node("base", "dinkster.load_checkpoint", {"checkpoint": models[0]}, column=0)
    graph.node("refiner", "dinkster.load_checkpoint", {"checkpoint": models[1]}, column=0, row=2)
    _text_conditioning(graph, "base", column=1)
    graph.node(
        "refiner_positive",
        "dinkster.clip_text_encode",
        {"text": "a cinematic landscape, detailed lighting"},
        column=1,
        row=2,
    )
    graph.node(
        "refiner_negative",
        "dinkster.clip_text_encode",
        {"text": "blurry, low quality"},
        column=1,
        row=3,
    )
    graph.node(
        "latent",
        "dinkster.empty_latent_image",
        {"width": 1024, "height": 1024, "batch_size": 1},
        column=1,
        row=4,
    )
    advanced = {
        "add_noise": "enable",
        "noise_seed": 91,
        "steps": 25,
        "cfg": 8.0,
        "sampler_name": "dinkster.euler",
        "scheduler": "dinkster.simple",
        "start_at_step": 0,
        "end_at_step": 20,
        "return_with_leftover_noise": "enable",
        "conditioning_batching": "auto",
        "max_fused_lanes": 2,
    }
    graph.node("base_sampler", "dinkster.ksampler_advanced", advanced, column=2, row=1)
    graph.node(
        "sampler",
        "dinkster.ksampler_advanced",
        {**advanced, "add_noise": "disable", "start_at_step": 20, "end_at_step": 25},
        column=3,
        row=1,
    )
    for source, target in (("base", "base_sampler"), ("refiner", "sampler")):
        graph.link(source, "model", target, "model")
    graph.link("positive", "conditioning", "base_sampler", "positive")
    graph.link("negative", "conditioning", "base_sampler", "negative")
    graph.link("latent", "latent", "base_sampler", "latent_image")
    graph.link("refiner", "clip", "refiner_positive", "clip")
    graph.link("refiner", "clip", "refiner_negative", "clip")
    graph.link("refiner_positive", "conditioning", "sampler", "positive")
    graph.link("refiner_negative", "conditioning", "sampler", "negative")
    graph.link("base_sampler", "latent", "sampler", "latent_image")
    _image_output(graph, slug, "refiner", column=4)


def _split_image(
    graph: Graph,
    slug: str,
    family: str,
    models: tuple[str, ...],
    *,
    pixel_space: bool,
    sd3_latent: bool,
) -> None:
    graph.node(
        "model",
        "dinkster.load_diffusion_model",
        {"diffusion_model": models[0]},
        column=0,
    )
    clip_types = {
        "dinkster.chroma": "chroma",
        "dinkster.chroma_radiance": "chroma",
        "dinkster.flux_dev": "flux",
        "dinkster.flux2_dev": "flux2",
        "dinkster.flux2_klein_9b": "flux2",
        "dinkster.flux2_klein_4b": "flux2",
        "dinkster.qwen_image": "qwen_image",
        "dinkster.z_image": "lumina2",
        "dinkster.z_image_pixel_space": "lumina2",
        "dinkster.krea2": "krea2",
        "dinkster.ideogram4": "ideogram4",
        "dinkster.anima": "stable_diffusion",
    }
    graph.node(
        "clip",
        "dinkster.load_clip",
        {"text_encoder": models[1], "type": clip_types[family]},
        column=0,
        row=1,
    )
    vae_values: dict[str, object] = {"pixel_space": True} if pixel_space else {"vae": models[-1]}
    graph.node("vae", "dinkster.load_vae", vae_values, column=0, row=2)
    if family == "dinkster.ideogram4":
        graph.node(
            "unconditional_model",
            "dinkster.load_diffusion_model",
            {"diffusion_model": models[1]},
            column=0,
            row=3,
        )
        graph.nodes["clip"]["values"] = {"text_encoder": models[2], "type": "ideogram4"}
    _text_conditioning(graph, "clip", column=1)
    latent_type = "dinkster.empty_sd3_latent_image" if sd3_latent else "dinkster.empty_latent_image"
    graph.node(
        "latent",
        latent_type,
        {"width": 1024, "height": 1024, "batch_size": 1},
        column=1,
        row=2,
    )
    graph.node("sampler", "dinkster.ksampler", _sampler_values(), column=2, row=1)
    graph.link("model", "model", "sampler", "model")
    graph.link("positive", "conditioning", "sampler", "positive")
    graph.link("negative", "conditioning", "sampler", "negative")
    graph.link("latent", "latent", "sampler", "latent_image")
    _image_output(graph, slug, "vae", column=3)


def _video_output(
    graph: Graph,
    slug: str,
    vae_node: str,
    latent_node: str = "sampler",
    latent_port: str = "latent",
    *,
    column: int,
    audio_node: str | None = None,
) -> None:
    graph.node("decode", "dinkster.vae_decode", column=column)
    graph.node("assemble", "dinkster.video.assemble", {"fps": 24.0}, column=column + 1)
    graph.node(
        "save",
        "dinkster.save_video",
        {"target": {"mount": "output", "prefix": slug}},
        column=column + 2,
    )
    graph.link(latent_node, latent_port, "decode", "samples")
    graph.link(vae_node, "vae", "decode", "vae")
    graph.link("decode", "image", "assemble", "images")
    if audio_node is not None:
        graph.link(audio_node, "audio", "assemble", "audio")
    graph.link("assemble", "video", "save", "video")


def _wan_video(graph: Graph, slug: str, models: tuple[str, ...]) -> None:
    graph.node("model", "dinkster.load_diffusion_model", {"diffusion_model": models[0]}, column=0)
    graph.node(
        "clip",
        "dinkster.load_clip",
        {"text_encoder": models[1], "type": "wan"},
        column=0,
        row=1,
    )
    graph.node("vae", "dinkster.load_vae", {"vae": models[2]}, column=0, row=2)
    _text_conditioning(graph, "clip", column=1)
    graph.node(
        "latent",
        "dinkster.empty_hunyuan_latent_video",
        {"width": 848, "height": 480, "length": 25, "batch_size": 1},
        column=1,
        row=2,
    )
    graph.node("sampler", "dinkster.ksampler", _sampler_values(cfg=5.0), column=2, row=1)
    graph.link("model", "model", "sampler", "model")
    graph.link("positive", "conditioning", "sampler", "positive")
    graph.link("negative", "conditioning", "sampler", "negative")
    graph.link("latent", "latent", "sampler", "latent_image")
    _video_output(graph, slug, "vae", column=3)


def _ltx_video(graph: Graph, slug: str, models: tuple[str, ...], *, av: bool) -> None:
    graph.node("checkpoint", "dinkster.load_checkpoint", {"checkpoint": models[0]}, column=0)
    if av:
        graph.node(
            "clip",
            "dinkster.load_ltxav_text_encoder",
            {"text_encoder": models[1], "ckpt_name": models[0]},
            column=0,
            row=1,
        )
        graph.node(
            "audio_vae",
            "dinkster.load_ltxav_audio_vae",
            {"ckpt_name": models[0]},
            column=0,
            row=2,
        )
    else:
        graph.node(
            "clip",
            "dinkster.load_clip",
            {"text_encoder": models[1], "type": "ltxv"},
            column=0,
            row=1,
        )
    _text_conditioning(graph, "clip", column=1)
    conditioning = "dinkster.ltxav_conditioning" if av else "dinkster.ltxv_conditioning"
    graph.node("condition", conditioning, {"frame_rate": 24.0}, column=2)
    graph.link("positive", "conditioning", "condition", "positive")
    graph.link("negative", "conditioning", "condition", "negative")
    latent_type = "dinkster.empty_ltxav_latent" if av else "dinkster.empty_ltxv_latent"
    latent_values = {"width": 768, "height": 512, "length": 97, "batch_size": 1}
    if av:
        latent_values["frame_rate"] = 24
    graph.node("latent", latent_type, latent_values, column=2, row=2)
    graph.link("checkpoint", "model", "latent", "model")
    graph.node("sampler", "dinkster.ksampler", _sampler_values(cfg=4.0), column=3, row=1)
    graph.link("checkpoint", "model", "sampler", "model")
    graph.link("condition", "positive", "sampler", "positive")
    graph.link("condition", "negative", "sampler", "negative")
    graph.link("latent", "latent", "sampler", "latent_image")
    if not av:
        _video_output(graph, slug, "checkpoint", column=4)
        return
    graph.node("separate", "dinkster.separate_av_latent", column=4, row=1)
    graph.node("audio_decode", "dinkster.ltxav_audio_vae_decode", column=5, row=2)
    graph.link("sampler", "latent", "separate", "latent")
    graph.link("separate", "audio_latent", "audio_decode", "samples")
    graph.link("audio_vae", "audio_vae", "audio_decode", "audio_vae")
    _video_output(
        graph,
        slug,
        "checkpoint",
        "separate",
        "video_latent",
        column=5,
        audio_node="audio_decode",
    )


def _h3_video(graph: Graph, slug: str, models: tuple[str, ...]) -> None:
    graph.node("model", "dinkster.load_diffusion_model", {"diffusion_model": models[0]}, column=0)
    graph.node(
        "clip",
        "dinkster.load_clip",
        {"text_encoder": models[1], "type": "minimax"},
        column=0,
        row=1,
    )
    graph.node("video_vae", "dinkster.load_vae", {"vae": models[2]}, column=0, row=2)
    graph.node("audio_vae", "dinkster.load_vae", {"vae": models[3]}, column=0, row=3)
    _text_conditioning(graph, "clip", column=1)
    graph.node(
        "latent",
        "dinkster.empty_minimax_h3_av",
        {"width": 864, "height": 480, "frame_count": 124},
        column=1,
        row=2,
    )
    graph.node("sampler", "dinkster.ksampler", _sampler_values(cfg=5.0, steps=30), column=2, row=1)
    graph.link("model", "model", "sampler", "model")
    graph.link("positive", "conditioning", "sampler", "positive")
    graph.link("negative", "conditioning", "sampler", "negative")
    graph.link("latent", "latent", "sampler", "latent_image")
    graph.node("separate", "dinkster.separate_av_latent", column=3, row=1)
    graph.node("audio_decode", "dinkster.vae_decode_audio", column=4, row=2)
    graph.link("sampler", "latent", "separate", "latent")
    graph.link("separate", "audio_latent", "audio_decode", "samples")
    graph.link("audio_vae", "vae", "audio_decode", "vae")
    _video_output(
        graph,
        slug,
        "video_vae",
        "separate",
        "video_latent",
        column=4,
        audio_node="audio_decode",
    )


def _music_audio(graph: Graph, slug: str, models: tuple[str, ...]) -> None:
    graph.node("model", "dinkster.load_diffusion_model", {"diffusion_model": models[0]}, column=0)
    graph.node(
        "clip",
        "dinkster.load_clip",
        {"text_encoder": models[1], "type": "minimax"},
        column=0,
        row=1,
    )
    graph.node("vae", "dinkster.load_vae", {"vae": models[2]}, column=0, row=2)
    graph.node(
        "condition",
        "dinkster.minimax_music3_text_encode",
        {"caption": "warm cinematic instrumental", "lyrics": ""},
        column=1,
    )
    graph.node(
        "latent",
        "dinkster.empty_minimax_music3_latent_audio",
        {"seconds": 30.0, "batch_size": 1},
        column=1,
        row=1,
    )
    graph.link("clip", "clip", "condition", "clip")
    graph.node("sampler", "dinkster.ksampler", _sampler_values(cfg=1.5), column=2)
    graph.link("model", "model", "sampler", "model")
    graph.link("condition", "conditioning", "sampler", "positive")
    graph.link("condition", "conditioning", "sampler", "negative")
    graph.link("latent", "latent", "sampler", "latent_image")
    graph.node("decode", "dinkster.vae_decode_audio", column=3)
    graph.node(
        "save",
        "dinkster.save_audio",
        {"target": {"mount": "output", "prefix": slug}},
        column=4,
    )
    graph.link("sampler", "latent", "decode", "samples")
    graph.link("vae", "vae", "decode", "vae")
    graph.link("decode", "audio", "save", "audio")


def _seedvr_image(graph: Graph, slug: str, models: tuple[str, ...]) -> None:
    graph.node("source", "dinkster.load_image", {"image": "input.png"}, column=0)
    graph.node(
        "model", "dinkster.load_diffusion_model", {"diffusion_model": models[0]}, column=0, row=1
    )
    graph.node("vae", "dinkster.load_vae", {"vae": models[1]}, column=0, row=2)
    graph.node("encode", "dinkster.vae_encode_tiled", column=1)
    graph.link("source", "image", "encode", "pixels")
    graph.link("vae", "vae", "encode", "vae")
    graph.node("condition", "dinkster.seedvr2_conditioning", column=2)
    graph.link("model", "model", "condition", "model")
    graph.link("encode", "latent", "condition", "vae_conditioning")
    graph.node("sampler", "dinkster.ksampler", _sampler_values(cfg=1.0, steps=1), column=3)
    graph.link("model", "model", "sampler", "model")
    graph.link("condition", "positive", "sampler", "positive")
    graph.link("condition", "negative", "sampler", "negative")
    graph.link("encode", "latent", "sampler", "latent_image")
    _image_output(graph, slug, "vae", column=4)


def _triposplat_3d(graph: Graph, slug: str, models: tuple[str, ...]) -> None:
    graph.node("source", "dinkster.load_image", {"image": "input.png"}, column=0)
    graph.node(
        "model", "dinkster.load_diffusion_model", {"diffusion_model": models[0]}, column=0, row=1
    )
    graph.node(
        "vision",
        "dinkster.load_triposplat_vision_encoder",
        {"vision_encoder": models[1]},
        column=0,
        row=2,
    )
    graph.node("vae", "dinkster.load_vae", {"vae": models[2]}, column=0, row=3)
    graph.node(
        "decoder",
        "dinkster.load_triposplat_decoder",
        {"decoder": models[3]},
        column=0,
        row=4,
    )
    graph.node("preprocess", "dinkster.triposplat_preprocess_image", column=1)
    graph.link("source", "image", "preprocess", "image")
    graph.link("source", "mask", "preprocess", "mask")
    graph.node("condition", "dinkster.triposplat_conditioning", column=2)
    graph.link("vision", "vision", "condition", "vision")
    graph.link("vae", "vae", "condition", "vae")
    graph.link("preprocess", "image", "condition", "image")
    graph.node("sampler", "dinkster.ksampler", _sampler_values(cfg=7.0, steps=20), column=3)
    graph.link("model", "model", "sampler", "model")
    graph.link("condition", "positive", "sampler", "positive")
    graph.link("condition", "negative", "sampler", "negative")
    graph.link("condition", "latent", "sampler", "latent_image")
    graph.node("decode", "dinkster.triposplat_decode", column=4)
    graph.node(
        "save",
        "dinkster.save_gaussian_splat",
        {"target": {"mount": "output", "prefix": slug}},
        column=5,
    )
    graph.link("sampler", "latent", "decode", "samples")
    graph.link("decoder", "decoder", "decode", "decoder")
    graph.link("decode", "splat", "save", "splat")


def _trellis_3d(graph: Graph, slug: str, models: tuple[str, ...]) -> None:
    graph.node("source", "dinkster.load_image", {"image": "input.png"}, column=0)
    graph.node(
        "model", "dinkster.load_diffusion_model", {"diffusion_model": models[0]}, column=0, row=1
    )
    graph.node("vision", "dinkster.load_vision", {"vision_encoder": models[1]}, column=0, row=2)
    graph.node("shape_vae", "dinkster.load_vae", {"vae": models[2]}, column=0, row=3)
    graph.node("texture_vae", "dinkster.load_vae", {"vae": models[3]}, column=0, row=4)
    graph.node("condition", "dinkster.trellis2_conditioning", column=1)
    graph.link("vision", "vision", "condition", "clip_vision_model")
    graph.link("source", "image", "condition", "image")
    graph.node("structure_latent", "dinkster.empty_trellis2_latent_structure", column=1, row=1)
    graph.node(
        "structure_sampler", "dinkster.ksampler", _sampler_values(cfg=7.5, steps=12), column=2
    )
    graph.link("model", "model", "structure_sampler", "model")
    graph.link("condition", "positive", "structure_sampler", "positive")
    graph.link("condition", "negative", "structure_sampler", "negative")
    graph.link("structure_latent", "latent", "structure_sampler", "latent_image")
    graph.node("structure_decode", "dinkster.vae_decode_structure_trellis2", column=3)
    graph.link("structure_sampler", "latent", "structure_decode", "samples")
    graph.link("shape_vae", "vae", "structure_decode", "vae")
    graph.node("shape_stage", "dinkster.trellis2_shape_stage", column=4)
    graph.link("condition", "positive", "shape_stage", "positive")
    graph.link("condition", "negative", "shape_stage", "negative")
    graph.link("structure_decode", "voxel", "shape_stage", "voxel")
    graph.node("shape_sampler", "dinkster.ksampler", _sampler_values(cfg=7.5, steps=12), column=5)
    graph.link("model", "model", "shape_sampler", "model")
    graph.link("shape_stage", "positive", "shape_sampler", "positive")
    graph.link("shape_stage", "negative", "shape_sampler", "negative")
    graph.link("shape_stage", "latent", "shape_sampler", "latent_image")
    graph.node("shape_decode", "dinkster.vae_decode_shape_trellis", column=6)
    graph.link("shape_sampler", "latent", "shape_decode", "samples")
    graph.link("shape_vae", "vae", "shape_decode", "vae")
    graph.node("texture_stage", "dinkster.trellis2_texture_stage", column=5, row=2)
    graph.link("shape_stage", "positive", "texture_stage", "positive")
    graph.link("shape_stage", "negative", "texture_stage", "negative")
    graph.link("shape_sampler", "latent", "texture_stage", "shape_latent")
    graph.node("sampler", "dinkster.ksampler", _sampler_values(cfg=7.5, steps=12), column=6, row=2)
    graph.link("model", "model", "sampler", "model")
    graph.link("texture_stage", "positive", "sampler", "positive")
    graph.link("texture_stage", "negative", "sampler", "negative")
    graph.link("texture_stage", "latent", "sampler", "latent_image")
    graph.node("texture_decode", "dinkster.vae_decode_texture_trellis", column=7, row=2)
    graph.link("sampler", "latent", "texture_decode", "samples")
    graph.link("texture_vae", "vae", "texture_decode", "vae")
    graph.link("shape_decode", "shape_subdivides", "texture_decode", "shape_subdivides")
    graph.node("paint", "dinkster.paint_mesh", column=8)
    graph.link("shape_decode", "mesh", "paint", "mesh")
    graph.link("texture_decode", "voxel_colors", "paint", "voxel_colors")
    graph.node("model3d", "dinkster.mesh_to_model3d", column=9)
    graph.node(
        "save",
        "dinkster.save_model3d",
        {"target": {"mount": "output", "prefix": slug}},
        column=10,
    )
    graph.link("paint", "mesh", "model3d", "mesh")
    graph.link("model3d", "model", "save", "model")


def workflow(
    slug: str,
    family: str,
    name: str,
    models: tuple[str, ...],
    graph_kind: str,
) -> dict[str, object]:
    graph = Graph()
    if graph_kind == "checkpoint-image":
        _checkpoint_image(graph, slug, models)
    elif graph_kind == "refiner-image":
        _refiner_image(graph, slug, models)
    elif graph_kind in {"image", "sd3-image", "pixel-image"}:
        _split_image(
            graph,
            slug,
            family,
            models,
            pixel_space=graph_kind == "pixel-image",
            sd3_latent=graph_kind != "image",
        )
    elif graph_kind == "wan-video":
        _wan_video(graph, slug, models)
    elif graph_kind in {"ltx-video", "ltxav-video"}:
        _ltx_video(graph, slug, models, av=graph_kind == "ltxav-video")
    elif graph_kind == "h3-video":
        _h3_video(graph, slug, models)
    elif graph_kind == "music-audio":
        _music_audio(graph, slug, models)
    elif graph_kind == "seedvr-image":
        _seedvr_image(graph, slug, models)
    elif graph_kind == "triposplat-3d":
        _triposplat_3d(graph, slug, models)
    elif graph_kind == "trellis-3d":
        _trellis_3d(graph, slug, models)
    else:
        raise ValueError(f"unsupported starter graph kind {graph_kind!r}")
    return {
        "format": "dinkster-workflow",
        "formatVersion": 1,
        "lineage": f"starter-{slug}",
        "root": "g0",
        "graphs": {
            "g0": {
                "id": "g0",
                "name": f"{name} Starter",
                "nodes": graph.nodes,
                "links": graph.links,
                "nets": {},
                "reroutes": {},
                "nextOrdinal": len(graph.nodes) + len(graph.links) + 1,
            }
        },
        "view": {"graphs": {"g0": {"nodes": graph.positions}}},
        "meta": {"title": f"{name} Starter", "family": family},
    }


def manifest_block(families: list[tuple[str, str, str, tuple[str, ...], str]], module: str) -> str:
    tables = []
    for slug, family, name, models, _graph_kind in families:
        lines = [
            "[[pack.templates]]",
            f'id = "{slug}"',
            f'name = "{name}"',
            f'description = "Starter native workflow for {name}."',
            f'family = "{family}"',
            f'tags = ["starter", "family", "{family}"]',
            "models = [" + ", ".join(json.dumps(model) for model in models) + "]",
            f'file = "src/{module}/templates/{slug}.json"',
            f'thumbnail = "src/{module}/templates/{slug}.png"',
        ]
        tables.append("\n".join(lines))
    return START + "\n\n" + "\n\n".join(tables) + "\n\n" + END


def main() -> None:
    grouped: dict[str, list[tuple[str, str, str, tuple[str, ...], str]]] = {
        owner: [] for owner in TARGETS
    }
    for index, (slug, family, name, models, graph_kind) in enumerate(FAMILIES):
        owner = OWNER_BY_SLUG.get(slug, "generation")
        grouped[owner].append((slug, family, name, models, graph_kind))
        package, module = TARGETS[owner]
        output = package / f"src/{module}/templates"
        output.mkdir(parents=True, exist_ok=True)
        document = workflow(slug, family, name, models, graph_kind)
        (output / f"{slug}.json").write_text(json.dumps(document, indent=2) + "\n")
        (output / f"{slug}.png").write_bytes(thumbnail(index))
    for owner, (package, module) in TARGETS.items():
        manifest = package / "dinkster-pack.toml"
        text = manifest.read_text()
        before, marker, remainder = text.partition(START)
        if not marker:
            raise RuntimeError(f"missing {START!r} in {manifest}")
        _, marker, after = remainder.partition(END)
        if not marker:
            raise RuntimeError(f"missing {END!r} in {manifest}")
        manifest.write_text(before + manifest_block(grouped[owner], module) + after)

    generation_package, generation_module = TARGETS["generation"]
    generation_output = generation_package / f"src/{generation_module}/templates"
    for slug in OWNER_BY_SLUG:
        (generation_output / f"{slug}.json").unlink(missing_ok=True)
        (generation_output / f"{slug}.png").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
