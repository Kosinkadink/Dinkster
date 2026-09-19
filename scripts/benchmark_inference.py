"""Measure one Dinkster inference benchmark cell on real hardware.

For exported workflows, pass --workflow (see tools/workflow_benchmark.py).
That HTTP contract derives family observations from plans and records a caller-
selected reference revision. The per-family cell format below is historical.

One invocation runs one cell - a model family workload on one backend -
and writes the JSON evidence report that
dinkster_workers.backend_env.validate_benchmark_report checks:

    .venv-rocm/bin/python scripts/benchmark_inference.py \\
        --backend rocm --family sd15 --checkpoint /models/sd15.safetensors \\
        --json sd15-bench.json

The report is one side of the Dinkster-vs-ComfyUI head-to-head comparison
(issue #667); a ComfyUI-side runner emits the same schema with
system="comfyui". Families: sd15 and sdxl (safetensors checkpoints),
lora (a checkpoint plus a LoRA applied before measurement), and the
split-source families zimage (Z-Image Turbo: --diffusion,
--text-encoder, --vae) and wan21 (Wan 2.1 text-to-video: the same three
sources plus --length frames, which must be 4k+1 so the requested count
equals what the causal video VAE decodes). Split-source workload
defaults match the official ComfyUI workflow templates; per its
template, zimage's negative conditioning is a zeroed copy of the
positive (ConditioningZeroOut), not a second encode. Wan 2.1 HuMo adds
the pinned LightX2V LoRA, Whisper audio encoder, reference image, and
input audio and drives the production provider conditioning path.
MiniMax H3 adds separate diffusion, conditioner, video VAE, and audio
VAE components for the official non-turbo text-to-audio-video graph.
Backends: rocm and xpu are the comparison lanes; cuda exists for
development hosts.
Eager execution is the head-to-head contract; --mode compile is a
Dinkster-only informational column that compiles the diffusion module
after load, so the cold sample phase includes compilation and the
warm runs show compiled steady state (checkpoint families only).

By default, assembled modules enroll in Dinkster's production residency
manager. Text encoding, sampling, and VAE decoding each run under a
role lease, so the manager keeps fitting models resident and partially
offloads models that exceed the device's real byte budget. The
--placement direct-diagnostic option retains whole-module placement as
an explicitly labeled isolation diagnostic rather than a headline
production measurement.

Timings are synchronized wall clock at phase boundaries:

- cold: the first image in this process - load (checkpoint read and
  placement setup), lora (patch decode and selected placement
  application, lora family only), encode (cond and uncond text
  encoding), sample, decode, and their total. Python package import
  time is recorded separately as import_s; ComfyUI pays its import
  cost at server boot, outside its measured node phases.
- warm: --warm-runs repetitions of sample+decode reusing the encoded
  conditioning, matching ComfyUI's caching of unchanged text-encode
  nodes. Warm run i uses seed+1+i so both systems generate fresh noise
  each run without output caching. Medians summarize the runs.

Per-step boundaries from the sampling callback are recorded as
step_wall_ms; they are dispatch-side (no per-step synchronize, which
would perturb the totals) and therefore informational. Memory telemetry:
allocator peak and reserved bytes across all runs, allocator residue
after unload, and peak resident process RAM. Exits nonzero when any check
fails or the report is incomplete.

Every report also records a residency section for offload-mechanism
comparison: the requested DINKSTER_AIMDO_ARM selector (off, auto, or on),
the actual routes selected for generic and H3 production handles, the VRAM
regime (open, or --regime constrained behind a ballast allocation that
pins device free memory down to --leave-free-mib), and best-effort GPU
shared (sysmem) usage samples - before load, after the cold run, and after
the warm runs - read through the counter sampler shared with
benchmark_residency.py. Post-warmup growth (after minus warm) is an
unattributed observation, not spill proof. shared_spill_detected is null
(unassessed), even for stable usage. Missing or invalid samples are null,
not zero. Passing this report's checks does not establish spill avoidance.
Ballast is held by the allocator across the measured phases, so
constrained-regime allocator peaks include it.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast


def _bootstrap_production_residency() -> tuple[str, bool | None]:
    """Apply the serving default and initialize Aimdo before torch imports."""
    mechanism = os.environ.get("DINKSTER_AIMDO_ARM", "auto")
    if mechanism not in ("off", "auto", "on"):
        sys.exit("error: DINKSTER_AIMDO_ARM must be one of ('off', 'auto', 'on')")
    os.environ["DINKSTER_AIMDO_ARM"] = mechanism
    if mechanism not in ("auto", "on"):
        return mechanism, None
    from dinkster_memory import AcceleratorMemoryPolicy
    from dinkster_workers.aimdo_bootstrap import bootstrap_aimdo

    succeeded, _effective_headroom = bootstrap_aimdo(
        True,
        simple_vram_headroom=AcceleratorMemoryPolicy().minimum_free_bytes,
    )
    return mechanism, succeeded


_BOOTSTRAP_MECHANISM: str | None = None
_AIMDO_BOOTSTRAP_SUCCEEDED: bool | None = None
if __name__ == "__main__":
    if any(arg == "--workflow" or arg.startswith("--workflow=") for arg in sys.argv[1:]):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools.workflow_benchmark import main as workflow_main

        sys.exit(workflow_main("dinkster"))
    _BOOTSTRAP_MECHANISM, _AIMDO_BOOTSTRAP_SUCCEEDED = _bootstrap_production_residency()

import torch  # noqa: E402 - Aimdo must initialize before torch imports
from dinkster_workers.backend_env import (  # noqa: E402 - follows pre-torch bootstrap
    BENCHMARK_ACCELERATORS,
    BENCHMARK_ANIMA_FALLBACK_VARIANT,
    BENCHMARK_ANIMA_PROMPT,
    BENCHMARK_DINKSTER_DIAGNOSTIC_PLACEMENT,
    BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT,
    BENCHMARK_FAMILIES,
    BENCHMARK_MINIMAX_H3_DINKSTER_EXECUTION_PATH,
    BENCHMARK_PRIMARY_VARIANT,
    BENCHMARK_REPORT_VERSION,
    BENCHMARK_RESIDENCY_MECHANISMS,
    BENCHMARK_RESIDENCY_REGIMES,
    FAMILY_RESIDUAL_CEILING_BYTES,
    FAMILY_VALIDATION_FAMILY_IDS,
    validate_benchmark_report,
)

if TYPE_CHECKING:
    from dinkster_assets.integrity import AssetVerificationRecord

#: Split-source families: the load_runtime keyword the text encoder
#: arrives through and the assembled field holding its module.
_SPLIT_TEXT_ENCODER_SLOTS = {
    "zimage": ("qwen3_4b", "qwen3_4b"),
    "wan21": ("t5xxl", "umt5xxl"),
}

_ANIMA_FAMILY = "anima"
_CHROMA_FAMILY = "chroma"
_FLUX_FAMILY = "flux"
_PROVIDER_WAN_FAMILIES = frozenset({"wan21_infinitetalk", "wan21_humo"})
_SPLIT_ARTIFACT_FAMILIES = frozenset(
    {
        *_SPLIT_TEXT_ENCODER_SLOTS,
        _ANIMA_FAMILY,
        _CHROMA_FAMILY,
        _FLUX_FAMILY,
        "minimax_h3",
        *_PROVIDER_WAN_FAMILIES,
    }
)
_VIDEO_FAMILIES = frozenset({"wan21", "minimax_h3", *_PROVIDER_WAN_FAMILIES})
_PLACEMENT_ARGUMENTS = ("residency", "direct-diagnostic")

_MIB = 1024 * 1024

#: Chunked ballast so a fragmented allocator cannot fail one huge
#: allocation.
_BALLAST_CHUNK_BYTES = 1024 * _MIB

#: Families whose official template zeroes the positive conditioning for
#: the negative input (ConditioningZeroOut) instead of encoding the
#: negative prompt; this runner mirrors that with a zeroed cond copy.
_ZERO_NEGATIVE_FAMILIES = frozenset({"zimage", "wan21_infinitetalk"})

#: Per-family workload defaults. The zimage and wan21 rows are the
#: official ComfyUI template settings (image_z_image_turbo.json and
#: text_to_video_wan.json); Dinkster's family-default sampling shifts
#: (3.0 for Z-Image, 8.0 for Wan 2.1) equal the templates' ModelSampling
#: shift values, so neither side overrides the schedule.
_WORKLOAD_DEFAULTS: dict[str, dict[str, object]] = {
    "zimage": {
        "steps": 8,
        "cfg": 1.0,
        "width": 1024,
        "height": 1024,
        "sampler": "dinkster.res_multistep",
        "scheduler": "dinkster.simple",
    },
    "wan21": {
        "steps": 30,
        "cfg": 6.0,
        "width": 832,
        "height": 480,
        "length": 33,
        "sampler": "dinkster.uni_pc",
        "scheduler": "dinkster.simple",
    },
    "wan21_infinitetalk": {
        "prompt": "The camera zooms in. Two characters are talking.",
        "negative_prompt": "",
        "seed": 0,
        "steps": 6,
        "cfg": 1.0,
        "width": 832,
        "height": 480,
        "length": 81,
        "sampler": "dinkster.euler",
        "scheduler": "dinkster.normal",
        "warm_runs": 3,
    },
    "wan21_humo": {
        "prompt": (
            "A young boy in sci-fi style clothing is talking to the camera in an alien desert."
        ),
        "negative_prompt": (
            "\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0c"
            "JPEG\u538b\u7f29\u6b8b\u7559\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c\u591a\u4f59\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c"
            "\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70"
        ),
        "seed": 0,
        "steps": 6,
        "cfg": 1.0,
        "width": 640,
        "height": 640,
        "length": 97,
        "sampler": "dinkster.uni_pc",
        "scheduler": "dinkster.simple",
        "warm_runs": 3,
    },
    "anima": {
        "prompt": BENCHMARK_ANIMA_PROMPT,
        "negative_prompt": "",
        "seed": 875817230929465,
        "steps": 30,
        "cfg": 4.0,
        "width": 1024,
        "height": 1024,
        "sampler": "dinkster.er_sde",
        "scheduler": "dinkster.simple",
        "warm_runs": 5,
    },
    "minimax_h3": {
        "prompt": "A red square centered on a black background.",
        "negative_prompt": "",
        "seed": 20260813,
        "steps": 20,
        "cfg": 1.0,
        "width": 1344,
        "height": 768,
        "length": 124,
        "sampler": "dinkster.res_multistep",
        "scheduler": "dinkster.simple",
        "warm_runs": 3,
    },
    # FLUX.1-dev at the official template settings; both sides run the
    # flat shift 1.15 schedule by default (ComfyUI Flux sampling_settings
    # and Dinkster FLUX_DEV shift agree), so neither overrides the schedule.
    "flux": {
        "steps": 20,
        "cfg": 1.0,
        "width": 1024,
        "height": 1024,
        "sampler": "dinkster.euler",
        "scheduler": "dinkster.simple",
        "guidance": 3.5,
    },
    # Chroma1-HD at the official template settings; the template pins
    # ModelSamplingAuraFlow shift 1.0, which matches Dinkster's native CHROMA
    # default, so only the ComfyUI graph carries a ModelSampling node. The
    # template's T5TokenizerOptions(min_padding=0, min_length=0) is applied
    # by both runners at encode time.
    "chroma": {
        "steps": 26,
        "cfg": 3.5,
        "width": 1024,
        "height": 1024,
        "sampler": "dinkster.euler",
        "scheduler": "dinkster.beta",
    },
}
_SD_ERA_DEFAULTS: dict[str, object] = {
    "prompt": "a photograph of an astronaut riding a horse",
    "negative_prompt": "",
    "seed": 667,
    "steps": 20,
    "cfg": 7.0,
    "width": 512,
    "height": 512,
    "sampler": "dinkster.euler",
    "scheduler": "dinkster.simple",
    "warm_runs": 5,
}

_INFINITETALK_ARTIFACT_PINS: dict[str, tuple[int, str, str]] = {
    "diffusion": (
        16_401_356_938,
        "b2de21b99b2e72cb0ff15253b07e926f26e7cf1b7e229efc32f94ad1f1ed9395",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/617a7633e636506f850e043bc4605f290a466a8e/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors",
    ),
    "text_encoder": (
        6_735_906_897,
        "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/617a7633e636506f850e043bc4605f290a466a8e/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    "vae": (
        253_815_318,
        "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/617a7633e636506f850e043bc4605f290a466a8e/split_files/vae/wan_2.1_vae.safetensors",
    ),
    "lora": (
        738_005_744,
        "85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/8260d429d19fd7a72304cad059160b95d843913f/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    ),
    "model_patch": (
        5_124_439_112,
        "4c2486cdfb6ff9a9f27408e98e11e20619136933b20411e0c365b1e84075d195",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/617a7633e636506f850e043bc4605f290a466a8e/split_files/model_patches/wan2.1_infiniteTalk_multi_fp16.safetensors",
    ),
    "audio_encoder": (
        190_115_368,
        "000813e441020f18cff844c969d2d5d4adc2a5ce46b2db1f23950b05d88805b4",
        "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/87847d3bc53702afda44078249e7c33e867827c4/wav2vec2-chinese-base_fp16.safetensors",
    ),
    "clip_vision": (
        1_264_219_396,
        "64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/617a7633e636506f850e043bc4605f290a466a8e/split_files/clip_vision/clip_vision_h.safetensors",
    ),
    "input_image": (
        1_297_223,
        "88a9d7bd3832304a5b66626c442886f0b82ddbce176089e504b8aeaf4cc3333e",
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/82b1954dc14c622be0f551d4020b4d8f961a5a48/input/two_character_talking.png",
    ),
    "input_audio_1": (
        105_996,
        "d008494976e34b05108f181942a6d4363e2bf1176ebabc10ecb69d2e61245afb",
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/82b1954dc14c622be0f551d4020b4d8f961a5a48/input/audio_speaker1_woman.mp3",
    ),
    "input_audio_2": (
        40_801,
        "632aecb453a9a58d37f9f9e70d07f6748ab604af59a564b84eb76031440d3545",
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/82b1954dc14c622be0f551d4020b4d8f961a5a48/input/audio_speaker2_man.mp3",
    ),
}

_HUMO_ARTIFACT_PINS: dict[str, tuple[int, str, str]] = {
    "diffusion": (
        17_058_372_152,
        "222ddeac4dea6b78363cb5be78c47660c92963a69386026cd6dc0de4d3094f66",
        "https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/2e746dc158c41696fd168accc7a3f19a6593fed6/split_files/diffusion_models/humo_17B_fp8_e4m3fn.safetensors",
    ),
    "text_encoder": (
        6_735_906_897,
        "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/617a7633e636506f850e043bc4605f290a466a8e/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    "vae": (
        253_815_318,
        "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/617a7633e636506f850e043bc4605f290a466a8e/split_files/vae/wan_2.1_vae.safetensors",
    ),
    "lora": (
        738_005_744,
        "85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/8260d429d19fd7a72304cad059160b95d843913f/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    ),
    "audio_encoder": (
        3_087_130_976,
        "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
        "https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/2e746dc158c41696fd168accc7a3f19a6593fed6/split_files/audio_encoders/whisper_large_v3_fp16.safetensors",
    ),
    "input_image": (
        1_117_950,
        "3a6662eba09c10b72d763cb947ca38e717998bafc55d7d0c14f72a1410ee1eb0",
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/82b1954dc14c622be0f551d4020b4d8f961a5a48/input/video_humo_reference_image.png",
    ),
    "input_audio": (
        5_264_684,
        "4e920892d3d33ebb8a04d772960a027f185fa55213ce0c60cfd0ec3faf191e8f",
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/82b1954dc14c622be0f551d4020b4d8f961a5a48/input/video_humo_input_audio.wav",
    ),
}

_ANIMA_ARTIFACT_PINS: dict[str, tuple[int, str, str]] = {
    "diffusion": (
        4_182_218_328,
        "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e",
        "https://huggingface.co/circlestone-labs/Anima/resolve/"
        "f973fc41ec7545364ac9776c2440285f43ff2a30/split_files/diffusion_models/"
        "anima-base-v1.0.safetensors",
    ),
    "text_encoder": (
        1_192_135_096,
        "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
        "https://huggingface.co/circlestone-labs/Anima/resolve/"
        "f973fc41ec7545364ac9776c2440285f43ff2a30/split_files/text_encoders/"
        "qwen_3_06b_base.safetensors",
    ),
    "vae": (
        253_806_246,
        "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
        "https://huggingface.co/circlestone-labs/Anima/resolve/"
        "f973fc41ec7545364ac9776c2440285f43ff2a30/split_files/vae/"
        "qwen_image_vae.safetensors",
    ),
}

_MINIMAX_H3_ARTIFACT_PINS: dict[str, tuple[int, str, str]] = {
    "diffusion": (
        34_038_892_334,
        "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/3f57e8291d2ef846f9a074b1b76d2767db434abe/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
    ),
    "text_encoder": (
        27_141_342_152,
        "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/3f57e8291d2ef846f9a074b1b76d2767db434abe/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    ),
    "video_vae": (
        5_207_808_496,
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/3f57e8291d2ef846f9a074b1b76d2767db434abe/vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    "audio_vae": (
        605_254_808,
        "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/3f57e8291d2ef846f9a074b1b76d2767db434abe/vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
}

_FLUX_ARTIFACT_PINS: dict[str, tuple[int, str, str]] = {
    "diffusion": (
        23_802_932_552,
        "4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7",
        "https://huggingface.co/Comfy-Org/flux1-dev/resolve/"
        "40a8a3d745c7d7adb077cb19879a975aa19c847b/flux1-dev.safetensors",
    ),
    "clip_l": (
        246_144_152,
        "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/"
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/clip_l.safetensors",
    ),
    "text_encoder": (
        9_787_841_024,
        "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/"
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp16.safetensors",
    ),
    "vae": (
        335_304_388,
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/"
        "08d04455279082882deaabc8d0d09fc914c071e1/split_files/vae/ae.safetensors",
    ),
}

_CHROMA_ARTIFACT_PINS: dict[str, tuple[int, str, str]] = {
    "diffusion": (
        9_193_379_316,
        "a2928ca6075f308f4d5e2182e2b96120fa8ad270ec6ea9b1b5c724c85c49a575",
        "https://huggingface.co/Comfy-Org/Chroma1-HD_repackaged/resolve/"
        "47f45ad2f72b2bccaa808418aeedca8c49d67974/split_files/diffusion_models/"
        "Chroma1-HD-fp8mixed.safetensors",
    ),
    "text_encoder": (
        5_157_348_688,
        "a498f0485dc9536735258018417c3fd7758dc3bccc0a645feaa472b34955557a",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/"
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    "vae": (
        335_304_388,
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/"
        "08d04455279082882deaabc8d0d09fc914c071e1/split_files/vae/ae.safetensors",
    ),
}

_ATTENTION_POLICIES = ("auto", "sdpa", "dinkster_kitchen_int8", "sage")


@dataclasses.dataclass(frozen=True)
class BackendAccess:
    """The per-backend torch device APIs this runner touches."""

    device: torch.device
    backend_runtime: str
    synchronize: Callable[[], None]
    reset_peak: Callable[[], None]
    peak_allocated: Callable[[], int]
    peak_reserved: Callable[[], int]
    allocated: Callable[[], int]
    empty_cache: Callable[[], None]


def _cuda_family_access(backend_runtime: str) -> BackendAccess:
    device = torch.device("cuda:0")
    return BackendAccess(
        device=device,
        backend_runtime=backend_runtime,
        synchronize=lambda: torch.cuda.synchronize(device),
        reset_peak=lambda: torch.cuda.reset_peak_memory_stats(device),
        peak_allocated=lambda: int(torch.cuda.max_memory_allocated(device)),
        peak_reserved=lambda: int(torch.cuda.max_memory_reserved(device)),
        allocated=lambda: int(torch.cuda.memory_allocated(device)),
        empty_cache=torch.cuda.empty_cache,
    )


def _admit_backend(backend: str) -> BackendAccess:
    """Admit the requested backend explicitly or exit; never fall through
    to whatever accelerator torch happens to see."""
    if backend == "rocm":
        hip = getattr(torch.version, "hip", None)
        if hip is None:
            sys.exit("error: this torch is not a HIP (ROCm) build")
        if not torch.cuda.is_available():
            sys.exit("error: torch cannot see a ROCm device on this machine")
        return _cuda_family_access(f"hip {hip}")
    if backend == "cuda":
        cuda = getattr(torch.version, "cuda", None)
        if cuda is None:
            sys.exit("error: this torch is not a CUDA build")
        if not torch.cuda.is_available():
            sys.exit("error: torch cannot see a CUDA device on this machine")
        return _cuda_family_access(f"cuda {cuda}")
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        sys.exit("error: torch cannot see an XPU device on this machine")
    build_xpu = str(getattr(torch.version, "xpu", "") or "").strip()
    device = torch.device("xpu:0")
    return BackendAccess(
        device=device,
        backend_runtime=f"xpu {build_xpu or torch.__version__}",
        synchronize=lambda: torch.xpu.synchronize(device),
        reset_peak=lambda: torch.xpu.reset_peak_memory_stats(device),
        peak_allocated=lambda: int(torch.xpu.max_memory_allocated(device)),
        peak_reserved=lambda: int(torch.xpu.max_memory_reserved(device)),
        allocated=lambda: int(torch.xpu.memory_allocated(device)),
        empty_cache=torch.xpu.empty_cache,
    )


def _load_residency_sampling() -> Any:
    """The sibling residency probe module, loaded by path so both
    benchmarks share one GPU shared-usage counter implementation."""
    path = Path(__file__).with_name("benchmark_residency.py")
    spec = importlib.util.spec_from_file_location("_benchmark_residency_sampling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the residency sampler from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ResidencySampler:
    """Best-effort GPU shared (sysmem) usage reader around the run."""

    def __init__(self, spill_scope: str) -> None:
        module = _load_residency_sampling()
        self.scope: str = module.resolve_spill_scope(
            spill_scope, platform.system(), platform.uname().release
        )
        self._read: Callable[[str, int], int | None] = module.gpu_shared_usage_bytes
        self._pid = os.getpid()

    def sample(self) -> int | None:
        if self.scope == "off":
            return None
        return self._read(self.scope, self._pid)


def _ballast_size_bytes(free_bytes: int, leave_free_mib: int) -> int:
    """Ballast that pins device free memory down to leave_free_mib."""
    return max(0, free_bytes - leave_free_mib * _MIB)


def _residency_section(
    *,
    mechanism: str,
    regime: str,
    leave_free_mib: int | None,
    ballast_bytes: int | None,
    spill_scope: str,
    shared_before_bytes: int | None,
    shared_warm_bytes: int | None,
    shared_after_bytes: int | None,
    aimdo_bootstrap_succeeded: bool | None = None,
    routes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Record raw shared usage; neither growth nor stability attributes driver spill."""
    growth = (
        shared_after_bytes - shared_warm_bytes
        if shared_after_bytes is not None and shared_warm_bytes is not None
        else None
    )
    return {
        "mechanism": mechanism,
        "aimdo_bootstrap_succeeded": aimdo_bootstrap_succeeded,
        "routes": dict(routes or {}),
        "regime": regime,
        "leave_free_mib": leave_free_mib,
        "ballast_bytes": ballast_bytes,
        "spill_scope": spill_scope,
        "shared_before_bytes": shared_before_bytes,
        "shared_warm_bytes": shared_warm_bytes,
        "shared_after_bytes": shared_after_bytes,
        "shared_growth_bytes": growth,
        "shared_spill_detected": None,
    }


def _device_entries(backend: str) -> list[dict[str, object]]:
    """Device identity entries, shaped exactly like the smoke reports'."""
    entries: list[dict[str, object]] = []
    if backend in ("rocm", "cuda"):
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            if backend == "rocm":
                architecture = str(getattr(properties, "gcnArchName", "") or "").strip()
            else:
                architecture = f"sm_{properties.major}{properties.minor}"
            entries.append(
                {
                    "index": index,
                    "name": properties.name,
                    "architecture": architecture,
                    "total_memory": int(properties.total_memory),
                }
            )
        return entries
    for index in range(torch.xpu.device_count()):
        properties = torch.xpu.get_device_properties(index)
        architecture = ""
        for attribute in ("architecture", "device_id", "platform_name"):
            value = getattr(properties, attribute, None)
            if value:
                architecture = f"{attribute}={value}"
                break
        entries.append(
            {
                "index": index,
                "name": properties.name,
                "architecture": architecture,
                "total_memory": int(properties.total_memory),
            }
        )
    return entries


def _windows_video_controllers() -> list[str]:
    command = (
        "Get-CimInstance Win32_VideoController | "
        "ForEach-Object { $_.Name + ' driver ' + $_.DriverVersion }"
    )
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return [line.strip() for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def _driver_identity(backend: str) -> str:
    """Best available display-driver identity, as the smoke scripts record."""
    parts: list[str] = []
    if backend == "xpu" and torch.xpu.device_count() > 0:
        properties = torch.xpu.get_device_properties(0)
        for attribute in ("driver_version", "platform_name"):
            value = str(getattr(properties, attribute, "") or "").strip()
            if value:
                parts.append(f"{attribute}={value}")
    if sys.platform == "win32":
        parts.extend(_windows_video_controllers())
    elif backend == "rocm":
        for label, candidate in (
            ("amdgpu kernel driver", "/sys/module/amdgpu/version"),
            ("rocm userspace", "/opt/rocm/.info/version"),
        ):
            try:
                text = Path(candidate).read_text().strip()
            except OSError:
                continue
            if text:
                parts.append(f"{label} {text}")
        if not parts and Path("/sys/module/amdgpu").is_dir():
            # The in-tree amdgpu module has no version file; for in-tree
            # builds the kernel release is the driver version.
            parts.append(f"amdgpu in-tree kernel driver, kernel {platform.uname().release}")
    elif backend == "cuda":
        try:
            text = Path("/proc/driver/nvidia/version").read_text().splitlines()[0].strip()
        except (OSError, IndexError):
            text = ""
        if text:
            parts.append(text)
    if parts:
        return "; ".join(parts)
    return "unknown (no driver identity source on this host)"


def _artifact_entry(
    role: str,
    path: Path,
    pin: tuple[int, str, str] | None = None,
    *,
    include_asset_preflight: bool = False,
) -> dict[str, object]:
    digest = hashlib.sha256()
    asset_hasher: Any = None
    if include_asset_preflight:
        from blake3 import blake3

        asset_hasher = blake3()
    with path.open("rb") as file:
        before_stat = os.fstat(file.fileno())
        for chunk in iter(lambda: file.read(1 << 22), b""):
            digest.update(chunk)
            if asset_hasher is not None:
                asset_hasher.update(chunk)
        after_stat = os.fstat(file.fileno())
    before_fingerprint = (
        before_stat.st_dev,
        before_stat.st_ino,
        before_stat.st_size,
        before_stat.st_mtime_ns,
        before_stat.st_ctime_ns,
    )
    after_fingerprint = (
        after_stat.st_dev,
        after_stat.st_ino,
        after_stat.st_size,
        after_stat.st_mtime_ns,
        after_stat.st_ctime_ns,
    )
    if before_fingerprint != after_fingerprint:
        raise ValueError(f"{role} artifact changed while its digests were computed: {path}")
    actual_size = after_stat.st_size
    actual_sha256 = digest.hexdigest()
    if pin is not None:
        expected_size, expected_sha256, _url = pin
        if actual_size != expected_size or actual_sha256 != expected_sha256:
            raise ValueError(
                f"{role} artifact does not match its pin:"
                f" expected {expected_size} bytes/{expected_sha256},"
                f" got {actual_size} bytes/{actual_sha256}"
            )
    entry: dict[str, object] = {
        "role": role,
        "path": str(path),
        "sha256": actual_sha256,
        "bytes": actual_size,
    }
    if pin is not None:
        entry["url"] = pin[2]
    if asset_hasher is not None:
        from dinkster_assets.integrity import verification_record

        asset_digest = f"blake3:{asset_hasher.hexdigest()}"
        verification = verification_record(asset_digest, after_stat)
        if verification is None:
            raise RuntimeError(f"cannot bind {role} artifact verification to its hashed file")
        entry["_asset_preflight"] = _AssetPreflight(
            path=path,
            digest=asset_digest,
            size=actual_size,
            verification=verification,
        )
    return entry


@dataclasses.dataclass(frozen=True)
class _AssetPreflight:
    path: Path
    digest: str
    size: int
    verification: AssetVerificationRecord


@dataclasses.dataclass(frozen=True)
class _FixedAssetResolver:
    path: Path
    digest: str
    verification: AssetVerificationRecord

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == self.digest else None

    def resolve_asset(self, digest: str) -> object | None:
        if digest != self.digest:
            return None
        from dinkster_assets.model import AssetResolution

        return AssetResolution(self.path, self.verification)


def _asset_ref(path: Path, preflight: _AssetPreflight | None = None) -> object:
    """Bind a benchmark path to the same verified AssetRef nodes receive."""
    from dinkster_assets import AssetRef
    from dinkster_assets.integrity import digest_file_with_record

    if preflight is None:
        digest, verification = digest_file_with_record(path)
        if verification is None:
            raise RuntimeError(f"cannot bind asset verification to its hashed file: {path}")
        preflight = _AssetPreflight(path, digest, verification.size, verification)
    elif preflight.path != path:
        raise ValueError(f"asset preflight path {preflight.path} does not match {path}")
    resolver = _FixedAssetResolver(path, preflight.digest, preflight.verification)
    return AssetRef(
        digest=preflight.digest,
        name=path.name,
        size=preflight.size,
        resolver=resolver,
    )


def _anima_component_load(path: Path, role: str, preflight: _AssetPreflight) -> tuple[Any, Any]:
    from dinkster_inference import (
        BFLOAT16,
        FLOAT32,
        anima_component_runtime_identity,
        load_safetensors_header,
        plan_anima_split_component,
        plan_qwen_image_official_component,
        qwen_image_component_runtime_identity,
    )
    from dinkster_workers.execution import ExecutionContext

    asset: Any = _asset_ref(path, preflight)
    source = load_safetensors_header(
        path,
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    if role == "vae":
        plan = plan_qwen_image_official_component(source, role="vae", path=path)
        identity = qwen_image_component_runtime_identity(plan, "vae", BFLOAT16)
        dtypes = ("unloaded", "unloaded", "bfloat16")
    else:
        component_role = "diffusion" if role == "diffusion" else "qwen3_06b"
        dtype = BFLOAT16 if role == "diffusion" else FLOAT32
        plan = plan_anima_split_component(source, role=component_role, path=path)
        identity = anima_component_runtime_identity(plan, component_role, dtype)
        dtypes = (
            "bfloat16" if role == "diffusion" else "unloaded",
            "float32" if role == "text_encoder" else "unloaded",
            "unloaded",
        )
    context = ExecutionContext(
        "native",
        identity,
        diffusion_dtype=dtypes[0],
        text_dtype=dtypes[1],
        vae_dtype=dtypes[2],
    )
    return asset, context


def _chroma_component_load(path: Path, role: str, preflight: _AssetPreflight) -> tuple[Any, Any]:
    from dinkster_inference import (
        BFLOAT16,
        chroma_component_runtime_identity,
        load_safetensors_header,
        plan_chroma_split_component,
    )
    from dinkster_workers.execution import ExecutionContext

    asset: Any = _asset_ref(path, preflight)
    source = load_safetensors_header(
        path,
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    component_role = "t5xxl" if role == "text_encoder" else role
    plan = plan_chroma_split_component(source, role=component_role, path=path)
    identity = chroma_component_runtime_identity(plan, component_role, BFLOAT16)
    context = ExecutionContext(
        "native",
        identity,
        diffusion_dtype="bfloat16" if role == "diffusion" else "unloaded",
        text_dtype="bfloat16" if role == "text_encoder" else "unloaded",
        vae_dtype="bfloat16" if role == "vae" else "unloaded",
    )
    return asset, context


def _minimax_h3_execution_identities(
    assets: dict[str, Any],
    *,
    attention_policy: str = "auto",
    attention_route_token: Any = None,
) -> dict[str, str]:
    """Plan the exact split H3 graph and derive each loader identity."""
    import dinkster_inference
    import dinkster_inference_torch

    paths = {
        "fl2va-dit": assets["diffusion"].local_path(),
        "qwen3vl-32b-conditioner": assets["text_encoder"].local_path(),
        "video-vae": assets["video_vae"].local_path(),
        "audio-vae": assets["audio_vae"].local_path(),
    }
    role_assets = {
        "fl2va-dit": assets["diffusion"],
        "qwen3vl-32b-conditioner": assets["text_encoder"],
        "video-vae": assets["video_vae"],
        "audio-vae": assets["audio_vae"],
    }
    sources = {
        role: dinkster_inference.load_safetensors_header(
            paths[role],
            asset_digest=asset.digest,
            asset_size=asset.size,
        )
        for role, asset in role_assets.items()
    }
    authority = dinkster_inference_torch.MiniMaxH3ArtifactPaths(
        "fl2va-dit",
        paths,
        role_assets,
    )
    plan = dinkster_inference_torch.plan_minimax_h3_split_assembly(
        diffusion=sources["fl2va-dit"],
        conditioner=sources["qwen3vl-32b-conditioner"],
        video_vae=sources["video-vae"],
        audio_vae=sources["audio-vae"],
        artifacts=authority,
    )
    diffusion = role_assets["fl2va-dit"]
    # The DiT identity must be the one load_minimax_h3_model constructs, so it
    # is planned through the same model-assembly path the loader uses. The
    # split plan's diffusion component carries extra artifact facts the loader
    # never includes, so its identity can never match.
    diffusion_plan = dinkster_inference_torch.plan_minimax_h3_model_assembly(
        sources["fl2va-dit"],
        role="fl2va-dit",
        path=paths["fl2va-dit"],
        attention_policy=attention_policy,
    ).diffusion
    return {
        "diffusion": dinkster_inference.minimax_h3_dit_runtime_identity(
            asset_digest=diffusion.digest,
            asset_size=diffusion.size,
            role="fl2va-dit",
            diffusion_dtype="bfloat16",
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            runtime_facts=diffusion_plan.identity_facts,
        ),
        "text_encoder": dinkster_inference_torch.minimax_h3_conditioner_runtime_identity(
            plan.conditioner,
            conditioner_dtype=torch.bfloat16,
        ),
        "video_vae": dinkster_inference_torch.minimax_h3_video_vae_runtime_identity(
            plan.video_vae,
            video_vae_dtype=torch.float16,
        ),
        "audio_vae": dinkster_inference_torch.minimax_h3_audio_vae_runtime_identity(
            plan.audio_vae,
            audio_vae_dtype=torch.float32,
        ),
    }


def speaker_mask_arrays(width: int, height: int) -> tuple[object, object]:
    """Build exact non-overlapping full-height half-frame masks."""
    if width <= 0 or width % 2 or height <= 0:
        raise ValueError(
            "speaker mask dimensions require a positive even width and positive height"
        )
    import numpy as np

    first = np.zeros((1, height, width), dtype=np.float32)
    second = np.zeros_like(first)
    half = width // 2
    first[:, :, :half] = 1.0
    second[:, :, half:] = 1.0
    return first, second


def _audio_frame_array(frame: object) -> object:
    import numpy as np

    typed = frame
    source = np.asarray(typed.to_ndarray())
    channels = len(typed.layout.channels)
    if source.ndim != 2 or channels < 1:
        raise ValueError(f"decoded audio frame has invalid shape {source.shape}")
    if typed.format.is_planar:
        if source.shape[0] != channels:
            raise ValueError("planar audio frame does not match its channel layout")
        planar = source
    else:
        if source.size % channels:
            raise ValueError("packed audio frame does not match its channel layout")
        planar = source.reshape(-1, channels).T
    if np.issubdtype(planar.dtype, np.signedinteger):
        converted = planar.astype(np.float32) / float(-np.iinfo(planar.dtype).min)
    elif np.issubdtype(planar.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(planar.dtype).max + 1) / 2.0
        converted = (planar.astype(np.float32) - midpoint) / midpoint
    else:
        converted = planar.astype(np.float32)
    return np.ascontiguousarray(converted)


def _decode_audio_file(path: Path) -> dict[str, object]:
    import av
    import numpy as np

    pieces: list[object] = []
    sample_rate: int | None = None
    channels: int | None = None
    total_bytes = 0
    with av.open(str(path), mode="r") as container:
        streams = tuple(container.streams.audio)
        if not streams:
            raise ValueError(f"audio input has no audio stream: {path}")
        stream = streams[0]
        for decoded in container.decode(stream):
            frame = decoded
            if frame.sample_rate is None or frame.sample_rate <= 0:
                raise ValueError("audio frame has no usable sample rate")
            rate = int(frame.sample_rate)
            if sample_rate is None:
                sample_rate = rate
            elif sample_rate != rate:
                raise ValueError("audio sample rate changes within the stream")
            array = _audio_frame_array(frame)
            if channels is None:
                channels = int(array.shape[0])
            elif channels != int(array.shape[0]):
                raise ValueError("audio channel count changes within the stream")
            total_bytes += int(array.nbytes)
            if total_bytes > 256 * 1024 * 1024:
                raise ValueError("decoded audio exceeds the 256 MiB limit")
            pieces.append(array)
    if not pieces or sample_rate is None:
        raise ValueError(f"audio input decodes to no samples: {path}")
    return {"waveform": np.concatenate(pieces, axis=1)[None, ...], "sample_rate": sample_rate}


def _decode_image_file(path: Path) -> object:
    from dinkster_values import decode_image_file

    return decode_image_file(_asset_ref(path))


def _zero_conditioning(value: object) -> object:
    from dinkster_inference import ConditioningCarrier, PayloadBinding, make_conditioning_carrier

    if type(value) is not ConditioningCarrier:
        raise TypeError("conditioning must be an exact ConditioningCarrier")
    carrier = value
    bindings = tuple(
        PayloadBinding(
            binding.reference_id,
            binding.shape,
            binding.dtype,
            binding.space,
            bytes(len(binding.data)),
        )
        for binding in carrier.bindings
    )
    return make_conditioning_carrier(carrier.conditioning, bindings)


def _peak_rss_bytes() -> int | None:
    """The process's lifetime peak resident set size, when the platform
    exposes one."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak <= 0:
        return None
    # Linux reports ru_maxrss in kilobytes, macOS in bytes.
    return peak * 1024 if sys.platform.startswith("linux") else peak


def _quality_file(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 22), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _capture_quality_tensor(
    tensor: Any,
    path: Path,
    *,
    spatial_stride: int | None,
) -> dict[str, object]:
    """Persist float32 NPY evidence one leading slice at a time."""
    import numpy as np

    source_shape = tuple(int(value) for value in tensor.shape)
    if not source_shape:
        raise ValueError("quality capture requires a non-scalar tensor")
    if spatial_stride is not None:
        if len(source_shape) != 4:
            raise ValueError("spatial quality capture requires a rank-4 THWC tensor")
        captured_shape = (
            source_shape[0],
            (source_shape[1] + spatial_stride - 1) // spatial_stride,
            (source_shape[2] + spatial_stride - 1) // spatial_stride,
            source_shape[3],
        )
    else:
        captured_shape = source_shape
    if path.exists():
        raise FileExistsError(f"quality capture refuses to overwrite {path}")
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=captured_shape)
    try:
        for index in range(source_shape[0]):
            source = tensor[index]
            if spatial_stride is not None:
                source = source[::spatial_stride, ::spatial_stride, :]
            output[index] = source.detach().to(device="cpu", dtype=torch.float32).numpy()
        output.flush()
    finally:
        del output
    return {
        **_quality_file(path),
        "dtype": "float32",
        "source_shape": list(source_shape),
        "captured_shape": list(captured_shape),
        "spatial_stride": spatial_stride,
    }


class BenchmarkRun:
    """One benchmark cell's mutable execution state."""

    def __init__(
        self,
        arguments: argparse.Namespace,
        access: BackendAccess,
        preflight_assets: Mapping[str, _AssetPreflight] | None = None,
    ) -> None:
        self.arguments = arguments
        self.access = access
        self.preflight_assets = dict(preflight_assets or {})
        self.checks: dict[str, dict[str, object]] = {}
        self.failed = False
        self.runtime: Any = None
        self.model_handle: Any = None
        self.clip_handle: Any = None
        self.model: Any = None
        self.vae: Any = None
        self.clip: Any = None
        self.audio_vae: Any = None
        self.model_patch: Any = None
        self.audio_encoder: Any = None
        self.component_publisher: Any = None
        self.executed_placement: str | None = None
        self.execution_path: str | None = None
        self.latent: Any = None
        self.family_id = "load-failed"
        self.cond: Any = None
        self.uncond: Any = None
        self.first_image: torch.Tensor | None = None
        self.first_audio: Any = None
        self.decoded_audio: Any = None
        attention_policy = getattr(arguments, "attention_policy", "auto")
        self.attention: dict[str, object] = {
            "requested_policy": attention_policy,
            "scope": ["diffusion:flux"],
            "route_token": None,
        }
        self.attention_route_token: Any = None
        self.quality_capture: dict[str, object] | None = None
        self.residency_routes: dict[str, object] = {}
        self.cold: dict[str, object] = {}
        self.warm_entries: list[dict[str, object]] = []
        self.residual_allocated = 0

    def record(self, name: str, action: Callable[[], str], *, always: bool = False) -> None:
        if self.failed and not always:
            self.checks[name] = {"ok": False, "detail": "not reached: an earlier check failed"}
            print(f"  {name:<14} NOT REACHED")
            return
        try:
            detail = action() or "ok"
            ok = True
        except Exception as error:  # the failure text is the check's result
            # The compact detail keeps only the first and last lines; for
            # nested errors the root cause sits mid-traceback, so preserve
            # it in the log.
            traceback.print_exc()
            lines = [line.strip() for line in str(error).splitlines() if line.strip()]
            detail = lines[0] if lines else type(error).__name__
            if len(lines) > 1:
                detail = f"{detail} [last line: {lines[-1]}]"
            detail = detail[:400]
            ok = False
        self.checks[name] = {"ok": ok, "detail": detail}
        if not ok:
            self.failed = True
        print(f"  {name:<14} {'ok (' + detail + ')' if ok else 'NO (' + detail + ')'}")

    def _timed(self, action: Callable[[], None]) -> float:
        start = time.perf_counter()
        action()
        self.access.synchronize()
        return time.perf_counter() - start

    def _capture_required_residency(
        self, family_label: str, route_handles: Mapping[str, object]
    ) -> None:
        for role, handle in route_handles.items():
            facts = getattr(handle, "residency_route", None)
            if facts is None:
                raise RuntimeError(
                    f"{family_label} {role} handle did not publish residency route facts"
                )
            self.residency_routes[role] = {
                "requested": facts.requested,
                "mechanism": facts.mechanism,
                "fallback_reason": facts.fallback_reason,
                "dynamic_components": list(facts.dynamic_components),
                "resident_components": list(facts.resident_components),
                "fallback_components": list(facts.fallback_components),
            }
        mechanism = os.environ.get("DINKSTER_AIMDO_ARM", "auto")
        required_mechanism = (
            "aimdo"
            if mechanism == "on" or (mechanism == "auto" and self.arguments.backend == "cuda")
            else None
        )
        if required_mechanism is None:
            return
        if required_mechanism == "aimdo" and _AIMDO_BOOTSTRAP_SUCCEEDED is not True:
            raise RuntimeError(
                f"{family_label} deep residency requires a successful pre-torch Aimdo bootstrap"
            )
        invalid = [
            role
            for role, route in self.residency_routes.items()
            if cast("Mapping[str, object]", route)["requested"] != mechanism
            or cast("Mapping[str, object]", route)["mechanism"] != required_mechanism
            or cast("Mapping[str, object]", route)["fallback_reason"] is not None
            or not cast("Mapping[str, object]", route)["dynamic_components"]
            or bool(cast("Mapping[str, object]", route)["fallback_components"])
        ]
        if invalid:
            raise RuntimeError(
                f"{family_label} {required_mechanism} residency did not own components: "
                + ", ".join(invalid)
            )

    def _stage(
        self,
        role: str,
        *,
        unload_before: tuple[str, ...] = (),
        observer_stage: Literal["condition", "load", "sample"] = "load",
    ) -> contextlib.AbstractContextManager[None]:
        if self.model_handle is None:
            return contextlib.nullcontext()
        return self.model_handle.stage(
            role,
            unload_before=unload_before,
            observer_stage=observer_stage,
        )

    # ---------------------------------------------------------------- checks

    def load(self) -> str:
        if self.arguments.family in _PROVIDER_WAN_FAMILIES:
            return self._load_provider_wan()
        if self.arguments.family == _ANIMA_FAMILY:
            return self._load_anima()
        if self.arguments.family == _CHROMA_FAMILY:
            return self._load_chroma()
        if self.arguments.family == "minimax_h3":
            return self._load_minimax_h3()
        if self.arguments.placement == "residency":
            return self._load_native_runtime()

        from dinkster_inference import load_safetensors_header
        from dinkster_inference_torch import load_runtime

        arguments = self.arguments
        split_slots = _SPLIT_TEXT_ENCODER_SLOTS.get(arguments.family)
        placed: list[str] = []

        def build() -> None:
            if split_slots is None:
                source = load_safetensors_header(arguments.checkpoint)
                self.runtime = load_runtime(checkpoint=source)  # family-default dtypes
            else:
                loader_kwarg, _ = split_slots
                self.runtime = load_runtime(  # family-default dtypes
                    diffusion=load_safetensors_header(arguments.diffusion),
                    vae=load_safetensors_header(arguments.vae),
                    **{loader_kwarg: load_safetensors_header(arguments.text_encoder)},
                )
            staged_field = None if split_slots is None else split_slots[1]
            for field in dataclasses.fields(self.runtime.assembled):
                value = getattr(self.runtime.assembled, field.name)
                if not isinstance(value, torch.nn.Module):
                    continue
                if field.name == staged_field:
                    # The text encoder stays on the CPU until encode_text.
                    placed.append(f"{field.name}={_module_dtype(value)} (cpu)")
                    continue
                if split_slots is not None and field.name == "diffusion":
                    # Placed after text encoding; see encode_text.
                    placed.append(f"{field.name}={_module_dtype(value)} (placed after encode)")
                    continue
                value.to(self.access.device)
                placed.append(f"{field.name}={_module_dtype(value)}")

        self.cold["load_s"] = round(self._timed(build), 4)
        if split_slots is None:
            self.executed_placement = BENCHMARK_DINKSTER_DIAGNOSTIC_PLACEMENT
        family_id = self.runtime.assembled.family.id
        if family_id not in FAMILY_VALIDATION_FAMILY_IDS[arguments.family]:
            raise RuntimeError(
                f"detected family {family_id!r} is not a {arguments.family} cell member"
            )
        self.family_id = family_id
        if arguments.mode == "compile":
            self.runtime.assembled.diffusion.compile()
        return f"{family_id}; " + ", ".join(placed) + f" in {self.cold['load_s']}s"

    def _load_native_runtime(self) -> str:
        from dinkster_compat_comfy import native_arm
        from dinkster_workers.execution import ExecutionContext, use_execution_context

        arguments = self.arguments
        if arguments.family == _FLUX_FAMILY:
            preflights = self.preflight_assets
            assets = {
                "diffusion": _asset_ref(arguments.diffusion, preflights["diffusion"]),
                "clip_l": _asset_ref(arguments.clip_l, preflights["clip_l"]),
                "t5xxl": _asset_ref(arguments.text_encoder, preflights["text_encoder"]),
                "vae": _asset_ref(arguments.vae, preflights["vae"]),
            }
        elif arguments.family in _SPLIT_TEXT_ENCODER_SLOTS:
            text_role = "qwen3_4b" if arguments.family == "zimage" else "t5xxl"
            assets = {
                "diffusion": _asset_ref(arguments.diffusion),
                text_role: _asset_ref(arguments.text_encoder),
                "vae": _asset_ref(arguments.vae),
            }
        else:
            assets = {"checkpoint": _asset_ref(arguments.checkpoint)}

        def build() -> None:
            with use_execution_context(ExecutionContext("native", None)):
                self.model_handle = native_arm.load_native_runtime_handle(assets)
            self.runtime = self.model_handle.runtime

        self.cold["load_s"] = round(self._timed(build), 4)
        self.executed_placement = BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT
        self._capture_required_residency(arguments.family, {"runtime": self.model_handle})
        family_id = self.runtime.assembled.family.id
        if family_id not in FAMILY_VALIDATION_FAMILY_IDS[arguments.family]:
            raise RuntimeError(
                f"detected family {family_id!r} is not a {arguments.family} cell member"
            )
        self.family_id = family_id
        if arguments.mode == "compile":
            self.runtime.assembled.diffusion.compile()
        placed = [
            f"{field.name}={_module_dtype(value)}"
            for field in dataclasses.fields(self.runtime.assembled)
            if isinstance((value := getattr(self.runtime.assembled, field.name)), torch.nn.Module)
        ]
        return (
            f"{family_id}; {', '.join(placed)} enrolled through the production native handle"
            f" in {self.cold['load_s']}s"
        )

    def _load_anima(self) -> str:
        from dinkster_compat_comfy import native_arm
        from dinkster_workers.execution import use_execution_context

        arguments = self.arguments
        diffusion, diffusion_context = _anima_component_load(
            arguments.diffusion, "diffusion", self.preflight_assets["diffusion"]
        )
        text_encoder, text_context = _anima_component_load(
            arguments.text_encoder, "text_encoder", self.preflight_assets["text_encoder"]
        )
        vae, vae_context = _anima_component_load(arguments.vae, "vae", self.preflight_assets["vae"])

        def build() -> None:
            created: list[Any] = []
            try:
                with use_execution_context(diffusion_context):
                    self.model_handle = native_arm.GenerationLoadDiffusionModel.execute(
                        diffusion_model=diffusion,
                        weight_dtype="default",
                    )["model"]
                created.append(self.model_handle)
                with use_execution_context(text_context):
                    self.clip_handle = native_arm.NativeLoadClip.execute(
                        text_encoder=text_encoder,
                        type="stable_diffusion",
                        device="default",
                    )["clip"]
                created.append(self.clip_handle)
                with use_execution_context(vae_context):
                    self.vae = native_arm.NativeLoadVae.execute(vae=vae)["vae"]
                created.append(self.vae)
            except BaseException:
                for handle in reversed(created):
                    handle.terminal_release()
                self.model_handle = None
                self.clip_handle = None
                self.vae = None
                raise
            self.model = self.model_handle
            self.runtime = self.model_handle.runtime

        self.cold["load_s"] = round(self._timed(build), 4)
        family_id = self.runtime.assembled.family.id
        if family_id not in FAMILY_VALIDATION_FAMILY_IDS[arguments.family]:
            raise RuntimeError(
                f"detected family {family_id!r} is not a {arguments.family} cell member"
            )
        self.family_id = family_id
        self.executed_placement = BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT
        return (
            f"{family_id}; diffusion, text encoder, and VAE enrolled through production"
            f" component handles in {self.cold['load_s']}s"
        )

    def _load_chroma(self) -> str:
        from dinkster_compat_comfy import native_arm
        from dinkster_workers.execution import use_execution_context

        arguments = self.arguments
        diffusion, diffusion_context = _chroma_component_load(
            arguments.diffusion, "diffusion", self.preflight_assets["diffusion"]
        )
        text_encoder, text_context = _chroma_component_load(
            arguments.text_encoder, "text_encoder", self.preflight_assets["text_encoder"]
        )
        vae, vae_context = _chroma_component_load(
            arguments.vae, "vae", self.preflight_assets["vae"]
        )

        def build() -> None:
            created: list[Any] = []
            try:
                with use_execution_context(diffusion_context):
                    self.model_handle = native_arm.GenerationLoadDiffusionModel.execute(
                        diffusion_model=diffusion,
                        weight_dtype="default",
                    )["model"]
                created.append(self.model_handle)
                with use_execution_context(text_context):
                    self.clip_handle = native_arm.NativeLoadClip.execute(
                        text_encoder=text_encoder,
                        type="chroma",
                        device="default",
                    )["clip"]
                created.append(self.clip_handle)
                with use_execution_context(vae_context):
                    self.vae = native_arm.NativeLoadVae.execute(vae=vae)["vae"]
                created.append(self.vae)
            except BaseException:
                for handle in reversed(created):
                    handle.terminal_release()
                self.model_handle = None
                self.clip_handle = None
                self.vae = None
                raise
            self.model = self.model_handle
            self.runtime = self.model_handle.runtime

        self.cold["load_s"] = round(self._timed(build), 4)
        family_id = self.runtime.assembled.family.id
        if family_id not in FAMILY_VALIDATION_FAMILY_IDS[arguments.family]:
            raise RuntimeError(
                f"detected family {family_id!r} is not a {arguments.family} cell member"
            )
        self.family_id = family_id
        self.executed_placement = BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT
        return (
            f"{family_id}; diffusion, text encoder, and VAE enrolled through production"
            f" component handles in {self.cold['load_s']}s"
        )

    def _load_provider_wan(self) -> str:
        from dinkster_compat_comfy import NativeComponentPublisher, native_arm
        from dinkster_inference_torch import use_component_publisher
        from dinkster_model_wan.provider import execute_load_wav2vec2_audio_encoder
        from dinkster_workers.execution import ExecutionContext, use_execution_context

        arguments = self.arguments
        infinitetalk = arguments.family == "wan21_infinitetalk"
        preflights = self.preflight_assets
        runtime_assets = {
            "diffusion": _asset_ref(arguments.diffusion, preflights["diffusion"]),
            "t5xxl": _asset_ref(arguments.text_encoder, preflights["text_encoder"]),
            "vae": _asset_ref(arguments.vae, preflights["vae"]),
        }
        if infinitetalk:
            runtime_assets["clip_vision"] = _asset_ref(
                arguments.clip_vision, preflights["clip_vision"]
            )
        lora = _asset_ref(arguments.lora, preflights["lora"])
        model_patch = (
            _asset_ref(arguments.model_patch, preflights["model_patch"]) if infinitetalk else None
        )
        audio_encoder = _asset_ref(arguments.audio_encoder, preflights["audio_encoder"])
        self.component_publisher = NativeComponentPublisher()

        def build() -> None:
            with use_execution_context(ExecutionContext("native", None)):
                base_handle = native_arm.load_native_runtime_handle(runtime_assets)
                try:
                    self.model_handle = native_arm.GenerationLoadLoraModelOnly.execute(
                        model=base_handle,
                        lora=lora,
                        strength_model=arguments.lora_strength_model,
                        execution_mode="precalculate",
                    )["model"]
                except BaseException:
                    base_handle.terminal_release()
                    raise
                if self.model_handle is not base_handle:
                    base_handle.terminal_release()
                self.model = self.model_handle
                self.vae = native_arm._NativeCodecHandle(self.model_handle)
                if model_patch is not None:
                    self.model_patch = native_arm.NativeLoadZImageControlPatch.execute(
                        model_patch=model_patch
                    )["model_patch"]
                with use_component_publisher(self.component_publisher):
                    self.audio_encoder = execute_load_wav2vec2_audio_encoder(
                        audio_encoder=audio_encoder
                    )["audio_encoder"]
            self.runtime = self.model_handle.runtime

        self.cold["load_s"] = round(self._timed(build), 4)
        self.executed_placement = BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT
        family_id = self.runtime.assembled.family.id
        if family_id not in FAMILY_VALIDATION_FAMILY_IDS[arguments.family]:
            raise RuntimeError(
                f"detected family {family_id!r} is not a {arguments.family} cell member"
            )
        self.family_id = family_id
        diffusion = self.runtime.assembled.diffusion
        extra_models = "model patch/audio encoder" if infinitetalk else "Whisper encoder"
        return (
            f"{family_id}; diffusion={_module_dtype(diffusion)}, LoRA/{extra_models}"
            f" loaded in {self.cold['load_s']}s"
        )

    def _load_minimax_h3(self) -> str:
        from dinkster_compat_comfy import native_arm
        from dinkster_protocol import attention_route_token_to_wire
        from dinkster_workers.execution import ExecutionContext, use_execution_context

        arguments = self.arguments
        if arguments.attention_policy != "auto":
            from dinkster_inference_torch import discover_attention_route_token

            self.attention_route_token = discover_attention_route_token(arguments.attention_policy)
            self.attention["route_token"] = attention_route_token_to_wire(
                self.attention_route_token
            )
        assets = {
            "diffusion": _asset_ref(arguments.diffusion),
            "text_encoder": _asset_ref(arguments.text_encoder),
            "video_vae": _asset_ref(arguments.vae),
            "audio_vae": _asset_ref(arguments.audio_vae),
        }
        identities = (
            _minimax_h3_execution_identities(assets)
            if arguments.attention_policy == "auto"
            else _minimax_h3_execution_identities(
                assets,
                attention_policy=arguments.attention_policy,
                attention_route_token=self.attention_route_token,
            )
        )
        loaded: list[Any] = []

        def load(
            role: str,
            action: Callable[[], Any],
            *,
            diffusion_dtype: str,
            text_dtype: str,
            vae_dtype: str,
        ) -> Any:
            if role == "diffusion":
                context = ExecutionContext(
                    "native",
                    identities[role],
                    diffusion_dtype=diffusion_dtype,
                    text_dtype=text_dtype,
                    vae_dtype=vae_dtype,
                    attention_policy=arguments.attention_policy,
                    attention_route_token=self.attention_route_token,
                )
            else:
                context = ExecutionContext(
                    "native",
                    identities[role],
                    diffusion_dtype=diffusion_dtype,
                    text_dtype=text_dtype,
                    vae_dtype=vae_dtype,
                )
            with use_execution_context(context):
                handle = action()
            loaded.append(handle)
            return handle

        def build() -> None:
            try:
                model = load(
                    "diffusion",
                    lambda: native_arm.GenerationLoadDiffusionModel.execute(
                        diffusion_model=assets["diffusion"],
                        weight_dtype="default",
                    )["model"],
                    diffusion_dtype="bfloat16",
                    text_dtype="unloaded",
                    vae_dtype="unloaded",
                )
                clip = load(
                    "text_encoder",
                    lambda: native_arm.NativeLoadClip.execute(
                        text_encoder=assets["text_encoder"],
                        type="minimax",
                        device="default",
                    )["clip"],
                    diffusion_dtype="unloaded",
                    text_dtype="bfloat16",
                    vae_dtype="unloaded",
                )
                video_vae = load(
                    "video_vae",
                    lambda: native_arm.NativeLoadVae.execute(vae=assets["video_vae"])["vae"],
                    diffusion_dtype="unloaded",
                    text_dtype="unloaded",
                    vae_dtype="float16",
                )
                audio_vae = load(
                    "audio_vae",
                    lambda: native_arm.NativeLoadVae.execute(vae=assets["audio_vae"])["vae"],
                    diffusion_dtype="unloaded",
                    text_dtype="unloaded",
                    vae_dtype="float32",
                )
            except BaseException:
                for handle in reversed(loaded):
                    handle.terminal_release()
                raise
            self.model_handle = model
            self.model = model
            self.runtime = model.runtime
            self.clip = clip
            self.vae = video_vae
            self.audio_vae = audio_vae

        self.cold["load_s"] = round(self._timed(build), 4)
        self.executed_placement = BENCHMARK_DINKSTER_PRODUCTION_PLACEMENT
        self.family_id = "dinkster.minimax_h3"
        self._capture_required_residency(
            "MiniMax H3",
            {
                "diffusion": self.model_handle,
                "conditioner": self.clip,
                "video_vae": self.vae,
                "audio_vae": self.audio_vae,
            },
        )
        return (
            "dinkster.minimax_h3; FL2VA DiT, conditioner, video VAE, and audio VAE "
            f"enrolled through production native handles in {self.cold['load_s']}s"
        )

    def lora_apply(self) -> str:
        if self.arguments.placement == "residency":
            return self._apply_native_lora()

        # The decode/route/patch flow mirrors family_validation.py
        # lora_apply, minus the effect and restore checks a benchmark
        # does not make claims about.
        from dinkster_inference import (
            PatchTarget,
            clip_lora_key_map,
            decode_lora,
            load_safetensors_header,
            native_unet_key_map,
        )
        from dinkster_inference_torch import (
            ModuleStateStore,
            build_patch_set,
            load_tensors,
            lora_compute_dtype,
            patch_weights,
        )

        arguments = self.arguments
        assembled = self.runtime.assembled
        counts: list[str] = []

        def apply() -> None:
            header = load_safetensors_header(arguments.lora)
            geometries = {key: header.entry(key).geometry for key in header.keys()}
            diffusion_keys = tuple(
                f"diffusion_model.{key}" for key in assembled.diffusion.state_dict()
            )
            key_map: dict[str, object] = dict(native_unet_key_map(diffusion_keys))
            clip_keys: list[str] = []
            for component in ("clip_l", "clip_g"):
                module = getattr(assembled, component, None)
                if module is None:
                    continue
                clip_keys.extend(f"{component}.transformer.{key}" for key in module.state_dict())
            key_map.update(clip_lora_key_map(clip_keys))
            decoded = decode_lora(geometries, key_map)
            if not decoded.patches:
                raise RuntimeError("LoRA decoded to zero patches matching this model")
            routes = ("diffusion", "clip_l", "clip_g")
            per_component: dict[str, dict[Any, object]] = {}
            for target, patch in decoded.patches.items():
                for component in routes:
                    prefix = (
                        "diffusion_model."
                        if component == "diffusion"
                        else f"{component}.transformer."
                    )
                    if (
                        target.key.startswith(prefix)
                        and getattr(assembled, component, None) is not None
                    ):
                        routed = PatchTarget(target.key.removeprefix(prefix), offset=target.offset)
                        per_component.setdefault(component, {})[routed] = patch
                        break
                else:
                    raise RuntimeError(f"decoded LoRA target {target.key!r} has no component route")
            tensors = load_tensors(arguments.lora)
            patch_dtype = lora_compute_dtype(self.access.device)
            for component in sorted(per_component):
                strength = (
                    arguments.lora_strength_model
                    if component == "diffusion"
                    else arguments.lora_strength_clip
                )
                patch_set = build_patch_set(per_component[component], tensors, strength=strength)
                store = ModuleStateStore(getattr(assembled, component))
                backup = patch_weights(
                    store,
                    patch_set,
                    weight_dtype=patch_dtype,
                    backup_device=torch.device("cpu"),
                )
                counts.append(f"{component}={len(backup)}")

        self.cold["lora_s"] = round(self._timed(apply), 4)
        return f"patched weights {', '.join(counts)} in {self.cold['lora_s']}s"

    def _apply_native_lora(self) -> str:
        from dinkster_compat_comfy import native_arm

        arguments = self.arguments
        base_handle = self.model_handle
        assert base_handle is not None

        def apply() -> None:
            loaded = native_arm.GenerationLoadLora.execute(
                model=base_handle,
                clip=base_handle,
                lora=_asset_ref(arguments.lora),
                strength_model=arguments.lora_strength_model,
                strength_clip=arguments.lora_strength_clip,
                execution_mode="precalculate",
            )
            self.model_handle = loaded["model"]
            self.runtime = self.model_handle.runtime
            if self.model_handle is not base_handle:
                base_handle.terminal_release()

        self.cold["lora_s"] = round(self._timed(apply), 4)
        return f"LoRA applied through the production native handle in {self.cold['lora_s']}s"

    def encode_text(self) -> str:
        arguments = self.arguments
        if arguments.family in _PROVIDER_WAN_FAMILIES:
            return self._encode_provider_text()
        if arguments.family == _ANIMA_FAMILY:
            return self._encode_anima_text()
        if arguments.family == _CHROMA_FAMILY:
            return self._encode_chroma_text()
        if arguments.family == "minimax_h3":
            return self._encode_minimax_h3()
        split_slots = _SPLIT_TEXT_ENCODER_SLOTS.get(arguments.family)

        def encode() -> None:
            encoder: torch.nn.Module | None = None
            if split_slots is not None and self.model_handle is None:
                encoder = getattr(self.runtime.assembled, split_slots[1])
                assert encoder is not None
                encoder.to(self.access.device)
            from dinkster_inference import Conditioning

            with self._stage("text"), torch.inference_mode():
                cond = self.runtime.encode_text(arguments.prompt)
                if arguments.family in _ZERO_NEGATIVE_FAMILIES:
                    uncond = Conditioning(
                        embeddings=torch.zeros_like(cond.embeddings),
                        pooled=None if cond.pooled is None else torch.zeros_like(cond.pooled),
                    )
                else:
                    uncond = self.runtime.encode_text(arguments.negative_prompt)
                if arguments.family == "wan21":
                    cond = self.runtime.prepare_text_conditioning(cond)
                    uncond = self.runtime.prepare_text_conditioning(uncond)
                self.cond = cond
                self.uncond = uncond
            if encoder is not None:
                encoder.to(torch.device("cpu"))
                self.access.empty_cache()

        self.cold["encode_s"] = round(self._timed(encode), 4)
        placement = ""
        if split_slots is not None and self.model_handle is None:

            def place() -> None:
                self.runtime.assembled.diffusion.to(self.access.device)

            place_s = round(self._timed(place), 4)
            self.cold["load_s"] = round(float(self.cold["load_s"]) + place_s, 4)
            self.executed_placement = BENCHMARK_DINKSTER_DIAGNOSTIC_PLACEMENT
            placement = f"; diffusion placed in {place_s}s (counted in load_s)"
        staged = (
            ""
            if split_slots is None
            else (
                "; text encoder leased through residency"
                if self.model_handle is not None
                else "; text encoder staged to device and back"
            )
        )
        encoded = (
            "cond encoded and uncond zeroed"
            if arguments.family in _ZERO_NEGATIVE_FAMILIES
            else "cond and uncond encoded"
        )
        return f"{encoded} in {self.cold['encode_s']}s{staged}{placement}"

    def _encode_anima_text(self) -> str:
        from dinkster_compat_comfy import native_arm

        def encode() -> None:
            self.cond = native_arm.GenerationClipTextEncode.execute(
                text=self.arguments.prompt,
                clip=self.clip_handle,
            )["conditioning"]
            self.uncond = native_arm.GenerationClipTextEncode.execute(
                text=self.arguments.negative_prompt,
                clip=self.clip_handle,
            )["conditioning"]

        self.cold["encode_s"] = round(self._timed(encode), 4)
        return (
            "cond and uncond encoded through the production text handle in "
            f"{self.cold['encode_s']}s"
        )

    def _encode_chroma_text(self) -> str:
        from dinkster_compat_comfy import native_arm

        def encode() -> None:
            # The reference graph pins T5 tokenizer padding off; mirror it here.
            clip = native_arm.GenerationT5TokenizerOptions.execute(
                clip=self.clip_handle,
                min_padding=0,
                min_length=0,
            )["clip"]
            self.cond = native_arm.GenerationClipTextEncode.execute(
                text=self.arguments.prompt,
                clip=clip,
            )["conditioning"]
            self.uncond = native_arm.GenerationClipTextEncode.execute(
                text=self.arguments.negative_prompt,
                clip=clip,
            )["conditioning"]

        self.cold["encode_s"] = round(self._timed(encode), 4)
        return (
            "cond and uncond encoded through the production text handle in "
            f"{self.cold['encode_s']}s"
        )

    def _encode_provider_text(self) -> str:
        from dinkster_compat_comfy import native_arm

        def encode() -> None:
            positive = native_arm.GenerationClipTextEncode.execute(
                text=self.arguments.prompt,
                clip=self.model_handle,
            )["conditioning"]
            self.cond = positive
            if self.arguments.family == "wan21_infinitetalk":
                self.uncond = _zero_conditioning(positive)
            else:
                self.uncond = native_arm.GenerationClipTextEncode.execute(
                    text=self.arguments.negative_prompt,
                    clip=self.model_handle,
                )["conditioning"]

        self.cold["encode_s"] = round(self._timed(encode), 4)
        detail = (
            "cond encoded and uncond zeroed"
            if self.arguments.family == "wan21_infinitetalk"
            else "cond and uncond encoded"
        )
        return f"{detail} in {self.cold['encode_s']}s"

    def _encode_minimax_h3(self) -> str:
        from dinkster_compat_comfy import native_arm

        arguments = self.arguments

        def encode() -> None:
            self.latent = native_arm.NativeEmptyMiniMaxH3AV.execute(
                width=arguments.width,
                height=arguments.height,
                frame_count=arguments.length,
            )["latent"]
            conditioned = native_arm.NativeMiniMaxH3T2VAConditioning.execute(
                clip=self.clip,
                target=self.latent,
                prompt=arguments.prompt,
            )
            self.cond = conditioned["positive"]
            self.uncond = conditioned["negative"]

        self.cold["encode_s"] = round(self._timed(encode), 4)
        if self.uncond != []:
            raise RuntimeError(
                "MiniMax H3 T2VA conditioning did not produce a neutral negative lane"
            )
        return f"positive-only T2VA conditioning encoded in {self.cold['encode_s']}s"

    def encode_audio(self) -> str:
        if self.arguments.family == "wan21_humo":
            return self._encode_humo_audio()

        from dinkster_model_wan.provider import (
            execute_encode_wav2vec2_audio,
            execute_wan_infinite_talk_to_video,
        )

        arguments = self.arguments

        def encode() -> None:
            first_audio = _decode_audio_file(arguments.input_audio_1)
            second_audio = _decode_audio_file(arguments.input_audio_2)
            image = _decode_image_file(arguments.input_image)
            first_mask, second_mask = speaker_mask_arrays(arguments.width, arguments.height)
            first_output = execute_encode_wav2vec2_audio(
                audio_encoder=self.audio_encoder,
                audio=first_audio,
            )["audio_encoder_output"]
            second_output = execute_encode_wav2vec2_audio(
                audio_encoder=self.audio_encoder,
                audio=second_audio,
            )["audio_encoder_output"]
            conditioned = execute_wan_infinite_talk_to_video(
                mode="two_speakers",
                model=self.model_handle,
                model_patch=self.model_patch,
                positive=self.cond,
                negative=self.uncond,
                vae=self.vae,
                width=arguments.width,
                height=arguments.height,
                length=arguments.length,
                audio_encoder_output_1=first_output,
                motion_frame_count=arguments.motion_frame_count,
                audio_scale=arguments.audio_scale,
                start_image=image,
                audio_encoder_output_2=second_output,
                mask_1=first_mask,
                mask_2=second_mask,
            )
            self.model = conditioned["model"]
            self.cond = conditioned["positive"]
            self.uncond = conditioned["negative"]
            self.latent = conditioned["latent"]

        self.cold["audio_encode_s"] = round(self._timed(encode), 4)
        return (
            "two audio streams encoded and InfiniteTalk conditioning prepared in "
            f"{self.cold['audio_encode_s']}s"
        )

    def _encode_humo_audio(self) -> str:
        from dinkster_model_wan.provider import (
            execute_encode_wav2vec2_audio,
            execute_wan21_humo,
        )

        arguments = self.arguments

        def encode() -> None:
            audio = _decode_audio_file(arguments.input_audio)
            image = _decode_image_file(arguments.input_image)
            audio_output = execute_encode_wav2vec2_audio(
                audio_encoder=self.audio_encoder,
                audio=audio,
            )["audio_encoder_output"]
            conditioned = execute_wan21_humo(
                positive=self.cond,
                negative=self.uncond,
                vae=self.vae,
                width=arguments.width,
                height=arguments.height,
                length=arguments.length,
                batch_size=1,
                audio_encoder_output=audio_output,
                ref_image=image,
            )
            self.cond = conditioned["positive"]
            self.uncond = conditioned["negative"]
            self.latent = conditioned["latent"]

        self.cold["audio_encode_s"] = round(self._timed(encode), 4)
        return f"Whisper audio and HuMo conditioning encoded in {self.cold['audio_encode_s']}s"

    def _guidance(self) -> object | None:
        """The uncond guidance, or None at cfg 1.0 (both systems skip it)."""
        from dinkster_inference import SamplingGuidance

        if self.arguments.cfg == 1.0:
            return None
        return SamplingGuidance(self.uncond, self.arguments.cfg)

    def _run_once(self, seed: int) -> tuple[torch.Tensor, float, float, list[float]]:
        if self.arguments.family in _PROVIDER_WAN_FAMILIES:
            return self._run_provider_video(seed)
        if self.arguments.family == _ANIMA_FAMILY:
            return self._run_anima(seed)
        if self.arguments.family == _CHROMA_FAMILY:
            return self._run_chroma(seed)
        if self.arguments.family == "minimax_h3":
            return self._run_minimax_h3(seed)
        if self.arguments.family == "wan21":
            return self._run_video(seed)
        return self._run_image(seed)

    def _run_anima(self, seed: int) -> tuple[torch.Tensor, float, float, list[float]]:
        from dinkster_compat_comfy import native_arm

        arguments = self.arguments
        if self.latent is None:
            self.latent = native_arm.GenerationEmptyLatentImage.execute(
                width=arguments.width,
                height=arguments.height,
                batch_size=1,
            )["latent"]
        start = time.perf_counter()
        sampled = native_arm.GenerationKSampler.execute(
            model=self.model,
            seed=seed,
            steps=arguments.steps,
            cfg=arguments.cfg,
            sampler_name=arguments.sampler,
            scheduler=arguments.scheduler,
            positive=self.cond,
            negative=self.uncond,
            latent_image=self.latent,
            denoise=1.0,
        )["latent"]
        self.access.synchronize()
        sample_s = time.perf_counter() - start
        start = time.perf_counter()
        image = native_arm.GenerationVAEDecode.execute(samples=sampled, vae=self.vae)["image"]
        self.access.synchronize()
        decode_s = time.perf_counter() - start
        return image, sample_s, decode_s, []

    def _run_chroma(self, seed: int) -> tuple[torch.Tensor, float, float, list[float]]:
        from dinkster_compat_comfy import native_arm
        from dinkster_schema import use_reporter

        arguments = self.arguments
        if self.latent is None:
            self.latent = native_arm.GenerationEmptySD3LatentImage.execute(
                width=arguments.width,
                height=arguments.height,
                batch_size=1,
            )["latent"]
        boundaries: list[tuple[int, float]] = []

        def report(name: str, data: Mapping[str, object], _blob: bytes | None) -> None:
            if name == "progress" and data.get("total") == arguments.steps:
                step = data.get("step")
                if isinstance(step, int) and not isinstance(step, bool):
                    boundaries.append((step, time.perf_counter()))

        start = time.perf_counter()
        with use_reporter(report):
            sampled = native_arm.GenerationKSampler.execute(
                model=self.model,
                seed=seed,
                steps=arguments.steps,
                cfg=arguments.cfg,
                sampler_name=arguments.sampler,
                scheduler=arguments.scheduler,
                positive=self.cond,
                negative=self.uncond,
                latent_image=self.latent,
                denoise=1.0,
            )["latent"]
        self.access.synchronize()
        sample_s = time.perf_counter() - start
        step_wall_ms: list[float] = []
        if [step for step, _boundary in boundaries] == list(range(1, arguments.steps + 1)):
            timestamps = [boundary for _step, boundary in boundaries]
            step_wall_ms = [
                round((later - earlier) * 1000, 3)
                for earlier, later in zip([start, *timestamps[:-1]], timestamps, strict=True)
            ]
        start = time.perf_counter()
        image = native_arm.GenerationVAEDecode.execute(samples=sampled, vae=self.vae)["image"]
        self.access.synchronize()
        decode_s = time.perf_counter() - start
        return image, sample_s, decode_s, step_wall_ms

    def _run_provider_video(self, seed: int) -> tuple[torch.Tensor, float, float, list[float]]:
        from dinkster_compat_comfy import native_arm

        arguments = self.arguments
        start = time.perf_counter()
        sampled = native_arm.GenerationKSampler.execute(
            model=self.model,
            seed=seed,
            steps=arguments.steps,
            cfg=arguments.cfg,
            sampler_name=arguments.sampler,
            scheduler=arguments.scheduler,
            positive=self.cond,
            negative=self.uncond,
            latent_image=self.latent,
            denoise=1.0,
        )["latent"]
        self.access.synchronize()
        sample_s = time.perf_counter() - start
        start = time.perf_counter()
        image = native_arm.GenerationVAEDecode.execute(samples=sampled, vae=self.vae)["image"]
        self.access.synchronize()
        decode_s = time.perf_counter() - start
        return image, sample_s, decode_s, []

    def _run_minimax_h3(self, seed: int) -> tuple[torch.Tensor, float, float, list[float]]:
        from dinkster_compat_comfy import native_arm
        from dinkster_schema import use_reporter
        from dinkster_workers.execution import ExecutionContext, use_execution_context

        arguments = self.arguments
        boundaries: list[tuple[int, float]] = []

        def report(name: str, data: Mapping[str, object], _blob: bytes | None) -> None:
            if name == "progress" and data.get("total") == arguments.steps:
                step = data.get("step")
                if isinstance(step, int) and not isinstance(step, bool):
                    boundaries.append((step, time.perf_counter()))

        start = time.perf_counter()
        # H3 MultiStreamLatent sampling is exposed through the production
        # KSampler path; issue #798 tracks SamplerCustomAdvanced support.
        with (
            use_execution_context(ExecutionContext("native", None, preview_mode="off")),
            use_reporter(report),
        ):
            sampled = native_arm.GenerationKSampler.execute(
                model=self.model,
                seed=seed,
                steps=arguments.steps,
                cfg=arguments.cfg,
                sampler_name=arguments.sampler,
                scheduler=arguments.scheduler,
                positive=self.cond,
                negative=self.uncond,
                latent_image=self.latent,
                denoise=1.0,
            )["latent"]
        self.execution_path = BENCHMARK_MINIMAX_H3_DINKSTER_EXECUTION_PATH
        expected_steps = list(range(1, arguments.steps + 1))
        actual_steps = [step for step, _boundary in boundaries]
        if actual_steps != expected_steps:
            raise RuntimeError(
                f"MiniMax H3 sampler reported progress steps {actual_steps}, "
                f"expected {expected_steps}"
            )
        timestamps = [boundary for _step, boundary in boundaries]
        step_wall_ms = [
            round((later - earlier) * 1000, 3)
            for earlier, later in zip([start, *timestamps[:-1]], timestamps, strict=True)
        ]
        self.access.synchronize()
        sample_s = time.perf_counter() - start
        start = time.perf_counter()
        decoded = native_arm.NativeMiniMaxH3AVDecode.execute(
            video_vae=self.vae,
            audio_vae=self.audio_vae,
            latent=sampled,
        )
        self.access.synchronize()
        decode_s = time.perf_counter() - start
        self.decoded_audio = decoded["audio"]
        frames = decoded["frames"]
        if not isinstance(frames, torch.Tensor):
            raise RuntimeError("MiniMax H3 decode did not return tensor frames")
        return frames, sample_s, decode_s, step_wall_ms

    def _run_image(self, seed: int) -> tuple[torch.Tensor, float, float, list[float]]:
        """One sample+decode; returns (image, sample_s, decode_s, step_wall_ms)."""
        arguments = self.arguments
        latent_space = self.runtime.assembled.family.single_stream_latent()
        latent = torch.zeros(
            (
                1,
                latent_space.channels,
                arguments.height // latent_space.spatial_downscale,
                arguments.width // latent_space.spatial_downscale,
            ),
            dtype=torch.float32,
            device=self.access.device,
        )
        # SD-era runs keep the reference fp16 compute pick explicit;
        # split-artifact families take the runtime's bfloat16 default.
        compute: dict[str, Any] = {}
        if arguments.family not in _SPLIT_ARTIFACT_FAMILIES:
            compute["compute_dtype"] = torch.float16
        if arguments.family == _FLUX_FAMILY:
            compute["guidance"] = arguments.guidance
        boundaries: list[float] = []

        def on_step(event: object) -> None:
            boundaries.append(time.perf_counter())

        start = time.perf_counter()
        with (
            self._stage("diffusion", unload_before=("text",), observer_stage="sample"),
            torch.inference_mode(),
        ):
            sampled = self.runtime.sample(
                latent,
                cond=self.cond,
                cfg=self._guidance(),
                sampler_id=arguments.sampler,
                scheduler_id=arguments.scheduler,
                steps=arguments.steps,
                denoise=1.0,
                seed=seed,
                device=self.access.device,
                on_step=on_step,
                **compute,
            )
        self.access.synchronize()
        sample_s = time.perf_counter() - start
        step_wall_ms = [
            round((later - earlier) * 1000, 3)
            for earlier, later in zip([start, *boundaries[:-1]], boundaries, strict=False)
        ]
        start = time.perf_counter()
        with self._stage("vae"), torch.inference_mode():
            image = self.runtime.decode_latent(sampled).permute(0, 2, 3, 1).clamp(0, 1)
        self.access.synchronize()
        decode_s = time.perf_counter() - start
        return image, sample_s, decode_s, step_wall_ms

    def _run_video(self, seed: int) -> tuple[torch.Tensor, float, float, list[float]]:
        """One video sample+decode; returns (frames, sample_s, decode_s, step_wall_ms)."""
        from dinkster_inference import WAN21_LATENT, MultiStreamLatent

        arguments = self.arguments
        latent_frames = (arguments.length - 1) // WAN21_LATENT.temporal_downscale + 1
        latent = torch.zeros(
            (
                1,
                WAN21_LATENT.channels,
                latent_frames,
                arguments.height // WAN21_LATENT.spatial_downscale,
                arguments.width // WAN21_LATENT.spatial_downscale,
            ),
            dtype=torch.float32,
            device=self.access.device,
        )
        boundaries: list[float] = []

        def on_step(event: object) -> None:
            boundaries.append(time.perf_counter())

        start = time.perf_counter()
        with (
            self._stage("diffusion", unload_before=("text",), observer_stage="sample"),
            torch.inference_mode(),
        ):
            sampled = self.runtime.sample_multistream(
                MultiStreamLatent.from_pairs((("video", latent),)),
                conditioning=self.cond,
                cfg=self._guidance(),
                sampler_id=arguments.sampler,
                scheduler_id=arguments.scheduler,
                steps=arguments.steps,
                denoise=1.0,
                seed=seed,
                device=self.access.device,
                on_step=on_step,
            ).by_role("video")
        self.access.synchronize()
        sample_s = time.perf_counter() - start
        step_wall_ms = [
            round((later - earlier) * 1000, 3)
            for earlier, later in zip([start, *boundaries[:-1]], boundaries, strict=False)
        ]
        start = time.perf_counter()
        with self._stage("vae"), torch.inference_mode():
            # decode_latent yields [1, 3, T, H, W] in 0..1; report [1, T, H, W, 3].
            video = self.runtime.decode_latent(sampled).permute(0, 2, 3, 4, 1).clamp(0, 1)
        self.access.synchronize()
        decode_s = time.perf_counter() - start
        return video, sample_s, decode_s, step_wall_ms

    def cold_run(self) -> str:
        arguments = self.arguments
        image, sample_s, decode_s, step_wall_ms = self._run_once(arguments.seed)
        self.first_image = image
        if arguments.family == "minimax_h3":
            self.first_audio = self.decoded_audio
        self.cold["sample_s"] = round(sample_s, 4)
        self.cold["decode_s"] = round(decode_s, 4)
        if len(step_wall_ms) == arguments.steps:
            self.cold["step_wall_ms"] = step_wall_ms
        phases = (
            "load_s",
            "lora_s",
            "encode_s",
            "audio_encode_s",
            "sample_s",
            "decode_s",
        )
        self.cold["total_s"] = round(
            sum(float(self.cold[name]) for name in phases if name in self.cold), 4
        )
        execution_path = (
            " via GenerationKSampler production path" if arguments.family == "minimax_h3" else ""
        )
        return (
            f"sample{execution_path} {self.cold['sample_s']}s, decode {self.cold['decode_s']}s,"
            f" total {self.cold['total_s']}s"
        )

    def finite_output(self) -> str:
        assert self.first_image is not None
        if not bool(torch.isfinite(self.first_image).all()):
            raise RuntimeError("cold image contains non-finite values")
        shape = tuple(self.first_image.shape)
        if self.arguments.family == "minimax_h3":
            detail = self._validate_minimax_h3_output(
                self.first_image,
                self.first_audio,
                "cold",
            )
            self.first_audio = None
        else:
            detail = f"image shape {shape}, all values finite"
        self.first_image = None
        return detail

    def _validate_minimax_h3_output(
        self,
        frames: torch.Tensor,
        audio: object,
        run_name: str,
    ) -> str:
        arguments = self.arguments
        expected_video = (arguments.length, arguments.height, arguments.width, 3)
        if tuple(frames.shape) != expected_video:
            raise RuntimeError(
                f"{run_name} MiniMax H3 video shape {tuple(frames.shape)} is not {expected_video}"
            )
        if not isinstance(audio, Mapping):
            raise RuntimeError(f"{run_name} MiniMax H3 audio output is not a mapping")
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if not isinstance(waveform, torch.Tensor) or (
            waveform.ndim != 3 or tuple(waveform.shape[:2]) != (1, 2) or waveform.shape[2] <= 0
        ):
            raise RuntimeError(f"{run_name} MiniMax H3 audio is not a batch-one stereo waveform")
        if sample_rate != 32_000:
            raise RuntimeError(
                f"{run_name} MiniMax H3 audio sample rate {sample_rate!r} is not 32000"
            )
        if not bool(torch.isfinite(waveform).all()):
            raise RuntimeError(f"{run_name} MiniMax H3 audio contains non-finite values")
        return (
            f"video shape {expected_video} and audio shape {tuple(waveform.shape)} at 32000 Hz, "
            "all values finite"
        )

    def warm_runs(self) -> str:
        arguments = self.arguments
        for index in range(arguments.warm_runs):
            image, sample_s, decode_s, step_wall_ms = self._run_once(arguments.seed + 1 + index)
            if not bool(torch.isfinite(image).all()):
                raise RuntimeError(f"warm run {index} image contains non-finite values")
            if arguments.family == "minimax_h3":
                self._validate_minimax_h3_output(image, self.decoded_audio, f"warm run {index}")
            entry: dict[str, object] = {
                "sample_s": round(sample_s, 4),
                "decode_s": round(decode_s, 4),
                "total_s": round(sample_s + decode_s, 4),
            }
            if len(step_wall_ms) == arguments.steps:
                entry["step_wall_ms"] = step_wall_ms
            self.warm_entries.append(entry)
        medians = ", ".join(
            f"{name}={statistics.median(float(entry[name]) for entry in self.warm_entries):.4f}"
            for name in ("sample_s", "decode_s", "total_s")
        )
        return f"{arguments.warm_runs} runs; median {medians}"

    def capture_quality(self) -> str:
        arguments = self.arguments
        if arguments.family != "minimax_h3" or arguments.quality_output_dir is None:
            raise RuntimeError("quality capture was not configured for MiniMax H3")
        capture_seed = arguments.seed + arguments.warm_runs + 1
        image, _sample_s, _decode_s, _step_wall_ms = self._run_once(capture_seed)
        detail = self._validate_minimax_h3_output(image, self.decoded_audio, "quality capture")
        assert isinstance(self.decoded_audio, Mapping)
        waveform = self.decoded_audio["waveform"]
        output_dir = arguments.quality_output_dir
        self.quality_capture = {
            "version": 1,
            "seed": capture_seed,
            "image": _capture_quality_tensor(
                image,
                output_dir / "capture_image.npy",
                spatial_stride=arguments.quality_spatial_stride,
            ),
            "audio": _capture_quality_tensor(
                waveform,
                output_dir / "capture_audio.npy",
                spatial_stride=None,
            ),
            "audio_sample_rate": self.decoded_audio["sample_rate"],
        }
        self.decoded_audio = None
        return detail

    def warm_section(self) -> dict[str, object]:
        section: dict[str, object] = {"runs": self.warm_entries}
        for name in ("sample_s", "decode_s", "total_s"):
            if self.warm_entries:
                section[f"median_{name}"] = round(
                    statistics.median(float(entry[name]) for entry in self.warm_entries), 4
                )
        return section

    def unload(self) -> str:
        if self.arguments.family in _PROVIDER_WAN_FAMILIES:
            return self._unload_provider_wan()
        if self.arguments.family == "minimax_h3":
            return self._unload_minimax_h3()
        self.model = None
        self.latent = None
        if self.clip_handle is not None:
            self.clip_handle.terminal_release()
            self.clip_handle = None
        if self.arguments.family in {_ANIMA_FAMILY, _CHROMA_FAMILY} and self.vae is not None:
            self.vae.terminal_release()
            self.vae = None
        if self.model_handle is not None:
            self.model_handle.terminal_release()
            self.model_handle = None
        elif self.runtime is not None:
            for field in dataclasses.fields(self.runtime.assembled):
                value = getattr(self.runtime.assembled, field.name)
                if isinstance(value, torch.nn.Module):
                    value.to(torch.device("cpu"))
        self.runtime = None
        self.cond = None
        self.uncond = None
        self.first_image = None
        if self.arguments.mode == "compile":
            # torch.compile caches (inductor constants, cudagraph pools)
            # hold device tensors beyond the module references.
            import torch._dynamo as torch_dynamo

            torch_dynamo.reset()
        # The cuBLAS/hipBLAS workspace is allocated through the caching
        # allocator (32 MiB observed on torch 2.13 cu130) and counts as
        # allocated bytes without being a leak; release it so the residual
        # ceiling measures only tensors the cell failed to free.
        if self.access.device.type == "cuda":
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
        gc.collect()
        self.access.empty_cache()
        self.access.synchronize()
        residual = self.access.allocated()
        self.residual_allocated = residual
        if residual > FAMILY_RESIDUAL_CEILING_BYTES:
            raise RuntimeError(f"allocator still holds {residual} B after unload")
        return f"residual_allocated={residual} B"

    def _unload_minimax_h3(self) -> str:
        self.latent = None
        self.cond = None
        self.uncond = None
        self.first_image = None
        self.first_audio = None
        self.decoded_audio = None
        for attribute in ("audio_vae", "vae", "clip", "model_handle"):
            handle = getattr(self, attribute)
            if handle is not None:
                handle.terminal_release()
                setattr(self, attribute, None)
        self.model = None
        self.runtime = None
        gc.collect()
        if self.access.device.type == "cuda":
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
        self.access.empty_cache()
        self.access.synchronize()
        residual = self.access.allocated()
        self.residual_allocated = residual
        if residual > FAMILY_RESIDUAL_CEILING_BYTES:
            raise RuntimeError(f"allocator still holds {residual} B after unload")
        return f"residual_allocated={residual} B"

    def _unload_provider_wan(self) -> str:
        self.model = None
        self.latent = None
        self.cond = None
        self.uncond = None
        self.first_image = None
        self.vae = None
        self.audio_encoder = None
        gc.collect()
        if self.component_publisher is not None:
            self.component_publisher.close()
            self.component_publisher = None
        if self.model_patch is not None:
            self.model_patch.terminal_release()
            self.model_patch = None
        if self.model_handle is not None:
            self.model_handle.terminal_release()
            self.model_handle = None
        self.runtime = None
        gc.collect()
        if self.access.device.type == "cuda":
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
        self.access.empty_cache()
        self.access.synchronize()
        residual = self.access.allocated()
        self.residual_allocated = residual
        if residual > FAMILY_RESIDUAL_CEILING_BYTES:
            raise RuntimeError(f"allocator still holds {residual} B after unload")
        return f"residual_allocated={residual} B"


def _module_dtype(module: torch.nn.Module) -> str:
    """The module's dominant parameter dtype by element count; small fp32
    islands (patch embeddings, norms) do not misreport a bf16 model."""
    elements: dict[torch.dtype, int] = {}
    for parameter in module.parameters():
        elements[parameter.dtype] = elements.get(parameter.dtype, 0) + parameter.numel()
    if not elements:
        return "no-parameters"
    dominant = max(elements, key=lambda dtype: elements[dtype])
    return str(dominant).removeprefix("torch.")


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BENCHMARK_ACCELERATORS, required=True)
    parser.add_argument("--family", choices=BENCHMARK_FAMILIES, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument(
        "--placement",
        choices=_PLACEMENT_ARGUMENTS,
        default="residency",
        help="production residency (default) or explicit direct-placement diagnostic",
    )
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint")
    parser.add_argument("--diffusion", type=Path, help="diffusion safetensors (split families)")
    parser.add_argument("--text-encoder", type=Path, help="text encoder safetensors (split)")
    parser.add_argument("--clip-l", type=Path, help="Flux CLIP-L text encoder safetensors")
    parser.add_argument("--vae", type=Path, help="VAE safetensors (split families)")
    parser.add_argument("--audio-vae", type=Path, help="MiniMax H3 audio VAE safetensors")
    parser.add_argument("--lora", type=Path, help="LoRA safetensors")
    parser.add_argument("--model-patch", type=Path, help="InfiniteTalk model patch safetensors")
    parser.add_argument("--audio-encoder", type=Path, help="Wan audio encoder safetensors")
    parser.add_argument("--clip-vision", type=Path, help="Wan CLIP vision safetensors")
    parser.add_argument("--input-image", type=Path, help="reference image input")
    parser.add_argument("--input-audio-1", type=Path, help="first speaker audio input")
    parser.add_argument("--input-audio-2", type=Path, help="second speaker audio input")
    parser.add_argument("--input-audio", type=Path, help="HuMo audio input")
    parser.add_argument("--lora-strength-model", type=float, default=1.0)
    parser.add_argument("--lora-strength-clip", type=float, default=1.0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--length", type=int, default=None, help="video frames")
    parser.add_argument(
        "--fallback-768",
        action="store_true",
        help="run the explicitly labeled 768x768 Anima fallback",
    )
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--guidance", type=float, default=None, help="Flux distilled guidance")
    parser.add_argument("--sampler", default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--warm-runs", type=int, default=None)
    parser.add_argument("--motion-frame-count", type=int, default=9)
    parser.add_argument("--audio-scale", type=float, default=1.0)
    parser.add_argument(
        "--attention-policy",
        choices=_ATTENTION_POLICIES,
        default="auto",
        help="MiniMax H3 diffusion attention implementation",
    )
    parser.add_argument(
        "--quality-output-dir",
        type=Path,
        default=None,
        help="write an unmeasured decoded output used for quality comparison",
    )
    parser.add_argument(
        "--quality-spatial-stride",
        type=int,
        default=4,
        help="spatial stride for decoded image quality evidence",
    )
    parser.add_argument(
        "--require-commit",
        default="",
        help="fail unless this checkout is clean and HEAD starts with this SHA",
    )
    parser.add_argument(
        "--regime",
        choices=BENCHMARK_RESIDENCY_REGIMES,
        default="open",
        help="the device as-is (default) or constrained behind a ballast allocation",
    )
    parser.add_argument(
        "--leave-free-mib",
        type=int,
        default=768,
        help="constrained regime: device MiB the ballast leaves free",
    )
    parser.add_argument(
        "--spill-scope",
        choices=("auto", "process", "machine", "off"),
        default="auto",
        help="GPU shared-usage counter scope (auto: process on Windows, machine on WSL, else off)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    arguments = parser.parse_args(argv)
    if arguments.leave_free_mib <= 0:
        parser.error("--leave-free-mib must be positive")
    if arguments.quality_spatial_stride < 1:
        parser.error("--quality-spatial-stride must be positive")
    if arguments.family != "minimax_h3":
        if arguments.attention_policy != "auto":
            parser.error("--attention-policy is currently supported only with MiniMax H3")
        if arguments.quality_output_dir is not None:
            parser.error("--quality-output-dir is currently supported only with MiniMax H3")
    if arguments.fallback_768:
        if arguments.family != _ANIMA_FAMILY:
            parser.error("--fallback-768 is only meaningful with --family anima")
        for name in ("width", "height"):
            supplied = getattr(arguments, name)
            if supplied not in (None, 768):
                parser.error(f"--{name} must be 768 with --fallback-768")
            setattr(arguments, name, 768)
    defaults = {**_SD_ERA_DEFAULTS, **_WORKLOAD_DEFAULTS.get(arguments.family, {})}
    for name, value in defaults.items():
        if getattr(arguments, name) is None:
            setattr(arguments, name, value)
    for name in ("cfg", "lora_strength_model", "lora_strength_clip", "audio_scale"):
        if not math.isfinite(getattr(arguments, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if arguments.warm_runs < 1:
        parser.error("--warm-runs must be at least 1")
    if arguments.family in _SPLIT_ARTIFACT_FAMILIES:
        for name in ("diffusion", "text_encoder", "vae"):
            if getattr(arguments, name) is None:
                parser.error(f"{arguments.family} family requires --{name.replace('_', '-')}")
        if arguments.checkpoint is not None:
            parser.error(f"--checkpoint is not meaningful with --family {arguments.family}")
        if arguments.mode == "compile":
            parser.error(f"--mode compile is not supported for {arguments.family} cells")
    else:
        if arguments.checkpoint is None:
            parser.error(f"{arguments.family} family requires --checkpoint")
        for name in ("diffusion", "text_encoder", "vae"):
            if getattr(arguments, name) is not None:
                parser.error(f"--{name.replace('_', '-')} is only meaningful with split families")
    if arguments.family == _FLUX_FAMILY:
        if arguments.clip_l is None:
            parser.error("flux family requires --clip-l")
        if not math.isfinite(arguments.guidance):
            parser.error("--guidance must be finite")
    else:
        if arguments.clip_l is not None:
            parser.error("--clip-l is only meaningful with --family flux")
        if arguments.guidance is not None:
            parser.error("--guidance is only meaningful with --family flux")
    lora_families = ("lora", "wan21_infinitetalk", "wan21_humo")
    if arguments.family in lora_families and arguments.lora is None:
        parser.error(f"{arguments.family} family requires --lora")
    if arguments.family not in lora_families and arguments.lora is not None:
        parser.error("--lora is only meaningful with a LoRA workload")
    if arguments.family == "minimax_h3":
        if arguments.audio_vae is None:
            parser.error("minimax_h3 family requires --audio-vae")
    elif arguments.audio_vae is not None:
        parser.error("--audio-vae is only meaningful with MiniMax H3")
    shared_provider_arguments = ("audio_encoder", "input_image")
    infinitetalk_arguments = (
        "model_patch",
        "clip_vision",
        "input_audio_1",
        "input_audio_2",
    )
    if arguments.family == "wan21_infinitetalk":
        for name in (*shared_provider_arguments, *infinitetalk_arguments):
            if getattr(arguments, name) is None:
                parser.error(f"wan21_infinitetalk family requires --{name.replace('_', '-')}")
        if arguments.input_audio is not None:
            parser.error("--input-audio is only meaningful with HuMo")
        if not 1 <= arguments.motion_frame_count <= 33:
            parser.error("--motion-frame-count must be in [1, 33]")
        if arguments.audio_scale <= 0:
            parser.error("--audio-scale must be positive")
        if arguments.length < 1 or (arguments.length - 1) % 4 != 0:
            parser.error("--length must be a positive 4k+1 frame count (e.g. 33)")
        pinned = {
            **_WORKLOAD_DEFAULTS["wan21_infinitetalk"],
            "lora_strength_model": 1.0,
            "motion_frame_count": 9,
            "audio_scale": 1.0,
        }
        for name, expected in pinned.items():
            if getattr(arguments, name) != expected:
                parser.error(
                    f"--{name.replace('_', '-')} must be {expected!r} for wan21_infinitetalk"
                )
    elif arguments.family == "wan21_humo":
        for name in (*shared_provider_arguments, "input_audio"):
            if getattr(arguments, name) is None:
                parser.error(f"wan21_humo family requires --{name.replace('_', '-')}")
        for name in infinitetalk_arguments:
            if getattr(arguments, name) is not None:
                parser.error(f"--{name.replace('_', '-')} is only meaningful with InfiniteTalk")
        pinned = {
            **_WORKLOAD_DEFAULTS["wan21_humo"],
            "lora_strength_model": 1.0,
        }
        for name, expected in pinned.items():
            if getattr(arguments, name) != expected:
                parser.error(f"--{name.replace('_', '-')} must be {expected!r} for wan21_humo")
        if arguments.motion_frame_count != 9:
            parser.error("--motion-frame-count is only meaningful with InfiniteTalk")
        if arguments.audio_scale != 1.0:
            parser.error("--audio-scale is only meaningful with InfiniteTalk")
    elif arguments.family == _ANIMA_FAMILY:
        pinned = {
            name: expected
            for name, expected in _WORKLOAD_DEFAULTS[_ANIMA_FAMILY].items()
            if name not in ("width", "height")
        }
        for name, expected in pinned.items():
            if getattr(arguments, name) != expected:
                parser.error(f"--{name.replace('_', '-')} must be {expected!r} for anima")
        expected_geometry = (768, 768) if arguments.fallback_768 else (1024, 1024)
        if (arguments.width, arguments.height) != expected_geometry:
            parser.error("768x768 Anima geometry requires --fallback-768")
        if arguments.placement != "residency":
            parser.error("--placement direct-diagnostic is not supported for anima")
        for name in (*shared_provider_arguments, *infinitetalk_arguments, "input_audio"):
            if getattr(arguments, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} is only meaningful with an audio workload"
                )
    elif arguments.family == "minimax_h3":
        for name in (*shared_provider_arguments, *infinitetalk_arguments, "input_audio"):
            if getattr(arguments, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} is only meaningful with an audio input workload"
                )
        for name, expected in _WORKLOAD_DEFAULTS["minimax_h3"].items():
            if getattr(arguments, name) != expected:
                parser.error(f"--{name.replace('_', '-')} must be {expected!r} for minimax_h3")
        if arguments.motion_frame_count != 9:
            parser.error("--motion-frame-count is only meaningful with InfiniteTalk")
        if arguments.audio_scale != 1.0:
            parser.error("--audio-scale is only meaningful with InfiniteTalk")
    elif arguments.family == _FLUX_FAMILY:
        for name in (*shared_provider_arguments, *infinitetalk_arguments, "input_audio"):
            if getattr(arguments, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} is only meaningful with an audio workload"
                )
        pinned = {**_SD_ERA_DEFAULTS, **_WORKLOAD_DEFAULTS[_FLUX_FAMILY]}
        for name, expected in pinned.items():
            if getattr(arguments, name) != expected:
                parser.error(f"--{name.replace('_', '-')} must be {expected!r} for flux")
        if arguments.motion_frame_count != 9:
            parser.error("--motion-frame-count is only meaningful with InfiniteTalk")
        if arguments.audio_scale != 1.0:
            parser.error("--audio-scale is only meaningful with InfiniteTalk")
        if arguments.placement != "residency":
            parser.error("--placement direct-diagnostic is not supported for flux")
    elif arguments.family == _CHROMA_FAMILY:
        for name in (*shared_provider_arguments, *infinitetalk_arguments, "input_audio"):
            if getattr(arguments, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} is only meaningful with an audio workload"
                )
        pinned = {**_SD_ERA_DEFAULTS, **_WORKLOAD_DEFAULTS[_CHROMA_FAMILY]}
        for name, expected in pinned.items():
            if getattr(arguments, name) != expected:
                parser.error(f"--{name.replace('_', '-')} must be {expected!r} for chroma")
        if arguments.motion_frame_count != 9:
            parser.error("--motion-frame-count is only meaningful with InfiniteTalk")
        if arguments.audio_scale != 1.0:
            parser.error("--audio-scale is only meaningful with InfiniteTalk")
        if arguments.placement != "residency":
            parser.error("--placement direct-diagnostic is not supported for chroma")
    else:
        for name in (*shared_provider_arguments, *infinitetalk_arguments, "input_audio"):
            if getattr(arguments, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} is only meaningful with an audio workload"
                )
        if arguments.motion_frame_count != 9:
            parser.error("--motion-frame-count is only meaningful with InfiniteTalk")
        if arguments.audio_scale != 1.0:
            parser.error("--audio-scale is only meaningful with InfiniteTalk")
    if arguments.family in (_VIDEO_FAMILIES - {"minimax_h3"}):
        # The causal video VAE decodes 4*T - 3 frames from T latent
        # frames, so only 4k+1 requests decode to exactly that count.
        if arguments.length < 1 or (arguments.length - 1) % 4 != 0:
            parser.error("--length must be a positive 4k+1 frame count (e.g. 33)")
    elif arguments.family == "minimax_h3":
        if arguments.length < 5 or (arguments.length - 5) % 17 != 0:
            parser.error("--length must be a positive 17k+5 frame count (e.g. 124)")
    elif arguments.length is not None:
        parser.error("--length is only meaningful with a video family")
    if arguments.family in _PROVIDER_WAN_FAMILIES and arguments.placement != "residency":
        parser.error("--placement direct-diagnostic is not supported for provider workloads")
    if arguments.family == "minimax_h3" and arguments.placement != "residency":
        parser.error("--placement direct-diagnostic is not supported for MiniMax H3")
    return arguments


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def checkout_identity(root: Path, require_commit: str) -> dict[str, object]:
    commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain")
    clean = dirty == "" if dirty is not None else False
    if require_commit:
        if commit is None or not commit.startswith(require_commit):
            sys.exit(
                f"error: checkout HEAD {commit or '(unknown)'} does not start"
                f" with required commit {require_commit}"
            )
        if not clean:
            sys.exit(
                f"error: checkout {root} has local modifications or could not be inspected;"
                " the report would record a commit it does not measure"
            )
    return {"commit": commit or "", "clean": clean}


def main() -> int:
    arguments = _parse_arguments()
    source = checkout_identity(Path(__file__).resolve().parent.parent, arguments.require_commit)
    if arguments.quality_output_dir is not None:
        arguments.quality_output_dir = arguments.quality_output_dir.resolve()
        arguments.quality_output_dir.mkdir(parents=True, exist_ok=True)
        if any(arguments.quality_output_dir.iterdir()):
            sys.exit(f"error: --quality-output-dir must be empty: {arguments.quality_output_dir}")
    mechanism = os.environ.get("DINKSTER_AIMDO_ARM", "auto")
    if mechanism not in BENCHMARK_RESIDENCY_MECHANISMS:
        sys.exit(f"error: DINKSTER_AIMDO_ARM must be one of {BENCHMARK_RESIDENCY_MECHANISMS}")
    os.environ["DINKSTER_AIMDO_ARM"] = mechanism
    access = _admit_backend(arguments.backend)

    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"python:   {platform.python_version()}")
    print(f"torch:    {torch.__version__} ({access.backend_runtime})")
    driver = _driver_identity(arguments.backend)
    print(f"driver:   {driver}")
    print(f"cell:     {arguments.family} / {arguments.mode} on {arguments.backend}")
    print(f"placement request: {arguments.placement}")

    sampler = _ResidencySampler(arguments.spill_scope)
    print(f"residency: mechanism={mechanism} regime={arguments.regime} spill_scope={sampler.scope}")

    preflight_assets: dict[str, _AssetPreflight] = {}
    if arguments.family == "wan21_infinitetalk":
        artifact_paths = {
            "diffusion": arguments.diffusion,
            "text_encoder": arguments.text_encoder,
            "vae": arguments.vae,
            "lora": arguments.lora,
            "model_patch": arguments.model_patch,
            "audio_encoder": arguments.audio_encoder,
            "clip_vision": arguments.clip_vision,
            "input_image": arguments.input_image,
            "input_audio_1": arguments.input_audio_1,
            "input_audio_2": arguments.input_audio_2,
        }
        artifacts = []
        for role, path in artifact_paths.items():
            entry = _artifact_entry(
                role,
                path,
                _INFINITETALK_ARTIFACT_PINS[role],
                include_asset_preflight=True,
            )
            preflight_assets[role] = cast("_AssetPreflight", entry.pop("_asset_preflight"))
            artifacts.append(entry)
    elif arguments.family == "wan21_humo":
        artifact_paths = {
            "diffusion": arguments.diffusion,
            "text_encoder": arguments.text_encoder,
            "vae": arguments.vae,
            "lora": arguments.lora,
            "audio_encoder": arguments.audio_encoder,
            "input_image": arguments.input_image,
            "input_audio": arguments.input_audio,
        }
        artifacts = []
        for role, path in artifact_paths.items():
            entry = _artifact_entry(
                role,
                path,
                _HUMO_ARTIFACT_PINS[role],
                include_asset_preflight=True,
            )
            preflight_assets[role] = cast("_AssetPreflight", entry.pop("_asset_preflight"))
            artifacts.append(entry)
    elif arguments.family == _ANIMA_FAMILY:
        artifact_paths = {
            "diffusion": arguments.diffusion,
            "text_encoder": arguments.text_encoder,
            "vae": arguments.vae,
        }
        artifacts = []
        for role, path in artifact_paths.items():
            entry = _artifact_entry(
                role,
                path,
                _ANIMA_ARTIFACT_PINS[role],
                include_asset_preflight=True,
            )
            preflight_assets[role] = cast("_AssetPreflight", entry.pop("_asset_preflight"))
            artifacts.append(entry)
    elif arguments.family == _CHROMA_FAMILY:
        artifact_paths = {
            "diffusion": arguments.diffusion,
            "text_encoder": arguments.text_encoder,
            "vae": arguments.vae,
        }
        artifacts = []
        for role, path in artifact_paths.items():
            entry = _artifact_entry(
                role,
                path,
                _CHROMA_ARTIFACT_PINS[role],
                include_asset_preflight=True,
            )
            preflight_assets[role] = cast("_AssetPreflight", entry.pop("_asset_preflight"))
            artifacts.append(entry)
    elif arguments.family == "minimax_h3":
        artifact_paths = {
            "diffusion": arguments.diffusion,
            "text_encoder": arguments.text_encoder,
            "video_vae": arguments.vae,
            "audio_vae": arguments.audio_vae,
        }
        artifacts = [
            _artifact_entry(role, path, _MINIMAX_H3_ARTIFACT_PINS[role])
            for role, path in artifact_paths.items()
        ]
    elif arguments.family == _FLUX_FAMILY:
        artifact_paths = {
            "diffusion": arguments.diffusion,
            "clip_l": arguments.clip_l,
            "text_encoder": arguments.text_encoder,
            "vae": arguments.vae,
        }
        artifacts = []
        for role, path in artifact_paths.items():
            entry = _artifact_entry(
                role,
                path,
                _FLUX_ARTIFACT_PINS[role],
                include_asset_preflight=True,
            )
            preflight_assets[role] = cast("_AssetPreflight", entry.pop("_asset_preflight"))
            artifacts.append(entry)
    elif arguments.family in _SPLIT_TEXT_ENCODER_SLOTS:
        artifacts = [
            _artifact_entry("diffusion", arguments.diffusion),
            _artifact_entry("text_encoder", arguments.text_encoder),
            _artifact_entry("vae", arguments.vae),
        ]
    else:
        artifacts = [_artifact_entry("checkpoint", arguments.checkpoint)]
        if arguments.family == "lora":
            artifacts.append(_artifact_entry("lora", arguments.lora))

    start = time.perf_counter()
    import dinkster_inference  # noqa: F401
    import dinkster_inference_torch  # noqa: F401

    if arguments.family in {
        *_PROVIDER_WAN_FAMILIES,
        _ANIMA_FAMILY,
        _CHROMA_FAMILY,
        _FLUX_FAMILY,
        "minimax_h3",
    }:
        import dinkster_assets  # noqa: F401
        import dinkster_compat_comfy.native_arm  # noqa: F401
        import dinkster_values  # noqa: F401

    if arguments.family in _PROVIDER_WAN_FAMILIES:
        import av  # noqa: F401
        import dinkster_model_wan.provider  # noqa: F401

    import_s = time.perf_counter() - start
    print(f"imports:  {import_s:.6g}s")

    # Synchronize first: resetting peak stats needs an initialized device
    # context, and nothing has touched the device yet.
    access.synchronize()

    ballast: list[Any] = []
    ballast_bytes: int | None = None
    if arguments.regime == "constrained":
        if access.device.type != "cuda":
            sys.exit("error: the constrained regime requires a CUDA-family device")
        free, _total = torch.cuda.mem_get_info(access.device)
        remaining = _ballast_size_bytes(int(free), arguments.leave_free_mib)
        ballast_bytes = remaining
        while remaining > 0:
            chunk = min(remaining, _BALLAST_CHUNK_BYTES)
            ballast.append(torch.empty(chunk, dtype=torch.uint8, device=access.device))
            remaining -= chunk
        print(f"ballast:  {ballast_bytes} bytes (leaving {arguments.leave_free_mib} MiB free)")

    access.reset_peak()
    shared_before = sampler.sample()
    run = BenchmarkRun(arguments, access, preflight_assets)
    print("checks:")
    run.record("load", run.load)
    if arguments.family == "lora":
        run.record("lora_apply", run.lora_apply)
    run.record("encode_text", run.encode_text)
    if arguments.family in _PROVIDER_WAN_FAMILIES:
        run.record("encode_audio", run.encode_audio)
    run.record("cold_run", run.cold_run)
    shared_warm = sampler.sample()
    run.record("finite_output", run.finite_output)
    run.record("warm_runs", run.warm_runs)
    peak_allocated = access.peak_allocated()
    peak_reserved = access.peak_reserved()
    peak_rss = _peak_rss_bytes()
    shared_after = sampler.sample()
    if arguments.quality_output_dir is not None:
        run.record("quality_capture", run.capture_quality)
    if ballast:
        ballast.clear()
        access.empty_cache()
    run.record("unload", run.unload, always=True)

    all_ok = all(bool(entry["ok"]) for entry in run.checks.values())
    report: dict[str, object] = {
        "report_version": BENCHMARK_REPORT_VERSION,
        "system": "dinkster",
        "accelerator": arguments.backend,
        "host": {
            "platform": platform.platform(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "driver": driver,
        "torch": {"version": str(torch.__version__), "backend_runtime": access.backend_runtime},
        "devices": _device_entries(arguments.backend),
        "dinkster": source,
        "family": arguments.family,
        "mode": arguments.mode,
        "placement": run.executed_placement,
        **(
            {
                "variant": BENCHMARK_ANIMA_FALLBACK_VARIANT
                if arguments.fallback_768
                else BENCHMARK_PRIMARY_VARIANT
            }
            if arguments.family == _ANIMA_FAMILY
            else {}
        ),
        **({"execution_path": run.execution_path} if arguments.family == "minimax_h3" else {}),
        "family_id": run.family_id,
        "workload": {
            "prompt": arguments.prompt,
            "negative_prompt": arguments.negative_prompt,
            "sampler_id": arguments.sampler,
            "scheduler_id": arguments.scheduler,
            "seed": arguments.seed,
            "steps": arguments.steps,
            "width": arguments.width,
            "height": arguments.height,
            "length": arguments.length if arguments.family in _VIDEO_FAMILIES else None,
            "cfg": arguments.cfg,
            "guidance": arguments.guidance if arguments.family == _FLUX_FAMILY else None,
            "warm_runs": arguments.warm_runs,
            "lora_strength_model": arguments.lora_strength_model
            if arguments.family in ("lora", "wan21_infinitetalk", "wan21_humo")
            else None,
            "lora_strength_clip": arguments.lora_strength_clip
            if arguments.family == "lora"
            else None,
            "motion_frame_count": arguments.motion_frame_count
            if arguments.family == "wan21_infinitetalk"
            else None,
            "audio_scale": arguments.audio_scale
            if arguments.family == "wan21_infinitetalk"
            else None,
            "speaker_mask_layout": "left_right_half"
            if arguments.family == "wan21_infinitetalk"
            else None,
        },
        "artifacts": artifacts,
        "attention": getattr(run, "attention", {"requested_policy": "auto"}),
        **(
            {"quality_capture": run.quality_capture}
            if getattr(run, "quality_capture", None) is not None
            else {}
        ),
        "timings": {
            "import_s": import_s,
            "cold": run.cold,
            "warm": run.warm_section(),
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "residual_allocated_bytes": run.residual_allocated,
            "peak_rss_bytes": peak_rss,
        },
        "residency": _residency_section(
            mechanism=mechanism,
            regime=arguments.regime,
            leave_free_mib=arguments.leave_free_mib if arguments.regime == "constrained" else None,
            ballast_bytes=ballast_bytes,
            spill_scope=sampler.scope,
            shared_before_bytes=shared_before,
            shared_warm_bytes=shared_warm,
            shared_after_bytes=shared_after,
            aimdo_bootstrap_succeeded=_AIMDO_BOOTSTRAP_SUCCEEDED,
            routes=getattr(run, "residency_routes", {}),
        ),
        "checks": run.checks,
        "all_ok": all_ok,
    }

    problems = validate_benchmark_report(
        report,
        accelerator=arguments.backend,
        canonical_evidence=arguments.placement == "residency",
    )
    for problem in problems:
        print(f"error: incomplete report: {problem}", file=sys.stderr)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report written: {arguments.json}")

    if not all_ok:
        print("error: a benchmark check failed", file=sys.stderr)
        return 1
    if problems:
        return 1
    print(f"benchmark cell {arguments.family}/{arguments.mode} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
