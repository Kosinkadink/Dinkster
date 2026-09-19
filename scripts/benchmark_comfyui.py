"""Measure one ComfyUI inference benchmark cell on real hardware.

For exported workflows, pass --workflow (see tools/workflow_benchmark.py).
That HTTP contract accepts a caller-selected reference revision and does not
route by family labels. The per-family cell format below is historical.

The ComfyUI half of the Dinkster-vs-ComfyUI head-to-head comparison
(issue #667): one invocation launches a fresh headless ComfyUI server
from a pinned checkout, runs one cell against it over HTTP, and writes
the same JSON evidence schema
dinkster_workers.backend_env.validate_benchmark_report checks, with
system="comfyui":

    .venv/bin/python scripts/benchmark_comfyui.py \\
        --backend rocm --family sd15 --checkpoint /models/sd15.safetensors \\
        --comfyui-root /work/ComfyUI \\
        --comfyui-python /work/ComfyUI/.venv/bin/python \\
        --json sd15-comfyui-bench.json

This driver runs under a Dinkster environment (it needs dinkster_workers for
report validation and never imports torch); the server runs under the
ComfyUI checkout's own venv. Instrumentation comes from the custom node
package in scripts/comfyui_benchmark_nodes/, injected through a
generated extra_model_paths config so the pinned checkout itself is
never modified. The graph is ComfyUI's canonical text-to-image path:
CheckpointLoaderSimple (plus LoraLoader for the lora family),
CLIPTextEncode for cond and uncond, EmptyLatentImage, KSampler,
VAEDecode, and the shim's no-output sink. The split-source families
zimage and wan21 (--diffusion, --text-encoder, --vae; wan21 adds
--length frames) instead load through UNETLoader, CLIPLoader, and
VAELoader, patch the schedule shift with the official template's
ModelSampling node, and build the latent with the template's empty
latent node; their workload defaults match the official ComfyUI
workflow templates. Per its template, zimage's negative input is
ConditioningZeroOut of the positive conditioning, not a second text
encode. InfiniteTalk adds the official model patch, dual Wav2Vec2 audio
conditioning, CLIP vision, image conditioning, and custom sampling path;
deterministic half-frame masks replace the template's interactive masks.
Wan 2.1 HuMo adds the official model-only LoRA, Whisper audio
conditioning, reference image conditioning, and HuMo latent builder.
MiniMax H3 adds the official non-turbo text-to-audio-video graph with
separate video and audio VAEs and the custom sampling nodes.

Phase timings come from the shim's server-side node-boundary
timestamps, which are device-synchronized, so they carry the same
synchronized-wall-clock contract as the Dinkster runner:

- cold: the first image in the fresh server process - load is the
  loader nodes' summed intervals (plus the ModelSampling patch for the
  split-source families), lora the LoraLoader interval (ComfyUI applies
  LoRA patches lazily inside the first sampler run, so cross-system
  lora comparisons use cold totals, not the lora phase alone), encode
  the sum of both text-encode intervals, sample the KSampler interval,
  decode the VAEDecode interval. ComfyUI also uploads weights to the
  device lazily inside the first encode/sample/decode, so cross-system
  cold comparisons use totals, not the per-phase split. Server boot
  (process start to HTTP ready) is recorded as import_s; ComfyUI pays
  its import cost at boot, outside its node phases.
- warm: --warm-runs resubmissions changing only the seed (seed+1+i), so
  ComfyUI's caching reuses the loaded model and encoded conditioning
  and only KSampler and VAEDecode re-execute - the same sample+decode
  warm scope the Dinkster runner measures. A warm run that re-executes a
  loader or encode node fails the cell instead of silently skewing the
  comparison.

Per-step boundaries reconstructed from the sampler's progress updates
are recorded as step_wall_ms when they resolve to exactly one boundary
per step; they are dispatch-side and informational, like the Dinkster
runner's. Compile mode does not exist here: the head-to-head is eager
against eager, and a comfyui report is eager by schema. Exits nonzero
when any check fails or the report is incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

if __name__ == "__main__" and any(
    arg == "--workflow" or arg.startswith("--workflow=") for arg in sys.argv[1:]
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.workflow_benchmark import main as workflow_main

    sys.exit(workflow_main("comfyui"))

from dinkster_workers.backend_env import (
    BENCHMARK_ACCELERATORS,
    BENCHMARK_ANIMA_FALLBACK_VARIANT,
    BENCHMARK_ANIMA_PROMPT,
    BENCHMARK_COMFYUI_COMMIT,
    BENCHMARK_COMFYUI_PLACEMENT,
    BENCHMARK_FAMILIES,
    BENCHMARK_MINIMAX_H3_COMFYUI_EXECUTION_PATH,
    BENCHMARK_PRIMARY_VARIANT,
    BENCHMARK_REPORT_VERSION,
    FAMILY_RESIDUAL_CEILING_BYTES,
    validate_benchmark_report,
)

#: The ComfyUI commit the head-to-head comparison is pinned to.
COMFYUI_PIN = BENCHMARK_COMFYUI_COMMIT

# Fixed node identifiers so phase extraction can name nodes exactly.
LOADER_NODE = "1"
LORA_NODE = "2"
POSITIVE_NODE = "3"
NEGATIVE_NODE = "4"
LATENT_NODE = "5"
SAMPLER_NODE = "6"
DECODE_NODE = "7"
SINK_NODE = "8"
CLIP_LOADER_NODE = "9"
VAE_LOADER_NODE = "10"
MODEL_SAMPLING_NODE = "11"
MODEL_PATCH_NODE = "12"
AUDIO_ENCODER_NODE = "13"
AUDIO_INPUT_1_NODE = "14"
AUDIO_INPUT_2_NODE = "15"
AUDIO_ENCODE_1_NODE = "16"
AUDIO_ENCODE_2_NODE = "17"
IMAGE_INPUT_NODE = "18"
MASK_BASE_NODE = "19"
MASK_HALF_NODE = "20"
MASK_1_NODE = "21"
MASK_2_NODE = "22"
CONDITIONING_NODE = "23"
NOISE_NODE = "24"
GUIDER_NODE = "25"
SAMPLER_SELECT_NODE = "26"
SCHEDULER_NODE = "27"
CLIP_VISION_LOADER_NODE = "28"
CLIP_VISION_ENCODE_NODE = "29"
AUDIO_VAE_LOADER_NODE = "30"
AUDIO_DECODE_NODE = "31"
FLUX_GUIDANCE_NODE = "32"
T5_TOKENIZER_NODE = "33"

# HuMo has one audio lane and reuses the first fixed audio node IDs.
AUDIO_INPUT_NODE = AUDIO_INPUT_1_NODE
AUDIO_ENCODE_NODE = AUDIO_ENCODE_1_NODE

#: Nodes ComfyUI's caching must not re-execute during a warm run; a warm
#: entry that includes one measured more than sample+decode.
COLD_ONLY_NODES = (
    LOADER_NODE,
    LORA_NODE,
    POSITIVE_NODE,
    NEGATIVE_NODE,
    LATENT_NODE,
    CLIP_LOADER_NODE,
    VAE_LOADER_NODE,
    MODEL_SAMPLING_NODE,
    MODEL_PATCH_NODE,
    AUDIO_ENCODER_NODE,
    AUDIO_INPUT_1_NODE,
    AUDIO_INPUT_2_NODE,
    AUDIO_ENCODE_1_NODE,
    AUDIO_ENCODE_2_NODE,
    IMAGE_INPUT_NODE,
    MASK_BASE_NODE,
    MASK_HALF_NODE,
    MASK_1_NODE,
    MASK_2_NODE,
    CONDITIONING_NODE,
    GUIDER_NODE,
    SAMPLER_SELECT_NODE,
    SCHEDULER_NODE,
    CLIP_VISION_LOADER_NODE,
    CLIP_VISION_ENCODE_NODE,
    AUDIO_VAE_LOADER_NODE,
    FLUX_GUIDANCE_NODE,
    T5_TOKENIZER_NODE,
)

#: Split-source families' graph pieces: the CLIPLoader clip type, the
#: ModelSampling patch node and its shift, and the empty-latent node,
#: all from the official workflow templates.
_SPLIT_GRAPH: dict[str, tuple[str, str, float, str]] = {
    "zimage": ("lumina2", "ModelSamplingAuraFlow", 3.0, "EmptySD3LatentImage"),
    "wan21": ("wan", "ModelSamplingSD3", 8.0, "EmptyHunyuanLatentVideo"),
    "chroma": ("chroma", "ModelSamplingAuraFlow", 1.0, "EmptySD3LatentImage"),
}

#: Families whose official template zeroes the positive conditioning for
#: the negative input (ConditioningZeroOut) instead of encoding the
#: negative prompt.
_ZERO_NEGATIVE_FAMILIES = frozenset({"zimage", "wan21_infinitetalk"})
_AUDIO_FAMILIES = frozenset({"wan21_infinitetalk", "wan21_humo"})
_ANIMA_FAMILY = "anima"
_CHROMA_FAMILY = "chroma"
_FLUX_FAMILY = "flux"
_ATTENTION_FLAGS = {
    "sdpa": "--use-pytorch-cross-attention",
    "dinkster_kitchen_int8": "--use-ck-attention",
    "sage": "--use-sage-attention",
}
_ATTENTION_POLICIES = ("auto", *_ATTENTION_FLAGS)


def attention_identity_problem(identity: object, requested_policy: str) -> str | None:
    if not isinstance(identity, dict):
        return "attention identity is missing"
    if identity.get("requested_policy") != requested_policy:
        return "the reported requested policy differs"
    if identity.get("selected_policy") != requested_policy:
        return "the selected policy differs"
    versions = identity.get("provider_versions")
    if not isinstance(versions, list) or not versions:
        return "provider versions are missing"
    if any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(value, str) and value for value in pair)
        for pair in versions
    ):
        return "a provider version is incomplete"
    if requested_policy == "sage":
        module = identity.get("provider_module")
        if not isinstance(module, dict) or module.get("authenticated") is not True:
            return "the imported Sage module is not bound to its reported distribution"
        if [module.get("distribution"), module.get("version")] not in versions:
            return "the imported Sage module differs from the reported provider version"
    return None


def attention_execution_problem(execution: object, requested_policy: str) -> str | None:
    if not isinstance(execution, dict) or execution.get("policy") != requested_policy:
        return "attention execution proof is missing"
    names = (
        "selected_calls",
        "provider_attempts",
        "provider_successes",
        "provider_exceptions",
        "fallback_calls",
    )
    if any(type(execution.get(name)) is not int or execution[name] < 0 for name in names):
        return "attention execution counters are malformed"
    if execution["provider_attempts"] != (
        execution["provider_successes"] + execution["provider_exceptions"]
    ):
        return "attention provider attempt counters are inconsistent"
    if execution["provider_successes"] <= 0:
        return "the requested attention provider did not execute successfully"
    if requested_policy == "sage":
        if execution["selected_calls"] != (
            execution["provider_successes"] + execution["fallback_calls"]
        ):
            return "Sage execution and fallback counters are inconsistent"
    elif execution["fallback_calls"] or (
        execution["selected_calls"] != execution["provider_successes"]
    ):
        return "attention execution counters are inconsistent"
    return None


#: Per-family workload defaults; the zimage and wan21 rows are the
#: official ComfyUI template settings, matching the Dinkster runner's.
_WORKLOAD_DEFAULTS: dict[str, dict[str, Any]] = {
    "zimage": {
        "steps": 8,
        "cfg": 1.0,
        "width": 1024,
        "height": 1024,
        "sampler": "res_multistep",
        "scheduler": "simple",
    },
    "wan21": {
        "steps": 30,
        "cfg": 6.0,
        "width": 832,
        "height": 480,
        "length": 33,
        "sampler": "uni_pc",
        "scheduler": "simple",
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
        "sampler": "euler",
        "scheduler": "normal",
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
        "sampler": "uni_pc",
        "scheduler": "simple",
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
        "sampler": "er_sde",
        "scheduler": "simple",
        "warm_runs": 5,
    },
    "minimax_h3": {
        "prompt": "A red square centered on a black background.",
        "negative_prompt": "",
        "seed": 20_260_813,
        "steps": 20,
        "cfg": 1.0,
        "width": 1344,
        "height": 768,
        "length": 124,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "warm_runs": 3,
    },
    # FLUX.1-dev at the official template settings; the pinned checkout's
    # Flux sampling defaults already apply the flat shift 1.15 schedule,
    # so the graph carries no ModelSampling node.
    "flux": {
        "steps": 20,
        "cfg": 1.0,
        "width": 1024,
        "height": 1024,
        "sampler": "euler",
        "scheduler": "simple",
        "guidance": 3.5,
    },
    # Chroma1-HD at the official template settings; the graph carries
    # ModelSamplingAuraFlow shift 1.0 and T5TokenizerOptions with padding
    # pinned off, matching the Dinkster runner's native defaults.
    "chroma": {
        "steps": 26,
        "cfg": 3.5,
        "width": 1024,
        "height": 1024,
        "sampler": "euler",
        "scheduler": "beta",
    },
}
_SD_ERA_DEFAULTS: dict[str, Any] = {
    "prompt": "a photograph of an astronaut riding a horse",
    "negative_prompt": "",
    "seed": 667,
    "steps": 20,
    "cfg": 7.0,
    "width": 512,
    "height": 512,
    "sampler": "euler",
    "scheduler": "simple",
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

_SHIM_DIRECTORY = Path(__file__).resolve().parent / "comfyui_benchmark_nodes"


def speaker_mask_regions(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    """The two non-overlapping full-height speaker regions."""
    if width <= 0 or width % 2 or height <= 0:
        raise ValueError(
            "speaker mask dimensions require a positive even width and positive height"
        )
    half = width // 2
    return ((0, 0, half, height), (half, 0, half, height))


def _build_infinitetalk_graph(
    *,
    diffusion_name: str | None,
    text_encoder_name: str | None,
    vae_name: str | None,
    lora_name: str | None,
    model_patch_name: str | None,
    audio_encoder_name: str | None,
    clip_vision_name: str | None,
    input_image_name: str | None,
    input_audio_1_name: str | None,
    input_audio_2_name: str | None,
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    length: int | None,
    cfg: float,
    sampler: str,
    scheduler: str,
    lora_strength_model: float,
    motion_frame_count: int,
    audio_scale: float,
) -> dict[str, Any]:
    names = {
        "diffusion": diffusion_name,
        "text encoder": text_encoder_name,
        "VAE": vae_name,
        "LoRA": lora_name,
        "model patch": model_patch_name,
        "audio encoder": audio_encoder_name,
        "CLIP vision": clip_vision_name,
        "input image": input_image_name,
        "first input audio": input_audio_1_name,
        "second input audio": input_audio_2_name,
    }
    missing = [name for name, value in names.items() if value is None]
    if missing:
        raise ValueError(f"wan21_infinitetalk family requires {', '.join(missing)} names")
    if length is None:
        raise ValueError("wan21_infinitetalk family requires a length")
    first_region, second_region = speaker_mask_regions(width, height)
    half_width = first_region[2]
    graph: dict[str, Any] = {
        LOADER_NODE: {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_name, "weight_dtype": "default"},
        },
        LORA_NODE: {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": [LOADER_NODE, 0],
                "lora_name": lora_name,
                "strength_model": lora_strength_model,
            },
        },
        CLIP_LOADER_NODE: {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder_name, "type": "wan", "device": "default"},
        },
        VAE_LOADER_NODE: {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        MODEL_PATCH_NODE: {
            "class_type": "ModelPatchLoader",
            "inputs": {"name": model_patch_name},
        },
        AUDIO_ENCODER_NODE: {
            "class_type": "AudioEncoderLoader",
            "inputs": {"audio_encoder_name": audio_encoder_name},
        },
        CLIP_VISION_LOADER_NODE: {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": clip_vision_name},
        },
        AUDIO_INPUT_1_NODE: {
            "class_type": "LoadAudio",
            "inputs": {"audio": input_audio_1_name},
        },
        AUDIO_INPUT_2_NODE: {
            "class_type": "LoadAudio",
            "inputs": {"audio": input_audio_2_name},
        },
        AUDIO_ENCODE_1_NODE: {
            "class_type": "AudioEncoderEncode",
            "inputs": {
                "audio_encoder": [AUDIO_ENCODER_NODE, 0],
                "audio": [AUDIO_INPUT_1_NODE, 0],
            },
        },
        AUDIO_ENCODE_2_NODE: {
            "class_type": "AudioEncoderEncode",
            "inputs": {
                "audio_encoder": [AUDIO_ENCODER_NODE, 0],
                "audio": [AUDIO_INPUT_2_NODE, 0],
            },
        },
        IMAGE_INPUT_NODE: {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name},
        },
        CLIP_VISION_ENCODE_NODE: {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": [CLIP_VISION_LOADER_NODE, 0],
                "image": [IMAGE_INPUT_NODE, 0],
                "crop": "center",
            },
        },
        MASK_BASE_NODE: {
            "class_type": "SolidMask",
            "inputs": {"value": 0.0, "width": width, "height": height},
        },
        MASK_HALF_NODE: {
            "class_type": "SolidMask",
            "inputs": {"value": 1.0, "width": half_width, "height": height},
        },
        MASK_1_NODE: {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": [MASK_BASE_NODE, 0],
                "source": [MASK_HALF_NODE, 0],
                "x": first_region[0],
                "y": first_region[1],
                "operation": "add",
            },
        },
        MASK_2_NODE: {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": [MASK_BASE_NODE, 0],
                "source": [MASK_HALF_NODE, 0],
                "x": second_region[0],
                "y": second_region[1],
                "operation": "add",
            },
        },
        POSITIVE_NODE: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": [CLIP_LOADER_NODE, 0]},
        },
        NEGATIVE_NODE: {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": [POSITIVE_NODE, 0]},
        },
        CONDITIONING_NODE: {
            "class_type": "WanInfiniteTalkToVideo",
            "inputs": {
                "mode": "two_speakers",
                "mode.audio_encoder_output_2": [AUDIO_ENCODE_2_NODE, 0],
                "mode.mask_1": [MASK_1_NODE, 0],
                "mode.mask_2": [MASK_2_NODE, 0],
                "model": [LORA_NODE, 0],
                "model_patch": [MODEL_PATCH_NODE, 0],
                "positive": [POSITIVE_NODE, 0],
                "negative": [NEGATIVE_NODE, 0],
                "vae": [VAE_LOADER_NODE, 0],
                "clip_vision_output": [CLIP_VISION_ENCODE_NODE, 0],
                "width": width,
                "height": height,
                "length": length,
                "audio_encoder_output_1": [AUDIO_ENCODE_1_NODE, 0],
                "motion_frame_count": motion_frame_count,
                "audio_scale": audio_scale,
                "start_image": [IMAGE_INPUT_NODE, 0],
            },
        },
        NOISE_NODE: {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        GUIDER_NODE: {
            "class_type": "CFGGuider",
            "inputs": {
                "model": [CONDITIONING_NODE, 0],
                "positive": [CONDITIONING_NODE, 1],
                "negative": [CONDITIONING_NODE, 2],
                "cfg": cfg,
            },
        },
        SAMPLER_SELECT_NODE: {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": sampler},
        },
        SCHEDULER_NODE: {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": [CONDITIONING_NODE, 0],
                "scheduler": scheduler,
                "steps": steps,
                "denoise": 1.0,
            },
        },
        SAMPLER_NODE: {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [NOISE_NODE, 0],
                "guider": [GUIDER_NODE, 0],
                "sampler": [SAMPLER_SELECT_NODE, 0],
                "sigmas": [SCHEDULER_NODE, 0],
                "latent_image": [CONDITIONING_NODE, 3],
            },
        },
        DECODE_NODE: {
            "class_type": "VAEDecode",
            "inputs": {"samples": [SAMPLER_NODE, 0], "vae": [VAE_LOADER_NODE, 0]},
        },
        SINK_NODE: {
            "class_type": "DinksterBenchmarkSink",
            "inputs": {"images": [DECODE_NODE, 0]},
        },
    }
    return graph


def _build_humo_graph(
    *,
    diffusion_name: str | None,
    text_encoder_name: str | None,
    vae_name: str | None,
    lora_name: str | None,
    audio_encoder_name: str | None,
    input_image_name: str | None,
    input_audio_name: str | None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    length: int | None,
    cfg: float,
    sampler: str,
    scheduler: str,
    lora_strength_model: float,
) -> dict[str, Any]:
    names = {
        "diffusion": diffusion_name,
        "text encoder": text_encoder_name,
        "VAE": vae_name,
        "LoRA": lora_name,
        "audio encoder": audio_encoder_name,
        "input image": input_image_name,
        "input audio": input_audio_name,
    }
    missing = [name for name, value in names.items() if value is None]
    if missing:
        raise ValueError(f"wan21_humo family requires {', '.join(missing)} names")
    if length is None:
        raise ValueError("wan21_humo family requires a length")
    return {
        LOADER_NODE: {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_name, "weight_dtype": "default"},
        },
        LORA_NODE: {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": [LOADER_NODE, 0],
                "lora_name": lora_name,
                "strength_model": lora_strength_model,
            },
        },
        CLIP_LOADER_NODE: {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder_name, "type": "wan", "device": "default"},
        },
        VAE_LOADER_NODE: {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        MODEL_SAMPLING_NODE: {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": [LORA_NODE, 0], "shift": 8.0},
        },
        AUDIO_ENCODER_NODE: {
            "class_type": "AudioEncoderLoader",
            "inputs": {"audio_encoder_name": audio_encoder_name},
        },
        AUDIO_INPUT_NODE: {
            "class_type": "LoadAudio",
            "inputs": {"audio": input_audio_name},
        },
        AUDIO_ENCODE_NODE: {
            "class_type": "AudioEncoderEncode",
            "inputs": {
                "audio_encoder": [AUDIO_ENCODER_NODE, 0],
                "audio": [AUDIO_INPUT_NODE, 0],
            },
        },
        IMAGE_INPUT_NODE: {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name},
        },
        POSITIVE_NODE: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": [CLIP_LOADER_NODE, 0]},
        },
        NEGATIVE_NODE: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": [CLIP_LOADER_NODE, 0]},
        },
        LATENT_NODE: {
            "class_type": "WanHuMoImageToVideo",
            "inputs": {
                "positive": [POSITIVE_NODE, 0],
                "negative": [NEGATIVE_NODE, 0],
                "vae": [VAE_LOADER_NODE, 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "audio_encoder_output": [AUDIO_ENCODE_NODE, 0],
                "ref_image": [IMAGE_INPUT_NODE, 0],
            },
        },
        SAMPLER_NODE: {
            "class_type": "KSampler",
            "inputs": {
                "model": [MODEL_SAMPLING_NODE, 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "positive": [LATENT_NODE, 0],
                "negative": [LATENT_NODE, 1],
                "latent_image": [LATENT_NODE, 2],
                "denoise": 1.0,
            },
        },
        DECODE_NODE: {
            "class_type": "VAEDecode",
            "inputs": {"samples": [SAMPLER_NODE, 0], "vae": [VAE_LOADER_NODE, 0]},
        },
        SINK_NODE: {
            "class_type": "DinksterBenchmarkSink",
            "inputs": {"images": [DECODE_NODE, 0]},
        },
    }


def _build_minimax_h3_graph(
    *,
    diffusion_name: str | None,
    text_encoder_name: str | None,
    video_vae_name: str | None,
    audio_vae_name: str | None,
    prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    length: int | None,
    sampler: str,
    scheduler: str,
) -> dict[str, Any]:
    names = {
        "diffusion": diffusion_name,
        "text encoder": text_encoder_name,
        "video VAE": video_vae_name,
        "audio VAE": audio_vae_name,
    }
    missing = [name for name, value in names.items() if value is None]
    if missing:
        raise ValueError(f"minimax_h3 family requires {', '.join(missing)} names")
    if length is None:
        raise ValueError("minimax_h3 family requires a length")
    return {
        LOADER_NODE: {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_name, "weight_dtype": "default"},
        },
        CLIP_LOADER_NODE: {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder_name,
                "type": "minimax",
                "device": "default",
            },
        },
        VAE_LOADER_NODE: {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae_name},
        },
        AUDIO_VAE_LOADER_NODE: {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae_name},
        },
        CONDITIONING_NODE: {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": [CLIP_LOADER_NODE, 0],
                "vae": [VAE_LOADER_NODE, 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        NOISE_NODE: {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        GUIDER_NODE: {
            "class_type": "BasicGuider",
            "inputs": {
                "model": [LOADER_NODE, 0],
                "conditioning": [CONDITIONING_NODE, 0],
            },
        },
        SAMPLER_SELECT_NODE: {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": sampler},
        },
        SCHEDULER_NODE: {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": [LOADER_NODE, 0],
                "scheduler": scheduler,
                "steps": steps,
                "denoise": 1.0,
            },
        },
        SAMPLER_NODE: {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [NOISE_NODE, 0],
                "guider": [GUIDER_NODE, 0],
                "sampler": [SAMPLER_SELECT_NODE, 0],
                "sigmas": [SCHEDULER_NODE, 0],
                "latent_image": [CONDITIONING_NODE, 1],
            },
        },
        DECODE_NODE: {
            "class_type": "VAEDecode",
            "inputs": {"samples": [SAMPLER_NODE, 0], "vae": [VAE_LOADER_NODE, 0]},
        },
        AUDIO_DECODE_NODE: {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": [SAMPLER_NODE, 0],
                "vae": [AUDIO_VAE_LOADER_NODE, 0],
            },
        },
        SINK_NODE: {
            "class_type": "DinksterBenchmarkSink",
            "inputs": {
                "images": [DECODE_NODE, 0],
                "audio": [AUDIO_DECODE_NODE, 0],
            },
        },
    }


def build_graph(
    family: str,
    *,
    ckpt_name: str | None = None,
    diffusion_name: str | None = None,
    text_encoder_name: str | None = None,
    clip_l_name: str | None = None,
    vae_name: str | None = None,
    audio_vae_name: str | None = None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    length: int | None = None,
    cfg: float,
    sampler: str,
    scheduler: str,
    lora_name: str | None = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 1.0,
    model_patch_name: str | None = None,
    audio_encoder_name: str | None = None,
    clip_vision_name: str | None = None,
    input_image_name: str | None = None,
    input_audio_1_name: str | None = None,
    input_audio_2_name: str | None = None,
    input_audio_name: str | None = None,
    motion_frame_count: int = 9,
    audio_scale: float = 1.0,
    guidance: float | None = None,
) -> dict[str, Any]:
    """The ComfyUI API-format prompt graph for one cell."""
    if family == "minimax_h3":
        return _build_minimax_h3_graph(
            diffusion_name=diffusion_name,
            text_encoder_name=text_encoder_name,
            video_vae_name=vae_name,
            audio_vae_name=audio_vae_name,
            prompt=prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            length=length,
            sampler=sampler,
            scheduler=scheduler,
        )
    if family == "wan21_infinitetalk":
        return _build_infinitetalk_graph(
            diffusion_name=diffusion_name,
            text_encoder_name=text_encoder_name,
            vae_name=vae_name,
            lora_name=lora_name,
            model_patch_name=model_patch_name,
            audio_encoder_name=audio_encoder_name,
            clip_vision_name=clip_vision_name,
            input_image_name=input_image_name,
            input_audio_1_name=input_audio_1_name,
            input_audio_2_name=input_audio_2_name,
            prompt=prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            length=length,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            lora_strength_model=lora_strength_model,
            motion_frame_count=motion_frame_count,
            audio_scale=audio_scale,
        )
    if family == "wan21_humo":
        return _build_humo_graph(
            diffusion_name=diffusion_name,
            text_encoder_name=text_encoder_name,
            vae_name=vae_name,
            lora_name=lora_name,
            audio_encoder_name=audio_encoder_name,
            input_image_name=input_image_name,
            input_audio_name=input_audio_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            length=length,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            lora_strength_model=lora_strength_model,
        )
    graph: dict[str, Any]
    model_source: list[Any]
    clip_source: list[Any]
    vae_source: list[Any]
    if family == _ANIMA_FAMILY:
        if diffusion_name is None or text_encoder_name is None or vae_name is None:
            raise ValueError("anima family requires diffusion, text encoder, and VAE names")
        graph = {
            LOADER_NODE: {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": diffusion_name, "weight_dtype": "default"},
            },
            CLIP_LOADER_NODE: {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": text_encoder_name,
                    "type": "stable_diffusion",
                    "device": "default",
                },
            },
            VAE_LOADER_NODE: {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_name},
            },
            LATENT_NODE: {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
        }
        model_source = [LOADER_NODE, 0]
        clip_source = [CLIP_LOADER_NODE, 0]
        vae_source = [VAE_LOADER_NODE, 0]
    elif family == _FLUX_FAMILY:
        if (
            diffusion_name is None
            or clip_l_name is None
            or text_encoder_name is None
            or vae_name is None
        ):
            raise ValueError("flux family requires diffusion, clip_l, text encoder, and VAE names")
        graph = {
            LOADER_NODE: {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": diffusion_name, "weight_dtype": "default"},
            },
            CLIP_LOADER_NODE: {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": clip_l_name,
                    "clip_name2": text_encoder_name,
                    "type": "flux",
                },
            },
            VAE_LOADER_NODE: {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_name},
            },
            LATENT_NODE: {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
        }
        model_source = [LOADER_NODE, 0]
        clip_source = [CLIP_LOADER_NODE, 0]
        vae_source = [VAE_LOADER_NODE, 0]
    elif family in _SPLIT_GRAPH:
        clip_type, model_sampling_class, shift, latent_class = _SPLIT_GRAPH[family]
        if diffusion_name is None or text_encoder_name is None or vae_name is None:
            raise ValueError(f"{family} family requires diffusion, text encoder, and VAE names")
        graph = {
            LOADER_NODE: {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": diffusion_name, "weight_dtype": "default"},
            },
            CLIP_LOADER_NODE: {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": text_encoder_name, "type": clip_type},
            },
            VAE_LOADER_NODE: {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_name},
            },
            MODEL_SAMPLING_NODE: {
                "class_type": model_sampling_class,
                "inputs": {"model": [LOADER_NODE, 0], "shift": shift},
            },
        }
        model_source = [MODEL_SAMPLING_NODE, 0]
        clip_source = [CLIP_LOADER_NODE, 0]
        vae_source = [VAE_LOADER_NODE, 0]
        if family == _CHROMA_FAMILY:
            # The official template pins T5 padding off; both encodes read
            # the options-applied clip.
            graph[T5_TOKENIZER_NODE] = {
                "class_type": "T5TokenizerOptions",
                "inputs": {
                    "clip": [CLIP_LOADER_NODE, 0],
                    "min_padding": 0,
                    "min_length": 0,
                },
            }
            clip_source = [T5_TOKENIZER_NODE, 0]
        latent_inputs: dict[str, Any] = {"width": width, "height": height, "batch_size": 1}
        if family == "wan21":
            if length is None:
                raise ValueError("wan21 family requires a length")
            latent_inputs["length"] = length
        graph[LATENT_NODE] = {"class_type": latent_class, "inputs": latent_inputs}
    else:
        if ckpt_name is None:
            raise ValueError(f"{family} family requires a ckpt_name")
        graph = {
            LOADER_NODE: {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ckpt_name},
            }
        }
        model_source = [LOADER_NODE, 0]
        clip_source = [LOADER_NODE, 1]
        vae_source = [LOADER_NODE, 2]
        if family == "lora":
            if lora_name is None:
                raise ValueError("lora family requires a lora_name")
            graph[LORA_NODE] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": model_source,
                    "clip": clip_source,
                    "lora_name": lora_name,
                    "strength_model": lora_strength_model,
                    "strength_clip": lora_strength_clip,
                },
            }
            model_source = [LORA_NODE, 0]
            clip_source = [LORA_NODE, 1]
        graph[LATENT_NODE] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
    graph[POSITIVE_NODE] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": clip_source},
    }
    if family in _ZERO_NEGATIVE_FAMILIES:
        graph[NEGATIVE_NODE] = {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": [POSITIVE_NODE, 0]},
        }
    else:
        graph[NEGATIVE_NODE] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": clip_source},
        }
    positive_source = [POSITIVE_NODE, 0]
    if family == _FLUX_FAMILY:
        if guidance is None:
            raise ValueError("flux family requires a guidance value")
        graph[FLUX_GUIDANCE_NODE] = {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": [POSITIVE_NODE, 0], "guidance": guidance},
        }
        positive_source = [FLUX_GUIDANCE_NODE, 0]
    graph[SAMPLER_NODE] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_source,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "positive": positive_source,
            "negative": [NEGATIVE_NODE, 0],
            "latent_image": [LATENT_NODE, 0],
            "denoise": 1.0,
        },
    }
    graph[DECODE_NODE] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [SAMPLER_NODE, 0], "vae": vae_source},
    }
    graph[SINK_NODE] = {
        "class_type": "DinksterBenchmarkSink",
        "inputs": {"images": [DECODE_NODE, 0]},
    }
    return graph


def prompt_events(events: list[dict[str, Any]], prompt_id: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("prompt_id") == prompt_id]


def node_intervals(events: list[dict[str, Any]], prompt_id: str) -> dict[str, float]:
    """Seconds each executed node spent, from the shim's synchronized
    start and finish timestamps.

    A cached node records only a finish and gets no interval, which is
    how a warm run proves the cold-only nodes were reused. A node that
    starts without finishing (the execution loop skips finish on
    failure) fails the extraction rather than producing partial phases.
    """
    starts: dict[str, float] = {}
    intervals: dict[str, float] = {}
    for event in prompt_events(events, prompt_id):
        kind = event.get("event")
        node = event.get("node")
        if node is None:
            continue
        node = str(node)
        if kind == "node_start":
            starts[node] = float(event["t"])
        elif kind == "node_finish":
            start = starts.pop(node, None)
            if start is not None:
                intervals[node] = intervals.get(node, 0.0) + (float(event["t"]) - start)
    if starts:
        raise RuntimeError(f"nodes {sorted(starts)} started but never finished")
    return intervals


def cold_phases(intervals: dict[str, float], family: str) -> dict[str, Any]:
    """Map cold-run node intervals onto the report's cold phase names."""
    load_nodes: tuple[str, ...] = (LOADER_NODE,)
    if family in (_ANIMA_FAMILY, _FLUX_FAMILY):
        load_nodes = (LOADER_NODE, CLIP_LOADER_NODE, VAE_LOADER_NODE)
    elif family in _SPLIT_GRAPH:
        # The ModelSampling patch belongs to load: the Dinkster runner's
        # load covers the equivalent family-default schedule setup.
        load_nodes = (LOADER_NODE, CLIP_LOADER_NODE, VAE_LOADER_NODE, MODEL_SAMPLING_NODE)
    encode_nodes: tuple[str, ...] = (POSITIVE_NODE, NEGATIVE_NODE)
    if family == _FLUX_FAMILY:
        encode_nodes = (POSITIVE_NODE, NEGATIVE_NODE, FLUX_GUIDANCE_NODE)
    elif family == _CHROMA_FAMILY:
        encode_nodes = (T5_TOKENIZER_NODE, POSITIVE_NODE, NEGATIVE_NODE)
    members: dict[str, tuple[str, ...]] = {
        "load_s": load_nodes,
        "encode_s": encode_nodes,
        "sample_s": (SAMPLER_NODE,),
        "decode_s": (DECODE_NODE,),
    }
    if family == "wan21_infinitetalk":
        members = {
            "load_s": (
                LOADER_NODE,
                LORA_NODE,
                CLIP_LOADER_NODE,
                VAE_LOADER_NODE,
                MODEL_PATCH_NODE,
                AUDIO_ENCODER_NODE,
                CLIP_VISION_LOADER_NODE,
            ),
            "encode_s": (POSITIVE_NODE, NEGATIVE_NODE),
            "audio_encode_s": (
                AUDIO_INPUT_1_NODE,
                AUDIO_INPUT_2_NODE,
                AUDIO_ENCODE_1_NODE,
                AUDIO_ENCODE_2_NODE,
                IMAGE_INPUT_NODE,
                CLIP_VISION_ENCODE_NODE,
                MASK_BASE_NODE,
                MASK_HALF_NODE,
                MASK_1_NODE,
                MASK_2_NODE,
                CONDITIONING_NODE,
            ),
            "sample_s": (
                GUIDER_NODE,
                SAMPLER_SELECT_NODE,
                SCHEDULER_NODE,
                NOISE_NODE,
                SAMPLER_NODE,
            ),
            "decode_s": (DECODE_NODE,),
        }
    if family == "wan21_humo":
        members = {
            "load_s": (
                LOADER_NODE,
                LORA_NODE,
                CLIP_LOADER_NODE,
                VAE_LOADER_NODE,
                MODEL_SAMPLING_NODE,
                AUDIO_ENCODER_NODE,
            ),
            "encode_s": (POSITIVE_NODE, NEGATIVE_NODE),
            "audio_encode_s": (
                AUDIO_INPUT_NODE,
                AUDIO_ENCODE_NODE,
                IMAGE_INPUT_NODE,
                LATENT_NODE,
            ),
            "sample_s": (SAMPLER_NODE,),
            "decode_s": (DECODE_NODE,),
        }
    if family == "minimax_h3":
        members = {
            "load_s": (
                LOADER_NODE,
                CLIP_LOADER_NODE,
                VAE_LOADER_NODE,
                AUDIO_VAE_LOADER_NODE,
            ),
            "encode_s": (CONDITIONING_NODE,),
            "sample_s": (
                NOISE_NODE,
                GUIDER_NODE,
                SAMPLER_SELECT_NODE,
                SCHEDULER_NODE,
                SAMPLER_NODE,
            ),
            "decode_s": (DECODE_NODE, AUDIO_DECODE_NODE),
        }
    if family == "lora":
        members["lora_s"] = (LORA_NODE,)
    phases: dict[str, Any] = {}
    total = 0.0
    for name, nodes in members.items():
        missing = [node for node in nodes if node not in intervals]
        if missing:
            raise RuntimeError(f"{name} nodes {missing} recorded no executing boundary")
        seconds = sum(intervals[node] for node in nodes)
        phases[name] = round(seconds, 4)
        total += seconds
    phases["total_s"] = round(total, 4)
    return phases


def warm_phases(intervals: dict[str, float], family: str | None = None) -> dict[str, Any]:
    """Map warm-run node intervals onto a warm entry, refusing runs where
    caching failed to hold the sample+decode warm scope."""
    stale = sorted(set(intervals) & set(COLD_ONLY_NODES))
    if stale:
        raise RuntimeError(
            f"warm run re-executed cold-only nodes {stale};"
            " caching did not hold the sample+decode warm scope"
        )
    sample_nodes = (
        (NOISE_NODE, SAMPLER_NODE)
        if family in ("wan21_infinitetalk", "minimax_h3")
        else (SAMPLER_NODE,)
    )
    decode_nodes = (DECODE_NODE, AUDIO_DECODE_NODE) if family == "minimax_h3" else (DECODE_NODE,)
    for node in (*sample_nodes, *decode_nodes):
        if node not in intervals:
            raise RuntimeError(f"warm run recorded no executing boundary for node {node}")
    entry: dict[str, Any] = {
        "sample_s": round(sum(intervals[node] for node in sample_nodes), 4),
        "decode_s": round(sum(intervals[node] for node in decode_nodes), 4),
    }
    entry["total_s"] = round(float(entry["sample_s"]) + float(entry["decode_s"]), 4)
    return entry


def sampler_step_wall_ms(
    events: list[dict[str, Any]], prompt_id: str, steps: int
) -> list[float] | None:
    """Per-step boundaries reconstructed from the sampler's progress
    updates; None when they do not resolve to exactly one per step.

    The sampler node also reports model-weight-loading progress through
    the same channel with a different max, so only ticks whose max is
    the step count belong to sampling."""
    start: float | None = None
    ticks: list[float] = []
    last: float | None = None
    for event in prompt_events(events, prompt_id):
        kind = event.get("event")
        node = event.get("node")
        if kind == "node_start" and node == SAMPLER_NODE:
            start = float(event["t"])
            ticks = []
            last = None
        elif kind == "progress" and node == SAMPLER_NODE:
            value = float(event.get("value", 0.0))
            if float(event.get("max", 0.0)) != float(steps):
                continue
            if value <= 0 or (last is not None and value <= last):
                continue
            ticks.append(float(event["t"]))
            last = value
    if start is None or len(ticks) != steps:
        return None
    return [
        round((later - earlier) * 1000, 3)
        for earlier, later in zip([start, *ticks[:-1]], ticks, strict=False)
    ]


def extra_model_paths_config(
    checkpoint_directory: Path | None = None,
    lora_directory: Path | None = None,
    *,
    diffusion_directory: Path | None = None,
    text_encoder_directory: Path | None = None,
    clip_l_directory: Path | None = None,
    vae_directory: Path | None = None,
    audio_vae_directory: Path | None = None,
    model_patch_directory: Path | None = None,
    audio_encoder_directory: Path | None = None,
    clip_vision_directory: Path | None = None,
) -> str:
    """The extra_model_paths YAML that points the pinned checkout at the
    cell's artifacts and the instrumentation package."""

    def quoted(path: Path) -> str:
        return "'" + str(path).replace("'", "''") + "'"

    lines = ["dinkster_benchmark:"]
    if checkpoint_directory is not None:
        lines.append(f"  checkpoints: {quoted(checkpoint_directory)}")
    lines.append(f"  custom_nodes: {quoted(_SHIM_DIRECTORY)}")
    if lora_directory is not None:
        lines.append(f"  loras: {quoted(lora_directory)}")
    if diffusion_directory is not None:
        lines.append(f"  diffusion_models: {quoted(diffusion_directory)}")
    text_encoder_directories = list(
        dict.fromkeys(
            directory
            for directory in (text_encoder_directory, clip_l_directory)
            if directory is not None
        )
    )
    if len(text_encoder_directories) == 1:
        lines.append(f"  text_encoders: {quoted(text_encoder_directories[0])}")
    elif text_encoder_directories:
        lines.append("  text_encoders: |")
        lines.extend(f"    {directory}" for directory in text_encoder_directories)
    vae_directories = list(
        dict.fromkeys(
            directory for directory in (vae_directory, audio_vae_directory) if directory is not None
        )
    )
    if len(vae_directories) == 1:
        lines.append(f"  vae: {quoted(vae_directories[0])}")
    elif vae_directories:
        lines.append("  vae: |")
        lines.extend(f"    {directory}" for directory in vae_directories)
    if model_patch_directory is not None:
        lines.append(f"  model_patches: {quoted(model_patch_directory)}")
    if audio_encoder_directory is not None:
        lines.append(f"  audio_encoders: {quoted(audio_encoder_directory)}")
    if clip_vision_directory is not None:
        lines.append(f"  clip_vision: {quoted(clip_vision_directory)}")
    return "\n".join(lines) + "\n"


def workload_section(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt": arguments.prompt,
        "negative_prompt": arguments.negative_prompt,
        "sampler_id": arguments.sampler,
        "scheduler_id": arguments.scheduler,
        "seed": arguments.seed,
        "steps": arguments.steps,
        "width": arguments.width,
        "height": arguments.height,
        "length": arguments.length
        if arguments.family in ("wan21", "wan21_infinitetalk", "wan21_humo", "minimax_h3")
        else None,
        "cfg": arguments.cfg,
        "guidance": arguments.guidance if arguments.family == _FLUX_FAMILY else None,
        "warm_runs": arguments.warm_runs,
        "lora_strength_model": arguments.lora_strength_model
        if arguments.family in ("lora", "wan21_infinitetalk", "wan21_humo")
        else None,
        "lora_strength_clip": arguments.lora_strength_clip if arguments.family == "lora" else None,
        "motion_frame_count": arguments.motion_frame_count
        if arguments.family == "wan21_infinitetalk"
        else None,
        "audio_scale": arguments.audio_scale if arguments.family == "wan21_infinitetalk" else None,
        "speaker_mask_layout": "left_right_half"
        if arguments.family == "wan21_infinitetalk"
        else None,
    }


def warm_section(warm_entries: list[dict[str, Any]]) -> dict[str, Any]:
    section: dict[str, Any] = {"runs": warm_entries}
    for name in ("sample_s", "decode_s", "total_s"):
        if warm_entries:
            section[f"median_{name}"] = round(
                statistics.median(float(entry[name]) for entry in warm_entries), 4
            )
    return section


def assemble_report(
    *,
    backend: str,
    family: str,
    identity: dict[str, Any],
    comfyui_commit: str,
    workload: dict[str, Any],
    artifacts: list[dict[str, Any]],
    import_s: float,
    cold: dict[str, Any],
    warm_entries: list[dict[str, Any]],
    memory: dict[str, Any],
    residual_allocated: int,
    checks: dict[str, dict[str, Any]],
    variant: str | None = None,
    execution_path: str | None = None,
    attention: dict[str, Any] | None = None,
    quality_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_ok = all(bool(entry["ok"]) for entry in checks.values())
    return {
        "report_version": BENCHMARK_REPORT_VERSION,
        "system": "comfyui",
        "accelerator": backend,
        "host": identity.get("host"),
        "driver": identity.get("driver"),
        "torch": identity.get("torch"),
        "devices": identity.get("devices"),
        "comfyui": {
            "version": str(identity.get("comfyui_version") or ""),
            "commit": comfyui_commit,
            "clean": True,
        },
        "family": family,
        "mode": "eager",
        "placement": BENCHMARK_COMFYUI_PLACEMENT,
        **({"variant": variant} if variant is not None else {}),
        **({"execution_path": execution_path} if family == "minimax_h3" else {}),
        "family_id": f"comfyui.{family}",
        "workload": workload,
        "artifacts": artifacts,
        "attention": attention or {"requested_policy": "auto"},
        **({"quality_capture": quality_capture} if quality_capture is not None else {}),
        "timings": {
            "import_s": round(import_s, 4),
            "cold": cold,
            "warm": warm_section(warm_entries),
        },
        "memory": {
            "peak_allocated_bytes": int(memory.get("peak_allocated_bytes") or 0),
            "peak_reserved_bytes": int(memory.get("peak_reserved_bytes") or 0),
            "residual_allocated_bytes": residual_allocated,
            "peak_rss_bytes": memory.get("peak_rss_bytes"),
            # Informational, ComfyUI-only: device-global used-bytes peak
            # (mem_get_info), which also sees weights ComfyUI's dynamic
            # VRAM loading holds outside the caching allocator. The Dinkster
            # runner has no counterpart field.
            "peak_device_used_bytes": memory.get("peak_device_used_bytes"),
        },
        "checks": checks,
        "all_ok": all_ok,
    }


def _artifact_entry(
    role: str, path: Path, pin: tuple[int, str, str] | None = None
) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 22), b""):
            digest.update(chunk)
    actual_size = path.stat().st_size
    actual_sha256 = digest.hexdigest()
    if pin is not None:
        expected_size, expected_sha256, _url = pin
        if actual_size != expected_size or actual_sha256 != expected_sha256:
            raise ValueError(
                f"{role} artifact does not match its pin:"
                f" expected {expected_size} bytes/{expected_sha256},"
                f" got {actual_size} bytes/{actual_sha256}"
            )
    entry = {
        "role": role,
        "path": str(path),
        "sha256": actual_sha256,
        "bytes": actual_size,
    }
    if pin is not None:
        entry["url"] = pin[2]
    return entry


class ComfyServer:
    """One fresh headless ComfyUI process driven over HTTP."""

    def __init__(
        self,
        *,
        root: Path,
        python: Path,
        port: int,
        config_path: Path,
        log_path: Path,
        boot_nonce: str,
        input_directory: Path | None = None,
        attention_policy: str = "auto",
        quality_output_dir: Path | None = None,
        quality_spatial_stride: int = 4,
    ) -> None:
        self.root = root
        self.python = python
        self.port = port
        self.config_path = config_path
        self.log_path = log_path
        self.boot_nonce = boot_nonce
        self.input_directory = input_directory
        self.attention_policy = attention_policy
        self.quality_output_dir = quality_output_dir
        self.quality_spatial_stride = quality_spatial_stride
        self.process: subprocess.Popen[bytes] | None = None
        self._log_file: Any = None
        self._launched_at = 0.0

    def launch(self) -> None:
        command = [
            str(self.python),
            "main.py",
            "--port",
            str(self.port),
            "--listen",
            "127.0.0.1",
            "--extra-model-paths-config",
            str(self.config_path),
        ]
        if self.input_directory is not None:
            command.extend(("--input-directory", str(self.input_directory)))
        attention_flag = _ATTENTION_FLAGS.get(self.attention_policy)
        if attention_flag is not None:
            command.append(attention_flag)
        self._log_file = self.log_path.open("wb")
        self._launched_at = time.monotonic()
        environment = dict(os.environ)
        environment["DINKSTER_BENCHMARK_BOOT_NONCE"] = self.boot_nonce
        environment["DINKSTER_BENCHMARK_ATTENTION_POLICY"] = self.attention_policy
        if self.quality_output_dir is not None:
            environment["DINKSTER_BENCHMARK_QUALITY_OUTPUT_DIR"] = str(self.quality_output_dir)
            environment["DINKSTER_BENCHMARK_QUALITY_SPATIAL_STRIDE"] = str(
                self.quality_spatial_stride
            )
        self.process = subprocess.Popen(
            command,
            cwd=str(self.root),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=environment,
        )

    def wait_ready(self, timeout_s: float) -> float:
        """Poll until the HTTP API answers; returns seconds since launch."""
        assert self.process is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(
                    f"server exited with code {code} during boot; see {self.log_path}"
                )
            try:
                self.get("/system_stats")
                return time.monotonic() - self._launched_at
            except (RuntimeError, OSError):
                time.sleep(0.25)
        raise RuntimeError(f"server not ready within {timeout_s}s; see {self.log_path}")

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, payload if payload is not None else {})

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:400]
            raise RuntimeError(f"{method} {path} -> HTTP {error.code}: {detail}") from error
        return json.loads(body) if body else {}

    def shutdown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


def run_prompt(server: ComfyServer, graph: dict[str, Any], timeout_s: float) -> str:
    """Submit one prompt and wait for its history entry to settle,
    raising the recorded exception message when execution failed."""
    prompt_id = str(uuid.uuid4())
    server.post("/prompt", {"prompt": graph, "prompt_id": prompt_id})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        entry = server.get(f"/history/{prompt_id}").get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            if status.get("completed"):
                return prompt_id
            if status.get("status_str") == "error":
                detail = ""
                for message in status.get("messages") or []:
                    if (
                        isinstance(message, list)
                        and len(message) == 2
                        and message[0] == "execution_error"
                        and isinstance(message[1], dict)
                    ):
                        detail = str(message[1].get("exception_message") or "")[:400]
                raise RuntimeError(f"execution failed: {detail or 'unknown error'}")
        time.sleep(0.25)
    raise RuntimeError(f"prompt {prompt_id} did not finish within {timeout_s}s")


class ComfyCell:
    """One benchmark cell's mutable execution state."""

    def __init__(self, arguments: argparse.Namespace, server: ComfyServer) -> None:
        self.arguments = arguments
        self.server = server
        self.checks: dict[str, dict[str, Any]] = {}
        self.failed = False
        self.cold: dict[str, Any] = {}
        self.warm_entries: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.residual_allocated = 0
        self.execution_path: str | None = None
        self.quality_capture: dict[str, Any] | None = None
        self.attention_execution: dict[str, Any] | None = None

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

    def _graph(self, seed: int) -> dict[str, Any]:
        arguments = self.arguments
        return build_graph(
            arguments.family,
            ckpt_name=arguments.checkpoint.name if arguments.checkpoint is not None else None,
            diffusion_name=arguments.diffusion.name if arguments.diffusion is not None else None,
            text_encoder_name=arguments.text_encoder.name
            if arguments.text_encoder is not None
            else None,
            clip_l_name=arguments.clip_l.name if arguments.clip_l is not None else None,
            vae_name=arguments.vae.name if arguments.vae is not None else None,
            audio_vae_name=arguments.audio_vae.name if arguments.audio_vae is not None else None,
            prompt=arguments.prompt,
            negative_prompt=arguments.negative_prompt,
            seed=seed,
            steps=arguments.steps,
            width=arguments.width,
            height=arguments.height,
            length=arguments.length,
            cfg=arguments.cfg,
            sampler=arguments.sampler,
            scheduler=arguments.scheduler,
            lora_name=arguments.lora.name if arguments.lora is not None else None,
            lora_strength_model=arguments.lora_strength_model,
            lora_strength_clip=arguments.lora_strength_clip,
            model_patch_name=arguments.model_patch.name
            if arguments.model_patch is not None
            else None,
            audio_encoder_name=arguments.audio_encoder.name
            if arguments.audio_encoder is not None
            else None,
            clip_vision_name=arguments.clip_vision.name
            if arguments.clip_vision is not None
            else None,
            input_image_name=arguments.input_image.name
            if arguments.input_image is not None
            else None,
            input_audio_1_name=arguments.input_audio_1.name
            if arguments.input_audio_1 is not None
            else None,
            input_audio_2_name=arguments.input_audio_2.name
            if arguments.input_audio_2 is not None
            else None,
            input_audio_name=arguments.input_audio.name
            if arguments.input_audio is not None
            else None,
            motion_frame_count=arguments.motion_frame_count,
            audio_scale=arguments.audio_scale,
            guidance=arguments.guidance,
        )

    def _state(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        state = self.server.get("/dinkster_benchmark/state")
        self.observations = list(state.get("observations") or [])
        return list(state.get("events") or []), self.observations

    # ---------------------------------------------------------------- checks

    def load(self) -> str:
        """The cold run: the first image in the fresh server process.

        One prompt execution covers load through decode, so this check
        extracts every cold phase; the later encode_text and cold_run
        checks report their slices of it.
        """
        arguments = self.arguments
        prompt_id = run_prompt(self.server, self._graph(arguments.seed), arguments.cell_timeout)
        events, _ = self._state()
        intervals = node_intervals(events, prompt_id)
        self.cold = cold_phases(intervals, arguments.family)
        if arguments.family == "minimax_h3":
            self.execution_path = BENCHMARK_MINIMAX_H3_COMFYUI_EXECUTION_PATH
        step_wall = sampler_step_wall_ms(events, prompt_id, arguments.steps)
        if step_wall is not None:
            self.cold["step_wall_ms"] = step_wall
        return f"models loaded in {self.cold['load_s']}s"

    def lora_apply(self) -> str:
        # cold_phases already required the LoraLoader boundary; this
        # check names its interval. ComfyUI applies the decoded patches
        # lazily inside the first sample, so the number is the decode
        # and registration cost only.
        return f"LoraLoader interval {self.cold['lora_s']}s (patches apply in the first sample)"

    def encode_text(self) -> str:
        if self.arguments.family == "minimax_h3":
            encoded = "positive conditioning and empty AV latent built"
        elif self.arguments.family in _ZERO_NEGATIVE_FAMILIES:
            encoded = "cond encoded and uncond zeroed"
        else:
            encoded = "cond and uncond encoded"
        return f"{encoded} in {self.cold['encode_s']}s"

    def encode_audio(self) -> str:
        detail = (
            "two audio streams and image conditioning"
            if self.arguments.family == "wan21_infinitetalk"
            else "Whisper audio and HuMo conditioning"
        )
        return f"{detail} encoded in {self.cold['audio_encode_s']}s"

    def cold_run(self) -> str:
        return (
            f"sample {self.cold['sample_s']}s, decode {self.cold['decode_s']}s,"
            f" total {self.cold['total_s']}s"
        )

    def finite_output(self) -> str:
        if not self.observations:
            raise RuntimeError("the sink recorded no image observation")
        observation = self.observations[0]
        self._validate_observation(observation, "cold")
        if self.arguments.family == "minimax_h3":
            return (
                f"video shape {tuple(observation.get('shape') or ())} and audio shape "
                f"{tuple(observation.get('audio_shape') or ())} at 32000 Hz, all values finite"
            )
        return f"image shape {tuple(observation.get('shape') or ())}, all values finite"

    def _validate_observation(self, observation: dict[str, Any], run_name: str) -> None:
        if not observation.get("finite"):
            raise RuntimeError(f"{run_name} image contains non-finite values")
        if self.arguments.family != "minimax_h3":
            return
        expected_video = [
            self.arguments.length,
            self.arguments.height,
            self.arguments.width,
            3,
        ]
        if observation.get("shape") != expected_video:
            raise RuntimeError(
                f"{run_name} MiniMax H3 video shape {observation.get('shape')!r} "
                f"is not {expected_video}"
            )
        audio_shape = observation.get("audio_shape")
        if not (
            isinstance(audio_shape, list)
            and len(audio_shape) == 3
            and audio_shape[:2] == [1, 2]
            and isinstance(audio_shape[2], int)
            and audio_shape[2] > 0
        ):
            raise RuntimeError(
                f"{run_name} MiniMax H3 audio shape {audio_shape!r} is not batch-one stereo"
            )
        if observation.get("audio_sample_rate") != 32_000:
            raise RuntimeError(
                f"{run_name} MiniMax H3 audio sample rate "
                f"{observation.get('audio_sample_rate')!r} is not 32000"
            )
        if not observation.get("audio_finite"):
            raise RuntimeError(f"{run_name} MiniMax H3 audio contains non-finite values")

    def warm_runs(self) -> str:
        arguments = self.arguments
        for index in range(arguments.warm_runs):
            prompt_id = run_prompt(
                self.server, self._graph(arguments.seed + 1 + index), arguments.cell_timeout
            )
            events, observations = self._state()
            if len(observations) != index + 2:
                raise RuntimeError(
                    f"warm run {index} recorded {len(observations)} sink observations;"
                    f" expected {index + 2}"
                )
            self._validate_observation(observations[index + 1], f"warm run {index}")
            entry = warm_phases(node_intervals(events, prompt_id), arguments.family)
            step_wall = sampler_step_wall_ms(events, prompt_id, arguments.steps)
            if step_wall is not None:
                entry["step_wall_ms"] = step_wall
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
        self.server.post("/dinkster_benchmark/arm_quality", {"seed": capture_seed})
        run_prompt(self.server, self._graph(capture_seed), arguments.cell_timeout)
        _events, observations = self._state()
        expected = arguments.warm_runs + 2
        if len(observations) != expected:
            raise RuntimeError(
                f"quality capture recorded {len(observations)} sink observations; "
                f"expected {expected}"
            )
        observation = observations[-1]
        self._validate_observation(observation, "quality capture")
        capture = observation.get("quality_capture")
        if not isinstance(capture, dict) or capture.get("seed") != capture_seed:
            raise RuntimeError("the sink did not authenticate the quality capture seed")
        self.quality_capture = capture
        return f"captured image and audio at seed {capture_seed}"

    def authenticate_attention_execution(self) -> str:
        state = self.server.get("/dinkster_benchmark/state")
        execution = state.get("attention_execution")
        problem = attention_execution_problem(execution, self.arguments.attention_policy)
        if problem is not None:
            raise RuntimeError(problem)
        assert isinstance(execution, dict)
        self.attention_execution = execution
        return (
            f"provider successes={execution['provider_successes']}, "
            f"fallbacks={execution['fallback_calls']}"
        )

    def unload(self) -> str:
        response = self.server.post(
            "/dinkster_benchmark/unload",
            {"drain_target_bytes": FAMILY_RESIDUAL_CEILING_BYTES},
        )
        residual = int(response.get("residual_allocated_bytes") or 0)
        self.residual_allocated = residual
        if residual > FAMILY_RESIDUAL_CEILING_BYTES:
            raise RuntimeError(f"allocator still holds {residual} B after unload")
        return f"residual_allocated={residual} B"


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BENCHMARK_ACCELERATORS, required=True)
    parser.add_argument("--family", choices=BENCHMARK_FAMILIES, required=True)
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint")
    parser.add_argument("--diffusion", type=Path, help="diffusion safetensors (split families)")
    parser.add_argument("--text-encoder", type=Path, help="text encoder safetensors (split)")
    parser.add_argument("--clip-l", type=Path, help="Flux CLIP-L text encoder safetensors")
    parser.add_argument("--vae", type=Path, help="VAE safetensors (split families)")
    parser.add_argument("--audio-vae", type=Path, help="MiniMax H3 audio VAE safetensors")
    parser.add_argument("--lora", type=Path, help="LoRA safetensors")
    parser.add_argument("--model-patch", type=Path, help="InfiniteTalk model patch safetensors")
    parser.add_argument("--audio-encoder", type=Path, help="audio encoder safetensors")
    parser.add_argument("--clip-vision", type=Path, help="CLIP vision safetensors")
    parser.add_argument("--input-image", type=Path, help="reference image input")
    parser.add_argument("--input-audio-1", type=Path, help="first audio input")
    parser.add_argument("--input-audio-2", type=Path, help="second audio input")
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
        help="MiniMax H3 attention implementation",
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
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--comfyui-root", type=Path, required=True, help="pinned ComfyUI checkout")
    parser.add_argument(
        "--comfyui-python", type=Path, required=True, help="the checkout's venv python"
    )
    parser.add_argument("--port", type=int, default=8299)
    parser.add_argument("--boot-timeout", type=float, default=240.0)
    parser.add_argument("--cell-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="server stdout+stderr (default: next to --json)",
    )
    parser.add_argument(
        "--require-commit",
        default=COMFYUI_PIN,
        help="fail unless the checkout's HEAD starts with this SHA"
        " (default: the documented pin; pass '' to disable for unpinned families)",
    )
    arguments = parser.parse_args(argv)
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
    if arguments.quality_spatial_stride < 1:
        parser.error("--quality-spatial-stride must be positive")
    if arguments.family != "minimax_h3":
        if arguments.attention_policy != "auto":
            parser.error("--attention-policy is currently supported only with MiniMax H3")
        if arguments.quality_output_dir is not None:
            parser.error("--quality-output-dir is currently supported only with MiniMax H3")
    split_family = (
        arguments.family in _SPLIT_GRAPH
        or arguments.family in _AUDIO_FAMILIES
        or arguments.family == _ANIMA_FAMILY
        or arguments.family == _FLUX_FAMILY
        or arguments.family == "minimax_h3"
    )
    if split_family:
        for name in ("diffusion", "text_encoder", "vae"):
            if getattr(arguments, name) is None:
                parser.error(f"{arguments.family} family requires --{name.replace('_', '-')}")
        if arguments.checkpoint is not None:
            parser.error(f"--checkpoint is not meaningful with --family {arguments.family}")
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
        parser.error("--lora is only meaningful with a LoRA workload family")
    if arguments.family == "minimax_h3":
        if arguments.audio_vae is None:
            parser.error("minimax_h3 family requires --audio-vae")
    elif arguments.audio_vae is not None:
        parser.error("--audio-vae is only meaningful with MiniMax H3")
    shared_audio_arguments = ("audio_encoder", "input_image")
    infinitetalk_arguments = (
        "model_patch",
        "clip_vision",
        "input_audio_1",
        "input_audio_2",
    )
    if arguments.family == "wan21_infinitetalk":
        for name in (*shared_audio_arguments, *infinitetalk_arguments):
            if getattr(arguments, name) is None:
                parser.error(f"wan21_infinitetalk family requires --{name.replace('_', '-')}")
        if arguments.input_audio is not None:
            parser.error("--input-audio is only meaningful with HuMo")
        if arguments.motion_frame_count < 1 or arguments.motion_frame_count > 33:
            parser.error("--motion-frame-count must be between 1 and 33")
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
        for name in (*shared_audio_arguments, "input_audio"):
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
        if arguments.require_commit != COMFYUI_PIN:
            parser.error(f"--require-commit must be {COMFYUI_PIN!r} for wan21_humo")
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
        if arguments.require_commit != COMFYUI_PIN:
            parser.error(f"--require-commit must be {COMFYUI_PIN!r} for anima")
        for name in (*shared_audio_arguments, *infinitetalk_arguments, "input_audio"):
            if getattr(arguments, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} is only meaningful with an audio workload"
                )
    elif arguments.family == "minimax_h3":
        for name in (*shared_audio_arguments, *infinitetalk_arguments, "input_audio"):
            if getattr(arguments, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} is only meaningful with an audio input workload"
                )
        pinned = _WORKLOAD_DEFAULTS["minimax_h3"]
        for name, expected in pinned.items():
            if getattr(arguments, name) != expected:
                parser.error(f"--{name.replace('_', '-')} must be {expected!r} for minimax_h3")
        if arguments.motion_frame_count != 9:
            parser.error("--motion-frame-count is only meaningful with InfiniteTalk")
        if arguments.audio_scale != 1.0:
            parser.error("--audio-scale is only meaningful with InfiniteTalk")
        if arguments.require_commit != COMFYUI_PIN:
            parser.error(f"--require-commit must be {COMFYUI_PIN!r} for minimax_h3")
    elif arguments.family == _FLUX_FAMILY:
        for name in (*shared_audio_arguments, *infinitetalk_arguments, "input_audio"):
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
        if arguments.require_commit != COMFYUI_PIN:
            parser.error(f"--require-commit must be {COMFYUI_PIN!r} for flux")
    elif arguments.family == _CHROMA_FAMILY:
        for name in (*shared_audio_arguments, *infinitetalk_arguments, "input_audio"):
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
        if arguments.require_commit != COMFYUI_PIN:
            parser.error(f"--require-commit must be {COMFYUI_PIN!r} for chroma")
    else:
        for name in (*shared_audio_arguments, *infinitetalk_arguments, "input_audio"):
            if getattr(arguments, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} is only meaningful with an audio workload"
                )
        if arguments.motion_frame_count != 9:
            parser.error("--motion-frame-count is only meaningful with InfiniteTalk")
        if arguments.audio_scale != 1.0:
            parser.error("--audio-scale is only meaningful with InfiniteTalk")
    if arguments.family in ("wan21", "wan21_infinitetalk", "wan21_humo"):
        # The causal video VAE decodes 4*T - 3 frames from T latent
        # frames, so only 4k+1 requests decode to exactly that count.
        if arguments.length < 1 or (arguments.length - 1) % 4 != 0:
            parser.error("--length must be a positive 4k+1 frame count (e.g. 33)")
    elif arguments.family == "minimax_h3":
        if arguments.length < 5 or (arguments.length - 5) % 17 != 0:
            parser.error("--length must be a positive 17k+5 frame count (e.g. 124)")
    elif arguments.length is not None:
        parser.error("--length is only meaningful with a video family")
    if not (arguments.comfyui_root / "main.py").is_file():
        parser.error(f"--comfyui-root {arguments.comfyui_root} has no main.py")
    return arguments


def _git(root: Path, *args: str) -> str | None:
    """The command's stripped stdout, or None when git itself failed."""
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


def enforce_checkout(root: Path, require_commit: str) -> str:
    """The checkout's HEAD commit, exiting unless it matches require_commit
    on a clean tree; an empty require_commit skips enforcement. A dirty tree
    or an unanswerable git query would make the recorded commit silently
    wrong, so both fail closed."""
    commit = _git(root, "rev-parse", "HEAD")
    if not require_commit:
        return commit or ""
    if commit is None or not commit.startswith(require_commit):
        sys.exit(
            f"error: checkout HEAD {commit or '(unknown)'} does not start"
            f" with required commit {require_commit}"
        )
    dirty = _git(root, "status", "--porcelain")
    if dirty is None:
        sys.exit(
            f"error: could not check {root} for local modifications;"
            " refusing to record a commit that was not verified clean"
        )
    if dirty:
        sys.exit(
            f"error: checkout {root} has local modifications;"
            " the report would record a commit it does not measure"
        )
    return commit


def main() -> int:
    arguments = _parse_arguments()

    commit = enforce_checkout(arguments.comfyui_root, arguments.require_commit)
    if arguments.quality_output_dir is not None:
        arguments.quality_output_dir = arguments.quality_output_dir.resolve()
        arguments.quality_output_dir.mkdir(parents=True, exist_ok=True)
        if any(arguments.quality_output_dir.iterdir()):
            sys.exit(f"error: --quality-output-dir must be empty: {arguments.quality_output_dir}")

    print(f"cell:     comfyui {arguments.family} / eager on {arguments.backend}")
    print(f"checkout: {arguments.comfyui_root} @ {commit or '(unknown)'}")

    # Artifacts are hashed driver-side; the server sees them by name
    # through the generated extra model paths config.
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
        artifacts = [
            _artifact_entry(role, path, _INFINITETALK_ARTIFACT_PINS[role])
            for role, path in artifact_paths.items()
        ]
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
        artifacts = [
            _artifact_entry(role, path, _HUMO_ARTIFACT_PINS[role])
            for role, path in artifact_paths.items()
        ]
    elif arguments.family == _ANIMA_FAMILY:
        artifact_paths = {
            "diffusion": arguments.diffusion,
            "text_encoder": arguments.text_encoder,
            "vae": arguments.vae,
        }
        artifacts = [
            _artifact_entry(role, path, _ANIMA_ARTIFACT_PINS[role])
            for role, path in artifact_paths.items()
        ]
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
        artifacts = [
            _artifact_entry(role, path, _FLUX_ARTIFACT_PINS[role])
            for role, path in artifact_paths.items()
        ]
    elif arguments.family == _CHROMA_FAMILY:
        artifact_paths = {
            "diffusion": arguments.diffusion,
            "text_encoder": arguments.text_encoder,
            "vae": arguments.vae,
        }
        artifacts = [
            _artifact_entry(role, path, _CHROMA_ARTIFACT_PINS[role])
            for role, path in artifact_paths.items()
        ]
    elif arguments.family in _SPLIT_GRAPH:
        artifacts = [
            _artifact_entry("diffusion", arguments.diffusion),
            _artifact_entry("text_encoder", arguments.text_encoder),
            _artifact_entry("vae", arguments.vae),
        ]
    else:
        artifacts = [_artifact_entry("checkpoint", arguments.checkpoint)]
        if arguments.family == "lora":
            artifacts.append(_artifact_entry("lora", arguments.lora))

    if arguments.server_log is not None:
        log_path = arguments.server_log
    elif arguments.json is not None:
        log_path = arguments.json.with_suffix(".server.log")
    else:
        log_path = Path(tempfile.gettempdir()) / f"comfyui-bench-{arguments.port}.server.log"

    config_text = extra_model_paths_config(
        arguments.checkpoint.resolve().parent if arguments.checkpoint is not None else None,
        arguments.lora.resolve().parent if arguments.lora is not None else None,
        diffusion_directory=arguments.diffusion.resolve().parent
        if arguments.diffusion is not None
        else None,
        text_encoder_directory=arguments.text_encoder.resolve().parent
        if arguments.text_encoder is not None
        else None,
        clip_l_directory=arguments.clip_l.resolve().parent
        if arguments.clip_l is not None
        else None,
        vae_directory=arguments.vae.resolve().parent if arguments.vae is not None else None,
        audio_vae_directory=arguments.audio_vae.resolve().parent
        if arguments.audio_vae is not None
        else None,
        model_patch_directory=arguments.model_patch.resolve().parent
        if arguments.model_patch is not None
        else None,
        audio_encoder_directory=arguments.audio_encoder.resolve().parent
        if arguments.audio_encoder is not None
        else None,
        clip_vision_directory=arguments.clip_vision.resolve().parent
        if arguments.clip_vision is not None
        else None,
    )
    config_file = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="dinkster-benchmark-paths-", delete=False
    )
    config_path = Path(config_file.name)
    with config_file:
        config_file.write(config_text)

    input_temp: tempfile.TemporaryDirectory[str] | None = None
    input_directory: Path | None = None
    server: ComfyServer | None = None
    try:
        if arguments.family in _AUDIO_FAMILIES:
            input_temp = tempfile.TemporaryDirectory(prefix="dinkster-benchmark-input-")
            input_directory = Path(input_temp.name)
            sources = (
                (arguments.input_image, arguments.input_audio_1, arguments.input_audio_2)
                if arguments.family == "wan21_infinitetalk"
                else (arguments.input_image, arguments.input_audio)
            )
            for source in sources:
                shutil.copy2(source, input_directory / source.name)

        server = ComfyServer(
            root=arguments.comfyui_root,
            python=arguments.comfyui_python,
            port=arguments.port,
            config_path=config_path,
            log_path=log_path,
            boot_nonce=str(uuid.uuid4()),
            input_directory=input_directory,
            attention_policy=arguments.attention_policy,
            quality_output_dir=arguments.quality_output_dir,
            quality_spatial_stride=arguments.quality_spatial_stride,
        )
        server.launch()
        import_s = server.wait_ready(arguments.boot_timeout)
        print(f"boot:     {import_s:.4f}s to HTTP ready (recorded as import_s)")

        try:
            identity = server.get(f"/dinkster_benchmark/identity?backend={arguments.backend}")
        except RuntimeError as error:
            sys.exit(f"error: backend admission failed: {error}")
        if identity.get("boot_nonce") != server.boot_nonce:
            # A stale server orphaned on the port would answer before the
            # fresh launch fails to bind; refuse to measure it.
            sys.exit(
                f"error: the server answering on port {arguments.port} is not"
                " the one this run launched"
            )
        attention = identity.get("attention")
        attention_problem = attention_identity_problem(attention, arguments.attention_policy)
        if attention_problem is not None:
            sys.exit(
                "error: ComfyUI did not authenticate the requested attention policy: "
                f"{attention_problem}; {attention!r}"
            )
        assert isinstance(attention, dict)
        print(f"driver:   {identity.get('driver')}")
        torch_section = identity.get("torch") or {}
        print(f"torch:    {torch_section.get('version')} ({torch_section.get('backend_runtime')})")

        server.post("/dinkster_benchmark/reset")

        cell = ComfyCell(arguments, server)
        print("checks:")
        cell.record("load", cell.load)
        if arguments.family == "lora":
            cell.record("lora_apply", cell.lora_apply)
        cell.record("encode_text", cell.encode_text)
        if arguments.family in _AUDIO_FAMILIES:
            cell.record("encode_audio", cell.encode_audio)
        cell.record("cold_run", cell.cold_run)
        cell.record("finite_output", cell.finite_output)
        cell.record("warm_runs", cell.warm_runs)
        try:
            memory = server.get("/dinkster_benchmark/memory")
        except (RuntimeError, OSError):
            # A dead server already failed the checks; the empty section
            # fails validation without masking the check details.
            memory = {}
        if arguments.quality_output_dir is not None:
            cell.record("quality_capture", cell.capture_quality)
        if arguments.attention_policy != "auto":
            cell.record("attention_execution", cell.authenticate_attention_execution)
        cell.record("unload", cell.unload, always=True)
    finally:
        if server is not None:
            server.shutdown()
        config_path.unlink(missing_ok=True)
        if input_temp is not None:
            input_temp.cleanup()

    report = assemble_report(
        backend=arguments.backend,
        family=arguments.family,
        identity=identity,
        comfyui_commit=commit,
        workload=workload_section(arguments),
        artifacts=artifacts,
        import_s=import_s,
        cold=cell.cold,
        warm_entries=cell.warm_entries,
        memory=memory,
        residual_allocated=cell.residual_allocated,
        checks=cell.checks,
        attention={
            **attention,
            **(
                {"execution": cell.attention_execution}
                if cell.attention_execution is not None
                else {}
            ),
        },
        quality_capture=cell.quality_capture,
        variant=(
            BENCHMARK_ANIMA_FALLBACK_VARIANT
            if arguments.fallback_768
            else BENCHMARK_PRIMARY_VARIANT
        )
        if arguments.family == _ANIMA_FAMILY
        else None,
        execution_path=cell.execution_path,
    )

    problems = validate_benchmark_report(report, accelerator=arguments.backend)
    for problem in problems:
        print(f"error: incomplete report: {problem}", file=sys.stderr)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report written: {arguments.json}")

    if not report["all_ok"]:
        print("error: a benchmark check failed", file=sys.stderr)
        return 1
    if problems:
        return 1
    print(f"benchmark cell comfyui/{arguments.family}/eager passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
