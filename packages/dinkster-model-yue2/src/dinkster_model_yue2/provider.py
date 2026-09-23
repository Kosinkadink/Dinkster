"""YuE2 node execution over admitted runtime handles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from dinkster_inference import require_inference_runtime_handle

from .declarations import FRAMES_PER_SECOND, LATENT_CHANNELS, LATENT_DOWNSCALE
from .runtime import YuE2TextRuntime


def _runtime(value: object) -> tuple[Any, Any, YuE2TextRuntime]:
    handle = require_inference_runtime_handle(value, "clip")
    runtime: Any = handle.runtime
    if runtime.family.id != "dinkster.yue2":
        raise TypeError("clip must be a YuE2 runtime")
    text_runtime = getattr(runtime, "_text_runtime", None)
    if not isinstance(text_runtime, YuE2TextRuntime):
        raise TypeError("clip must retain the YuE2 text runtime")
    return handle, runtime, text_runtime


def execute_generate_abc(**inputs: object) -> Mapping[str, object]:
    clip = inputs.pop("clip")
    handle, _runtime_owner, text_runtime = _runtime(clip)
    with handle.stage("text"), torch.inference_mode():
        abc = text_runtime.generate_abc(**inputs)
    return {"abc": abc}


def execute_generate_music(**inputs: object) -> Mapping[str, object]:
    clip = inputs.pop("clip")
    handle, runtime, text_runtime = _runtime(clip)
    with handle.stage("text"), torch.inference_mode():
        conditioning = text_runtime.generate_music(**inputs)
    return {
        "conditioning": runtime.text_conditioning_carrier(conditioning),
        "seconds": conditioning.frames / FRAMES_PER_SECOND,
    }


def execute_empty_latent(*, seconds: object, batch_size: object) -> Mapping[str, object]:
    if not isinstance(seconds, (int, float)) or not 0.04 <= seconds <= 1000.0:
        raise ValueError("seconds must be in [0.04, 1000.0]")
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError("batch_size must be an integer in [1, 4096]")
    frames = max(1, round(float(seconds) * FRAMES_PER_SECOND))
    return {
        "latent": {
            "samples": torch.zeros((batch_size, LATENT_CHANNELS, frames)),
            "type": "audio",
            "downscale_ratio_temporal": LATENT_DOWNSCALE,
        }
    }


def execute_decode_audio(*, samples: object, vae: object) -> Mapping[str, object]:
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a latent mapping")
    latent = cast("Mapping[object, object]", samples).get("samples")
    if type(latent) is not torch.Tensor:
        raise TypeError("samples['samples'] must be an exact torch.Tensor")
    if latent.ndim != 3 or latent.shape[0] < 1 or latent.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"samples['samples'] must be nonempty [batch,{LATENT_CHANNELS},frames]")
    handle = require_inference_runtime_handle(vae, "vae")
    runtime: Any = handle.runtime
    if runtime.family.id != "dinkster.yue2":
        raise TypeError("vae must be a YuE2 runtime")
    with handle.stage("vae"), torch.inference_mode():
        waveform = runtime.decode_latent(latent.to(cast("Any", handle.load_device)))
    if (
        type(waveform) is not torch.Tensor
        or waveform.ndim != 3
        or waveform.shape[0] != latent.shape[0]
        or waveform.shape[1] != 2
        or waveform.shape[2] < 1
        or not waveform.is_floating_point()
        or waveform.layout is not torch.strided
    ):
        raise ValueError("YuE2 audio decode must return nonempty floating [batch,2,samples]")
    waveform = waveform.to(device="cpu", dtype=torch.float32, copy=True)
    return {"audio": {"waveform": waveform, "sample_rate": 48_000}}


__all__ = [
    "execute_decode_audio",
    "execute_empty_latent",
    "execute_generate_abc",
    "execute_generate_music",
]
