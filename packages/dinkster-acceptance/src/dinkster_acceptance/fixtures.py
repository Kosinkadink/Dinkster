from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from typing import TypeVar

import torch
from dinkster_inference import FLUX_DEV, ClipTextConfig, FluxConfig, KLConfig, T5Config
from dinkster_inference_torch import (
    AssembledFlux,
    AutoencoderKL,
    ClipTextModel,
    Flux,
    T5TextModel,
)

TINY_CLIP = ClipTextConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    hidden_act="quick_gelu",
    vocab_size=49408,
    eos_token_id=49407,
)
TINY_T5 = T5Config(
    d_model=48,
    d_ff=96,
    d_kv=12,
    num_heads=4,
    num_layers=2,
    vocab_size=32128,
    dense_act_fn="gelu_pytorch_tanh",
    is_gated_act=True,
)
TINY_FLUX = FluxConfig(
    in_channels=16,
    out_channels=16,
    vec_in_dim=TINY_CLIP.hidden_size,
    context_in_dim=TINY_T5.d_model,
    hidden_size=32,
    depth=1,
    depth_single_blocks=1,
    num_heads=2,
    axes_dim=(4, 6, 6),
    guidance_embed=True,
)
TINY_KL = KLConfig(
    in_channels=3,
    out_channels=3,
    ch=32,
    decoder_ch=32,
    ch_mult=(1, 2),
    num_res_blocks=1,
    z_channels=16,
    embed_dim=16,
)

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)
_MASK32 = 0xFFFFFFFF


def _hash_uniform(seed: int, count: int) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.int64) + (seed & _MASK32)
    values = (values * 1664525 + 1013904223) & _MASK32
    values = values ^ (values >> 13)
    values = (values * 214013 + 2531011) & _MASK32
    values = values ^ (values >> 17)
    values = (values * 69069 + 1) & _MASK32
    values = values ^ (values >> 5)
    return (values.to(torch.float64) / float(_MASK32 + 1)).to(torch.float32)


def _fill_value(key: str, shape: Sequence[int]) -> torch.Tensor:
    count = math.prod(shape) if shape else 1
    uniform = _hash_uniform(zlib.crc32(key.encode("utf-8")), count)
    if key.endswith(".weight") and "layer_norm" in key:
        values = 1.0 + (uniform - 0.5) * 0.1
    else:
        values = (uniform - 0.5) * 0.4
    return values.reshape(tuple(shape))


def _filled(module: ModuleT) -> ModuleT:
    state = {
        key: _fill_value(key, tuple(tensor.shape)) for key, tensor in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True)
    return module


def assembled_flux() -> AssembledFlux:
    torch.manual_seed(0)
    return AssembledFlux(
        family=FLUX_DEV,
        diffusion=_filled(Flux(TINY_FLUX)),
        clip_l=_filled(ClipTextModel(TINY_CLIP)),
        t5xxl=_filled(T5TextModel(TINY_T5)),
        vae=_filled(AutoencoderKL(TINY_KL)),
    )


def tiny_latent() -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(99)
    return torch.randn(1, TINY_FLUX.in_channels, 4, 4, generator=generator)
