"""Native SeedVR2 diffusion transformers from ComfyUI 8a33128f2f8c5585c57486c07de481241e70a39c."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate, chain
from math import ceil, pi
from typing import Any, TypeVar, cast

import torch
import torch.nn.functional as F
from torch import nn

from .attention import AttentionKernel, select_attention
from .flux import apply_rope1
from .operations import INITLESS, Operations, ResidencyRouted
from .ops import cast_weight
from .seedvr2_attention import optimized_var_attention
from .seedvr2_constants import (
    BYTEDANCE_720P_REF_AREA,
    BYTEDANCE_MAX_TEMPORAL_WINDOW,
    BYTEDANCE_ROPE_MAX_FREQ,
    BYTEDANCE_SINUSOIDAL_DIM,
    ROPE_THETA,
    SEEDVR2_7B_MLP_CHUNK,
    SEEDVR2_7B_VID_DIM,
    SEEDVR2_LATENT_CHANNELS,
    SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS,
)

_DEFAULT_ATTENTION = select_attention("flux").kernel
T = TypeVar("T")
Device = torch.device | str | None
WindowSlices = list[tuple[slice, slice, slice]]
WindowOp = Callable[[tuple[int, int, int], tuple[int, int, int]], WindowSlices]


def _as_triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (value, value, value) if isinstance(value, int) else value


class _ComfyOperations:
    """Comfy-style factory names over Dinkster's typed Operations seam."""

    def __init__(self, operations: Operations) -> None:
        self._operations = operations

    def Linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Linear:
        return self._operations.linear(in_features, out_features, bias=bias)

    def RMSNorm(
        self,
        normalized_shape: int,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> nn.RMSNorm:
        if not elementwise_affine:
            return nn.RMSNorm(normalized_shape, eps=eps, elementwise_affine=False)
        return self._operations.rms_norm(normalized_shape, eps=eps)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = math.log(10000) / (half_dim - downscale_freq_shift)
    frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -exponent).to(
        device=timesteps.device
    )
    embedding = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


class Cache:
    def __init__(
        self,
        disable: bool = False,
        prefix: str = "",
        cache: dict[str, Any] | None = None,
    ) -> None:
        self.cache = cache if cache is not None else {}
        self.disable = disable
        self.prefix = prefix

    def __call__(self, key: str, fn: Callable[[], T]) -> T:
        if self.disable:
            return fn()

        key = self.prefix + key
        if key not in self.cache:
            result = fn()
            self.cache[key] = result
        return self.cache[key]

    def namespace(self, namespace: str) -> Cache:
        return Cache(
            disable=self.disable,
            prefix=self.prefix + namespace + ".",
            cache=self.cache,
        )


def repeat_concat(
    vid: torch.Tensor,  # (VL ... c)
    txt: torch.Tensor,  # (TL ... c)
    vid_len: torch.Tensor,  # (n*b)
    txt_len: torch.Tensor,  # (b)
    txt_repeat: list[int],  # (n)
) -> torch.Tensor:  # (L ... c)
    vid_parts = torch.split(vid, vid_len.tolist())
    txt_parts = torch.split(txt, txt_len.tolist())
    repeated_txt = [[x] * n for x, n in zip(txt_parts, txt_repeat, strict=True)]
    flat_txt = list(chain(*repeated_txt))
    return torch.cat(list(chain(*zip(vid_parts, flat_txt, strict=True))))


def repeat_concat_idx(
    vid_len: torch.Tensor,  # (n*b)
    txt_len: torch.Tensor,  # (b)
    txt_repeat: torch.Tensor,  # (n)
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
]:
    device = vid_len.device
    vid_idx = torch.arange(int(vid_len.sum()), device=device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + int(txt_len.sum()), device=device)
    txt_repeat_list = txt_repeat.tolist()
    tgt_idx = repeat_concat(vid_idx, txt_idx, vid_len, txt_len, txt_repeat_list)
    src_idx = torch.argsort(tgt_idx)
    txt_idx_len = len(tgt_idx) - len(vid_idx)
    repeat_txt_len = (txt_len * txt_repeat).tolist()

    def concat(vid: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        return torch.cat([vid, txt])[tgt_idx]

    def unconcat_coalesce(all_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vid_out, txt_out = all_values[src_idx].split([len(vid_idx), txt_idx_len])
        txt_out_coalesced = []
        for txt, repeat_time in zip(txt_out.split(repeat_txt_len), txt_repeat_list, strict=True):
            txt = txt.reshape(-1, repeat_time, *txt.shape[1:]).mean(1)
            txt_out_coalesced.append(txt)
        return vid_out, torch.cat(txt_out_coalesced)

    return (
        concat,
        unconcat_coalesce,
    )


def cumulative_lengths(lengths: Sequence[int]) -> list[int]:
    return [0, *accumulate(lengths)]


@dataclass
class MMArg:
    vid: Any
    txt: Any


def get_args(key: str, args: Sequence[Any]) -> list[Any]:
    return [getattr(v, key) if isinstance(v, MMArg) else v for v in args]


def get_kwargs(key: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: getattr(v, key) if isinstance(v, MMArg) else v for k, v in kwargs.items()}


def get_window_op(name: str) -> WindowOp:
    if name == "720pwin_by_size_bysize":
        return make_720Pwindows_bysize
    if name == "720pswin_by_size_bysize":
        return make_shifted_720Pwindows_bysize
    raise ValueError(f"Unknown windowing method: {name}")


def make_720Pwindows_bysize(
    size: tuple[int, int, int], num_windows: tuple[int, int, int]
) -> WindowSlices:
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = math.sqrt(BYTEDANCE_720P_REF_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)
    wt = ceil(min(t, BYTEDANCE_MAX_TEMPORAL_WINDOW) / resized_nt)
    nt, nh, nw = ceil(t / wt), ceil(h / wh), ceil(w / ww)
    return [
        (
            slice(it * wt, min((it + 1) * wt, t)),
            slice(ih * wh, min((ih + 1) * wh, h)),
            slice(iw * ww, min((iw + 1) * ww, w)),
        )
        for iw in range(nw)
        if min((iw + 1) * ww, w) > iw * ww
        for ih in range(nh)
        if min((ih + 1) * wh, h) > ih * wh
        for it in range(nt)
        if min((it + 1) * wt, t) > it * wt
    ]


def make_shifted_720Pwindows_bysize(
    size: tuple[int, int, int], num_windows: tuple[int, int, int]
) -> WindowSlices:
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    scale = math.sqrt(BYTEDANCE_720P_REF_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)
    wt = ceil(min(t, BYTEDANCE_MAX_TEMPORAL_WINDOW) / resized_nt)

    st, sh, sw = (
        0.5 if wt < t else 0,
        0.5 if wh < h else 0,
        0.5 if ww < w else 0,
    )
    nt, nh, nw = ceil((t - st) / wt), ceil((h - sh) / wh), ceil((w - sw) / ww)
    nt, nh, nw = (
        nt + 1 if st > 0 else 1,
        nh + 1 if sh > 0 else 1,
        nw + 1 if sw > 0 else 1,
    )
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]


class RotaryEmbedding(ResidencyRouted, nn.Module):
    freqs: torch.Tensor

    def __init__(
        self,
        dim: int,
        freqs_for: str = "lang",
        theta: float = 10000,
        max_freq: float = 10,
    ) -> None:
        super().__init__()

        self.freqs_for = freqs_for

        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        else:
            raise ValueError(f"Unknown rotary frequency type: {freqs_for}")

        self.register_buffer("freqs", freqs)

    @property
    def device(self) -> torch.device:
        return self.freqs.device

    def get_axial_freqs(
        self,
        *dims: int,
        offsets: Sequence[int | float] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        Colon = slice(None)
        all_freqs = []
        target_device = self.device if device is None else device

        if offsets is not None:
            if len(offsets) != len(dims):
                raise ValueError(
                    "SeedVR2 rotary offsets length must match dims length, got "
                    f"{len(offsets)} and {len(dims)}."
                )

        for ind, dim in enumerate(dims):
            offset = 0
            if offsets is not None:
                offset = offsets[ind]

            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=target_device)
            else:
                pos = torch.arange(dim, device=target_device)

            pos = pos + offset

            freqs = self.forward(pos)

            all_axis: list[slice | None] = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        stored = self.freqs

        def apply(freqs: torch.Tensor) -> torch.Tensor:
            freqs = freqs.to(device=t.device, dtype=torch.float32)
            positions = t.float()
            result = torch.einsum("..., f -> ... f", positions, freqs)
            return result.unsqueeze(-1).expand(*result.shape, 2).flatten(-2)

        binding = self._offloaded_residency()
        if binding is None:
            return apply(stored)
        with binding.lease() as lease:
            return apply(lease.get("freqs", dtype=stored.dtype))


class RotaryEmbeddingBase(nn.Module):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__()
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="pixel",
            max_freq=BYTEDANCE_ROPE_MAX_FREQ,
        )

    def get_axial_freqs(self, *dims: int, device: torch.device | None = None) -> torch.Tensor:
        return self.rope.get_axial_freqs(*dims, device=device)


class RotaryEmbedding3d(RotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)
        self.mm = False


class NaRotaryEmbedding3d(RotaryEmbedding3d):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        shape: torch.Tensor,
        cache: Cache,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        get_freqs = cast(Callable[[torch.Tensor], torch.Tensor], self.get_freqs)
        freqs = cache("rope_freqs_3d", lambda: get_freqs(shape))
        freqs = freqs.to(device=q.device)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        q = _apply_seedvr2_rotary_emb(freqs, q.float()).to(q.dtype)
        k = _apply_seedvr2_rotary_emb(freqs, k.float()).to(k.dtype)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        return q, k

    @torch.compiler.disable
    def get_freqs(
        self,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        # Primary provenance: ByteDance-Seed/SeedVR models/dit/rope.py builds
        # 7B pixel RoPE with the interleaved-angle convention, not Comfy's
        # Flux freqs_cis matrix.
        plain_rope = RotaryEmbedding(
            dim=self.rope.freqs.numel() * 2,
            freqs_for="pixel",
            max_freq=BYTEDANCE_ROPE_MAX_FREQ,
        )
        plain_rope = plain_rope.to(self.rope.device)
        freq_list = []
        for f, h, w in shape.tolist():
            freqs = plain_rope.get_axial_freqs(f, h, w, device=shape.device)
            freq_list.append(freqs.view(-1, freqs.size(-1)))
        return torch.cat(freq_list, dim=0)


class MMRotaryEmbeddingBase(RotaryEmbeddingBase):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__(dim, rope_dim)
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="lang",
            theta=ROPE_THETA,
        )
        self.mm = True


def slice_at_dim(t: torch.Tensor, dim_slice: slice, *, dim: int) -> torch.Tensor:
    dim += t.ndim if dim < 0 else 0
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def exists(val: object | None) -> bool:
    return val is not None


def _apply_seedvr2_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
    freqs_seq_dim: int | None = None,
) -> torch.Tensor:
    dtype = t.dtype
    if freqs_seq_dim is None and (freqs.ndim == 2 or t.ndim == 3):
        freqs_seq_dim = 0

    if t.ndim == 3 or freqs_seq_dim is not None:
        seq_len = t.shape[seq_dim]
        assert freqs_seq_dim is not None
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

    rot_feats = freqs.shape[-1]
    end_index = start_index + rot_feats

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    freqs = freqs.to(device=t_middle.device, dtype=t_middle.dtype)
    cos = freqs.cos() * scale
    sin = freqs.sin() * scale
    t_middle = (t_middle * cos) + (rotate_half(t_middle) * sin)
    return torch.cat((t_left, t_middle, t_right), dim=-1).to(dtype)


def _to_flux_freqs_cis(freqs_interleaved: torch.Tensor) -> torch.Tensor:
    angles = freqs_interleaved[..., ::2].float()
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2)


def _apply_rope1_partial(t: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    out = t.clone() if torch.is_grad_enabled() and t.requires_grad else t
    rot_d = 2 * freqs_cis.shape[-3]
    seq_len = out.shape[-2]
    for start in range(0, seq_len, SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS):
        end = min(start + SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS, seq_len)
        freqs_chunk = freqs_cis[start:end]
        if rot_d == out.shape[-1]:
            out[..., start:end, :] = apply_rope1(out[..., start:end, :], freqs_chunk).to(out.dtype)
        else:
            out[..., start:end, :rot_d] = apply_rope1(out[..., start:end, :rot_d], freqs_chunk).to(
                out.dtype
            )
    return out


class NaMMRotaryEmbedding3d(MMRotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)

    def forward(
        self,
        vid_q: torch.Tensor,  # L h d
        vid_k: torch.Tensor,  # L h d
        vid_shape: torch.Tensor,  # B 3
        txt_q: torch.Tensor,  # L h d
        txt_k: torch.Tensor,  # L h d
        txt_shape: torch.Tensor,  # B 1
        cache: Cache,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        get_freqs = cast(
            Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
            self.get_freqs,
        )
        vid_freqs, txt_freqs = cache(
            "mmrope_freqs_3d",
            lambda: get_freqs(vid_shape, txt_shape),
        )
        target_device = vid_q.device
        if vid_freqs.device != target_device:
            vid_freqs = vid_freqs.to(target_device)
        if txt_freqs.device != target_device:
            txt_freqs = txt_freqs.to(target_device)
        vid_q = vid_q.transpose(0, 1)
        vid_k = vid_k.transpose(0, 1)
        vid_q = _apply_rope1_partial(vid_q, vid_freqs)
        vid_k = _apply_rope1_partial(vid_k, vid_freqs)
        vid_q = vid_q.transpose(0, 1)
        vid_k = vid_k.transpose(0, 1)

        txt_q = txt_q.transpose(0, 1)
        txt_k = txt_k.transpose(0, 1)
        txt_q = _apply_rope1_partial(txt_q, txt_freqs)
        txt_k = _apply_rope1_partial(txt_k, txt_freqs)
        txt_q = txt_q.transpose(0, 1)
        txt_k = txt_k.transpose(0, 1)
        return vid_q, vid_k, txt_q, txt_k

    # .tolist() is data-dependent and causes graph breaks.
    @torch.compiler.disable
    def get_freqs(
        self,
        vid_shape: torch.Tensor,
        txt_shape: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        max_temporal = 0
        max_height = 0
        max_width = 0
        max_txt_len = 0

        for (f, h, w), text_length in zip(
            vid_shape.tolist(), txt_shape[:, 0].tolist(), strict=True
        ):
            max_temporal = max(max_temporal, text_length + f)
            max_height = max(max_height, h)
            max_width = max(max_width, w)
            max_txt_len = max(max_txt_len, text_length)

        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(autocast_device, enabled=False):
            vid_freqs = self.get_axial_freqs(
                max_temporal + 16,
                max_height + 4,
                max_width + 4,
                device=vid_shape.device,
            ).float()
            txt_freqs = self.get_axial_freqs(max_txt_len + 16, device=txt_shape.device)

        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), text_length in zip(
            vid_shape.tolist(), txt_shape[:, 0].tolist(), strict=True
        ):
            vid_freq = vid_freqs[text_length : text_length + f, :h, :w].reshape(
                -1, vid_freqs.size(-1)
            )
            txt_freq = txt_freqs[:text_length].repeat(1, 3).reshape(-1, vid_freqs.size(-1))
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        vid_freqs_interleaved = torch.cat(vid_freq_list, dim=0)
        txt_freqs_interleaved = torch.cat(txt_freq_list, dim=0)

        return _to_flux_freqs_cis(vid_freqs_interleaved), _to_flux_freqs_cis(txt_freqs_interleaved)


class MMModule(nn.Module):
    def __init__(
        self,
        module: Callable[..., nn.Module],
        *args: Any,
        shared_weights: bool = False,
        vid_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        self.all: nn.Module | None = None
        self.vid: nn.Module | None = None
        self.txt: nn.Module | None = None
        if self.shared_weights:
            if get_args("vid", args) != get_args("txt", args):
                raise ValueError("SeedVR2 shared MMModule requires matching vid/txt args.")
            if get_kwargs("vid", kwargs) != get_kwargs("txt", kwargs):
                raise ValueError("SeedVR2 shared MMModule requires matching vid/txt kwargs.")
            self.all = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
        else:
            self.vid = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
            self.txt = (
                module(*get_args("txt", args), **get_kwargs("txt", kwargs))
                if not vid_only
                else None
            )

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        vid_module = self.vid if not self.shared_weights else self.all
        assert vid_module is not None
        vid = vid_module(vid, *get_args("vid", args), **get_kwargs("vid", kwargs))
        if not self.vid_only:
            txt_module = self.txt if not self.shared_weights else self.all
            assert txt_module is not None
            txt = txt.to(device=vid.device, dtype=vid.dtype)
            txt = txt_module(txt, *get_args("txt", args), **get_kwargs("txt", kwargs))
        return vid, txt


def get_na_rope(
    rope_type: str | None, dim: int
) -> NaRotaryEmbedding3d | NaMMRotaryEmbedding3d | None:
    if rope_type is None:
        return None
    if rope_type == "rope3d":
        return NaRotaryEmbedding3d(dim=dim)
    if rope_type == "mmrope3d":
        return NaMMRotaryEmbedding3d(dim=dim)
    raise ValueError(f"Unknown SeedVR2 rope type: {rope_type}")


class NaMMAttention(nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm: Callable[..., nn.Module],
        qk_norm_eps: float,
        rope_type: str | None,
        rope_dim: int,
        shared_weights: bool,
        attention_kernel: AttentionKernel,
        device: Device,
        dtype: torch.dtype | None,
        operations: _ComfyOperations,
    ) -> None:
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.heads = heads
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            operations.Linear,
            dim,
            qkv_dim,
            bias=qk_bias,
            shared_weights=shared_weights,
            device=device,
            dtype=dtype,
        )
        self.proj_out = MMModule(
            operations.Linear,
            inner_dim,
            dim,
            shared_weights=shared_weights,
            device=device,
            dtype=dtype,
        )
        self.norm_q = MMModule(
            qk_norm,
            normalized_shape=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
            device=device,
            dtype=dtype,
        )
        self.norm_k = MMModule(
            qk_norm,
            normalized_shape=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
            device=device,
            dtype=dtype,
        )

        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)
        object.__setattr__(self, "_attention_kernel", attention_kernel)


def window(
    hid: torch.Tensor,  # (L c)
    hid_shape: torch.Tensor,  # (b n)
    window_fn: Callable[[torch.Tensor], list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    hid_parts = unflatten(hid, hid_shape)
    nested_windows = list(map(window_fn, hid_parts))
    hid_windows_list = [len(x) for x in nested_windows]
    hid_windows = torch.as_tensor(hid_windows_list, device=hid_shape.device)
    hid_windows_flat = list(chain(*nested_windows))
    hid_len_list = [math.prod(x.shape[:-1]) for x in hid_windows_flat]
    flat_hid, flat_shape = flatten(hid_windows_flat)
    return flat_hid, flat_shape, hid_windows, hid_len_list, hid_windows_list


def window_idx(
    hid_shape: torch.Tensor,  # (b n)
    window_fn: Callable[[torch.Tensor], list[torch.Tensor]],
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
]:
    hid_idx = torch.arange(int(hid_shape.prod(-1).sum()), device=hid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape, tgt_windows, tgt_len_list, tgt_windows_list = window(
        hid_idx, hid_shape, window_fn
    )
    tgt_idx = tgt_idx.squeeze(-1)
    src_idx = torch.argsort(tgt_idx)

    def partition(value: torch.Tensor) -> torch.Tensor:
        return torch.index_select(value, 0, tgt_idx)

    def reverse(value: torch.Tensor) -> torch.Tensor:
        return torch.index_select(value, 0, src_idx)

    return (
        partition,
        reverse,
        tgt_shape,
        tgt_windows,
        tgt_len_list,
        tgt_windows_list,
    )


class NaSwinAttention(NaMMAttention):
    def __init__(
        self,
        *args: Any,
        window: int | tuple[int, int, int],
        window_method: str,
        version: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.version_7b = version
        self.window = _as_triple(window)
        self.window_method = window_method
        if not all(v >= 0 for v in self.window):
            raise ValueError(
                f"SeedVR2 window must contain non-negative integers, got {self.window}."
            )

        self.window_op = get_window_op(window_method)

    def forward(
        self,
        vid: torch.Tensor,  # l c
        txt: torch.Tensor,  # l c
        vid_shape: torch.Tensor,  # b 3
        txt_shape: torch.Tensor,  # b 1
        cache: Cache,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)

        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor) -> list[torch.Tensor]:
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        (
            window_partition,
            window_reverse,
            window_shape,
            window_count,
            vid_len_win_list,
            window_count_list,
        ) = cache_win(
            "win_transform",
            lambda: window_idx(vid_shape, make_window),
        )
        vid_qkv_win = window_partition(vid_qkv)

        vid_qkv_win = vid_qkv_win.reshape(vid_qkv_win.shape[0], 3, self.heads, self.head_dim)
        txt_qkv = txt_qkv.reshape(txt_qkv.shape[0], 3, self.heads, self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len = txt_len.to(window_count.device)

        if self.rope is not None:
            if self.version_7b:
                assert isinstance(self.rope, NaRotaryEmbedding3d)
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)
            elif isinstance(self.rope, NaMMRotaryEmbedding3d):
                _, num_h, _ = txt_q.shape
                txt_q_repeat = txt_q.flatten(1, 2)
                txt_q_repeat = unflatten(txt_q_repeat, txt_shape)
                txt_q_repeat = [
                    [x] * n for x, n in zip(txt_q_repeat, window_count_list, strict=True)
                ]
                txt_q_repeat = list(chain(*txt_q_repeat))
                txt_q_repeat, txt_shape_repeat = flatten(txt_q_repeat)
                txt_q_repeat = txt_q_repeat.reshape(txt_q_repeat.shape[0], num_h, self.head_dim)

                txt_k_repeat = txt_k.flatten(1, 2)
                txt_k_repeat = unflatten(txt_k_repeat, txt_shape)
                txt_k_repeat = [
                    [x] * n for x, n in zip(txt_k_repeat, window_count_list, strict=True)
                ]
                txt_k_repeat = list(chain(*txt_k_repeat))
                txt_k_repeat, _ = flatten(txt_k_repeat)
                txt_k_repeat = txt_k_repeat.reshape(txt_k_repeat.shape[0], num_h, self.head_dim)

                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q,
                    vid_k,
                    window_shape,
                    txt_q_repeat,
                    txt_k_repeat,
                    txt_shape_repeat,
                    cache_win,
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)

        def build_txt_len_win_list() -> list[int]:
            return [
                txt_len
                for txt_len, window_count in zip(txt_len.tolist(), window_count_list, strict=True)
                for _ in range(window_count)
            ]

        txt_len_win_list = cache_win("txt_len_list", build_txt_len_win_list)

        def build_all_len_win() -> list[int]:
            return [
                vid_len + txt_len
                for vid_len, txt_len in zip(vid_len_win_list, txt_len_win_list, strict=True)
            ]

        all_len_win = cache_win("all_len", build_all_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: repeat_concat_idx(vid_len_win, txt_len, window_count)
        )
        out = optimized_var_attention(
            q=concat_win(vid_q, txt_q),
            k=concat_win(vid_k, txt_k),
            v=concat_win(vid_v, txt_v),
            heads=self.heads,
            skip_reshape=True,
            skip_output_reshape=True,
            cu_seqlens_q=cache_win("vid_seqlens_q", lambda: cumulative_lengths(all_len_win)),
            cu_seqlens_k=cache_win("vid_seqlens_k", lambda: cumulative_lengths(all_len_win)),
            attention_kernel=self._attention_kernel,
        )
        vid_out, txt_out = unconcat_win(out)

        vid_out = vid_out.flatten(1, 2)
        txt_out = txt_out.flatten(1, 2)
        vid_out = window_reverse(vid_out)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        device: Device,
        dtype: torch.dtype | None,
        operations: _ComfyOperations,
    ) -> None:
        super().__init__()
        self.proj_in = operations.Linear(dim, dim * expand_ratio, device=device, dtype=dtype)
        self.act = nn.GELU("tanh")
        self.proj_out = operations.Linear(dim * expand_ratio, dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        multiple_of: int = 256,
        device: Device = None,
        dtype: torch.dtype | None = None,
        operations: _ComfyOperations | None = None,
    ) -> None:
        super().__init__()
        operations = cast(_ComfyOperations, operations)
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = operations.Linear(
            dim, hidden_dim, bias=False, device=device, dtype=dtype
        )
        self.proj_out = operations.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.proj_in = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))


def get_mlp(mlp_type: str | None = "normal") -> type[MLP] | type[SwiGLUMLP]:
    if mlp_type == "normal":
        return MLP
    if mlp_type == "swiglu":
        return SwiGLUMLP
    raise ValueError(f"Unknown SeedVR2 MLP type: {mlp_type}")


class NaMMSRTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: Callable[..., nn.Module],
        norm_eps: float,
        ada: type[AdaSingle],
        qk_bias: bool,
        qk_norm: Callable[..., nn.Module],
        mlp_type: str,
        shared_weights: bool,
        rope_type: str,
        rope_dim: int,
        is_last_layer: bool,
        window: int | tuple[int, int, int],
        window_method: str,
        version: bool,
        attention_kernel: AttentionKernel,
        device: Device,
        dtype: torch.dtype | None,
        operations: _ComfyOperations,
    ) -> None:
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(
            norm,
            normalized_shape=dim,
            eps=norm_eps,
            elementwise_affine=False,
            shared_weights=shared_weights,
            device=device,
            dtype=dtype,
        )

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            rope_type=rope_type,
            rope_dim=rope_dim,
            shared_weights=shared_weights,
            window=window,
            window_method=window_method,
            version=version,
            attention_kernel=attention_kernel,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        self.mlp_norm = MMModule(
            norm,
            normalized_shape=dim,
            eps=norm_eps,
            elementwise_affine=False,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
            device=device,
            dtype=dtype,
        )
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.ada = MMModule(
            ada,
            dim=dim,
            emb_dim=emb_dim,
            layers=["attn", "mlp"],
            shared_weights=shared_weights,
            vid_only=is_last_layer,
            device=device,
            dtype=dtype,
        )
        self.is_last_layer = is_last_layer
        self.version = version

    def _seedvr2_7b_mlp(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        vid_module = self.mlp.vid if not self.mlp.shared_weights else self.mlp.all
        assert vid_module is not None
        if torch.is_grad_enabled() and vid.requires_grad:
            vid = torch.cat(
                [vid_module(chunk) for chunk in vid.split(SEEDVR2_7B_MLP_CHUNK, dim=0)], dim=0
            )
        else:
            vid_out: torch.Tensor | None = None
            offset = 0
            for chunk in vid.split(SEEDVR2_7B_MLP_CHUNK, dim=0):
                chunk_out = vid_module(chunk)
                if vid_out is None:
                    vid_out = chunk_out.new_empty((vid.shape[0], *chunk_out.shape[1:]))
                assert vid_out is not None
                vid_out[offset : offset + chunk_out.shape[0]] = chunk_out
                offset += chunk_out.shape[0]
            assert vid_out is not None
            vid = vid_out
        if not self.mlp.vid_only:
            txt_module = self.mlp.txt if not self.mlp.shared_weights else self.mlp.all
            assert txt_module is not None
            txt = txt.to(device=vid.device, dtype=vid.dtype)
            txt = txt_module(txt)
        return vid, txt

    def forward(
        self,
        vid: torch.Tensor,  # l c
        txt: torch.Tensor,  # l c
        vid_shape: torch.Tensor,  # b 3
        txt_shape: torch.Tensor,  # b 1
        emb: torch.Tensor,
        cache: Cache,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hid_len = MMArg(
            cache("vid_len", lambda: vid_shape.prod(-1)),
            cache("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        if self.version:
            vid_mlp, txt_mlp = self._seedvr2_7b_mlp(vid_mlp, txt_mlp)
        else:
            vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp, vid_shape, txt_shape


class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: int | tuple[int, int, int],
        dim: int,
        device: Device,
        dtype: torch.dtype | None,
        operations: _ComfyOperations,
    ) -> None:
        super().__init__()
        t, h, w = _as_triple(patch_size)
        self.patch_size = t, h, w
        self.proj = operations.Linear(dim, out_channels * t * h * w, device=device, dtype=dtype)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = self.proj(vid)
        b, T, H, W, channels = vid.shape
        c = channels // (t * h * w)
        vid = (
            vid.view(b, T, H, W, t, h, w, c)
            .permute(0, 7, 1, 4, 2, 5, 3, 6)
            .reshape(b, c, T * t, H * h, W * w)
        )
        if t > 1:
            vid = vid[:, :, (t - 1) :]
        return vid


class NaPatchOut(PatchOut):
    def forward(  # type: ignore[override]
        self,
        vid: torch.Tensor,  # l c
        vid_shape: torch.Tensor,
        cache: Cache | None = None,
        vid_shape_before_patchify: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if cache is None:
            cache = Cache(disable=True)

        t, h, w = self.patch_size
        vid = self.proj(vid)

        if not (t == h == w == 1):
            vid_parts = unflatten(vid, vid_shape)
            assert vid_shape_before_patchify is not None
            for i in range(len(vid_parts)):
                T, H, W, channels = vid_parts[i].shape
                c = channels // (t * h * w)
                vid_parts[i] = (
                    vid_parts[i]
                    .view(T, H, W, t, h, w, c)
                    .permute(0, 3, 1, 4, 2, 5, 6)
                    .reshape(T * t, H * h, W * w, c)
                )
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid_parts[i] = vid_parts[i][(t - vid_shape_before_patchify[i, 0] % t) :]
            vid, vid_shape = flatten(vid_parts)

        return vid, vid_shape


class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int | tuple[int, int, int],
        dim: int,
        device: Device,
        dtype: torch.dtype | None,
        operations: _ComfyOperations,
    ) -> None:
        super().__init__()
        t, h, w = _as_triple(patch_size)
        self.patch_size = t, h, w
        self.proj = operations.Linear(in_channels * t * h * w, dim, device=device, dtype=dtype)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if t > 1:
            if vid.size(2) % t != 1:
                raise ValueError(
                    "SeedVR2 patch input temporal size must satisfy "
                    f"T % {t} == 1, got {vid.size(2)}."
                )
            vid = torch.cat([vid[:, :, :1]] * (t - 1) + [vid], dim=2)
        b, c, Tt, Hh, Ww = vid.shape
        vid = (
            vid.view(b, c, Tt // t, t, Hh // h, h, Ww // w, w)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(b, Tt // t, Hh // h, Ww // w, t * h * w * c)
        )
        vid = self.proj(vid)
        return vid


class NaPatchIn(PatchIn):
    def forward(  # type: ignore[override]
        self,
        vid: torch.Tensor,  # l c
        vid_shape: torch.Tensor,
        cache: Cache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache is None:
            cache = Cache(disable=True)
        cache = cache.namespace("patch")
        vid_shape_before_patchify = cache("vid_shape_before_patchify", lambda: vid_shape)
        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid_parts = unflatten(vid, vid_shape)
            for i in range(len(vid_parts)):
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid_parts[i] = torch.cat(
                        [vid_parts[i][:1]] * (t - vid_parts[i].size(0) % t) + [vid_parts[i]],
                        dim=0,
                    )
                Tt, Hh, Ww, c = vid_parts[i].shape
                vid_parts[i] = (
                    vid_parts[i]
                    .view(Tt // t, t, Hh // h, h, Ww // w, w, c)
                    .permute(0, 2, 4, 1, 3, 5, 6)
                    .reshape(Tt // t, Hh // h, Ww // w, t * h * w * c)
                )
            vid, vid_shape = flatten(vid_parts)

        vid = self.proj(vid)
        return vid, vid_shape


def expand_dims(x: torch.Tensor, dim: int, ndim: int) -> torch.Tensor:
    shape = x.shape
    shape = shape[:dim] + (1,) * (ndim - len(shape)) + shape[dim:]
    return x.reshape(shape)


class AdaSingle(ResidencyRouted, nn.Module):
    def __init__(
        self,
        dim: int,
        emb_dim: int,
        layers: list[str],
        modes: tuple[str, ...] = ("in", "out"),
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if emb_dim != 6 * dim:
            raise ValueError(
                f"SeedVR2 AdaSingle requires emb_dim == 6 * dim, got emb_dim={emb_dim}, dim={dim}."
            )
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = layers

        for layer in layers:
            if "in" in modes:
                self.register_parameter(
                    f"{layer}_shift",
                    nn.Parameter(torch.empty(dim, device=device, dtype=dtype)),
                )
                self.register_parameter(
                    f"{layer}_scale",
                    nn.Parameter(torch.empty(dim, device=device, dtype=dtype)),
                )
            if "out" in modes:
                self.register_parameter(
                    f"{layer}_gate",
                    nn.Parameter(torch.empty(dim, device=device, dtype=dtype)),
                )

    def forward(
        self,
        hid: torch.Tensor,  # b ... c
        emb: torch.Tensor,  # b d
        layer: str,
        mode: str,
        cache: Cache | None = None,
        branch_tag: str = "",
        hid_len: torch.Tensor | None = None,  # b
    ) -> torch.Tensor:
        if cache is None:
            cache = Cache(disable=True)
        idx = self.layers.index(layer)
        emb = emb.reshape(emb.shape[0], -1, len(self.layers), 3)[:, :, idx, :]
        emb = expand_dims(emb, 1, hid.ndim + 1)

        if hid_len is not None:
            emb = cache(
                f"emb_repeat_{idx}_{branch_tag}",
                lambda: torch.repeat_interleave(emb, hid_len, dim=0),
            )

        shiftA, scaleA, gateA = emb.unbind(-1)
        names = (f"{layer}_shift", f"{layer}_scale", f"{layer}_gate")
        binding = self._offloaded_residency()

        def apply(
            shiftB: torch.Tensor | None,
            scaleB: torch.Tensor | None,
            gateB: torch.Tensor | None,
        ) -> torch.Tensor:
            if mode == "in":
                if shiftB is None or scaleB is None:
                    raise ValueError(f"SeedVR2 AdaSingle layer {layer!r} has no input modulation")
                return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
            if mode == "out":
                return hid.mul_(gateA if gateB is None else gateA + gateB)
            raise ValueError(f"Unknown AdaSingle mode: {mode}")

        if binding is not None:
            with binding.lease() as lease:
                values = tuple(
                    lease.get(name, dtype=hid.dtype)
                    if isinstance(getattr(self, name, None), nn.Parameter)
                    else None
                    for name in names
                )
                return apply(*values)

        values = tuple(
            cast_weight(value, dtype=hid.dtype, device=hid.device)
            if isinstance((value := getattr(self, name, None)), nn.Parameter)
            else None
            for name in names
        )
        return apply(*values)


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: Device,
        dtype: torch.dtype | None,
        operations: _ComfyOperations,
    ) -> None:
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = operations.Linear(sinusoidal_dim, hidden_dim, device=device, dtype=dtype)
        self.proj_hid = operations.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        self.proj_out = operations.Linear(hidden_dim, output_dim, device=device, dtype=dtype)
        self.act = nn.SiLU()

    def forward(
        self,
        timestep: int | float | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]

        emb = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        ).to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        emb = self.proj_out(emb)
        return emb


def flatten(
    hid: list[torch.Tensor],  # List of (*** c)
) -> tuple[
    torch.Tensor,  # (L c)
    torch.Tensor,  # (b n)
]:
    if len(hid) == 0:
        raise ValueError("SeedVR2 flatten requires at least one tensor.")
    shape = torch.as_tensor([x.shape[:-1] for x in hid], device=hid[0].device)
    flat_hid = torch.cat([x.flatten(0, -2) for x in hid])
    return flat_hid, shape


def unflatten(
    hid: torch.Tensor,  # (L c) or (L ... c)
    hid_shape: torch.Tensor,  # (b n)
) -> list[torch.Tensor]:  # List of (*** c) or (*** ... c)
    hid_len = hid_shape.prod(-1)
    hid_parts = hid.split(hid_len.tolist())
    return [x.unflatten(0, s.tolist()) for x, s in zip(hid_parts, hid_shape, strict=True)]


class NaDiT(ResidencyRouted, nn.Module):
    positive_conditioning: torch.Tensor
    negative_conditioning: torch.Tensor

    def __init__(
        self,
        norm_eps: float,
        num_layers: int,
        mlp_type: str,
        vid_in_channels: int = 33,
        vid_out_channels: int = SEEDVR2_LATENT_CHANNELS,
        vid_dim: int = 2560,
        txt_in_dim: int = 5120,
        heads: int = 20,
        head_dim: int = 128,
        mm_layers: int | Sequence[bool] = 10,
        expand_ratio: int = 4,
        qk_bias: bool = False,
        patch_size: int | tuple[int, int, int] = (1, 2, 2),
        rope_dim: int = 128,
        rope_type: str = "mmrope3d",
        vid_out_norm: str | None = None,
        image_model: str | None = None,
        device: Device = None,
        dtype: torch.dtype | None = None,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        if image_model not in (None, "seedvr2"):
            raise ValueError(f"SeedVR2 NaDiT expected image_model='seedvr2', got {image_model!r}.")
        self._7b_version = vid_dim == SEEDVR2_7B_VID_DIM
        if self._7b_version:
            rope_type = "rope3d"
        self.dtype = dtype
        window_method = num_layers // 2 * ["720pwin_by_size_bysize", "720pswin_by_size_bysize"]
        txt_dim = vid_dim
        emb_dim = vid_dim * 6
        window: list[tuple[int, int, int]] = num_layers * [(4, 3, 3)]
        ada = AdaSingle
        comfy_operations = _ComfyOperations(operations)
        norm = comfy_operations.RMSNorm
        qk_norm = comfy_operations.RMSNorm
        super().__init__()
        self.register_buffer(
            "positive_conditioning", torch.empty((58, 5120), device=device, dtype=dtype)
        )
        self.register_buffer(
            "negative_conditioning", torch.empty((64, 5120), device=device, dtype=dtype)
        )
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
            device=device,
            dtype=dtype,
            operations=comfy_operations,
        )
        self.txt_in = (
            comfy_operations.Linear(txt_in_dim, txt_dim, device=device, dtype=dtype)
            if txt_in_dim and txt_in_dim != txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=BYTEDANCE_SINUSOIDAL_DIM,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
            device=device,
            dtype=dtype,
            operations=comfy_operations,
        )

        self.blocks = nn.ModuleList(
            [
                NaMMSRTransformerBlock(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_norm=qk_norm,
                    mlp_type=mlp_type,
                    rope_dim=rope_dim,
                    window=window[i],
                    window_method=window_method[i],
                    version=self._7b_version,
                    is_last_layer=(i == num_layers - 1) and not self._7b_version,
                    rope_type=rope_type,
                    shared_weights=not (
                        (i < mm_layers) if isinstance(mm_layers, int) else mm_layers[i]
                    ),
                    attention_kernel=attention_kernel,
                    operations=comfy_operations,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ]
        )
        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
            device=device,
            dtype=dtype,
            operations=comfy_operations,
        )

        self.vid_out_norm: nn.RMSNorm | None = None
        if vid_out_norm is not None:
            self.vid_out_norm = comfy_operations.RMSNorm(
                normalized_shape=vid_dim,
                eps=norm_eps,
                elementwise_affine=True,
                device=device,
                dtype=dtype,
            )
            self.vid_out_ada = ada(
                dim=vid_dim,
                emb_dim=emb_dim,
                layers=["out"],
                modes=("in",),
                device=device,
                dtype=dtype,
            )

    @contextmanager
    def materialized_text_conditioning(
        self,
        branch: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Generator[torch.Tensor, None, None]:
        if branch not in ("positive", "negative"):
            raise ValueError(
                f"SeedVR2 conditioning branch must be positive or negative, got {branch!r}."
            )
        name = f"{branch}_conditioning"
        stored = cast(torch.Tensor, getattr(self, name))
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                yield lease.get(name, dtype=dtype)
            return
        yield stored.to(device=device, dtype=dtype)

    def _resolve_text_conditioning(
        self, context: torch.Tensor | None, cond_or_uncond: Sequence[object] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is None or context.numel() == 0:
            context = self.positive_conditioning
            return flatten([context])
        if NaDiT._seedvr2_is_single_conditioning_branch(cond_or_uncond):
            if context.shape[0] == 1:
                context = context.squeeze(0)
                return flatten([context])
            return flatten(list(context.unbind(0)))
        if context.shape[0] % 2 != 0:
            raise ValueError(
                "SeedVR2 expected an even text-conditioning batch, got shape "
                f"{tuple(context.shape)}"
            )
        neg_cond, pos_cond = context.chunk(2, dim=0)
        if pos_cond.shape[0] == 1:
            pos_cond, neg_cond = pos_cond.squeeze(0), neg_cond.squeeze(0)
            return flatten([pos_cond, neg_cond])
        return flatten([*pos_cond.unbind(0), *neg_cond.unbind(0)])

    @staticmethod
    def _seedvr2_is_single_conditioning_branch(
        cond_or_uncond: Sequence[object] | None,
    ) -> bool:
        if cond_or_uncond is None or len(cond_or_uncond) == 0:
            return False
        first = cond_or_uncond[0]
        return all(entry == first for entry in cond_or_uncond)

    @staticmethod
    def _check_seedvr2_video_latent(x: torch.Tensor, channels: int, name: str) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"SeedVR2 expected {name} to be 5-D native latent, got shape {tuple(x.shape)}."
            )
        if x.shape[1] != channels:
            raise ValueError(
                f"SeedVR2 expected {name} channels to be {channels}, got shape {tuple(x.shape)}."
            )
        return x

    def _swap_pos_neg_halves(
        self, out: torch.Tensor, cond_or_uncond: Sequence[object] | None = None
    ) -> torch.Tensor:
        if NaDiT._seedvr2_is_single_conditioning_branch(cond_or_uncond):
            return out
        pos, neg = out.chunk(2, dim=0)
        return torch.cat([neg, pos], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None,  # l c
        disable_cache: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        transformer_options = cast(dict[str, Any], kwargs.get("transformer_options", {}))
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        conditions_value = kwargs.get("condition")
        if not isinstance(conditions_value, torch.Tensor):
            raise ValueError(
                "SeedVR2 requires conditioning latents from the SeedVR2Conditioning node."
            )
        x = self._check_seedvr2_video_latent(x, SEEDVR2_LATENT_CHANNELS, "latent")
        conditions = self._check_seedvr2_video_latent(
            conditions_value, SEEDVR2_LATENT_CHANNELS + 1, "conditioning"
        )
        b, _, t, h, w = x.shape
        if conditions.shape[0] != b or conditions.shape[2:] != (t, h, w):
            raise ValueError(
                "SeedVR2 conditioning shape must match latent batch/temporal/spatial "
                f"dimensions; got latent {tuple(x.shape)} and conditioning "
                f"{tuple(conditions.shape)}."
            )
        x = x.movedim(1, -1)
        conditions = conditions.movedim(1, -1)
        cache = Cache(disable=disable_cache)

        txt, txt_shape = self._resolve_text_conditioning(
            context, transformer_options.get("cond_or_uncond")
        )

        vid, vid_shape = flatten(list(x.unbind(0)))
        cond_latent, _ = flatten(list(conditions.unbind(0)))

        vid = torch.cat([vid, cond_latent], dim=-1)

        txt = self.txt_in(txt)

        vid_shape_before_patchify = vid_shape
        vid, vid_shape = self.vid_in(vid, vid_shape, cache=cache)

        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        for i, block in enumerate(self.blocks):
            if ("block", i) in blocks_replace:

                def block_wrap(
                    args: dict[str, Any], selected_block: nn.Module = block
                ) -> dict[str, torch.Tensor]:
                    out: dict[str, torch.Tensor] = {}
                    out["vid"], out["txt"], out["vid_shape"], out["txt_shape"] = selected_block(
                        vid=args["vid"],
                        txt=args["txt"],
                        vid_shape=args["vid_shape"],
                        txt_shape=args["txt_shape"],
                        emb=args["emb"],
                        cache=args["cache"],
                    )
                    return out

                out = blocks_replace[("block", i)](
                    {
                        "vid": vid,
                        "txt": txt,
                        "vid_shape": vid_shape,
                        "txt_shape": txt_shape,
                        "emb": emb,
                        "cache": cache,
                    },
                    {"original_block": block_wrap},
                )
                vid, txt, vid_shape, txt_shape = (
                    out["vid"],
                    out["txt"],
                    out["vid_shape"],
                    out["txt_shape"],
                )
            else:
                vid, txt, vid_shape, txt_shape = block(
                    vid=vid,
                    txt=txt,
                    vid_shape=vid_shape,
                    txt_shape=txt_shape,
                    emb=emb,
                    cache=cache,
                )

        if self.vid_out_norm:
            vid = self.vid_out_norm(vid)
            vid = self.vid_out_ada(
                vid,
                emb=emb,
                layer="out",
                mode="in",
                hid_len=cache("vid_len", lambda: vid_shape.prod(-1)),
                cache=cache,
                branch_tag="vid",
            )

        vid, vid_shape = self.vid_out(
            vid, vid_shape, cache, vid_shape_before_patchify=vid_shape_before_patchify
        )
        vid = unflatten(vid, vid_shape)
        out = torch.stack(vid)
        out = out.movedim(-1, 1)
        return self._swap_pos_neg_halves(out, transformer_options.get("cond_or_uncond"))
