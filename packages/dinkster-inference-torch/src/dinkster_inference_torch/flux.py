"""Flux diffusion transformer: the native torch architecture.

Faithful port of the reference classic-Flux model
(comfy/ldm/flux/model.py Flux / comfy/ldm/flux/layers.py
DoubleStreamBlock / SingleStreamBlock / Modulation / MLPEmbedder /
QKNorm / LastLayer / EmbedND / timestep_embedding and
comfy/ldm/flux/math.py rope / apply_rope @ b78cec87), constructed
from a torch-free :class:`~dinkster_inference.flux.FluxConfig` through
the typed :class:`~dinkster_inference_torch.operations.Operations` seam.
State-dict keys are IDENTICAL to the reference's bare BFL layout:
img_in.*, time_in.*, vector_in.*, guidance_in.*, txt_in.*,
double_blocks.N.*, single_blocks.N.*, final_layer.*.

Paired RoPE application capability-probes dinkster-kitchen's combined
operation (ComfyUI's pinned inference path, whose CUDA backends
contract the pair rotation into fused multiply-adds one ulp from the
reference), then the reference's pure-torch math. The single-input operation
uses dinkster-kitchen before the same fallback. These kernels do not register an
autograd formula (the reference routes training through the pure-torch path via
a global ``in_training`` flag); Dinkster gates on the autograd facts themselves:
inputs that require grad under an enabled grad mode take the pure-torch path.

Scope pins (everything else is ledgered in ROADMAP "Native
inference", never silently dropped):

- Classic Flux plus the row-28 context-normalized and gated variants:
  optional txt_norm applies RMS normalization before txt_in; optional
  yak_mlp selects gated double-block MLPs and packed SiLU-gated
  single-block MLPs. Header detection admits their composition only for
  the exact vector-free Ovis architecture, which constructs no vector
  embedder, needs no y, and advances text positions on axes 1 and 2.
- The Flux2 configuration (comfy/ldm/flux/model.py @ b78cec87 with
  global_modulation / mlp_silu_act / operations_bias False): shared
  double_stream_modulation_img/txt and single_stream_modulation
  computed once from the time+guidance vector (final_layer still
  receives the raw vector), SiLU-gated MLPs whose gate is the FIRST
  chunk half, bias-free linears everywhere except vector_in (which
  Flux2 does not construct), four rope axes with text positions on
  axis 3. Header detection lives in dinkster_inference.flux2 and admits
  only the exact frozen dev / Klein 9B / Klein 4B geometries.
  No Chroma distilled guidance and no unproven text-position
  semantics.
- Flux2 reference-latent conditioning uses the reference's default
  offset placement. The alternate index/uxo methods and timestep-zero
  modulation slicing are not exposed. No ControlNet residual injection,
  no attention masking
  (``attention_mask``), and no ``transformer_options``
  patch/wrapper seams. ``image_position_ids`` optionally supplies a
  validated declared global image grid for windowed evaluation; the
  ordinary forward path derives the same local grid as the reference.
- Attention runs through PyTorch SDPA with the reference's head
  layout and SDP_BATCH_LIMIT batch chunking
  (comfy/ldm/modules/attention.py attention_pytorch @ b78cec87).
- RoPE frequencies are computed in float64 on the input device, as
  the reference does everywhere fp64 exists; the MPS carve-out
  (supports_fp64) is not ported - CPU and CUDA are the validated
  targets.

Deliberate strictness beyond the reference: ``y`` is REQUIRED with
exactly ``vec_in_dim`` columns when the vector embedder exists (the
reference zero-fills a missing y and silently slices a wide one -
silent conditioning drift). The approved vector-free model follows
the reference's conditional exactly: no embedder is constructed and y
is not required or consumed. ``guidance`` is required exactly when
``guidance_embed`` is set (the reference silently skips the embedder
when guidance is None).

Deliberate preservation: no ``inference_mode``/``no_grad`` anywhere
(training program, docs/native-inference-plan.md 3.1); residual adds
and fp16 nan_to_num are out-of-place so autograd and torch.compile
see plain dataflow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from threading import Lock
from typing import NamedTuple, Protocol, cast

import torch
import torch.nn.functional as F
from dinkster_inference.flux import FluxConfig

from .attention import AttentionKernel, select_attention
from .model_prefetch import (
    close_prefetch_queue,
    make_prefetch_queue,
    prefetch_queue_pop,
)
from .operations import INITLESS, Operations

_DEFAULT_FLUX_ATTENTION = select_attention("flux").kernel

__all__ = [
    "DoubleStreamBlock",
    "EmbedND",
    "Flux",
    "LastLayer",
    "MLPEmbedder",
    "Modulation",
    "ModulationOut",
    "QKNorm",
    "SelfAttention",
    "SingleStreamBlock",
    "apply_rope",
    "apply_rope1",
    "flux_timestep_embedding",
    "rope",
]


class _RopeKernel(Protocol):
    def __call__(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class _Rope1Kernel(Protocol):
    def __call__(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor: ...


def _probe_kitchen_apply_rope() -> _RopeKernel | None:
    """Capability probe for the combined inference operation.
    Missing or incompatible kitchen -> pure torch (same degrade-never-
    break contract as rounding._probe_kitchen_fp8_kernel)."""
    try:
        import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

        kernel: _RopeKernel = dinkster_kitchen.apply_rope
        if not callable(kernel):
            return None
        return kernel
    except (AttributeError, ImportError):
        return None


_KITCHEN_ROPE_UNPROBED = object()
_ck_apply_rope: _RopeKernel | None | object = _KITCHEN_ROPE_UNPROBED
_ck_apply_rope_lock = Lock()


def _probe_kitchen_apply_rope1() -> _Rope1Kernel | None:
    try:
        import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

        kernel: _Rope1Kernel = dinkster_kitchen.apply_rope1
        if not callable(kernel):
            return None
        return kernel
    except (AttributeError, ImportError):
        return None


_ck_apply_rope1: _Rope1Kernel | None | object = _KITCHEN_ROPE_UNPROBED
_ck_apply_rope1_lock = Lock()


def _kitchen_apply_rope() -> _RopeKernel | None:
    global _ck_apply_rope
    if _ck_apply_rope is _KITCHEN_ROPE_UNPROBED:
        with _ck_apply_rope_lock:
            if _ck_apply_rope is _KITCHEN_ROPE_UNPROBED:
                _ck_apply_rope = _probe_kitchen_apply_rope()
    return cast(_RopeKernel, _ck_apply_rope) if callable(_ck_apply_rope) else None


def _kitchen_apply_rope1() -> _Rope1Kernel | None:
    global _ck_apply_rope1
    if _ck_apply_rope1 is _KITCHEN_ROPE_UNPROBED:
        with _ck_apply_rope1_lock:
            if _ck_apply_rope1 is _KITCHEN_ROPE_UNPROBED:
                _ck_apply_rope1 = _probe_kitchen_apply_rope1()
    return cast(_Rope1Kernel, _ck_apply_rope1) if callable(_ck_apply_rope1) else None


def flux_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
) -> torch.Tensor:
    """Flux's sinusoidal embedding (comfy/ldm/flux/layers.py
    timestep_embedding @ b78cec87): inputs are 0..1 flow timesteps
    scaled by ``time_factor``, laid out cos-then-sin, zero-padded for
    odd dim, and cast back to a floating input's dtype."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """Rotation matrices for one position axis
    (comfy/ldm/flux/math.py rope @ b78cec87): float64 frequency math,
    float32 result, laid out [..., n, dim/2, 2, 2]. MPS cannot hold
    float64 tensors, so there the frequency math runs on cpu (the
    reference's supports_fp64 fallback) and only the float32 result
    lands on pos.device."""
    if dim % 2:
        raise ValueError(f"rope dim must be even, got {dim}")
    device = torch.device("cpu") if pos.device.type == "mps" else pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
        dim=-1,
    )
    return out.unflatten(-1, (2, 2)).to(dtype=torch.float32, device=pos.device)


def _apply_rope1_torch(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """The reference pure-torch rotation (comfy/ldm/flux/math.py
    _apply_rope1 @ b78cec87, out-of-place); also exactly the kitchen
    kernel's eager backend."""
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = torch.addcmul(freqs_cis[..., 0] * x_[..., 0], freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(x.shape).type_as(x)


def _apply_rope_torch(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply_rope1_torch(xq, freqs_cis), _apply_rope1_torch(xk, freqs_cis)


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate one tensor through ComfyUI's single-input operation."""
    needs_grad = torch.is_grad_enabled() and (x.requires_grad or freqs_cis.requires_grad)
    kitchen_layout = x.ndim == 4 and freqs_cis.ndim == 6
    if kitchen_layout and not needs_grad and not torch.compiler.is_compiling():
        kitchen = _kitchen_apply_rope1()
        if kitchen is not None:
            if x.device.type == "cuda":
                with torch.cuda.device(x.device):
                    return kitchen(x, freqs_cis)
            return kitchen(x, freqs_cis)
    return _apply_rope1_torch(x, freqs_cis)


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k through the fastest eligible combined operation."""
    return apply_rope_comfy(xq, xk, freqs_cis)


def apply_rope_comfy(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k through ComfyUI's kitchen-first execution route."""
    needs_grad = torch.is_grad_enabled() and (
        xq.requires_grad or xk.requires_grad or freqs_cis.requires_grad
    )
    if not needs_grad and not torch.compiler.is_compiling():
        kitchen = _kitchen_apply_rope()
        if kitchen is not None:
            if xq.device.type == "cuda":
                with torch.cuda.device(xq.device):
                    return kitchen(xq, xk, freqs_cis)
            return kitchen(xq, xk, freqs_cis)
    return _apply_rope_torch(xq, xk, freqs_cis)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pe: torch.Tensor,
    attention_kernel: AttentionKernel,
    rope_kernel: _RopeKernel,
) -> torch.Tensor:
    """RoPE + SDPA over [batch, heads, seq, head_dim] inputs, output
    re-fused to [batch, seq, heads * head_dim]
    (comfy/ldm/flux/math.py attention -> attention_pytorch with
    skip_reshape @ b78cec87). The injected adapter owns the
    SDP_BATCH_LIMIT leg."""
    q, k = rope_kernel(q, k, pe)
    batch, heads, _, dim_head = q.shape
    out = attention_kernel(q, k, v)
    return out.transpose(1, 2).reshape(batch, -1, heads * dim_head)


class EmbedND(torch.nn.Module):
    """Per-axis RoPE frequencies concatenated over the id axes
    (comfy/ldm/flux/layers.py EmbedND @ b78cec87)."""

    def __init__(self, dim: int, theta: int, axes_dim: tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])],
            dim=-3,
        )
        return emb.unsqueeze(1)


class MLPEmbedder(torch.nn.Module):
    """Linear - SiLU - Linear vector embedder (keys in_layer /
    out_layer)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        bias: bool = True,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.in_layer = operations.linear(in_dim, hidden_dim, bias=bias)
        self.out_layer = operations.linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))


class QKNorm(torch.nn.Module):
    """Per-head RMS normalization of q and k, cast to v's dtype."""

    def __init__(self, dim: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.query_norm = operations.rms_norm(dim)
        self.key_norm = operations.rms_norm(dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SelfAttention(torch.nn.Module):
    """The qkv / norm / proj parameter bundle of one stream. The
    reference gives it no forward - DoubleStreamBlock drives the
    pieces so both streams share one attention call."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        proj_bias: bool = True,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = operations.linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(dim // num_heads, operations=operations)
        self.proj = operations.linear(dim, dim, bias=proj_bias)


class ModulationOut(NamedTuple):
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(torch.nn.Module):
    """SiLU + linear producing (shift, scale, gate) once or twice
    (key ``lin``). The reference pads the single case with a None
    second element; returning exactly the produced sets instead lets
    both call sites unpack without narrowing."""

    def __init__(
        self,
        dim: int,
        *,
        double: bool,
        bias: bool = True,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.linear(dim, self.multiplier * dim, bias=bias)

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ...]:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        out = self.lin(F.silu(vec)).chunk(self.multiplier, dim=-1)
        return tuple(ModulationOut(*out[i : i + 3]) for i in range(0, self.multiplier, 3))


def _modulate(x: torch.Tensor, mod: ModulationOut) -> torch.Tensor:
    """shift + x * (1 + scale), fused (the reference's apply_mod
    without modulation-dim slicing @ b78cec87)."""
    return torch.addcmul(mod.shift, x, 1 + mod.scale)


class _SiLUGate(torch.nn.Module):
    """chunk(2) -> silu(first) * second (the reference's
    SiLUActivation @ b78cec87; note the gate is the FIRST half, the
    opposite split order from the Ovis packed single-block MLP)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


def _mlp(
    hidden_size: int,
    mlp_hidden_dim: int,
    *,
    silu_act: bool = False,
    operations: Operations = INITLESS,
) -> torch.nn.Sequential:
    if silu_act:
        return torch.nn.Sequential(
            operations.linear(hidden_size, 2 * mlp_hidden_dim, bias=False),
            _SiLUGate(),
            operations.linear(mlp_hidden_dim, hidden_size, bias=False),
        )
    return torch.nn.Sequential(
        operations.linear(hidden_size, mlp_hidden_dim),
        torch.nn.GELU(approximate="tanh"),
        operations.linear(mlp_hidden_dim, hidden_size),
    )


class _YakMLP(torch.nn.Module):
    """SiLU-gated MLP used by Ovis Flux double-stream blocks."""

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_dim: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.gate_proj = operations.linear(hidden_size, mlp_hidden_dim)
        self.up_proj = operations.linear(hidden_size, mlp_hidden_dim)
        self.down_proj = operations.linear(mlp_hidden_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _heads_first(
    qkv: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[b, seq, 3*hidden] -> three [b, heads, seq, head_dim]."""
    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, num_heads, -1).permute(2, 0, 3, 1, 4)
    return q, k, v


class DoubleStreamBlock(torch.nn.Module):
    """Joint image/text attention with per-stream modulation, norms,
    and MLPs (comfy/ldm/flux/layers.py DoubleStreamBlock @ b78cec87).
    With ``modulation=False`` (Flux2's global-modulation
    configuration) the block owns no modulation modules and ``vec``
    is the precomputed ((img_mod1, img_mod2), (txt_mod1, txt_mod2))
    shared across all double blocks."""

    _attention_kernel: AttentionKernel
    _rope_kernel: _RopeKernel

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_hidden_dim: int,
        *,
        qkv_bias: bool,
        yak_mlp: bool = False,
        mlp_silu_act: bool = False,
        modulation: bool = True,
        ops_bias: bool = True,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_FLUX_ATTENTION,
        rope_kernel: _RopeKernel = apply_rope,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        object.__setattr__(self, "_rope_kernel", rope_kernel)

        def build_mlp() -> torch.nn.Module:
            if yak_mlp:
                return _YakMLP(hidden_size, mlp_hidden_dim, operations=operations)
            return _mlp(hidden_size, mlp_hidden_dim, silu_act=mlp_silu_act, operations=operations)

        self.img_mod: Modulation | None = (
            Modulation(hidden_size, double=True, operations=operations) if modulation else None
        )
        self.img_norm1 = operations.layer_norm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.img_attn = SelfAttention(
            hidden_size, num_heads, qkv_bias=qkv_bias, proj_bias=ops_bias, operations=operations
        )
        self.img_norm2 = operations.layer_norm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.img_mlp = build_mlp()
        self.txt_mod: Modulation | None = (
            Modulation(hidden_size, double=True, operations=operations) if modulation else None
        )
        self.txt_norm1 = operations.layer_norm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.txt_attn = SelfAttention(
            hidden_size, num_heads, qkv_bias=qkv_bias, proj_bias=ops_bias, operations=operations
        )
        self.txt_norm2 = operations.layer_norm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.txt_mlp = build_mlp()

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor | tuple[tuple[ModulationOut, ...], tuple[ModulationOut, ...]],
        pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(vec, torch.Tensor):
            if self.img_mod is None or self.txt_mod is None:
                raise ValueError("this block uses global modulation; pass precomputed mods")
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            if self.img_mod is not None:
                raise ValueError("this block owns its modulation; pass the modulation vector")
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        img_q, img_k, img_v = _heads_first(
            self.img_attn.qkv(_modulate(self.img_norm1(img), img_mod1)),
            self.num_heads,
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_q, txt_k, txt_v = _heads_first(
            self.txt_attn.qkv(_modulate(self.txt_norm1(txt), txt_mod1)),
            self.num_heads,
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v
        attn = _attention(
            q,
            k,
            v,
            pe,
            self._attention_kernel,
            self._rope_kernel,
        )
        del q, k, v
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        grad_enabled = torch.is_grad_enabled()
        update = img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + update if grad_enabled else img.add_(update)
        del img_attn, update
        update = img_mod2.gate * self.img_mlp(_modulate(self.img_norm2(img), img_mod2))
        img = img + update if grad_enabled else img.add_(update)
        del update
        update = txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + update if grad_enabled else txt.add_(update)
        del txt_attn, update
        update = txt_mod2.gate * self.txt_mlp(_modulate(self.txt_norm2(txt), txt_mod2))
        txt = txt + update if grad_enabled else txt.add_(update)
        del update
        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)
        return img, txt


class SingleStreamBlock(torch.nn.Module):
    """Fused-stream DiT block with parallel attention and MLP
    projections (comfy/ldm/flux/layers.py SingleStreamBlock
    @ b78cec87). With ``modulation=False`` (Flux2's
    global-modulation configuration) the block owns no modulation
    module and ``vec`` is the precomputed ModulationOut shared across
    all single blocks."""

    _attention_kernel: AttentionKernel
    _rope_kernel: _RopeKernel

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_hidden_dim: int,
        *,
        yak_mlp: bool = False,
        mlp_silu_act: bool = False,
        modulation: bool = True,
        ops_bias: bool = True,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_FLUX_ATTENTION,
        rope_kernel: _RopeKernel = apply_rope,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.yak_mlp = yak_mlp
        self.mlp_silu_act = mlp_silu_act
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        object.__setattr__(self, "_rope_kernel", rope_kernel)
        self.packed_mlp_dim = 2 * mlp_hidden_dim if yak_mlp or mlp_silu_act else mlp_hidden_dim
        self.linear1 = operations.linear(
            hidden_size, hidden_size * 3 + self.packed_mlp_dim, bias=ops_bias
        )
        self.linear2 = operations.linear(hidden_size + mlp_hidden_dim, hidden_size, bias=ops_bias)
        self.norm = QKNorm(hidden_size // num_heads, operations=operations)
        self.pre_norm = operations.layer_norm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.modulation: Modulation | None = (
            Modulation(hidden_size, double=False, operations=operations) if modulation else None
        )

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor | ModulationOut,
        pe: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(vec, torch.Tensor):
            if self.modulation is None:
                raise ValueError("this block uses global modulation; pass a precomputed mod")
            (mod,) = self.modulation(vec)
        else:
            if self.modulation is not None:
                raise ValueError("this block owns its modulation; pass the modulation vector")
            mod = vec
        qkv, mlp = torch.split(
            self.linear1(_modulate(self.pre_norm(x), mod)),
            [3 * self.hidden_size, self.packed_mlp_dim],
            dim=-1,
        )
        q, k, v = _heads_first(qkv, self.num_heads)
        del qkv
        q, k = self.norm(q, k, v)
        attn = _attention(q, k, v, pe, self._attention_kernel, self._rope_kernel)
        del q, k, v
        if self.yak_mlp:
            up, gate = mlp.split(self.mlp_hidden_dim, dim=-1)
            mlp = F.silu(gate) * up
        elif self.mlp_silu_act:
            gate, up = mlp.chunk(2, dim=-1)
            mlp = F.silu(gate) * up
        else:
            mlp = F.gelu(mlp, approximate="tanh")
        output = self.linear2(torch.cat((attn, mlp), 2))
        update = mod.gate * output
        x = x + update if torch.is_grad_enabled() else x.add_(update)
        del update
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


class LastLayer(torch.nn.Module):
    """adaLN-modulated output projection (keys norm_final / linear /
    adaLN_modulation.1)."""

    def __init__(
        self,
        hidden_size: int,
        out_features: int,
        *,
        bias: bool = True,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.norm_final = operations.layer_norm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = operations.linear(hidden_size, out_features, bias=bias)
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            operations.linear(hidden_size, 2 * hidden_size, bias=bias),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(torch.addcmul(shift, self.norm_final(x), 1 + scale))


class Flux(torch.nn.Module):
    """The classic Flux flow-matching transformer.

    ``forward(x, timesteps, context, y, guidance)``: x is the latent
    image [N x in_channels x H x W] (H/W need not be patch-size
    multiples: inputs are circularly padded and the output cropped,
    the reference's pad_to_patch_size behavior), timesteps the 0..1
    flow timestep per batch element, context the T5 token sequence
    [N x T x context_in_dim], y the CLIP pooled vector
    [N x vec_in_dim] when the config has a vector embedder (otherwise
    None), guidance the distilled-CFG strength per batch element
    (exactly when the config has ``guidance_embed``)."""

    def __init__(
        self,
        config: FluxConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_FLUX_ATTENTION,
    ) -> None:
        super().__init__()
        if len(config.axes_dim) not in (3, 4):
            raise ValueError(
                "Flux positions ids over (index, h, w) or Flux2's (index, h, w, txt);"
                f" got {len(config.axes_dim)} axes"
            )
        self.config = config
        hidden = config.hidden_size
        self.pe_embedder = EmbedND(
            dim=config.head_dim, theta=config.theta, axes_dim=config.axes_dim
        )
        patch = config.patch_size * config.patch_size
        self.img_in = operations.linear(config.in_channels * patch, hidden, bias=config.ops_bias)
        self.time_in = MLPEmbedder(256, hidden, bias=config.ops_bias, operations=operations)
        self.vector_in: MLPEmbedder | None = (
            MLPEmbedder(config.vec_in_dim, hidden, operations=operations)
            if config.vec_in_dim is not None
            else None
        )
        self.guidance_in: MLPEmbedder | None = (
            MLPEmbedder(256, hidden, bias=config.ops_bias, operations=operations)
            if config.guidance_embed
            else None
        )
        self.txt_norm: torch.nn.RMSNorm | None = (
            operations.rms_norm(config.context_in_dim) if config.txt_norm else None
        )
        self.txt_in = operations.linear(config.context_in_dim, hidden, bias=config.ops_bias)
        self.double_stream_modulation_img: Modulation | None = None
        self.double_stream_modulation_txt: Modulation | None = None
        self.single_stream_modulation: Modulation | None = None
        if config.global_modulation:
            self.double_stream_modulation_img = Modulation(
                hidden, double=True, bias=False, operations=operations
            )
            self.double_stream_modulation_txt = Modulation(
                hidden, double=True, bias=False, operations=operations
            )
            self.single_stream_modulation = Modulation(
                hidden, double=False, bias=False, operations=operations
            )
        self.double_blocks = torch.nn.ModuleList(
            DoubleStreamBlock(
                hidden,
                config.num_heads,
                config.mlp_hidden_dim,
                qkv_bias=config.qkv_bias,
                yak_mlp=config.yak_mlp,
                mlp_silu_act=config.mlp_silu_act,
                modulation=not config.global_modulation,
                ops_bias=config.ops_bias,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.depth)
        )
        self.single_blocks = torch.nn.ModuleList(
            SingleStreamBlock(
                hidden,
                config.num_heads,
                config.mlp_hidden_dim,
                yak_mlp=config.yak_mlp,
                mlp_silu_act=config.mlp_silu_act,
                modulation=not config.global_modulation,
                ops_bias=config.ops_bias,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.depth_single_blocks)
        )
        self.final_layer = LastLayer(
            hidden, config.out_channels * patch, bias=config.ops_bias, operations=operations
        )

    def _patchify(
        self,
        x: torch.Tensor,
        image_position_ids: torch.Tensor | None = None,
        *,
        index: float = 0.0,
        height_offset: int = 0,
        width_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad to patch multiples, tokenize, and build the (index, h,
        w[, txt]) position ids with one column per rope axis (the
        reference's process_img @ b78cec87; h/w live on axes 1/2,
        every other axis stays 0)."""
        batch, channels, height, width = x.shape
        axes = len(self.config.axes_dim)
        patch = self.config.patch_size
        pad_h = (patch - height % patch) % patch
        pad_w = (patch - width % patch) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        h_len = (height + (patch // 2)) // patch
        w_len = (width + (patch // 2)) // patch
        img = (
            x.view(batch, channels, h_len, patch, w_len, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, h_len * w_len, channels * patch * patch)
        )
        if image_position_ids is None:
            img_ids = torch.zeros((h_len, w_len, axes), device=x.device, dtype=torch.float32)
            img_ids[:, :, 0] = index
            img_ids[:, :, 1] = torch.linspace(
                height_offset,
                h_len - 1 + height_offset,
                steps=h_len,
                device=x.device,
                dtype=torch.float32,
            ).unsqueeze(1)
            img_ids[:, :, 2] = torch.linspace(
                width_offset,
                w_len - 1 + width_offset,
                steps=w_len,
                device=x.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            return img, img_ids.reshape(1, h_len * w_len, axes).expand(batch, -1, -1)
        if index != 0.0 or height_offset != 0 or width_offset != 0:
            raise ValueError("declared image positions cannot be combined with position offsets")
        if (
            type(image_position_ids) is not torch.Tensor
            or image_position_ids.shape != (batch, h_len * w_len, axes)
            or image_position_ids.dtype is not torch.float32
            or image_position_ids.device != x.device
        ):
            raise ValueError(
                "image_position_ids must be float32 on the input device with shape"
                f" ({batch}, {h_len * w_len}, {axes})"
            )
        grid_ids = image_position_ids.reshape(batch, h_len, w_len, axes)
        if torch.count_nonzero(grid_ids[..., 0]) or (
            axes > 3 and torch.count_nonzero(grid_ids[..., 3:])
        ):
            raise ValueError("image_position_ids non-spatial axes must be zero")
        if (
            not torch.isfinite(grid_ids).all()
            or not torch.equal(grid_ids, grid_ids.round())
            or torch.any(grid_ids < 0)
        ):
            raise ValueError("image_position_ids must contain non-negative exact integers")
        height_ids = grid_ids[:, :, :1, 1]
        width_ids = grid_ids[:, :1, :, 2]
        if not torch.equal(grid_ids[..., 1], height_ids.expand(-1, -1, w_len)) or not torch.equal(
            grid_ids[..., 2], width_ids.expand(-1, h_len, -1)
        ):
            raise ValueError("image_position_ids must be a row-major Cartesian product")
        if batch > 1 and not torch.equal(grid_ids, grid_ids[:1].expand_as(grid_ids)):
            raise ValueError("image_position_ids must declare one shared grid for the batch")
        return img, image_position_ids

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        ref_latents: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        config = self.config
        batch, _, height, width = x.shape
        if timesteps.shape != (batch,):
            # A (1,)-shaped timestep against a larger batch would
            # broadcast silently through the modulation vec.
            raise ValueError(
                f"timesteps must be [batch] = ({batch},), got {tuple(timesteps.shape)}"
            )
        if (
            context.ndim != 3
            or context.shape[0] != batch
            or (context.shape[2] != config.context_in_dim)
        ):
            raise ValueError(
                f"context must be [batch x tokens x context_in_dim] ="
                f" ({batch}, T, {config.context_in_dim}), got"
                f" {tuple(context.shape)}"
            )
        if self.vector_in is not None:
            vec_in_dim = config.vec_in_dim
            assert vec_in_dim is not None
            if y is None or y.shape != (batch, vec_in_dim):
                found = None if y is None else tuple(y.shape)
                raise ValueError(
                    f"y must be [batch x vec_in_dim] = ({batch}, {vec_in_dim}), got {found}"
                )
        if self.guidance_in is None and guidance is not None:
            raise ValueError("this Flux model has no guidance embedder; guidance must be None")
        if guidance is not None and guidance.shape != (batch,):
            raise ValueError(f"guidance must be [batch] = ({batch},), got {tuple(guidance.shape)}")

        img, img_ids = self._patchify(x, image_position_ids)
        img_tokens = img.shape[1]
        if ref_latents:
            if image_position_ids is not None:
                raise ValueError(
                    "reference latents cannot be combined with declared image positions"
                )
            accumulated_height = 0
            accumulated_width = 0
            patch = config.patch_size
            for reference in ref_latents:
                if (
                    reference.ndim != 4
                    or reference.shape[0] != batch
                    or reference.shape[1] != config.in_channels
                ):
                    raise ValueError(
                        "reference latents must be [batch x in_channels x height x width]"
                    )
                height_offset = 0
                width_offset = 0
                if reference.shape[-2] + accumulated_height > (
                    reference.shape[-1] + accumulated_width
                ):
                    width_offset = accumulated_width
                else:
                    height_offset = accumulated_height
                accumulated_height = max(accumulated_height, reference.shape[-2] + height_offset)
                accumulated_width = max(accumulated_width, reference.shape[-1] + width_offset)
                reference_img, reference_ids = self._patchify(
                    reference,
                    index=1.0,
                    height_offset=(height_offset + patch // 2) // patch,
                    width_offset=(width_offset + patch // 2) // patch,
                )
                img = torch.cat((img, reference_img), dim=1)
                img_ids = torch.cat((img_ids, reference_ids), dim=1)
        txt_ids = torch.zeros(
            (batch, context.shape[1], len(config.axes_dim)),
            device=x.device,
            dtype=torch.float32,
        )
        for index in config.txt_ids_dims:
            txt_ids[:, :, index] = torch.linspace(
                0,
                context.shape[1] - 1,
                steps=context.shape[1],
                device=x.device,
                dtype=torch.float32,
            )

        img = self.img_in(img)
        vec = self.time_in(flux_timestep_embedding(timesteps, 256).to(img.dtype))
        if self.guidance_in is not None and guidance is not None:
            vec = vec + self.guidance_in(flux_timestep_embedding(guidance, 256).to(img.dtype))
        if self.vector_in is not None:
            assert y is not None
            vec = vec + self.vector_in(y)
        txt = self.txt_in(self.txt_norm(context) if self.txt_norm is not None else context)

        double_vec: torch.Tensor | tuple[tuple[ModulationOut, ...], tuple[ModulationOut, ...]]
        single_vec: torch.Tensor | ModulationOut
        if config.global_modulation:
            assert self.double_stream_modulation_img is not None
            assert self.double_stream_modulation_txt is not None
            assert self.single_stream_modulation is not None
            double_vec = (
                self.double_stream_modulation_img(vec),
                self.double_stream_modulation_txt(vec),
            )
            (single_vec,) = self.single_stream_modulation(vec)
        else:
            double_vec = vec
            single_vec = vec

        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        double_prefetch = make_prefetch_queue(self.double_blocks)
        try:
            for block in self.double_blocks:
                prefetch_queue_pop(double_prefetch, block)
                img, txt = block(img, txt, double_vec, pe)
            prefetch_queue_pop(double_prefetch, None)
        finally:
            close_prefetch_queue(double_prefetch)
        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        tokens = torch.cat((txt, img), dim=1)
        single_prefetch = make_prefetch_queue(self.single_blocks)
        try:
            for block in self.single_blocks:
                prefetch_queue_pop(single_prefetch, block)
                tokens = block(tokens, single_vec, pe)
            prefetch_queue_pop(single_prefetch, None)
        finally:
            close_prefetch_queue(single_prefetch)
        img = tokens[:, txt.shape[1] : txt.shape[1] + img_tokens]

        out = self.final_layer(img, vec)
        patch = config.patch_size
        h_len = (height + (patch // 2)) // patch
        w_len = (width + (patch // 2)) // patch
        return (
            out.view(batch, h_len, w_len, config.out_channels, patch, patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, config.out_channels, h_len * patch, w_len * patch)[
                :, :, :height, :width
            ]
        )
