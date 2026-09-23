"""MiniMax H3 Fun ControlNet-Union model patch.

Ports the official ComfyUI H3 Fun control surface (commit 95539f563449):
v1 checkpoints place one control block per ten base blocks, Union 2.0
checkpoints place ten control blocks evenly across the fifty base blocks,
and v2 checkpoints declare post-normalization inpaint masking so source
holes are filled at the pixel the VAE normalizes to zero instead of black.
"""

from __future__ import annotations

import ctypes
import json
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import torch
from blake3 import blake3
from dinkster_inference import ControlApplication
from dinkster_inference.minimax_h3_codecs import IMAGENET_MEAN
from dinkster_inference.minimax_h3_dit import MiniMaxH3TimeEmbeddingKind

from .attention import AttentionKernel
from .minimax_h3_dit import (
    MiniMaxH3AttentionProviderEvidence,
    MiniMaxH3BlockShapes,
    MiniMaxH3ControlBlockContext,
    _MiniMaxH3Block,  # pyright: ignore[reportPrivateUsage]
    _PackedLayout,  # pyright: ignore[reportPrivateUsage]
    _patchify_video,  # pyright: ignore[reportPrivateUsage]
)
from .operations import INITLESS, Operations

# Every video-kind packed row carries this many control columns per token:
# 49 latent channels over the (1, 2, 2) video patch.
CONTROL_IN_DIM = 49
_FUN_CONTROL_PATCH = (1, 2, 2)
_BASE_BLOCK_COUNT = 50

_STATE_DICT_PROBES = (
    "control_proj_in.weight",
    "control_blocks.0.adaln_proj.linear.weight",
    "control_blocks.0.after_proj.weight",
    "control_blocks.0.before_proj.weight",
    "control_blocks.0.attn.qkv_proj.weight",
    "control_blocks.0.attn.q_norm.weight",
    "control_blocks.0.mlp.fc1.weight",
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3FunControlShapes:
    """Checkpoint-derived block widths for one Fun control model."""

    hidden_width: int
    attention_heads: int
    attention_head_dim: int
    ffn_width: int


class ControlDiTBlock(_MiniMaxH3Block):
    """One control block: the base H3 block plus its residual projections."""

    def __init__(
        self,
        shapes: MiniMaxH3BlockShapes,
        attention_kernel: AttentionKernel,
        evidence: MiniMaxH3AttentionProviderEvidence,
        *,
        operations: Operations,
        fp32_operations: Operations,
        rotary_dim: int,
        apply_silu: bool,
        first_block: bool,
    ) -> None:
        super().__init__(
            shapes,
            attention_kernel,
            evidence,
            operations=operations,
            fp32_operations=fp32_operations,
            rotary_dim=rotary_dim,
            apply_silu=apply_silu,
        )
        if first_block:
            self.before_proj = operations.linear(shapes.hidden_width, shapes.hidden_width)
        self.after_proj = operations.linear(shapes.hidden_width, shapes.hidden_width)


def is_minimax_h3_fun_state_dict(state_dict: Mapping[str, object]) -> bool:
    """Detect the H3 Fun control checkpoint layout by its key set."""
    return all(key in state_dict for key in _STATE_DICT_PROBES)


def minimax_h3_fun_injection_layers(
    num_blocks: int,
    control_blocks_places: str | None = None,
) -> tuple[int, ...]:
    """Derive control injection layers from the checkpoint block count.

    Without metadata the blocks spread evenly over the fifty base blocks:
    v1 checkpoints carry five blocks (every tenth base block) and Union 2.0
    carries ten (every fifth). The "control_blocks_places" metadata, when
    present, names the exact placement instead and must match the checkpoint.
    """
    if type(num_blocks) is not int or num_blocks < 1:
        raise ValueError("MiniMax H3 Fun control checkpoint must contain at least one block")
    if control_blocks_places is None:
        return tuple(range(0, _BASE_BLOCK_COUNT, _BASE_BLOCK_COUNT // num_blocks))
    injection_layers = tuple(json.loads(control_blocks_places))
    if len(injection_layers) != num_blocks:
        raise ValueError(
            "MiniMax H3 Fun control_blocks_places metadata does not match the checkpoint"
        )
    return injection_layers


def minimax_h3_fun_inpaint_post_norm(metadata: Mapping[str, str] | None) -> bool:
    """Whether the checkpoint masks source video after VAE normalization."""
    if metadata is None:
        return False
    return metadata.get("inpaint_masked_pixel_mode") == "post_norm"


def minimax_h3_inpaint_mask_fill(
    source: torch.Tensor,
    visibility: torch.Tensor,
    post_norm: bool,
) -> torch.Tensor:
    """Mask the source video for control encoding.

    Visible pixels keep the source; hidden pixels fall back to black, or to
    the ImageNet-mean pixel when the checkpoint masks after VAE
    normalization, so holes encode to zero instead of a black-video code.
    """
    visibility = visibility.to(device=source.device)
    masked = source * visibility
    if post_norm:
        masked = masked + (1.0 - visibility) * torch.tensor(
            IMAGENET_MEAN, dtype=source.dtype, device=source.device
        ).view(1, 3, 1, 1)
    return masked


def minimax_h3_fun_control_hint_digest(hint: torch.Tensor) -> str:
    """Content digest for one canonical H3 Fun control latent."""
    if type(hint) is not torch.Tensor:
        raise TypeError("MiniMax H3 Fun control hint must be an exact torch.Tensor")
    if hint.layout is not torch.strided:
        raise TypeError("MiniMax H3 Fun control hint must use strided tensor storage")
    value = hint.detach().resolve_conj().resolve_neg().contiguous().cpu()
    raw: bytes | bytearray = ctypes.string_at(
        value.data_ptr(), value.numel() * value.element_size()
    )
    width = value.element_size()
    if sys.byteorder == "big" and width > 1:
        raw = bytearray(raw)
        for start in range(0, len(raw), width):
            raw[start : start + width] = reversed(raw[start : start + width])
    hasher = blake3()
    hasher.update(b"dinkster.minimax-h3-fun-control-hint.v1\n")
    hasher.update(f"shape={','.join(str(dim) for dim in value.shape)}\n".encode("ascii"))
    hasher.update(f"dtype={str(value.dtype).removeprefix('torch.')}\n".encode("ascii"))
    hasher.update(b"byte_order=little\n\n")
    hasher.update(raw)
    return hasher.hexdigest()


class MiniMaxH3FunControl(torch.nn.Module):
    """H3 Fun ControlNet-Union model patch for v1 and v2 checkpoints."""

    def __init__(
        self,
        shapes: MiniMaxH3FunControlShapes,
        attention_kernel: AttentionKernel,
        evidence: MiniMaxH3AttentionProviderEvidence,
        *,
        injection_layers: tuple[int, ...],
        inpaint_post_norm: bool = False,
        operations: Operations = INITLESS,
        fp32_operations: Operations | None = None,
        time_embedding_kind: MiniMaxH3TimeEmbeddingKind = "curve",
    ) -> None:
        super().__init__()
        if not injection_layers or injection_layers[0] != 0:
            raise ValueError("MiniMax H3 Fun control injection layers must start at layer 0")
        if injection_layers != tuple(sorted(set(injection_layers))):
            raise ValueError(
                "MiniMax H3 Fun control injection layers must be unique and increasing"
            )
        fp32_operations = operations if fp32_operations is None else fp32_operations
        self.shapes = shapes
        self.injection_layers = tuple(injection_layers)
        self.inpaint_post_norm = inpaint_post_norm
        rotary_dim = min(96, shapes.attention_head_dim // 6 * 6)
        if rotary_dim < 6:
            raise ValueError("MiniMax H3 attention head dimension must support three RoPE axes")
        patch_dim = CONTROL_IN_DIM * math.prod(_FUN_CONTROL_PATCH)
        self.control_proj_in = fp32_operations.linear(patch_dim, shapes.hidden_width)
        adaln_operations = fp32_operations if time_embedding_kind == "curve" else operations
        self.control_blocks = torch.nn.ModuleList(
            ControlDiTBlock(
                shapes,
                attention_kernel,
                evidence,
                operations=operations,
                fp32_operations=adaln_operations,
                rotary_dim=rotary_dim,
                apply_silu=time_embedding_kind == "mlp",
                first_block=index == 0,
            )
            for index in range(len(self.injection_layers))
        )

    def init_stream(
        self,
        hidden: torch.Tensor,
        control_latent: torch.Tensor,
        layout: _PackedLayout,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Open the control stream from the first injected base block's input."""
        first_block = cast("ControlDiTBlock", self.control_blocks[0])
        adaln_in = first_block.adaln_proj.linear.in_features  # pyright: ignore[reportPrivateUsage]
        if time_embedding.shape[-1] != adaln_in:
            raise RuntimeError(
                f"MiniMax H3 controlnet adaln width {adaln_in} does not match the base model's "
                f"timestep embedding width {time_embedding.shape[-1]}: the controlnet and "
                "base checkpoint use different adaln forms (curve basis vs full), convert "
                "the controlnet to match the base model."
            )
        patch_dim = CONTROL_IN_DIM * math.prod(_FUN_CONTROL_PATCH)
        pad_t = (-control_latent.shape[2]) % _FUN_CONTROL_PATCH[0]
        pad_h = (-control_latent.shape[3]) % _FUN_CONTROL_PATCH[1]
        pad_w = (-control_latent.shape[4]) % _FUN_CONTROL_PATCH[2]
        latent = torch.nn.functional.pad(
            control_latent.to(torch.float32), (0, pad_w, 0, pad_h, 0, pad_t)
        )
        target_rows = _patchify_video(latent, _FUN_CONTROL_PATCH)
        if target_rows.shape[1] < patch_dim:
            target_rows = torch.nn.functional.pad(
                target_rows, (0, patch_dim - target_rows.shape[1])
            )
        elif target_rows.shape[1] > patch_dim:
            raise ValueError(
                "MiniMax H3 control input has "
                f"{target_rows.shape[1]} columns but the model patch expects {patch_dim}"
            )
        video_update = layout.video_update.to(hidden.device)
        rows = torch.zeros(
            video_update.shape[0], patch_dim, dtype=torch.float32, device=hidden.device
        )
        rows[video_update] = target_rows.to(hidden.device)
        stream = hidden.clone()
        stream[0, _video_row_positions(layout)] = self.control_proj_in(rows).to(hidden.dtype)
        return first_block.before_proj(stream) + hidden

    def step(
        self,
        index: int,
        stream: torch.Tensor,
        time_embedding: torch.Tensor,
        segments: tuple[tuple[int, int, int | torch.Tensor], ...],
        rope_table: torch.Tensor,
        *,
        attention_kernel: AttentionKernel | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the control stream and return it with this block's residual."""
        block = cast("ControlDiTBlock", self.control_blocks[index])
        stream = block(
            stream, time_embedding, segments, rope_table, attention_kernel=attention_kernel
        )
        return stream, block.after_proj(stream)


def _video_row_positions(layout: _PackedLayout) -> torch.Tensor:
    """Absolute packed-row indices of every video-kind segment."""
    positions = [
        torch.arange(start, stop, device=layout.video_update.device)
        for start, stop, kind in layout.segments
        if kind in ("condition", "reference_video", "video")
    ]
    return torch.cat(positions)


class MiniMaxH3FunControlPatch:
    """Runtime control application for one prepared Fun control latent.

    Implements the DiT control seam: the control stream opens from the first
    injected base block's input, each control block consumes the base
    block's modulation context, and its residual is added to the controlled
    hidden state everywhere except audio rows.
    """

    def __init__(
        self,
        model: MiniMaxH3FunControl,
        control_latent: torch.Tensor,
        strength: float,
    ) -> None:
        if type(strength) is not float or not math.isfinite(strength) or strength < 0.0:
            raise ValueError("MiniMax H3 Fun control strength must be a non-negative finite float")
        self.model = model
        self.control_latent = control_latent
        self.strength = strength
        self._control_stream: torch.Tensor | None = None
        self._pristine_stream: torch.Tensor | None = None

    def before_base_block(self, hidden: torch.Tensor, block_index: int) -> None:
        # Stash only: control weight loads here would clobber the base
        # block's freshly staged weights.
        if block_index == self.model.injection_layers[0]:
            self._pristine_stream = hidden.clone()

    def after_base_block(
        self,
        hidden: torch.Tensor,
        block_index: int,
        context: MiniMaxH3ControlBlockContext,
    ) -> torch.Tensor:
        layers = self.model.injection_layers
        if block_index not in layers:
            return hidden
        control_index = layers.index(block_index)
        if control_index == 0:
            if self._pristine_stream is None:
                raise RuntimeError("MiniMax H3 Fun control stream opened without its base input")
            stream = self.model.init_stream(
                self._pristine_stream, self.control_latent, context.layout, context.time_embedding
            )
            self._pristine_stream = None
        else:
            if self._control_stream is None:
                raise RuntimeError("MiniMax H3 Fun control stream stepped out of order")
            stream = self._control_stream
        self._control_stream, skip = self.model.step(
            control_index,
            stream,
            context.time_embedding,
            context.segments,
            context.rope_table,
            attention_kernel=context.attention_kernel,
        )
        skip = _zero_audio_rows(skip, context.layout)
        return hidden + skip * self.strength


@dataclass(frozen=True, slots=True)
class MiniMaxH3FunControlConditioning:
    """Prepared H3 control input and its sampling window."""

    application: ControlApplication
    model: MiniMaxH3FunControl
    control_latent: torch.Tensor

    def __post_init__(self) -> None:
        if type(self.application) is not ControlApplication:
            raise TypeError("MiniMax H3 control application must be an exact ControlApplication")
        if type(self.model) is not MiniMaxH3FunControl:
            raise TypeError("MiniMax H3 control model must be an exact MiniMaxH3FunControl")
        if type(self.control_latent) is not torch.Tensor:
            raise TypeError("MiniMax H3 control latent must be an exact torch.Tensor")

    def patch_for_sigma(
        self, sigma: float, percent_to_sigma: Callable[[float], float]
    ) -> MiniMaxH3FunControlPatch | None:
        window = self.application.window
        start_sigma = percent_to_sigma(window.start_percent)
        end_sigma = percent_to_sigma(window.end_percent)
        if not end_sigma <= sigma <= start_sigma:
            return None
        return MiniMaxH3FunControlPatch(
            self.model,
            self.control_latent,
            float(self.application.strength),
        )


def _zero_audio_rows(skip: torch.Tensor, layout: _PackedLayout) -> torch.Tensor:
    # Audio rows never carry control: the residual keeps them at zero.
    keep = torch.ones(1, skip.shape[1], 1, dtype=skip.dtype, device=skip.device)
    for start, stop, kind in layout.segments:
        if kind in ("audio", "condition_audio", "reference_audio"):
            keep[:, start:stop] = 0.0
    return skip * keep


def load_minimax_h3_fun_control(
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str] | None,
    *,
    attention_kernel: AttentionKernel,
    evidence: MiniMaxH3AttentionProviderEvidence,
    operations: Operations = INITLESS,
    fp32_operations: Operations | None = None,
    time_embedding_kind: MiniMaxH3TimeEmbeddingKind = "curve",
) -> MiniMaxH3FunControl:
    """Strict-load one Fun control checkpoint into its model patch."""
    if not is_minimax_h3_fun_state_dict(state_dict):
        raise ValueError("state dict is not a MiniMax H3 Fun control checkpoint")
    num_blocks = 0
    while f"control_blocks.{num_blocks}.after_proj.weight" in state_dict:
        num_blocks += 1
    places = None if metadata is None else metadata.get("control_blocks_places")
    injection_layers = minimax_h3_fun_injection_layers(num_blocks, places)
    qkv = state_dict["control_blocks.0.attn.qkv_proj.weight"]
    head_dim = state_dict["control_blocks.0.attn.q_norm.weight"].shape[0]
    shapes = MiniMaxH3FunControlShapes(
        hidden_width=state_dict["control_proj_in.weight"].shape[0],
        attention_heads=qkv.shape[0] // (3 * head_dim),
        attention_head_dim=head_dim,
        ffn_width=state_dict["control_blocks.0.mlp.fc1.weight"].shape[0] // 2,
    )
    model = MiniMaxH3FunControl(
        shapes,
        attention_kernel,
        evidence,
        injection_layers=injection_layers,
        inpaint_post_norm=minimax_h3_fun_inpaint_post_norm(metadata),
        operations=operations,
        fp32_operations=fp32_operations,
        time_embedding_kind=time_embedding_kind,
    )
    model.load_state_dict(state_dict, strict=True)
    return model


__all__ = [
    "CONTROL_IN_DIM",
    "ControlDiTBlock",
    "MiniMaxH3FunControl",
    "MiniMaxH3FunControlConditioning",
    "MiniMaxH3FunControlPatch",
    "MiniMaxH3FunControlShapes",
    "is_minimax_h3_fun_state_dict",
    "load_minimax_h3_fun_control",
    "minimax_h3_fun_control_hint_digest",
    "minimax_h3_fun_injection_layers",
    "minimax_h3_fun_inpaint_post_norm",
    "minimax_h3_inpaint_mask_fill",
]
