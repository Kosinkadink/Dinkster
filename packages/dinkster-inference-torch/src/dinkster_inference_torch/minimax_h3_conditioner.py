"""Direct-import MiniMax H3 Qwen3-VL-32B layer-50 conditioner source."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from dinkster_inference.minimax_h3_conditioner import MINIMAX_H3_CONDITIONER_CONFIG

from .attention import AttentionKernel, select_attention
from .minimax_h3_conditioning import MiniMaxH3ConditionerInputs
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations, bound_compute_device
from .quant_linear import linear_input_act
from .qwen_image_text import QwenImageLanguageModel, QwenImageVisionAttention

_DEFAULT_ATTENTION = select_attention("qwen").kernel


class _PatchEmbed(torch.nn.Module):
    def __init__(self, hidden_size: int, patch: tuple[int, int, int], operations: Operations):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch = patch
        self.proj = operations.conv3d(3, hidden_size, patch, stride=patch, bias=True)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        width = 3 * math.prod(self.patch)
        if patches.ndim != 2 or patches.shape[1] != width:
            raise ValueError(f"MiniMax H3 vision patches must be [patches, {width}]")
        return self.proj(patches.reshape(-1, 3, *self.patch)).reshape(-1, self.hidden_size)


class _VisionMlp(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, operations: Operations):
        super().__init__()
        self.linear_fc1 = operations.linear(hidden_size, intermediate_size)
        self.linear_fc2 = operations.linear(intermediate_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return linear_input_act(self.linear_fc2, self.linear_fc1(hidden), "gelu_tanh")


class _VisionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        heads: int,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.norm1 = operations.layer_norm(hidden_size, eps=1e-6)
        self.norm2 = operations.layer_norm(hidden_size, eps=1e-6)
        self.attn = QwenImageVisionAttention(
            hidden_size=hidden_size,
            num_heads=heads,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.mlp = _VisionMlp(hidden_size, intermediate_size, operations)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cumulative_lengths: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), position_embeddings, cumulative_lengths)
        return hidden + self.mlp(self.norm2(hidden))


class _Merger(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        merge_size: int,
        *,
        postshuffle_norm: bool,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.merge_dim = hidden_size * merge_size**2
        norm_size = self.merge_dim if postshuffle_norm else hidden_size
        self.norm = operations.layer_norm(norm_size, eps=1e-6)
        self.linear_fc1 = operations.linear(self.merge_dim, self.merge_dim)
        self.linear_fc2 = operations.linear(self.merge_dim, output_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.norm.normalized_shape == (self.merge_dim,):
            hidden = self.norm(hidden.reshape(-1, self.merge_dim))
        else:
            hidden = self.norm(hidden).reshape(-1, self.merge_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden)))


@dataclass(frozen=True)
class _VisionShape:
    hidden_size: int
    output_size: int
    intermediate_size: int
    heads: int
    layers: int
    patch: tuple[int, int, int]
    merge_size: int
    position_embeddings: int
    deepstack_layers: tuple[int, ...]


class MiniMaxH3VisionModel(torch.nn.Module):
    def __init__(
        self,
        *,
        operations: Operations = INITLESS,
        position_operations: Operations | None = None,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
        _shape: _VisionShape | None = None,
    ) -> None:
        super().__init__()
        position_operations = operations if position_operations is None else position_operations
        config = MINIMAX_H3_CONDITIONER_CONFIG
        shape = _shape or _VisionShape(
            config.vision_hidden_size,
            config.hidden_size,
            config.vision_intermediate_size,
            config.vision_heads,
            config.vision_layers,
            (config.vision_temporal_patch, config.vision_patch_size, config.vision_patch_size),
            config.vision_merge_size,
            config.vision_position_embeddings,
            config.deepstack_layers,
        )
        position_side = math.isqrt(shape.position_embeddings)
        if shape.hidden_size % shape.heads or position_side**2 != shape.position_embeddings:
            raise ValueError("MiniMax H3 vision configuration is inconsistent")
        self.shape = shape
        self.patch_embed = _PatchEmbed(shape.hidden_size, shape.patch, operations)
        self.pos_embed = position_operations.embedding(shape.position_embeddings, shape.hidden_size)
        self.blocks = torch.nn.ModuleList(
            _VisionBlock(
                shape.hidden_size,
                shape.intermediate_size,
                shape.heads,
                operations,
                attention_kernel,
            )
            for _ in range(shape.layers)
        )
        self.merger = _Merger(
            shape.hidden_size,
            shape.output_size,
            shape.merge_size,
            postshuffle_norm=False,
            operations=operations,
        )
        self.deepstack_merger_list = torch.nn.ModuleList(
            _Merger(
                shape.hidden_size,
                shape.output_size,
                shape.merge_size,
                postshuffle_norm=True,
                operations=operations,
            )
            for _ in shape.deepstack_layers
        )

    @classmethod
    def reduced(
        cls,
        *,
        hidden_size: int,
        output_size: int,
        intermediate_size: int,
        heads: int,
        layers: int,
        patch: tuple[int, int, int],
        merge_size: int,
        position_embeddings: int,
        deepstack_layers: tuple[int, ...],
        operations: Operations = INITLESS,
        position_operations: Operations | None = None,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> MiniMaxH3VisionModel:
        return cls(
            operations=operations,
            position_operations=position_operations,
            attention_kernel=attention_kernel,
            _shape=_VisionShape(
                hidden_size,
                output_size,
                intermediate_size,
                heads,
                layers,
                patch,
                merge_size,
                position_embeddings,
                deepstack_layers,
            ),
        )

    def _validate(self, patches: torch.Tensor, grid: torch.Tensor) -> list[tuple[int, int, int]]:
        width = 3 * math.prod(self.shape.patch)
        if patches.ndim != 2 or patches.shape[1] != width:
            raise ValueError(f"MiniMax H3 vision patches must be [patches, {width}]")
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
            raise ValueError("MiniMax H3 vision grid must be non-empty [items, 3]")
        if grid.dtype == torch.bool or grid.is_floating_point():
            raise ValueError("MiniMax H3 vision grid must use an integer dtype")
        rows = [(int(row[0]), int(row[1]), int(row[2])) for row in grid.tolist()]
        if any(value <= 0 for row in rows for value in row):
            raise ValueError("MiniMax H3 vision grid dimensions must be positive")
        if any(
            height % self.shape.merge_size or width % self.shape.merge_size
            for _, height, width in rows
        ):
            raise ValueError("MiniMax H3 vision grid must divide by the merge size")
        if sum(math.prod(row) for row in rows) != patches.shape[0]:
            raise ValueError("MiniMax H3 vision grid must account for every patch")
        return rows

    def _position_values(
        self, rows: list[tuple[int, int, int]], device: torch.device
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        side = math.isqrt(self.shape.position_embeddings)
        embedding_chunks: list[torch.Tensor] = []
        rotary_chunks: list[torch.Tensor] = []
        lengths: list[int] = []
        rotary_dim = (self.shape.hidden_size // self.shape.heads) // 2
        inverse = 1.0 / (
            10_000.0
            ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
        )
        max_side = max(max(height, width) for _, height, width in rows)
        frequency_table = torch.outer(
            torch.arange(max_side, device=device, dtype=torch.float32), inverse
        )
        for time, height, width in rows:
            h = torch.linspace(0, side - 1, height, device=device)
            w = torch.linspace(0, side - 1, width, device=device)
            h0, w0 = h.floor().long(), w.floor().long()
            h1, w1 = (h0 + 1).clamp(max=side - 1), (w0 + 1).clamp(max=side - 1)
            dh, dw = h - h0, w - w0
            indices = (
                h0[:, None] * side + w0[None, :],
                h0[:, None] * side + w1[None, :],
                h1[:, None] * side + w0[None, :],
                h1[:, None] * side + w1[None, :],
            )
            lookups = tuple(self.pos_embed(index.flatten()) for index in indices)
            weights = tuple(
                weight.to(lookups[0].dtype)
                for weight in (
                    (1 - dh)[:, None] * (1 - dw)[None, :],
                    (1 - dh)[:, None] * dw[None, :],
                    dh[:, None] * (1 - dw)[None, :],
                    dh[:, None] * dw[None, :],
                )
            )
            corners = tuple(
                lookup * weight.flatten()[:, None]
                for lookup, weight in zip(lookups, weights, strict=True)
            )
            position = corners[0] + corners[1] + corners[2] + corners[3]
            row_ids = torch.arange(height, device=device)[:, None].expand(-1, width)
            col_ids = torch.arange(width, device=device)[None, :].expand(height, -1)
            coords = torch.stack((row_ids, col_ids), -1).reshape(-1, 2)
            rotary = frequency_table[coords].flatten(1)
            merge = self.shape.merge_size
            position = (
                position.repeat(time, 1)
                .reshape(time, height // merge, merge, width // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            rotary = (
                rotary.repeat(time, 1)
                .reshape(time, height // merge, merge, width // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            embedding_chunks.append(position)
            rotary_chunks.append(torch.cat((rotary, rotary), -1))
            lengths.extend([height * width] * time)
        positions = torch.cat(embedding_chunks)
        rotary = torch.cat(rotary_chunks)
        cumulative = F.pad(
            torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0),
            (1, 0),
            value=0,
        )
        return positions, (rotary.cos(), rotary.sin()), cumulative

    def forward(
        self, patches: torch.Tensor, grid: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        rows = self._validate(patches, grid)
        hidden = self.patch_embed(patches)
        positions, rotary, cumulative = self._position_values(rows, hidden.device)
        hidden = hidden + positions.to(hidden)
        deepstack: list[torch.Tensor] = []
        queue = make_prefetch_queue(self.blocks)
        try:
            for index, block in enumerate(self.blocks):
                prefetch_queue_pop(queue, block)
                hidden = block(hidden, rotary, cumulative)
                if index in self.shape.deepstack_layers:
                    merger = self.deepstack_merger_list[self.shape.deepstack_layers.index(index)]
                    deepstack.append(merger(hidden))
            prefetch_queue_pop(queue, None)
            return self.merger(hidden), tuple(deepstack)
        finally:
            close_prefetch_queue(queue)


class MiniMaxH3ConditionerModel(torch.nn.Module):
    def __init__(
        self,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
        language: QwenImageLanguageModel | None = None,
        visual: MiniMaxH3VisionModel | None = None,
    ) -> None:
        super().__init__()
        config = MINIMAX_H3_CONDITIONER_CONFIG
        self.model = language or QwenImageLanguageModel.reduced(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=config.layers,
            num_heads=config.attention_heads,
            num_kv_heads=config.key_value_heads,
            rope_dims=config.rope_dimensions,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            qkv_bias=False,
            qk_norm=True,
            final_norm=False,
            interleaved_mrope=True,
            max_position_embeddings=config.max_position_embeddings,
            architecture="MiniMax H3",
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.visual = visual or MiniMaxH3VisionModel(
            operations=operations,
            attention_kernel=attention_kernel,
        )

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
        image_patches: torch.Tensor | None = None,
        image_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("MiniMax H3 conditioner IDs must be one non-empty row")
        if ids.dtype == torch.bool or ids.is_floating_point():
            raise ValueError("MiniMax H3 conditioner IDs must use an integer dtype")
        if attention_mask is not None and attention_mask.shape != ids.shape:
            raise ValueError("MiniMax H3 attention mask must match IDs")
        if position_ids is not None and (
            position_ids.dtype == torch.bool or position_ids.is_floating_point()
        ):
            raise ValueError("MiniMax H3 position IDs must use an integer dtype")
        if visual_mask is not None and visual_mask.dtype != torch.bool:
            raise ValueError("MiniMax H3 visual mask must use bool dtype")
        if (image_patches is None) != (image_grid is None):
            raise ValueError("MiniMax H3 vision patches and grid must be provided together")
        if image_patches is not None and (
            position_ids is None or position_ids.shape != (3, ids.shape[1])
        ):
            raise ValueError("MiniMax H3 vision inputs require three-axis position IDs")
        weight = self.model.embed_tokens.weight
        device = bound_compute_device(self.model.embed_tokens) or weight.device
        ids = ids.to(device)
        attention_mask = None if attention_mask is None else attention_mask.to(device)
        position_ids = None if position_ids is None else position_ids.to(device)
        visual_mask = None if visual_mask is None else visual_mask.to(device)
        embeds = self.model.embed(ids).float()
        deepstack: tuple[torch.Tensor, ...] = ()
        if image_patches is not None:
            if visual_mask is None or visual_mask.shape != ids.shape:
                raise ValueError("MiniMax H3 visual mask must match IDs")
            assert image_grid is not None
            visual, deepstack = self.visual(
                image_patches.to(device=embeds.device, dtype=torch.float32),
                image_grid.to(device=embeds.device),
            )
            if visual.shape != (int(visual_mask.count_nonzero()), embeds.shape[-1]):
                raise ValueError("MiniMax H3 vision output must match visual placeholders")
            embeds = embeds.clone()
            embeds[visual_mask] = visual.to(embeds)
        elif visual_mask is not None:
            raise ValueError("MiniMax H3 visual mask requires vision inputs")
        if position_ids is None:
            position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        return self.model.forward_embeds(
            embeds,
            attention_mask,
            position_ids,
            deepstack_features=deepstack,
            visual_mask=visual_mask,
        )

    def encode(self, inputs: MiniMaxH3ConditionerInputs) -> torch.Tensor:
        """Encode one fully validated H3 token/vision realization."""
        return self(
            inputs.ids,
            position_ids=inputs.position_ids,
            visual_mask=inputs.visual_mask if inputs.patches is not None else None,
            image_patches=inputs.patches,
            image_grid=inputs.grids,
        )


__all__ = ["MiniMaxH3ConditionerModel", "MiniMaxH3VisionModel"]
