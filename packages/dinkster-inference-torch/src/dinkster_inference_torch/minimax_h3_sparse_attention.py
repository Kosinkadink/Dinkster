# pyright: reportPrivateUsage=false

"""Execution-scoped MiniMax H3 BlockSparseAttention producer."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from dinkster_inference import MINIMAX_H3_SIGMAS, MiniMaxH3SparseAttentionConfig

from .minimax_h3_attention import MiniMaxH3PackedSequenceFacts
from .minimax_h3_dit import MiniMaxH3Attention
from .operations import materialized_rms_norm_weight

_BLOCK_SIZE = 64
_PRODUCER_CHUNK = 4096
_VSA_CUBE = (4, 4, 4)


@dataclass(frozen=True, slots=True)
class _VSAPlan:
    source_rows: torch.Tensor
    inverse_rows: torch.Tensor
    block_lengths: torch.Tensor
    prefix_blocks: int

    @property
    def padded_rows(self) -> int:
        return int(self.source_rows.numel())


def _vsa_plan(facts: MiniMaxH3PackedSequenceFacts, device: torch.device) -> _VSAPlan:
    if facts.video_grid is None:
        raise ValueError("VSA requires the target video token grid")
    video_grid = facts.video_grid
    tiles: list[torch.Tensor] = []
    prefix_blocks = 0
    for start, stop, kind in facts.segments:
        rows = stop - start
        if kind != "video":
            blocks = math.ceil(rows / _BLOCK_SIZE)
            segment = torch.full((blocks * _BLOCK_SIZE,), -1, dtype=torch.int64, device=device)
            segment[:rows] = torch.arange(start, stop, device=device)
            tiles.append(segment.view(blocks, _BLOCK_SIZE))
            prefix_blocks += blocks
            continue
        padded_grid = tuple(
            math.ceil(size / cube) * cube for size, cube in zip(video_grid, _VSA_CUBE, strict=True)
        )
        padded = torch.full(padded_grid, -1, dtype=torch.int64, device=device)
        padded[: video_grid[0], : video_grid[1], : video_grid[2]] = torch.arange(
            start, stop, device=device
        ).view(video_grid)
        ct, ch, cw = _VSA_CUBE
        cubes = (
            padded.view(
                padded_grid[0] // ct,
                ct,
                padded_grid[1] // ch,
                ch,
                padded_grid[2] // cw,
                cw,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, _BLOCK_SIZE)
        )
        order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
        tiles.append(torch.gather(cubes, 1, order))
    tiled = torch.cat(tiles)
    source = tiled.reshape(-1)
    live = source >= 0
    inverse = torch.empty(facts.sequence_length, dtype=torch.int64, device=device)
    inverse[source[live]] = torch.nonzero(live).flatten()
    return _VSAPlan(
        source,
        inverse,
        (tiled >= 0).sum(1).to(torch.int32),
        prefix_blocks,
    )


class MiniMaxH3SparseAttention:
    """Sampling-run state shared by every H3 block and guidance lane."""

    def __init__(self, config: MiniMaxH3SparseAttentionConfig) -> None:
        if type(config) is not MiniMaxH3SparseAttentionConfig:
            raise TypeError("config must be an exact MiniMaxH3SparseAttentionConfig")
        self.config = config
        self._pooled: dict[tuple[int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._plans: dict[tuple[MiniMaxH3PackedSequenceFacts, torch.device], _VSAPlan] = {}
        self._logged: set[object] = set()

    def _log_once(self, key: object, message: str) -> None:
        if self.config.verbose and key not in self._logged:
            self._logged.add(key)
            logging.info("BlockSparseAttention: %s", message)

    def bind(
        self,
        *,
        sigma: float,
        lane: str,
        facts: MiniMaxH3PackedSequenceFacts,
    ) -> _BoundMiniMaxH3SparseAttention:
        return _BoundMiniMaxH3SparseAttention(self, sigma, lane, facts)


@dataclass(frozen=True, slots=True)
class _BoundMiniMaxH3SparseAttention:
    owner: MiniMaxH3SparseAttention
    sigma: float
    lane: str
    facts: MiniMaxH3PackedSequenceFacts

    def _eligible(self, hidden: torch.Tensor, block_index: int) -> bool:
        config = self.owner.config
        start = MINIMAX_H3_SIGMAS.percent_to_sigma(config.start_percent)
        end = MINIMAX_H3_SIGMAS.percent_to_sigma(config.end_percent)
        if self.sigma > start or self.sigma < end:
            self.owner._log_once(
                ("dense", "sigma", self.sigma),
                f"dense: sigma {self.sigma:.3g} outside the start/end window",
            )
            return False
        if hidden.shape[1] < config.min_tokens:
            self.owner._log_once(
                ("dense", "tokens", hidden.shape[1]),
                f"dense: {hidden.shape[1]} tokens < min_tokens {config.min_tokens}",
            )
            return False
        if block_index in config.dense_blocks:
            self.owner._log_once(
                ("dense", "block", block_index),
                f"dense: block {block_index} in dense_blocks",
            )
            return False
        if hidden.dtype != torch.bfloat16 or hidden.device.type != "cuda":
            self.owner._log_once(
                ("dense", "tensor", hidden.dtype, hidden.device.type),
                f"dense: requires CUDA bfloat16, got {hidden.device.type} {hidden.dtype}",
            )
            return False
        kitchen = __import__("dinkster_kitchen")
        available = bool(kitchen.sol_attn_is_available(hidden.device))
        if not available:
            self.owner._log_once(
                ("dense", "kernel", hidden.device),
                "dense: no compiled sol_attn kernel for this GPU",
            )
        return available

    def __call__(
        self,
        attention: MiniMaxH3Attention,
        hidden: torch.Tensor,
        rope_table: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor | None:
        if not self._eligible(hidden, block_index):
            return None
        kitchen = __import__("dinkster_kitchen")
        config = self.owner.config
        plan = None
        rows = self.facts.sequence_length
        active_rope = rope_table
        if config.selection == "vsa":
            key = (self.facts, hidden.device)
            plan = self.owner._plans.get(key)
            if plan is None:
                plan = _vsa_plan(self.facts, hidden.device)
                self.owner._plans[key] = plan
            rows = plan.padded_rows
            active_rope = rope_table.new_zeros((1, rows, *rope_table.shape[2:]))
            active_rope[0, plan.inverse_rows] = rope_table[0]
        sink_blocks, sink_queries = self._sinks()
        extra: dict[str, object] = {}
        gate = attention.to_gate_compress
        if plan is not None:
            sink_blocks = sink_queries = (0, plan.prefix_blocks)
            extra = {"tail": False, "block_len": plan.block_lengths}
            if gate is not None:
                extra["coarse_gate"] = hidden.new_empty(
                    (1, rows, attention.geometry.heads, attention.geometry.head_dim)
                )

        source = hidden[0]

        def chunks() -> Any:
            for offset in range(0, rows, _PRODUCER_CHUNK):
                if plan is None:
                    yield attention.qkv_proj(source[offset : offset + _PRODUCER_CHUNK])
                    continue
                indices = plan.source_rows[offset : offset + _PRODUCER_CHUNK]
                live = indices >= 0
                values = source[indices.clamp_min(0)] * live.unsqueeze(1).to(source.dtype)
                coarse = extra.get("coarse_gate")
                if gate is not None and isinstance(coarse, torch.Tensor):
                    coarse.view(rows, -1)[offset : offset + values.shape[0]] = gate(values)
                yield attention.qkv_proj(values)

        pool_key = (block_index, self.lane, rows)
        pooled = self.owner._pooled.get(pool_key)
        with (
            materialized_rms_norm_weight(attention.q_norm) as query_weight,
            materialized_rms_norm_weight(attention.k_norm) as key_weight,
        ):
            output, key_mean, value_scale = kitchen.sol_attn_chunked(
                chunks,
                rows,
                attention.geometry.heads,
                active_rope,
                (query_weight.detach(), key_weight.detach()),
                kmean=None if pooled is None else pooled[0],
                vscale=None if pooled is None else pooled[1],
                tau=config.tau if config.selection == "sol-attn" else 0.0,
                topk_ratio=(0.0 if config.selection == "sol-attn" else config.keep_percent / 100.0),
                sink_blocks=list(sink_blocks),
                sink_q=list(sink_queries),
                rope_eps=attention.geometry.norm_eps,
                token_aug=0 if config.selection == "vsa" else config.extra_tokens,
                **extra,
            )
        self.owner._pooled[pool_key] = (key_mean, value_scale)
        self.owner._log_once(
            ("sparse", rows),
            f"sparse producer path: {self.facts.sequence_length} tokens, {rows} kernel rows",
        )
        output = output.view(1, rows, attention.geometry.inner_width)
        if plan is not None:
            output = output[:, plan.inverse_rows]
        return attention.out_proj(output)

    def _sinks(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.owner.config.sink_conditioning == "off":
            return (0, 0), (0, 0)
        video_start = next(start for start, _stop, kind in self.facts.segments if kind == "video")
        blocks = (0, math.ceil(video_start / _BLOCK_SIZE))
        if self.owner.config.sink_conditioning == "exact_kv":
            return blocks, (0, 0)
        audio_start = next(start for start, _stop, kind in self.facts.segments if kind == "audio")
        return blocks, (audio_start // _BLOCK_SIZE, blocks[1])


__all__ = ["MiniMaxH3SparseAttention"]
