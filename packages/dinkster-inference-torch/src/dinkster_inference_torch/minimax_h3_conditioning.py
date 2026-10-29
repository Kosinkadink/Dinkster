"""MiniMax H3 raw-token and Qwen3-VL vision-input realization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from dinkster_inference.minimax_h3 import (
    MiniMaxH3ConditionerPlan,
    MiniMaxH3PresentationKind,
    MiniMaxH3TokenTag,
)
from dinkster_inference.qwen_bpe import QWEN_PAD, load_qwen_bpe

_IMAGE_TOKEN = 151655
_PATCH_SIZE = 16
_TEMPORAL_PATCH = 2
_MERGE_SIZE = 2
_MIN_PIXELS = 3136
_MAX_PIXELS = 12845056
_VISION_START_TOKEN = 151652
_VISION_END_TOKEN = 151653


@dataclass(frozen=True)
class MiniMaxH3VisionValue:
    kind: MiniMaxH3PresentationKind
    pixels: torch.Tensor

    def __post_init__(self) -> None:
        if self.kind not in (
            MiniMaxH3PresentationKind.IMAGE_CONTENT,
            MiniMaxH3PresentationKind.VIDEO_CONTENT,
        ):
            raise ValueError("vision value kind must be image_content or video_content")
        expected_frames = 1 if self.kind is MiniMaxH3PresentationKind.IMAGE_CONTENT else 2
        if (
            self.pixels.ndim != 4
            or self.pixels.shape[0] != expected_frames
            or self.pixels.shape[-1] != 3
            or self.pixels.shape[1] <= 0
            or self.pixels.shape[2] <= 0
        ):
            raise ValueError(
                f"{self.kind.value} pixels must be [{expected_frames}, height, width, 3]"
            )
        if not self.pixels.is_floating_point():
            raise TypeError("vision pixels must use a floating dtype")


@dataclass(frozen=True)
class MiniMaxH3ConditionerInputs:
    ids: torch.Tensor
    position_ids: torch.Tensor
    visual_mask: torch.Tensor
    token_tags: torch.Tensor
    patches: torch.Tensor | None
    grids: torch.Tensor | None
    declared_token_count: int

    def __post_init__(self) -> None:
        if type(self.declared_token_count) is not int or self.declared_token_count < 1:
            raise ValueError("declared conditioner token count must be an exact int >= 1")
        if self.ids.ndim != 2 or self.ids.shape[0] != 1 or self.ids.shape[1] == 0:
            raise ValueError("conditioner IDs must be one non-empty row")
        if self.ids.shape[1] != self.declared_token_count:
            raise ValueError("conditioner IDs must match the declared token count")
        if self.ids.dtype != torch.long:
            raise TypeError("conditioner IDs must use torch.long")
        if self.position_ids.shape not in ((1, self.ids.shape[1]), (3, self.ids.shape[1])):
            raise ValueError("conditioner position IDs must match the token row")
        if self.position_ids.dtype != torch.long:
            raise TypeError("conditioner position IDs must use torch.long")
        if self.visual_mask.shape != self.ids.shape or self.visual_mask.dtype != torch.bool:
            raise ValueError("conditioner visual mask must be a bool token row")
        if self.token_tags.shape != self.ids.shape or self.token_tags.dtype != torch.long:
            raise ValueError("conditioner token tags must be a long token row")
        if (self.patches is None) != (self.grids is None):
            raise ValueError("conditioner patches and grids must be provided together")
        if self.patches is None:
            if bool(self.visual_mask.any()) or self.position_ids.shape[0] != 1:
                raise ValueError("text-only conditioner inputs cannot carry visual state")
            if not torch.equal(
                self.position_ids,
                torch.arange(self.ids.shape[1], device=self.ids.device).unsqueeze(0),
            ):
                raise ValueError("text-only conditioner positions must be sequential")
            if not bool(torch.all(self.token_tags == int(MiniMaxH3TokenTag.TEXT))):
                raise ValueError("text-only conditioner tags must be text")
        else:
            assert self.grids is not None
            if self.patches.ndim != 2 or self.patches.shape[1] != 1536:
                raise ValueError("conditioner patches must be [patches, 1536]")
            if not self.patches.is_floating_point():
                raise TypeError("conditioner patches must use a floating dtype")
            if (
                self.grids.ndim != 2
                or self.grids.shape[0] == 0
                or self.grids.shape[1] != 3
                or self.grids.dtype != torch.long
            ):
                raise ValueError("conditioner grids must be [items, 3]")
            grid_rows = [(int(row[0]), int(row[1]), int(row[2])) for row in self.grids.tolist()]
            if any(
                time != 1
                or height <= 0
                or width <= 0
                or height % _MERGE_SIZE
                or width % _MERGE_SIZE
                for time, height, width in grid_rows
            ):
                raise ValueError("conditioner grids must be positive mergeable single blocks")
            if self.patches.shape[0] != sum(math.prod(row) for row in grid_rows):
                raise ValueError("conditioner patch count must match every grid")
            if self.position_ids.shape[0] != 3:
                raise ValueError("vision conditioner inputs require three-axis positions")
            expected_visual = int(sum(math.prod(row) // (_MERGE_SIZE**2) for row in grid_rows))
            if int(self.visual_mask.count_nonzero()) != expected_visual:
                raise ValueError("visual mask count must match merged grid tokens")
            visual_indices = torch.nonzero(self.visual_mask[0]).flatten().tolist()
            spans: list[tuple[int, int, tuple[int, int, int]]] = []
            cursor = 0
            for grid in grid_rows:
                merged_tokens = math.prod(grid) // (_MERGE_SIZE**2)
                indices = visual_indices[cursor : cursor + merged_tokens]
                if len(indices) != merged_tokens:
                    raise ValueError("every conditioner grid requires one visual span")
                start = indices[0]
                end = start + merged_tokens
                if indices != list(range(start, end)):
                    raise ValueError("conditioner visual spans must be contiguous")
                if start == 0 or end >= self.ids.shape[1]:
                    raise ValueError("conditioner visual spans require flanking markers")
                if (
                    int(self.ids[0, start - 1]) != _VISION_START_TOKEN
                    or int(self.ids[0, end]) != _VISION_END_TOKEN
                    or not bool(torch.all(self.ids[0, start:end] == _IMAGE_TOKEN))
                ):
                    raise ValueError("conditioner visual spans require exact IDs and markers")
                spans.append((start, end, grid))
                cursor += merged_tokens
            expected_tags = torch.full_like(self.token_tags, int(MiniMaxH3TokenTag.TEXT))
            for start, end, _grid in spans:
                expected_tags[0, start - 1 : end + 1] = int(MiniMaxH3TokenTag.VISION)
            if not torch.equal(self.token_tags, expected_tags):
                raise ValueError("conditioner token tags must match visual spans")
            expected_positions = _mrope_positions(self.ids.shape[1], tuple(spans), self.ids.device)
            if not torch.equal(self.position_ids, expected_positions):
                raise ValueError("conditioner positions must match visual spans")
        devices = {
            tensor.device
            for tensor in (
                self.ids,
                self.position_ids,
                self.visual_mask,
                self.token_tags,
                self.patches,
                self.grids,
            )
            if tensor is not None
        }
        if len(devices) != 1:
            raise ValueError("conditioner input tensors must share one device")


def _resize_shape(height: int, width: int) -> tuple[int, int]:
    factor = _PATCH_SIZE * _MERGE_SIZE
    resized_h = round(height / factor) * factor
    resized_w = round(width / factor) * factor
    if resized_h * resized_w > _MAX_PIXELS:
        beta = math.sqrt((height * width) / _MAX_PIXELS)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < _MIN_PIXELS:
        beta = math.sqrt(_MIN_PIXELS / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    return resized_h, resized_w


def _vision_patches(value: MiniMaxH3VisionValue) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = value.pixels
    _, height, width, _ = pixels.shape
    resized_h, resized_w = _resize_shape(height, width)
    images = F.interpolate(
        pixels.permute(0, 3, 1, 2),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    if value.kind is MiniMaxH3PresentationKind.IMAGE_CONTENT:
        images = images.repeat(2, 1, 1, 1)
    mean = torch.full(
        (1, 3, 1, 1),
        0.5,
        device=images.device,
        dtype=(
            torch.float32 if value.kind is MiniMaxH3PresentationKind.VIDEO_CONTENT else images.dtype
        ),
    )
    images = (images - mean) / mean
    grid_h = resized_h // _PATCH_SIZE
    grid_w = resized_w // _PATCH_SIZE
    patches = images.reshape(
        1,
        _TEMPORAL_PATCH,
        3,
        grid_h // _MERGE_SIZE,
        _MERGE_SIZE,
        _PATCH_SIZE,
        grid_w // _MERGE_SIZE,
        _MERGE_SIZE,
        _PATCH_SIZE,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(grid_h * grid_w, 1536)
    grid = torch.tensor(((1, grid_h, grid_w),), device=images.device, dtype=torch.long)
    return patches, grid


def _mrope_positions(
    length: int,
    spans: tuple[tuple[int, int, tuple[int, int, int]], ...],
    device: torch.device,
) -> torch.Tensor:
    if not spans:
        return torch.arange(length, device=device).unsqueeze(0)
    positions = torch.zeros((3, length), device=device, dtype=torch.long)
    offset = 0
    for span_index, (start, end, grid) in enumerate(spans):
        time, height, width = grid
        if span_index == 0:
            positions[:, :start] = torch.arange(start, device=device)
        maximum = max(time, height, width) // _MERGE_SIZE
        next_start = start + maximum
        positions[:, end:] = torch.arange(
            next_start + offset, next_start + (length - end) + offset, device=device
        )
        positions[0, start:end] = start + offset
        merged_h = height // _MERGE_SIZE
        merged_w = width // _MERGE_SIZE
        positions[1, start:end] = (
            torch.arange(start + offset, start + merged_h + offset, device=device)
            .unsqueeze(1)
            .repeat(1, math.ceil((end - start) / merged_h))
            .flatten()[: end - start]
        )
        positions[2, start:end] = (
            torch.arange(start + offset, start + merged_w + offset, device=device)
            .unsqueeze(0)
            .repeat(math.ceil((end - start) / merged_w), 1)
            .flatten()[: end - start]
        )
        offset += maximum - (end - start)
    return positions


def realize_minimax_h3_conditioner_inputs(
    plan: MiniMaxH3ConditionerPlan,
    vision: tuple[MiniMaxH3VisionValue, ...] = (),
) -> MiniMaxH3ConditionerInputs:
    """Tokenize a normalized plan and realize its ordered vision tensors."""
    if type(plan) is not MiniMaxH3ConditionerPlan:
        raise TypeError("plan must be an exact MiniMaxH3ConditionerPlan")
    if type(vision) is not tuple or any(
        type(value) is not MiniMaxH3VisionValue for value in vision
    ):
        raise TypeError("vision must be a tuple of exact MiniMaxH3VisionValue values")
    expected_kinds = tuple(
        segment.kind
        for segment in plan.presentation
        if segment.kind
        in (
            MiniMaxH3PresentationKind.IMAGE_CONTENT,
            MiniMaxH3PresentationKind.VIDEO_CONTENT,
        )
    )
    if tuple(value.kind for value in vision) != expected_kinds:
        raise ValueError("realized vision values must exactly match the conditioner plan")
    if vision and any(value.pixels.device != vision[0].pixels.device for value in vision):
        raise ValueError("all MiniMax H3 vision values must share one device")
    tokenizer = load_qwen_bpe()
    ids: list[int] = []
    tags: list[int] = []
    visual_flags: list[bool] = []
    spans: list[tuple[int, int, tuple[int, int, int]]] = []
    patches: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    vision_index = 0
    device = vision[0].pixels.device if vision else torch.device("cpu")
    for segment in plan.presentation:
        if segment.kind in (
            MiniMaxH3PresentationKind.TEXT,
            MiniMaxH3PresentationKind.VISION_START,
            MiniMaxH3PresentationKind.VISION_END,
        ):
            assert segment.text is not None
            segment_ids = tokenizer.encode(segment.text)
            ids.extend(segment_ids)
            tags.extend([int(segment.token_tag)] * len(segment_ids))
            visual_flags.extend([False] * len(segment_ids))
            continue
        value = vision[vision_index]
        vision_index += 1
        flattened, grid = _vision_patches(value)
        grid_values = grid[0].tolist()
        grid_tuple = (int(grid_values[0]), int(grid_values[1]), int(grid_values[2]))
        merged_tokens = math.prod(grid_tuple) // (_MERGE_SIZE**2)
        start = len(ids)
        ids.extend([_IMAGE_TOKEN] * merged_tokens)
        tags.extend([int(MiniMaxH3TokenTag.VISION)] * merged_tokens)
        visual_flags.extend([True] * merged_tokens)
        spans.append((start, len(ids), grid_tuple))
        patches.append(flattened)
        grids.append(grid)
    if not ids:
        ids.append(QWEN_PAD)
        tags.append(int(MiniMaxH3TokenTag.TEXT))
        visual_flags.append(False)
    id_tensor = torch.tensor((ids,), device=device, dtype=torch.long)
    tag_tensor = torch.tensor((tags,), device=device, dtype=torch.long)
    visual_mask = torch.tensor((visual_flags,), device=device, dtype=torch.bool)
    return MiniMaxH3ConditionerInputs(
        ids=id_tensor,
        position_ids=_mrope_positions(len(ids), tuple(spans), device),
        visual_mask=visual_mask,
        token_tags=tag_tensor,
        patches=torch.cat(patches) if patches else None,
        grids=torch.cat(grids) if grids else None,
        declared_token_count=len(ids),
    )


__all__ = [
    "MiniMaxH3ConditionerInputs",
    "MiniMaxH3VisionValue",
    "realize_minimax_h3_conditioner_inputs",
]
