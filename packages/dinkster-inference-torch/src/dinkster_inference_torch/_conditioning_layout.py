"""Internal conditioning carriers for declared model-token geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from dinkster_inference import (
    Conditioning,
    ModelTokenLayout,
    ModelTokenSegment,
    TokenGridTransform,
    TokenLayoutError,
)

CROSS_ATTN_REPEAT_LIMIT = 4


def cross_attn_repeat(lengths: list[int]) -> list[int] | None:
    """Return bounded repeat factors to the token-length least common multiple."""
    target = 1
    for length in lengths:
        target = math.lcm(target, length)
    factors = [target // length for length in lengths]
    if any(factor > CROSS_ATTN_REPEAT_LIMIT for factor in factors):
        return None
    return factors


def repeat_cross_attn(context: torch.Tensor, factor: int) -> torch.Tensor:
    """Tile a complete conditioning sequence along only its token axis."""
    if factor == 1:
        return context
    return context.repeat(1, factor, 1)


@dataclass(frozen=True, kw_only=True)
class DeclaredConditioning(Conditioning[torch.Tensor]):
    source_token_count: int
    model_token_layout: ModelTokenLayout | None = None
    token_transforms: tuple[TokenGridTransform, ...] = ()

    def __post_init__(self) -> None:
        if type(self.source_token_count) is not int or self.source_token_count < 1:
            raise TokenLayoutError("declared conditioning token count must be an exact int >= 1")
        if (
            self.model_token_layout is not None
            and type(self.model_token_layout) is not ModelTokenLayout
        ):
            raise TokenLayoutError("conditioning layout must be an exact ModelTokenLayout or None")
        if type(self.token_transforms) is not tuple or any(
            type(transform) is not TokenGridTransform for transform in self.token_transforms
        ):
            raise TokenLayoutError(
                "conditioning transforms must be exact TokenGridTransform values"
            )
        if self.model_token_layout is None and self.token_transforms:
            raise TokenLayoutError("conditioning transforms require a declared layout")


def declare_text_conditioning(
    conditioning: Conditioning[torch.Tensor], token_count: int
) -> Conditioning[torch.Tensor]:
    return DeclaredConditioning(
        embeddings=conditioning.embeddings,
        pooled=conditioning.pooled,
        source_token_count=token_count,
    )


def declared_token_count(conditioning: Conditioning[torch.Tensor]) -> int | None:
    if type(conditioning) is not DeclaredConditioning:
        return None
    return conditioning.source_token_count


def conditioning_layout(value: object) -> ModelTokenLayout | None:
    if type(value) is not DeclaredConditioning:
        return None
    return value.model_token_layout


def conditioning_token_transforms(value: object) -> tuple[TokenGridTransform, ...]:
    if type(value) is not DeclaredConditioning:
        return ()
    return value.token_transforms


def bind_flux_layout(
    conditioning: object,
    *,
    latent_height: int,
    latent_width: int,
    patch_size: int,
) -> object:
    if type(conditioning) is not DeclaredConditioning:
        return conditioning
    if (
        type(patch_size) is not int
        or patch_size != 2
        or latent_height % patch_size
        or latent_width % patch_size
    ):
        return replace(conditioning, model_token_layout=None, token_transforms=())
    text_tokens = conditioning.source_token_count
    image_grid = (latent_height // patch_size, latent_width // patch_size)
    image_tokens = image_grid[0] * image_grid[1]
    layout = ModelTokenLayout(
        (
            ModelTokenSegment("text", "text", "context", 0, text_tokens, (text_tokens,)),
            ModelTokenSegment(
                "latent_image",
                "image",
                "latent",
                text_tokens,
                text_tokens + image_tokens,
                image_grid,
            ),
        ),
        0,
    )
    return replace(
        conditioning,
        model_token_layout=layout,
        token_transforms=(
            TokenGridTransform(
                "flux.text-context-identity.v1",
                "text",
                "text",
                (text_tokens,),
                None,
            ),
            TokenGridTransform(
                "flux.latent-image-pack-2x2.v1",
                "image",
                "latent_image",
                (latent_height, latent_width),
                None,
            ),
        ),
    )


def flux_fused_layout(layouts: tuple[ModelTokenLayout, ...]) -> ModelTokenLayout:
    """Describe a Flux call after bounded whole-sequence text repetition."""
    first_text, first_image = layouts[0].segments
    repeats = cross_attn_repeat([layout.segments[0].rows for layout in layouts])
    if repeats is None:
        raise TokenLayoutError("Flux text rows exceed the bounded repeat limit")
    if any(
        len(layout.segments) != 2
        or layout.padded_rows != layouts[0].padded_rows
        or layout.segments[0].identity != first_text.identity
        or layout.segments[0].modality != first_text.modality
        or layout.segments[0].role != first_text.role
        or layout.segments[1].identity != first_image.identity
        or layout.segments[1].modality != first_image.modality
        or layout.segments[1].role != first_image.role
        or layout.segments[1].grid != first_image.grid
        for layout in layouts
    ):
        raise TokenLayoutError("Flux fused layouts must differ only in text rows")
    text_rows = first_text.rows * repeats[0]
    return ModelTokenLayout(
        (
            ModelTokenSegment(
                first_text.identity,
                first_text.modality,
                first_text.role,
                0,
                text_rows,
                (text_rows,),
            ),
            ModelTokenSegment(
                first_image.identity,
                first_image.modality,
                first_image.role,
                text_rows,
                text_rows + first_image.rows,
                first_image.grid,
            ),
        ),
        layouts[0].padded_rows,
    )


def bind_sd_layout(conditioning: object, target_token_count: int) -> object:
    if type(conditioning) is not DeclaredConditioning:
        return conditioning
    source_tokens = conditioning.source_token_count
    layout = ModelTokenLayout(
        (
            ModelTokenSegment(
                "text_context",
                "text",
                "context",
                0,
                target_token_count,
                (target_token_count,),
            ),
        ),
        0,
    )
    return replace(
        conditioning,
        model_token_layout=layout,
        token_transforms=(
            TokenGridTransform(
                "sd.text-context-repeat.v1",
                "text",
                "text_context",
                (source_tokens,),
                None,
            ),
        ),
    )


def validate_flux_layout(
    condition: tuple[torch.Tensor, torch.Tensor | None],
    layout: ModelTokenLayout,
    *,
    latent_height: int,
    latent_width: int,
    patch_size: int,
) -> None:
    if tuple(segment.identity for segment in layout.segments) != ("text", "latent_image"):
        raise TokenLayoutError("Flux layout requires text then latent_image segments")
    text, image = layout.segments
    if text.modality != "text" or text.role != "context":
        raise TokenLayoutError("Flux text segment has the wrong semantic role")
    if condition[0].shape[1] != text.rows:
        raise TokenLayoutError("Flux context rows do not match the declared text segment")
    expected_grid = (latent_height // patch_size, latent_width // patch_size)
    if image.modality != "image" or image.role != "latent" or image.grid != expected_grid:
        raise TokenLayoutError("Flux latent image rows do not match the declared packed grid")


def validate_sd_layout(
    condition: tuple[torch.Tensor, torch.Tensor | None],
    layout: ModelTokenLayout,
    *,
    repeat_limit: int,
) -> None:
    if len(layout.segments) != 1 or layout.segments[0].identity != "text_context":
        raise TokenLayoutError("SD layout requires one text_context segment")
    segment = layout.segments[0]
    if segment.modality != "text" or segment.role != "context":
        raise TokenLayoutError("SD text context segment has the wrong semantic role")
    source_tokens = condition[0].shape[1]
    if segment.rows % source_tokens or segment.rows // source_tokens > repeat_limit:
        raise TokenLayoutError("SD context rows are not an admissible whole-sequence repeat")
