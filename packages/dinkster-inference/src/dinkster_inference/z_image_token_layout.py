"""Z-Image's declared caption, learned-padding, and image token sequence."""

from __future__ import annotations

from dataclasses import dataclass

from .token_layout import ModelTokenLayout, ModelTokenSegment, TokenRowTable


@dataclass(frozen=True, slots=True)
class ZImageTokenPlan:
    layout: ModelTokenLayout
    positions: TokenRowTable
    caption_rows: int
    caption_padding_rows: int
    image_grid: tuple[int, int]
    image_padding_rows: int

    def __post_init__(self) -> None:
        self.positions.validate_for(self.layout)


def plan_z_image_token_layout(
    caption_rows: int,
    image_grid: tuple[int, int],
    *,
    pad_multiple: int = 32,
) -> ZImageTokenPlan:
    """Declare the exact normal non-Omni sequence and three-axis positions."""
    if type(caption_rows) is not int or caption_rows < 1:
        raise ValueError("caption_rows must be a positive exact int")
    if (
        type(image_grid) is not tuple
        or len(image_grid) != 2
        or any(type(size) is not int or size < 1 for size in image_grid)
    ):
        raise ValueError("image_grid must contain two positive exact ints")
    if type(pad_multiple) is not int or pad_multiple < 1:
        raise ValueError("pad_multiple must be a positive exact int")

    caption_pad = (-caption_rows) % pad_multiple
    image_rows = image_grid[0] * image_grid[1]
    image_pad = (-image_rows) % pad_multiple
    segments: list[ModelTokenSegment] = []
    positions: list[tuple[float, float, float]] = []
    offset = 0

    segments.append(
        ModelTokenSegment("caption", "text", "conditioning", offset, caption_rows, (caption_rows,))
    )
    positions.extend((float(index), 0.0, 0.0) for index in range(1, caption_rows + 1))
    offset += caption_rows
    if caption_pad:
        segments.append(
            ModelTokenSegment(
                "caption_pad",
                "text",
                "learned_padding",
                offset,
                offset + caption_pad,
                (caption_pad,),
            )
        )
        positions.extend(
            (float(index), 0.0, 0.0)
            for index in range(caption_rows + 1, caption_rows + caption_pad + 1)
        )
        offset += caption_pad

    padded_caption = caption_rows + caption_pad
    segments.append(
        ModelTokenSegment(
            "image",
            "image",
            "target",
            offset,
            offset + image_rows,
            image_grid,
        )
    )
    positions.extend(
        (float(padded_caption + 1), float(row), float(column))
        for row in range(image_grid[0])
        for column in range(image_grid[1])
    )
    offset += image_rows
    if image_pad:
        segments.append(
            ModelTokenSegment(
                "image_pad",
                "image",
                "learned_padding",
                offset,
                offset + image_pad,
                (image_pad,),
            )
        )
        positions.extend((0.0, 0.0, 0.0) for _ in range(image_pad))

    layout = ModelTokenLayout(tuple(segments), padded_rows=0)
    return ZImageTokenPlan(
        layout,
        TokenRowTable(tuple(positions)),
        caption_rows,
        caption_pad,
        image_grid,
        image_pad,
    )


__all__ = ["ZImageTokenPlan", "plan_z_image_token_layout"]
