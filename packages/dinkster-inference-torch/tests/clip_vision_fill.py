"""Deterministic CLIP vision state shared by golden generation and replay."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from unet_fill import fill_value


def fill_state_dict(
    entries: Sequence[tuple[str, Sequence[int]]],
) -> dict[str, torch.Tensor]:
    values = {key: fill_value(key, shape) for key, shape in entries}
    position = values.get("vision_model.embeddings.position_ids")
    if position is not None:
        values["vision_model.embeddings.position_ids"] = torch.arange(
            position.shape[1], dtype=torch.int64
        ).unsqueeze(0)
    return values


def image_input(shape: Sequence[int]) -> torch.Tensor:
    return (fill_value("input:clip-vision:image", shape) + 0.2).clamp(0, 1)
