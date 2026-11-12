from __future__ import annotations

import torch
from dinkster_hidream.hidream_patches import (
    HiDreamO1ReferenceImages,
)


def test_reference_order_append_and_shallow_identity() -> None:
    token, negative_token, marker = object(), object(), object()
    prior, negative_prior = torch.tensor([-7.0]), torch.tensor([-13.0])
    positive = [[token, {"reference_latents": [prior], "marker": marker}]]
    negative = [[negative_token, {"reference_latents": [negative_prior], "prompt_type": "old"}]]
    images = {
        name: torch.full((1, 2, 3, 3), float(index))
        for name, index in (
            ("extra", 201),
            ("image_100", 100),
            ("image_10", 10),
            ("image_2", 2),
            ("tail", 301),
        )
    }
    images["image_1"] = None
    result = HiDreamO1ReferenceImages.execute(positive=positive, negative=negative, images=images)
    for output, original, previous in (
        ("positive", positive, prior),
        ("negative", negative, negative_prior),
    ):
        entry = result[output][0]
        assert entry[0] is original[0][0] and entry[1] is not original[0][1]
        refs = entry[1]["reference_latents"]
        assert refs[0] is previous and refs[1] is None
        assert all(
            actual is images[name]
            for actual, name in zip(
                refs[2:], ("image_2", "image_10", "image_100", "extra", "tail"), strict=True
            )
        )
        assert len(original[0][1]["reference_latents"]) == 1
    assert result["positive"][0][1]["marker"] is marker
    assert result["negative"][0][1]["prompt_type"] == "negative"
    assert negative[0][1]["prompt_type"] == "old"
    for empty in (None, {}):
        unchanged = HiDreamO1ReferenceImages.execute(
            positive=positive, negative=negative, images=empty
        )
        assert unchanged["positive"] is positive and unchanged["negative"] is negative
