from __future__ import annotations

import numpy as np
import pytest
from dinkster_api.v1 import (
    ReplacementPredicate,
    annotate_image,
    media_semantics,
)
from dinkster_image._draft_initialization import (
    ImageAlphaJoin,
    ImageAlphaPremultiply,
    ImageAlphaUnpremultiply,
    ImageChannelSplit,
    MaskPolarity,
)


def test_alpha_conversion_is_numerical_nonmutating_and_zero_safe() -> None:
    rgba = np.array([[[[2, -1, 0.5, 0], [2, -1, 0.5, 0.25], [1, 0.5, 0.25, 1e-8]]]], np.float32)
    original = rgba.copy()
    premultiplied = np.asarray(ImageAlphaPremultiply.execute(image=rgba)["image"])
    np.testing.assert_array_equal(premultiplied[..., :3], rgba[..., :3] * rgba[..., 3:4])
    np.testing.assert_array_equal(premultiplied[..., 3], rgba[..., 3])
    restored = np.asarray(ImageAlphaUnpremultiply.execute(image=premultiplied)["image"])
    expected = rgba.copy()
    expected[..., 0, :3] = 0
    np.testing.assert_array_equal(restored, expected)
    np.testing.assert_array_equal(rgba, original)
    assert premultiplied.flags.c_contiguous and restored.flags.c_contiguous


@pytest.mark.parametrize("polarity", ["coverage", "transparency"])
def test_mask_polarity_inverts_without_clipping(polarity: str) -> None:
    mask = np.array([[[-0.5, 0, 0.25, 1, 2]]], np.float32)
    converted = np.asarray(MaskPolarity.execute(mask=mask, mask_polarity=polarity)["mask"])
    np.testing.assert_array_equal(converted, 1.0 - mask)
    np.testing.assert_array_equal(MaskPolarity.execute(mask=converted)["mask"], mask)


def test_alpha_conversion_returns_annotated_arrays_and_preserves_color() -> None:
    color = {"primaries": 1, "transfer": 8, "range": 2}
    image = annotate_image(np.full((1, 2, 3, 4), 0.5, np.float32), color=color)
    premultiplied = ImageAlphaPremultiply.execute(image=image)["image"]
    assert isinstance(premultiplied, np.ndarray)
    assert media_semantics(premultiplied) == {"alpha": "premultiplied", "color": color}
    restored = ImageAlphaUnpremultiply.execute(image=premultiplied)["image"]
    assert media_semantics(restored) == {"color": color}
    np.testing.assert_array_equal(restored, image)
    assert media_semantics(image) == {"color": color}
    split = ImageChannelSplit.execute(image=premultiplied)
    assert media_semantics(split["image"]) == {"color": color}
    joined = ImageAlphaJoin.execute(image=image, alpha_mask=split["alpha_mask"])["image"]
    assert media_semantics(joined) == {"color": color}
    joined = ImageAlphaJoin.execute(image=premultiplied, alpha_mask=split["alpha_mask"])["image"]
    assert media_semantics(joined) == {"color": color}
    np.testing.assert_array_equal(joined, image)


@pytest.mark.parametrize("opacity", [0, 0.25, 1])
def test_alpha_replacement_and_split_preserve_unassociated_rgb(opacity: float) -> None:
    pixels = annotate_image(
        np.array([[[[0.5, 0, 0, 0.5], [0, 0, 0, 0]]]], np.float32), alpha="premultiplied"
    )
    joined = ImageAlphaJoin.execute(
        image=pixels, alpha_mask=np.full((1, 1, 2), 1 - opacity, np.float32)
    )["image"]
    np.testing.assert_array_equal(joined, [[[[1, 0, 0, opacity], [0, 0, 0, opacity]]]])
    assert media_semantics(joined) == {}
    split = ImageChannelSplit.execute(image=pixels)
    np.testing.assert_array_equal(split["image"], [[[[1, 0, 0], [0, 0, 0]]]])
    np.testing.assert_array_equal(split["alpha_mask"], [[[0.5, 1]]])


def _matches(
    predicate: ReplacementPredicate | None, values: dict[str, object], links: set[str]
) -> bool:
    if predicate is None or predicate.kind == "always":
        return True
    if predicate.kind == "valueEquals":
        return values.get(predicate.input) == predicate.value
    if predicate.kind == "valuePresent":
        return predicate.input in values
    if predicate.kind == "inputConnected":
        return predicate.input in links
    if predicate.kind == "not":
        return not _matches(predicate.of[0], values, links)
    children = [_matches(child, values, links) for child in predicate.of]
    return all(children) if predicate.kind == "all" else any(children)
