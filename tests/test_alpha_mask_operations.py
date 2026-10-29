from __future__ import annotations

import numpy as np
import pytest
from dinkster_api.v1 import (
    Node,
    ReplacementPredicate,
    annotate_image,
    annotate_mask,
    media_semantics,
)
from dinkster_nodes_image import (
    IMAGE_NODES,
    GlslShader,
    ImageAlphaJoin,
    ImageAlphaPremultiply,
    ImageAlphaUnpremultiply,
    ImageChannelMerge,
    ImageChannelSplit,
    ImageComposite,
    ImageToMask,
    MaskCombine,
    MaskInfo,
    MaskMorphology,
    MaskPolarity,
    MaskToImage,
    PorterDuffComposite,
)
from dinkster_nodes_media_io.image import LoadImage, LoadImageOutput, LoadMask, SaveImage, SaveMask
from dinkster_schema import validate_replacement_references


@pytest.mark.parametrize("node", [ImageAlphaPremultiply, ImageAlphaUnpremultiply])
def test_alpha_conversion_requires_rgba(node: type[Node]) -> None:
    for channels in (1, 3):
        with pytest.raises(ValueError, match="RGBA"):
            node.execute(image=np.ones((1, 2, 3, channels), np.float32))


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


@pytest.mark.parametrize("polarity", ["coverage", "transparency"])
def test_alpha_masks_return_selected_polarity(polarity: str) -> None:
    image = np.full((1, 2, 3, 4), 0.5, np.float32)
    expected = {
        "semantic": "alpha",
        **({"polarity": polarity} if polarity == "transparency" else {}),
    }
    split = ImageChannelSplit.execute(image=image, mask_polarity=polarity)
    assert media_semantics(split["alpha_mask"]) == expected
    composed = PorterDuffComposite.execute(
        source=image[..., :3],
        destination=image[..., :3],
        source_alpha_mask=split["alpha_mask"],
        destination_alpha_mask=split["alpha_mask"],
        mask_polarity=polarity,
    )
    assert media_semantics(composed["alpha_mask"]) == expected
    mask = annotate_mask(np.ones((1, 2, 3), np.float32), semantic="alpha")
    converted = MaskPolarity.execute(mask=mask, mask_polarity=polarity)["mask"]
    assert media_semantics(converted) == expected
    assert media_semantics(mask) == {"semantic": "alpha"}


@pytest.mark.parametrize("channel", ["alpha", "transparency", "red"])
@pytest.mark.parametrize("invert", [False, True])
def test_image_to_mask_marks_alpha_extraction(channel: str, invert: bool) -> None:
    result = ImageToMask.execute(
        image=np.ones((1, 2, 3, 4), np.float32), channel=channel, invert=invert
    )["mask"]
    expected = {} if channel == "red" else {"semantic": "alpha"}
    if channel != "red" and ((channel == "transparency") != invert):
        expected["polarity"] = "transparency"
    assert media_semantics(result) == expected


def test_preserving_nodes_restore_semantics_after_array_coercion() -> None:
    mask = annotate_mask(
        np.full((1, 2, 3), 0.5, np.float32), polarity="transparency", semantic="other"
    )
    assert media_semantics(np.asarray(mask)) == {}
    outputs = [MaskInfo.execute(mask=mask)["mask"]]
    outputs.extend(MaskMorphology.execute(mask=mask, operation="invert").values())
    outputs.append(MaskCombine.execute(destination=mask, source=np.ones_like(mask))["mask"])
    for output in outputs:
        assert media_semantics(output) == media_semantics(mask)
    image = annotate_image(np.full((1, 2, 3, 4), 0.5, np.float32), alpha="premultiplied")
    for x in (0, 100):
        output = ImageComposite.execute(destination=image, source=np.asarray(image), x=x)["image"]
        assert media_semantics(output) == {"alpha": "premultiplied"}


@pytest.mark.parametrize(
    "node",
    [node for node in IMAGE_NODES if not node.__module__.endswith(("geometry", "compositor"))],
    ids=lambda node: node.__name__,
)
def test_image_operation_alpha_policy_inventory(node: type[Node]) -> None:
    schema = node.define_schema()
    for output in schema.outputs:
        expected = "preserve"
        if node is ImageChannelSplit and output.type.runtime_type_id() == "dinkster.image":
            expected = "drop"
        elif node.__module__.endswith("preprocess") and output.id == "image":
            expected = "drop"
        elif node in (ImageAlphaJoin, GlslShader):
            expected = "create_if_missing"
        elif node in (ImageAlphaPremultiply, ImageAlphaUnpremultiply):
            expected = "require"
        assert output.alpha_policy == expected, (schema.node_type, output.id)
    if node is ImageChannelMerge:
        assert all(spec.alpha_policy == "drop" for spec in schema.inputs[:3])
    if node in (ImageAlphaPremultiply, ImageAlphaUnpremultiply):
        assert schema.inputs[0].alpha_policy == "require"


def test_media_policy_inventory() -> None:
    for node in (LoadImage, LoadImageOutput):
        assert node.schema().outputs[0].alpha_policy == "drop"
        assert node.schema().outputs[1].mask_semantic == "alpha"
    for node in (LoadMask, SaveMask, SaveImage):
        assert all(spec.alpha_policy == "preserve" for spec in node.schema().outputs)


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


@pytest.mark.parametrize(
    "node,source,target,choices",
    [
        (ImageChannelSplit, "alpha_mask_polarity", "mask_polarity", {}),
        (ImageChannelMerge, "alpha_mask_polarity", "mask_polarity", {}),
        (ImageAlphaJoin, "alpha_mask_polarity", "mask_polarity", {}),
        (PorterDuffComposite, "alpha_mask_polarity", "mask_polarity", {}),
        (
            ImageComposite,
            "mask.mask_polarity",
            "mask.mask_polarity",
            {"mask": 1, "source_resize": "none"},
        ),
        (MaskToImage, "channels.alpha_polarity", "channels.mask_polarity", {"channels": "rgba"}),
        (LoadMask, "polarity", "mask_polarity", {}),
        (SaveMask, "mask_mode", "mask_polarity", {}),
    ],
)
@pytest.mark.parametrize(
    "legacy,canonical",
    [
        ("opacity", "coverage"),
        ("transparency", "transparency"),
        ("mask_is_opacity", "coverage"),
        ("mask_is_transparency", "transparency"),
    ],
)
def test_polarity_migrations_preserve_values_and_links(
    node: type[Node],
    source: str,
    target: str,
    choices: dict[str, object],
    legacy: str,
    canonical: str,
) -> None:
    schema = node.define_schema()
    assert schema.version >= 2
    assert validate_replacement_references({schema.node_type: schema}) == ()
    rule = schema.replacements[0]
    values = {**choices, source: legacy}
    for links in (set(), {source}):
        case = next(case for case in rule.cases if _matches(case.when, values, links))
        mapping = dict(case.inputs)[target]
        assert mapping.input == source
        if links:
            assert mapping.kind == "copy"
        else:
            assert mapping.transform is not None
            assert dict(mapping.transform.map)[legacy] == canonical


@pytest.mark.parametrize(
    "legacy,canonical",
    [("inverted", "transparency"), ("source", "coverage"), ("direct", "coverage")],
)
@pytest.mark.parametrize("node", [LoadMask, SaveMask])
def test_mask_file_migrations_preserve_old_pixel_inversion(
    node: type[Node], legacy: str, canonical: str
) -> None:
    rule = node.schema().replacements[0]
    for source in ("polarity", "mask_mode"):
        case = next(case for case in rule.cases if _matches(case.when, {source: legacy}, set()))
        transform = dict(case.inputs)["mask_polarity"].transform
        assert transform is not None
        assert dict(transform.map)[legacy] == canonical
