"""Torch-free source/effect mask role and compiled-field contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from dinkster_inference.conditioning import PayloadDescriptor, PayloadReference
from dinkster_inference.effect_mask import (
    EffectMaskInput,
    MaskInputError,
    MaskMediaPlacement,
    SourceMaskInput,
    compile_effect_mask_field,
)
from dinkster_inference.token_layout import (
    ModelTokenLayout,
    ModelTokenSegment,
    TokenGridTransform,
    TokenRowTable,
)

SOURCE_DIGEST = "blake3:" + "a" * 64


def _payload() -> PayloadDescriptor:
    return PayloadDescriptor(PayloadReference(SOURCE_DIGEST), (1, 2, 2), "float32", "mask")


def _effect() -> EffectMaskInput:
    return EffectMaskInput(
        _payload(),
        SOURCE_DIGEST,
        ("batch", "height", "width"),
        MaskMediaPlacement.FULL_DOMAIN,
        "residual",
        "test.bilinear.v1",
    )


def test_source_and_effect_masks_are_distinct_exact_types() -> None:
    source = SourceMaskInput(
        _payload(),
        SOURCE_DIGEST,
        ("batch", "height", "width"),
        MaskMediaPlacement.FULL_DOMAIN,
    )
    effect = _effect()

    assert type(source) is SourceMaskInput
    assert type(effect) is EffectMaskInput
    assert not isinstance(source, EffectMaskInput)
    assert effect.digest == _effect().digest
    with pytest.raises(FrozenInstanceError):
        effect.target_segment = "other"  # type: ignore[misc]


def test_effect_mask_input_binds_canonical_asset_identity_and_geometry() -> None:
    with pytest.raises(MaskInputError, match="BLAKE3"):
        EffectMaskInput(
            _payload(),
            "a" * 64,
            ("batch", "height", "width"),
            MaskMediaPlacement.FULL_DOMAIN,
            "residual",
            "test.bilinear.v1",
        )
    with pytest.raises(MaskInputError, match="name every"):
        EffectMaskInput(
            _payload(),
            SOURCE_DIGEST,
            ("height", "width"),
            MaskMediaPlacement.FULL_DOMAIN,
            "residual",
            "test.bilinear.v1",
        )


def test_compiled_field_binds_source_transform_layout_and_values() -> None:
    layout = ModelTokenLayout(
        (ModelTokenSegment("residual", "image", "control-residual", 0, 4, (2, 2)),),
        0,
    )
    transform = TokenGridTransform("test.bilinear.v1", "image", "residual", (2, 2), None)
    table = TokenRowTable(((0.0,), (0.25,), (0.5,), (1.0,)))

    field = compile_effect_mask_field(_effect(), layout, transform, table)

    assert field.input_digest == _effect().digest
    assert field.source_digest == SOURCE_DIGEST
    assert field.layout_digest == layout.digest
    assert field.transform_digest == transform.digest
    assert field.table is table
    assert field.digest == compile_effect_mask_field(_effect(), layout, transform, table).digest
    changed = compile_effect_mask_field(
        _effect(), layout, transform, TokenRowTable(((0.0,), (0.25,), (0.75,), (1.0,)))
    )
    assert changed.digest != field.digest
    with pytest.raises(TypeError, match="created by"):
        type(field)(
            field.input_digest,
            field.source_digest,
            field.layout_digest,
            field.transform_digest,
            field.target_segment,
            field.table,
            field.canonical_preimage,
            field.digest,
        )


def test_compiled_field_contains_only_target_rows_and_canonical_float_values() -> None:
    layout = ModelTokenLayout(
        (
            ModelTokenSegment("context", "text", "conditioning", 0, 2, (2,)),
            ModelTokenSegment("residual", "image", "control-residual", 2, 6, (2, 2)),
        ),
        3,
    )
    transform = TokenGridTransform("test.bilinear.v1", "image", "residual", (2, 2), None)
    table = TokenRowTable(((-0.0,), (0.1,), (0.5,), (1.0,)))

    field = compile_effect_mask_field(_effect(), layout, transform, table)

    assert field.table is table
    assert '"table":[[0.0],[0.1],[0.5],[1.0]]' in field.canonical_preimage
    assert "-0.0" not in field.canonical_preimage
    unpadded = ModelTokenLayout(layout.segments, 0)
    same_field = compile_effect_mask_field(_effect(), unpadded, transform, table)
    assert field.layout_digest == unpadded.digest
    assert field.digest == same_field.digest
    assert field.digest == "1cf469e6c151978d1ca1a14760731aa7bba0a7fb08029583cb89b13b4339a8b9"
    with pytest.raises(MaskInputError, match="target segment"):
        compile_effect_mask_field(
            _effect(),
            layout,
            transform,
            TokenRowTable(((1.0,),) * layout.valid_rows),
        )


def test_compiled_field_uses_named_media_axes_instead_of_batch_position() -> None:
    payload = PayloadDescriptor(PayloadReference(SOURCE_DIGEST), (2, 2, 1), "float32", "mask")
    source = EffectMaskInput(
        payload,
        SOURCE_DIGEST,
        ("height", "width", "batch"),
        MaskMediaPlacement.FULL_DOMAIN,
        "residual",
        "test.bilinear.v1",
    )
    layout = ModelTokenLayout(
        (ModelTokenSegment("residual", "image", "control-residual", 0, 4, (2, 2)),),
        0,
    )
    transform = TokenGridTransform("test.bilinear.v1", "image", "residual", (2, 2), None)

    field = compile_effect_mask_field(source, layout, transform, TokenRowTable(((1.0,),) * 4))

    assert field.input_digest == source.digest


@pytest.mark.parametrize(
    ("table", "message"),
    (
        (TokenRowTable(((0.0,), (1.0,))), "mask_layout_mismatch"),
        (TokenRowTable(((0.0,), (0.5,), (1.0,), (1.1,))), r"in \[0, 1\]"),
        (TokenRowTable(((0.0,), (0.5,), (1.0,), (float("nan"),))), "finite"),
    ),
)
def test_compiled_field_refuses_incomplete_or_invalid_values(
    table: TokenRowTable, message: str
) -> None:
    layout = ModelTokenLayout(
        (ModelTokenSegment("residual", "image", "control-residual", 0, 4, (2, 2)),),
        0,
    )
    transform = TokenGridTransform("test.bilinear.v1", "image", "residual", (2, 2), None)
    with pytest.raises(MaskInputError, match=message):
        compile_effect_mask_field(_effect(), layout, transform, table)
