"""Proving tests for header-level OpenCLIP text-encoder conversion.

The synthetic OpenCLIP layout was verified byte-exact against the
real sd_xl_base_1.0.safetensors header (conditioner.embedders.1.model
subtree: identical key set and shapes, logit_scale the only inert
sibling), so these tests prove the conversion on the layout real SDXL
checkpoints ship. The reference conversion is comfy/utils.py
clip_text_transformers_convert @ 947c2749.
"""

from __future__ import annotations

import pytest
from dinkster_inference import (
    BFLOAT16,
    CLIP_G_TEXT_CONFIG,
    CLIP_L_TEXT_CONFIG,
    FLOAT16,
    FLOAT32,
    OPENCLIP_TEXT_INERT_KEYS,
    DType,
    OpenClipTextDetectError,
    RowChunk,
    TensorGeometry,
    Transpose2D,
    clip_text_layout,
    convert_openclip_text,
    is_openclip_text,
    openclip_text_layout,
    transformed_geometry,
)


def geometries(dtype: DType = FLOAT16) -> dict[str, TensorGeometry]:
    return {
        key: TensorGeometry(shape, dtype)
        for key, shape in openclip_text_layout(CLIP_G_TEXT_CONFIG).items()
    }


# ------------------------------------------------------------ routing


def test_is_openclip_text_routes_on_resblocks() -> None:
    assert is_openclip_text(geometries())
    transformers = {
        key: TensorGeometry(shape, FLOAT16)
        for key, shape in clip_text_layout(CLIP_L_TEXT_CONFIG).items()
    }
    assert not is_openclip_text(transformers)


# --------------------------------------------------------- conversion


def test_convert_covers_the_full_transformers_layout() -> None:
    conversion = convert_openclip_text(geometries())
    assert conversion.config == CLIP_G_TEXT_CONFIG
    expected = clip_text_layout(CLIP_G_TEXT_CONFIG)
    assert set(conversion.keys) == set(expected)
    for model_key, entry in conversion.keys.items():
        assert entry.geometry.shape == expected[model_key], model_key


def test_plain_renames_have_no_transform() -> None:
    conversion = convert_openclip_text(geometries())
    plain = {
        "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
        "text_model.embeddings.position_embedding.weight": "positional_embedding",
        "text_model.final_layer_norm.weight": "ln_final.weight",
        "text_model.encoder.layers.3.layer_norm1.weight": "transformer.resblocks.3.ln_1.weight",
        "text_model.encoder.layers.3.mlp.fc1.bias": "transformer.resblocks.3.mlp.c_fc.bias",
        "text_model.encoder.layers.3.mlp.fc2.weight": "transformer.resblocks.3.mlp.c_proj.weight",
        "text_model.encoder.layers.3.self_attn.out_proj.weight": (
            "transformer.resblocks.3.attn.out_proj.weight"
        ),
    }
    for model_key, source_key in plain.items():
        entry = conversion.keys[model_key]
        assert entry.source == source_key
        assert entry.transform is None


def test_fused_in_proj_fans_out_to_qkv_row_chunks() -> None:
    conversion = convert_openclip_text(geometries())
    hidden = CLIP_G_TEXT_CONFIG.hidden_size
    for part, proj in enumerate(("q_proj", "k_proj", "v_proj")):
        weight = conversion.keys[f"text_model.encoder.layers.0.self_attn.{proj}.weight"]
        assert weight.source == "transformer.resblocks.0.attn.in_proj_weight"
        assert weight.transform == RowChunk(part=part, parts=3)
        assert weight.geometry.shape == (hidden, hidden)
        bias = conversion.keys[f"text_model.encoder.layers.0.self_attn.{proj}.bias"]
        assert bias.source == "transformer.resblocks.0.attn.in_proj_bias"
        assert bias.transform == RowChunk(part=part, parts=3)
        assert bias.geometry.shape == (hidden,)


def test_text_projection_is_transposed() -> None:
    conversion = convert_openclip_text(geometries())
    entry = conversion.keys["text_projection.weight"]
    assert entry.source == "text_projection"
    assert entry.transform == Transpose2D()
    hidden = CLIP_G_TEXT_CONFIG.hidden_size
    assert entry.geometry.shape == (hidden, hidden)


def test_text_projection_weight_spelling_renames_without_transpose() -> None:
    # The reference accepts both spellings: bare text_projection is
    # transposed, text_projection.weight is already the Linear layout
    # (comfy/utils.py clip_text_transformers_convert @ 947c2749).
    sd = geometries()
    sd["text_projection.weight"] = sd.pop("text_projection")
    conversion = convert_openclip_text(sd)
    entry = conversion.keys["text_projection.weight"]
    assert entry.source == "text_projection.weight"
    assert entry.transform is None
    hidden = CLIP_G_TEXT_CONFIG.hidden_size
    assert entry.geometry.shape == (hidden, hidden)
    assert set(conversion.keys) == set(clip_text_layout(CLIP_G_TEXT_CONFIG))


def test_dtypes_ride_through_unchanged() -> None:
    conversion = convert_openclip_text(geometries(BFLOAT16))
    assert all(entry.geometry.dtype == BFLOAT16 for entry in conversion.keys.values())


def test_inert_siblings_are_ignored_not_errors() -> None:
    sd = geometries()
    sd["logit_scale"] = TensorGeometry((), FLOAT32)
    sd["attn_mask"] = TensorGeometry((77, 77), FLOAT16)
    conversion = convert_openclip_text(sd)
    assert set(conversion.ignored) == {"logit_scale", "attn_mask"}
    assert set(conversion.ignored) <= OPENCLIP_TEXT_INERT_KEYS


# ----------------------------------------------------------- refusals


def test_empty_header_refuses() -> None:
    with pytest.raises(OpenClipTextDetectError, match="empty"):
        convert_openclip_text({})


def test_non_openclip_header_refuses() -> None:
    with pytest.raises(OpenClipTextDetectError, match="token_embedding"):
        convert_openclip_text({"foo": TensorGeometry((1,), FLOAT16)})


def test_unknown_width_refuses() -> None:
    sd = geometries()
    sd["token_embedding.weight"] = TensorGeometry((49408, 1024), FLOAT16)
    with pytest.raises(OpenClipTextDetectError, match="unknown OpenCLIP"):
        convert_openclip_text(sd)


def test_wrong_rank_token_embedding_refuses() -> None:
    sd = geometries()
    sd["token_embedding.weight"] = TensorGeometry((49408,), FLOAT16)
    with pytest.raises(OpenClipTextDetectError, match="rank"):
        convert_openclip_text(sd)


def test_missing_key_refuses_by_name() -> None:
    sd = geometries()
    del sd["transformer.resblocks.5.attn.in_proj_weight"]
    with pytest.raises(OpenClipTextDetectError, match="resblocks.5.attn.in_proj_weight"):
        convert_openclip_text(sd)


def test_malformed_fused_projection_refuses() -> None:
    hidden = CLIP_G_TEXT_CONFIG.hidden_size
    sd = geometries()
    sd["transformer.resblocks.0.attn.in_proj_weight"] = TensorGeometry(
        (3 * hidden - 1, hidden), FLOAT16
    )
    with pytest.raises(OpenClipTextDetectError, match="in_proj_weight"):
        convert_openclip_text(sd)


def test_both_projection_spellings_refuse() -> None:
    sd = geometries()
    sd["text_projection.weight"] = sd["text_projection"]
    with pytest.raises(
        OpenClipTextDetectError,
        match="both text_projection and text_projection.weight",
    ):
        convert_openclip_text(sd)


def test_missing_projection_refuses() -> None:
    sd = geometries()
    del sd["text_projection"]
    with pytest.raises(OpenClipTextDetectError, match="missing text_projection"):
        convert_openclip_text(sd)


def test_wrong_shape_projection_refuses() -> None:
    hidden = CLIP_G_TEXT_CONFIG.hidden_size
    sd = geometries()
    sd["text_projection"] = TensorGeometry((hidden, hidden - 1), FLOAT16)
    with pytest.raises(OpenClipTextDetectError, match="text_projection"):
        convert_openclip_text(sd)


def test_unexpected_key_refuses_by_name() -> None:
    sd = geometries()
    sd["visual.proj"] = TensorGeometry((1280, 1024), FLOAT16)
    with pytest.raises(OpenClipTextDetectError, match="visual.proj"):
        convert_openclip_text(sd)


# ----------------------------------------------- transform primitives


def test_row_chunk_validates_its_indices() -> None:
    with pytest.raises(ValueError, match="parts"):
        RowChunk(part=0, parts=1)
    with pytest.raises(ValueError, match="part"):
        RowChunk(part=3, parts=3)


def test_transformed_geometry_row_chunk() -> None:
    chunk = transformed_geometry(TensorGeometry((3840, 1280), FLOAT16), RowChunk(part=1, parts=3))
    assert chunk == TensorGeometry((1280, 1280), FLOAT16)


def test_transformed_geometry_row_chunk_refuses_indivisible() -> None:
    with pytest.raises(ValueError, match="equal chunks"):
        transformed_geometry(TensorGeometry((3841, 1280), FLOAT16), RowChunk(part=0, parts=3))
    with pytest.raises(ValueError, match="at least one axis"):
        transformed_geometry(TensorGeometry((), FLOAT16), RowChunk(part=0, parts=3))


def test_transformed_geometry_transpose() -> None:
    assert transformed_geometry(TensorGeometry((2, 5), FLOAT16), Transpose2D()) == TensorGeometry(
        (5, 2), FLOAT16
    )
    with pytest.raises(ValueError, match="rank-2"):
        transformed_geometry(TensorGeometry((2, 5, 1), FLOAT16), Transpose2D())
