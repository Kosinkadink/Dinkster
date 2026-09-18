from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT8,
    MINIMAX_MUSIC3,
    MINIMAX_MUSIC3_CONFIG,
    MINIMAX_MUSIC3_DAV_DESCRIPTOR,
    MINIMAX_MUSIC3_LATENT,
    OFFICIAL_ARTIFACT_REVISION,
    OFFICIAL_ARTIFACTS,
    UINT8,
    DType,
    MiniMaxMusic3TextConfig,
    Parameterization,
    TensorGeometry,
    WeightEntry,
    build_music_prompt,
    builtin_families,
    clean_music_caption,
    default_text_dtype,
    derive_music_seed,
    detect_minimax_music3,
    detect_minimax_music3_text_config,
    minimax_music3_dav_layout,
    minimax_music3_diffusion_layout,
    minimax_music3_latent_length,
    minimax_music3_text_layout,
    normalize_music_lyrics,
    plan_minimax_music3_component,
    plan_minimax_music3_split_component,
)


class HeaderSource:
    def __init__(
        self,
        path: Path,
        geometries: Mapping[str, TensorGeometry],
        *,
        asset_digest: str | None = None,
        asset_size: int | None = None,
    ) -> None:
        self.path = path
        self.geometries = dict(geometries)
        self.asset_digest = asset_digest
        self.asset_size = asset_size

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def _geometries(
    layout: Mapping[str, tuple[int, ...]],
    dtype: DType,
    *,
    prefix: str = "",
) -> dict[str, TensorGeometry]:
    return {prefix + key: TensorGeometry(shape, dtype) for key, shape in layout.items()}


def _text_geometries(
    config: MiniMaxMusic3TextConfig,
    floating_dtype: DType = BFLOAT16,
) -> dict[str, TensorGeometry]:
    result: dict[str, TensorGeometry] = {}
    for key, shape in minimax_music3_text_layout(config).items():
        if key == "tokenizer_json":
            dtype = UINT8
        elif config.storage_format == "int8_pruned" and key.endswith(
            (
                ".self_attn.qkv_proj.weight",
                ".self_attn.o_proj.weight",
                ".mlp.gate_up_proj.weight",
                ".mlp.down_proj.weight",
            )
        ):
            dtype = INT8
        else:
            dtype = floating_dtype
        result[key] = TensorGeometry(shape, dtype)
    return result


def test_official_artifact_receipt_is_immutable_and_complete() -> None:
    assert OFFICIAL_ARTIFACT_REVISION == "6baad88896848433857c170ba4f05d2ea9d5f218"
    assert set(OFFICIAL_ARTIFACTS) == {
        "diffusion_models/minimax_music3_dit_fp16.safetensors",
        "diffusion_models/minimax_music3_dit_fp32.safetensors",
        "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
        "text_encoders/minimax_music3_text_encoder_bf16.safetensors",
        "text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors",
        "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        "vae/minimax_music3_dav.safetensors",
    }
    for path, (url, size, digest) in OFFICIAL_ARTIFACTS.items():
        assert url.endswith(f"/{OFFICIAL_ARTIFACT_REVISION}/{path}")
        assert size > 0
        assert len(digest) == 64 and digest == digest.lower()
    with pytest.raises(TypeError):
        OFFICIAL_ARTIFACTS["other"] = ("url", 1, "0" * 64)  # type: ignore[index]


def test_prompt_cleanup_and_seed_match_source_contract() -> None:
    caption = "# **Fast** song\n\n* <|tempo 120|>\n---\n\u2022 bright"
    lyrics = "[VERSE 1] Hello ^ World [CHORUS] Go"
    assert clean_music_caption(caption) == "Fast song\ntempo is 120\nbright"
    assert normalize_music_lyrics(lyrics) == "[start]\n[verse 1]\nHello\nWorld\n[chorus]\nGo"
    assert build_music_prompt(caption, lyrics) == (
        "<|im_start|><|caption_start|>Fast song\ntempo is 120\nbright"
        "<|caption_end|><|lyrics_start|>[start]\n[verse 1]\nHello\nWorld\n"
        "[chorus]\nGo<|lyrics_end|><|im_end|><|audio_start|>"
    )
    assert derive_music_seed(42, "ar") == 146933486985881370
    assert derive_music_seed(42, "ar") == derive_music_seed(42, "ar")
    assert derive_music_seed(42, "ar") != derive_music_seed(42, "depth")


@pytest.mark.parametrize(
    ("frames", "length"),
    ((0, 1), (1, 3), (25, 86), (100, 344), (200, 689), (9000, 31007)),
)
def test_latent_geometry_matches_source(frames: int, length: int) -> None:
    assert minimax_music3_latent_length(frames) == length


def test_family_catalog_and_codec_descriptors_are_native_audio() -> None:
    assert MINIMAX_MUSIC3 in builtin_families()
    assert MINIMAX_MUSIC3.id == "dinkster.minimax_music3"
    assert MINIMAX_MUSIC3.sampling.parameterization is Parameterization.FLOW
    assert MINIMAX_MUSIC3.memory_factor == 2.0
    assert MINIMAX_MUSIC3.latent is MINIMAX_MUSIC3_LATENT
    assert (MINIMAX_MUSIC3_LATENT.channels, MINIMAX_MUSIC3_LATENT.dimensions) == (128, 1)
    descriptor = MINIMAX_MUSIC3_DAV_DESCRIPTOR
    assert descriptor.kind == "audio"
    assert descriptor.content_channels == 2
    assert descriptor.supported_dtypes == frozenset({FLOAT32})
    assert descriptor.tiling is not None
    assert (descriptor.tiling.decode_tile, descriptor.tiling.decode_overlap) == ((1536,), (64,))


def test_exact_diffusion_layout_detects_bare_and_prefixed_sources() -> None:
    layout = minimax_music3_diffusion_layout()
    assert len(layout) == 374
    for prefix in ("", "model.diffusion_model."):
        source = HeaderSource(Path("dit.safetensors"), _geometries(layout, FLOAT16, prefix=prefix))
        evidence = detect_minimax_music3(source)
        assert evidence is not None
        assert evidence.config is MINIMAX_MUSIC3_CONFIG
        assert evidence.key_prefix == prefix
        assert evidence.fields["blocks"] == 36
        assert len(evidence.matched_keys) == len(layout)
        plan = plan_minimax_music3_component(source, "diffusion")
        assert plan.component == "diffusion"
        assert set(plan.keys) == set(layout)


def test_diffusion_detection_refuses_near_match() -> None:
    geometries = _geometries(minimax_music3_diffusion_layout(), FLOAT16)
    geometries.pop("diffusion_transformer.transformer.layers.35.self_attn.to_out.weight")
    assert detect_minimax_music3(HeaderSource(Path("near.safetensors"), geometries)) is None


def test_diffusion_planner_refuses_unquantized_integer_storage() -> None:
    geometries = _geometries(minimax_music3_diffusion_layout(), FLOAT16)
    key = next(iter(geometries))
    geometries[key] = TensorGeometry(geometries[key].shape, UINT8)
    with pytest.raises(ValueError, match="require floating-point storage"):
        plan_minimax_music3_component(
            HeaderSource(Path("integer.safetensors"), geometries), "diffusion"
        )


@pytest.mark.parametrize(
    ("config", "floating_dtype"),
    (
        (MiniMaxMusic3TextConfig(False, False, False, False, False, "floating"), BFLOAT16),
        (MiniMaxMusic3TextConfig(False, False, False, False, False, "floating"), FLOAT64),
        (MiniMaxMusic3TextConfig(True, True, True, True, True, "floating"), BFLOAT16),
        (MiniMaxMusic3TextConfig(True, True, True, True, True, "int8_pruned"), BFLOAT16),
    ),
)
def test_every_official_text_layout_is_exactly_detected_and_planned(
    config: MiniMaxMusic3TextConfig,
    floating_dtype: DType,
) -> None:
    geometries = _text_geometries(config, floating_dtype)
    assert len(geometries) == (447 if not config.pruned else 328)
    assert detect_minimax_music3_text_config(geometries) == config
    if config.storage_format == "floating":
        plan = plan_minimax_music3_component(
            HeaderSource(Path("text.safetensors"), geometries), "text"
        )
        assert plan.config == config
        assert "tokenizer_json" in plan.ignored
        assert "tokenizer_json" not in plan.keys
    else:
        with pytest.raises(ValueError, match="require ConvRot metadata"):
            plan_minimax_music3_component(
                HeaderSource(Path("text.safetensors"), geometries), "text"
            )


def test_text_compute_uses_the_component_default() -> None:
    assert default_text_dtype("dinkster.minimax_music3") is BFLOAT16


def test_floating_text_storage_is_validated_by_kind_per_tensor() -> None:
    config = MiniMaxMusic3TextConfig(False, False, False, False, False, "floating")
    geometries = _text_geometries(config, FLOAT64)
    key = "model.layers.0.self_attn.q_proj.weight"
    geometries[key] = TensorGeometry(geometries[key].shape, BFLOAT16)

    assert detect_minimax_music3_text_config(geometries) == config
    plan = plan_minimax_music3_component(
        HeaderSource(Path("mixed-floating-text.safetensors"), geometries), "text"
    )
    assert set(plan.dtypes.values()) == {BFLOAT16, FLOAT64}


def test_unpruned_text_layout_refuses_int8_with_declared_reason() -> None:
    config = MiniMaxMusic3TextConfig(False, False, False, False, False, "floating")
    geometries = _text_geometries(config, FLOAT32)
    key = "model.layers.0.self_attn.q_proj.weight"
    geometries[key] = TensorGeometry(geometries[key].shape, INT8)

    with pytest.raises(ValueError, match="INT8 storage requires the pruned text layout"):
        detect_minimax_music3_text_config(geometries)


def test_tokenizer_dtype_does_not_select_the_int8_weight_format() -> None:
    config = MiniMaxMusic3TextConfig(False, False, False, False, False, "floating")
    geometries = _text_geometries(config)
    tokenizer = geometries["tokenizer_json"]
    geometries["tokenizer_json"] = TensorGeometry(tokenizer.shape, INT8)

    with pytest.raises(ValueError, match="tokenizer_json must be uint8"):
        detect_minimax_music3_text_config(geometries)


def test_dav_layout_is_exact_float32_and_decode_only() -> None:
    layout = minimax_music3_dav_layout()
    assert len(layout) == 121
    source = HeaderSource(Path("dav.safetensors"), _geometries(layout, FLOAT32))
    plan = plan_minimax_music3_component(source, "vae")
    assert plan.component == "vae"
    assert set(plan.keys) == set(layout)
    wrong = _geometries(layout, FLOAT16)
    with pytest.raises(ValueError, match="exact MiniMax Music 3 DAV"):
        plan_minimax_music3_component(HeaderSource(Path("dav.safetensors"), wrong), "vae")


def test_split_component_identity_binds_asset_receipt() -> None:
    path = Path("dit.safetensors")
    digest = "blake3:" + "1" * 64
    source = HeaderSource(
        path,
        _geometries(minimax_music3_diffusion_layout(), FLOAT16),
        asset_digest=digest,
        asset_size=123,
    )
    plan = plan_minimax_music3_split_component(source, role="diffusion", path=path)
    assert f"asset_digest={digest}" in plan.identity_facts
    assert "asset_size=123" in plan.identity_facts
    with pytest.raises(ValueError, match="path differs"):
        plan_minimax_music3_split_component(source, role="diffusion", path=Path("other"))


def test_config_is_immutable_and_exact() -> None:
    with pytest.raises(FrozenInstanceError):
        MINIMAX_MUSIC3_CONFIG.blocks = 35  # type: ignore[misc]
    with pytest.raises(ValueError, match="exact supported profile"):
        type(MINIMAX_MUSIC3_CONFIG)(blocks=35)
