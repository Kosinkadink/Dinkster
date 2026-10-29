"""Qwen Image S1 inert profile and header-detection tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, fields, replace

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    NATIVE_WIRED_FAMILY_IDS,
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_2511_CONFIG,
    QWEN_IMAGE_LAYERED_CONFIG,
    QwenImageConfig,
    TensorGeometry,
    WeightEntry,
    builtin_families,
    detect_minimax_h3,
    detect_qwen_image,
)


class HeaderSource:
    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = dict(shapes)
        self.entries_read: list[str] = []
        self.metadata_reads = 0
        self.payload_reads = 0

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        self.entries_read.append(key)
        geometry = TensorGeometry(self.shapes[key], FLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        self.metadata_reads += 1
        raise AssertionError("Qwen Image detection must not read metadata")

    def read_float_scalar(self, key: str) -> float:
        self.payload_reads += 1
        raise AssertionError(f"Qwen Image detection must not read payload {key}")


def qwen_image_shapes(prefix: str = "") -> dict[str, tuple[int, ...]]:
    shapes = {
        "txt_norm.weight": (3584,),
        "img_in.weight": (3072, 64),
        "txt_in.weight": (3072, 3584),
        "proj_out.weight": (64, 3072),
        "time_text_embed.timestep_embedder.linear_2.weight": (3072, 3072),
    }
    for index in range(60):
        shapes[f"transformer_blocks.{index}.attn.norm_q.weight"] = (128,)
    return {prefix + key: shape for key, shape in shapes.items()}


def test_qwen_image_config_is_exact_and_immutable() -> None:
    config = QWEN_IMAGE_CONFIG
    assert config == QwenImageConfig()
    assert config.family_id == "dinkster.qwen_image"
    assert (
        config.transformer_blocks,
        config.hidden_width,
        config.attention_heads,
        config.attention_head_dim,
        config.text_width,
        config.pooled_width,
        config.patchified_input_channels,
        config.output_latent_channels,
    ) == (60, 3072, 24, 128, 3584, 768, 64, 16)
    assert config.patch == (2, 2)
    assert config.rope_axes == (16, 56, 56)
    assert (
        config.latent_id,
        config.latent_channels,
        config.latent_dimensions,
        config.temporal_downscale,
    ) == ("Wan21", 16, 3, 4)
    assert (config.sampling_multiplier, config.sampling_shift) == (1.0, 1.15)
    assert config.inference_dtypes == (BFLOAT16, FLOAT32)
    assert config.memory_factor == 1.8
    assert config.text_encoder_id == "Qwen2.5-VL-7B"
    assert config.default_ref_method == "index"
    assert config.use_additional_t_cond is False
    with pytest.raises(FrozenInstanceError):
        config.hidden_width = 1  # type: ignore[misc]


def test_qwen_image_variant_configs_are_exact_and_immutable() -> None:
    assert QWEN_IMAGE_EDIT_2511_CONFIG == QwenImageConfig(default_ref_method="index_timestep_zero")
    assert QWEN_IMAGE_EDIT_2511_CONFIG.use_additional_t_cond is False
    assert QWEN_IMAGE_LAYERED_CONFIG == QwenImageConfig(
        default_ref_method="negative_index", use_additional_t_cond=True
    )


@pytest.mark.parametrize(
    "field",
    (
        "transformer_blocks",
        "hidden_width",
        "attention_heads",
        "attention_head_dim",
        "text_width",
        "pooled_width",
        "patchified_input_channels",
        "output_latent_channels",
        "latent_channels",
        "latent_dimensions",
        "temporal_downscale",
    ),
)
def test_qwen_image_config_refuses_bool_and_float_at_integer_fields(field: str) -> None:
    value = getattr(QWEN_IMAGE_CONFIG, field)
    for replacement in (True, float(value)):
        with pytest.raises(ValueError, match="exact supported profile"):
            replace(QWEN_IMAGE_CONFIG, **{field: replacement})


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("sampling_multiplier", 1),
        ("sampling_multiplier", True),
        ("sampling_shift", True),
        ("memory_factor", True),
        ("patch", (2.0, 2)),
        ("patch", (True, 2)),
        ("rope_axes", (16.0, 56, 56)),
        ("rope_axes", (True, 56, 56)),
        ("inference_dtypes", (FLOAT32, BFLOAT16)),
        ("default_ref_method", "negative_index"),
        ("use_additional_t_cond", 0),
    ),
)
def test_qwen_image_config_refuses_equal_or_wrong_typed_substitutions(
    field: str, replacement: object
) -> None:
    with pytest.raises(ValueError, match="exact supported profile"):
        replace(QWEN_IMAGE_CONFIG, **{field: replacement})


def test_qwen_image_every_config_field_is_part_of_exact_validation() -> None:
    assert tuple(field.name for field in fields(QwenImageConfig)) == (
        "family_id",
        "transformer_blocks",
        "hidden_width",
        "attention_heads",
        "attention_head_dim",
        "text_width",
        "pooled_width",
        "patchified_input_channels",
        "output_latent_channels",
        "patch",
        "rope_axes",
        "latent_id",
        "latent_channels",
        "latent_dimensions",
        "temporal_downscale",
        "sampling_multiplier",
        "sampling_shift",
        "inference_dtypes",
        "memory_factor",
        "text_encoder_id",
        "default_ref_method",
        "use_additional_t_cond",
    )


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_qwen_image_exact_headers_match_deterministically(prefix: str) -> None:
    source = HeaderSource(qwen_image_shapes(prefix))
    original_shapes = dict(source.shapes)
    first = detect_qwen_image(source)
    second = detect_qwen_image(source)
    assert first == second
    assert first is not None
    assert first.config is QWEN_IMAGE_CONFIG
    assert first.key_prefix == prefix
    assert first.fields == {
        "attention_head_dim": 128,
        "attention_heads": 24,
        "default_ref_method": "index",
        "hidden_width": 3072,
        "key_prefix": prefix,
        "output_latent_channels": 16,
        "patch": "2x2",
        "patchified_input_channels": 64,
        "pooled_width": 768,
        "text_width": 3584,
        "transformer_blocks": 60,
        "use_additional_t_cond": False,
    }
    assert first.matched_keys == tuple(sorted(first.matched_keys))
    assert source.shapes == original_shapes
    assert source.metadata_reads == source.payload_reads == 0


def test_qwen_image_evidence_snapshots_inputs_immutably() -> None:
    source = HeaderSource(qwen_image_shapes())
    evidence = detect_qwen_image(source)
    assert evidence is not None
    source.shapes.clear()
    assert evidence.matched_keys
    assert evidence.fields["transformer_blocks"] == 60
    with pytest.raises(TypeError):
        evidence.fields["transformer_blocks"] = 59  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        evidence.key_prefix = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "missing",
    (
        "txt_norm.weight",
        "img_in.weight",
        "txt_in.weight",
        "proj_out.weight",
        "time_text_embed.timestep_embedder.linear_2.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.59.attn.norm_q.weight",
    ),
)
def test_qwen_image_every_dereferenced_key_is_guarded(missing: str) -> None:
    shapes = qwen_image_shapes()
    del shapes[missing]
    assert detect_qwen_image(HeaderSource(shapes)) is None


def test_qwen_image_inconsistent_source_fails_closed_without_key_error() -> None:
    class InconsistentSource(HeaderSource):
        def entry(self, key: str) -> WeightEntry:
            raise KeyError(key)

    assert detect_qwen_image(InconsistentSource(qwen_image_shapes())) is None


@pytest.mark.parametrize(
    ("key", "shape"),
    (
        ("txt_norm.weight", (3583,)),
        ("txt_norm.weight", (3584, 1)),
        ("img_in.weight", (3071, 64)),
        ("img_in.weight", (3072,)),
        ("txt_in.weight", (3071, 3584)),
        ("proj_out.weight", (63, 3072)),
        ("time_text_embed.timestep_embedder.linear_2.weight", (3071, 3072)),
        ("transformer_blocks.0.attn.norm_q.weight", (127,)),
    ),
)
def test_qwen_image_foreign_geometry_fails_closed(key: str, shape: tuple[int, ...]) -> None:
    shapes = qwen_image_shapes()
    shapes[key] = shape
    assert detect_qwen_image(HeaderSource(shapes)) is None


@pytest.mark.parametrize(
    "key",
    (
        "img_in.weight",
        "txt_in.weight",
        "proj_out.weight",
        "time_text_embed.timestep_embedder.linear_2.weight",
    ),
)
def test_qwen_image_linear_detection_ignores_packed_second_axis(key: str) -> None:
    shapes = qwen_image_shapes()
    shapes[key] = (shapes[key][0], 1)
    assert detect_qwen_image(HeaderSource(shapes)) is not None


@pytest.mark.parametrize("index", (0, 17, 59))
def test_qwen_image_missing_block_index_fails_closed(index: int) -> None:
    shapes = qwen_image_shapes()
    del shapes[f"transformer_blocks.{index}.attn.norm_q.weight"]
    assert detect_qwen_image(HeaderSource(shapes)) is None


@pytest.mark.parametrize("namespace", ("60", "100", "01", "00", "-1", "1x", ""))
def test_qwen_image_extra_malformed_or_duplicate_spelled_block_refuses(
    namespace: str,
) -> None:
    shapes = qwen_image_shapes()
    shapes[f"transformer_blocks.{namespace}.attn.norm_q.weight"] = (128,)
    assert detect_qwen_image(HeaderSource(shapes)) is None


def test_qwen_image_unbounded_block_namespace_fails_closed() -> None:
    shapes = qwen_image_shapes()
    shapes[f"transformer_blocks.{'9' * 5000}.attn.norm_q.weight"] = (128,)
    assert detect_qwen_image(HeaderSource(shapes)) is None


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_qwen_image_edit_2511_profile_is_detected(prefix: str) -> None:
    shapes = qwen_image_shapes(prefix)
    marker = prefix + "__index_timestep_zero__"
    shapes[marker] = (0,)
    source = HeaderSource(shapes)
    evidence = detect_qwen_image(source)
    assert evidence is not None
    assert evidence.config is QWEN_IMAGE_EDIT_2511_CONFIG
    assert evidence.fields["default_ref_method"] == "index_timestep_zero"
    assert evidence.fields["use_additional_t_cond"] is False
    assert marker in evidence.matched_keys
    assert marker in source.entries_read


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_qwen_image_layered_profile_is_detected_and_takes_precedence(prefix: str) -> None:
    shapes = qwen_image_shapes(prefix)
    index_marker = prefix + "__index_timestep_zero__"
    layered_marker = prefix + "time_text_embed.addition_t_embedding.weight"
    shapes[index_marker] = (0,)
    shapes[layered_marker] = (2, 3072)
    evidence = detect_qwen_image(HeaderSource(shapes))
    assert evidence is not None
    assert evidence.config is QWEN_IMAGE_LAYERED_CONFIG
    assert evidence.fields["default_ref_method"] == "negative_index"
    assert evidence.fields["use_additional_t_cond"] is True
    assert index_marker in evidence.matched_keys
    assert layered_marker in evidence.matched_keys


@pytest.mark.parametrize(
    ("marker", "shape"),
    (
        ("__index_timestep_zero__", (1,)),
        ("time_text_embed.addition_t_embedding.weight", (1, 3072)),
        ("time_text_embed.addition_t_embedding.weight", (2, 3071)),
    ),
)
def test_qwen_image_variant_marker_geometry_fails_closed(
    marker: str, shape: tuple[int, ...]
) -> None:
    shapes = qwen_image_shapes()
    shapes[marker] = shape
    assert detect_qwen_image(HeaderSource(shapes)) is None


def test_qwen_image_mage_flow_and_truncated_headers_refuse() -> None:
    mage = qwen_image_shapes()
    mage["txt_norm.weight"] = (2560,)
    mage["proj_out.weight"] = (128, 3072)
    assert detect_qwen_image(HeaderSource(mage)) is None
    assert detect_qwen_image(HeaderSource({"txt_norm.weight": (3584,)})) is None


def test_qwen_image_catalog_registration_is_native_wired() -> None:
    family_ids = tuple(family.id for family in builtin_families())
    assert family_ids == (
        "dinkster.sd15",
        "dinkster.sdxl",
        "dinkster.sdxl_refiner",
        "dinkster.chroma",
        "dinkster.chroma_radiance",
        "dinkster.flux_dev",
        "dinkster.flux_schnell",
        "dinkster.flux2_dev",
        "dinkster.flux2_klein_9b",
        "dinkster.flux2_klein_4b",
        "dinkster.wan21",
        "dinkster.wan22",
        "dinkster.ltxv",
        "dinkster.ltxav",
        "dinkster.qwen_image",
        "dinkster.z_image",
        "dinkster.z_image_pixel_space",
        "dinkster.minimax_h3",
        "dinkster.minimax_music3",
        "dinkster.krea2",
        "dinkster.ideogram4",
        "dinkster.seedvr2",
        "dinkster.anima",
        "dinkster.lumina2",
        "dinkster.triposplat",
        "dinkster.trellis2",
    )
    assert QWEN_IMAGE_CONFIG.family_id in family_ids
    assert QWEN_IMAGE_CONFIG.family_id in NATIVE_WIRED_FAMILY_IDS
    assert not hasattr(QWEN_IMAGE_CONFIG, "runtime")


def test_qwen_image_detector_does_not_absorb_existing_h3_geometry() -> None:
    h3 = {
        "txt_norm.weight": (3584,),
        **{f"blocks.{index}.attn.q_norm.weight": (128,) for index in range(50)},
    }
    source = HeaderSource(h3)
    assert detect_qwen_image(source) is None
    assert detect_minimax_h3(source) is None
