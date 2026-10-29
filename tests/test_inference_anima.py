"""Anima inert profile and header-detection tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, fields, replace
from typing import cast

import pytest
from dinkster_inference import (
    ANIMA_CONFIG,
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    AnimaConfig,
    Parameterization,
    TensorGeometry,
    WeightEntry,
    anima_layout,
    builtin_families,
    builtin_family_registry,
    detect_anima,
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
        raise AssertionError("Anima detection must not read metadata")

    def read_float_scalar(self, key: str) -> float:
        self.payload_reads += 1
        raise AssertionError(f"Anima detection must not read payload {key}")


def anima_shapes(prefix: str = "") -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "x_embedder.proj.1.weight": (2048, 68),
        "t_embedder.1.linear_1.weight": (2048, 2048),
        "t_embedder.1.linear_2.weight": (6144, 2048),
        "t_embedding_norm.weight": (2048,),
        "final_layer.linear.weight": (64, 2048),
        "final_layer.adaln_modulation.1.weight": (256, 2048),
        "final_layer.adaln_modulation.2.weight": (4096, 256),
        "llm_adapter.embed.weight": (32128, 1024),
        "llm_adapter.out_proj.weight": (1024, 1024),
        "llm_adapter.norm.weight": (1024,),
    }
    for index in range(28):
        shapes[f"blocks.{index}.mlp.layer1.weight"] = (8192, 2048)
        shapes[f"blocks.{index}.self_attn.q_norm.weight"] = (128,)
        shapes[f"blocks.{index}.cross_attn.k_proj.weight"] = (2048, 1024)
    for index in range(6):
        shapes[f"llm_adapter.blocks.{index}.self_attn.q_norm.weight"] = (64,)
        shapes[f"llm_adapter.blocks.{index}.cross_attn.q_proj.weight"] = (1024, 1024)
    return {prefix + key: shape for key, shape in shapes.items()}


def test_anima_config_is_exact_and_immutable() -> None:
    config = ANIMA_CONFIG
    assert config == AnimaConfig()
    assert config.family_id == "dinkster.anima"
    assert (
        config.blocks,
        config.hidden_width,
        config.attention_heads,
        config.attention_head_dim,
        config.context_width,
        config.adaln_lora_dim,
        config.patchified_input_channels,
        config.output_latent_channels,
    ) == (28, 2048, 16, 128, 1024, 256, 68, 16)
    assert config.patch == (1, 2, 2)
    assert (
        config.adapter_blocks,
        config.adapter_width,
        config.adapter_heads,
        config.adapter_head_dim,
        config.adapter_vocabulary,
    ) == (6, 1024, 16, 64, 32128)
    assert (
        config.latent_id,
        config.latent_channels,
        config.latent_dimensions,
        config.temporal_downscale,
    ) == ("Wan21", 16, 3, 4)
    assert (config.sampling_multiplier, config.sampling_shift) == (1.0, 3.0)
    assert config.inference_dtypes == (BFLOAT16, FLOAT16, FLOAT32)
    assert config.memory_factor == 1.0
    assert config.text_encoder_id == "Qwen3-0.6B"
    with pytest.raises(FrozenInstanceError):
        config.hidden_width = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    (
        "blocks",
        "hidden_width",
        "attention_heads",
        "attention_head_dim",
        "context_width",
        "adaln_lora_dim",
        "patchified_input_channels",
        "output_latent_channels",
        "adapter_blocks",
        "adapter_width",
        "adapter_heads",
        "adapter_head_dim",
        "adapter_source_width",
        "adapter_vocabulary",
        "latent_channels",
        "latent_dimensions",
        "temporal_downscale",
    ),
)
def test_anima_config_refuses_bool_and_float_at_integer_fields(field: str) -> None:
    value = getattr(ANIMA_CONFIG, field)
    for replacement in (True, float(value)):
        with pytest.raises(ValueError, match="exact supported profile"):
            replace(ANIMA_CONFIG, **{field: replacement})


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("sampling_multiplier", 1),
        ("sampling_multiplier", True),
        ("sampling_shift", True),
        ("memory_factor", True),
        ("patch", (1.0, 2, 2)),
        ("patch", (True, 2, 2)),
        ("patch", (1, 2.0, 2)),
        ("inference_dtypes", (FLOAT32, FLOAT16, BFLOAT16)),
    ),
)
def test_anima_config_refuses_equal_or_wrong_typed_substitutions(
    field: str, replacement: object
) -> None:
    with pytest.raises(ValueError, match="exact supported profile"):
        replace(ANIMA_CONFIG, **{field: replacement})


def test_anima_every_config_field_is_part_of_exact_validation() -> None:
    assert tuple(field.name for field in fields(AnimaConfig)) == (
        "family_id",
        "blocks",
        "hidden_width",
        "attention_heads",
        "attention_head_dim",
        "context_width",
        "adaln_lora_dim",
        "patchified_input_channels",
        "output_latent_channels",
        "patch",
        "adapter_blocks",
        "adapter_width",
        "adapter_heads",
        "adapter_head_dim",
        "adapter_source_width",
        "adapter_vocabulary",
        "latent_id",
        "latent_channels",
        "latent_dimensions",
        "temporal_downscale",
        "sampling_multiplier",
        "sampling_shift",
        "inference_dtypes",
        "memory_factor",
        "text_encoder_id",
    )


@pytest.mark.parametrize("prefix", ("", "net.", "model.diffusion_model."))
def test_anima_exact_headers_match_deterministically(prefix: str) -> None:
    source = HeaderSource(anima_shapes(prefix))
    original_shapes = dict(source.shapes)
    first = detect_anima(source)
    second = detect_anima(source)
    assert first == second
    assert first is not None
    assert first.config is ANIMA_CONFIG
    assert first.key_prefix == prefix
    assert first.fields == {
        "adaln_lora_dim": 256,
        "adapter_blocks": 6,
        "adapter_head_dim": 64,
        "adapter_heads": 16,
        "adapter_vocabulary": 32128,
        "adapter_width": 1024,
        "attention_head_dim": 128,
        "attention_heads": 16,
        "blocks": 28,
        "context_width": 1024,
        "hidden_width": 2048,
        "key_prefix": prefix,
        "output_latent_channels": 16,
        "patch": "1x2x2",
        "patchified_input_channels": 68,
    }
    assert first.matched_keys == tuple(sorted(first.matched_keys))
    assert set(first.matched_keys) == set(anima_shapes(prefix))
    assert source.shapes == original_shapes
    assert source.metadata_reads == source.payload_reads == 0


def test_anima_evidence_snapshots_inputs_immutably() -> None:
    source = HeaderSource(anima_shapes())
    evidence = detect_anima(source)
    assert evidence is not None
    source.shapes.clear()
    assert evidence.matched_keys
    assert evidence.fields["blocks"] == 28
    with pytest.raises(TypeError):
        evidence.fields["blocks"] = 27  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        evidence.key_prefix = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "missing",
    (
        "x_embedder.proj.1.weight",
        "t_embedder.1.linear_1.weight",
        "t_embedder.1.linear_2.weight",
        "t_embedding_norm.weight",
        "final_layer.linear.weight",
        "final_layer.adaln_modulation.1.weight",
        "final_layer.adaln_modulation.2.weight",
        "llm_adapter.embed.weight",
        "llm_adapter.out_proj.weight",
        "llm_adapter.norm.weight",
        "blocks.0.mlp.layer1.weight",
        "blocks.0.self_attn.q_norm.weight",
        "blocks.13.cross_attn.k_proj.weight",
        "blocks.27.mlp.layer1.weight",
        "blocks.27.cross_attn.k_proj.weight",
        "llm_adapter.blocks.0.self_attn.q_norm.weight",
        "llm_adapter.blocks.3.cross_attn.q_proj.weight",
        "llm_adapter.blocks.5.cross_attn.q_proj.weight",
    ),
)
def test_anima_every_dereferenced_key_is_guarded(missing: str) -> None:
    shapes = anima_shapes()
    del shapes[missing]
    assert detect_anima(HeaderSource(shapes)) is None


def test_anima_inconsistent_source_fails_closed_without_key_error() -> None:
    class InconsistentSource(HeaderSource):
        def entry(self, key: str) -> WeightEntry:
            raise KeyError(key)

    assert detect_anima(InconsistentSource(anima_shapes())) is None


@pytest.mark.parametrize(
    ("key", "shape"),
    (
        ("x_embedder.proj.1.weight", (2048, 67)),
        ("x_embedder.proj.1.weight", (2048,)),
        ("t_embedder.1.linear_1.weight", (2047, 2048)),
        ("t_embedder.1.linear_2.weight", (6143, 2048)),
        ("t_embedding_norm.weight", (2048, 1)),
        ("final_layer.linear.weight", (64, 2047)),
        ("final_layer.adaln_modulation.1.weight", (255, 2048)),
        ("final_layer.adaln_modulation.2.weight", (4096, 255)),
        ("llm_adapter.embed.weight", (32127, 1024)),
        ("llm_adapter.embed.weight", (32128, 1025)),
        ("llm_adapter.out_proj.weight", (1024, 1023)),
        ("llm_adapter.norm.weight", (1023,)),
        ("blocks.0.mlp.layer1.weight", (8191, 2048)),
        ("blocks.0.self_attn.q_norm.weight", (127,)),
        ("blocks.13.mlp.layer1.weight", (8192, 2047)),
        ("blocks.13.self_attn.q_norm.weight", (129,)),
        ("blocks.13.cross_attn.k_proj.weight", (2047, 1024)),
        ("blocks.27.cross_attn.k_proj.weight", (2048, 1023)),
        ("llm_adapter.blocks.0.self_attn.q_norm.weight", (128,)),
        ("llm_adapter.blocks.3.self_attn.q_norm.weight", (63,)),
        ("llm_adapter.blocks.3.cross_attn.q_proj.weight", (1023, 1024)),
        ("llm_adapter.blocks.5.cross_attn.q_proj.weight", (1024, 1023)),
    ),
)
def test_anima_foreign_geometry_fails_closed(key: str, shape: tuple[int, ...]) -> None:
    shapes = anima_shapes()
    shapes[key] = shape
    assert detect_anima(HeaderSource(shapes)) is None


@pytest.mark.parametrize("index", (0, 13, 27))
def test_anima_missing_dit_block_index_fails_closed(index: int) -> None:
    shapes = anima_shapes()
    del shapes[f"blocks.{index}.mlp.layer1.weight"]
    del shapes[f"blocks.{index}.self_attn.q_norm.weight"]
    del shapes[f"blocks.{index}.cross_attn.k_proj.weight"]
    assert detect_anima(HeaderSource(shapes)) is None


@pytest.mark.parametrize("index", (0, 3, 5))
def test_anima_missing_adapter_block_index_fails_closed(index: int) -> None:
    shapes = anima_shapes()
    del shapes[f"llm_adapter.blocks.{index}.self_attn.q_norm.weight"]
    del shapes[f"llm_adapter.blocks.{index}.cross_attn.q_proj.weight"]
    assert detect_anima(HeaderSource(shapes)) is None


@pytest.mark.parametrize("namespace", ("28", "100", "01", "00", "-1", "1x", ""))
def test_anima_extra_malformed_or_duplicate_spelled_dit_block_refuses(
    namespace: str,
) -> None:
    shapes = anima_shapes()
    shapes[f"blocks.{namespace}.self_attn.q_norm.weight"] = (128,)
    assert detect_anima(HeaderSource(shapes)) is None


@pytest.mark.parametrize("namespace", ("6", "10", "01", "00", "-1", "1x", ""))
def test_anima_extra_malformed_or_duplicate_spelled_adapter_block_refuses(
    namespace: str,
) -> None:
    shapes = anima_shapes()
    shapes[f"llm_adapter.blocks.{namespace}.self_attn.q_norm.weight"] = (64,)
    assert detect_anima(HeaderSource(shapes)) is None


def test_anima_unbounded_block_namespace_fails_closed() -> None:
    shapes = anima_shapes()
    shapes[f"blocks.{'9' * 5000}.self_attn.q_norm.weight"] = (128,)
    assert detect_anima(HeaderSource(shapes)) is None


def test_anima_detector_does_not_absorb_other_families() -> None:
    source = HeaderSource(anima_shapes())
    assert detect_qwen_image(source) is None
    assert detect_minimax_h3(source) is None

    qwen_like = {
        "txt_norm.weight": (3584,),
        "img_in.weight": (3072, 64),
        "txt_in.weight": (3072, 3584),
        "proj_out.weight": (64, 3072),
        "time_text_embed.timestep_embedder.linear_2.weight": (3072, 3072),
        **{f"transformer_blocks.{index}.attn.norm_q.weight": (128,) for index in range(60)},
    }
    assert detect_anima(HeaderSource(qwen_like)) is None


def test_anima_predict2_without_adapter_is_not_anima() -> None:
    shapes = {
        key: shape for key, shape in anima_shapes().items() if not key.startswith("llm_adapter.")
    }
    assert detect_anima(HeaderSource(shapes)) is None


def test_anima_catalog_preserves_architecture_and_sampling() -> None:
    families = {family.id: family for family in builtin_families()}
    anima = families["dinkster.anima"]
    assert anima.display_name == "Anima"
    assert anima.specificity == 100
    latent = anima.single_stream_latent()
    assert (latent.channels, latent.dimensions) == (16, 3)
    assert anima.sampling.parameterization is Parameterization.FLOW
    assert anima.sampling.shift == 3.0
    assert anima.sampling.sigma_max == 1.0
    # ModelSamplingDiscreteFlow @ shift=3, multiplier=1: first step,
    # 3 * 0.001 / (1 + 2 * 0.001)
    assert anima.sampling.sigma_min == 0.0029940119760479044
    assert anima.wiring.vae_prefix == "vae."
    assert anima.wiring.text_encoder_prefix == "text_encoders."
    assert anima.wiring.text_encoders == ("dinkster.qwen3_06b",)
    assert anima.supported_dtypes == frozenset({BFLOAT16, FLOAT16, FLOAT32})
    assert anima.memory_factor == 1.0


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_anima_registry_detection_is_unambiguous(prefix: str) -> None:
    class RegistrySource(HeaderSource):
        """Other registered detectors may read metadata; only Anima's own
        detection is held to the header-only contract."""

        def metadata(self) -> Mapping[str, str]:
            return {}

    result = builtin_family_registry().detect(RegistrySource(anima_shapes(prefix)))
    assert result.best is not None
    assert result.best.family_id == "dinkster.anima"
    assert result.ambiguous == ()


def test_anima_layout_covers_every_detection_pin_exactly() -> None:
    layout = anima_layout()
    for key, shape in anima_shapes().items():
        assert layout[key] == shape


def test_anima_layout_key_and_bias_counts() -> None:
    layout = anima_layout()
    # 7 backbone top-level + 28 blocks * 20 + embed + 6 adapter blocks * 19
    # + out_proj weight/bias + final norm.
    assert len(layout) == 7 + 28 * 20 + 1 + 6 * 19 + 3
    biases = sorted(key for key in layout if key.endswith(".bias"))
    assert len(biases) == 6 * 2 + 1
    assert all(key.startswith("llm_adapter.") for key in biases)


def test_anima_layout_shapes_are_all_int_tuples() -> None:
    for key, shape in anima_layout().items():
        assert type(shape) is tuple, key
        assert all(type(extent) is int and extent > 0 for extent in shape), key


def test_anima_layout_refuses_inconsistent_geometry() -> None:
    with pytest.raises(ValueError, match="attention_heads"):
        anima_layout(replace_config(attention_heads=12))
    with pytest.raises(ValueError, match="padding-mask"):
        anima_layout(replace_config(patchified_input_channels=64))
    with pytest.raises(ValueError, match="adapter_heads"):
        anima_layout(replace_config(adapter_heads=8))
    with pytest.raises(ValueError, match="input projection"):
        anima_layout(replace_config(context_width=2048))


def replace_config(**overrides: int) -> AnimaConfig:
    """A structurally inconsistent config for refusal tests; bypasses
    the exact-profile validator the way reduced torch-test geometries
    do (dataclass cast, no construction)."""

    class _Duck:
        def __init__(self) -> None:
            for field in fields(AnimaConfig):
                setattr(self, field.name, getattr(ANIMA_CONFIG, field.name))
            for name, value in overrides.items():
                setattr(self, name, value)

    return cast("AnimaConfig", _Duck())
