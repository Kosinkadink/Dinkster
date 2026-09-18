"""Proving tests for the torch-free Krea 2 DiT layer.

Layout generation and detection run over TensorGeometry mappings only
(headers, never payloads). The single accepted geometry (RAW and
Turbo share one tensor layout) is pinned against the EXECUTED
reference's own full-size state-dict listing
(goldens/krea2_dit_goldens.json "layouts", produced by the reference
SingleStreamDiT with its constructor defaults @ the audited baseline
on the meta device), and every documented rejection branch refuses
with a Krea2DetectError naming what it found.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLUX_DEV_CONFIG,
    KREA2,
    KREA2_CONFIG,
    KREA2_SIGMAS,
    WAN21_LATENT,
    FluxFlowSigmas,
    Krea2Config,
    Krea2DetectError,
    LatentDescriptor,
    Parameterization,
    TensorGeometry,
    WeightEntry,
    builtin_families,
    builtin_family_registry,
    detect_krea2,
    detect_krea2_config,
    flux_layout,
    krea2_layout,
)

GOLDENS = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "krea2_dit_goldens.json"
)


class HeaderSource:
    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = dict(shapes)

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], BFLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        raise AssertionError("Krea 2 detection must not read metadata")


def golden_layout() -> list[tuple[str, list[int]]]:
    payload = json.loads(GOLDENS.read_text())
    return [(key, list(shape)) for key, shape in payload["layouts"]["krea2"]]


def geometries_of(
    layout: list[tuple[str, list[int]]],
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in layout}


# ------------------------------------------------------------ config


def test_config_is_the_reference_geometry() -> None:
    config = KREA2_CONFIG
    assert config.family_id == "dinkster.krea2"
    assert config.features == 6144
    assert config.transformer_blocks == 28
    assert config.attention_heads == 48
    assert config.kv_heads == 12
    assert config.attention_head_dim == 128
    assert config.mlp_width == 16384
    assert config.time_width == 256
    assert config.text_width == 2560
    assert config.text_layers == 12
    assert config.text_heads == 20
    assert config.text_kv_heads == 20
    assert config.text_mlp_width == 6912
    assert config.text_fusion_layerwise_blocks == 2
    assert config.text_fusion_refiner_blocks == 2
    assert config.latent_channels == 16
    assert config.patch == (2, 2)
    assert config.rope_axes == (32, 48, 48)
    assert config.rope_theta == 1000.0
    assert config.rms_norm_eps == 1e-5
    assert config.latent_id == "Wan21"
    assert config.latent_dimensions == 3
    assert config.temporal_downscale == 4
    assert config.sampling_multiplier == 1.0
    assert config.sampling_shift == 1.15
    assert config.inference_dtypes == (BFLOAT16, FLOAT16, FLOAT32)
    assert config.memory_factor == 2.2
    assert config.text_encoder_id == "Qwen3-VL-4B"


def test_swiglu_widths_follow_the_reference_rounding_rule() -> None:
    """int(2 * width / 3) times the reference constructor's multiplier
    (4), rounded up to a multiple of 128."""

    def swiglu(width: int) -> int:
        raw = int(2 * width / 3) * 4
        return -(-raw // 128) * 128

    assert KREA2_CONFIG.mlp_width == swiglu(KREA2_CONFIG.features)
    assert KREA2_CONFIG.text_mlp_width == swiglu(KREA2_CONFIG.text_width)


def test_rope_axes_split_the_full_head() -> None:
    assert sum(KREA2_CONFIG.rope_axes) == KREA2_CONFIG.attention_head_dim
    assert KREA2_CONFIG.attention_heads * KREA2_CONFIG.attention_head_dim == KREA2_CONFIG.features


@pytest.mark.parametrize("field", sorted(field.name for field in dataclasses.fields(Krea2Config)))
def test_config_refuses_any_other_geometry(field: str) -> None:
    value = getattr(KREA2_CONFIG, field)
    if isinstance(value, str):
        mutated: object = value + "x"
    elif isinstance(value, int):
        mutated = value + 1
    elif isinstance(value, float):
        mutated = value + 1.0
    else:
        mutated = (*value, 1)
    with pytest.raises(ValueError, match="exact published Krea 2"):
        dataclasses.replace(KREA2_CONFIG, **{field: mutated})


def test_sigmas_pin_the_reference_sampling() -> None:
    """ModelType.FLUX: ModelSamplingFlux over 10000 timesteps of
    flux_time_shift(1.15, 1.0, t) (comfy/model_base.py Krea2,
    comfy/model_sampling.py ModelSamplingFlux @ 947c2749)."""
    assert KREA2_SIGMAS == FluxFlowSigmas(shift=1.15, timesteps=10000)
    assert KREA2_SIGMAS.sigma_max == 1.0
    assert KREA2_SIGMAS.sigma_min == math.exp(1.15) / (math.exp(1.15) + 9999)


# ------------------------------------------------------------ layout


def test_layout_matches_the_executed_reference_listing() -> None:
    predicted = sorted((key, list(shape)) for key, shape in krea2_layout().items())
    assert len(predicted) == 430
    assert predicted == sorted(golden_layout())


def test_layout_is_immutable() -> None:
    layout = krea2_layout()
    with pytest.raises(TypeError):
        layout["first.weight"] = (1,)  # type: ignore[index]


# --------------------------------------------------------- detection


def test_detects_the_published_geometry() -> None:
    assert detect_krea2_config(geometries_of(golden_layout())) is KREA2_CONFIG


def test_rejects_empty_headers() -> None:
    with pytest.raises(Krea2DetectError, match="empty state dict"):
        detect_krea2_config({})


def test_rejects_foreign_headers() -> None:
    geometries = {
        key: TensorGeometry(shape, FLOAT32) for key, shape in flux_layout(FLUX_DEV_CONFIG).items()
    }
    with pytest.raises(Krea2DetectError, match="not a Krea 2 DiT"):
        detect_krea2_config(geometries)


def test_rejects_quantized_repackages_with_extra_scale_tensors() -> None:
    """An fp8-style repackage adds input_scale / weight_scale tensors;
    an exact key-set match refuses them instead of misreading the
    checkpoint."""
    geometries = geometries_of(golden_layout())
    geometries["blocks.0.attn.wq.input_scale"] = TensorGeometry((), FLOAT32)
    with pytest.raises(Krea2DetectError, match="unexpected key"):
        detect_krea2_config(geometries)


def test_rejects_truncated_headers() -> None:
    geometries = geometries_of(golden_layout())
    del geometries["blocks.27.mlp.down.weight"]
    with pytest.raises(Krea2DetectError, match="missing blocks.27.mlp.down.weight"):
        detect_krea2_config(geometries)


def test_rejects_mismatched_shapes() -> None:
    geometries = geometries_of(golden_layout())
    geometries["first.weight"] = TensorGeometry((6144, 65), FLOAT32)
    with pytest.raises(Krea2DetectError, match="first.weight: expected shape"):
        detect_krea2_config(geometries)


# ------------------------------------------------ family detector seam


def test_family_detector_returns_evidence_for_bare_checkpoints() -> None:
    evidence = detect_krea2(HeaderSource({key: tuple(shape) for key, shape in golden_layout()}))
    assert evidence is not None
    assert evidence.config is KREA2_CONFIG
    assert evidence.key_prefix == ""
    assert len(evidence.matched_keys) == 430
    assert dict(evidence.fields) == {
        "features": 6144,
        "key_prefix": "",
        "kv_heads": 12,
        "sampling_shift": 1.15,
        "text_layers": 12,
        "text_width": 2560,
        "transformer_blocks": 28,
    }


def test_family_detector_scans_the_combined_checkpoint_prefix() -> None:
    evidence = detect_krea2(
        HeaderSource(
            {"model.diffusion_model." + key: tuple(shape) for key, shape in golden_layout()}
        )
    )
    assert evidence is not None
    assert evidence.config is KREA2_CONFIG
    assert evidence.key_prefix == "model.diffusion_model."
    assert all(key.startswith("model.diffusion_model.") for key in evidence.matched_keys)


def test_family_detector_returns_none_for_foreign_checkpoints() -> None:
    assert detect_krea2(HeaderSource({"weird.weight": (4, 4)})) is None
    classic = {key: shape for key, shape in flux_layout(FLUX_DEV_CONFIG).items()}
    assert detect_krea2(HeaderSource(classic)) is None


# ------------------------------------------------------ family catalog


class GenericHeaderSource(HeaderSource):
    def metadata(self) -> Mapping[str, str]:
        return {}


def test_krea2_registers_family_detection() -> None:
    assert KREA2.id == KREA2_CONFIG.family_id == "dinkster.krea2"
    assert KREA2.id in tuple(family.id for family in builtin_families())
    assert KREA2.id in builtin_family_registry().ids()


def test_registry_routes_the_golden_layout_unambiguously() -> None:
    registry = builtin_family_registry()
    for prefix in ("", "model.diffusion_model."):
        source = GenericHeaderSource({prefix + key: tuple(shape) for key, shape in golden_layout()})
        result = registry.detect(source)
        assert result.best is not None
        assert result.best.family_id == "dinkster.krea2"
        assert len(result.candidates) == 1


def test_krea2_pins_the_reference_sampling() -> None:
    assert KREA2.sampling.parameterization is Parameterization.FLOW
    assert KREA2.sampling.shift == 1.15
    assert KREA2.sampling.sigma_max == 1.0
    assert KREA2.sampling.sigma_min == math.exp(1.15) / (math.exp(1.15) + 9999)


def test_krea2_uses_the_wan21_latent() -> None:
    assert KREA2.latent is WAN21_LATENT
    assert isinstance(KREA2.latent, LatentDescriptor)
    assert KREA2.latent.channels == 16
    assert KREA2.latent.dimensions == 3
    assert KREA2.latent.spatial_downscale == 8
    assert KREA2.latent.temporal_downscale == 4
    assert KREA2.latent.temporal_causal is True
    assert KREA2.latent.rgb_factors is not None
    assert KREA2.latent.rgb_bias is not None
    assert KREA2.latent.taesd_decoder == "lighttaew2_1"
    assert KREA2.single_stream_latent() is KREA2.latent


def test_krea2_dtypes_and_memory_factor() -> None:
    assert KREA2.supported_dtypes == frozenset({FLOAT16, BFLOAT16, FLOAT32})
    assert KREA2.memory_factor == 2.2


def test_krea2_wiring_names_the_reference_text_encoder() -> None:
    assert KREA2.wiring.vae_prefix == "vae."
    assert KREA2.wiring.text_encoder_prefix == "text_encoders."
    assert KREA2.wiring.text_encoders == ("dinkster.qwen3vl_4b",)
