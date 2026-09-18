"""Proving tests for the torch-free Flux2 layer.

Layout generation and detection run over TensorGeometry mappings only
(headers, never payloads). The three accepted geometries (dev,
Klein 9B, Klein 4B) are pinned against the EXECUTED reference's own
full-size state-dict listings (goldens/flux2_goldens.json "layouts",
produced by the reference Flux in its Flux2 configuration @ the
audited baseline on the meta device), and every documented rejection
branch refuses with a Flux2DetectError naming what it found.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLUX2_AXES_DIM,
    FLUX2_DEV,
    FLUX2_DEV_CONFIG,
    FLUX2_KLEIN_4B,
    FLUX2_KLEIN_4B_CONFIG,
    FLUX2_KLEIN_9B,
    FLUX2_KLEIN_9B_CONFIG,
    FLUX2_LATENT_CHANNELS,
    FLUX2_MLP_RATIO,
    FLUX2_PATCH_SIZE,
    FLUX2_THETA,
    FLUX2_TXT_IDS_DIMS,
    FLUX_DEV_CONFIG,
    KNOWN_FLUX2_CONFIGS,
    NATIVE_WIRED_FAMILY_IDS,
    Flux2DetectError,
    LatentDescriptor,
    Parameterization,
    TensorGeometry,
    WeightEntry,
    builtin_families,
    builtin_family_registry,
    detect_flux2,
    detect_flux2_config,
    flux2_empirical_mu,
    flux2_layout,
    flux_layout,
)

GOLDENS = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "flux2_goldens.json"
)

NAMED = {
    "flux2_dev": FLUX2_DEV_CONFIG,
    "flux2_klein_9b": FLUX2_KLEIN_9B_CONFIG,
    "flux2_klein_4b": FLUX2_KLEIN_4B_CONFIG,
}


class HeaderSource:
    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = dict(shapes)

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], BFLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        raise AssertionError("Flux2 detection must not read metadata")


def golden_layout(name: str) -> list[tuple[str, list[int]]]:
    payload = json.loads(GOLDENS.read_text())
    return [(key, list(shape)) for key, shape in payload["layouts"][name]]


def geometries_of(
    layout: list[tuple[str, list[int]]],
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in layout}


# ------------------------------------------------------------ config


def test_known_configs_are_the_reference_geometries() -> None:
    for config in NAMED.values():
        assert config.in_channels == 128
        assert config.out_channels == 128
        assert config.vec_in_dim is None
        assert config.axes_dim == (32, 32, 32, 32)
        assert config.theta == 2000
        assert config.patch_size == 1
        assert config.mlp_ratio == 3.0
        assert not config.qkv_bias
        assert config.txt_ids_dims == (3,)
        assert config.global_modulation
        assert config.mlp_silu_act
        assert not config.ops_bias
        assert config.head_dim == 128
        assert config.num_heads == config.hidden_size // 128
        assert config.mlp_hidden_dim == config.hidden_size * 3
    assert FLUX2_DEV_CONFIG.context_in_dim == 15360
    assert FLUX2_DEV_CONFIG.hidden_size == 6144
    assert FLUX2_DEV_CONFIG.depth == 8
    assert FLUX2_DEV_CONFIG.depth_single_blocks == 48
    assert FLUX2_DEV_CONFIG.guidance_embed
    assert FLUX2_KLEIN_9B_CONFIG.context_in_dim == 12288
    assert FLUX2_KLEIN_9B_CONFIG.hidden_size == 4096
    assert FLUX2_KLEIN_9B_CONFIG.depth == 8
    assert FLUX2_KLEIN_9B_CONFIG.depth_single_blocks == 24
    assert not FLUX2_KLEIN_9B_CONFIG.guidance_embed
    assert FLUX2_KLEIN_4B_CONFIG.context_in_dim == 7680
    assert FLUX2_KLEIN_4B_CONFIG.hidden_size == 3072
    assert FLUX2_KLEIN_4B_CONFIG.depth == 5
    assert FLUX2_KLEIN_4B_CONFIG.depth_single_blocks == 20
    assert not FLUX2_KLEIN_4B_CONFIG.guidance_embed
    assert KNOWN_FLUX2_CONFIGS == NAMED


def test_pinned_constants_match_the_reference_detection() -> None:
    """The facts comfy/model_detection.py pins rather than derives."""
    assert FLUX2_AXES_DIM == (32, 32, 32, 32)
    assert FLUX2_THETA == 2000
    assert FLUX2_PATCH_SIZE == 1
    assert FLUX2_MLP_RATIO == 3.0
    assert FLUX2_LATENT_CHANNELS == 128
    assert FLUX2_TXT_IDS_DIMS == (3,)


# ------------------------------------------------------------ layout


@pytest.mark.parametrize("name", sorted(NAMED))
def test_layout_matches_the_executed_reference_listing(name: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in flux2_layout(NAMED[name]).items())
    assert predicted == sorted(golden_layout(name))


def test_layout_refuses_classic_flux_configs() -> None:
    with pytest.raises(ValueError, match="requires a Flux2 config"):
        flux2_layout(FLUX_DEV_CONFIG)


# --------------------------------------------------------- detection


@pytest.mark.parametrize("name", sorted(NAMED))
def test_detects_each_published_geometry(name: str) -> None:
    assert detect_flux2_config(geometries_of(golden_layout(name))) is NAMED[name]


def test_detects_bare_bfl_scale_spelling() -> None:
    geometries = {
        (
            key[: -len(".weight")] + ".scale"
            if key.endswith(("query_norm.weight", "key_norm.weight"))
            else key
        ): geometry
        for key, geometry in geometries_of(golden_layout("flux2_klein_4b")).items()
    }
    assert any(key.endswith(".scale") for key in geometries)
    assert detect_flux2_config(geometries) is FLUX2_KLEIN_4B_CONFIG


def test_rejects_empty_headers() -> None:
    with pytest.raises(Flux2DetectError, match="empty state dict"):
        detect_flux2_config({})


def test_rejects_classic_flux_headers() -> None:
    geometries = {
        key: TensorGeometry(shape, FLOAT32) for key, shape in flux_layout(FLUX_DEV_CONFIG).items()
    }
    with pytest.raises(Flux2DetectError, match="not a Flux2 DiT"):
        detect_flux2_config(geometries)


def test_rejects_quantized_repackages_with_extra_scale_tensors() -> None:
    """The Comfy-Org fp8-mixed dev export adds input_scale /
    weight_scale tensors; an exact key-set match refuses them instead
    of misreading the checkpoint."""
    geometries = geometries_of(golden_layout("flux2_dev"))
    geometries["double_blocks.0.img_attn.qkv.input_scale"] = TensorGeometry((), FLOAT32)
    with pytest.raises(Flux2DetectError, match="unexpected key"):
        detect_flux2_config(geometries)


def test_rejects_truncated_headers() -> None:
    geometries = geometries_of(golden_layout("flux2_klein_9b"))
    del geometries["single_blocks.23.linear2.weight"]
    with pytest.raises(Flux2DetectError, match="missing single_blocks.23.linear2.weight"):
        detect_flux2_config(geometries)


def test_rejects_mismatched_shapes() -> None:
    geometries = geometries_of(golden_layout("flux2_klein_4b"))
    geometries["txt_in.weight"] = TensorGeometry((3072, 7681), FLOAT32)
    with pytest.raises(Flux2DetectError, match="txt_in.weight: expected shape"):
        detect_flux2_config(geometries)


def test_rejects_unpublished_global_modulation_geometries() -> None:
    geometries = geometries_of(golden_layout("flux2_klein_4b"))
    resized = {
        key: TensorGeometry(
            (geometry.shape[0], 64) if key == "img_in.weight" else geometry.shape, FLOAT32
        )
        for key, geometry in geometries.items()
    }
    with pytest.raises(Flux2DetectError, match="no published Flux2 geometry"):
        detect_flux2_config(resized)


# ------------------------------------------------ family detector seam


def test_family_detector_returns_evidence_for_bare_checkpoints() -> None:
    evidence = detect_flux2(
        HeaderSource({key: tuple(shape) for key, shape in golden_layout("flux2_dev")})
    )
    assert evidence is not None
    assert evidence.config is FLUX2_DEV_CONFIG
    assert evidence.key_prefix == ""
    assert len(evidence.matched_keys) == 299
    assert evidence.fields == {
        "context_in_dim": 15360,
        "depth": 8,
        "depth_single_blocks": 48,
        "guidance_embed": True,
        "hidden_size": 6144,
        "key_prefix": "",
    }


def test_family_detector_scans_the_combined_checkpoint_prefix() -> None:
    evidence = detect_flux2(
        HeaderSource(
            {
                "model.diffusion_model." + key: tuple(shape)
                for key, shape in golden_layout("flux2_klein_9b")
            }
        )
    )
    assert evidence is not None
    assert evidence.config is FLUX2_KLEIN_9B_CONFIG
    assert evidence.key_prefix == "model.diffusion_model."
    assert all(key.startswith("model.diffusion_model.") for key in evidence.matched_keys)


def test_family_detector_returns_none_for_foreign_checkpoints() -> None:
    assert detect_flux2(HeaderSource({"weird.weight": (4, 4)})) is None
    classic = {key: shape for key, shape in flux_layout(FLUX_DEV_CONFIG).items()}
    assert detect_flux2(HeaderSource(classic)) is None


# ------------------------------------------------------ family catalog


class GenericHeaderSource(HeaderSource):
    def metadata(self) -> Mapping[str, str]:
        return {}


FAMILIES = {
    "flux2_dev": FLUX2_DEV,
    "flux2_klein_9b": FLUX2_KLEIN_9B,
    "flux2_klein_4b": FLUX2_KLEIN_4B,
}


def test_flux2_families_register_with_native_runtime_wiring() -> None:
    ids = tuple(family.id for family in builtin_families())
    registry_ids = builtin_family_registry().ids()
    for family in FAMILIES.values():
        assert family.id in ids
        assert family.id in registry_ids
        assert family.id in NATIVE_WIRED_FAMILY_IDS


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_registry_routes_each_golden_layout_unambiguously(name: str) -> None:
    registry = builtin_family_registry()
    for prefix in ("", "model.diffusion_model."):
        source = GenericHeaderSource(
            {prefix + key: tuple(shape) for key, shape in golden_layout(name)}
        )
        result = registry.detect(source)
        assert result.best is not None
        assert result.best.family_id == FAMILIES[name].id
        assert len(result.candidates) == 1


def test_registry_keeps_classic_flux_and_flux2_disjoint() -> None:
    registry = builtin_family_registry()

    classic = GenericHeaderSource(dict(flux_layout(FLUX_DEV_CONFIG)))
    result = registry.detect(classic)
    assert result.best is not None
    assert result.best.family_id == "dinkster.flux_dev"
    assert all(
        not candidate.family_id.startswith("dinkster.flux2_") for candidate in result.candidates
    )

    flux2 = GenericHeaderSource({key: tuple(shape) for key, shape in golden_layout("flux2_dev")})
    result = registry.detect(flux2)
    assert all(
        candidate.family_id not in ("dinkster.flux_dev", "dinkster.flux_schnell")
        for candidate in result.candidates
    )


def test_flux2_families_pin_the_reference_sampling_and_latent() -> None:
    for family in FAMILIES.values():
        assert family.sampling.parameterization is Parameterization.FLOW
        assert family.sampling.shift == 2.02
        assert family.sampling.sigma_max == 1.0
        assert family.sampling.sigma_min == math.exp(2.02) / (math.exp(2.02) + 9999)
        assert isinstance(family.latent, LatentDescriptor)
        assert family.latent.channels == 128
        assert family.latent.dimensions == 2
        assert family.latent.spatial_downscale == 16
        assert family.latent.scale_factor == 1.0
        assert family.latent.shift_factor == 0.0
        assert family.latent.rgb_factors is None
        assert family.latent.rgb_bias is None
        assert family.latent.taesd_decoder is None
        assert family.single_stream_latent() is family.latent
        assert family.supported_dtypes == frozenset({FLOAT16, BFLOAT16, FLOAT32})


def test_flux2_memory_factors_scale_with_hidden_size() -> None:
    assert FLUX2_DEV.memory_factor == 3.1 * 4.0 * (6144 / 2604)
    assert FLUX2_KLEIN_9B.memory_factor == 3.1 * 4.0 * (4096 / 2604)
    assert FLUX2_KLEIN_4B.memory_factor == 3.1 * 4.0 * (3072 / 2604)


def test_flux2_wiring_names_the_reference_text_encoders() -> None:
    for family in FAMILIES.values():
        assert family.wiring.vae_prefix == "vae."
        assert family.wiring.text_encoder_prefix == "text_encoders."
    assert FLUX2_DEV.wiring.text_encoders == ("dinkster.mistral3_24b",)
    assert FLUX2_KLEIN_9B.wiring.text_encoders == ("dinkster.qwen3_8b",)
    assert FLUX2_KLEIN_4B.wiring.text_encoders == ("dinkster.qwen3_4b",)


@pytest.mark.parametrize(
    ("image_seq_len", "num_steps", "expected"),
    (
        (1024, 28, 1.8591765771805644),
        (4096, 20, 2.1980220725551165),
        (4300, 10, 2.27407142532),
        (4300, 200, 1.18452766),
        (4301, 7, 1.18469693),
        (9216, 50, 2.01665898),
        (16384, 4, 3.22998634),
    ),
)
def test_empirical_mu_is_bit_exact_to_the_executed_reference(
    image_seq_len: int, num_steps: int, expected: float
) -> None:
    """Literals executed from compute_empirical_mu (comfy_extras/
    nodes_flux.py @ 947c2749); pure float math, so equality is exact.
    Above 4300 tokens the step count does not move the fit line."""
    assert flux2_empirical_mu(image_seq_len, num_steps) == expected
    if image_seq_len > 4300:
        assert flux2_empirical_mu(image_seq_len, 200) == expected


def test_empirical_mu_refuses_non_positive_or_non_int_arguments() -> None:
    with pytest.raises(ValueError, match="image token count"):
        flux2_empirical_mu(0, 20)
    with pytest.raises(ValueError, match="image token count"):
        flux2_empirical_mu(4096.0, 20)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step count"):
        flux2_empirical_mu(4096, 0)
    with pytest.raises(ValueError, match="step count"):
        flux2_empirical_mu(4096, True)  # type: ignore[arg-type]
