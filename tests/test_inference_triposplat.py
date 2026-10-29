"""TripoSplat torch-free family, component contracts, and detection tests.

The three exact layouts (DiT, octree gaussian decoder, DINOv3 ViT-H/16+)
are pinned against the published VAST-AI/TripoSplat artifact headers
(goldens/triposplat_headers.json), and every documented rejection branch
refuses with an error naming what it found.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    DINOV3_VITH,
    FLOAT16,
    FLOAT32,
    NATIVE_WIRED_FAMILY_IDS,
    TRIPOSPLAT,
    TRIPOSPLAT_CONFIG,
    TRIPOSPLAT_FAMILY,
    TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG,
    TRIPOSPLAT_SIGMAS,
    DINOv3DetectError,
    LatentDescriptor,
    MultiStreamLatentDescriptor,
    Parameterization,
    SamplingDescriptor,
    TensorGeometry,
    TripoSplatDetectError,
    WeightEntry,
    builtin_families,
    builtin_family_registry,
    detect_dinov3_vith,
    detect_triposplat,
    detect_triposplat_config,
    detect_triposplat_gaussian_decoder,
    dinov3_vith_layout,
    probe_native,
    triposplat_gaussian_decoder_layout,
    triposplat_layout,
)

GOLDENS = Path(__file__).parent / "goldens" / "triposplat_headers.json"


class HeaderSource:
    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = dict(shapes)

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], FLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        raise AssertionError("TripoSplat detection must not read metadata")


class GenericHeaderSource(HeaderSource):
    path = Path("triposplat.safetensors")

    def metadata(self) -> Mapping[str, str]:
        return {}


def golden_shapes(section: str) -> dict[str, tuple[int, ...]]:
    payload = json.loads(GOLDENS.read_text())
    return {key: tuple(shape) for key, shape in payload[section].items()}


def geometries_of(
    shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, FLOAT16) for key, shape in shapes.items()}


# ------------------------------------------------------------ layouts


def test_dit_layout_is_the_published_artifact_header() -> None:
    assert triposplat_layout() == golden_shapes("diffusion")


def test_gaussian_decoder_layout_is_the_published_artifact_header() -> None:
    assert triposplat_gaussian_decoder_layout() == golden_shapes("gaussian_decoder")


def test_dinov3_layout_is_the_published_artifact_header() -> None:
    assert dinov3_vith_layout() == golden_shapes("dinov3_vith")


def test_layout_spot_facts() -> None:
    dit = triposplat_layout()
    assert dit["input_layer.weight"] == (1024, 16)
    assert dit["cond_embedder.weight"] == (1024, 1280)
    assert dit["cond_embedder2.weight"] == (1024, 128)
    assert dit["cam_out_layer.weight"] == (5, 1024)
    assert dit["repo_layers.0.final_map.weight"] == (48, 128)
    # head_dim 64 splits 20/20/24 across the three rotary axes.
    assert dit["repo_layers.23.freqs_0"] == (10,)
    assert dit["repo_layers.23.freqs_2"] == (12,)
    assert "pos_emb" not in dit

    decoder = triposplat_gaussian_decoder_layout()
    assert decoder["octree.out_proj.weight"] == (8, 1024)
    assert decoder["gs.out_proj.weight"] == (480, 1024)
    assert decoder["gs.base_offset_scale"] == ()
    assert decoder["gs.points_offset_perturbation"] == (32, 3)

    dino = dinov3_vith_layout()
    assert dino["embeddings.register_tokens"] == (1, 4, 1280)
    assert dino["layer.0.mlp.gate_proj.weight"] == (5120, 1280)
    assert "layer.0.attention.k_proj.bias" not in dino


# ------------------------------------------------------------ configs


def test_config_pins_the_published_architecture() -> None:
    assert TRIPOSPLAT_CONFIG.family_id == "dinkster.triposplat"
    assert TRIPOSPLAT_CONFIG.q_token_length == 8192
    assert TRIPOSPLAT_CONFIG.latent_channels == 16
    assert TRIPOSPLAT_CONFIG.cam_channels == 5
    assert TRIPOSPLAT_CONFIG.model_channels == 1024
    assert TRIPOSPLAT_CONFIG.num_blocks == 24
    assert TRIPOSPLAT_CONFIG.num_refiner_blocks == 2
    assert TRIPOSPLAT_CONFIG.attention_heads == 16
    assert TRIPOSPLAT_CONFIG.attention_head_dim == 64
    assert TRIPOSPLAT_CONFIG.sampling_shift == 3.0
    assert TRIPOSPLAT_CONFIG.memory_factor == 0.6
    assert TRIPOSPLAT_CONFIG.inference_dtypes == (FLOAT16, BFLOAT16, FLOAT32)
    with pytest.raises(ValueError, match="exact published architecture"):
        type(TRIPOSPLAT_CONFIG)(num_blocks=25)


def test_gaussian_decoder_config_pins_the_published_architecture() -> None:
    config = TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG
    assert config.latent_channels == TRIPOSPLAT_CONFIG.latent_channels
    assert config.gaussians_per_point == 32
    # 32 gaussians x (xyz 3 + color 3 + scaling 3 + rotation 4 +
    # opacity 1) + 32 offset scales.
    assert config.feature_channels == 32 * (3 + 3 + 3 + 4 + 1) + 32
    assert config.max_voxel_level == 8
    with pytest.raises(ValueError, match="exact published architecture"):
        type(config)(gaussians_per_point=16)


def test_dinov3_config_pins_the_published_architecture() -> None:
    assert DINOV3_VITH.hidden_size == 1280
    assert DINOV3_VITH.num_hidden_layers == 32
    assert DINOV3_VITH.num_attention_heads == 20
    assert DINOV3_VITH.num_register_tokens == 4
    assert DINOV3_VITH.intermediate_size == 5120
    assert DINOV3_VITH.patch_size == 16
    assert DINOV3_VITH.image_size == 1024
    assert DINOV3_VITH.rope_theta == 100.0
    assert DINOV3_VITH.tokens == 1 + 4 + (1024 // 16) ** 2
    with pytest.raises(ValueError, match="divisible"):
        type(DINOV3_VITH)(num_attention_heads=21)


def test_sigmas_are_the_shift_3_discrete_flow_span() -> None:
    assert TRIPOSPLAT_SIGMAS.shift == 3.0
    assert TRIPOSPLAT_SIGMAS.sigma_max == 1.0
    assert TRIPOSPLAT_SIGMAS.sigma_min == pytest.approx(0.003 / 1.002)


# ------------------------------------------------------------ DiT detection


def test_detect_config_accepts_the_golden_header() -> None:
    geometries = geometries_of(golden_shapes("diffusion"))
    assert detect_triposplat_config(geometries) is TRIPOSPLAT_CONFIG


def test_detect_config_refuses_empty_headers() -> None:
    with pytest.raises(TripoSplatDetectError, match="empty state dict"):
        detect_triposplat_config({})


def test_detect_config_refuses_foreign_checkpoints_by_marker() -> None:
    geometries = geometries_of({"input_blocks.0.0.weight": (320, 4, 3, 3)})
    with pytest.raises(TripoSplatDetectError, match="not a TripoSplat DiT"):
        detect_triposplat_config(geometries)


def test_detect_config_refuses_missing_keys() -> None:
    geometries = geometries_of(golden_shapes("diffusion"))
    del geometries["blocks.23.mlp.mlp.2.weight"]
    with pytest.raises(TripoSplatDetectError, match="missing blocks.23.mlp.mlp.2.weight"):
        detect_triposplat_config(geometries)


def test_detect_config_refuses_unexpected_keys() -> None:
    geometries = geometries_of(golden_shapes("diffusion"))
    geometries["pos_emb"] = TensorGeometry((1, 8192, 1024), FLOAT16)
    with pytest.raises(TripoSplatDetectError, match="unexpected key pos_emb"):
        detect_triposplat_config(geometries)


def test_detect_config_refuses_resized_shapes() -> None:
    geometries = geometries_of(golden_shapes("diffusion"))
    geometries["input_layer.weight"] = TensorGeometry((1024, 32), FLOAT16)
    with pytest.raises(TripoSplatDetectError, match="input_layer.weight: expected shape"):
        detect_triposplat_config(geometries)


def test_detect_scans_bare_and_prefixed_checkpoints() -> None:
    shapes = golden_shapes("diffusion")
    bare = detect_triposplat(HeaderSource(shapes))
    assert bare is not None
    assert bare.key_prefix == ""
    assert bare.config is TRIPOSPLAT_CONFIG
    assert bare.fields["q_token_length"] == 8192
    assert bare.fields["cam_channels"] == 5

    combined = {f"model.diffusion_model.{key}": shape for key, shape in shapes.items()}
    combined["first_stage_model.octree.out_proj.weight"] = (8, 1024)
    combined["conditioner.embeddings.cls_token"] = (1, 1, 1280)
    prefixed = detect_triposplat(HeaderSource(combined))
    assert prefixed is not None
    assert prefixed.key_prefix == "model.diffusion_model."
    assert len(prefixed.matched_keys) == len(shapes)
    assert all(key.startswith("model.diffusion_model.") for key in prefixed.matched_keys)


def test_detect_returns_none_for_foreign_sources() -> None:
    assert detect_triposplat(HeaderSource({})) is None
    assert detect_triposplat(HeaderSource({"weird.weight": (4, 4)})) is None
    markers_only = {
        "cam_out_layer.weight": (5, 1024),
        "repo_layers.0.final_map.weight": (48, 128),
    }
    assert detect_triposplat(HeaderSource(markers_only)) is None


# ------------------------------------------------ component detection


def test_gaussian_decoder_detection_accepts_the_golden_header() -> None:
    geometries = geometries_of(golden_shapes("gaussian_decoder"))
    assert detect_triposplat_gaussian_decoder(geometries) is (TRIPOSPLAT_GAUSSIAN_DECODER_CONFIG)


def test_gaussian_decoder_detection_refuses_foreign_and_damaged_headers() -> None:
    with pytest.raises(TripoSplatDetectError, match="empty state dict"):
        detect_triposplat_gaussian_decoder({})
    kl = geometries_of({"decoder.conv_in.weight": (512, 16, 3, 3)})
    with pytest.raises(TripoSplatDetectError, match="not a TripoSplat gaussian decoder"):
        detect_triposplat_gaussian_decoder(kl)
    geometries = geometries_of(golden_shapes("gaussian_decoder"))
    del geometries["gs.blocks.15.norm2.bias"]
    with pytest.raises(TripoSplatDetectError, match="missing gs.blocks.15.norm2.bias"):
        detect_triposplat_gaussian_decoder(geometries)


def test_dinov3_detection_accepts_the_golden_header() -> None:
    geometries = geometries_of(golden_shapes("dinov3_vith"))
    assert detect_dinov3_vith(geometries) is DINOV3_VITH


def test_dinov3_detection_refuses_foreign_and_damaged_headers() -> None:
    with pytest.raises(DINOv3DetectError, match="empty state dict"):
        detect_dinov3_vith({})
    clip = geometries_of({"vision_model.embeddings.class_embedding": (1280,)})
    with pytest.raises(DINOv3DetectError, match="not a DINOv3 ViT-H/16"):
        detect_dinov3_vith(clip)
    geometries = geometries_of(golden_shapes("dinov3_vith"))
    geometries["layer.0.attention.k_proj.bias"] = TensorGeometry((1280,), FLOAT16)
    with pytest.raises(DINOv3DetectError, match="unexpected key layer.0.attention.k_proj.bias"):
        detect_dinov3_vith(geometries)
    geometries = geometries_of(golden_shapes("dinov3_vith"))
    del geometries["layer.31.mlp.down_proj.bias"]
    with pytest.raises(DINOv3DetectError, match="missing layer.31.mlp.down_proj.bias"):
        detect_dinov3_vith(geometries)
    geometries = geometries_of(golden_shapes("dinov3_vith"))
    geometries["embeddings.patch_embeddings.weight"] = TensorGeometry((1280, 3, 14, 14), FLOAT16)
    with pytest.raises(
        DINOv3DetectError, match="embeddings.patch_embeddings.weight: expected shape"
    ):
        detect_dinov3_vith(geometries)


# ------------------------------------------------------------ catalog


def test_triposplat_registers_in_the_family_catalog_without_wiring() -> None:
    ids = tuple(family.id for family in builtin_families())
    assert TRIPOSPLAT_CONFIG.family_id in ids
    registry = builtin_family_registry()
    assert TRIPOSPLAT_CONFIG.family_id in registry.ids()
    assert TRIPOSPLAT_CONFIG.family_id not in NATIVE_WIRED_FAMILY_IDS
    assert TRIPOSPLAT_FAMILY.config is TRIPOSPLAT_CONFIG
    assert TRIPOSPLAT_FAMILY.sigmas is TRIPOSPLAT_SIGMAS
    assert TRIPOSPLAT_FAMILY.component_roles == (
        "dit",
        "dinov3-vision-conditioner",
        "reference-latent-vae",
        "gaussian-decoder",
    )
    assert TRIPOSPLAT_FAMILY.detect(HeaderSource(golden_shapes("diffusion"))) is not None


def test_triposplat_family_has_token_stream_flow_facts() -> None:
    assert TRIPOSPLAT.supported_dtypes == frozenset({FLOAT16, BFLOAT16, FLOAT32})
    assert TRIPOSPLAT.memory_factor == 0.6
    assert isinstance(TRIPOSPLAT.sampling, SamplingDescriptor)
    assert TRIPOSPLAT.sampling.parameterization is Parameterization.FLOW
    assert TRIPOSPLAT.sampling.shift == 3.0
    assert TRIPOSPLAT.sampling.sigma_min == TRIPOSPLAT_SIGMAS.sigma_min
    assert TRIPOSPLAT.sampling.sigma_max == TRIPOSPLAT_SIGMAS.sigma_max
    assert isinstance(TRIPOSPLAT.latent, MultiStreamLatentDescriptor)
    assert TRIPOSPLAT.latent.streams == (
        ("latent", LatentDescriptor(channels=16, dimensions=1)),
        ("camera", LatentDescriptor(channels=5, dimensions=1)),
    )
    assert TRIPOSPLAT.wiring.text_encoders == ()


def test_registry_detection_resolves_the_golden_header_unambiguously() -> None:
    registry = builtin_family_registry()
    detection = registry.detect(GenericHeaderSource(golden_shapes("diffusion")))
    assert detection.best is not None
    assert detection.best.family_id == TRIPOSPLAT_CONFIG.family_id
    assert detection.ambiguous == ()

    capability = probe_native(diffusion=GenericHeaderSource(golden_shapes("diffusion")))
    assert capability.family_id == TRIPOSPLAT_CONFIG.family_id
    assert capability.native is False
    reason = "; ".join(capability.reasons)
    assert "checkpoint components do not match an executable architecture" in reason
    assert f"detected labels=({TRIPOSPLAT_CONFIG.family_id!r},)" in reason
    assert "has no native runtime wiring" not in reason
