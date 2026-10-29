"""Proving tests for the torch-free classic-Flux layer.

Layout generation and detection run over TensorGeometry mappings only
(headers, never payloads). The two accepted geometries (dev and
schnell) are pinned against the EXECUTED reference's own full-size
state-dict listings (goldens/flux_goldens.json "layouts", produced by
the reference Flux @ the audited baseline on the meta device), and
every documented rejection branch - the variant legs the reference's
flux detection routes elsewhere - refuses with a FluxDetectError
naming what it found.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLUX_AXES_DIM,
    FLUX_DEV_CONFIG,
    FLUX_LATENT_CHANNELS,
    FLUX_MLP_RATIO,
    FLUX_PATCH_SIZE,
    FLUX_SCHNELL_CONFIG,
    FLUX_THETA,
    KNOWN_FLUX_CONFIGS,
    FluxConfig,
    FluxDetectError,
    TensorGeometry,
    detect_flux_config,
    flux_layout,
    normalize_flux_keys,
)

GOLDENS = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "flux_goldens.json"
)
GATED_GOLDENS = GOLDENS.with_name("flux_gated_goldens.json")

NAMED = {
    "flux_dev": FLUX_DEV_CONFIG,
    "flux_schnell": FLUX_SCHNELL_CONFIG,
}

OVIS_VECTOR_FREE_CONFIG = replace(
    FLUX_SCHNELL_CONFIG,
    vec_in_dim=None,
    context_in_dim=2048,
    depth=6,
    depth_single_blocks=27,
    txt_norm=True,
    yak_mlp=True,
    txt_ids_dims=(1, 2),
)


def golden_layout(name: str) -> list[tuple[str, list[int]]]:
    payload = json.loads(GOLDENS.read_text())
    return [(key, list(shape)) for key, shape in payload["layouts"][name]]


def geometries_of(
    layout: list[tuple[str, list[int]]],
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in layout}


# ------------------------------------------------------------ config


def test_known_configs_are_the_reference_geometries() -> None:
    for config in (FLUX_DEV_CONFIG, FLUX_SCHNELL_CONFIG):
        assert config.in_channels == 16
        assert config.out_channels == 16
        assert config.vec_in_dim == 768
        assert config.context_in_dim == 4096
        assert config.hidden_size == 3072
        assert config.depth == 19
        assert config.depth_single_blocks == 38
        assert config.num_heads == 24
        assert config.axes_dim == (16, 56, 56)
        assert config.theta == 10000
        assert config.patch_size == 2
        assert config.mlp_ratio == 4.0
        assert config.qkv_bias
        assert config.head_dim == 128
        assert config.mlp_hidden_dim == 12288
    assert FLUX_DEV_CONFIG.guidance_embed
    assert not FLUX_SCHNELL_CONFIG.guidance_embed
    assert KNOWN_FLUX_CONFIGS == NAMED
    assert repr(FLUX_DEV_CONFIG) == (
        "FluxConfig(in_channels=16, out_channels=16, vec_in_dim=768, "
        "context_in_dim=4096, hidden_size=3072, depth=19, "
        "depth_single_blocks=38, num_heads=24, axes_dim=(16, 56, 56), "
        "theta=10000, patch_size=2, mlp_ratio=4.0, qkv_bias=True, "
        "guidance_embed=True)"
    )


def test_pinned_constants_match_the_reference_detection() -> None:
    """The facts comfy/model_detection.py pins rather than derives."""
    assert FLUX_AXES_DIM == (16, 56, 56)
    assert FLUX_THETA == 10000
    assert FLUX_PATCH_SIZE == 2
    assert FLUX_MLP_RATIO == 4.0
    assert FLUX_LATENT_CHANNELS == 16


def test_config_rejects_non_positive_fields() -> None:
    with pytest.raises(ValueError, match="depth must be >= 1"):
        replace(FLUX_DEV_CONFIG, depth=0)


def test_config_allows_absent_vector_only_as_an_explicit_fact() -> None:
    assert OVIS_VECTOR_FREE_CONFIG.vec_in_dim is None
    with pytest.raises(ValueError, match="vec_in_dim must be"):
        replace(FLUX_DEV_CONFIG, vec_in_dim=0)


def test_config_rejects_invalid_text_position_axes() -> None:
    with pytest.raises(ValueError, match="txt_ids_dims"):
        replace(FLUX_DEV_CONFIG, txt_ids_dims=(1, 1))
    with pytest.raises(ValueError, match="txt_ids_dims"):
        replace(FLUX_DEV_CONFIG, txt_ids_dims=(3,))


def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        replace(FLUX_DEV_CONFIG, num_heads=23)


def test_config_rejects_odd_axes_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        replace(FLUX_DEV_CONFIG, axes_dim=(16, 56, 55), num_heads=24)


def test_config_rejects_axes_dim_head_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="expected the per-head dim"):
        replace(FLUX_DEV_CONFIG, axes_dim=(16, 56, 58))


def test_config_rejects_empty_mlp() -> None:
    with pytest.raises(ValueError, match="empty MLP"):
        replace(
            FLUX_DEV_CONFIG,
            mlp_ratio=1e-9,
        )


# ------------------------------------------------------------ layout


@pytest.mark.parametrize("name", sorted(NAMED))
def test_layout_matches_executed_reference(name: str) -> None:
    """Same key -> shape mapping as the executed reference's listing
    (the golden is stored sorted; generation order is free)."""
    layout = flux_layout(NAMED[name])
    golden = golden_layout(name)
    assert len(golden) == len(layout)  # no duplicates on either side
    assert {key: list(shape) for key, shape in layout.items()} == dict(golden)


def test_dev_and_schnell_differ_only_in_guidance_keys() -> None:
    dev = flux_layout(FLUX_DEV_CONFIG)
    schnell = flux_layout(FLUX_SCHNELL_CONFIG)
    extra = set(dev) - set(schnell)
    assert extra == {
        "guidance_in.in_layer.weight",
        "guidance_in.in_layer.bias",
        "guidance_in.out_layer.weight",
        "guidance_in.out_layer.bias",
    }
    assert {key: dev[key] for key in schnell} == schnell


def test_vector_free_ovis_layout_is_the_exact_397_key_artifact_geometry() -> None:
    layout = flux_layout(OVIS_VECTOR_FREE_CONFIG)
    assert len(layout) == 397
    assert layout["txt_in.weight"] == (3072, 2048)
    assert layout["txt_norm.weight"] == (2048,)
    assert layout["double_blocks.0.img_mlp.gate_proj.weight"] == (
        12288,
        3072,
    )
    assert not any(key.startswith("vector_in.") for key in layout)
    assert not any(key.startswith("guidance_in.") for key in layout)


# ------------------------------------------------- key normalization


def test_normalize_renames_norm_scale_to_weight() -> None:
    mapping = {
        "double_blocks.0.img_attn.norm.query_norm.scale": 1,
        "single_blocks.3.norm.key_norm.scale": 2,
        "img_in.weight": 3,
        "vector_in.in_layer.bias": 4,
    }
    assert normalize_flux_keys(mapping) == {
        "double_blocks.0.img_attn.norm.query_norm.weight": 1,
        "single_blocks.3.norm.key_norm.weight": 2,
        "img_in.weight": 3,
        "vector_in.in_layer.bias": 4,
    }


@pytest.mark.parametrize("scale_first", [False, True])
def test_normalize_rejects_alias_collisions(scale_first: bool) -> None:
    entries = [
        ("txt_norm.weight", 1),
        ("txt_norm.scale", 2),
    ]
    if scale_first:
        entries.reverse()
    with pytest.raises(ValueError, match="both normalize to 'txt_norm.weight'"):
        normalize_flux_keys(dict(entries))


@pytest.mark.parametrize("scale_first", [False, True])
def test_detector_rejects_txt_norm_alias_collisions(scale_first: bool) -> None:
    geometries = geometries_of(golden_layout("flux_dev"))
    entries = [
        ("txt_norm.weight", TensorGeometry((4096,), FLOAT32)),
        ("txt_norm.scale", TensorGeometry((4096,), FLOAT32)),
    ]
    if scale_first:
        entries.reverse()
    geometries.update(entries)
    with pytest.raises(FluxDetectError, match="both normalize"):
        detect_flux_config(geometries)


# --------------------------------------------------------- detection


@pytest.mark.parametrize("name", sorted(NAMED))
def test_detects_reference_geometry(name: str) -> None:
    assert detect_flux_config(geometries_of(golden_layout(name))) == (NAMED[name])


def test_detects_fp16_checkpoints() -> None:
    geometries = {
        key: TensorGeometry(tuple(shape), FLOAT16) for key, shape in golden_layout("flux_dev")
    }
    assert detect_flux_config(geometries) == FLUX_DEV_CONFIG


def test_detects_bare_bfl_scale_spelling() -> None:
    """Bare BFL exports spell RMSNorm parameters *_norm.scale; the
    detector normalizes internally exactly like the reference's
    process_unet_state_dict."""
    geometries = {
        (
            key[: -len(".weight")] + ".scale"
            if key.endswith(("query_norm.weight", "key_norm.weight"))
            else key
        ): TensorGeometry(tuple(shape), FLOAT32)
        for key, shape in golden_layout("flux_schnell")
    }
    assert detect_flux_config(geometries) == FLUX_SCHNELL_CONFIG


@pytest.mark.parametrize("spelling", ["weight", "scale"])
def test_detects_txt_norm_variant(spelling: str) -> None:
    config = replace(FLUX_DEV_CONFIG, txt_norm=True)
    geometries = geometries_of([(key, list(shape)) for key, shape in flux_layout(config).items()])
    if spelling == "scale":
        geometries["txt_norm.scale"] = geometries.pop("txt_norm.weight")
    assert detect_flux_config(geometries) == config
    assert detect_flux_config(geometries).txt_norm


def test_txt_norm_layout_adds_only_required_weight() -> None:
    classic = flux_layout(FLUX_DEV_CONFIG)
    normalized = flux_layout(replace(FLUX_DEV_CONFIG, txt_norm=True))
    assert set(normalized) - set(classic) == {"txt_norm.weight"}
    assert normalized["txt_norm.weight"] == (4096,)
    assert {key: normalized[key] for key in classic} == classic


def test_rejects_empty_header() -> None:
    with pytest.raises(FluxDetectError, match="empty"):
        detect_flux_config({})


def test_rejects_non_flux_families() -> None:
    geometries = {
        "joint_blocks.0.context_block.attn.qkv.weight": TensorGeometry((1536, 1536), FLOAT32)
    }
    with pytest.raises(FluxDetectError, match="not a Flux-lineage"):
        detect_flux_config(geometries)


def test_rejects_flux2_global_modulation() -> None:
    geometries = geometries_of(golden_layout("flux_dev"))
    geometries["double_stream_modulation_img.lin.weight"] = TensorGeometry((18432, 3072), FLOAT32)
    with pytest.raises(FluxDetectError, match="Flux2 detection lives in dinkster_inference.flux2"):
        detect_flux_config(geometries)


@pytest.mark.parametrize("spelling", ["weight", "scale"])
def test_rejects_chroma_distilled_guidance(spelling: str) -> None:
    """Chroma's marker is scanned as a prefix so both RMSNorm
    spellings of its norms.N keys hit the Chroma rejection, never the
    vector_in one."""
    geometries = {
        key: value
        for key, value in geometries_of(golden_layout("flux_dev")).items()
        if not key.startswith(("vector_in.", "guidance_in."))
    }
    geometries[f"distilled_guidance_layer.norms.0.{spelling}"] = TensorGeometry((5120,), FLOAT32)
    with pytest.raises(FluxDetectError, match="Chroma"):
        detect_flux_config(geometries)


@pytest.mark.parametrize("reverse", [False, True])
def test_refuses_unproven_vector_present_gated_txt_norm_geometry(
    reverse: bool,
) -> None:
    config = replace(FLUX_DEV_CONFIG, txt_norm=True, yak_mlp=True)
    entries = list(flux_layout(config).items())
    if reverse:
        entries.reverse()
    geometries = geometries_of([(key, list(shape)) for key, shape in entries])
    with pytest.raises(FluxDetectError, match="unproven text-position semantics"):
        detect_flux_config(geometries)


@pytest.mark.parametrize("reverse", [False, True])
def test_detects_exact_vector_free_ovis_geometry(reverse: bool) -> None:
    entries = list(flux_layout(OVIS_VECTOR_FREE_CONFIG).items())
    if reverse:
        entries.reverse()
    geometries = {
        (
            key[: -len(".weight")] + ".scale" if key.endswith("_norm.weight") else key
        ): TensorGeometry(shape, BFLOAT16)
        for key, shape in entries
    }
    assert len(geometries) == 397
    assert detect_flux_config(geometries) == OVIS_VECTOR_FREE_CONFIG


@pytest.mark.parametrize(
    "config",
    [
        replace(OVIS_VECTOR_FREE_CONFIG, txt_norm=False),
        replace(OVIS_VECTOR_FREE_CONFIG, yak_mlp=False),
        replace(OVIS_VECTOR_FREE_CONFIG, guidance_embed=True),
        replace(OVIS_VECTOR_FREE_CONFIG, depth=7),
        replace(OVIS_VECTOR_FREE_CONFIG, depth_single_blocks=26),
        replace(OVIS_VECTOR_FREE_CONFIG, hidden_size=2944, num_heads=23),
    ],
)
def test_rejects_non_approved_vector_free_flux_geometry(
    config: FluxConfig,
) -> None:
    geometries = geometries_of([(key, list(shape)) for key, shape in flux_layout(config).items()])
    with pytest.raises(FluxDetectError, match="only the approved Ovis"):
        detect_flux_config(geometries)


def test_gated_layout_matches_executed_reference() -> None:
    payload = json.loads(GATED_GOLDENS.read_text())
    spec = dict(payload["config"])
    spec["axes_dim"] = tuple(spec["axes_dim"])
    config = FluxConfig(**spec)
    expected = {key: tuple(shape) for key, shape in payload["layout"]}
    assert flux_layout(config) == expected


def test_gated_layout_replaces_double_mlps_and_widens_single_mlps() -> None:
    classic = flux_layout(FLUX_DEV_CONFIG)
    gated = flux_layout(replace(FLUX_DEV_CONFIG, yak_mlp=True))
    assert "double_blocks.0.img_mlp.0.weight" in classic
    assert "double_blocks.0.img_mlp.0.weight" not in gated
    assert gated["double_blocks.0.img_mlp.gate_proj.weight"] == (12288, 3072)
    assert gated["double_blocks.0.img_mlp.up_proj.weight"] == (12288, 3072)
    assert gated["double_blocks.0.img_mlp.down_proj.weight"] == (3072, 12288)
    assert classic["single_blocks.0.linear1.weight"] == (21504, 3072)
    assert gated["single_blocks.0.linear1.weight"] == (33792, 3072)
    assert gated["single_blocks.0.linear2.weight"] == (classic["single_blocks.0.linear2.weight"])


@pytest.mark.parametrize("gated_block", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_rejects_gate_keys_present_in_only_some_blocks(reverse: bool, gated_block: int) -> None:
    classic = flux_layout(FLUX_DEV_CONFIG)
    gated = flux_layout(replace(FLUX_DEV_CONFIG, yak_mlp=True))
    prefix = f"double_blocks.{gated_block}."
    for key in list(classic):
        if key.startswith(prefix) and "_mlp." in key:
            del classic[key]
    classic.update(
        (key, shape) for key, shape in gated.items() if key.startswith(prefix) and "_mlp." in key
    )
    entries = list(classic.items())
    if reverse:
        entries.reverse()
    with pytest.raises(FluxDetectError, match="geometry does not match"):
        detect_flux_config(geometries_of([(key, list(shape)) for key, shape in entries]))


@pytest.mark.parametrize("reverse", [False, True])
def test_rejects_incomplete_gate_projection_set(reverse: bool) -> None:
    gated = flux_layout(replace(FLUX_DEV_CONFIG, yak_mlp=True))
    del gated["double_blocks.0.img_mlp.up_proj.weight"]
    del gated["double_blocks.0.img_mlp.down_proj.bias"]
    entries = list(gated.items())
    if reverse:
        entries.reverse()
    with pytest.raises(FluxDetectError, match="img_mlp.up_proj.weight"):
        detect_flux_config(geometries_of([(key, list(shape)) for key, shape in entries]))


def test_rejects_malformed_txt_norm_shape() -> None:
    geometries = geometries_of(golden_layout("flux_dev"))
    geometries["txt_norm.weight"] = TensorGeometry((3072,), FLOAT32)
    with pytest.raises(FluxDetectError, match="txt_norm.weight"):
        detect_flux_config(geometries)


def test_rejects_vector_free_lineages() -> None:
    """LongCat-Image: context width 3584 (Qwen2.5-VL) AND no
    vector_in - the reference's exact discriminator."""
    layout = flux_layout(replace(FLUX_DEV_CONFIG, context_in_dim=3584))
    geometries = {
        key: TensorGeometry(tuple(shape), FLOAT32)
        for key, shape in layout.items()
        if not key.startswith("vector_in.")
    }
    with pytest.raises(FluxDetectError, match="LongCat"):
        detect_flux_config(geometries)


@pytest.mark.parametrize("reverse", [False, True])
def test_truncated_classic_flux_refuses_in_both_header_orders(
    reverse: bool,
) -> None:
    """A classic-width (4096) checkpoint missing all vector_in keys
    is truncated, not a vector-free lineage: the reference only
    discriminates LongCat at context width 3584."""
    entries = [
        (key, value)
        for key, value in geometries_of(golden_layout("flux_dev")).items()
        if not key.startswith("vector_in.")
    ]
    if reverse:
        entries.reverse()
    geometries = dict(entries)
    with pytest.raises(FluxDetectError, match="truncated classic Flux"):
        detect_flux_config(geometries)


def test_missing_vector_in_linear_reports_key() -> None:
    """Only vector_in.in_layer.weight gone (other vector_in keys
    intact): a missing-key refusal, never a lineage claim."""
    geometries = geometries_of(golden_layout("flux_dev"))
    del geometries["vector_in.in_layer.weight"]
    with pytest.raises(FluxDetectError, match="missing vector_in.in_layer.weight"):
        detect_flux_config(geometries)


def test_rejects_widened_inpainting_img_in() -> None:
    geometries = geometries_of(golden_layout("flux_dev"))
    geometries["img_in.weight"] = TensorGeometry((3072, 384), FLOAT32)
    with pytest.raises(FluxDetectError, match="FluxInpaint"):
        detect_flux_config(geometries)


def test_rejects_malformed_img_in_rank() -> None:
    """Malformed geometry refuses with FluxDetectError, never an
    incidental IndexError."""
    geometries = geometries_of(golden_layout("flux_dev"))
    geometries["img_in.weight"] = TensorGeometry((3072,), FLOAT32)
    with pytest.raises(FluxDetectError, match="img_in.weight has rank 1"):
        detect_flux_config(geometries)


def test_rejects_incoherent_txt_in() -> None:
    geometries = geometries_of(golden_layout("flux_dev"))
    geometries["txt_in.weight"] = TensorGeometry((1536, 4096), FLOAT32)
    with pytest.raises(FluxDetectError, match="not a coherent"):
        detect_flux_config(geometries)


def test_rejects_underivable_heads() -> None:
    """hidden_size not a multiple of sum(axes_dim): heads cannot be
    derived the way the reference derives them."""
    layout = flux_layout(replace(FLUX_DEV_CONFIG, num_heads=24))
    geometries = {
        key: TensorGeometry(tuple(100 if size == 3072 else size for size in shape), FLOAT32)
        for key, shape in layout.items()
    }
    with pytest.raises(FluxDetectError, match="not a multiple"):
        detect_flux_config(geometries)


def test_rejects_missing_key_against_layout() -> None:
    geometries = geometries_of(golden_layout("flux_dev"))
    del geometries["single_blocks.37.linear2.bias"]
    with pytest.raises(FluxDetectError, match="missing single_blocks.37"):
        detect_flux_config(geometries)


def test_rejects_unexpected_key_against_layout() -> None:
    geometries = geometries_of(golden_layout("flux_dev"))
    geometries["double_blocks.0.stray.weight"] = TensorGeometry((3072,), FLOAT32)
    with pytest.raises(FluxDetectError, match="unexpected key"):
        detect_flux_config(geometries)


def test_rejects_drifted_shape_against_layout() -> None:
    geometries = geometries_of(golden_layout("flux_dev"))
    geometries["time_in.in_layer.weight"] = TensorGeometry((3072, 512), FLOAT32)
    with pytest.raises(FluxDetectError, match="time_in.in_layer.weight"):
        detect_flux_config(geometries)


def test_rejects_truncated_double_blocks() -> None:
    """A pruned checkpoint (fewer double blocks than the contiguous
    count implies elsewhere) fails the full-layout comparison."""
    geometries = {
        key: value
        for key, value in geometries_of(golden_layout("flux_dev")).items()
        if not key.startswith("double_blocks.18.img_attn.")
    }
    with pytest.raises(FluxDetectError, match="missing double_blocks.18"):
        detect_flux_config(geometries)
