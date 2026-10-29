"""Proving tests for the torch-free SD/SDXL AutoencoderKL layer.

Detection runs over TensorGeometry mappings only (headers, never
payloads): the two accepted layouts are read from the executed
reference's own state-dict listing (goldens/kl_goldens.json, key
names and shapes produced by comfy.ldm.models.autoencoder.AutoencoderKL
@ the audited baseline), and every documented rejection branch
refuses with a KLDetectError naming what it found.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT32,
    KL_PREFIX_RENAMES,
    KL_STANDARD_CH_MULT,
    KL_X4_CH_MULT,
    KLConfig,
    KLDetectError,
    KLMemoryEstimator,
    TensorGeometry,
    detect_kl_config,
    kl_descriptor,
    normalize_kl_keys,
)
from dinkster_inference.autoencoder_kl import DIFFUSERS_KL_MARKER

GOLDENS = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "kl_goldens.json"
)


def golden_geometries(case: str) -> dict[str, TensorGeometry]:
    payload = json.loads(GOLDENS.read_text())
    return {
        key: TensorGeometry(tuple(shape), FLOAT32)
        for key, shape in payload["cases"][case]["state_dict"]
    }


def kl_geometries(
    *,
    ch: int = 128,
    decoder_ch: int | None = None,
    ch_mult: tuple[int, ...] = KL_STANDARD_CH_MULT,
    num_res_blocks: int = 2,
    in_channels: int = 3,
    out_channels: int = 3,
    z_channels: int = 4,
    embed_dim: int = 4,
    batch_norm_latent: bool = False,
) -> dict[str, TensorGeometry]:
    """A synthetic KL header mirroring the constructed model's full
    conv-weight topology (detection now validates all of it)."""

    def g(*shape: int) -> TensorGeometry:
        return TensorGeometry(shape, FLOAT32)

    dec_ch = ch if decoder_ch is None else decoder_ch
    levels = len(ch_mult)
    sd = {
        "encoder.conv_in.weight": g(ch, in_channels, 3, 3),
        "encoder.conv_out.weight": g(2 * z_channels, ch * ch_mult[-1], 3, 3),
        "decoder.conv_in.weight": g(dec_ch * ch_mult[-1], z_channels, 3, 3),
        "decoder.conv_out.weight": g(out_channels, dec_ch * ch_mult[0], 3, 3),
        "quant_conv.weight": g(2 * embed_dim, 2 * z_channels, 1, 1),
        "post_quant_conv.weight": g(z_channels, embed_dim, 1, 1),
    }

    def add_block(prefix: str, block_in: int, block_out: int) -> None:
        sd[f"{prefix}.conv1.weight"] = g(block_out, block_in, 3, 3)
        sd[f"{prefix}.conv2.weight"] = g(block_out, block_out, 3, 3)
        if block_in != block_out:
            sd[f"{prefix}.nin_shortcut.weight"] = g(block_out, block_in, 1, 1)

    def add_mid(prefix: str, width: int) -> None:
        add_block(f"{prefix}.block_1", width, width)
        add_block(f"{prefix}.block_2", width, width)
        for name in ("q", "k", "v", "proj_out"):
            sd[f"{prefix}.attn_1.{name}.weight"] = g(width, width, 1, 1)

    in_mults = (1, *ch_mult)
    for level in range(levels):
        width = ch * ch_mult[level]
        for block in range(num_res_blocks):
            block_in = ch * in_mults[level] if block == 0 else width
            add_block(f"encoder.down.{level}.block.{block}", block_in, width)
        if level < levels - 1:
            sd[f"encoder.down.{level}.downsample.conv.weight"] = g(width, width, 3, 3)
    add_mid("encoder.mid", ch * ch_mult[-1])

    add_mid("decoder.mid", dec_ch * ch_mult[-1])
    running_in = dec_ch * ch_mult[-1]
    for level in reversed(range(levels)):
        width = dec_ch * ch_mult[level]
        for block in range(num_res_blocks + 1):
            block_in = running_in if block == 0 else width
            add_block(f"decoder.up.{level}.block.{block}", block_in, width)
        if level > 0:
            sd[f"decoder.up.{level}.upsample.conv.weight"] = g(width, width, 3, 3)
        running_in = width
    if batch_norm_latent:
        sd["bn.running_mean"] = g(4 * z_channels)
        sd["bn.running_var"] = g(4 * z_channels)
        sd["bn.num_batches_tracked"] = g()
    return sd


def diffusers_geometries(
    canonical: dict[str, TensorGeometry],
) -> dict[str, TensorGeometry]:
    """Independent inverse of ComfyUI's finite Diffusers VAE map.

    Attention uses the modern to_q/to_k/to_v/to_out.0 spelling and
    rank-2 Linear weights, matching real Diffusers AutoencoderKL files.
    """
    renames: list[tuple[str, str]] = [
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]
    for level in range(4):
        for block in range(2):
            renames.append(
                (
                    f"encoder.down.{level}.block.{block}.",
                    f"encoder.down_blocks.{level}.resnets.{block}.",
                )
            )
        if level < 3:
            renames.extend(
                (
                    (
                        f"down.{level}.downsample.",
                        f"down_blocks.{level}.downsamplers.0.",
                    ),
                    (
                        f"up.{3 - level}.upsample.",
                        f"up_blocks.{level}.upsamplers.0.",
                    ),
                )
            )
        for block in range(3):
            renames.append(
                (
                    f"decoder.up.{3 - level}.block.{block}.",
                    f"decoder.up_blocks.{level}.resnets.{block}.",
                )
            )
    for block in range(2):
        renames.append((f"mid.block_{block + 1}.", f"mid_block.resnets.{block}."))
    attention = (
        ("norm.", "group_norm."),
        ("q.", "to_q."),
        ("k.", "to_k."),
        ("v.", "to_v."),
        ("proj_out.", "to_out.0."),
    )
    converted: dict[str, TensorGeometry] = {}
    for key, geometry in canonical.items():
        diffusers_key = key
        if ".attn_1." in diffusers_key:
            for canonical_part, diffusers_part in attention:
                diffusers_key = diffusers_key.replace(canonical_part, diffusers_part)
            if key.endswith((".q.weight", ".k.weight", ".v.weight", ".proj_out.weight")):
                geometry = TensorGeometry(geometry.shape[:2], geometry.dtype)
        for canonical_part, diffusers_part in renames:
            diffusers_key = diffusers_key.replace(canonical_part, diffusers_part)
        converted[diffusers_key] = geometry
    assert DIFFUSERS_KL_MARKER in converted
    return converted


# ------------------------------------------------------------ detection


def test_detects_standard_layout_from_executed_reference_listing() -> None:
    config = detect_kl_config(golden_geometries("standard"))
    assert config == KLConfig(
        in_channels=3,
        out_channels=3,
        ch=32,
        decoder_ch=32,
        ch_mult=KL_STANDARD_CH_MULT,
        num_res_blocks=1,
        z_channels=3,
        embed_dim=3,
    )
    assert config.spatial_downscale == 8


def test_detects_x4_layout_from_executed_reference_listing() -> None:
    config = detect_kl_config(golden_geometries("x4"))
    assert config.ch_mult == KL_X4_CH_MULT
    assert config.num_res_blocks == 2
    assert config.spatial_downscale == 4


def test_detects_sd_sized_synthetic_header() -> None:
    config = detect_kl_config(kl_geometries())
    assert config.ch == 128
    assert config.decoder_ch == 128
    assert config.z_channels == 4
    assert config.embed_dim == 4


def test_detects_nested_quant_conv_keys() -> None:
    sd = kl_geometries()
    sd["encoder.quant_conv.weight"] = sd.pop("quant_conv.weight")
    sd["decoder.post_quant_conv.weight"] = sd.pop("post_quant_conv.weight")
    assert detect_kl_config(sd).embed_dim == 4


@pytest.mark.parametrize("family", ["sd15", "sdxl"])
def test_diffusers_complete_key_map_matches_canonical_golden(family: str) -> None:
    # SD1.5 and SDXL artifacts share the standard AutoencoderKL layout;
    # retain two named family cases so both accepted wiring roles stay pinned.
    assert family in {"sd15", "sdxl"}
    canonical = golden_geometries("standard")
    diffusers = diffusers_geometries(canonical)
    assert normalize_kl_keys(diffusers) == canonical
    assert detect_kl_config(diffusers) == detect_kl_config(canonical)


def test_diffusers_real_sd_shape_detects_without_payload_reads() -> None:
    canonical = kl_geometries(ch=128, z_channels=4, embed_dim=4)
    canonical["decoder.up.3.block.0.norm1.weight"] = TensorGeometry((512,), FLOAT32)
    config = detect_kl_config(diffusers_geometries(canonical))
    assert config == detect_kl_config(canonical)
    assert config.ch == 128
    assert config.embed_dim == 4


def test_diffusers_legacy_attention_aliases_match_modern() -> None:
    modern = diffusers_geometries(golden_geometries("standard"))
    legacy = {
        key.replace(".to_q.", ".query.")
        .replace(".to_k.", ".key.")
        .replace(".to_v.", ".value.")
        .replace(".to_out.0.", ".proj_attn."): geometry
        for key, geometry in modern.items()
    }
    assert normalize_kl_keys(legacy) == normalize_kl_keys(modern)


def test_diffusers_incomplete_mixed_unknown_and_ambiguous_refuse() -> None:
    canonical = kl_geometries()
    canonical["decoder.up.3.block.0.norm1.weight"] = TensorGeometry((512,), FLOAT32)
    diffusers = diffusers_geometries(canonical)
    incomplete = dict(diffusers)
    del incomplete[DIFFUSERS_KL_MARKER]
    with pytest.raises(KLDetectError, match="incomplete"):
        detect_kl_config(incomplete)

    mixed = dict(diffusers)
    mixed["encoder.down.0.block.0.conv1.weight"] = TensorGeometry((128, 128, 3, 3), FLOAT32)
    with pytest.raises(KLDetectError, match="mixed"):
        detect_kl_config(mixed)

    unknown = dict(diffusers)
    unknown["decoder.up_blocks.4.resnets.0.norm1.weight"] = TensorGeometry((128,), FLOAT32)
    with pytest.raises(KLDetectError, match="unknown"):
        detect_kl_config(unknown)

    ambiguous = dict(diffusers)
    query = "encoder.mid_block.attentions.0.to_q.weight"
    ambiguous["encoder.mid_block.attentions.0.query.weight"] = ambiguous[query]
    with pytest.raises(KLDetectError, match="ambiguous"):
        detect_kl_config(ambiguous)

    for canonical_key in (
        "encoder.mid.attn_1.q.weight",
        "encoder.mid.block_1.conv1.weight",
        "encoder.down.0.block.0.conv1.weight",
        "encoder.down.0.block.0.nin_shortcut.weight",
        "encoder.norm_out.weight",
    ):
        mixed_alias = dict(diffusers)
        mixed_alias[canonical_key] = TensorGeometry((128, 128), FLOAT32)
        with pytest.raises(KLDetectError, match="mixed"):
            detect_kl_config(mixed_alias)

    unknown_attention = dict(diffusers)
    unknown_attention["encoder.mid_block.attentions.0.to_out.1.weight"] = TensorGeometry(
        (128, 128), FLOAT32
    )
    with pytest.raises(KLDetectError, match="unknown"):
        detect_kl_config(unknown_attention)


def test_kl_prefix_alias_collision_refuses_before_normalization() -> None:
    canonical = kl_geometries()
    canonical["encoder.quant_conv.weight"] = canonical["quant_conv.weight"]
    with pytest.raises(KLDetectError, match="ambiguous"):
        detect_kl_config(canonical)


def test_detects_wider_decoder_base() -> None:
    assert detect_kl_config(kl_geometries(decoder_ch=256)).decoder_ch == 256


def test_detects_regularizer_only_from_executed_reference_listing() -> None:
    config = detect_kl_config(golden_geometries("regularizer"))
    assert config == KLConfig(
        in_channels=3,
        out_channels=3,
        ch=32,
        decoder_ch=32,
        ch_mult=KL_STANDARD_CH_MULT,
        num_res_blocks=1,
        z_channels=16,
        embed_dim=16,
        quant_convs=False,
    )


def test_detects_regularizer_only_synthetic_flux_shape() -> None:
    """The classic Flux ae layout: no quant convs, z_channels 16;
    embed_dim IS z_channels for this variant."""
    sd = kl_geometries(z_channels=16, embed_dim=16)
    del sd["quant_conv.weight"]
    del sd["post_quant_conv.weight"]
    config = detect_kl_config(sd)
    assert not config.quant_convs
    assert config.z_channels == 16
    assert config.embed_dim == 16
    assert config.spatial_downscale == 8


def test_config_rejects_quant_convless_embed_mismatch() -> None:
    with pytest.raises(ValueError, match="quant_convs=False requires"):
        KLConfig(
            in_channels=3,
            out_channels=3,
            ch=128,
            decoder_ch=128,
            ch_mult=KL_STANDARD_CH_MULT,
            num_res_blocks=2,
            z_channels=4,
            embed_dim=8,
            quant_convs=False,
        )


# ---------------------------------------------------- batch-norm latent


def test_detects_batch_norm_from_executed_reference_listing() -> None:
    config = detect_kl_config(golden_geometries("batch_norm"))
    assert config == KLConfig(
        in_channels=3,
        out_channels=3,
        ch=32,
        decoder_ch=32,
        ch_mult=KL_STANDARD_CH_MULT,
        num_res_blocks=1,
        z_channels=4,
        embed_dim=4,
        batch_norm_latent=True,
    )
    assert config.spatial_downscale == 16
    assert config.latent_channels == 16


def test_detects_batch_norm_synthetic_flux2_shape() -> None:
    """The real Flux2 VAE geometry: ch 128, z_channels 32, embed_dim
    32, bn over the packed 128 channels, external downscale 16."""
    config = detect_kl_config(kl_geometries(z_channels=32, embed_dim=32, batch_norm_latent=True))
    assert config.batch_norm_latent
    assert config.z_channels == 32
    assert config.embed_dim == 32
    assert config.spatial_downscale == 16
    assert config.latent_channels == 128


def test_diffusers_layout_batch_norm_detects() -> None:
    """The diffusers-named Flux2 VAE file carries the same top-level
    bn buffers; they pass through key normalization untouched."""
    canonical = kl_geometries(z_channels=32, embed_dim=32, batch_norm_latent=True)
    canonical["decoder.up.3.block.0.norm1.weight"] = TensorGeometry((512,), FLOAT32)
    diffusers = diffusers_geometries(canonical)
    assert "bn.running_mean" in diffusers
    config = detect_kl_config(diffusers)
    assert config == detect_kl_config(canonical)
    assert config.batch_norm_latent


def test_batch_norm_without_quant_convs_refuses() -> None:
    sd = kl_geometries(z_channels=16, embed_dim=16, batch_norm_latent=True)
    del sd["quant_conv.weight"]
    del sd["post_quant_conv.weight"]
    with pytest.raises(KLDetectError, match="without quant convs"):
        detect_kl_config(sd)


def test_batch_norm_embed_mismatch_refuses() -> None:
    sd = kl_geometries(z_channels=4, embed_dim=8, batch_norm_latent=True)
    with pytest.raises(KLDetectError, match="embed_dim == z_channels"):
        detect_kl_config(sd)


def test_batch_norm_wrong_extent_refuses() -> None:
    for shape in ((4,), (32,)):
        sd = kl_geometries(batch_norm_latent=True)
        sd["bn.running_mean"] = TensorGeometry(shape, FLOAT32)
        sd["bn.running_var"] = TensorGeometry(shape, FLOAT32)
        with pytest.raises(KLDetectError, match="packed 2x2 latent"):
            detect_kl_config(sd)


def test_config_rejects_batch_norm_without_quant_convs() -> None:
    with pytest.raises(ValueError, match="batch_norm_latent requires quant_convs"):
        KLConfig(
            in_channels=3,
            out_channels=3,
            ch=128,
            decoder_ch=128,
            ch_mult=KL_STANDARD_CH_MULT,
            num_res_blocks=2,
            z_channels=32,
            embed_dim=32,
            quant_convs=False,
            batch_norm_latent=True,
        )


def test_config_rejects_batch_norm_embed_mismatch() -> None:
    with pytest.raises(ValueError, match="embed_dim == z_channels"):
        KLConfig(
            in_channels=3,
            out_channels=3,
            ch=128,
            decoder_ch=128,
            ch_mult=KL_STANDARD_CH_MULT,
            num_res_blocks=2,
            z_channels=32,
            embed_dim=16,
            batch_norm_latent=True,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda sd: sd.update(
                {"decoder.up_blocks.0.resnets.0.norm1.weight": (TensorGeometry((4,), FLOAT32))}
            ),
            "diffusers",
        ),
        (
            lambda sd: sd.update({"decoder.mid.block_1.mix_factor": TensorGeometry((1,), FLOAT32)}),
            "video",
        ),
        (
            lambda sd: sd.update({"taesd_decoder.1.weight": TensorGeometry((4, 4, 3, 3), FLOAT32)}),
            "TAESD",
        ),
        (
            lambda sd: sd.update({"bn.running_mean": TensorGeometry((16,), FLOAT32)}),
            "present together",
        ),
        (
            lambda sd: sd.update({"bn.running_var": TensorGeometry((16,), FLOAT32)}),
            "present together",
        ),
        (lambda sd: sd.pop("decoder.conv_in.weight"), "not a KL autoencoder"),
        (lambda sd: sd.pop("encoder.conv_in.weight"), "decoder-only"),
        (
            lambda sd: sd.update(
                {"decoder.conv_in.weight": TensorGeometry((512, 4, 1, 3, 3), FLOAT32)}
            ),
            "conv3d",
        ),
        (lambda sd: sd.pop("quant_conv.weight"), "quant_conv"),
        (lambda sd: sd.pop("post_quant_conv.weight"), "quant_conv"),
        (
            lambda sd: sd.update({"post_quant_conv.weight": TensorGeometry((6, 4, 1, 1), FLOAT32)}),
            "decoder.conv_in expects",
        ),
        (
            lambda sd: sd.update({"quant_conv.weight": TensorGeometry((4, 4, 1, 1), FLOAT32)}),
            "double_z",
        ),
        (
            lambda sd: sd.update({"quant_conv.weight": TensorGeometry((8, 6, 1, 1), FLOAT32)}),
            "quant_conv geometry",
        ),
        (
            lambda sd: sd.update(
                {"encoder.conv_out.weight": TensorGeometry((4, 512, 3, 3), FLOAT32)}
            ),
            "encoder.conv_out",
        ),
        (
            lambda sd: sd.pop("encoder.down.0.block.0.conv1.weight"),
            "no encoder.down.0 residual blocks",
        ),
        (
            lambda sd: sd.pop("decoder.up.0.block.2.conv1.weight"),
            "expected num_res_blocks",
        ),
        (
            lambda sd: [sd.pop(f"decoder.up.3.block.{block}.conv1.weight") for block in range(3)],
            "disagree on resolution level count",
        ),
        (
            lambda sd: sd.update(
                {"encoder.down.1.block.0.conv1.weight": (TensorGeometry((200, 128, 3, 3), FLOAT32))}
            ),
            "not a multiple of base ch",
        ),
        (
            lambda sd: sd.update(
                {"encoder.down.1.block.0.conv1.weight": (TensorGeometry((384, 128, 3, 3), FLOAT32))}
            ),
            "not a known SD/SDXL KL layout",
        ),
        (
            lambda sd: sd.update(
                {
                    "encoder.down.3.downsample.conv.weight": (
                        TensorGeometry((512, 512, 3, 3), FLOAT32)
                    )
                }
            ),
            "downsample presence contradicts",
        ),
        (
            lambda sd: sd.pop("encoder.down.0.downsample.conv.weight"),
            "downsample presence contradicts",
        ),
        (
            lambda sd: sd.update(
                {"decoder.conv_in.weight": TensorGeometry((514, 4, 3, 3), FLOAT32)}
            ),
            "decoder.conv_in width",
        ),
        (lambda sd: sd.pop("decoder.conv_out.weight"), "decoder.conv_out"),
        (
            lambda sd: sd.update({"decoder.conv_in.weight": TensorGeometry((512,), FLOAT32)}),
            "rank-4 conv weight",
        ),
        (
            lambda sd: sd.update({"quant_conv.weight": TensorGeometry((8, 8, 3, 3), FLOAT32)}),
            "kernel",
        ),
        (
            lambda sd: sd.update(
                {
                    "encoder.down.0.block.0.conv_shortcut.weight": (
                        TensorGeometry((128, 128, 3, 3), FLOAT32)
                    )
                }
            ),
            "conv_shortcut",
        ),
        (
            lambda sd: sd.update(
                {"encoder.down.3.attn.0.q.weight": (TensorGeometry((512, 512, 1, 1), FLOAT32))}
            ),
            "per-resolution attention",
        ),
        (
            lambda sd: sd.pop("encoder.down.1.block.1.conv1.weight"),
            "expected 2 like level 0",
        ),
        (
            lambda sd: sd.pop("decoder.up.1.block.2.conv1.weight"),
            "decoder level 1 has",
        ),
        (
            lambda sd: sd.pop("decoder.up.3.upsample.conv.weight"),
            "upsample presence contradicts",
        ),
        (
            lambda sd: sd.update(
                {"decoder.up.0.upsample.conv.weight": (TensorGeometry((128, 128, 3, 3), FLOAT32))}
            ),
            "upsample presence contradicts",
        ),
        (
            lambda sd: sd.pop("encoder.down.1.block.0.nin_shortcut.weight"),
            "nin_shortcut presence contradicts",
        ),
        (
            lambda sd: sd.update(
                {
                    "encoder.down.0.block.0.nin_shortcut.weight": (
                        TensorGeometry((128, 128, 1, 1), FLOAT32)
                    )
                }
            ),
            "nin_shortcut presence contradicts",
        ),
        (
            lambda sd: sd.pop("encoder.down.0.block.0.conv2.weight"),
            "missing encoder.down.0.block.0.conv2",
        ),
        (
            lambda sd: sd.update(
                {"decoder.up.2.block.0.conv1.weight": (TensorGeometry((512, 128, 3, 3), FLOAT32))}
            ),
            "takes 128 channels, expected 512",
        ),
        (
            lambda sd: sd.pop("encoder.mid.attn_1.q.weight"),
            "missing encoder.mid.attn_1.q",
        ),
        (
            lambda sd: sd.pop("decoder.mid.block_1.conv1.weight"),
            "missing decoder.mid.block_1.conv1",
        ),
        (
            lambda sd: sd.update({"encoder.conv_out.weight": TensorGeometry((), FLOAT32)}),
            "rank-4 conv weight",
        ),
        (
            lambda sd: sd.update(
                {"encoder.down.0.block.0.conv1.weight": (TensorGeometry((128,), FLOAT32))}
            ),
            "rank-4 conv weight",
        ),
        (
            lambda sd: sd.update({"encoder.conv_in.weight": TensorGeometry((0, 3, 3, 3), FLOAT32)}),
            "zero-sized dimension",
        ),
        (
            lambda sd: sd.update(
                {"encoder.down.0.block.3.conv1.weight": (TensorGeometry((128, 128, 3, 3), FLOAT32))}
            ),
            "outside the detected layout",
        ),
        (
            lambda sd: sd.update(
                {"decoder.up.5.block.0.conv1.weight": (TensorGeometry((512, 512, 3, 3), FLOAT32))}
            ),
            "outside the detected layout",
        ),
        (
            lambda sd: sd.update(
                {"encoder.down.0.block.0.temb_proj.weight": (TensorGeometry((128, 512), FLOAT32))}
            ),
            "time-conditioned",
        ),
        (
            lambda sd: sd.update(
                {"encoder.down.x.block.0.conv1.weight": (TensorGeometry((128, 128, 3, 3), FLOAT32))}
            ),
            "not an integer",
        ),
        (
            lambda sd: sd.update(
                {"decoder.up.01.block.0.conv1.weight": (TensorGeometry((128, 128, 3, 3), FLOAT32))}
            ),
            "not an integer",
        ),
        (
            # unicode superscript two: isdigit-true but int() refuses
            lambda sd: sd.update(
                {
                    "encoder.down.\u00b2.block.0.conv1.weight": (
                        TensorGeometry((128, 128, 3, 3), FLOAT32)
                    )
                }
            ),
            "not an integer",
        ),
        (
            lambda sd: sd.update(
                {
                    "encoder.down.-1.block.0.conv1.weight": (
                        TensorGeometry((128, 128, 3, 3), FLOAT32)
                    )
                }
            ),
            "outside the detected layout",
        ),
        (
            lambda sd: sd.update(
                {"encoder.down.0.upsample.conv.weight": (TensorGeometry((128, 128, 3, 3), FLOAT32))}
            ),
            "unsupported member",
        ),
    ],
)
def test_rejects_unsupported_layouts(mutate, match: str) -> None:
    sd = kl_geometries()
    mutate(sd)
    with pytest.raises(KLDetectError, match=match):
        detect_kl_config(sd)


def test_rejection_is_error_not_guess_for_empty_mapping() -> None:
    with pytest.raises(KLDetectError):
        detect_kl_config({})


# -------------------------------------------------------- normalization


def test_normalize_kl_keys_renames_nested_quant_convs() -> None:
    out = normalize_kl_keys(
        {
            "decoder.post_quant_conv.weight": "a",
            "encoder.quant_conv.bias": "b",
            "decoder.conv_in.weight": "c",
        }
    )
    assert out == {
        "post_quant_conv.weight": "a",
        "quant_conv.bias": "b",
        "decoder.conv_in.weight": "c",
    }


def test_prefix_renames_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        KL_PREFIX_RENAMES["x"] = "y"  # type: ignore[index]


# ------------------------------------------------------------- config


def test_config_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="z_channels"):
        KLConfig(
            in_channels=3,
            out_channels=3,
            ch=128,
            decoder_ch=128,
            ch_mult=KL_STANDARD_CH_MULT,
            num_res_blocks=2,
            z_channels=0,
            embed_dim=4,
        )


def test_config_rejects_bad_ch_mult_and_dropout() -> None:
    with pytest.raises(ValueError, match="ch_mult"):
        KLConfig(
            in_channels=3,
            out_channels=3,
            ch=128,
            decoder_ch=128,
            ch_mult=(),
            num_res_blocks=2,
            z_channels=4,
            embed_dim=4,
        )
    with pytest.raises(ValueError, match="dropout"):
        KLConfig(
            in_channels=3,
            out_channels=3,
            ch=128,
            decoder_ch=128,
            ch_mult=KL_STANDARD_CH_MULT,
            num_res_blocks=2,
            z_channels=4,
            embed_dim=4,
            dropout=1.0,
        )


# ---------------------------------------------------------- descriptor


def test_descriptor_carries_reference_tiling_and_dtypes() -> None:
    config = detect_kl_config(kl_geometries())
    descriptor = kl_descriptor(config)
    assert descriptor.id == "dinkster.autoencoder_kl"
    assert descriptor.latent.channels == 4
    assert descriptor.latent.dimensions == 2
    assert descriptor.latent.spatial_downscale == 8
    assert descriptor.content_channels == 3
    assert descriptor.supported_dtypes == frozenset({BFLOAT16, FLOAT32})
    assert descriptor.supports_tiling
    tiling = descriptor.tiling
    assert tiling is not None
    assert tiling.decode_tile == (64, 64)
    assert tiling.decode_overlap == (16, 16)
    assert tiling.encode_tile == (512, 512)
    assert tiling.encode_overlap == (64, 64)


def test_x4_descriptor_downscale() -> None:
    config = detect_kl_config(golden_geometries("x4"))
    assert kl_descriptor(config).latent.spatial_downscale == 4


def test_batch_norm_descriptor_exposes_packed_geometry() -> None:
    config = detect_kl_config(kl_geometries(z_channels=32, embed_dim=32, batch_norm_latent=True))
    descriptor = kl_descriptor(config)
    assert descriptor.latent.channels == 128
    assert descriptor.latent.spatial_downscale == 16


# -------------------------------------------------------------- memory


def test_memory_estimator_matches_reference_formulas() -> None:
    estimator = KLMemoryEstimator()
    content = TensorGeometry((1, 3, 512, 768), FLOAT32)
    latent = TensorGeometry((1, 4, 64, 96), BFLOAT16)
    assert estimator.encode_bytes(content) == 1767 * 512 * 768 * 4
    assert estimator.decode_bytes(latent) == 2178 * 64 * 96 * 64 * 2


def test_memory_estimator_ratio_scales() -> None:
    content = TensorGeometry((1, 3, 8, 8), FLOAT32)
    assert KLMemoryEstimator(ratio=2.0).encode_bytes(
        content
    ) == 2 * KLMemoryEstimator().encode_bytes(content)


def test_memory_estimator_decode_multiplier_scales_decode_only() -> None:
    content = TensorGeometry((1, 3, 8, 8), FLOAT32)
    latent = TensorGeometry((1, 128, 4, 4), FLOAT32)
    packed = KLMemoryEstimator(decode_multiplier=4.0)
    base = KLMemoryEstimator()
    assert packed.decode_bytes(latent) == 4 * base.decode_bytes(latent)
    assert packed.encode_bytes(content) == base.encode_bytes(content)
