"""Stage 5 slice 7: the native classic-T5 text model (Flux stack).

Every golden in goldens/t5_text_goldens.json was produced by RUNNING
the reference stack @ the audited baseline (tools/gen_t5_text_goldens
.py): comfy.text_encoders.t5.T5 for the architecture, comfy.sd1_clip
.SDClipModel.encode_token_weights with Flux's T5XXLModel knobs for
the encode policy, comfy.text_encoders.flux.FluxClipModel for the
two-model composition. Weights come from the shared deterministic
hash (clip_fill.py), so both sides run bit-identical parameters
without storing megabytes.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from clip_fill import embedding_vectors, fill_state_dict
from dinkster_inference import (
    T5_XXL_CONFIG,
    T5_XXL_FLUX_PROFILE,
    T5_XXL_LTXV_PROFILE,
    UMT5_XXL_CONFIG,
    UMT5_XXL_WAN_PROFILE,
    Chunk,
    ClipTextConfig,
    Conditioning,
    EmbeddingSlot,
    PackedToken,
    T5Config,
    TokenizerProfile,
    WeightedSpan,
    t5_layout,
)
from dinkster_inference_torch import (
    SD1_CLIP_L_POLICY,
    CastOperations,
    ClipTextEncoder,
    ClipTextModel,
    T5EncodeError,
    T5TextEncoder,
    T5TextModel,
    compose_flux_conditioning,
    compose_flux_t5_conditioning,
    relative_position_bucket,
    select_attention,
)
from dinkster_inference_torch._conditioning_layout import (
    declare_text_conditioning,
    declared_token_count,
)

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "t5_text_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])

T5_PROFILE = T5_XXL_FLUX_PROFILE


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> T5Config:
    stored = dict(GOLDENS["cases"][case]["config"])
    return T5Config(**stored)


def case_chunks(case: str) -> tuple[Chunk, ...]:
    def unit(stored: Any) -> int | EmbeddingSlot:
        if isinstance(stored, dict):
            return EmbeddingSlot(stored["embedding"], stored["row"])
        return int(stored)

    return tuple(
        tuple(PackedToken(unit(stored), float(weight), 0) for stored, weight in chunk)
        for chunk in GOLDENS["cases"][case]["chunks"]
    )


def build_model(case: str) -> T5TextModel:
    model = T5TextModel(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def build_encoder(case: str) -> T5TextEncoder:
    spec = GOLDENS["cases"][case]
    config = case_config(case)

    def lookup(name: str) -> torch.Tensor | None:
        rows = spec["embeddings"].get(name)
        if rows is None:
            return None
        return embedding_vectors(name, rows, config.d_model)

    return T5TextEncoder(
        build_model(case),
        profile=T5_PROFILE,
        embeddings=lookup,
    )


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


def test_t5_injects_scaled_masked_kernel_without_changing_state_or_output() -> None:
    case = CASES[0]
    baseline = build_model(case)
    spy = CallableModuleKernel(select_attention("t5").kernel)
    model = T5TextModel(case_config(case), attention_kernel=spy)
    model.load_state_dict(baseline.state_dict(), strict=True)
    embeds = baseline.embed_tokens(torch.tensor([[3, 1]]))
    assert set(model.state_dict()) == set(baseline.state_dict())
    assert_kernel_is_not_model_state(model, spy)
    torch.testing.assert_close(model(embeds), baseline(embeds))
    assert spy.calls
    assert all(
        call["mask"] is not None and call["scale"] == 1.0 and not call["causal"]
        for call in spy.calls
    )


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in t5_layout(case_config(case)).items())
    assert predicted == golden_entries(case)


def test_full_size_module_matches_reference_layout() -> None:
    """The real T5-XXL architecture, constructed on the meta device
    (initless factories never touch the storage), against the
    reference model's own full-size listing."""
    with torch.device("meta"):
        model = T5TextModel(T5_XXL_CONFIG)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["t5_xxl"]]
    assert ours == golden


def test_full_size_umt5_module_matches_reference_layout() -> None:
    with torch.device("meta"):
        model = T5TextModel(UMT5_XXL_CONFIG)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"]["umt5_xxl"]]
    assert ours == golden


# ---------------------------------------------------- bucket math


def test_relative_position_bucket_matches_executed_reference() -> None:
    spec = GOLDENS["buckets"]
    positions = torch.arange(
        spec["first_position"],
        spec["first_position"] + len(spec["buckets"]),
        dtype=torch.long,
    )
    got = relative_position_bucket(
        positions,
        num_buckets=spec["num_buckets"],
        max_distance=spec["max_distance"],
    )
    assert got.tolist() == spec["buckets"]


def test_bucket_zero_is_zero_and_sign_claims_half() -> None:
    got = relative_position_bucket(torch.tensor([0, 1, -1, 1000, -1000]))
    assert got.tolist() == [0, 17, 1, 31, 15]


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_encode_matches_executed_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    got = build_encoder(case).encode_chunks(case_chunks(case))
    assert got.embeddings.dtype == torch.float32
    assert got.pooled is None
    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=1e-4, atol=1e-5)


def test_umt5_encode_matches_executed_reference_and_zeros_padding() -> None:
    spec = GOLDENS["umt5"]
    config = T5Config(**spec["config"])
    model = T5TextModel(config)
    entries = [(key, list(shape)) for key, shape in spec["state_dict"]]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    chunks = tuple(
        tuple(PackedToken(int(unit), float(weight), 0) for unit, weight in chunk)
        for chunk in spec["chunks"]
    )
    encoder = T5TextEncoder(model)
    assert encoder.profile is UMT5_XXL_WAN_PROFILE
    got = encoder.encode_chunks(chunks)
    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=1e-4, atol=1e-5)
    assert torch.count_nonzero(got.embeddings[:, 4:]) == 0


def test_umt5_attention_uses_one_bias_per_block_and_padding_mask() -> None:
    spec = GOLDENS["umt5"]
    config = T5Config(**spec["config"])
    spy = CallableModuleKernel(select_attention("t5").kernel)
    model = T5TextModel(config, attention_kernel=spy)
    entries = [(key, list(shape)) for key, shape in spec["state_dict"]]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    ids = torch.tensor([[100, 200, 300, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    model(model.embed_tokens(ids), mask)
    assert len(spy.calls) == config.num_layers
    assert all(call["mask"].shape == (1, config.num_heads, 6, 6) for call in spy.calls)
    assert not torch.equal(spy.calls[0]["mask"], spy.calls[1]["mask"])


def test_flux_composition_matches_executed_reference() -> None:
    spec = GOLDENS["flux"]
    l_config = ClipTextConfig(**spec["l_config"])
    l_model = ClipTextModel(l_config)
    entries = sorted((key, list(value.shape)) for key, value in l_model.state_dict().items())
    l_model.load_state_dict(fill_state_dict(entries), strict=True)
    l_chunks = tuple(
        tuple(PackedToken(int(unit), float(weight), 0) for unit, weight in chunk)
        for chunk in spec["l_chunks"]
    )
    clip_l = ClipTextEncoder(l_model, policy=SD1_CLIP_L_POLICY).encode_chunks(l_chunks)
    t5 = build_encoder(spec["t5_case"]).encode_chunks(case_chunks(spec["t5_case"]))
    got = compose_flux_conditioning(t5, clip_l)
    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=1e-4, atol=1e-5)
    assert got.pooled is not None
    torch.testing.assert_close(got.pooled, dec(spec["pooled"]), rtol=1e-4, atol=1e-5)


def test_flux_composition_requires_clip_pooled() -> None:
    cond = Conditioning(torch.randn(1, 16, 8), None)
    with pytest.raises(T5EncodeError, match="pooled"):
        compose_flux_conditioning(cond, Conditioning(torch.randn(1, 4, 8)))


def test_flux_t5_composition_supplies_neutral_clip_vector() -> None:
    embeddings = torch.randn(2, 16, 8, dtype=torch.float64)
    t5 = declare_text_conditioning(Conditioning(embeddings), 11)

    got = compose_flux_t5_conditioning(t5)

    assert got.embeddings is embeddings
    assert got.pooled is not None
    assert got.pooled.shape == (2, 768)
    assert got.pooled.dtype == embeddings.dtype
    assert got.pooled.device == embeddings.device
    assert torch.count_nonzero(got.pooled).item() == 0
    assert declared_token_count(got) == 11


# ----------------------------------------------------- encode policy


def test_unweighted_prompt_skips_the_empty_chunk_batch_row() -> None:
    """No weight deviates from 1.0 -> the encoder must not batch an
    empty-prompt row (the reference only appends one when needed)."""
    model = build_model("t5_unweighted")
    seen: list[int] = []
    original = model.forward

    def spy(embeds: torch.Tensor) -> torch.Tensor:
        seen.append(embeds.shape[0])
        return original(embeds)

    model.forward = spy  # type: ignore[method-assign]
    T5TextEncoder(model, profile=T5_PROFILE).encode_chunks(case_chunks("t5_unweighted"))
    assert seen == [1]


def test_encode_spans_equals_packed_chunks() -> None:
    """``encode`` must be exactly pack_spans + encode_chunks with a
    short-prompt profile (min_length pads like the Flux profile)."""
    profile = TokenizerProfile(
        max_length=99999999,
        start_token=None,
        end_token=1,
        pad_token=0,
        pad_to_max_length=False,
        min_length=16,
    )
    encoder = T5TextEncoder(build_model("t5_weighted"), profile=profile)
    spans = (
        WeightedSpan((100,), 1.0),
        WeightedSpan((200, 300), 1.3),
    )
    from dinkster_inference import pack_spans

    direct = encoder.encode(spans)
    packed = encoder.encode_chunks(pack_spans(spans, profile))
    torch.testing.assert_close(direct.embeddings, packed.embeddings)


def test_encode_packing_overrides_are_scoped_to_one_call() -> None:
    profile = TokenizerProfile(
        max_length=99999999,
        start_token=None,
        end_token=1,
        pad_token=0,
        pad_to_max_length=False,
        min_length=16,
    )
    encoder = T5TextEncoder(build_model("t5_unweighted"), profile=profile)
    spans = (WeightedSpan((100,), 1.0),)

    overridden = encoder.encode(spans, min_padding=3, min_length=8)
    default = encoder.encode(spans)

    assert overridden.embeddings.shape[1] == 8
    assert default.embeddings.shape[1] == 16
    assert encoder.profile is profile


def test_ltxv_mask_policy_masks_through_eos_without_zeroing_padding() -> None:
    """Classic T5 with LTXV's explicit policy: the model sees a mask
    covering the prompt through EOS, the padded rows keep the model
    output (no zero-out), and the declared token count is the
    through-EOS length so the runtime can rebuild the mask."""
    model = build_model("t5_unweighted")
    masks: list[torch.Tensor | None] = []
    original = model.forward

    def spy(embeds: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        masks.append(attention_mask)
        return original(embeds) if attention_mask is None else original(embeds, attention_mask)

    model.forward = spy  # type: ignore[method-assign]
    encoder = T5TextEncoder(
        model,
        profile=T5_XXL_LTXV_PROFILE,
        attention_masked=True,
        zero_out_masked=False,
    )
    got = encoder.encode((WeightedSpan((100,), 1.0),))
    floor = T5_XXL_LTXV_PROFILE.min_length
    assert floor is not None
    assert got.embeddings.shape[1] == floor
    assert declared_token_count(got) == 2
    assert len(masks) == 1
    assert masks[0] is not None
    assert masks[0].tolist() == [[1, 1] + [0] * (floor - 2)]
    assert torch.count_nonzero(got.embeddings[:, 2:]) > 0


def test_masked_not_zeroed_declares_only_contiguous_prefixes() -> None:
    """The declared token count rebuilds the mask as a contiguous
    prefix, so it can describe padding only on the final chunk's tail;
    padding before the last attended token must refuse."""
    encoder = T5TextEncoder(
        build_model("t5_unweighted"),
        profile=T5_PROFILE,
        attention_masked=True,
        zero_out_masked=False,
    )

    def chunk(units: tuple[int, ...]) -> Chunk:
        return tuple(PackedToken(unit, 1.0, 0) for unit in units)

    tail_padded = (chunk((100, 100, 1)), chunk((100, 1, 0)))
    assert declared_token_count(encoder.encode_chunks(tail_padded)) == 5

    interior_padded = (chunk((100, 1, 0)), chunk((100, 100, 1)))
    with pytest.raises(T5EncodeError, match="contiguous"):
        encoder.encode_chunks(interior_padded)


def test_classic_t5_defaults_stay_unmasked_and_declare_every_row() -> None:
    encoder = T5TextEncoder(build_model("t5_unweighted"), profile=T5_XXL_LTXV_PROFILE)
    assert (encoder.attention_masked, encoder.zero_out_masked) == (False, False)
    got = encoder.encode((WeightedSpan((100,), 1.0),))
    assert declared_token_count(got) == T5_XXL_LTXV_PROFILE.min_length


def test_zero_out_masked_without_attention_mask_refuses() -> None:
    with pytest.raises(T5EncodeError, match="needs attention_masked"):
        T5TextEncoder(
            build_model("t5_unweighted"),
            profile=T5_PROFILE,
            attention_masked=False,
            zero_out_masked=True,
        )


def test_left_padded_profile_refuses() -> None:
    profile = TokenizerProfile(
        max_length=99999999,
        start_token=None,
        end_token=1,
        pad_token=0,
        pad_to_max_length=False,
        min_length=16,
        pad_left=True,
    )
    with pytest.raises(T5EncodeError, match="left-padded"):
        T5TextEncoder(build_model("t5_unweighted"), profile=profile)


def test_empty_chunk_list_refuses() -> None:
    encoder = T5TextEncoder(build_model("t5_unweighted"), profile=T5_PROFILE)
    with pytest.raises(ValueError, match="no chunks"):
        encoder.encode_chunks(())


def test_mismatched_chunk_lengths_refuse() -> None:
    encoder = T5TextEncoder(build_model("t5_unweighted"), profile=T5_PROFILE)
    chunks = case_chunks("t5_unweighted")
    with pytest.raises(ValueError, match="share one length"):
        encoder.encode_chunks([chunks[0], chunks[0][:-1]])


def test_unresolvable_embedding_refuses() -> None:
    encoder = T5TextEncoder(build_model("t5_unweighted"), profile=T5_PROFILE)
    chunk = (
        PackedToken(EmbeddingSlot("ghost", 0), 1.0, 0),
        PackedToken(1, 1.0, 0),
    )
    with pytest.raises(T5EncodeError, match="ghost"):
        encoder.encode_chunks((chunk,))


def test_mis_shaped_embedding_refuses_instead_of_dropping() -> None:
    def lookup(name: str) -> torch.Tensor | None:
        return torch.zeros(2, 7)

    encoder = T5TextEncoder(build_model("t5_unweighted"), profile=T5_PROFILE, embeddings=lookup)
    chunk = (
        PackedToken(EmbeddingSlot("wide", 0), 1.0, 0),
        PackedToken(1, 1.0, 0),
    )
    with pytest.raises(T5EncodeError, match="wide"):
        encoder.encode_chunks((chunk,))


# ---------------------------------------------- fp16 storage policy
#
# The reference always runs text encoders at fp32 compute over the
# checkpoint's storage dtype (sd1_clip.SDClipModel @ 947c2749:
# manual_cast ops + a hardcoded dtype=torch.float32 forward). fp16
# COMPUTE is not a supported T5-XXL shape - its activations exceed
# fp16 range, the RMS variance overflows to inf, rsqrt(inf) = 0, and
# every block past the overflow contributes signed zeros. The GPU
# suite proves the full-size checkpoint under this policy;
# CastOperations semantics are pinned per-layer in test_ops.py.


def test_fp16_storage_at_fp32_compute_tracks_the_golden() -> None:
    case = "t5_weighted"
    spec = GOLDENS["cases"][case]
    config = case_config(case)
    model = T5TextModel(config, operations=CastOperations(torch.float32))
    model.load_state_dict(
        {
            key: value.to(torch.float16)
            for key, value in fill_state_dict(golden_entries(case)).items()
        },
        strict=True,
        assign=True,
    )

    def lookup(name: str) -> torch.Tensor | None:
        rows = spec["embeddings"].get(name)
        if rows is None:
            return None
        return embedding_vectors(name, rows, config.d_model)

    encoder = T5TextEncoder(model, profile=T5_PROFILE, embeddings=lookup)
    got = encoder.encode_chunks(case_chunks(case))
    assert got.embeddings.dtype == torch.float32
    # fp16 rounds the stored weights; the forward itself is fp32, so
    # the drift is bounded by the weight quantization alone.
    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=0.02, atol=0.02)


#: Tiny UMT5 (Wan's model_type) in the golden cases' geometry. The
#: golden fill keeps activations in fp16 range, so the overflow model
#: scales the embedding table until magnitudes exceed sqrt(fp16 max)
#: ~ 256 - the range the full-size official umt5_xxl weights reach.
UMT5_OVERFLOW_CONFIG = T5Config(
    d_model=48,
    d_ff=96,
    d_kv=12,
    num_heads=4,
    num_layers=3,
    vocab_size=512,
    dense_act_fn="gelu_pytorch_tanh",
    is_gated_act=True,
    model_type="umt5",
)

UMT5_OVERFLOW_SCALE = 4096.0


def umt5_overflow_state(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Deterministic weights for the tiny UMT5 whose embedding rows
    overflow the fp16 RMS variance, stored in ``dtype``; the scaled
    values themselves stay finite in fp16."""
    entries = [
        (key, list(value.shape))
        for key, value in T5TextModel(UMT5_OVERFLOW_CONFIG).state_dict().items()
    ]
    filled = fill_state_dict(entries)
    filled["shared.weight"] = filled["shared.weight"] * UMT5_OVERFLOW_SCALE
    return {key: value.to(dtype) for key, value in filled.items()}


def build_umt5_overflow_model(dtype: torch.dtype) -> T5TextModel:
    """The overflow UMT5 with all weights stored (and therefore
    computed) in ``dtype``."""
    model = T5TextModel(UMT5_OVERFLOW_CONFIG)
    model.load_state_dict(umt5_overflow_state(dtype), strict=True, assign=True)
    return model


def build_umt5_overflow_model_via_wan_policy() -> T5TextModel:
    """The fp16-stored overflow weights loaded the way assemble_wan21
    loads Wan's UMT5 tower: the compute dtype comes from
    default_text_dtype("dinkster.wan21") and the storage/compute cast
    decision from the production _pick_operations. A revert of the Wan
    float32 text policy makes this construction compute in fp16."""
    from dinkster_inference import default_text_dtype
    from dinkster_inference_torch.assemble import (
        _pick_operations,  # pyright: ignore[reportPrivateUsage]
    )

    compute_dtype = getattr(torch, default_text_dtype("dinkster.wan21").name)
    stored = umt5_overflow_state(torch.float16)
    model = T5TextModel(
        UMT5_OVERFLOW_CONFIG, operations=_pick_operations(stored, frozenset(), compute_dtype)
    )
    model.load_state_dict(stored, strict=True, assign=True)
    return model


def umt5_overflow_chunks() -> tuple[Chunk, ...]:
    """One Wan-profile chunk: real tokens, EOS, then pads (exercising
    the UMT5 mask and zero-out policy alongside the collapse)."""
    profile = UMT5_XXL_WAN_PROFILE
    assert profile.end_token is not None
    words = tuple(
        PackedToken(unit, 1.0, word_id)
        for word_id, unit in enumerate((7, 23, 101, 202, 303), start=1)
    )
    tail = (PackedToken(profile.end_token, 1.0, 0),) + (PackedToken(profile.pad_token, 1.0, 0),) * 4
    return (words + tail,)


def test_umt5_fp16_compute_collapses_to_zeros_while_fp32_stays_finite() -> None:
    """Executed regression for the documented Wan UMT5 fp16 failure:
    T5LayerNorm computes the RMS variance in the INPUT dtype, so
    activations past sqrt(fp16 max) square to inf, rsqrt(inf) = 0
    zeroes every normed input, and the final layer norm collapses the
    whole encoding to exact zeros - while the same weights and chunks
    at fp32 encode finite and non-degenerate. This arithmetic is why
    default_text_dtype pins Wan text encode to float32."""
    chunks = umt5_overflow_chunks()

    fp32 = T5TextEncoder(build_umt5_overflow_model(torch.float32)).encode_chunks(chunks)
    assert fp32.embeddings.dtype == torch.float32
    assert fp32.embeddings.shape == (1, 10, 48)
    assert bool(torch.isfinite(fp32.embeddings).all())
    assert bool(fp32.embeddings.abs().max() > 0)

    model = build_umt5_overflow_model(torch.float16)
    token_ids = torch.tensor(
        [[packed.unit if isinstance(packed.unit, int) else 0 for packed in chunks[0]]]
    )
    variance = model.embed_tokens(token_ids).pow(2).mean(-1)
    assert bool(torch.isinf(variance).all())

    fp16 = T5TextEncoder(model).encode_chunks(chunks)
    assert fp16.embeddings.dtype == torch.float32
    assert fp16.embeddings.shape == (1, 10, 48)
    assert bool((fp16.embeddings == 0).all())


def test_umt5_fp16_storage_under_wan_text_policy_encodes_finite() -> None:
    """Executed pin of the production policy connection behind #760:
    Wan's UMT5 tower computes at default_text_dtype("dinkster.wan21")
    over fp16 checkpoint storage via cast-at-use (assemble_wan21 ->
    _pick_operations). The overflow weights that collapse under fp16
    compute must encode finite and nonzero through that exact
    boundary; a policy revert to float16 turns the cast decision into
    same-dtype fp16 compute and fails the nonzero assert."""
    got = T5TextEncoder(build_umt5_overflow_model_via_wan_policy()).encode_chunks(
        umt5_overflow_chunks()
    )
    assert got.embeddings.dtype == torch.float32
    assert got.embeddings.shape == (1, 10, 48)
    assert bool(torch.isfinite(got.embeddings).all())
    assert bool(got.embeddings.abs().max() > 0)


# ------------------------------------------------- training program


def test_gradients_flow_to_every_parameter_kind() -> None:
    """No inference_mode/no_grad anywhere: the encode path must be
    trainable end to end, including the relative bias table."""
    encoder = build_encoder("t5_weighted")
    got = encoder.encode_chunks(case_chunks("t5_weighted"))
    got.embeddings.square().mean().backward()
    for name, parameter in encoder.model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


# ------------------------------------------------ real checkpoints


T5_XXL_FP16 = Path("/home/kosin/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors")


@pytest.mark.skipif(not T5_XXL_FP16.exists(), reason="real T5-XXL checkpoint not present")
def test_real_checkpoint_header_matches_module_layout() -> None:
    """The real fp16 T5-XXL header (shapes only; nothing loads) must
    be exactly the module layout plus the optional embed_tokens
    duplicate the reference tolerates via strict=False."""
    from dinkster_inference import (
        T5_TEXT_OPTIONAL_KEYS,
        detect_t5_config,
        load_safetensors_header,
    )

    source = load_safetensors_header(T5_XXL_FP16)
    geometries = {key: entry.geometry for key, entry in source.entries.items()}
    config = detect_t5_config(geometries)
    with torch.device("meta"):
        model = T5TextModel(config)
    listing = model.state_dict()
    assert set(geometries) - set(listing) == T5_TEXT_OPTIONAL_KEYS
    for key, value in listing.items():
        assert tuple(value.shape) == geometries[key].shape


# ------------------------------------------------- fp8 NaN safeguard


def test_embed_tokens_scrubs_fp8_nan_rows() -> None:
    """The reference's fp8-T5 fix (t5.py @ 947c2749): fp8 storage can
    hold NaN encodings, and one poisoned row would sink the whole
    encoding. Cast-at-use dequantizes first, then the scrub zeroes the
    poison; clean rows keep their dequantized values exactly."""
    case = CASES[0]
    model = T5TextModel(case_config(case), operations=CastOperations(torch.float32))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    quantized = model.shared.weight.data.to(torch.float8_e4m3fn)
    quantized[5] = torch.tensor(float("nan")).to(torch.float8_e4m3fn)
    model.shared.weight = torch.nn.Parameter(quantized, requires_grad=False)

    embeds = model.embed_tokens(torch.tensor([[5, 1, 2]]))
    assert embeds.dtype == torch.float32
    assert not embeds.isnan().any()
    assert torch.equal(embeds[0, 0], torch.zeros_like(embeds[0, 0]))
    assert torch.equal(embeds[0, 1], quantized[1].to(torch.float32))


def test_embed_tokens_keeps_nan_for_ordinary_storage() -> None:
    """The scrub is gated on fp8 STORAGE: a NaN in an ordinary-dtype
    embedding is real data corruption and must stay visible, exactly
    like the reference (whose fix is gated on the checkpoint dtype)."""
    case = CASES[0]
    model = build_model(case)
    with torch.no_grad():
        model.shared.weight[5] = float("nan")
    embeds = model.embed_tokens(torch.tensor([[5]]))
    assert embeds.isnan().any()
