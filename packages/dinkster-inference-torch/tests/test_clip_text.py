"""Stage 5 slice 4: the native SD1/SDXL CLIP text model.

Every golden in goldens/clip_text_goldens.json was produced by
RUNNING the reference stack @ the audited baseline
(tools/gen_clip_text_goldens.py): comfy.clip_model.CLIPTextModel for
the architecture, comfy.sd1_clip.SDClipModel.encode_token_weights for
the weighted-chunk policy, comfy.sdxl_clip.SDXLClipModel for the
two-tower composition. Weights come from the shared deterministic
hash (clip_fill.py), so both sides run bit-identical parameters
without storing megabytes.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from clip_fill import embedding_vectors, fill_state_dict
from dinkster_inference import (
    CLIP_G_PROFILE,
    CLIP_G_TEXT_CONFIG,
    CLIP_L_PROFILE,
    CLIP_L_TEXT_CONFIG,
    Chunk,
    ClipTextConfig,
    Conditioning,
    EmbeddingSlot,
    PackedToken,
    TokenizerProfile,
    WeightedSpan,
    clip_text_layout,
    pack_spans,
)
from dinkster_inference_torch import (
    SD1_CLIP_L_POLICY,
    SDXL_CLIP_POLICY,
    ClipAttention,
    ClipEncodeError,
    ClipEncodePolicy,
    ClipTextEncoder,
    ClipTextModel,
    apply_span_weights,
    compose_sdxl_conditioning,
    select_attention,
)

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "clip_text_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def case_config(case: str) -> ClipTextConfig:
    return ClipTextConfig(**GOLDENS["cases"][case]["config"])


def case_profile(case: str) -> TokenizerProfile:
    specials = GOLDENS["cases"][case]["special_tokens"]
    profile = CLIP_L_PROFILE if specials["pad"] == CLIP_L_PROFILE.pad_token else CLIP_G_PROFILE
    assert specials == {
        "start": profile.start_token,
        "end": profile.end_token,
        "pad": profile.pad_token,
    }
    return profile


def case_policy(case: str) -> ClipEncodePolicy:
    spec = GOLDENS["cases"][case]
    return ClipEncodePolicy(
        hidden_layer=None if spec["layer"] == "last" else spec["layer_idx"],
        layer_norm_hidden_state=spec["layer_norm_hidden_state"],
        projected_pooled=spec["return_projected_pooled"],
    )


def case_chunks(case: str) -> tuple[Chunk, ...]:
    def unit(stored: Any) -> int | EmbeddingSlot:
        if isinstance(stored, dict):
            return EmbeddingSlot(stored["embedding"], stored["row"])
        return int(stored)

    return tuple(
        tuple(PackedToken(unit(stored), float(weight), 0) for stored, weight in chunk)
        for chunk in GOLDENS["cases"][case]["chunks"]
    )


def case_spans(case: str) -> tuple[WeightedSpan, ...]:
    spans = []
    profile = case_profile(case)
    for packed in case_chunks(case)[0][1:]:
        if packed.unit == profile.end_token:
            break
        if isinstance(packed.unit, EmbeddingSlot):
            raise AssertionError(f"{case} uses embedding slots")
        spans.append(WeightedSpan((packed.unit,), packed.weight))
    return tuple(spans)


def build_model(case: str) -> ClipTextModel:
    model = ClipTextModel(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def build_encoder(case: str) -> ClipTextEncoder:
    spec = GOLDENS["cases"][case]
    config = case_config(case)

    def lookup(name: str) -> torch.Tensor | None:
        rows = spec["embeddings"].get(name)
        if rows is None:
            return None
        return embedding_vectors(name, rows, config.hidden_size)

    return ClipTextEncoder(
        build_model(case),
        profile=case_profile(case),
        policy=case_policy(case),
        embeddings=lookup,
    )


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


def test_clip_injects_causal_kernel_without_changing_state_or_output() -> None:
    case = CASES[0]
    baseline = build_model(case)
    spy = CallableModuleKernel(select_attention("clip").kernel)
    model = ClipTextModel(case_config(case), attention_kernel=spy)
    model.load_state_dict(baseline.state_dict(), strict=True)
    embeds = baseline.embed_tokens(torch.tensor([[1, 2, 3]]))
    eos = torch.tensor([2])
    assert set(model.state_dict()) == set(baseline.state_dict())
    assert_kernel_is_not_model_state(model, spy)
    torch.testing.assert_close(model(embeds, eos).last_hidden, baseline(embeds, eos).last_hidden)
    assert len(spy.calls) == len(model.text_model.encoder.layers)
    assert all(call["mask"] is None and call["causal"] for call in spy.calls)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted(
        (key, list(shape)) for key, shape in clip_text_layout(case_config(case)).items()
    )
    assert predicted == golden_entries(case)


@pytest.mark.parametrize(
    ("name", "config"),
    [("clip_l", CLIP_L_TEXT_CONFIG), ("clip_g", CLIP_G_TEXT_CONFIG)],
)
def test_full_size_module_matches_reference_layout(name: str, config: ClipTextConfig) -> None:
    """The real CLIP-L/G architectures, constructed on the meta
    device (initless factories never touch the storage), against the
    reference model's own full-size listing."""
    with torch.device("meta"):
        model = ClipTextModel(config)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert ours == golden


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_encode_matches_executed_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    got = build_encoder(case).encode_chunks(case_chunks(case))
    assert got.embeddings.dtype == torch.float32
    assert got.pooled is not None
    assert got.pooled.dtype == torch.float32
    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got.pooled, dec(spec["pooled"]), rtol=1e-4, atol=1e-5)


def test_sdxl_composition_matches_executed_reference() -> None:
    spec = GOLDENS["sdxl"]
    got = compose_sdxl_conditioning(
        build_encoder(spec["l_case"]).encode_chunks(case_chunks(spec["l_case"])),
        build_encoder(spec["g_case"]).encode_chunks(case_chunks(spec["g_case"])),
    )
    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=1e-4, atol=1e-5)
    assert got.pooled is not None
    torch.testing.assert_close(got.pooled, dec(spec["pooled"]), rtol=1e-4, atol=1e-5)


def test_sd1_hidden_layer_override_matches_executed_reference() -> None:
    case = "l_clip_skip_2"
    spec = GOLDENS["cases"][case]
    encoder = ClipTextEncoder(
        build_model(case),
        profile=case_profile(case),
        policy=SD1_CLIP_L_POLICY,
    )

    got = encoder.encode(case_spans(case), hidden_layer=-2)

    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=1e-4, atol=1e-5)
    assert got.pooled is not None
    torch.testing.assert_close(got.pooled, dec(spec["pooled"]), rtol=1e-4, atol=1e-5)
    assert encoder.policy is SD1_CLIP_L_POLICY


def test_sdxl_hidden_layer_override_matches_both_executed_towers() -> None:
    spec = GOLDENS["sdxl_override"]
    l_case = spec["l_case"]
    g_case = spec["g_case"]
    clip_l = ClipTextEncoder(
        build_model(l_case),
        profile=case_profile(l_case),
        policy=SDXL_CLIP_POLICY,
    ).encode(case_spans(l_case), hidden_layer=spec["layer"])
    clip_g = ClipTextEncoder(
        build_model(g_case),
        profile=case_profile(g_case),
        policy=SDXL_CLIP_POLICY,
    ).encode(case_spans(g_case), hidden_layer=spec["layer"])

    got = compose_sdxl_conditioning(clip_l, clip_g)

    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=1e-4, atol=1e-5)
    assert got.pooled is not None
    torch.testing.assert_close(got.pooled, dec(spec["pooled"]), rtol=1e-4, atol=1e-5)


def test_sdxl_hidden_layer_override_clamps_shallower_tower_to_last() -> None:
    spec = GOLDENS["sdxl_mixed_depth_override"]
    l_case = spec["l_case"]
    g_case = spec["g_case"]
    clip_l = ClipTextEncoder(
        build_model(l_case),
        profile=case_profile(l_case),
        policy=SDXL_CLIP_POLICY,
    ).encode(case_spans(l_case), hidden_layer=spec["layer"])
    clip_g = ClipTextEncoder(
        build_model(g_case),
        profile=case_profile(g_case),
        policy=SDXL_CLIP_POLICY,
    ).encode(case_spans(g_case), hidden_layer=spec["layer"])

    got = compose_sdxl_conditioning(clip_l, clip_g)

    torch.testing.assert_close(got.embeddings, dec(spec["cond"]), rtol=1e-4, atol=1e-5)
    assert got.pooled is not None
    torch.testing.assert_close(got.pooled, dec(spec["pooled"]), rtol=1e-4, atol=1e-5)


def test_sd1_hidden_layer_override_beyond_tower_depth_uses_last() -> None:
    case = "l_unweighted"
    hidden_policy = replace(SD1_CLIP_L_POLICY, hidden_layer=-2)
    encoder = ClipTextEncoder(
        build_model(case),
        profile=case_profile(case),
        policy=hidden_policy,
    )
    last_encoder = ClipTextEncoder(
        build_model(case),
        profile=case_profile(case),
        policy=replace(hidden_policy, hidden_layer=None),
    )

    out_of_range = encoder.encode(case_spans(case), hidden_layer=-24)
    last = last_encoder.encode(case_spans(case))

    assert torch.equal(out_of_range.embeddings, last.embeddings)
    assert out_of_range.pooled is not None
    assert last.pooled is not None
    assert torch.equal(out_of_range.pooled, last.pooled)
    assert encoder.policy is hidden_policy


def test_sdxl_composition_cuts_to_the_shorter_tower() -> None:
    l_cond = torch.randn(1, 154, 8)
    g_cond = torch.randn(1, 77, 12)
    pooled = torch.randn(1, 12)
    got = compose_sdxl_conditioning(
        Conditioning(l_cond, torch.randn(1, 8)),
        Conditioning(g_cond, pooled),
    )
    assert got.embeddings.shape == (1, 77, 20)
    torch.testing.assert_close(got.embeddings[..., :8], l_cond[:, :77])
    torch.testing.assert_close(got.embeddings[..., 8:], g_cond)
    assert got.pooled is pooled


def test_exported_policies_are_the_golden_case_policies() -> None:
    """The shipped constants match the knobs the reference cases were
    executed with: SD1 reads the final layer + raw pooled, SDXL reads
    the un-normed penultimate layer + projected pooled."""
    assert case_policy("l_unweighted") == SD1_CLIP_L_POLICY
    assert case_policy("l_weighted") == SD1_CLIP_L_POLICY
    assert case_policy("sdxl_l") == SDXL_CLIP_POLICY
    assert case_policy("sdxl_g") == SDXL_CLIP_POLICY
    assert case_policy("g_hidden") == SDXL_CLIP_POLICY


# ------------------------------------------------- model semantics


def test_hidden_layer_negative_index_matches_positive() -> None:
    model = build_model("g_hidden")
    embeds = torch.randn(1, 77, model.config.hidden_size)
    eos = torch.tensor([3])
    layers = model.config.num_hidden_layers
    negative = model(embeds, eos, hidden_layer=-2)
    positive = model(embeds, eos, hidden_layer=layers - 2)
    assert negative.hidden is not None
    assert positive.hidden is not None
    torch.testing.assert_close(negative.hidden, positive.hidden)


def test_hidden_layer_out_of_range_refuses() -> None:
    model = build_model("l_unweighted")
    embeds = torch.randn(1, 77, model.config.hidden_size)
    eos = torch.tensor([3])
    with pytest.raises(ValueError, match="out of range"):
        model(embeds, eos, hidden_layer=model.config.num_hidden_layers)


def test_hidden_state_layer_norm_is_the_final_norm() -> None:
    model = build_model("l_unweighted")
    embeds = torch.randn(1, 77, model.config.hidden_size)
    eos = torch.tensor([3])
    normed = model(embeds, eos, hidden_layer=-2)
    raw = model(embeds, eos, hidden_layer=-2, layer_norm_hidden_state=False)
    assert normed.hidden is not None
    assert raw.hidden is not None
    torch.testing.assert_close(normed.hidden, model.text_model.final_layer_norm(raw.hidden))


def test_pooled_is_the_eos_position_of_the_last_hidden_state() -> None:
    model = build_model("l_unweighted")
    embeds = torch.randn(2, 77, model.config.hidden_size)
    eos = torch.tensor([5, 11])
    out = model(embeds, eos)
    torch.testing.assert_close(out.pooled[0], out.last_hidden[0, 5])
    torch.testing.assert_close(out.pooled[1], out.last_hidden[1, 11])
    torch.testing.assert_close(out.projected, model.text_projection(out.pooled))


def test_sequence_longer_than_position_table_refuses() -> None:
    model = build_model("l_unweighted")
    embeds = torch.randn(1, 78, model.config.hidden_size)
    with pytest.raises(ValueError, match="position"):
        model(embeds, torch.tensor([0]))


def test_attention_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        ClipAttention(64, 5)


def test_causal_attention_blocks_future_positions() -> None:
    """Changing a LATER position must not change an earlier one."""
    model = build_model("l_unweighted")
    base = torch.randn(1, 77, model.config.hidden_size)
    changed = base.clone()
    changed[0, 40] += 1.0
    eos = torch.tensor([3])
    torch.testing.assert_close(
        model(base, eos).last_hidden[0, :40],
        model(changed, eos).last_hidden[0, :40],
    )


# ------------------------------------------------- weighting policy


def test_span_weights_of_one_are_bit_exact() -> None:
    z = torch.randn(2, 5, 3)
    z_empty = torch.randn(5, 3)
    weights = torch.ones(2, 5, 1)
    assert torch.equal(apply_span_weights(z, weights, z_empty), z)


def test_span_weights_interpolate_against_the_empty_encoding() -> None:
    z = torch.randn(1, 4, 3)
    z_empty = torch.randn(4, 3)
    weights = torch.full((1, 4, 1), 0.25)
    torch.testing.assert_close(
        apply_span_weights(z, weights, z_empty),
        (z - z_empty) * 0.25 + z_empty,
    )


def test_unweighted_prompt_skips_the_empty_chunk_batch_row() -> None:
    """weight-1.0-everywhere encodes exactly like a bare model forward
    over the single chunk (the reference only appends the empty chunk
    to the batch when a weight deviates)."""
    encoder = build_encoder("l_unweighted")
    (chunk,) = case_chunks("l_unweighted")
    direct = encoder.encode_chunks((chunk,))
    assert all(isinstance(packed.unit, int) for packed in chunk)
    ids = torch.tensor(
        [[packed.unit for packed in chunk if isinstance(packed.unit, int)]],
        dtype=torch.long,
    )
    # BOS + [1000, 2000] + EOS: the first end token sits at index 3.
    out = encoder.model(encoder.model.embed_tokens(ids), torch.tensor([3]))
    torch.testing.assert_close(direct.embeddings, out.last_hidden.float())
    assert direct.pooled is not None
    torch.testing.assert_close(direct.pooled, out.pooled.float())


# ------------------------------------------------ spans and errors


def test_encode_spans_equals_packed_chunks() -> None:
    encoder = build_encoder("l_weighted")
    spans = (
        WeightedSpan((1000,), 1.0),
        WeightedSpan((2000, 3000), 1.2),
        WeightedSpan((4000,), 0.8),
    )
    via_spans = encoder.encode(spans)
    via_chunks = encoder.encode_chunks(pack_spans(spans, encoder.profile))
    assert torch.equal(via_spans.embeddings, via_chunks.embeddings)
    assert via_spans.pooled is not None
    assert via_chunks.pooled is not None
    assert torch.equal(via_spans.pooled, via_chunks.pooled)


def test_left_padded_profile_refuses() -> None:
    model = build_model("l_unweighted")
    with pytest.raises(ClipEncodeError, match="left-padded"):
        ClipTextEncoder(model, profile=TokenizerProfile(pad_left=True))


def test_empty_chunk_list_refuses() -> None:
    with pytest.raises(ValueError, match="no chunks"):
        build_encoder("l_unweighted").encode_chunks(())


def test_mismatched_chunk_lengths_refuse() -> None:
    chunks = case_chunks("l_weighted")
    with pytest.raises(ValueError, match="one length"):
        build_encoder("l_weighted").encode_chunks((chunks[0], chunks[1][:-1]))


def test_unresolvable_embedding_refuses() -> None:
    chunks = (
        (
            PackedToken(49406, 1.0, 0),
            PackedToken(EmbeddingSlot("ghost", 0), 1.0, 1),
            PackedToken(49407, 1.0, 0),
        ),
    )
    with pytest.raises(ClipEncodeError, match="ghost"):
        build_encoder("l_unweighted").encode_chunks(chunks)


def test_mis_shaped_embedding_refuses_instead_of_dropping() -> None:
    """The reference warns and DROPS a width-mismatched embedding
    (comfy/sd1_clip.py process_tokens); Dinkster refuses loudly - a
    deliberate divergence, silent conditioning drift is worse."""
    model = build_model("l_unweighted")
    encoder = ClipTextEncoder(
        model,
        embeddings=lambda name: torch.zeros(2, 999),
    )
    chunks = (
        (
            PackedToken(49406, 1.0, 0),
            PackedToken(EmbeddingSlot("wide", 0), 1.0, 1),
            PackedToken(49407, 1.0, 0),
        ),
    )
    with pytest.raises(ClipEncodeError, match="wide"):
        encoder.encode_chunks(chunks)


# ------------------------------------------------------- training


def test_gradients_flow_to_every_parameter_kind() -> None:
    """No inference_mode/no_grad anywhere: the encode path must be
    trainable end to end (docs/native-inference-plan.md 3.1)."""
    encoder = build_encoder("l_embedding")
    out = encoder.encode_chunks(case_chunks("l_embedding"))
    assert out.pooled is not None
    loss = out.embeddings.square().mean() + out.pooled.square().mean()
    loss.backward()
    model = encoder.model
    for name, parameter in model.named_parameters():
        if name == "text_projection.weight":
            # SD1 policy reads the raw pooled output; the projection
            # legitimately receives no gradient.
            assert parameter.grad is None
            continue
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
