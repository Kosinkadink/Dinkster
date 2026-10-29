"""Native Gemma 3 text math and LTX-2 encode-policy goldens."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from clip_fill import fill_state_dict, fill_value
from dinkster_inference import (
    GEMMA3_LTX_12B_CONFIG,
    Conditioning,
    GemmaTextConfig,
    LtxGemmaPromptTokens,
    gemma_text_layout,
    tokenize_ltx_gemma_prompt,
)
from dinkster_inference_torch import (
    GemmaTextModel,
    LtxDualTextProjection,
    LtxGemmaTextEncoder,
    select_attention,
)
from dinkster_inference_torch.gemma_text import (
    _apply_rope,  # pyright: ignore[reportPrivateUsage]
    _rms_norm,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.operations import INITLESS
from dinkster_inference_torch.quant_linear import Int8Linear
from golden_files import assert_reference_tensor, load_platform_golden

GOLDENS = load_platform_golden(
    Path(__file__).parents[3] / "tests" / "goldens" / "ltx_gemma_text_goldens.json"
)


def dec(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def tiny_config() -> GemmaTextConfig:
    spec = dict(GOLDENS["model"]["config"])
    spec["sliding_pattern"] = tuple(spec["sliding_pattern"])
    return GemmaTextConfig(**spec)


def build_tiny() -> GemmaTextModel:
    model = GemmaTextModel(tiny_config())
    model.load_state_dict(fill_state_dict(GOLDENS["model"]["state_dict"]), strict=True)
    return model


def test_tiny_state_layout_matches_executed_reference() -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_tiny().state_dict().items())
    assert ours == [(key, list(shape)) for key, shape in GOLDENS["model"]["state_dict"]]


def test_full_state_layout_and_strict_load_match_executed_reference() -> None:
    with torch.device("meta"):
        model = GemmaTextModel(GEMMA3_LTX_12B_CONFIG)
    expected = [(key, tuple(shape)) for key, shape in GOLDENS["layout"]["model"]]
    assert sorted(gemma_text_layout(GEMMA3_LTX_12B_CONFIG).items()) == expected
    assert (
        sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == expected
    )
    state = {key: torch.empty(shape, device="meta") for key, shape in expected}
    model.load_state_dict(state, strict=True, assign=True)
    del state["layers.47.self_attn.q_norm.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True, assign=True)


@pytest.mark.parametrize("case", range(len(GOLDENS["model"]["cases"])))
def test_model_stack_matches_executed_reference(case: int) -> None:
    spec = GOLDENS["model"]["cases"][case]
    ids = torch.tensor(spec["ids"], dtype=torch.long)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long)
    with torch.no_grad():
        got = build_tiny()(ids, mask)
    assert_reference_tensor(got, dec(spec["stack"]), rtol=1e-5, atol=1e-6)


def test_gemma_injects_masked_noncausal_kernel_without_state_drift() -> None:
    baseline = build_tiny()
    spy = CallableModuleKernel(select_attention("qwen").kernel)
    model = GemmaTextModel(tiny_config(), attention_kernel=spy)
    model.load_state_dict(baseline.state_dict(), strict=True)
    spec = GOLDENS["model"]["cases"][0]
    ids = torch.tensor(spec["ids"], dtype=torch.long)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long)
    assert set(model.state_dict()) == set(baseline.state_dict())
    assert_kernel_is_not_model_state(model, spy)
    torch.testing.assert_close(model(ids, mask), baseline(ids, mask))
    assert len(spy.calls) == tiny_config().num_hidden_layers
    assert all(
        call["mask"] is not None
        and not call["causal"]
        and call["q_shape"][1] == call["k_shape"][1] == call["v_shape"][1]
        for call in spy.calls
    )


def test_model_validates_ids_and_attention_mask_shapes() -> None:
    model = build_tiny()
    with pytest.raises(ValueError, match="ids must be"):
        model(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError, match="attention mask must match"):
        model(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1]]))


def test_gemma4_rope_matches_reference_multiply_add_order() -> None:
    query = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 8).reshape(2, 3, 4, 8)
    key = query[:, :2] * 1.7
    angle = torch.linspace(-0.5, 0.5, 4 * 4).reshape(1, 1, 4, 4)
    cosine = angle.cos()
    sine = angle.sin()
    frequencies = torch.stack(
        (torch.stack((cosine, -sine), dim=-1), torch.stack((sine, cosine), dim=-1)),
        dim=-2,
    )

    def reference(hidden: torch.Tensor) -> torch.Tensor:
        pairs = hidden.reshape(*hidden.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        output = frequencies[..., 0] * pairs[..., 0] + frequencies[..., 1] * pairs[..., 1]
        return output.movedim(-1, -2).reshape(hidden.shape)

    rotated_query, rotated_key = _apply_rope(query, key, frequencies)
    assert torch.equal(rotated_query, reference(query))
    assert torch.equal(rotated_key, reference(key))


@pytest.mark.parametrize(
    ("architecture", "rms_norm_add"),
    (("gemma3_ltx_12b", True), ("gemma4_ltx_12b", False)),
)
def test_gemma_casts_bfloat16_embeddings_to_float32_before_stack(
    architecture: str, rms_norm_add: bool
) -> None:
    config = replace(
        tiny_config(),
        architecture=architecture,
        num_hidden_layers=1,
        rms_norm_add=rms_norm_add,
    )
    model = GemmaTextModel(config).to(torch.bfloat16)

    class Passthrough(torch.nn.Module):
        def forward(self, hidden: torch.Tensor, *_args: object) -> torch.Tensor:
            return hidden

    model.layers = torch.nn.ModuleList([Passthrough()])
    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            torch.arange(config.vocab_size * config.hidden_size).reshape(
                config.vocab_size, config.hidden_size
            )
            / 101
        )
        model.norm.weight.copy_(torch.linspace(0.5, 1.5, config.hidden_size))
    ids = torch.tensor([[1, 3]])
    embedded = torch.nn.functional.embedding(ids, model.embed_tokens.weight).float()
    hidden = embedded * math.sqrt(config.hidden_size)
    norm_weight = model.norm.weight + 1.0 if rms_norm_add else model.norm.weight
    expected = torch.nn.functional.rms_norm(
        hidden,
        (config.hidden_size,),
        norm_weight.float(),
        model.norm.eps,
    )
    expected = torch.stack((hidden, expected), dim=1)

    with torch.no_grad():
        got = model(ids)

    assert got.dtype == torch.float32
    assert torch.equal(got, expected)


def test_added_rms_norm_shifts_before_compute_dtype_cast() -> None:
    module = INITLESS.rms_norm(2, eps=1e-6).to(torch.float16)
    hidden = torch.tensor([[0.75, -1.25]], dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([0.0006, -0.0006]))
        expected_weight = (module.weight + 1.0).float()
        expected = torch.nn.functional.rms_norm(hidden, (2,), expected_weight, module.eps)
        actual = _rms_norm(module, hidden, add_weight=True)

    assert torch.equal(actual, expected)


def _single_projection() -> torch.nn.Linear:
    ((key, shape),) = GOLDENS["encoder"]["projection_fill"]["single"]
    projection = torch.nn.Linear(shape[1], shape[0], bias=False)
    with torch.no_grad():
        projection.weight.copy_(fill_value(key, shape))
    return projection


def _dual_projection() -> LtxDualTextProjection:
    entries = GOLDENS["encoder"]["projection_fill"]["dual"]
    shapes = {key.removeprefix("text_embedding_projection."): shape for key, shape in entries}
    projection = LtxDualTextProjection(
        in_features=shapes["video_aggregate_embed.weight"][1],
        video_features=shapes["video_aggregate_embed.weight"][0],
        audio_features=shapes["audio_aggregate_embed.weight"][0],
    )
    projection.load_state_dict(
        {
            key.removeprefix("text_embedding_projection."): fill_value(key, shape)
            for key, shape in entries
        },
        strict=True,
    )
    return projection


class _StackModel:
    """Returns a pinned reference stack; records what the encoder sent."""

    def __init__(self, stack: torch.Tensor) -> None:
        self.config = tiny_config()
        self.embed_tokens = type("Embedding", (), {"weight": torch.empty(0)})()
        self.stack = stack
        self.ids: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None

    def __call__(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert attention_mask is not None
        self.ids = ids
        self.mask = attention_mask
        return self.stack.clone()


def _golden_prompt_tokens() -> LtxGemmaPromptTokens:
    spec = GOLDENS["model"]["cases"][0]
    return LtxGemmaPromptTokens(
        tuple(spec["ids"][0]),
        tuple(spec["attention_mask"][0]),
        (0,) * len(spec["ids"][0]),
    )


@pytest.mark.parametrize("kind", ("single_linear", "dual_linear"))
def test_encoder_projection_matches_executed_reference(kind: str) -> None:
    model = _StackModel(dec(GOLDENS["model"]["cases"][0]["stack"]))
    projection = _single_projection() if kind == "single_linear" else _dual_projection()
    encoder = LtxGemmaTextEncoder(cast("Any", model), projection, cast("Any", None))
    tokens = _golden_prompt_tokens()
    got: Conditioning[torch.Tensor] = encoder.encode_tokens(tokens)
    assert model.ids is not None and model.mask is not None
    assert model.ids.tolist() == [list(tokens.ids)]
    assert model.mask.tolist() == [list(tokens.attention_mask)]
    assert got.pooled is None
    assert got.embeddings.shape[1] == GOLDENS["encoder"]["attended_tokens"]
    torch.testing.assert_close(got.embeddings, dec(GOLDENS["encoder"][kind]), rtol=1e-5, atol=1e-6)


def test_encoder_casts_the_stack_to_the_projection_dtype() -> None:
    model = _StackModel(dec(GOLDENS["model"]["cases"][0]["stack"]))
    projection = _single_projection().to(torch.bfloat16)
    encoder = LtxGemmaTextEncoder(cast("Any", model), projection, cast("Any", None))
    got: Conditioning[torch.Tensor] = encoder.encode_tokens(_golden_prompt_tokens())
    assert got.embeddings.dtype == torch.float32
    assert torch.isfinite(got.embeddings).all()

    # The cast happens before the fold and range normalization, as in the
    # reference; replicate that order on the pinned stack in bfloat16.
    attended = GOLDENS["encoder"]["attended_tokens"]
    stack = dec(GOLDENS["model"]["cases"][0]["stack"])
    folded = stack[:, :, stack.shape[2] - attended :].to(torch.bfloat16).movedim(1, -1)
    folded = (
        8.0
        * (folded - folded.mean(dim=(1, 2), keepdim=True))
        / (folded.amax(dim=(1, 2), keepdim=True) - folded.amin(dim=(1, 2), keepdim=True) + 1e-6)
    )
    folded = folded.reshape(folded.shape[0], folded.shape[1], -1)
    expected = torch.nn.functional.linear(folded, projection.weight).float()
    assert torch.equal(got.embeddings, expected)


def test_encoder_uses_int8_projection_compute_dtype() -> None:
    model = _StackModel(dec(GOLDENS["model"]["cases"][0]["stack"]))
    projection = _dual_projection()
    video = projection.video_aggregate_embed
    cast("Any", projection).video_aggregate_embed = Int8Linear(
        video.in_features,
        video.out_features,
        bias=False,
        compute_dtype=torch.bfloat16,
        convrot=False,
        convrot_groupsize=0,
    )
    encoder = LtxGemmaTextEncoder(cast("Any", model), projection, cast("Any", None))
    assert encoder._projection_dtype() == torch.bfloat16  # pyright: ignore[reportPrivateUsage]


def _fake_encode(_text: str) -> list[int]:
    return [7]


def test_encoder_encode_tokenizes_with_the_ltx_policy() -> None:
    model = _StackModel(dec(GOLDENS["model"]["cases"][0]["stack"]))
    encoder = LtxGemmaTextEncoder(
        cast("Any", model),
        _single_projection(),
        cast("Any", type("Tok", (), {"encode": staticmethod(_fake_encode)})()),
    )
    got = encoder.encode("cat")
    expected = tokenize_ltx_gemma_prompt("cat", encode=_fake_encode)
    assert model.ids is not None
    assert model.ids.tolist() == [list(expected.ids)]
    assert got.embeddings.shape[1] == sum(expected.attention_mask)


def test_encoder_refuses_other_gemma_profiles() -> None:
    model = _StackModel(dec(GOLDENS["model"]["cases"][0]["stack"]))
    model.config = replace(tiny_config(), architecture="other_gemma")
    with pytest.raises(ValueError, match="requires the gemma3_ltx_12b"):
        LtxGemmaTextEncoder(cast("Any", model), _single_projection(), cast("Any", None))
