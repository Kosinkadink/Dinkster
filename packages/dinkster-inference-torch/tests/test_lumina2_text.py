"""Lumina2 Gemma 2 text math and weighted conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from attention_spy import CallableModuleKernel
from clip_fill import fill_state_dict
from dinkster_inference import (
    GEMMA2_LUMINA_2B_CONFIG,
    GemmaTextConfig,
    Lumina2PromptTokens,
    gemma_text_layout,
)
from dinkster_inference_torch import (
    CastOperations,
    GemmaTextModel,
    Lumina2TextRuntime,
    select_attention,
)
from dinkster_inference_torch import gemma_text as gemma_text_mod
from dinkster_inference_torch import lumina2_runtime as runtime_mod
from dinkster_inference_torch.lumina2_component import LUMINA2_TOKENIZER_ATTRIBUTE
from golden_files import assert_reference_tensor, load_platform_golden

GOLDENS = load_platform_golden(
    Path(__file__).parent / "goldens" / "lumina2_text_goldens.json",
    allow_portable_fallback=True,
)
CASES = sorted(GOLDENS["cases"])


def decode(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def tiny_config(case: str) -> GemmaTextConfig:
    spec = GOLDENS["cases"][case]["config"]
    return GemmaTextConfig(
        architecture="gemma2_lumina_2b",
        vocab_size=spec["vocab_size"],
        hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_hidden_layers=spec["num_hidden_layers"],
        num_attention_heads=spec["num_attention_heads"],
        num_key_value_heads=spec["num_key_value_heads"],
        head_dim=spec["head_dim"],
        rms_norm_eps=spec["rms_norm_eps"],
        rope_theta_global=spec["rope_theta"],
        rope_theta_local=spec["rope_theta"],
        rope_scale_global=1.0,
        rope_scale_local=1.0,
        sliding_window=spec["max_position_embeddings"],
        sliding_pattern=(False,),
        prompt_template="{}",
        min_tokens=1,
        pad_token_id=0,
        bos_token_id=2,
        end_of_turn_token_id=107,
        image_soft_token_id=None,
    )


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def build_model(case: str) -> GemmaTextModel:
    model = GemmaTextModel(tiny_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


@pytest.mark.parametrize("case", CASES)
def test_gemma2_stack_matches_executed_reference(case: str) -> None:
    spec = GOLDENS["cases"][case]
    ids = torch.tensor(spec["ids"], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        stack = build_model(case)(ids, mask)
    assert_reference_tensor(stack, decode(spec["stack"]), rtol=1e-5, atol=1e-6)
    assert_reference_tensor(stack[:, -1], decode(spec["final"]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("case", CASES)
def test_tiny_gemma2_layout_matches_reference(case: str) -> None:
    model = GemmaTextModel(tiny_config(case))
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert ours == golden_entries(case)


def test_full_gemma2_layout_matches_reference_and_detector() -> None:
    with torch.device("meta"):
        model = GemmaTextModel(GEMMA2_LUMINA_2B_CONFIG)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    golden = [(key, list(shape)) for key, shape in GOLDENS["layout"]]
    predicted = sorted(
        (key, list(shape)) for key, shape in gemma_text_layout(GEMMA2_LUMINA_2B_CONFIG).items()
    )
    assert ours == golden == predicted
    assert len(ours) == 288
    assert not any(key.endswith(("q_norm.weight", "k_norm.weight")) for key, _ in ours)


def test_gemma2_preserves_grouped_heads_for_the_attention_kernel() -> None:
    case = CASES[0]
    spec = GOLDENS["cases"][case]
    config = tiny_config(case)
    spy = CallableModuleKernel(select_attention("qwen").kernel)
    model = GemmaTextModel(config, attention_kernel=spy)
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    ids = torch.tensor(spec["ids"], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        model(ids, mask)
    assert len(spy.calls) == config.num_hidden_layers
    assert all(
        call["q_shape"][1] == config.num_attention_heads
        and call["k_shape"][1] == call["v_shape"][1] == config.num_key_value_heads
        and call["enable_gqa"]
        for call in spy.calls
    )


def test_gemma_rms_norm_shifts_storage_before_compute_cast() -> None:
    norm = CastOperations(torch.float32).rms_norm(4, eps=1e-6).to(torch.float16)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor((0.0003, 0.001, 0.1234, -0.1234)))
    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 10.0
    expected_weight = (norm.weight + 1.0).float()
    wrong_weight = norm.weight.float() + 1.0
    with torch.no_grad():
        actual = gemma_text_mod._rms_norm(  # pyright: ignore[reportPrivateUsage]
            norm,
            hidden,
            add_weight=True,
        )
    expected = torch.nn.functional.rms_norm(hidden, (4,), expected_weight, 1e-6)
    wrong = torch.nn.functional.rms_norm(hidden, (4,), wrong_weight, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(actual, wrong)


class _FakeTokenizer:
    def __init__(self, model: bytes, *, expected_vocab_size: int) -> None:
        assert model == b"sentencepiece"
        assert expected_vocab_size == 256000

    def encode(self, text: str) -> tuple[int, ...]:
        del text
        return (5, 6)


class _RecordingGemma(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = GEMMA2_LUMINA_2B_CONFIG
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.__dict__[LUMINA2_TOKENIZER_ATTRIBUTE] = b"sentencepiece"

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.calls.append((ids.clone(), mask.clone()))
        values = ids.float().unsqueeze(-1).expand(-1, -1, 2304)
        return torch.stack((torch.zeros_like(values), values, values * 2.0), dim=1)


def test_weighted_runtime_uses_equal_length_empty_prompt_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _RecordingGemma()
    monkeypatch.setattr(runtime_mod, "GemmaSentencePieceTokenizer", _FakeTokenizer)

    def tokenize(*_args: object, **_kwargs: object) -> Lumina2PromptTokens:
        return Lumina2PromptTokens((2, 5, 6), (1, 1, 1), (1.0, 0.5, 2.0))

    monkeypatch.setattr(runtime_mod, "tokenize_lumina2_prompt", tokenize)
    runtime = Lumina2TextRuntime(cast("Any", model))
    condition = runtime.encode_text("weighted prompt")
    assert len(model.calls) == 2
    assert model.calls[0][0].tolist() == [[2, 5, 6]]
    assert model.calls[1][0].tolist() == [[2, 0, 0]]
    assert model.calls[1][1].tolist() == [[1, 0, 0]]
    assert condition.pooled is None
    expected = torch.tensor((2.0, 2.5, 12.0)).reshape(1, 3, 1).expand(1, 3, 2304)
    torch.testing.assert_close(condition.embeddings, expected)


def test_text_runtime_requires_exact_profile_and_tokenizer() -> None:
    model = _RecordingGemma()
    del model.__dict__[LUMINA2_TOKENIZER_ATTRIBUTE]
    with pytest.raises(ValueError, match="no SentencePiece model"):
        Lumina2TextRuntime(cast("Any", model))
    model.__dict__[LUMINA2_TOKENIZER_ATTRIBUTE] = b"sentencepiece"
    model.config = tiny_config(CASES[0])
    with pytest.raises(ValueError, match="Gemma 2 2B profile"):
        Lumina2TextRuntime(cast("Any", model))
