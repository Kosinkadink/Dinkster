"""Flux2 text-tower math and stacked-conditioning goldens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from clip_fill import fill_state_dict
from dinkster_inference import (
    KLEIN_QWEN3_4B_CONFIG,
    KLEIN_QWEN3_8B_CONFIG,
    MISTRAL3_24B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
    Conditioning,
    QwenTextConfig,
    load_flux2_tekken_bpe,
    qwen_text_layout,
    tokenize_flux2_dev_prompt,
    tokenize_flux2_klein_prompt,
)
from dinkster_inference_torch import Flux2DevTextEncoder, Flux2KleinTextEncoder, QwenTextModel

GOLDENS = json.loads(
    (Path(__file__).parents[3] / "tests" / "goldens" / "flux2_text_goldens.json").read_text()
)

TINY_CASES = ("mistral3_full", "mistral3_pruned", "klein")

FULL_CONFIGS = (
    MISTRAL3_24B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
    KLEIN_QWEN3_4B_CONFIG,
    KLEIN_QWEN3_8B_CONFIG,
)


def dec(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def tiny_config(name: str) -> QwenTextConfig:
    spec = dict(GOLDENS["models"][name]["config"])
    spec["output_hidden_layers"] = tuple(spec["output_hidden_layers"])
    return QwenTextConfig(**spec)


def build_tiny(name: str) -> QwenTextModel:
    # Filled over the model's OWN entries: the fill is per-key
    # deterministic, so shared keys carry the reference values and the
    # pruned tower's never-applied norm.weight fills independently.
    model = QwenTextModel(tiny_config(name))
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


@pytest.mark.parametrize("name", TINY_CASES)
def test_tiny_captures_match_executed_reference(name: str) -> None:
    spec = GOLDENS["models"][name]
    ids = torch.tensor(spec["ids"], dtype=torch.long)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long)
    with torch.no_grad():
        got = build_tiny(name)(ids, mask)
    torch.testing.assert_close(got, dec(spec["captures"]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", TINY_CASES)
def test_tiny_layout_matches_reference_plus_unused_final_norm(name: str) -> None:
    config = tiny_config(name)
    model = QwenTextModel(config)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    reference = [(key, list(shape)) for key, shape in GOLDENS["models"][name]["state_dict"]]
    if config.final_norm:
        assert ours == reference
    else:
        norm_entry = ("norm.weight", [config.hidden_size])
        assert norm_entry not in reference
        assert [entry for entry in ours if entry != norm_entry] == reference
        assert norm_entry in ours


@pytest.mark.parametrize("config", FULL_CONFIGS, ids=lambda config: config.architecture)
def test_full_size_layout_matches_qwen_text_layout(config: QwenTextConfig) -> None:
    with torch.device("meta"):
        model = QwenTextModel(config)
    assert sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == sorted(
        qwen_text_layout(config).items()
    )


@pytest.mark.parametrize(
    "config",
    (MISTRAL3_24B_CONFIG, MISTRAL3_24B_PRUNED_CONFIG),
    ids=lambda config: config.architecture,
)
def test_mistral_profiles_refuse_sequences_past_8192(config: QwenTextConfig) -> None:
    assert config.max_position_embeddings == 8192
    with torch.device("meta"):
        model = QwenTextModel(config)
    ids = torch.empty((1, 8193), dtype=torch.long, device="meta")
    with pytest.raises(ValueError, match=rf"{config.architecture} received 8193 tokens"):
        model(ids)


class _RecordingStackModel:
    """Fake tower returning a distinguishable (batch, captures, tokens, hidden) stack."""

    hidden = 2

    def __init__(self, config: QwenTextConfig) -> None:
        self.config = config
        self.embed_tokens = type("Embedding", (), {"weight": torch.empty(0)})()
        self.ids: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    def __call__(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert attention_mask is not None
        self.ids = ids
        self.mask = attention_mask
        captures = len(self.config.output_hidden_layers or ())
        count = captures * ids.shape[1] * self.hidden
        self.output = torch.arange(count, dtype=torch.float32).reshape(
            1, captures, ids.shape[1], self.hidden
        )
        return self.output


def assert_folds_captures_into_width(
    model: _RecordingStackModel, got: Conditioning[torch.Tensor]
) -> None:
    assert model.output is not None
    captures = model.output.shape[1]
    tokens = model.output.shape[2]
    assert got.pooled is None
    assert got.embeddings.shape == (1, tokens, captures * model.hidden)
    for capture in range(captures):
        torch.testing.assert_close(
            got.embeddings[0, :, capture * model.hidden : (capture + 1) * model.hidden],
            model.output[0, capture],
        )


@pytest.mark.parametrize(
    "config",
    (MISTRAL3_24B_CONFIG, MISTRAL3_24B_PRUNED_CONFIG),
    ids=lambda config: config.architecture,
)
def test_dev_encoder_forwards_tekken_tokens_and_folds_captures(config: QwenTextConfig) -> None:
    model = _RecordingStackModel(config)
    tokenizer = load_flux2_tekken_bpe()
    encoder = Flux2DevTextEncoder(cast("Any", model), tokenizer)
    got: Conditioning[torch.Tensor] = encoder.encode("Hello, world!")
    expected = tokenize_flux2_dev_prompt("Hello, world!", tokenizer=tokenizer)
    assert model.ids is not None and model.mask is not None
    assert model.ids.tolist() == [list(expected.ids)]
    assert model.mask.tolist() == [list(expected.attention_mask)]
    assert_folds_captures_into_width(model, got)


@pytest.mark.parametrize(
    "config",
    (KLEIN_QWEN3_4B_CONFIG, KLEIN_QWEN3_8B_CONFIG),
    ids=lambda config: config.architecture,
)
def test_klein_encoder_forwards_padded_tokens_and_folds_captures(config: QwenTextConfig) -> None:
    model = _RecordingStackModel(config)
    encoder = Flux2KleinTextEncoder(cast("Any", model))
    got: Conditioning[torch.Tensor] = encoder.encode("cat")
    expected = tokenize_flux2_klein_prompt("cat")
    assert model.ids is not None and model.mask is not None
    assert model.ids.tolist() == [list(expected.ids)]
    assert model.mask.tolist() == [list(expected.attention_mask)]
    assert_folds_captures_into_width(model, got)


def test_dev_encoder_refuses_non_mistral_profiles() -> None:
    with pytest.raises(ValueError, match="requires a mistral3_24b"):
        Flux2DevTextEncoder(
            cast("Any", _RecordingStackModel(KLEIN_QWEN3_4B_CONFIG)), load_flux2_tekken_bpe()
        )


def test_klein_encoder_refuses_non_klein_profiles() -> None:
    with pytest.raises(ValueError, match="requires a Klein Qwen3"):
        Flux2KleinTextEncoder(cast("Any", _RecordingStackModel(MISTRAL3_24B_CONFIG)))
