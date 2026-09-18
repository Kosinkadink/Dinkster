"""Native Anima Qwen3-0.6B text tower math goldens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from clip_fill import fill_state_dict
from dinkster_inference import ANIMA_QWEN3_06B_CONFIG, QwenTextConfig, qwen_text_layout
from dinkster_inference_torch import CastOperations, QwenTextModel

GOLDENS = json.loads(
    (Path(__file__).parents[3] / "tests" / "goldens" / "anima_text_goldens.json").read_text()
)


def dec(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def build_tiny() -> QwenTextModel:
    model = QwenTextModel(QwenTextConfig(**GOLDENS["model"]["config"]))
    model.load_state_dict(fill_state_dict(GOLDENS["model"]["state_dict"]), strict=True)
    return model


def test_tiny_state_layout_matches_executed_reference() -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_tiny().state_dict().items())
    assert ours == [(key, list(shape)) for key, shape in GOLDENS["model"]["state_dict"]]


def test_model_math_matches_executed_reference() -> None:
    spec = GOLDENS["model"]
    ids = torch.tensor(spec["ids"], dtype=torch.long)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long)
    with torch.no_grad():
        got = build_tiny()(ids, mask)
    torch.testing.assert_close(got, dec(spec["output"]), rtol=1e-5, atol=1e-6)


def test_causal_prefill_and_incremental_decode_match_full_forward() -> None:
    model = build_tiny()
    ids = torch.tensor([[3, 5, 7, 11, 13]], dtype=torch.long)
    config = model.config
    key = torch.empty(
        config.num_hidden_layers,
        1,
        config.num_key_value_heads,
        ids.shape[1],
        config.head_dim,
    )
    value = torch.empty_like(key)
    cache = tuple((key[index], value[index]) for index in range(config.num_hidden_layers))

    with torch.no_grad():
        full = model(ids)
        prefill, prefill_kv = model.forward_causal(ids)
        first, first_kv = model.forward_causal(ids[:, :3], cache)
        second, second_kv = model.forward_causal(ids[:, 3:], cache, cache_position=3)
        frequency_cache = model.causal_frequencies(ids.shape[1], device=ids.device)
        cached_prefill, _ = model.forward_causal(ids, frequencies=frequency_cache)
        cached_second, _ = model.forward_causal(
            ids[:, 3:],
            cache,
            cache_position=3,
            frequencies=(
                frequency_cache[0][:, :, 3:],
                frequency_cache[1][:, :, 3:],
                frequency_cache[2][:, :, 3:],
            ),
        )

    torch.testing.assert_close(prefill, full)
    torch.testing.assert_close(cached_prefill, prefill, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first, full[:, :3])
    torch.testing.assert_close(second, full[:, 3:])
    torch.testing.assert_close(cached_second, second, rtol=0.0, atol=0.0)
    for layer in range(config.num_hidden_layers):
        torch.testing.assert_close(first_kv[layer][0], prefill_kv[layer][0][:, :, :3])
        torch.testing.assert_close(first_kv[layer][1], prefill_kv[layer][1][:, :, :3])
        torch.testing.assert_close(second_kv[layer][0], prefill_kv[layer][0][:, :, 3:])
        torch.testing.assert_close(second_kv[layer][1], prefill_kv[layer][1][:, :, 3:])
        torch.testing.assert_close(cache[layer][0], prefill_kv[layer][0])
        torch.testing.assert_close(cache[layer][1], prefill_kv[layer][1])


def test_causal_path_checks_cache_shape_and_total_position_limit() -> None:
    model = build_tiny()
    config = model.config
    key = torch.empty(1, config.num_key_value_heads, 2, config.head_dim)
    cache = tuple((key, key.clone()) for _ in range(config.num_hidden_layers))

    with pytest.raises(ValueError, match="cache position requires"):
        model.forward_causal(torch.tensor([[1]]), cache_position=1)
    with pytest.raises(ValueError, match="capacity"):
        model.forward_causal(torch.tensor([[1, 2, 3]]), cache)
    half_cache = tuple((key.half(), key.half()) for _ in range(config.num_hidden_layers))
    with pytest.raises(ValueError, match="execution dtype"):
        model.forward_causal(torch.tensor([[1]]), half_cache)
    key = torch.empty(1, config.num_key_value_heads, 33, config.head_dim)
    cache = tuple((key, key.clone()) for _ in range(config.num_hidden_layers))
    with pytest.raises(ValueError, match="maximum is 32"):
        model.forward_causal(torch.tensor([[1, 2]]), cache, cache_position=31)
    with pytest.raises(ValueError, match="causal frequencies"):
        model.forward_causal(
            torch.tensor([[1]]),
            frequencies=(torch.empty(1), torch.empty(1), torch.empty(1)),
        )


def test_tied_language_head_matches_explicit_projection() -> None:
    model = build_tiny()
    ids = torch.tensor([[3, 5, 7]], dtype=torch.long)
    with torch.no_grad():
        hidden, _ = model.forward_causal(ids)
        got = model.logits(hidden)
        expected = torch.nn.functional.linear(hidden, model.embed_tokens.weight)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_tied_language_head_materializes_storage_at_the_bound_compute_dtype() -> None:
    source = build_tiny()
    state = {key: value.half() for key, value in source.state_dict().items()}
    with torch.device("meta"):
        model = QwenTextModel(
            source.config,
            operations=CastOperations(torch.float32),
        )
    model.load_state_dict(state, strict=True, assign=True)
    hidden = torch.randn(1, 2, source.config.hidden_size)

    with torch.no_grad():
        got = model.logits(hidden)
        expected = torch.nn.functional.linear(hidden, state["embed_tokens.weight"].float())

    assert model.embed_tokens.weight.dtype is torch.float16
    assert got.dtype is torch.float32
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_full_state_layout_and_strict_load_match_executed_reference() -> None:
    with torch.device("meta"):
        model = QwenTextModel(ANIMA_QWEN3_06B_CONFIG)
    expected = [(key, tuple(shape)) for key, shape in GOLDENS["layout"]]
    assert sorted(qwen_text_layout(ANIMA_QWEN3_06B_CONFIG).items()) == expected
    assert (
        sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == expected
    )
    state = {key: torch.empty(shape, device="meta") for key, shape in expected}
    model.load_state_dict(state, strict=True, assign=True)
    del state["layers.27.self_attn.q_norm.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True, assign=True)
