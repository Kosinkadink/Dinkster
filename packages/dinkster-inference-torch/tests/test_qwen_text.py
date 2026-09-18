"""Native Ovis Qwen3-2B text math and encode-policy goldens."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from clip_fill import fill_state_dict
from dinkster_inference import (
    OVIS_QWEN3_2B_CONFIG,
    Z_IMAGE_QWEN3_4B_CONFIG,
    Conditioning,
    QwenTextConfig,
    qwen_text_layout,
    tokenize_ovis_prompt,
)
from dinkster_inference_torch import (
    OvisTextEncoder,
    QwenTextModel,
    ZImageTextEncoder,
    qwen_text,
    select_attention,
)
from dinkster_inference_torch.qwen_text import _apply_rope  # pyright: ignore[reportPrivateUsage]

GOLDENS = json.loads(
    (Path(__file__).parents[3] / "tests" / "goldens" / "ovis_text_goldens.json").read_text()
)


def dec(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def tiny_config() -> QwenTextConfig:
    return QwenTextConfig(**GOLDENS["model"]["config"])


def build_tiny() -> QwenTextModel:
    model = QwenTextModel(tiny_config())
    model.load_state_dict(fill_state_dict(GOLDENS["model"]["state_dict"]), strict=True)
    return model


def test_tiny_state_layout_matches_executed_reference() -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_tiny().state_dict().items())
    assert ours == [(key, list(shape)) for key, shape in GOLDENS["model"]["state_dict"]]


def test_full_state_layout_and_strict_load_match_executed_reference() -> None:
    with torch.device("meta"):
        model = QwenTextModel(OVIS_QWEN3_2B_CONFIG)
    expected = [(key, tuple(shape)) for key, shape in GOLDENS["layout"]]
    assert sorted(qwen_text_layout(OVIS_QWEN3_2B_CONFIG).items()) == expected
    assert (
        sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == expected
    )
    state = {key: torch.empty(shape, device="meta") for key, shape in expected}
    model.load_state_dict(state, strict=True, assign=True)
    del state["layers.27.mlp.down_proj.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True, assign=True)


def test_z_image_projection_geometry_matches_qwen3_4b_layout() -> None:
    with torch.device("meta"):
        model = QwenTextModel(Z_IMAGE_QWEN3_4B_CONFIG)
    expected = qwen_text_layout(Z_IMAGE_QWEN3_4B_CONFIG)
    assert sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == sorted(
        expected.items()
    )


def test_configured_hidden_layer_is_returned_without_final_norm() -> None:
    config = replace(tiny_config(), output_hidden_layer=-2, layer_norm_hidden_state=False)
    model = QwenTextModel(config)
    model.load_state_dict(build_tiny().state_dict(), strict=True)
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _args: tuple[object, ...], output: object) -> None:
        captured.append(cast("torch.Tensor", output))

    handle = model.layers[-2].register_forward_hook(capture)
    ids = torch.tensor(GOLDENS["model"]["ids"], dtype=torch.long)
    mask = torch.tensor(GOLDENS["model"]["attention_mask"], dtype=torch.long)
    try:
        output = model(ids, mask)
    finally:
        handle.remove()
    assert len(captured) == 1
    torch.testing.assert_close(output, captured[0], rtol=0, atol=0)


def test_model_math_matches_executed_reference() -> None:
    spec = GOLDENS["model"]
    ids = torch.tensor(spec["ids"], dtype=torch.long)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long)
    with torch.no_grad():
        got = build_tiny()(ids, mask)
    torch.testing.assert_close(got, dec(spec["output"]), rtol=1e-5, atol=1e-6)


def test_qwen_rope_matches_reference_addcmul_order() -> None:
    query = torch.linspace(-2.0, 2.0, 32, dtype=torch.float32).reshape(1, 2, 2, 8)
    key = query[:, :1] * 0.75
    cosine = torch.linspace(-0.9, 0.9, 16, dtype=torch.float32).reshape(1, 1, 2, 8)
    sine = torch.linspace(-0.7, 0.7, 8, dtype=torch.float32).reshape(1, 1, 2, 4)
    negative_sine = -sine

    def reference(value: torch.Tensor) -> torch.Tensor:
        result = value * cosine
        half = result.shape[-1] // 2
        result[..., :half].addcmul_(value[..., half:], negative_sine)
        result[..., half:].addcmul_(value[..., :half], sine)
        return result

    actual_query, actual_key = _apply_rope(query, key, (cosine, sine, negative_sine))
    assert torch.equal(actual_query, reference(query))
    assert torch.equal(actual_key, reference(key))


def test_qwen_uses_native_gqa_and_additive_mask_without_state_drift() -> None:
    baseline = build_tiny()
    spy = CallableModuleKernel(select_attention("qwen").kernel)
    model = QwenTextModel(tiny_config(), attention_kernel=spy)
    model.load_state_dict(baseline.state_dict(), strict=True)
    ids = torch.tensor(GOLDENS["model"]["ids"], dtype=torch.long)
    mask = torch.tensor(GOLDENS["model"]["attention_mask"], dtype=torch.long)
    assert set(model.state_dict()) == set(baseline.state_dict())
    assert_kernel_is_not_model_state(model, spy)
    torch.testing.assert_close(model(ids, mask), baseline(ids, mask))
    assert spy.calls
    assert all(
        call["mask"] is not None
        and not call["causal"]
        and call["enable_gqa"]
        and call["q_shape"][1] == 2 * call["k_shape"][1]
        and call["k_shape"][1] == call["v_shape"][1]
        for call in spy.calls
    )


def test_model_validates_ids_and_attention_mask_shapes() -> None:
    model = build_tiny()
    with pytest.raises(ValueError, match="ids must be"):
        model(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError, match="attention mask must match"):
        model(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1]]))


def test_model_admits_position_limit_and_refuses_longer_sequence() -> None:
    config = replace(tiny_config(), max_position_embeddings=3)
    model = QwenTextModel(config)
    model.load_state_dict(build_tiny().state_dict(), strict=True)
    assert model(torch.tensor([[1, 2, 3]])).shape[:2] == (1, 3)
    with pytest.raises(ValueError, match=rf"{config.architecture} received 4 tokens; maximum is 3"):
        model(torch.tensor([[1, 2, 3, 4]]))


class _RecordingModel:
    config = OVIS_QWEN3_2B_CONFIG

    def __init__(self) -> None:
        self.embed_tokens = type("Embedding", (), {"weight": torch.empty(0)})()
        self.ids: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None

    def __call__(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert attention_mask is not None
        self.ids = ids
        self.mask = attention_mask
        positions = torch.arange(ids.shape[1], dtype=torch.float32)
        return positions.reshape(1, -1, 1)


def test_ovis_encoder_applies_mask_and_post_template_slice() -> None:
    model = _RecordingModel()
    encoder = OvisTextEncoder(cast("Any", model))
    got: Conditioning[torch.Tensor] = encoder.encode("cat")
    expected = tokenize_ovis_prompt("cat")
    assert model.ids is not None and model.mask is not None
    assert model.ids.tolist() == [list(expected.ids)]
    assert model.mask.tolist() == [list(expected.attention_mask)]
    assert got.pooled is None
    assert got.embeddings.shape == (1, 284 - expected.slice_start, 1)
    assert got.embeddings[0, 0, 0].item() == expected.slice_start
    assert got.embeddings[0, -1, 0].item() == 0.0


def test_ovis_encoder_refuses_other_qwen_profiles() -> None:
    model = _RecordingModel()
    model.config = replace(
        OVIS_QWEN3_2B_CONFIG,
        architecture="other_qwen",
    )
    with pytest.raises(ValueError, match="requires the ovis_qwen3_2b"):
        OvisTextEncoder(cast("Any", model))


def test_z_image_encoder_preserves_attention_and_returns_hidden_layer() -> None:
    model = _RecordingModel()
    model.config = Z_IMAGE_QWEN3_4B_CONFIG
    encoder = ZImageTextEncoder(cast("Any", model))
    got = encoder.encode("cat")
    assert model.ids is not None and model.mask is not None
    assert model.ids.shape == model.mask.shape
    assert model.mask.tolist() == [[1] * model.ids.shape[1]]
    assert got.pooled is None
    assert got.embeddings.shape == (1, model.ids.shape[1], 1)


def test_z_image_encoder_refuses_other_qwen_profiles() -> None:
    with pytest.raises(ValueError, match="requires the z_image_qwen3_4b"):
        ZImageTextEncoder(cast("Any", _RecordingModel()))


def build_capture_model(*, qk_norm: bool = False) -> QwenTextModel:
    model = QwenTextModel(
        replace(
            tiny_config(),
            num_hidden_layers=4,
            qk_norm=qk_norm,
            output_hidden_layer=-3,
            layer_norm_hidden_state=False,
        )
    )
    generator = torch.Generator().manual_seed(1261)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.rand(parameter.shape, generator=generator) * 0.1)
    return model


@pytest.mark.parametrize("layer", (None, -3, -1))
def test_dense_ids_equal_preembedded_without_reusing_embedding_table(
    monkeypatch: pytest.MonkeyPatch, layer: int | None
) -> None:
    model = build_capture_model()
    model.config = replace(model.config, output_hidden_layer=layer)
    ids = torch.tensor([[2, 3, 4, 0, 0], [8, 9, 10, 11, 0]])
    mask = ids.ne(0).long()
    events: list[str] = []
    original = qwen_text._execution_device  # pyright: ignore[reportPrivateUsage]

    def device(module: torch.nn.Module) -> torch.device:
        events.append("embedding device" if module is model.embed_tokens else "other device")
        return original(module)

    def record_embedding(*_args: Any) -> None:
        events.append("embedding")

    monkeypatch.setattr(qwen_text, "_execution_device", device)
    hook = model.embed_tokens.register_forward_hook(record_embedding)
    try:
        with torch.no_grad():
            expected = model(ids, mask)
        assert events[:3] == ["embedding device", "embedding", "other device"]
        with torch.no_grad():
            embeds = model.embed_tokens(ids)
        events.clear()
        with torch.no_grad():
            actual = model(None, mask, embeds=embeds)
        assert "embedding" not in events and "embedding device" not in events
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    finally:
        hook.remove()


def test_arbitrary_dense_embeddings_preserve_padding_and_causal_masks() -> None:
    model = build_capture_model()
    embeds = torch.randn(1, 8, 16, generator=torch.Generator().manual_seed(42))
    mask = torch.tensor([[1, 1, 0, 1, 1, 1, 0, 0]])
    captures: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        captures.append(cast(torch.Tensor, output).clone())

    hook = model.layers[-3].register_forward_hook(capture)
    try:
        with torch.no_grad():
            result = model(None, mask, embeds=embeds)
    finally:
        hook.remove()
    torch.testing.assert_close(result, captures[0], rtol=0, atol=0)
    assert not torch.equal(result, model.norm(result))
    changed = embeds.clone()
    changed[:, 2] += 100
    changed[:, 6:] += 100
    future = embeds.clone()
    future[:, 5] += 100
    with torch.no_grad():
        masked = model(None, mask, embeds=changed)
        causal = model(None, mask, embeds=future)
    torch.testing.assert_close(result[:, 3:6], masked[:, 3:6], rtol=0, atol=0)
    torch.testing.assert_close(result[:, :5], causal[:, :5], rtol=0, atol=0)


@pytest.mark.parametrize("layer", (-4, -3, -1))
@pytest.mark.parametrize("qk_norm", (False, True))
@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("final_norm", (False, True))
def test_per_call_hidden_selection_preserves_config_and_default_ids(
    layer: int, qk_norm: bool, normalize: bool, final_norm: bool
) -> None:
    model = build_capture_model(qk_norm=qk_norm)
    model.config = replace(
        model.config,
        output_hidden_layer=None,
        layer_norm_hidden_state=normalize,
        final_norm=final_norm,
    )
    config = model.config
    ids = torch.tensor([[2, 3, 4, 0, 0]])
    mask = ids.ne(0).long()
    captures: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        captures.append(cast(torch.Tensor, output).clone())

    hook = model.layers[layer].register_forward_hook(capture)
    try:
        with torch.no_grad():
            default = model(ids, mask)
            expected = model.norm(captures[0]) if normalize and final_norm else captures[0]
            from_ids = model(ids, mask, hidden_layer=layer)
            from_embeds = model(None, mask, embeds=model.embed_tokens(ids), hidden_layer=layer)
            positive_index = model(ids, mask, hidden_layer=len(model.layers) + layer)
            after = model(ids, mask, hidden_layer=None)
    finally:
        hook.remove()
    for actual in (from_ids, from_embeds, positive_index):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(default, after, rtol=0, atol=0)
    assert model.config is config


@pytest.mark.parametrize("layer", (-5, 4))
def test_per_call_hidden_index_validation(layer: int) -> None:
    model = build_capture_model()
    ids = torch.tensor([[2, 3, 4]])
    with pytest.raises(ValueError, match="must address a Qwen model layer"):
        model(ids, hidden_layer=layer)
    with pytest.raises(ValueError, match="must address a Qwen model layer"):
        model(None, embeds=model.embed_tokens(ids), hidden_layer=layer)


@pytest.mark.parametrize("layer", (True, False, 1.0, 1.5, -1.0, 4.0))
@pytest.mark.parametrize("preembedded", (False, True))
def test_per_call_hidden_layer_requires_exact_int(layer: object, preembedded: bool) -> None:
    model = build_capture_model()
    ids = torch.tensor([[2, 3, 4]])
    with pytest.raises(ValueError, match="hidden_layer must be an integer or None"):
        if preembedded:
            model(None, embeds=model.embed_tokens(ids), hidden_layer=cast(int, layer))
        else:
            model(ids, hidden_layer=cast(int, layer))


@pytest.mark.parametrize("normalize", (False, True))
@pytest.mark.parametrize("final_norm", (False, True))
def test_per_call_hidden_rejects_multi_capture_without_changing_default(
    normalize: bool, final_norm: bool
) -> None:
    model = build_capture_model()
    model.config = replace(
        model.config,
        output_hidden_layer=None,
        output_hidden_layers=(0, 2),
        layer_norm_hidden_state=normalize,
        final_norm=final_norm,
    )
    config = model.config
    ids = torch.tensor([[2, 3, 4, 0]])
    mask = ids.ne(0).long()
    with torch.no_grad():
        default = model(ids, mask)
        from_embeds = model(None, mask, embeds=model.embed_tokens(ids))
    assert default.shape == (1, 2, 4, 16)
    torch.testing.assert_close(default, from_embeds, rtol=0, atol=0)
    with pytest.raises(ValueError, match="cannot override a multi-capture"):
        model(ids, mask, hidden_layer=-3)
    with pytest.raises(ValueError, match="cannot override a multi-capture"):
        model(None, mask, embeds=model.embed_tokens(ids), hidden_layer=-1)
    with torch.no_grad():
        torch.testing.assert_close(default, model(ids, mask), rtol=0, atol=0)
    assert model.config is config


@pytest.mark.parametrize(
    ("ids", "mask", "embeds", "message"),
    [
        (None, None, None, "exactly one"),
        (torch.ones(1, 2).long(), None, torch.zeros(1, 2, 16), "exactly one"),
        (torch.ones(2).long(), None, None, "batch x tokens"),
        (None, None, torch.zeros(1, 2), "tokens x hidden"),
        (None, None, torch.zeros(1, 2, 15), "tokens x hidden"),
        (None, torch.ones(1, 3), torch.zeros(1, 2, 16), "match input"),
        (torch.ones(1, 2).long(), torch.ones(1, 3), None, "match input"),
        (None, None, torch.zeros(1, 33, 16), "maximum is 32"),
    ],
)
def test_dense_exclusive_inputs_and_shapes(
    ids: torch.Tensor | None,
    mask: torch.Tensor | None,
    embeds: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_capture_model()(ids, mask, embeds=embeds)


def test_preembedded_input_admits_position_limit() -> None:
    model = build_capture_model()
    embeds = torch.randn(1, 32, 16, generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        assert model(None, embeds=embeds).shape == embeds.shape
