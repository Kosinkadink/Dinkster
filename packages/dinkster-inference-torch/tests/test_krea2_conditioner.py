"""Krea 2 language-tower math and encode-policy goldens."""

from __future__ import annotations

import json
from dataclasses import fields, make_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import dinkster_inference_torch.krea2_conditioner as krea2_conditioner_module
import pytest
import torch
from clip_fill import fill_state_dict
from dinkster_inference import (
    KREA2_TAP_LAYERS,
    Conditioning,
    krea2_language_layout,
    select_krea2_output,
    tokenize_krea2_prompt,
)
from dinkster_inference_torch import Krea2TextEncoder, krea2_language_model
from dinkster_inference_torch.qwen_image_text import QwenImageLanguageModel

GOLDENS = json.loads(
    (Path(__file__).parents[3] / "tests" / "goldens" / "krea2_text_goldens.json").read_text()
)


def dec(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def build_tiny() -> QwenImageLanguageModel:
    spec = GOLDENS["model"]["config"]
    model = QwenImageLanguageModel.reduced(
        vocab_size=spec["vocab_size"],
        hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_layers=spec["num_hidden_layers"],
        num_heads=spec["num_attention_heads"],
        num_kv_heads=spec["num_key_value_heads"],
        rope_dims=tuple(spec["rope_dims"]),
        head_dim=spec["head_dim"],
        rope_theta=spec["rope_theta"],
        qkv_bias=spec["qkv_bias"],
        qk_norm=spec["qk_norm"],
        final_norm=spec["final_norm"],
        interleaved_mrope=spec["interleaved_mrope"],
    )
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert entries == [(key, list(shape)) for key, shape in GOLDENS["model"]["state_dict"]]
    model.load_state_dict(fill_state_dict(GOLDENS["model"]["state_dict"]), strict=True)
    return model


def golden_ids_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(GOLDENS["model"]["ids"], dtype=torch.long),
        torch.tensor(GOLDENS["model"]["attention_mask"], dtype=torch.long),
    )


def test_full_meta_state_layout_strict_loads_the_language_tower() -> None:
    with torch.device("meta"):
        model = krea2_language_model()
    assert model.shape.max_position_embeddings == 262144
    assert model.shape.architecture == "Krea 2"
    expected = sorted(krea2_language_layout().items())
    assert sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == (
        expected
    )
    state = {key: torch.empty(shape, device="meta") for key, shape in expected}
    model.load_state_dict(state, strict=True, assign=True)
    del state["layers.35.mlp.down_proj.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True, assign=True)


def test_tapped_states_match_executed_reference() -> None:
    ids, mask = golden_ids_and_mask()
    tap_layers = tuple(GOLDENS["model"]["config"]["tap_layers"])
    with torch.no_grad():
        got = build_tiny().tapped_states(ids, mask, tap_layers=tap_layers)
    torch.testing.assert_close(got, dec(GOLDENS["model"]["taps"]), rtol=1e-5, atol=1e-6)


def test_taps_capture_each_layer_input_and_skip_layers_past_the_last_tap() -> None:
    model = build_tiny()
    ids, mask = golden_ids_and_mask()
    tap_layers = (1, 3, 5)
    inputs: dict[int, torch.Tensor] = {}
    outputs: dict[int, torch.Tensor] = {}
    ran: list[int] = []
    handles = []
    for index, layer in enumerate(model.layers):

        def pre_hook(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            index: int = index,
        ) -> None:
            ran.append(index)
            inputs[index] = cast("torch.Tensor", args[0])

        def post_hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            index: int = index,
        ) -> None:
            outputs[index] = cast("torch.Tensor", output)

        handles.append(layer.register_forward_pre_hook(pre_hook))
        handles.append(layer.register_forward_hook(post_hook))
    try:
        with torch.no_grad():
            stacked = model.tapped_states(ids, mask, tap_layers=tap_layers)
    finally:
        for handle in handles:
            handle.remove()
    # Tap k is the input to layer k; tap 5 is captured before layer 5
    # would run, and no later layer can affect any capture, so layer 5
    # itself never executes.
    assert ran == [0, 1, 2, 3, 4]
    torch.testing.assert_close(stacked[:, 0], inputs[1], rtol=0, atol=0)
    torch.testing.assert_close(stacked[:, 1], inputs[3], rtol=0, atol=0)
    torch.testing.assert_close(stacked[:, 2], outputs[4], rtol=0, atol=0)


def test_tapped_states_validate_ids_mask_and_tap_layers() -> None:
    model = build_tiny()
    ids, mask = golden_ids_and_mask()
    with pytest.raises(ValueError, match="rank 2"):
        model.tapped_states(ids[0], tap_layers=(1,))
    with pytest.raises(ValueError, match="mask must match"):
        model.tapped_states(ids, mask[:, :-1], tap_layers=(1,))
    for tap_layers in ((), (True,), (-1,), (7,)):
        with pytest.raises(ValueError, match="decoder layer indices"):
            model.tapped_states(ids, mask, tap_layers=tap_layers)  # type: ignore[arg-type]
    for tap_layers in ((3, 1), (1, 1, 3)):
        with pytest.raises(ValueError, match="strictly increasing"):
            model.tapped_states(ids, mask, tap_layers=tap_layers)

    terminal = model.tapped_states(ids, mask, tap_layers=(6,))
    assert terminal.shape == (ids.shape[0], 1, ids.shape[1], model.shape.hidden_size)

    limited = QwenImageLanguageModel.reduced(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        rope_dims=(2, 1, 1),
        final_norm=False,
        max_position_embeddings=3,
        architecture="Krea test tower",
    )
    with pytest.raises(ValueError, match="Krea test tower received 4 tokens; maximum is 3"):
        limited.tapped_states(torch.zeros((1, 4), dtype=torch.long), tap_layers=(0,))


class _RecordingKrea2Model:
    def __init__(self, shape: Any) -> None:
        self.shape = shape
        self.embed_tokens = type("Embedding", (), {"weight": torch.empty(0)})()
        self.ids: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.tap_layers: tuple[int, ...] | None = None

    def tapped_states(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        tap_layers: tuple[int, ...],
    ) -> torch.Tensor:
        self.ids = ids
        self.mask = attention_mask
        self.tap_layers = tap_layers
        length = ids.shape[1]
        positions = torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1)
        return positions.expand(1, len(tap_layers), length, 3).clone()

    def validate_sequence_length(self, length: int) -> None:
        if length > self.shape.max_position_embeddings:
            raise ValueError(
                f"{self.shape.architecture} received {length} tokens; "
                f"maximum is {self.shape.max_position_embeddings}"
            )


def full_profile_shape() -> Any:
    with torch.device("meta"):
        return krea2_language_model().shape


def test_krea2_encoder_strips_template_and_stacks_taps_per_sequence_position() -> None:
    model = _RecordingKrea2Model(full_profile_shape())
    encoder = Krea2TextEncoder(cast("Any", model))
    got: Conditioning[torch.Tensor] = encoder.encode("cat")
    expected = tokenize_krea2_prompt("cat")
    selection = select_krea2_output([list(expected.ids)], [list(expected.attention_mask)])
    assert model.ids is not None and model.mask is not None
    assert model.ids.tolist() == [list(expected.ids)]
    assert model.mask.tolist() == [list(expected.attention_mask)]
    assert model.tap_layers == KREA2_TAP_LAYERS
    assert got.pooled is None
    assert got.embeddings.dtype == torch.float32
    length = len(expected.ids) - selection.slice_start
    assert got.embeddings.shape == (1, length, 12 * 3)
    # Row i holds tap-major copies of sequence position slice_start + i.
    assert got.embeddings[0, 0].unique().tolist() == [float(selection.slice_start)]
    assert got.embeddings[0, -1].unique().tolist() == [float(len(expected.ids) - 1)]


def test_krea2_encoder_refuses_overflow_before_output_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_selection(*_args: object, **_kwargs: object) -> None:
        pytest.fail("output selection ran before length validation")

    model = _RecordingKrea2Model(full_profile_shape())
    count = model.shape.max_position_embeddings + 1
    tokens = SimpleNamespace(ids=(0,) * count, attention_mask=(1,) * count)

    def tokenize(_text: str) -> SimpleNamespace:
        return tokens

    monkeypatch.setattr(krea2_conditioner_module, "tokenize_krea2_prompt", tokenize)
    monkeypatch.setattr(krea2_conditioner_module, "select_krea2_output", fail_selection)

    with pytest.raises(ValueError, match=rf"Krea 2 received {count} tokens; maximum is"):
        Krea2TextEncoder(cast("Any", model)).encode("oversized")


def test_krea2_encoder_refuses_other_language_profiles() -> None:
    with pytest.raises(ValueError, match="exact Krea 2 language profile"):
        Krea2TextEncoder(cast("Any", build_tiny()))


@pytest.mark.parametrize(
    ("field", "value"),
    (("max_position_embeddings", None), ("architecture", "other Qwen tower")),
)
def test_krea2_encoder_refuses_position_policy_drift(field: str, value: object) -> None:
    shape = replace(full_profile_shape(), **{field: value})
    model = _RecordingKrea2Model(shape)
    with pytest.raises(ValueError, match="exact Krea 2 language profile"):
        Krea2TextEncoder(cast("Any", model))


def test_krea2_encoder_refuses_unrecognized_profile_fields() -> None:
    current = full_profile_shape()
    extended_type = make_dataclass(
        "ExtendedLanguageShape",
        [(field.name, field.type) for field in fields(current)] + [("normalization_variant", str)],
        frozen=True,
    )
    shape = extended_type(
        **{field.name: getattr(current, field.name) for field in fields(current)},
        normalization_variant="future-profile-drift",
    )
    model = _RecordingKrea2Model(shape)
    with pytest.raises(ValueError, match="exact Krea 2 language profile"):
        Krea2TextEncoder(cast("Any", model))
