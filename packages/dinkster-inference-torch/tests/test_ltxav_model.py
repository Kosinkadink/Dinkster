"""The native LTX-2 audio-video transformer against the executed reference.

Every golden in goldens/ltxav_model_goldens.json was produced by
RUNNING the reference LTXAVModel (comfy/ldm/lightricks/av_model.py @
the audited baseline, tools/gen_ltxav_model_goldens.py) with attention
forced to pytorch SDPA. Weights come from the shared deterministic
hash (unet_fill.py) and inputs from its ``hashed_input`` namespace.

The 19B reference builds text-side embeddings connectors unconditionally,
so its golden listings carry connector keys that Dinkster filters. LTX-2.3
22B owns asymmetric connectors in the diffusion model, and its golden
listings keep those keys.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]
import pytest
import torch
from dinkster_inference import (
    LTXAV_19B_CONFIG,
    LTXAV_22B_V23_CONFIG,
    LTXAV_22B_V25_CONFIG,
    LTXAV_CONNECTOR_PREFIXES,
    LTXAVConfig,
    LtxConnectorConfig,
    LTXGeneratedKeyframes,
    ltxav_layout,
)
from dinkster_inference_torch import LTXAVModel, pack_av_latents, unpack_av_latents
from dinkster_inference_torch import ltxav_model as ltxav_model_module
from dinkster_inference_torch.module_residency import enroll_component
from unet_fill import fill_state_dict, hashed_input

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "ltxav_model_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def split_connector_entries(
    entries: list[tuple[str, list[int]]],
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    kept = [entry for entry in entries if not entry[0].startswith(LTXAV_CONNECTOR_PREFIXES)]
    dropped = [entry for entry in entries if entry[0].startswith(LTXAV_CONNECTOR_PREFIXES)]
    return kept, dropped


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    entries = [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]
    if GOLDENS["cases"][case]["config"].get("caption_proj_before_connector", False):
        return entries
    kept, dropped = split_connector_entries(entries)
    assert dropped, "the reference always builds the connectors"
    return kept


def case_config(case: str) -> LTXAVConfig:
    spec = GOLDENS["cases"][case]["config"]
    connector_layers = spec.get("connector_num_layers")
    video_connector = None
    audio_connector = None
    if spec.get("caption_proj_before_connector", False):
        video_connector = LtxConnectorConfig(
            num_attention_heads=spec["connector_num_attention_heads"],
            attention_head_dim=spec["connector_attention_head_dim"],
            num_layers=connector_layers,
            num_learnable_registers=128,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=4096,
            gated_attention=spec["connector_apply_gated_attention"],
        )
        audio_connector = LtxConnectorConfig(
            num_attention_heads=spec["audio_connector_num_attention_heads"],
            attention_head_dim=spec["audio_connector_attention_head_dim"],
            num_layers=connector_layers,
            num_learnable_registers=128,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=4096,
            gated_attention=spec["connector_apply_gated_attention"],
        )
    return LTXAVConfig(
        in_channels=spec["in_channels"],
        cross_attention_dim=spec["cross_attention_dim"],
        attention_head_dim=spec["attention_head_dim"],
        num_attention_heads=spec["num_attention_heads"],
        audio_in_channels=spec["audio_in_channels"],
        audio_cross_attention_dim=spec["audio_cross_attention_dim"],
        audio_attention_head_dim=spec["audio_attention_head_dim"],
        audio_num_attention_heads=spec["audio_num_attention_heads"],
        caption_channels=spec["caption_channels"],
        num_layers=spec["num_layers"],
        causal_temporal_positioning=spec["causal_temporal_positioning"],
        use_middle_indices_grid=spec["use_middle_indices_grid"],
        av_ca_timestep_scale_multiplier=spec["av_ca_timestep_scale_multiplier"],
        cross_attention_adaln=spec["cross_attention_adaln"],
        caption_proj_before_connector=spec.get("caption_proj_before_connector", False),
        gated_attention=spec.get("apply_gated_attention", False),
        video_connector=video_connector,
        audio_connector=audio_connector,
    )


def build_model(case: str) -> LTXAVModel:
    model = LTXAVModel(case_config(case))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


def case_inputs(case: str) -> dict[str, Any]:
    spec = GOLDENS["cases"][case]
    config = spec["config"]
    batch = spec["batch"]
    video = hashed_input(
        f"{case}:x",
        (batch, config["in_channels"], spec["frames"], spec["height"], spec["width"]),
    )
    if spec["audio_length"]:
        audio = hashed_input(f"{case}:audio", (batch, 8, spec["audio_length"], 16))
    else:
        audio = torch.zeros(batch, 8, 0, 16, dtype=torch.float32)
    context_width = (
        config["cross_attention_dim"] + config["audio_cross_attention_dim"]
        if config.get("caption_proj_before_connector", False)
        else 2 * config["caption_channels"]
    )
    context = hashed_input(f"{case}:context", (batch, spec["context_len"], context_width))
    attention_mask = None
    if spec["attention_mask"] is not None:
        attention_mask = torch.tensor(spec["attention_mask"], dtype=torch.int64)
    denoise_mask = None
    if spec["denoise_mask"] is not None:
        denoise_mask = torch.tensor(spec["denoise_mask"], dtype=torch.float32)
    ref_audio_tokens = None
    if spec["ref_audio_tokens"] is not None:
        ref_audio_tokens = hashed_input(f"{case}:ref_audio", tuple(spec["ref_audio_tokens"]))
    return {
        "video": video,
        "audio": audio,
        "timestep": torch.tensor(spec["timestep"], dtype=torch.float32),
        "audio_timestep": torch.tensor(spec["audio_timestep"], dtype=torch.float32),
        "context": context,
        "attention_mask": attention_mask,
        "frame_rate": spec["frame_rate"],
        "denoise_mask": denoise_mask,
        "ref_audio_tokens": ref_audio_tokens,
    }


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_model(case).state_dict().items())
    assert ours == golden_entries(case)


@pytest.mark.parametrize("case", CASES)
def test_torch_free_layout_predicts_the_module(case: str) -> None:
    predicted = sorted((key, list(shape)) for key, shape in ltxav_layout(case_config(case)).items())
    assert predicted == golden_entries(case)


@pytest.mark.parametrize(
    ("name", "config"),
    (("ltxav_19b", LTXAV_19B_CONFIG), ("ltxav_22b_v23", LTXAV_22B_V23_CONFIG)),
)
def test_full_size_module_matches_reference_layout(name: str, config: LTXAVConfig) -> None:
    """Real 19B and 22B architectures against the reference layouts."""
    entries = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    if name == "ltxav_19b":
        entries, dropped = split_connector_entries(entries)
        assert len(dropped) == 58
    with torch.device("meta"):
        model = LTXAVModel(config)
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert ours == entries


@pytest.mark.parametrize(
    ("name", "config"),
    (("ltxav_19b", LTXAV_19B_CONFIG), ("ltxav_22b_v23", LTXAV_22B_V23_CONFIG)),
)
def test_torch_free_layout_predicts_the_full_size_reference(name: str, config: LTXAVConfig) -> None:
    entries = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    if name == "ltxav_19b":
        entries, _ = split_connector_entries(entries)
    predicted = sorted((key, list(shape)) for key, shape in ltxav_layout(config).items())
    assert predicted == entries


def test_v23_model_state_is_admitted_by_component_residency() -> None:
    model = build_model("ltxav_22b_v23")

    enroll_component(model, load_device="cpu", offload_device="cpu")


def test_v25_full_size_module_omits_only_video_feed_forward_biases() -> None:
    with torch.device("meta"):
        model = LTXAVModel(LTXAV_22B_V25_CONFIG)

    assert set(model.state_dict()) == set(ltxav_layout(LTXAV_22B_V25_CONFIG))
    block = cast("Any", model.transformer_blocks[0])
    assert block.ff.net[0].proj.bias is None
    assert block.ff.net[2].bias is None
    assert block.audio_ff.net[0].proj.bias is not None
    assert block.audio_ff.net[2].bias is not None


def test_optional_keyframe_embedding_marks_causal_and_generated_frames() -> None:
    config = replace(case_config("ltxav_per_frame"), use_keyframes_abs_pos_embedding=True)
    model = LTXAVModel(config)
    state = fill_state_dict(
        sorted((key, list(shape)) for key, shape in ltxav_layout(config).items())
    )
    embedding = torch.arange(config.hidden_size, dtype=torch.float32).reshape(1, -1)
    state["keyframes_abs_pos_embedding"] = embedding
    model.load_state_dict(state, strict=True)
    inputs = case_inputs("ltxav_per_frame")
    video = cast("torch.Tensor", inputs["video"])
    tokens_per_frame = video.shape[3] * video.shape[4]
    inputs["generated_keyframes"] = LTXGeneratedKeyframes(tokens_per_frame, 2, 1)
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        captured.append(cast("torch.Tensor", args[0]).detach().clone())

    hook = model.transformer_blocks[0].register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        hook.remove()

    projected = model.patchify_proj(video.flatten(2).transpose(1, 2))
    expected = projected.clone()
    expected[:, :tokens_per_frame] += embedding
    expected[:, 2 * tokens_per_frame : 3 * tokens_per_frame] += embedding
    assert torch.equal(captured[0], expected)
    assert "keyframes_abs_pos_embedding" in model.state_dict()


def test_generated_keyframes_require_the_same_latent_grid() -> None:
    config = replace(case_config("ltxav_per_frame"), use_keyframes_abs_pos_embedding=True)
    model = LTXAVModel(config)
    model.load_state_dict(
        fill_state_dict(sorted((key, list(shape)) for key, shape in ltxav_layout(config).items())),
        strict=True,
    )
    inputs = case_inputs("ltxav_per_frame")
    inputs["generated_keyframes"] = LTXGeneratedKeyframes(5, 1, 1)

    with pytest.raises(ValueError, match="recorded at 5 tokens.*latent has 4"):
        model(**inputs)


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_forward_and_block_match_executed_reference(case: str) -> None:
    model = build_model(case)
    observed: dict[str, torch.Tensor] = {}

    def record_block(
        _module: torch.nn.Module, _inputs: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        observed["block0_video"] = cast(torch.Tensor, output[0])
        observed["block0_audio"] = cast(torch.Tensor, output[1])

    hook = model.transformer_blocks[0].register_forward_hook(record_block)
    inputs = case_inputs(case)
    try:
        with torch.no_grad():
            video_out, audio_out = model(**inputs)
    finally:
        hook.remove()
    golden = GOLDENS["cases"][case]
    torch.testing.assert_close(
        observed["block0_video"],
        dec(golden["block_outputs"]["block0_video"]),
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        observed["block0_audio"],
        dec(golden["block_outputs"]["block0_audio"]),
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(video_out, dec(golden["output_video"]), rtol=1e-4, atol=1e-5)
    if golden["output_audio"] is None:
        assert audio_out.numel() == 0
    else:
        torch.testing.assert_close(audio_out, dec(golden["output_audio"]), rtol=1e-4, atol=1e-5)


def test_execution_controls_match_reference_transformer_options() -> None:
    model = build_model("ltxav_22b_v23")
    inputs = case_inputs("ltxav_22b_v23")
    block = cast("Any", model.transformer_blocks[0])
    q_calls: list[str] = []
    a2v_calls: list[str] = []
    v2a_calls: list[str] = []

    def record(calls: list[str], name: str) -> Any:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], _output: object) -> None:
            calls.append(name)

        return hook

    hooks = (
        block.attn1.to_q.register_forward_hook(record(q_calls, "video")),
        block.audio_attn1.to_q.register_forward_hook(record(q_calls, "audio")),
        block.audio_to_video_attn.register_forward_hook(record(a2v_calls, "a2v")),
        block.video_to_audio_attn.register_forward_hook(record(v2a_calls, "v2a")),
    )
    try:
        with torch.no_grad():
            baseline = model(**inputs)
            explicit_default = model(
                **inputs,
                stg_self_attn_blocks=frozenset(),
                a2v_cross_attention=True,
                v2a_cross_attention=True,
            )
            stg = model(**inputs, stg_self_attn_blocks=frozenset({0}))
            decoupled = model(
                **inputs,
                a2v_cross_attention=False,
                v2a_cross_attention=False,
            )
    finally:
        for hook in hooks:
            hook.remove()

    assert all(
        torch.equal(left, right) for left, right in zip(baseline, explicit_default, strict=True)
    )
    assert not torch.equal(stg[0], baseline[0])
    assert not torch.equal(decoupled[0], baseline[0])
    assert q_calls == ["video", "audio"] * 3
    assert a2v_calls == ["a2v"] * 3
    assert v2a_calls == ["v2a"] * 3


def test_split_rope_uses_paired_and_single_kitchen_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import ltx_model as ltx_model_module

    q_matrix = ltx_model_module._rope_matrix(  # pyright: ignore[reportPrivateUsage]
        torch.randn(1, 3, 4), 0, True, 2, torch.float32
    )
    k_matrix = ltx_model_module._rope_matrix(  # pyright: ignore[reportPrivateUsage]
        torch.randn(1, 2, 4), 0, True, 2, torch.float32
    )
    q = torch.randn(1, 3, 8)
    same_k = torch.randn(1, 3, 8)
    short_k = torch.randn(1, 2, 8)
    calls: list[str] = []

    def paired(
        query: torch.Tensor, key: torch.Tensor, table: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("paired")
        return (
            ltx_model_module._apply_rope_torch(  # pyright: ignore[reportPrivateUsage]
                query, table, True
            ),
            ltx_model_module._apply_rope_torch(  # pyright: ignore[reportPrivateUsage]
                key, table, True
            ),
        )

    def single(value: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        calls.append("single")
        return ltx_model_module._apply_rope_torch(  # pyright: ignore[reportPrivateUsage]
            value, table, True
        )

    monkeypatch.setattr(comfy_kitchen, "apply_rope_split_half", paired)
    monkeypatch.setattr(comfy_kitchen, "apply_rope_split_half1", single)
    with torch.no_grad():
        ltxav_model_module._apply_split_rope_qk(  # pyright: ignore[reportPrivateUsage]
            q, same_k, (q_matrix, True), None
        )
        ltxav_model_module._apply_split_rope_qk(  # pyright: ignore[reportPrivateUsage]
            q, short_k, (q_matrix, True), (k_matrix, True)
        )
    assert calls == ["paired", "single", "single"]


def test_spatial_timestep_detection_checks_every_batch_row() -> None:
    model = build_model("ltxav_spatial_mask")
    inputs = case_inputs("ltxav_spatial_mask")
    mask = torch.ones((2, 1, 2, 2, 3), dtype=torch.float32)
    mask[0, :, 1] = 0.5
    mask[1, :, 0, 0, 0] = 0.0
    inputs["denoise_mask"] = mask
    inputs["timestep"] = mask.flatten(1) * torch.tensor((0.35, 0.8)).reshape(2, 1)

    with torch.no_grad():
        batched_video, batched_audio = model(**inputs)

    single_inputs = {
        name: value[1:2] if type(value) is torch.Tensor and value.shape[0] == 2 else value
        for name, value in inputs.items()
    }
    with torch.no_grad():
        single_video, single_audio = model(**single_inputs)

    torch.testing.assert_close(batched_video[1:2], single_video, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched_audio[1:2], single_audio, rtol=1e-5, atol=1e-6)


# ------------------------------------------------- packed wire format


def test_pack_matches_reference_and_round_trips() -> None:
    example = GOLDENS["pack_example"]
    video = hashed_input("pack:video", tuple(example["video_shape"]))
    audio = hashed_input("pack:audio", tuple(example["audio_shape"]))
    packed, shapes = pack_av_latents([video, audio])
    assert [list(shape) for shape in shapes] == example["latent_shapes"]
    torch.testing.assert_close(packed, dec(example["packed"]), rtol=0.0, atol=0.0)
    unpacked = unpack_av_latents(packed, shapes)
    assert len(unpacked) == 2
    assert torch.equal(unpacked[0], video)
    assert torch.equal(unpacked[1], audio)


def test_unpack_single_shape_passes_through() -> None:
    packed = hashed_input("pack:single", (2, 1, 12))
    unpacked = unpack_av_latents(packed, [torch.Size((2, 4, 3))])
    assert len(unpacked) == 1
    assert torch.equal(unpacked[0], packed)


# ------------------------------------------------------- fail closed


def test_unprocessed_context_width_is_refused() -> None:
    case = "ltxav_t2av_scalar"
    model = build_model(case)
    inputs = case_inputs(case)
    inputs["context"] = inputs["context"][..., :-1]
    with pytest.raises(ValueError, match="embeddings-connector"):
        model(**inputs)


def test_config_rejects_mismatched_video_context_width() -> None:
    with pytest.raises(ValueError, match="cross_attention_dim"):
        LTXAVConfig(cross_attention_dim=1024)


def test_config_rejects_mismatched_audio_context_width() -> None:
    with pytest.raises(ValueError, match="audio_cross_attention_dim"):
        LTXAVConfig(audio_cross_attention_dim=1024)


def test_config_rejects_non_mel_audio_channels() -> None:
    with pytest.raises(ValueError, match="audio_in_channels"):
        LTXAVConfig(audio_in_channels=64)
