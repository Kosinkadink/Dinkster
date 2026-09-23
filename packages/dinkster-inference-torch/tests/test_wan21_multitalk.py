"""Native Wan 2.1 InfiniteTalk/MultiTalk model, math, and binding contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from dinkster_inference import WAN21_MULTITALK, Wan21MultiTalkConfig
from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
from dinkster_inference.wan21_multitalk import wan21_multitalk_model_layout
from dinkster_inference_torch import assemble as assemble_module
from dinkster_inference_torch import assemble_wan21_multitalk
from dinkster_inference_torch import wan21_model as wan21_model_module
from dinkster_inference_torch import wan21_multitalk as multitalk_module
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.wan21_model import Wan21Config, Wan21Model
from dinkster_inference_torch.wan21_multitalk import (
    MultiTalkAudioProjection,
    MultiTalkCrossAttention,
    MultiTalkRotaryEmbedding1D,
    Wan21MultiTalk,
    Wan21MultiTalkBindingError,
    Wan21MultiTalkExecution,
    _split_audio_windows,  # pyright: ignore[reportPrivateUsage]
    multitalk_attention_map,
    multitalk_audio_windows,
    project_audio_features,
    snapshot_wan21_multitalk_execution,
    validate_wan21_multitalk_resource,
    wan21_multitalk_resource_digest,
    wan21_multitalk_tensor_digest,
)
from golden_files import cpu_identity, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "wan21_model_goldens.json")

# Pinned ComfyUI on Linux differs from the stored Windows golden by at most
# 1.6689301e-6 while Dinkster matches the same-process reference within 1e-6.
# This limit leaves at least 1.49x headroom over the cross-host golden drift.
MULTITALK_GOLDEN_ATOL = 2.5e-6

# Batched versus independent execution drifts by at most 1.3057e-6 on Windows
# and 1.847744e-6 on Linux.
# This separate limit leaves at least 1.35x headroom over batch reduction order.
MULTITALK_BATCH_ATOL = 2.5e-6


def _tensor_diagnostic(value: torch.Tensor) -> dict[str, object]:
    flat = value.detach().to(dtype=torch.float32, device="cpu").flatten()
    return {
        "digest": wan21_multitalk_tensor_digest(value),
        "max": float(flat.max()),
        "min": float(flat.min()),
        "shape": tuple(value.shape),
    }


def _fill_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17 - 8) / 64)


def _small_sealed_model() -> tuple[Wan21MultiTalk, str]:
    model = Wan21MultiTalk.__new__(Wan21MultiTalk)
    torch.nn.Module.__init__(model)
    test_model = cast(Any, model)
    test_model.config = WAN21_MULTITALK
    test_model.layer = assemble_module.INITLESS.linear(2, 2, bias=False)
    model.load_state_dict(
        {"layer.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        strict=True,
        assign=True,
    )
    digest = wan21_multitalk_resource_digest("blake3:" + "0" * 64, torch.bfloat16)
    multitalk_module._bind_wan21_multitalk_resource(  # pyright: ignore[reportPrivateUsage]
        model, digest
    )
    return model, digest


def test_exact_meta_layout_matches_all_40_maintained_patch_blocks() -> None:
    with torch.device("meta"):
        model = Wan21MultiTalk()
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    executed_reference = {key: tuple(shape) for key, shape in GOLDENS["layouts"]["multitalk_patch"]}

    assert model.config is WAN21_MULTITALK
    assert actual == wan21_multitalk_model_layout()
    assert actual == executed_reference
    assert len(model.blocks) == 40
    assert len(actual) == 330
    assert actual["audio_proj.proj1.weight"] == (512, 46080)
    assert actual["audio_proj.proj1_vf.weight"] == (512, 73728)
    assert actual["blocks.39.audio_cross_attn.kv_linear.weight"] == (10240, 768)
    assert actual["blocks.39.norm_x.weight"] == (5120,)


def test_five_and_eight_frame_audio_windows_and_projection_match_reference_values() -> None:
    encoded = torch.arange(9, dtype=torch.float32).reshape(9, 1, 1)
    windows = multitalk_audio_windows((encoded,), 0, 9)
    first, latter = _split_audio_windows(windows)

    assert first.shape == (1, 1, 5, 1, 1)
    assert latter.shape == (1, 2, 8, 1, 1)
    assert first[0, 0, :, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert latter[0, 0, :, 0, 0].tolist() == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert latter[0, 1, :, 0, 0].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0]

    projection = MultiTalkAudioProjection(
        seq_len=5,
        seq_len_vf=8,
        blocks=1,
        channels=1,
        intermediate_dim=1,
        out_dim=2,
        context_tokens=1,
    )
    with torch.no_grad():
        projection.proj1.weight.fill_(1.0)
        projection.proj1.bias.zero_()
        projection.proj1_vf.weight.fill_(1.0)
        projection.proj1_vf.bias.zero_()
        projection.proj2.weight.fill_(1.0)
        projection.proj2.bias.zero_()
        projection.proj3.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        projection.proj3.bias.zero_()
        projection.norm.weight.fill_(1.0)
        projection.norm.bias.zero_()

    actual = project_audio_features(projection, (encoded,), 0, 9)
    pre_norm = torch.tensor(((3.0, -3.0), (21.0, -21.0), (49.0, -49.0)))
    expected = F.layer_norm(pre_norm, (2,)).reshape(1, 3, 1, 2)

    assert actual.shape == (1, 3, 1, 2)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    initial_only = project_audio_features(projection, (encoded,), 0, 1)
    assert initial_only.shape == (1, 1, 1, 2)
    torch.testing.assert_close(initial_only, expected[:, :1], rtol=0.0, atol=0.0)


def test_short_audio_pads_with_its_first_frame_and_refuses_invalid_frame_groups() -> None:
    encoded = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1)
    windows = multitalk_audio_windows((encoded,), 4, 9)

    assert windows[0, -1, :, 0, 0].tolist() == [0.0] * 5
    with pytest.raises(ValueError, match="groups of four"):
        _split_audio_windows(multitalk_audio_windows((encoded,), 0, 6))


def test_attention_map_matches_chunked_masked_softmax_math() -> None:
    visual_q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 1.0], [1.0, -0.5]]]])
    ref_k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])
    masks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    actual = multitalk_attention_map(visual_q, ref_k, (1, 1, 2), masks, split_num=2)
    scaled_q = visual_q.transpose(1, 2) / (visual_q.shape[-1] ** 0.5)
    logits = scaled_q @ ref_k.permute(0, 2, 3, 1)
    probabilities = (logits - logits.max(dim=-1, keepdim=True).values).exp()
    probabilities = probabilities / (probabilities.sum(dim=-1, keepdim=True) + 1e-8)
    expected = torch.stack(
        tuple(
            ((probabilities * mask.reshape(1, 1, 1, -1)).sum(-1) / (mask.sum() + 1e-8))
            .mean(dim=1)
            .squeeze(0)
            for mask in masks
        )
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_single_and_two_speaker_attention_use_optimized_kernel_and_stay_finite() -> None:
    rotary = MultiTalkRotaryEmbedding1D(2)
    rotary_actual = rotary(
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        torch.tensor([0.0, torch.pi / 2]),
    )
    torch.testing.assert_close(
        rotary_actual,
        torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]]),
        rtol=1e-6,
        atol=1e-7,
    )

    attention = MultiTalkCrossAttention(
        4,
        3,
        2,
        class_range=24,
        class_interval=4,
    )
    _fill_parameters(attention)
    hidden = torch.tensor(
        [
            [
                [0.1, -0.2, 0.3, -0.4],
                [0.2, 0.1, -0.3, 0.4],
                [-0.5, 0.6, 0.7, -0.8],
                [0.9, -1.0, 1.1, -1.2],
            ]
        ]
    )
    single_context = torch.linspace(-0.5, 0.5, 18).reshape(1, 2, 3, 3)

    single = attention(hidden, single_context, (2, 1, 2))
    reshaped = hidden.reshape(2, 2, 4)
    query = attention.q_linear(reshaped).reshape(2, 2, 2, 2).transpose(1, 2)
    key, value = attention.kv_linear(single_context.squeeze(0)).reshape(2, 3, 2, 2, 2).unbind(2)
    expected = F.scaled_dot_product_attention(
        query, key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)
    expected = attention.proj(expected.reshape(2, 2, 4)).reshape(1, 4, 4)

    assert single.shape == hidden.shape
    assert torch.isfinite(single).all()
    torch.testing.assert_close(single, expected, rtol=1e-6, atol=1e-7)

    batched_hidden = torch.cat((hidden, hidden + 0.25))
    batched_single = attention(batched_hidden, single_context, (2, 1, 2))
    batched_reshaped = batched_hidden.reshape(4, 2, 4)
    batched_query = attention.q_linear(batched_reshaped).reshape(4, 2, 2, 2).transpose(1, 2)
    expanded_context = single_context.expand(2, -1, -1, -1).reshape(4, 3, 3)
    batched_key, batched_value = (
        attention.kv_linear(expanded_context).reshape(4, 3, 2, 2, 2).unbind(2)
    )
    batched_expected = F.scaled_dot_product_attention(
        batched_query,
        batched_key.transpose(1, 2),
        batched_value.transpose(1, 2),
    ).transpose(1, 2)
    batched_expected = attention.proj(batched_expected.reshape(4, 2, 4)).reshape(2, 4, 4)
    torch.testing.assert_close(batched_single, batched_expected, rtol=1e-6, atol=1e-7)

    two_context = torch.linspace(-0.75, 0.75, 36).reshape(1, 2, 6, 3)
    speaker_map = torch.tensor([[0.9, 0.7, 0.2, 0.1], [0.1, 0.3, 0.8, 0.9]])
    two = attention(hidden, two_context, (2, 1, 2), speaker_map)

    speaker_one = (
        (speaker_map[0] - speaker_map[0].min())
        / (speaker_map[0].max() - speaker_map[0].min() + 1e-8)
        * 4
    )
    speaker_two = (speaker_map[1] - speaker_map[1].min()) / (
        speaker_map[1].max() - speaker_map[1].min() + 1e-8
    ) * 4 + 20
    positions = torch.stack((speaker_one, speaker_two, torch.full((4,), 12.0)), dim=1)[
        torch.arange(4), speaker_map.argmax(dim=0)
    ]
    two_query = attention.q_linear(reshaped).reshape(2, 2, 2, 2).permute(0, 2, 1, 3)
    two_query = two_query.reshape(1, 2, 2, 2, 2).permute(0, 2, 1, 3, 4).reshape(1, 2, 4, 2)
    two_query = attention.rope_1d(two_query, positions)
    two_query = two_query.reshape(1, 2, 2, 2, 2).permute(0, 2, 1, 3, 4).reshape(2, 2, 2, 2)
    two_key, two_value = (
        attention.kv_linear(two_context.squeeze(0)).reshape(2, 6, 2, 2, 2).permute(2, 0, 3, 1, 4)
    ).unbind(0)
    key_positions = torch.tensor([2.0, 2.0, 2.0, 22.0, 22.0, 22.0] * 2)
    two_key = two_key.reshape(1, 2, 2, 6, 2).permute(0, 2, 1, 3, 4).reshape(1, 2, 12, 2)
    two_key = attention.rope_1d(two_key, key_positions)
    two_key = two_key.reshape(1, 2, 2, 6, 2).permute(0, 2, 1, 3, 4).reshape(2, 2, 6, 2)
    two_expected = F.scaled_dot_product_attention(two_query, two_key, two_value).transpose(1, 2)
    two_expected = attention.proj(two_expected.reshape(2, 2, 4)).reshape(1, 4, 4)

    assert two.shape == hidden.shape
    assert torch.isfinite(two).all()
    torch.testing.assert_close(two, two_expected, rtol=1e-6, atol=1e-7)


def test_reduced_model_and_patch_forward_match_executed_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = GOLDENS["cases"]["infinite_talk_reduced"]
    model_config = Wan21Config(**spec["config"])
    model = Wan21Model(model_config)
    model.load_state_dict(fill_state_dict(spec["state_dict"]), strict=True)

    patch_values = {
        "audio_window": spec["patch_config"]["audio_window"],
        "latter_audio_window": (
            spec["patch_config"]["audio_window"] + spec["patch_config"]["vae_scale"] - 1
        ),
        "audio_encoder_blocks": 12,
        "audio_input_width": 768,
        "audio_hidden_width": spec["patch_config"]["intermediate_dim"],
        "audio_context_width": spec["patch_config"]["out_dim"],
        "audio_context_tokens": spec["patch_config"]["context_tokens"],
        "patch_width": spec["patch_config"]["in_dim"],
        "attention_heads": 40,
        "layers": spec["patch_config"]["num_layers"],
        "speaker_class_range": 24,
        "speaker_class_interval": 4,
    }
    patch_config = object.__new__(Wan21MultiTalkConfig)
    for field, value in patch_values.items():
        object.__setattr__(patch_config, field, value)
    monkeypatch.setattr(multitalk_module, "WAN21_MULTITALK", patch_config)
    patch = Wan21MultiTalk(patch_config)
    patch.load_state_dict(fill_state_dict(spec["patch_state_dict"]), strict=True)
    model_digest = wan21_multitalk_resource_digest("blake3:" + "3" * 64, torch.float32)
    multitalk_module._bind_wan21_multitalk_resource(  # pyright: ignore[reportPrivateUsage]
        patch, model_digest
    )

    audio = tuple(
        hashed_input(f"infinite_talk_reduced:audio:{index}", shape)
        for index, shape in enumerate(spec["audio_shapes"])
    )
    audio_context = patch.project_audio(audio, spec["audio_start"], spec["audio_end"])
    target_masks = torch.tensor(spec["target_masks"], dtype=torch.float32)
    execution = Wan21MultiTalkExecution(
        patch,
        audio_context,
        target_masks,
        float(spec["strength"]),
        model_digest,
        wan21_multitalk_tensor_digest(audio_context),
        wan21_multitalk_tensor_digest(target_masks),
    )
    monkeypatch.setattr(wan21_model_module, "WAN21_I2V_14B", model_config)
    actual = model(
        hashed_input("infinite_talk_reduced:x", spec["input_shape"]),
        torch.tensor(spec["timesteps"], dtype=torch.float32),
        hashed_input("infinite_talk_reduced:context", spec["context_shape"]),
        hashed_input("infinite_talk_reduced:vision", spec["vision_shape"]),
        multitalk=execution,
    )
    repeated = model(
        hashed_input("infinite_talk_reduced:x", spec["input_shape"]),
        torch.tensor(spec["timesteps"], dtype=torch.float32),
        hashed_input("infinite_talk_reduced:context", spec["context_shape"]),
        hashed_input("infinite_talk_reduced:vision", spec["vision_shape"]),
        multitalk=execution,
    )
    expected = torch.tensor(
        spec["output"]["data"], dtype=getattr(torch, spec["output"]["dtype"])
    ).reshape(spec["output"]["shape"])
    torch.testing.assert_close(actual, repeated, rtol=0.0, atol=0.0)
    golden_drift = float((actual - expected).detach().abs().max())
    assert golden_drift <= MULTITALK_GOLDEN_ATOL, (
        f"MultiTalk max drift {golden_drift} exceeds {MULTITALK_GOLDEN_ATOL}; "
        f"cpu={cpu_identity()}; torch={torch.__version__}; "
        f"audio_context={_tensor_diagnostic(audio_context)}; "
        f"target_masks={_tensor_diagnostic(target_masks)}; "
        f"actual={_tensor_diagnostic(actual)}; expected={_tensor_diagnostic(expected)}"
    )

    first_x = hashed_input("infinite_talk_reduced:x", spec["input_shape"])
    first_context = hashed_input("infinite_talk_reduced:context", spec["context_shape"])
    first_vision = hashed_input("infinite_talk_reduced:vision", spec["vision_shape"])
    second_x = first_x + 0.125
    second_context = first_context - 0.25
    second_vision = first_vision + 0.375
    batched = model(
        torch.cat((first_x, second_x)),
        torch.tensor(spec["timesteps"] * 2, dtype=torch.float32),
        torch.cat((first_context, second_context)),
        torch.cat((first_vision, second_vision)),
        multitalk=execution,
    )
    independent = torch.cat(
        (
            model(
                first_x,
                torch.tensor(spec["timesteps"], dtype=torch.float32),
                first_context,
                first_vision,
                multitalk=execution,
            ),
            model(
                second_x,
                torch.tensor(spec["timesteps"], dtype=torch.float32),
                second_context,
                second_vision,
                multitalk=execution,
            ),
        )
    )
    torch.testing.assert_close(
        batched,
        independent,
        rtol=1e-5,
        atol=MULTITALK_BATCH_ATOL,
    )


def test_assembly_builds_exact_profile_selects_sdpa_and_seals_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = SimpleNamespace(config=WAN21_MULTITALK)

    def load(
        value: object,
        build: Any,
        *,
        compute_dtype: torch.dtype,
        fp8_matmul: bool,
    ) -> torch.nn.Module:
        assert value is component
        assert compute_dtype == torch.bfloat16
        assert fp8_matmul is False
        with torch.device("meta"):
            model = build(
                WAN21_MULTITALK,
                operations=assemble_module.CastOperations(compute_dtype),
            )
        assert type(model) is Wan21MultiTalk
        assert set(model.state_dict()) == set(wan21_multitalk_model_layout())
        assert assemble_module.bound_compute_dtype(model.audio_proj.proj1) is torch.float16
        first_block = cast(Any, model.blocks[0])
        assert (
            assemble_module.bound_compute_dtype(first_block.audio_cross_attn.q_linear)
            is torch.bfloat16
        )
        return model

    monkeypatch.setattr(assemble_module, "_load_component", load)
    plan = cast(
        Any,
        SimpleNamespace(patch=component, asset_digest="blake3:" + "1" * 64),
    )
    assembled = assemble_wan21_multitalk(plan)

    assert assembled.compute_dtype == torch.bfloat16
    assert assembled.attention_status.role == "flux"
    assert assembled.attention_status.primary == "sdpa"
    assert assembled.resource_digest == wan21_multitalk_resource_digest(
        plan.asset_digest, torch.bfloat16
    )

    with pytest.raises(TypeError, match="compute dtype must be floating"):
        assemble_wan21_multitalk(plan, compute_dtype=torch.int32)


def test_resource_binding_accepts_whole_component_residency_load_and_unload() -> None:
    model, digest = _small_sealed_model()
    test_model = cast(Any, model)
    original = test_model.layer.weight
    mechanism = enroll_component(
        model,
        load_device="cpu",
        offload_device="cpu",
        patch_set=PatchSet(
            {"layer.weight": (PatchEntry(DiffPatch(torch.ones_like(test_model.layer.weight))),)}
        ),
    )

    mechanism.partially_load(None)
    assert test_model.layer.weight is not original
    validate_wan21_multitalk_resource(model, digest)

    mechanism.unload()
    assert test_model.layer.weight is original
    validate_wan21_multitalk_resource(model, digest)


def test_execution_descriptor_owns_inputs_and_rejects_tensor_or_resource_mutation() -> None:
    model, digest = _small_sealed_model()
    audio = torch.zeros((1, 2, 32, 768))
    audio_digest = wan21_multitalk_tensor_digest(audio)
    execution = Wan21MultiTalkExecution(
        model,
        audio,
        None,
        -1.25,
        digest,
        audio_digest,
        None,
    )
    snapshot = snapshot_wan21_multitalk_execution(execution)

    audio.fill_(1.0)
    assert torch.count_nonzero(snapshot.audio_context).item() == 0
    with pytest.raises(Wan21MultiTalkBindingError, match="audio context identity changed"):
        Wan21MultiTalkExecution(model, audio, None, -1.25, digest, audio_digest, None)

    masks = torch.tensor([[True, False], [False, True]])
    two_audio = torch.zeros((1, 2, 64, 768))
    Wan21MultiTalkExecution(
        model,
        two_audio,
        masks,
        1.0,
        digest,
        wan21_multitalk_tensor_digest(two_audio),
        wan21_multitalk_tensor_digest(masks),
    )

    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(Wan21MultiTalkBindingError, match="binding-mismatch"):
        validate_wan21_multitalk_resource(model, digest)
