# pyright: basic
"""Windowed conditioning evaluation and FreeNoise tensor parity tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    AttentionGuidanceDescriptor,
    CancellationToken,
    GuidanceCondition,
    GuidanceContractError,
    GuidanceContribution,
    GuidanceRole,
    LatentStream,
    MultiStreamLatent,
    Parameterization,
    ProgressScope,
    SamplingExecutionContext,
)
from dinkster_inference.context_windows import (
    ContextFuseMethod,
    ContextWindowSchedule,
    ContextWindowsError,
    ContextWindowsSpec,
    plan_windows,
    relative_bias,
)
from dinkster_inference_torch import GuidanceExecutor, GuidanceRegistry, run_sampler_engine
from dinkster_inference_torch.context_windows import (
    PackedContextWindowAxis,
    PackedContextWindows,
    PackedContextWindowScale,
    PackedContextWindowStream,
    apply_freenoise,
    windowed_conditioning_evaluation,
)
from dinkster_inference_torch.guidance import ConditioningEvaluation
from dinkster_inference_torch.latent_streams import pack_latent_streams, unpack_latent_streams
from dinkster_inference_torch.sampling_execution import SamplingGuidancePlan, guided_denoiser
from dinkster_inference_torch.solvers import euler

GOLDENS = json.loads(
    (Path(__file__).parents[3] / "tests" / "goldens" / "context_windows_b78cec87.json").read_text(
        encoding="utf-8"
    )
)


def execution() -> SamplingExecutionContext:
    token = CancellationToken(lambda: False)
    return SamplingExecutionContext((1.0, 0.5, 0.0), 0, 0, 1.0, 7, token, ProgressScope(token), {})


def spec_with(**overrides: Any) -> ContextWindowsSpec:
    values: dict[str, Any] = {
        "schedule": ContextWindowSchedule.STATIC_STANDARD,
        "fuse_method": ContextFuseMethod.PYRAMID,
        "length": 16,
        "overlap": 4,
        "dim": 2,
    }
    values.update(overrides)
    return ContextWindowsSpec(**values)


def frame_latent(num_frames: int) -> torch.Tensor:
    """A (1, 1, T) latent whose value at frame t is t, so window inputs
    reveal the frame indices the wrapper selected."""

    return torch.arange(num_frames, dtype=torch.float32).reshape(1, 1, num_frames)


def identity_evaluation(seen: list[list[int]] | None = None) -> ConditioningEvaluation[float]:
    def evaluate(x: torch.Tensor, _sigma: float, _condition: float) -> torch.Tensor:
        if seen is not None:
            seen.append([int(value) for value in x[0, 0].tolist()])
        return x

    return ConditioningEvaluation(lambda _value, _role: 1.0, evaluate)


def packed_av_latent(
    frames: int = 7, audio_steps: int = 37
) -> tuple[torch.Tensor, PackedContextWindows, MultiStreamLatent[torch.Tensor]]:
    video = torch.arange(frames * 6, dtype=torch.float32).reshape(1, 1, frames, 2, 3)
    audio = torch.arange(audio_steps * 2, dtype=torch.float32).reshape(1, 1, 2, audio_steps)
    streams = MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))
    packed, layout = pack_latent_streams(streams)
    windows = PackedContextWindows(
        layout,
        (
            PackedContextWindowAxis(
                2,
                (
                    PackedContextWindowStream("video", 2),
                    PackedContextWindowStream("audio", 3, PackedContextWindowScale.PROPORTIONAL),
                ),
            ),
            PackedContextWindowAxis(4, (PackedContextWindowStream("video", 4),)),
        ),
    )
    return packed, windows, streams


@pytest.mark.parametrize(
    "case",
    GOLDENS["freenoise"],
    ids=lambda case: f"f{case['num_frames']}-l{case['length']}-o{case['overlap']}",
)
def test_apply_freenoise_matches_reference(case) -> None:
    num_frames = case["num_frames"]
    base = torch.arange(num_frames, dtype=torch.float32).reshape(1, 1, num_frames, 1, 1)
    shuffled = apply_freenoise(base, 2, case["length"], case["overlap"], case["seed"])
    assert [int(value) for value in shuffled.flatten().tolist()] == case["shuffled_indices"]
    assert torch.equal(
        base, torch.arange(num_frames, dtype=torch.float32).reshape(1, 1, num_frames, 1, 1)
    )


def test_apply_freenoise_rejects_out_of_range_dim() -> None:
    with pytest.raises(ContextWindowsError):
        apply_freenoise(torch.zeros(1, 1, 8), 3, 4, 1, 0)


@pytest.mark.parametrize("fuse", list(ContextFuseMethod))
def test_windowed_identity_evaluation_reconstructs_input(fuse: ContextFuseMethod) -> None:
    spec = spec_with(fuse_method=fuse)
    wrapped = windowed_conditioning_evaluation(identity_evaluation(), spec, (1.0, 0.0))
    x = frame_latent(33)
    # The relative blend's float32 weighted average rounds at ~1e-7 relative.
    torch.testing.assert_close(wrapped.evaluate(x, 1.0, 1.0), x, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fuse", list(ContextFuseMethod))
def test_packed_temporal_windows_restore_asymmetric_stream_extents(
    fuse: ContextFuseMethod,
) -> None:
    packed, windows, streams = packed_av_latent()
    wrapped = windowed_conditioning_evaluation(
        identity_evaluation(),
        spec_with(fuse_method=fuse, length=4, overlap=2),
        (1.0, 0.0),
        windows,
    )
    restored = unpack_latent_streams(wrapped.evaluate(packed, 1.0, 1.0), windows.layout)
    torch.testing.assert_close(restored.by_role("video"), streams.by_role("video"))
    torch.testing.assert_close(restored.by_role("audio"), streams.by_role("audio"))


@pytest.mark.parametrize("fuse", list(ContextFuseMethod))
def test_packed_spatial_windows_keep_unmapped_audio_whole(fuse: ContextFuseMethod) -> None:
    packed, windows, streams = packed_av_latent()
    wrapped = windowed_conditioning_evaluation(
        identity_evaluation(),
        spec_with(fuse_method=fuse, length=2, overlap=1, dim=4),
        (1.0, 0.0),
        windows,
    )
    restored = unpack_latent_streams(wrapped.evaluate(packed, 1.0, 1.0), windows.layout)
    torch.testing.assert_close(restored.by_role("video"), streams.by_role("video"))
    torch.testing.assert_close(restored.by_role("audio"), streams.by_role("audio"))


def test_windowed_evaluation_selects_planned_windows() -> None:
    seen: list[list[int]] = []
    spec = spec_with()
    wrapped = windowed_conditioning_evaluation(identity_evaluation(seen), spec, (1.0, 0.0))
    wrapped.evaluate(frame_latent(33), 1.0, 1.0)
    assert seen == [list(window) for window in plan_windows(spec, 33, 0)]


def test_causal_anchor_prepends_prior_frame_and_strips_it() -> None:
    seen: list[list[int]] = []
    spec = spec_with(causal_anchor=True)
    wrapped = windowed_conditioning_evaluation(identity_evaluation(seen), spec, (1.0, 0.0))
    x = frame_latent(33)
    torch.testing.assert_close(wrapped.evaluate(x, 1.0, 1.0), x, rtol=0.0, atol=1e-6)
    windows = [list(window) for window in plan_windows(spec, 33, 0)]
    assert seen == [window if window[0] == 0 else [window[0] - 1, *window] for window in windows]


def test_latent_retain_overwrites_window_positions_with_full_content() -> None:
    seen: list[list[int]] = []
    spec = spec_with(latent_retain_indices=(0,))
    wrapped = windowed_conditioning_evaluation(identity_evaluation(seen), spec, (1.0, 0.0))
    wrapped.evaluate(frame_latent(33), 1.0, 1.0)
    windows = [list(window) for window in plan_windows(spec, 33, 0)]
    assert seen == [[0, *window[1:]] for window in windows]


def test_latent_retain_applies_after_causal_anchor_prepend() -> None:
    seen: list[list[int]] = []
    spec = spec_with(causal_anchor=True, latent_retain_indices=(0,))
    wrapped = windowed_conditioning_evaluation(identity_evaluation(seen), spec, (1.0, 0.0))
    wrapped.evaluate(frame_latent(33), 1.0, 1.0)
    expected = []
    for window in plan_windows(spec, 33, 0):
        anchored = [window[0] - 1, *window] if window[0] > 0 else list(window)
        anchored[0] = 0
        expected.append(anchored)
    assert seen == expected


def test_latent_retain_refuses_windows_shorter_than_the_retained_index() -> None:
    spec = spec_with(schedule=ContextWindowSchedule.BATCHED, overlap=0, latent_retain_indices=(15,))
    wrapped = windowed_conditioning_evaluation(identity_evaluation(), spec, (1.0, 0.0))
    # 20 frames batch into a 16-frame window and a 4-frame tail the
    # retained index cannot address.
    with pytest.raises(ContextWindowsError):
        wrapped.evaluate(frame_latent(20), 1.0, 1.0)


def test_cond_retain_does_not_touch_the_latent_path() -> None:
    seen: list[list[int]] = []
    spec = spec_with(cond_retain_indices=(0,))
    wrapped = windowed_conditioning_evaluation(identity_evaluation(seen), spec, (1.0, 0.0))
    x = frame_latent(33)
    torch.testing.assert_close(wrapped.evaluate(x, 1.0, 1.0), x, rtol=0.0, atol=1e-6)
    assert seen == [list(window) for window in plan_windows(spec, 33, 0)]


def test_off_schedule_sigma_keeps_prior_step_windows() -> None:
    spec = spec_with(schedule=ContextWindowSchedule.UNIFORM_STANDARD)
    step_windows = [[list(window) for window in plan_windows(spec, 33, step)] for step in range(2)]
    assert step_windows[0] != step_windows[1]
    seen: list[list[int]] = []
    sigmas = (4.0, 2.0, 1.0, 0.0)
    wrapped = windowed_conditioning_evaluation(identity_evaluation(seen), spec, sigmas)
    x = frame_latent(33)
    wrapped.evaluate(x, 4.0, 1.0)
    assert seen == step_windows[0]
    seen.clear()
    wrapped.evaluate(x, 2.0, 1.0)
    assert seen == step_windows[1]
    seen.clear()
    # A solver substep sigma is off schedule: the step-1 windows persist.
    wrapped.evaluate(x, 1.7, 1.0)
    assert seen == step_windows[1]


def test_relative_fusing_matches_reference_blend() -> None:
    spec = spec_with(fuse_method=ContextFuseMethod.RELATIVE, length=4, overlap=2)
    calls: list[int] = []

    def evaluate(x: torch.Tensor, _sigma: float, _condition: float) -> torch.Tensor:
        calls.append(len(calls))
        return x * float(len(calls) + 1)

    inner = ConditioningEvaluation(lambda _value, _role: 1.0, evaluate)
    wrapped = windowed_conditioning_evaluation(inner, spec, (1.0, 0.0))
    num_frames = 10
    fused = wrapped.evaluate(frame_latent(num_frames), 1.0, 1.0)

    expected = [0.0] * num_frames
    totals = [0.0] * num_frames
    for call, window in enumerate(plan_windows(spec, num_frames, 0)):
        multiplier = float(call + 2)
        for index in window:
            bias = relative_bias(index, window[0], window[-1])
            total = totals[index]
            expected[index] = expected[index] * (total / (total + bias)) + float(
                index
            ) * multiplier * (bias / (total + bias))
            totals[index] = total + bias
    torch.testing.assert_close(
        fused,
        torch.tensor(expected, dtype=torch.float32).reshape(1, 1, num_frames),
        rtol=1e-6,
        atol=1e-6,
    )


def test_evaluate_only_family_windows_the_single_condition_path() -> None:
    seen: list[list[int]] = []
    wrapped = windowed_conditioning_evaluation(identity_evaluation(seen), spec_with(), (1.0, 0.0))
    assert wrapped.evaluate_batch is None
    assert wrapped.batchable is None
    x = frame_latent(33)
    torch.testing.assert_close(wrapped.evaluate(x, 1.0, 1.0), x, rtol=0.0, atol=1e-6)
    assert len(seen) == len(plan_windows(spec_with(), 33, 0))


def test_wrapper_declares_no_attention_evaluation() -> None:
    inner = ConditioningEvaluation(
        lambda _value, _role: 1.0,
        lambda x, _sigma, _condition: x,
        lambda _conditions: True,
        lambda x, _sigma, conditions: tuple(x for _ in conditions),
        evaluate_batch_attention=lambda x, _sigma, conditions, _roles, _attention: tuple(
            x for _ in conditions
        ),
    )
    wrapped = windowed_conditioning_evaluation(inner, spec_with(), (1.0, 0.0))
    assert wrapped.evaluate_batch_attention is None
    assert wrapped.evaluate_batch is not None
    assert wrapped.batchable is not None


def test_attention_contributions_refuse_before_sampling() -> None:
    wrapped = windowed_conditioning_evaluation(identity_evaluation(), spec_with(), (1.0, 0.0))
    registry = GuidanceRegistry(
        (
            (
                "owner",
                GuidanceContribution(
                    attention=AttentionGuidanceDescriptor(
                        "x.att", lambda positive, negative: positive
                    )
                ),
            ),
        )
    )
    plan = SamplingGuidancePlan(
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast(Any, 0.9)),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, cast(Any, 0.4)),
        ),
        2.0,
        False,
    )
    with pytest.raises(GuidanceContractError, match="attention-kind"):
        guided_denoiser(
            wrapped,
            input=torch.zeros(1, 2, 8),
            executor=GuidanceExecutor(registry),
            plan=plan,
            execution=execution(),
        )


def test_windowed_sampling_matches_unwindowed_for_a_linear_model() -> None:
    batch_calls: list[int] = []

    def evaluate(x: torch.Tensor, _sigma: float, condition: float) -> torch.Tensor:
        return x * condition

    def evaluate_batch(
        x: torch.Tensor, _sigma: float, conditions: tuple[float, ...]
    ) -> tuple[torch.Tensor, ...]:
        batch_calls.append(len(conditions))
        return tuple(x * condition for condition in conditions)

    inner = ConditioningEvaluation(
        lambda value, _role: cast(float, value),
        evaluate,
        lambda _conditions: True,
        evaluate_batch,
        standard_activation_memory_factor=1.0,
    )
    sigmas = (2.0, 1.0, 0.0)
    spec = spec_with(length=8, overlap=2)
    plan = SamplingGuidancePlan(
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, cast(Any, 0.9)),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, cast(Any, 0.4)),
        ),
        2.0,
        False,
    )
    num_frames = 21
    latent = torch.zeros(1, 2, num_frames)
    noise = torch.linspace(-1.0, 1.0, latent.numel()).reshape_as(latent)

    def sample(evaluation: ConditioningEvaluation[float]) -> torch.Tensor:
        return run_sampler_engine(
            guided_denoiser(
                evaluation, input=latent, executor=None, plan=plan, execution=execution()
            ),
            euler(),
            latent=latent,
            noise=noise,
            sigmas=sigmas,
            parameterization=Parameterization.EPS,
            sigma_max=2.0,
            process_in=lambda value: value,
            process_out=lambda value: value,
        )

    reference = sample(inner)
    windowed = sample(windowed_conditioning_evaluation(inner, spec, sigmas))
    torch.testing.assert_close(windowed, reference, rtol=1e-5, atol=1e-5)
    # The reference run makes one batched call per solver step; the windowed
    # run makes windows-many batched calls per step, each carrying both lanes.
    steps = len(sigmas) - 1
    windows_per_step = len(plan_windows(spec, num_frames, 0))
    assert batch_calls == [2] * (steps + steps * windows_per_step)
