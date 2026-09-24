from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from dinkster_inference import (
    CancellationToken,
    Conditioning,
    Denoiser,
    GuidanceCondition,
    GuidanceContribution,
    GuidanceEvaluateNext,
    GuidanceEvaluationRequest,
    GuidanceEvaluationWrapperDescriptor,
    GuidanceExtensionError,
    GuidancePredictions,
    GuidanceRole,
    Parameterization,
    ProgressScope,
    SamplerInfo,
    SamplingExecutionContext,
)
from dinkster_inference_torch import EasyCacheConfig, LazyCacheConfig, SamplingCacheError
from dinkster_inference_torch import sampling_cache as sampling_cache_module
from dinkster_inference_torch.guidance import (
    ConditioningEvaluation,
    GuidanceExecutor,
    GuidanceRegistry,
    GuidedDenoiser,
)


class _Denoiser:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        del sigma
        self.calls += 1
        return x * 2.0


class _UncondDenoiser(_Denoiser):
    def call_with_uncond(self, x: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        del sigma
        self.calls += 1
        return x * 2.0, x * 3.0


class _AutoregressiveDenoiser(_Denoiser):
    def prepare_autoregressive(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("autoregressive sampling must not start")


def _wrap(
    denoiser: Denoiser[torch.Tensor],
    *,
    threshold: float = 0.2,
) -> Denoiser[torch.Tensor]:
    return LazyCacheConfig(reuse_threshold=threshold).wrap_denoiser(
        denoiser,
        (1.0, 0.8, 0.7, 0.0),
        SamplerInfo(Parameterization.EPS, percent_to_sigma=lambda percent: 1.0 - percent),
    )


def test_lazy_cache_reuses_only_inside_configured_window() -> None:
    inner = _Denoiser()
    cached = _wrap(inner)

    assert torch.equal(cached(torch.full((1, 1, 8, 8), 1.0), 1.0), torch.full((1, 1, 8, 8), 2.0))
    assert torch.equal(cached(torch.full((1, 1, 8, 8), 1.1), 0.8), torch.full((1, 1, 8, 8), 2.2))
    assert torch.allclose(
        cached(torch.full((1, 1, 8, 8), 1.11), 0.7),
        torch.full((1, 1, 8, 8), 2.21),
    )
    assert inner.calls == 2
    assert torch.equal(cached(torch.full((1, 1, 8, 8), 1.2), 0.05), torch.full((1, 1, 8, 8), 2.4))
    assert inner.calls == 3


def test_lazy_cache_threshold_miss_refreshes_the_residual() -> None:
    inner = _Denoiser()
    cached = _wrap(inner, threshold=0.001)

    cached(torch.full((1, 1, 8, 8), 1.0), 1.0)
    cached(torch.full((1, 1, 8, 8), 1.1), 0.8)
    output = cached(torch.full((1, 1, 8, 8), 1.2), 0.7)

    assert torch.equal(output, torch.full((1, 1, 8, 8), 2.4))
    assert inner.calls == 3


def test_lazy_cache_preserves_cfgpp_unconditional_output() -> None:
    inner = _UncondDenoiser()
    cached = _wrap(inner)

    cached.call_with_uncond(torch.full((1, 1, 8, 8), 1.0), 1.0)  # type: ignore[attr-defined]
    cached.call_with_uncond(torch.full((1, 1, 8, 8), 1.1), 0.8)  # type: ignore[attr-defined]
    output, uncond = cached.call_with_uncond(  # type: ignore[attr-defined]
        torch.full((1, 1, 8, 8), 1.11), 0.7
    )

    assert torch.allclose(output, torch.full((1, 1, 8, 8), 2.21))
    assert torch.allclose(uncond, torch.full((1, 1, 8, 8), 3.31))
    assert inner.calls == 2


def test_lazy_cache_refuses_cfgpp_without_unconditional_output() -> None:
    cached = _wrap(_Denoiser())

    with pytest.raises(SamplingCacheError, match="unconditional output"):
        cached.call_with_uncond(torch.ones((1, 1, 8, 8)), 0.8)  # type: ignore[attr-defined]


def test_lazy_cache_state_is_scoped_to_each_wrapper() -> None:
    inner = _Denoiser()
    config = LazyCacheConfig()
    info = SamplerInfo(Parameterization.EPS, percent_to_sigma=lambda percent: 1.0 - percent)
    first = config.wrap_denoiser(inner, (1.0, 0.0), info)
    second = config.wrap_denoiser(inner, (1.0, 0.0), info)

    first(torch.ones((1, 1, 8, 8)), 0.8)
    second(torch.ones((1, 1, 8, 8)), 0.8)

    assert inner.calls == 2


def test_lazy_cache_resets_on_tensor_metadata_change() -> None:
    inner = _Denoiser()
    cached = _wrap(inner)

    cached(torch.ones((1, 1, 8, 8)), 0.8)
    output = cached(torch.ones((1, 1, 16, 16)), 0.7)

    assert torch.equal(output, torch.full((1, 1, 16, 16), 2.0))
    assert inner.calls == 2


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"reuse_threshold": float("nan")}, TypeError),
        ({"reuse_threshold": -0.1}, ValueError),
        ({"start_percent": 0.8, "end_percent": 0.2}, ValueError),
        ({"subsample_factor": 0}, ValueError),
        ({"subsample_factor": True}, ValueError),
    ],
)
def test_lazy_cache_validates_config(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        LazyCacheConfig(**kwargs)  # type: ignore[arg-type]


def test_lazy_cache_requires_percent_to_sigma_mapping() -> None:
    with pytest.raises(SamplingCacheError, match="percent-to-sigma"):
        LazyCacheConfig().wrap_denoiser(
            _Denoiser(),
            (1.0, 0.0),
            SamplerInfo(Parameterization.EPS),
        )


def _easy_guided_denoiser(
    calls: list[tuple[GuidanceRole, float]],
    *,
    executor: GuidanceExecutor | None = None,
    cfg_scale: float = 2.0,
    force_uncond: bool = True,
) -> GuidedDenoiser:
    conditional = Conditioning(torch.tensor(4.0))
    unconditional = Conditioning(torch.tensor(2.0))

    def prepare(value: object, role: GuidanceRole) -> tuple[GuidanceRole, float]:
        source = value
        assert isinstance(source, Conditioning)
        return role, float(source.embeddings)

    def evaluate(
        x: torch.Tensor,
        _sigma: float,
        value: tuple[GuidanceRole, float],
    ) -> torch.Tensor:
        calls.append(value)
        return x * value[1]

    cancellation = CancellationToken(lambda: False)
    execution = SamplingExecutionContext(
        (1.0, 0.8, 0.7, 0.6, 0.0),
        0,
        0,
        1.0,
        7,
        cancellation,
        ProgressScope(cancellation),
        {},
    )
    return GuidedDenoiser(
        ConditioningEvaluation(prepare, evaluate),
        executor or GuidanceExecutor(GuidanceRegistry()),
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, conditional),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, unconditional),
        ),
        cfg_scale=cfg_scale,
        force_uncond=force_uncond,
        input=torch.zeros(1, 1, 8, 8),
        execution=execution,
    )


def _wrap_easy(
    denoiser: Denoiser[torch.Tensor],
    *,
    threshold: float = 0.2,
) -> Denoiser[torch.Tensor]:
    return EasyCacheConfig(reuse_threshold=threshold).wrap_denoiser(
        denoiser,
        (1.0, 0.8, 0.7, 0.6, 0.0),
        SamplerInfo(Parameterization.EPS, percent_to_sigma=lambda percent: 1.0 - percent),
    )


def test_easy_cache_reuses_all_guidance_lanes_by_stable_id() -> None:
    calls: list[tuple[GuidanceRole, float]] = []
    cached = _wrap_easy(_easy_guided_denoiser(calls))

    cached.call_with_uncond(torch.full((1, 1, 8, 8), 1.0), 1.0)  # type: ignore[attr-defined]
    cached.call_with_uncond(torch.full((1, 1, 8, 8), 1.1), 0.8)  # type: ignore[attr-defined]
    cached.call_with_uncond(torch.full((1, 1, 8, 8), 1.2), 0.7)  # type: ignore[attr-defined]
    denoised, uncond = cached.call_with_uncond(  # type: ignore[attr-defined]
        torch.full((1, 1, 8, 8), 1.21), 0.6
    )

    assert len(calls) == 6
    assert torch.allclose(denoised, torch.full((1, 1, 8, 8), 7.21))
    assert torch.allclose(uncond, torch.full((1, 1, 8, 8), 2.41))


def test_easy_cache_cfg_off_uses_one_stable_model_lane() -> None:
    calls: list[tuple[GuidanceRole, float]] = []
    cached = _wrap_easy(_easy_guided_denoiser(calls, cfg_scale=1.0, force_uncond=False))

    cached(torch.full((1, 1, 8, 8), 1.0), 1.0)
    cached(torch.full((1, 1, 8, 8), 1.1), 0.8)
    cached(torch.full((1, 1, 8, 8), 1.2), 0.7)
    output = cached(torch.full((1, 1, 8, 8), 1.21), 0.6)

    assert len(calls) == 3
    assert torch.allclose(output, torch.full((1, 1, 8, 8), 4.81))


def test_easy_cache_uses_first_physical_conditioning_lane_for_decisions() -> None:
    calls: list[tuple[GuidanceRole, ...]] = []
    conditional = Conditioning(torch.tensor(1.0))
    unconditional = Conditioning(torch.tensor(2.0))

    def prepare(value: object, role: GuidanceRole) -> GuidanceRole:
        assert isinstance(value, Conditioning)
        return role

    def evaluate(
        x: torch.Tensor,
        _sigma: float,
        values: tuple[GuidanceRole, ...],
    ) -> tuple[torch.Tensor, ...]:
        calls.append(values)
        return tuple(x.square() if role is GuidanceRole.CONDITIONAL else x for role in values)

    cancellation = CancellationToken(lambda: False)
    guided = GuidedDenoiser(
        ConditioningEvaluation(
            prepare,
            lambda _x, _sigma, _value: pytest.fail("lanes must evaluate together"),
            lambda _values: True,
            evaluate,
            standard_activation_memory_factor=1.0,
        ),
        GuidanceExecutor(GuidanceRegistry()),
        (
            GuidanceCondition("positive", GuidanceRole.CONDITIONAL, conditional),
            GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, unconditional),
        ),
        cfg_scale=2.0,
        force_uncond=True,
        input=torch.zeros(1, 1, 8, 8),
        execution=SamplingExecutionContext(
            (1.0, 0.8, 0.7, 0.6, 0.0),
            0,
            0,
            1.0,
            7,
            cancellation,
            ProgressScope(cancellation),
            {},
        ),
    )
    cached = _wrap_easy(guided, threshold=0.01)

    cached(torch.full((1, 1, 8, 8), 1.0), 1.0)
    cached(torch.full((1, 1, 8, 8), 1.1), 0.8)
    cached(torch.full((1, 1, 8, 8), 1.2), 0.7)
    output = cached(torch.full((1, 1, 8, 8), 1.21), 0.6)

    assert calls == [
        (GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL),
        (GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL),
        (GuidanceRole.UNCONDITIONAL, GuidanceRole.CONDITIONAL),
    ]
    assert torch.allclose(output, torch.full((1, 1, 8, 8), 1.69))


def test_easy_cache_identical_runs_make_identical_decisions() -> None:
    def run() -> tuple[tuple[torch.Tensor, ...], int]:
        calls: list[tuple[GuidanceRole, float]] = []
        cached = _wrap_easy(_easy_guided_denoiser(calls))
        outputs = tuple(
            cached(torch.full((1, 1, 8, 8), value), sigma)
            for value, sigma in ((1.0, 1.0), (1.1, 0.8), (1.2, 0.7), (1.21, 0.6))
        )
        return outputs, len(calls)

    first_outputs, first_calls = run()
    second_outputs, second_calls = run()

    assert first_calls == second_calls == 6
    assert all(
        torch.equal(first, second)
        for first, second in zip(first_outputs, second_outputs, strict=True)
    )


def test_easy_cache_conditioning_change_forces_a_miss() -> None:
    calls: list[tuple[GuidanceRole, float]] = []
    replacement = Conditioning(torch.tensor(5.0))

    def replace_positive(
        request: GuidanceEvaluationRequest[torch.Tensor],
        next: GuidanceEvaluateNext[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        if float(request.sigma) > 0.65:
            return next(request)
        lanes = tuple(
            replace(lane, conditioning=replacement) if lane.id == request.plan.primary_id else lane
            for lane in request.plan.lanes
        )
        return next(replace(request, plan=replace(request.plan, lanes=lanes)))

    executor = GuidanceExecutor(
        GuidanceRegistry(
            (
                (
                    "test",
                    GuidanceContribution(
                        evaluation_wrappers=(
                            GuidanceEvaluationWrapperDescriptor(
                                "test.conditioning-switch", replace_positive
                            ),
                        )
                    ),
                ),
            )
        )
    )
    cached = _wrap_easy(_easy_guided_denoiser(calls, executor=executor))

    cached(torch.full((1, 1, 8, 8), 1.0), 1.0)
    cached(torch.full((1, 1, 8, 8), 1.1), 0.8)
    cached(torch.full((1, 1, 8, 8), 1.2), 0.7)
    output = cached(torch.full((1, 1, 8, 8), 1.21), 0.6)

    assert len(calls) == 8
    assert torch.equal(output, torch.full((1, 1, 8, 8), 9.68))


def test_easy_cache_tensor_metadata_change_resets_before_reuse() -> None:
    calls: list[tuple[GuidanceRole, float]] = []
    cached = _wrap_easy(_easy_guided_denoiser(calls))

    cached(torch.full((1, 1, 8, 8), 1.0), 1.0)
    cached(torch.full((1, 1, 8, 8), 1.1), 0.8)
    cached(torch.full((1, 1, 8, 8), 1.2), 0.7)
    output = cached(torch.full((1, 1, 16, 16), 1.21), 0.6)

    assert len(calls) == 8
    assert torch.allclose(output, torch.full((1, 1, 16, 16), 7.26))


def test_easy_cache_resets_before_reentering_the_active_sigma_window() -> None:
    calls: list[tuple[GuidanceRole, float]] = []
    cached = _wrap_easy(_easy_guided_denoiser(calls))

    cached(torch.full((1, 1, 8, 8), 1.1), 0.8)
    cached(torch.full((1, 1, 8, 8), 1.2), 0.7)
    cached(torch.full((1, 1, 8, 8), 1.21), 0.6)
    cached(torch.full((1, 1, 8, 8), 10.0), 0.05)
    output = cached(torch.full((1, 1, 8, 8), 1.21), 0.6)

    assert len(calls) == 8
    assert torch.allclose(output, torch.full((1, 1, 8, 8), 7.26))


def test_easy_cache_state_is_scoped_to_each_wrapper() -> None:
    calls: list[tuple[GuidanceRole, float]] = []
    guided = _easy_guided_denoiser(calls)
    config = EasyCacheConfig()
    info = SamplerInfo(Parameterization.EPS, percent_to_sigma=lambda percent: 1.0 - percent)
    first = config.wrap_denoiser(guided, (1.0, 0.0), info)
    second = config.wrap_denoiser(guided, (1.0, 0.0), info)

    first(torch.ones((1, 1, 8, 8)), 0.8)
    second(torch.ones((1, 1, 8, 8)), 0.8)

    assert len(calls) == 4


def test_easy_cache_refuses_unsupported_denoiser_and_clears_binding() -> None:
    inner = _Denoiser()
    cached = _wrap_easy(inner)

    with pytest.raises(SamplingCacheError, match="shared guidance evaluation seam"):
        cached(torch.ones((1, 1, 8, 8)), 0.8)

    assert inner.calls == 1
    assert sampling_cache_module.active_guidance_evaluation_cache() is None


def test_easy_cache_clears_binding_when_guidance_fails() -> None:
    def fail(
        request: GuidanceEvaluationRequest[torch.Tensor],
        next: GuidanceEvaluateNext[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        del request, next
        raise RuntimeError("failure")

    executor = GuidanceExecutor(
        GuidanceRegistry(
            (
                (
                    "test",
                    GuidanceContribution(
                        evaluation_wrappers=(
                            GuidanceEvaluationWrapperDescriptor("test.failure", fail),
                        )
                    ),
                ),
            )
        )
    )
    cached = _wrap_easy(_easy_guided_denoiser([], executor=executor))

    with pytest.raises(GuidanceExtensionError, match="failure"):
        cached(torch.ones((1, 1, 8, 8)), 0.8)

    assert sampling_cache_module.active_guidance_evaluation_cache() is None


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"reuse_threshold": float("nan")}, TypeError),
        ({"reuse_threshold": -0.1}, ValueError),
        ({"start_percent": 0.8, "end_percent": 0.2}, ValueError),
        ({"subsample_factor": 0}, ValueError),
        ({"subsample_factor": True}, ValueError),
    ],
)
def test_easy_cache_validates_config(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        EasyCacheConfig(**kwargs)  # type: ignore[arg-type]


def test_easy_cache_requires_percent_to_sigma_mapping() -> None:
    with pytest.raises(SamplingCacheError, match="percent-to-sigma"):
        EasyCacheConfig().wrap_denoiser(
            _Denoiser(),
            (1.0, 0.0),
            SamplerInfo(Parameterization.EPS),
        )


def test_easy_cache_refuses_autoregressive_denoiser_before_sampling() -> None:
    with pytest.raises(SamplingCacheError, match="autoregressive"):
        _wrap_easy(_AutoregressiveDenoiser())
