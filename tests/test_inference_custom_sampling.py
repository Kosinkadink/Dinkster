from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pytest
from dinkster_inference import (
    Conditioning,
    ConditioningCarrier,
    ContextWindowsSpec,
    CustomSamplingRequest,
    CustomSamplingResult,
    CustomSamplingRuntime,
    Denoiser,
    DualSamplingGuidance,
    FamilyRuntime,
    InpaintConditioning,
    ModelFamily,
    MultiStreamLatent,
    OptionValue,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    SamplerDescriptor,
    SamplerInfo,
    SamplingCache,
    SamplingGuidance,
    SamplingStateCallback,
    SamplingTimelineSchedule,
    SolverFn,
    SparseLatent,
    StepCallback,
    builtin_family_registry,
    builtin_sampler_registry,
    current_realized_sampling_row,
    current_realized_sampling_timeline,
    realize_sampling_timeline,
    use_sampling_cache,
    use_sampling_timeline,
)
from dinkster_inference.sampling import (
    NoiseSampler,
    Parameterization,
    StepBeginCallback,
    run_step_begin_solver,
    solver_sampling_cache,
    solver_supports_step_begin,
)


@dataclass(frozen=True)
class _FakeTensor:
    value: float

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def _other(self, other: _FakeTensor | float) -> float:
        return other.value if isinstance(other, _FakeTensor) else other

    def __add__(self, other: _FakeTensor | float) -> _FakeTensor:
        return _FakeTensor(self.value + self._other(other))

    def __sub__(self, other: _FakeTensor | float) -> _FakeTensor:
        return _FakeTensor(self.value - self._other(other))

    def __mul__(self, other: _FakeTensor | float) -> _FakeTensor:
        return _FakeTensor(self.value * self._other(other))


class _CustomRuntime:
    def __init__(self) -> None:
        self.schedule_devices: list[object | None] = []

    @property
    def family(self) -> ModelFamily:
        family = builtin_family_registry().get("dinkster.sd15")
        assert family is not None
        return family

    @property
    def runtime_identity(self) -> str:
        return "native:custom:" + "0" * 64

    def custom_sampling_sigmas(
        self,
        scheduler_id: str,
        steps: int,
        denoise: float,
        *,
        device: object | None = None,
    ) -> tuple[float, ...]:
        self.schedule_devices.append(device)
        del scheduler_id, denoise
        return tuple(float(steps - index) for index in range(steps + 1))

    def custom_sampling_beta_sigmas(
        self,
        steps: int,
        alpha: float,
        beta: float,
        *,
        device: object | None = None,
    ) -> tuple[float, ...]:
        self.schedule_devices.append(device)
        return (float(steps), alpha, beta, 0.0)

    def custom_sampling_sd_turbo_sigmas(
        self,
        steps: int,
        denoise: float,
        *,
        device: object | None = None,
    ) -> tuple[float, ...]:
        self.schedule_devices.append(device)
        return (float(steps), denoise, 0.0)

    def custom_sampling_percent_to_sigma(
        self,
        percent: float,
        *,
        return_actual_sigma: bool,
    ) -> float:
        return percent if return_actual_sigma else 1.0 - percent

    def check_custom_sampling(
        self,
        request: CustomSamplingRequest[_FakeTensor],
        *,
        has_denoise_mask: bool,
        has_inpaint: bool,
        has_context_windows: bool,
        guidance: float | None = None,
    ) -> None:
        del request, has_denoise_mask, has_inpaint, has_context_windows, guidance

    def sample_custom(
        self,
        latent: _FakeTensor | MultiStreamLatent[_FakeTensor] | SparseLatent[_FakeTensor],
        *,
        noise: _FakeTensor | MultiStreamLatent[_FakeTensor] | SparseLatent[_FakeTensor],
        cond: Conditioning[_FakeTensor] | ConditioningCarrier | PreparedMultiStreamConditioning,
        cfg: (
            SamplingGuidance[Conditioning[_FakeTensor]]
            | SamplingGuidance[ConditioningCarrier]
            | SamplingGuidance[PreparedMultiStreamConditioning]
            | DualSamplingGuidance[Conditioning[_FakeTensor]]
            | DualSamplingGuidance[PreparedMultiStreamConditioning]
            | PerpNegSamplingGuidance[Conditioning[_FakeTensor]]
            | PerpNegSamplingGuidance[PreparedMultiStreamConditioning]
            | None
        ),
        request: CustomSamplingRequest[_FakeTensor],
        seed: int = 0,
        guidance: float | None = None,
        denoise_mask: (
            _FakeTensor | MultiStreamLatent[_FakeTensor] | SparseLatent[_FakeTensor] | None
        ) = None,
        inpaint: InpaintConditioning[_FakeTensor] | None = None,
        context_windows: ContextWindowsSpec | None = None,
        on_step: StepCallback | None = None,
        on_state: SamplingStateCallback | None = None,
    ) -> (
        CustomSamplingResult[_FakeTensor]
        | CustomSamplingResult[MultiStreamLatent[_FakeTensor]]
        | CustomSamplingResult[SparseLatent[_FakeTensor]]
    ):
        del noise, cond, cfg, request, seed, guidance, denoise_mask, inpaint, on_step, on_state
        del context_windows
        if type(latent) is MultiStreamLatent:
            return CustomSamplingResult(latent, latent)
        if type(latent) is SparseLatent:
            return CustomSamplingResult(latent, latent)
        narrowed = cast("_FakeTensor", latent)
        return CustomSamplingResult(narrowed, narrowed)


class _FakeCache:
    def __init__(self) -> None:
        self.wraps = 0
        self.calls = 0

    def wrap_denoiser(
        self,
        denoiser: Denoiser[_FakeTensor],
        sigmas: Sequence[float],
        info: SamplerInfo,
    ) -> Denoiser[_FakeTensor]:
        del sigmas, info
        self.wraps += 1

        def wrapped(x: _FakeTensor, sigma: float) -> _FakeTensor:
            self.calls += 1
            return denoiser(x, sigma)

        return wrapped


def _step_begin_sampler() -> SamplerDescriptor[_FakeTensor]:
    def make(_options: Mapping[str, OptionValue]) -> SolverFn[_FakeTensor]:
        def solve(
            denoiser: Denoiser[_FakeTensor],
            x: _FakeTensor,
            sigmas: Sequence[float],
            info: SamplerInfo,
            *,
            noise: NoiseSampler[_FakeTensor] | None = None,
            on_step: StepCallback | None = None,
            on_step_begin: StepBeginCallback | None = None,
        ) -> _FakeTensor:
            del info, noise, on_step
            if on_step_begin is not None:
                on_step_begin(0)
            return denoiser(x, sigmas[0])

        return cast("SolverFn[_FakeTensor]", solve)

    return SamplerDescriptor(
        "test.cached",
        "Cached",
        make,
        supports_step_begin=True,
    )


def test_custom_sampling_request_normalizes_options_and_sigmas() -> None:
    descriptor = builtin_sampler_registry().get("dinkster.euler_ancestral")
    assert descriptor is not None

    request = CustomSamplingRequest(descriptor, (("eta", 0.5),), (1, 0.25, 0))

    assert request.options == (("eta", 0.5), ("s_noise", 1.0))
    assert request.sigmas == (1.0, 0.25, 0.0)
    assert callable(request.build_solver())


@pytest.mark.parametrize("sigmas", [(float("nan"),), (-1.0,), (True,)])
def test_custom_sampling_request_rejects_invalid_sigmas(sigmas: object) -> None:
    descriptor = builtin_sampler_registry().get("dinkster.euler")
    assert descriptor is not None
    with pytest.raises((TypeError, ValueError), match="sigmas"):
        CustomSamplingRequest(descriptor, (), cast("tuple[float, ...]", sigmas))


def test_custom_sampling_request_rejects_unknown_options() -> None:
    descriptor = builtin_sampler_registry().get("dinkster.euler")
    assert descriptor is not None
    with pytest.raises(ValueError, match="unknown option"):
        CustomSamplingRequest(descriptor, (("missing", 1.0),), ())


def test_custom_sampling_request_applies_explicit_cache() -> None:
    cache = _FakeCache()
    request = CustomSamplingRequest(
        _step_begin_sampler(),
        (),
        (1.0, 0.0),
        cache=cast("SamplingCache[_FakeTensor]", cache),
    )
    solver = request.build_solver()
    steps: list[int] = []

    output = run_step_begin_solver(
        solver,
        lambda x, sigma: x + sigma,
        _FakeTensor(2.0),
        request.sigmas,
        SamplerInfo(Parameterization.EPS),
        noise=None,
        on_step=None,
        on_step_begin=steps.append,
    )

    assert solver_supports_step_begin(solver)
    assert output == _FakeTensor(3.0)
    assert cache.wraps == 1
    assert cache.calls == 1
    assert steps == [0]


def test_custom_sampling_request_captures_ambient_cache() -> None:
    cache = _FakeCache()
    sampler = _step_begin_sampler()

    with use_sampling_cache(cast("SamplingCache[_FakeTensor]", cache)):
        cached = CustomSamplingRequest(sampler, (), (1.0, 0.0))
    uncached = CustomSamplingRequest(sampler, (), (1.0, 0.0))

    assert cached.cache is cache
    assert uncached.cache is None


def test_solver_sampling_cache_finds_cache_inside_timeline_adapter() -> None:
    cache = _FakeCache()
    request = CustomSamplingRequest(
        _step_begin_sampler(),
        (),
        (1.0, 0.0),
        cache=cast("SamplingCache[_FakeTensor]", cache),
        timeline=SamplingTimelineSchedule("sage", 0.0, 1.0),
    )

    assert solver_sampling_cache(request.build_solver()) is cache
    assert solver_sampling_cache(_step_begin_sampler().build()) is None


def test_custom_sampling_request_rejects_invalid_cache() -> None:
    with pytest.raises(TypeError, match="SamplingCache"):
        CustomSamplingRequest(
            _step_begin_sampler(),
            (),
            (1.0, 0.0),
            cache=cast("SamplingCache[_FakeTensor]", object()),
        )


def test_custom_sampling_request_installs_timeline_before_control_callback() -> None:
    sampler = _step_begin_sampler()
    schedule = SamplingTimelineSchedule("sage", 0.0, 1.0)
    observations: list[tuple[str, int, str]] = []
    with use_sampling_timeline(schedule):
        request = CustomSamplingRequest(sampler, (), (1.0, 0.0))

    def control(step_index: int) -> None:
        row = current_realized_sampling_row()
        assert row is not None
        observations.append(("control", step_index, row.attention_plan.provider))

    def denoiser(x: _FakeTensor, sigma: float) -> _FakeTensor:
        row = current_realized_sampling_row()
        assert row is not None
        observations.append(("denoiser", row.anchors.step_index, row.attention_plan.provider))
        return x + sigma

    output = run_step_begin_solver(
        request.build_solver(),
        denoiser,
        _FakeTensor(2.0),
        request.sigmas,
        SamplerInfo(Parameterization.EPS),
        noise=None,
        on_step=None,
        on_step_begin=control,
    )

    assert output == _FakeTensor(3.0)
    assert observations == [("control", 0, "sage"), ("denoiser", 0, "sage")]
    assert current_realized_sampling_row() is None


def test_custom_sampling_reuses_precompiled_timeline_for_all_consumers() -> None:
    schedule = SamplingTimelineSchedule("sage", 0.0, 1.0)
    timeline = realize_sampling_timeline(schedule, (1.0, 0.0))
    request = CustomSamplingRequest(_step_begin_sampler(), (), (1.0, 0.0), timeline=schedule)
    observed: list[object] = []

    def denoiser(x: _FakeTensor, sigma: float) -> _FakeTensor:
        del sigma
        observed.append(current_realized_sampling_timeline())
        return x

    request.build_solver(realized_timeline=timeline)(
        denoiser,
        _FakeTensor(2.0),
        request.sigmas,
        SamplerInfo(Parameterization.EPS),
    )

    assert observed == [timeline]


def test_custom_sampling_timeline_cleans_up_after_solver_exception() -> None:
    request = CustomSamplingRequest(
        _step_begin_sampler(),
        (),
        (1.0, 0.0),
        timeline=SamplingTimelineSchedule("sage", 0.0, 1.0),
    )

    def fail(x: _FakeTensor, sigma: float) -> _FakeTensor:
        del x, sigma
        assert current_realized_sampling_row() is not None
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        request.build_solver()(
            fail,
            _FakeTensor(2.0),
            request.sigmas,
            SamplerInfo(Parameterization.EPS),
        )

    assert current_realized_sampling_row() is None


def test_custom_sampling_runtime_narrows_only_custom_runtimes() -> None:
    concrete = _CustomRuntime()
    runtime: CustomSamplingRuntime[_FakeTensor] = concrete
    schedule_device = object()

    assert isinstance(runtime, CustomSamplingRuntime)
    assert not isinstance(runtime, FamilyRuntime)
    assert runtime.custom_sampling_sigmas("dinkster.normal", 2, 1.0, device=schedule_device) == (
        2.0,
        1.0,
        0.0,
    )
    assert runtime.custom_sampling_beta_sigmas(2, 0.6, 0.7, device=schedule_device) == (
        2.0,
        0.6,
        0.7,
        0.0,
    )
    assert runtime.custom_sampling_sd_turbo_sigmas(2, 0.75, device=schedule_device) == (
        2.0,
        0.75,
        0.0,
    )
    assert concrete.schedule_devices == [schedule_device] * 3
    assert runtime.custom_sampling_percent_to_sigma(0.25, return_actual_sigma=False) == 0.75
