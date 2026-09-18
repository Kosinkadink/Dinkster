"""Sampling contracts: the solver boundary, typed.

The portable heart of ComfyUI's sampling stack is one interface: a
denoiser called as ``model(x, sigma)`` inside a solver loop over a
sigma schedule (comfy/k_diffusion/sampling.py @ b78cec87). Everything
wrong with it is around that interface - name lists as registration,
solvers probing ``model.inner_model.inner_model``, capabilities
discovered by introspection.

Here the boundary is explicit:

- Denoiser: the callable solvers see. Nothing else - no wrapper guts.
- SamplerInfo: what the model needs the solver to know (parameterization,
  noise scaling behavior), passed as data instead of probed.
- NoiseSampler / NoiseKind: noise as an injected callable plus a
  descriptor-declared requirement, replacing hardcoded torch.randn_like
  and default-constructed BrownianTreeNoiseSampler.
- SolverFn / StepEvent: the solver contract and its progress callback.
- SigmaScheduleFn: schedules as pure float functions over a SigmaSpace
  (they need no tensors at all - comfy/samplers.py karras/exponential/
  beta/kl_optimal @ b78cec87 are already pure math; the ports live in
  schedules.py, the spaces in spaces.py).
- OptionSpec / resolve_options: typed per-solver option schemas
  (eta, s_noise, solver_type) as data on descriptors, replacing
  KSAMPLER's untyped extra_options dict expanded blind into ``**kwargs``
  (comfy/samplers.py KSAMPLER @ b78cec87).
- Sampler/SchedulerDescriptor: registry entries with provenance,
  replacing KSAMPLER_NAMES/SCHEDULER_HANDLERS. The solver ports
  themselves live in solvers.py.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum
from types import CodeType, FunctionType, MappingProxyType
from typing import Any, Generic, Literal, Protocol, Self, TypeVar, cast, final, runtime_checkable

from .sampling_timeline import (
    RealizedSamplingTimeline,
    SamplingTimelineSchedule,
    executed_sampling_timeline,
    realize_sampling_timeline,
    use_realized_sampling_timeline,
)
from .spaces import SigmaSpace


class ArithTensor(Protocol):
    """The arithmetic a tensor must support for sampling math: scaling
    by floats and elementwise combination with peers. torch.Tensor
    satisfies this structurally; tests use plain value types. Scalar
    transcendentals (log/exp/sqrt) happen on float sigmas, never on
    tensors, so this is the whole demand.

    Solver-specific operations live in narrow runtime-checkable
    capability protocols below, so unrelated tensor implementations
    need only provide the arithmetic used by the common solver path."""

    def __add__(self, other: Self | float) -> Self: ...

    def __sub__(self, other: Self | float) -> Self: ...

    def __mul__(self, other: Self | float) -> Self: ...


CapabilityT = TypeVar("CapabilityT", bound=ArithTensor)
CapabilityT_co = TypeVar("CapabilityT_co", bound=ArithTensor, covariant=True)


@runtime_checkable
class DivTensor(Protocol[CapabilityT]):
    """Tensor division required by only a few specialized solvers."""

    def __truediv__(self, other: CapabilityT | float) -> CapabilityT: ...


@runtime_checkable
class AdaptiveTensor(DivTensor[CapabilityT], Protocol[CapabilityT]):
    """Extra tensor operations required only by ``dpm_adaptive``."""

    def abs(self) -> CapabilityT: ...

    def maximum(self, other: CapabilityT, /) -> CapabilityT: ...

    def clamp(self, minimum: float, maximum: float, /) -> CapabilityT: ...

    def numel(self) -> int: ...

    def sum(self) -> Any: ...


@runtime_checkable
class LCMClipTensor(Protocol[CapabilityT_co]):
    """Extra tensor operations required only by LCM noise clipping."""

    def clamp(self, minimum: float, maximum: float, /) -> CapabilityT_co: ...

    def std(self) -> Any: ...


TensorT = TypeVar("TensorT", bound=ArithTensor)
StateT = TypeVar("StateT")
_ResultT = TypeVar("_ResultT")
TensorT_co = TypeVar("TensorT_co", bound=ArithTensor, covariant=True)


class Parameterization(Enum):
    """What the diffusion core's raw output means (comfy/model_sampling.py
    prediction mixins @ b78cec87)."""

    EPS = "eps"
    V_PREDICTION = "v_prediction"
    EDM = "edm"
    FLOW = "flow"
    IMAGE_TO_IMAGE_FLOW = "image_to_image_flow"
    X0 = "x0"


def is_flow_parameterization(parameterization: Parameterization) -> bool:
    """Whether solvers must use the reference's CONST flow branches."""

    return parameterization in (
        Parameterization.FLOW,
        Parameterization.IMAGE_TO_IMAGE_FLOW,
    )


class SamplingSpace(Enum):
    """The sigma-space implementation selected by a model variant."""

    DISCRETE = "discrete"
    CONTINUOUS_EDM = "continuous_edm"


@dataclass(frozen=True)
class SamplingDescriptor:
    """A model family's sampling parameters: how its sigmas span and what
    its outputs mean. Schedule-family specifics (beta tables, flow shift
    curves) live in the spaces module's SigmaSpace implementations."""

    parameterization: Parameterization
    sigma_min: float
    sigma_max: float
    shift: float = 1.0
    zsnr: bool = field(default=False, repr=False)
    space: SamplingSpace = field(default=SamplingSpace.DISCRETE, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_min) or not math.isfinite(self.sigma_max):
            raise ValueError("sigma range must be finite")
        if self.sigma_min <= 0 or self.sigma_max <= 0:
            raise ValueError("sigma range must be positive (schedules append the terminal 0.0)")
        if self.sigma_min >= self.sigma_max:
            raise ValueError(f"sigma_min ({self.sigma_min}) must be < sigma_max ({self.sigma_max})")
        if self.zsnr and self.space is not SamplingSpace.DISCRETE:
            raise ValueError("zero-terminal-SNR is only valid for a discrete sigma space")


@dataclass(frozen=True, slots=True)
class SamplingSegment:
    """One validated contiguous segment of a full sampling schedule.

    A sampling call must use the same ``steps`` value or refuse before it
    builds the schedule.
    """

    steps: int
    start_step: int
    end_step: int
    add_noise: bool
    return_with_leftover_noise: bool

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("sampling segment steps must be a positive integer")
        if type(self.start_step) is not int or type(self.end_step) is not int:
            raise TypeError("sampling segment bounds must be exact integers")
        if not 0 <= self.start_step < self.end_step <= self.steps:
            raise ValueError("sampling segment requires 0 <= start_step < end_step <= steps")
        if type(self.add_noise) is not bool or type(self.return_with_leftover_noise) is not bool:
            raise TypeError("sampling segment noise policies must be exact booleans")


class Denoiser(Protocol[TensorT]):
    """What a solver calls: denoised prediction for ``x`` at ``sigma``.

    ``sigma`` is a plain float - solvers are torch-free, so the
    executing denoiser owns lifting it to whatever batch tensor its
    framework wants (k_diffusion does ``model(x, sigma * s_in)`` with
    ``s_in = ones([batch])`` @ b78cec87; that lift is wrapper business,
    not solver business). Conditioning, CFG, batching, control - all of
    that lives behind this callable (stage 4+ builds it); solvers never
    see through it.
    """

    def __call__(self, x: TensorT, sigma: float) -> TensorT: ...


@runtime_checkable
class SamplingCache(Protocol[TensorT]):
    """Invocation-scoped denoiser cache applied by the sampling engine."""

    def wrap_denoiser(
        self,
        denoiser: Denoiser[TensorT],
        sigmas: Sequence[float],
        info: SamplerInfo,
    ) -> Denoiser[TensorT]: ...


@runtime_checkable
class AutoregressiveDenoiser(Protocol[TensorT]):
    """Denoiser capability for model-owned temporal block sampling."""

    def sample_autoregressive(
        self,
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        num_frame_per_block: int,
        on_step: StepCallback | None = None,
    ) -> TensorT: ...


@runtime_checkable
class UncondDenoiser(Protocol[TensorT]):
    """The optional denoiser capability CFG++ solvers require: one
    evaluation returning both the CFG-combined denoised prediction AND
    the raw unconditioned one.

    The reference smuggles the uncond out of sampling_function through
    a post-CFG hook closure (set_model_options_post_cfg_function with
    disable_cfg1_optimization=True, comfy/k_diffusion/sampling.py
    sample_*_cfg_pp @ b78cec87); here it is a declared method on the
    executing denoiser. Calling it forces the uncond evaluation even at
    cfg 1 - the hook's disable_cfg1_optimization flag as method
    semantics - and with no uncond conditioning at all the uncond slot
    is all zeros, exactly the reference's untouched calc_cond_batch
    accumulator (comfy/samplers.py @ b78cec87). Descriptors whose
    solver needs this declare ``needs_uncond``; the solver itself
    narrows via isinstance and refuses plain Denoisers loudly.
    """

    def call_with_uncond(self, x: TensorT, sigma: float) -> tuple[TensorT, TensorT]:
        """(cfg-combined denoised, uncond denoised) for ``x`` at ``sigma``."""
        ...


class NoiseSampler(Protocol[TensorT_co]):
    """Fresh noise for the step from ``sigma_from`` down to ``sigma_to``,
    shaped like the latent being sampled.

    The injected replacement for k_diffusion's default_noise_sampler
    (seeded gaussian) and BrownianTreeNoiseSampler (@ b78cec87): the
    executing backend constructs the right kind (see NoiseKind on the
    descriptor) with its own framework and seed; solvers only call it.
    Deviation from the reference: euler/heun/dpm_2 churn noise was
    hardcoded ``torch.randn_like`` there - here it comes through this
    same injected callable, so churn is seedable and deterministic.
    """

    def __call__(self, sigma_from: float, sigma_to: float) -> TensorT_co: ...


class NoiseKind(Enum):
    """What noise a solver consumes, declared as data on its descriptor
    so the executing backend knows which NoiseSampler to construct."""

    NONE = "none"
    """Fully deterministic - no NoiseSampler needed."""
    GAUSSIAN = "gaussian"
    """Independent unit gaussian per call (default_noise_sampler
    @ b78cec87). euler/heun/dpm_2 need it only when s_churn > 0."""
    BROWNIAN = "brownian"
    """Brownian-tree increments scaled by 1/sqrt(|t1-t0|)
    (BrownianTreeNoiseSampler @ b78cec87)."""
    BROWNIAN_GPU = "brownian_gpu"
    """Same Brownian-tree math, but the tree (and its RNG stream) lives
    on the latent's device instead of the CPU (BrownianTreeNoiseSampler
    cpu=False, the *_gpu samplers' only difference @ b78cec87). The
    solver math is identical; only the noise stream differs, and it is
    device-RNG-dependent, so cross-device reproducibility is
    deliberately NOT promised - exactly the reference's contract."""
    RES4LYF_GAUSSIAN = "res4lyf_gaussian"
    """Two seeded float64 gaussian streams, globally standardized at
    generation and channelwise zscore-normalized per draw, for the
    RES4LYF RK engine's step and substep noise swaps (NoiseGenerator
    gaussian + normalize_zscore @ 26036f64). The outer stream is seeded
    at run seed + 1 and the substep stream at run seed + 10001, matching
    the reference's seeded path. The constructed sampler satisfies the
    RK engine's RKNoiseSampler protocol."""


@dataclass(frozen=True)
class SolverStateEvent(Generic[StateT]):
    """Packed solver state at one algorithm-defined callback point."""

    step: int
    total: int
    sigma: float
    phase: Literal["pre_update", "post_update"]
    current: StateT
    denoised: StateT | None = None


SolverStateCallback = Callable[[SolverStateEvent[object]], None]


@dataclass(frozen=True)
class SamplerInfo:
    """Facts a solver may legitimately need about the model - passed as
    data, replacing ComfyUI solvers' ``model.inner_model.inner_model``
    capability probing (comfy/k_diffusion/sampling.py @ b78cec87)."""

    parameterization: Parameterization
    seed: int = 0
    noise_scale: float = 1.0
    """Multiplies s_noise in ancestral/SDE solvers - the reference's
    ``getattr(model_sampling, "noise_scale", 1.0)`` probe as data
    (comfy/k_diffusion/sampling.py @ b78cec87)."""
    percent_to_sigma: Callable[[float], float] | None = None
    """The model's schedule-percent -> sigma map (SigmaSpace
    .percent_to_sigma), needed only by sa_solver's default stochastic
    interval (percent_to_sigma(0.2)/(0.8), sample_sa_solver @ b78cec87
    - another model_sampling probe carried as data). Solvers that need
    it and find None refuse loudly instead of guessing."""
    sigma_min: float | None = None
    """The model's smallest positive sigma (SigmaSpace.sigma_min),
    needed by the RES4LYF RK engine's schedule preprocessing and
    BONGMATH guard - the reference's ``model_sampling.sigma_min`` probe
    as data. Solvers that need it and find None refuse loudly."""
    sigma_max: float | None = None
    """The model's largest sigma (SigmaSpace.sigma_max), needed by the
    RES4LYF RK engine's variance-preserving SDE coefficients and
    BONGMATH guard. Solvers that need it and find None refuse loudly."""
    on_state: SolverStateCallback | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class StepEvent:
    """Per-step progress: the typed form of k_diffusion's callback dict
    ``{x, i, sigma, sigma_hat, denoised}`` minus the tensors (previews
    subscribe to a richer stage-3 event carrying them)."""

    step: int
    total: int
    sigma: float


StepCallback = Callable[[StepEvent], None]
StepBeginCallback = Callable[[int], None]


@dataclass(frozen=True)
class SamplingStateEvent(Generic[StateT]):
    """Owned sampler state snapshot for previews and diagnostics."""

    step: int
    total: int
    sigma: float
    phase: Literal["pre_update", "post_update"]
    current: StateT
    denoised: StateT | None = None


SamplingStateCallback = Callable[[SamplingStateEvent[object]], None]


class SamplingCancelled(RuntimeError):
    """Cooperative sampling cancellation observed at a safe loop point."""


_additional_sampling_cancellation: ContextVar[Callable[[], bool] | None] = ContextVar(
    "dinkster_additional_sampling_cancellation", default=None
)


@dataclass(frozen=True)
class CancellationToken:
    """Invocation-scoped cooperative cancellation."""

    cancelled: Callable[[], bool]

    def check(self) -> None:
        additional = _additional_sampling_cancellation.get()
        if (additional is not None and additional()) or self.cancelled():
            raise SamplingCancelled("sampling cancelled")


@dataclass(frozen=True)
class ProgressScope:
    """The existing flat step callback lifted into the execution context."""

    cancellation: CancellationToken
    on_step: StepCallback | None = None
    on_state: SamplingStateCallback | None = None

    def report(self, event: StepEvent, state: SamplingStateEvent[object] | None = None) -> None:
        self.cancellation.check()
        if state is not None and self.on_state is not None:
            self.on_state(state)
        if self.on_step is not None:
            self.on_step(event)
        self.cancellation.check()


@dataclass(frozen=True)
class SamplingExecutionContext:
    """Honest invocation-local facts available to extension samplers.

    The context fields and extension namespace set are frozen for one call.
    Each namespace contains mutable invocation scratch shared by every model
    evaluation in that call; session state is deliberately not represented in
    S1.
    """

    sigma_schedule: tuple[float, ...]
    outer_step: int
    model_evaluation: int
    current_sigma: float
    seed: int
    cancellation: CancellationToken
    progress: ProgressScope
    extension_state: Mapping[str, MutableMapping[str, object]]

    def __post_init__(self) -> None:
        if self.outer_step < 0 or self.model_evaluation < 0:
            raise ValueError("sampling ordinals must be non-negative")
        object.__setattr__(
            self,
            "sigma_schedule",
            tuple(float(sigma) for sigma in self.sigma_schedule),
        )
        frozen_state: dict[str, MutableMapping[str, object]] = {}
        for extension_id, state in self.extension_state.items():
            if not extension_id:
                raise ValueError("extension state ids must be non-empty")
            frozen_state[extension_id] = (
                state
                if isinstance(state, _ExtensionStateNamespace)
                else _ExtensionStateNamespace(state)
            )
        object.__setattr__(self, "extension_state", MappingProxyType(frozen_state))

    def derive_seed(self, purpose: str) -> int:
        """Derive one stable unsigned 64-bit seed for a named RNG stream."""
        if not purpose:
            raise ValueError("RNG stream purpose must be non-empty")
        payload = f"dinkster.sampling-rng-v1\0{self.seed}\0{purpose}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class ModelEvaluation(Generic[TensorT]):
    """One denoiser result paired with the exact context of that call."""

    value: TensorT
    context: SamplingExecutionContext


class ContextDenoiser(Protocol[TensorT]):
    """Denoiser adapter that owns the model-evaluation ordinal."""

    def __call__(
        self, x: TensorT, sigma: float, *, outer_step: int
    ) -> ModelEvaluation[TensorT]: ...


class ContextSolverFn(Protocol[TensorT]):
    """Extension solver shape; the established SolverFn remains unchanged."""

    def __call__(
        self,
        denoiser: ContextDenoiser[TensorT],
        x: TensorT,
        context: SamplingExecutionContext,
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None = None,
    ) -> TensorT: ...


class SolverFn(Protocol[TensorT]):
    """One sampling algorithm: drive ``x`` through ``sigmas``.

    The typed equivalent of k_diffusion's ``sample_*(model, x, sigmas,
    extra_args, callback, disable)`` signature (@ b78cec87); per-solver
    options are bound at construction by the descriptor's factory.
    ``noise`` is required exactly when the descriptor's NoiseKind (or
    an option like s_churn > 0) demands it - stochastic solvers raise
    a loud ValueError instead of silently sampling nothing.
    """

    def __call__(
        self,
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None = None,
        on_step: StepCallback | None = None,
    ) -> TensorT: ...


class StepBeginSolverFn(Protocol[TensorT]):
    """Internal solver capability for engine-owned outer-step activation."""

    def __call__(
        self,
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: StepBeginCallback | None = None,
    ) -> TensorT: ...


@dataclass(frozen=True)
class _StepBeginSolverAdapter(Generic[TensorT]):
    solver: StepBeginSolverFn[TensorT]

    def __call__(
        self,
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None = None,
        on_step: StepCallback | None = None,
    ) -> TensorT:
        return self.solver(denoiser, x, sigmas, info, noise=noise, on_step=on_step)

    def run_with_step_begin(
        self,
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None,
        on_step: StepCallback | None,
        on_step_begin: StepBeginCallback,
    ) -> TensorT:
        return self.solver(
            denoiser,
            x,
            sigmas,
            info,
            noise=noise,
            on_step=on_step,
            on_step_begin=on_step_begin,
        )


@dataclass(frozen=True)
class _SamplingCacheSolverAdapter(Generic[TensorT]):
    solver: SolverFn[TensorT]
    cache: SamplingCache[TensorT]

    def __call__(
        self,
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None = None,
        on_step: StepCallback | None = None,
    ) -> TensorT:
        return self.solver(
            self.cache.wrap_denoiser(denoiser, sigmas, info),
            x,
            sigmas,
            info,
            noise=noise,
            on_step=on_step,
        )

    def run_with_step_begin(
        self,
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None,
        on_step: StepCallback | None,
        on_step_begin: StepBeginCallback,
    ) -> TensorT:
        return run_step_begin_solver(
            self.solver,
            self.cache.wrap_denoiser(denoiser, sigmas, info),
            x,
            sigmas,
            info,
            noise=noise,
            on_step=on_step,
            on_step_begin=on_step_begin,
        )


@dataclass(frozen=True)
class _SamplingTimelineSolverAdapter(Generic[TensorT]):
    solver: SolverFn[TensorT]
    schedule: SamplingTimelineSchedule
    realized: RealizedSamplingTimeline | None = None

    def _timeline(self, sigmas: Sequence[float]) -> RealizedSamplingTimeline:
        executed = executed_sampling_timeline(tuple(float(sigma) for sigma in sigmas))
        if self.realized is None:
            return realize_sampling_timeline(
                self.schedule,
                tuple(float(sigma) for sigma in sigmas),
            )
        if (
            self.realized.schedule_digest != self.schedule.digest
            or self.realized.executed != executed
        ):
            raise ValueError("realized sampling timeline does not match the executed request")
        return self.realized

    def __call__(
        self,
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None = None,
        on_step: StepCallback | None = None,
    ) -> TensorT:
        if len(sigmas) < 2:
            return self.solver(
                denoiser,
                x,
                sigmas,
                info,
                noise=noise,
                on_step=on_step,
            )
        timeline = self._timeline(sigmas)
        with use_realized_sampling_timeline(timeline) as activate:
            return run_step_begin_solver(
                self.solver,
                denoiser,
                x,
                sigmas,
                info,
                noise=noise,
                on_step=on_step,
                on_step_begin=activate,
            )

    def run_with_step_begin(
        self,
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None,
        on_step: StepCallback | None,
        on_step_begin: StepBeginCallback,
    ) -> TensorT:
        if len(sigmas) < 2:
            return run_step_begin_solver(
                self.solver,
                denoiser,
                x,
                sigmas,
                info,
                noise=noise,
                on_step=on_step,
                on_step_begin=on_step_begin,
            )
        timeline = self._timeline(sigmas)
        with use_realized_sampling_timeline(timeline) as activate:

            def activate_and_forward(step_index: int) -> None:
                activate(step_index)
                on_step_begin(step_index)

            return run_step_begin_solver(
                self.solver,
                denoiser,
                x,
                sigmas,
                info,
                noise=noise,
                on_step=on_step,
                on_step_begin=activate_and_forward,
            )


def solver_supports_step_begin(solver: SolverFn[TensorT]) -> bool:
    """Whether a descriptor explicitly adapted this solver for step activation."""
    if type(solver) in (_SamplingCacheSolverAdapter, _SamplingTimelineSolverAdapter):
        return solver_supports_step_begin(cast("Any", solver).solver)
    return type(solver) is _StepBeginSolverAdapter


def solver_has_sampling_timeline(solver: SolverFn[TensorT]) -> bool:
    if type(solver) is _SamplingTimelineSolverAdapter:
        return True
    if type(solver) is _SamplingCacheSolverAdapter:
        return solver_has_sampling_timeline(
            cast("_SamplingCacheSolverAdapter[TensorT]", solver).solver
        )
    return False


def solver_sampling_cache(solver: SolverFn[TensorT]) -> SamplingCache[TensorT] | None:
    if type(solver) is _SamplingCacheSolverAdapter:
        return cast("_SamplingCacheSolverAdapter[TensorT]", solver).cache
    if type(solver) is _SamplingTimelineSolverAdapter:
        return solver_sampling_cache(cast("_SamplingTimelineSolverAdapter[TensorT]", solver).solver)
    return None


def run_step_begin_solver(
    solver: SolverFn[TensorT],
    denoiser: Denoiser[TensorT],
    x: TensorT,
    sigmas: Sequence[float],
    info: SamplerInfo,
    *,
    noise: NoiseSampler[TensorT] | None,
    on_step: StepCallback | None,
    on_step_begin: StepBeginCallback,
) -> TensorT:
    """Invoke one solver after its explicit step-begin capability was checked."""
    if type(solver) in (_SamplingCacheSolverAdapter, _SamplingTimelineSolverAdapter):
        return cast("Any", solver).run_with_step_begin(
            denoiser,
            x,
            sigmas,
            info,
            noise=noise,
            on_step=on_step,
            on_step_begin=on_step_begin,
        )
    if type(solver) is not _StepBeginSolverAdapter:
        raise TypeError("solver does not declare step-begin capability")
    return cast("_StepBeginSolverAdapter[TensorT]", solver).run_with_step_begin(
        denoiser,
        x,
        sigmas,
        info,
        noise=noise,
        on_step=on_step,
        on_step_begin=on_step_begin,
    )


@dataclass(frozen=True)
class _SamplingEnvironment:
    extension_ids: tuple[str, ...]
    cancelled: Callable[[], bool]


class _ExtensionStateNamespace(dict[str, object]):
    """One invocation's mutable scratch, shared by derived contexts."""


_sampling_environment: ContextVar[_SamplingEnvironment | None] = ContextVar(
    "dinkster_sampling_environment", default=None
)
_sampling_cache: ContextVar[SamplingCache[Any] | None] = ContextVar(
    "dinkster_sampling_cache", default=None
)
_sampling_timeline: ContextVar[SamplingTimelineSchedule | None] = ContextVar(
    "dinkster_sampling_timeline", default=None
)


def _not_cancelled() -> bool:
    return False


@final
class CancellationFlag:
    """Set-only cooperative cancellation whose check executes no caller
    code: ``cancel()`` latches one slot bool and calling the flag reads
    it. Call dispatch goes through the type-owned ``__call__`` and
    ``__slots__`` leaves no instance dict, so no per-instance state can
    shadow the check with caller code. The read compares the slot value
    by identity to ``True`` rather than truth-testing it: the slot
    itself is writable, and a foreign object planted there must never
    have its ``__bool__`` executed by a receipt-gated consumer - any
    non-``True`` content simply reads as not cancelled."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def __call__(self) -> bool:
        return self._cancelled is True


@contextmanager
def use_sampling_environment(extension_ids: Sequence[str], cancelled: Callable[[], bool]):
    """Install invocation-local extension namespaces and cancellation."""
    normalized = tuple(sorted(set(extension_ids)))
    token = _sampling_environment.set(_SamplingEnvironment(normalized, cancelled))
    try:
        yield
    finally:
        _sampling_environment.reset(token)


@contextmanager
def use_additional_sampling_cancellation(cancelled: Callable[[], bool]):
    """Add a dynamic cancellation source to every token checked in this scope."""
    existing = _additional_sampling_cancellation.get()

    def combined() -> bool:
        return cancelled() or (existing is not None and existing())

    token = _additional_sampling_cancellation.set(combined)
    try:
        yield
    finally:
        _additional_sampling_cancellation.reset(token)


@contextmanager
def use_sampling_cache(cache: SamplingCache[Any] | None):
    """Bind one immutable cache configuration while a request is constructed."""
    if cache is not None and not isinstance(cast("object", cache), SamplingCache):
        raise TypeError("sampling cache must implement SamplingCache")
    token = _sampling_cache.set(cache)
    try:
        yield
    finally:
        _sampling_cache.reset(token)


@contextmanager
def use_sampling_timeline(schedule: SamplingTimelineSchedule | None):
    """Bind one immutable timeline declaration while a request is constructed."""
    if schedule is not None and type(schedule) is not SamplingTimelineSchedule:
        raise TypeError("sampling timeline must be an exact SamplingTimelineSchedule")
    token = _sampling_timeline.set(schedule)
    try:
        yield
    finally:
        _sampling_timeline.reset(token)


def sampling_environment_cancellation() -> Callable[[], bool]:
    """The ambient sampling environment's cancellation callable.

    Custom-sampling runtimes resolve an omitted ``cancelled`` argument
    here, so cooperative cancellation installed by the executing node
    layer (:func:`use_sampling_environment`) reaches the solver loop
    without a dedicated seam parameter. Absent an environment, the
    default no-op callable is returned."""
    environment = _sampling_environment.get()
    return _not_cancelled if environment is None else environment.cancelled


def sampling_cancellation_is_trusted() -> bool:
    """True when the ambient sampling environment's cancellation check
    executes no caller code: the environment is absent, carries the
    default no-op, or carries an exact :class:`CancellationFlag`.
    Receipt-gated distributed sampling consults this because a
    cancellation callable is invoked between solver steps and an
    arbitrary closure there could mutate receipt-covered state."""
    environment = _sampling_environment.get()
    if environment is None:
        return True
    cancelled = environment.cancelled
    return cancelled is _not_cancelled or type(cancelled) is CancellationFlag


def sampling_execution_context(
    sigmas: Sequence[float],
    seed: int,
    on_step: StepCallback | None = None,
    on_state: SamplingStateCallback | None = None,
) -> SamplingExecutionContext:
    """Build the invocation context used by model-side extension phases."""
    environment = _sampling_environment.get()
    extension_ids = environment.extension_ids if environment is not None else ()
    cancelled = environment.cancelled if environment is not None else _not_cancelled
    cancellation = CancellationToken(cancelled)
    schedule = tuple(float(sigma) for sigma in sigmas)
    return SamplingExecutionContext(
        sigma_schedule=schedule,
        outer_step=0,
        model_evaluation=0,
        current_sigma=schedule[0] if schedule else 0.0,
        seed=seed,
        cancellation=cancellation,
        progress=ProgressScope(cancellation, on_step, on_state),
        extension_state={extension_id: {} for extension_id in extension_ids},
    )


class _ContextDenoiser(Generic[TensorT]):
    def __init__(
        self,
        denoiser: Denoiser[TensorT],
        base: SamplingExecutionContext,
        on_step_begin: StepBeginCallback | None,
    ) -> None:
        self._denoiser = denoiser
        self._base = base
        self._evaluation = 0
        self._on_step_begin = on_step_begin
        self._current_step: int | None = None

    def __call__(self, x: TensorT, sigma: float, *, outer_step: int) -> ModelEvaluation[TensorT]:
        self._base.cancellation.check()
        if outer_step != self._current_step:
            if self._on_step_begin is not None:
                self._on_step_begin(outer_step)
            self._current_step = outer_step
        context = SamplingExecutionContext(
            sigma_schedule=self._base.sigma_schedule,
            outer_step=outer_step,
            model_evaluation=self._evaluation,
            current_sigma=float(sigma),
            seed=self._base.seed,
            cancellation=self._base.cancellation,
            progress=self._base.progress,
            extension_state=self._base.extension_state,
        )
        self._evaluation += 1
        value = self._denoiser(x, sigma)
        context.cancellation.check()
        return ModelEvaluation(value, context)


def _adapt_context_solver(
    solver: ContextSolverFn[TensorT],
) -> SolverFn[TensorT]:
    def adapted(
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: StepBeginCallback | None = None,
    ) -> TensorT:
        environment = _sampling_environment.get()
        extension_ids = environment.extension_ids if environment is not None else ()
        cancelled = environment.cancelled if environment is not None else _not_cancelled
        cancellation = CancellationToken(cancelled)
        schedule = tuple(float(sigma) for sigma in sigmas)
        progress = ProgressScope(cancellation, on_step)
        context = SamplingExecutionContext(
            sigma_schedule=schedule,
            outer_step=0,
            model_evaluation=0,
            current_sigma=schedule[0] if schedule else 0.0,
            seed=info.seed,
            cancellation=cancellation,
            progress=progress,
            extension_state={extension_id: {} for extension_id in extension_ids},
        )
        cancellation.check()
        result = solver(
            _ContextDenoiser(denoiser, context, on_step_begin),
            x,
            context,
            info,
            noise=noise,
        )
        cancellation.check()
        return result

    return adapted


OptionValue = float | int | bool | str | None
"""A resolved solver option value: FLOAT options carry floats, CHOICE
options carry one of their declared strings."""


class OptionKind(Enum):
    """The closed set of option value shapes solvers actually use.
    Grows only when a ported solver needs a new one."""

    FLOAT = "float"
    INT = "int"
    OPTIONAL_FLOAT = "optional_float"
    CHOICE = "choice"
    BOOL = "bool"


@dataclass(frozen=True)
class OptionSpec:
    """One typed solver option: the schema KSAMPLER's untyped
    ``extra_options`` dict never had (comfy/samplers.py @ b78cec87).

    ``name`` is bare (``eta``, not ``dinkster.eta``) - options are scoped
    by their descriptor. FLOAT options may carry inclusive bounds;
    CHOICE options carry the closed value set.
    """

    name: str
    kind: OptionKind
    default: OptionValue
    doc: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or "." in self.name:
            raise ValueError(f"option name must be bare and nonempty, got {self.name!r}")
        if self.kind is OptionKind.FLOAT:
            if self.choices:
                raise ValueError(f"option {self.name!r}: FLOAT options carry no choices")
            self.check(self.default)
        elif self.kind is OptionKind.OPTIONAL_FLOAT:
            if self.choices:
                raise ValueError(f"option {self.name!r}: OPTIONAL_FLOAT options carry no choices")
            self.check(self.default)
        elif self.kind is OptionKind.INT:
            if self.choices:
                raise ValueError(f"option {self.name!r}: INT options carry no choices")
            self.check(self.default)
        elif self.kind is OptionKind.CHOICE:
            if not self.choices:
                raise ValueError(f"option {self.name!r}: CHOICE options need choices")
            if self.minimum is not None or self.maximum is not None:
                raise ValueError(f"option {self.name!r}: CHOICE options carry no bounds")
            self.check(self.default)
        elif self.kind is OptionKind.BOOL:
            if self.choices or self.minimum is not None or self.maximum is not None:
                raise ValueError(f"option {self.name!r}: BOOL options carry no choices or bounds")
            self.check(self.default)

    def check(self, value: object) -> OptionValue:
        """Validate one value against this spec; returns it normalized
        (ints become floats). Loud on type, bound, and choice errors."""
        if self.kind is OptionKind.OPTIONAL_FLOAT and value is None:
            return None
        if self.kind in (OptionKind.FLOAT, OptionKind.OPTIONAL_FLOAT):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"option {self.name!r} expects a float, got {value!r}")
            out = float(value)
            if math.isnan(out):
                raise ValueError(f"option {self.name!r} must not be NaN")
            if self.minimum is not None and out < self.minimum:
                raise ValueError(f"option {self.name!r} must be >= {self.minimum}, got {out}")
            if self.maximum is not None and out > self.maximum:
                raise ValueError(f"option {self.name!r} must be <= {self.maximum}, got {out}")
            return out
        if self.kind is OptionKind.INT:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"option {self.name!r} expects an int, got {value!r}")
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"option {self.name!r} must be >= {self.minimum}, got {value}")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"option {self.name!r} must be <= {self.maximum}, got {value}")
            return value
        if self.kind is OptionKind.BOOL:
            if not isinstance(value, bool):
                raise ValueError(f"option {self.name!r} expects a bool, got {value!r}")
            return value
        if not isinstance(value, str) or value not in self.choices:
            raise ValueError(f"option {self.name!r} expects one of {self.choices}, got {value!r}")
        return value


def resolve_options(
    schema: Sequence[OptionSpec], overrides: Mapping[str, object]
) -> dict[str, OptionValue]:
    """Defaults overlaid with validated overrides; unknown names are
    loud errors, never silently dropped (KSAMPLER passed extra_options
    blind into ``**kwargs`` @ b78cec87 - a typo'd option name there
    was a TypeError deep in the solver, or worse, accepted)."""
    by_name = {spec.name: spec for spec in schema}
    if len(by_name) != len(schema):
        raise ValueError("option schema has duplicate names")
    unknown = sorted(set(overrides) - set(by_name))
    if unknown:
        known = ", ".join(sorted(by_name)) or "(none)"
        raise ValueError(f"unknown option(s) {unknown}; known: {known}")
    resolved: dict[str, OptionValue] = {spec.name: spec.default for spec in schema}
    for name, value in overrides.items():
        resolved[name] = by_name[name].check(value)
    return resolved


@dataclass(frozen=True)
class SamplerDescriptor(Generic[TensorT]):
    """A registered sampling algorithm.

    ``aliases`` carry legacy ComfyUI names ("euler", "dpmpp_2m") so
    ported workflows resolve; ``options`` is the typed schema of what
    ``make`` accepts; ``noise`` declares what NoiseSampler the executing
    backend must construct. ``discard_penultimate`` declares the sigma
    correction the reference keeps in a hardcoded name set
    (KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS @ b78cec87); step
    planning (steps.sampling_sigmas) consumes it. Build solvers through
    :meth:`build`, which validates against the schema - ``make`` itself
    trusts its input.
    """

    id: str
    display_name: str
    make: (
        Callable[[Mapping[str, OptionValue]], SolverFn[TensorT]]
        | Callable[[Mapping[str, OptionValue]], ContextSolverFn[TensorT]]
    )
    options: tuple[OptionSpec, ...] = ()
    noise: NoiseKind = NoiseKind.NONE
    aliases: tuple[str, ...] = ()
    discard_penultimate: bool = False
    requires_snr_offset: bool = False
    """Flow/CONST models offset the first sigma before this solver runs,
    matching ComfyUI's ``offset_first_sigma_for_snr`` sampler set. Brownian
    tree bounds must still be derived from the pre-offset schedule."""
    needs_uncond: bool = False
    """This solver consumes the unconditioned prediction (CFG++): the
    executing denoiser must provide the UncondDenoiser capability. The
    reference's equivalent is implicit - sample_*_cfg_pp installing a
    post-CFG hook with disable_cfg1_optimization=True @ b78cec87."""
    random_inpaint_noise: bool = False
    """Use a seed+1 CPU float32 noise stream for the masked latent source.
    ComfyUI sets this only for ``sampler_object('ddim')`` through
    ``inpaint_options={'random': True}``; solver math remains Euler."""
    context_aware: bool = False
    """Whether ``make`` builds ContextSolverFn instead of legacy SolverFn."""
    supports_step_begin: bool = False
    """Whether ``make`` accepts the engine's internal outer-step callback."""

    def build(self, **overrides: object) -> SolverFn[TensorT]:
        """A SolverFn with ``overrides`` validated against the option
        schema and defaults filled in."""
        built = self.make(resolve_options(self.options, overrides))
        if self.context_aware:
            solver = _adapt_context_solver(cast("ContextSolverFn[TensorT]", built))
        else:
            solver = cast("SolverFn[TensorT]", built)
        if self.supports_step_begin:
            return _StepBeginSolverAdapter(cast("StepBeginSolverFn[TensorT]", solver))
        return solver


@dataclass(frozen=True, slots=True)
class BuiltinSamplerSelection:
    """Validated immutable selection of one built-in sampler and its options."""

    sampler_id: str
    options: tuple[tuple[str, OptionValue], ...]

    def __post_init__(self) -> None:
        if type(self.sampler_id) is not str or not self.sampler_id:
            raise ValueError("built-in sampler selection requires a sampler id")
        raw_options = cast("object", self.options)
        if not isinstance(raw_options, tuple):
            raise TypeError("built-in sampler options must be a tuple")
        names: list[str] = []
        for raw_item in cast("tuple[object, ...]", raw_options):
            if not isinstance(raw_item, tuple):
                raise TypeError("built-in sampler options must be (name, value) tuples")
            item = cast("tuple[object, ...]", raw_item)
            if len(item) != 2 or type(item[0]) is not str:
                raise TypeError("built-in sampler options must be (name, value) tuples")
            names.append(item[0])
        if len(names) != len(set(names)):
            raise ValueError("built-in sampler option names must be unique")


@dataclass(frozen=True, slots=True)
class CustomSamplingRequest(Generic[TensorT]):
    """One custom sampler invocation over an exact pre-offset sigma sequence."""

    sampler: SamplerDescriptor[TensorT]
    options: tuple[tuple[str, OptionValue], ...]
    sigmas: tuple[float, ...]
    cache: SamplingCache[TensorT] | None = field(default=None, repr=False)
    timeline: SamplingTimelineSchedule | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.sampler) is not SamplerDescriptor:
            raise TypeError("custom sampling requires an exact SamplerDescriptor")
        cache = self.cache
        if cache is None:
            cache = cast("SamplingCache[TensorT] | None", _sampling_cache.get())
            object.__setattr__(self, "cache", cache)
        if cache is not None and not isinstance(cast("object", cache), SamplingCache):
            raise TypeError("custom sampling cache must implement SamplingCache")
        timeline = self.timeline
        if timeline is None:
            timeline = _sampling_timeline.get()
            object.__setattr__(self, "timeline", timeline)
        if timeline is not None and type(timeline) is not SamplingTimelineSchedule:
            raise TypeError("custom sampling timeline must be an exact SamplingTimelineSchedule")
        raw_options = cast("object", self.options)
        valid_options = isinstance(raw_options, tuple)
        if valid_options:
            for raw_item in cast("tuple[object, ...]", raw_options):
                if not isinstance(raw_item, tuple):
                    valid_options = False
                    break
                item = cast("tuple[object, ...]", raw_item)
                if len(item) != 2 or type(item[0]) is not str:
                    valid_options = False
                    break
        if not valid_options:
            raise TypeError("custom sampler options must be (name, value) tuples")
        option_names = tuple(name for name, _value in self.options)
        if len(option_names) != len(set(option_names)):
            raise ValueError("custom sampler option names must be unique")
        resolved = resolve_options(self.sampler.options, dict(self.options))
        object.__setattr__(
            self,
            "options",
            tuple((spec.name, resolved[spec.name]) for spec in self.sampler.options),
        )
        raw_sigmas = cast("object", self.sigmas)
        if not isinstance(raw_sigmas, tuple):
            raise TypeError("custom sampling sigmas must be a tuple")
        normalized: list[float] = []
        for sigma in self.sigmas:
            raw_sigma = cast("object", sigma)
            if isinstance(raw_sigma, bool) or not isinstance(raw_sigma, (int, float)):
                raise TypeError("custom sampling sigmas must contain only real numbers")
            value = float(raw_sigma)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("custom sampling sigmas must be finite and nonnegative")
            normalized.append(value)
        object.__setattr__(self, "sigmas", tuple(normalized))

    def build_solver(
        self,
        *,
        realized_timeline: RealizedSamplingTimeline | None = None,
    ) -> SolverFn[TensorT]:
        if realized_timeline is not None and self.timeline is None:
            raise ValueError("a realized sampling timeline requires a timeline schedule")
        if (
            realized_timeline is not None
            and self.timeline is not None
            and realized_timeline.schedule_digest != self.timeline.digest
        ):
            raise ValueError("realized sampling timeline does not match the timeline schedule")
        solver = self.sampler.build(**dict(self.options))
        if self.cache is not None:
            solver = _SamplingCacheSolverAdapter(solver, self.cache)
        if self.timeline is not None:
            solver = _SamplingTimelineSolverAdapter(solver, self.timeline, realized_timeline)
        return solver


@dataclass(frozen=True, slots=True)
class CustomSamplingResult(Generic[_ResultT]):
    """Custom sampling output and the last denoised state, when one ran.

    The payload is unbounded: single-stream families return one tensor,
    multi-stream families return a :class:`MultiStreamLatent` pack."""

    output: _ResultT
    denoised_output: _ResultT | None


SigmaScheduleFn = Callable[[int, SigmaSpace], tuple[float, ...]]
"""(steps, space) -> descending sigmas, ending at 0.0 - pure."""


@dataclass(frozen=True)
class SchedulerDescriptor:
    """A registered sigma schedule."""

    id: str
    display_name: str
    make_sigmas: SigmaScheduleFn
    aliases: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Catalog snapshots: builtin sampler/scheduler catalogs reconstruct per call
# from values captured at import, before any caller code runs. A shallow
# by-reference snapshot is not enough - a frozen dataclass without slots
# still accepts object.__setattr__, so a shared nested OptionSpec or a
# factory whose __code__ was swapped would ride along into every "fresh"
# descriptor. The snapshot therefore reduces every field to caller-
# unreachable primitives, and callables to import-time behavior evidence
# that is re-verified on every reconstruction.


@dataclass(frozen=True)
class CallableEvidence:
    """Import-time behavior evidence for one catalog callable.

    A function object's identity does not pin its behavior: ``__code__``
    and ``__defaults__`` are reassignable, ``__kwdefaults__`` is a
    mutable dict, and closure cell contents are writable. ``capture``
    records those at import; ``resolve`` re-verifies them and returns
    the function, raising loudly when any changed. Code objects
    themselves are immutable, so identity of ``__code__`` pins the
    executable behavior (module-global rebinding stays out of scope, as
    everywhere else in the receipt gate)."""

    function: Callable[..., Any]
    code: CodeType
    defaults: tuple[Any, ...] | None
    kwdefault_items: tuple[tuple[str, Any], ...] | None
    closure_contents: tuple[Any, ...]

    @staticmethod
    def capture(function: Callable[..., Any]) -> CallableEvidence:
        if type(function) is not FunctionType:
            raise RuntimeError(f"catalog callable {function!r} must be a plain function")
        kwdefaults = function.__kwdefaults__
        return CallableEvidence(
            function=function,
            code=function.__code__,
            defaults=function.__defaults__,
            kwdefault_items=(
                None
                if kwdefaults is None
                else tuple(sorted(kwdefaults.items(), key=lambda item: item[0]))
            ),
            closure_contents=tuple(cell.cell_contents for cell in function.__closure__ or ()),
        )

    def resolve(self) -> Callable[..., Any]:
        function = self.function
        live_closure = tuple(cell.cell_contents for cell in function.__closure__ or ())
        live_kwdefaults = function.__kwdefaults__
        live_items = (
            None
            if live_kwdefaults is None
            else tuple(sorted(live_kwdefaults.items(), key=lambda item: item[0]))
        )
        kept_items = self.kwdefault_items
        intact = (
            function.__code__ is self.code
            and function.__defaults__ is self.defaults
            and len(live_closure) == len(self.closure_contents)
            and all(
                live is kept for live, kept in zip(live_closure, self.closure_contents, strict=True)
            )
            and (live_items is None) == (kept_items is None)
            and (
                live_items is None
                or kept_items is None
                or (
                    len(live_items) == len(kept_items)
                    and all(
                        live_name == kept_name and live_value is kept_value
                        for (live_name, live_value), (kept_name, kept_value) in zip(
                            live_items, kept_items, strict=True
                        )
                    )
                )
            )
        )
        if not intact:
            raise RuntimeError(f"catalog callable {function.__qualname__} was mutated after import")
        return function


@dataclass(frozen=True)
class _OptionSnapshot:
    """One OptionSpec reduced to primitives; rebuilt fresh per catalog call."""

    name: str
    kind: OptionKind
    default: OptionValue
    doc: str
    minimum: float | None
    maximum: float | None
    choices: tuple[str, ...]

    @staticmethod
    def capture(spec: OptionSpec) -> _OptionSnapshot:
        return _OptionSnapshot(
            name=_primitive(spec.name, str),
            kind=spec.kind,
            default=cast(
                "OptionValue",
                spec.default if spec.default is None else _primitive_value(spec.default),
            ),
            doc=_primitive(spec.doc, str),
            minimum=None if spec.minimum is None else _primitive_number(spec.minimum),
            maximum=None if spec.maximum is None else _primitive_number(spec.maximum),
            choices=tuple(_primitive(choice, str) for choice in spec.choices),
        )

    def rebuild(self) -> OptionSpec:
        return OptionSpec(
            name=self.name,
            kind=self.kind,
            default=self.default,
            doc=self.doc,
            minimum=self.minimum,
            maximum=self.maximum,
            choices=self.choices,
        )


@dataclass(frozen=True)
class _TupleSnapshot:
    """A tuple field's element snapshots; rebuilt as a fresh tuple."""

    items: tuple[Any, ...]


_PrimitiveT = TypeVar("_PrimitiveT")


def _primitive(value: object, expected: type[_PrimitiveT]) -> _PrimitiveT:
    if type(value) is not expected:
        raise RuntimeError(f"catalog value {value!r} is not exactly {expected.__name__}")
    return cast("_PrimitiveT", value)


def _primitive_value(value: object) -> object:
    if type(value) not in (str, int, float, bool):
        raise RuntimeError(f"catalog value {value!r} is not a primitive")
    return value


def _primitive_number(value: object) -> float:
    if type(value) not in (int, float):
        raise RuntimeError(f"catalog value {value!r} is not a number")
    return cast("float", value)


def catalog_value_snapshot(value: object) -> object:
    """Capture one descriptor field for later trusted reconstruction."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Enum):
        # Enum members are singletons compared by identity downstream,
        # so keeping the member itself is exact.
        return value
    if type(value) is tuple:
        items = cast("tuple[object, ...]", value)
        return _TupleSnapshot(tuple(catalog_value_snapshot(item) for item in items))
    if type(value) is OptionSpec:
        return _OptionSnapshot.capture(value)
    if type(value) is FunctionType:
        return CallableEvidence.capture(value)
    raise RuntimeError(f"catalog field value {value!r} has no snapshot form")


def catalog_value_rebuild(snapshot: object) -> object:
    """Reconstruct one descriptor field from its import-time snapshot."""
    if type(snapshot) is _TupleSnapshot:
        return tuple(catalog_value_rebuild(item) for item in snapshot.items)
    if type(snapshot) is _OptionSnapshot:
        return snapshot.rebuild()
    if type(snapshot) is CallableEvidence:
        return snapshot.resolve()
    return snapshot


def catalog_descriptor_snapshot(descriptor: object) -> tuple[tuple[str, object], ...]:
    """Snapshot every dataclass field of one pristine catalog constant."""
    return tuple(
        (field_info.name, catalog_value_snapshot(getattr(descriptor, field_info.name)))
        for field_info in dataclass_fields(descriptor)  # type: ignore[arg-type]
    )


def catalog_descriptor_rebuild(
    snapshot: tuple[tuple[str, object], ...],
) -> dict[str, Any]:
    """Field values for one fresh descriptor, verified against import."""
    return {name: catalog_value_rebuild(value) for name, value in snapshot}


def catalog_value_is_canonical(resolved: object, builtin: object) -> bool:
    """True when ``resolved`` carries exactly the behavior of the
    canonical ``builtin`` field value.

    Canonical catalogs rebuild their nested containers fresh on every
    call, so admission cannot demand object identity for tuples and
    OptionSpecs. Behavioral identity is instead structural over exactly
    the shapes the catalog snapshot admits: primitives by exact type and
    value, enum members and functions by identity (never ``==``, which a
    caller can override), tuples element-wise, and OptionSpecs field-wise
    with no extra instance state. Anything outside those shapes refuses."""
    if resolved is None or builtin is None:
        return resolved is builtin
    if type(resolved) is not type(builtin):
        return False
    if isinstance(builtin, Enum):
        return resolved is builtin
    kind = type(builtin)
    if kind in (str, int, float, bool):
        return resolved == builtin
    if kind is tuple:
        resolved_items = cast("tuple[object, ...]", resolved)
        builtin_items = cast("tuple[object, ...]", builtin)
        return len(resolved_items) == len(builtin_items) and all(
            catalog_value_is_canonical(resolved_item, builtin_item)
            for resolved_item, builtin_item in zip(resolved_items, builtin_items, strict=True)
        )
    if kind is OptionSpec:
        field_names = {field_info.name for field_info in dataclass_fields(OptionSpec)}
        state = vars(resolved)
        if set(state) != field_names:
            return False
        return all(
            catalog_value_is_canonical(state[name], getattr(builtin, name)) for name in field_names
        )
    if kind is FunctionType:
        return resolved is builtin
    return False


__all__ = [
    "AutoregressiveDenoiser",
    "BuiltinSamplerSelection",
    "ArithTensor",
    "CallableEvidence",
    "CancellationFlag",
    "CancellationToken",
    "ContextDenoiser",
    "ContextSolverFn",
    "CustomSamplingRequest",
    "CustomSamplingResult",
    "Denoiser",
    "ModelEvaluation",
    "NoiseKind",
    "NoiseSampler",
    "OptionKind",
    "OptionSpec",
    "OptionValue",
    "Parameterization",
    "ProgressScope",
    "SamplerDescriptor",
    "SamplerInfo",
    "SamplingCancelled",
    "SamplingCache",
    "SamplingTimelineSchedule",
    "SamplingDescriptor",
    "SamplingExecutionContext",
    "SamplingSegment",
    "SamplingStateCallback",
    "SamplingStateEvent",
    "SamplingSpace",
    "SchedulerDescriptor",
    "SigmaScheduleFn",
    "SolverFn",
    "SolverStateCallback",
    "SolverStateEvent",
    "StepCallback",
    "StepEvent",
    "UncondDenoiser",
    "catalog_descriptor_rebuild",
    "catalog_descriptor_snapshot",
    "catalog_value_is_canonical",
    "resolve_options",
    "sampling_cancellation_is_trusted",
    "sampling_execution_context",
    "solver_sampling_cache",
    "use_additional_sampling_cancellation",
    "use_sampling_cache",
    "use_sampling_timeline",
    "use_sampling_environment",
]
