"""Out-of-tree S1 proof pack A: a deterministic scaled Euler sampler."""

from collections.abc import Callable, Mapping, Sequence

from dinkster_api.v1 import (
    NoiseKind,
    NoiseSampler,
    OptionKind,
    OptionSpec,
    OptionValue,
    SamplerContribution,
    SamplerDescriptor,
    SamplerInfo,
    StepCallback,
)


def _make_scaled_euler(options: Mapping[str, OptionValue]):
    raw_scale = options["scale"]
    if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
        raise TypeError("scale must resolve to a float")
    scale = float(raw_scale)

    def solve(
        denoiser,
        x,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ):
        del info, noise
        for step, (sigma, sigma_next) in enumerate(zip(sigmas, sigmas[1:], strict=False)):
            if on_step_begin is not None:
                on_step_begin(step)
            denoised = denoiser(x, sigma)
            if sigma != 0.0:
                x = x + (x - denoised) * (scale * (float(sigma_next) - float(sigma)) / float(sigma))
            if on_step is not None:
                from dinkster_api.v1 import StepEvent

                on_step(StepEvent(step, max(0, len(sigmas) - 1), float(sigma)))
        return x

    return solve


SCALED_EULER = SamplerDescriptor(
    id="proof_a.scaled_euler",
    display_name="S1 Scaled Euler",
    make=_make_scaled_euler,
    options=(OptionSpec("scale", OptionKind.FLOAT, 0.5, minimum=0.0),),
    noise=NoiseKind.NONE,
    aliases=("s1_scaled_euler",),
)


def register() -> SamplerContribution:
    return SamplerContribution((SCALED_EULER,))
