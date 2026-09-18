"""Out-of-tree S1 proof pack B: a context-consuming sampler."""

from collections.abc import Mapping

from dinkster_api.v1 import (
    ContextDenoiser,
    NoiseKind,
    NoiseSampler,
    OptionValue,
    SamplerContribution,
    SamplerDescriptor,
    SamplerInfo,
    SamplingExecutionContext,
    StepEvent,
)


def _make_context_probe(options: Mapping[str, OptionValue]):
    if options:
        raise ValueError("context probe has no options")

    def solve(
        denoiser: ContextDenoiser,
        x,
        context: SamplingExecutionContext,
        info: SamplerInfo,
        *,
        noise: NoiseSampler | None = None,
    ):
        del info, noise
        if "proof_b" not in context.extension_state:
            raise RuntimeError("proof_b invocation state is missing")
        state = context.extension_state["proof_b"]
        stream_bit = context.derive_seed("proof-b") & 1
        for step, (sigma, sigma_next) in enumerate(
            zip(
                context.sigma_schedule,
                context.sigma_schedule[1:],
                strict=False,
            )
        ):
            evaluated = denoiser(x, sigma, outer_step=step)
            if evaluated.context.outer_step != step:
                raise RuntimeError("outer step did not reach the denoiser context")
            if evaluated.context.model_evaluation != step:
                raise RuntimeError("model evaluation ordinal did not advance")
            if evaluated.context.current_sigma != sigma:
                raise RuntimeError("current sigma did not reach the denoiser context")
            if evaluated.context.extension_state["proof_b"] is not state:
                raise RuntimeError("invocation state was not shared with model evaluation")
            if state.get("completed_steps", 0) != step:
                raise RuntimeError("invocation state did not persist between steps")
            if sigma != 0.0:
                scale = 0.25 + 0.125 * stream_bit
                x = x + (x - evaluated.value) * (
                    scale * (float(sigma_next) - float(sigma)) / float(sigma)
                )
            context.progress.report(StepEvent(step, len(context.sigma_schedule) - 1, float(sigma)))
            state["completed_steps"] = step + 1
        return x

    return solve


CONTEXT_PROBE = SamplerDescriptor(
    id="proof_b.context_probe",
    display_name="S1 Context Probe",
    make=_make_context_probe,
    noise=NoiseKind.NONE,
    aliases=("s1_context_probe",),
    context_aware=True,
)


def register() -> SamplerContribution:
    return SamplerContribution((CONTEXT_PROBE,))
