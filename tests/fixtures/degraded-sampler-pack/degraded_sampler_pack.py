"""Fixture pack that ships samplers alongside ordinary node, route and event surfaces.

Composed on a host with no native sampling worker, its node, route and event
surfaces must still serve while its inference surface is reported unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    InferenceContribution,
    InputSpec,
    JsonField,
    JsonObjectSchema,
    Node,
    NodeSchema,
    NoiseKind,
    NoiseSampler,
    OptionValue,
    OutputSpec,
    PackEvent,
    SamplerDescriptor,
    SamplerInfo,
    SchedulerDescriptor,
    StepCallback,
    TypeExpr,
    report_pack_event,
)

EVENT = PackEvent(
    "degraded.sampler.executed",
    JsonObjectSchema((JsonField("steps", "integer"),)),
)


def _make_solver(_options: Mapping[str, OptionValue]):
    def solve(
        _denoiser,
        x,
        _sigmas: Sequence[float],
        _info: SamplerInfo,
        *,
        noise: NoiseSampler | None = None,
        on_step: StepCallback | None = None,
    ):
        del noise, on_step
        return x

    return solve


SOLVER = SamplerDescriptor(
    id="degraded.fast_solver",
    display_name="Fast Solver",
    make=_make_solver,
    noise=NoiseKind.NONE,
)

SCHEDULE = SchedulerDescriptor(
    id="degraded.stepped_schedule",
    display_name="Stepped Schedule",
    make_sigmas=lambda steps, _space: (float(steps), 0.0),
)


def register_inference() -> InferenceContribution:
    return InferenceContribution(samplers=(SOLVER,), schedulers=(SCHEDULE,))


def route(_request: Mapping[str, object]) -> dict[str, object]:
    return {"message": "Degraded sampler pack route ready"}


class DegradedSolverProbe(Node):
    """Reports which solver id the plan selected, the way a sampler node does.

    The id arrives as an ordinary string input so the composition layer can
    refuse a plan naming a solver that its degraded pack can no longer run.
    """

    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete("core.string")
        return NodeSchema(
            node_type="degraded.solver_probe",
            display_name="Degraded Solver Probe",
            inputs=(InputSpec("solver", string, default="degraded.fast_solver"),),
            outputs=(OutputSpec("solver", string),),
        )

    @classmethod
    def execute(cls, *, solver: str) -> Mapping[str, object]:
        report_pack_event(EVENT, {"steps": 2})
        return cls.outputs(solver=solver)


NODES = (DegradedSolverProbe,)
