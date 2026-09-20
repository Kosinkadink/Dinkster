"""Minimal native-arm host used to prove worker-local sampler activation."""

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class KSamplerHost(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ksampler",
            outputs=(OutputSpec("value", TypeExpr.concrete("core.float")),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        from dinkster_inference import Parameterization, SamplerInfo
        from dinkster_workers import current_execution_context

        context = current_execution_context()
        if context is None or context.inference_registries is None:
            return cls.outputs(value=0.0)
        registries = cast("Any", context.inference_registries)
        scheduler = registries.schedulers.get("proof_a.scheduler")
        sampler = registries.samplers.get("proof_a.scaled_euler")
        if scheduler is None or sampler is None:
            return cls.outputs(value=0.0)
        return cls.outputs(
            value=sampler.build()(
                lambda _value, _sigma: 0.0,
                8.0,
                scheduler.make_sigmas(2, object()),
                SamplerInfo(Parameterization.EPS, seed=7),
            )
        )


class CancellationProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete("core.string")
        return NodeSchema(
            node_type="dinkster.cancel_probe",
            inputs=(InputSpec("marker", string),),
            outputs=(OutputSpec("value", string),),
        )

    @classmethod
    def execute(cls, *, marker: str) -> Mapping[str, object]:
        from dinkster_inference import (
            Parameterization,
            SamplerInfo,
            SamplingCancelled,
            use_sampling_environment,
        )
        from dinkster_workers import current_execution_context
        from s1_sampler_pack_b import CONTEXT_PROBE

        context = current_execution_context()
        if context is None:
            raise RuntimeError("cancellation probe requires worker execution context")

        def denoiser(x: float, sigma: float) -> float:
            del x, sigma
            time.sleep(0.01)
            return 0.0

        sigmas = tuple(float(value) for value in range(100, -1, -1))
        try:
            with use_sampling_environment(("proof_b",), context.cancelled):
                CONTEXT_PROBE.build()(
                    denoiser,
                    8.0,
                    sigmas,
                    SamplerInfo(Parameterization.EPS, seed=7),
                )
        except SamplingCancelled:
            Path(marker).write_text("cancelled", encoding="ascii")
            raise
        return cls.outputs(value="completed")


NODES = [KSamplerHost, CancellationProbe]
ARM_NODES = {"native": [KSamplerHost]}


def choices() -> dict[str, tuple[str, ...]]:
    return {
        "comfy.samplers": ("euler",),
        "comfy.schedulers": ("normal",),
    }
