"""Ordinary third-party pack fixture for the extension contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    FLOAT32,
    AssemblyRegistration,
    ComponentDescriptor,
    ComponentWiring,
    DetectionEvidence,
    InferenceContribution,
    InputSpec,
    JsonField,
    JsonObjectSchema,
    LatentDescriptor,
    ModelFamily,
    Node,
    NodeSchema,
    OutputSpec,
    PackEvent,
    Parameterization,
    SamplingDescriptor,
    TypeExpr,
    TypeRegistry,
    report_pack_event,
)

VALUE_TYPE = "fixture.extension.value"
FAMILY_ID = "fixture.toy-image"


class ToyFamilyDetector:
    def detect(self, source: Any) -> DetectionEvidence | None:
        key = "toy.denoiser.weight"
        if key not in source.keys():
            return None
        return DetectionEvidence(FAMILY_ID, (key,), {"channels": 4})


TOY_FAMILY = ModelFamily(
    id=FAMILY_ID,
    display_name="Fixture Toy Image",
    detector=ToyFamilyDetector(),
    specificity=100,
    latent=LatentDescriptor(channels=4, scale_factor=1.0, shift_factor=0.0),
    sampling=SamplingDescriptor(Parameterization.FLOW, sigma_min=0.01, sigma_max=1.0),
    wiring=ComponentWiring(),
    supported_dtypes=frozenset({FLOAT32}),
    denoiser="extension_contract_pack:toy_denoiser",
    text_encoder="extension_contract_pack:encode_text",
    latent_codec="extension_contract_pack:decode_latent",
    loader="extension_contract_pack:load_toy",
)


def detect_components(
    source: Any, _path: object, **_options: object
) -> tuple[tuple[str, object], ...]:
    return (("diffusion", {"family": FAMILY_ID}),) if "toy.denoiser.weight" in source.keys() else ()


TOY_COMPONENT = ComponentDescriptor(
    family=TOY_FAMILY,
    detector=detect_components,
    roles=("diffusion",),
    text_encoder_roles=(),
    codec_roles=(),
    loader="extension_contract_pack:load_toy",
    runtime_class="builtins:object",
)


def plan_toy(**_sources: object) -> object:
    return {"family": FAMILY_ID}


TOY_ASSEMBLY = AssemblyRegistration(
    id=FAMILY_ID,
    plan=plan_toy,
    load="extension_contract_pack:load_toy",
)


def toy_denoiser(_runtime: object, _dtype: object, _context: object) -> object:
    from dinkster_inference_torch.sampling_execution import (
        SamplingDenoiserAdapter,
        SamplingDenoiserExecution,
    )

    class Adapter:
        evaluator_identity = "fixture.toy-image.denoiser.v1"

        @staticmethod
        def prepare_conditioning(value: object, _role: object) -> object:
            return value

        @staticmethod
        def evaluate_conditioning(value: Any, _sigma: float, context: Any) -> Any:
            return value * 0.25 + context.embeddings.mean()

        @staticmethod
        def batchable(_conditions: tuple[object, ...]) -> bool:
            return True

        def evaluate_batch(
            self,
            value: Any,
            sigma: float,
            conditions: tuple[Any, ...],
            _context: object | None = None,
        ) -> tuple[Any, ...]:
            return tuple(self.evaluate_conditioning(value, sigma, item) for item in conditions)

        evaluate_conditioning_batch = evaluate_batch

    return SamplingDenoiserExecution(cast("SamplingDenoiserAdapter", Adapter()))


def encode_text(text: str) -> tuple[int, ...]:
    return tuple(text.encode("utf-8"))


def decode_latent(latent: object) -> object:
    return latent


def load_toy(plan: object) -> object:
    return plan


def register_inference() -> InferenceContribution:
    return InferenceContribution(
        families=(TOY_FAMILY,),
        components=(TOY_COMPONENT,),
        assemblies=(TOY_ASSEMBLY,),
    )


register = register_inference


EVENT = PackEvent(
    "fixture.extension-contract.executed",
    JsonObjectSchema(
        (
            JsonField("height", "integer"),
            JsonField("mean", "number"),
            JsonField("width", "integer"),
        )
    ),
)


def route(_request: Mapping[str, object]) -> dict[str, object]:
    return {"message": "Third-party pack route ready"}


def register_types(registry: TypeRegistry) -> None:
    registry.register(
        VALUE_TYPE,
        encode=lambda value: json.dumps(value).encode(),
        decode=lambda data: json.loads(data),
    )


class ExtensionContractValue(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.extension.value",
            display_name="Extension Contract Value",
            inputs=(
                InputSpec("width", TypeExpr.concrete("core.int"), default=13),
                InputSpec("height", TypeExpr.concrete("core.int"), default=7),
            ),
            outputs=(OutputSpec("sample", TypeExpr.concrete(VALUE_TYPE)),),
        )

    @classmethod
    def execute(cls, *, width: int, height: int) -> Mapping[str, object]:
        return cls.outputs(
            sample={
                "width": width,
                "height": height,
                "mean": (width + height) / 40,
            }
        )


class ExtensionContractProof(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.extension.contract",
            display_name="Extension Contract Proof",
            inputs=(InputSpec("sample", TypeExpr.concrete(VALUE_TYPE)),),
            outputs=(OutputSpec("mean", TypeExpr.concrete("core.float")),),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, sample: Mapping[str, int | float]) -> Mapping[str, object]:
        width = int(sample["width"])
        height = int(sample["height"])
        mean = float(sample["mean"])
        report_pack_event(EVENT, {"height": height, "mean": mean, "width": width})
        return cls.outputs(mean=mean)


NODES = (ExtensionContractValue, ExtensionContractProof)
