"""Native audio latent construction and dimension inspection."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    first_party_schema,
    initialize_inference_process,
)


def _duration_inputs(
    seconds: float, step: float, *, minimum: float = 1.0, batch_doc: str = ""
) -> tuple[InputSpec, ...]:
    return (
        InputSpec(
            "seconds",
            TypeExpr.concrete(CORE_FLOAT),
            default=seconds,
            widget=NumberWidget(min=minimum, max=1000.0, step=step),
        ),
        InputSpec(
            "batch_size",
            TypeExpr.concrete(CORE_INT),
            default=1,
            widget=NumberWidget(min=1, max=4096),
            doc=batch_doc,
        ),
    )


class EmptyAceStepLatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="EmptyAceStepLatentAudio",
            display_name="Empty Ace Step 1.0 Latent Audio",
            category="model/latent/ace",
            inputs=_duration_inputs(
                120.0, 0.1, batch_doc="The number of latent images in the batch."
            ),
            outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, seconds: float, batch_size: int) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        length = int(seconds * 44100 / 512 / 8)
        latent = torch.zeros(
            [batch_size, 8, 16, length],
            device=management.intermediate_device(),
            dtype=management.intermediate_dtype(),
        )
        return cls.outputs(LATENT={"samples": latent, "type": "audio"})


class EmptyAceStep15LatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="EmptyAceStep1.5LatentAudio",
            display_name="Empty Ace Step 1.5 Latent Audio",
            category="model/latent/ace",
            inputs=_duration_inputs(
                120.0, 0.01, batch_doc="The number of latent images in the batch."
            ),
            outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, seconds: float, batch_size: int) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        length = round(seconds * 48000 / 1920)
        latent = torch.zeros(
            [batch_size, 64, length],
            device=management.intermediate_device(),
            dtype=management.intermediate_dtype(),
        )
        return cls.outputs(
            LATENT={"samples": latent, "type": "audio", "downscale_ratio_temporal": 1764}
        )


class EmptyYuE2LatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="EmptyYuE2LatentAudio",
            display_name="Empty YuE2 Latent Audio",
            category="model/latent/yue2",
            inputs=_duration_inputs(120.0, 0.04, minimum=0.04),
            outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, seconds: float, batch_size: int) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        yue2 = importlib.import_module("dinkster_inference.text_encoders.yue2")
        latent = torch.zeros(
            [batch_size, 64, max(1, round(seconds * yue2.FRAMES_PER_SECOND))],
            device=management.intermediate_device(),
            dtype=management.intermediate_dtype(),
        )
        return cls.outputs(
            LATENT={"samples": latent, "type": "audio", "downscale_ratio_temporal": 1920}
        )
