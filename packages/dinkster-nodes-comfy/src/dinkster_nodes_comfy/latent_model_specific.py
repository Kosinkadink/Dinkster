"""Empty latent layouts for image and 3D model families."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


def _integer(
    name: str, default: int, maximum: int, minimum: int = 1, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_INT),
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _image_inputs() -> tuple[InputSpec, ...]:
    return (
        _integer("width", 1024, 16384, 16, 16),
        _integer("height", 1024, 16384, 16, 16),
        _integer("batch_size", 1, 4096),
    )


def _schema(
    name: str,
    category: str,
    inputs: tuple[InputSpec, ...],
    display: str | None = None,
    description: str = "",
) -> NodeSchema:
    return NodeSchema(
        node_type=name,
        display_name=display or name,
        category=category,
        description=description,
        inputs=inputs,
        outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
    )


def _runtime() -> tuple[Any, Any]:
    initialize_inference_process()
    return importlib.import_module("torch"), importlib.import_module(
        "dinkster_inference.model_management"
    )


class EmptySD3LatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("EmptySD3LatentImage", "model/latent/stable diffusion", _image_inputs())

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            [batch_size, 16, height // 8, width // 8],
            device=management.intermediate_device(),
            dtype=management.intermediate_dtype(),
        )
        return cls.outputs(LATENT={"samples": samples, "downscale_ratio_spacial": 8})


class EmptyFlux2LatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "EmptyFlux2LatentImage", "model/latent/flux", _image_inputs(), "Empty Flux 2 Latent"
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            [batch_size, 128, height // 16, width // 16], device=management.intermediate_device()
        )
        return cls.outputs(LATENT={"samples": samples})


class EmptyChromaRadianceLatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "EmptyChromaRadianceLatentImage", "model/latent/chroma radiance", _image_inputs()
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            (batch_size, 3, height, width), device=management.intermediate_device()
        )
        return cls.outputs(LATENT={"samples": samples})


class EmptyHiDreamO1LatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "EmptyHiDreamO1LatentImage",
            "model/latent/hidream",
            (
                _integer("width", 2048, 4096, 64, 32),
                _integer("height", 2048, 4096, 64, 32),
                _integer("batch_size", 1, 64),
            ),
            "Empty HiDream-O1 Latent Image",
            "Empty pixel-space latent for HiDream-O1-Image. The model was "
            "trained at ~4 megapixels; lower resolutions go off-distribution "
            "and quality regresses noticeably. Trained resolutions: "
            "2048x2048, 2304x1728, 1728x2304, 2560x1440, 1440x2560, "
            "2496x1664, 1664x2496, 3104x1312, 1312x3104, 2304x1792, 1792x2304.",
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            (batch_size, 3, height, width), device=management.intermediate_device()
        )
        return cls.outputs(LATENT={"samples": samples})


class EmptyQwenImageLayeredLatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "EmptyQwenImageLayeredLatentImage",
            "model/latent/qwen",
            (
                _integer("width", 640, 16384, 16, 16),
                _integer("height", 640, 16384, 16, 16),
                _integer("layers", 3, 16384, 0, 1),
                _integer("batch_size", 1, 4096),
            ),
            "Empty Qwen Image Layered Latent",
        )

    @classmethod
    def execute(
        cls, *, width: int, height: int, layers: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros(
            [batch_size, 16, layers + 1, height // 8, width // 8],
            device=management.intermediate_device(),
        )
        return cls.outputs(LATENT={"samples": samples})


class EmptyLatentHunyuan3Dv2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "EmptyLatentHunyuan3Dv2",
            "model/latent/hunyuan 3d",
            (_integer("resolution", 3072, 8192), _integer("batch_size", 1, 4096)),
            "Empty Hunyuan 3D v2 Latent",
        )

    @classmethod
    def execute(cls, *, resolution: int, batch_size: int) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros([batch_size, 64, resolution], device=management.intermediate_device())
        return cls.outputs(LATENT={"samples": samples, "type": "hunyuan3dv2"})


class EmptyTrellis2LatentStructure(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "EmptyTrellis2LatentStructure",
            "model/latent/trellis",
            (_integer("batch_size", 1, 4096),),
            "Empty Trellis2 Latent Structure",
        )

    @classmethod
    def execute(cls, *, batch_size: int) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        samples = torch.zeros(batch_size, 32, 16, 16, 16)
        return cls.outputs(LATENT={"samples": samples, "type": "trellis2"})


MODEL_LATENT_NODES = (
    EmptySD3LatentImage,
    EmptyFlux2LatentImage,
    EmptyChromaRadianceLatentImage,
    EmptyHiDreamO1LatentImage,
    EmptyQwenImageLayeredLatentImage,
    EmptyLatentHunyuan3Dv2,
    EmptyTrellis2LatentStructure,
)
