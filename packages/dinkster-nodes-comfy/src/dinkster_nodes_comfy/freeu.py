"""FreeU model output-block patches."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_CATEGORY = "model/patch/unet"


def _float(name: str, default: float) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_FLOAT),
        default=default,
        advanced=True,
        widget=NumberWidget(min=0.0, max=10.0, step=0.01),
    )


def fourier_filter(x: Any, threshold: int, scale: float) -> Any:
    torch = importlib.import_module("torch")
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    batch, channels, height, width = x_freq.shape
    mask = torch.ones((batch, channels, height, width), device=x.device)
    row, column = height // 2, width // 2
    mask[..., row - threshold : row + threshold, column - threshold : column + threshold] = scale
    x_freq = x_freq * mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    return torch.fft.ifftn(x_freq, dim=(-2, -1)).real.to(x.dtype)


def _filter_skip(skip: Any, scale: float, cpu_devices: set[Any]) -> Any:
    if skip.device in cpu_devices:
        return fourier_filter(skip.cpu(), threshold=1, scale=scale).to(skip.device)
    try:
        return fourier_filter(skip, threshold=1, scale=scale)
    except Exception:
        logging.warning(
            "Device %s does not support the torch.fft functions used in the FreeU node, "
            "switching to CPU.",
            skip.device,
        )
        cpu_devices.add(skip.device)
        return fourier_filter(skip.cpu(), threshold=1, scale=scale).to(skip.device)


class _FreeU(Node):
    defaults: tuple[float, float, float, float]
    normalized = False

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=cls.__name__,
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                *(
                    _float(name, default)
                    for name, default in zip(("b1", "b2", "s1", "s2"), cls.defaults, strict=True)
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, b1: float, b2: float, s1: float, s2: float
    ) -> Mapping[str, object]:
        model_channels = model.model.model_config.unet_config["model_channels"]
        scale_by_channels = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
        cpu_devices: set[Any] = set()

        def output_block_patch(hidden: Any, skip: Any, transformer_options: Any) -> tuple[Any, Any]:
            scale = scale_by_channels.get(int(hidden.shape[1]))
            if scale is None:
                return hidden, skip
            if cls.normalized:
                torch = importlib.import_module("torch")
                hidden_mean = hidden.mean(1).unsqueeze(1)
                batch = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(batch, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(batch, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
                    hidden_max - hidden_min
                ).unsqueeze(2).unsqueeze(3)
                hidden[:, : hidden.shape[1] // 2] *= (scale[0] - 1) * hidden_mean + 1
            else:
                hidden[:, : hidden.shape[1] // 2] *= scale[0]
            return hidden, _filter_skip(skip, scale[1], cpu_devices)

        patched = model.clone()
        patched.set_model_output_block_patch(output_block_patch)
        return cls.outputs(MODEL=patched)


class FreeU(_FreeU):
    defaults = (1.1, 1.2, 0.9, 0.2)


class FreeU_V2(_FreeU):
    defaults = (1.3, 1.4, 0.9, 0.2)
    normalized = True


FREEU_NODES = (FreeU, FreeU_V2)
