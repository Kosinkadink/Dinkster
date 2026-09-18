"""Control architectures registered through the shared component catalog."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import catalog
from .assembly import AssemblyError, ComponentPlan
from .component_registry import ComponentDescriptor
from .controlnet import (
    detect_sdxl_control_lora,
    normalize_sd15_controlnet,
    normalize_sdxl_controlnet,
    normalize_sdxl_controlnet_union,
)
from .devices import BFLOAT16, FLOAT16, FLOAT32
from .t2i_adapter import normalize_sd15_t2i_adapter
from .weights import AssetIdentifiedSource, TensorGeometry, WeightSource


@dataclass(frozen=True)
class ControlComponentDescriptor(ComponentDescriptor):
    requires_base: bool = False
    hint_channels: int = 3


@dataclass(frozen=True)
class ControlComponentPlan:
    component: ComponentPlan[Any]
    asset_digest: str | None
    source_layout: str = "canonical"

    @property
    def role(self) -> str:
        return "controlnet"

    @property
    def identity_components(self) -> tuple[ComponentPlan[Any], ...]:
        return (self.component,)


def _control_lora_geometry(
    geometries: Mapping[str, TensorGeometry],
) -> tuple[Any, dict[str, str]]:
    return detect_sdxl_control_lora(geometries), {key: key for key in geometries}


def _detector(
    normalize: Callable[..., Any], component_role: str
) -> Callable[..., tuple[tuple[str, ControlComponentPlan], ...]]:
    def detect(
        source: WeightSource, path: Path, *, bind_asset_identity: bool = True
    ) -> tuple[tuple[str, ControlComponentPlan], ...]:
        geometries = {key: source.entry(key).geometry for key in source.keys()}
        try:
            normalized = normalize(geometries)
        except ValueError:
            return ()
        config, *layout, key_map = normalized
        digest = None
        if bind_asset_identity:
            if not isinstance(source, AssetIdentifiedSource):
                raise AssemblyError("control component source must carry asset identity")
            digest = source.asset_digest
        component = ComponentPlan(
            component_role,
            path,
            config,
            {model: source_key for source_key, model in key_map.items()},
            {model: geometries[source_key].dtype for source_key, model in key_map.items()},
            {},
        )
        return (("controlnet", ControlComponentPlan(component, digest, *layout)),)

    return detect


def control_component_descriptors() -> tuple[ComponentDescriptor, ...]:
    return tuple(
        ControlComponentDescriptor(
            family=replace(family, id=identifier, display_name=title, aliases=()),
            detector=_detector(normalize, role),
            roles=("controlnet",),
            text_encoder_roles=(),
            codec_roles=(),
            loader=f"dinkster_inference_torch.control_components:{loader}",
            runtime_class="dinkster_inference_torch.controlnet:SDControlConditioning",
            default_diffusion_dtype=FLOAT16,
            requires_base=requires_base,
            hint_channels=hint_channels,
            attention_roles=(("controlnet",) if role == "controlnet_union" else ()),
            attention_requires_route=role == "controlnet_union",
        )
        for family, identifier, title, normalize, role, loader, requires_base, hint_channels in (
            (
                catalog.SD15,
                "dinkster.sd15_controlnet",
                "SD1.5 ControlNet",
                normalize_sd15_controlnet,
                "controlnet",
                "load_sd15_controlnet",
                False,
                3,
            ),
            (
                replace(
                    catalog.SD15,
                    supported_dtypes=frozenset({FLOAT16, BFLOAT16, FLOAT32}),
                ),
                "dinkster.sd15_t2i_adapter",
                "SD1.5 T2I Adapter",
                normalize_sd15_t2i_adapter,
                "t2i_adapter",
                "load_sd15_t2i_adapter",
                False,
                1,
            ),
            (
                catalog.SDXL,
                "dinkster.sdxl_controlnet",
                "SDXL ControlNet",
                normalize_sdxl_controlnet,
                "controlnet",
                "load_sdxl_controlnet",
                False,
                3,
            ),
            (
                catalog.SDXL,
                "dinkster.sdxl_controlnet_union",
                "SDXL ControlNet Union",
                normalize_sdxl_controlnet_union,
                "controlnet_union",
                "load_sdxl_controlnet_union",
                False,
                3,
            ),
            (
                catalog.SDXL,
                "dinkster.sdxl_control_lora",
                "SDXL Control-LoRA",
                _control_lora_geometry,
                "control_lora",
                "load_sdxl_control_lora",
                True,
                3,
            ),
        )
    )
