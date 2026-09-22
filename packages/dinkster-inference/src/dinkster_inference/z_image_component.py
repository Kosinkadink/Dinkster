"""Z-Image per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .assembly import ComponentPlan, plan_z_image_component
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource
from .z_image import ZImageConfig


class ZImageComponentAssemblyError(ValueError):
    """A source is not the exact latent Z-Image component requested."""


def plan_z_image_split_component(
    source: WeightSource,
    *,
    role: str,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[ZImageConfig]:
    """Plan latent Z-Image diffusion and bind its immutable asset identity."""
    if role != "diffusion":
        raise ZImageComponentAssemblyError(f"unsupported Z-Image component role {role!r}")
    if getattr(source, "path", None) != path:
        raise ZImageComponentAssemblyError("Z-Image diffusion source path differs from selection")
    try:
        planned = plan_z_image_component(source)
    except ValueError as error:
        raise ZImageComponentAssemblyError(f"Z-Image diffusion: {error}") from error
    if not bind_asset_identity:
        return planned
    if not isinstance(source, AssetIdentifiedSource):
        raise ZImageComponentAssemblyError("Z-Image diffusion source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise ZImageComponentAssemblyError("Z-Image diffusion source must carry asset identity")
    return replace(
        planned,
        identity_facts=(
            *planned.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )


def z_image_component_runtime_identity(
    planned: ComponentPlan[ZImageConfig], compute_dtype: DType
) -> str:
    """Build the native identity for independently loaded latent Z-Image diffusion."""
    if planned.component != "diffusion":
        raise ValueError("Z-Image component identity requires the diffusion plan")
    return build_runtime_identity_from_facts(
        "dinkster.z_image",
        runtime_component_identity("dinkster.z_image", (planned,)),
        diffusion_dtype=compute_dtype.name,
        text_dtype="unloaded",
        vae_dtype="unloaded",
        fp8_matmul=False,
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "ZImageComponentAssemblyError",
    "plan_z_image_split_component",
    "z_image_component_runtime_identity",
]
