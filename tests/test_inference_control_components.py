"""Control loading selection is driven by tensor geometry and declared roles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from dinkster_inference import TensorGeometry, WeightEntry
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.component_registry import ComponentRegistry
from dinkster_inference.control_components import control_component_descriptors
from dinkster_protocol import ATTENTION_ROLES, AttentionRoute, AttentionRouteToken
from test_inference_controlnet import (
    adapter_geometries,
    canonical_geometries,
    classic_sdxl_controlnet_geometries,
    control_lora_geometries,
    controlnet_union_geometries,
)


@dataclass(frozen=True)
class ControlHeader:
    geometries: dict[str, TensorGeometry]
    path: Path = Path("control.safetensors")
    asset_digest: str = "blake3:" + "1" * 64
    asset_size: int = 1

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return {}


@pytest.mark.parametrize(
    "identifier,geometries",
    [
        ("dinkster.sd15_controlnet", canonical_geometries()),
        ("dinkster.sd15_t2i_adapter", adapter_geometries()),
        ("dinkster.sdxl_controlnet", classic_sdxl_controlnet_geometries()),
        ("dinkster.sdxl_controlnet_union", controlnet_union_geometries()),
        ("dinkster.sdxl_control_lora", control_lora_geometries()),
    ],
)
def test_control_descriptor_preserves_source_claims(
    identifier: str, geometries: dict[str, TensorGeometry]
) -> None:
    source = ControlHeader(geometries)
    registry = default_component_registry()
    descriptor, role, plan = registry.select(source, source.path, "controlnet")
    assert descriptor.id == identifier
    assert role == "controlnet"
    assert plan.asset_digest == source.asset_digest
    assert set(plan.component.keys.values()) == set(source.keys())
    assert plan.component.path == source.path
    unbound = descriptor.detector(source, source.path, bind_asset_identity=False)[0][1]
    assert unbound.component == plan.component
    assert unbound.asset_digest is None
    with pytest.raises(ValueError, match="no matching component architecture"):
        registry.select(source, source.path, "model")


def test_control_descriptor_from_an_unknown_family_is_selectable() -> None:
    descriptor = control_component_descriptors()[0]
    descriptor = replace(descriptor, family=replace(descriptor.family, id="test.new_control"))
    registry = ComponentRegistry()
    registry.register(descriptor)
    source = ControlHeader(canonical_geometries())
    selected, role, _plan = registry.select(source, source.path, "controlnet")
    assert selected is descriptor
    assert role == "controlnet"


def test_union_control_descriptor_declares_its_attention_role() -> None:
    descriptors = {descriptor.id: descriptor for descriptor in control_component_descriptors()}

    union = descriptors["dinkster.sdxl_controlnet_union"]
    assert union.attention_roles == ("controlnet",)
    assert union.attention_requires_route
    assert all(
        descriptor.attention_roles == ()
        for identifier, descriptor in descriptors.items()
        if identifier != "dinkster.sdxl_controlnet_union"
    )

    with pytest.raises(ValueError, match="attention roles must be component roles"):
        replace(union, attention_roles=("unet",))
    with pytest.raises(ValueError, match="need at least one attention role"):
        replace(union, attention_roles=())


def test_union_control_identity_binds_authenticated_attention_route() -> None:
    source = ControlHeader(controlnet_union_geometries())
    descriptor, role, plan = default_component_registry().select(source, source.path, "controlnet")
    token = AttentionRouteToken(
        version=1,
        routes=tuple(AttentionRoute(attention_role, "sdpa") for attention_role in ATTENTION_ROLES),
        provider_versions=(("torch", "2.13.0"),),
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        requested_policy="auto",
    )

    knobs = descriptor.knobs(
        role,
        plan.identity_components,
        "float16",
        attention_route_token=token,
    )
    base_identity = descriptor.component_identity(role, plan, "float16")
    routed_identity = descriptor.component_identity(
        role,
        plan,
        "float16",
        attention_route_token=token,
    )

    assert knobs.attention_route_token is token
    assert routed_identity != base_identity
