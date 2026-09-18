from __future__ import annotations

from typing import Any, cast

import pytest
from dinkster_inference_torch import NativeResidencyPolicy


def _policy(**overrides: Any) -> NativeResidencyPolicy:
    values: dict[str, Any] = {
        "enrollment_components": ("dit", "text", "vae"),
        "enrollment_orders": (("dit", "text", "vae"),),
        "component_roles": {"dit": "diffusion", "text": "text", "vae": "vae"},
        "diffusion_roles": frozenset({"diffusion"}),
        "unload_after_stage": frozenset({"vae"}),
        "resident_components": frozenset({"vae"}),
    }
    values.update(overrides)
    return NativeResidencyPolicy(**values)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    (
        ("enrollment_components", ["dit"], TypeError),
        ("enrollment_components", (), ValueError),
        ("enrollment_components", ("dit", 1), TypeError),
        ("enrollment_components", ("dit", ""), ValueError),
        ("enrollment_components", ("dit", "dit"), ValueError),
        ("enrollment_orders", [["dit"]], TypeError),
        ("enrollment_orders", (["dit"],), TypeError),
        ("enrollment_orders", ((1,),), TypeError),
        ("enrollment_orders", (("unknown",),), ValueError),
        ("component_roles", (), TypeError),
        ("component_roles", {1: "diffusion"}, TypeError),
        ("component_roles", {"unknown": "diffusion"}, ValueError),
        ("component_roles", {"dit": 1}, TypeError),
        ("component_roles", {"dit": ""}, ValueError),
        ("diffusion_roles", {"diffusion"}, TypeError),
        ("diffusion_roles", frozenset({1}), TypeError),
        ("diffusion_roles", frozenset({""}), ValueError),
        ("diffusion_roles", frozenset({"unknown"}), ValueError),
        ("unload_after_stage", {"vae"}, TypeError),
        ("unload_after_stage", frozenset({1}), TypeError),
        ("unload_after_stage", frozenset({""}), ValueError),
        ("unload_after_stage", frozenset({"unknown"}), ValueError),
        ("resident_components", {"vae"}, TypeError),
        ("resident_components", frozenset({1}), TypeError),
        ("resident_components", frozenset({"unknown"}), ValueError),
    ),
)
def test_native_residency_policy_validates_every_field(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match=field):
        _policy(**{field: value})


def test_native_residency_policy_copies_and_freezes_component_roles() -> None:
    component_roles = {"dit": "diffusion", "text": "text", "vae": "vae"}
    policy = _policy(component_roles=component_roles)

    component_roles["extra"] = "text"
    assert dict(policy.component_roles) == {
        "dit": "diffusion",
        "text": "text",
        "vae": "vae",
    }
    with pytest.raises(TypeError):
        cast("Any", policy.component_roles)["dit"] = "text"
