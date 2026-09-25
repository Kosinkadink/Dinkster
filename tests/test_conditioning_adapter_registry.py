"""Generation-scoped conditioning adapter registration contracts."""

from __future__ import annotations

import pytest
from dinkster_inference.conditioning_adapters import ConditioningAdapter
from dinkster_inference.extensions import (
    INFERENCE_CONDITIONING_ADAPTERS_SURFACE,
    InferenceContribution,
    conditioning_adapter_declaration,
)
from dinkster_inference.registries import builtin_registries, merge
from dinkster_inference.registry import RegistryError


def _adapter(id_: str, aliases: tuple[str, ...] = ()) -> ConditioningAdapter:
    return ConditioningAdapter(
        id=id_,
        aliases=aliases,
        prepare=lambda carrier: carrier,
        release=lambda carrier: None,
    )


def test_contributed_conditioning_adapter_is_generation_scoped() -> None:
    base = builtin_registries()
    adapter = _adapter("test.family", ("test.legacy-family",))

    first = merge(base, (InferenceContribution(conditioning_adapters=(adapter,)),))
    second = merge(base, ())

    assert first.conditioning_adapters.get("test.family") is adapter
    assert first.conditioning_adapters.get("test.legacy-family") is adapter
    assert second.conditioning_adapters.get("test.family") is None


def test_conditioning_adapter_collision_fails_without_replacing_owner() -> None:
    first = _adapter("test.family")
    second = _adapter("test.other", ("test.family",))

    with pytest.raises(RegistryError, match="already registered"):
        merge(
            builtin_registries(),
            (
                InferenceContribution(conditioning_adapters=(first,)),
                InferenceContribution(conditioning_adapters=(second,)),
            ),
        )


def test_conditioning_adapter_declaration_names_family_identity() -> None:
    declaration = conditioning_adapter_declaration(_adapter("test.family"))

    assert declaration.surface_id == INFERENCE_CONDITIONING_ADAPTERS_SURFACE
    assert declaration.id == "test.family"
    assert declaration.behavior_metadata == ()


def test_conditioning_adapter_rejects_non_callable_operations() -> None:
    with pytest.raises(TypeError, match="prepare must be callable"):
        ConditioningAdapter("test.family", None, lambda carrier: None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="release must be callable"):
        ConditioningAdapter("test.family", lambda carrier: carrier, None)  # type: ignore[arg-type]
