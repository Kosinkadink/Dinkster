"""Generation-scoped conditioning adapter registration contracts."""

from __future__ import annotations

import pytest
from dinkster_inference import (
    ConditioningAdapter,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningLayoutIncompatibility,
    ConditioningRecord,
    ConditioningSet,
    PayloadBinding,
    PayloadDescriptor,
    PayloadReference,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    make_conditioning_carrier,
    prepare_conditioning,
    release_conditioning,
)
from dinkster_inference.extensions import (
    INFERENCE_CONDITIONING_ADAPTERS_SURFACE,
    InferenceContribution,
    conditioning_adapter_declaration,
)
from dinkster_inference.registries import builtin_registries, merge
from dinkster_inference.registry import RegistryError


def _carrier(*family_ids: str) -> ConditioningCarrier:
    records = tuple(
        ConditioningRecord(
            channels=(
                (
                    ConditioningChannel.TEXT,
                    PayloadDescriptor(PayloadReference(f"text-{index}"), (1,), "F32", "text"),
                ),
            ),
            token_layout=TokenLayoutDescriptor(
                family_id,
                1,
                ("text",),
                (TokenSegmentDescriptor("prompt", "text", 0, 1),),
            ),
        )
        for index, family_id in enumerate(family_ids)
    )
    return make_conditioning_carrier(
        ConditioningSet(records),
        tuple(
            PayloadBinding(f"text-{index}", (1,), "F32", "text", b"\0\0\0\0")
            for index in range(len(records))
        ),
    )


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


def test_conditioning_adapter_dispatches_by_token_layout_family() -> None:
    prepared: list[ConditioningCarrier] = []
    released: list[ConditioningCarrier] = []
    adapter = ConditioningAdapter(
        "test.family",
        lambda carrier: prepared.append(carrier) or carrier,
        released.append,
    )
    registries = merge(
        builtin_registries(),
        (InferenceContribution(conditioning_adapters=(adapter,)),),
    )
    carrier = _carrier("test.family")

    assert prepare_conditioning(carrier, registries.conditioning_adapters) is carrier
    release_conditioning(carrier, registries.conditioning_adapters)

    assert prepared == [carrier]
    assert released == [carrier]


def test_unadapted_family_keeps_the_canonical_carrier() -> None:
    carrier = _carrier("test.unadapted")
    adapters = builtin_registries().conditioning_adapters

    assert prepare_conditioning(carrier, adapters) is carrier
    release_conditioning(carrier, adapters)


def test_mixed_token_layout_families_are_explicitly_incompatible() -> None:
    carrier = _carrier("test.first", "test.second")

    with pytest.raises(ConditioningLayoutIncompatibility, match="test.first, test.second"):
        prepare_conditioning(carrier, builtin_registries().conditioning_adapters)


def test_conditioning_adapter_cannot_change_canonical_records() -> None:
    carrier = _carrier("test.family")
    replacement = _carrier("test.other")
    adapter = ConditioningAdapter(
        "test.family", lambda _carrier: replacement, lambda _carrier: None
    )
    registries = merge(
        builtin_registries(),
        (InferenceContribution(conditioning_adapters=(adapter,)),),
    )

    with pytest.raises(ConditioningLayoutIncompatibility, match="changed canonical"):
        prepare_conditioning(carrier, registries.conditioning_adapters)
