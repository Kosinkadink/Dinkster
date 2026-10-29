"""Torch-free execution composition contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from dinkster_inference import (
    COMPONENT_CONDITIONING_METADATA_KEY,
    CONDITIONING_TYPE_ID,
    ComponentBinding,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ExecutionComposition,
    PayloadBinding,
    PayloadDescriptor,
    PayloadReference,
    bind_component_conditioning,
    compose_execution,
    decode_conditioning_carrier,
    encode_conditioning_carrier,
    extend_runtime_identity,
    make_conditioning_carrier,
    register_conditioning_type,
    split_component_conditioning,
)
from dinkster_values import TypeRegistry

FAMILY = "dinkster.minimax_h3"


def identity(digit: str, *, family_id: str = FAMILY) -> str:
    return f"native:{family_id}:{digit * 64}"


def binding(role: str, digit: str) -> ComponentBinding:
    return ComponentBinding(role, FAMILY, identity(digit))


def test_execution_identity_is_byte_stable_and_order_independent() -> None:
    conditioner = binding("conditioner", "1")
    dit = binding("fl2va-dit", "2")
    first = ExecutionComposition(FAMILY, (dit, conditioner))
    second = ExecutionComposition(FAMILY, (conditioner, dit))
    mapped_first = compose_execution(
        FAMILY,
        {"fl2va-dit": dit.identity, "conditioner": conditioner.identity},
    )
    mapped_second = compose_execution(
        FAMILY,
        {"conditioner": conditioner.identity, "fl2va-dit": dit.identity},
    )

    assert tuple(component.role for component in first.components) == (
        "conditioner",
        "fl2va-dit",
    )
    assert first.execution_identity == (
        "native:dinkster.minimax_h3:"
        "2cbec3e827f2cae212d4270b8e75fd6c917bb6666573611def67a44d55b2f5f0"
    )
    assert first.execution_identity == second.execution_identity
    assert first.execution_identity == mapped_first.execution_identity
    assert first.execution_identity == mapped_second.execution_identity


def test_execution_identity_rotates_with_component_changes() -> None:
    conditioner = binding("conditioner", "1")
    dit = binding("fl2va-dit", "2")
    base = ExecutionComposition(FAMILY, (conditioner, dit))
    changed = ExecutionComposition(FAMILY, (conditioner, binding("fl2va-dit", "3")))
    added = ExecutionComposition(FAMILY, (*base.components, binding("video-vae", "4")))
    removed = ExecutionComposition(FAMILY, (dit,))

    assert (
        len(
            {
                base.execution_identity,
                changed.execution_identity,
                added.execution_identity,
                removed.execution_identity,
            }
        )
        == 4
    )


def test_composition_values_are_frozen() -> None:
    component = binding("conditioner", "1")
    composition = ExecutionComposition(FAMILY, (component,))
    with pytest.raises(FrozenInstanceError):
        component.role = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        composition.components = ()  # type: ignore[misc]


def _record(metadata: tuple[tuple[str, object], ...] = ()) -> ConditioningRecord:
    payload = PayloadDescriptor(PayloadReference("text"), (1, 2), "F32", "text")
    return ConditioningRecord(
        channels=((ConditioningChannel.TEXT, payload),),
        extension_metadata=metadata,  # type: ignore[arg-type]
    )


def _carrier(*records: ConditioningRecord) -> ConditioningCarrier:
    return make_conditioning_carrier(
        ConditioningSet(records or (_record(),)),
        (PayloadBinding("text", (1, 2), "F32", "text", bytes(range(8))),),
    )


def test_component_binding_round_trips_through_metadata_and_the_wire() -> None:
    component = binding("qwen2_5_vl_7b", "1")
    carrier = _carrier()
    bound = bind_component_conditioning(carrier, component)

    assert type(bound) is ConditioningCarrier
    for record in bound.conditioning.records:
        assert COMPONENT_CONDITIONING_METADATA_KEY in dict(record.extension_metadata)

    registry = TypeRegistry()
    register_conditioning_type(registry)
    registry.wrap(CONDITIONING_TYPE_ID, bound)

    decoded = decode_conditioning_carrier(encode_conditioning_carrier(bound))
    rebuilt = make_conditioning_carrier(decoded.conditioning, decoded.bindings)
    stripped, recovered = split_component_conditioning(rebuilt)
    assert recovered == component
    assert stripped.conditioning == carrier.conditioning

    unstamped_carrier, missing = split_component_conditioning(carrier)
    assert missing is None
    assert unstamped_carrier.conditioning == carrier.conditioning


def test_component_binding_restamp_is_idempotent_but_conflicts_are_refused() -> None:
    component = binding("qwen2_5_vl_7b", "1")
    bound = bind_component_conditioning(_carrier(), component)
    rebound = bind_component_conditioning(bound, component)
    assert split_component_conditioning(rebound)[1] == component

    with pytest.raises(ValueError, match="different component binding"):
        bind_component_conditioning(bound, binding("qwen2_5_vl_7b", "2"))
    with pytest.raises(ValueError, match="empty conditioning"):
        bind_component_conditioning(ConditioningCarrier(ConditioningSet(()), ()), component)

    class CarrierSubclass(ConditioningCarrier):
        pass

    class BindingSubclass(ComponentBinding):
        pass

    with pytest.raises(TypeError, match="exact ConditioningCarrier"):
        bind_component_conditioning(CarrierSubclass(ConditioningSet(()), ()), component)
    with pytest.raises(TypeError, match="exact ComponentBinding"):
        bind_component_conditioning(
            _carrier(),
            BindingSubclass(component.role, component.family_id, component.identity),
        )


def test_split_component_conditioning_requires_record_agreement() -> None:
    component = binding("qwen2_5_vl_7b", "1")
    stamp = {
        "role": component.role,
        "family_id": component.family_id,
        "identity": component.identity,
    }
    other = dict(stamp, identity=identity("2"))

    mixed = _carrier(_record(((COMPONENT_CONDITIONING_METADATA_KEY, stamp),)), _record())
    with pytest.raises(ValueError, match="disagree on their component binding"):
        split_component_conditioning(mixed)

    conflicting = _carrier(
        _record(((COMPONENT_CONDITIONING_METADATA_KEY, stamp),)),
        _record(((COMPONENT_CONDITIONING_METADATA_KEY, other),)),
    )
    with pytest.raises(ValueError, match="disagree on their component binding"):
        split_component_conditioning(conflicting)

    malformed = _carrier(_record(((COMPONENT_CONDITIONING_METADATA_KEY, "bogus"),)))
    with pytest.raises(ValueError, match="must be a mapping"):
        split_component_conditioning(malformed)


def test_empty_composition_is_refused_with_exact_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        ExecutionComposition(FAMILY, ())
    assert str(exc_info.value) == "execution composition components must be non-empty"


def test_duplicate_component_role_is_refused_with_exact_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        ExecutionComposition(FAMILY, (binding("dit", "1"), binding("dit", "2")))
    assert str(exc_info.value) == "execution composition component roles must be unique"


def test_component_family_mismatch_is_refused_with_role_in_exact_message() -> None:
    component = ComponentBinding(
        "video-vae", "dinkster.other", identity("1", family_id="dinkster.other")
    )
    with pytest.raises(ValueError) as exc_info:
        ExecutionComposition(FAMILY, (component,))
    assert str(exc_info.value) == (
        "execution component 'video-vae' family_id must match composition family_id"
    )


def test_explicit_shared_component_family_is_accepted() -> None:
    component = ComponentBinding(
        "t5xxl", "dinkster.chroma", identity("1", family_id="dinkster.chroma")
    )
    composition = ExecutionComposition(
        "dinkster.chroma_radiance",
        (component,),
        shared_component_families=frozenset(("dinkster.chroma",)),
    )

    assert composition.components == (component,)


@pytest.mark.parametrize("shared", (set(), ("dinkster.chroma",), frozenset((1,))))
def test_shared_component_families_require_exact_frozen_string_set(shared: object) -> None:
    component = binding("dit", "1")
    with pytest.raises(TypeError, match="frozenset of strings"):
        ExecutionComposition(
            FAMILY,
            (component,),
            shared_component_families=shared,  # type: ignore[arg-type]
        )


def test_shared_component_family_ids_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="family ids must be non-empty"):
        ExecutionComposition(
            FAMILY,
            (binding("dit", "1"),),
            shared_component_families=frozenset(("",)),
        )


def test_identity_family_mismatch_is_refused_with_exact_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        ComponentBinding("dit", FAMILY, identity("1", family_id="dinkster.other"))
    assert str(exc_info.value) == "component binding identity family must match family_id"


@pytest.mark.parametrize(
    ("malformed", "message"),
    (
        (
            f"native:{FAMILY}",
            "component binding identity must be a 3-part native identity",
        ),
        (
            f"foreign:{FAMILY}:{'1' * 64}",
            "component binding identity must be a 3-part native identity",
        ),
        (
            f"native::{'1' * 64}",
            "component binding identity family must be non-empty",
        ),
        (
            f"native:{FAMILY}:extra:{'1' * 64}",
            "component binding identity must be a 3-part native identity",
        ),
        (
            f"native:{FAMILY}:{'A' * 64}",
            "component binding identity digest must be a lowercase sha256 hex digest",
        ),
        (
            f"native:{FAMILY}:{'1' * 63}",
            "component binding identity digest must be a lowercase sha256 hex digest",
        ),
    ),
)
def test_malformed_identity_is_refused_with_exact_message(malformed: str, message: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        ComponentBinding("dit", FAMILY, malformed)
    assert str(exc_info.value) == message


@pytest.mark.parametrize("role", ("", "Uppercase", ".leading", "_leading"))
def test_bad_role_is_refused_with_exact_message(role: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        ComponentBinding(role, FAMILY, identity("1"))
    assert str(exc_info.value) == "component binding role must be a canonical id"


def test_component_binding_non_string_fields_are_refused_with_exact_messages() -> None:
    with pytest.raises(TypeError) as exc_info:
        ComponentBinding(1, FAMILY, identity("1"))  # type: ignore[arg-type]
    assert str(exc_info.value) == "component binding role must be a string"

    with pytest.raises(TypeError) as exc_info:
        ComponentBinding("dit", 1, identity("1"))  # type: ignore[arg-type]
    assert str(exc_info.value) == "component binding family_id must be a string"

    with pytest.raises(TypeError) as exc_info:
        ComponentBinding("dit", FAMILY, 1)  # type: ignore[arg-type]
    assert str(exc_info.value) == "component binding identity must be a string"


def test_execution_composition_field_types_are_refused_with_exact_messages() -> None:
    component = binding("dit", "1")
    with pytest.raises(TypeError) as exc_info:
        ExecutionComposition(1, (component,))  # type: ignore[arg-type]
    assert str(exc_info.value) == "execution composition family_id must be a string"

    with pytest.raises(TypeError) as exc_info:
        ExecutionComposition(FAMILY, [component])  # type: ignore[arg-type]
    assert str(exc_info.value) == (
        "execution composition components must be a tuple of ComponentBinding"
    )

    with pytest.raises(TypeError) as exc_info:
        ExecutionComposition(FAMILY, ("dit",))  # type: ignore[arg-type]
    assert str(exc_info.value) == (
        "execution composition components must be a tuple of ComponentBinding"
    )


def test_empty_family_ids_are_refused_with_exact_messages() -> None:
    with pytest.raises(ValueError) as exc_info:
        ComponentBinding("dit", "", "native::" + "1" * 64)
    assert str(exc_info.value) == "component binding family_id must be non-empty"

    with pytest.raises(ValueError) as exc_info:
        ExecutionComposition("", (binding("dit", "1"),))
    assert str(exc_info.value) == "execution composition family_id must be non-empty"


def test_compose_execution_requires_a_mapping() -> None:
    with pytest.raises(TypeError) as exc_info:
        compose_execution(FAMILY, (("dit", identity("1")),))  # type: ignore[arg-type]
    assert str(exc_info.value) == "components must be a mapping of role to identity"


@pytest.mark.parametrize(
    ("malformed", "message"),
    (
        (
            f"native:{FAMILY}",
            "component binding identity must be a 3-part native identity",
        ),
        (1, "component binding identity must be a string"),
    ),
)
def test_compose_execution_refuses_malformed_identity_without_incidental_errors(
    malformed: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError)) as exc_info:
        compose_execution(FAMILY, {"dit": malformed})  # type: ignore[dict-item]
    assert str(exc_info.value) == message


def test_execution_identity_is_a_valid_extendable_runtime_identity() -> None:
    composition = compose_execution(FAMILY, {"dit": identity("1")})
    extended = extend_runtime_identity(composition.execution_identity, ("fact=x",))

    assert extended.split(":")[:2] == composition.execution_identity.split(":")[:2]
    assert extended != composition.execution_identity
