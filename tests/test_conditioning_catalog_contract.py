"""Catalog-wide acceptance contract for the native conditioning value."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dinkster_inference import (
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    PayloadBinding,
    PayloadDescriptor,
    PayloadReference,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    builtin_families,
    builtin_registries,
    conditioning,
    make_conditioning_carrier,
    prepare_conditioning,
)
from dinkster_schema import (
    DynamicComboSpec,
    DynamicEntry,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    NodeSchema,
    TypeExpr,
)
from dinkster_workers import load_manifest
from dinkster_workers.catalog import read_catalog

from dinkster.comfy_compose import comfy_compat_specs
from dinkster.compose import default_pack_specs

CONDITIONING_TYPE = TypeExpr.concrete("dinkster.conditioning")


def _dynamic_inputs(entries: Sequence[DynamicEntry]) -> Iterable[InputSpec]:
    for entry in entries:
        if isinstance(entry, InputSpec):
            yield entry
        elif isinstance(entry, InputFamilySpec):
            yield from _dynamic_inputs(entry.template)
        elif isinstance(entry, DynamicComboSpec):
            for option in entry.options:
                yield from _dynamic_inputs(option.inputs)
        elif isinstance(entry, DynamicSlotSpec):
            yield from _dynamic_inputs(entry.inputs)
            for variant in entry.variants or ():
                yield from _dynamic_inputs(variant.inputs)


def _conditioning_input_ids(schema: NodeSchema) -> tuple[str, ...]:
    inputs = (
        *schema.inputs,
        *_dynamic_inputs((*schema.input_families, *schema.combos, *schema.slots)),
    )
    return tuple(sorted({input_.id for input_ in inputs if input_.type == CONDITIONING_TYPE}))


def _family_carrier(family_id: str) -> ConditioningCarrier:
    payload = PayloadDescriptor(PayloadReference("text"), (1,), "F32", "text")
    layout = TokenLayoutDescriptor(
        family_id,
        1,
        ("text",),
        (TokenSegmentDescriptor("prompt", "text", 0, 1),),
    )
    return make_conditioning_carrier(
        ConditioningSet(
            (
                ConditioningRecord(
                    channels=((ConditioningChannel.TEXT, payload),), token_layout=layout
                ),
            )
        ),
        (PayloadBinding("text", (1,), "F32", "text", b"\0\0\0\0"),),
    )


def test_every_catalog_conditioning_input_accepts_every_registered_family(
    installed_default_catalogs: None,
) -> None:
    input_ids: set[str] = set()
    for spec in (*default_pack_specs(), *comfy_compat_specs()):
        catalog = read_catalog(load_manifest(spec.manifest))
        assert catalog is not None
        for schema in catalog.schemas.values():
            input_ids.update(
                f"{schema.node_type}.{input_id}" for input_id in _conditioning_input_ids(schema)
            )

    assert input_ids
    registries = builtin_registries()
    family_ids = tuple(family.id for family in builtin_families())
    assert family_ids
    for family_id in family_ids:
        carrier = prepare_conditioning(
            _family_carrier(family_id),
            registries.conditioning_adapters,
        )
        for input_id in input_ids:
            assert conditioning(carrier, input_id) is carrier
