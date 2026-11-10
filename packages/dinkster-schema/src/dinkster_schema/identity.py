"""Schema-derived replacement declarations for pack authors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .model import DynamicComboSpec, DynamicEntry, InputSpec, NodeSchema
from .replace import (
    InputFamilyMapping,
    MappingSource,
    OutputFamilyMapping,
    ReplacementCase,
    ReplacementPredicate,
    ReplacementRule,
)


def _interfaces(
    entries: Sequence[DynamicEntry], prefix: str = ""
) -> list[tuple[tuple[str, ...], tuple[tuple[str, str], ...], tuple[ReplacementPredicate, ...]]]:
    """Enumerate materialized combo paths, keeping each branch's inputs separate."""
    states: list[
        tuple[tuple[str, ...], tuple[tuple[str, str], ...], tuple[ReplacementPredicate, ...]]
    ] = [((), (), ())]
    for entry in entries:
        path = f"{prefix}.{entry.id}" if prefix else entry.id
        choices: list[
            tuple[tuple[str, ...], tuple[tuple[str, str], ...], tuple[ReplacementPredicate, ...]]
        ]
        if isinstance(entry, InputSpec):
            choices = [((path,), (), ())]
        elif isinstance(entry, DynamicComboSpec):
            choices = []
            for index, option in enumerate(entry.options):
                predicate = ReplacementPredicate.value_equals(path, option.key)
                if index == 0:
                    predicate = ReplacementPredicate.any_of(
                        predicate,
                        ReplacementPredicate.not_(ReplacementPredicate.value_present(path)),
                    )
                for inputs, variants, predicates in _interfaces(option.inputs, path):
                    choices.append(
                        (inputs, ((path, option.key), *variants), (predicate, *predicates))
                    )
        else:
            raise ValueError(f"identity replacement does not support nested {type(entry).__name__}")
        states = [
            (inputs + extra_inputs, variants + extra_variants, predicates + extra_predicates)
            for inputs, variants, predicates in states
            for extra_inputs, extra_variants, extra_predicates in choices
        ]
    return states


def _identity_rule(source: str, schema: NodeSchema) -> ReplacementRule:
    if schema.slots:
        raise ValueError("identity replacement does not support dynamic slots")
    families: dict[str, InputFamilyMapping] = {}
    for family in schema.input_families:
        if any(not isinstance(entry, InputSpec) for entry in family.template):
            raise ValueError("identity replacement requires static input-family templates")
        names = tuple(entry.id for entry in family.template)
        families[family.id] = InputFamilyMapping.copy(
            family.id,
            inputs={name: MappingSource.copy(name) for name in names},
            source_template=names,
        )
    interfaces = _interfaces((*schema.inputs, *schema.combos))
    cases: list[ReplacementCase] = []
    # The first declared option is the frontend's wire-schema default. Keep
    # it as the required fallback, after all authored branch combinations.
    for index in (*range(1, len(interfaces)), 0):
        inputs, variants, predicates = interfaces[index]
        cases.append(
            ReplacementCase.build(
                schema.node_type,
                when=ReplacementPredicate.all_of(*predicates) if index else None,
                inputs={name: MappingSource.copy(name) for name in inputs},
                outputs={spec.id: spec.id for spec in schema.outputs},
                slot_variants=dict(variants),
                input_families=families,
                output_families={
                    family.id: OutputFamilyMapping.copy(family.id)
                    for family in schema.output_families
                },
            )
        )
    return ReplacementRule(from_type=source, cases=tuple(cases))


def with_identity_replacement(schema: NodeSchema, source_type: str) -> NodeSchema:
    """Declare an identity replacement without changing the authored node id."""
    declared = replace(
        schema,
        display_name=schema.display_name or source_type,
        aliases=(),
        search_terms=tuple(dict.fromkeys((*schema.search_terms, source_type))),
        replacements=(),
    )
    return replace(declared, replacements=(_identity_rule(source_type, declared),))
