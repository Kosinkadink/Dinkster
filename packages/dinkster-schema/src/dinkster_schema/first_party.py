"""First-party identities and schema-declared ComfyUI replacement rules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from functools import cache
from importlib.resources import files

from .model import DynamicComboSpec, DynamicEntry, InputSpec, NodeSchema
from .replace import (
    InputFamilyMapping,
    MappingSource,
    OutputFamilyMapping,
    ReplacementCase,
    ReplacementPredicate,
    ReplacementRule,
    rule_to_wire,
)


@cache
def _identities() -> dict[str, tuple[str, str, bool]]:
    data = json.loads(files("dinkster_schema").joinpath("comfy_node_ids.json").read_text())
    return {
        row["from"]: (row["pack"], row["nodeType"], row["backendPureRename"])
        for row in data["nodes"]
    }


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


def first_party_schema(schema: NodeSchema) -> NodeSchema:
    """Move an original class schema to its frozen identity without changing its interface."""
    source = schema.node_type
    _, target, backend = _identities()[source]
    if backend == bool(
        schema.input_families or schema.output_families or schema.combos or schema.slots
    ):
        raise ValueError(f"{source}: backendPureRename disagrees with the schema")
    if any(rule.from_type == source for rule in schema.replacements):
        raise ValueError(f"{source}: existing same-type migration requires explicit integration")
    renamed = replace(
        schema,
        node_type=target,
        aliases=(),
        search_terms=tuple(dict.fromkeys((*schema.search_terms, source))),
        replacements=tuple(
            replace(
                rule,
                cases=tuple(
                    replace(case, to=target) if case.to == source else case for case in rule.cases
                ),
            )
            for rule in schema.replacements
        ),
    )
    return replace(renamed, replacements=(*renamed.replacements, _identity_rule(source, renamed)))


def collect_comfy_translations(
    packs: Iterable[tuple[str, str, str, Iterable[NodeSchema]]],
) -> dict[str, object]:
    """Collect unchanged rules from (authenticated publisher, pack, version, schemas)."""
    identities = _identities()
    sources = {target: source for source, (_, target, _) in identities.items()}
    claimed: set[str] = set()
    items: dict[str, dict[str, object]] = {}
    for publisher, pack, version, schemas in packs:
        for schema in schemas:
            source = sources.get(schema.node_type)
            if source is not None and not any(
                rule.from_type == source and rule.migration is None for rule in schema.replacements
            ):
                raise ValueError(f"comfy-translations.missing: {source}")
            for rule in schema.replacements:
                identity = identities.get(rule.from_type)
                if identity is None or rule.migration is not None:
                    continue
                if publisher != "dinkster":
                    raise ValueError(
                        "comfy-translations.publisher: only dinkster may claim source classes"
                    )
                owner, target, _ = identity
                if pack != owner or schema.node_type != target:
                    raise ValueError(f"comfy-translations.owner: wrong owner for {rule.from_type}")
                if rule.from_type in claimed:
                    raise ValueError(f"comfy-translations.duplicate: {rule.from_type}")
                if rule != _identity_rule(rule.from_type, schema):
                    raise ValueError(
                        f"comfy-translations.rule: non-identity rule for {rule.from_type}"
                    )
                claimed.add(rule.from_type)
                items[rule.from_type] = {
                    "pack": pack,
                    "version": version,
                    "rule": rule_to_wire(rule),
                }
    return {"schemaVersion": 1, "items": [items[source] for source in sorted(items)]}
