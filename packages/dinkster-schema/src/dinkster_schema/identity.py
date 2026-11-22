"""Schema-derived replacement declarations for pack authors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .body import (
    BodyRow,
    Button,
    Choice,
    Editor,
    Group,
    Note,
    Row,
    Section,
    State,
    Use,
    scope_rows,
)
from .flatten import apply_use
from .model import Input, NodeSchema, Output
from .replace import (
    InputFamilyMapping,
    MappingSource,
    OutputFamilyMapping,
    ReplacementCase,
    ReplacementPredicate,
    ReplacementRule,
)

type _Interface = tuple[
    tuple[str, ...], tuple[tuple[str, str], ...], tuple[ReplacementPredicate, ...]
]


def _body_interfaces(rows: Sequence[BodyRow | Use], prefix: str = "") -> list[_Interface]:
    """The inputs of one body scope, branched once per option of each choice,
    with the shared rows an option names and the choices inside it."""
    states: list[_Interface] = [((), (), ())]
    for row in scope_rows(tuple(rows)):
        path = (
            f"{prefix}.{row.id}" if prefix and not isinstance(row, Row) else getattr(row, "id", "")
        )
        branches: list[_Interface]
        if isinstance(row, (Input, State)):
            branches = [((path,), (), ())]
        elif isinstance(row, Choice):
            branches = []
            shared = {item.id: item for item in scope_rows(row.shared) if not isinstance(item, Row)}
            for index, option in enumerate(row.options):
                predicate = ReplacementPredicate.value_equals(path, option.key)
                if index == 0:
                    predicate = ReplacementPredicate.any_of(
                        predicate,
                        ReplacementPredicate.not_(ReplacementPredicate.value_present(path)),
                    )
                option_rows = [
                    *option.rows,
                    # A Use may narrow a shared choice's options with only=.
                    *(apply_use(shared[use.id], use) for use in _uses(option.rows)),
                ]
                for inputs, variants, predicates in _body_interfaces(option_rows, path):
                    branches.append(
                        (inputs, ((path, option.key), *variants), (predicate, *predicates))
                    )
        elif isinstance(row, (Output, Button, Note, Editor)):
            continue
        elif isinstance(row, Group) and not prefix:
            continue
        else:
            raise ValueError(
                f"identity replacement does not support {type(row).__name__} '{path}' here"
            )
        states = [
            (inputs + extra_inputs, variants + extra_variants, predicates + extra_predicates)
            for inputs, variants, predicates in states
            for extra_inputs, extra_variants, extra_predicates in branches
        ]
    return states


def _uses(rows: Sequence[BodyRow | Use]) -> list[Use]:
    found: list[Use] = []
    for row in rows:
        if isinstance(row, Use):
            found.append(row)
        elif isinstance(row, (Section, Row)):
            found.extend(_uses(row.rows))
    return found


def _body_families(
    schema: NodeSchema,
) -> tuple[dict[str, InputFamilyMapping], dict[str, OutputFamilyMapping]]:
    """The body's own groups as copied input and output families."""
    inputs: dict[str, InputFamilyMapping] = {}
    outputs: dict[str, OutputFamilyMapping] = {}
    for row in scope_rows(schema.body):
        if not isinstance(row, Group):
            continue
        if isinstance(row.value, Output):
            outputs[row.id] = OutputFamilyMapping.copy(row.id)
            continue
        member = (row.value,) if row.value is not None else tuple(row.member or ())
        flat = list(scope_rows(member))
        if not all(isinstance(item, Input) for item in flat):
            raise ValueError(f"identity replacement needs group '{row.id}' to hold inputs only")
        if row.value is None and len(flat) == 1:
            # The replacement language reads a one-row template as a group of values.
            raise ValueError(
                f"identity replacement does not support group '{row.id}', "
                "a group of records with one row"
            )
        names = tuple(item.id for item in flat if isinstance(item, Input))
        inputs[row.id] = InputFamilyMapping.copy(
            row.id,
            inputs={name: MappingSource.copy(name) for name in names},
            source_template=names,
        )
    return inputs, outputs


def _identity_rule(source: str, schema: NodeSchema) -> ReplacementRule:
    families, output_families = _body_families(schema)
    interfaces = _body_interfaces(schema.body)
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
                output_families=output_families,
            )
        )
    return ReplacementRule(from_type=source, cases=tuple(cases))


def with_identity_replacement(schema: NodeSchema, source_type: str) -> NodeSchema:
    """Declare an identity replacement without changing the authored node id."""
    declared = replace(
        schema,
        display_name=schema.display_name or source_type,
        search_terms=tuple(dict.fromkeys((*schema.search_terms, source_type))),
        replacements=(),
    )
    return replace(declared, replacements=(_identity_rule(source_type, declared),))
