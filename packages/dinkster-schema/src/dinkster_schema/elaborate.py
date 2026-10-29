"""Elaboration: dynamic interfaces become concrete, once, before all consumers.

This is the backend half of the frontend's elaboration model (hazard H10):

    elaborate(schema, stored input ids, stored output members)
        -> effective concrete NodeSchema

- **Pure and deterministic.** Its inputs are restricted by construction: the
  base schema and *document state* (stored inputs and output-member lists) -
  never runtime values or solved types - so elaboration cannot
  oscillate with type resolution and always runs before validation, planning,
  caching, and invocation.
- **Output membership is document-determined.** The document stores an
  ordered member-suffix list per output family; a node can never change its
  interface by executing. Data-dependent cardinality belongs in a
  collection-typed value on one static output; true runtime structural
  fan-out is a future explicit graph-expansion feature with its own contract,
  never interface mutation mid-run.
- **The result is an ordinary NodeSchema.** Family members become plain
  InputSpecs/OutputSpecs with their stored ids. Validation, cache keys (via
  schema_signature of the effective schema), and worker output wrapping all
  consume the effective schema and never branch on "is this dynamic".
- **Unbound member order is document order.** Unbound families keep the order
  the document stores (order can be semantically meaningful), while
  count-bound families require their canonical index order.
- **Static schemas elaborate to themselves** (identical object), so static
  nodes pay nothing and their signatures/cache keys are unchanged.
- **Hard budgets, deterministic failure** (mirror of the frontend's
  ElabBudget): malformed or oversized membership raises ElaborationError -
  one clear error, before members are materialized - which validation
  surfaces as a diagnostic. Elaboration never clamps or repairs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import cast

from .model import (
    STRUCTURAL_ID_PATTERN,
    ComboWidget,
    DynamicComboSpec,
    DynamicEntry,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    Widget,
    WidgetRepresentations,
)
from .output_descriptors import elaborate_output_descriptors

MAX_FAMILY_MEMBERS = 512
"""Hard cap on members per family (matches the frontend's budget)."""

MAX_INTERFACE_ITEMS = 1024
"""Hard cap on total effective interface items (inputs + outputs)."""


class ElaborationError(ValueError):
    """Document state cannot elaborate against this schema (deterministic)."""


def elaborate(
    schema: NodeSchema,
    stored_inputs: Mapping[str, object] | Iterable[str],
    output_members: Mapping[str, Sequence[str]] | None = None,
    slot_variants: Mapping[str, str] | None = None,
) -> NodeSchema:
    """The effective concrete interface of one node.

    stored_inputs is either the graph node's input mapping or, for callers
    that only need document-driven interfaces, its input keys. Count-bound
    output families require the mapping so they can verify the stored literal.
    output_members is the node's stored output membership: family id ->
    ordered member suffixes. A missing family key means an empty family.
    slot_variants is the node's stored slot choices: slot id -> active
    variant key. A missing key means "no choice" - legal only for an
    optional slot, which then contributes nothing to the interface.
    """
    if schema.is_static:
        if output_members:
            raise ElaborationError(f"{schema.node_type}: output members stored for a static schema")
        if slot_variants:
            raise ElaborationError(f"{schema.node_type}: slot variants stored for a static schema")
        return schema

    if isinstance(stored_inputs, Mapping):
        stored_values = cast("Mapping[str, object]", stored_inputs)
        stored_ids = tuple(stored_values)
    else:
        stored_values = None
        stored_ids = tuple(stored_inputs)
    choices = slot_variants or {}
    inputs = list(schema.inputs)
    projected, active_choices, consumed_choices = _elaborate_dynamic_entries(
        (*schema.input_families, *schema.combos, *schema.slots),
        "",
        stored_ids,
        choices,
    )
    inputs.extend(projected)
    inputs = [
        replace(spec, widget=_materialize_family_options(spec.widget, inputs))
        if spec.widget is not None
        else spec
        for spec in inputs
    ]
    unknown_choices = set(choices) - consumed_choices
    if unknown_choices:
        raise ElaborationError(
            f"{schema.node_type}: dynamic choice(s) stored for undeclared or "
            f"inactive construct(s): {', '.join(sorted(unknown_choices))}"
        )
    type_bindings = _slot_type_bindings(schema, choices)
    if type_bindings:
        inputs = [
            replace(
                spec,
                type=_substitute_type(
                    spec.type,
                    type_bindings,
                    f"{schema.node_type}: input '{spec.id}'",
                ),
            )
            for spec in inputs
        ]
    _validate_projected_namespace(schema, inputs)
    outputs = _elaborate_outputs(schema, output_members or {}, stored_values)
    if type_bindings:
        outputs = [
            replace(
                spec,
                type=_substitute_type(
                    spec.type,
                    type_bindings,
                    f"{schema.node_type}: output '{spec.id}'",
                ),
            )
            for spec in outputs
        ]

    chosen = dict(active_choices)

    def scope_is_covered(applies: Mapping[str, tuple[str, ...]]) -> bool:
        return all(chosen[combo_id] in covered for combo_id, covered in applies.items())

    if schema.chunk_safe_applies is not None and not scope_is_covered(schema.chunk_safe_applies):
        assert schema.chunk_safe is not None
        inputs = [
            replace(spec, accepts_stream=False) if spec.id in schema.chunk_safe[0] else spec
            for spec in inputs
        ]

    mirror = schema.mirror
    if mirror is not None and mirror.applies is not None:
        # applies keys name required top-level combos, so elaboration always
        # has a consumed choice. Covered metadata loses its resolved scope;
        # uncovered metadata is not a sound estimate and is dropped.
        mirror = replace(mirror, applies=None) if scope_is_covered(mirror.applies) else None

    for index, output in enumerate(outputs):
        represents = output.represents
        if represents is None or represents.applies is None:
            continue
        outputs[index] = replace(
            output,
            represents=(
                replace(represents, applies=None) if scope_is_covered(represents.applies) else None
            ),
        )

    total = len(inputs) + len(outputs)
    if total > MAX_INTERFACE_ITEMS:
        raise ElaborationError(
            f"{schema.node_type}: effective interface has {total} items, "
            f"budget is {MAX_INTERFACE_ITEMS}"
        )

    return replace(
        schema,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        mirror=mirror,
        chunk_safe=(
            (
                tuple(
                    name for name in schema.chunk_safe[0] if name in {spec.id for spec in inputs}
                ),
                schema.chunk_safe[1],
            )
            if schema.chunk_safe is not None
            and (schema.chunk_safe_applies is None or scope_is_covered(schema.chunk_safe_applies))
            else None
        ),
        chunk_safe_applies=None,
        input_families=(),
        output_families=(),
        output_descriptors=None,
        combos=(),
        slots=(),
        slot_choices=active_choices,
    )


def _materialize_family_options(widget: Widget, inputs: Sequence[InputSpec]) -> Widget:
    if isinstance(widget, ComboWidget) and widget.option_source is not None:
        prefix = widget.option_source.input_family + "."
        suffixes: list[str] = []
        for spec in inputs:
            if not spec.id.startswith(prefix):
                continue
            suffix = spec.id[len(prefix) :].split(".", 1)[0]
            if suffix and suffix not in suffixes:
                suffixes.append(suffix)
        return replace(widget, options=tuple(suffixes), option_source=None)
    if isinstance(widget, WidgetRepresentations):
        return replace(
            widget,
            representations=tuple(
                replace(
                    representation,
                    widget=_materialize_family_options(representation.widget, inputs),
                )
                for representation in widget.representations
            ),
        )
    return widget


def _join_path(parent: str, local: str) -> str:
    return f"{parent}.{local}" if parent else local


def _slot_type_bindings(
    schema: NodeSchema,
    choices: Mapping[str, str],
) -> dict[str, TypeExpr]:
    bindings: dict[str, TypeExpr] = {}
    for slot in schema.slots:
        if not slot.type_template_id:
            continue
        key = choices.get(slot.id)
        assert isinstance(key, str)  # required closed slot was validated above
        variant = slot.variant(key)
        assert variant is not None  # selected key was validated above
        bindings[slot.type_template_id] = variant.type
    return bindings


def _substitute_type(
    expr: TypeExpr,
    bindings: Mapping[str, TypeExpr],
    subject: str,
) -> TypeExpr:
    if expr.kind == "variable":
        bound = bindings.get(expr.template_id)
        if bound is None:
            return expr
        type_id = bound.runtime_type_id()
        assert type_id is not None
        if not expr.accepts_concrete(type_id):
            raise ElaborationError(
                f"{subject} type variable '{expr.template_id}' does not allow selected type "
                f"{type_id!r}"
            )
        return bound
    if expr.kind in ("list", "asset", "stream"):
        assert expr.element is not None
        element = _substitute_type(expr.element, bindings, subject)
        return expr if element is expr.element else replace(expr, element=element)
    return expr


def _elaborate_dynamic_entries(
    entries: Sequence[DynamicEntry],
    parent: str,
    stored_input_ids: Sequence[str],
    choices: Mapping[str, str],
) -> tuple[list[InputSpec], tuple[tuple[str, str], ...], set[str]]:
    inputs: list[InputSpec] = []
    active: list[tuple[str, str]] = []
    consumed: set[str] = set()
    for entry in entries:
        path = _join_path(parent, entry.id)
        if isinstance(entry, InputSpec):
            inputs.append(replace(entry, id=path))
        elif isinstance(entry, InputFamilySpec):
            family_inputs, family_active, family_consumed = _elaborate_family(
                entry, path, stored_input_ids, choices
            )
            inputs.extend(family_inputs)
            active.extend(family_active)
            consumed.update(family_consumed)
        elif isinstance(entry, DynamicComboSpec):
            combo_inputs, combo_active, combo_consumed = _elaborate_combo(
                entry, path, stored_input_ids, choices
            )
            inputs.extend(combo_inputs)
            active.extend(combo_active)
            consumed.update(combo_consumed)
        else:
            slot_inputs, slot_active, slot_consumed = _elaborate_slot(
                entry, path, stored_input_ids, choices
            )
            inputs.extend(slot_inputs)
            active.extend(slot_active)
            consumed.update(slot_consumed)
    return inputs, tuple(active), consumed


def _elaborate_family(
    family: InputFamilySpec,
    path: str,
    stored_input_ids: Sequence[str],
    choices: Mapping[str, str],
) -> tuple[list[InputSpec], tuple[tuple[str, str], ...], set[str]]:
    prefix = path + "."
    suffixes: list[str] = []
    for input_id in stored_input_ids:
        if not input_id.startswith(prefix):
            continue
        suffix = input_id[len(prefix) :].split(".", 1)[0]
        if suffix and suffix not in suffixes:
            suffixes.append(suffix)
    for choice_path in choices:
        if not choice_path.startswith(prefix):
            continue
        suffix = choice_path[len(prefix) :].split(".", 1)[0]
        if suffix and suffix not in suffixes:
            suffixes.append(suffix)
    if len(suffixes) > MAX_FAMILY_MEMBERS:
        raise ElaborationError(
            f"input family '{path}' has {len(suffixes)} members, budget is {MAX_FAMILY_MEMBERS}"
        )
    if len(suffixes) < family.min_members:
        raise ElaborationError(
            f"input family '{path}' has {len(suffixes)} members, needs >= {family.min_members}"
        )
    if family.max_members is not None and len(suffixes) > family.max_members:
        raise ElaborationError(
            f"input family '{path}' has {len(suffixes)} members, allows <= {family.max_members}"
        )
    inputs: list[InputSpec] = []
    active: list[tuple[str, str]] = []
    consumed: set[str] = set()
    for suffix in suffixes:
        if not STRUCTURAL_ID_PATTERN.match(suffix):
            raise ElaborationError(f"input family '{path}' has invalid member suffix {suffix!r}")
        if family.member_names is not None and suffix not in family.member_names:
            raise ElaborationError(
                f"input family '{path}' member suffix {suffix!r} is not in member_names"
            )
        member_path = _join_path(path, suffix)
        if len(family.template) == 1 and isinstance(family.template[0], InputSpec):
            inputs.append(replace(family.template[0], id=member_path))
            continue
        nested_inputs, nested_active, nested_consumed = _elaborate_dynamic_entries(
            family.template, member_path, stored_input_ids, choices
        )
        inputs.extend(nested_inputs)
        active.extend(nested_active)
        consumed.update(nested_consumed)
    return inputs, tuple(active), consumed


def _choice(
    path: str,
    raw: object,
    required: bool,
) -> str | None:
    if raw is None:
        if required:
            raise ElaborationError(f"required dynamic construct '{path}' has no stored choice")
        return None
    if not isinstance(raw, str):
        raise ElaborationError(f"dynamic construct '{path}' has a non-string choice: {raw!r}")
    return raw


def _elaborate_combo(
    combo: DynamicComboSpec,
    path: str,
    stored_input_ids: Sequence[str],
    choices: Mapping[str, str],
) -> tuple[list[InputSpec], tuple[tuple[str, str], ...], set[str]]:
    consumed: set[str] = {path} if path in choices else set()
    if not combo.options and path not in choices:
        return [], (), consumed
    key = _choice(path, cast("object", choices.get(path)), combo.required)
    if key is None:
        return [], (), consumed
    option = combo.option(key)
    if option is None:
        raise ElaborationError(f"dynamic combo '{path}' has unknown option choice '{key}'")
    inputs, active, nested_consumed = _elaborate_dynamic_entries(
        option.inputs, path, stored_input_ids, choices
    )
    consumed.update(nested_consumed)
    return inputs, ((path, key), *active), consumed


def _elaborate_slot(
    slot: DynamicSlotSpec,
    path: str,
    stored_input_ids: Sequence[str],
    choices: Mapping[str, str],
) -> tuple[list[InputSpec], tuple[tuple[str, str], ...], set[str]]:
    if slot.variants is None:
        assert slot.slot_type is not None
        inputs = [
            InputSpec(
                id=path,
                type=slot.slot_type,
                required=bool(slot.required),
                doc=slot.doc,
                display_name=slot.display_name,
                force_input=slot.force_input,
            )
        ]
        if path not in stored_input_ids:
            return inputs, (), set()
        nested, active, consumed = _elaborate_dynamic_entries(
            slot.inputs, path, stored_input_ids, choices
        )
        inputs.extend(nested)
        return inputs, active, consumed
    if path not in choices and slot.required:
        raise ElaborationError(f"required slot '{path}' has no stored variant choice")
    key = _choice(path, cast("object", choices.get(path)), bool(slot.required))
    consumed: set[str] = {path} if path in choices else set()
    if key is None:
        return [], (), consumed
    variant = slot.variant(key)
    if variant is None:
        raise ElaborationError(f"dynamic slot '{path}' has unknown variant choice '{key}'")
    inputs = [
        InputSpec(
            id=path,
            type=variant.type,
            required=True,
            doc=slot.doc,
            display_name=slot.display_name,
        )
    ]
    nested, active, nested_consumed = _elaborate_dynamic_entries(
        (*slot.inputs, *variant.inputs), path, stored_input_ids, choices
    )
    inputs.extend(nested)
    consumed.update(nested_consumed)
    return inputs, ((path, key), *active), consumed


def _validate_projected_namespace(schema: NodeSchema, inputs: Sequence[InputSpec]) -> None:
    ids = [spec.id for spec in inputs]
    if len(set(ids)) != len(ids):
        duplicates = sorted({input_id for input_id in ids if ids.count(input_id) > 1})
        raise ElaborationError(
            f"{schema.node_type}: projected input namespace collision: {', '.join(duplicates)}"
        )


def _elaborate_outputs(
    schema: NodeSchema,
    output_members: Mapping[str, Sequence[str]],
    stored_inputs: Mapping[str, object] | None,
) -> list[OutputSpec]:
    declared = {fam.id for fam in schema.output_families}
    unknown = set(output_members) - declared
    if unknown:
        raise ElaborationError(
            f"{schema.node_type}: output members stored for undeclared "
            f"family(ies): {', '.join(sorted(unknown))}"
        )

    static_ids = {out.id for out in schema.outputs}
    outputs = list(schema.outputs)
    for fam in schema.output_families:
        suffixes = tuple(output_members.get(fam.id, ()))
        if fam.count is not None:
            value = None if stored_inputs is None else stored_inputs.get(fam.count.input)
            if type(value) is not int:
                raise ElaborationError(
                    f"{schema.node_type}: output family '{fam.id}' count input "
                    f"'{fam.count.input}' must be a stored untyped integer literal"
                )
            if value < 0:
                raise ElaborationError(
                    f"{schema.node_type}: output family '{fam.id}' count must be >= 0"
                )
            if value > MAX_FAMILY_MEMBERS:
                raise ElaborationError(
                    f"{schema.node_type}: output family '{fam.id}' count exceeds "
                    f"the {MAX_FAMILY_MEMBERS} member budget"
                )
            if value < fam.min_members or (fam.max_members is not None and value > fam.max_members):
                raise ElaborationError(
                    f"{schema.node_type}: output family '{fam.id}' count is outside its bounds"
                )
            expected = tuple(str(index) for index in range(value))
            if suffixes != expected:
                raise ElaborationError(
                    f"{schema.node_type}: output family '{fam.id}' members must exactly equal "
                    f"the canonical count suffixes {expected!r}"
                )
        if len(suffixes) > MAX_FAMILY_MEMBERS:
            raise ElaborationError(
                f"{schema.node_type}: output family '{fam.id}' has "
                f"{len(suffixes)} members, budget is {MAX_FAMILY_MEMBERS}"
            )
        seen: set[str] = set()
        # Membership comes from the document, so the declared element type
        # is a promise the outside world has not made: check it for real.
        for suffix in cast("tuple[object, ...]", suffixes):
            if not isinstance(suffix, str) or not suffix:
                raise ElaborationError(
                    f"{schema.node_type}: output family '{fam.id}' has an "
                    f"empty or non-string member suffix: {suffix!r}"
                )
            if suffix in seen:
                raise ElaborationError(
                    f"{schema.node_type}: output family '{fam.id}' has "
                    f"duplicate member suffix '{suffix}'"
                )
            seen.add(suffix)
            member_id = fam.member_id(suffix)
            if member_id in static_ids:
                raise ElaborationError(
                    f"{schema.node_type}: output member '{member_id}' collides with a static output"
                )
            outputs.append(
                OutputSpec(id=member_id, type=fam.type, doc=fam.doc, preview=fam.preview)
            )
    if schema.output_descriptors is not None:
        spec = schema.output_descriptors
        value = None if stored_inputs is None else stored_inputs.get(spec.input)
        try:
            descriptors = elaborate_output_descriptors(spec, value)
        except ValueError as error:
            raise ElaborationError(f"{schema.node_type}: {error}") from error
        occupied = {output.id for output in outputs} | declared
        for descriptor in descriptors:
            if descriptor.id in occupied:
                raise ElaborationError(
                    f"{schema.node_type}: descriptor '{descriptor.id}' collides with an output"
                )
        outputs.extend(descriptors)
    return outputs
