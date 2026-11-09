"""Same-type migrations for the version-1 static image-node interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from dinkster_api.v1 import (
    ComboWidget,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilyMapping,
    InputFamilyMember,
    InputFamilySpec,
    InputSpec,
    MappingSource,
    NodeSchema,
    OutputFamilyMapping,
    ReplacementCase,
    ReplacementLink,
    ReplacementMigration,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    ValueTransform,
)


def with_mask_polarity(schema: NodeSchema) -> NodeSchema:
    """Rename polarity controls and migrate both flat and materialized input paths."""
    names = {"alpha_mask_polarity", "alpha_polarity", "mask_polarity"}
    values = {
        "opacity": "coverage",
        "coverage": "coverage",
        "transparency": "transparency",
        "mask_is_opacity": "coverage",
        "mask_is_transparency": "transparency",
    }
    transform = ValueTransform.enum_rename(values)

    def path(value: str) -> str:
        prefix, _, local = value.rpartition(".")
        return _path(prefix, "mask_polarity") if local in names else value

    def entry[T: InputSpec | InputFamilySpec | DynamicComboSpec | DynamicSlotSpec](value: T) -> T:
        if isinstance(value, InputSpec) and value.id in names:
            assert isinstance(value.widget, ComboWidget)
            return cast(
                T,
                replace(
                    value,
                    id="mask_polarity",
                    default=values[str(value.default)],
                    widget=replace(value.widget, options=("coverage", "transparency")),
                ),
            )
        if isinstance(value, DynamicComboSpec):
            return cast(
                T,
                replace(
                    value,
                    options=tuple(
                        replace(option, inputs=tuple(entry(child) for child in option.inputs))
                        for option in value.options
                    ),
                ),
            )
        if isinstance(value, DynamicSlotSpec):
            return cast(
                T,
                replace(
                    value,
                    inputs=tuple(entry(child) for child in value.inputs),
                    variants=tuple(
                        replace(variant, inputs=tuple(entry(child) for child in variant.inputs))
                        for variant in value.variants or ()
                    ),
                ),
            )
        return value

    def migrate_cases(case: ReplacementCase, *, dotted: bool) -> tuple[ReplacementCase, ...]:
        mappings: list[tuple[str, MappingSource]] = []
        polarity_source = ""
        polarity_target = ""
        for target, source in case.inputs:
            if dotted:
                source = replace(source, input=target)
            if target.rsplit(".", 1)[-1] in names:
                polarity_source, polarity_target = source.input, path(target)
                source = MappingSource.from_value(source.input, transform)
            mappings.append((path(target), source))
        migrated = replace(case, inputs=tuple(mappings))
        if not polarity_source:
            return (migrated,)
        linked = replace(
            migrated,
            when=ReplacementPredicate.all_of(
                case.when or ReplacementPredicate.always(),
                ReplacementPredicate.input_connected(polarity_source),
            ),
            inputs=tuple(
                (
                    target,
                    MappingSource.copy(polarity_source) if target == polarity_target else source,
                )
                for target, source in mappings
            ),
        )
        return linked, migrated

    defaults = _default_choices((*schema.combos, *schema.slots))
    states = _states((*schema.combos, *schema.slots))
    states = [state for state in states if state != defaults] + [defaults]
    cases = tuple(
        case
        for state in states
        for case in migrate_cases(
            _case(schema, state, fallback=state == defaults, current_only_inputs=frozenset()),
            dotted=True,
        )
    )
    historical = tuple(
        target
        for state in states
        for target, _ in _case(schema, state, fallback=True, current_only_inputs=frozenset()).inputs
        if target not in {spec.id for spec in schema.inputs} or target in names
    )
    rule = ReplacementRule(
        from_type=schema.node_type,
        cases=cases,
        migration=ReplacementMigration(
            tuple(
                dict.fromkeys(
                    (
                        *historical,
                        *(combo.id for combo in schema.combos),
                    )
                )
            )
        ),
    )
    return replace(
        schema,
        version=schema.version + 1,
        inputs=tuple(entry(spec) for spec in schema.inputs),
        combos=tuple(entry(combo) for combo in schema.combos),
        slots=tuple(entry(slot) for slot in schema.slots),
        replacements=(
            rule,
            *(
                replace(
                    old_rule,
                    cases=tuple(
                        migrated
                        for case in old_rule.cases
                        for migrated in migrate_cases(case, dotted=False)
                    ),
                )
                for old_rule in schema.replacements
            ),
        ),
    )


def _path(prefix: str, local: str) -> str:
    return f"{prefix}.{local}" if prefix else local


def _merge_states(left: list[dict[str, str]], right: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict((*first.items(), *second.items())) for first in left for second in right]


def _entry_states(
    entry: object,
    prefix: str,
    current_only_choices: Mapping[str, frozenset[str]],
) -> list[dict[str, str]]:
    if isinstance(entry, DynamicComboSpec):
        construct = _path(prefix, entry.id)
        states: list[dict[str, str]] = []
        for option in entry.options:
            if option.key in current_only_choices.get(construct, frozenset()):
                continue
            children = _states(option.inputs, construct, current_only_choices)
            states.extend(dict(((construct, option.key), *child.items())) for child in children)
        return states
    if isinstance(entry, DynamicSlotSpec):
        if entry.variants is None:
            raise ValueError("version-1 image migrations require closed dynamic slots")
        construct = _path(prefix, entry.id)
        states = [{}] if not entry.required else []
        for variant in entry.variants:
            children = _states(
                (*entry.inputs, *variant.inputs),
                construct,
                current_only_choices,
            )
            states.extend(dict(((construct, variant.key), *child.items())) for child in children)
        return states
    return [{}]


def _states(
    entries: Sequence[object],
    prefix: str = "",
    current_only_choices: Mapping[str, frozenset[str]] | None = None,
) -> list[dict[str, str]]:
    current_only_choices = current_only_choices or {}
    states: list[dict[str, str]] = [{}]
    for entry in entries:
        states = _merge_states(
            states,
            _entry_states(entry, prefix, current_only_choices),
        )
    return states


def _default_choices(entries: Sequence[object], prefix: str = "") -> dict[str, str]:
    choices: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, DynamicComboSpec):
            continue
        if entry.default is None:
            raise ValueError(f"version-1 image migration requires a default for {entry.id}")
        construct = _path(prefix, entry.id)
        choices[construct] = entry.default
        option = entry.option(entry.default)
        assert option is not None
        choices.update(_default_choices(option.inputs, construct))
    return choices


def _dynamic_ids(entries: Sequence[object]) -> tuple[str, ...]:
    ids: list[str] = []

    def collect(children: Sequence[object]) -> None:
        for entry in children:
            if isinstance(entry, (InputSpec, InputFamilySpec, DynamicComboSpec, DynamicSlotSpec)):
                if entry.id not in ids:
                    ids.append(entry.id)
            if isinstance(entry, DynamicComboSpec):
                for option in entry.options:
                    collect(option.inputs)
            elif isinstance(entry, DynamicSlotSpec):
                collect(entry.inputs)
                for variant in entry.variants or ():
                    collect(variant.inputs)

    collect(entries)
    return tuple(ids)


def _active_interface(
    entries: Sequence[object], choices: dict[str, str], prefix: str = ""
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, bool]]]:
    inputs: list[tuple[str, str]] = []
    constructs: list[tuple[str, str, str, bool]] = []
    for entry in entries:
        path = (
            _path(prefix, entry.id)
            if isinstance(entry, (InputSpec, InputFamilySpec, DynamicComboSpec, DynamicSlotSpec))
            else ""
        )
        if isinstance(entry, InputSpec):
            inputs.append((path, entry.id))
        elif isinstance(entry, DynamicComboSpec):
            choice = choices[path]
            constructs.append((path, entry.id, choice, False))
            option = entry.option(choice)
            assert option is not None
            child_inputs, child_constructs = _active_interface(option.inputs, choices, path)
            inputs.extend(child_inputs)
            constructs.extend(child_constructs)
        elif isinstance(entry, DynamicSlotSpec) and path in choices:
            choice = choices[path]
            constructs.append((path, entry.id, choice, True))
            inputs.append((path, entry.id))
            variant = entry.variant(choice)
            assert variant is not None
            child_inputs, child_constructs = _active_interface(
                (*entry.inputs, *variant.inputs), choices, path
            )
            inputs.extend(child_inputs)
            constructs.extend(child_constructs)
    return inputs, constructs


def _active_source(input_id: str) -> ReplacementPredicate:
    return ReplacementPredicate.any_of(
        ReplacementPredicate.input_connected(input_id),
        ReplacementPredicate.all_of(
            ReplacementPredicate.value_present(input_id),
            ReplacementPredicate.not_(ReplacementPredicate.value_equals(input_id, None)),
        ),
    )


def _input_family_mappings(schema: NodeSchema) -> dict[str, InputFamilyMapping]:
    mappings: dict[str, InputFamilyMapping] = {}
    for family in schema.input_families:
        template = {
            entry.id: MappingSource.copy(entry.id)
            for entry in family.template
            if isinstance(entry, InputSpec)
        }
        if len(template) != len(family.template):
            raise ValueError("version-1 image migrations require static input-family templates")
        mappings[family.id] = InputFamilyMapping.copy(family.id, inputs=template)
    return mappings


def _case(
    schema: NodeSchema,
    choices: dict[str, str],
    *,
    fallback: bool,
    current_only_inputs: frozenset[str],
) -> ReplacementCase:
    active_inputs, constructs = _active_interface((*schema.combos, *schema.slots), choices)
    mappings = {
        spec.id: MappingSource.copy(spec.id)
        for spec in schema.inputs
        if spec.id not in current_only_inputs
    }
    mappings.update(
        {
            path: MappingSource.copy(input_id)
            for path, input_id in active_inputs
            if input_id not in current_only_inputs
        }
    )
    predicates: list[ReplacementPredicate] = []
    if not fallback:
        historical_constructs: list[str] = []
        for _path_value, logical, _choice, slot in constructs:
            if (
                not slot
                and not (schema.node_type == "dinkster.image.crop" and logical == "source")
                and logical not in historical_constructs
            ):
                historical_constructs.append(logical)
        predicates.extend(
            ReplacementPredicate.not_(ReplacementPredicate.input_connected(input_id))
            for input_id in historical_constructs
        )
        for _path, logical, choice, slot in constructs:
            if slot:
                predicates.append(_active_source(logical))
            elif schema.node_type == "dinkster.image.crop" and logical == "source":
                if choice in ("region", "mask"):
                    predicates.append(_active_source(choice))
            else:
                predicates.append(ReplacementPredicate.value_equals(logical, choice))
    return ReplacementCase.build(
        schema.node_type,
        when=None if fallback else ReplacementPredicate.all_of(*predicates),
        slot_variants=choices,
        inputs=mappings,
        input_families=_input_family_mappings(schema),
        output_families={
            family.id: OutputFamilyMapping.copy(family.id) for family in schema.output_families
        },
        outputs={output.id: output.id for output in schema.outputs},
    )


def with_v1_migration(
    schema: NodeSchema,
    *,
    current_only_inputs: frozenset[str] = frozenset(),
    current_only_choices: Mapping[str, frozenset[str]] | None = None,
) -> NodeSchema:
    """Attach a flat-version-1 migration rule to a dynamic image schema."""
    current_only_choices = current_only_choices or {}
    entries = (*schema.combos, *schema.slots)
    historical = [
        input_id for input_id in _dynamic_ids(entries) if input_id not in current_only_inputs
    ]
    if schema.node_type == "dinkster.image.crop":
        historical.remove("source")
    defaults = _default_choices(entries)
    states = _states(entries, current_only_choices=current_only_choices)
    states.sort(
        key=lambda choices: (
            0 if choices.get("source") == "region" else 1 if choices.get("source") == "mask" else 2,
            -sum(path in {slot.id for slot in schema.slots} for path in choices),
        )
    )
    cases = [
        _case(
            schema,
            choices,
            fallback=False,
            current_only_inputs=current_only_inputs,
        )
        for choices in states
        if choices != defaults
    ]
    cases.append(
        _case(
            schema,
            defaults,
            fallback=True,
            current_only_inputs=current_only_inputs,
        )
    )
    rule = ReplacementRule(
        from_type=schema.node_type,
        cases=tuple(cases),
        migration=ReplacementMigration(tuple(historical)),
    )
    return replace(schema, replacements=(*schema.replacements, rule))


_RESIZE_NOTE = (
    "Imported KJNodes center-fit uses centered fill; divisibility is finalized by "
    "cropping after resize; no companion mask is emitted when no mask or padding exists."
)
_RESIZE_OUTPUTS = {"image": "image", "mask": "mask"}
_RESIZE_ANCHORS = ("disabled", "center", "top", "bottom", "left", "right")


def _selected(path: str, choice: str, default: str) -> ReplacementPredicate:
    selected = ReplacementPredicate.value_equals(path, choice)
    if choice != default:
        return selected
    return ReplacementPredicate.any_of(
        selected,
        ReplacementPredicate.not_(ReplacementPredicate.value_present(path)),
    )


def _all(*predicates: ReplacementPredicate) -> ReplacementPredicate:
    return ReplacementPredicate.all_of(*predicates)


def _resize_case(
    *,
    when: ReplacementPredicate | None,
    choices: Mapping[str, str],
    inputs: Mapping[str, MappingSource],
    interpolation: str,
    mask: str | None,
    nodes: Mapping[str, ReplacementNode] | None = None,
    input_families: Mapping[str, InputFamilyMapping] | None = None,
    links: Sequence[ReplacementLink] = (),
) -> ReplacementCase:
    mappings = {
        "image": MappingSource.copy("image"),
        "interpolation": MappingSource.copy(interpolation),
        **inputs,
    }
    if mask is not None:
        mappings["mask"] = MappingSource.copy(mask)
    return ReplacementCase.build(
        "dinkster.image.resize",
        when=when,
        nodes=nodes,
        slot_variants=choices,
        inputs=mappings,
        input_families=input_families,
        links=links,
        outputs=_RESIZE_OUTPUTS,
    )


def _pixel_resize_cases(
    *,
    when: ReplacementPredicate,
    width: str,
    height: str,
    interpolation: str,
    mask: str | None,
    divisibility: str,
) -> tuple[ReplacementCase, ...]:
    cases: list[ReplacementCase] = []
    for width_linked, height_linked in ((True, True), (True, False), (False, True), (False, False)):
        predicates = [when]
        for source, linked in ((width, width_linked), (height, height_linked)):
            connected = ReplacementPredicate.input_connected(source)
            predicates.append(connected if linked else ReplacementPredicate.not_(connected))
        nodes = {
            "pixels": ReplacementNode.build(
                "dinkster.math.expression",
                values={"expression": "a * b / 1048576"},
            )
        }
        inputs: dict[str, MappingSource] = {}
        links = [ReplacementLink("pixels:float", "target.megapixels")]
        for local_id, source, linked, member in (
            ("width", width, width_linked, "a"),
            ("height", height, height_linked, "b"),
        ):
            if linked:
                continue
            nodes[local_id] = ReplacementNode.build("dinkster.int", values={"value": 512})
            inputs[f"{local_id}:value"] = MappingSource.from_value(source)
            links.append(ReplacementLink(f"{local_id}:value", f"pixels:values.{member}"))
        cases.append(
            _resize_case(
                when=_all(*predicates),
                choices={
                    "target": "total_pixels",
                    "mode": "stretch",
                    "divisibility": "crop",
                },
                inputs={
                    "divisibility.multiple_of": MappingSource.copy(divisibility),
                    **inputs,
                },
                interpolation=interpolation,
                mask=mask,
                nodes=nodes,
                input_families={
                    "pixels:values": InputFamilyMapping.from_members(
                        InputFamilyMember.build(
                            "a",
                            inputs={"value": MappingSource.link(width)},
                        ),
                        InputFamilyMember.build(
                            "b",
                            inputs={"value": MappingSource.link(height)},
                        ),
                    )
                },
                links=links,
            )
        )
    return tuple(cases)


def _resize_migration_rule(*, dotted: bool) -> ReplacementRule:
    def path(compatibility: str, input_id: str, *, construct: str = "") -> str:
        if not dotted:
            return input_id
        prefix = compatibility
        if construct:
            prefix = f"{prefix}.{construct}"
        return f"{prefix}.{input_id}"

    compatibility_path = "compatibility"
    cases: list[ReplacementCase] = []

    def compatibility(choice: str) -> ReplacementPredicate:
        return _selected(compatibility_path, choice, "native")

    native = "compatibility"
    native_target = path(native, "target") if dotted else "target"
    native_mode = path(native, "mode") if dotted else "mode"
    native_interpolation = path(native, "interpolation")
    native_mask = path(native, "mask")
    for target in (
        "dimensions",
        "width",
        "height",
        "longest",
        "shortest",
        "factor",
        "total_pixels",
        "match",
        "multiple_of",
    ):
        target_choices = {
            "target": "dimensions" if target == "multiple_of" else target,
            "divisibility": "none",
        }
        target_inputs: dict[str, MappingSource] = {}
        if target == "dimensions":
            target_inputs = {
                "target.width": MappingSource.copy(path(native, "width", construct="target")),
                "target.height": MappingSource.copy(path(native, "height", construct="target")),
            }
        elif target in ("width", "height"):
            target_inputs[f"target.{target}"] = MappingSource.copy(
                path(native, target, construct="target")
            )
        elif target in ("longest", "shortest"):
            target_inputs["target.size"] = MappingSource.copy(
                path(native, "size", construct="target")
            )
        elif target == "factor":
            target_inputs["target.factor"] = MappingSource.copy(
                path(native, "factor", construct="target")
            )
        elif target == "total_pixels":
            target_inputs = {
                "target.megapixels": MappingSource.copy(
                    path(native, "megapixels", construct="target")
                ),
                "target.resolution_steps": MappingSource.copy(
                    path(native, "resolution_steps", construct="target")
                ),
            }
        elif target == "match":
            target_inputs["target.reference"] = MappingSource.copy(
                path(native, "reference_image", construct="target")
            )
        else:
            target_inputs = {
                "target.width": MappingSource.constant(0),
                "target.height": MappingSource.constant(0),
                "divisibility.multiple_of": MappingSource.copy(
                    path(native, "multiple_of", construct="target")
                ),
            }
            target_choices["divisibility"] = "crop"
        for mode in ("stretch", "fit", "fill", "pad"):
            choices = {**target_choices, "mode": mode}
            inputs = dict(target_inputs)
            if mode == "fill":
                inputs["mode.mode_anchor"] = MappingSource.constant("center")
            elif mode == "pad":
                choices["mode.mode_padding"] = "constant"
                inputs["mode.mode_anchor"] = MappingSource.constant("center")
                inputs["mode.mode_padding.pad_value"] = MappingSource.copy(
                    path(native, "pad_value", construct="mode")
                )
            cases.append(
                _resize_case(
                    when=_all(
                        compatibility("native"),
                        _selected(native_target, target, "dimensions"),
                        _selected(native_mode, mode, "stretch"),
                    ),
                    choices=choices,
                    inputs=inputs,
                    interpolation=native_interpolation,
                    mask=native_mask,
                )
            )

    for compatibility_name in ("kjnodes_v1", "kjnodes_v2", "essentials"):
        prefix = "compatibility" if dotted else compatibility_name
        mode_path = path(prefix, "mode") if dotted else "mode"
        interpolation = path(prefix, "interpolation")
        width = path(prefix, "width")
        height = path(prefix, "height")
        mask = path(prefix, "mask") if compatibility_name == "kjnodes_v2" else None
        dimensions = {
            "target.width": MappingSource.copy(width),
            "target.height": MappingSource.copy(height),
        }
        compatibility_predicate = compatibility(compatibility_name)

        if compatibility_name == "kjnodes_v1":
            reference = path(prefix, "reference_image")
            reference_active = ReplacementPredicate.any_of(
                ReplacementPredicate.input_connected(reference),
                ReplacementPredicate.value_present(reference),
            )
            for has_reference in (True, False):
                reference_predicate = (
                    reference_active
                    if has_reference
                    else ReplacementPredicate.not_(reference_active)
                )
                target_choices = {"target": "match" if has_reference else "dimensions"}
                target_inputs = (
                    {"target.reference": MappingSource.copy(reference)}
                    if has_reference
                    else dimensions
                )
                for mode in ("stretch", "fit", "fill"):
                    for anchor in _RESIZE_ANCHORS:
                        natural_anchor = "center" if anchor == "disabled" else anchor
                        natural_mode = "fill" if mode == "fit" and anchor == "center" else mode
                        choices = {
                            **target_choices,
                            "mode": natural_mode,
                            "divisibility": "crop",
                        }
                        case_inputs = {
                            **target_inputs,
                            "divisibility.multiple_of": MappingSource.copy(
                                path(prefix, "divisible_by")
                            ),
                        }
                        if natural_mode == "fill":
                            case_inputs["mode.mode_anchor"] = MappingSource.constant(natural_anchor)
                        cases.append(
                            _resize_case(
                                when=_all(
                                    compatibility_predicate,
                                    reference_predicate,
                                    _selected(mode_path, mode, "stretch"),
                                    _selected(path(prefix, "anchor"), anchor, "disabled"),
                                ),
                                choices=choices,
                                inputs=case_inputs,
                                interpolation=interpolation,
                                mask=None,
                            )
                        )
            continue

        if compatibility_name == "kjnodes_v2":
            divisible_by = path(prefix, "divisible_by")
            if dotted:
                cases.extend(
                    _pixel_resize_cases(
                        when=_all(
                            compatibility_predicate,
                            _selected(mode_path, "total_pixels", "stretch"),
                        ),
                        width=width,
                        height=height,
                        interpolation=interpolation,
                        mask=mask,
                        divisibility=divisible_by,
                    )
                )
            mode_map: dict[str, tuple[str, str | None]] = {
                "stretch": ("stretch", None),
                "fit": ("fit", None),
                "fill": ("fill", None),
            }
            if dotted:
                mode_map.update(
                    {
                        "pad": ("pad", "constant"),
                        "pad_edge": ("pad", "edge_average"),
                        "pad_edge_pixel": ("pad", "edge_pixel"),
                        "pillarbox_blur": ("pad", "blurred_background"),
                    }
                )
            for source_mode, (mode, padding) in mode_map.items():
                anchors = _RESIZE_ANCHORS if mode in ("fill", "pad") else ("disabled",)
                for anchor in anchors:
                    natural_anchor = "center" if anchor == "disabled" else anchor
                    choices = {
                        "target": "dimensions",
                        "mode": mode,
                        "divisibility": "crop",
                    }
                    inputs = {
                        **dimensions,
                        "divisibility.multiple_of": MappingSource.copy(divisible_by),
                    }
                    if mode in ("fill", "pad"):
                        inputs["mode.mode_anchor"] = MappingSource.constant(natural_anchor)
                    if padding is not None:
                        choices["mode.mode_padding"] = padding
                    if padding == "constant":
                        inputs["mode.mode_padding.pad_color"] = MappingSource.copy(
                            path(prefix, "pad_color", construct="mode")
                        )
                    cases.append(
                        _resize_case(
                            when=_all(
                                compatibility_predicate,
                                _selected(mode_path, source_mode, "stretch"),
                                _selected(path(prefix, "anchor"), anchor, "disabled"),
                            ),
                            choices=choices,
                            inputs=inputs,
                            interpolation=interpolation,
                            mask=mask,
                        )
                    )
            continue

        apply_map = {
            "always": "always",
            "downscale_if_bigger": "only_if_bigger",
            "upscale_if_smaller": "only_if_smaller",
            "if_bigger_area": "only_if_bigger_area",
            "if_smaller_area": "only_if_smaller_area",
        }
        for mode in ("stretch", "fit", "fill", "pad"):
            for source_apply, apply_value in apply_map.items():
                choices = {
                    "target": "dimensions",
                    "mode": mode,
                    "divisibility": "crop",
                }
                inputs = {
                    **dimensions,
                    "apply": MappingSource.constant(apply_value),
                    "divisibility.multiple_of": MappingSource.copy(path(prefix, "divisible_by")),
                }
                if mode in ("fill", "pad"):
                    inputs["mode.mode_anchor"] = MappingSource.constant("center")
                if mode == "pad":
                    choices["mode.mode_padding"] = "constant"
                cases.append(
                    _resize_case(
                        when=_all(
                            compatibility_predicate,
                            _selected(mode_path, mode, "stretch"),
                            _selected(path(prefix, "condition"), source_apply, "always"),
                        ),
                        choices=choices,
                        inputs=inputs,
                        interpolation=interpolation,
                        mask=None,
                    )
                )

    cases.append(
        _resize_case(
            when=None,
            choices={"target": "dimensions", "mode": "stretch", "divisibility": "none"},
            inputs={},
            interpolation="interpolation" if not dotted else "compatibility.interpolation",
            mask="mask" if not dotted else "compatibility.mask",
        )
    )
    historical = (
        (
            "target",
            "width",
            "height",
            "size",
            "factor",
            "megapixels",
            "multiple_of",
            "resolution_steps",
            "mode",
            "pad_value",
            "reference_image",
            "compatibility",
            "divisible_by",
            "anchor",
            "condition",
        )
        if not dotted
        else (
            "compatibility",
            "compatibility.target",
            "compatibility.target.width",
            "compatibility.target.height",
            "compatibility.target.size",
            "compatibility.target.factor",
            "compatibility.target.megapixels",
            "compatibility.target.multiple_of",
            "compatibility.target.resolution_steps",
            "compatibility.target.reference_image",
            "compatibility.mode",
            "compatibility.mode.pad_value",
            "compatibility.mode.pad_color",
            "compatibility.width",
            "compatibility.height",
            "compatibility.interpolation",
            "compatibility.mask",
            "compatibility.divisible_by",
            "compatibility.anchor",
            "compatibility.reference_image",
            "compatibility.condition",
        )
    )
    return ReplacementRule(
        from_type="dinkster.image.resize",
        cases=tuple(cases),
        note=_RESIZE_NOTE,
        migration=ReplacementMigration(historical),
    )


def with_resize_migrations(schema: NodeSchema) -> NodeSchema:
    """Attach migrations for the flat v1 and compatibility-grouped v3 shapes."""
    return replace(
        schema,
        replacements=(
            *schema.replacements,
            _resize_migration_rule(dotted=True),
            _resize_migration_rule(dotted=False),
        ),
    )
