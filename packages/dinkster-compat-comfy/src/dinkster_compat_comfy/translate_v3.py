"""ComfyUI V3 (comfy_entrypoint) -> Dinkster schema translation, pure.

The legacy quarantine runs v1 packs; V3 packs are the port-them-properly
case (DESIGN 3.8). This module is the porting half of that promise: given
a V3 ``Schema`` object (what ``ComfyNode.define_schema()`` returns), it
produces an honest Dinkster ``NodeSchema`` for ``dinkster port`` to emit as a
native pack skeleton. It never manufactures runtime wrappers - V3 nodes
are not executed under compat, deliberately - and it never imports
ComfyUI: everything is read duck-typed off the schema object, so the
conversion is testable with plain fakes and only ever runs against real
``comfy_api`` objects inside the disposable probe subprocess.

Translation rules (the v1 rules extended with what V3 can actually say):

- io_type strings map exactly like v1 type strings: INT/FLOAT/STRING/
  BOOLEAN -> core envelope types, COMBO -> core.combo (choice vocabulary is
  UI affordance within that identity), ``*`` -> wildcard, anything else -> opaque
  ``comfy.T``. COMBO ``options`` that are all non-empty strings ride
  along as a static ComboWidget (the v1 combo rule), skipped under
  ``is_input_list`` where the socket is a list. Disguised-boolean combos
  (enable/disable and kin) become core.boolean with a BooleanWidget, and
  BOOLEAN inputs keep their label_on/label_off as the same widget - the
  port skeleton's execute is the human's to adapt, and the porting fact
  is spelled in the schema instead of lost.
- MultiType inputs serialize their io_type as a comma-join ("A,B"); that
  becomes a real union TypeExpr. A ``*`` member collapses to wildcard.
- MatchType inputs/outputs carry a template (template_id + allowed
  types); that is exactly a Dinkster type variable, so it translates to
  ``TypeExpr.variable(template_id, allowed)``.
- Autogrow inputs are dynamic families; Dinkster has that shape natively
  (InputFamilySpec, hazard H10), so prefix and names vocabularies plus bounds
  survive. DynamicCombo options and DynamicSlot dependents recurse through
  the same DynamicEntry grammar.
- ``is_input_list`` is v1's INPUT_IS_LIST under a new name: every input
  wraps in list<T> and declared defaults wrap in a one-element list.
  Per-output ``is_output_list`` wraps only that output.
- V3 lifecycle flags map to the Dinkster fields that mean the same thing:
  ``is_output_node`` -> output_node (and never cached, like v1),
  ``not_idempotent`` -> idempotent=False, ``is_api_node`` -> io_bound
  (a network-waiting node, exempt from the compute lane),
  ``is_deprecated`` -> a Deprecation record with a placeholder message
  (V3 has no deprecation prose; the port TODO says write it),
  ``is_dev_only`` -> search_visibility="hidden".
- Tooltips become InputSpec/OutputSpec doc strings - V3 carries them per
  socket, so the port keeps them.
- ``search_aliases`` have no Dinkster schema home (aliases are submission
  resolution, not search synonyms); the probe records them as a porting
  fact instead of misfiling them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, cast

from dinkster_schema import (
    BooleanWidget,
    Deprecation,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicEntry,
    DynamicSlotSpec,
    InputFamilySpec,
    InputSpec,
    MultiComboWidget,
    NodeSchema,
    OutputSpec,
    SelectorSpec,
    SourceFilenameSpec,
    TypeExpr,
    Widget,
)
from dinkster_values import CORE_COMBO, CORE_INT, CORE_STRING

from .translate import (
    CUSTOM_COMBO_FAMILY_ID,
    CUSTOM_COMBO_NODE_ID,
    CUSTOM_COMBO_OPTION_NAMES,
    PRIMITIVES,
    CompatError,
    CompatTranslation,
    boolean_combo,
    combo_widget,
    comfy_type_id,
    dynamic_entries_occupy_gpu,
    number_widget,
    selector_lazy_inputs,
    source_asset_widget,
    string_widget,
    trusted_v3_remote_combo,
    trusted_v3_remote_multicombo,
)

#: io_type markers of the V3 structural input kinds (comfy_api _io.py).
V3_AUTOGROW_IO_TYPE = "COMFY_AUTOGROW_V3"
V3_DYNAMICCOMBO_IO_TYPE = "COMFY_DYNAMICCOMBO_V3"
V3_DYNAMICSLOT_IO_TYPE = "COMFY_DYNAMICSLOT_V3"
V3_MATCHTYPE_IO_TYPE = "COMFY_MATCHTYPE_V3"


def _is_structural_marker(io_type: str) -> bool:
    """comfy_api marks structural (dynamic) port kinds ``COMFY_*_V3``.
    Autogrow/MatchType translate above; any OTHER marker (DynamicCombo,
    DynamicSlot, future kinds) must refuse loudly - falling through to the
    opaque branch would mint a type id like ``comfy.COMFY_DYNAMICCOMBO_V3``
    that looks real and means nothing."""
    return io_type.startswith("COMFY_") and io_type.endswith("_V3")


#: io_type whose vocabulary is widget data within core.combo identity.
_COMBO_IO_TYPE = "COMBO"


def _exact_enum_value(
    value: object,
    *,
    class_name: str,
    members: dict[str, str],
    field: str,
) -> str:
    if not isinstance(value, Enum) or type(value).__name__ != class_name:
        raise CompatError(f"{field} must be an exact {class_name}")
    enum_type = type(value)
    if {name: member.value for name, member in enum_type.__members__.items()} != members:
        raise CompatError(f"{field} must be an exact {class_name}")
    raw_value = value.value
    if type(raw_value) is not str:
        raise CompatError(f"{field} must be an exact {class_name}")
    return raw_value


def _io_type_of(port: object) -> str:
    """The declared io_type string of one V3 input/output, via the API's
    own accessor (widget overrides and MultiType comma-joins included)."""
    get = getattr(port, "get_io_type", None)
    if callable(get):
        return str(get())
    return str(getattr(port, "io_type", ""))


def _member_type_id(io_type: str, translation: CompatTranslation) -> str:
    """One union/variable member io_type -> a concrete Dinkster type id."""
    marker = next(
        (
            member
            for member in (part.strip() for part in io_type.split(","))
            if _is_structural_marker(member)
        ),
        None,
    )
    if marker is not None:
        raise CompatError(f"V3 dynamic type marker {marker} is not portable")
    if io_type == _COMBO_IO_TYPE:
        return CORE_COMBO
    if io_type == "COLOR":
        return CORE_STRING
    primitive = PRIMITIVES.get(io_type)
    if primitive is not None:
        return primitive
    opaque = comfy_type_id(io_type)
    translation.opaque_types.add(opaque)
    return opaque


def translate_v3_type(io_type: str, translation: CompatTranslation) -> TypeExpr:
    """A V3 io_type string -> TypeExpr, registering opaque comfy.* types.

    Comma-joined strings (MultiType's serialization) become unions; a
    ``*`` anywhere collapses to the wildcard, exactly as permissive as
    the declaration."""
    members = [part.strip() for part in io_type.split(",") if part.strip()]
    if not members:
        raise CompatError("empty V3 io_type declaration")
    marker = next((member for member in members if _is_structural_marker(member)), None)
    if marker is not None:
        raise CompatError(f"V3 dynamic type marker {marker} is not portable")
    if "*" in members:
        return TypeExpr.wildcard()
    if len(members) == 1:
        return TypeExpr.concrete(_member_type_id(members[0], translation))
    seen: dict[str, None] = {}
    for member in members:
        seen.setdefault(_member_type_id(member, translation), None)
    ids = list(seen)
    if len(ids) == 1:
        return TypeExpr.concrete(ids[0])
    return TypeExpr.union(*ids)


def _template_expr(port: object, translation: CompatTranslation) -> TypeExpr:
    """A MatchType input/output -> a Dinkster type variable."""
    template = getattr(port, "template", None)
    template_id = str(getattr(template, "template_id", "") or "")
    if template is None or not template_id:
        raise CompatError("MatchType port lacks a template_id")
    allowed_decl = cast("Iterable[object]", getattr(template, "allowed_types", ()) or ())
    allowed: dict[str, None] = {}
    for entry in allowed_decl:
        io_type = str(getattr(entry, "io_type", "") or "")
        if not io_type or io_type == "*":
            # An AnyType member means unconstrained; drop the allow-list.
            return TypeExpr.variable(template_id)
        allowed.setdefault(_member_type_id(io_type, translation), None)
    return TypeExpr.variable(template_id, tuple(allowed))


def _autogrow_family(
    input_obj: object,
    translation: CompatTranslation,
    *,
    input_is_list: bool,
) -> InputFamilySpec:
    """An Autogrow input -> a Dinkster input family. Dinkster members are
    ``family.suffix`` stored ids (never renumbered), so member naming
    changes on purpose - the family's type and bounds carry over."""
    family_id = str(getattr(input_obj, "id", "") or "")
    template = getattr(input_obj, "template", None)
    inner = getattr(template, "input", None)
    if template is None or inner is None:
        raise CompatError(f"autogrow input {family_id!r} lacks a template input")
    if _is_structural_marker(_io_type_of(inner)) and _io_type_of(inner) != V3_MATCHTYPE_IO_TYPE:
        raise CompatError(f"autogrow input {family_id!r} has unsupported nested dynamic template")
    member = _ordinary_v3_input(inner, translation, input_is_list=input_is_list)
    min_members = int(getattr(template, "min", 0) or 0)
    max_decl = getattr(template, "max", None)
    names = cast("Sequence[object] | None", getattr(template, "names", None))
    prefix = getattr(template, "prefix", None)
    member_prefix = str(prefix) if prefix is not None else None
    member_names = (
        tuple(str(name) for name in names) if member_prefix is None and names is not None else None
    )
    max_members: int | None = None
    if isinstance(max_decl, int):
        max_members = max_decl
    if member_names is not None:
        max_members = None
    return InputFamilySpec(
        family_id,
        (member,),
        min_members=min_members,
        max_members=max_members,
        doc=str(getattr(inner, "tooltip", "") or ""),
        member_prefix=member_prefix,
        member_names=member_names,
    )


def _ordinary_v3_input(
    input_obj: object,
    translation: CompatTranslation,
    *,
    input_is_list: bool,
    allowed_lazy: frozenset[str] = frozenset(),
) -> InputSpec:
    input_id = str(getattr(input_obj, "id", "") or "")
    if not input_id:
        raise CompatError("V3 input lacks an id")
    if getattr(input_obj, "lazy", None) and input_id not in allowed_lazy:
        raise CompatError(f"V3 input {input_id!r} uses unsupported lazy semantics")
    if getattr(input_obj, "rawLink", None):
        raise CompatError(f"V3 input {input_id!r} uses unsupported rawLink semantics")
    io_type = _io_type_of(input_obj)
    if io_type == V3_MATCHTYPE_IO_TYPE:
        type_expr = _template_expr(input_obj, translation)
    elif _is_structural_marker(io_type):
        raise CompatError(f"V3 dynamic input kind {io_type} is not portable yet")
    else:
        type_expr = translate_v3_type(io_type, translation)
    raw_multiselect = getattr(input_obj, "multiselect", None)
    if (
        io_type == _COMBO_IO_TYPE
        and raw_multiselect is not None
        and type(raw_multiselect) is not bool
    ):
        raise CompatError(f"V3 combo {input_id!r} multiselect must be a bool")
    if (
        io_type == _COMBO_IO_TYPE
        and hasattr(input_obj, "multi_select")
        and raw_multiselect is not True
    ):
        raise CompatError(f"V3 combo {input_id!r} multi_select requires multiselect true")
    is_multicombo = io_type == _COMBO_IO_TYPE and raw_multiselect is True
    if is_multicombo and input_is_list:
        raise CompatError(f"V3 MultiCombo {input_id!r} conflicts with is_input_list")
    upload = getattr(input_obj, "upload", None)
    image_folder = getattr(input_obj, "image_folder", None)
    source_filename: SourceFilenameSpec | None = None
    if upload is not None:
        if io_type != _COMBO_IO_TYPE:
            raise CompatError(f"V3 input {input_id!r} source upload requires an exact COMBO input")
        source_kinds = {
            "image_upload": "media/image",
            "audio_upload": "media/audio",
            "video_upload": "media/video",
        }
        upload_value = _exact_enum_value(
            upload,
            class_name="UploadType",
            members={
                "image": "image_upload",
                "audio": "audio_upload",
                "video": "video_upload",
                "model": "file_upload",
            },
            field=f"V3 input {input_id!r} upload",
        )
        if upload_value == "file_upload":
            raise CompatError(f"V3 input {input_id!r} model/file_upload is not supported")
        if type(upload_value) is not str or upload_value not in source_kinds:
            raise CompatError(f"V3 input {input_id!r} upload type is not supported")
        if input_is_list:
            raise CompatError(f"V3 input {input_id!r} source upload conflicts with is_input_list")
        if is_multicombo:
            raise CompatError(f"V3 input {input_id!r} source upload conflicts with multiselect")
        if getattr(input_obj, "default", None) is not None:
            raise CompatError(
                f"V3 input {input_id!r} source upload cannot declare an ambient default"
            )
        category: object = "input"
        if image_folder is not None:
            category = _exact_enum_value(
                image_folder,
                class_name="FolderType",
                members={"input": "input", "output": "output", "temp": "temp"},
                field=f"V3 input {input_id!r} image_folder",
            )
        source_filename = SourceFilenameSpec(
            cast("Any", source_kinds[upload_value]),
            cast("Any", category),
        )
        type_expr = TypeExpr.concrete("dinkster.asset")
        translation.require_asset_type()
    elif image_folder is not None:
        raise CompatError(f"V3 input {input_id!r} image_folder has no UploadType")
    default: object = (
        getattr(input_obj, "default", None) if is_multicombo else _input_default(input_obj, io_type)
    )
    optional = bool(getattr(input_obj, "optional", False))
    widget: Widget | None = None
    if source_filename is not None:
        widget = source_asset_widget(source_filename)
        default = None
    elif not input_is_list:
        if is_multicombo:
            options = getattr(input_obj, "options", None)
            if not isinstance(options, list) or any(
                type(option) is not str or not option for option in cast("list[object]", options)
            ):
                raise CompatError(f"V3 MultiCombo {input_id!r} options must be non-empty strings")
            if default is not None and (
                type(default) is not list
                or any(type(value) is not str for value in cast("list[object]", default))
            ):
                raise CompatError(f"V3 MultiCombo {input_id!r} default must be an array of strings")
            if getattr(input_obj, "control_after_generate", None) is not None:
                raise CompatError(
                    f"V3 MultiCombo {input_id!r} control_after_generate is not supported"
                )
            placeholder = getattr(input_obj, "placeholder", None)
            chip = getattr(input_obj, "chip", None)
            if placeholder is not None and type(placeholder) is not str:
                raise CompatError(f"V3 MultiCombo {input_id!r} placeholder must be a string")
            if chip is not None and type(chip) is not bool:
                raise CompatError(f"V3 MultiCombo {input_id!r} chip must be a bool")
            type_expr = TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))
            option_values = tuple(cast("list[str]", options))
            remote = getattr(input_obj, "remote", None)
            if not option_values and remote is None:
                raise CompatError(
                    f"V3 MultiCombo {input_id!r} requires static options or a remote source"
                )
            if default is not None:
                default = list(cast("list[str]", default))
            widget = MultiComboWidget(
                options=option_values or (("remote",) if remote is not None else ()),
                placeholder=placeholder,
                chip=chip,
            )
            if remote is not None:
                widget = trusted_v3_remote_multicombo(
                    widget,
                    remote,
                    set(translation.listing_snapshots),
                )
                if not option_values:
                    widget = MultiComboWidget(
                        remote_route=widget.remote_route,
                        refresh_button=widget.refresh_button,
                        control_after_refresh=widget.control_after_refresh,
                        remote_timeout_ms=widget.remote_timeout_ms,
                        remote_max_retries=widget.remote_max_retries,
                        remote_refresh_ms=widget.remote_refresh_ms,
                        placeholder=widget.placeholder,
                        chip=widget.chip,
                    )
        elif io_type == _COMBO_IO_TYPE:
            options = getattr(input_obj, "options", None)
            pair = boolean_combo(options)
            if pair is not None:
                on_string, off_string = pair
                type_expr = TypeExpr.concrete(PRIMITIVES["BOOLEAN"])
                widget = BooleanWidget(label_on=on_string, label_off=off_string)
                if isinstance(default, str):
                    default = default == on_string
            else:
                declared_control = getattr(input_obj, "control_after_generate", None)
                widget = combo_widget(
                    options,
                    {
                        "control_after_generate": getattr(
                            declared_control, "value", declared_control
                        )
                    },
                )
                remote = getattr(input_obj, "remote", None)
                if remote is not None:
                    if widget is None:
                        raise CompatError(f"V3 remote combo {input_id!r} is not representable")
                    widget = trusted_v3_remote_combo(
                        widget,
                        remote,
                        set(translation.listing_snapshots),
                    )
        elif io_type == "BOOLEAN":
            label_on = getattr(input_obj, "label_on", None)
            label_off = getattr(input_obj, "label_off", None)
            on_text = label_on if isinstance(label_on, str) else ""
            off_text = label_off if isinstance(label_off, str) else ""
            if on_text or off_text:
                widget = BooleanWidget(label_on=on_text, label_off=off_text)
        elif io_type in {"INT", "FLOAT"}:
            numeric_config = {
                field: getattr(input_obj, field, None)
                for field in ("min", "max", "step", "round", "control_after_generate")
            }
            display_mode = getattr(input_obj, "display_mode", None)
            numeric_config["display"] = getattr(display_mode, "value", None)
            widget = number_widget(input_id, io_type, numeric_config)
        elif io_type in {"STRING", "COLOR"}:
            widget = string_widget(
                io_type,
                {
                    "multiline": getattr(input_obj, "multiline", None),
                    "placeholder": getattr(input_obj, "placeholder", None),
                    "dynamicPrompts": getattr(input_obj, "dynamic_prompts", None),
                },
            )
    if input_is_list:
        type_expr = TypeExpr.list_of(type_expr)
        if default is not None:
            default = [cast("object", default)]
    return InputSpec(
        id=input_id,
        type=type_expr,
        required=not optional and default is None,
        default=cast("object", default),
        doc=str(getattr(input_obj, "tooltip", "") or ""),
        widget=widget,
        display_name=str(getattr(input_obj, "display_name", "") or ""),
        force_input=bool(getattr(input_obj, "force_input", False)),
        advanced=bool(getattr(input_obj, "advanced", False)),
        lazy=input_id in allowed_lazy,
        source_filename=source_filename,
    )


def _translate_v3_entry(
    input_obj: object,
    translation: CompatTranslation,
    *,
    input_is_list: bool,
    allowed_lazy: frozenset[str] = frozenset(),
) -> DynamicEntry:
    io_type = _io_type_of(input_obj)
    input_id = str(getattr(input_obj, "id", "") or "")
    if getattr(input_obj, "lazy", None) and input_id not in allowed_lazy:
        raise CompatError(f"V3 input {input_id!r} uses unsupported lazy semantics")
    if getattr(input_obj, "rawLink", None):
        raise CompatError(f"V3 input {input_id!r} uses unsupported rawLink semantics")
    if io_type == V3_AUTOGROW_IO_TYPE:
        return _autogrow_family(input_obj, translation, input_is_list=input_is_list)
    if io_type == V3_DYNAMICCOMBO_IO_TYPE:
        options: list[DynamicComboOption] = []
        for option_obj in cast("Sequence[object]", getattr(input_obj, "options", ()) or ()):
            key = str(getattr(option_obj, "key", "") or "")
            option_inputs = tuple(
                _translate_v3_entry(
                    child,
                    translation,
                    input_is_list=input_is_list,
                )
                for child in cast("Sequence[object]", getattr(option_obj, "inputs", ()) or ())
            )
            options.append(DynamicComboOption(key, option_inputs))
        required = not bool(getattr(input_obj, "optional", False))
        if required and not options:
            raise CompatError(
                f"V3 DynamicCombo input {input_id!r} is required but has zero options"
            )
        default = getattr(input_obj, "default", None)
        return DynamicComboSpec(
            input_id,
            tuple(options),
            default=str(default) if default is not None else None,
            required=required,
            doc=str(getattr(input_obj, "tooltip", "") or ""),
            display_name=str(getattr(input_obj, "display_name", "") or ""),
        )
    if io_type == V3_DYNAMICSLOT_IO_TYPE:
        slot = getattr(input_obj, "slot", None)
        if slot is None:
            raise CompatError(f"V3 DynamicSlot input {input_id!r} lacks a slot")
        slot_io_type = _io_type_of(slot)
        if _is_structural_marker(slot_io_type):
            raise CompatError(f"V3 DynamicSlot input {input_id!r} has dynamic slot type")
        slot_type = translate_v3_type(slot_io_type, translation)
        if input_is_list:
            slot_type = TypeExpr.list_of(slot_type)
        dependents = tuple(
            _translate_v3_entry(
                child,
                translation,
                input_is_list=input_is_list,
            )
            for child in cast("Sequence[object]", getattr(input_obj, "inputs", ()) or ())
        )
        return DynamicSlotSpec(
            input_id,
            slot_type=slot_type,
            required=False,
            inputs=dependents,
            force_input=bool(getattr(input_obj, "force_input", False)),
            doc=str(getattr(input_obj, "tooltip", "") or ""),
            display_name=str(getattr(input_obj, "display_name", "") or ""),
        )
    return _ordinary_v3_input(
        input_obj, translation, input_is_list=input_is_list, allowed_lazy=allowed_lazy
    )


def _input_default(input_obj: object, io_type: str) -> object:
    default = getattr(input_obj, "default", None)
    if default is not None:
        return default
    if io_type == _COMBO_IO_TYPE:
        options = cast("Sequence[object] | None", getattr(input_obj, "options", None))
        if options and isinstance(options[0], str):
            return options[0]
    return None


def _output_ids(outputs: Sequence[object]) -> list[str]:
    """Stable output ids: the declared id, else display_name, else the
    io_type lowercased; duplicates get positional suffixes (v1 rule)."""
    ids: list[str] = []
    seen: dict[str, int] = {}
    for output in outputs:
        declared = getattr(output, "id", None)
        display = getattr(output, "display_name", None)
        if isinstance(declared, str) and declared:
            base = declared
        elif isinstance(display, str) and display:
            base = display
        else:
            base = _io_type_of(output).lower() or "out"
        count = seen.get(base, 0)
        seen[base] = count + 1
        ids.append(base if count == 0 else f"{base}_{count + 1}")
    return ids


def translate_v3_schema(
    schema: object,
    translation: CompatTranslation,
    *,
    namespace: str = "",
) -> NodeSchema:
    """One V3 Schema object -> a Dinkster NodeSchema (no runtime class).

    ``namespace`` scopes the node type exactly like the v1 translator:
    ``comfy.<pack>.<node_id>``. Reads only duck-typed attributes so fakes
    (tests) and real comfy_api objects (the probe subprocess) both work."""
    raw_node_id = getattr(schema, "node_id", "")
    node_id = str(raw_node_id or "")
    if not node_id:
        raise CompatError("V3 schema lacks a node_id")
    raw_input_is_list = getattr(schema, "is_input_list", False)
    input_is_list = bool(raw_input_is_list)
    inputs_decl = cast("Sequence[object]", getattr(schema, "inputs", ()) or ())
    outputs_decl = cast("Sequence[object]", getattr(schema, "outputs", ()) or ())
    output_display_names = tuple(getattr(output, "display_name", None) for output in outputs_decl)
    raw_output_ids = tuple(getattr(output, "id", None) for output in outputs_decl)
    custom_combo = (
        not namespace
        and type(raw_node_id) is str
        and node_id == CUSTOM_COMBO_NODE_ID
        and getattr(schema, "accept_all_inputs", False) is True
        and raw_input_is_list is False
        and len(inputs_decl) == 1
        and type(getattr(inputs_decl[0], "id", None)) is str
        and getattr(inputs_decl[0], "id", None) == "choice"
        and _io_type_of(inputs_decl[0]) == _COMBO_IO_TYPE
        and type(getattr(inputs_decl[0], "options", None)) is list
        and len(cast("list[object]", getattr(inputs_decl[0], "options", None))) == 0
        and getattr(inputs_decl[0], "multiselect", None) is False
        and getattr(inputs_decl[0], "default", None) is None
        and getattr(inputs_decl[0], "optional", None) is False
        and getattr(inputs_decl[0], "lazy", False) is None
        and getattr(inputs_decl[0], "rawLink", False) is None
        and len(outputs_decl) == 2
        and tuple(_io_type_of(output) for output in outputs_decl) == ("STRING", "INT")
        and all(type(name) is str for name in output_display_names)
        and output_display_names == ("STRING", "INDEX")
        and (
            all(output_id is None for output_id in raw_output_ids)
            or (
                all(type(output_id) is str for output_id in raw_output_ids)
                and raw_output_ids == ("_0_STRING_", "_1_INT_")
            )
        )
        and all(getattr(output, "is_output_list", None) is False for output in outputs_decl)
    )
    if bool(getattr(schema, "accept_all_inputs", False)) and not custom_combo:
        raise CompatError(f"{node_id}: V3 accept_all_inputs is not portable")

    pending = CompatTranslation()
    pending.listing_snapshots.update(translation.listing_snapshots)
    selector_lazy = selector_lazy_inputs(
        node_id, schema, inputs_decl, outputs_decl, input_is_list=input_is_list
    )
    allowed_lazy = selector_lazy
    if input_is_list:
        allowed_lazy |= frozenset(
            str(getattr(input_obj, "id", "") or "")
            for input_obj in inputs_decl
            if getattr(input_obj, "lazy", None)
            and (
                _io_type_of(input_obj) == V3_MATCHTYPE_IO_TYPE
                or not _is_structural_marker(_io_type_of(input_obj))
            )
        )
    specs: list[InputSpec] = []
    families: list[InputFamilySpec] = []
    combos: list[DynamicComboSpec] = []
    slots: list[DynamicSlotSpec] = []
    occupies_gpu = False
    for input_obj in inputs_decl:
        io_type = _io_type_of(input_obj)
        entry = _translate_v3_entry(
            input_obj, pending, input_is_list=input_is_list, allowed_lazy=allowed_lazy
        )
        if isinstance(entry, InputSpec):
            specs.append(entry)
        elif isinstance(entry, InputFamilySpec):
            families.append(entry)
        elif isinstance(entry, DynamicComboSpec):
            combos.append(entry)
        else:
            slots.append(entry)
        if dynamic_entries_occupy_gpu((entry,)):
            occupies_gpu = True

    if custom_combo:
        specs.append(
            InputSpec(
                id="index",
                type=TypeExpr.concrete(CORE_INT),
                required=False,
                default=0,
            )
        )
        families.append(
            InputFamilySpec(
                CUSTOM_COMBO_FAMILY_ID,
                TypeExpr.concrete(CORE_STRING),
                member_names=CUSTOM_COMBO_OPTION_NAMES,
            )
        )

    output_ids = ["STRING", "INDEX"] if custom_combo else _output_ids(outputs_decl)
    outputs: list[OutputSpec] = []
    for output_id, output_obj in zip(output_ids, outputs_decl, strict=True):
        io_type = _io_type_of(output_obj)
        if io_type == V3_MATCHTYPE_IO_TYPE:
            type_expr = _template_expr(output_obj, pending)
        elif _is_structural_marker(io_type):
            raise CompatError(f"{node_id}: V3 dynamic output kind {io_type} is not portable yet")
        else:
            type_expr = translate_v3_type(io_type, pending)
        if bool(getattr(output_obj, "is_output_list", False)):
            type_expr = TypeExpr.list_of(type_expr)
        outputs.append(
            OutputSpec(
                id=output_id,
                type=type_expr,
                doc=str(getattr(output_obj, "tooltip", "") or ""),
            )
        )

    is_output_node = bool(getattr(schema, "is_output_node", False))
    not_idempotent = bool(getattr(schema, "not_idempotent", False))
    io_bound = bool(getattr(schema, "is_api_node", False))
    deprecation: Deprecation | None = None
    if bool(getattr(schema, "is_deprecated", False)):
        # V3 deprecation is a bare flag; Dinkster deprecation is authored
        # prose plus an optional successor. The port carries the fact and
        # leaves the message for the human (TODO(port) in the skeleton).
        deprecation = Deprecation(
            message="Deprecated in the source ComfyUI pack (V3 is_deprecated)."
        )

    result = NodeSchema(
        node_type=comfy_type_id(f"{namespace}.{node_id}" if namespace else node_id),
        display_name=str(getattr(schema, "display_name", "") or "") or node_id,
        category="comfy/" + (str(getattr(schema, "category", "") or "") or "uncategorized"),
        description=str(getattr(schema, "description", "") or ""),
        inputs=tuple(specs),
        outputs=tuple(outputs),
        input_families=tuple(families),
        combos=tuple(combos),
        slots=tuple(slots),
        idempotent=not (is_output_node or not_idempotent),
        occupies=("gpu",) if occupies_gpu and not io_bound else (),
        io_bound=io_bound,
        deprecation=deprecation,
        search_visibility="hidden" if bool(getattr(schema, "is_dev_only", False)) else "normal",
        aliases=(node_id,),
        # ComfyUI itself refuses NodeOutput.expand unless the source schema
        # declares enable_expand, so this is the exact V3 classification; the
        # loud runtime refusal (_unwrap_node_output) stays for anything that
        # slips through.
        may_expand_graph=bool(getattr(schema, "enable_expand", False)),
        output_node=is_output_node,
        selector=(
            SelectorSpec("switch", {"false": "on_false", "true": "on_true"})
            if selector_lazy
            else None
        ),
    )
    translation.opaque_types.update(pending.opaque_types)
    return result


__all__ = [
    "V3_AUTOGROW_IO_TYPE",
    "V3_MATCHTYPE_IO_TYPE",
    "translate_v3_schema",
    "translate_v3_type",
]
