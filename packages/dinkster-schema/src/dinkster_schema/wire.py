"""Wire encoding of node schemas: the /object_info successor.

Shape is designed to be near-isomorphic to Dinkster-Frontend's normalized
NodeSchema model: one ordered interface list, real output ids, structured
type expressions, and an explicit schemaVersion field.

"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Final, Literal, cast

from dinkster_values import CustomWidgetDescriptor, JsonValue

from . import body
from .formula import FORMULA_LANGUAGE_VERSION
from .model import (
    CONTROL_AFTER_GENERATE,
    AbsentPolicy,
    AssetWidget,
    BooleanWidget,
    ColorWidget,
    ComboOption,
    ComboWidget,
    CompositorWidget,
    ConditionalWidgetCondition,
    ConditionalWidgetGroup,
    ControlAfterGenerate,
    CurveWidget,
    Deprecation,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicEntry,
    DynamicSlotSpec,
    Input,
    InputFamilyOptionSource,
    InputFamilySpec,
    MirrorSpec,
    MirrorTolerance,
    MountWidget,
    MultiComboWidget,
    NodeSchema,
    NumberWidget,
    Output,
    OutputCountSpec,
    OutputDescriptorsSpec,
    OutputFamilySpec,
    OutputKnownValue,
    OutputProbeSpec,
    OutputRepresents,
    SaveTargetWidget,
    SearchVisibility,
    SelectorSpec,
    SlotVariant,
    SourceFilenameSpec,
    StringWidget,
    TextCompletionItem,
    TextCompletions,
    TypeExpr,
    Widget,
    WidgetDescriptor,
    WidgetRepresentation,
    WidgetRepresentations,
)
from .replace import rule_from_wire, rule_to_wire

SCHEMA_WIRE_VERSION = 2
"""Version 2 carries the node body tree and speaks formula language version 1."""

if FORMULA_LANGUAGE_VERSION != 1:
    raise ImportError("a new formula language version needs a new SCHEMA_WIRE_VERSION")
_JSON_SAFE_INT = 2**53 - 1
_DECIMAL_WIRE_INT_MIN = -(2**63)
_DECIMAL_WIRE_INT_MAX = 2**64 - 1
_CANONICAL_DECIMAL_INT = re.compile("^-?(?:0|[1-9][0-9]*)$")


def _number_constraint_to_wire(value: int | float, wire_version: int) -> int | float | str | None:
    if isinstance(value, int) and abs(value) > _JSON_SAFE_INT:
        return str(value)
    return value


def _number_constraint_from_wire(value: object, field: str, wire_version: int) -> int | float:
    if isinstance(value, str):
        if _CANONICAL_DECIMAL_INT.fullmatch(value) is None:
            raise ValueError(f"NUMBER widget {field} must be a number")
        decoded = int(value)
        if abs(decoded) <= _JSON_SAFE_INT:
            raise ValueError(
                f"NUMBER widget {field} decimal string must encode an integer "
                "outside the JSON-double-safe range"
            )
        if not _DECIMAL_WIRE_INT_MIN <= decoded <= _DECIMAL_WIRE_INT_MAX:
            raise ValueError(f"NUMBER widget {field} decimal string is outside the supported range")
        return decoded
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"NUMBER widget {field} must be a number")
    if isinstance(value, int) and abs(value) > _JSON_SAFE_INT:
        raise ValueError(
            f"NUMBER widget {field} integer outside the JSON-double-safe range "
            "must use a decimal string"
        )
    return value


def _combo_option_to_wire(
    option: str | bool | int | float | ComboOption, wire_version: int
) -> object:
    if not isinstance(option, ComboOption):
        return option
    wire: dict[str, object] = {"value": option.value}
    for field_name in ("label", "info", "folder"):
        field_value = getattr(option, field_name)
        if field_value is not None:
            wire[field_name] = field_value
    return wire


def _widget_descriptor_to_wire(
    widget: WidgetDescriptor, wire_version: int
) -> dict[str, object] | None:
    if isinstance(widget, CustomWidgetDescriptor):
        return {
            "type": widget.widget_type,
            **{key: _json_value_to_wire(value) for key, value in widget.params.items()},
        }
    if isinstance(widget, AssetWidget):
        asset_wire: dict[str, object] = {"type": "ASSET", "accept": list(widget.accept)}
        if widget.kind:
            asset_wire["kind"] = widget.kind
        if widget.allow_upload:
            asset_wire["allowUpload"] = True
        if widget.selection:
            asset_wire["selection"] = True
        return asset_wire
    if isinstance(widget, ComboWidget):
        combo_wire: dict[str, object] = {"type": "COMBO"}
        if widget.editable_options:
            combo_wire["editableOptions"] = True
        if widget.options:
            combo_wire["options"] = [
                _combo_option_to_wire(option, wire_version) for option in widget.options
            ]
        if widget.option_source is not None:
            combo_wire["optionSource"] = {"inputFamily": widget.option_source.input_family}
        if widget.remote_route:
            remote: dict[str, object] = {"route": widget.remote_route}
            if widget.refresh_button:
                remote["refreshButton"] = True
            if widget.control_after_refresh is not None:
                remote["controlAfterRefresh"] = widget.control_after_refresh
            if widget.remote_timeout_ms is not None:
                remote["timeoutMs"] = widget.remote_timeout_ms
            if widget.remote_max_retries is not None:
                remote["maxRetries"] = widget.remote_max_retries
            if widget.remote_refresh_ms is not None:
                remote["refreshMs"] = widget.remote_refresh_ms
            combo_wire["remote"] = remote
        if widget.control_after_generate is not None:
            combo_wire["controlAfterGenerate"] = widget.control_after_generate
        return combo_wire
    if isinstance(widget, MultiComboWidget):
        multi_wire: dict[str, object] = {"type": "MULTI_COMBO"}
        if widget.options:
            multi_wire["options"] = [
                _combo_option_to_wire(option, wire_version) for option in widget.options
            ]
        if widget.remote_route:
            remote = {"route": widget.remote_route}
            if widget.refresh_button:
                remote["refreshButton"] = True
            if widget.control_after_refresh is not None:
                remote["controlAfterRefresh"] = widget.control_after_refresh
            if widget.remote_timeout_ms is not None:
                remote["timeoutMs"] = widget.remote_timeout_ms
            if widget.remote_max_retries is not None:
                remote["maxRetries"] = widget.remote_max_retries
            if widget.remote_refresh_ms is not None:
                remote["refreshMs"] = widget.remote_refresh_ms
            multi_wire["remote"] = remote
        if widget.placeholder is not None:
            multi_wire["placeholder"] = widget.placeholder
        if widget.chip is not None:
            multi_wire["chip"] = widget.chip
        return multi_wire
    if isinstance(widget, BooleanWidget):
        boolean_wire: dict[str, object] = {"type": "BOOLEAN"}
        if widget.label_on:
            boolean_wire["labelOn"] = widget.label_on
        if widget.label_off:
            boolean_wire["labelOff"] = widget.label_off
        return boolean_wire
    if isinstance(widget, NumberWidget):
        number_wire: dict[str, object] = {"type": "NUMBER"}
        for field in ("min", "max", "step"):
            constraint = getattr(widget, field)
            if constraint is None:
                continue
            encoded = _number_constraint_to_wire(constraint, wire_version)
            if encoded is not None:
                number_wire[field] = encoded
        if widget.round is not None:
            number_wire["round"] = widget.round
        if widget.control_after_generate is not None:
            number_wire["controlAfterGenerate"] = widget.control_after_generate
        if widget.display is not None:
            number_wire["display"] = widget.display
        if len(number_wire) == 1:
            return None
        return number_wire
    if isinstance(widget, StringWidget):
        string_wire: dict[str, object] = {"type": "STRING"}
        if widget.multiline is not None:
            string_wire["multiline"] = widget.multiline
        if widget.placeholder is not None:
            string_wire["placeholder"] = widget.placeholder
        if widget.dynamic_prompts is not None:
            string_wire["dynamicPrompts"] = widget.dynamic_prompts
        if widget.completions is not None:
            completions: dict[str, object] = {}
            if widget.completions.items:
                completions["items"] = [
                    {
                        "value": item.value,
                        **({"label": item.label} if item.label else {}),
                        **({"insertText": item.insert_text} if item.insert_text else {}),
                        **({"detail": item.detail} if item.detail else {}),
                        **({"kind": item.kind} if item.kind != "identifier" else {}),
                    }
                    for item in widget.completions.items
                ]
            if widget.completions.input_families:
                completions["inputFamilies"] = list(widget.completions.input_families)
            string_wire["completions"] = completions
        if len(string_wire) == 1:
            return None
        return string_wire
    if isinstance(widget, ColorWidget):
        return {"type": "COLOR"}
    if isinstance(widget, CurveWidget):
        return {"type": "CURVE"}
    if isinstance(widget, CompositorWidget):
        return {"type": "COMPOSITOR"}
    if isinstance(widget, MountWidget):
        return {"type": "MOUNT"}
    remaining = cast("object", widget)
    if isinstance(remaining, SaveTargetWidget):
        wire: dict[str, object] = {"type": "SAVE_TARGET"}
        if remaining.suffix:
            wire["suffix"] = remaining.suffix
        return wire
    raise TypeError(f"unsupported widget descriptor {widget!r}")


def _json_value_to_wire(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value_to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value_to_wire(item) for item in value]
    return value


def _widget_to_wire(widget: Widget, wire_version: int) -> dict[str, object] | None:
    if isinstance(widget, WidgetRepresentations):
        representations: list[dict[str, object]] = []
        for representation in widget.representations:
            representation_wire = _widget_descriptor_to_wire(representation.widget, wire_version)
            if representation_wire is None:
                continue
            entry: dict[str, object] = {"id": representation.id, "widget": representation_wire}
            if representation.display_name:
                entry["displayName"] = representation.display_name
            representations.append(entry)
        if not representations:
            return None
        surviving_ids = {cast("str", entry["id"]) for entry in representations}
        return {
            "type": "REPRESENTATIONS",
            "default": widget.default
            if widget.default in surviving_ids
            else cast("str", representations[0]["id"]),
            "userSwitchable": widget.user_switchable,
            "representations": representations,
        }
    return _widget_descriptor_to_wire(widget, wire_version)


def _reject_unknown_fields(data: dict[str, object], allowed: frozenset[str], subject: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{subject} has unknown fields: {unknown}")


def _combo_option_from_wire(
    value: object, wire_version: int, *, subject: str
) -> str | bool | int | float | ComboOption:
    if not isinstance(value, dict):
        scalar = cast("str | bool | int | float", value)
        ComboOption(scalar)
        return scalar
    option = cast("dict[str, object]", value)
    _reject_unknown_fields(option, frozenset({"value", "label", "info", "folder"}), subject)
    raw_value = cast("str | bool | int | float", option.get("value"))
    presentation: dict[str, str] = {}
    for field_name in ("label", "info", "folder"):
        if field_name not in option:
            continue
        field_value = option[field_name]
        if type(field_value) is not str or not field_value:
            raise ValueError(f"{subject}.{field_name} must be a non-empty string")
        presentation[field_name] = field_value
    return ComboOption(value=raw_value, **presentation)


def _widget_descriptor_from_wire(
    widget_data: dict[str, object], wire_version: int, *, strict: bool
) -> WidgetDescriptor:
    kind = widget_data.get("type")
    if kind == "ASSET":
        if strict:
            allowed = {"type", "accept", "kind"}
            allowed.add("allowUpload")
            allowed.add("selection")
            _reject_unknown_fields(widget_data, frozenset(allowed), "ASSET widget")
        accept = widget_data.get("accept", ())
        if not isinstance(accept, list):
            raise ValueError("ASSET widget accept must be a list of strings")
        accept_values = cast("list[object]", accept)
        if not all(isinstance(value, str) for value in accept_values):
            raise ValueError("ASSET widget accept must be a list of strings")
        asset_kind = widget_data.get("kind", "")
        if not isinstance(asset_kind, str):
            raise ValueError("ASSET widget kind must be a string")
        allow_upload = widget_data.get("allowUpload", False)
        if type(allow_upload) is not bool:
            raise ValueError("ASSET widget allowUpload must be a boolean")
        return AssetWidget(
            accept=tuple(cast("list[str]", accept_values)),
            kind=asset_kind,
            allow_upload=allow_upload,
            selection=cast("bool", widget_data.get("selection", False)),
        )
    if kind == "MOUNT":
        _reject_unknown_fields(widget_data, frozenset({"type"}), "MOUNT widget")
        return MountWidget()
    if kind == "SAVE_TARGET":
        if strict:
            _reject_unknown_fields(widget_data, frozenset({"type", "suffix"}), "SAVE_TARGET widget")
        suffix = widget_data.get("suffix", "")
        if not isinstance(suffix, str):
            raise ValueError("SAVE_TARGET widget suffix must be a string")
        return SaveTargetWidget(suffix=suffix)
    if kind == "COMBO":
        if strict:
            allowed = {"type", "options", "remote"}
            allowed.add("controlAfterGenerate")
            allowed.add("optionSource")
            allowed.add("editableOptions")
            _reject_unknown_fields(widget_data, frozenset(allowed), "COMBO widget")
        options = widget_data.get("options", [])
        if not isinstance(options, list):
            raise ValueError("COMBO widget options must be a list")
        option_values = cast("list[object]", options)
        decoded_options = tuple(
            (
                _combo_option_from_wire(
                    value, wire_version, subject=f"COMBO widget options[{index}]"
                )
                for index, value in enumerate(option_values)
            )
        )
        option_source = None
        raw_option_source = widget_data.get("optionSource")
        if raw_option_source is not None:
            if not isinstance(raw_option_source, dict):
                raise ValueError("COMBO widget optionSource must be an object")
            source_data = cast("dict[str, object]", raw_option_source)
            if strict:
                _reject_unknown_fields(
                    source_data, frozenset({"inputFamily"}), "COMBO widget optionSource"
                )
            input_family = source_data.get("inputFamily")
            if not isinstance(input_family, str):
                raise ValueError("COMBO widget optionSource inputFamily must be a string")
            option_source = InputFamilyOptionSource(input_family)
        remote_route = ""
        refresh_button = False
        control_after_refresh: object = None
        remote_timeout_ms: object = None
        remote_max_retries: object = None
        remote_refresh_ms: object = None
        remote = widget_data.get("remote")
        if remote is not None:
            if not isinstance(remote, dict):
                raise ValueError("COMBO widget remote must be an object")
            remote_data = cast("dict[str, object]", remote)
            if strict:
                allowed_remote = {"route", "refreshButton"}
                allowed_remote.update(
                    ("controlAfterRefresh", "timeoutMs", "maxRetries", "refreshMs")
                )
                _reject_unknown_fields(
                    remote_data, frozenset(allowed_remote), "COMBO widget remote"
                )
            route = remote_data.get("route")
            if not isinstance(route, str) or not route:
                raise ValueError("COMBO widget remote route must be a string")
            remote_route = route
            refresh = remote_data.get("refreshButton", False)
            if not isinstance(refresh, bool):
                raise ValueError("COMBO widget refreshButton must be a boolean")
            refresh_button = refresh
            control_after_refresh = remote_data.get("controlAfterRefresh")
            if "controlAfterRefresh" in remote_data and (
                type(control_after_refresh) is not str
                or control_after_refresh not in {"first", "last"}
            ):
                raise ValueError("COMBO widget controlAfterRefresh must be 'first' or 'last'")
            for field, lower, upper in (
                ("timeoutMs", 1, 60000),
                ("maxRetries", 0, 5),
                ("refreshMs", 0, 86400000),
            ):
                value = remote_data.get(field)
                if field in remote_data and (type(value) is not int or not lower <= value <= upper):
                    raise ValueError(f"COMBO widget {field} must be an integer in {lower}..{upper}")
            remote_timeout_ms = remote_data.get("timeoutMs")
            remote_max_retries = remote_data.get("maxRetries")
            remote_refresh_ms = remote_data.get("refreshMs")
        control = widget_data.get("controlAfterGenerate")
        if "controlAfterGenerate" in widget_data and (
            type(control) is not str or control not in CONTROL_AFTER_GENERATE
        ):
            raise ValueError(
                f"COMBO widget controlAfterGenerate must be one of {sorted(CONTROL_AFTER_GENERATE)}"
            )
        return ComboWidget(
            options=decoded_options,
            option_source=option_source,
            editable_options=_expect_bool(
                widget_data.get("editableOptions", False), "COMBO widget editableOptions"
            ),
            remote_route=remote_route,
            refresh_button=refresh_button,
            control_after_generate=cast("ControlAfterGenerate | None", control),
            control_after_refresh=cast("Literal['first', 'last'] | None", control_after_refresh),
            remote_timeout_ms=cast("int | None", remote_timeout_ms),
            remote_max_retries=cast("int | None", remote_max_retries),
            remote_refresh_ms=cast("int | None", remote_refresh_ms),
        )
    if kind == "MULTI_COMBO":
        if strict:
            _reject_unknown_fields(
                widget_data,
                frozenset({"type", "options", "remote", "placeholder", "chip"}),
                "MULTI_COMBO widget",
            )
        options = widget_data.get("options", [])
        if not isinstance(options, list):
            raise ValueError("MULTI_COMBO widget options must be a list")
        decoded_options = tuple(
            (
                _combo_option_from_wire(
                    value, wire_version, subject=f"MULTI_COMBO widget options[{index}]"
                )
                for index, value in enumerate(cast("list[object]", options))
            )
        )
        remote_route = ""
        refresh_button = False
        control_after_refresh: object = None
        remote_timeout_ms: object = None
        remote_max_retries: object = None
        remote_refresh_ms: object = None
        remote = widget_data.get("remote")
        if remote is not None:
            if not isinstance(remote, dict):
                raise ValueError("MULTI_COMBO widget remote must be an object")
            remote_data = cast("dict[str, object]", remote)
            if strict:
                _reject_unknown_fields(
                    remote_data,
                    frozenset(
                        {
                            "route",
                            "refreshButton",
                            "controlAfterRefresh",
                            "timeoutMs",
                            "maxRetries",
                            "refreshMs",
                        }
                    ),
                    "MULTI_COMBO widget remote",
                )
            route = remote_data.get("route")
            if type(route) is not str or not route:
                raise ValueError("MULTI_COMBO widget remote route must be a string")
            remote_route = route
            refresh = remote_data.get("refreshButton", False)
            if type(refresh) is not bool:
                raise ValueError("MULTI_COMBO widget refreshButton must be a boolean")
            refresh_button = refresh
            control_after_refresh = remote_data.get("controlAfterRefresh")
            if "controlAfterRefresh" in remote_data and (
                type(control_after_refresh) is not str
                or control_after_refresh not in {"first", "last"}
            ):
                raise ValueError("MULTI_COMBO widget controlAfterRefresh must be 'first' or 'last'")
            for field, lower, upper in (
                ("timeoutMs", 1, 60000),
                ("maxRetries", 0, 5),
                ("refreshMs", 0, 86400000),
            ):
                value = remote_data.get(field)
                if field in remote_data and (type(value) is not int or not lower <= value <= upper):
                    raise ValueError(
                        f"MULTI_COMBO widget {field} must be an integer in {lower}..{upper}"
                    )
            remote_timeout_ms = remote_data.get("timeoutMs")
            remote_max_retries = remote_data.get("maxRetries")
            remote_refresh_ms = remote_data.get("refreshMs")
        placeholder = widget_data.get("placeholder")
        if "placeholder" in widget_data and type(placeholder) is not str:
            raise ValueError("MULTI_COMBO widget placeholder must be a string")
        chip = widget_data.get("chip")
        if "chip" in widget_data and type(chip) is not bool:
            raise ValueError("MULTI_COMBO widget chip must be a boolean")
        return MultiComboWidget(
            options=cast("tuple[str | ComboOption, ...]", decoded_options),
            remote_route=remote_route,
            refresh_button=refresh_button,
            control_after_refresh=cast("Literal['first', 'last'] | None", control_after_refresh),
            remote_timeout_ms=cast("int | None", remote_timeout_ms),
            remote_max_retries=cast("int | None", remote_max_retries),
            remote_refresh_ms=cast("int | None", remote_refresh_ms),
            placeholder=cast("str | None", placeholder),
            chip=cast("bool | None", chip),
        )
    if kind == "BOOLEAN":
        if strict:
            _reject_unknown_fields(
                widget_data, frozenset({"type", "labelOn", "labelOff"}), "BOOLEAN widget"
            )
        label_on = widget_data.get("labelOn", "")
        label_off = widget_data.get("labelOff", "")
        if not isinstance(label_on, str) or not isinstance(label_off, str):
            raise ValueError("BOOLEAN widget labels must be strings")
        return BooleanWidget(label_on=label_on, label_off=label_off)
    if kind == "NUMBER":
        if strict:
            allowed = {"type", "min", "max", "step", "controlAfterGenerate"}
            allowed.add("display")
            allowed.add("round")
            _reject_unknown_fields(widget_data, frozenset(allowed), "NUMBER widget")
        bounds: dict[str, int | float | None] = {}
        for field in ("min", "max", "step"):
            raw = widget_data.get(field)
            bounds[field] = (
                None if raw is None else _number_constraint_from_wire(raw, field, wire_version)
            )
        round_value = widget_data.get("round")
        if (
            "round" in widget_data
            and round_value is not False
            and type(round_value) not in (int, float)
        ):
            raise ValueError("NUMBER widget round must be False or a number")
        control = widget_data.get("controlAfterGenerate")
        if control is not None and (
            not isinstance(control, str) or control not in CONTROL_AFTER_GENERATE
        ):
            raise ValueError(
                "NUMBER widget controlAfterGenerate must be one of "
                f"{sorted(CONTROL_AFTER_GENERATE)}"
            )
        display = widget_data.get("display")
        if "display" in widget_data:
            if type(display) is not str:
                raise ValueError("NUMBER widget display must be a string")
        return NumberWidget(
            min=bounds["min"],
            max=bounds["max"],
            step=bounds["step"],
            round=cast("int | float | Literal[False] | None", round_value),
            control_after_generate=cast("ControlAfterGenerate | None", control),
            display=cast("Any", display),
        )
    if kind == "STRING":
        if strict:
            allowed = {"type", "multiline"}
            allowed.update(("placeholder", "dynamicPrompts"))
            allowed.add("completions")
            _reject_unknown_fields(widget_data, frozenset(allowed), "STRING widget")
        multiline = widget_data.get("multiline")
        if "multiline" in widget_data and type(multiline) is not bool:
            raise ValueError("STRING widget multiline must be a boolean")
        placeholder = widget_data.get("placeholder")
        if "placeholder" in widget_data and type(placeholder) is not str:
            raise ValueError("STRING widget placeholder must be a string")
        dynamic_prompts = widget_data.get("dynamicPrompts")
        if "dynamicPrompts" in widget_data and type(dynamic_prompts) is not bool:
            raise ValueError("STRING widget dynamicPrompts must be a boolean")
        completions = None
        if "completions" in widget_data:
            completion_data = widget_data["completions"]
            if not isinstance(completion_data, dict):
                raise ValueError("STRING widget completions must be an object")
            completion_data = cast("dict[str, object]", completion_data)
            _reject_unknown_fields(
                completion_data, frozenset({"items", "inputFamilies"}), "STRING widget completions"
            )
            raw_items = completion_data.get("items", [])
            raw_families = completion_data.get("inputFamilies", [])
            if not isinstance(raw_items, list):
                raise ValueError("STRING widget completion items must be an array")
            if not isinstance(raw_families, list):
                raise ValueError("STRING widget completion inputFamilies must be a string array")
            families = cast("list[object]", raw_families)
            if any(not isinstance(family, str) for family in families):
                raise ValueError("STRING widget completion inputFamilies must be a string array")
            items: list[TextCompletionItem] = []
            for raw_item in cast("list[object]", raw_items):
                if not isinstance(raw_item, dict):
                    raise ValueError("STRING widget completion items must be objects")
                item = cast("dict[str, object]", raw_item)
                _reject_unknown_fields(
                    item,
                    frozenset({"value", "label", "insertText", "detail", "kind"}),
                    "STRING widget completion item",
                )
                value = item.get("value")
                if not isinstance(value, str):
                    raise ValueError("STRING widget completion item value must be a string")
                label = item.get("label", "")
                insert_text = item.get("insertText", "")
                detail = item.get("detail", "")
                if not all(isinstance(field, str) for field in (label, insert_text, detail)):
                    raise ValueError("STRING widget completion item text fields must be strings")
                kind_value = item.get("kind", "identifier")
                if kind_value not in ("identifier", "operator"):
                    raise ValueError("STRING widget completion item kind is invalid")
                items.append(
                    TextCompletionItem(
                        value=value,
                        label=cast("str", label),
                        insert_text=cast("str", insert_text),
                        detail=cast("str", detail),
                        kind=kind_value,
                    )
                )
            completions = TextCompletions(
                items=tuple(items), input_families=tuple(cast("list[str]", families))
            )
        return StringWidget(
            multiline=cast("bool | None", multiline),
            placeholder=cast("str | None", placeholder),
            dynamic_prompts=cast("bool | None", dynamic_prompts),
            completions=completions,
        )
    if kind == "COLOR":
        if strict:
            _reject_unknown_fields(widget_data, frozenset({"type"}), "COLOR widget")
        return ColorWidget()
    if kind == "CURVE":
        if strict:
            _reject_unknown_fields(widget_data, frozenset({"type"}), "CURVE widget")
        return CurveWidget()
    if kind == "COMPOSITOR":
        if strict:
            _reject_unknown_fields(widget_data, frozenset({"type"}), "COMPOSITOR widget")
        return CompositorWidget()
    if not isinstance(kind, str) or not kind:
        raise ValueError("custom widget type must be a non-empty string")
    return CustomWidgetDescriptor(
        kind,
        cast(
            "Mapping[str, JsonValue]",
            {key: value for key, value in widget_data.items() if key != "type"},
        ),
    )


def _widget_from_wire(widget_wire: object, wire_version: int) -> Widget:
    if not isinstance(widget_wire, dict):
        raise ValueError(f"unsupported input widget: {widget_wire!r}")
    widget_data = cast("dict[str, object]", widget_wire)
    if widget_data.get("type") != "REPRESENTATIONS":
        return _widget_descriptor_from_wire(widget_data, wire_version, strict=True)
    _reject_unknown_fields(
        widget_data,
        frozenset({"type", "default", "userSwitchable", "representations"}),
        "REPRESENTATIONS widget",
    )
    default = widget_data.get("default")
    if not isinstance(default, str):
        raise ValueError("REPRESENTATIONS widget default must be a string")
    user_switchable = widget_data.get("userSwitchable")
    if type(user_switchable) is not bool:
        raise ValueError("REPRESENTATIONS widget userSwitchable must be a boolean")
    raw_representations = widget_data.get("representations")
    if not isinstance(raw_representations, list):
        raise ValueError("REPRESENTATIONS widget representations must be an array")
    representations: list[WidgetRepresentation] = []
    for raw_representation in cast("list[object]", raw_representations):
        if not isinstance(raw_representation, dict):
            raise ValueError("REPRESENTATIONS widget entries must be objects")
        representation = cast("dict[str, object]", raw_representation)
        _reject_unknown_fields(
            representation,
            frozenset({"id", "displayName", "widget"}),
            "REPRESENTATIONS widget entry",
        )
        representation_id = representation.get("id")
        if not isinstance(representation_id, str):
            raise ValueError("REPRESENTATIONS widget entry id must be a string")
        display_name = representation.get("displayName", "")
        if not isinstance(display_name, str):
            raise ValueError("REPRESENTATIONS widget entry displayName must be a string")
        descriptor = representation.get("widget")
        if not isinstance(descriptor, dict):
            raise ValueError("REPRESENTATIONS widget entry widget must be an object")
        representations.append(
            WidgetRepresentation(
                id=representation_id,
                display_name=display_name,
                widget=_widget_descriptor_from_wire(
                    cast("dict[str, object]", descriptor), wire_version, strict=True
                ),
            )
        )
    return WidgetRepresentations(
        representations=tuple(representations), default=default, user_switchable=user_switchable
    )


def type_expr_to_wire(expr: TypeExpr, wire_version: int = SCHEMA_WIRE_VERSION) -> dict[str, object]:
    wire: dict[str, object] = {"kind": expr.kind}
    if expr.kind == "tuple":
        wire["elements"] = [type_expr_to_wire(item, wire_version) for item in expr.elements]
    if expr.kind == "variable":
        wire["templateId"] = expr.template_id
        if expr.types:
            wire["allowed"] = list(expr.types)
        return wire
    if expr.types:
        wire["types"] = list(expr.types)
    if expr.template_id:
        wire["templateId"] = expr.template_id
    if expr.element is not None:
        wire["element"] = type_expr_to_wire(expr.element, wire_version)
    if expr.input_id:
        wire["input"] = expr.input_id
    if expr.index is not None:
        wire["index"] = expr.index
    return wire


def _media_policy_to_wire(spec: Input | Output, wire_version: int) -> dict[str, object]:
    result: dict[str, object] = {}
    if spec.alpha_policy != "preserve":
        result["alphaPolicy"] = spec.alpha_policy
    if spec.mask_polarity is not None:
        result["maskPolarity"] = spec.mask_polarity
    if spec.mask_semantic is not None:
        result["maskSemantic"] = spec.mask_semantic
    return result


def _media_policy_from_wire(entry: dict[str, Any], wire_version: int) -> dict[str, Any]:
    return {
        "alpha_policy": _expect_str(entry.get("alphaPolicy", "preserve"), "alphaPolicy"),
        "mask_polarity": _expect_str(entry["maskPolarity"], "maskPolarity")
        if "maskPolarity" in entry
        else None,
        "mask_semantic": _expect_str(entry["maskSemantic"], "maskSemantic")
        if "maskSemantic" in entry
        else None,
    }


def _input_entry_to_wire(spec: Input, wire_version: int) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": spec.id,
        "type": type_expr_to_wire(spec.type, wire_version),
        "required": spec.required,
    }
    entry.update(_media_policy_to_wire(spec, wire_version))
    if spec.default is not None:
        entry["default"] = spec.default
    if spec.on_absent is not None:
        entry["onAbsent"] = spec.on_absent
    if spec.widget is not None:
        try:
            widget_wire = _widget_to_wire(spec.widget, wire_version)
        except TypeError as exc:
            raise ValueError(f"input {spec.id!r}: {exc}") from exc
        if widget_wire is not None:
            entry["widget"] = widget_wire
    if spec.tooltip:
        entry["tooltip"] = spec.tooltip
    if spec.label:
        entry["label"] = spec.label
    if spec.adjacent_to is not None:
        entry["adjacentTo"] = spec.adjacent_to
    if spec.pin_first:
        entry["pinFirst"] = True
    if not spec.pin:
        entry["pin"] = False
    if spec.on_edit:
        entry["onEdit"] = _steps_to_wire(spec.on_edit, wire_version)
    if spec.advanced:
        entry["advanced"] = True
    if spec.hidden:
        entry["hidden"] = True
    if spec.lazy:
        entry["lazy"] = True
    if spec.accepts_storage:
        entry["acceptsStorage"] = True
    if spec.accepts_stream:
        entry["acceptsStream"] = True
    if spec.accepts_list:
        entry["acceptsList"] = True
    if spec.source_filename is not None:
        entry["sourceFilename"] = {
            "kind": spec.source_filename.kind,
            "category": spec.source_filename.category,
        }
    return entry


def _dynamic_entry_to_wire(
    spec: DynamicEntry, wire_version: int = SCHEMA_WIRE_VERSION
) -> dict[str, object]:
    if isinstance(spec, Input):
        return {"role": "input", **_input_entry_to_wire(spec, wire_version)}
    if isinstance(spec, InputFamilySpec):
        entry: dict[str, object] = {
            "role": "inputFamily",
            "id": spec.id,
            "template": [_dynamic_entry_to_wire(item, wire_version) for item in spec.template],
            "minMembers": spec.min_members,
            "required": spec.required,
        }
        if spec.independent_types:
            entry["independentTypes"] = True
        if spec.optional_after_minimum:
            entry["optionalAfterMinimum"] = True
        if spec.member_prefix is not None:
            entry["memberPrefix"] = spec.member_prefix
        if spec.member_names is not None:
            entry["memberNames"] = list(spec.member_names)
        elif spec.max_members is not None:
            entry["maxMembers"] = spec.max_members
        if spec.doc:
            entry["doc"] = spec.doc
        if spec.display_name:
            entry["displayName"] = spec.display_name
        return entry
    if isinstance(spec, DynamicComboSpec):
        entry = {
            "role": "dynamicCombo",
            "id": spec.id,
            "options": [
                {
                    "key": option.key,
                    "inputs": [
                        _dynamic_entry_to_wire(item, wire_version) for item in option.inputs
                    ],
                }
                for option in spec.options
            ],
            "required": spec.required,
        }
        if spec.default is not None:
            entry["default"] = spec.default
        if spec.hidden:
            entry["hidden"] = True
        if spec.doc:
            entry["doc"] = spec.doc
        if spec.display_name:
            entry["displayName"] = spec.display_name
        return entry
    if not isinstance(cast("object", spec), DynamicSlotSpec):
        raise TypeError(f"unsupported dynamic entry: {type(spec).__name__}")
    entry = {
        "role": "dynamicSlot",
        "id": spec.id,
        "required": spec.required,
        "inputs": [_dynamic_entry_to_wire(item, wire_version) for item in spec.inputs],
    }
    if spec.variants is not None:
        entry["variants"] = [_variant_to_wire(variant, wire_version) for variant in spec.variants]
    else:
        assert spec.slot_type is not None
        entry["slotType"] = type_expr_to_wire(spec.slot_type, wire_version)
        if spec.force_input:
            entry["forceInput"] = True
    if spec.type_template_id:
        entry["typeTemplateId"] = spec.type_template_id
    if spec.doc:
        entry["doc"] = spec.doc
    if spec.display_name:
        entry["displayName"] = spec.display_name
    return entry


def _variant_to_wire(variant: SlotVariant, wire_version: int) -> dict[str, object]:
    wire: dict[str, object] = {
        "key": variant.key,
        "type": type_expr_to_wire(variant.type, wire_version),
        "inputs": [_dynamic_entry_to_wire(dep, wire_version) for dep in variant.inputs],
    }
    if variant.doc:
        wire["doc"] = variant.doc
    return wire


def _output_entry_to_wire(out: Output, wire_version: int) -> dict[str, object]:
    out_entry: dict[str, object] = {
        "id": out.id,
        "type": type_expr_to_wire(out.type, wire_version),
    }
    out_entry.update(_media_policy_to_wire(out, wire_version))
    if out.optional:
        out_entry["optional"] = True
    if out.tooltip:
        out_entry["tooltip"] = out.tooltip
    if out.label:
        out_entry["label"] = out.label
    if out.preview:
        out_entry["preview"] = True
    if out.represents is not None:
        represents: dict[str, object] = {
            "input": out.represents.input,
            "rendition": out.represents.rendition,
        }
        if out.represents.applies is not None:
            represents["applies"] = {
                combo_id: list(values) for combo_id, values in out.represents.applies.items()
            }
        out_entry["represents"] = represents
    if out.known_value is not None:
        out_entry["knownValue"] = {"input": out.known_value.input}
    return out_entry


def schema_to_wire(
    schema: NodeSchema,
    *,
    wire_version: int = SCHEMA_WIRE_VERSION,
    replacement_schemas: Mapping[str, NodeSchema] | None = None,
) -> dict[str, object]:
    if type(wire_version) is not int or wire_version != SCHEMA_WIRE_VERSION:
        raise ValueError(f"unsupported schemaVersion: {wire_version!r}")
    interface: list[dict[str, object]] = []
    has_body = bool(schema.body)
    for spec in () if has_body else schema.input_entries:
        try:
            interface.append(_dynamic_entry_to_wire(spec, wire_version))
        except ValueError as exc:
            raise ValueError(f"node {schema.node_type!r}, {exc}") from exc
    for out in () if has_body else schema.outputs:
        interface.append({"role": "output", **_output_entry_to_wire(out, wire_version)})
    for out_fam in schema.output_families:
        out_fam_entry: dict[str, object] = {
            "role": "outputFamily",
            "id": out_fam.id,
            "type": type_expr_to_wire(out_fam.type, wire_version),
            "minMembers": out_fam.min_members,
        }
        if out_fam.max_members is not None:
            out_fam_entry["maxMembers"] = out_fam.max_members
        if out_fam.doc:
            out_fam_entry["doc"] = out_fam.doc
        if out_fam.preview:
            out_fam_entry["preview"] = True
        if out_fam.count is not None:
            out_fam_entry["count"] = {"input": out_fam.count.input, "suffix": out_fam.count.suffix}
        interface.append(out_fam_entry)
    descriptors = schema.output_descriptors
    if descriptors is not None:
        descriptor_entry: dict[str, object] = {
            "role": "outputDescriptors",
            "input": descriptors.input,
            "minEntries": descriptors.min_entries,
            "maxEntries": descriptors.max_entries,
            "fixedIds": descriptors.fixed_ids,
            "choices": [
                {
                    "id": choice.id,
                    "type": type_expr_to_wire(choice.type, wire_version),
                    **({"displayName": choice.label} if choice.label else {}),
                    **({"doc": choice.tooltip} if choice.tooltip else {}),
                    **({"optional": True} if choice.optional else {}),
                    **({"preview": True} if choice.preview else {}),
                    **_media_policy_to_wire(choice, wire_version),
                }
                for choice in descriptors.choices
            ],
        }
        if descriptors.probe is not None:
            descriptor_entry["probe"] = {
                "input": descriptors.probe.input,
                "kind": descriptors.probe.kind,
                "revision": descriptors.probe.revision,
            }
        interface.append(descriptor_entry)
    wire: dict[str, object] = {
        "schemaVersion": wire_version,
        "nodeType": schema.node_type,
        "version": schema.version,
        "displayName": schema.display_name,
        "category": schema.category,
        "description": schema.description,
        "idempotent": schema.idempotent,
        "interface": interface,
    }
    if has_body:
        try:
            _body_fields_to_wire(schema, wire, wire_version)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"node {schema.node_type!r}, {exc}") from exc
    if schema.editor_role is not None:
        wire["editorRole"] = schema.editor_role
    if schema.additional_inputs is not None:
        wire["additionalInputs"] = type_expr_to_wire(schema.additional_inputs, wire_version)
    if schema.slot_choices:
        wire["slotChoices"] = [[slot_id, key] for slot_id, key in schema.slot_choices]
    if schema.occupies:
        wire["occupies"] = list(schema.occupies)
    if schema.io_bound:
        wire["ioBound"] = True
    if schema.dispatch_affinity is not None:
        wire["dispatchAffinity"] = schema.dispatch_affinity
    if schema.deprecation is not None:
        dep: dict[str, object] = {"message": schema.deprecation.message}
        if schema.deprecation.since:
            dep["since"] = schema.deprecation.since
        if schema.deprecation.replacement:
            dep["replacement"] = schema.deprecation.replacement
        wire["deprecation"] = dep
    if schema.search_visibility != "normal":
        wire["searchVisibility"] = schema.search_visibility
    if schema.search_terms:
        wire["searchTerms"] = list(schema.search_terms)
    if schema.replacements:
        replacements = schema.replacements
        if replacements:
            wire["replacements"] = [rule_to_wire(rule) for rule in replacements]
    if schema.output_node:
        wire["outputNode"] = True
    if schema.chunk_safe is not None:
        wire["chunkSafe"] = {
            "inputs": list(schema.chunk_safe[0]),
            "outputs": list(schema.chunk_safe[1]),
            **(
                {
                    "applies": {
                        key: list(values) for key, values in schema.chunk_safe_applies.items()
                    }
                }
                if schema.chunk_safe_applies is not None
                else {}
            ),
        }
    if schema.emits_previews:
        wire["emitsPreviews"] = True
    if schema.may_expand_graph:
        wire["mayExpandGraph"] = True
    if schema.admission_validation:
        wire["admissionValidation"] = True
    if schema.execution_context:
        wire["executionContext"] = True
    if schema.input_values:
        wire["inputValues"] = True
    if schema.widget_groups:
        groups: list[dict[str, object]] = []
        for group in schema.widget_groups:
            encoded: dict[str, object] = {
                "input": group.input,
                "values": list(group.values),
                "members": list(group.members),
            }
            if group.requires:
                encoded["requires"] = [
                    {"input": condition.input, "values": list(condition.values)}
                    for condition in group.requires
                ]
            groups.append(encoded)
        wire["widgetGroups"] = groups
    if schema.mirror is not None:
        mirror: dict[str, object] = {
            "kind": schema.mirror.kind,
            "precision": schema.mirror.precision,
        }
        if schema.mirror.tolerance is not None:
            tolerance: dict[str, object] = {}
            if schema.mirror.tolerance.relative is not None:
                tolerance["relative"] = schema.mirror.tolerance.relative
            if schema.mirror.tolerance.per_channel is not None:
                tolerance["perChannel"] = schema.mirror.tolerance.per_channel
            mirror["tolerance"] = tolerance
        if schema.mirror.grammar_version is not None:
            mirror["grammarVersion"] = schema.mirror.grammar_version
        if schema.mirror.source is not None:
            mirror["source"] = schema.mirror.source
        if schema.mirror.applies is not None:
            mirror["applies"] = {
                combo_id: list(values) for combo_id, values in schema.mirror.applies.items()
            }
        wire["mirror"] = mirror
    if schema.selector is not None:
        wire["selector"] = {
            "input": schema.selector.input,
            "branches": dict(schema.selector.branches),
        }
    return wire


def schema_signature(schema: NodeSchema) -> str:
    """Stable content hash of the wire form; a component of every cache key
    (hazard H4: keys derive from schema signature + input fingerprints).

    Admission hints (occupies, ioBound, dispatchAffinity) are excluded: they
    say where/how a node may be scheduled, never what it computes, so changing
    them must not invalidate caches - entries stay shareable across differently
    configured hosts. Lifecycle/presentation metadata (deprecation, searchVisibility,
    searchTerms, replacements) is excluded for the same reason: deprecating
    or hiding a node, adding search keywords, or shipping migration advice
    for its predecessors, changes how documents evolve or how the node is
    found, never what this node computes. Targeting metadata (outputNode)
    is excluded too: whether submission formats target a node by default
    never changes what it computes. Capability metadata (emitsPreviews, mayExpandGraph) is
    excluded for the same reason: whether a node ships live previews while
    running, or whether its execution may return a runtime graph expansion
    payload, never changes what it computes. Mirror, output representation,
    and pre-execution
    known-value declarations are excluded for the same reason: declaring,
    changing, or removing one never changes what the node computes.
    Presentation prose (displayName, category, description,
    per-port doc and displayName) is excluded for the same reason (hazard H15:
    presentation is pixels, never identity) - renaming a badge or rewording
    a tooltip must never invalidate caches or migrate workflows. Explicit
    dynamicPrompts state is the sole widget exception because it changes
    client serialization before submission."""

    def strip_editor_widget(entry: DynamicEntry) -> DynamicEntry:
        if isinstance(entry, Input):
            if (
                isinstance(entry.widget, (CurveWidget, CompositorWidget))
                or (
                    isinstance(entry.widget, ComboWidget) and entry.widget.option_source is not None
                )
                or (
                    isinstance(entry.widget, WidgetRepresentations)
                    and any(
                        isinstance(representation.widget, (CurveWidget, CompositorWidget))
                        or (
                            isinstance(representation.widget, ComboWidget)
                            and representation.widget.option_source is not None
                        )
                        for representation in entry.widget.representations
                    )
                )
            ):
                return replace(entry, widget=None)
            return entry
        if isinstance(entry, InputFamilySpec):
            template = tuple(strip_editor_widget(item) for item in entry.template)
            if template == entry.template:
                return entry
            return InputFamilySpec(
                entry.id,
                template,
                min_members=entry.min_members,
                max_members=None if entry.member_names is not None else entry.max_members,
                doc=entry.doc,
                display_name=entry.display_name,
                required=entry.required,
                member_prefix=entry.member_prefix,
                member_names=entry.member_names,
                independent_types=entry.independent_types,
                optional_after_minimum=entry.optional_after_minimum,
            )
        if isinstance(entry, DynamicComboSpec):
            options = tuple(
                replace(option, inputs=tuple(strip_editor_widget(item) for item in option.inputs))
                for option in entry.options
            )
            if options == entry.options:
                return entry
            return replace(entry, options=options)
        inputs = tuple(strip_editor_widget(item) for item in entry.inputs)
        variants = (
            tuple(
                replace(
                    variant,
                    inputs=tuple(strip_editor_widget(item) for item in variant.inputs),
                )
                for variant in entry.variants
            )
            if entry.variants is not None
            else None
        )
        if inputs == entry.inputs and variants == entry.variants:
            return entry
        return replace(entry, inputs=inputs, variants=variants)

    signature_schema = (
        schema
        if schema.body
        else replace(
            schema,
            inputs=tuple(
                cast("Input", strip_editor_widget(input_spec)) for input_spec in schema.inputs
            ),
            input_families=tuple(
                cast("InputFamilySpec", strip_editor_widget(family))
                for family in schema.input_families
            ),
            combos=tuple(
                cast("DynamicComboSpec", strip_editor_widget(combo)) for combo in schema.combos
            ),
            slots=tuple(
                cast("DynamicSlotSpec", strip_editor_widget(slot)) for slot in schema.slots
            ),
            replacements=(),
            outputs=tuple(replace(output, known_value=None) for output in schema.outputs),
        )
    )
    wire = schema_to_wire(signature_schema)
    wire.pop("occupies", None)
    wire.pop("ioBound", None)
    wire.pop("deprecation", None)
    wire.pop("searchVisibility", None)
    wire.pop("searchTerms", None)
    wire.pop("dispatchAffinity", None)
    wire.pop("replacements", None)
    wire.pop("outputNode", None)
    wire.pop("emitsPreviews", None)
    wire.pop("mayExpandGraph", None)
    wire.pop("widgetGroups", None)
    wire.pop("mirror", None)
    wire.pop("displayName", None)
    wire.pop("editorRole", None)
    wire.pop("category", None)
    wire.pop("description", None)

    def strip_presentation(entry: dict[str, object]) -> None:
        entry.pop("doc", None)
        entry.pop("tooltip", None)
        if "onAbsent" in entry and entry["onAbsent"] == (
            "skip" if entry.get("required", True) else "omit"
        ):
            entry.pop("onAbsent")
        widget = entry.pop("widget", None)
        dynamic_identity = dynamic_prompts_identity(widget)
        if dynamic_identity is not None:
            entry["dynamicPrompts"] = dynamic_identity
        asset_identity = _asset_widget_identity(widget)
        if asset_identity is not None:
            entry["asset"] = asset_identity
        entry.pop("displayName", None)
        entry.pop("label", None)
        entry.pop("adjacentTo", None)
        entry.pop("forceInput", None)
        entry.pop("pinFirst", None)
        entry.pop("pin", None)
        entry.pop("onEdit", None)
        entry.pop("advanced", None)
        entry.pop("hidden", None)
        entry.pop("preview", None)
        entry.pop("represents", None)
        entry.pop("knownValue", None)
        for child in cast("list[dict[str, object]]", entry.get("template", ())):
            strip_presentation(child)
        for option in cast("list[dict[str, object]]", entry.get("options", ())):
            for child in cast("list[dict[str, object]]", option.get("inputs", ())):
                strip_presentation(child)
        for child in cast("list[dict[str, object]]", entry.get("inputs", ())):
            strip_presentation(child)
        for variant in cast("list[dict[str, object]]", entry.get("variants", ())):
            variant.pop("doc", None)
            for dep in cast("list[dict[str, object]]", variant.get("inputs", ())):
                strip_presentation(dep)
        for choice in cast("list[dict[str, object]]", entry.get("choices", ())):
            strip_presentation(choice)

    def dynamic_prompts_identity(widget: object) -> object | None:
        if not isinstance(widget, dict):
            return None
        descriptor = cast("dict[str, object]", widget)
        if descriptor.get("type") == "STRING" and "dynamicPrompts" in descriptor:
            return descriptor["dynamicPrompts"]
        if descriptor.get("type") != "REPRESENTATIONS":
            return None
        identities: list[list[object]] = []
        for raw in cast("list[dict[str, object]]", descriptor.get("representations", [])):
            identity = dynamic_prompts_identity(raw.get("widget"))
            if identity is not None:
                identities.append([raw.get("id"), identity])
        if not identities:
            return None
        return {"default": descriptor.get("default"), "representations": identities}

    for entry in cast("list[dict[str, object]]", wire["interface"]):
        strip_presentation(entry)
    if schema.body:
        for key in ("body", "tables", "derived", "validate", "titleSuffix", "menu"):
            wire.pop(key, None)
        wire["body"] = _signature_rows(schema.body, schema, strip_presentation)
    canonical = json.dumps(wire, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=20).hexdigest()


def _asset_widget_identity(widget: object) -> object | None:
    """The asset kind and selection a run resolves an asset by; they join both signatures."""
    if not isinstance(widget, dict):
        return None
    descriptor = cast("dict[str, object]", widget)
    if descriptor.get("type") == "ASSET":
        identity = {key: descriptor[key] for key in ("kind", "selection") if key in descriptor}
        return identity or None
    if descriptor.get("type") == "REPRESENTATIONS":
        found = [
            [raw.get("id"), _asset_widget_identity(raw.get("widget"))]
            for raw in cast("list[dict[str, object]]", descriptor.get("representations", []))
        ]
        found = [item for item in found if item[1] is not None]
        return found or None
    return None


_FORMULA_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _structure_formula(formula: body.Formula, schema: NodeSchema) -> dict[str, object]:
    """A structure formula with the derived formulas and tables it names written out."""
    derived: dict[str, str] = {}
    tables: dict[str, object] = {}
    pending = [formula.text]
    while pending:
        for name in _FORMULA_NAME.findall(pending.pop()):
            if name in schema.derived and name not in derived:
                derived[name] = schema.derived[name].text
                pending.append(derived[name])
            elif name in schema.tables and name not in tables:
                tables[name] = _table_to_wire(schema.tables[name])
    wire: dict[str, object] = {"formula": formula.text}
    if derived:
        wire["derived"] = derived
    if tables:
        wire["tables"] = tables
    return wire


def _signature_rows(
    rows: Sequence[body.BodyRow | body.Use],
    schema: NodeSchema,
    strip: Any,
) -> list[dict[str, object]]:
    """The structure and execution fields of a scope's rows, in one canonical order.

    Sections and Rows open no scope and are layout, so their rows join the
    scope around them; notes, buttons and a state that is not part of a run
    drop out. Rows are
    sorted, so reordering rows or moving one between sections changes
    neither signature."""
    found: list[dict[str, object]] = []
    for row in rows:
        if isinstance(row, (body.Section, body.Row)):
            found.extend(_signature_rows(row.rows, schema, strip))
            continue
        if isinstance(row, (body.Note, body.Button)) or (
            isinstance(row, body.State) and not row.part_of_run
        ):
            continue
        found.append(_signature_row(row, schema, strip))
    return sorted(found, key=lambda entry: (str(entry["kind"]), str(entry.get("id", ""))))


def _signature_row(
    row: body.BodyRow | body.Use, schema: NodeSchema, strip: Any
) -> dict[str, object]:
    wire_version = SCHEMA_WIRE_VERSION
    if isinstance(row, Input):
        entry = {"kind": "input", **_input_entry_to_wire(row, wire_version)}
        strip(entry)
        return entry
    if isinstance(row, Output):
        entry = {"kind": "output", **_output_entry_to_wire(row, wire_version)}
        strip(entry)
        return entry
    if isinstance(row, body.State):
        return body_row_to_wire(row, wire_version)
    if isinstance(row, body.Use):
        entry: dict[str, object] = {"kind": "use", "id": row.id}
        if row.default is not body.UNSET:
            entry["default"] = row.default
        if row.required is not body.UNSET:
            entry["required"] = row.required
        if row.only is not None:
            entry["only"] = sorted(row.only)
        if row.inside:
            entry["inside"] = _signature_rows(row.inside, schema, strip)
        return entry
    if isinstance(row, body.Group):
        assert row.keys is not None
        entry = {"kind": "group", "id": row.id, "keys": _keys_to_wire(row.keys)}
        if row.value is not None:
            entry["value"] = _signature_row(row.value, schema, strip)
        else:
            assert row.member is not None
            entry["member"] = _signature_rows(row.member, schema, strip)
        if isinstance(row.grows, body.Count):
            entry["count"] = _structure_formula(row.grows.formula, schema)
        entry["minMembers"] = row.min_members
        if row.max_members is not None:
            entry["maxMembers"] = row.max_members
        for flag in ("independent_types", "optional_after_minimum"):
            if getattr(row, flag):
                entry[_camel(flag)] = True
        return entry
    if isinstance(row, body.Choice):
        entry = {
            "kind": "choice",
            "id": row.id,
            "required": row.required,
            "options": sorted(
                (
                    {"key": option.key, "rows": _signature_rows(option.rows, schema, strip)}
                    for option in row.options
                ),
                key=lambda option: str(option["key"]),
            ),
        }
        if row.shared:
            entry["shared"] = _signature_rows(row.shared, schema, strip)
        if row.leader is not None:
            entry["leader"] = _structure_formula(row.leader, schema)
        return entry
    if isinstance(row, body.TypeChoice):
        entry = {"kind": "typeChoice", "id": row.id, "required": bool(row.required)}
        if row.variants is not None:
            entry["variants"] = sorted(
                (
                    {
                        "key": variant.key,
                        "type": type_expr_to_wire(variant.type, wire_version),
                        "rows": _signature_rows(variant.rows, schema, strip),
                    }
                    for variant in row.variants
                ),
                key=lambda variant: str(variant["key"]),
            )
        else:
            assert row.type is not None
            entry["type"] = type_expr_to_wire(row.type, wire_version)
        entry["rows"] = _signature_rows(row.rows, schema, strip)
        if row.type_variable:
            entry["typeVariable"] = row.type_variable
        return entry
    assert isinstance(row, body.Described)
    entry = {
        "kind": "described",
        "id": row.id,
        "kinds": _signature_rows(row.kinds, schema, strip),
        "minEntries": row.min_entries,
        "maxEntries": row.max_entries,
        "fixedIds": row.fixed_ids,
    }
    if row.probe is not None:
        entry["probe"] = {
            "input": row.probe.input,
            "kind": row.probe.kind,
            "revision": row.probe.revision,
        }
    return entry


def _expect_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast("dict[str, Any]", value)


def _expect_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast("list[Any]", value)


def _expect_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _expect_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _expect_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _expect_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _expect_strings(value: object, field: str) -> tuple[str, ...]:
    values = _expect_list(value, field)
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"{field} must be an array of strings")
    return tuple(cast("list[str]", values))


def type_expr_from_wire(wire: dict[str, Any], wire_version: int = SCHEMA_WIRE_VERSION) -> TypeExpr:
    kind = _expect_str(wire.get("kind"), "type.kind")
    if kind == "variable":
        if "types" in wire:
            raise ValueError("type.types is not a variable field; use allowed")
        types = _expect_strings(wire.get("allowed", []), "type.allowed")
    else:
        if "allowed" in wire:
            raise ValueError("type.allowed is only legal on variable TypeExprs; use types")
        types = _expect_strings(wire.get("types", []), "type.types")
    template_id = _expect_str(wire.get("templateId", ""), "type.templateId")
    element_wire = wire.get("element")
    element = (
        None
        if element_wire is None
        else type_expr_from_wire(_expect_object(element_wire, "type.element"), wire_version)
    )
    elements_wire = wire.get("elements", [])
    if not isinstance(elements_wire, list):
        raise ValueError("type.elements must be an array")
    elements = tuple(
        type_expr_from_wire(_expect_object(item, "type.elements position"), wire_version)
        for item in cast(list[object], elements_wire)
    )
    return TypeExpr(
        kind=cast("Any", kind),
        types=types,
        template_id=template_id,
        element=element,
        elements=elements,
        input_id=_expect_str(wire.get("input", ""), "type.input"),
        index=wire.get("index"),
    )


def _input_entry_from_wire(entry: dict[str, Any], wire_version: int) -> Input:
    on_absent = entry.get("onAbsent")
    widget_wire = entry.get("widget")
    widget = None if widget_wire is None else _widget_from_wire(widget_wire, wire_version)
    source_wire = entry.get("sourceFilename")
    source_filename = None
    if source_wire is not None:
        source_data = _expect_object(source_wire, "input.sourceFilename")
        _reject_unknown_fields(source_data, frozenset({"kind", "category"}), "input.sourceFilename")
        source_filename = SourceFilenameSpec(
            kind=cast("Any", _expect_str(source_data.get("kind"), "input.sourceFilename.kind")),
            category=cast(
                "Any", _expect_str(source_data.get("category"), "input.sourceFilename.category")
            ),
        )
    return Input(
        id=_expect_str(entry.get("id"), "input.id"),
        type=type_expr_from_wire(_expect_object(entry.get("type"), "input.type"), wire_version),
        required=_expect_bool(entry.get("required", True), "input.required"),
        default=entry.get("default"),
        tooltip=_expect_str(entry.get("tooltip", ""), "input.tooltip"),
        on_absent=None
        if on_absent is None
        else cast("AbsentPolicy", _expect_str(on_absent, "input.onAbsent")),
        widget=widget,
        label=_expect_str(entry.get("label", ""), "input.label"),
        adjacent_to=None
        if "adjacentTo" not in entry
        else _expect_str(entry["adjacentTo"], "input.adjacentTo"),
        pin_first=_expect_bool(entry.get("pinFirst", False), "input.pinFirst"),
        pin=_expect_bool(entry.get("pin", True), "input.pin"),
        on_edit=_steps_from_wire(entry.get("onEdit", []), "input.onEdit", wire_version),
        advanced=_expect_bool(entry.get("advanced", False), "input.advanced"),
        hidden=_expect_bool(entry.get("hidden", False), "input.hidden"),
        lazy=_expect_bool(entry.get("lazy", False), "input.lazy"),
        source_filename=source_filename,
        accepts_storage=_expect_bool(entry.get("acceptsStorage", False), "input.acceptsStorage"),
        accepts_stream=_expect_bool(entry.get("acceptsStream", False), "input.acceptsStream"),
        accepts_list=_expect_bool(entry.get("acceptsList", False), "input.acceptsList"),
        **_media_policy_from_wire(entry, wire_version),
    )


def _variant_from_wire(variant: dict[str, Any], wire_version: int) -> SlotVariant:
    return SlotVariant(
        key=_expect_str(variant.get("key"), "variant.key"),
        type=type_expr_from_wire(_expect_object(variant.get("type"), "variant.type"), wire_version),
        inputs=tuple(
            _dynamic_entry_from_wire(_expect_object(dep, "variant.inputs entry"), wire_version)
            for dep in _expect_list(variant.get("inputs"), "variant.inputs")
        ),
        doc=_expect_str(variant.get("doc", ""), "variant.doc"),
    )


def _dynamic_entry_from_wire(entry: dict[str, Any], wire_version: int) -> DynamicEntry:
    role = _expect_str(entry.get("role"), "dynamic entry role")
    if role == "input":
        return _input_entry_from_wire(entry, wire_version)
    if role == "inputFamily":
        if "type" in entry and "template" in entry:
            raise ValueError("inputFamily must not carry both type and template")
        if "type" in entry:
            raise ValueError("inputFamily.type is not supported")
        max_members = entry.get("maxMembers")
        names = entry.get("memberNames")
        prefix = entry.get("memberPrefix")
        return InputFamilySpec(
            id=_expect_str(entry.get("id"), "inputFamily.id"),
            template=tuple(
                _dynamic_entry_from_wire(
                    _expect_object(item, "inputFamily.template entry"), wire_version
                )
                for item in _expect_list(entry.get("template"), "inputFamily.template")
            ),
            min_members=_expect_int(entry.get("minMembers", 0), "inputFamily.minMembers"),
            max_members=None
            if max_members is None
            else _expect_int(max_members, "inputFamily.maxMembers"),
            doc=_expect_str(entry.get("doc", ""), "inputFamily.doc"),
            display_name=_expect_str(entry.get("displayName", ""), "inputFamily.displayName"),
            required=_expect_bool(entry.get("required", True), "inputFamily.required"),
            independent_types=_expect_bool(
                entry.get("independentTypes", False), "inputFamily.independentTypes"
            ),
            optional_after_minimum=_expect_bool(
                entry.get("optionalAfterMinimum", False), "inputFamily.optionalAfterMinimum"
            ),
            member_prefix=None
            if prefix is None
            else _expect_str(prefix, "inputFamily.memberPrefix"),
            member_names=None
            if names is None
            else _expect_strings(names, "inputFamily.memberNames"),
        )
    if role == "dynamicCombo":
        options: list[DynamicComboOption] = []
        for raw_option in _expect_list(entry.get("options"), "dynamicCombo.options"):
            option = _expect_object(raw_option, "dynamicCombo.options entry")
            options.append(
                DynamicComboOption(
                    key=_expect_str(option.get("key"), "dynamicCombo option key"),
                    inputs=tuple(
                        _dynamic_entry_from_wire(
                            _expect_object(item, "dynamicCombo option input"), wire_version
                        )
                        for item in _expect_list(option.get("inputs"), "dynamicCombo option inputs")
                    ),
                )
            )
        default = entry.get("default")
        return DynamicComboSpec(
            id=_expect_str(entry.get("id"), "dynamicCombo.id"),
            options=tuple(options),
            default=None if default is None else _expect_str(default, "dynamicCombo.default"),
            required=_expect_bool(entry.get("required", True), "dynamicCombo.required"),
            doc=_expect_str(entry.get("doc", ""), "dynamicCombo.doc"),
            display_name=_expect_str(entry.get("displayName", ""), "dynamicCombo.displayName"),
            hidden=_expect_bool(entry.get("hidden", False), "dynamicCombo.hidden"),
        )
    if role == "dynamicSlot":
        has_variants = "variants" in entry
        has_slot_type = "slotType" in entry
        if has_variants == has_slot_type:
            raise ValueError("dynamicSlot requires exactly one of variants or slotType")
        type_template_id = _expect_str(
            entry.get("typeTemplateId", ""), "dynamicSlot.typeTemplateId"
        )
        if "typeTemplateId" in entry and (not type_template_id):
            raise ValueError("dynamicSlot.typeTemplateId must be a non-empty string")
        return DynamicSlotSpec(
            id=_expect_str(entry.get("id"), "dynamicSlot.id"),
            variants=tuple(
                _variant_from_wire(
                    _expect_object(variant, "dynamicSlot.variants entry"), wire_version
                )
                for variant in _expect_list(entry.get("variants"), "dynamicSlot.variants")
            )
            if has_variants
            else None,
            slot_type=type_expr_from_wire(
                _expect_object(entry.get("slotType"), "dynamicSlot.slotType"), wire_version
            )
            if has_slot_type
            else None,
            inputs=tuple(
                _dynamic_entry_from_wire(
                    _expect_object(item, "dynamicSlot.inputs entry"), wire_version
                )
                for item in _expect_list(
                    entry.get("inputs") if has_slot_type else entry.get("inputs", []),
                    "dynamicSlot.inputs",
                )
            ),
            required=_expect_bool(entry.get("required", has_variants), "dynamicSlot.required"),
            force_input=_expect_bool(entry.get("forceInput", False), "dynamicSlot.forceInput"),
            doc=_expect_str(entry.get("doc", ""), "dynamicSlot.doc"),
            display_name=_expect_str(entry.get("displayName", ""), "dynamicSlot.displayName"),
            type_template_id=type_template_id,
        )
    raise ValueError(f"unsupported dynamic entry role: {role!r}")


def _output_descriptors_from_wire(
    entry: dict[str, Any], wire_version: int
) -> OutputDescriptorsSpec:
    _reject_unknown_fields(
        entry,
        frozenset({"role", "input", "choices", "minEntries", "maxEntries", "fixedIds", "probe"}),
        "outputDescriptors",
    )
    choices: list[Output] = []
    for raw in _expect_list(entry.get("choices"), "outputDescriptors.choices"):
        choice = _expect_object(raw, "output descriptor choice")
        _reject_unknown_fields(
            choice,
            frozenset(
                {
                    "id",
                    "type",
                    "displayName",
                    "doc",
                    "optional",
                    "preview",
                    "alphaPolicy",
                    "maskPolarity",
                    "maskSemantic",
                }
            ),
            "output descriptor choice",
        )
        choices.append(
            Output(
                id=_expect_str(choice.get("id"), "output descriptor choice id"),
                type=type_expr_from_wire(
                    _expect_object(choice.get("type"), "choice type"), wire_version
                ),
                label=_expect_str(choice.get("displayName", ""), "choice displayName"),
                tooltip=_expect_str(choice.get("doc", ""), "choice doc"),
                optional=_expect_bool(choice.get("optional", False), "choice optional"),
                preview=_expect_bool(choice.get("preview", False), "choice preview"),
                **_media_policy_from_wire(choice, wire_version),
            )
        )
    probe = None
    if "probe" in entry:
        data = _expect_object(entry["probe"], "outputDescriptors.probe")
        _reject_unknown_fields(data, frozenset({"input", "kind", "revision"}), "output probe")
        probe = OutputProbeSpec(
            input=_expect_str(data.get("input"), "output probe input"),
            kind=_expect_str(data.get("kind"), "output probe kind"),
            revision=_expect_str(data.get("revision"), "output probe revision"),
        )
    return OutputDescriptorsSpec(
        input=_expect_str(entry.get("input"), "outputDescriptors.input"),
        choices=tuple(choices),
        min_entries=_expect_int(entry.get("minEntries"), "outputDescriptors.minEntries"),
        max_entries=_expect_int(entry.get("maxEntries"), "outputDescriptors.maxEntries"),
        fixed_ids=_expect_bool(entry.get("fixedIds"), "outputDescriptors.fixedIds"),
        probe=probe,
    )


def _output_entry_from_wire(entry: dict[str, Any], wire_version: int) -> Output:
    expr = type_expr_from_wire(_expect_object(entry.get("type"), "output.type"), wire_version)
    preview = False
    preview = _expect_bool(entry.get("preview", False), "output.preview")
    represents = None
    if "represents" in entry:
        data = _expect_object(entry.get("represents"), "output.represents")
        _reject_unknown_fields(
            data, frozenset({"input", "rendition", "applies"}), "output.represents"
        )
        applies = None
        if "applies" in data:
            applies = _applies_from_wire(data.get("applies"), "output.represents.applies")
        represents = OutputRepresents(
            input=_expect_str(data.get("input"), "output.represents.input"),
            rendition=_expect_str(data.get("rendition"), "output.represents.rendition"),
            applies=applies,
        )
    known_value = None
    if "knownValue" in entry:
        data = _expect_object(entry.get("knownValue"), "output.knownValue")
        _reject_unknown_fields(data, frozenset({"input"}), "output.knownValue")
        known_value = OutputKnownValue(
            input=_expect_str(data.get("input"), "output.knownValue.input")
        )
    return Output(
        id=_expect_str(entry.get("id"), "output.id"),
        type=expr,
        tooltip=_expect_str(entry.get("tooltip", ""), "output.tooltip"),
        optional=_expect_bool(entry.get("optional", False), "output.optional"),
        preview=preview,
        represents=represents,
        known_value=known_value,
        label=_expect_str(entry.get("label", ""), "output.label"),
        **_media_policy_from_wire(entry, wire_version),
    )


def schema_from_wire(wire: dict[str, Any]) -> NodeSchema:
    wire_version = wire.get("schemaVersion")
    if type(wire_version) is not int or wire_version != SCHEMA_WIRE_VERSION:
        raise ValueError(f"unsupported schemaVersion: {wire.get('schemaVersion')!r}")
    dispatch_affinity = None
    if "dispatchAffinity" in wire:
        raw_dispatch_affinity = _expect_str(wire.get("dispatchAffinity"), "dispatchAffinity")
        if raw_dispatch_affinity != "native":
            raise ValueError(f"unsupported dispatchAffinity: {raw_dispatch_affinity!r}")
        dispatch_affinity = raw_dispatch_affinity
    inputs: list[Input] = []
    outputs: list[Output] = []
    input_families: list[InputFamilySpec] = []
    output_families: list[OutputFamilySpec] = []
    output_descriptors = None
    combos: list[DynamicComboSpec] = []
    slots: list[DynamicSlotSpec] = []
    input_order: list[str] = []
    for raw_entry in _expect_list(wire.get("interface"), "interface"):
        entry = _expect_object(raw_entry, "interface entry")
        role = _expect_str(entry.get("role"), "interface role")
        if role == "outputDescriptors":
            if output_descriptors is not None:
                raise ValueError("only one outputDescriptors construct is allowed")
            output_descriptors = _output_descriptors_from_wire(entry, wire_version)
            continue
        if role not in {
            "input",
            "inputFamily",
            "dynamicCombo",
            "dynamicSlot",
            "output",
            "outputFamily",
        }:
            raise ValueError(f"unsupported interface role: {role!r}")
        if role in {"input", "inputFamily", "dynamicCombo", "dynamicSlot"}:
            dynamic = _dynamic_entry_from_wire(entry, cast("int", wire_version))
            input_order.append(dynamic.id)
            if isinstance(dynamic, Input):
                inputs.append(dynamic)
            elif isinstance(dynamic, InputFamilySpec):
                input_families.append(dynamic)
            elif isinstance(dynamic, DynamicComboSpec):
                combos.append(dynamic)
            else:
                slots.append(dynamic)
            continue
        expr = type_expr_from_wire(
            _expect_object(entry.get("type"), "interface type"), wire_version
        )
        if role == "outputFamily":
            out_max = entry.get("maxMembers")
            has_count = "count" in entry
            raw_count = entry.get("count")
            count = None
            if has_count:
                count_data = _expect_object(raw_count, "outputFamily.count")
                if set(count_data) != {"input", "suffix"}:
                    raise ValueError("outputFamily.count must contain exactly input and suffix")
                count = OutputCountSpec(
                    input=_expect_str(count_data.get("input"), "outputFamily.count.input"),
                    suffix=cast(
                        "Literal['index']",
                        _expect_str(count_data.get("suffix"), "outputFamily.count.suffix"),
                    ),
                )
            preview = False
            preview = _expect_bool(entry.get("preview", False), "outputFamily.preview")
            output_families.append(
                OutputFamilySpec(
                    id=_expect_str(entry.get("id"), "outputFamily.id"),
                    type=expr,
                    min_members=_expect_int(entry.get("minMembers", 0), "outputFamily.minMembers"),
                    max_members=None
                    if out_max is None
                    else _expect_int(out_max, "outputFamily.maxMembers"),
                    doc=_expect_str(entry.get("doc", ""), "outputFamily.doc"),
                    preview=preview,
                    count=count,
                )
            )
        elif role == "output":
            outputs.append(_output_entry_from_wire(entry, wire_version))
    slot_choices: list[tuple[str, str]] = []
    for raw_choice in _expect_list(wire.get("slotChoices", []), "slotChoices"):
        choice = _expect_list(raw_choice, "slotChoices entry")
        if len(choice) != 2:
            raise ValueError("slotChoices entry must contain a slot id and key")
        slot_choices.append(
            (
                _expect_str(choice[0], "slotChoices slot id"),
                _expect_str(choice[1], "slotChoices key"),
            )
        )
    selector = None
    if wire.get("selector") is not None:
        data = _expect_object(wire.get("selector"), "selector")
        if set(data) != {"input", "branches"}:
            raise ValueError("selector must contain exactly input and branches")
        branches = _expect_object(data.get("branches"), "selector.branches")
        if set(branches) != {"false", "true"}:
            raise ValueError("selector.branches must contain exactly false and true")
        selector = SelectorSpec(
            _expect_str(data.get("input"), "selector.input"),
            {
                "false": _expect_str(branches.get("false"), "selector.branches.false"),
                "true": _expect_str(branches.get("true"), "selector.branches.true"),
            },
        )
    replacements = tuple(
        rule_from_wire(_expect_object(rule, "replacements entry"))
        for rule in _expect_list(wire.get("replacements", []), "replacements")
    )
    widget_groups = _widget_groups_from_wire(wire.get("widgetGroups"))
    chunk_safe = None
    chunk_safe_applies = None
    if "chunkSafe" in wire:
        declaration = _expect_object(wire["chunkSafe"], "chunkSafe")
        _reject_unknown_fields(
            declaration, frozenset({"inputs", "outputs", "applies"}), "chunkSafe"
        )
        if "applies" in declaration:
            chunk_safe_applies = _applies_from_wire(declaration["applies"], "chunkSafe.applies")
        chunk_safe = (
            _expect_strings(declaration.get("inputs"), "chunkSafe.inputs"),
            _expect_strings(declaration.get("outputs"), "chunkSafe.outputs"),
        )
    body_fields: dict[str, object] = {}
    if "body" in wire:
        if inputs or outputs or input_families or output_families or combos or slots:
            raise ValueError("a schema carries body or interface rows, not both")
        if output_descriptors is not None:
            raise ValueError("a schema carries body or interface rows, not both")
        body_fields = _body_fields_from_wire(wire, wire_version)
        input_order = []
    return NodeSchema(
        **cast("Any", body_fields),
        node_type=_expect_str(wire.get("nodeType"), "nodeType"),
        version=_expect_int(wire.get("version"), "version"),
        display_name=_expect_str(wire.get("displayName", ""), "displayName"),
        category=_expect_str(wire.get("category", ""), "category"),
        description=_expect_str(wire.get("description", ""), "description"),
        editor_role=(
            _expect_str(wire["editorRole"], "editorRole") if "editorRole" in wire else None
        ),
        additional_inputs=(
            type_expr_from_wire(_expect_object(wire["additionalInputs"], "additionalInputs"))
            if "additionalInputs" in wire
            else None
        ),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        input_families=tuple(input_families),
        output_families=tuple(output_families),
        output_descriptors=output_descriptors,
        combos=tuple(combos),
        slots=tuple(slots),
        input_order=tuple(input_order),
        slot_choices=tuple(slot_choices),
        idempotent=_expect_bool(wire.get("idempotent", True), "idempotent"),
        occupies=_expect_strings(wire.get("occupies", []), "occupies"),
        io_bound=_expect_bool(wire.get("ioBound", False), "ioBound"),
        dispatch_affinity=dispatch_affinity,
        deprecation=_deprecation_from_wire(wire.get("deprecation")),
        search_visibility=cast(
            "SearchVisibility",
            _expect_str(wire.get("searchVisibility", "normal"), "searchVisibility"),
        ),
        search_terms=_expect_strings(wire.get("searchTerms", []), "searchTerms"),
        replacements=replacements,
        output_node=_expect_bool(wire.get("outputNode", False), "outputNode"),
        emits_previews=_expect_bool(wire.get("emitsPreviews", False), "emitsPreviews"),
        may_expand_graph=_expect_bool(wire.get("mayExpandGraph", False), "mayExpandGraph"),
        admission_validation=_expect_bool(
            wire.get("admissionValidation", False), "admissionValidation"
        ),
        execution_context=_expect_bool(wire.get("executionContext", False), "executionContext"),
        input_values=_expect_bool(wire.get("inputValues", False), "inputValues"),
        mirror=_mirror_from_wire(wire.get("mirror"), wire_version),
        selector=selector,
        widget_groups=widget_groups,
        chunk_safe=chunk_safe,
        chunk_safe_applies=chunk_safe_applies,
    )


def _widget_groups_from_wire(raw: Any) -> tuple[ConditionalWidgetGroup, ...]:
    if raw is None:
        return ()
    entries = _expect_list(raw, "widgetGroups")
    if not entries:
        raise ValueError("widgetGroups must be a non-empty array")

    def condition_from_wire(value: Any, where: str) -> ConditionalWidgetCondition:
        data = _expect_object(value, where)
        if set(data) != {"input", "values"}:
            raise ValueError(f"{where} must contain exactly input and values")
        values = _expect_list(data.get("values"), f"{where}.values")
        return ConditionalWidgetCondition(
            input=_expect_str(data.get("input"), f"{where}.input"), values=tuple(values)
        )

    result: list[ConditionalWidgetGroup] = []
    for index, value in enumerate(entries):
        where = f"widgetGroups[{index}]"
        data = _expect_object(value, where)
        _reject_unknown_fields(data, frozenset({"input", "values", "members", "requires"}), where)
        condition = condition_from_wire(
            {"input": data.get("input"), "values": data.get("values")}, where
        )
        members = _expect_strings(data.get("members"), f"{where}.members")
        requires = tuple(
            (
                condition_from_wire(item, f"{where}.requires[{required_index}]")
                for required_index, item in enumerate(
                    _expect_list(data.get("requires", []), f"{where}.requires")
                )
            )
        )
        if "requires" in data and (not requires):
            raise ValueError(f"{where}.requires must be a non-empty array")
        result.append(ConditionalWidgetGroup(condition.input, condition.values, members, requires))
    return tuple(result)


def _mirror_from_wire(wire: Any, wire_version: int) -> MirrorSpec | None:
    if wire is None:
        return None
    data = _expect_object(wire, "mirror")
    _reject_unknown_fields(
        data,
        frozenset({"kind", "precision", "tolerance", "grammarVersion", "source", "applies"}),
        "mirror",
    )
    tolerance = None
    if data.get("tolerance") is not None:
        bounds = _expect_object(data.get("tolerance"), "mirror.tolerance")
        _reject_unknown_fields(bounds, frozenset({"relative", "perChannel"}), "mirror.tolerance")
        tolerance = MirrorTolerance(
            relative=_expect_float(bounds.get("relative"), "mirror.tolerance.relative")
            if bounds.get("relative") is not None
            else None,
            per_channel=_expect_float(bounds.get("perChannel"), "mirror.tolerance.perChannel")
            if bounds.get("perChannel") is not None
            else None,
        )
    applies: dict[str, tuple[str, ...]] | None = None
    if data.get("applies") is not None:
        applies = _applies_from_wire(data.get("applies"), "mirror.applies")
    return MirrorSpec(
        kind=cast("Any", _expect_str(data.get("kind"), "mirror.kind")),
        precision=cast("Any", _expect_str(data.get("precision"), "mirror.precision")),
        tolerance=tolerance,
        grammar_version=_expect_int(data.get("grammarVersion"), "mirror.grammarVersion")
        if data.get("grammarVersion") is not None
        else None,
        source=_expect_str(data.get("source"), "mirror.source")
        if data.get("source") is not None
        else None,
        applies=applies,
    )


def _applies_from_wire(wire: Any, subject: str) -> dict[str, tuple[str, ...]]:
    entries = _expect_object(wire, subject)
    return {
        _expect_str(combo_id, f"{subject} key"): tuple(
            _expect_strings(values, f"{subject}[{combo_id!r}]")
        )
        for combo_id, values in entries.items()
    }


def _deprecation_from_wire(wire: Any) -> Deprecation | None:
    if wire is None:
        return None
    data = _expect_object(wire, "deprecation")
    return Deprecation(
        message=_expect_str(data.get("message"), "deprecation.message"),
        since=_expect_str(data.get("since", ""), "deprecation.since"),
        replacement=_expect_str(data.get("replacement", ""), "deprecation.replacement"),
    )


# --- the node body tree -----------------------------------------------------------
#
# A body row is an object with "kind". A formula is {"formula": text}; a
# lookup is {"lookup": wire name, "args": [...]}; a step value is
# {"value": literal}, a formula or a lookup. Fields at their default are left out.

_STEP_KINDS: Final = {
    "set": body.Set,
    "toggle": body.Toggle,
    "reset": body.Reset,
    "swap": body.Swap,
    "append": body.Append,
    "randomize": body.Randomize,
    "setMode": body.SetMode,
    "addMember": body.AddMember,
    "removeMember": body.RemoveMember,
    "clearMembers": body.ClearMembers,
    "setEach": body.SetEach,
    "disconnect": body.Disconnect,
    "addNode": body.AddNode,
    "pick": body.Pick,
    "ask": body.Ask,
    "openPicker": body.OpenPicker,
    "openEditor": body.OpenEditor,
    "pasteImage": body.PasteImage,
    "call": body.Call,
    "queue": body.Queue,
    "interrupt": body.Interrupt,
    "copy": body.Copy,
    "download": body.Download,
    "openLink": body.OpenLink,
}
_STEP_NAMES: Final = {cls: name for name, cls in _STEP_KINDS.items()}


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


def _formula_to_wire(value: object) -> object:
    """Text or a flag as itself, a formula or a lookup as an object."""
    if isinstance(value, body.Formula):
        return {"formula": value.text}
    if isinstance(value, body.Lookup):
        return _lookup_to_wire(value)
    return value


def _formula_from_wire(value: object, field: str) -> object:
    if isinstance(value, dict):
        data = cast("dict[str, Any]", value)
        if set(data) == {"formula"}:
            return body.Formula(_expect_str(data["formula"], f"{field}.formula"))
        if "lookup" in data:
            return _lookup_from_wire(data, field)
        raise ValueError(f"{field} must be a value, a formula or a lookup")
    return value


def _arguments_to_wire(args: tuple[body.Argument, ...]) -> list[object]:
    return [_formula_to_wire(argument) for argument in args]


def _arguments_from_wire(value: object, field: str) -> tuple[body.Argument, ...]:
    args: list[body.Argument] = []
    for item in _expect_list(value, field):
        argument = _formula_from_wire(item, field)
        if not isinstance(argument, (str, body.Formula)):
            raise ValueError(f"{field} holds field ids and formulas")
        args.append(argument)
    return tuple(args)


def _lookup_to_wire(lookup: body.Lookup) -> dict[str, object]:
    wire: dict[str, object] = {"lookup": lookup.name}
    if lookup.args:
        wire["args"] = _arguments_to_wire(lookup.args)
    return wire


def _lookup_from_wire(data: dict[str, Any], field: str) -> body.Lookup:
    _reject_unknown_fields(data, frozenset({"lookup", "args"}), field)
    return body.Lookup(
        _expect_str(data["lookup"], f"{field}.lookup"),
        _arguments_from_wire(data.get("args", []), f"{field}.args"),
    )


def _table_to_wire(value: object) -> object:
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value).items()
        return {"table": [[key, _table_to_wire(item)] for key, item in items]}
    if isinstance(value, (list, tuple)):
        return [_table_to_wire(item) for item in cast("Sequence[object]", value)]
    return value


def _table_from_wire(value: object) -> object:
    if isinstance(value, dict):
        data = cast("dict[str, Any]", value)
        if set(data) != {"table"}:
            raise ValueError("a table dictionary is {'table': [[key, value], ...]}")
        pairs = _expect_list(data["table"], "table")
        result: dict[object, object] = {}
        for pair in pairs:
            entry = _expect_list(pair, "table entry")
            if (
                len(entry) != 2
                or not isinstance(entry[0], (str, int))
                or isinstance(entry[0], bool)
            ):
                raise ValueError("a table entry is [text or whole-number key, value]")
            result[entry[0]] = _table_from_wire(entry[1])
        return result
    if isinstance(value, list):
        return tuple(_table_from_wire(item) for item in cast("list[object]", value))
    return value


def _step_value_to_wire(value: object) -> object:
    if isinstance(value, (body.Formula, body.Lookup)):
        return _formula_to_wire(value)
    return {"value": _table_to_wire(value)}


def _step_value_from_wire(value: object, field: str) -> object:
    data = _expect_object(value, field)
    if set(data) == {"value"}:
        return _table_from_wire(data["value"])
    return _formula_from_wire(data, field)


def _step_to_wire(step: body.Step, wire_version: int) -> dict[str, object]:
    wire: dict[str, object] = {"step": _STEP_NAMES[type(step)]}
    for item in dataclasses.fields(cast("Any", step)):
        value = getattr(step, item.name)
        if value == item.default or (
            item.default_factory is not dataclasses.MISSING and value == item.default_factory()
        ):
            continue
        key = "options" if item.name == "options" else _camel(item.name)
        if isinstance(step, body.Ask) and item.name == "rows":
            wire[key] = [_input_entry_to_wire(row, wire_version) for row in step.rows]
        elif isinstance(step, body.Call) and item.name == "function":
            wire["function"] = step.name
        elif isinstance(step, body.Call) and item.name == "args":
            wire["args"] = _arguments_to_wire(step.args)
        elif item.name in ("value", "low", "high", "url", "options"):
            wire[key] = _step_value_to_wire(value)
        elif item.name == "values":
            values = cast("Mapping[str, object]", value)
            wire[key] = {name: _step_value_to_wire(item) for name, item in values.items()}
        elif item.name == "which":
            wire[key] = _formula_to_wire(value)
        elif isinstance(value, tuple):
            wire[key] = list(cast("tuple[object, ...]", value))
        else:
            wire[key] = value
    return wire


def _step_from_wire(value: object, wire_version: int) -> body.Step:
    data = _expect_object(value, "step")
    name = _expect_str(data.get("step"), "step.step")
    cls = _STEP_KINDS.get(name)
    if cls is None:
        raise ValueError(f"unknown step {name!r}")
    fields_by_key = {_camel(item.name): item for item in dataclasses.fields(cast("Any", cls))}
    _reject_unknown_fields(data, frozenset({"step", *fields_by_key}), f"step {name}")
    kwargs: dict[str, object] = {}
    for key, raw in data.items():
        if key == "step":
            continue
        field_name = fields_by_key[key].name
        where = f"step {name}.{key}"
        if cls is body.Ask and field_name == "rows":
            kwargs[field_name] = tuple(
                _input_entry_from_wire(_expect_object(row, where), wire_version)
                for row in _expect_list(raw, where)
            )
        elif cls is body.Call and field_name == "function":
            kwargs[field_name] = _expect_str(raw, where)
        elif cls is body.Call and field_name == "args":
            kwargs[field_name] = _arguments_from_wire(raw, where)
        elif field_name in ("value", "low", "high", "url", "options"):
            kwargs[field_name] = _step_value_from_wire(raw, where)
        elif field_name == "values":
            kwargs[field_name] = {
                item_name: _step_value_from_wire(item, where)
                for item_name, item in _expect_object(raw, where).items()
            }
        elif field_name == "which":
            kwargs[field_name] = _formula_from_wire(raw, where)
        elif isinstance(raw, list):
            kwargs[field_name] = _expect_strings(cast("object", raw), where)
        else:
            kwargs[field_name] = raw
    return cls(**cast("Any", kwargs))


def _steps_to_wire(steps: tuple[body.Step, ...], wire_version: int) -> list[object]:
    return [_step_to_wire(step, wire_version) for step in steps]


def _steps_from_wire(value: object, field: str, wire_version: int) -> tuple[body.Step, ...]:
    return tuple(_step_from_wire(item, wire_version) for item in _expect_list(value, field))


def _rule_to_wire(rule: body.Rule, wire_version: int) -> dict[str, object]:
    wire: dict[str, object] = {"check": _formula_to_wire(rule.check), "message": rule.message}
    if rule.fix:
        wire["fix"] = _steps_to_wire(rule.fix, wire_version)
    return wire


def _rule_from_wire(value: object, wire_version: int) -> body.Rule:
    data = _expect_object(value, "rule")
    _reject_unknown_fields(data, frozenset({"check", "message", "fix"}), "rule")
    check = _formula_from_wire(data.get("check"), "rule.check")
    if not isinstance(check, body.Formula):
        raise ValueError("rule.check is a formula")
    return body.Rule(
        check,
        _expect_str(data.get("message"), "rule.message"),
        _steps_from_wire(data.get("fix", []), "rule.fix", wire_version),
    )


def _common_to_wire(row: object, wire: dict[str, object]) -> dict[str, object]:
    """The label, tooltip and visible a box carries, each only when set."""
    for name in ("label", "tooltip"):
        value = getattr(row, name, "")
        if value:
            wire[name] = _formula_to_wire(value)
    visible = getattr(row, "visible", True)
    if visible is not True:
        wire["visible"] = _formula_to_wire(visible)
    return wire


def _common_from_wire(data: dict[str, Any], field: str) -> dict[str, object]:
    common: dict[str, object] = {}
    for name in ("label", "tooltip"):
        if name in data:
            common[name] = _formula_from_wire(data[name], f"{field}.{name}")
    if "visible" in data:
        common["visible"] = _formula_from_wire(data["visible"], f"{field}.visible")
    return common


_COMMON: Final = frozenset({"label", "tooltip", "visible"})


def _keys_to_wire(keys: body.Numbered | body.Named) -> dict[str, object]:
    if isinstance(keys, body.Named):
        return {"named": list(keys.names)}
    return {"numbered": {"prefix": keys.prefix, "start": keys.start, "pad": keys.pad}}


def _keys_from_wire(value: object) -> body.Numbered | body.Named:
    data = _expect_object(value, "group.keys")
    if set(data) == {"named"}:
        return body.Named(_expect_strings(data["named"], "group.keys.named"))
    if set(data) != {"numbered"}:
        raise ValueError("group.keys is {'numbered': {...}} or {'named': [...]}")
    numbered = _expect_object(data["numbered"], "group.keys.numbered")
    _reject_unknown_fields(numbered, frozenset({"prefix", "start", "pad"}), "group.keys.numbered")
    return body.Numbered(
        _expect_str(numbered.get("prefix"), "group.keys.numbered.prefix"),
        _expect_int(numbered.get("start", 1), "group.keys.numbered.start"),
        _expect_int(numbered.get("pad", 0), "group.keys.numbered.pad"),
    )


def _grows_to_wire(grows: body.Grows) -> object:
    if isinstance(grows, body.Count):
        return {"count": grows.formula.text}
    if isinstance(grows, body.FollowList):
        return {"followList": grows.field}
    return grows


def _grows_from_wire(value: object) -> body.Grows:
    if isinstance(value, str):
        if value not in body.GROWS_WORDS:
            raise ValueError(f"unknown group.grows {value!r}")
        return cast("body.Grows", value)
    data = _expect_object(value, "group.grows")
    if set(data) == {"count"}:
        return body.Count(body.Formula(_expect_str(data["count"], "group.grows.count")))
    if set(data) == {"followList"}:
        return body.FollowList(_expect_str(data["followList"], "group.grows.followList"))
    raise ValueError("group.grows is a word, {'count': formula} or {'followList': field}")


def _use_to_wire(use: body.Use, wire_version: int) -> dict[str, object]:
    wire: dict[str, object] = {"kind": "use", "id": use.id}
    if use.label is not body.UNSET:
        wire["label"] = use.label
    if use.tooltip is not body.UNSET:
        wire["tooltip"] = use.tooltip
    if use.widget is not body.UNSET:
        widget = cast("Widget | None", use.widget)
        wire["widget"] = None if widget is None else _widget_to_wire(widget, wire_version)
    if use.default is not body.UNSET:
        wire["default"] = use.default
    if use.required is not body.UNSET:
        wire["required"] = use.required
    if use.only is not None:
        wire["only"] = list(use.only)
    if use.inside:
        wire["inside"] = [_use_to_wire(inner, wire_version) for inner in use.inside]
    return wire


def _use_from_wire(data: dict[str, Any], wire_version: int) -> body.Use:
    allowed = {"kind", "id", "label", "tooltip", "widget", "default", "required", "only", "inside"}
    _reject_unknown_fields(data, frozenset(allowed), "use")
    kwargs: dict[str, object] = {"id": _expect_str(data.get("id"), "use.id")}
    for name in ("label", "tooltip"):
        if name in data:
            kwargs[name] = _expect_str(data[name], f"use.{name}")
    if "widget" in data:
        raw = data["widget"]
        kwargs["widget"] = None if raw is None else _widget_from_wire(raw, wire_version)
    if "default" in data:
        kwargs["default"] = data["default"]
    if "required" in data:
        kwargs["required"] = _expect_bool(data["required"], "use.required")
    if "only" in data:
        kwargs["only"] = _expect_strings(data["only"], "use.only")
    if "inside" in data:
        kwargs["inside"] = tuple(
            _use_from_wire(_expect_object(item, "use.inside"), wire_version)
            for item in _expect_list(data["inside"], "use.inside")
        )
    return body.Use(**cast("Any", kwargs))


def body_row_to_wire(row: body.BodyRow | body.Use, wire_version: int) -> dict[str, object]:
    if isinstance(row, body.Use):
        return _use_to_wire(row, wire_version)
    if isinstance(row, Input):
        return {"kind": "input", **_input_entry_to_wire(row, wire_version)}
    if isinstance(row, Output):
        return {"kind": "output", **_output_entry_to_wire(row, wire_version)}
    if isinstance(row, body.State):
        state: dict[str, object] = {
            "kind": "state",
            "id": row.id,
            "type": type_expr_to_wire(row.type, wire_version),
        }
        if row.default is not None:
            state["default"] = row.default
        if not row.part_of_run:
            state["partOfRun"] = False
        return state
    if isinstance(row, body.Row):
        return {"kind": "row", "rows": _rows_to_wire(row.rows, wire_version)}
    if isinstance(row, body.Section):
        wire = _common_to_wire(row, {"kind": "section", "id": row.id})
        wire["rows"] = _rows_to_wire(row.rows, wire_version)
        if row.collapsed:
            wire["collapsed"] = True
        if not row.open_name:
            wire["openName"] = False
        return wire
    if isinstance(row, body.Note):
        wire = {"kind": "note", "id": row.id, "text": _formula_to_wire(row.text)}
        if row.unknown:
            wire["unknown"] = row.unknown
        return _common_to_wire(row, wire)
    if isinstance(row, body.Button):
        wire = _common_to_wire(row, {"kind": "button", "id": row.id, "label": row.label})
        wire["steps"] = _steps_to_wire(row.steps, wire_version)
        if row.enabled is not True:
            wire["enabled"] = _formula_to_wire(row.enabled)
        if row.disabled_reason:
            wire["disabledReason"] = _formula_to_wire(row.disabled_reason)
        if row.icon is not None:
            wire["icon"] = row.icon
        return wire
    if isinstance(row, body.Group):
        assert row.keys is not None
        wire = _common_to_wire(row, {"kind": "group", "id": row.id})
        if row.value is not None:
            wire["value"] = body_row_to_wire(row.value, wire_version)
        else:
            assert row.member is not None
            wire["member"] = _rows_to_wire(row.member, wire_version)
        wire["keys"] = _keys_to_wire(row.keys)
        wire["grows"] = _grows_to_wire(row.grows)
        if row.min_members:
            wire["minMembers"] = row.min_members
        if row.max_members is not None:
            wire["maxMembers"] = row.max_members
        for flag in ("independent_types", "optional_after_minimum", "reorderable"):
            if getattr(row, flag):
                wire[_camel(flag)] = True
        if row.add_label:
            wire["addLabel"] = row.add_label
        if row.member_label:
            wire["memberLabel"] = _formula_to_wire(row.member_label)
        if not row.open_name:
            wire["openName"] = False
        return wire
    if isinstance(row, body.Choice):
        wire = _common_to_wire(row, {"kind": "choice", "id": row.id})
        wire["options"] = [
            {
                "key": option.key,
                "rows": _rows_to_wire(option.rows, wire_version),
                **({"label": option.label} if option.label else {}),
            }
            for option in row.options
        ]
        if row.shared:
            wire["shared"] = _rows_to_wire(row.shared, wire_version)
        if row.leader is not None:
            wire["leader"] = row.leader.text
        if row.default is not None:
            wire["default"] = row.default
        if not row.required:
            wire["required"] = False
        if row.hidden:
            wire["hidden"] = True
        if row.on_edit:
            wire["onEdit"] = _steps_to_wire(row.on_edit, wire_version)
        return wire
    if isinstance(row, body.TypeChoice):
        wire = _common_to_wire(row, {"kind": "typeChoice", "id": row.id})
        if row.variants is not None:
            wire["variants"] = [
                {
                    "key": variant.key,
                    "type": type_expr_to_wire(variant.type, wire_version),
                    "rows": _rows_to_wire(variant.rows, wire_version),
                    **({"tooltip": variant.tooltip} if variant.tooltip else {}),
                }
                for variant in row.variants
            ]
        else:
            assert row.type is not None
            wire["type"] = type_expr_to_wire(row.type, wire_version)
        if row.rows:
            wire["rows"] = _rows_to_wire(row.rows, wire_version)
        wire["required"] = bool(row.required)
        if row.type_variable:
            wire["typeVariable"] = row.type_variable
        return wire
    assert isinstance(row, body.Described)
    wire = _common_to_wire(row, {"kind": "described", "id": row.id})
    wire["kinds"] = _rows_to_wire(row.kinds, wire_version)
    wire["maxEntries"] = row.max_entries
    if row.min_entries:
        wire["minEntries"] = row.min_entries
    if row.fixed_ids:
        wire["fixedIds"] = True
    if row.probe is not None:
        wire["probe"] = {
            "input": row.probe.input,
            "kind": row.probe.kind,
            "revision": row.probe.revision,
        }
    return wire


def _rows_to_wire(
    rows: Sequence[body.BodyRow | body.Use], wire_version: int
) -> list[dict[str, object]]:
    return [body_row_to_wire(row, wire_version) for row in rows]


def _rows_from_wire(value: object, field: str, wire_version: int) -> tuple[Any, ...]:
    return tuple(
        body_row_from_wire(_expect_object(item, field), wire_version)
        for item in _expect_list(value, field)
    )


def body_row_from_wire(data: dict[str, Any], wire_version: int) -> body.BodyRow | body.Use:
    kind = _expect_str(data.get("kind"), "row.kind")
    rest = {key: value for key, value in data.items() if key != "kind"}
    if kind == "use":
        return _use_from_wire(data, wire_version)
    if kind == "input":
        return _input_entry_from_wire(rest, wire_version)
    if kind == "output":
        return _output_entry_from_wire(rest, wire_version)
    if kind == "state":
        _reject_unknown_fields(rest, frozenset({"id", "type", "default", "partOfRun"}), "state")
        return body.State(
            _expect_str(rest.get("id"), "state.id"),
            type_expr_from_wire(_expect_object(rest.get("type"), "state.type"), wire_version),
            rest.get("default"),
            _expect_bool(rest.get("partOfRun", True), "state.partOfRun"),
        )
    if kind == "row":
        _reject_unknown_fields(rest, frozenset({"rows"}), "row")
        return body.Row(_rows_from_wire(rest.get("rows"), "row.rows", wire_version))
    if kind == "section":
        _reject_unknown_fields(rest, _COMMON | {"id", "rows", "collapsed", "openName"}, "section")
        return body.Section(
            _expect_str(rest.get("id"), "section.id"),
            _rows_from_wire(rest.get("rows"), "section.rows", wire_version),
            collapsed=_expect_bool(rest.get("collapsed", False), "section.collapsed"),
            open_name=_expect_bool(rest.get("openName", True), "section.openName"),
            **cast("Any", _common_from_wire(rest, "section")),
        )
    if kind == "note":
        _reject_unknown_fields(rest, _COMMON | {"id", "text", "unknown"}, "note")
        common = _common_from_wire(rest, "note")
        return body.Note(
            _expect_str(rest.get("id"), "note.id"),
            cast("Any", _formula_from_wire(rest.get("text"), "note.text")),
            _expect_str(rest.get("unknown", ""), "note.unknown"),
            **cast("Any", {key: common[key] for key in ("visible",) if key in common}),
        )
    if kind == "button":
        _reject_unknown_fields(
            rest, _COMMON | {"id", "steps", "enabled", "disabledReason", "icon"}, "button"
        )
        common = _common_from_wire(rest, "button")
        return body.Button(
            _expect_str(rest.get("id"), "button.id"),
            _expect_str(common.pop("label", None), "button.label"),
            _steps_from_wire(rest.get("steps"), "button.steps", wire_version),
            enabled=cast("Any", _formula_from_wire(rest.get("enabled", True), "button.enabled")),
            disabled_reason=cast(
                "Any", _formula_from_wire(rest.get("disabledReason", ""), "button.disabledReason")
            ),
            icon=None if "icon" not in rest else _expect_str(rest["icon"], "button.icon"),
            **cast("Any", common),
        )
    if kind == "group":
        allowed = _COMMON | {
            "id",
            "member",
            "value",
            "keys",
            "grows",
            "minMembers",
            "maxMembers",
            "independentTypes",
            "optionalAfterMinimum",
            "reorderable",
            "addLabel",
            "memberLabel",
            "openName",
        }
        _reject_unknown_fields(rest, frozenset(allowed), "group")
        value = None
        if "value" in rest:
            value = body_row_from_wire(_expect_object(rest["value"], "group.value"), wire_version)
        return body.Group(
            _expect_str(rest.get("id"), "group.id"),
            member=None
            if "member" not in rest
            else _rows_from_wire(rest["member"], "group.member", wire_version),
            value=cast("Any", value),
            keys=_keys_from_wire(rest.get("keys")),
            grows=_grows_from_wire(rest.get("grows", "link")),
            min_members=_expect_int(rest.get("minMembers", 0), "group.minMembers"),
            max_members=None
            if "maxMembers" not in rest
            else _expect_int(rest["maxMembers"], "group.maxMembers"),
            independent_types=_expect_bool(
                rest.get("independentTypes", False), "group.independentTypes"
            ),
            optional_after_minimum=_expect_bool(
                rest.get("optionalAfterMinimum", False), "group.optionalAfterMinimum"
            ),
            reorderable=_expect_bool(rest.get("reorderable", False), "group.reorderable"),
            add_label=_expect_str(rest.get("addLabel", ""), "group.addLabel"),
            member_label=cast(
                "Any", _formula_from_wire(rest.get("memberLabel", ""), "group.memberLabel")
            ),
            open_name=_expect_bool(rest.get("openName", True), "group.openName"),
            **cast("Any", _common_from_wire(rest, "group")),
        )
    if kind == "choice":
        allowed = _COMMON | {
            "id",
            "options",
            "shared",
            "leader",
            "default",
            "required",
            "hidden",
            "onEdit",
        }
        _reject_unknown_fields(rest, frozenset(allowed), "choice")
        options: list[body.Option] = []
        for raw in _expect_list(rest.get("options"), "choice.options"):
            option = _expect_object(raw, "choice.options")
            _reject_unknown_fields(option, frozenset({"key", "rows", "label"}), "choice.options")
            options.append(
                body.Option(
                    _expect_str(option.get("key"), "choice.options.key"),
                    _rows_from_wire(option.get("rows", []), "choice.options.rows", wire_version),
                    _expect_str(option.get("label", ""), "choice.options.label"),
                )
            )
        return body.Choice(
            _expect_str(rest.get("id"), "choice.id"),
            tuple(options),
            shared=_rows_from_wire(rest.get("shared", []), "choice.shared", wire_version),
            leader=None
            if "leader" not in rest
            else body.Formula(_expect_str(rest["leader"], "choice.leader")),
            default=None
            if "default" not in rest
            else _expect_str(rest["default"], "choice.default"),
            required=_expect_bool(rest.get("required", True), "choice.required"),
            hidden=_expect_bool(rest.get("hidden", False), "choice.hidden"),
            on_edit=_steps_from_wire(rest.get("onEdit", []), "choice.onEdit", wire_version),
            **cast("Any", _common_from_wire(rest, "choice")),
        )
    if kind == "typeChoice":
        allowed = _COMMON | {"id", "variants", "type", "rows", "required", "typeVariable"}
        _reject_unknown_fields(rest, frozenset(allowed), "typeChoice")
        variants: list[body.Variant] | None = None
        if "variants" in rest:
            variants = []
            for raw in _expect_list(rest["variants"], "typeChoice.variants"):
                variant = _expect_object(raw, "typeChoice.variants")
                _reject_unknown_fields(
                    variant, frozenset({"key", "type", "rows", "tooltip"}), "typeChoice.variants"
                )
                variants.append(
                    body.Variant(
                        _expect_str(variant.get("key"), "typeChoice.variants.key"),
                        type_expr_from_wire(
                            _expect_object(variant.get("type"), "typeChoice.variants.type"),
                            wire_version,
                        ),
                        _rows_from_wire(
                            variant.get("rows", []), "typeChoice.variants.rows", wire_version
                        ),
                        _expect_str(variant.get("tooltip", ""), "typeChoice.variants.tooltip"),
                    )
                )
        return body.TypeChoice(
            _expect_str(rest.get("id"), "typeChoice.id"),
            variants=None if variants is None else tuple(variants),
            type=None
            if "type" not in rest
            else type_expr_from_wire(_expect_object(rest["type"], "typeChoice.type"), wire_version),
            rows=_rows_from_wire(rest.get("rows", []), "typeChoice.rows", wire_version),
            required=_expect_bool(rest.get("required"), "typeChoice.required"),
            type_variable=_expect_str(rest.get("typeVariable", ""), "typeChoice.typeVariable"),
            **cast("Any", _common_from_wire(rest, "typeChoice")),
        )
    if kind == "described":
        allowed = _COMMON | {"id", "kinds", "maxEntries", "minEntries", "fixedIds", "probe"}
        _reject_unknown_fields(rest, frozenset(allowed), "described")
        probe = None
        if "probe" in rest:
            data_probe = _expect_object(rest["probe"], "described.probe")
            _reject_unknown_fields(
                data_probe, frozenset({"input", "kind", "revision"}), "described.probe"
            )
            probe = OutputProbeSpec(
                _expect_str(data_probe.get("input"), "described.probe.input"),
                _expect_str(data_probe.get("kind"), "described.probe.kind"),
                _expect_str(data_probe.get("revision"), "described.probe.revision"),
            )
        return body.Described(
            _expect_str(rest.get("id"), "described.id"),
            _rows_from_wire(rest.get("kinds"), "described.kinds", wire_version),
            _expect_int(rest.get("maxEntries"), "described.maxEntries"),
            min_entries=_expect_int(rest.get("minEntries", 0), "described.minEntries"),
            fixed_ids=_expect_bool(rest.get("fixedIds", False), "described.fixedIds"),
            probe=probe,
            **cast("Any", _common_from_wire(rest, "described")),
        )
    raise ValueError(f"unknown row kind {kind!r}")


def _body_fields_to_wire(schema: NodeSchema, wire: dict[str, object], wire_version: int) -> None:
    wire["body"] = _rows_to_wire(schema.body, wire_version)
    if schema.tables:
        wire["tables"] = {name: _table_to_wire(value) for name, value in schema.tables.items()}
    if schema.derived:
        wire["derived"] = {name: formula.text for name, formula in schema.derived.items()}
    if schema.validate:
        wire["validate"] = [_rule_to_wire(rule, wire_version) for rule in schema.validate]
    if schema.title_suffix:
        wire["titleSuffix"] = _formula_to_wire(schema.title_suffix)
    if schema.menu:
        wire["menu"] = _rows_to_wire(schema.menu, wire_version)


def _body_fields_from_wire(wire: dict[str, Any], wire_version: int) -> dict[str, object]:
    tables = {
        name: _table_from_wire(value)
        for name, value in _expect_object(wire.get("tables", {}), "tables").items()
    }
    derived = {
        name: body.Formula(_expect_str(text, f"derived.{name}"))
        for name, text in _expect_object(wire.get("derived", {}), "derived").items()
    }
    menu = _rows_from_wire(wire.get("menu", []), "menu", wire_version)
    if any(not isinstance(item, body.Button) for item in menu):
        raise ValueError("menu holds buttons")
    return {
        "body": _rows_from_wire(wire.get("body"), "body", wire_version),
        "tables": tables,
        "derived": derived,
        "validate": tuple(
            _rule_from_wire(rule, wire_version)
            for rule in _expect_list(wire.get("validate", []), "validate")
        ),
        "title_suffix": _formula_from_wire(wire.get("titleSuffix", ""), "titleSuffix"),
        "menu": menu,
    }
