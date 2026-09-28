"""Wire encoding of node schemas: the /object_info successor.

Shape is designed to be near-isomorphic to Dinkster-Frontend's normalized
NodeSchema model: one ordered interface list, real output ids, structured
type expressions, and an explicit schemaVersion field.

"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal, cast

from dinkster_values import CustomWidgetDescriptor, JsonValue

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
    InputFamilyOptionSource,
    InputFamilySpec,
    InputSpec,
    MirrorSpec,
    MirrorTolerance,
    MultiComboWidget,
    NodeSchema,
    NumberWidget,
    OutputCountSpec,
    OutputDescriptorsSpec,
    OutputFamilySpec,
    OutputKnownValue,
    OutputProbeSpec,
    OutputRepresents,
    OutputSpec,
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

SCHEMA_WIRE_VERSION = 1
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


def _combo_option_to_wire(option: str | ComboOption, wire_version: int) -> object:
    if isinstance(option, str):
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
        return asset_wire
    if isinstance(widget, ComboWidget):
        combo_wire: dict[str, object] = {"type": "COMBO"}
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


def _combo_option_from_wire(value: object, wire_version: int, *, subject: str) -> str | ComboOption:
    if type(value) is str and value:
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{subject} must be a non-empty string" + " or structured choice")
    option = cast("dict[str, object]", value)
    _reject_unknown_fields(option, frozenset({"value", "label", "info", "folder"}), subject)
    raw_value = option.get("value")
    if type(raw_value) is not str or not raw_value:
        raise ValueError(f"{subject}.value must be a non-empty string")
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
        )
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
            options=decoded_options,
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
        if "round" in widget_data and type(round_value) not in (int, float):
            raise ValueError("NUMBER widget round must be a number")
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
            round=cast("int | float | None", round_value),
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
    return wire


def _media_policy_to_wire(spec: InputSpec | OutputSpec, wire_version: int) -> dict[str, object]:
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


def _input_entry_to_wire(spec: InputSpec, wire_version: int) -> dict[str, object]:
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
    if spec.doc:
        entry["doc"] = spec.doc
    if spec.display_name:
        entry["displayName"] = spec.display_name
    if spec.force_input:
        entry["forceInput"] = True
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
    if spec.source_filename is not None:
        entry["sourceFilename"] = {
            "kind": spec.source_filename.kind,
            "category": spec.source_filename.category,
        }
    return entry


def _dynamic_entry_to_wire(
    spec: DynamicEntry, wire_version: int = SCHEMA_WIRE_VERSION
) -> dict[str, object]:
    if isinstance(spec, InputSpec):
        return {"role": "input", **_input_entry_to_wire(spec, wire_version)}
    if isinstance(spec, InputFamilySpec):
        entry: dict[str, object] = {
            "role": "inputFamily",
            "id": spec.id,
            "template": [_dynamic_entry_to_wire(item, wire_version) for item in spec.template],
            "minMembers": spec.min_members,
            "required": spec.required,
        }
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


def schema_to_wire(
    schema: NodeSchema,
    *,
    wire_version: int = SCHEMA_WIRE_VERSION,
    replacement_schemas: Mapping[str, NodeSchema] | None = None,
) -> dict[str, object]:
    if type(wire_version) is not int or wire_version != SCHEMA_WIRE_VERSION:
        raise ValueError(f"unsupported schemaVersion: {wire_version!r}")
    interface: list[dict[str, object]] = []
    for spec in schema.inputs:
        try:
            interface.append(_dynamic_entry_to_wire(spec, wire_version))
        except ValueError as exc:
            raise ValueError(f"node {schema.node_type!r}, {exc}") from exc
    for fam in schema.input_families:
        interface.append(_dynamic_entry_to_wire(fam, wire_version))
    for combo in schema.combos:
        interface.append(_dynamic_entry_to_wire(combo, wire_version))
    for slot in schema.slots:
        interface.append(_dynamic_entry_to_wire(slot, wire_version))
    for out in schema.outputs:
        out_entry: dict[str, object] = {
            "role": "output",
            "id": out.id,
            "type": type_expr_to_wire(out.type, wire_version),
        }
        out_entry.update(_media_policy_to_wire(out, wire_version))
        if out.optional:
            out_entry["optional"] = True
        if out.doc:
            out_entry["doc"] = out.doc
        if out.display_name:
            out_entry["displayName"] = out.display_name
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
        interface.append(out_entry)
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
                    **({"displayName": choice.display_name} if choice.display_name else {}),
                    **({"doc": choice.doc} if choice.doc else {}),
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
    if schema.editor_role is not None:
        wire["editorRole"] = schema.editor_role
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
    if schema.aliases:
        wire["aliases"] = list(schema.aliases)
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
    found, never what this node computes. Resolution/targeting
    metadata (aliases, outputNode) is excluded too: what names resolve to a
    node and whether submission formats target it by default never change
    what it computes. Capability metadata (emitsPreviews, mayExpandGraph) is
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
        if isinstance(entry, InputSpec):
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

    signature_schema = replace(
        schema,
        inputs=tuple(
            cast("InputSpec", strip_editor_widget(input_spec)) for input_spec in schema.inputs
        ),
        input_families=tuple(
            cast("InputFamilySpec", strip_editor_widget(family)) for family in schema.input_families
        ),
        combos=tuple(
            cast("DynamicComboSpec", strip_editor_widget(combo)) for combo in schema.combos
        ),
        slots=tuple(cast("DynamicSlotSpec", strip_editor_widget(slot)) for slot in schema.slots),
        replacements=(),
        outputs=tuple(replace(output, known_value=None) for output in schema.outputs),
    )
    wire = schema_to_wire(signature_schema)
    wire.pop("occupies", None)
    wire.pop("ioBound", None)
    wire.pop("deprecation", None)
    wire.pop("searchVisibility", None)
    wire.pop("searchTerms", None)
    wire.pop("dispatchAffinity", None)
    wire.pop("replacements", None)
    wire.pop("aliases", None)
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
        widget = entry.pop("widget", None)
        dynamic_identity = dynamic_prompts_identity(widget)
        if dynamic_identity is not None:
            entry["dynamicPrompts"] = dynamic_identity
        entry.pop("displayName", None)
        entry.pop("forceInput", None)
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
    canonical = json.dumps(wire, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=20).hexdigest()


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
    return TypeExpr(kind=cast("Any", kind), types=types, template_id=template_id, element=element)


def _input_entry_from_wire(entry: dict[str, Any], wire_version: int) -> InputSpec:
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
    return InputSpec(
        id=_expect_str(entry.get("id"), "input.id"),
        type=type_expr_from_wire(_expect_object(entry.get("type"), "input.type"), wire_version),
        required=_expect_bool(entry.get("required", True), "input.required"),
        default=entry.get("default"),
        doc=_expect_str(entry.get("doc", ""), "input.doc"),
        on_absent=None
        if on_absent is None
        else cast("AbsentPolicy", _expect_str(on_absent, "input.onAbsent")),
        widget=widget,
        display_name=_expect_str(entry.get("displayName", ""), "input.displayName"),
        force_input=_expect_bool(entry.get("forceInput", False), "input.forceInput"),
        advanced=_expect_bool(entry.get("advanced", False), "input.advanced"),
        hidden=_expect_bool(entry.get("hidden", False), "input.hidden"),
        lazy=_expect_bool(entry.get("lazy", False), "input.lazy"),
        source_filename=source_filename,
        accepts_storage=_expect_bool(entry.get("acceptsStorage", False), "input.acceptsStorage"),
        accepts_stream=_expect_bool(entry.get("acceptsStream", False), "input.acceptsStream"),
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
    choices: list[OutputSpec] = []
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
            OutputSpec(
                id=_expect_str(choice.get("id"), "output descriptor choice id"),
                type=type_expr_from_wire(
                    _expect_object(choice.get("type"), "choice type"), wire_version
                ),
                display_name=_expect_str(choice.get("displayName", ""), "choice displayName"),
                doc=_expect_str(choice.get("doc", ""), "choice doc"),
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
    inputs: list[InputSpec] = []
    outputs: list[OutputSpec] = []
    input_families: list[InputFamilySpec] = []
    output_families: list[OutputFamilySpec] = []
    output_descriptors = None
    combos: list[DynamicComboSpec] = []
    slots: list[DynamicSlotSpec] = []
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
            if isinstance(dynamic, InputSpec):
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
            outputs.append(
                OutputSpec(
                    id=_expect_str(entry.get("id"), "output.id"),
                    type=expr,
                    doc=_expect_str(entry.get("doc", ""), "output.doc"),
                    optional=_expect_bool(entry.get("optional", False), "output.optional"),
                    preview=preview,
                    represents=represents,
                    known_value=known_value,
                    display_name=_expect_str(entry.get("displayName", ""), "output.displayName"),
                    **_media_policy_from_wire(entry, wire_version),
                )
            )
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
    return NodeSchema(
        node_type=_expect_str(wire.get("nodeType"), "nodeType"),
        version=_expect_int(wire.get("version"), "version"),
        display_name=_expect_str(wire.get("displayName", ""), "displayName"),
        category=_expect_str(wire.get("category", ""), "category"),
        description=_expect_str(wire.get("description", ""), "description"),
        editor_role=(
            _expect_str(wire["editorRole"], "editorRole") if "editorRole" in wire else None
        ),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        input_families=tuple(input_families),
        output_families=tuple(output_families),
        output_descriptors=output_descriptors,
        combos=tuple(combos),
        slots=tuple(slots),
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
        aliases=_expect_strings(wire.get("aliases", []), "aliases"),
        output_node=_expect_bool(wire.get("outputNode", False), "outputNode"),
        emits_previews=_expect_bool(wire.get("emitsPreviews", False), "emitsPreviews"),
        may_expand_graph=_expect_bool(wire.get("mayExpandGraph", False), "mayExpandGraph"),
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
