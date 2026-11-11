"""Compare an audit ComfyUI core dump with the served /api/nodes registry.

Derived from the node-interface comparison in comfy-vibe-station issue 504.
Classes match schema-declared replacement sources for packs selected in the scope
data, and their original ids otherwise. Aliases and labels never match an id
or a value. The only additional input is an appended optional save mount.
Source-marked file combos and strings may use an ASSET socket and asset picker.
Scalar STRING files with an empty default use an optional ASSET with no default
file to represent no paint; multiline text metadata does not apply to ASSET files.
Marked directory, pair and literal choices require their exact selection kind;
ordinary combo types, options and defaults remain exact.
V3 inputs follow the recorded declaration sequence, including optional inputs
between required inputs. V1 inputs retain required-then-optional group order.
Hidden required/optional declarations remain part of the input contract.
Operational hidden-group sentinels and unmatched native hidden inputs are
excluded. Output names count only when the reference declares RETURN_NAMES
(or explicit V3 output labels); dumps must leave return_names null when names
are synthesized.
Unnamed outputs compare by type and position. A reference wildcard
socket matches an unrestricted native type-template socket, not a concrete
type. List flags never wrap the reference declaration. On a list-flagged
port only, the native type may be the declared type or one list wrapper
around it. Unflagged and nested-list differences remain differences.
Declared FLOAT bounds, step and round compare exactly after projecting the
reference numbers to float; INT and field presence remain exact. Boolean
round sentinels are not numeric precision values.
API nodes and nodes originating in nodes_train.py or nodes_dataset.py are
excluded from required coverage. Training-origin nodes must also be absent
from the served registry; their exclusion does not hide pending deletions.

Usage: python tools/compare_nodes.py COMFY_JSON REGISTRY_JSON OUT_CSV OUT_JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

FORBIDDEN_PREFIXES = ("dinkster.", "comfy.")
SCOPE_RULES = Path(__file__).with_name("node_comparison_scope.json")
INPUT_ROLES = {"input", "inputFamily", "dynamicCombo", "dynamicSlot"}
VALUE_OPTIONS = (
    "default",
    "min",
    "max",
    "step",
    "round",
    "multiline",
    "forceInput",
    "control_after_generate",
    "multiselect",
    "conditional_options",
    "environment_options",
    "baseline_options",
    "possible_options",
)


def _type(expr: dict[str, Any]) -> Any:
    kind = expr.get("kind")
    if kind == "concrete":
        values = expr.get("types", [])
        return values[0] if len(values) == 1 else {"concrete": values}
    if kind == "union":
        return ",".join(expr["types"])
    if kind == "wildcard":
        return "*"
    if kind == "variable":
        return {
            "template_id": expr.get("templateId"),
            "allowed_types": ",".join(expr.get("allowed") or ["*"]),
        }
    if kind in ("list", "asset", "stream"):
        return {kind: _type(expr["element"])}
    return {"unknown_type": expr}


def _reference_inputs(groups: dict[str, Any], list_inputs: bool = False) -> list[dict[str, Any]]:
    result = []
    for group in ("required", "optional"):
        parameters = groups.get(group, {})
        if not isinstance(parameters, dict):
            raise ValueError(f"{group}: input group must be an object")
        for name, spec in parameters.items():
            value_type = spec[0] if isinstance(spec, list) else spec
            options = spec[1] if isinstance(spec, list) and len(spec) > 1 else {}
            if not isinstance(options, dict):
                raise ValueError(f"{name}: input options must be an object")
            entry: dict[str, Any] = {"name": name, "group": group, "type": value_type}
            if options.get("hidden"):
                entry["hidden"] = True
            if value_type == "STRING" and "socketless" in options:
                entry["socketless"] = options["socketless"]
            if isinstance(value_type, list):
                entry.update(type="COMBO", options=value_type)
            elif value_type == "COMBO":
                entry["options"] = options.get("options", [])
            elif value_type == "COMFY_MATCHTYPE_V3":
                entry["type"] = options["template"]
            elif value_type == "COMFY_DYNAMICCOMBO_V3":
                entry["type"] = "dynamicCombo"
                entry["options"] = [
                    {
                        "key": option["key"],
                        "inputs": _reference_inputs(option["inputs"], list_inputs),
                    }
                    for option in options["options"]
                ]
            elif value_type == "COMFY_AUTOGROW_V3":
                template = options["template"]
                entry["type"] = "inputFamily"
                entry["template"] = _reference_inputs(template["input"], list_inputs)
                entry["min"] = template.get("min", 1)
                if entry["min"] == 0:
                    for member in entry["template"]:
                        member["group"] = "optional"
                elif any(member["group"] == "required" for member in entry["template"]):
                    entry["optional_after_minimum"] = True
                if "names" in template:
                    entry["names"] = template["names"]
                else:
                    entry["prefix"] = template["prefix"]
                    entry["max"] = template["max"]
            elif value_type == "COMFY_DYNAMICGROUP_V3":
                entry.update(
                    type="inputFamily",
                    template=_reference_inputs(options["template"], list_inputs),
                    min=options["min"],
                    max=options["max"],
                    prefix=None,
                )
            elif str(value_type).startswith("COMFY_"):
                # Unknown dynamic forms must fail rather than lose their child inputs.
                entry["dynamic"] = options
            for key in VALUE_OPTIONS:
                if key in options:
                    entry[key] = options[key]
            if entry["type"] in ("FLOAT", {"list": "FLOAT"}):
                for key in ("min", "max", "step", "round"):
                    value = entry.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        entry[key] = float(value)
            if entry.get("control_after_generate") is True:
                entry["control_after_generate"] = "fixed"
            if entry.get("multiselect") is False:
                entry.pop("multiselect")
            if options.get("file_input"):
                entry["file_input"] = True
            if "asset_selection_kind" in options:
                entry["asset_selection_kind"] = options["asset_selection_kind"]
            if "options" in entry and entry["type"] == "COMBO" and entry["options"]:
                entry.setdefault("default", entry["options"][0])
            if entry["type"] == "inputFamily" and not entry["template"]:
                continue
            if list_inputs and entry["type"] not in ("inputFamily", "dynamicCombo"):
                entry["_list_flag"] = True
            result.append(entry)
    return result


def _native_inputs(
    items: list[dict[str, Any]], public_inputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    public_by_name = {entry["name"]: entry for entry in public_inputs}
    for item in items:
        role = item["role"]
        public = public_by_name.get(item["id"], {})
        if role not in INPUT_ROLES or (item.get("hidden") and not public):
            continue
        entry: dict[str, Any] = {
            "name": item["id"],
            "group": "required" if item.get("required", True) else "optional",
        }
        if item.get("hidden"):
            entry["hidden"] = True
        if "socketless" in public:
            entry["socketless"] = item.get("socketless", False)
        if role == "input":
            entry["type"] = _type(item["type"])
            widget = item.get("widget") or {}
            if widget.get("type") == "ASSET" and widget.get("selection") is True:
                entry["selection_picker"] = widget.get("kind")
            for key in VALUE_OPTIONS:
                wire_key = "controlAfterGenerate" if key == "control_after_generate" else key
                if wire_key in widget:
                    value = widget[wire_key]
                    # Schema wire encodes integers outside JS's safe range as decimal strings.
                    if key in ("min", "max", "step") and isinstance(value, str):
                        value = int(value)
                    entry[key] = value
                if wire_key in item:
                    entry[key] = item[wire_key]
            if "default" in item:
                entry["default"] = item["default"]
            if entry["type"] in ("COMBO", {"list": "COMBO"}):
                if isinstance(widget.get("remote"), dict) and widget["remote"].get("route"):
                    entry["remote_route"] = widget["remote"]["route"]
                if "options" in widget:
                    entry["options"] = [
                        option["value"] if isinstance(option, dict) else option
                        for option in widget["options"]
                    ]
                elif (
                    widget.get("type") == "COMBO"
                    and widget.get("editableOptions") is True
                    and "remote_route" not in entry
                ):
                    entry["options"] = []
                elif widget.get("type") == "ASSET":
                    entry["file_picker"] = True
                if entry.get("options"):
                    entry.setdefault("default", entry["options"][0])
            elif entry["type"] in ("ASSET", {"list": "ASSET"}) and widget.get("type") == "ASSET":
                entry["file_picker"] = True
            if entry.get("multiselect") is False:
                entry.pop("multiselect")
        elif role == "dynamicCombo":
            public_options = (
                {option["key"]: option["inputs"] for option in public.get("options", [])}
                if public.get("type") == role
                else {}
            )
            entry.update(
                type=role,
                options=[
                    {
                        "key": option["key"],
                        "inputs": _native_inputs(
                            option["inputs"], public_options.get(option["key"], [])
                        ),
                    }
                    for option in item["options"]
                ],
            )
            if "default" in item:
                entry["default"] = item["default"]
        elif role == "inputFamily":
            entry.update(
                type=role,
                template=_native_inputs(item["template"], public.get("template", [])),
                min=item.get("minMembers", 0),
            )
            if item.get("optionalAfterMinimum") and entry["min"] == 0:
                for member in entry["template"]:
                    member["group"] = "optional"
            if (
                item.get("optionalAfterMinimum")
                and entry["min"] > 0
                and any(member["group"] == "required" for member in entry["template"])
            ):
                entry["optional_after_minimum"] = True
            if "memberNames" in item:
                entry["names"] = item["memberNames"]
            else:
                entry["prefix"] = item.get("memberPrefix")
                entry["max"] = item.get("maxMembers")
        else:
            entry.update(type=role, dynamic=item)
        if entry["type"] == "inputFamily" and not entry["template"]:
            continue
        result.append(entry)
    return result


def _differences(expected: Any, actual: Any, path: str) -> list[str]:
    if (
        path.endswith(".type")
        and expected == "*"
        and isinstance(actual, dict)
        and set(actual) == {"template_id", "allowed_types"}
        and isinstance(actual["template_id"], str)
        and actual["template_id"]
        and actual["allowed_types"] == "*"
    ):
        return []
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected, actual = dict(expected), dict(actual)
        if any(
            key in expected
            for key in (
                "conditional_options",
                "environment_options",
                "baseline_options",
                "possible_options",
            )
        ):
            return [f"{path}: invalid conditional option declaration"]
        actual.pop("remote_route", None)
        list_flag = expected.pop("_list_flag", False)
        native_type = actual.get("type")
        if (
            list_flag
            and isinstance(native_type, dict)
            and set(native_type) == {"list"}
            and not (isinstance(native_type["list"], dict) and "list" in native_type["list"])
            and not (isinstance(expected.get("type"), dict) and "list" in expected["type"])
        ):
            actual["type"] = native_type["list"]
        file_input = expected.pop("file_input", False)
        selection_kind = expected.pop("asset_selection_kind", None)
        selection_picker = actual.pop("selection_picker", None)
        file_picker = actual.pop("file_picker", False)
        if selection_kind is not None or selection_picker is not None:
            if (
                not isinstance(selection_kind, str)
                or selection_kind
                not in {"model/vae", "model/diffusers", "directory/input", "directory/dataset"}
                or selection_picker != selection_kind
                or expected.get("type") != "COMBO"
            ):
                return [f"{path}: asset selection requires matching source-marked purpose kinds"]
            native_type = actual.get("type")
            if native_type == "COMBO" or (
                selection_kind == "model/vae"
                and isinstance(native_type, str)
                and set(native_type.split(",")) == {"COMBO", "ASSET"}
                and len(native_type.split(",")) == 2
            ):
                actual["type"] = "COMBO"
            else:
                return [f"{path}: asset selection has an unsupported socket type"]
            expected.pop("options", None)
            expected.pop("default", None)
            actual.pop("default", None)
        elif file_picker:
            if not file_input or expected.get("type") not in (
                "COMBO",
                "STRING",
                {"list": "COMBO"},
                {"list": "STRING"},
            ):
                return [f"{path}: asset picker on a non-file input"]
            # Mount contents vary, but the widget still consumes the original file slot.
            file_type = expected["type"]
            if file_type == "STRING" and actual.get("type") == "ASSET":
                if (
                    list_flag
                    or expected.get("default") != ""
                    or actual.get("group") != "optional"
                    or actual.get("default") is not None
                ):
                    return [f"{path}: empty file requires an optional ASSET with no default"]
                if expected.get("group") == "required":
                    expected["group"] = "optional"
                expected.pop("multiline", None)
            if isinstance(file_type, dict):
                file_type = file_type["list"]
            if actual.get("type") == "ASSET":
                actual["type"] = file_type
            elif actual.get("type") == {"list": "ASSET"}:
                actual["type"] = {"list": file_type}
            expected.pop("options", None)
            expected.pop("default", None)
            actual.pop("default", None)
        missing = expected.keys() ^ actual.keys()
        result = [
            f"{path}.{key}: missing {'native' if key in expected else 'reference'} field"
            for key in sorted(missing)
        ]
        return result + [
            difference
            for key in sorted(expected.keys() & actual.keys())
            for difference in _differences(expected.get(key), actual.get(key), f"{path}.{key}")
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        result = []
        if len(expected) != len(actual):
            result.append(f"{path}: length {len(expected)} vs {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            result.extend(_differences(left, right, f"{path}[{index}]"))
        return result
    # Python's True == 1 must not hide corrupted positional widget values.
    if (isinstance(expected, bool) != isinstance(actual, bool)) or expected != actual:
        return [f"{path}: {expected!r} vs {actual!r}"]
    return []


def compare(
    row: dict[str, Any], schema: dict[str, Any], *, expected_node_type: str | None = None
) -> list[str]:
    if row.get("input_types_error") or not isinstance(row.get("input_types"), dict):
        return [f"reference input schema unavailable: {row.get('input_types_error')}"]
    if "interface" not in schema:
        return ["native interface missing"]
    if any(item.get("role") not in INPUT_ROLES | {"output"} for item in schema["interface"]):
        return ["native interface contains an unsupported role or a variable output family"]
    expected = _reference_inputs(row["input_types"], bool(row.get("input_is_list")))
    for item in expected:
        if not any(
            key in item
            for key in (
                "conditional_options",
                "environment_options",
                "baseline_options",
                "possible_options",
            )
        ):
            continue
        allowed = None
        if row["class_name"] == "ModelAttentionBackend" and item["name"] == "attention":
            allowed = [
                {
                    "value": "comfy kitchen attention",
                    "condition": "comfy_kitchen.int8_attention_is_available()",
                }
            ]
        elif (
            row["class_name"] in ("SelectModelDevice", "SelectCLIPDevice", "SelectVAEDevice")
            and item["name"] == "device"
        ):
            allowed = [
                {
                    "pattern": r"gpu:[0-9]+",
                    "condition": "len(comfy.model_management.get_all_torch_devices()) > 1",
                }
            ]
        if (
            allowed is None
            or "conditional_options" in item
            or item.get("environment_options") is not True
            or item.get("possible_options") != allowed
            or item.get("baseline_options") != item.get("options")
            or not isinstance(item.get("options"), list)
            or not all(isinstance(option, str) for option in item["options"])
        ):
            return [f"{item['name']}: invalid conditional option declaration"]
    if row.get("is_v3") or "v3_inputs" in row:
        declarations = row.get("v3_inputs")
        if not isinstance(declarations, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("id"), str)
            for item in declarations
        ):
            return ["reference V3 input sequence unavailable or malformed"]
        names = [item["id"] for item in declarations]
        by_name = {item["name"]: item for item in expected}
        if (
            len(names) != len(set(names))
            or len(names) != len(expected)
            or set(names) != set(by_name)
        ):
            return ["reference V3 input sequence does not match public input groups"]
        expected = [by_name[name] for name in names]
    actual = _native_inputs(schema.get("interface", []), expected)
    differences = []
    for item in expected:
        conditional = item.pop("possible_options", [])
        if not conditional:
            continue
        item.pop("environment_options")
        baseline = item.pop("baseline_options")
        for entry in (other for other in actual if other["name"] == item["name"]):
            options = entry.get("options", [])
            if not entry.get("remote_route"):
                if any(option not in options for option in baseline):
                    differences.append(f"{item['name']}: native options omit baseline choices")
                for declaration in conditional:
                    if not any(
                        option == declaration["value"]
                        if "value" in declaration
                        else isinstance(option, str)
                        and re.fullmatch(declaration["pattern"], option) is not None
                        for option in options
                    ):
                        differences.append(f"{item['name']}: native options omit possible choices")
                if any(
                    option not in baseline
                    and not any(
                        option == declaration["value"]
                        if "value" in declaration
                        else isinstance(option, str)
                        and re.fullmatch(declaration["pattern"], option) is not None
                        for declaration in conditional
                    )
                    for option in options
                ):
                    differences.append(
                        f"{item['name']}: native options include unsupported choices"
                    )
            entry["options"] = item["options"]
    save_node = row.get("output_node") is True and any(
        item["name"] == "filename_prefix"
        and item["type"] == "STRING"
        and not item.get("_list_flag")
        for item in expected
    )
    if actual and actual[-1]["name"] == "mount" and save_node:
        mount = actual.pop()
        if mount.get("multiline") is False:
            mount.pop("multiline")
        # The production serializer omits the configured-output None sentinel.
        mount.setdefault("default", None)
        differences.extend(
            _differences(
                {"name": "mount", "group": "optional", "type": "STRING", "default": None},
                mount,
                "mount",
            )
        )
    for item in expected:
        if item["name"] in row.get("file_inputs", []):
            item["file_input"] = True
    differences.extend(_differences(expected, actual, "inputs"))
    expected_outputs = []
    names = row.get("return_names")
    output_metadata = row.get("v3_outputs") or []
    output_list_flags = row.get("output_is_list") or []
    for index, output_type in enumerate(row.get("return_types") or []):
        if output_type == "COMFY_MATCHTYPE_V3":
            metadata = output_metadata[index] if index < len(output_metadata) else {}
            output_type = metadata.get("template", {"missing_output_template": True})
        output = {"type": output_type}
        if index < len(output_list_flags) and output_list_flags[index]:
            output["_list_flag"] = True
        if names is not None and names[index] is not None:
            output["name"] = names[index]
        expected_outputs.append(output)
    actual_outputs = [
        {
            **(
                {"name": item.get("displayName") or item["id"]}
                if names is not None and index < len(names) and names[index] is not None
                else {}
            ),
            "type": _type(item["type"]),
        }
        for index, item in enumerate(
            item
            for item in schema.get("interface", [])
            if item["role"] in ("output", "outputFamily")
        )
    ]
    differences.extend(_differences(expected_outputs, actual_outputs, "outputs"))
    for key, reference, native in (
        ("node id", expected_node_type or row["class_name"], schema.get("nodeType")),
        ("category", row.get("category") or "", schema.get("category") or ""),
        (
            "display name",
            row.get("display_name") or row["class_name"],
            schema.get("displayName") or schema.get("nodeType"),
        ),
        ("output node", bool(row.get("output_node")), bool(schema.get("outputNode"))),
    ):
        differences.extend(_differences(reference, native, key))
    return differences


def compare_registry(
    reference: dict[str, Any],
    registry: dict[str, Any],
    *,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if scope is None:
        scope = cast(
            "dict[str, Any]", json.loads(SCOPE_RULES.read_text(), object_pairs_hook=_unique_object)
        )
    if scope["schemaVersion"] != 1:
        raise ValueError("unsupported scope version")
    migrated = set(scope["migratedPacks"])
    nodes = reference["nodes"]
    if not nodes:
        raise ValueError("reference contains no node classes")
    classes = [row["class_name"] for row in nodes]
    if len(classes) != len(set(classes)):
        raise ValueError("duplicate reference class id")
    served = registry["nodes"]
    results = []
    errors = []
    identities: dict[str, tuple[str, str]] = {}
    # A nested official namespace belongs to the most specific selected pack.
    moved_owners = {
        key: pack
        for key in served
        for pack in sorted(migrated, key=len)
        if key.startswith(pack.replace("-", ".") + ".")
    }
    for key, schema in served.items():
        if key not in moved_owners:
            continue
        for rule in schema.get("replacements", []):
            if rule.get("migration") is not None or rule["from"] == key:
                continue
            if not rule.get("cases") or any(case.get("to") != key for case in rule["cases"]):
                errors.append(f"{key}: replacement must target its declaring node")
                continue
            source = rule["from"]
            if source in identities:
                raise ValueError(f"duplicate replacement source class: {source}")
            identities[source] = (key, moved_owners[key])
    conditional_options = {}
    if reference.get("builtin_import_failures"):
        errors.append(
            "reference builtin imports failed: " + ", ".join(reference["builtin_import_failures"])
        )
    expected = {name: identities.get(name, (name, scope["legacyPack"])) for name in classes}
    exclusions = {
        row["class_name"]: next(
            (
                rule
                for rule in scope["exclusions"]
                if any(row.get(flag) for flag in rule.get("flags", []))
                or row.get("file") in rule.get("files", [])
                or any(
                    str(row.get("file", "")).startswith(prefix)
                    for prefix in rule.get("filePrefixes", [])
                )
            ),
            None,
        )
        for row in nodes
    }
    eligible_targets = {target for name, (target, _) in expected.items() if not exclusions[name]}
    owners = {node_type: pack for node_type, pack in expected.values()}
    ids = set(owners)
    owners.update(moved_owners)
    moved_ids = set(moved_owners)
    for key, schema in served.items():
        if key != schema.get("nodeType"):
            errors.append(f"registry key {key!r} differs from nodeType {schema.get('nodeType')!r}")
        owner = owners.get(key)
        actual_owner = (
            schema.get("pack") if key in moved_ids else schema.get("pack", scope["legacyPack"])
        )
        if key in owners and actual_owner != owner:
            errors.append(f"core class id {key!r} is claimed by pack {schema.get('pack')!r}")
        if key in identities:
            errors.append(f"retired class id remains registered: {key}")
        if key.startswith(FORBIDDEN_PREFIXES) and key not in moved_ids:
            errors.append(f"prefixed node id: {key}")
        for item in schema.get("interface", []):
            for value in _type_names(item):
                if value.startswith(FORBIDDEN_PREFIXES):
                    errors.append(f"{key}: prefixed type name: {value}")
    for row in nodes:
        name = row["class_name"]
        node_type, pack = expected[name]
        inputs = row.get("input_types")
        if not isinstance(inputs, dict):
            inputs = {}
        row_conditionals = {
            parameter: {
                key: spec[1].get(key)
                for key in ("environment_options", "baseline_options", "possible_options")
            }
            for group in ("required", "optional")
            if isinstance(inputs.get(group), dict)
            for parameter, spec in inputs[group].items()
            if isinstance(spec, list)
            and len(spec) > 1
            and isinstance(spec[1], dict)
            and spec[1].get("environment_options")
        }
        if row_conditionals:
            conditional_options[name] = row_conditionals
        exclusion = ""
        if rule := exclusions[name]:
            exclusion = rule["reason"]
            # An import-only rule can share a target with an eligible class;
            # the excluded source must still never be registered itself.
            if rule["mustBeAbsent"] and (
                name in served or (node_type in served and node_type not in eligible_targets)
            ):
                errors.append(f"excluded {exclusion} remains registered: {name}")
        differences = []
        if exclusion:
            status = "excluded"
        elif node_type not in served:
            status = "absent"
        else:
            try:
                differences = compare(row, served[node_type], expected_node_type=node_type)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                differences = [f"invalid interface: {type(exc).__name__}: {exc}"]
            status = (
                "differs" if differences else "conditional" if row_conditionals else "identical"
            )
        results.append(
            {
                "class_name": name,
                "node_type": node_type,
                "pack": pack,
                "file": row.get("file", ""),
                "status": status,
                "exclusion": exclusion,
                "differences": " | ".join(differences),
            }
        )
    for name in sorted(served.keys() - ids):
        results.append(
            {
                "class_name": name,
                "node_type": name,
                "pack": served[name].get("pack", ""),
                "file": "",
                "status": "Dinkster-only",
                "exclusion": "",
                "differences": "",
            }
        )
    counts = dict(Counter(row["status"] for row in results))
    return {
        "reference_commit": reference.get("source_commit"),
        "reference_environment": reference.get("environment"),
        "eligible_node_count": sum(
            row["status"] in ("identical", "conditional", "differs", "absent") for row in results
        ),
        "counts": counts,
        "pack_counts": {
            pack: dict(Counter(row["status"] for row in results if row["pack"] == pack))
            for pack in sorted({row["pack"] for row in results if row["pack"]})
        },
        "engine_exclusions": {},
        "conditional_options": conditional_options,
        "registry_errors": sorted(set(errors)),
        "rows": results,
        "passed": not errors and not counts.get("differs") and not counts.get("absent"),
    }


def _type_names(value: Any) -> list[str]:
    result = []
    if isinstance(value, dict):
        if value.get("kind") in ("concrete", "union", "variable"):
            result.extend(value.get("types", []))
            result.extend(value.get("allowed", []))
        for nested in value.values():
            result.extend(_type_names(nested))
    elif isinstance(value, list):
        for nested in value:
            result.extend(_type_names(nested))
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "registry", "csv_output", "summary_output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--scope", type=Path, default=SCOPE_RULES)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(), object_pairs_hook=_unique_object)
    registry = json.loads(args.registry.read_text(), object_pairs_hook=_unique_object)
    summary = compare_registry(
        reference,
        registry,
        scope=json.loads(args.scope.read_text(), object_pairs_hook=_unique_object),
    )
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_name",
                "node_type",
                "pack",
                "file",
                "status",
                "exclusion",
                "differences",
            ],
        )
        writer.writeheader()
        writer.writerows(summary.pop("rows"))
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
