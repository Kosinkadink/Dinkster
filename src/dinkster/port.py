"""dinkster-port - generate a native Dinkster pack skeleton from a legacy ComfyUI pack.

Porting is the endorsed migration path (DESIGN 3.8); this tool makes it
cheap. Point it at an unmodified custom-node pack - v1 NODE_CLASS_MAPPINGS,
V3 comfy_entrypoint, or a mixed pack shipping both - and it emits a
doctor-clean native pack: grammar-valid manifest, one Node subclass per
translatable source node with the schema fully translated (v1 through the
SAME translate.py rules the compat worker runs, V3 through translate_v3.py's
direct Schema conversion), a typed execute() stub carrying the source
reference, register_types for the opaque envelope types the schemas need,
tests, and a README that lists exactly what ported, what was skipped and
why, and what remains TODO. Faithfully translated schema metadata is
emitted as real code; everything that needs a human lands as a TODO(port)
marker - the generator never pretends fidelity it does not have.

Reading a ComfyUI pack means importing arbitrary pack code, so discovery runs
in a disposable subprocess (``dinkster_compat_comfy.port_probe``) against a
real ComfyUI environment (``--comfyui-root`` / ``DINKSTER_COMFYUI_ROOT``),
the same quarantine posture as ``dinkster doctor``'s import probe. This
process only ever consumes the probe's JSON.

Usage::

    dinkster-port <legacy-pack-path> --name my-pack [--out DIR]
               [--comfyui-root PATH] [--nodes A,B] [--force]
"""

from __future__ import annotations

import argparse
import json
import keyword
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dinkster_schema import reserved_root, validate_name
from dinkster_workers.interpreter import InterpreterPreflightError, preflight_interpreter

from .comfy_compose import comfy_python, dinkster_pythonpath, find_compat_manifest

_PROBE_TIMEOUT_S = 600.0
_LINE_WIDTH = 100
_CHOICE_PREVIEW = 8

_SCALAR_ANNOTATIONS = {
    "core.int": "int",
    "core.float": "float",
    "core.string": "str",
    "core.combo": "str",
    "core.boolean": "bool",
}


class PortError(Exception):
    """The port cannot proceed; the message says why and what to do."""


# --- naming ---------------------------------------------------------------


def _module_name(pack_name: str) -> str:
    """``my-pack`` -> ``my_pack_nodes`` (template convention)."""
    return re.sub(r"[-.]", "_", pack_name) + "_nodes"


def _pack_display_name(pack_name: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[-_.]", pack_name))


def _node_slug(source_name: str) -> str:
    """A grammar-friendly node-type tail from a source node name: lowercase,
    non-alphanumeric runs collapse to ``_``. The original source name
    survives in ``aliases`` so submission formats still resolve it."""
    slug = re.sub(r"[^a-z0-9]+", "_", source_name.lower()).strip("_")
    if not slug:
        slug = "node"
    if not slug[0].isalpha():
        slug = "n" + slug
    return slug


def _class_identifier(source_name: str) -> str:
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", source_name) if t]
    name = "".join(t[:1].upper() + t[1:] for t in tokens) or "Node"
    if not name[0].isalpha():
        name = "N" + name
    return name


def _dedupe(candidate: str, used: set[str], *, sep: str) -> str:
    name = candidate
    counter = 2
    while name in used:
        name = f"{candidate}{sep}{counter}"
        counter += 1
    used.add(name)
    return name


# --- wire -> source fragments ----------------------------------------------


def _type_expr_code(wire: dict[str, Any]) -> str:
    kind = wire.get("kind")
    types = cast("list[str]", wire.get("types", []))
    if kind == "concrete":
        return f"TypeExpr.concrete({types[0]!r})"
    if kind == "wildcard":
        return "TypeExpr.wildcard()"
    if kind == "union":
        return "TypeExpr.union(" + ", ".join(repr(t) for t in types) + ")"
    if kind == "variable":
        # Wire-15 spells the variable constraint set "allowed", not "types".
        allowed_ids = cast("list[str]", wire.get("allowed", []))
        allowed = ", ".join(repr(t) for t in allowed_ids)
        template = cast("str", wire.get("templateId", ""))
        if allowed:
            trailing = "," if len(allowed_ids) == 1 else ""
            return f"TypeExpr.variable({template!r}, ({allowed}{trailing}))"
        return f"TypeExpr.variable({template!r})"
    if kind == "list":
        return f"TypeExpr.list_of({_type_expr_code(cast('dict[str, Any]', wire['element']))})"
    raise PortError(f"probe reported an unknown TypeExpr kind: {kind!r}")


def _annotation(wire: dict[str, Any]) -> str:
    kind = wire.get("kind")
    if kind == "concrete":
        types = cast("list[str]", wire["types"])
        return _SCALAR_ANNOTATIONS.get(types[0], "object")
    if kind == "list":
        return f"list[{_annotation(cast('dict[str, Any]', wire['element']))}]"
    return "object"


def _family_annotation(entry: dict[str, Any]) -> str:
    template = cast("list[dict[str, Any]]", entry.get("template", []))
    if len(template) != 1 or template[0].get("role") != "input":
        return "object"
    type_wire = template[0].get("type")
    if not isinstance(type_wire, dict):
        return "object"
    return _annotation(cast("dict[str, Any]", type_wire))


def _collect_atoms(wire: dict[str, Any], out: set[str]) -> None:
    for type_id in cast("list[str]", wire.get("types", [])):
        out.add(type_id)
    for type_id in cast("list[str]", wire.get("allowed", [])):
        out.add(type_id)
    element = wire.get("element")
    if isinstance(element, dict):
        _collect_atoms(cast("dict[str, Any]", element), out)


def _collect_entry_atoms(entry: dict[str, Any], out: set[str]) -> None:
    type_wire = entry.get("type")
    if isinstance(type_wire, dict):
        _collect_atoms(cast("dict[str, Any]", type_wire), out)
    slot_type = entry.get("slotType")
    if isinstance(slot_type, dict):
        _collect_atoms(cast("dict[str, Any]", slot_type), out)
    for field in ("template", "inputs"):
        for child in cast("list[dict[str, Any]]", entry.get(field, [])):
            _collect_entry_atoms(child, out)
    for option in cast("list[dict[str, Any]]", entry.get("options", [])):
        for child in cast("list[dict[str, Any]]", option.get("inputs", [])):
            _collect_entry_atoms(child, out)
    for variant in cast("list[dict[str, Any]]", entry.get("variants", [])):
        _collect_atoms(cast("dict[str, Any]", variant["type"]), out)
        for child in cast("list[dict[str, Any]]", variant.get("inputs", [])):
            _collect_entry_atoms(child, out)


def _py_literal(value: object) -> str:
    """``repr`` that stays a valid Python expression for everything JSON
    can carry: non-finite floats (JSON's Infinity/NaN) repr as bare
    ``inf``/``nan``, which would be NameErrors in generated source."""
    if isinstance(value, float) and not math.isfinite(value):
        return f"float({str(value)!r})"
    return repr(value)


def _literal_lines(value: object, indent: str, key: str) -> list[str]:
    """``key=value,`` as source, chunking long strings into adjacent
    literals so generated files stay within the pack lint's line width."""
    text = _py_literal(value)
    single = f"{indent}{key}={text},"
    if len(single) <= _LINE_WIDTH - 2:
        return [single]
    if isinstance(value, str):
        lines = [f"{indent}{key}=("]
        chunk_width = _LINE_WIDTH - len(indent) - 12
        for start in range(0, len(value), chunk_width):
            lines.append(f"{indent}    {value[start : start + chunk_width]!r}")
        lines.append(f"{indent}),")
        return lines
    return [f"{single}  # noqa: E501"]


def _comment_lines(text: str, indent: str, prefix: str = "# ") -> list[str]:
    """Comment lines wrapped on word boundaries so v1 reference source and
    choice lists never break the generated pack's own lint."""
    lines: list[str] = []
    budget = _LINE_WIDTH - len(indent) - len(prefix)
    for raw in text.splitlines() or [""]:
        line = raw.rstrip()
        if not line:
            lines.append(f"{indent}{prefix}".rstrip())
            continue
        lead = line[: len(line) - len(line.lstrip())]
        for piece in textwrap.wrap(
            line.strip(),
            budget,
            initial_indent=lead,
            subsequent_indent=lead,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]:
            lines.append(f"{indent}{prefix}{piece}")
    return lines


# --- node emission ----------------------------------------------------------


@dataclass(frozen=True)
class _PortedNode:
    source_name: str
    source_api: str
    node_type: str
    class_name: str


@dataclass(frozen=True)
class _Interface:
    inputs: list[dict[str, Any]]
    input_families: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    output_families: list[dict[str, Any]]


def _source_api(node: dict[str, Any]) -> str:
    return cast("str", node.get("sourceApi", "v1"))


def _split_interface(node: dict[str, Any]) -> _Interface:
    interface = _Interface([], [], [], [])
    by_role = {
        "input": interface.inputs,
        "inputFamily": interface.input_families,
        "output": interface.outputs,
        "outputFamily": interface.output_families,
    }
    schema = cast("dict[str, Any]", node["schema"])
    for entry in cast("list[dict[str, Any]]", schema["interface"]):
        bucket = by_role.get(cast("str", entry.get("role", "")))
        if bucket is None:
            # Refusing beats silently dropping part of an interface.
            raise PortError(
                f"{node['sourceName']}: unexpected interface role "
                f"{entry.get('role')!r} in probe output"
            )
        bucket.append(entry)
    return interface


def _widget_code(wire: dict[str, Any]) -> str:
    """Render a wire widget descriptor back into constructor source, so
    generated ports keep the UI affordances translation preserved instead
    of dropping them on the codegen floor."""
    kind = wire.get("type")
    if kind == "COMBO":
        parts: list[str] = []
        options = cast("list[str]", wire.get("options", []))
        if options:
            option_items = "".join(f"{o!r}, " for o in options)
            parts.append(f"options=({option_items})")
        remote = cast("dict[str, Any]", wire.get("remote") or {})
        if remote.get("route"):
            parts.append(f"remote_route={remote['route']!r}")
            if remote.get("refreshButton"):
                parts.append("refresh_button=True")
        return f"ComboWidget({', '.join(parts)})"
    if kind == "BOOLEAN":
        parts = []
        if wire.get("labelOn"):
            parts.append(f"label_on={wire['labelOn']!r}")
        if wire.get("labelOff"):
            parts.append(f"label_off={wire['labelOff']!r}")
        return f"BooleanWidget({', '.join(parts)})"
    if kind == "ASSET":
        accept = cast("list[str]", wire.get("accept", []))
        accept_items = "".join(f"{a!r}, " for a in accept)
        parts = [f"accept=({accept_items})"]
        if wire.get("kind"):
            parts.append(f"kind={wire['kind']!r}")
        return f"AssetWidget({', '.join(parts)})"
    if kind == "SAVE_TARGET":
        suffix = f"suffix={wire['suffix']!r}" if wire.get("suffix") else ""
        return f"SaveTargetWidget({suffix})"
    raise PortError(f"probe reported an unknown widget kind: {kind!r}")


_WIDGET_CLASS_NAMES = {
    "COMBO": "ComboWidget",
    "BOOLEAN": "BooleanWidget",
    "ASSET": "AssetWidget",
    "SAVE_TARGET": "SaveTargetWidget",
}


def _emit_input_spec(entry: dict[str, Any], indent: str) -> list[str]:
    lines = [f"{indent}InputSpec("]
    inner = indent + "    "
    lines.append(f"{inner}{entry['id']!r},")
    lines.append(f"{inner}{_type_expr_code(cast('dict[str, Any]', entry['type']))},")
    if not entry.get("required", True):
        lines.append(f"{inner}required=False,")
    if "default" in entry:
        lines.extend(_literal_lines(entry["default"], inner, "default"))
    if entry.get("doc"):
        lines.extend(_literal_lines(entry["doc"], inner, "doc"))
    if entry.get("widget"):
        lines.append(f"{inner}widget={_widget_code(cast('dict[str, Any]', entry['widget']))},")
    lines.append(f"{indent}),")
    return lines


def _emit_output_spec(entry: dict[str, Any], indent: str) -> list[str]:
    lines = [f"{indent}OutputSpec("]
    inner = indent + "    "
    lines.append(f"{inner}{entry['id']!r},")
    lines.append(f"{inner}{_type_expr_code(cast('dict[str, Any]', entry['type']))},")
    if entry.get("doc"):
        lines.extend(_literal_lines(entry["doc"], inner, "doc"))
    lines.append(f"{indent}),")
    return lines


def _emit_family_spec(entry: dict[str, Any], indent: str, class_name: str) -> list[str]:
    lines = [f"{indent}{class_name}("]
    inner = indent + "    "
    lines.append(f"{inner}{entry['id']!r},")
    type_wire = entry.get("type")
    if class_name == "InputFamilySpec":
        template = cast("list[dict[str, Any]]", entry.get("template", []))
        if len(template) != 1 or template[0].get("role") != "input":
            raise PortError("grouped recursive input families are not ported yet")
        type_wire = template[0].get("type")
    if not isinstance(type_wire, dict):
        raise PortError(f"{class_name} lacks a member type")
    lines.append(f"{inner}{_type_expr_code(cast('dict[str, Any]', type_wire))},")
    min_members = cast("int", entry.get("minMembers", 0))
    if min_members:
        lines.append(f"{inner}min_members={min_members},")
    if "maxMembers" in entry:
        lines.append(f"{inner}max_members={entry['maxMembers']},")
    if entry.get("memberPrefix") is not None:
        lines.append(f"{inner}member_prefix={entry['memberPrefix']!r},")
    if entry.get("memberNames") is not None:
        names = tuple(cast("list[str]", entry["memberNames"]))
        lines.append(f"{inner}member_names={names!r},")
    if entry.get("doc"):
        lines.extend(_literal_lines(entry["doc"], inner, "doc"))
    lines.append(f"{indent}),")
    return lines


def _choices_comment(node: dict[str, Any], input_id: str, indent: str) -> list[str]:
    choices = cast("dict[str, list[object]]", node.get("choices", {}))
    values = choices.get(input_id)
    if not values:
        return []
    preview = ", ".join(repr(v) for v in values[:_CHOICE_PREVIEW])
    if len(values) > _CHOICE_PREVIEW:
        preview += f", ... (+{len(values) - _CHOICE_PREVIEW} more)"
    return _comment_lines(
        f"source combo choices for {input_id!r} (core.combo identity with "
        f"UI-only vocabulary - consider a closed enum or an asset input): {preview}",
        indent,
    )


def _execute_signature(interface: _Interface, class_name: str) -> list[str]:
    """The typed keyword-only stub signature, or a ``**inputs`` fallback
    when a source input id is not a Python parameter name. Input-family
    members arrive grouped as one mapping per family id (the worker's
    calling convention); output families add the reserved output_spec."""
    named = interface.inputs + interface.input_families
    ids = [cast("str", e["id"]) for e in named]
    if any(not i.isidentifier() or keyword.iskeyword(i) for i in ids):
        lines = ["    @classmethod"]
        lines.extend(
            _comment_lines(
                f"source input ids are not all valid parameter names ({', '.join(ids)})",
                "    ",
            )
        )
        lines.append("    def execute(cls, **inputs: object) -> Mapping[str, object]:")
        return lines
    params: list[str] = []
    for entry in interface.inputs:
        input_id = cast("str", entry["id"])
        annotation = _annotation(cast("dict[str, Any]", entry["type"]))
        if "default" in entry:
            default = _py_literal(entry["default"])
            if len(default) > 40:
                default = "..."  # long defaults stay in define_schema only
            params.append(f"{input_id}: {annotation} = {default}")
        elif not entry.get("required", True):
            params.append(f"{input_id}: {annotation} | None = None")
        else:
            params.append(f"{input_id}: {annotation}")
    for entry in interface.input_families:
        annotation = _family_annotation(entry)
        params.append(f"{entry['id']}: Mapping[str, {annotation}]")
    if interface.output_families:
        params.append("output_spec: OutputInterface")
    header = "    def execute(cls, *, " + ", ".join(params) + ") -> Mapping[str, object]:"
    if len(header) <= _LINE_WIDTH and params:
        return ["    @classmethod", header]
    if not params:
        return ["    @classmethod", "    def execute(cls) -> Mapping[str, object]:"]
    lines = ["    @classmethod", "    def execute(", "        cls,", "        *,"]
    lines.extend(f"        {param}," for param in params)
    lines.append("    ) -> Mapping[str, object]:")
    return lines


def _emit_node(node: dict[str, Any], node_type: str, class_name: str) -> str:
    schema = cast("dict[str, Any]", node["schema"])
    interface = _split_interface(node)
    source_name = cast("str", node["sourceName"])
    api = _source_api(node)

    lines: list[str] = [f"class {class_name}(Node):"]
    doc = (
        f'    """{schema.get("displayName", source_name)} '
        f'(ported from {api} node {source_name!r})."""'
    )
    if len(doc) > _LINE_WIDTH:
        lines.append(f'    """Ported from {api} node {source_name!r}."""')
    else:
        lines.append(doc)
    lines.append("")
    lines.append("    @classmethod")
    lines.append("    def define_schema(cls) -> NodeSchema:")
    lines.append("        return NodeSchema(")
    lines.append(f"            node_type={node_type!r},")
    lines.extend(
        _literal_lines(schema.get("displayName", source_name), "            ", "display_name")
    )
    category = cast("str", schema.get("category", ""))
    category = category.removeprefix("comfy/")
    lines.extend(_literal_lines(category, "            ", "category"))
    description = cast("str", schema.get("description", ""))
    if description:
        lines.extend(_literal_lines(description, "            ", "description"))
    if interface.inputs:
        lines.append("            inputs=(")
        for entry in interface.inputs:
            if not entry.get("widget"):
                # Choices that became a widget (static combo, labeled
                # boolean) already live in the spec; only widgetless
                # inputs need the prose reminder.
                lines.extend(_choices_comment(node, cast("str", entry["id"]), "                "))
            lines.extend(_emit_input_spec(entry, "                "))
        lines.append("            ),")
    if interface.outputs:
        lines.append("            outputs=(")
        for entry in interface.outputs:
            lines.extend(_emit_output_spec(entry, "                "))
        lines.append("            ),")
    if interface.input_families:
        lines.append("            input_families=(")
        for entry in interface.input_families:
            lines.extend(_emit_family_spec(entry, "                ", "InputFamilySpec"))
        lines.append("            ),")
    if interface.output_families:
        lines.append("            output_families=(")
        for entry in interface.output_families:
            lines.extend(_emit_family_spec(entry, "                ", "OutputFamilySpec"))
        lines.append("            ),")
    if not schema.get("idempotent", True):
        if api == "v1":
            lines.append("            # TODO(port): v1 signaled OUTPUT_NODE/IS_CHANGED, so the")
            lines.append(
                "            # translation never caches this node. Decide real idempotence."
            )
        else:
            lines.append("            # TODO(port): V3 declared is_output_node/not_idempotent, so")
            lines.append("            # this node is never cached. Decide real idempotence.")
        lines.append("            idempotent=False,")
    occupies = cast("list[str]", schema.get("occupies", []))
    if occupies:
        entries = ", ".join(repr(lane) for lane in occupies)
        lines.append(f"            occupies=({entries},),")
    if schema.get("ioBound"):
        lines.append("            # V3 declared is_api_node: a network-waiting node.")
        lines.append("            io_bound=True,")
    deprecation = cast("dict[str, Any] | None", schema.get("deprecation"))
    if deprecation is not None:
        lines.append("            # TODO(port): the source pack declared a bare deprecation")
        lines.append("            # flag; write real prose (and a successor, if one exists).")
        lines.append("            deprecation=Deprecation(")
        lines.extend(_literal_lines(deprecation.get("message", ""), "                ", "message"))
        if deprecation.get("since"):
            lines.extend(_literal_lines(deprecation["since"], "                ", "since"))
        if deprecation.get("replacement"):
            lines.extend(
                _literal_lines(deprecation["replacement"], "                ", "replacement")
            )
        lines.append("            ),")
    visibility = cast("str", schema.get("searchVisibility", "normal"))
    if visibility != "normal":
        lines.append("            # V3 declared is_dev_only: hidden from node search.")
        lines.append(f"            search_visibility={visibility!r},")
    if schema.get("outputNode"):
        lines.append("            output_node=True,")
    lines.append("            # The source name stays resolvable for API submissions.")
    lines.append(f"            aliases=({source_name!r},),")
    lines.append("        )")
    lines.append("")

    lines.extend(_execute_signature(interface, class_name))
    hidden = cast("list[str]", node.get("hidden", []))
    if hidden:
        lines.extend(
            _comment_lines(
                f"source declared hidden inputs ({', '.join(hidden)}); they "
                "never enter Dinkster schemas. Port prompt/id introspection "
                "properly or drop it.",
                "        ",
            )
        )
    if node.get("inputIsList"):
        lines.extend(
            _comment_lines(
                "source used INPUT_IS_LIST/is_input_list: the function "
                "received every input as a whole list, which the schema now "
                "declares as list<T> sockets.",
                "        ",
            )
        )
    search_aliases = cast("list[str]", node.get("searchAliases", []))
    if search_aliases:
        lines.extend(
            _comment_lines(
                f"V3 declared search aliases ({', '.join(search_aliases)}); "
                "Dinkster aliases are submission resolution, not search "
                "synonyms, so they were not carried over.",
                "        ",
            )
        )
    if interface.outputs:
        output_ids = ", ".join(f"{cast('str', e['id'])}=..." for e in interface.outputs)
        lines.extend(
            _comment_lines(
                f"TODO(port): implement natively and return cls.outputs({output_ids})",
                "        ",
            )
        )
    else:
        lines.extend(
            _comment_lines("TODO(port): implement natively (no declared outputs)", "        ")
        )
    lines.append("        raise NotImplementedError(")
    lines.append(f'            "TODO(port): implement {node_type}"')
    lines.append("        )")

    source = cast("str", node.get("source", ""))
    function = cast("str", node.get("function", ""))
    if source:
        lines.append("")
        lines.extend(
            _comment_lines(
                f"--- {api} reference: {source_name}.{function} (delete after porting) ---",
                "    ",
            )
        )
        lines.extend(_comment_lines(source.rstrip(), "    ", prefix="#   "))
    return "\n".join(lines)


# --- file emission -----------------------------------------------------------


def _emit_module(
    pack_name: str,
    source_label: str,
    ported: list[_PortedNode],
    node_payloads: list[dict[str, Any]],
    opaque_types: list[str],
) -> str:
    def _interface(node: dict[str, Any]) -> list[dict[str, Any]]:
        return cast("list[dict[str, Any]]", cast("dict[str, Any]", node["schema"])["interface"])

    def _has_role(role: str) -> bool:
        return any(
            entry.get("role") == role for node in node_payloads for entry in _interface(node)
        )

    has_inputs = _has_role("input")
    has_outputs = _has_role("output")
    has_input_families = _has_role("inputFamily")
    has_output_families = _has_role("outputFamily")
    has_deprecation = any(
        cast("dict[str, Any]", node["schema"]).get("deprecation") is not None
        for node in node_payloads
    )
    api_names = ["Node", "NodeSchema", "TypeRegistry"]
    if has_inputs:
        api_names.append("InputSpec")
    if has_outputs:
        api_names.append("OutputSpec")
    if has_input_families:
        api_names.append("InputFamilySpec")
    if has_output_families:
        api_names.extend(["OutputFamilySpec", "OutputInterface"])
    if has_inputs or has_outputs or has_input_families or has_output_families:
        api_names.append("TypeExpr")
    if has_deprecation:
        api_names.append("Deprecation")
    widget_kinds = {
        cast("dict[str, Any]", entry["widget"]).get("type")
        for node in node_payloads
        for entry in _interface(node)
        if entry.get("widget")
    }
    api_names.extend(name for kind, name in _WIDGET_CLASS_NAMES.items() if kind in widget_kinds)

    header = [
        f'"""{_pack_display_name(pack_name)}: native Dinkster pack ported from {source_label}.',
        "",
        "Generated by 'dinkster port'. Schemas below are faithful translations of",
        "the source declarations; every execute() is a stub carrying its",
        "source reference. Work the TODO(port) markers, then delete them.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from collections.abc import Mapping",
        "",
        "from dinkster_api.v1 import (",
    ]
    header.extend(f"    {name}," for name in sorted(api_names))
    header.append(")")

    blocks = ["\n".join(header)]
    blocks.extend(
        _emit_node(payload, node.node_type, node.class_name)
        for node, payload in zip(ported, node_payloads, strict=True)
    )

    register_lines = ["def register_types(registry: TypeRegistry) -> None:"]
    if opaque_types:
        register_lines.extend(
            [
                '    """Envelope types this pack\'s schemas reference, carried over',
                "    from ComfyUI type strings.",
                "",
                "    TODO(port): declare encode/decode codecs so values cross process",
                "    boundaries and caches without the fallback codec (dinkster doctor",
                "    warns until then) - or drop entries owned by a pack you depend",
                '    on and let that pack register them."""',
            ]
        )
        register_lines.extend(f"    registry.register({t!r})" for t in opaque_types)
    else:
        register_lines.append('    """The translated schemas need no pack-owned value types."""')
    blocks.append("\n".join(register_lines))

    nodes_lines = ["NODES = ["]
    nodes_lines.extend(f"    {node.class_name}," for node in ported)
    nodes_lines.append("]")
    blocks.append("\n".join(nodes_lines))

    return "\n\n\n".join(blocks) + "\n"


def _emit_manifest(pack_name: str, module_name: str, source_label: str) -> str:
    return f"""# Generated by 'dinkster port' from {source_label}.
# Review every TODO(port) marker before publishing.

[pack]
name = "{pack_name}"
# Node types live under "{pack_name}" (the default namespace claim - the
# pack's own name); declare claims explicitly only if you rename them.
# TODO(port): pin the ported code's runtime deps here (doctor warns on
# unpinned entries):
requires = []

[pack.sandbox]

[pack.entry]
nodes = "{module_name}:NODES"
types = "{module_name}:register_types"

[pack.presentation]
display_name = "{_pack_display_name(pack_name)}"
# TODO(port): declare abbr/color/mark and a 64x64 icon (docs/pack-authoring.md).
"""


def _emit_pyproject(pack_name: str, source_label: str) -> str:
    return f"""# Dev tooling ONLY - the pack's real metadata lives in dinkster-pack.toml.

[project]
name = "{pack_name}"
version = "0.1.0"
description = "Dinkster pack ported from {source_label} by dinkster port"
requires-python = ">=3.12"
dependencies = []  # runtime deps go in dinkster-pack.toml requires, pinned

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.8", "pyright>=1.1.390"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "standard"
"""


def _emit_readme(
    pack_name: str,
    source_label: str,
    ported: list[_PortedNode],
    skipped: dict[str, str],
    opaque_types: list[str],
) -> str:
    lines = [
        f"# {_pack_display_name(pack_name)}",
        "",
        f"Native Dinkster pack skeleton generated by `dinkster port` from `{source_label}`.",
        "",
        f"## Ported nodes ({len(ported)})",
        "",
        "| Node type | Source node | API |",
        "|-----------|-------------|-----|",
    ]
    lines.extend(f"| `{n.node_type}` | `{n.source_name}` | {n.source_api} |" for n in ported)
    if skipped:
        lines.extend(["", f"## Skipped by the translator ({len(skipped)})", ""])
        lines.append("| Source node | Reason |")
        lines.append("|-------------|--------|")
        lines.extend(f"| `{name}` | {reason} |" for name, reason in sorted(skipped.items()))
    lines.extend(
        [
            "",
            "## Porting checklist",
            "",
            "- [ ] Implement every `execute()` marked TODO(port); the source",
            "      function is inlined as a reference comment.",
        ]
    )
    if opaque_types:
        lines.append(
            "- [ ] Declare codecs in `register_types()` for: "
            + ", ".join(f"`{t}`" for t in opaque_types)
        )
    lines.extend(
        [
            "- [ ] Review generated combo/boolean widgets; core.combo keeps",
            "      choice vocabulary as UI affordance, so promote to closed",
            "      enums or asset inputs where choices are actual identity.",
            "- [ ] Revisit `idempotent=False` nodes - OUTPUT_NODE/IS_CHANGED",
            "      is a blunt signal, not a real idempotence declaration.",
            "- [ ] Pin `requires` in `dinkster-pack.toml`.",
            "- [ ] Add presentation (abbr, color, 64x64 icon).",
            "- [ ] Replace the generated smoke tests with real behavior tests.",
            "- [ ] Run `dinkster-doctor .` until clean.",
        ]
    )
    return "\n".join(lines) + "\n"


_CONFTEST = '''"""Put the pack directory on sys.path, exactly as the worker host does
when it resolves manifest entries."""

from __future__ import annotations

import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))
'''


def _emit_tests(module_name: str, ported: list[_PortedNode]) -> str:
    lines = [
        '"""Generated smoke tests: schemas build and types register.',
        "",
        "TODO(port): replace with real behavior tests as you implement",
        'execute() (see templates/pack for the style)."""',
        "",
        "from __future__ import annotations",
        "",
        "from dinkster_api.v1 import TypeRegistry",
        "",
        f"from {module_name} import NODES, register_types",
        "",
        "",
        "def test_schemas_build() -> None:",
        "    node_types = [node.define_schema().node_type for node in NODES]",
        "    assert node_types == [",
    ]
    lines.extend(f"        {n.node_type!r}," for n in ported)
    lines.extend(
        [
            "    ]",
            "",
            "",
            "def test_types_register() -> None:",
            "    register_types(TypeRegistry())",
        ]
    )
    return "\n".join(lines) + "\n"


# --- generation --------------------------------------------------------------


@dataclass(frozen=True)
class PortResult:
    out_dir: Path
    ported: tuple[_PortedNode, ...]
    skipped: dict[str, str]
    opaque_types: tuple[str, ...]


def generate_pack(
    probe: dict[str, Any],
    pack_name: str,
    out_dir: Path,
    *,
    only: list[str] | None = None,
    source_label: str = "",
) -> PortResult:
    """Emit the native pack skeleton from one probe report. Pure codegen:
    deterministic output, no pack code in this process."""
    report = cast("dict[str, Any]", probe.get("report", {}))
    status = cast("str", report.get("status", ""))
    if status not in ("loaded", "v3-entrypoint"):
        # "loaded" is the v1/mixed case; "v3-entrypoint" means the import
        # succeeded and the pack is pure V3 - exactly what porting is for.
        error = cast("str", report.get("error", ""))
        raise PortError(
            f"legacy pack did not load (status: {status or 'unknown'})"
            + (f": {error}" if error else "")
        )
    skipped = dict(cast("dict[str, str]", report.get("nodes_skipped", {})))
    skipped.update(cast("dict[str, str]", probe.get("v3Skipped", {})))
    v3_error = cast("str", probe.get("v3Error", ""))
    node_payloads = list(cast("list[dict[str, Any]]", probe.get("nodes", [])))
    if only is not None:
        by_name = {cast("str", n["sourceName"]): n for n in node_payloads}
        selected: list[dict[str, Any]] = []
        for name in only:
            if name in by_name:
                selected.append(by_name[name])
            elif name in skipped:
                raise PortError(
                    f"requested node {name!r} was skipped by the translator: {skipped[name]}"
                )
            else:
                raise PortError(f"unknown source node name: {name!r}")
        node_payloads = selected
    if not node_payloads:
        if v3_error:
            raise PortError(f"no translatable nodes to port (V3 entrypoint failed: {v3_error})")
        raise PortError("no translatable nodes to port")

    source_label = source_label or cast("str", report.get("path", "a ComfyUI pack"))
    # The module docstring and pyproject description use the short pack id;
    # long absolute paths (README, manifest comments) never enter lint-bound
    # Python lines.
    short_label = cast("str", report.get("pack_id", "")) or Path(source_label).name
    module_name = _module_name(pack_name)

    used_slugs: set[str] = set()
    used_classes: set[str] = set()
    ported: list[_PortedNode] = []
    atoms: set[str] = set()
    for payload in node_payloads:
        source_name = cast("str", payload["sourceName"])
        slug = _dedupe(_node_slug(source_name), used_slugs, sep="_")
        class_name = _dedupe(_class_identifier(source_name), used_classes, sep="V")
        ported.append(
            _PortedNode(
                source_name=source_name,
                source_api=_source_api(payload),
                node_type=f"{pack_name}.{slug}",
                class_name=class_name,
            )
        )
        for entry in cast(
            "list[dict[str, Any]]", cast("dict[str, Any]", payload["schema"])["interface"]
        ):
            _collect_entry_atoms(entry, atoms)

    # Only the opaque types the SELECTED schemas reference get registered.
    opaque_types = sorted(set(cast("list[str]", probe.get("opaqueTypes", []))) & atoms)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tests").mkdir(exist_ok=True)
    files = {
        "dinkster-pack.toml": _emit_manifest(pack_name, module_name, source_label),
        f"{module_name}.py": _emit_module(
            pack_name, short_label, ported, node_payloads, opaque_types
        ),
        "pyproject.toml": _emit_pyproject(pack_name, short_label),
        "README.md": _emit_readme(pack_name, source_label, ported, skipped, opaque_types),
        "tests/conftest.py": _CONFTEST,
        "tests/test_pack.py": _emit_tests(module_name, ported),
    }
    for relative, content in files.items():
        (out_dir / relative).write_text(content, encoding="utf-8")
    return PortResult(
        out_dir=out_dir,
        ported=tuple(ported),
        skipped=skipped,
        opaque_types=tuple(opaque_types),
    )


# --- CLI ---------------------------------------------------------------------


def _run_probe(pack_path: Path, comfyui_root: Path, *, python: str | None = None) -> dict[str, Any]:
    """Probe under the same environment the compat workers use: the
    ComfyUI install's own interpreter (packs import torch and friends),
    with Dinkster's pure-stdlib packages reachable from source."""
    interpreter = comfy_python(comfyui_root, python)
    try:
        preflight_interpreter(interpreter)
    except InterpreterPreflightError as exc:
        raise PortError(f"probe interpreter refused: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix="dinkster-port-") as tmp:
        out_json = Path(tmp) / "probe.json"
        env = dict(os.environ)
        env["DINKSTER_COMFYUI_ROOT"] = str(comfyui_root)
        pythonpath = dinkster_pythonpath(find_compat_manifest())
        if pythonpath:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{existing}" if existing else pythonpath
        try:
            proc = subprocess.run(
                [
                    interpreter,
                    "-m",
                    "dinkster_compat_comfy.port_probe",
                    str(pack_path),
                    str(out_json),
                ],
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_S,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise PortError(f"probe exceeded {_PROBE_TIMEOUT_S:.0f}s importing the pack") from exc
        if proc.returncode != 0:
            raise PortError(f"probe failed:\n{proc.stderr.strip()[-1500:]}")
        if not out_json.is_file():
            raise PortError("probe exited cleanly but wrote no report")
        return cast("dict[str, Any]", json.loads(out_json.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dinkster-port",
        description=(
            "Generate a native Dinkster pack skeleton from a legacy ComfyUI custom-node pack."
        ),
    )
    parser.add_argument(
        "pack",
        help="legacy pack path: a directory containing __init__.py, or a single .py file",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="native pack name (lowercase segments separated by '-', '_' or '.')",
    )
    parser.add_argument("--out", help="output directory (default: ./<name>)")
    parser.add_argument(
        "--comfyui-root",
        help="ComfyUI install the pack was written against (default: DINKSTER_COMFYUI_ROOT)",
    )
    parser.add_argument(
        "--comfy-python",
        default="",
        metavar="PATH",
        help="interpreter for the probe subprocess "
        "(default: $DINKSTER_COMFYUI_PYTHON, else <comfy-root>/venv/bin/python)",
    )
    parser.add_argument(
        "--nodes",
        help="comma-separated source node names to port (default: everything translatable)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write into an existing non-empty output directory",
    )
    args = parser.parse_args(argv)

    try:
        pack_name = cast("str", args.name)
        problem = validate_name(pack_name)
        if problem is not None:
            raise PortError(f"--name {pack_name!r} {problem}")
        root_claim = reserved_root(pack_name)
        if root_claim is not None:
            raise PortError(
                f"--name {pack_name!r} falls under the reserved root "
                f"'{root_claim}' (std, comfy, core, dinkster); pick a name of "
                "your own"
            )
        pack_path = Path(cast("str", args.pack))
        if not pack_path.exists():
            raise PortError(f"legacy pack path does not exist: {pack_path}")
        out_dir = Path(cast("str | None", args.out) or pack_name)
        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            raise PortError(
                f"output directory {out_dir} exists and is not empty; pass --force to write into it"
            )
        only = None
        if args.nodes:
            only = [n.strip() for n in cast("str", args.nodes).split(",") if n.strip()]
        root_text = cast("str | None", args.comfyui_root) or os.environ.get(
            "DINKSTER_COMFYUI_ROOT", ""
        )
        if not root_text:
            raise PortError(
                "no ComfyUI installation: pass --comfyui-root or set "
                "DINKSTER_COMFYUI_ROOT (the probe imports the pack in the "
                "environment it was written against)"
            )
        probe = _run_probe(
            pack_path,
            Path(root_text),
            python=cast("str", args.comfy_python) or None,
        )
        result = generate_pack(probe, pack_name, out_dir, only=only, source_label=str(pack_path))
    except PortError as exc:
        print(f"dinkster-port: {exc}", file=sys.stderr)
        return 1

    print(f"dinkster-port: {pack_name}: {len(result.ported)} node(s) -> {result.out_dir}")
    for node in result.ported:
        print(f"  {node.node_type}  ({node.source_api} {node.source_name!r})")
    if result.opaque_types:
        print("  opaque types needing codecs: " + ", ".join(result.opaque_types))
    if result.skipped:
        print(f"  skipped by the translator ({len(result.skipped)}):")
        for name, reason in sorted(result.skipped.items()):
            print(f"    {name}: {reason}")
    print(f"next: work the TODO(port) markers, then run 'dinkster-doctor {result.out_dir}'")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
