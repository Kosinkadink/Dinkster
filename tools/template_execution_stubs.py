"""Explicit CPU computation substitutes for intact official workflow tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from dinkster_inference_wire.conditioning import ConditioningSet
from dinkster_inference_wire.conditioning_wire import make_conditioning_carrier
from dinkster_inference_wire.sampling_wire import (
    BuiltinSamplerSelection,
    NoiseSelection,
    SigmaSchedule,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_schema import (
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicEntry,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)
from dinkster_values import TypeRegistry

CONTROLS = {
    "ComfySwitchNode",
    "StartLoop",
    "EndLoop",
    "LoopIteration",
    "LoopProgress",
    "LoopResult",
}
PURE = CONTROLS | {
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveBoolean",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "ComfyMathExpression",
    "ComfyNumberConvert",
    "StringFormat",
    "StringLength",
    "RegexExtract",
    "RegexReplace",
    "ConvertDictionaryToString",
    "ConvertArrayToString",
}
NODES: list[type[Node]] = []
STUBBED: list[str] = []
_TYPES: set[str] = set()


def _input(name: str, descriptor: Sequence[Any], required: bool) -> DynamicEntry:
    atom = descriptor[0]
    options = descriptor[1] if len(descriptor) > 1 else {}
    default = options.get("default")
    widget = None
    if atom == "COMFY_MATCHTYPE_V3":
        template = options["template"]
        names = template.get("allowed_types", "").split(",")
        _TYPES.update(name for name in names if name)
        return InputSpec(name, TypeExpr.variable(template["template_id"]), required=required)
    if atom == "COMFY_AUTOGROW_V3":
        template = options["template"]
        entries = template["input"].get("required", {})
        item_name, item_descriptor = next(iter(entries.items()))
        item = _input(item_name, item_descriptor, True)
        assert isinstance(item, InputSpec)
        return InputFamilySpec(
            name,
            item.type,
            min_members=template.get("min", 0),
            member_names=tuple(template["names"]) if "names" in template else None,
            member_prefix=template.get("prefix"),
        )
    if atom == "COMFY_DYNAMICCOMBO_V3":
        return DynamicComboSpec(
            name,
            tuple(
                DynamicComboOption(
                    option["key"],
                    tuple(
                        _input(key, value, section == "required")
                        for section in ("required", "optional")
                        for key, value in option["inputs"].get(section, {}).items()
                    ),
                )
                for option in options["options"]
            ),
            default=default,
            required=required,
        )
    if isinstance(atom, list) or atom == "COMBO":
        choices = tuple(
            str(item) for item in (atom if isinstance(atom, list) else options["options"])
        )
        # File inventories are empty on the reference CPU host. A file name
        # remains a COMBO string; the stub never opens the selected file.
        widget = ComboWidget(options=choices or ("test-fixture",))
        atom = "COMBO"
        default = default if default is not None else widget.options[0]
    elif atom in ("INT", "FLOAT"):
        if not options.get("forceInput"):
            widget = NumberWidget(
                step=options.get("step", 1 if atom == "INT" else 0.01),
                control_after_generate="fixed" if options.get("control_after_generate") else None,
            )
    elif atom == "STRING":
        widget = StringWidget(multiline=bool(options.get("multiline")))
    elif atom == "*":
        return InputSpec(name, TypeExpr.variable("value"), required=required)
    if "," in atom:
        names = tuple(part.strip() for part in atom.split(","))
        _TYPES.update(names)
        return InputSpec(name, TypeExpr.union(*names), required=required, default=default)
    _TYPES.add(atom)
    return InputSpec(
        name,
        TypeExpr.concrete(atom),
        required=required,
        default=default,
        widget=widget,
        force_input=bool(options.get("forceInput")),
    )


def _stub(row: Mapping[str, Any]) -> type[Node]:
    node_type = row["class_name"]
    entries = tuple(
        _input(name, descriptor, section == "required")
        for section in ("required", "optional")
        for name, descriptor in row["input_types"].get(section, {}).items()
    )
    inputs = tuple(item for item in entries if isinstance(item, InputSpec))
    returns = row.get("return_types") or []
    names = row.get("return_names") or returns
    output_metadata = row.get("v3_outputs") or [{} for _ in returns]
    outputs = tuple(
        OutputSpec(
            name or atom,
            TypeExpr.variable(metadata["template"]["template_id"])
            if atom == "COMFY_MATCHTYPE_V3"
            else TypeExpr.concrete(atom),
        )
        for name, atom, metadata in zip(names, returns, output_metadata, strict=True)
    )
    _TYPES.update(atom for atom in returns if atom != "COMFY_MATCHTYPE_V3")
    schema = NodeSchema(
        node_type,
        inputs=inputs,
        input_families=tuple(item for item in entries if isinstance(item, InputFamilySpec)),
        combos=tuple(item for item in entries if isinstance(item, DynamicComboSpec)),
        input_order=tuple(item.id for item in entries),
        outputs=outputs,
        output_node=bool(row.get("output_node")),
    )

    def define_schema(cls: type[Node]) -> NodeSchema:
        return schema

    def execute(cls: type[Node], **values: object) -> Mapping[str, object]:
        result: dict[str, object] = {}
        for output, atom in zip(outputs, returns, strict=True):
            matching = next(
                (
                    values[item.id]
                    for item in inputs
                    if item.id in values and item.type == output.type
                ),
                None,
            )
            if atom == "INT":
                value: object = 2
            elif atom == "FLOAT":
                value = 1.0
            elif atom == "BOOLEAN":
                value = True
            elif atom in ("STRING", "COMBO"):
                value = f"fixture:{node_type}"
            elif atom == "CONDITIONING":
                value = make_conditioning_carrier(ConditioningSet(), ())
            elif atom == "IMAGE":
                value = np.zeros((1, 2, 2, 3), dtype=np.float32)
            elif atom == "MASK":
                value = np.zeros((1, 2, 2), dtype=np.float32)
            elif atom == "LATENT":
                value = {"samples": np.zeros((1, 4, 2, 2), dtype=np.float32)}
            elif atom == "SIGMAS":
                value = SigmaSchedule((1.0, 0.0))
            elif atom == "NOISE":
                value = NoiseSelection(1)
            elif atom == "SAMPLER":
                value = BuiltinSamplerSelection("euler", ())
            else:
                value = matching if matching is not None else {"fixtureType": atom}
            result[output.id] = value
        return result

    return type(
        node_type,
        (Node,),
        {
            "define_schema": classmethod(define_schema),
            "execute": classmethod(execute),
        },
    )


def configure(reference: Path, templates: Sequence[Path]) -> None:
    """Build one ordinary pack; workflow subgraph ids are never node stubs."""
    rows = {row["class_name"]: row for row in json.loads(reference.read_text())["nodes"]}
    needed = set(CONTROLS)
    for template in templates:
        workflow = json.loads(template.read_text())
        definitions = workflow.get("definitions", {}).get("subgraphs", [])
        subgraphs = {graph["id"] for graph in definitions}
        needed.update(
            node["type"]
            for graph in (workflow, *definitions)
            for node in graph["nodes"]
            if node["type"] not in subgraphs
            and node["type"] not in ("Note", "MarkdownNote", "Reroute")
        )
    real = {node.schema().node_type: node for node in COMFY_NODES}
    NODES.clear()
    STUBBED.clear()
    _TYPES.clear()
    NODES.extend(
        node
        for node in FOUNDATION_NODES
        if node.schema().node_type
        in {
            "std.list.make",
            "std.list.concat",
            "std.list.element",
            "std.list.length",
            "std.list.range",
            "std.math.add_ints",
            "dinkster.value.compare",
            "dinkster.value.select",
        }
    )
    for name in sorted(needed):
        if name in {"CreateList", "GetItemFromList"}:
            continue
        if name in PURE:
            if name not in real:
                raise RuntimeError(f"template requires native computation node {name}")
            NODES.append(real[name])
        else:
            NODES.append(_stub(rows[name]))
            STUBBED.append(name)


def register_types(registry: TypeRegistry) -> None:
    for name in sorted(_TYPES):
        if name not in registry:
            registry.register(name)
