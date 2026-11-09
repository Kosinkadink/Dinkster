"""Numeric conversion, seed, and value inspection nodes."""

from __future__ import annotations

import importlib
import json
import math
import sys
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    report_progress,
)


class ComfyNumberConvert(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ComfyNumberConvert",
            display_name="Convert Number",
            category="utilities",
            inputs=(
                InputSpec(
                    "value",
                    TypeExpr.union(CORE_INT, CORE_FLOAT, CORE_STRING, CORE_BOOLEAN),
                    display_name="value",
                ),
            ),
            outputs=(
                OutputSpec("FLOAT", TypeExpr.concrete(CORE_FLOAT)),
                OutputSpec("INT", TypeExpr.concrete(CORE_INT)),
            ),
        )

    @classmethod
    def execute(cls, *, value: object) -> Mapping[str, object]:
        if isinstance(value, bool):
            float_val = 1.0 if value else 0.0
            int_val = 1 if value else 0
        elif isinstance(value, int):
            float_val = float(value)
            int_val = value
        elif isinstance(value, float):
            float_val = value
            int_val = int(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("Cannot convert empty string to number.")
            try:
                float_val = float(text)
            except ValueError:
                raise ValueError(f"Cannot convert string to number: {value!r}") from None
            if not math.isfinite(float_val):
                raise ValueError(f"Cannot convert non-finite value to number: {float_val}")
            try:
                int_val = int(text)
            except ValueError:
                int_val = int(float_val)
        else:
            raise TypeError(f"Unsupported input type: {type(value).__name__}")

        if not math.isfinite(float_val):
            raise ValueError(f"Cannot convert non-finite value to number: {float_val}")
        return cls.outputs(FLOAT=float_val, INT=int_val)


class SeedNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SeedNode",
            display_name="Seed",
            category="utilities",
            inputs=(
                InputSpec(
                    "seed",
                    TypeExpr.concrete(CORE_INT),
                    widget=NumberWidget(min=0, max=sys.maxsize, control_after_generate="fixed"),
                ),
            ),
            outputs=(OutputSpec("seed", TypeExpr.concrete(CORE_INT)),),
        )

    @classmethod
    def execute(cls, *, seed: int) -> Mapping[str, object]:
        return cls.outputs(seed=seed)


class PreviewAny(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PreviewAny",
            display_name="Preview as Text",
            category="utilities",
            description="Preview any input value as text.",
            inputs=(InputSpec("source", TypeExpr.wildcard()),),
            outputs=(OutputSpec("STRING", TypeExpr.concrete(CORE_STRING), preview=True),),
            output_node=True,
            search_terms=(
                "preview",
                "preview text",
                "show output",
                "inspect",
                "debug",
                "print value",
                "show text",
            ),
        )

    @classmethod
    def execute(cls, *, source: object = None) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        torch.set_printoptions(edgeitems=6)
        value = "None"
        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = json.dumps(source, indent=4, ensure_ascii=False)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = "source exists, but could not be serialized."
        torch.set_printoptions()
        report_progress(1, 1, text=value)
        return cls.outputs(STRING=value)
