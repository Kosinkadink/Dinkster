"""Node authoring surface (hazard H9: the boundary is free for node authors).

A standard node is define_schema() plus an execute() that takes and returns
plain Python values. Outputs are returned as a mapping keyed by output id -
never positional tuples. execute() may be sync or async.
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .model import NodeSchema


class NodeOutputError(ValueError):
    """execute() returned outputs that do not match the declared schema."""


@dataclass(frozen=True)
class AbsentOutput:
    """Deliberate "no value" for an output declared ``optional=True``
    (DESIGN 3.15): ``return cls.outputs(vae=ABSENT)`` or
    ``cls.outputs(vae=AbsentOutput("model has no VAE"))``. The worker turns
    it into a typed absent envelope carrying this node as the origin;
    returning it for a non-optional output is a contract error."""

    reason: str = ""


ABSENT = AbsentOutput()
"""The bare marker for reason-less absence."""


class Node:
    """Base class for node authoring. Subclass, implement both classmethods."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        raise NotImplementedError

    @classmethod
    def execute(
        cls, *args: Any, **inputs: Any
    ) -> Mapping[str, object] | Awaitable[Mapping[str, object]]:
        # (*args: Any, **kwargs: Any) is deliberate: type checkers treat it
        # as a gradual signature, so concrete nodes declare plain named
        # parameters (sync or async) without tripping override checks -
        # zero typing ceremony for node authors (hazard H9). Nodes are only
        # ever invoked with their schema-declared inputs by the worker shim,
        # never polymorphically through this base signature.
        raise NotImplementedError

    @classmethod
    def schema(cls) -> NodeSchema:
        """The class's schema, computed once and cached per subclass."""
        cached = cls.__dict__.get("_cached_schema")
        if cached is None:
            cached = cls.define_schema()
            cls._cached_schema = cached
        return cached

    @classmethod
    def outputs(cls, **values: object) -> Mapping[str, object]:
        """Return helper: build the output mapping, validated against the
        schema at the return site. ``return cls.outputs(sum=a + b)`` fails
        here, on the offending line, if an output id is missing or typo'd -
        not later at the worker shim. Sugar only: returning a plain mapping
        stays equally valid (the mapping is the canonical representation).

        Output families are returned as one nested mapping per family:
        ``cls.outputs(parts={"a": ..., "b": ...})``. Exact member checking
        against the document's elaborated membership happens at the worker
        (the base schema cannot know which members exist)."""
        schema = cls.schema()
        static = {out.id for out in schema.outputs}
        families = {fam.id for fam in schema.output_families}
        returned = set(values)
        problems: list[str] = []
        missing = static - returned
        extra = returned - static - families
        if missing:
            problems.append("missing outputs: " + ", ".join(sorted(missing)))
        if extra:
            problems.append("undeclared outputs: " + ", ".join(sorted(extra)))
        for family_id in families & returned:
            if not isinstance(values[family_id], Mapping):
                problems.append(
                    f"output family '{family_id}' must be a mapping of "
                    f"member suffix -> value, got {type(values[family_id]).__name__}"
                )
        if problems:
            raise NodeOutputError(f"{schema.node_type}: " + "; ".join(problems))
        return values


def build_schemas(node_classes: Iterable[type[Node]]) -> dict[str, NodeSchema]:
    """Collect schemas from node classes, keyed by node_type."""
    schemas: dict[str, NodeSchema] = {}
    for cls in node_classes:
        schema = cls.schema()
        if schema.node_type in schemas:
            raise ValueError(f"duplicate node type: {schema.node_type}")
        schemas[schema.node_type] = schema
    return schemas


def build_node_types(node_classes: Iterable[type[Node]]) -> dict[str, type[Node]]:
    """Map node_type -> class, using each class's own schema declaration."""
    node_types: dict[str, type[Node]] = {}
    for cls in node_classes:
        node_type = cls.schema().node_type
        if node_type in node_types:
            raise ValueError(f"duplicate node type: {node_type}")
        node_types[node_type] = cls
    return node_types
