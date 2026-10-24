"""Plain-value graph declarations for node authors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ExpansionOutput:
    node_id: str | None
    output_id: str | int


@dataclass(frozen=True)
class Expansion:
    """A finalized local graph and the expanding node's named return values.

    Values are ordinary node outputs or references from ``node.out()``.
    The worker owns wrapping present values and validating declarations.
    """

    nodes: tuple[tuple[str, str, tuple[tuple[str, object], ...], str | None], ...]
    outputs: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


class ExpansionNode:
    """A mutable local node while its builder is being authored."""

    def __init__(self, node_type: str, local_id: str, inputs: Mapping[str, object]) -> None:
        self.node_type = node_type
        self.id = local_id
        self.inputs = dict(inputs)
        self.override_display_id: str | None = None

    def out(self, index: str | int) -> ExpansionOutput:
        return ExpansionOutput(self.id, index)

    def set_input(self, key: str, value: object) -> None:
        if value is None:
            self.inputs.pop(key, None)
        else:
            self.inputs[key] = value

    def get_input(self, key: str) -> object:
        return self.inputs.get(key)

    def set_override_display_id(self, node_id: str | None) -> None:
        self.override_display_id = node_id


class ExpansionBuilder:
    """Build a run-local graph without assigning process-global prefixes."""

    def __init__(self) -> None:
        self.nodes: dict[str, ExpansionNode] = {}
        self._next_id = 1

    def node(self, node_type: str, local_id: str | None = None, **inputs: object) -> ExpansionNode:
        if local_id is None:
            while str(self._next_id) in self.nodes:
                self._next_id += 1
            local_id = str(self._next_id)
            self._next_id += 1
        if local_id in self.nodes:
            return self.nodes[local_id]
        node = ExpansionNode(node_type, local_id, inputs)
        self.nodes[local_id] = node
        return node

    def lookup_node(self, local_id: str) -> ExpansionNode | None:
        return self.nodes.get(local_id)

    @staticmethod
    def current(output_id: str) -> ExpansionOutput:
        """Refer to a present value in this same expansion's parent outputs."""
        return ExpansionOutput(None, output_id)

    def finalize(self, **outputs: object) -> Expansion:
        return Expansion(
            tuple(
                (
                    node.id,
                    node.node_type,
                    tuple(sorted(node.inputs.items())),
                    node.override_display_id,
                )
                for _, node in sorted(self.nodes.items())
            ),
            outputs,
        )
