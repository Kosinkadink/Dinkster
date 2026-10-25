"""Plain-value graph declarations for node authors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class ExpansionOutput:
    node_id: str | None
    output_id: str | int
    external: bool = False


@dataclass(frozen=True)
class ExpansionNodeDeclaration:
    local_id: str
    node_type: str
    inputs: tuple[tuple[str, object], ...]
    display: tuple[str, bool] | None
    output_members: tuple[tuple[str, tuple[str, ...]], ...]
    slot_variants: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Expansion:
    """A finalized local graph and the expanding node's named return values.

    Values are ordinary node outputs or references from ``node.out()``.
    The worker owns wrapping present values and validating declarations.
    Cache policy inherits the enclosing expansion unless explicitly set;
    reuse allows a nested cached loop inside a rerunning outer loop.
    """

    nodes: tuple[ExpansionNodeDeclaration, ...]
    outputs: Mapping[str, object]
    cache_policy: Literal["inherit", "reuse", "rerun"] = "inherit"

    def __post_init__(self) -> None:
        if self.cache_policy not in ("inherit", "reuse", "rerun"):
            raise ValueError("expansion cache_policy must be inherit, reuse or rerun")
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


class ExpansionNode:
    """A mutable local node while its builder is being authored."""

    def __init__(self, node_type: str, local_id: str, inputs: Mapping[str, object]) -> None:
        self.node_type = node_type
        self.id = local_id
        self.inputs = dict(inputs)
        self.override_display_id: tuple[str, bool] | None = None
        self.output_members: dict[str, tuple[str, ...]] = {}
        self.slot_variants: dict[str, str] = {}

    def out(self, index: str | int) -> ExpansionOutput:
        return ExpansionOutput(self.id, index)

    def set_input(self, key: str, value: object) -> None:
        if value is None:
            self.inputs.pop(key, None)
        else:
            self.inputs[key] = value

    def get_input(self, key: str) -> object:
        return self.inputs.get(key)

    def set_override_display_id(self, node_id: str | None, *, external: bool = False) -> None:
        self.override_display_id = (node_id, external) if node_id is not None else None


class ExpansionBuilder:
    """Build a run-local graph without assigning process-global prefixes."""

    def __init__(self, *, cache_policy: Literal["inherit", "reuse", "rerun"] = "inherit") -> None:
        self.nodes: dict[str, ExpansionNode] = {}
        self._next_id = 1
        self.cache_policy: Literal["inherit", "reuse", "rerun"] = cache_policy

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

    @staticmethod
    def external(node_id: str, output_id: str | int) -> ExpansionOutput:
        """Refer to an existing node in the live graph, outside this fragment."""
        return ExpansionOutput(node_id, output_id, external=True)

    def finalize(self, **outputs: object) -> Expansion:
        return Expansion(
            tuple(
                ExpansionNodeDeclaration(
                    node.id,
                    node.node_type,
                    tuple(sorted(node.inputs.items())),
                    node.override_display_id,
                    tuple(sorted(node.output_members.items())),
                    tuple(sorted(node.slot_variants.items())),
                )
                for _, node in sorted(self.nodes.items())
            ),
            outputs,
            self.cache_policy,
        )
