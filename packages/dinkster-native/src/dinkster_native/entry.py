"""Manifest entry points for native execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_schema import Node
from dinkster_values import TypeRegistry

from .native import NATIVE_NODES as BASE_NATIVE_NODES
from .native import register_native_types
from .native_arm import GENERATION_PROVIDER_NODES, NATIVE_ARM_NODES, NATIVE_SCHEDULING_NODES
from .native_catalog import COMFY_RUNTIME_NODE_IDS

_default_nodes = (*BASE_NATIVE_NODES, *NATIVE_SCHEDULING_NODES, *GENERATION_PROVIDER_NODES)
NATIVE_NODES: tuple[type[Node], ...] = tuple(
    {
        node.schema().node_type: node
        for node in (*_default_nodes, *NATIVE_ARM_NODES)
        if node.schema().node_type not in COMFY_RUNTIME_NODE_IDS
    }.values()
)
ARM_NODES = {
    "native": tuple(
        node for node in NATIVE_ARM_NODES if node.schema().node_type not in COMFY_RUNTIME_NODE_IDS
    )
}


def combo_choices() -> Mapping[str, Sequence[str]]:
    from .native import SAMPLER_CHOICES, SCHEDULER_CHOICES

    return {"comfy.samplers": SAMPLER_CHOICES, "comfy.schedulers": SCHEDULER_CHOICES}


def register_types(registry: TypeRegistry) -> None:
    register_native_types(registry)
