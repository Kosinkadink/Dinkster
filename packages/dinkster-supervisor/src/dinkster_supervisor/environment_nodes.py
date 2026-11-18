"""Managed node exclusions, independent of retained software and pack admission."""

from __future__ import annotations

from typing import cast

from .environment_state import EnvironmentError


def read_node_exclusions(value: object) -> list[str]:
    if not isinstance(value, list):
        raise EnvironmentError("excludedNodeTypes must be an array of node type ids")
    nodes = cast("list[object]", value)
    if any(not isinstance(node, str) or not node or node.strip() != node for node in nodes):
        raise EnvironmentError(
            "excludedNodeTypes must contain nonempty node type ids without surrounding whitespace"
        )
    selected = cast("list[str]", nodes)
    if len(set(selected)) != len(selected):
        raise EnvironmentError("excludedNodeTypes must contain distinct node type ids")
    return list(selected)
