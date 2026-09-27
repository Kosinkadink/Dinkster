"""Manifest entry points for native execution."""

from __future__ import annotations

from dinkster_values import TypeRegistry

from .native import NATIVE_NODES, register_native_types

ARM_NODES = {"native": NATIVE_NODES}


def register_types(registry: TypeRegistry) -> None:
    register_native_types(registry)
