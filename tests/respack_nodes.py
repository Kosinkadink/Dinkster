"""Test pack for worker-resident value tests. Imported by the worker HOST
process (via a manifest entry), never by the test process's engine side.

``res.heavy`` values contain a threading.Lock, so any attempt to move them
across the boundary with the default pickle codec would raise - the tests
pass only if the resident codec keeps the object in this process and sends
a stub instead (the compat pack's MODEL/CLIP/VAE story, minus torch).
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from dinkster_compat_comfy import register_resident_type
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import CORE_BOOLEAN, CORE_INT, CORE_STRING, TypeRegistry

HEAVY = TypeExpr.concrete("res.heavy")
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


def register_types(registry: TypeRegistry) -> None:
    register_resident_type(registry, "res.heavy")


class HeavyThing:
    """Stands in for a loaded model: identity matters, pickling must not."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.lock = threading.Lock()  # unpicklable on purpose


class HeavyLoad(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="res.load",
            display_name="Heavy Load",
            category="test",
            inputs=(InputSpec("token", STRING, default="secret"),),
            outputs=(OutputSpec("heavy", HEAVY), OutputSpec("oid", INT)),
        )

    @classmethod
    def execute(cls, *, token: str) -> Mapping[str, object]:
        thing = HeavyThing(token)
        return cls.outputs(heavy=thing, oid=id(thing))


class HeavyUse(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="res.use",
            display_name="Heavy Use",
            category="test",
            inputs=(InputSpec("heavy", HEAVY), InputSpec("oid", INT)),
            outputs=(OutputSpec("same", BOOLEAN), OutputSpec("token", STRING)),
        )

    @classmethod
    def execute(cls, *, heavy: object, oid: int) -> Mapping[str, object]:
        assert isinstance(heavy, HeavyThing)
        return cls.outputs(same=id(heavy) == oid, token=heavy.token)


NODES = (HeavyLoad, HeavyUse)
