import os
from importlib.metadata import version

from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class Touch(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="versionconflict.loaded_model",
            inputs=(InputSpec("model", TypeExpr.concrete("MODEL")),),
            outputs=(OutputSpec("value", TypeExpr.concrete("INT")),),
        )

    @classmethod
    def execute(cls, *, model):
        raise AssertionError("Conflicting resident node must not execute")


class Plain(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="versionconflict.plain",
            inputs=(),
            outputs=(
                OutputSpec("version", TypeExpr.concrete("STRING")),
                OutputSpec("pid", TypeExpr.concrete("INT")),
            ),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(version=version("humanize"), pid=os.getpid())


NODES = [Touch, Plain]
