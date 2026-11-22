import os
from importlib.metadata import version

from dinkster_api.v1 import Input, Node, NodeSchema, Output, TypeExpr


class Touch(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="versionconflict.loaded_model",
            inputs=(Input("model", TypeExpr.concrete("MODEL")),),
            outputs=(Output("value", TypeExpr.concrete("INT")),),
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
                Output("version", TypeExpr.concrete("STRING")),
                Output("pid", TypeExpr.concrete("INT")),
            ),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(version=version("humanize"), pid=os.getpid())


NODES = [Touch, Plain]
