import os

from dinkster_api.v1 import Input, Node, NodeSchema, Output, TypeExpr


class Touch(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="refused.loaded_model",
            body=(
                Input("model", TypeExpr.concrete("MODEL")),
                Output("value", TypeExpr.concrete("INT")),
            ),
        )

    @classmethod
    def execute(cls, *, model):
        raise AssertionError("Refused resident node must not execute")


class Plain(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="refused.plain",
            body=(
                Output("value", TypeExpr.concrete("INT")),
                Output("pid", TypeExpr.concrete("INT")),
            ),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(value=37, pid=os.getpid())


NODES = [Touch, Plain]
