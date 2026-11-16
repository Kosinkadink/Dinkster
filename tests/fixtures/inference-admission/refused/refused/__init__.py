import os

from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class Touch(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="refused.loaded_model",
            inputs=(InputSpec("model", TypeExpr.concrete("MODEL")),),
            outputs=(OutputSpec("value", TypeExpr.concrete("INT")),),
        )

    @classmethod
    def execute(cls, *, model):
        raise AssertionError("Refused resident node must not execute")


class Plain(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="refused.plain",
            inputs=(),
            outputs=(
                OutputSpec("value", TypeExpr.concrete("INT")),
                OutputSpec("pid", TypeExpr.concrete("INT")),
            ),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(value=37, pid=os.getpid())


NODES = [Touch, Plain]
