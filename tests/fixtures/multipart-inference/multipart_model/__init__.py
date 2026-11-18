import os

from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr


class Plain(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="multipart-model.plain",
            inputs=(),
            outputs=(
                OutputSpec("value", TypeExpr.concrete("INT")),
                OutputSpec("pid", TypeExpr.concrete("INT")),
            ),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(value=37, pid=os.getpid())


NODES = [Plain]
