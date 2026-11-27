import os

from dinkster_api.v1 import Node, NodeSchema, Output, TypeExpr


class Plain(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="multipart-model.plain",
            body=(
                Output("value", TypeExpr.concrete("INT")),
                Output("pid", TypeExpr.concrete("INT")),
            ),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(value=37, pid=os.getpid())


NODES = [Plain]
