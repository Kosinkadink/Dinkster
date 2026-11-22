import json
import os

from dinkster_api.v1 import Input, Node, NodeSchema, Output, TypeExpr


class Inspect(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="multipart-model.loaded_model",
            inputs=(
                Input("model", TypeExpr.concrete("MODEL")),
                Input("value", TypeExpr.concrete("INT")),
                Input("producer_pid", TypeExpr.concrete("INT")),
            ),
            outputs=(Output("summary", TypeExpr.concrete("STRING")),),
        )

    @classmethod
    def execute(cls, *, model, value, producer_pid):
        from dinkster_inference.model_patcher import ModelPatcher

        assert isinstance(model, ModelPatcher), type(model)
        assert value == 37, value
        assert producer_pid != os.getpid(), producer_pid
        parameters = list(model.model.parameters())
        count = sum(parameter.numel() for parameter in parameters)
        assert count > 800_000_000, count
        first = parameters[0]
        return cls.outputs(
            summary=json.dumps(
                {
                    "pid": os.getpid(),
                    "producerPid": producer_pid,
                    "value": value,
                    "class": type(model).__qualname__,
                    "parameters": count,
                    "sample": float(first.flatten()[0].item()),
                    "shape": list(first.shape),
                }
            )
        )


NODES = [Inspect]
