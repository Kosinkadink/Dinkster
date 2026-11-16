import json
import os
from importlib.metadata import version

from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class Inspect(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="inspector.loaded_model",
            inputs=(InputSpec("model", TypeExpr.concrete("MODEL")),),
            outputs=(OutputSpec("summary", TypeExpr.concrete("STRING")),),
        )

    @classmethod
    def execute(cls, *, model):
        from dinkster_inference.model_patcher import ModelPatcher

        assert isinstance(model, ModelPatcher), type(model)
        parameters = list(model.model.parameters())
        count = sum(p.numel() for p in parameters)
        assert count > 800_000_000, count
        first = parameters[0]
        sample = float(first.flatten()[0].item())
        return cls.outputs(
            summary=json.dumps(
                {
                    "pid": os.getpid(),
                    "dependency": version("humanize"),
                    "class": type(model).__qualname__,
                    "parameters": count,
                    "sample": sample,
                    "device": str(first.device),
                    "shape": list(first.shape),
                }
            )
        )


NODES = [Inspect]
