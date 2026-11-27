"""CPU resident model fixtures for cross-pack composition."""

from dinkster_inference_wire.types import MODEL_TYPE_ID
from dinkster_schema import Input, Node, NodeSchema, Output, TypeExpr

MODEL = TypeExpr.concrete(MODEL_TYPE_ID)
FLOAT = TypeExpr.concrete("FLOAT")


class ModelSource(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="source.model",
            body=(Input("weight", FLOAT), Output("model", MODEL)),
        )

    @classmethod
    def execute(cls, weight):
        return cls.outputs(model={"diffusion_model.weight": weight})


class ModelMerge(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="merge.simple",
            body=(
                Input("a", MODEL),
                Input("b", MODEL),
                Input("ratio", FLOAT),
                Output("model", MODEL),
            ),
        )

    @classmethod
    def execute(cls, a, b, ratio):
        key = "diffusion_model.weight"
        return cls.outputs(model={key: a[key] * ratio + b[key] * (1 - ratio)})


class ModelWeight(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="source.weight",
            body=(Input("model", MODEL), Output("weight", FLOAT)),
        )

    @classmethod
    def execute(cls, model):
        return cls.outputs(weight=model["diffusion_model.weight"])


SOURCE_NODES = [ModelSource, ModelWeight]
MERGE_NODES = [ModelMerge]
