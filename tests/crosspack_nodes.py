"""CPU resident model fixtures for cross-pack composition."""

from dinkster_inference_wire.types import MODEL_TYPE_ID
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr

MODEL = TypeExpr.concrete(MODEL_TYPE_ID)
FLOAT = TypeExpr.concrete("FLOAT")


class ModelSource(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="source.model",
            inputs=(InputSpec("weight", FLOAT),),
            outputs=(OutputSpec("model", MODEL),),
        )

    @classmethod
    def execute(cls, weight):
        return cls.outputs(model={"diffusion_model.weight": weight})


class ModelMerge(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="merge.simple",
            inputs=(InputSpec("a", MODEL), InputSpec("b", MODEL), InputSpec("ratio", FLOAT)),
            outputs=(OutputSpec("model", MODEL),),
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
            inputs=(InputSpec("model", MODEL),),
            outputs=(OutputSpec("weight", FLOAT),),
        )

    @classmethod
    def execute(cls, model):
        return cls.outputs(weight=model["diffusion_model.weight"])


SOURCE_NODES = [ModelSource, ModelWeight]
MERGE_NODES = [ModelMerge]
