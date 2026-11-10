"""Torch-free autoregressive start-image conditioning contract."""

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    NumberWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_misc_models import NODES
from dinkster_misc_models.ar_video import ARVideoI2V


def test_ar_video_conditioning_schema_and_cold_discovery() -> None:
    schema = ARVideoI2V.schema()
    assert schema.node_type == "dinkster.misc.models.ar_video_i2_v"
    assert schema.category == "model/conditioning/autoregressive"
    assert [(i.id, i.type) for i in schema.inputs[:3]] == [
        ("model", TypeExpr.concrete(MODEL_TYPE)),
        ("vae", TypeExpr.concrete(VAE_TYPE)),
        ("start_image", TypeExpr.concrete(IMAGE_TYPE)),
    ]
    assert [(i.id, i.type, i.default, i.widget) for i in schema.inputs[3:]] == [
        ("width", TypeExpr.concrete(CORE_INT), 832, NumberWidget(min=16, max=8192, step=16)),
        ("height", TypeExpr.concrete(CORE_INT), 480, NumberWidget(min=16, max=8192, step=16)),
        ("length", TypeExpr.concrete(CORE_INT), 81, NumberWidget(min=1, max=1024, step=4)),
        ("batch_size", TypeExpr.concrete(CORE_INT), 1, NumberWidget(min=1, max=64)),
    ]
    assert [(o.id, o.type) for o in schema.outputs] == [
        ("MODEL", TypeExpr.concrete(MODEL_TYPE)),
        ("LATENT", TypeExpr.concrete(LATENT_TYPE)),
    ]
    assert all(i.required for i in schema.inputs)
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert NODES.count(ARVideoI2V) == 1
