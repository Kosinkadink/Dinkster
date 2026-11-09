"""Torch-free Hunyuan image latent node interfaces."""

from typing import cast

from dinkster_api.v1 import NumberWidget, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.hunyuan_latent import HUNYUAN_LATENT_NODES


def test_hunyuan_image_latent_schemas_match_the_fixed_reference_contract() -> None:
    empty, refiner = HUNYUAN_LATENT_NODES
    empty_schema = empty.schema()
    assert empty in COMFY_NODES
    assert empty_schema.node_type == "EmptyHunyuanImageLatent"
    assert empty_schema.display_name == "Empty Hunyuan Image Latent"
    assert empty_schema.category == "model/latent/hunyuan image"
    assert [(input_spec.id, input_spec.default) for input_spec in empty_schema.inputs] == [
        ("width", 2048),
        ("height", 2048),
        ("batch_size", 1),
    ]
    assert [
        (
            cast(NumberWidget, input_spec.widget).min,
            cast(NumberWidget, input_spec.widget).max,
            cast(NumberWidget, input_spec.widget).step,
        )
        for input_spec in empty_schema.inputs
    ] == [(64, 16384, 32), (64, 16384, 32), (1, 4096, None)]
    assert [(output.id, output.type.types) for output in empty_schema.outputs] == [
        ("LATENT", ("LATENT",))
    ]

    refiner_schema = refiner.schema()
    assert refiner in COMFY_NODES
    assert refiner_schema.node_type == "HunyuanRefinerLatent"
    assert refiner_schema.display_name == "Hunyuan Latent Refiner"
    assert refiner_schema.category == "model/conditioning/hunyuan video"
    assert [(input_spec.id, input_spec.default) for input_spec in refiner_schema.inputs] == [
        ("positive", None),
        ("negative", None),
        ("latent", None),
        ("noise_augmentation", 0.1),
    ]
    noise = refiner_schema.inputs[-1]
    assert cast(NumberWidget, noise.widget) == NumberWidget(min=0.0, max=1.0, step=0.01)
    assert noise.advanced is True
    assert [(output.id, output.type.types) for output in refiner_schema.outputs] == [
        ("positive", ("CONDITIONING",)),
        ("negative", ("CONDITIONING",)),
        ("latent", ("LATENT",)),
    ]
    assert schema_to_wire(empty_schema)
    assert schema_to_wire(refiner_schema)
