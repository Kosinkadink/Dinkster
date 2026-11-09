"""Torch-free SeedVR2 output post-processing contract."""

from dinkster_api.v1 import (
    CORE_COMBO,
    IMAGE_TYPE,
    ComboWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy.seedvr_postprocess import SeedVR2PostProcessing


def test_seedvr_postprocess_schema_wire_and_cold_discovery() -> None:
    schema = SeedVR2PostProcessing.schema()
    assert schema.node_type == "SeedVR2PostProcessing"
    assert schema.display_name == "Post-Process SeedVR2 Output"
    assert schema.category == "image/post-processors"
    assert schema.search_terms == (
        "seedvr2",
        "upscale",
        "color correction",
        "color match",
        "postprocess",
    )
    assert [i.id for i in schema.inputs] == [
        "images",
        "original_resized_images",
        "color_correction_method",
    ]
    image = TypeExpr.concrete(IMAGE_TYPE)
    assert all(i.type == image and i.required for i in schema.inputs[:2])
    assert schema.inputs[2].type == TypeExpr.concrete(CORE_COMBO)
    assert schema.inputs[2].widget == ComboWidget(("lab", "wavelet", "adain", "none"))
    assert schema.inputs[2].default == "lab"
    assert schema.outputs[0].id == "images" and schema.outputs[0].type == image
    assert schema_from_wire(schema_to_wire(schema)) == schema
