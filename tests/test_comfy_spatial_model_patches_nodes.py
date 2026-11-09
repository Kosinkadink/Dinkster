"""Public schemas for spatial model patch nodes."""

from dinkster_api.v1 import CORE_BOOLEAN, CORE_COMBO, CORE_FLOAT, CORE_INT, MODEL_TYPE, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.spatial_model_patches import SPATIAL_MODEL_PATCH_NODES


def test_spatial_model_patches_use_native_model_types_without_torch() -> None:
    expected = {
        "TomePatchModel": ("model", "ratio"),
        "HyperTile": ("model", "tile_size", "swap_size", "max_depth", "scale_depth"),
        "PatchModelAddDownscale": (
            "model",
            "block_number",
            "downscale_factor",
            "start_percent",
            "end_percent",
            "downscale_after_skip",
            "downscale_method",
            "upscale_method",
        ),
    }
    assert {node.define_schema().node_type for node in SPATIAL_MODEL_PATCH_NODES} == set(expected)
    for node in SPATIAL_MODEL_PATCH_NODES:
        schema = node.define_schema()
        assert node in COMFY_NODES
        assert tuple(input_spec.id for input_spec in schema.inputs) == expected[schema.node_type]
        assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert schema.outputs[0].type == TypeExpr.concrete(MODEL_TYPE)

    tome, hypertile, downscale = (node.define_schema() for node in SPATIAL_MODEL_PATCH_NODES)
    assert tome.inputs[1].type == TypeExpr.concrete(CORE_FLOAT)
    assert [input_spec.type for input_spec in hypertile.inputs[1:]] == [
        TypeExpr.concrete(CORE_INT),
        TypeExpr.concrete(CORE_INT),
        TypeExpr.concrete(CORE_INT),
        TypeExpr.concrete(CORE_BOOLEAN),
    ]
    assert downscale.display_name == "PatchModelAddDownscale (Kohya Deep Shrink)"
    assert [input_spec.type for input_spec in downscale.inputs[1:]] == [
        TypeExpr.concrete(CORE_INT),
        TypeExpr.concrete(CORE_FLOAT),
        TypeExpr.concrete(CORE_FLOAT),
        TypeExpr.concrete(CORE_FLOAT),
        TypeExpr.concrete(CORE_BOOLEAN),
        TypeExpr.concrete(CORE_COMBO),
        TypeExpr.concrete(CORE_COMBO),
    ]
