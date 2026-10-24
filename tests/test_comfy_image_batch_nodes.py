"""Image dimensions use scalar sockets; rebatching uses explicit domain lists."""

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    NumberWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy.image_dimensions import GetImageSize, GetLatentSize
from dinkster_nodes_comfy.rebatch import RebatchImages, RebatchLatents


def test_image_utility_socket_contracts_roundtrip() -> None:
    for node, input_name, value_type, output_names in (
        (GetImageSize, "image", IMAGE_TYPE, ("width", "height", "batch_size")),
        (GetLatentSize, "latent", LATENT_TYPE, ("width", "height", "frames", "batch_size")),
        (RebatchImages, "images", IMAGE_TYPE, ("IMAGE",)),
        (RebatchLatents, "latents", LATENT_TYPE, ("LATENT",)),
    ):
        schema = node.schema()
        assert schema.inputs[0].id == input_name
        assert tuple(output.id for output in schema.outputs) == output_names
        if node in (RebatchImages, RebatchLatents):
            domain_type = TypeExpr.list_of(TypeExpr.concrete(value_type))
            assert schema.inputs[0].type == domain_type
            assert schema.outputs[0].type == domain_type
            size = schema.inputs[1]
            assert size.id == "batch_size" and size.type == TypeExpr.concrete(CORE_INT)
            assert size.default == 1
            assert isinstance(size.widget, NumberWidget)
            assert size.widget.min == 1 and size.widget.max == 4096
        else:
            assert schema.inputs[0].type == TypeExpr.concrete(value_type)
            assert all(output.type == TypeExpr.concrete(CORE_INT) for output in schema.outputs)
        assert schema_to_wire(schema_from_wire(schema_to_wire(schema))) == schema_to_wire(schema)
