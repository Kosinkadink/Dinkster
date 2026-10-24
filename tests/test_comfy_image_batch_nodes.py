"""Image utility sockets retain their declared scalar workflow types."""

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy.image_dimensions import GetImageSize, GetLatentSize
from dinkster_nodes_comfy.rebatch import RebatchImages, RebatchLatents


def test_image_utility_scalar_socket_contracts_roundtrip() -> None:
    for node, input_name, value_type, output_names in (
        (GetImageSize, "image", IMAGE_TYPE, ("width", "height", "batch_size")),
        (GetLatentSize, "latent", LATENT_TYPE, ("width", "height", "frames", "batch_size")),
        (RebatchImages, "images", IMAGE_TYPE, ("IMAGE",)),
        (RebatchLatents, "latents", LATENT_TYPE, ("LATENT",)),
    ):
        schema = node.schema()
        assert schema.inputs[0].id == input_name
        assert schema.inputs[0].type == TypeExpr.concrete(value_type)
        assert tuple(output.id for output in schema.outputs) == output_names
        expected_type = value_type if node in (RebatchImages, RebatchLatents) else CORE_INT
        assert all(output.type == TypeExpr.concrete(expected_type) for output in schema.outputs)
        assert schema_to_wire(schema_from_wire(schema_to_wire(schema))) == schema_to_wire(schema)
