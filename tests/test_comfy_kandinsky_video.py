"""Torch-free Kandinsky video discovery and public schema."""

from dinkster_api.v1 import schema_from_wire, schema_to_wire
from dinkster_misc_models import NODES
from dinkster_misc_models.kandinsky_video import Kandinsky5ImageToVideo


def test_kandinsky_video_public_schema() -> None:
    schema = Kandinsky5ImageToVideo.schema()
    assert NODES.count(Kandinsky5ImageToVideo) == 1
    assert schema.node_type == "dinkster.misc.models.kandinsky5_image_to_video"
    assert schema.category == "model/conditioning/kandinsky"
    assert tuple(item.id for item in schema.inputs) == (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
    )
    assert tuple(item.default for item in schema.inputs[3:7]) == (768, 512, 121, 1)
    assert not schema.inputs[-1].required
    assert tuple(item.id for item in schema.outputs) == (
        "positive",
        "negative",
        "latent",
        "cond_latent",
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
