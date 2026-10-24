"""Torch-free public schema for native Stable Video image conditioning."""

from dinkster_api.v1 import NumberWidget, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.video_conditioning import SVD_img2vid_Conditioning


def test_stable_video_conditioning_schema_matches_comfyui() -> None:
    schema = SVD_img2vid_Conditioning.schema()
    assert SVD_img2vid_Conditioning in COMFY_NODES
    assert schema.node_type == "SVD_img2vid_Conditioning"
    assert schema.display_name == ""
    assert schema.category == "model/conditioning/stable video"
    assert tuple(item.id for item in schema.inputs) == (
        "clip_vision",
        "init_image",
        "vae",
        "width",
        "height",
        "video_frames",
        "motion_bucket_id",
        "fps",
        "augmentation_level",
    )
    assert tuple(item.id for item in schema.outputs) == ("positive", "negative", "latent")
    assert all(item.required for item in schema.inputs)
    assert [item.default for item in schema.inputs] == [
        None,
        None,
        None,
        1024,
        576,
        14,
        127,
        6,
        0.0,
    ]
    assert [item.type.runtime_type_id() for item in schema.inputs] == [
        "CLIP_VISION",
        "IMAGE",
        "VAE",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
    ]
    assert schema.inputs[6].advanced and schema.inputs[8].advanced
    assert not schema.inputs[3].advanced and not schema.inputs[7].advanced
    widgets = [
        widget for item in schema.inputs[3:] if isinstance(widget := item.widget, NumberWidget)
    ]
    assert len(widgets) == 6
    assert [(widget.min, widget.max, widget.step) for widget in widgets] == [
        (16, 16384, 8),
        (16, 16384, 8),
        (1, 4096, None),
        (1, 1023, None),
        (1, 1024, None),
        (0.0, 10.0, 0.01),
    ]
    assert schema_to_wire(schema)
