"""Torch-free schema for the native empty LTXV audio latent node."""

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    LATENT_TYPE,
    VAE_TYPE,
    NumberWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_ltx import NODES
from dinkster_ltx.ltx_audio import LTXVEmptyLatentAudio


def test_ltx_empty_audio_schema_matches_comfy() -> None:
    assert NODES.count(LTXVEmptyLatentAudio) == 1
    schema = LTXVEmptyLatentAudio.schema()
    assert (schema.node_type, schema.display_name, schema.category) == (
        "dinkster.ltx.ltxv_empty_latent_audio",
        "Empty LTXV Latent Audio",
        "model/latent/ltxv",
    )
    assert tuple((item.id, *item.type.types) for item in schema.inputs) == (
        ("frames_number", CORE_INT),
        ("frame_rate", CORE_FLOAT, CORE_INT),
        ("batch_size", CORE_INT),
        ("audio_vae", VAE_TYPE),
    )
    assert tuple((item.id, item.type.types[0]) for item in schema.outputs) == (
        ("LATENT", LATENT_TYPE),
    )
    assert all(item.required for item in schema.inputs)
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert tuple((item.default, item.doc) for item in schema.inputs[:3]) == (
        (97, "Number of frames."),
        (25.0, "Number of frames per second."),
        (1, "The number of latent audio samples in the batch."),
    )
    assert (
        schema.inputs[3].display_name,
        schema.inputs[3].doc,
        schema.outputs[0].display_name,
    ) == (
        "Audio VAE",
        "The Audio VAE model to get configuration from.",
        "Latent",
    )
    for setting, bounds in zip(
        schema.inputs[:3], ((1, 1000, 1), (1.0, 1000.0, 0.01), (1, 4096, None)), strict=True
    ):
        assert isinstance(setting.widget, NumberWidget)
        assert (
            setting.widget.min,
            setting.widget.max,
            setting.widget.step,
            setting.widget.display,
        ) == (
            *bounds,
            "number",
        )
