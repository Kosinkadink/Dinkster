"""Torch-free autoregressive sampler schema and discovery."""

from dinkster_api.v1 import (
    CORE_INT,
    SAMPLER_TYPE,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_misc_models import NODES
from dinkster_misc_models.ar_video import SamplerARVideo


def test_ar_video_sampler_schema_and_cold_discovery() -> None:
    schema = SamplerARVideo.schema()
    assert schema.node_type == "dinkster.misc.models.sampler_ar_video"
    assert schema.display_name == "Sampler AR Video"
    assert schema.category == "model/sampling/samplers"
    (input_,) = schema.inputs
    assert input_.id == "num_frame_per_block"
    assert input_.type == TypeExpr.concrete(CORE_INT)
    assert input_.default == 1
    assert input_.widget == NumberWidget(min=1, max=64)
    assert input_.doc == (
        "Frames per autoregressive block. 1 = framewise, "
        "3 = chunkwise. Must match the checkpoint's training mode."
    )
    assert schema.outputs == (OutputSpec("SAMPLER", TypeExpr.concrete(SAMPLER_TYPE)),)
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert NODES.count(SamplerARVideo) == 1
