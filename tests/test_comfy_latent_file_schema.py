"""Public latent file schemas remain usable without Torch."""

from dinkster_api.v1 import ASSET_TYPE, LATENT_TYPE, MountWidget, TypeExpr
from dinkster_nodes_comfy.latent_file_tools import LoadLatent, SaveLatent


def test_latent_file_slots_preserve_source_order_and_mount_layout() -> None:
    load = LoadLatent.schema()
    assert [(item.id, item.type) for item in load.inputs] == [
        ("latent", TypeExpr.concrete(ASSET_TYPE)),
    ]
    assert load.inputs[0].source_filename is not None
    assert load.inputs[0].source_filename.category == "input"
    assert load.outputs[0].type == TypeExpr.concrete(LATENT_TYPE)
    save = SaveLatent.schema()
    assert [item.id for item in save.inputs] == ["samples", "filename_prefix", "mount"]
    assert save.inputs[1].default == "latents/ComfyUI"
    assert save.inputs[2].widget == MountWidget()
    assert save.inputs[2].adjacent_to == "filename_prefix"
    assert save.inputs[2].default is None and not save.inputs[2].required
    assert save.output_node and not save.idempotent
    assert save.outputs[0].id == "samples"
