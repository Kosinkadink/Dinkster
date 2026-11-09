"""Public schemas for native model-specific scheduler nodes."""

from dinkster_api.v1 import SIGMAS_TYPE, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.model_specific_schedulers import MODEL_SPECIFIC_SCHEDULER_NODES


def test_model_specific_schedulers_are_registered_with_exact_public_schemas() -> None:
    expected = {
        "AlignYourStepsScheduler": ("model_type", "steps", "denoise"),
        "GITSScheduler": ("coeff", "steps", "denoise"),
        "OptimalStepsScheduler": ("model_type", "steps", "denoise"),
        "Ideogram4Scheduler": ("steps", "width", "height", "mu", "std"),
    }

    for node in MODEL_SPECIFIC_SCHEDULER_NODES:
        schema = node.schema()
        assert schema.node_type in expected
        assert tuple(input.id for input in schema.inputs) == expected[schema.node_type]
        assert tuple(output.id for output in schema.outputs) == ("SIGMAS",)
        assert schema.outputs[0].type == TypeExpr.concrete(SIGMAS_TYPE)
        assert node in COMFY_NODES
