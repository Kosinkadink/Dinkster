"""Public schemas for native model-specific scheduler nodes."""

from dinkster_api.v1 import SIGMAS_TYPE, TypeExpr
from dinkster_misc_models import NODES as MISC_NODES
from dinkster_misc_models.model_specific_schedulers import Ideogram4Scheduler
from dinkster_sampling import NODES as SAMPLING_NODES
from dinkster_sampling.model_specific_schedulers import (
    AlignYourStepsScheduler,
    GITSScheduler,
    OptimalStepsScheduler,
)


def test_model_specific_schedulers_are_registered_with_exact_public_schemas() -> None:
    expected = {
        "dinkster.sampling.align_your_steps_scheduler": ("model_type", "steps", "denoise"),
        "dinkster.sampling.gits_scheduler": ("coeff", "steps", "denoise"),
        "dinkster.sampling.optimal_steps_scheduler": ("model_type", "steps", "denoise"),
        "dinkster.misc.models.ideogram4_scheduler": ("steps", "width", "height", "mu", "std"),
    }

    for node in (AlignYourStepsScheduler, GITSScheduler, OptimalStepsScheduler, Ideogram4Scheduler):
        schema = node.schema()
        assert schema.node_type in expected
        assert tuple(input.id for input in schema.inputs) == expected[schema.node_type]
        assert tuple(output.id for output in schema.outputs) == ("SIGMAS",)
        assert schema.outputs[0].type == TypeExpr.concrete(SIGMAS_TYPE)
        assert node in (MISC_NODES if node is Ideogram4Scheduler else SAMPLING_NODES)
