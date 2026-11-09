"""Public interfaces for native ComfyUI sigma schedule nodes."""

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    MODEL_TYPE,
    SIGMAS_TYPE,
    NumberWidget,
    TypeExpr,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.sigma_operations import SIGMA_OPERATION_NODES


def test_sigma_operation_interfaces_are_registered_in_canonical_order() -> None:
    assert tuple(node.__name__ for node in SIGMA_OPERATION_NODES) == (
        "SplitSigmas",
        "SplitSigmasDenoise",
        "FlipSigmas",
        "SetFirstSigma",
        "ExtendIntermediateSigmas",
        "SamplingPercentToSigma",
        "ManualSigmas",
    )
    assert all(node in COMFY_NODES for node in SIGMA_OPERATION_NODES)
    assert [node.schema().node_type for node in SIGMA_OPERATION_NODES] == [
        node.__name__ for node in SIGMA_OPERATION_NODES
    ]


def test_sigma_operation_schemas_preserve_core_inputs_outputs_and_widgets() -> None:
    sigmas = TypeExpr.concrete(SIGMAS_TYPE)
    scalar_types = {
        "step": TypeExpr.concrete(CORE_INT),
        "denoise": TypeExpr.concrete(CORE_FLOAT),
        "sigma": TypeExpr.concrete(CORE_FLOAT),
        "steps": TypeExpr.concrete(CORE_INT),
        "start_at_sigma": TypeExpr.concrete(CORE_FLOAT),
        "end_at_sigma": TypeExpr.concrete(CORE_FLOAT),
        "sampling_percent": TypeExpr.concrete(CORE_FLOAT),
        "return_actual_sigma": TypeExpr.concrete(CORE_BOOLEAN),
        "sigmas_text": TypeExpr.concrete(CORE_STRING),
    }
    expected = {
        "SplitSigmas": (("sigmas", "step"), ("high_sigmas", "low_sigmas")),
        "SplitSigmasDenoise": (("sigmas", "denoise"), ("high_sigmas", "low_sigmas")),
        "FlipSigmas": (("sigmas",), ("sigmas",)),
        "SetFirstSigma": (("sigmas", "sigma"), ("sigmas",)),
        "ExtendIntermediateSigmas": (
            ("sigmas", "steps", "start_at_sigma", "end_at_sigma", "spacing"),
            ("sigmas",),
        ),
        "SamplingPercentToSigma": (
            ("model", "sampling_percent", "return_actual_sigma"),
            ("sigma_value",),
        ),
        "ManualSigmas": (("sigmas",), ("sigmas",)),
    }
    for node in SIGMA_OPERATION_NODES:
        schema = node.schema()
        input_ids, output_ids = expected[schema.node_type]
        assert schema.category == "model/sampling/sigmas"
        assert tuple(item.id for item in schema.inputs) == input_ids
        assert tuple(item.id for item in schema.outputs) == output_ids
        assert all(item.type == sigmas for item in schema.outputs if item.id != "sigma_value")
    split = SIGMA_OPERATION_NODES[0].schema()
    assert split.inputs[0].type == sigmas
    assert split.inputs[1].type == scalar_types["step"]
    assert split.inputs[1].widget == NumberWidget(min=0, max=10_000)
    denoise = SIGMA_OPERATION_NODES[1].schema().inputs[1]
    assert denoise.type == scalar_types["denoise"]
    assert denoise.default == 1.0
    set_first = SIGMA_OPERATION_NODES[3].schema().inputs[1]
    assert set_first.type == scalar_types["sigma"]
    assert set_first.widget == NumberWidget(min=0.0, max=20_000.0, step=0.001, round=False)
    extend = SIGMA_OPERATION_NODES[4].schema()
    assert [item.type for item in extend.inputs[1:4]] == [
        scalar_types["steps"],
        scalar_types["start_at_sigma"],
        scalar_types["end_at_sigma"],
    ]
    assert extend.inputs[1].widget == NumberWidget(min=1, max=100)
    assert extend.inputs[2].widget == NumberWidget(min=-1.0, max=20_000.0, step=0.01, round=False)
    assert extend.inputs[3].widget == NumberWidget(min=0.0, max=20_000.0, step=0.01, round=False)
    sampling = SIGMA_OPERATION_NODES[5].schema()
    assert sampling.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
    assert sampling.inputs[1].type == scalar_types["sampling_percent"]
    assert sampling.inputs[2].type == scalar_types["return_actual_sigma"]
    assert sampling.outputs[0].type == scalar_types["sampling_percent"]
    manual = SIGMA_OPERATION_NODES[6].schema()
    assert manual.inputs[0].type == scalar_types["sigmas_text"]
    assert manual.inputs[0].default == "1, 0.5"
    assert manual.search_terms == ("custom noise schedule", "define sigmas")
