"""Public interfaces for native model-sampling and model patches."""

from dinkster_api.v1 import MODEL_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.device_routing import DEVICE_ROUTING_NODES
from dinkster_nodes_comfy.model_advanced import MODEL_ADVANCED_NODES


def test_advanced_model_patch_interfaces() -> None:
    expected = {
        "ModelSamplingDiscrete": ("model", "sampling", "zsnr"),
        "ModelSamplingContinuousEDM": ("model", "sampling", "sigma_max", "sigma_min"),
        "ModelSamplingContinuousV": ("model", "sampling", "sigma_max", "sigma_min"),
        "ModelSamplingStableCascade": ("model", "shift"),
        "ModelSamplingSD3": ("model", "shift"),
        "ModelSamplingAuraFlow": ("model", "shift", "sampling"),
        "ModelSamplingFlux": ("model", "max_shift", "base_shift", "width", "height"),
        "ModelNoiseScale": ("model", "noise_scale"),
        "RescaleCFG": ("model", "multiplier"),
        "ModelComputeDtype": ("model", "dtype"),
        "ModelAttentionBackend": ("model", "attention"),
    }
    assert {node.define_schema().node_type for node in MODEL_ADVANCED_NODES} == set(expected)
    for node in MODEL_ADVANCED_NODES:
        assert node in COMFY_NODES
        schema = node.define_schema()
        assert tuple(item.id for item in schema.inputs) == expected[schema.node_type]
        assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert [(item.id, item.type) for item in schema.outputs] == [
            ("MODEL", TypeExpr.concrete(MODEL_TYPE))
        ]
        if schema.node_type in ("ModelSamplingContinuousEDM", "ModelSamplingContinuousV"):
            for item in schema.inputs[2:]:
                assert item.widget == NumberWidget(min=0.0, max=1000.0, step=0.001, round=False)
    aura = next(node for node in MODEL_ADVANCED_NODES if node.__name__ == "ModelSamplingAuraFlow")
    assert not aura.define_schema().inputs[-1].required
    attention = MODEL_ADVANCED_NODES[-1].define_schema()
    assert attention.outputs[0].display_name == "model"


def test_device_routing_required_inputs_and_cold_discovery() -> None:
    expected = {
        "MultiGPU_WorkUnits": ("model", "max_gpus"),
        "SelectModelDevice": ("model", "device"),
        "SelectCLIPDevice": ("clip", "device"),
        "SelectVAEDevice": ("vae", "device"),
    }
    assert {node.schema().node_type for node in DEVICE_ROUTING_NODES} == set(expected)
    for node in DEVICE_ROUTING_NODES:
        assert node in COMFY_NODES
        schema = node.schema()
        assert tuple(item.id for item in schema.inputs) == expected[schema.node_type]
        assert all(item.required for item in schema.inputs)
        assert len(schema.outputs) == 1
        assert schema.inputs[0].type == schema.outputs[0].type
