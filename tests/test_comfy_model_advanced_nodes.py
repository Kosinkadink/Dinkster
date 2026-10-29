"""Public interfaces for native model-sampling and model patches."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
from dinkster_api.v1 import MODEL_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES, model_advanced
from dinkster_nodes_comfy.device_routing import DEVICE_ROUTING_NODES
from dinkster_nodes_comfy.model_advanced import MODEL_ADVANCED_NODES


def test_advanced_model_patch_interfaces() -> None:
    # Named classes below, F bc5f880d, define_schema() without inputs.
    # MODEL uses the canonical native public identity.
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
    # Named device-routing classes below, F bc5f880d, define_schema() without inputs.
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


def test_attention_schema_discovery_never_requests_torch_or_inference() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _COLD_ATTENTION],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_attention_backend_resolves_pytorch_only_when_executing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(initialized=False)
    pytorch = object()
    runtime = SimpleNamespace(
        get_attention_function=lambda name, default=None: pytorch if name == "pytorch" else default
    )

    def initialize() -> None:
        state.initialized = True

    def load(name: str) -> object:
        assert name == "dinkster_inference.ldm.modules.attention"
        assert state.initialized
        return runtime

    monkeypatch.setattr(model_advanced, "importlib", SimpleNamespace(import_module=load))
    monkeypatch.setattr(model_advanced, "initialize_inference_process", initialize)
    clone = SimpleNamespace(attention=None)
    clone.set_model_optimized_attention = lambda callback: setattr(clone, "attention", callback)
    original = SimpleNamespace(attention=None, clone=lambda: clone)
    model_advanced.ModelAttentionBackend.schema()
    assert not state.initialized
    result = model_advanced.ModelAttentionBackend.execute(
        model=original, attention="pytorch attention"
    )["MODEL"]
    assert result is clone
    assert clone.attention is pytorch
    assert original.attention is None


@pytest.mark.parametrize("attention", ("comfy kitchen attention", "unknown"))
def test_attention_backend_rejects_values_outside_its_widget(
    monkeypatch: pytest.MonkeyPatch, attention: str
) -> None:
    def initialize() -> None:
        pytest.fail("invalid attention must not initialize inference")

    monkeypatch.setattr(model_advanced, "initialize_inference_process", initialize)
    with pytest.raises(ValueError, match="Unknown attention backend"):
        model_advanced.ModelAttentionBackend.execute(model=object(), attention=attention)


_COLD_ATTENTION = """
import importlib.abc
import sys

class NoTensorRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'dinkster_inference'}:
            raise AssertionError('cold discovery requested ' + fullname)

sys.meta_path.insert(0, NoTensorRuntime())
from dinkster_nodes_comfy.model_advanced import ModelAttentionBackend
from dinkster_schema import schema_to_wire

schema = schema_to_wire(ModelAttentionBackend.schema())
assert schema
widget = ModelAttentionBackend.schema().inputs[1].widget
# ModelAttentionBackend, F bc5f880d: finalized ordinary combo options.
assert widget.options == ('pytorch attention',)
assert not widget.remote_route
assert not any(name.split('.')[0] in {'torch', 'dinkster_inference'} for name in sys.modules)
"""
