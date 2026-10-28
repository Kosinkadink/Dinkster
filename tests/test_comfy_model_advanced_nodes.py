"""Public interfaces for native model-sampling and model patches."""

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_api.v1 import MODEL_TYPE, NumberWidget, TypeExpr, TypeRegistry
from dinkster_nodes_comfy import COMFY_NODES, model_advanced
from dinkster_nodes_comfy.device_routing import DEVICE_ROUTING_NODES
from dinkster_nodes_comfy.model_advanced import MODEL_ADVANCED_NODES
from dinkster_workers import InProcessWorker
from dinkster_workers.host import load_choices
from dinkster_workers.manifest import load_manifest

MANIFEST = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"


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


def test_attention_schema_and_choice_discovery_never_request_torch() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _COLD_ATTENTION, str(MANIFEST)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_attention_choices_follow_live_device_and_kernel_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(cuda=False, hip=None, kernel=True, initialized=False)

    def initialize() -> None:
        state.initialized = True

    def load(name: str) -> object:
        if name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: state.cuda),
                version=SimpleNamespace(hip=state.hip),
            )
        assert name == "dinkster_inference.ldm.modules.attention"
        assert state.initialized
        return SimpleNamespace(COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE=state.kernel)

    monkeypatch.setattr(model_advanced, "importlib", SimpleNamespace(import_module=load))
    monkeypatch.setattr(model_advanced, "initialize_inference_process", initialize)
    choices = load_choices(load_manifest(MANIFEST))
    worker = InProcessWorker({}, TypeRegistry(), lazy_choices=choices.lazy)

    async def scenario() -> None:
        # ModelAttentionBackend, F bc5f880d, define_schema() with controlled
        # CUDA/HIP/kernel states below: CPU, CUDA missing kernel, CUDA, HIP.
        assert await worker.fetch_choices("model-attention.backends") == ("pytorch attention",)
        assert not state.initialized
        state.cuda, state.kernel = True, False
        assert await worker.fetch_choices("model-attention.backends") == ("pytorch attention",)
        state.kernel = True
        assert await worker.fetch_choices("model-attention.backends") == (
            "pytorch attention",
            "comfy kitchen attention",
        )
        state.cuda, state.hip = False, "test-hip"
        assert await worker.fetch_choices("model-attention.backends") == (
            "pytorch attention",
            "comfy kitchen attention",
        )
        state.hip = None
        assert await worker.fetch_choices("model-attention.backends") == ("pytorch attention",)

    asyncio.run(scenario())


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
from dinkster_values.names import claim_covers
from dinkster_workers.host import load_choices
from dinkster_workers.manifest import load_manifest

manifest = load_manifest(sys.argv[1])
choices = load_choices(manifest)
assert not choices.static
schema = schema_to_wire(ModelAttentionBackend.schema())
assert schema
widget = ModelAttentionBackend.schema().inputs[1].widget
# Native remote-choice contract: immediate CPU fallback and deferred live choices.
assert widget.options == ('pytorch attention',)
choice_id = widget.remote_route.removeprefix('/api/choices/')
assert choice_id in choices.lazy
assert any(claim_covers(claim, choice_id) for claim in manifest.namespaces)
assert not any(name.split('.')[0] in {'torch', 'dinkster_inference'} for name in sys.modules)
"""
