"""Shared BiRefNet construction and direct-state residency contracts."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from dinkster_inference_torch.birefnet import (
    BasicLayer,
    BiRefNet,
    DeformableConv2d,
    SwinTransformerBlock,
    WindowAttention,
)
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import CastOperations, ResidencyRouted


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention", [False, True], ids=["deform-conv", "window-attention"])
def test_direct_state_routes_preserve_storage_and_outputs(
    storage_dtype: torch.dtype, attention: bool
) -> None:
    operations = CastOperations(torch.float32)
    module = (
        (
            WindowAttention(8, (2, 2), 2, operations=operations)
            if attention
            else DeformableConv2d(2, 3, operations=operations)
        )
        .to(dtype=storage_dtype)
        .eval()
    )
    generator = torch.Generator().manual_seed(73)
    state = {
        name: (
            torch.randn(tensor.shape, generator=generator).mul(0.1).to(storage_dtype)
            if tensor.is_floating_point()
            else tensor.clone()
        )
        for name, tensor in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True)
    value = torch.randn((2, 4, 8) if attention else (2, 2, 5, 7), generator=generator)
    with torch.inference_mode():
        expected = module(value)
        cpu = torch.device("cpu")
        mechanism = enroll_component(module, load_device=cpu, offload_device=cpu)
        mechanism.partially_load(None)
        assert torch.equal(module(value), expected)
        mechanism.unload()
        assert mechanism.loaded_bytes() == 0
        with mechanism.execution_context():
            actual = module(value)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        assert mechanism.loaded_bytes() == 0
    for name, tensor in module.state_dict().items():
        assert tensor.dtype == state[name].dtype
        assert torch.equal(tensor, state[name])


def test_every_birefnet_state_owner_supports_residency() -> None:
    with torch.device("meta"):
        module = BiRefNet(operations=CastOperations(torch.float32))
    for name, child in module.named_modules():
        owns_state = tuple(child.parameters(recurse=False)) or tuple(child.buffers(recurse=False))
        if owns_state:
            assert isinstance(child, ResidencyRouted), name
    layers = list(module.bb.layers)
    assert all(isinstance(layer, BasicLayer) for layer in layers)
    assert isinstance(layers[2], BasicLayer) and len(layers[2].blocks) == 18
    assert isinstance(layers[0], BasicLayer)
    block = layers[0].blocks[0]
    assert isinstance(block, SwinTransformerBlock) and block.attn.window_size == (12, 12)
    assert module.state_dict()["squeeze_module.0.conv_in.weight"].shape == (64, 5760, 3, 3)


def test_generation_loader_is_strict_and_independent_of_comfyui() -> None:
    code = """
import importlib.abc
import sys

class NoComfy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'comfy', 'comfy_extras', 'nodes'}:
            raise AssertionError(f'upstream import: {fullname}')

sys.meta_path.insert(0, NoComfy())
import torch
from dinkster_compat_comfy.native_arm import _load_background_removal_component
from dinkster_inference_torch.birefnet import BiRefNet
from dinkster_inference_torch.operations import bound_compute_dtype

torch.set_num_threads(1)
with torch.device('meta'):
    template = BiRefNet()
state = {
    key: torch.zeros(value.shape, dtype=torch.float16 if value.is_floating_point() else value.dtype)
    for key, value in template.state_dict().items()
}
model, dtype, size, mean, std = _load_background_removal_component(state, torch.device('cpu'))
assert type(model) is BiRefNet
assert dtype == torch.float32
assert (size, mean, std) == (1024, (0., 0., 0.), (1., 1., 1.))
assert not model.training
assert set(model.state_dict()) == set(state)
for key, value in model.state_dict().items():
    assert value.dtype == state[key].dtype, key
    assert value.shape == state[key].shape, key
    assert value.device.type == 'cpu', key
assert bound_compute_dtype(model.bb.patch_embed.proj) == torch.float32
assert bound_compute_dtype(model.squeeze_module[0].bn_in) == torch.float32
del model
del state['bb.patch_embed.proj.weight']
try:
    _load_background_removal_component(state, torch.device('cpu'))
except RuntimeError as error:
    assert 'Missing key(s)' in str(error)
else:
    raise AssertionError('loader accepted missing checkpoint state')
assert not any(name.split('.')[0] in {'comfy', 'comfy_extras', 'nodes'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
