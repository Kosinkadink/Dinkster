"""Native MoGe loading, execution, and residency contracts."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from dinkster_inference_torch.dinov2 import Dinov2Model
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.moge import MoGeModelV1, build_from_state_dict
from dinkster_inference_torch.moge_geometry import (
    normalized_view_plane_uv,
    recover_focal_shift,
)
from dinkster_inference_torch.operations import CastOperations, ResidencyRouted

# https://huggingface.co/Comfy-Org/MoGe/resolve/7f19f325949e1e7bb31c720333936f03cf932f57/geometry_estimation/moge_2_vitl_normal_fp16.safetensors
_MODEL_SIZE = 661_859_924
_MODEL_SHA256 = "cb1a692d03235671e959e81360d7b4d9f44aefadb1f852d6ca6aa17799d5e31f"


def _fill_state(module: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(143)
    state = {
        name: (
            torch.randn(value.shape, generator=generator).mul(0.05)
            if value.is_floating_point()
            else value.clone()
        )
        for name, value in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True)


def test_dinov2_direct_state_survives_residency_offload() -> None:
    model = Dinov2Model(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "layer_norm_eps": 1e-6,
            "position_tokens": 17,
            "use_mask_token": True,
        },
        operations=CastOperations(torch.float32),
    ).eval()
    _fill_state(model)
    image = torch.randn((2, 3, 42, 56), generator=torch.Generator().manual_seed(151))

    for name, child in model.named_modules():
        owns_state = tuple(child.parameters(recurse=False)) or tuple(child.buffers(recurse=False))
        if owns_state:
            assert isinstance(child, ResidencyRouted), name

    with torch.inference_mode():
        expected = model.get_intermediate_layers(image, [0, 1])
        cpu = torch.device("cpu")
        mechanism = enroll_component(model, load_device=cpu, offload_device=cpu)
        mechanism.partially_load(None)
        mechanism.unload()
        assert mechanism.loaded_bytes() == 0
        with mechanism.execution_context():
            actual = model.get_intermediate_layers(image, [0, 1])
        assert mechanism.loaded_bytes() == 0

    for expected_layer, actual_layer in zip(expected, actual, strict=True):
        for expected_value, actual_value in zip(expected_layer, actual_layer, strict=True):
            assert torch.equal(actual_value, expected_value)


def test_focal_recovery_matches_known_asymmetric_projection() -> None:
    height, width = 37, 53
    uv = normalized_view_plane_uv(width, height, dtype=torch.float64)
    depth = torch.linspace(1.0, 2.0, height * width, dtype=torch.float64).reshape(height, width)
    expected_focal = torch.tensor(0.8, dtype=torch.float64)
    expected_shift = torch.tensor(0.4, dtype=torch.float64)
    xy = uv * (depth + expected_shift).unsqueeze(-1) / expected_focal
    points = torch.cat([xy, depth.unsqueeze(-1)], dim=-1).unsqueeze(0)
    mask = torch.ones((1, height, width), dtype=torch.bool)

    focal, shift = recover_focal_shift(points, mask)
    assert torch.allclose(focal, expected_focal.unsqueeze(0), atol=1e-7, rtol=0)
    assert torch.allclose(shift, expected_shift.unsqueeze(0), atol=1e-7, rtol=0)
    fixed_focal, fixed_shift = recover_focal_shift(points, mask, focal=expected_focal.unsqueeze(0))
    assert torch.equal(fixed_focal, expected_focal.unsqueeze(0))
    assert torch.allclose(fixed_shift, expected_shift.unsqueeze(0), atol=1e-7, rtol=0)


def test_moge_v1_state_detection_preserves_outputs_without_mask_token() -> None:
    model = MoGeModelV1(
        {
            "hidden_size": 64,
            "num_attention_heads": 1,
            "num_hidden_layers": 4,
            "layer_norm_eps": 1e-6,
            "position_tokens": 17,
            "use_mask_token": False,
        },
        dim_upsample=(8,),
        operations=CastOperations(torch.float32),
    ).eval()
    _fill_state(model)
    restored = cast(
        MoGeModelV1,
        build_from_state_dict(model.state_dict(), operations=CastOperations(torch.float32)).eval(),
    )
    image = torch.rand((1, 3, 42, 56), generator=torch.Generator().manual_seed(163))
    with torch.inference_mode():
        expected = model(image, num_tokens=12)
        actual = restored(image, num_tokens=12)
    assert set(actual) == {"points", "mask"}
    assert all(torch.equal(actual[name], expected[name]) for name in expected)
    assert restored.backbone.embeddings.mask_token is None


def _configured_model() -> Path:
    configured = os.environ.get("DINKSTER_MOGE_MODEL")
    if not configured:
        pytest.skip("DINKSTER_MOGE_MODEL is not set")
    path = Path(configured)
    if not path.is_file():
        pytest.skip("DINKSTER_MOGE_MODEL does not name a file")
    return path


def test_real_moge_loader_is_strict_and_independent_of_comfyui() -> None:
    model_path = _configured_model()
    assert model_path.stat().st_size == _MODEL_SIZE
    with model_path.open("rb") as model_file:
        assert hashlib.file_digest(model_file, "sha256").hexdigest() == _MODEL_SHA256
    code = """
import importlib.abc
import sys
from pathlib import Path

class NoComfy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'comfy', 'comfy_extras', 'nodes'}:
            raise AssertionError(f'upstream import: {fullname}')

sys.meta_path.insert(0, NoComfy())
import torch
from dinkster_assets import AssetRef
from dinkster_compat_comfy.native_arm import (
    _NativeGeometryModel,
    _enroll_auxiliary_model,
    _load_geometry_component,
    _run_geometry_model,
)
from dinkster_inference_torch.checkpoint import load_checkpoint
from dinkster_inference_torch.moge import MoGeModelV2

state = load_checkpoint(Path(sys.argv[1]))
model, dtype, version, threshold, token_range = _load_geometry_component(state)
assert type(model) is MoGeModelV2
assert dtype == torch.float32
assert version == 'v2'
assert threshold == 0.5
assert token_range == (1200, 3600)
assert not model.training
assert {value.device.type for value in model.state_dict().values()} == {'cpu'}
asset = AssetRef('blake3:' + '1' * 64, 'moge.safetensors', Path(sys.argv[1]).stat().st_size)
handle = _enroll_auxiliary_model(
    asset, model, 'geometry', load_device=torch.device('cpu')
)
resource = _NativeGeometryModel(handle, dtype, version, threshold, (1200, 1200))
image = torch.rand((1, 56, 70, 3), generator=torch.Generator().manual_seed(157))
output = _run_geometry_model(resource, image, 0, 60.0, 1, True, True)
assert {name: tuple(value.shape) for name, value in output.items()} == {
    'image': (1, 56, 70, 3),
    'points': (1, 56, 70, 3),
    'depth': (1, 56, 70),
    'intrinsics': (1, 3, 3),
    'mask': (1, 56, 70),
    'normal': (1, 56, 70, 3),
}
assert torch.isfinite(output['intrinsics']).all()
assert torch.isfinite(output['mask']).all()
del model
del resource
del handle
del state['encoder.backbone.blocks.0.attn.proj.weight']
try:
    _load_geometry_component(state)
except RuntimeError as error:
    assert 'Missing key(s)' in str(error)
else:
    raise AssertionError('loader accepted missing checkpoint state')
assert not any(
    name.split('.')[0] in {'comfy', 'comfy_extras', 'nodes'} for name in sys.modules
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(model_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
