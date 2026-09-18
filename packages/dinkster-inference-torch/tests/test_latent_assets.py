"""Native latent serialization and mounted node execution."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import pytest
import torch
from dinkster_inference import MultiStreamLatent
from dinkster_inference_torch.latent_assets import load_latent, serialize_native_latent
from safetensors.torch import save


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_native_round_trip_preserves_dtype_layout_and_owned_storage(dtype: torch.dtype) -> None:
    video = torch.arange(24, dtype=dtype).reshape(2, 3, 4).transpose(1, 2)
    audio = torch.tensor([[-0.0, 1.5]], dtype=dtype)
    samples = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    with serialize_native_latent(samples) as source:
        encoded = source.read()
        source.seek(0)
        loaded, hint = load_latent(source, torch)
    assert type(loaded) is MultiStreamLatent
    assert loaded.roles == samples.roles
    assert hint == ""
    for stream in samples.streams:
        actual = loaded.by_role(stream.role)
        expected = stream.payload
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.is_contiguous()
        assert actual.data_ptr() != expected.data_ptr()
        assert torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))
    with serialize_native_latent(loaded) as source:
        assert source.read() == encoded


@pytest.mark.parametrize("marker", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_stock_serialization_matches_source_load_arithmetic(
    marker: bool, dtype: torch.dtype
) -> None:
    # SaveLatent/LoadLatent in ComfyUI nodes.py at e20d433a4966dcc88fa5abbae6ace824cb78b263.
    tensor = torch.tensor([[-1.5, 0.18215, 0.0, 2.25]], dtype=dtype)
    tensors = {"latent_tensor": tensor}
    if marker:
        tensors["latent_format_version_0"] = torch.tensor([])
    loaded, hint = load_latent(BytesIO(save(tensors)), torch)
    expected = tensor.float() * (1.0 if marker else 1 / 0.18215)
    assert isinstance(loaded, torch.Tensor)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, expected)
    assert hint == ""


def test_codec_import_and_execution_without_comfyui_or_compatibility() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoCompat(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'comfy', 'comfy_extras', 'folder_paths',
                                     'nodes', 'dinkster_compat_comfy'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoCompat())
import torch
from dinkster_inference_torch.latent_assets import load_latent, serialize_native_latent
with serialize_native_latent(torch.tensor([[1.25, -2.5]])) as source:
    samples, hint = load_latent(source, torch)
assert torch.equal(samples, torch.tensor([[1.25, -2.5]]))
assert hint == ''
assert 'dinkster_compat_comfy' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_mounted_load_save_load_preserves_identity_bytes_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, multi: bool, dtype: torch.dtype
) -> None:
    from dinkster_assets import (
        AssetRef,
        MountDef,
        MountTable,
        parse_latent_asset,
        resolver_from_env,
    )
    from dinkster_native import native
    from dinkster_native.pool import ResidentPool
    from dinkster_protocol import ExportSnapshot
    from dinkster_workers import ExecutionContext
    from dinkster_workers.execution import use_execution_context

    output = tmp_path / "output"
    output.mkdir()
    video = torch.tensor([[0.0, -0.0], [1.5, -2.5]], dtype=dtype).T
    audio = torch.arange(3, dtype=dtype)
    samples = MultiStreamLatent.from_pairs((("video", video), ("audio", audio))) if multi else video
    pool = ResidentPool(cost_of=lambda _: {})
    vae = object()
    pool.label(vae, "vae.safetensors")
    pool.label_source(
        vae, digest="blake3:" + "a" * 64, name="vae.safetensors", latent_space="dinkster.sd15"
    )
    source = pool.source_for(vae)
    assert source is not None
    monkeypatch.setattr(native, "default_pool", lambda: pool)
    prompt = {"1": {"class_type": "LoadLatent", "inputs": {"latent": "source.latent"}}}
    workflow = {"nodes": [{"id": 1, "type": "LoadLatent"}]}
    snapshot = ExportSnapshot(prompt=prompt, extra_pnginfo={"workflow": workflow})
    with serialize_native_latent(samples, snapshot=snapshot, vae_hint=source.hint()) as stream:
        encoded = stream.read()
    (output / "source.latent").write_bytes(encoded)
    mounts = tmp_path / "mounts.json"
    table = MountTable(mounts)
    table.add(MountDef(id="out", path=output, mode="readwrite"))
    table.scan("out")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(mounts))
    initial = table.ref("mounts/out/source.latent")
    loaded = native.LoadLatent.execute(asset=initial)
    with use_execution_context(ExecutionContext(None, None, export_snapshot=snapshot)):
        saved = native.SaveLatent.execute(
            samples=loaded["samples"], target={"mount": "out", "prefix": "saved"}, vae=vae
        )
    assert saved["samples"] is loaded["samples"]
    asset = saved["asset"]
    assert isinstance(asset, AssetRef)
    assert asset.digest == initial.digest
    assert asset.virtual_path.startswith("mounts/out/saved_")
    assert (output / asset.name).read_bytes() == encoded
    bound = AssetRef.from_wire(asset.to_wire(), resolver=resolver_from_env())
    reloaded = native.LoadLatent.execute(asset=bound)
    assert reloaded["vae_hint"] == loaded["vae_hint"] == source.hint()
    latent = reloaded["samples"]
    assert isinstance(latent, Mapping)
    actual = latent["samples"]
    if multi:
        assert type(actual) is MultiStreamLatent
        assert actual.roles == ("video", "audio")
        received = (actual.by_role("video"), actual.by_role("audio"))
        expected = (video, audio)
    else:
        assert isinstance(actual, torch.Tensor)
        received = (actual,)
        expected = (video,)
    for value, original in zip(received, expected, strict=True):
        assert isinstance(value, torch.Tensor)
        assert value.dtype == original.dtype
        assert value.shape == original.shape
        assert value.is_contiguous()
        assert torch.equal(value.view(torch.uint8), original.contiguous().view(torch.uint8))
    with bound.open() as stream:
        descriptor = parse_latent_asset(stream)
    assert json.loads(descriptor.metadata["prompt"]) == prompt
    assert json.loads(descriptor.metadata["workflow"]) == workflow
    assert pool.source_for(vae) is source
    assert len(pool) == 1
