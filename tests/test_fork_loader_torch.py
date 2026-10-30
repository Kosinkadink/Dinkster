"""Real native loader checkpoints, independent of model downloads."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import pytest

torch = cast("Any", pytest.importorskip("torch"))

from dinkster_assets import AssetRef, AssetVault, MountDef, MountTable, digest_file  # noqa: E402
from dinkster_inference_wire.process import initialize_inference_process  # noqa: E402
from dinkster_native.fork_nodes import (  # noqa: E402
    GenerationLoadCheckpoint,
    GenerationLoadDiffusionModel,
    NativeLoadClip,
    NativeLoadVae,
)


@pytest.fixture(autouse=True)
def cpu_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")
    initialize_inference_process()


def _asset(path: Path, kind: str, vault_backed: bool) -> Any:
    if not vault_backed:
        mounts = MountTable()
        mounts.add(MountDef(id="weights", path=path.parent), kind=kind)
        asset = mounts.bind_name(path.name, kind)
        assert asset is not None
        assert asset.local_path() == path
        return asset
    vault = AssetVault(path.parent / "vault")
    digest = digest_file(path)
    with vault.writer(digest) as writer:
        writer.write(path.read_bytes())
        writer.commit()
    asset = AssetRef(digest, path.name, path.stat().st_size, resolver=vault)
    local = asset.local_path()
    assert local.name == digest.removeprefix("blake3:")
    assert local.parent.name == local.name[:2]
    assert not local.suffix
    path.unlink()
    return asset


def _diffusion_state() -> dict[str, Any]:
    ops = importlib.import_module("dinkster_inference.ops")
    Flux = importlib.import_module("dinkster_inference.ldm.flux.model").Flux

    model = Flux(
        in_channels=4,
        out_channels=16,
        vec_in_dim=None,
        context_in_dim=8,
        hidden_size=128,
        mlp_ratio=4.0,
        num_heads=1,
        depth=1,
        depth_single_blocks=0,
        axes_dim=[16, 56, 56],
        theta=10000,
        patch_size=2,
        qkv_bias=True,
        guidance_embed=False,
        txt_ids_dims=[],
        operations=ops.manual_cast,
        dtype=torch.float32,
        device="cpu",
    )
    state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    state["img_in.bias"].fill_(0.125)
    state["final_layer.linear.bias"] = torch.arange(64, dtype=torch.float32) / 64
    return {key: value.half() for key, value in state.items()}


def _vae_state() -> dict[str, Any]:
    TAESD = importlib.import_module("dinkster_inference.taesd.taesd").TAESD

    model = TAESD(latent_channels=4)
    state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    state["vae_scale"].fill_(1)
    state["taesd_decoder.19.bias"] = torch.tensor([0.125, 0.5, 0.875])
    state["taesd_encoder.14.bias"] = torch.tensor([0.25, 0.5, 0.75, 1.0])
    return {key: value.half() for key, value in state.items()}


def _check_diffusion(patcher: Any) -> None:
    model = patcher.model.diffusion_model
    assert torch.equal(model.img_in.bias, torch.full((128,), 0.125))
    assert torch.equal(model.final_layer.linear.bias, torch.arange(64) / 64)
    dtype = patcher.model.get_dtype_inference()
    with torch.inference_mode():
        result = model(
            torch.ones((1, 4, 2, 2), dtype=dtype),
            torch.tensor([0.5], dtype=dtype),
            torch.ones((1, 2, 8), dtype=dtype),
            None,
        )
    expected = (torch.arange(64) / 64).reshape(1, 16, 2, 2)
    assert torch.equal(result, expected)


def _check_vae(vae: Any) -> None:
    vae.throw_exception_if_invalid()
    assert torch.equal(
        vae.first_stage_model.taesd_decoder[19].bias, torch.tensor([0.125, 0.5, 0.875])
    )
    with torch.inference_mode():
        decoded = vae.decode(torch.ones((1, 4, 1, 1)))
        encoded = vae.encode(torch.ones((1, 8, 8, 3)))
    assert torch.equal(decoded, torch.tensor([0.125, 0.5, 0.875]).expand(1, 8, 8, 3))
    assert torch.equal(encoded, torch.tensor([0.25, 0.5, 0.75, 1.0]).reshape(1, 4, 1, 1))


@pytest.mark.parametrize("vault_backed", [False, True], ids=["named-mount", "vault"])
def test_native_diffusion_model_real_checkpoint(tmp_path: Path, vault_backed: bool) -> None:
    save_file = importlib.import_module("safetensors.torch").save_file

    path = tmp_path / "diffusion.safetensors"
    save_file(_diffusion_state(), path)
    assert path.stat().st_size < 2_000_000
    asset = _asset(path, "model/diffusion", vault_backed)
    loaded = GenerationLoadDiffusionModel.execute(diffusion_model=asset, weight_dtype="default")
    _check_diffusion(loaded["model"])


@pytest.mark.parametrize("vault_backed", [False, True], ids=["named-mount", "vault"])
def test_native_vae_real_checkpoint(tmp_path: Path, vault_backed: bool) -> None:
    save_file = importlib.import_module("safetensors.torch").save_file

    path = tmp_path / "vae.safetensors"
    save_file(_vae_state(), path)
    assert path.stat().st_size < 10_000_000
    asset = _asset(path, "model/vae", vault_backed)
    _check_vae(NativeLoadVae.execute(vae=asset)["vae"])


@pytest.mark.parametrize("vault_backed", [False, True], ids=["named-mount", "vault"])
def test_native_combined_real_checkpoint(tmp_path: Path, vault_backed: bool) -> None:
    save_file = importlib.import_module("safetensors.torch").save_file

    path = tmp_path / "checkpoint.safetensors"
    state = {f"model.diffusion_model.{key}": value for key, value in _diffusion_state().items()}
    state.update({f"vae.{key}": value for key, value in _vae_state().items()})
    save_file(state, path)
    assert path.stat().st_size < 12_000_000
    asset = _asset(path, "model/checkpoint", vault_backed)
    loaded = GenerationLoadCheckpoint.execute(checkpoint=asset)
    _check_diffusion(loaded["model"])
    _check_vae(loaded["vae"])
    assert loaded["clip"] is None


@pytest.mark.parametrize("vault_backed", [False, True], ids=["named-mount", "vault"])
def test_native_clip_real_checkpoint(tmp_path: Path, vault_backed: bool) -> None:
    SD1ClipModel = importlib.import_module("dinkster_inference.sd1_clip").SD1ClipModel

    # CLIP's file loader fixes the architecture. Expanded tensors retain its full
    # state with scalar backing storage; torch serialization preserves strides.
    model = SD1ClipModel(device="meta", dtype=torch.float32)
    zero = torch.tensor(0.0)
    state = {
        key.removeprefix("clip_l.transformer."): zero.expand(value.shape)
        for key, value in model.state_dict().items()
    }
    bias = torch.arange(768, dtype=torch.float32) / 1024
    state["text_model.final_layer_norm.bias"] = bias
    path = tmp_path / "clip.pt"
    torch.save(state, path)
    assert path.stat().st_size < 100_000
    asset = _asset(path, "model/text-encoder", vault_backed)
    loaded = NativeLoadClip.execute(text_encoder=asset, type="stable_diffusion", device="cpu")
    clip = cast("Any", loaded["clip"])
    transformer = clip.cond_stage_model.clip_l.transformer
    assert torch.equal(transformer.text_model.final_layer_norm.bias, bias)
    assert torch.count_nonzero(transformer.text_model.encoder.layers[11].mlp.fc2.weight) == 0
    tokens = clip.tokenize("a cat")
    assert [token for token, weight in tokens["l"][0][:4]] == [49406, 320, 2368, 49407]
    with torch.inference_mode():
        encoded, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    assert torch.equal(encoded, bias.expand(1, 77, 768))
    assert torch.equal(pooled, bias.expand(1, 768))
