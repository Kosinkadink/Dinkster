"""Checkpoint hook behavior with actual CPU patchers and controlled loader I/O."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api import v1
from dinkster_api.v1 import AssetRef, LocalAssetRef, initialize_inference_process
from dinkster_assets import AssetError, MountDef, MountTable, digest_file
from dinkster_nodes_comfy.hook_checkpoint import (
    CreateHookModelAsLora,
    CreateHookModelAsLoraModelOnly,
)


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _asset(path: Path, verified: bool) -> AssetRef | LocalAssetRef:
    if verified:
        return AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=_Resolver(path))
    mounts = MountTable()
    mounts.add(MountDef(id="checkpoints", path=path.parent), kind="model/checkpoint")
    ref = mounts.bind_name(path.name, "model/checkpoint")
    assert isinstance(ref, LocalAssetRef)
    return ref


def _patcher(values: list[list[float]]) -> Any:
    from dinkster_inference.model_patcher import ModelPatcher

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor(values))
    model.model_sampling = torch.nn.Linear(1, 1, bias=False)
    return ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("has_clip", [False, True])
def test_checkpoint_hooks_preserve_weights_prior_hooks_and_strength_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verified: bool, has_clip: bool
) -> None:
    initialize_inference_process()
    from dinkster_inference import hooks, sd

    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"cpu checkpoint loader seam")
    ref = _asset(checkpoint, verified)
    roots = (tmp_path / "embedding-b", tmp_path / "embedding-a")

    def mount_roots(kind: str) -> tuple[Path, ...]:
        assert kind == "model/embedding"
        return roots

    monkeypatch.setattr(v1, "mount_roots_for_kind", mount_roots, raising=False)
    model = _patcher([[2.0, -7.0], [11.0, 0.5]])
    clip = _patcher([[-3.0, 13.0]])
    calls = []

    def load(path: str, **options: Any) -> tuple[Any, Any, object, None]:
        assert path == str(checkpoint)
        assert options == {
            "output_vae": True,
            "output_clip": True,
            "embedding_directory": [str(root) for root in roots],
        }
        calls.append(path)
        return model, SimpleNamespace(patcher=clip) if has_clip else None, object(), None

    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    previous = hooks.HookGroup()
    prior = hooks.WeightHook(strength_model=-0.25, strength_clip=0.75)
    previous.add(prior)
    result = CreateHookModelAsLora.execute(
        ckpt_name=ref, strength_model=-0.7, strength_clip=1.3, prev_hooks=previous
    )["HOOKS"]
    assert isinstance(result, hooks.HookGroup)
    assert len(previous.hooks) == 1 and previous.hooks[0] is prior
    assert len(result.hooks) == 2
    assert result.hooks[0] is not prior
    assert result.hooks[0].strength_model == -0.25
    added = result.hooks[1]
    assert added.strength_model == -0.7 and added.strength_clip == 1.3
    assert not added.need_weight_init
    assert set(added.weights) == {"weight"}
    assert added.weights["weight"][0] == "model_as_lora"
    assert torch.equal(added.weights["weight"][1][0], torch.tensor([[2.0, -7.0], [11.0, 0.5]]))
    if has_clip:
        assert set(added.weights_clip) == {"weight"}
        assert torch.equal(added.weights_clip["weight"][1][0], torch.tensor([[-3.0, 13.0]]))
    else:
        assert added.weights_clip is None
    assert "model_sampling.weight" in model.model.state_dict()
    assert "model_sampling.weight" in clip.model.state_dict()

    model_only = CreateHookModelAsLoraModelOnly.execute(ckpt_name=ref, strength_model=0.0)["HOOKS"]
    assert isinstance(model_only, hooks.HookGroup)
    assert len(model_only.hooks) == 1
    assert model_only.hooks[0].strength_model == 0.0
    assert model_only.hooks[0].strength_clip == 0.0
    assert torch.equal(model_only.hooks[0].weights["weight"][1][0], model.model.weight)
    assert len(calls) == 1
    assert added.strength_model == -0.7 and added.strength_clip == 1.3


def test_checkpoint_hook_refreshes_embedding_authority_and_revalidates_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    initialize_inference_process()
    from dinkster_inference import hooks, sd

    checkpoint = tmp_path / "refresh.bin"
    checkpoint.write_bytes(b"first revision")
    ref = _asset(checkpoint, False)
    roots = (tmp_path / "first",)
    monkeypatch.setattr(v1, "mount_roots_for_kind", lambda kind: roots, raising=False)

    def load(path: str, **options: Any) -> tuple[Any, None, None, None]:
        assert options["embedding_directory"] == [str(root) for root in roots]
        assert options["output_vae"] is True and options["output_clip"] is True
        return _patcher([[3.0 if roots else -17.0]]), None, None, None

    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    first = CreateHookModelAsLoraModelOnly.execute(ckpt_name=ref, strength_model=0.8)["HOOKS"]
    roots = ()
    revoked = CreateHookModelAsLoraModelOnly.execute(ckpt_name=ref, strength_model=0.8)["HOOKS"]
    assert isinstance(first, hooks.HookGroup) and isinstance(revoked, hooks.HookGroup)
    assert first.hooks[0].weights["weight"][1][0].item() == 3.0
    assert revoked.hooks[0].weights["weight"][1][0].item() == -17.0
    checkpoint.write_bytes(b"different checkpoint revision")
    with pytest.raises(AssetError, match="unavailable|revoked|changed"):
        CreateHookModelAsLoraModelOnly.execute(ckpt_name=ref, strength_model=0.8)


def test_checkpoint_weights_share_residency_across_assets_nodes_and_strengths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    initialize_inference_process()
    from dinkster_inference import sd
    from dinkster_memory.resident import default_pool

    paths = (tmp_path / "first.bin", tmp_path / "second.bin", tmp_path / "same-first.bin")
    paths[0].write_bytes(b"first checkpoint")
    paths[1].write_bytes(b"second checkpoint")
    paths[2].write_bytes(paths[0].read_bytes())
    assets = [_asset(path, True) for path in paths]
    monkeypatch.setattr(v1, "mount_roots_for_kind", lambda kind: (), raising=False)
    model_a = _patcher([[2.0, -7.0], [11.0, 0.5]])
    model_b = _patcher([[-13.0, 3.0, 8.0]])
    calls: list[str] = []

    def load(path: str, **options: Any) -> tuple[Any, Any, None, None]:
        assert options == {"output_vae": True, "output_clip": True, "embedding_directory": []}
        calls.append(path)
        model = model_b if path == str(paths[1]) else model_a
        return model, SimpleNamespace(patcher=model), None, None

    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    baseline = default_pool().footprint("ram")
    retained = []
    previous = None
    for index, strength in ((0, -0.75), (1, 1.25), (2, 0.0), (1, -2.5), (0, 3.0)):
        result = CreateHookModelAsLora.execute(
            ckpt_name=assets[index],
            strength_model=strength,
            strength_clip=-strength,
            prev_hooks=previous,
        )["HOOKS"]
        added = result.hooks[-1]
        assert added.strength_model == strength and added.strength_clip == -strength
        expected = model_b.model.weight if index == 1 else model_a.model.weight
        torch.testing.assert_close(added.weights["weight"][1][0], expected, rtol=0, atol=0)
        assert added.weights_clip["weight"][1][0].data_ptr() == expected.data_ptr()
        retained.append(result._dinkster_resident_refs[-1])
        previous = result
    assert retained[0] is retained[2] is retained[4]
    assert retained[1] is retained[3] and retained[0] is not retained[1]
    assert calls == [str(paths[0]), str(paths[1])]
    # Each pair aliases model/CLIP storage, so charge each retained storage once.
    assert default_pool().footprint("ram") - baseline == 28


def test_checkpoint_hook_rejects_unbound_filename() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        CreateHookModelAsLoraModelOnly.execute(ckpt_name="unbound.ckpt", strength_model=1.0)
