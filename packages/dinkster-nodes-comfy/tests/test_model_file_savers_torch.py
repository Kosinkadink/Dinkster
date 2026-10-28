"""Encoder export writes actual safetensors through mounted output authority."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.model_file_savers import CheckpointSave, CLIPSave, ModelSave, VAESave
from safetensors import safe_open


class _Clip:
    def load_model(self) -> None:
        pass

    def state_dict_for_saving(self) -> dict[str, torch.Tensor]:
        return {
            "clip_l.transformer.left": torch.tensor([1.25, -3.5]),
            "clip_g.right": torch.tensor([[8, -2]], dtype=torch.int32),
            "transformer.other": torch.tensor([9.0], dtype=torch.float64),
        }


class _Vae:
    def get_sd(self) -> dict[str, torch.Tensor]:
        return {"decoder.weight": torch.tensor([[1.25, -3.5]], dtype=torch.float64)}


def test_encoder_exports_partition_clip_keys_and_preserve_vae_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "output", "root": str(output), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                ],
                "outputMount": "output",
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    assert CLIPSave.execute(clip=_Clip(), mount="alternate") == {}
    assert VAESave.execute(vae=_Vae()) == {}
    expected = (
        (alternate / "clip/ComfyUI_clip_l_00001.safetensors", "left", torch.tensor([1.25, -3.5])),
        (
            alternate / "clip/ComfyUI_clip_g_00001.safetensors",
            "right",
            torch.tensor([[8, -2]], dtype=torch.int32),
        ),
        (
            alternate / "clip/ComfyUI_00001.safetensors",
            "other",
            torch.tensor([9.0], dtype=torch.float64),
        ),
        (
            output / "vae/ComfyUI_vae_00001.safetensors",
            "decoder.weight",
            torch.tensor([[1.25, -3.5]], dtype=torch.float64),
        ),
    )
    for path, key, tensor in expected:
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert list(handle.keys()) == [key]
            assert handle.metadata() == (
                {"prompt": ""} if key == "decoder.weight" else {"format": "pt", "prompt": ""}
            )
            saved = handle.get_tensor(key)
            assert saved.dtype == tensor.dtype
            torch.testing.assert_close(saved, tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("model_class", "architecture", "checkpoint", "vpred", "disable_metadata"),
    (
        ("SDXL", "stable-diffusion-xl-v1-base", True, False, False),
        ("SDXL_instructpix2pix", "stable-diffusion-xl-v1-edit", False, False, False),
        ("SDXLRefiner", "stable-diffusion-xl-v1-refiner", False, False, False),
        ("SVD_img2vid", "stable-video-diffusion-img2vid-v1", True, False, False),
        ("SD3", "stable-diffusion-v3-medium", False, False, True),
        (None, None, False, False, False),
        (None, None, True, True, False),
    ),
)
def test_checkpoint_exports_preserve_metadata_counters_and_tensors(
    tmp_path: Path, monkeypatch, model_class, architecture, checkpoint, vpred, disable_metadata
) -> None:
    initialize_inference_process()
    from dinkster_inference import cli_args, model_base, model_management, model_sampling

    monkeypatch.setattr(cli_args.args, "disable_metadata", disable_metadata)
    monkeypatch.setattr(model_management, "load_models_gpu", lambda _models: None)
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "output", "root": str(output), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                ],
                "outputMount": "output",
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    body = object.__new__(getattr(model_base, model_class)) if model_class else SimpleNamespace()
    if model_class:
        torch.nn.Module.__init__(body)
    body.model_type = model_base.ModelType.V_PREDICTION if vpred else model_base.ModelType.EPS
    if vpred:

        class Sampling(model_sampling.ModelSamplingContinuousEDM, model_sampling.V_PREDICTION):
            pass

        sampling = Sampling()
        sampling.zsnr = True
        sampling.sigmas = torch.tensor([0.125, 37.5])
    else:
        sampling = SimpleNamespace()
    weight = torch.tensor([[1.25, -3.5, 9.75], [-0.5, 7.0, 2.0]], dtype=torch.float64).T

    def state_dict(clip_sd, vae_sd, vision_sd):
        assert vision_sd is None
        state = {"model.weight": weight}
        if clip_sd is not None:
            state.update(clip_sd)
        if vae_sd is not None:
            state.update(vae_sd)
        return state

    model = SimpleNamespace(
        model=body, get_model_object=lambda _name: sampling, state_dict_for_saving=state_dict
    )
    prefix = "checkpoints/model" if checkpoint else "diffusion_models/model"
    root = alternate if checkpoint else output
    folder = root / prefix.split("/")[0]
    folder.mkdir()
    existing = folder / "model_00004_.safetensors"
    existing.write_bytes(b"existing")
    node = CheckpointSave if checkpoint else ModelSave
    options = {"clip": _Clip(), "vae": _Vae(), "mount": "alternate"} if checkpoint else {}
    expected = {"model.weight": weight}
    if checkpoint:
        expected.update(_Clip().state_dict_for_saving())
        expected.update(_Vae().get_sd())
    if vpred:
        expected.update(
            {
                "edm_vpred.sigma_min": torch.tensor(0.125),
                "edm_vpred.sigma_max": torch.tensor(37.5),
                "v_pred": torch.tensor([]),
                "ztsnr": torch.tensor([]),
            }
        )
    for counter in (5, 6):
        assert node.execute(model=model, filename_prefix=prefix, **options) == {}
        with safe_open(folder / f"model_{counter:05d}_.safetensors", framework="pt") as saved:
            metadata = {"modelspec.predict_key": "v" if vpred else "epsilon"}
            if architecture:
                metadata.update(
                    {
                        "modelspec.architecture": architecture,
                        "modelspec.sai_model_spec": "1.0.0",
                        "modelspec.implementation": "sgm",
                        "modelspec.title": f"model {counter}",
                    }
                )
            if not disable_metadata:
                metadata["prompt"] = ""
            assert saved.metadata() == metadata
            assert set(saved.keys()) == set(expected)
            for key, value in expected.items():
                actual = saved.get_tensor(key)
                assert actual.dtype == value.dtype
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
    assert existing.read_bytes() == b"existing"
    assert len(list(folder.glob("*.safetensors"))) == 3
    assert not list((output if checkpoint else alternate).rglob("*.safetensors"))
    assert not list(folder.glob(".dinkster-save-*"))
