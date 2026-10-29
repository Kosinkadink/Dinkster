"""Controlled CPU checkpoint export through real safetensors and mounted output.

Metadata expectations follow ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f,
comfy_extras/nodes_model_merging.py:170-227 and nodes_video_model.py:110-123.
Tensor values and metadata are hand-derived, not fixed-reference captures.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import AssetError, MountWidget, initialize_inference_process
from dinkster_nodes_comfy.checkpoint_save import ImageOnlyCheckpointSave
from safetensors import safe_open
from safetensors.torch import load_file


class _Weights:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values

    def get_sd(self) -> dict[str, torch.Tensor]:
        return self.values.copy()


class _Model:
    def __init__(self, model: Any, sampling: Any) -> None:
        self.model = model
        self.sampling = sampling
        self.weight = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]).T

    def get_model_object(self, name: str) -> Any:
        assert name == "model_sampling"
        return self.sampling

    def state_dict_for_saving(
        self,
        clip: dict[str, torch.Tensor] | None,
        vae: dict[str, torch.Tensor],
        vision: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert clip is None
        return {"diffusion.weight": self.weight, **vae, **vision}


@pytest.fixture
def checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, _Model, _Weights, _Weights]:
    initialize_inference_process()
    base = importlib.import_module("dinkster_inference.model_base")
    management = importlib.import_module("dinkster_inference.model_management")
    args = importlib.import_module("dinkster_inference.cli_args").args
    monkeypatch.setattr(args, "disable_metadata", False)
    monkeypatch.setattr(management, "load_models_gpu", lambda models: None)
    model = base.SVD_img2vid.__new__(base.SVD_img2vid)
    torch.nn.Module.__init__(model)
    model.model_type = base.ModelType.EPS
    wrapped = _Model(model, SimpleNamespace())
    root = tmp_path / "output"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "outputMount": "out",
                "mounts": [{"id": "out", "root": str(root), "mode": "readwrite"}],
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return (
        root,
        snapshot,
        wrapped,
        _Weights({"vision.weight": torch.tensor([7.0, -8.0])}),
        _Weights({"vae.weight": torch.tensor([[9.0, -10.0]])}),
    )


def test_checkpoint_bytes_metadata_and_reserved_counter(
    checkpoint: tuple[Path, Path, _Model, _Weights, _Weights],
) -> None:
    root, _, model, vision, vae = checkpoint
    original = model.weight.clone()
    for index, mount in enumerate((None, "out"), start=1):
        assert (
            ImageOnlyCheckpointSave.execute(
                model=model,
                clip_vision=vision,
                vae=vae,
                filename_prefix="checkpoints/image",
                mount=mount,
                prompt={"node": 2},
                extra_pnginfo={"workflow": {"id": "test"}},
            )
            == {}
        )
        path = root / "checkpoints" / f"image_{index:05}_.safetensors"
        values = load_file(path)
        assert set(values) == {"diffusion.weight", "vision.weight", "vae.weight"}
        assert torch.equal(values["diffusion.weight"], original)
        assert torch.equal(values["vision.weight"], torch.tensor([7.0, -8.0]))
        assert torch.equal(values["vae.weight"], torch.tensor([[9.0, -10.0]]))
        with safe_open(path, framework="pt") as saved:
            assert saved.metadata() == {
                "modelspec.architecture": "stable-video-diffusion-img2vid-v1",
                "modelspec.sai_model_spec": "1.0.0",
                "modelspec.implementation": "sgm",
                "modelspec.title": f"image {index}",
                "modelspec.predict_key": "epsilon",
                "prompt": '{"node": 2}',
                "workflow": '{"id": "test"}',
            }
    assert torch.equal(model.weight, original)
    assert not model.weight.is_contiguous()
    assert not list(root.rglob(".dinkster-save-*"))


def test_edm_prediction_keys_survive_disabled_workflow_metadata(
    checkpoint: tuple[Path, Path, _Model, _Weights, _Weights], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, model, vision, vae = checkpoint
    sampling = importlib.import_module("dinkster_inference.model_sampling")
    base = importlib.import_module("dinkster_inference.model_base")
    args = importlib.import_module("dinkster_inference.cli_args").args
    edm = type("EDMV", (sampling.ModelSamplingContinuousEDM, sampling.V_PREDICTION), {})()
    edm.set_parameters(0.125, 8.0, 1.0)
    edm.zsnr = True
    model.sampling = edm
    model.model = SimpleNamespace(model_type=base.ModelType.V_PREDICTION)
    monkeypatch.setattr(args, "disable_metadata", True)
    ImageOnlyCheckpointSave.execute(
        model=model,
        clip_vision=vision,
        vae=vae,
        filename_prefix="edm",
        prompt={"not": "saved"},
        extra_pnginfo={"workflow": "not saved"},
    )
    path = root / "edm_00001_.safetensors"
    values = load_file(path)
    assert values["edm_vpred.sigma_max"].item() == 8.0
    assert values["edm_vpred.sigma_min"].item() == 0.125
    assert values["v_pred"].numel() == values["ztsnr"].numel() == 0
    with safe_open(path, framework="pt") as saved:
        assert saved.metadata() == {"modelspec.predict_key": "v"}


@pytest.mark.parametrize("failure", ("traversal", "revoked", "readonly", "missing_snapshot"))
def test_checkpoint_requires_current_writable_mount(
    checkpoint: tuple[Path, Path, _Model, _Weights, _Weights],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root, snapshot, model, vision, vae = checkpoint
    prefix = "../escape" if failure == "traversal" else "safe"
    if failure == "revoked":
        snapshot.write_text(json.dumps({"mounts": []}))
    elif failure == "readonly":
        data = json.loads(snapshot.read_text())
        data["mounts"][0]["mode"] = "read"
        snapshot.write_text(json.dumps(data))
    elif failure == "missing_snapshot":
        monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT")
    with pytest.raises((AssetError, ValueError)):
        ImageOnlyCheckpointSave.execute(
            model=model, clip_vision=vision, vae=vae, filename_prefix=prefix, mount="out"
        )
    assert not list(root.rglob("*.safetensors"))


def test_checkpoint_schema_and_cold_discovery() -> None:
    schema = ImageOnlyCheckpointSave.schema()
    assert [input.id for input in schema.inputs] == [
        "model",
        "clip_vision",
        "vae",
        "filename_prefix",
        "mount",
    ]
    assert schema.inputs[3].default == "checkpoints/ComfyUI"
    assert isinstance(schema.inputs[4].widget, MountWidget)
    assert schema.inputs[4].adjacent_to == "filename_prefix"
    assert schema.output_node and not schema.idempotent and not schema.outputs
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster_nodes_comfy.checkpoint_save; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stderr
