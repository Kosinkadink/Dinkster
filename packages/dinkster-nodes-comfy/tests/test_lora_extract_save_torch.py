"""CPU coverage for native LoRA extraction and mounted saving."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DINKSTER_ACCELERATOR"] = "cpu"

import dinkster_nodes_comfy.lora_extract_save as lora
import torch
from safetensors import safe_open


class _Patcher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            diffusion_model=SimpleNamespace(linear=object(), scalar=object())
        )
        self.load_device = "cpu"
        self.weights = {
            "diffusion_model.linear.weight": torch.tensor([[3.0, -1.0], [2.0, 4.0]]),
            "diffusion_model.linear.bias": torch.tensor([0.25, -0.5]),
            "diffusion_model.scalar.weight": torch.tensor([7.0]),
        }

    def model_state_dict(self, *, filter_prefix: str) -> dict[str, torch.Tensor]:
        return {key: value for key, value in self.weights.items() if key.startswith(filter_prefix)}

    def patch_weight_to_device(self, key: str, device: str, *, return_weight: bool) -> torch.Tensor:
        assert device == "cpu" and return_weight
        return self.weights[key]


def _runtime(monkeypatch: Any) -> list[object]:
    loaded: list[object] = []
    real_import = lora.importlib.import_module
    safetensors = real_import("safetensors.torch")

    def load_models_gpu(models: list[object]) -> None:
        loaded.extend(models)

    modules = {
        "dinkster_inference.model_management": SimpleNamespace(load_models_gpu=load_models_gpu),
        "dinkster_inference.utils": SimpleNamespace(
            get_attr=lambda value, name: _get_attr(value, name),
            save_torch_file=safetensors.save_file,
        ),
    }
    monkeypatch.setattr(
        lora.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else real_import(name),
    )
    monkeypatch.setattr(lora, "initialize_inference_process", lambda: None)
    return loaded


def _get_attr(value: object, name: str) -> object:
    for segment in name.split("."):
        value = getattr(value, segment)
    return value


def test_schema_stays_torch_free_and_preserves_source_inputs() -> None:
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from dinkster_api.v1 import CLIP_TYPE, MODEL_TYPE, TypeExpr
from dinkster_nodes_comfy.lora_extract_save import LoraSave

schema = LoraSave.schema()
assert schema.node_type == "LoraSave"
assert schema.display_name == "Extract and Save Lora"
assert schema.category == "experimental"
assert schema.search_terms == ("export lora",)
assert tuple(item.id for item in schema.inputs) == (
    "filename_prefix", "rank", "lora_type", "bias_diff", "model_diff", "text_encoder_diff", "mount"
)
assert schema.inputs[4].type == TypeExpr.concrete(MODEL_TYPE)
assert schema.inputs[4].required is False
assert schema.inputs[5].type == TypeExpr.concrete(CLIP_TYPE)
assert schema.inputs[5].required is False
assert schema.inputs[6].adjacent_to == "filename_prefix"
assert schema.output_node and not schema.idempotent
assert "torch" not in sys.modules
assert not any(name.startswith("dinkster_inference.") for name in sys.modules)
""",
        ],
        check=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )


def test_extract_lora_and_calculation_preserve_tiny_cpu_tensor_semantics(monkeypatch: Any) -> None:
    loaded = _runtime(monkeypatch)
    patcher = _Patcher()
    standard = lora.calc_lora_model(
        patcher,
        1,
        "diffusion_model.",
        "diffusion_model.",
        {},
        lora.LORAType.STANDARD,
        bias_diff=True,
    )
    assert loaded == [patcher]
    assert tuple(standard) == (
        "diffusion_model.linear.lora_up.weight",
        "diffusion_model.linear.lora_down.weight",
        "diffusion_model.linear.diff_b",
        "diffusion_model.scalar.diff",
    )
    assert all(
        value.device.type == "cpu" and value.dtype == torch.float16 for value in standard.values()
    )
    assert standard["diffusion_model.linear.lora_up.weight"].shape == (2, 1)
    assert standard["diffusion_model.linear.lora_down.weight"].shape == (1, 2)
    full = lora.calc_lora_model(
        patcher,
        8,
        "diffusion_model.",
        "diffusion_model.",
        {},
        lora.LORAType.FULL_DIFF,
        bias_diff=True,
    )
    assert tuple(full) == (
        "diffusion_model.linear.diff",
        "diffusion_model.linear.diff_b",
        "diffusion_model.scalar.diff",
    )
    assert torch.equal(
        full["diffusion_model.linear.diff"], patcher.weights["diffusion_model.linear.weight"].half()
    )


def test_execute_writes_safetensors_to_selected_mount_and_cleans_temporary_file(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _runtime(monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "outputMount": "out",
                "mounts": [{"id": "out", "root": str(output), "mode": "readwrite"}],
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    created: list[Path] = []
    real_tempfile = lora.tempfile.NamedTemporaryFile

    def temporary_file(*args: Any, **kwargs: Any) -> Any:
        temporary = real_tempfile(*args, **kwargs)
        created.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(lora.tempfile, "NamedTemporaryFile", temporary_file)
    lora.LoraSave.execute(
        filename_prefix="loras/tiny",
        rank=1,
        lora_type="full_diff",
        bias_diff=True,
        model_diff=_Patcher(),
        mount="out",
    )
    saved = list((output / "loras").glob("tiny_*.safetensors"))
    assert len(saved) == 1
    with safe_open(saved[0], framework="pt", device="cpu") as checkpoint:
        assert checkpoint.metadata() is None
        assert tuple(checkpoint.keys()) == (
            "diffusion_model.linear.diff",
            "diffusion_model.linear.diff_b",
            "diffusion_model.scalar.diff",
        )
    assert created and all(not path.exists() for path in created)
