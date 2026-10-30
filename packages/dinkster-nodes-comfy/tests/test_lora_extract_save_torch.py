"""CPU coverage for native LoRA extraction and mounted saving."""

from __future__ import annotations

import asyncio
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
import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    MODEL_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema.reporting import capture_saved_artifacts
from safetensors import safe_open

from dinkster.compose import PackSpec, ServingComposer


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
    with capture_saved_artifacts():
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


class LoraDifferenceSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-save-host-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP

        model = torch.nn.Module()
        model.diffusion_model = torch.nn.Module()
        model.diffusion_model.linear = torch.nn.Linear(2, 2)
        model.diffusion_model.linear.weight.data.copy_(torch.tensor([[3.0, -1.0], [2.0, 4.0]]))
        model.diffusion_model.linear.bias.data.copy_(torch.tensor([0.25, -0.5]))
        encoder = torch.nn.Module()
        encoder.linear = torch.nn.Linear(2, 1)
        encoder.linear.weight.data.copy_(torch.tensor([[-7.0, 1.5]]))
        encoder.linear.bias.data.copy_(torch.tensor([0.75]))
        clip = CLIP(no_init=True)
        clip.patcher = ModelPatcher(encoder, torch.device("cpu"), torch.device("cpu"))
        clip.cond_stage_model = encoder
        clip.tokenizer = object()
        clip.layer_idx = None
        clip.tokenizer_options = {}
        clip.use_clip_schedule = True
        clip.apply_hooks_to_conds = None
        return cls.outputs(
            model=ModelPatcher(model, torch.device("cpu"), torch.device("cpu")), clip=clip
        )


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("explicit_mount", (False, True))
def test_lora_save_preserves_both_differences_through_native_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, explicit_mount: bool
) -> None:
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    for name, directory in (("output", output), ("alternate", alternate)):
        table.add(MountDef(name, directory, mode="readwrite"))
        table.scan(name)
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        module = f"lora_save_host_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_lora_extract_save_torch import LoraDifferenceSource\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (LoraDifferenceSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="lora-save-host-fixture"\npublisher="dinkster"\n'
            'namespaces=["lora-save-host-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifests = (package / "dinkster-pack.toml", fixture)
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(checkout / "src"),
                        *(str(path) for path in sorted(checkout.glob("packages/*/src"))),
                    )
                ),
            }
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "lora-save-host-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            inputs: dict[str, Any] = {
                "filename_prefix": "loras/asymmetric",
                "rank": 1,
                "lora_type": "full_diff",
                "bias_diff": True,
                "model_diff": Link("source", "model"),
                "text_encoder_diff": Link("source", "clip"),
            }
            if explicit_mount:
                inputs["mount"] = "alternate"
            graph = Graph(
                nodes={
                    "source": GraphNode("lora-save-host-fixture.source"),
                    "save": GraphNode("LoraSave", inputs),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["save"])
                if attempt:
                    assert "source" in result.cached and "save" not in result.cached
        finally:
            await composer.close()

    asyncio.run(scenario())
    selected, other = (alternate, output) if explicit_mount else (output, alternate)
    files = sorted(selected.glob("loras/asymmetric_*.safetensors"))
    assert len(files) == 2 and not list(other.rglob("*.safetensors"))
    expected = {
        "diffusion_model.linear.diff": torch.tensor([[3.0, -1.0], [2.0, 4.0]]),
        "diffusion_model.linear.diff_b": torch.tensor([0.25, -0.5]),
        "text_encoders.linear.diff": torch.tensor([[-7.0, 1.5]]),
        "text_encoders.linear.diff_b": torch.tensor([0.75]),
    }
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert set(handle.keys()) == set(expected) and handle.metadata() is None
            for key, tensor in expected.items():
                actual = handle.get_tensor(key)
                assert actual.dtype == torch.float16
                torch.testing.assert_close(actual, tensor.half(), rtol=0, atol=0)


@pytest.mark.parametrize("in_process", (True, False))
def test_lora_exports_attach_exact_mounted_files_on_each_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    for name, root in (("output", output), ("alternate", alternate)):
        table.add(MountDef(name, root, mode="readwrite"))
        table.scan(name)
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"lora_attachment_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_lora_extract_save_torch import LoraDifferenceSource\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (LoraDifferenceSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="lora-attachment-fixture"\npublisher="dinkster"\n'
            'namespaces=["lora-save-host-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (str(tmp_path), str(Path(__file__).parent), *(p for p in sys.path if p))
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(paths),
            }
        )
        try:
            manifests = (native, fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "lora-attachment-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            options = {
                "rank": 1,
                "lora_type": "full_diff",
                "bias_diff": True,
                "model_diff": Link("source", "model"),
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("lora-save-host-fixture.source"),
                    "combined": GraphNode(
                        "LoraSave",
                        {
                            **options,
                            "filename_prefix": "loras/combined",
                            "text_encoder_diff": Link("source", "clip"),
                        },
                    ),
                    "model": GraphNode(
                        "LoraSave",
                        {**options, "filename_prefix": "loras/model", "mount": "alternate"},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["combined", "model"])
                assert len(result.artifacts) == 2
                assert {artifact.node_id for artifact in result.artifacts} == {"combined", "model"}
                for node, mount, root in (
                    ("combined", "output", output),
                    ("model", "alternate", alternate),
                ):
                    relative = f"loras/{node}_{attempt + 1:05d}_.safetensors"
                    path = root / relative
                    artifact = next(item for item in result.artifacts if item.node_id == node)
                    assert artifact.virtual_path == f"mounts/{mount}/{relative}"
                    assert artifact.name == path.name
                    assert artifact.digest == digest_file(path)
                    assert artifact.size == path.stat().st_size
                    assert artifact.media_type == "application/octet-stream"
                    expected = {
                        "diffusion_model.linear.diff": torch.tensor([[3.0, -1.0], [2.0, 4.0]]),
                        "diffusion_model.linear.diff_b": torch.tensor([0.25, -0.5]),
                    }
                    if node == "combined":
                        expected.update(
                            {
                                "text_encoders.linear.diff": torch.tensor([[-7.0, 1.5]]),
                                "text_encoders.linear.diff_b": torch.tensor([0.75]),
                            }
                        )
                    with safe_open(path, framework="pt", device="cpu") as saved:
                        assert saved.metadata() is None
                        assert set(saved.keys()) == set(expected)
                        for key, tensor in expected.items():
                            actual = saved.get_tensor(key)
                            assert actual.dtype == torch.float16
                            torch.testing.assert_close(actual, tensor.half(), rtol=0, atol=0)
                if attempt:
                    assert "source" in result.cached
                    assert not {"combined", "model"} & set(result.cached)
            assert len(list(output.rglob("*.safetensors"))) == 2
            assert len(list(alternate.rglob("*.safetensors"))) == 2
        finally:
            await composer.close()

    asyncio.run(scenario())
