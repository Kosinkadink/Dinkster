"""Actual safetensors latent files across the public asset boundary."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    VAE_TYPE,
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_assets import MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.latent_file_tools import LoadLatent, SaveLatent
from safetensors import safe_open
from safetensors.torch import save_file
from test_model_file_savers_torch import _Clip, _Vae

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize("versioned", (False, True))
def test_load_latent_converts_to_float_and_scales_only_legacy_files(
    tmp_path: Path, versioned: bool
) -> None:
    path = tmp_path / "selected.latent"
    tensor = torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]], dtype=torch.float64)
    state = {"latent_tensor": tensor}
    if versioned:
        state["latent_format_version_0"] = torch.tensor([])
    save_file(state, str(path))
    asset = AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=SimpleNamespace(resolve=lambda _digest: path),
    )
    actual = LoadLatent.execute(latent=asset)["LATENT"]
    expected = torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]])
    if not versioned:
        expected *= 1.0 / 0.18215
    assert set(actual) == {"samples"}
    assert actual["samples"].dtype == torch.float32
    assert actual["samples"].device.type == "cpu"
    torch.testing.assert_close(actual["samples"], expected, rtol=0, atol=0)


def test_save_latent_preserves_source_and_writes_versioned_tensors_to_selected_mount(
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
    tensor = torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]], dtype=torch.float64)
    samples = {"samples": tensor.transpose(-1, -2), "batch_index": [7], "origin": "source"}
    assert not samples["samples"].is_contiguous()
    assert SaveLatent.execute(samples=samples)["samples"] is samples
    assert (
        SaveLatent.execute(samples=samples, filename_prefix="chosen/latent", mount="alternate")[
            "samples"
        ]
        is samples
    )
    for path in (
        output / "latents/ComfyUI_00001.latent",
        alternate / "chosen/latent_00001.latent",
    ):
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert set(handle.keys()) == {"latent_tensor", "latent_format_version_0"}
            assert handle.metadata() == {"prompt": ""}
            saved = handle.get_tensor("latent_tensor")
            assert saved.dtype == torch.float64
            assert saved.is_contiguous()
            torch.testing.assert_close(saved, tensor.transpose(-1, -2), rtol=0, atol=0)
            assert handle.get_tensor("latent_format_version_0").numel() == 0
    assert samples["batch_index"] == [7] and samples["origin"] == "source"
    assert not samples["samples"].is_contiguous()


class EncoderSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="file-export-fixture.source",
            outputs=(
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(clip=_Clip(), vae=_Vae())


class LatentInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="file-export-fixture.inspect",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: Any) -> dict[str, object]:
        samples = materialize_latent(latent)["samples"]
        # Reciprocal multiplication has different float32 rounding from division.
        expected = torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]]) * (1.0 / 0.18215)
        assert samples.dtype == torch.float32 and samples.device.type == "cpu"
        torch.testing.assert_close(samples, expected, rtol=0, atol=0)
        return cls.outputs(summary="latent-values-exact")


HOST_NODES = (EncoderSource, LatentInspect)


@pytest.mark.parametrize("verified", (False, True))
@pytest.mark.parametrize("in_process", (True, False))
def test_latent_and_encoder_exports_write_exact_files_through_native_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verified: bool, in_process: bool
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")
    source, output, alternate = (tmp_path / name for name in ("input", "output", "alternate"))
    for directory in (source, output, alternate):
        directory.mkdir()
    save_file(
        {"latent_tensor": torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]])},
        source / "selected.latent",
    )
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    table.add(MountDef("input", source), kind="data/latent")
    table.add(MountDef("output", output, mode="readwrite"))
    table.add(MountDef("alternate", alternate, mode="readwrite"))
    table.select_output_mount("output")
    table.scan("output")
    table.scan("alternate")
    if verified:
        table.scan("input")
    asset = table.bind_name("selected.latent", "data/latent")
    assert type(asset).__name__ == ("AssetRef" if verified else "LocalAssetRef")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        module = f"file_export_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_latent_files_torch import HOST_NODES\n"
            "from dinkster_nodes_comfy import register_inference_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="file-export-fixture"\npublisher="dinkster"\n'
            'namespaces=["file-export-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:HOST_NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        manifests = (native, fixture)
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
                        worker_group=None if in_process else "file-export-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("file-export-fixture.source"),
                    "load": GraphNode("LoadLatent", {"latent": asset.to_wire()}),
                    "save": GraphNode("SaveLatent", {"samples": Link("load", "LATENT")}),
                    "inspect": GraphNode(
                        "file-export-fixture.inspect", {"latent": Link("save", "samples")}
                    ),
                    "clip": GraphNode(
                        "CLIPSave", {"clip": Link("source", "clip"), "mount": "alternate"}
                    ),
                    "vae": GraphNode("VAESave", {"vae": Link("source", "vae")}),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect", "clip", "vae"])
                assert result.outputs["inspect"]["summary"].resolve() == "latent-values-exact"
                if attempt:
                    assert {"source", "load"} <= set(result.cached)
                    assert not {"save", "clip", "vae"} & set(result.cached)
        finally:
            await composer.close()

    asyncio.run(scenario())
    expected = {
        "latents/ComfyUI": (
            "latent_tensor",
            torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]]) * (1.0 / 0.18215),
        ),
        "vae/ComfyUI_vae": ("decoder.weight", torch.tensor([[1.25, -3.5]], dtype=torch.float64)),
    }
    for prefix, (key, tensor) in expected.items():
        paths = sorted(
            output.glob(
                f"{prefix}_*.safetensors" if prefix.startswith("vae/") else f"{prefix}_*.latent"
            )
        )
        assert len(paths) == 2
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as handle:
                torch.testing.assert_close(handle.get_tensor(key), tensor, rtol=0, atol=0)
                assert handle.metadata() == {"prompt": ""}
    for prefix, key, tensor in (
        ("clip_l", "left", torch.tensor([1.25, -3.5])),
        ("clip_g", "right", torch.tensor([[8, -2]], dtype=torch.int32)),
        ("", "other", torch.tensor([9.0], dtype=torch.float64)),
    ):
        name = f"ComfyUI_{prefix}" if prefix else "ComfyUI"
        paths = sorted((alternate / "clip").glob(f"{name}_*.safetensors"))
        if not prefix:
            paths = [path for path in paths if path.stem.rsplit("_", 1)[0] == name]
        assert len(paths) == 2
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as handle:
                assert list(handle.keys()) == [key]
                torch.testing.assert_close(handle.get_tensor(key), tensor, rtol=0, atol=0)
                assert handle.metadata() == {"format": "pt", "prompt": ""}
    assert not (output / "clip").exists() and not (alternate / "vae").exists()
