"""Actual safetensors latent files across the public asset boundary."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    AssetRef,
    ExportSnapshot,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_assets import MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_latent.latent_file_tools import LoadLatent, SaveLatent
from dinkster_schema.reporting import capture_saved_artifacts
from safetensors import safe_open
from safetensors.torch import save_file

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


@pytest.mark.parametrize("mount", (None, "alternate"))
def test_save_latent_preserves_filename_tail_and_existing_publications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount: str | None
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
    root = alternate if mount else output
    folder = root / "chosen"
    folder.mkdir()
    existing = folder / "tail__00007_.latent"
    existing.write_bytes(b"complete existing publication")
    samples = {"samples": torch.tensor([[[[-1.25, 7.5, 0.125]]]])}
    previous = {}
    for counter in (8, 9):
        with capture_saved_artifacts() as claims:
            result = SaveLatent.execute(
                samples=samples, filename_prefix="chosen/tail_", mount=mount
            )
        assert result["samples"] is samples
        path = folder / f"tail__{counter:05d}_.latent"
        assert path.is_file()
        assert claims == [
            {
                "digest": digest_file(path),
                "name": path.name,
                "size": path.stat().st_size,
                "mediaType": "application/octet-stream",
                "virtualPath": f"mounts/{mount or 'output'}/chosen/{path.name}",
            }
        ]
        previous[path] = path.read_bytes()
    assert existing.read_bytes() == b"complete existing publication"
    assert len(list(folder.glob("*.latent"))) == 3
    for path, encoded in previous.items():
        assert path.read_bytes() == encoded
    other = output if mount else alternate
    assert not list(other.rglob("*.latent"))


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
    with capture_saved_artifacts():
        assert SaveLatent.execute(samples=samples)["samples"] is samples
        assert (
            SaveLatent.execute(samples=samples, filename_prefix="chosen/latent", mount="alternate")[
                "samples"
            ]
            is samples
        )
    for path in (
        output / "latents/ComfyUI_00001_.latent",
        alternate / "chosen/latent_00001_.latent",
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


HOST_NODES = (LatentInspect,)


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("disabled", (False, True))
@pytest.mark.parametrize("prompt", ({"load": {"inputs": {"seed": 67}}}, {}, None))
def test_latent_export_consumes_submission_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_process: bool,
    disabled: bool,
    prompt: dict | None,
) -> None:
    from dinkster_api.v1 import initialize_inference_process

    initialize_inference_process()
    from dinkster_inference.cli_args import args

    monkeypatch.setattr(args, "disable_metadata", disabled)
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    tensor = torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]])
    save_file(
        {"latent_tensor": tensor, "latent_format_version_0": torch.tensor([])},
        source / "selected.latent",
    )
    mounts = tmp_path / "mounts.json"
    table = MountTable(mounts, index_root=tmp_path / "index")
    table.add(MountDef("input", source), kind="data/latent")
    table.add(MountDef("output", output, mode="readwrite"))
    table.scan("input")
    table.scan("output")
    table.select_output_mount("output")
    asset = table.bind_name("selected.latent", "data/latent")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(mounts))
    extra = {"workflow": {"nodes": [{"id": 37, "title": "latent: caf\u00e9"}]}, "seed": 139}

    async def scenario():
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"latent_metadata_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_latent_files_torch import HOST_NODES\n"
            "from dinkster_inference_wire import register_inference_types\n"
            "from dinkster_api.v1 import initialize_inference_process\n"
            "initialize_inference_process()\n"
            "from dinkster_inference.cli_args import args\n"
            f"args.disable_metadata = {disabled!r}\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="latent-metadata-fixture"\npublisher="dinkster"\n'
            'namespaces=["file-export-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:HOST_NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (str(tmp_path), str(Path(__file__).parent), *(p for p in sys.path if p))
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(mounts),
                "PYTHONPATH": os.pathsep.join(paths),
            }
        )
        try:
            manifests = (native, fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        trust_reserved=True,
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "latent-metadata-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            engine = composer.composition.make_engine(lambda _event: None)
            await engine.run(
                Graph(
                    nodes={
                        "load": GraphNode(
                            "dinkster.latent.load_latent", {"latent": asset.to_wire()}
                        ),
                        "save": GraphNode(
                            "dinkster.latent.save_latent", {"samples": Link("load", "LATENT")}
                        ),
                    }
                ),
                ["save"],
                export_snapshot=ExportSnapshot(prompt=prompt, extra_pnginfo=extra),
            )
        finally:
            await composer.close()

    asyncio.run(scenario())
    paths = list(output.rglob("*.latent"))
    assert len(paths) == 1
    with safe_open(paths[0], framework="pt", device="cpu") as handle:
        expected = (
            None
            if disabled
            else {
                "prompt": json.dumps(prompt) if prompt is not None else "",
                **{key: json.dumps(value) for key, value in extra.items()},
            }
        )
        assert handle.metadata() == expected
        torch.testing.assert_close(handle.get_tensor("latent_tensor"), tensor, rtol=0, atol=0)
        assert handle.get_tensor("latent_format_version_0").numel() == 0


@pytest.mark.parametrize("in_process", (True, False))
def test_latent_exports_attach_exact_mounted_files_on_each_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    source, output, alternate = (tmp_path / name for name in ("input", "output", "alternate"))
    for directory in (source, output, alternate):
        directory.mkdir()
    tensor = torch.tensor([[[[-1.25, 7.5, 0.125], [3.75, -0.5, 9.0]]]])
    save_file(
        {"latent_tensor": tensor, "latent_format_version_0": torch.tensor([])},
        source / "selected.latent",
    )
    mounts = tmp_path / "mounts.json"
    table = MountTable(mounts, index_root=tmp_path / "index")
    table.add(MountDef("input", source), kind="data/latent")
    for name, root in (("output", output), ("alternate", alternate)):
        table.add(MountDef(name, root, mode="readwrite"))
        table.scan(name)
    table.scan("input")
    table.select_output_mount("output")
    asset = table.bind_name("selected.latent", "data/latent")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(mounts))

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        paths = tuple(p for p in sys.path if p)
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(mounts),
                "PYTHONPATH": os.pathsep.join(paths),
            }
        )
        try:
            await composer.add_pack(
                PackSpec(trust_reserved=True, manifest=native, in_process=in_process)
            )
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode("dinkster.latent.load_latent", {"latent": asset.to_wire()}),
                    "default": GraphNode(
                        "dinkster.latent.save_latent", {"samples": Link("load", "LATENT")}
                    ),
                    "alternate": GraphNode(
                        "dinkster.latent.save_latent",
                        {
                            "samples": Link("load", "LATENT"),
                            "filename_prefix": "chosen/asymmetric",
                            "mount": "alternate",
                        },
                    ),
                }
            )
            for attempt in range(2):
                result = await engine.run(graph, ["default", "alternate"])
                assert len(result.artifacts) == 2
                by_node = {artifact.node_id: artifact for artifact in result.artifacts}
                assert set(by_node) == {"default", "alternate"}
                for node, mount, root, prefix in (
                    ("default", "output", output, "latents/ComfyUI"),
                    ("alternate", "alternate", alternate, "chosen/asymmetric"),
                ):
                    relative = f"{prefix}_{attempt + 1:05d}_.latent"
                    path = root / relative
                    artifact = by_node[node]
                    assert artifact.virtual_path == f"mounts/{mount}/{relative}"
                    assert artifact.name == path.name
                    assert artifact.digest == digest_file(path)
                    assert artifact.size == path.stat().st_size
                    assert artifact.media_type == "application/octet-stream"
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        assert set(handle.keys()) == {"latent_tensor", "latent_format_version_0"}
                        torch.testing.assert_close(
                            handle.get_tensor("latent_tensor"), tensor, rtol=0, atol=0
                        )
                        assert handle.get_tensor("latent_format_version_0").numel() == 0
                if attempt:
                    assert "load" in result.cached
                    assert not {"default", "alternate"} & set(result.cached)
            assert len(list(output.rglob("*.latent"))) == 2
            assert len(list(alternate.rglob("*.latent"))) == 2
        finally:
            await composer.close()

    asyncio.run(scenario())
