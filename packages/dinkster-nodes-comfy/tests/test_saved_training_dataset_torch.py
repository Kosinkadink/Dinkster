"""Saved training shards preserve ordered tensors and read-only folder authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    LATENT_TYPE,
    AssetError,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.dataset_shards import LoadTrainingDataset

from dinkster.compose import PackSpec, ServingComposer


def seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "datasets"
    directory = root / "saved"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text("{}", encoding="ascii")
    # Hand-derived fixture values distinguish sorted shards, dtypes and nested metadata.
    for name, marker, dtype in (
        ("shard_20.pkl", 19, torch.float16),
        ("shard_03.pkl", 7, torch.float64),
    ):
        torch.save(
            {
                "latents": [{"samples": torch.tensor([[[[marker, -marker]]]], dtype=dtype)}],
                "conditioning": [
                    [
                        [
                            torch.tensor([[[marker + 2.0]]]),
                            {"pooled_output": torch.tensor([marker - 3])},
                        ]
                    ]
                ],
            },
            directory / name,
        )
    (directory / "ignored.pkl").write_bytes(b"not a shard")
    (directory / "shard_wrong.pt").write_bytes(b"wrong suffix")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "datasets",
                        "kind": "directory/dataset",
                        "root": str(root),
                        "index": str(tmp_path / "no-index.json"),
                        "priority": 0,
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return directory


def digests(directory: Path) -> dict[Path, str]:
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def check_outputs(latents: Any, conditioning: Any) -> None:
    assert len(latents) == len(conditioning) == 2
    # Hand-derived: shard_03 precedes shard_20, preserving the seeded marker arithmetic.
    for index, (marker, dtype) in enumerate(((7, torch.float64), (19, torch.float16))):
        samples = materialize_latent(latents[index])["samples"]
        assert samples.dtype == dtype
        assert torch.equal(samples, torch.tensor([[[[marker, -marker]]]], dtype=dtype))
        embedding, metadata = conditioning[index][0]
        assert torch.equal(embedding, torch.tensor([[[marker + 2.0]]]))
        assert set(metadata) == {"pooled_output"}
        assert torch.equal(metadata["pooled_output"], torch.tensor([marker - 3]))


def test_saved_shards_preserve_order_values_and_input_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = seed(tmp_path, monkeypatch)
    before = digests(directory)
    result = LoadTrainingDataset.execute(folder_name="saved")
    check_outputs(result["latents"], result["conditioning"])
    assert digests(directory) == before
    for path in directory.glob("shard_*.pkl"):
        path.unlink()
    with pytest.raises(ValueError, match="No shard files"):
        LoadTrainingDataset.execute(folder_name="saved")


def test_saved_shards_reject_escaped_revoked_and_unsafe_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = seed(tmp_path, monkeypatch)
    with pytest.raises(AssetError):
        LoadTrainingDataset.execute(folder_name="../saved")
    (directory / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(AssetError):
        LoadTrainingDataset.execute(folder_name="saved")
    (directory / "outside").unlink()
    torch.save({"latents": [Path("unsafe")], "conditioning": []}, directory / "shard_00.pkl")
    with pytest.raises(Exception, match="Weights only load failed"):
        LoadTrainingDataset.execute(folder_name="saved")
    (tmp_path / "mounts.json").write_text('{"mounts":[]}', encoding="ascii")
    with pytest.raises(AssetError):
        LoadTrainingDataset.execute(folder_name="saved")


class SavedDatasetInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="saved-dataset-fixture.inspect",
            inputs=(
                InputSpec("latents", TypeExpr.list_of(TypeExpr.concrete(LATENT_TYPE))),
                InputSpec("conditioning", TypeExpr.list_of(TypeExpr.concrete(CONDITIONING_TYPE))),
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, latents: Any, conditioning: Any) -> Any:
        check_outputs(latents, conditioning)
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_saved_shards_ordinary_hosts_reread_without_mutating_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    directory = seed(tmp_path, monkeypatch)
    before = digests(directory)

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        module = f"saved_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_saved_training_dataset_torch import SavedDatasetInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (SavedDatasetInspect,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="saved-dataset-fixture"\npublisher="dinkster"\n'
            'namespaces=["saved-dataset-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (checkout / "src", *sorted(checkout.glob("packages/*/src")))
        composer = ServingComposer(
            worker_env={
                "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts.json"),
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        *(str(p) for p in paths),
                        *(p for p in sys.path if p),
                    )
                ),
            }
        )
        try:
            manifests = (package / "dinkster-pack.toml", fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "saved-dataset-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "load": GraphNode("LoadTrainingDataset", {"folder_name": "saved"}),
                    "inspect": GraphNode(
                        "saved-dataset-fixture.inspect",
                        {name: Link("load", name) for name in ("latents", "conditioning")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
                assert "load" not in result.cached
                assert digests(directory) == before
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert digests(directory) == before
