"""CPU contracts for loading RAFT optical-flow assets."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_STRING,
    OPTICAL_FLOW_TYPE,
    AssetIntegrityError,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_assets import AssetVault, MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.optical_flow import OpticalFlowLoader

from dinkster.compose import PackSpec, ServingComposer


class _TinyRaft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_encoder = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        self.context_encoder = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
        self.update_block = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)


def _raft_large(*, weights: object, progress: bool) -> _TinyRaft:
    assert weights is None and progress is False
    return _TinyRaft()


def _state() -> dict[str, torch.Tensor]:
    return {
        "feature_encoder.weight": torch.tensor([[0.125, -3.5], [7.25, -0.5]], dtype=torch.float64),
        "context_encoder.weight": torch.tensor([[11.0, -13.0]], dtype=torch.float64),
        "update_block.weight": torch.tensor([[19.5]], dtype=torch.float64),
    }


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, _digest: str) -> Path:
        return self.path


def _checkpoint(
    path: Path, state: dict[str, torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    state = _state() if state is None else state
    torch.save(state, path)
    return state


def _local_asset(path: Path) -> LocalAssetRef:
    mounts = MountTable()
    mounts.add(MountDef(id="optical-flow", path=path.parent), kind="model/optical-flow")
    asset = mounts.bind_name(path.name, "model/optical-flow")
    assert isinstance(asset, LocalAssetRef)
    return asset


def _assert_patcher(patcher: object, expected: dict[str, torch.Tensor]) -> None:
    from dinkster_inference.model_patcher import CoreModelPatcher

    assert type(patcher) is CoreModelPatcher
    assert type(patcher.model) is _TinyRaft
    assert patcher.load_device == patcher.offload_device == torch.device("cpu")
    assert not patcher.model.training
    actual = patcher.model.state_dict()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key].dtype is torch.float32
        torch.testing.assert_close(actual[key], value.float(), rtol=0, atol=0)


def test_schema_is_canonical_and_cold_imports_no_tensor_runtime(tmp_path: Path) -> None:
    assert OpticalFlowLoader in COMFY_NODES
    schema = OpticalFlowLoader.schema()
    assert schema.node_type == "OpticalFlowLoader"
    assert schema.display_name == "Load Optical Flow Model"
    assert schema.category == "model/loaders"
    assert len(schema.inputs) == 1
    assert schema.inputs[0].id == "model_name"
    assert schema.inputs[0].type.types == (ASSET_TYPE,)
    assert isinstance(schema.inputs[0].widget, AssetWidget)
    assert schema.inputs[0].widget.kind == "model/optical-flow"
    assert [(output.id, output.type.types) for output in schema.outputs] == [
        ("OPTICAL_FLOW", (OPTICAL_FLOW_TYPE,))
    ]

    root = Path(__file__).resolve().parents[3]
    python_path = os.pathsep.join(
        (str(root), str(root / "src"), *(str(path) for path in sorted(root.glob("packages/*/src"))))
    )
    cold_discovery = "\n".join(
        (
            "import json",
            "import sys",
            "import dinkster_nodes_comfy.optical_flow as module",
            "module.OpticalFlowLoader.schema()",
            "forbidden = sorted(",
            "    name for name in sys.modules",
            "    if name == 'torch' or name.startswith('torch.')",
            "    or name == 'torchvision' or name.startswith('torchvision.')",
            "    or name == 'dinkster_inference' or name.startswith('dinkster_inference.')",
            ")",
            "print(json.dumps({'path': module.__file__, 'forbidden': forbidden}))",
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            cold_discovery,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": python_path,
        },
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    discovery = json.loads(result.stdout)
    assert (
        Path(discovery["path"]).resolve()
        == root / "packages/dinkster-nodes-comfy/src/dinkster_nodes_comfy/optical_flow.py"
    )
    assert discovery["forbidden"] == []


@pytest.mark.parametrize("verified", (False, True))
def test_loader_preserves_real_safe_checkpoint_weights_and_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verified: bool
) -> None:
    checkpoint = tmp_path / "tiny-raft.pt"
    state = _checkpoint(checkpoint)
    source = {key: value.clone() for key, value in state.items()}
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    asset: AssetRef | LocalAssetRef = _local_asset(checkpoint)
    if verified:
        asset = AssetRef(
            digest_file(checkpoint),
            checkpoint.name,
            checkpoint.stat().st_size,
            resolver=_Resolver(checkpoint),
        )

    import torchvision.models.optical_flow as optical_flow

    monkeypatch.setattr(optical_flow, "raft_large", _raft_large)
    loaded = OpticalFlowLoader.execute(model_name=asset)["OPTICAL_FLOW"]
    _assert_patcher(loaded, state)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    for key, value in state.items():
        assert value.dtype is torch.float64
        torch.testing.assert_close(value, source[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "missing_prefix", ("feature_encoder.", "context_encoder.", "update_block.")
)
def test_loader_rejects_invalid_or_unbound_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_prefix: str
) -> None:
    checkpoint = tmp_path / "invalid.pt"
    _checkpoint(
        checkpoint,
        {key: value for key, value in _state().items() if not key.startswith(missing_prefix)},
    )
    monkeypatch.setattr("torchvision.models.optical_flow.raft_large", _raft_large)
    with pytest.raises(ValueError, match="feature_encoder.*context_encoder.*update_block"):
        OpticalFlowLoader.execute(model_name=_local_asset(checkpoint))
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        OpticalFlowLoader.execute(model_name=str(checkpoint))


def test_verified_asset_rejects_tampered_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "tampered.pt"
    _checkpoint(checkpoint)
    asset = AssetRef(
        digest_file(checkpoint),
        checkpoint.name,
        checkpoint.stat().st_size,
        resolver=_Resolver(checkpoint),
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    monkeypatch.setattr("torchvision.models.optical_flow.raft_large", _raft_large)
    with pytest.raises(AssetIntegrityError):
        OpticalFlowLoader.execute(model_name=asset)


class OpticalFlowFixtureSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="optical-flow-fixture.source",
            outputs=(OutputSpec("asset", TypeExpr.concrete(ASSET_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        if os.environ.get("OPTICAL_FLOW_FIXTURE_INSTALL_FACTORY") == "1":
            import torchvision.models.optical_flow as optical_flow

            optical_flow.raft_large = _raft_large
        return cls.outputs(
            asset=AssetRef(
                os.environ["OPTICAL_FLOW_DIGEST"],
                "tiny-raft.pt",
                int(os.environ["OPTICAL_FLOW_SIZE"]),
            ).to_wire()
        )


class OpticalFlowFixtureConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="optical-flow-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(OPTICAL_FLOW_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: object) -> dict[str, object]:
        _assert_patcher(model, _state())
        return cls.outputs(summary="resident RAFT weights are float32, eval, and CPU-offloaded")


@pytest.mark.parametrize("in_process", (True, False))
def test_loader_crosses_ordinary_hosts_twice_with_resident_optical_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, in_process: bool
) -> None:
    checkpoint = tmp_path / "tiny-raft.pt"
    _checkpoint(checkpoint)
    vault = AssetVault(tmp_path / "vault")
    digest = digest_file(checkpoint)
    with vault.writer(digest) as writer:
        writer.write(checkpoint.read_bytes())
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    monkeypatch.setenv("OPTICAL_FLOW_DIGEST", digest)
    monkeypatch.setenv("OPTICAL_FLOW_SIZE", str(checkpoint.stat().st_size))
    if in_process:
        monkeypatch.setattr("torchvision.models.optical_flow.raft_large", _raft_large)

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        native = root / "dinkster-pack.toml"
        module = "optical_flow_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from test_optical_flow_torch import (\n"
            "    OpticalFlowFixtureConsumer, OpticalFlowFixtureSource,\n"
            ")\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (OpticalFlowFixtureSource, OpticalFlowFixtureConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "optical-flow-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["optical-flow-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "DINKSTER_ACCELERATOR": "cpu",
            "DINKSTER_ASSET_VAULT": str(vault.root),
            "OPTICAL_FLOW_DIGEST": digest,
            "OPTICAL_FLOW_SIZE": str(checkpoint.stat().st_size),
            "OPTICAL_FLOW_FIXTURE_INSTALL_FACTORY": "0" if in_process else "1",
            "PYTHONPATH": os.pathsep.join(
                (str(tmp_path), str(Path(__file__).parent), os.environ["PYTHONPATH"])
            ),
        }
        composer = ServingComposer(worker_env=environment)
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest,
                        worker_group="optical-flow-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("optical-flow-fixture.source"),
                    "load": GraphNode("OpticalFlowLoader", {"model_name": Link("source", "asset")}),
                    "consumer": GraphNode(
                        "optical-flow-fixture.consumer", {"model": Link("load", "OPTICAL_FLOW")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for run in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == (
                    "resident RAFT weights are float32, eval, and CPU-offloaded"
                )
                if run == 1:
                    assert set(result.cached) == {"source", "load", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
