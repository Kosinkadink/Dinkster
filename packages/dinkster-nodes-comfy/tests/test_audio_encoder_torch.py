"""CPU audio encoder node contracts with a controlled checkpoint/model seam."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_TYPE,
    CORE_STRING,
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_assets import AssetVault, MountDef, MountTable, digest_bytes
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.audio_encoder import (
    AudioEncoderEncode,
    AudioEncoderLoader,
    SheetSage2AudioToABC,
)

from dinkster.compose import PackSpec, ServingComposer


def test_bound_loader_encode_and_abc_preserve_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_inference_process()
    utils = importlib.import_module("dinkster_inference.utils")
    encoders = importlib.import_module("dinkster_inference.audio_encoders.audio_encoders")
    waveform = torch.tensor([[[0.25, -0.75, 0.125]], [[-0.5, 0.625, 0.0]]])
    output = {
        "encoded_audio": torch.tensor([[2.0, -3.0]], dtype=torch.float16),
        "encoded_audio_all_layers": (torch.empty((2, 0)), torch.tensor([7.5])),
        "audio_samples": 3,
        "metadata": {"tag": "unchanged", "extra": (False, 17)},
    }
    scores = {True: ["X:1\nK:C\nC2", "X:2\nK:G\nG"], False: ["X:1\nK:C\n[CEG]"]}

    class Encoder:
        patcher = object()

        def encode_audio(self, audio, sample_rate):
            assert audio is waveform
            assert sample_rate == 22051
            return output

        def generate_abc(self, audio, sample_rate, *, melody_only):
            assert audio is waveform
            assert sample_rate == 22051
            return scores[melody_only]

    encoder = Encoder()
    path = tmp_path / "encoder.safetensors"
    from safetensors.torch import save_file

    save_file({"fixture.weight": torch.tensor([[-2.0, 0.75]])}, path)
    data = path.read_bytes()
    vault = AssetVault(tmp_path / "vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    asset = AssetRef(digest, path.name, len(data), resolver=vault)
    original = utils.load_torch_file

    def safe_load(path, *, safe_load):
        assert safe_load is True
        return original(path, safe_load=safe_load)

    def load(state_dict):
        assert torch.equal(state_dict["fixture.weight"], torch.tensor([[-2.0, 0.75]]))
        return encoder

    monkeypatch.setattr(utils, "load_torch_file", safe_load)
    monkeypatch.setattr(encoders, "load_audio_encoder_from_sd", load)
    loaded = AudioEncoderLoader.execute(audio_encoder_name=asset)["AUDIO_ENCODER"]
    assert loaded is encoder
    assert loaded._dinkster_resident_owner is encoder.patcher
    audio = {"waveform": waveform, "sample_rate": 22051}
    assert (
        AudioEncoderEncode.execute(audio_encoder=loaded, audio=audio)["AUDIO_ENCODER_OUTPUT"]
        is output
    )
    for mode, melody in (("full", False), ("melody", True)):
        assert (
            SheetSage2AudioToABC.execute(audio_encoder=loaded, audio=audio, mode=mode)["abc"]
            is scores[melody]
        )

    monkeypatch.setattr(encoders, "load_audio_encoder_from_sd", lambda _state_dict: None)
    with pytest.raises(RuntimeError, match="does not contain a valid model"):
        AudioEncoderLoader.execute(audio_encoder_name=asset)
    with pytest.raises(TypeError, match="AssetRef"):
        AudioEncoderLoader.execute(audio_encoder_name=str(path))


class _HostEncoder:
    def __init__(self) -> None:
        self.patcher = object()
        self.pid = os.getpid()

    def encode_audio(self, audio: torch.Tensor, sample_rate: int) -> dict[str, object]:
        assert os.getpid() == self.pid
        assert sample_rate == 22051
        assert torch.equal(audio, torch.tensor([[[0.25, -0.75, 0.125]]]))
        return {
            "encoded_audio": audio * -2,
            "encoded_audio_all_layers": (audio + 0.5, torch.empty((1, 0))),
            "audio_samples": 3,
            "metadata": {"rate": sample_rate, "tuple": (True, "kept")},
        }

    def generate_abc(
        self, audio: torch.Tensor, sample_rate: int, *, melody_only: bool
    ) -> list[str]:
        self.encode_audio(audio, sample_rate)
        return ["melody", "C2"] if melody_only else ["full", "[CEG]"]


class _HostSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-encoder-fixture.source",
            inputs=(InputSpec("asset", TypeExpr.concrete(ASSET_TYPE)),),
            outputs=(
                OutputSpec("asset", TypeExpr.concrete(ASSET_TYPE)),
                OutputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, asset: object) -> dict[str, object]:
        initialize_inference_process()
        encoders = importlib.import_module("dinkster_inference.audio_encoders.audio_encoders")

        def load(state_dict: Any) -> _HostEncoder:
            assert torch.equal(state_dict["fixture.weight"], torch.tensor([-2.0, 0.75]))
            return _HostEncoder()

        encoders.load_audio_encoder_from_sd = load
        return cls.outputs(
            asset=asset,
            audio={"waveform": torch.tensor([[[0.25, -0.75, 0.125]]]), "sample_rate": 22051},
        )


class _HostInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-encoder-fixture.inspect",
            inputs=(
                InputSpec("encoded", TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)),
                InputSpec("melody", TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))),
                InputSpec("full", TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, encoded: object, melody: object, full: object) -> dict[str, object]:
        output: Any = materialize_latent(encoded)
        assert torch.equal(output["encoded_audio"], torch.tensor([[[-0.5, 1.5, -0.25]]]))
        assert isinstance(output["encoded_audio_all_layers"], tuple)
        assert torch.equal(
            output["encoded_audio_all_layers"][0], torch.tensor([[[0.75, -0.25, 0.625]]])
        )
        assert output["encoded_audio_all_layers"][1].shape == (1, 0)
        assert output["audio_samples"] == 3
        assert output["metadata"] == {"rate": 22051, "tuple": (True, "kept")}
        assert melody == ["melody", "C2"]
        assert full == ["full", "[CEG]"]
        return cls.outputs(summary="audio encoder affinity and outputs preserved")


_HOST_NODES = (_HostSource, _HostInspect)


@pytest.mark.parametrize("verified,in_process", ((False, True), (True, True), (True, False)))
def test_audio_encoder_ordinary_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, verified: bool
) -> None:
    from safetensors.torch import save_file

    initialize_inference_process()
    encoders = importlib.import_module("dinkster_inference.audio_encoders.audio_encoders")
    monkeypatch.setattr(encoders, "load_audio_encoder_from_sd", encoders.load_audio_encoder_from_sd)
    save_file({"fixture.weight": torch.tensor([-2.0, 0.75])}, tmp_path / "encoder.safetensors")
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    table.add(MountDef(id="encoders", path=tmp_path), kind="model/audio-encoder")
    if verified:
        table.scan("encoders")
    asset = table.bind_name("encoder.safetensors", "model/audio-encoder")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        module = f"audio_encoder_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_audio_encoder_torch import _HOST_NODES\n"
            "from dinkster_nodes_comfy import register_inference_types\nNODES = _HOST_NODES\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            '[pack]\nname="audio-encoder-fixture"\npublisher="dinkster"\n'
            'namespaces=["audio-encoder-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (
            tmp_path,
            Path(__file__).parent,
            checkout / "src",
            *sorted(checkout.glob("packages/*/src")),
            *(Path(p) for p in sys.path if p and Path(p).is_dir()),
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(map(str, paths)),
            }
        )
        manifests = (package / "dinkster-pack.toml", fixture_manifest)
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "audio-encoder-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            nodes = {
                "source": GraphNode("audio-encoder-fixture.source", {"asset": asset.to_wire()}),
                "loader": GraphNode(
                    "AudioEncoderLoader", {"audio_encoder_name": Link("source", "asset")}
                ),
                "encode": GraphNode(
                    "AudioEncoderEncode",
                    {
                        "audio_encoder": Link("loader", "AUDIO_ENCODER"),
                        "audio": Link("source", "audio"),
                    },
                ),
            }
            for mode in ("melody", "full"):
                nodes[mode] = GraphNode(
                    "SheetSage2AudioToABC",
                    {
                        "audio_encoder": Link("loader", "AUDIO_ENCODER"),
                        "audio": Link("source", "audio"),
                        "mode": mode,
                    },
                )
            nodes["inspect"] = GraphNode(
                "audio-encoder-fixture.inspect",
                {
                    "encoded": Link("encode", "AUDIO_ENCODER_OUTPUT"),
                    "melody": Link("melody", "abc"),
                    "full": Link("full", "abc"),
                },
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(Graph(nodes=nodes), ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "audio encoder affinity and outputs preserved"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
