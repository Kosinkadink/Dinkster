"""Execute the serialized audio replacement graphs against pinned source receipts.

The backend has no workflow replacement applier; the graph builder below is a
test consumer of the public replacement wire. Source prompt import is tested
separately, without claiming a browser replacement/import integration run.
MP3 encoding compares exact bytes within one codec build; decoding separately
compares fixed source-minted files against exact PCM profiles for known decoder
builds.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dinkster_assets import (
    AssetRef,
    AssetVault,
    digest_bytes,
    register_asset_type,
    resolver_from_env,
)
from dinkster_assets.audio import register_audio_value_type
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import translate_prompt
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import Graph, GraphNode, Link, validate
from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_nodes_media_io import MEDIA_IO_NODES, LoadAudio, register_media_types
from dinkster_schema import (
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_from_wire,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_values.audio_codec import audio_window, effective_audio_facts
from dinkster_workers import InProcessWorker

from tests.test_audio_io import _mount
from tools.gen_audio_alias_receipts import encode_contract

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = json.loads((ROOT / "tests/goldens/audio_aliases_b78cec87.json").read_text("utf-8"))
REGISTRY = json.loads(
    (ROOT / "packages/dinkster-nodes-media-io/comfy-aliases.json").read_text("utf-8")
)
RECORDS = {record["source"]["nodeClass"]: record for record in REGISTRY["records"]}

# PyAV's bundled mp3float decoder can differ by binary build and CPU dispatch.
# Any unrecorded libavcodec binary or output refuses until profiled in issue #1436.
MP3_LINUX_WINDOWS_BUILD_OUTPUTS = {
    ("V0", 0): "62176983aa99946c747d934d6218f967e22ecacdd66041487766d42e2ba9cef3",
    ("V0", 1): "555fa6d9cbc326dd5e211a59f63281b4d00a3c8c1e943d8444291e4ace4a385b",
    ("128k", 0): "662e3b65fb456f7b30bf441939f1f9e76b250453d62a316f61f5d2103206b657",
    ("128k", 1): "971513a801ee730464247e5ab9cec9ada08258e9ae7159ddadf4de606fd82609",
    ("320k", 0): "ee0600dfdf604e249081a69824a1abebbc887d2d58a6d073335b4bbdb2420bfa",
    ("320k", 1): "4f7f23da94b83a11ec2edd04131c13688823bf335f4c2b3d74deaef0a8961e7a",
}
# PyAV 16.0.1 cp312/cp313 manylinux_2_28_x86_64.
LINUX_AVCODEC_SHA256 = "9457aae3915cf55e35c198e496b0eee9bfedc48cb724d810be06f01c054dba68"
# PyAV 18.1.0 cp311-abi3 manylinux_2_28_x86_64.
LINUX_AV18_AVCODEC_SHA256 = "d0f0b65dd8d194278309dce64a9661040f0f7bd77d0d20733a0d7a06bdb26a5b"
# PyAV 16.0.1 cp312 macosx_14_0_arm64.
MACOS_AVCODEC_SHA256 = "e207c496c6ab21a132e0b9519c4d2057b364cc4055a13825fa1ea1d58b4f80f7"
# PyAV 16.0.1 cp312/cp313 win_amd64.
WINDOWS_AVCODEC_SHA256 = "273eb25aa4f81c9b60be7bab35cb83341c3acb2eb57e4c7973e3ed80a6039181"
MP3_DECODER_BUILD_OUTPUTS = {
    LINUX_AVCODEC_SHA256: MP3_LINUX_WINDOWS_BUILD_OUTPUTS,
    LINUX_AV18_AVCODEC_SHA256: MP3_LINUX_WINDOWS_BUILD_OUTPUTS,
    MACOS_AVCODEC_SHA256: {
        ("V0", 0): "44f2b7177542b900d58ef72f7591f3166c4bb81f836b3c4414d884eb1e4d57cf",
        ("V0", 1): "524e9f40e13f11b6249569e68fab26f9977e930e2269ac3875fecc76c6841291",
        ("128k", 0): "9cbc0c32473984be15200e1914c6a623b9bd2985cda0af51e24c3252e5a6ed25",
        ("128k", 1): "fae107d8d397070e55d86101649cbb99fef51355a60ea7a3d72e514c221d44c8",
        ("320k", 0): "c5b9e1983a5e38e2ff5d6d75c573295b3a8604b51d2e25e8c3c20faeaa950a37",
        ("320k", 1): "d34415e655b723b80e233eb07e9c459b634b71983d18db1c8a0c492d62694a8a",
    },
    WINDOWS_AVCODEC_SHA256: MP3_LINUX_WINDOWS_BUILD_OUTPUTS,
}


def _audio(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_rate": record["sample_rate"],
        "waveform": np.frombuffer(base64.b64decode(record["data"]), dtype="<f4")
        .reshape(record["shape"])
        .copy(),
    }


def _mp3_source(ref: AssetRef, recorded: dict[str, Any], tmp_path: Path) -> AssetRef:
    assert ref.read_bytes() == encode_contract(recorded["contract"], recorded["submitted"])
    data = base64.b64decode(recorded["data"])
    vault = AssetVault(tmp_path / "source")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    return AssetRef(digest, "source.mp3", len(data), "audio/mpeg", resolver=vault)


@cache
def _libavcodec_sha256() -> str:
    package = distribution("av")
    binaries = [
        Path(str(package.locate_file(file)))
        for file in package.files or ()
        if (file.name.lower().startswith("libavcodec") or file.name.lower().startswith("avcodec"))
        and (
            file.name.lower().endswith(".dll")
            or file.name.lower().endswith(".dylib")
            or ".so." in file.name.lower()
        )
    ]
    assert len(binaries) == 1, f"expected one bundled libavcodec binary, found {binaries}"
    with binaries[0].open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _assert_mp3_decode_exact(waveform: np.ndarray, quality: str, batch: int) -> None:
    build = _libavcodec_sha256()
    actual = hashlib.sha256(np.ascontiguousarray(waveform, dtype="<f4").tobytes()).hexdigest()
    expected = MP3_DECODER_BUILD_OUTPUTS.get(build, {}).get((quality, batch))
    assert expected is not None, (
        f"unrecognized libavcodec {build} MP3 output {quality}/{batch}={actual}; "
        "record its exact decoder profile in issue #1436"
    )
    assert actual == expected, (
        f"unrecorded libavcodec {build} MP3 output {quality}/{batch}: "
        f"expected {expected}, got {actual}"
    )


def _endpoint(reference: str) -> tuple[str, str]:
    if ":" in reference:
        name, port = reference.split(":", 1)
        return name, port
    return "main", reference


def _run_case(node_class: str, inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rule = RECORDS[node_class]["replacement"]
    case = next(
        case
        for case in rule["cases"]
        if case.get("when", {"kind": "always"})["kind"] == "always"
        or inputs.get(case["when"]["input"]) == case["when"]["value"]
    )
    value = inputs.get("audio", inputs.get("audio_file"))
    audio_type = TypeExpr.concrete("comfy.AUDIO")
    value_type = TypeExpr.asset_of(audio_type) if isinstance(value, AssetRef) else audio_type

    class Fixture(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.audio_alias_fixture",
                outputs=(OutputSpec("value", value_type),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            return cls.outputs(value=value)

    native_nodes = [*MEDIA_IO_NODES, *FOUNDATION_NODES, Fixture]
    schemas = build_schemas(native_nodes)
    definitions = {"main": {"type": case["to"]}, **case.get("nodes", {})}
    bound = {name: dict(node.get("values", {})) for name, node in definitions.items()}
    for target, mapping in case["inputs"].items():
        assert mapping["kind"] == "copy"
        name, port = _endpoint(target)
        if mapping["input"] in inputs:
            incoming = inputs[mapping["input"]]
            bound[name][port] = Link("fixture", "value") if incoming is value else incoming
    for target, family in case.get("inputFamilies", {}).items():
        name, port = _endpoint(target)
        assert family["kind"] == "members"
        for member in family["members"]:
            mapping = member["inputs"]["value"]
            assert mapping["kind"] == "constant"
            bound[name][port + "." + member["suffix"]] = mapping["value"]
    for link in case.get("links", []):
        name, port = _endpoint(link["to"])
        bound[name][port] = Link(*_endpoint(link["from"]))
    graph = Graph(
        nodes={
            "fixture": GraphNode(Fixture.schema().node_type, {}),
            **{name: GraphNode(node["type"], bound[name]) for name, node in definitions.items()},
        }
    )
    targets = {"main", *(_endpoint(port)[0] for port in case.get("outputs", {}))}
    assert validate(graph, schemas, sorted(targets)) == []
    registry = TypeRegistry()
    register_core_types(registry)
    resolver = value.resolver if isinstance(value, AssetRef) else resolver_from_env()
    register_asset_type(registry, resolver)
    register_audio_value_type(registry, "comfy.AUDIO", resolver)
    register_media_types(registry)
    engine = Engine(
        schemas=schemas,
        registry=registry,
        worker=InProcessWorker(build_node_types(native_nodes), registry),
        cache=MemoryLRUCache(),
    )
    result = asyncio.run(engine.run(graph, sorted(targets)))
    outputs = {
        source: result.outputs[name][port].resolve()
        for target, source in case.get("outputs", {}).items()
        for name, port in [_endpoint(target)]
    }
    return outputs, {name: value.resolve() for name, value in result.outputs["main"].items()}


@pytest.mark.parametrize("case", RECEIPTS["loadCases"])
def test_vhs_load_replay(case: dict[str, Any], tmp_path: Path) -> None:
    data = base64.b64decode(RECEIPTS["loadWav"])
    vault = AssetVault(tmp_path / "vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    ref = AssetRef(digest, "input.wav", len(data), "audio/wav", resolver=vault)
    inputs = {"duration": case["duration"]}
    if case["nodeClass"] == "VHS_LoadAudio":
        inputs.update(audio_file=ref, seek_seconds=case["start"])
    else:
        inputs.update(audio=ref, start_time=case["start"])
    outputs, _ = _run_case(case["nodeClass"], inputs)
    expected = _audio(case["output"])
    facts = effective_audio_facts(outputs["audio"])
    assert facts["frames"] / facts["sample_rate"] == case["loadedDuration"]
    np.testing.assert_array_equal(
        audio_window(outputs["audio"], 0, facts["frames"])["waveform"], expected["waveform"]
    )
    assert set(outputs) == {"audio"}
    assert "duration output is not mapped" in RECORDS[case["nodeClass"]]["replacement"]["note"]


@pytest.mark.parametrize("case", RECEIPTS["cropCases"])
def test_crop_replay(case: dict[str, Any]) -> None:
    inputs = {
        "audio": _audio(RECEIPTS["cropInput"]),
        "start_time": case["start_time"],
        "end_time": case["end_time"],
    }
    expected = _audio(case["output"])["waveform"]
    if expected.shape[-1] == 0:
        with pytest.raises(ExecutionError, match="start must land before"):
            _run_case("AudioCrop", inputs)
        return
    outputs, _ = _run_case("AudioCrop", inputs)
    actual = audio_window(outputs["audio"], 0, effective_audio_facts(outputs["audio"])["frames"])[
        "waveform"
    ]
    if case["end_time"] == "30":
        assert actual.shape[-1] == expected.shape[-1] + 1
        np.testing.assert_array_equal(actual[..., :-1], expected)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", RECEIPTS["saveCases"])
def test_advanced_save_replay(
    case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _mount(tmp_path, monkeypatch)
    audio = _audio(RECEIPTS["saveInput"])
    inputs = {"audio": audio, "format": case["format"], "filename_prefix": "aliases/receipt"}
    if case["quality"] is not None:
        inputs["quality"] = case["quality"]
    outputs, native = _run_case("SaveAudioAdvanced", inputs)
    np.testing.assert_array_equal(outputs["audio"]["waveform"], audio["waveform"])
    refs = native["audios"]
    assert len(refs) == len(case["outputs"]) == 2
    for batch, (ref, expected) in enumerate(zip(refs, case["outputs"], strict=True)):
        assert ref.name.startswith("receipt")
        assert (root / "aliases" / ref.name).is_file()
        if case["format"] == "mp3":
            assert len(case["encoded"]) == len(refs)
            ref = _mp3_source(ref, case["encoded"][batch], tmp_path)
        actual = LoadAudio.execute(audio=ref)["audio"]
        facts = effective_audio_facts(actual)
        assert facts["frames"] == expected["shape"][-1]
        assert facts["sample_rate"] == expected["sample_rate"]
        waveform = audio_window(actual, 0, expected["shape"][-1])["waveform"]
        assert waveform.shape == tuple(expected["shape"])
        if case["format"] == "opus":
            # Reuse the existing libopus-version/resampler bound, without widening it.
            np.testing.assert_allclose(waveform, _audio(expected)["waveform"], rtol=0, atol=0.05)
        elif case["format"] == "mp3":
            assert isinstance(case["quality"], str)
            _assert_mp3_decode_exact(waveform, case["quality"], batch)
        else:
            np.testing.assert_array_equal(waveform, _audio(expected)["waveform"])


@pytest.mark.parametrize("format", [None, "unrecognized", "mp3", "opus"])
def test_advanced_save_defaults(
    format: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mount(tmp_path, monkeypatch)
    inputs: dict[str, Any] = {"audio": _audio(RECEIPTS["saveInput"])}
    if format is not None:
        inputs["format"] = format
    _, native = _run_case("SaveAudioAdvanced", inputs)
    expected_format = format if format in {"mp3", "opus"} else "flac"
    quality = {"flac": None, "mp3": "V0", "opus": "128k"}[expected_format]
    (receipt,) = [
        case
        for case in RECEIPTS["saveCases"]
        if case["format"] == expected_format and case["quality"] == quality
    ]
    for batch, (ref, expected) in enumerate(zip(native["audios"], receipt["outputs"], strict=True)):
        assert ref.name.endswith("." + expected_format)
        if expected_format == "mp3":
            assert len(receipt["encoded"]) == len(native["audios"])
            ref = _mp3_source(ref, receipt["encoded"][batch], tmp_path)
        loaded = LoadAudio.execute(audio=ref)["audio"]
        actual = audio_window(loaded, 0, effective_audio_facts(loaded)["frames"])["waveform"]
        if expected_format == "opus":
            np.testing.assert_allclose(actual, _audio(expected)["waveform"], rtol=0, atol=0.05)
        elif expected_format == "mp3":
            assert quality is not None
            _assert_mp3_decode_exact(actual, quality, batch)
        else:
            np.testing.assert_array_equal(actual, _audio(expected)["waveform"])
    assert "Missing quality uses native" in RECORDS["SaveAudioAdvanced"]["replacement"]["note"]


def test_crop_template_import() -> None:
    template = RECEIPTS["template"]
    assert template["repository"] == "Comfy-Org/workflow_templates"
    assert template["commit"] == "2e56ca49dfae00aa500b220a6ef6cedb0d8b51d3"
    (node,) = template["nodes"]
    start, end = node["widgets_values"]
    source_schemas = {
        schema.node_type: schema for schema in map(schema_from_wire, REGISTRY["sourceSchemas"])
    }
    imported = translate_prompt(
        {
            "crop": {
                "class_type": node["type"],
                "inputs": {"audio": ["source", 0], "start_time": start, "end_time": end},
            },
            "source": {"class_type": "EmptyAudio", "inputs": {"duration": 30.0}},
            "preview": {"class_type": "PreviewAudio", "inputs": {"audio": ["crop", 0]}},
        },
        source_schemas,
    )
    crop = imported.graph.nodes["crop"]
    assert isinstance(crop, GraphNode)
    assert crop.node_type == "comfy.AudioCrop"
    outputs, _ = _run_case(
        "AudioCrop", {"audio": _audio(RECEIPTS["cropInput"]), "start_time": start, "end_time": end}
    )
    np.testing.assert_array_equal(
        audio_window(outputs["audio"], 0, 160)["waveform"],
        _audio(RECEIPTS["cropCases"][0]["output"])["waveform"],
    )


@pytest.mark.parametrize("format", ["flac", "mp3", "opus"])
def test_advanced_save_prompt_import(format: str) -> None:
    schemas = {
        schema.node_type: schema for schema in map(schema_from_wire, REGISTRY["sourceSchemas"])
    }
    imported = translate_prompt(
        {
            "load": {"class_type": "VHS_LoadAudioUpload", "inputs": {"audio": "input.wav"}},
            "save": {
                "class_type": "SaveAudioAdvanced",
                "inputs": {"audio": ["load", 0], "format": format},
            },
        },
        schemas,
    )
    assert imported.targets == ("save",)
    assert imported.graph.nodes["save"].inputs["audio"] == Link("load", "audio")


def test_audio_vae_dependencies_and_annotation_are_not_execution_claims() -> None:
    assert "sampler_rate is annotation-only" in RECEIPTS["audioKeyNote"]
    assert set(RECEIPTS["dependencies"]) == {
        "VAEEncodeAudio",
        "VAEDecodeAudio",
        "VAEDecodeAudioTiled",
        "EmptyLatentAudio",
    }
    assert not set(RECEIPTS["dependencies"]).intersection(RECORDS)
