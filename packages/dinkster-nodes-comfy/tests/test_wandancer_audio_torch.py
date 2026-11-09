"""CPU contracts and ordinary-host transport for native WanDancer audio features."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_TYPE,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.wandancer_audio import WanDancerEncodeAudio

from dinkster.compose import PackSpec, ServingComposer


def _audio(waveform: torch.Tensor, sample_rate: int = 15360) -> dict[str, object]:
    return {"waveform": waveform, "sample_rate": sample_rate}


def _assert_output(
    result: dict[str, object], *, fps: float, scale: float, fps_string: str
) -> torch.Tensor:
    encoded = result["audio_encoder_output"]
    assert isinstance(encoded, dict)
    feature = encoded["audio_feature"]
    assert isinstance(feature, torch.Tensor)
    assert feature.ndim == 3 and feature.shape[0] == 1 and feature.shape[2] == 35
    assert feature.dtype == torch.float32 and feature.device.type == "cpu"
    assert encoded["fps"] == fps and encoded["audio_inject_scale"] == scale
    assert result["fps_string"] == fps_string
    return feature


def test_schema_and_cold_source_tree_discovery() -> None:
    schema = WanDancerEncodeAudio.define_schema()
    assert schema.node_type == "WanDancerEncodeAudio"
    assert schema.category == "model/conditioning/wan/dancer"
    assert [(item.id, item.type.types) for item in schema.inputs] == [
        ("audio", (AUDIO_TYPE,)),
        ("video_frames", (CORE_INT,)),
        ("audio_inject_scale", (CORE_FLOAT,)),
    ]
    assert [item.default for item in schema.inputs[1:]] == [149, 1.0]
    assert [(item.widget.min, item.widget.max, item.widget.step) for item in schema.inputs[1:]] == [
        (1, 16384, 4),
        (0.0, 10.0, 0.01),
    ]
    assert (
        schema.inputs[2].doc
        == "The scale for the audio features when injected into the video model."
    )
    assert [(item.id, item.type.types, item.display_name, item.doc) for item in schema.outputs] == [
        ("audio_encoder_output", (AUDIO_ENCODER_OUTPUT_TYPE,), "audio_encoder_output", ""),
        (
            "fps_string",
            (CORE_STRING,),
            "fps_string",
            "The calculated fps based on the audio length and the number of video frames. "
            "Used in the prompt.",
        ),
    ]

    root = Path(__file__).resolve().parents[3]
    python_path = os.pathsep.join(
        (str(root), str(root / "src"), *(str(path) for path in sorted(root.glob("packages/*/src"))))
    )
    script = "\n".join(
        (
            "import json",
            "import sys",
            "import dinkster_nodes_comfy.wandancer_audio as module",
            "module.WanDancerEncodeAudio.define_schema()",
            "prefixes = ('torch', 'torchvision', 'dinkster_inference', 'scipy')",
            "forbidden = sorted(name for name in sys.modules if any("
            "name == prefix or name.startswith(prefix + '.') for prefix in prefixes))",
            "print(json.dumps({'path': module.__file__, 'forbidden': forbidden}))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": python_path},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    discovery = json.loads(completed.stdout)
    assert Path(discovery["path"]).resolve() == (
        root / "packages/dinkster-nodes-comfy/src/dinkster_nodes_comfy/wandancer_audio.py"
    )
    assert discovery["forbidden"] == []


def test_silence_features_have_only_the_constant_spectrum_mfcc_offset() -> None:
    silence = torch.zeros((1, 1, 15360), dtype=torch.float32)
    before = silence.clone()
    result = dict(
        WanDancerEncodeAudio.execute(audio=_audio(silence), video_frames=4, audio_inject_scale=1.0)
    )
    feature = _assert_output(result, fps=3.75, scale=1.0, fps_string=" \u5e27\u7387\u662f3.7500")
    assert feature.shape == (1, 31, 35)
    assert torch.equal(silence, before)
    torch.testing.assert_close(feature[:, :, 0], torch.zeros_like(feature[:, :, 0]), rtol=0, atol=0)
    torch.testing.assert_close(
        feature[:, :, 1], torch.full_like(feature[:, :, 1], -100 * np.sqrt(128)), rtol=0, atol=2e-4
    )
    torch.testing.assert_close(
        feature[:, :, 2:], torch.zeros_like(feature[:, :, 2:]), rtol=0, atol=0
    )


def test_real_mono_stereo_first_batch_features_and_input_immutability() -> None:
    samples = torch.arange(15360, dtype=torch.float32) / 15360
    first = torch.stack((samples, samples * 3)).unsqueeze(0)
    ignored = torch.full((1, 2, samples.numel()), 99.0)
    batched = torch.cat((first, ignored), dim=0)
    before = batched.clone()
    mono = first.mean(dim=1, keepdim=True)
    stereo = dict(
        WanDancerEncodeAudio.execute(audio=_audio(batched), video_frames=3, audio_inject_scale=2.5)
    )
    expected = dict(
        WanDancerEncodeAudio.execute(audio=_audio(mono), video_frames=3, audio_inject_scale=2.5)
    )
    stereo_feature = _assert_output(
        stereo, fps=3.0, scale=2.5, fps_string=" \u5e27\u7387\u662f3.0000"
    )
    expected_feature = _assert_output(
        expected, fps=3.0, scale=2.5, fps_string=" \u5e27\u7387\u662f3.0000"
    )
    torch.testing.assert_close(stereo_feature, expected_feature, rtol=0, atol=0)
    assert torch.isfinite(stereo_feature).all()
    assert torch.count_nonzero(stereo_feature[:, :, 21:33]) > 0
    assert torch.equal(batched, before)


@pytest.mark.parametrize(
    ("video_frames", "fps", "text"),
    (
        (1, 10.0, " \u5e27\u7387\u662f10.0000"),
        (2, 15.0, " \u5e27\u7387\u662f15.0000"),
        (3, 30.0, ", \u5e27\u7387\u662f30fps\u3002"),
    ),
)
def test_controlled_feature_rows_preserve_column_order_and_fps_branches(
    monkeypatch: pytest.MonkeyPatch, video_frames: int, fps: float, text: str
) -> None:
    feature_module = types.ModuleType("dinkster_nodes_comfy._wandancer_audio_features")
    feature_module.quick_tempo_estimate = lambda *_args, **_kwargs: 120.0
    feature_module._compute_mel_spectrogram = lambda *_args, **_kwargs: np.ones(
        (128, 3), dtype=np.float32
    )
    feature_module.power_to_db = lambda values, **_kwargs: values
    feature_module.compute_onset_envelope = lambda *_args, **_kwargs: np.array(
        [1, 2, 3], dtype=np.float32
    )
    feature_module.compute_mfcc = lambda *_args, **_kwargs: (
        np.arange(60, dtype=np.float32).reshape(3, 20) + 10
    )
    feature_module.compute_chroma_cens = lambda **_kwargs: (
        np.arange(36, dtype=np.float32).reshape(12, 3) + 100
    )
    feature_module.detect_onset_peaks = lambda *_args, **_kwargs: np.array([1])
    feature_module.estimate_tempo_from_onset = lambda *_args, **_kwargs: 120.0
    feature_module.track_beats = lambda *_args, **_kwargs: np.array([2])
    audio_module = types.ModuleType("dinkster_inference.audio")
    audio_module.resample = lambda waveform, *_args: waveform
    management_module = types.ModuleType("dinkster_inference.model_management")
    management_module.intermediate_device = lambda: torch.device("cpu")
    monkeypatch.setitem(sys.modules, feature_module.__name__, feature_module)
    monkeypatch.setitem(sys.modules, audio_module.__name__, audio_module)
    monkeypatch.setitem(sys.modules, management_module.__name__, management_module)

    waveform = torch.ones((1, 1, 512), dtype=torch.float32)
    result = dict(
        WanDancerEncodeAudio.execute(
            audio=_audio(waveform), video_frames=video_frames, audio_inject_scale=0.25
        )
    )
    feature = _assert_output(result, fps=fps, scale=0.25, fps_string=text)
    envelope = np.array([1, 2, 3], dtype=np.float32)[:, None]
    mfcc = np.arange(60, dtype=np.float32).reshape(3, 20) + 10
    chroma = (np.arange(36, dtype=np.float32).reshape(12, 3) + 100).T
    expected = np.concatenate(
        (
            envelope,
            mfcc,
            chroma,
            np.array([[0], [1], [0]], dtype=np.float32),
            np.array([[0], [0], [1]], dtype=np.float32),
        ),
        axis=1,
    )
    torch.testing.assert_close(feature, torch.from_numpy(expected).unsqueeze(0), rtol=0, atol=0)


class AudioSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-audio-fixture.source",
            outputs=(OutputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        waveform = torch.stack((-torch.ones(15360), torch.ones(15360))).reshape(1, 2, 15360)
        return cls.outputs(audio=_audio(waveform))


class AudioConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-audio-fixture.consumer",
            inputs=(
                InputSpec("encoded", TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)),
                InputSpec("fps_string", TypeExpr.concrete(CORE_STRING)),
                InputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, encoded: object, fps_string: str, audio: object) -> dict[str, object]:
        source = materialize_audio(audio)
        waveform = source["waveform"]
        assert isinstance(waveform, torch.Tensor) and waveform.shape == (1, 2, 15360)
        assert torch.equal(waveform[0, 0], -torch.ones(15360))
        assert torch.equal(waveform[0, 1], torch.ones(15360))
        value = materialize_latent(encoded)
        feature = value["audio_feature"]
        assert isinstance(feature, torch.Tensor) and feature.shape == (1, 31, 35)
        assert feature.dtype is torch.float32 and feature.device.type == "cpu"
        assert value["fps"] == 3.0 and value["audio_inject_scale"] == 0.5
        assert fps_string == " \u5e27\u7387\u662f3.0000"
        expected = torch.zeros((1, 31, 35), dtype=torch.float32)
        expected[:, :, 1] = -100 * np.sqrt(128)
        torch.testing.assert_close(feature[:, :, 1], expected[:, :, 1], rtol=0, atol=2e-4)
        torch.testing.assert_close(feature[:, :, 0], expected[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(feature[:, :, 2:], expected[:, :, 2:], rtol=0, atol=0)
        return cls.outputs(summary="wandancer-audio-host-values")


HOST_NODES = (AudioSource, AudioConsumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_wandancer_audio_runs_twice_through_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        root = package.parents[1]
        source_paths = (
            str(root),
            str(root / "src"),
            *(str(path) for path in sorted(root.glob("packages/*/src"))),
        )
        source_module = package / "src/dinkster_nodes_comfy/wandancer_audio.py"
        entry = tmp_path / "wandancer_audio_host_entry.py"
        entry.write_text(
            "from pathlib import Path\n"
            "import dinkster_nodes_comfy.wandancer_audio as native\n"
            f"assert Path(native.__file__).resolve() == Path({str(source_module)!r}).resolve()\n"
            "from test_wandancer_audio_torch import HOST_NODES\n"
            "from dinkster_api.v1 import (AUDIO_TYPE, register_audio_value_type,\n"
            "    register_inference_types, resolver_from_env)\n"
            "def register_types(registry):\n"
            "    register_inference_types(registry)\n"
            "    if AUDIO_TYPE not in registry:\n"
            "        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="wandancer-audio-fixture"\npublisher="dinkster"\n'
            'namespaces=["wandancer-audio-fixture"]\n[pack.entry]\n'
            'nodes="wandancer_audio_host_entry:HOST_NODES"\n'
            'types="wandancer_audio_host_entry:register_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), *source_paths)
                ),
            }
        )
        try:
            native = package / "dinkster-pack.toml"
            native_before = native.read_bytes()
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "wandancer-audio-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("wandancer-audio-fixture.source"),
                    "encode": GraphNode(
                        "WanDancerEncodeAudio",
                        {
                            "audio": Link("source", "audio"),
                            "video_frames": 3,
                            "audio_inject_scale": 0.5,
                        },
                    ),
                    "consumer": GraphNode(
                        "wandancer-audio-fixture.consumer",
                        {
                            "encoded": Link("encode", "audio_encoder_output"),
                            "fps_string": Link("encode", "fps_string"),
                            "audio": Link("source", "audio"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert (
                    result.outputs["consumer"]["summary"].resolve() == "wandancer-audio-host-values"
                )
                assert set(result.cached) == (
                    set() if attempt == 0 else {"source", "encode", "consumer"}
                )
            assert native.read_bytes() == native_before
        finally:
            await composer.close()

    asyncio.run(scenario())
