"""Execute pinned audio alias sources with .venv-torch/bin/python.

Set COMFY_AUDIO_ROOT, VHS_ROOT and AUDIO_CROP_ROOT to clean pinned checkouts.
Set AUDIO_TEMPLATE_ROOT to a workflow_templates clone containing the template pin.
The generating environment uses torch 2.13.0+cpu, av 18.1.0 and numpy 2.5.2.
VHS classes and their local-file helpers are compiled unchanged from their AST
to avoid importing unrelated video/server nodes. ComfyUI's saver runs normally.
Run twice and compare the printed digest before committing the fixture.
MP3 encoding is replayed exactly within the installed codec build; fixed
source-minted files separately preserve the historical decoder PCM evidence.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import av
import numpy as np

COMFY_PIN = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
COMFY_AUDIT_PIN = "f00bfd610cb001381603669e2cc01160ae37aaf3"
VHS_PIN = "4d907bee61e92c2e65af3bd6383a4e4d356126d1"
CROP_PIN = "ac339561973f0c1e56db2f9d40f11b0fddda6763"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests/goldens/audio_aliases_b78cec87.json"


def packed(waveform: Any, rate: int) -> dict[str, Any]:
    array = np.ascontiguousarray(waveform, dtype="<f4")
    return {
        "shape": list(array.shape),
        "sample_rate": rate,
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def encode_contract(contract: dict[str, Any], submitted: dict[str, Any]) -> bytes:
    """Replay source-recorded encoder inputs without importing a Dinkster saver."""
    array = np.frombuffer(base64.b64decode(submitted["data"]), dtype="<f4").reshape(
        submitted["shape"]
    )
    assert submitted["sample_rate"] == contract["rate"]
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=contract["container"]) as container:
        stream = container.add_stream(
            contract["codec"], rate=contract["rate"], layout=contract["layout"]
        )
        if "qscale" in contract:
            stream.codec_context.qscale = contract["qscale"]
        else:
            stream.bit_rate = contract["bitRate"]
        frame = av.AudioFrame.from_ndarray(
            array, format=contract["sampleFormat"], layout=contract["layout"]
        )
        frame.sample_rate = contract["rate"]
        frame.pts = contract["pts"]
        container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return buffer.getvalue()


@contextmanager
def _capture_mp3() -> Iterator[list[dict[str, Any]]]:
    """Observe the unmodified pinned source's encoder calls and submitted frames."""
    records: list[dict[str, Any]] = []
    original_open = av.open

    def recorded_open(*args: Any, **kwargs: Any) -> Any:
        container = original_open(*args, **kwargs)

        class Container:
            def __getattr__(self, name: str) -> Any:
                return getattr(container, name)

            def add_stream(self, codec: str, **settings: Any) -> Any:
                stream = container.add_stream(codec, **settings)
                contract = {
                    "container": kwargs["format"],
                    "codec": codec,
                    "rate": settings["rate"],
                    "layout": settings["layout"],
                }
                record: dict[str, Any] = {"contract": contract}
                records.append(record)

                class Stream:
                    def __getattr__(self, name: str) -> Any:
                        return getattr(stream, name)

                    def __setattr__(self, name: str, value: Any) -> None:
                        setattr(stream, name, value)

                    def encode(self, frame: Any) -> Any:
                        if frame is not None:
                            assert "submitted" not in record
                            assert frame.sample_rate == contract["rate"]
                            assert frame.layout.name == contract["layout"]
                            contract.update(sampleFormat=frame.format.name, pts=frame.pts)
                            if stream.codec_context.qscale:
                                contract["qscale"] = bool(stream.codec_context.qscale)
                            else:
                                contract["bitRate"] = stream.bit_rate
                            record["submitted"] = packed(frame.to_ndarray(), frame.sample_rate)
                        return stream.encode(frame)

                return Stream()

        return Container()

    with patch.object(av, "open", recorded_open):
        yield records


def _checkout(variable: str, pin: str) -> Path:
    root = Path(os.environ[variable]).resolve()
    for arguments, expected in ((["rev-parse", "HEAD"], pin), (["status", "--porcelain"], "")):
        actual = subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"{variable} must be clean and pinned to {pin}")
    return root


def _definitions(path: Path, names: set[str], namespace: dict[str, Any]) -> None:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in selected} != names:
        raise RuntimeError(f"missing source definitions in {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def load_sources() -> tuple[Any, dict[str, Any], type[Any], dict[str, object]]:
    comfy = _checkout("COMFY_AUDIO_ROOT", COMFY_PIN)
    vhs = _checkout("VHS_ROOT", VHS_PIN)
    crop = _checkout("AUDIO_CROP_ROOT", CROP_PIN)
    for path, symbols in (
        ("comfy_extras/nodes_audio.py", {"SaveAudioAdvanced", "load"}),
        ("comfy_api/latest/_ui.py", {"AudioSaveHelper"}),
    ):
        audit_source = subprocess.check_output(
            ["git", "-C", str(comfy), "show", f"{COMFY_AUDIT_PIN}:{path}"], text=True
        )
        trees = [ast.parse(text) for text in (audit_source, (comfy / path).read_text("utf-8"))]
        definitions = [
            {
                node.name: ast.dump(node)
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in symbols
            }
            for tree in trees
        ]
        if definitions[0] != definitions[1] or definitions[0].keys() != symbols:
            raise RuntimeError(f"audio implementation differs from audit pin: {path}")
    sys.path.insert(0, str(comfy))
    import torch  # pyright: ignore[reportMissingImports]
    from comfy.cli_args import args  # pyright: ignore[reportMissingImports]

    args.cpu = True
    args.disable_metadata = True
    import folder_paths  # pyright: ignore[reportMissingImports]
    from comfy_extras import nodes_audio  # pyright: ignore[reportMissingImports]
    from imageio_ffmpeg import get_ffmpeg_exe  # pyright: ignore[reportMissingImports]

    namespace: dict[str, Any] = {
        "os": os,
        "re": re,
        "subprocess": subprocess,
        "torch": torch,
        "folder_paths": folder_paths,
        "ffmpeg_path": get_ffmpeg_exe(),
        "ENCODE_ARGS": ("utf-8", "backslashreplace"),
        "audio_extensions": ["mp3", "mp4", "wav", "ogg"],
    }
    _definitions(
        vhs / "videohelpersuite/utils.py",
        {"get_audio", "strip_path", "validate_path", "is_url", "is_safe_path"},
        namespace,
    )
    _definitions(vhs / "videohelpersuite/nodes.py", {"LoadAudio", "LoadAudioUpload"}, namespace)
    spec = importlib.util.spec_from_file_location("pinned_audio_crop", crop / "src/crop.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provenance = {
        "comfyCommit": COMFY_PIN,
        "comfyAuditCommit": COMFY_AUDIT_PIN,
        "auditAudioDefinitionsIdentical": True,
        "vhsCommit": VHS_PIN,
        "cropCommit": CROP_PIN,
        "files": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "ComfyUI/comfy_extras/nodes_audio.py": comfy / "comfy_extras/nodes_audio.py",
                "ComfyUI/comfy_api/latest/_ui.py": comfy / "comfy_api/latest/_ui.py",
                "VHS/videohelpersuite/nodes.py": vhs / "videohelpersuite/nodes.py",
                "VHS/videohelpersuite/utils.py": vhs / "videohelpersuite/utils.py",
                "audio-separation-nodes-comfyui/src/crop.py": crop / "src/crop.py",
            }.items()
        },
    }
    return nodes_audio, namespace, module.AudioCrop, provenance


def build_receipts() -> dict[str, object]:
    nodes, vhs, crop, provenance = load_sources()
    import folder_paths  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    def signal(batch: int, channels: int, frames: int) -> Any:
        return (
            ((np.arange(batch * channels * frames) % 127 - 63) / 128)
            .astype(np.float32)
            .reshape(batch, channels, frames)
        )

    crop_input = signal(1, 2, 480)
    crop_cases = []
    for start, end in [
        ("0:10", "0:20"),
        ("10", "20"),
        ("-1", "2"),
        ("0", "30"),
        ("2", "2"),
        ("00:05", "00:09"),
        ("005", "009"),
        ("40", "50"),
    ]:
        output = crop().main(
            {"waveform": torch.from_numpy(crop_input), "sample_rate": 16}, start, end
        )[0]
        crop_cases.append(
            {"start_time": start, "end_time": end, "output": packed(output["waveform"], 16)}
        )

    with tempfile.TemporaryDirectory(prefix="dinkster-audio-alias-source-") as directory:
        root = Path(directory)
        folder_paths.set_input_directory(directory)
        folder_paths.set_output_directory(directory)
        raw = signal(1, 2, 2400)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(8000)
            writer.writeframes((raw[0].T * 32768).astype("<i2").tobytes())
        wav = buffer.getvalue()
        (root / "input.wav").write_bytes(wav)
        loads = []
        for node_name in ("LoadAudio", "LoadAudioUpload"):
            for start, duration in [(0, 0), (0.05, 0.1), (0.1, 0)]:
                kwargs = (
                    {"audio_file": str(root / "input.wav"), "seek_seconds": start}
                    if node_name == "LoadAudio"
                    else {"audio": "input.wav", "start_time": start}
                )
                audio, seconds = vhs[node_name]().load_audio(**kwargs, duration=duration)
                loads.append(
                    {
                        "nodeClass": "VHS_" + node_name,
                        "start": start,
                        "duration": duration,
                        "loadedDuration": seconds,
                        "output": packed(audio["waveform"], audio["sample_rate"]),
                    }
                )

        save_input = signal(2, 2, 2048)
        saves = []
        for format, qualities in [
            ("flac", [None]),
            ("mp3", ["V0", "128k", "320k"]),
            ("opus", ["64k", "96k", "128k", "192k", "320k"]),
        ]:
            for quality in qualities:
                settings = {"format": format}
                if quality is not None:
                    settings["quality"] = quality
                audio = {"waveform": torch.from_numpy(save_input.copy()), "sample_rate": 48000}
                with _capture_mp3() if format == "mp3" else nullcontext([]) as encoded:
                    result = nodes.SaveAudioAdvanced.execute(
                        audio=audio, filename_prefix="clip", format=settings
                    )
                assert result[0] is audio
                outputs = []
                paths = sorted(root.glob(f"clip_*.{format}"))
                if format == "mp3":
                    assert len(encoded) == len(paths) == 2
                    for record, path in zip(encoded, paths, strict=True):
                        source_bytes = path.read_bytes()
                        assert (
                            encode_contract(record["contract"], record["submitted"]) == source_bytes
                        )
                        record["data"] = base64.b64encode(source_bytes).decode("ascii")
                for path in paths:
                    decoded, rate = nodes.load(str(path))
                    outputs.append(packed(decoded.numpy()[None], rate))
                    path.unlink()
                assert len(outputs) == 2
                case = {"format": format, "quality": quality, "outputs": outputs}
                if format == "mp3":
                    case["encoded"] = encoded
                saves.append(case)

    template_pin = "2e56ca49dfae00aa500b220a6ef6cedb0d8b51d3"
    template_path = "templates/audio-chatterbox_tts_dialog.json"
    template_bytes = subprocess.check_output(
        [
            "git",
            "-C",
            os.environ["AUDIO_TEMPLATE_ROOT"],
            "show",
            f"{template_pin}:{template_path}",
        ]
    )
    template = json.loads(template_bytes)
    return {
        "format": "dinkster-audio-alias-replay/2",
        "source": provenance,
        "runtime": {
            "torch": torch.__version__,
            "av": av.__version__,
            "numpy": np.__version__,
            "libraries": av.library_versions,
        },
        "template": {
            "repository": "Comfy-Org/workflow_templates",
            "commit": template_pin,
            "path": template_path,
            "sha256": hashlib.sha256(template_bytes).hexdigest(),
            "nodes": [node for node in template["nodes"] if node["type"] == "AudioCrop"],
        },
        "audioKeyNote": (
            "sampler_rate is annotation-only in ComfyUI AudioDict; runtime uses sample_rate."
        ),
        "dependencies": {
            "VAEEncodeAudio": "Requires a generic audio VAE encoder or Stable Audio binding.",
            "EmptyLatentAudio": (
                "Requires the Stable Audio latent contract (64 channels, temporal ratio 2048)."
            ),
            "VAEDecodeAudio": (
                "Existing native alias executes MiniMax Music 3 DAV only; "
                "generic/Stable Audio decode remains a dependency."
            ),
            "VAEDecodeAudioTiled": (
                "Existing native alias executes MiniMax Music 3 DAV only; "
                "generic/Stable Audio tiled decode remains a dependency."
            ),
        },
        "cropInput": packed(crop_input, 16),
        "cropCases": crop_cases,
        "loadWav": base64.b64encode(wav).decode("ascii"),
        "loadCases": loads,
        "saveInput": packed(save_input, 48000),
        "saveCases": saves,
    }


if __name__ == "__main__":
    content = (json.dumps(build_receipts(), indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"{OUT}: sha256={hashlib.sha256(content).hexdigest()}")
