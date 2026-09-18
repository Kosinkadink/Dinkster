"""Generate audio I/O goldens from ComfyUI e20d433a.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies (torch, torchaudio, av):

    COMFYUI_ROOT=/path/to/ComfyUI \
      /path/to/python tools/gen_audio_io_goldens.py

The ComfyUI checkout must be clean and pinned to the commit below. Run the
generator twice and compare the printed sha256 before committing the fixture.

Each case stores the exact fixture bytes it fed the pinned code (waveform or
container bytes, base64) plus the pinned outputs, so the Dinkster-side tests
replay identical inputs instead of regenerating them.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

BASELINE = "e20d433a4966dcc88fa5abbae6ace824cb78b263"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "audio_io_e20d433a.json"


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _tone(np: Any, samples: int, sample_rate: int, channels: int) -> Any:
    """A deterministic tonal mix that is nontrivial for lossy encoders."""
    timeline = np.arange(samples, dtype=np.float64) / sample_rate
    left = 0.55 * np.sin(2.0 * np.pi * 440.0 * timeline) + 0.25 * np.sin(
        2.0 * np.pi * 1318.5 * timeline
    )
    if channels == 1:
        return np.asarray(left[None, :], dtype=np.float32)
    right = 0.45 * np.sin(2.0 * np.pi * 554.37 * timeline) + 0.2 * np.sin(
        2.0 * np.pi * 220.0 * timeline
    )
    return np.asarray(np.stack([left, right]), dtype=np.float32)


def _wav_bytes(np: Any, waveform: Any, sample_rate: int) -> bytes:
    quantized = np.clip(np.rint(waveform * 32767.0), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(int(waveform.shape[0]))
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(quantized.T.tobytes())
    return buffer.getvalue()


def build_goldens(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    import av  # pyright: ignore[reportMissingImports]
    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from comfy_api.latest import _ui  # pyright: ignore[reportMissingImports]
    from comfy_extras import nodes_audio  # pyright: ignore[reportMissingImports]

    def pinned_encode(waveform: Any, sample_rate: int, format: str, quality: str) -> bytes:
        """AudioSaveHelper.save_audio's encode path for one batch element,
        without the output-directory and metadata plumbing."""
        clip = torch.from_numpy(np.ascontiguousarray(waveform))
        encode_rate = sample_rate
        if format == "opus":
            if encode_rate > 48000:
                encode_rate = 48000
            elif encode_rate not in _ui.AudioSaveHelper._OPUS_RATES:
                for rate in sorted(_ui.AudioSaveHelper._OPUS_RATES):
                    if rate > encode_rate:
                        encode_rate = rate
                        break
                if encode_rate not in _ui.AudioSaveHelper._OPUS_RATES:
                    encode_rate = 48000
            if encode_rate != sample_rate:
                import torchaudio  # pyright: ignore[reportMissingImports]

                clip = torchaudio.functional.resample(clip, sample_rate, encode_rate)
        layout = "mono" if clip.shape[0] == 1 else "stereo"
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format=format)
        if format == "opus":
            # The Ogg muxer randomizes its stream serial, so the raw pinned
            # bytes are not reproducible; bitexact pins the serial without
            # changing any encoded audio. flac and mp3 are already stable.
            from av.container import Flags  # pyright: ignore[reportMissingImports]

            container.flags = container.flags | Flags.bitexact.value
        if format == "opus":
            stream = container.add_stream("libopus", rate=encode_rate, layout=layout)
            stream.bit_rate = {
                "64k": 64000,
                "96k": 96000,
                "128k": 128000,
                "192k": 192000,
                "320k": 320000,
            }[quality]
        elif format == "mp3":
            stream = container.add_stream("libmp3lame", rate=encode_rate, layout=layout)
            if quality == "V0":
                stream.codec_context.qscale = 1
            elif quality == "128k":
                stream.bit_rate = 128000
            elif quality == "320k":
                stream.bit_rate = 320000
        else:
            stream = container.add_stream("flac", rate=encode_rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(
            clip.movedim(0, 1).reshape(1, -1).float().numpy(), format="flt", layout=layout
        )
        frame.sample_rate = encode_rate
        frame.pts = 0
        container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
        container.close()
        return buffer.getvalue()

    def pinned_load(data: bytes, suffix: str) -> tuple[Any, int]:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            path = handle.name
        try:
            waveform, rate = nodes_audio.load(path)
        finally:
            os.unlink(path)
        return waveform.numpy(), int(rate)

    def decoded_record(data: bytes, suffix: str) -> dict[str, object]:
        decoded, rate = pinned_load(data, suffix)
        return {
            "decodedSha256": hashlib.sha256(
                np.ascontiguousarray(decoded, dtype=np.float32).tobytes()
            ).hexdigest(),
            "decodedShape": list(decoded.shape),
            "decodedRate": rate,
        }

    cases: dict[str, object] = {}

    wav_wave = _tone(np, 11025, 44100, 2)
    wav_data = _wav_bytes(np, wav_wave, 44100)
    cases["load_wav_stereo_44100"] = {
        "wavB64": _b64(wav_data),
        **decoded_record(wav_data, ".wav"),
    }

    save_cases = [
        ("save_flac_stereo_44100", "flac", "", 44100, 2, ".flac"),
        ("save_mp3_v0_mono_44100", "mp3", "V0", 44100, 1, ".mp3"),
        ("save_mp3_320k_stereo_48000", "mp3", "320k", 48000, 2, ".mp3"),
        ("save_opus_128k_stereo_44100", "opus", "128k", 44100, 2, ".opus"),
        ("save_opus_64k_mono_16000", "opus", "64k", 16000, 1, ".opus"),
    ]
    for name, format_name, quality, sample_rate, channels, suffix in save_cases:
        waveform = _tone(np, sample_rate // 4, sample_rate, channels)
        encoded = pinned_encode(waveform, sample_rate, format_name, quality)
        cases[name] = {
            "waveformB64": _b64(np.ascontiguousarray(waveform).tobytes()),
            "waveformShape": list(waveform.shape),
            "sampleRate": sample_rate,
            "quality": quality,
            "encodedB64": _b64(encoded),
            **decoded_record(encoded, suffix),
        }

    empty = []
    for duration, sample_rate, channels in [
        (60.0, 44100, 2),
        (1.5, 8000, 1),
        (0.25, 2, 2),
        (0.75, 2, 1),
        (0.0, 44100, 2),
    ]:
        result = nodes_audio.EmptyAudio.execute(duration, sample_rate, channels)
        value = result.result[0]
        assert not bool(value["waveform"].any())
        empty.append(
            {
                "duration": duration,
                "sampleRate": sample_rate,
                "channels": channels,
                "shape": list(value["waveform"].shape),
                "outputRate": int(value["sample_rate"]),
            }
        )
    cases["empty_audio"] = empty

    return {
        "revision": BASELINE,
        "generator": "tools/gen_audio_io_goldens.py",
        "avVersion": av.__version__,
        "cases": cases,
    }


def main() -> None:
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    content = (json.dumps(build_goldens(comfy_root.resolve()), indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"wrote {OUT} sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
