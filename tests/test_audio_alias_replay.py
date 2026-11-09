"""Execute first-party advanced audio saves against pinned source receipts.

MP3 encoding compares exact bytes within one codec build; decoding separately
compares fixed source-minted files against exact PCM profiles for known decoder
builds.
"""

from __future__ import annotations

import base64
import hashlib
import json
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
)
from dinkster_audio.audio_io import LoadAudio
from dinkster_audio.audio_save import SaveAudioAdvanced
from dinkster_values.audio_codec import audio_window, effective_audio_facts

from tests.test_audio_io import _bound, _mount, _save
from tools.gen_audio_alias_receipts import encode_contract

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = json.loads((ROOT / "tests/goldens/audio_aliases_b78cec87.json").read_text("utf-8"))

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
# PyAV 18.1.0 cp311-abi3 win_amd64. Only outputs measured by hosted Windows
# validation are recorded; every unmeasured tuple remains fail-closed.
WINDOWS_AV18_AVCODEC_SHA256 = "777704d8914bb99e5d78f3a0e7424031b5376621f2e7c60e83b84e77fcc7e2d7"
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
    WINDOWS_AV18_AVCODEC_SHA256: {
        # Batch 1 was measured by hosted full-validation run 36015503535.
        ("V0", 0): "62176983aa99946c747d934d6218f967e22ecacdd66041487766d42e2ba9cef3",
        ("V0", 1): "555fa6d9cbc326dd5e211a59f63281b4d00a3c8c1e943d8444291e4ace4a385b",
        ("128k", 0): "662e3b65fb456f7b30bf441939f1f9e76b250453d62a316f61f5d2103206b657",
        ("128k", 1): "971513a801ee730464247e5ab9cec9ada08258e9ae7159ddadf4de606fd82609",
        ("320k", 0): "ee0600dfdf604e249081a69824a1abebbc887d2d58a6d073335b4bbdb2420bfa",
        ("320k", 1): "4f7f23da94b83a11ec2edd04131c13688823bf335f4c2b3d74deaef0a8961e7a",
    },
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


@pytest.mark.parametrize("case", RECEIPTS["saveCases"])
def test_advanced_save_replay(
    case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, snapshot = _mount(tmp_path, monkeypatch)
    audio = _audio(RECEIPTS["saveInput"])
    original = audio["waveform"].copy()
    refs = _save(
        SaveAudioAdvanced,
        audio=audio,
        format=case["format"],
        filename_prefix="aliases/receipt",
        mount="comfy-output",
        quality=case["quality"],
    )
    np.testing.assert_array_equal(audio["waveform"], original)
    assert len(refs) == len(case["outputs"]) == 2
    for batch, (ref, expected) in enumerate(zip(refs, case["outputs"], strict=True)):
        ref = _bound(ref, snapshot)
        assert ref.name.startswith("receipt")
        assert (root / "aliases" / ref.name).is_file()
        if case["format"] == "mp3":
            assert len(case["encoded"]) == len(refs)
            ref = _mp3_source(ref, case["encoded"][batch], tmp_path)
        actual = LoadAudio.execute(audio=ref)["AUDIO"]
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
