"""Real pinned VHS source info, with independent alpha and timestamp fixtures.

Run with DINKSTER_COMFYUI_ROOT, DINKSTER_COMFYUI_PYTHON and DINKSTER_VHS_ROOT set.
The worker interpreter needs the pinned ComfyUI and VHS requirements, matching
CPU torch/torchvision/torchaudio, and Dinkster's compat numba dependency. The host
uses the normal locked Dinkster environment (including its own PyAV version).
VHS_FORCE_FFMPEG_PATH selects the actual FFmpeg executable used by VHS.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_nodes_media_io.video_ops import AssembleVideo, DisassembleVideo, VideoInfo
from dinkster_values import TypeRegistry, edit_video, video_from_source

COMFY_ROOT = os.environ.get("DINKSTER_COMFYUI_ROOT", "")
VHS_ROOT = os.environ.get("DINKSTER_VHS_ROOT", "")
PINNED_VHS_COMMIT = "4d907bee61e92c2e65af3bd6383a4e4d356126d1"
CASES = ("rgb", "ten_bit", "alpha", "vfr_mp4", "vfr_mkv")
pytestmark = pytest.mark.skipif(
    not COMFY_ROOT or not VHS_ROOT,
    reason="DINKSTER_COMFYUI_ROOT and DINKSTER_VHS_ROOT not set (live VHS comparison)",
)


def _pixels(index: int, *, alpha: bool = False, ten_bit: bool = False) -> np.ndarray:
    pixels = np.zeros((32, 64, 4 if alpha else 3), np.uint16 if ten_bit else np.uint8)
    pixels[..., 0] = index * (4000 if ten_bit else 20)
    pixels[..., 1] = np.arange(64) * (997 if ten_bit else 3)
    pixels[..., 2] = np.arange(32)[:, None] * (1973 if ten_bit else 7)
    if alpha:
        pixels[..., 3] = np.arange(64) * 4
    return pixels


def _write_source(path: Path, case: str) -> None:
    alpha, ten_bit = case == "alpha", case == "ten_bit"
    timestamps = [0, 1, 4, 5, 9, 10] if case.startswith("vfr") else list(range(10))
    with av.open(path, "w", options={"fflags": "+bitexact"}) as opened:
        output = cast(Any, opened)
        stream = output.add_stream("ffv1" if alpha else "libx264", rate=10)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "bgra" if alpha else "yuv420p10le" if ten_bit else "yuv420p"
        stream.codec_context.thread_count = 1
        stream.codec_context.max_b_frames = 0
        stream.time_base = Fraction(1, 10)
        if not alpha:
            stream.options = {"crf": "0", "x264-params": "scenecut=0"}
        for index, pts in enumerate(timestamps):
            frame = av.VideoFrame.from_ndarray(
                _pixels(index, alpha=alpha, ten_bit=ten_bit),
                format="rgba" if alpha else "rgb48le" if ten_bit else "rgb24",
            )
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _getters(video: Any) -> dict[str, Any]:
    return {
        "width": video.get_dimensions()[0],
        "height": video.get_dimensions()[1],
        "fps": float(video.get_frame_rate()),
        "frame_count": video.get_frame_count(),
        "duration": video.get_duration(),
        "bit_depth": video.get_bit_depth(),
        "trim": list(video.get_active_trim_window()),
    }


def _live_receipt(paths: dict[str, str], destination: str) -> None:
    """Run in a separate interpreter so real ComfyUI globals cannot leak into tests."""
    sys.path[:0] = [COMFY_ROOT, VHS_ROOT]
    sys.argv = [sys.argv[0], "--cpu"]
    importlib.import_module("comfy.options").enable_args_parsing()
    loop = asyncio.new_event_loop()
    # VHS imports the real prompt queue even for standalone file loading.
    importlib.import_module("server").PromptServer(loop)
    loaders = importlib.import_module("videohelpersuite.load_video_nodes")
    info_nodes = importlib.import_module("videohelpersuite.nodes")
    info_node = info_nodes.VideoInfoSource()
    combined_info_node = info_nodes.VideoInfo()
    impl = importlib.import_module("comfy_api.input_impl")
    compat = importlib.import_module("dinkster_compat_comfy.video")
    torch = importlib.import_module("torch")
    assert not torch.cuda.is_initialized()
    registry = TypeRegistry()
    compat.register_video_type(registry, "comfy.VIDEO")
    spec = registry.spec("comfy.VIDEO")
    assert spec.coerce is not None
    result: dict[str, Any] = {}
    for case, path in paths.items():
        common = {
            "video": path,
            "force_rate": 0,
            "frame_load_cap": 0,
            "custom_width": 0,
            "custom_height": 0,
        }
        cv = loaders.load_video(**common, skip_first_frames=0, select_every_nth=1)
        ffmpeg = loaders.load_video(
            **common, start_time=0, generator=loaders.ffmpeg_frame_generator
        )
        upstream = impl.VideoFromFile(path)
        wrapper = cast(Any, spec.coerce(upstream))
        cropped = wrapper.as_trimmed(0.1, 0.4, True).as_cropped(4, 2, 32, 16)
        converted_crop = cast(
            Any, spec.coerce(upstream.as_trimmed(0.1, 0.4, True).as_cropped(4, 2, 32, 16))
        )
        parts = wrapper.get_components()
        crop_parts = cropped.get_components()
        converted_parts = converted_crop.get_components()
        np.testing.assert_array_equal(crop_parts.images.numpy(), converted_parts.images.numpy())
        # Coercion and wire decoding must retain the real VideoInput interface.
        decoded = cast(Any, spec.decode(spec.encode(cropped)))
        assert isinstance(decoded, impl.VideoFromFile)
        assert _getters(decoded) == _getters(cropped) == _getters(converted_crop)
        result[case] = {
            "cv": cv[3],
            "ffmpeg": ffmpeg[3],
            "cv_info_node": info_node.get_video_info(cv[3]),
            "ffmpeg_info_node": info_node.get_video_info(ffmpeg[3]),
            "cv_combined_info_node": combined_info_node.get_video_info(cv[3]),
            "ffmpeg_combined_info_node": combined_info_node.get_video_info(ffmpeg[3]),
            "upstream": _getters(upstream),
            "wrapper": _getters(wrapper),
            "crop": _getters(cropped),
            "shape": list(parts.images.shape),
            "crop_shape": list(crop_parts.images.shape),
            "alpha": None if parts.alpha is None else parts.alpha[0, 0].tolist(),
            "crop_alpha": None if crop_parts.alpha is None else crop_parts.alpha[0, 0].tolist(),
        }
    ffmpeg_path = importlib.import_module("videohelpersuite.utils").ffmpeg_path
    result["environment"] = {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "torchaudio", "av", "numpy", "opencv-python")
        },
        "ffmpeg": subprocess.check_output([ffmpeg_path, "-version"], text=True).splitlines()[0],
        "ffmpeg_path": ffmpeg_path,
    }
    assert not torch.cuda.is_initialized()
    Path(destination).write_text(json.dumps(result, indent=2), "utf-8")
    loop.close()


@pytest.fixture(scope="module")
def live_corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Path], dict[str, Any]]:
    from tests.test_compat_video_live import (
        PINNED_COMFY_COMMIT,
        REPO_ROOT,
        _comfy_python,
        _dinkster_pythonpath,
    )

    for root, pin in ((COMFY_ROOT, PINNED_COMFY_COMMIT), (VHS_ROOT, PINNED_VHS_COMMIT)):
        assert (
            subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
            == pin
        )
        subprocess.run(["git", "-C", root, "diff", "--quiet", "HEAD"], check=True)
    directory = tmp_path_factory.mktemp("vhs-corpus")
    paths = {
        case: directory / f"{case}.{'mkv' if case in ('alpha', 'vfr_mkv') else 'mp4'}"
        for case in CASES
    }
    for case, path in paths.items():
        _write_source(path, case)
    receipt = directory / "receipt.json"
    completed = subprocess.run(
        [
            _comfy_python(),
            "-c",
            "import json, sys; from tests.test_video_vhs_live import _live_receipt; "
            "_live_receipt(json.loads(sys.argv[1]), sys.argv[2])",
            json.dumps({case: str(path) for case, path in paths.items()}),
            str(receipt),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(REPO_ROOT), str(REPO_ROOT / "src"), _dinkster_pythonpath())
            ),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return paths, json.loads(receipt.read_text("utf-8"))


@pytest.mark.parametrize("case", CASES[:3])
def test_video_info_outputs_match_actual_vhs_video_info(
    live_corpus: tuple[dict[str, Path], dict[str, Any]],
    case: str,
) -> None:
    paths, receipt = live_corpus
    value = video_from_source(paths[case].read_bytes())
    info = VideoInfo.execute(video=value)
    metadata = json.loads(cast(str, info["info"]))
    expected = {"width": 64, "height": 32, "fps": 10, "frame_count": 10, "duration": 1}
    for key, expected_value in expected.items():
        assert info[key] == expected_value
        for loader in ("cv", "ffmpeg"):
            assert receipt[case][loader][f"source_{key}"] == expected_value
        assert receipt[case]["wrapper"][key] == expected_value
        if case != "alpha":
            assert receipt[case]["upstream"][key] == expected_value
    assert metadata["probe"]["frame_count_kind"] == ("estimated" if case == "alpha" else "header")
    assert metadata["probe"]["duration_kind"] == ("container" if case == "alpha" else "stream")
    assert metadata["probe"]["bit_depth"] == (10 if case == "ten_bit" else 8)
    assert metadata["probe"]["alpha"] == (case == "alpha")
    actual = tuple(info[key] for key in ("fps", "frame_count", "duration", "width", "height"))
    for loader in ("cv", "ffmpeg"):
        assert receipt[case][f"{loader}_info_node"] == [10, 10, 1, 64, 32]
        source_and_loaded = tuple(receipt[case][f"{loader}_combined_info_node"])
        assert source_and_loaded == actual + actual


@pytest.mark.parametrize("case", CASES[:3])
def test_disassemble_assemble_and_real_wrapper_trim_crop(
    live_corpus: tuple[dict[str, Path], dict[str, Any]],
    case: str,
) -> None:
    paths, receipt = live_corpus
    value = video_from_source(paths[case].read_bytes())
    parts = DisassembleVideo.execute(video=value)
    images = cast(np.ndarray, parts["images"])
    assert images.shape == (10, 32, 64, 4 if case == "alpha" else 3)
    assert parts["bit_depth"] == (10 if case == "ten_bit" else 8)
    rebuilt = AssembleVideo.execute(images=images, fps=10, bit_depth=str(parts["bit_depth"]))[
        "video"
    ]
    np.testing.assert_array_equal(DisassembleVideo.execute(video=rebuilt)["images"], images)
    cropped = edit_video(value, {"trim": {"start_time": 0.1, "duration": 0.4}})
    cropped = edit_video(cropped, {"crop": {"x": 4, "y": 2, "width": 32, "height": 16}})
    cropped_metadata = json.loads(cast(str, VideoInfo.execute(video=cropped)["info"]))
    assert (
        cropped_metadata["probe"]
        == json.loads(cast(str, VideoInfo.execute(video=value)["info"]))["probe"]
    )
    assert cropped_metadata["effective"]["frame_count_kind"] == "estimated"
    np.testing.assert_array_equal(
        DisassembleVideo.execute(video=cropped)["images"],
        images[1:5, 2:18, 4:36],
    )
    assert receipt[case]["crop"] == {
        "width": 32,
        "height": 16,
        "fps": 10,
        "frame_count": 4,
        "duration": 0.4,
        "bit_depth": 10 if case == "ten_bit" else 8,
        "trim": [0.1, 0.4],
    }
    assert receipt[case]["shape"] == [10, 32, 64, 3]
    assert receipt[case]["crop_shape"] == [4, 16, 32, 3]
    if case == "alpha":
        # The golden is the generated RGBA array, never ComfyUI's alpha decoder.
        expected = np.stack([_pixels(i, alpha=True) for i in range(10)])
        np.testing.assert_array_equal(np.rint(images * 255).astype(np.uint8), expected)
        np.testing.assert_array_equal(
            np.rint(np.array(receipt[case]["alpha"]) * 255), expected[0, 0, :, 3]
        )
        np.testing.assert_array_equal(
            np.rint(np.array(receipt[case]["crop_alpha"]) * 255), expected[1, 2, 4:36, 3]
        )


def test_vfr_source_counts_are_estimates_not_decoded_counts(
    live_corpus: tuple[dict[str, Path], dict[str, Any]],
) -> None:
    paths, receipt = live_corpus
    for case in ("vfr_mp4", "vfr_mkv"):
        value = video_from_source(paths[case].read_bytes())
        info = VideoInfo.execute(video=value)
        probe = json.loads(cast(str, info["info"]))["probe"]
        timestamps: list[Fraction] = []
        with av.open(paths[case]) as container:
            for frame in container.decode(video=0):
                assert frame.pts is not None and frame.time_base is not None
                timestamps.append(frame.pts * frame.time_base)
        assert timestamps == [Fraction(t, 10) for t in (0, 1, 4, 5, 9, 10)]
        assert len(timestamps) == 6
        assert info["duration"] == 1.1
        assert receipt[case]["ffmpeg"]["source_duration"] == 1.1
        assert DisassembleVideo.execute(video=value)["frame_count"] == 6
        assert receipt[case]["cv"]["loaded_frame_count"] == 6
        # FFmpeg's default output synchronization duplicates frames to CFR.
        assert receipt[case]["ffmpeg"]["loaded_frame_count"] == 11
        if case == "vfr_mp4":
            assert probe["frame_count_kind"] == "header"
            assert probe["duration_kind"] == "stream"
            assert info["frame_count"] == 6
            assert info["fps"] == 60 / 11
            assert receipt[case]["cv"]["source_frame_count"] == 6
            assert receipt[case]["cv"]["source_fps"] == 60 / 11
            # VHS parses FFmpeg's two-decimal FPS, then multiplies by duration.
            assert receipt[case]["ffmpeg"]["source_fps"] == 5.45
            assert receipt[case]["ffmpeg"]["source_frame_count"] == 5.45 * 1.1
            assert receipt[case]["ffmpeg"]["source_frame_count"] != 6
        else:
            assert probe["frame_count_kind"] == "estimated"
            assert probe["duration_kind"] == "stream"
            assert info["fps"] == 10
            assert info["frame_count"] == 11
            for loader in ("cv", "ffmpeg"):
                assert receipt[case][loader]["source_fps"] == 10
                assert receipt[case][loader]["source_frame_count"] == 11
        for key in ("width", "height", "fps", "duration", "frame_count"):
            assert receipt[case]["wrapper"][key] == info[key]
