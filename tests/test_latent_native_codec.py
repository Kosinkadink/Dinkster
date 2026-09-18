from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import types
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_assets import AssetError, MountDef, MountTable, parse_latent_asset
from dinkster_inference import MultiStreamLatent
from dinkster_protocol import ExportSnapshot


def _codec_module() -> types.ModuleType:
    # Exercise the lazy codec without the torch package's eager public exports.
    path = (
        Path(__file__).resolve().parents[1]
        / "packages/dinkster-inference-torch/src/dinkster_inference_torch/latent_assets.py"
    )
    spec = importlib.util.spec_from_file_location("native_latent_codec", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CODEC = _codec_module()
load_latent = _CODEC.load_latent
serialize_native_latent = _CODEC.serialize_native_latent


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"torch.{self.name}"


class _ByteView:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def numpy(self) -> np.ndarray:
        return self.value.view(np.uint8)


class _Tensor:
    def __init__(self, value: np.ndarray, dtype_name: str = "float32") -> None:
        self.value = value
        self.dtype = _DType(dtype_name)
        self.shape = value.shape

    def detach(self) -> _Tensor:
        return self

    def to(self, *, device: str) -> _Tensor:
        assert device == "cpu"
        return self

    def contiguous(self) -> _Tensor:
        return self

    def view(self, dtype: object) -> _ByteView:
        assert dtype is _FAKE_TORCH.uint8
        return _ByteView(self.value)


class _Loaded:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def reshape(self, shape: tuple[int, ...]) -> _Loaded:
        return _Loaded(self.value.reshape(shape))

    def clone(self) -> _Loaded:
        return _Loaded(self.value.copy())

    def float(self) -> _Loaded:
        return _Loaded(self.value.astype(np.float32))

    def __mul__(self, scale: float) -> _Loaded:
        return _Loaded(self.value * scale)


def _frombuffer(data: bytearray, *, dtype: object) -> _Loaded:
    return _Loaded(np.frombuffer(data, dtype=cast("Any", dtype)))


_FAKE_TORCH = types.ModuleType("torch")
_FAKE_TORCH.uint8 = object()  # type: ignore[attr-defined]
_FAKE_TORCH.float16 = np.float16  # type: ignore[attr-defined]
_FAKE_TORCH.bfloat16 = np.uint16  # type: ignore[attr-defined]
_FAKE_TORCH.float32 = np.float32  # type: ignore[attr-defined]
_FAKE_TORCH.float64 = np.float64  # type: ignore[attr-defined]
_FAKE_TORCH.frombuffer = _frombuffer  # type: ignore[attr-defined]


def _serialize(samples: object, **kwargs: object) -> bytes:
    previous = sys.modules.get("torch")
    sys.modules["torch"] = _FAKE_TORCH
    try:
        source = serialize_native_latent(samples, **kwargs)  # type: ignore[arg-type]
        try:
            return source.read()
        finally:
            source.close()
    finally:
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous


def test_native_single_golden_round_trip_and_metadata() -> None:
    tensor = _Tensor(np.array([[1.25, -2.5]], dtype=np.float32))
    snapshot = ExportSnapshot(
        prompt={"1": {"class_type": "dinkster.save_latent"}},
        extra_pnginfo={"workflow": {"nodes": []}},
    )
    hint = json.dumps(
        {"sourceDigest": "blake3:" + "a" * 64, "version": 1},
        sort_keys=True,
        separators=(",", ":"),
    )
    first = _serialize(tensor, snapshot=snapshot, vae_hint=hint)
    assert first == _serialize(tensor, snapshot=snapshot, vae_hint=hint)
    assert hashlib.sha256(first).hexdigest() == (
        "7fae1abeac322432974d660d57f15f54c6ac69b1328ccee848700f33dbaa1772"
    )
    descriptor = parse_latent_asset(BytesIO(first))
    assert descriptor.tensors[0].name == "dinkster_samples"
    assert descriptor.metadata["dinkster_vae_hint"] == hint
    loaded, loaded_hint = load_latent(BytesIO(first), _FAKE_TORCH)
    assert isinstance(loaded, _Loaded)
    np.testing.assert_array_equal(loaded.value, tensor.value)
    assert loaded_hint == hint


def test_native_multi_round_trip_preserves_public_roles_and_order() -> None:
    value = MultiStreamLatent.from_pairs(
        (
            ("video", _Tensor(np.array([1.0, 2.0], dtype=np.float32))),
            ("audio", _Tensor(np.array([3.0], dtype=np.float32))),
        )
    )
    encoded = _serialize(value)
    descriptor = parse_latent_asset(BytesIO(encoded))
    assert tuple(item.name for item in descriptor.tensors) == (
        "dinkster_stream_0000",
        "dinkster_stream_0001",
    )
    loaded, _hint = load_latent(BytesIO(encoded), _FAKE_TORCH)
    assert type(loaded) is MultiStreamLatent
    assert loaded.roles == ("video", "audio")
    np.testing.assert_array_equal(loaded.by_role("video").value, [1.0, 2.0])
    np.testing.assert_array_equal(loaded.by_role("audio").value, [3.0])


def test_comfyui_marker_and_legacy_multiplier_are_exactly_distinct() -> None:
    body = struct.pack("<f", 0.18215)

    def fixture(marker: bool) -> bytes:
        header: dict[str, object] = {
            "latent_tensor": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        }
        if marker:
            header["latent_format_version_0"] = {
                "dtype": "F32",
                "shape": [0],
                "data_offsets": [4, 4],
            }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        return struct.pack("<Q", len(encoded)) + encoded + body

    current, _ = load_latent(BytesIO(fixture(True)), _FAKE_TORCH)
    legacy, _ = load_latent(BytesIO(fixture(False)), _FAKE_TORCH)
    assert isinstance(current, _Loaded)
    assert isinstance(legacy, _Loaded)
    np.testing.assert_array_equal(current.value, np.array([0.18215], dtype=np.float32))
    np.testing.assert_allclose(legacy.value, np.array([1.0], dtype=np.float32), rtol=1e-6)


def test_comfyui_codec_ignores_metadata_format_but_rejects_extra_tensors() -> None:
    body = struct.pack("<f", 1.0)
    header: dict[str, object] = {
        "__metadata__": {"format": "anything", "prompt": "{}"},
        "latent_tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "latent_format_version_0": {
            "dtype": "F32",
            "shape": [0],
            "data_offsets": [4, 4],
        },
    }

    def encoded(table: dict[str, object], tensor_body: bytes) -> bytes:
        raw = json.dumps(table, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        return struct.pack("<Q", len(raw)) + raw + tensor_body

    loaded, _ = load_latent(BytesIO(encoded(header, body)), _FAKE_TORCH)
    assert isinstance(loaded, _Loaded)
    np.testing.assert_array_equal(loaded.value, np.array([1.0], dtype=np.float32))

    header["model.weight"] = {
        "dtype": "F32",
        "shape": [1],
        "data_offsets": [4, 8],
    }
    header["latent_format_version_0"] = {
        "dtype": "F32",
        "shape": [0],
        "data_offsets": [8, 8],
    }
    with pytest.raises(AssetError, match="schema-less"):
        load_latent(BytesIO(encoded(header, body + body)), _FAKE_TORCH)


def test_malformed_vae_hint_is_informational_and_metadata_bounds_refuse() -> None:
    tensor = _Tensor(np.array([2.0], dtype=np.float32))
    encoded = _serialize(tensor, vae_hint='{"version":2}')
    loaded, hint = load_latent(BytesIO(encoded), _FAKE_TORCH)
    assert isinstance(loaded, _Loaded)
    np.testing.assert_array_equal(loaded.value, tensor.value)
    assert hint == ""

    huge_version = '{"version":' + "9" * 5000 + "}"
    loaded, hint = load_latent(BytesIO(_serialize(tensor, vae_hint=huge_version)), _FAKE_TORCH)
    assert isinstance(loaded, _Loaded)
    np.testing.assert_array_equal(loaded.value, tensor.value)
    assert hint == ""

    nested_version = '{"version":' + "[" * 3900 + "1" + "]" * 3900 + "}"
    loaded, hint = load_latent(BytesIO(_serialize(tensor, vae_hint=nested_version)), _FAKE_TORCH)
    assert isinstance(loaded, _Loaded)
    np.testing.assert_array_equal(loaded.value, tensor.value)
    assert hint == ""

    oversized = ExportSnapshot(prompt={"value": "x" * (1024 * 1024)})
    with pytest.raises(AssetError, match="prompt metadata exceeds"):
        _serialize(tensor, snapshot=oversized)


def test_native_save_node_returns_the_exact_latent_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_compat_comfy.native as native

    latent = {"samples": object(), "noise_mask": object()}
    asset = object()
    seen: list[object] = []

    def serialize(samples: object, **_kwargs: object) -> BytesIO:
        seen.append(samples)
        return BytesIO(b"native")

    class Writer:
        def save_stream(self, *_args: object, **_kwargs: object) -> object:
            return asset

    monkeypatch.setitem(sys.modules, "dinkster_inference_torch.latent_assets", _CODEC)
    monkeypatch.setattr(_CODEC, "serialize_native_latent", serialize)
    monkeypatch.setattr(native, "mount_writer", Writer)
    result = native.SaveLatent.execute(samples=latent, target={"mount": "out", "prefix": "x"})
    assert result["samples"] is latent
    assert result["asset"] is asset
    assert seen == [latent["samples"]]


def test_native_load_node_materializes_asset_and_returns_both_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_compat_comfy.native import LoadLatent

    tensor = _Tensor(np.array([[1.25, -2.5]], dtype=np.float32))
    (tmp_path / "sample.latent").write_bytes(_serialize(tensor))
    (tmp_path / "malformed.latent").write_bytes(b"invalid")
    table = MountTable(tmp_path / "mounts.json")
    table.add(MountDef(id="out", path=tmp_path))
    table.scan("out")
    monkeypatch.setitem(sys.modules, "dinkster_inference_torch.latent_assets", _CODEC)
    monkeypatch.setitem(sys.modules, "torch", _FAKE_TORCH)
    result = LoadLatent.execute(asset=table.ref("mounts/out/sample.latent"))
    assert set(result) == {"samples", "vae_hint"}
    loaded = cast(dict[str, _Loaded], result["samples"])["samples"]
    np.testing.assert_array_equal(loaded.value, tensor.value)
    assert result["vae_hint"] == ""
    with pytest.raises(AssetError, match="truncated safetensors framing"):
        LoadLatent.execute(asset=table.ref("mounts/out/malformed.latent"))
