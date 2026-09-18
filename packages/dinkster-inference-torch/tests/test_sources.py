"""Payload reads: safetensors bytes into torch tensors.

The header contract (parsing, validation, byte ranges) is proven
torch-free in tests/test_inference_inspection.py; these tests cover
the executing half only - that load_tensors turns the promised ranges
into lifetime-owned copy-on-write views and refuses what torch cannot
represent. No numpy and no safetensors package anywhere.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import gc
import json
import os
import struct
from pathlib import Path

import pytest
import torch
from dinkster_inference import (
    Q4_K,
    Q8_0,
    admit_gguf_encoded_storage,
    decode_gguf_encoded_storage,
)
from dinkster_inference.sources import MalformedSafetensors
from dinkster_inference_torch import SourceReadError, load_tensors
from dinkster_inference_torch.gguf_linear import GGUF_BLOCK_DECODERS, decode_q8_0_blocks
from dinkster_inference_torch.sources import (
    load_gguf_encoded_blocks,
    load_gguf_tensors,
    tensor_file_slice,
)

from tests.test_inference_gguf import (  # pyright: ignore[reportMissingImports]
    _decode_vectors,
    _encoded_storage,
)

BITS = {"F32": 32, "F16": 16, "BF16": 16, "I32": 32, "U8": 8, "F4": 4, "F6_E2M3": 6}


def write(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    *,
    drop_payload_bytes: int = 0,
) -> Path:
    header: dict[str, object] = {}
    payload = bytearray()
    for key, (dtype, shape, data) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        assert len(data) == numel * BITS[dtype] // 8
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    raw = json.dumps(header).encode()
    body = struct.pack("<Q", len(raw)) + raw + bytes(payload)
    if drop_payload_bytes:
        body = body[:-drop_payload_bytes]
    path.write_bytes(body)
    return path


def f32(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def mapping_state(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return Linux mapping and descriptor entries naming ``path``."""
    maps_path = Path("/proc/self/maps")
    fd_path = Path("/proc/self/fd")
    if not maps_path.is_file() or not fd_path.is_dir():
        pytest.skip("mapping lifetime proof requires Linux /proc")
    target = str(path.resolve())
    mappings = tuple(line for line in maps_path.read_text().splitlines() if line.endswith(target))
    descriptors = []
    for descriptor in fd_path.iterdir():
        try:
            if os.readlink(descriptor) == target:
                descriptors.append(descriptor.name)
        except OSError:
            continue
    return mappings, tuple(sorted(descriptors))


def test_reads_values_shapes_and_dtypes(tmp_path: Path) -> None:
    path = write(
        tmp_path / "ok.safetensors",
        {
            "a": ("F32", (2, 2), f32(1.0, -2.0, 3.5, 0.25)),
            "b": ("I32", (3,), struct.pack("<3i", -1, 0, 7)),
            "c": ("U8", (2,), bytes([0, 255])),
        },
    )
    out = load_tensors(path)
    assert set(out) == {"a", "b", "c"}
    torch.testing.assert_close(out["a"], torch.tensor([[1.0, -2.0], [3.5, 0.25]]))
    assert out["b"].dtype == torch.int32
    assert out["b"].tolist() == [-1, 0, 7]
    assert out["c"].dtype == torch.uint8
    assert out["c"].tolist() == [0, 255]


def test_reads_half_precision_payloads(tmp_path: Path) -> None:
    half = torch.tensor([1.5, -0.25], dtype=torch.float16)
    bf16 = torch.tensor([2.0, -8.0], dtype=torch.bfloat16)

    def raw(t: torch.Tensor) -> bytes:
        return bytes(t.contiguous().untyped_storage())[: t.numel() * t.element_size()]

    path = write(
        tmp_path / "half.safetensors",
        {"h": ("F16", (2,), raw(half)), "b": ("BF16", (2,), raw(bf16))},
    )
    out = load_tensors(path)
    assert out["h"].dtype == torch.float16
    torch.testing.assert_close(out["h"], half)
    assert out["b"].dtype == torch.bfloat16
    torch.testing.assert_close(out["b"], bf16)


def test_gguf_payload_loader_admits_and_decodes_reference_values(tmp_path: Path) -> None:
    authority, model_key, _payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    storage = admit_gguf_encoded_storage(authority, model_key)
    expected = torch.tensor(decode_gguf_encoded_storage(storage)).reshape(storage.logical_shape)

    loaded = load_gguf_tensors(authority, (model_key,))

    assert tuple(loaded) == (model_key,)
    assert loaded[model_key].dtype == torch.float32
    assert torch.equal(loaded[model_key], expected)


def test_gguf_payload_loader_fails_closed_after_file_mutation(tmp_path: Path) -> None:
    authority, model_key, payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    storage = admit_gguf_encoded_storage(authority, model_key)
    expected = torch.tensor(decode_gguf_encoded_storage(storage)).reshape(storage.logical_shape)
    raw = bytearray(authority.path.read_bytes())
    raw[authority.source.tensor(model_key).offset + 2] ^= 1
    try:
        authority.path.write_bytes(raw)
    except PermissionError:
        if os.name != "nt":
            raise
        loaded = load_gguf_tensors(authority, (model_key,))[model_key]
        assert torch.equal(loaded, expected)
        assert (
            payload
            == authority.file_slice(authority.source.tensor(model_key).offset, len(payload)).read()
        )
        return

    with pytest.raises(SourceReadError, match="changed after identity verification"):
        load_gguf_tensors(
            authority,
            (model_key,),
            expected_runtime_facts=authority.runtime_facts,
        )


def test_gguf_encoded_block_reader_returns_raw_q8_0_payload(tmp_path: Path) -> None:
    authority, model_key, payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])

    blocks = load_gguf_encoded_blocks(
        authority, model_key, expected_runtime_facts=authority.runtime_facts
    )

    assert blocks.dtype == torch.uint8
    assert blocks.shape == (len(payload) // Q8_0.block_bytes, Q8_0.block_bytes)
    assert bytes(blocks.reshape(-1).tolist()) == payload
    storage = admit_gguf_encoded_storage(authority, model_key)
    expected = torch.tensor(decode_gguf_encoded_storage(storage)).reshape(storage.logical_shape)
    assert torch.equal(decode_q8_0_blocks(blocks, storage.logical_shape), expected)


def test_gguf_encoded_block_reader_returns_raw_q4_k_payload(tmp_path: Path) -> None:
    authority, model_key, payload = _encoded_storage(tmp_path, Q4_K, _decode_vectors()[2][1])

    blocks = load_gguf_encoded_blocks(
        authority, model_key, expected_runtime_facts=authority.runtime_facts
    )

    assert blocks.dtype == torch.uint8
    assert blocks.shape == (len(payload) // Q4_K.block_bytes, Q4_K.block_bytes)
    assert bytes(blocks.reshape(-1).tolist()) == payload
    storage = admit_gguf_encoded_storage(authority, model_key)
    expected = torch.tensor(decode_gguf_encoded_storage(storage)).reshape(storage.logical_shape)
    decoded = GGUF_BLOCK_DECODERS["Q4_K"](blocks, storage.logical_shape)
    assert torch.equal(decoded, expected)


def test_gguf_encoded_block_reader_names_refusals(tmp_path: Path) -> None:
    authority, model_key, _payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    with pytest.raises(SourceReadError, match="identity changed after planning"):
        load_gguf_encoded_blocks(authority, model_key, expected_runtime_facts=("forged.fact=1",))
    with pytest.raises(SourceReadError, match="no mapped GGUF tensor named 'nope'"):
        load_gguf_encoded_blocks(authority, "nope")


def test_subset_selection_reads_only_named_keys(tmp_path: Path) -> None:
    path = write(
        tmp_path / "subset.safetensors",
        {
            "keep": ("F32", (1,), f32(4.0)),
            "skip": ("F32", (1,), f32(5.0)),
        },
    )
    out = load_tensors(path, keys=["keep"])
    assert set(out) == {"keep"}


def test_empty_selection_returns_no_tensors(tmp_path: Path) -> None:
    path = write(
        tmp_path / "empty-selection.safetensors",
        {"skip": ("F32", (1,), f32(5.0))},
    )
    assert load_tensors(path, keys=[]) == {}


def test_unknown_key_refuses(tmp_path: Path) -> None:
    path = write(tmp_path / "missing.safetensors", {"a": ("F32", (1,), f32(1.0))})
    with pytest.raises(SourceReadError, match="no tensor named 'nope'"):
        load_tensors(path, keys=["nope"])


def test_unmappable_dtype_refuses(tmp_path: Path) -> None:
    path = write(
        tmp_path / "quant.safetensors", {"q": ("F6_E2M3", (4,), bytes([0x12, 0x34, 0x56]))}
    )
    with pytest.raises(SourceReadError, match="no packed torch"):
        load_tensors(path)


@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
def test_packed_f4_requires_even_final_axis(tmp_path: Path, shape: tuple[int, ...]) -> None:
    path = write(tmp_path / "odd.safetensors", {"q": ("F4", shape, bytes(3 if shape[0] else 0))})
    with pytest.raises(SourceReadError, match="even final F4 dimension"):
        load_tensors(path)


@pytest.mark.parametrize("mixed", [False, True])
def test_empty_packed_f4_keeps_physical_shape_and_cpu_placement(
    tmp_path: Path, mixed: bool
) -> None:
    tensors = {"empty": ("F4", (0, 6), b"")}
    if mixed:
        tensors["full"] = ("F4", (2, 6), bytes(range(6)))
    path = write(tmp_path / "packed.safetensors", tensors)
    with torch.device("meta"):
        actual = load_tensors(path)
    assert actual["empty"].shape == (0, 3)
    assert actual["empty"].dtype == torch.float4_e2m1fn_x2
    assert actual["empty"].device.type == "cpu"
    if mixed:
        assert actual["full"].shape == (2, 3)
        assert actual["full"].device.type == "cpu"
        assert actual["full"].view(torch.uint8).flatten().tolist() == list(range(6))


def test_truncated_file_refuses_at_header_parse(tmp_path: Path) -> None:
    """A truncated file never reaches the payload reader: the header
    parser validates every data_offsets range against the actual
    payload size first."""
    path = write(
        tmp_path / "short.safetensors",
        {"a": ("F32", (2,), f32(1.0, 2.0))},
        drop_payload_bytes=3,
    )
    with pytest.raises(MalformedSafetensors, match="fall outside"):
        load_tensors(path)


def test_file_shrunk_between_opens_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader validates the mapped file's actual extent before
    creating any tensor views."""
    path = write(tmp_path / "race.safetensors", {"a": ("F32", (2,), f32(1.0, 2.0))})
    from dinkster_inference.sources import load_safetensors_header

    source = load_safetensors_header(path)
    path.write_bytes(path.read_bytes()[:-3])

    def stale_header(_path: Path) -> object:
        return source

    monkeypatch.setattr(
        "dinkster_inference_torch.sources.load_safetensors_header",
        stale_header,
    )
    with pytest.raises(SourceReadError, match="payload truncated"):
        load_tensors(path)


def test_zero_element_tensor_needs_no_payload(tmp_path: Path) -> None:
    path = write(tmp_path / "empty.safetensors", {"z": ("F32", (0, 4), b"")})
    out = load_tensors(path)
    assert out["z"].shape == (0, 4)
    assert out["z"].dtype == torch.float32


def test_returned_tensor_keeps_mapping_alive_after_gc(tmp_path: Path) -> None:
    path = write(
        tmp_path / "lifetime.safetensors",
        {"a": ("F32", (2,), f32(1.0, 2.0))},
    )
    tensor = load_tensors(path)["a"]
    gc.collect()
    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0]))


def test_copy_on_write_mutation_leaves_checkpoint_unchanged(tmp_path: Path) -> None:
    path = write(tmp_path / "own.safetensors", {"a": ("F32", (2,), f32(1.0, 2.0))})
    checkpoint_bytes = path.read_bytes()
    first = load_tensors(path)["a"]
    first.mul_(100.0)
    assert path.read_bytes() == checkpoint_bytes
    second = load_tensors(path)["a"]
    torch.testing.assert_close(second, torch.tensor([1.0, 2.0]))


def test_mapping_and_file_descriptor_release_after_last_tensor(
    tmp_path: Path,
) -> None:
    path = write(
        tmp_path / "release.safetensors",
        {
            "a": ("F32", (1,), f32(1.0)),
            "b": ("F32", (1,), f32(2.0)),
        },
    )
    assert mapping_state(path) == ((), ())

    loaded = load_tensors(path)
    first = loaded.pop("a")
    last = loaded.pop("b")
    del loaded
    mappings, descriptors = mapping_state(path)
    assert len(mappings) == 1
    assert len(descriptors) == 2
    first_slice = tensor_file_slice(first)
    last_slice = tensor_file_slice(last)
    assert first_slice is not None
    assert last_slice is not None
    assert first_slice.file is last_slice.file
    assert first_slice.lock is last_slice.lock
    del first_slice, last_slice

    del first
    gc.collect()
    assert mapping_state(path) == (mappings, descriptors)

    del last
    gc.collect()
    assert mapping_state(path) == ((), ())


def test_malformed_header_stays_the_parsers_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 4) + b"nope")
    with pytest.raises(MalformedSafetensors):
        load_tensors(path)
