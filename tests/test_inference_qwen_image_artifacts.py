from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference.qwen_image_layout import qwen_image_dit_layout
from dinkster_inference.qwen_image_text import qwen_image_text_layout
from tools.inference_parity import qwen_image_receipts  # pyright: ignore[reportMissingImports]
from tools.inference_parity.qwen_image_receipts import (  # pyright: ignore[reportMissingImports]
    ArtifactReceiptError,
    _open_artifact,
    canonical_header_digest,
    load_manifest,
    read_safetensors_header,
    validate_dit_header,
    validate_manifest,
    validate_text_header,
    validate_vae_header,
    verify_receipts,
)

from tools.evidence_paths import EVIDENCE_ROOT

MANIFEST_PATH = EVIDENCE_ROOT / "tools/inference_parity/qwen_image_artifacts.json"
pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptor APIs")
_pread = cast("Callable[[int, int, int], bytes]", getattr(os, "pread", None))


def _entry(dtype: str, shape: tuple[int, ...]) -> dict[str, object]:
    return {"data_offsets": [0, 0], "dtype": dtype, "shape": list(shape)}


def _write_safetensors(path: Path, header: Mapping[str, object], payload: bytes = b"") -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("ascii")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_qwen_image_artifact_manifest_pins_official_immutable_graph() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    validate_manifest(manifest)
    assert manifest["schema"] == 1
    assert manifest["artifact_root"] == "models"
    assert manifest["provider"] == {
        "api_url": (
            "https://huggingface.co/api/models/Comfy-Org/Qwen-Image_ComfyUI/"
            "revision/46839d338df81ce625d5fae27d7e370314c0fbc9"
        ),
        "gated": False,
        "license": "Apache-2.0",
        "model_card_url": (
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/raw/"
            "46839d338df81ce625d5fae27d7e370314c0fbc9/README.md"
        ),
        "repository": "Comfy-Org/Qwen-Image_ComfyUI",
        "revision": "46839d338df81ce625d5fae27d7e370314c0fbc9",
    }
    assert manifest["license_receipt"] == {
        "bytes": 11343,
        "name": "Apache License 2.0",
        "repository": "Qwen/Qwen-Image",
        "revision": "75e0b4be04f60ec59a75f475837eced720f823b6",
        "sha256": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
        "source_url": (
            "https://huggingface.co/Qwen/Qwen-Image/raw/"
            "75e0b4be04f60ec59a75f475837eced720f823b6/LICENSE"
        ),
    }
    artifacts = {item["role"]: item for item in manifest["artifacts"]}
    assert set(artifacts) == {"qwen-image-dit", "qwen2.5-vl-7b-text", "wan21-vae"}
    assert artifacts["qwen-image-dit"] == {
        **artifacts["qwen-image-dit"],
        "blake3": "d834b2740ee10a114d4472a541a8e0427a733a028194fb907d3df5454a4d5ac7",
        "bytes": 40861031488,
        "header_sha256": "9356eb06d3b193fa894c2921ad8f61b19bb87823d54b0f57e22bdd76bc3a5b9f",
        "local_path": "diffusion_models/qwen_image_bf16.safetensors",
        "sha256": "d08fb5d68026c0d87325f9a7b3ad6454061113a2bc73cc883114dae172937ae7",
    }
    assert artifacts["qwen2.5-vl-7b-text"] == {
        **artifacts["qwen2.5-vl-7b-text"],
        "blake3": "6f5249d557373ca144505c1d4193adffdf33e4d3a8ded719de65a8cb77780722",
        "bytes": 9384670680,
        "header_sha256": "c4e6e0abbd46c2216857d21eaef3e85ed56553e9ddd3103dbbeff243b8a38d73",
        "local_path": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "sha256": "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4",
    }
    assert artifacts["wan21-vae"] == {
        **artifacts["wan21-vae"],
        "blake3": "1018b941788940c4dfc449d5d749950cae58c9a7f6a794b00a2f8321abc885d1",
        "bytes": 253806246,
        "header_sha256": "5fcff35e07ec3899a69d23ddec25bc5ec29c092d2902a66c0135bf36f3ec7cc5",
        "local_path": "vae/qwen_image_vae.safetensors",
        "sha256": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    }
    for artifact in artifacts.values():
        assert artifact["acquisition"] == "reused-existing"
        assert f"/resolve/{manifest['provider']['revision']}/" in artifact["source_url"]


def test_manifest_refuses_mutable_provider_license_and_artifact_claims() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    cases = []
    mutable_provider = deepcopy(manifest)
    mutable_provider["provider"]["revision"] = "main"
    cases.append(mutable_provider)
    mutable_url = deepcopy(manifest)
    mutable_url["artifacts"][0]["source_url"] = mutable_url["artifacts"][0]["source_url"].replace(
        manifest["provider"]["revision"], "main"
    )
    cases.append(mutable_url)
    wrong_license = deepcopy(manifest)
    wrong_license["license_receipt"]["name"] = "unknown"
    cases.append(wrong_license)
    duplicate_role = deepcopy(manifest)
    duplicate_role["artifacts"][1]["role"] = duplicate_role["artifacts"][0]["role"]
    cases.append(duplicate_role)
    absolute_path = deepcopy(manifest)
    absolute_path["artifacts"][0]["local_path"] = "/tmp/model.safetensors"
    cases.append(absolute_path)
    wrong_repository = deepcopy(manifest)
    wrong_repository["provider"]["repository"] = "Qwen/Qwen-Image"
    cases.append(wrong_repository)
    wrong_provider_path = deepcopy(manifest)
    wrong_provider_path["artifacts"][0]["provider_path"] = "qwen_image_bf16.safetensors"
    cases.append(wrong_provider_path)
    unknown_manifest_field = deepcopy(manifest)
    unknown_manifest_field["note"] = "unverified"
    cases.append(unknown_manifest_field)
    unknown_artifact_field = deepcopy(manifest)
    unknown_artifact_field["artifacts"][0]["etag"] = "unverified"
    cases.append(unknown_artifact_field)
    boolean_schema = deepcopy(manifest)
    boolean_schema["schema"] = True
    cases.append(boolean_schema)
    integer_gated = deepcopy(manifest)
    integer_gated["provider"]["gated"] = 0
    cases.append(integer_gated)
    float_license_bytes = deepcopy(manifest)
    float_license_bytes["license_receipt"]["bytes"] = 11343.0
    cases.append(float_license_bytes)
    float_tensor_count = deepcopy(manifest)
    float_tensor_count["artifacts"][0]["header_contract"]["tensor_count"] = 1933.0
    cases.append(float_tensor_count)
    spaced_digest = deepcopy(manifest)
    spaced_digest["artifacts"][0]["sha256"] = " " + "0" * 63
    cases.append(spaced_digest)
    for broken in cases:
        with pytest.raises(ArtifactReceiptError):
            validate_manifest(broken)


def test_verify_receipts_uses_one_descriptor_for_digests_header_and_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    root = tmp_path / "models"
    root.mkdir()
    expected_header = {"value": {"data_offsets": [0, 1], "dtype": "F8_E4M3", "shape": [1]}}
    for artifact in manifest["artifacts"]:
        artifact["local_path"] = f"{artifact['role']}.safetensors"
        path = root / artifact["local_path"]
        _write_safetensors(path, expected_header, b"x")
        fd = os.open(path, os.O_RDONLY)
        try:
            artifact["sha256"], artifact["blake3"] = qwen_image_receipts._digests_fd(fd)
        finally:
            os.close(fd)
        artifact["bytes"] = path.stat().st_size
        artifact["header_sha256"] = canonical_header_digest(expected_header)
    manifest["artifact_root"] = "models"
    manifest["storage_receipt"] = {
        "after_free_bytes": 1,
        "before_free_bytes": 1,
        "minimum_free_bytes": 1,
        "required_download_bytes": 0,
        "reused_existing_bytes": sum(item["bytes"] for item in manifest["artifacts"]),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    seen: list[str] = []
    monkeypatch.setattr(
        qwen_image_receipts, "validate_dit_header", lambda _header: seen.append("dit")
    )
    monkeypatch.setattr(
        qwen_image_receipts, "validate_text_header", lambda _header: seen.append("text")
    )
    monkeypatch.setattr(
        qwen_image_receipts, "validate_vae_header", lambda _header: seen.append("vae")
    )
    result = verify_receipts(manifest_path, root)
    assert seen == ["dit", "text", "vae"]
    assert {item["role"] for item in result["verified"]} == {
        "qwen-image-dit",
        "qwen2.5-vl-7b-text",
        "wan21-vae",
    }
    manifest["artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    with pytest.raises(ArtifactReceiptError, match="SHA256"):
        verify_receipts(manifest_path, root)


def test_artifact_path_requires_ordinary_contained_exact_size(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "model.safetensors"
    artifact.write_bytes(b"artifact")
    with _open_artifact(root, "model.safetensors", 8) as (fd, path):
        assert path == artifact
        assert _pread(fd, 8, 0) == b"artifact"
    link = root / "link.safetensors"
    link.symlink_to(artifact)
    with pytest.raises(ArtifactReceiptError, match="ordinary file"):
        with _open_artifact(root, "link.safetensors", 8):
            pass
    with pytest.raises(ArtifactReceiptError, match="contained"):
        with _open_artifact(root, "../model.safetensors", 8):
            pass
    with pytest.raises(ArtifactReceiptError, match="size"):
        with _open_artifact(root, "model.safetensors", 7):
            pass
    nested = root / "nested"
    nested.mkdir()
    (nested / "model.safetensors").write_bytes(b"artifact")
    nested_link = root / "nested-link"
    nested_link.symlink_to(nested, target_is_directory=True)
    with pytest.raises(ArtifactReceiptError, match="ordinary file"):
        with _open_artifact(root, "nested-link/model.safetensors", 8):
            pass
    root_link = tmp_path / "root-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ArtifactReceiptError, match="root"):
        with _open_artifact(root_link, "model.safetensors", 8):
            pass


def test_open_artifact_keeps_one_inode_and_detects_in_place_changes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "model.safetensors"
    artifact.write_bytes(b"original")
    with pytest.raises(ArtifactReceiptError, match="changed"):
        with _open_artifact(root, "model.safetensors", 8) as (fd, _):
            artifact.rename(root / "old.safetensors")
            artifact.write_bytes(b"replaced")
            assert _pread(fd, 8, 0) == b"original"
    with pytest.raises(ArtifactReceiptError, match="changed"):
        with _open_artifact(root, "model.safetensors", 8):
            artifact.write_bytes(b"mutated!")


def test_open_artifact_detects_root_and_intermediate_replacement(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "model.safetensors").write_bytes(b"artifact")
    with pytest.raises(ArtifactReceiptError, match="root changed"):
        with _open_artifact(root, "nested/model.safetensors", 8):
            root.rename(tmp_path / "old-root")
            replacement = tmp_path / "root" / "nested"
            replacement.mkdir(parents=True)
            (replacement / "model.safetensors").write_bytes(b"artifact")

    root = tmp_path / "second-root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "model.safetensors").write_bytes(b"artifact")
    with pytest.raises(ArtifactReceiptError, match="directory changed"):
        with _open_artifact(root, "nested/model.safetensors", 8):
            nested.rename(root / "old-nested")
            replacement = root / "nested"
            replacement.mkdir()
            (replacement / "model.safetensors").write_bytes(b"artifact")


def test_safetensors_header_reader_refuses_truncation_payload_overlap_and_metadata(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.safetensors"
    _write_safetensors(
        valid,
        {"value": {"data_offsets": [0, 12], "dtype": "BF16", "shape": [2, 3]}},
        bytes(12),
    )
    header = read_safetensors_header(valid)
    assert header["value"]["shape"] == [2, 3]
    assert canonical_header_digest(header) == (
        "f399d7bfe6fba0c70fa0e951a18ebb7d957e1b35872a2a6483de6382187b7f70"
    )
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(struct.pack("<Q", 99) + b"{}")
    with pytest.raises(ArtifactReceiptError, match="truncated"):
        read_safetensors_header(truncated)
    overlapping = tmp_path / "overlapping.safetensors"
    _write_safetensors(
        overlapping,
        {
            "a": {"data_offsets": [0, 4], "dtype": "F32", "shape": [1]},
            "b": {"data_offsets": [3, 7], "dtype": "F32", "shape": [1]},
        },
        b"1234567",
    )
    with pytest.raises(ArtifactReceiptError, match="overlap"):
        read_safetensors_header(overlapping)
    metadata = tmp_path / "metadata.safetensors"
    _write_safetensors(metadata, {"__metadata__": {"format": "pt"}})
    with pytest.raises(ArtifactReceiptError, match="metadata"):
        read_safetensors_header(metadata)
    wrong_span = tmp_path / "wrong-span.safetensors"
    _write_safetensors(
        wrong_span,
        {"value": {"data_offsets": [0, 3], "dtype": "F32", "shape": [1]}},
        bytes(3),
    )
    with pytest.raises(ArtifactReceiptError, match="span"):
        read_safetensors_header(wrong_span)
    unsupported_dtype = tmp_path / "unsupported.safetensors"
    _write_safetensors(
        unsupported_dtype,
        {"value": {"data_offsets": [0, 2], "dtype": "I16", "shape": [1]}},
        bytes(2),
    )
    with pytest.raises(ArtifactReceiptError, match="geometry"):
        read_safetensors_header(unsupported_dtype)
    duplicate = tmp_path / "duplicate.safetensors"
    encoded = (
        b'{"value":{"data_offsets":[0,0],"dtype":"BF16","shape":[0]},'
        b'"value":{"data_offsets":[0,0],"dtype":"BF16","shape":[0]}}'
    )
    duplicate.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    with pytest.raises(ArtifactReceiptError, match="duplicate"):
        read_safetensors_header(duplicate)
    gap = tmp_path / "gap.safetensors"
    _write_safetensors(
        gap,
        {"value": {"data_offsets": [1, 2], "dtype": "F8_E4M3", "shape": [1]}},
        bytes(2),
    )
    with pytest.raises(ArtifactReceiptError, match="gap"):
        read_safetensors_header(gap)
    trailing = tmp_path / "trailing.safetensors"
    _write_safetensors(
        trailing,
        {"value": {"data_offsets": [0, 1], "dtype": "F8_E4M3", "shape": [1]}},
        bytes(2),
    )
    with pytest.raises(ArtifactReceiptError, match="trailing"):
        read_safetensors_header(trailing)
    malformed = tmp_path / "malformed.safetensors"
    _write_safetensors(
        malformed,
        {"value": {"data_offsets": [-1, 0], "dtype": "BF16", "shape": [0]}},
    )
    with pytest.raises(ArtifactReceiptError, match="geometry"):
        read_safetensors_header(malformed)


def test_dit_header_requires_exact_public_s2_role() -> None:
    header = {key: _entry("BF16", shape) for key, shape in qwen_image_dit_layout().keys.items()}
    validate_dit_header(header)
    header["img_in.weight"] = _entry("BF16", (3072, 63))
    with pytest.raises(ArtifactReceiptError, match="img_in.weight"):
        validate_dit_header(header)
    header = {key: _entry("F32", shape) for key, shape in qwen_image_dit_layout().keys.items()}
    with pytest.raises(ArtifactReceiptError, match="BF16"):
        validate_dit_header(header)


def test_scaled_text_header_maps_exactly_to_public_s4a_role() -> None:
    header: dict[str, dict[str, object]] = {}
    for index, (key, shape) in enumerate(qwen_image_text_layout().items()):
        dtype = "F8_E4M3" if index < 358 else "BF16"
        header[key] = _entry(dtype, shape)
        if dtype == "F8_E4M3":
            stem = key.removesuffix(".weight")
            header[f"{stem}.scale_input"] = _entry("F32", ())
            header[f"{stem}.scale_weight"] = _entry("F32", ())
    header["lm_head.weight"] = _entry("BF16", (152064, 3584))
    header["scaled_fp8"] = _entry("F8_E4M3", (0,))
    validate_text_header(header)
    valid_header = deepcopy(header)
    del header[next(key for key in header if key.endswith(".scale_weight"))]
    with pytest.raises(ArtifactReceiptError, match="scale"):
        validate_text_header(header)
    header = valid_header
    removed_scale = next(key for key in header if key.endswith(".scale_input"))
    header[removed_scale] = _entry("BF16", ())
    with pytest.raises(ArtifactReceiptError):
        validate_text_header(header)


def test_vae_header_requires_exact_public_s5a_role_anchors() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    contract = next(
        item["header_contract"] for item in manifest["artifacts"] if item["role"] == "wan21-vae"
    )
    header = {
        key: _entry("BF16", tuple(shape)) for key, shape in contract["required_tensors"].items()
    }
    for index in range(contract["tensor_count"] - len(header)):
        header[f"provider.tensor.{index}"] = _entry("BF16", ())
    validate_vae_header(header)
    header["conv2.weight"] = _entry("BF16", (15, 16, 1, 1, 1))
    with pytest.raises(ArtifactReceiptError, match="conv2.weight"):
        validate_vae_header(header)
    header["conv2.weight"] = _entry("BF16", (16, 16, 1, 1, 1))
    header["provider.tensor.0"] = _entry("F32", ())
    with pytest.raises(ArtifactReceiptError, match="storage"):
        validate_vae_header(header)
    del header["provider.tensor.0"]
    with pytest.raises(ArtifactReceiptError, match="count"):
        validate_vae_header(header)
