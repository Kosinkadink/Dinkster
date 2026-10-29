"""Canonical BitTorrent v2 descriptor conformance and validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import libtorrent as lt
import pytest
from dinkster_assets import (
    P2P_BLOCK_LENGTH,
    P2P_PIECE_LENGTH,
    P2P_PROTOCOL,
    AssetVault,
    P2PDescriptorBuilder,
    P2PDescriptorError,
    P2PDescriptorResult,
    P2PDescriptorV1,
    derive_p2p_descriptor,
    digest_file,
    validate_p2p_descriptor,
    verify_p2p_descriptor,
)

LIBTORRENT_CONFORMANCE_VERSION = "2.1.1.0"
_PATTERN = bytes(range(251))

_GOLDENS = [
    (
        1,
        "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
        "e8f3d05e065ee066bdf67fad4613b17c159b0bcad58ff07f40162575ef8b16d5",
        "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
        "",
    ),
    (
        16 * 1024 - 1,
        "7529418ecb789a30254899f229522ccde05234d2019c5ec5072bc4685a57658d",
        "8e3e6b98edae11041a8a5f99f946f0ac2d567abb558d7aebb47b6234931718da",
        "e08c7d58e58b9318263144a618d6f2b6f6825974decd9e6f9371567354b6f566",
        "",
    ),
    (
        16 * 1024,
        "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde4",
        "0eae85ffd4ddd657920f2cf6368fd8c7e7d2b4578ece0b23592e9a8468962c84",
        "4348e3b98e8a327b34ced39c1da9e67cdb4cd5e48e4d7960607a3ae403d35f0c",
        "",
    ),
    (
        16 * 1024 + 1,
        "1dabe216be2578830263b049de1639f39f05a4da616b9b78c7a5e4e41662fd1f",
        "e10b422c4d4aff087403f76b88f367f07e2db257bec06d13bff6b7be6603d1eb",
        "9d7887c65d577a0237fb3c0998b87b3a62762d03796889a2caea01db914ccbb8",
        "",
    ),
    (
        P2P_PIECE_LENGTH - 1,
        "be5bd5bd53dedb92269e1fe52a5451c500bb76cdd6da82b9a4b0a4c8b7d1e3e3",
        "71453005d1f1d34e64080256a5e7547db5ff70ecf676bf8d3218c0dbd9405f18",
        "8329c7b4c0647c91756b563830514bf1aa3f26f78ebedf829e9c52c7e546080c",
        "",
    ),
    (
        P2P_PIECE_LENGTH,
        "1adedad9735f565ac6e22dab203db63b960c27098f2c0f0fda9adf9238d4c0c9",
        "faac2f12778f92445b0b10bf7dbac1c1be40cb5928b3f16bce63c602bf6ec544",
        "0da00444b10294dfdc7500f31586092c84898d891cb10cda9af185f3730eb03c",
        "",
    ),
    (
        P2P_PIECE_LENGTH + 1,
        "249ef5d5043bde86396029c96497c8ac6c0e7f2f5ef97b7f9afabdf23167b7fb",
        "ca3f9898b219858884dd7afdaf73f08b89a22cb033abe32ffcb47d056691c2a3",
        "1ab58ed57c22ed388f1f47867567fd4d885fb56f2d01bae5cfa49d83cb255ae7",
        "0da00444b10294dfdc7500f31586092c84898d891cb10cda9af185f3730eb03c"
        "a13a0fbfdb69b7f222926162b030e57bf5e2dfd9d1087185c1005d478f9b8374",
    ),
    (
        2 * P2P_PIECE_LENGTH + 12345,
        "215dc3702f9872c22c5b6c071420d976649a9b564dddeae05a974794551f762a",
        "60f2b05f7e1f40ef243890ae306cd1d61832cea2ca37192e8564aedc6e7127dc",
        "90a9d9507ce37c22858ab9d255d984e4e632a1e555b912813064e85b3c951fd4",
        "0da00444b10294dfdc7500f31586092c84898d891cb10cda9af185f3730eb03c"
        "67cac263433086c1a2f41d5b667f98c4a2be4d66da0afc9c9d5f95362ab3526b"
        "82274cdd8e62dae1e80c5e2f37ebe6441f622f6b230e00fbc6f43048b9d810da",
    ),
]


def _write_fixture(path: Path, size: int) -> None:
    with path.open("wb") as handle:
        remaining = size
        chunk = _PATTERN * (1024 * 1024 // len(_PATTERN))
        while remaining:
            written = min(remaining, len(chunk))
            handle.write(chunk[:written])
            remaining -= written


def _libtorrent_descriptor(path: Path, name: str) -> tuple[bytes, str, bytes]:
    assert lt.version == LIBTORRENT_CONFORMANCE_VERSION
    canonical_path = path.with_name(name)
    path.rename(canonical_path)
    files = lt.file_storage()
    files.add_file(name, canonical_path.stat().st_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torrent = lt.create_torrent(files, P2P_PIECE_LENGTH, lt.create_torrent.v2_only)
    lt.set_piece_hashes(torrent, str(canonical_path.parent))
    metainfo = cast("dict[bytes, Any]", torrent.generate())
    info = cast("dict[bytes, Any]", metainfo[b"info"])
    encoded_info = bytes(lt.bencode(info))
    file_tree = cast("dict[bytes, Any]", info[b"file tree"])
    file_entry = cast("dict[bytes, Any]", file_tree[name.encode("ascii")][b""])
    file_root = bytes(file_entry[b"pieces root"])
    piece_layers = cast("dict[bytes, bytes]", metainfo.get(b"piece layers", {}))
    piece_layer = bytes(next(iter(piece_layers.values()))) if piece_layers else b""
    return encoded_info, file_root.hex(), piece_layer


@pytest.mark.parametrize(
    ("size", "digest_hex", "info_hash", "file_root", "piece_layer_hex"), _GOLDENS
)
def test_golden_vectors_match_libtorrent_2_1_1(
    tmp_path: Path,
    size: int,
    digest_hex: str,
    info_hash: str,
    file_root: str,
    piece_layer_hex: str,
) -> None:
    path = tmp_path / "arbitrary-local-name.bin"
    _write_fixture(path, size)

    result = derive_p2p_descriptor(path)
    assert result.asset_digest == f"blake3:{digest_hex}"
    assert result.descriptor.to_wire() == {
        "protocol": P2P_PROTOCOL,
        "infoHash": info_hash,
        "fileRoot": file_root,
        "pieceLength": P2P_PIECE_LENGTH,
    }
    assert result.piece_layer.hex() == piece_layer_hex

    libtorrent_info, libtorrent_root, libtorrent_layer = _libtorrent_descriptor(path, digest_hex)
    assert result.info == libtorrent_info
    assert result.descriptor.info_hash == hashlib.sha256(libtorrent_info).hexdigest()
    assert result.descriptor.file_root == libtorrent_root
    assert result.piece_layer == libtorrent_layer


def _derived(tmp_path: Path, size: int = P2P_PIECE_LENGTH + 1) -> tuple[Path, P2PDescriptorResult]:
    path = tmp_path / "asset.bin"
    _write_fixture(path, size)
    return path, derive_p2p_descriptor(path)


def test_descriptor_wire_form_is_exact_and_round_trips(tmp_path: Path) -> None:
    _, result = _derived(tmp_path, 1)
    wire = result.descriptor.to_wire()
    assert P2PDescriptorV1.from_wire(wire) == result.descriptor
    with pytest.raises(P2PDescriptorError, match="fields must be exactly"):
        P2PDescriptorV1.from_wire({**wire, "tracker": "https://unexpected.example"})
    with pytest.raises(P2PDescriptorError, match="fields must be exactly"):
        P2PDescriptorV1.from_wire({key: value for key, value in wire.items() if key != "fileRoot"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol", "bittorrent-v1", "protocol"),
        ("infoHash", "A" * 64, "lowercase"),
        ("infoHash", "0" * 63, "64 lowercase"),
        ("fileRoot", "g" * 64, "64 lowercase"),
        ("pieceLength", 16 * 1024, "8388608"),
        ("pieceLength", True, "8388608"),
        ("pieceLength", float(P2P_PIECE_LENGTH), "8388608"),
    ],
)
def test_descriptor_rejects_malformed_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _, result = _derived(tmp_path, 1)
    wire = result.descriptor.to_wire()
    wire[field] = value
    with pytest.raises(P2PDescriptorError, match=message):
        P2PDescriptorV1.from_wire(wire)


def _mutated_info(result: P2PDescriptorResult, mutate: Callable[[dict[bytes, Any]], None]) -> bytes:
    decoded = cast("dict[bytes, Any]", lt.bdecode(result.info))
    mutate(decoded)
    return bytes(lt.bencode(decoded))


def test_published_info_must_be_exact_single_file_profile(tmp_path: Path) -> None:
    _, result = _derived(tmp_path, 16 * 1024 + 1)

    def add_file(info: dict[bytes, Any]) -> None:
        file_tree = info[b"file tree"]
        file_tree[b"other"] = file_tree[next(iter(file_tree))]

    mutations = [
        _mutated_info(result, lambda info: info.update({b"extra": b"value"})),
        _mutated_info(result, add_file),
        _mutated_info(result, lambda info: info.update({b"name": b"0" * 64})),
        _mutated_info(result, lambda info: info.update({b"piece length": 16 * 1024})),
    ]
    for info in mutations:
        with pytest.raises(P2PDescriptorError, match="exact canonical single-file"):
            validate_p2p_descriptor(
                result.descriptor,
                asset_digest=result.asset_digest,
                size=result.size,
                info=info,
            )


def test_descriptor_binding_rejects_mutated_identity_size_root_and_hash(tmp_path: Path) -> None:
    _, result = _derived(tmp_path, 16 * 1024 + 1)
    arguments = {
        "asset_digest": result.asset_digest,
        "size": result.size,
        "info": result.info,
    }
    with pytest.raises(P2PDescriptorError, match="infoHash"):
        validate_p2p_descriptor(
            result.descriptor, **{**arguments, "asset_digest": "blake3:" + "0" * 64}
        )
    with pytest.raises(P2PDescriptorError, match="infoHash"):
        validate_p2p_descriptor(result.descriptor, **{**arguments, "size": result.size + 1})

    root_wire = result.descriptor.to_wire()
    root_wire["fileRoot"] = "0" * 64
    with pytest.raises(P2PDescriptorError, match="infoHash"):
        validate_p2p_descriptor(root_wire, **arguments)

    hash_wire = result.descriptor.to_wire()
    hash_wire["infoHash"] = "0" * 64
    with pytest.raises(P2PDescriptorError, match="infoHash"):
        validate_p2p_descriptor(hash_wire, **arguments)


def test_piece_layer_is_bound_to_size_and_file_root(tmp_path: Path) -> None:
    _, result = _derived(tmp_path)
    validate_p2p_descriptor(
        result.descriptor,
        asset_digest=result.asset_digest,
        size=result.size,
        piece_layer=result.piece_layer,
    )
    for mutated in (result.piece_layer[:-32], bytes(32) + result.piece_layer[32:]):
        with pytest.raises(P2PDescriptorError, match="piece layer"):
            validate_p2p_descriptor(
                result.descriptor,
                asset_digest=result.asset_digest,
                size=result.size,
                piece_layer=mutated,
            )


def test_mutated_bytes_fail_verification(tmp_path: Path) -> None:
    path, result = _derived(tmp_path, 16 * 1024 + 1)
    path.write_bytes(b"changed" + path.read_bytes()[7:])
    with pytest.raises(P2PDescriptorError, match="BLAKE3 identity"):
        verify_p2p_descriptor(
            path,
            result.descriptor,
            asset_digest=result.asset_digest,
            size=result.size,
            info=result.info,
            piece_layer=result.piece_layer,
        )


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.touch()
    with pytest.raises(P2PDescriptorError, match="must not be empty"):
        derive_p2p_descriptor(path)


def test_incremental_builder_matches_file_derivation_across_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "streamed.bin"
    _write_fixture(path, P2P_PIECE_LENGTH + P2P_BLOCK_LENGTH + 37)
    expected = derive_p2p_descriptor(path)
    builder = P2PDescriptorBuilder()

    with path.open("rb") as handle:
        for size in (
            P2P_BLOCK_LENGTH - 3,
            11,
            P2P_PIECE_LENGTH - P2P_BLOCK_LENGTH - 13,
            17,
        ):
            builder.update(handle.read(size))
        while chunk := handle.read(7001):
            builder.update(chunk)

    assert builder.finalize() == expected


def test_incremental_builder_rejects_empty_input_and_reuse() -> None:
    empty = P2PDescriptorBuilder()
    empty.update(b"")
    with pytest.raises(P2PDescriptorError, match="must not be empty"):
        empty.finalize()
    with pytest.raises(P2PDescriptorError, match="already finalized"):
        empty.update(b"late")

    builder = P2PDescriptorBuilder()
    builder.update(b"positive bytes")
    builder.finalize()
    with pytest.raises(P2PDescriptorError, match="already finalized"):
        builder.finalize()
    with pytest.raises(P2PDescriptorError, match="already finalized"):
        builder.update(b"late")


def test_identity_and_vault_layout_remain_authoritative(tmp_path: Path) -> None:
    data = b"vault asset bytes"
    vault = AssetVault(tmp_path / "vault")
    expected_digest = "blake3:158b984e04c8126ff6f0c1279433633270df7373b093fd75e95bdd91f42c882c"
    with vault.writer(expected_digest) as writer:
        writer.write(data)
        path = writer.commit()

    result = derive_p2p_descriptor(path)
    digest_hex = expected_digest.removeprefix("blake3:")
    assert result.asset_digest == expected_digest == digest_file(path)
    assert path == vault.root / digest_hex[:2] / digest_hex


def test_fixture_command_prints_and_verifies_identical_json(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    _write_fixture(path, 16 * 1024 + 1)
    command = [sys.executable, "-m", "dinkster_assets.p2p_fixture", str(path)]
    generated = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    fixture_path = tmp_path / "descriptor.json"
    fixture_path.write_text(generated, "utf-8")
    verified = subprocess.run(
        [*command, "--verify", str(fixture_path)], check=True, capture_output=True, text=True
    ).stdout
    assert json.loads(generated)["descriptor"]["protocol"] == P2P_PROTOCOL
    assert verified == generated


def test_descriptor_does_not_depend_on_local_name(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "nested" / "renamed.safetensors"
    second.parent.mkdir()
    _write_fixture(first, 16 * 1024 + 1)
    second.write_bytes(first.read_bytes())
    assert derive_p2p_descriptor(first) == derive_p2p_descriptor(second)
