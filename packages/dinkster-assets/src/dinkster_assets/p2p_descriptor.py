"""Canonical BitTorrent v2 descriptors for BLAKE3-identified assets."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from .identity import DIGEST_PREFIX, AssetError, new_hasher, require_digest

P2P_PROTOCOL = "bittorrent-v2"
P2P_BLOCK_LENGTH = 16 * 1024
P2P_PIECE_LENGTH = 8 * 1024 * 1024

_BLOCKS_PER_PIECE = P2P_PIECE_LENGTH // P2P_BLOCK_LENGTH
_HEX_256_RE = re.compile(r"^[0-9a-f]{64}$")
_DESCRIPTOR_FIELDS = {"protocol", "infoHash", "fileRoot", "pieceLength"}
_ZERO_HASH = bytes(32)

_BencodeValue: TypeAlias = int | bytes | Mapping[bytes, "_BencodeValue"]


class P2PDescriptorError(AssetError):
    """A BitTorrent v2 descriptor or its asset binding is invalid."""


def _require_sha256_hex(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HEX_256_RE.fullmatch(value) is None:
        raise P2PDescriptorError(f"{field_name} must be 64 lowercase SHA-256 hex characters")
    return value


def _require_protocol(value: object) -> str:
    if value != P2P_PROTOCOL or not isinstance(value, str):
        raise P2PDescriptorError(f"protocol must be {P2P_PROTOCOL!r}")
    return value


def _require_piece_length(value: object) -> int:
    if type(value) is not int or value != P2P_PIECE_LENGTH:
        raise P2PDescriptorError(f"pieceLength must be the integer {P2P_PIECE_LENGTH}")
    return value


def _require_size(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise P2PDescriptorError("BitTorrent v2 assets must have a positive integer size")
    return value


@dataclass(frozen=True)
class P2PDescriptorV1:
    """Strict JSON-facing descriptor fields for the canonical profile."""

    protocol: str
    info_hash: str
    file_root: str
    piece_length: int

    def __post_init__(self) -> None:
        _require_protocol(self.protocol)
        _require_sha256_hex(self.info_hash, "infoHash")
        _require_sha256_hex(self.file_root, "fileRoot")
        _require_piece_length(self.piece_length)

    def to_wire(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "infoHash": self.info_hash,
            "fileRoot": self.file_root,
            "pieceLength": self.piece_length,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> P2PDescriptorV1:
        if set(wire) != _DESCRIPTOR_FIELDS:
            missing = sorted(_DESCRIPTOR_FIELDS - set(wire))
            extra = sorted(set(wire) - _DESCRIPTOR_FIELDS)
            raise P2PDescriptorError(
                f"descriptor fields must be exactly {sorted(_DESCRIPTOR_FIELDS)}; "
                f"missing={missing}, extra={extra}"
            )
        return cls(
            protocol=_require_protocol(wire["protocol"]),
            info_hash=_require_sha256_hex(wire["infoHash"], "infoHash"),
            file_root=_require_sha256_hex(wire["fileRoot"], "fileRoot"),
            piece_length=_require_piece_length(wire["pieceLength"]),
        )


@dataclass(frozen=True)
class P2PDescriptorResult:
    """One scan's identity, descriptor, info bytes, and BEP 52 piece layer."""

    asset_digest: str
    size: int
    descriptor: P2PDescriptorV1
    info: bytes
    piece_layer: bytes


class P2PDescriptorBuilder:
    """Incrementally derive canonical identity and BEP 52 descriptor material."""

    def __init__(self) -> None:
        self._hasher = new_hasher()
        self._block = bytearray()
        self._piece_blocks: list[bytes] = []
        self._piece_roots: list[bytes] = []
        self._size = 0
        self._finalized = False

    def update(self, data: bytes) -> None:
        """Consume the next bytes without retaining the complete asset."""
        if self._finalized:
            raise P2PDescriptorError("descriptor builder is already finalized")
        if type(data) is not bytes:
            raise TypeError("descriptor builder data must be bytes")

        self._hasher.update(data)
        self._size += len(data)
        offset = 0

        if self._block:
            take = min(P2P_BLOCK_LENGTH - len(self._block), len(data))
            self._block.extend(data[:take])
            offset = take
            if len(self._block) == P2P_BLOCK_LENGTH:
                self._add_block_hash(hashlib.sha256(self._block).digest())
                self._block.clear()

        while len(data) - offset >= P2P_BLOCK_LENGTH:
            end = offset + P2P_BLOCK_LENGTH
            self._add_block_hash(hashlib.sha256(data[offset:end]).digest())
            offset = end
        self._block.extend(data[offset:])

    def _add_block_hash(self, block_hash: bytes) -> None:
        self._piece_blocks.append(block_hash)
        if len(self._piece_blocks) == _BLOCKS_PER_PIECE:
            self._piece_roots.append(_merkle_root(self._piece_blocks))
            self._piece_blocks.clear()

    def finalize(self) -> P2PDescriptorResult:
        """Finish the descriptor; the builder cannot be reused afterward."""
        if self._finalized:
            raise P2PDescriptorError("descriptor builder is already finalized")
        self._finalized = True
        if self._size == 0:
            raise P2PDescriptorError("BitTorrent v2 assets must not be empty")

        if self._block:
            self._piece_blocks.append(hashlib.sha256(self._block).digest())
            self._block.clear()
        if self._piece_blocks:
            if self._piece_roots:
                self._piece_blocks.extend(
                    [_ZERO_HASH] * (_BLOCKS_PER_PIECE - len(self._piece_blocks))
                )
            self._piece_roots.append(_merkle_root(self._piece_blocks))
            self._piece_blocks.clear()

        return _descriptor_result(self._hasher.hexdigest(), self._size, self._piece_roots)


def _bencode(value: _BencodeValue) -> bytes:
    if isinstance(value, bool):
        raise TypeError("booleans are not canonical bencode integers")
    if isinstance(value, int):
        return f"i{value}e".encode("ascii")
    if isinstance(value, bytes):
        return str(len(value)).encode("ascii") + b":" + value
    encoded = bytearray(b"d")
    for key in sorted(value):
        encoded.extend(_bencode(key))
        encoded.extend(_bencode(value[key]))
    encoded.extend(b"e")
    return bytes(encoded)


def _merkle_root(hashes: Sequence[bytes], *, pad_hash: bytes = _ZERO_HASH) -> bytes:
    if not hashes:
        raise ValueError("a Merkle tree requires at least one hash")
    level = list(hashes)
    target_count = 1 << (len(level) - 1).bit_length()
    level.extend([pad_hash] * (target_count - len(level)))
    while len(level) > 1:
        level = [
            hashlib.sha256(level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0]


def _zero_piece_root() -> bytes:
    root = _ZERO_HASH
    for _ in range(_BLOCKS_PER_PIECE.bit_length() - 1):
        root = hashlib.sha256(root + root).digest()
    return root


_ZERO_PIECE_ROOT = _zero_piece_root()


def canonical_p2p_info(*, asset_digest: str, size: int, file_root: str) -> bytes:
    """Encode the only accepted v2 single-file info dictionary."""
    digest = require_digest(asset_digest)
    _require_size(size)
    root = bytes.fromhex(_require_sha256_hex(file_root, "fileRoot"))
    name = digest.removeprefix(DIGEST_PREFIX).encode("ascii")
    return _bencode(
        {
            b"file tree": {name: {b"": {b"length": size, b"pieces root": root}}},
            b"meta version": 2,
            b"name": name,
            b"piece length": P2P_PIECE_LENGTH,
        }
    )


def _descriptor_result(
    digest_hex: str, size: int, piece_roots: Sequence[bytes]
) -> P2PDescriptorResult:
    if len(piece_roots) == 1:
        file_root_bytes = piece_roots[0]
        piece_layer = b""
    else:
        file_root_bytes = _merkle_root(piece_roots, pad_hash=_ZERO_PIECE_ROOT)
        piece_layer = b"".join(piece_roots)
    asset_digest = DIGEST_PREFIX + digest_hex
    info = canonical_p2p_info(
        asset_digest=asset_digest,
        size=size,
        file_root=file_root_bytes.hex(),
    )
    descriptor = P2PDescriptorV1(
        protocol=P2P_PROTOCOL,
        info_hash=hashlib.sha256(info).hexdigest(),
        file_root=file_root_bytes.hex(),
        piece_length=P2P_PIECE_LENGTH,
    )
    return P2PDescriptorResult(asset_digest, size, descriptor, info, piece_layer)


def derive_p2p_descriptor(path: Path | str) -> P2PDescriptorResult:
    """Derive BLAKE3 identity and BEP 52 material in one bounded scan."""
    source = Path(path)
    builder = P2PDescriptorBuilder()
    total = 0

    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise P2PDescriptorError(f"descriptor source must be a regular file: {source}")
        if before.st_size <= 0:
            raise P2PDescriptorError("BitTorrent v2 assets must not be empty")
        while chunk := handle.read(P2P_PIECE_LENGTH):
            builder.update(chunk)
            total += len(chunk)
        after = os.fstat(handle.fileno())

    if total != before.st_size or (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise P2PDescriptorError(f"descriptor source changed while it was being hashed: {source}")

    return builder.finalize()


def validate_p2p_descriptor(
    descriptor: P2PDescriptorV1 | Mapping[str, object],
    *,
    asset_digest: str,
    size: int,
    info: bytes | None = None,
    piece_layer: bytes | None = None,
) -> P2PDescriptorV1:
    """Validate descriptor fields and their canonical asset binding."""
    parsed = (
        descriptor
        if isinstance(descriptor, P2PDescriptorV1)
        else P2PDescriptorV1.from_wire(descriptor)
    )
    canonical_info = canonical_p2p_info(
        asset_digest=asset_digest,
        size=size,
        file_root=parsed.file_root,
    )
    if hashlib.sha256(canonical_info).hexdigest() != parsed.info_hash:
        raise P2PDescriptorError(
            "infoHash does not match the canonical single-file info dictionary"
        )
    if info is not None:
        if type(info) is not bytes or info != canonical_info:
            raise P2PDescriptorError("info must be the exact canonical single-file v2 dictionary")
    if piece_layer is not None:
        if type(piece_layer) is not bytes:
            raise P2PDescriptorError("piece layer must be bytes")
        expected_hashes = (size + P2P_PIECE_LENGTH - 1) // P2P_PIECE_LENGTH
        if expected_hashes == 1:
            if piece_layer:
                raise P2PDescriptorError("single-piece assets must have an empty piece layer")
        else:
            if len(piece_layer) != expected_hashes * 32:
                raise P2PDescriptorError("piece layer length does not match the asset size")
            roots = [piece_layer[offset : offset + 32] for offset in range(0, len(piece_layer), 32)]
            if _merkle_root(roots, pad_hash=_ZERO_PIECE_ROOT).hex() != parsed.file_root:
                raise P2PDescriptorError("piece layer does not match fileRoot")
    return parsed


def verify_p2p_descriptor(
    path: Path | str,
    descriptor: P2PDescriptorV1 | Mapping[str, object],
    *,
    asset_digest: str,
    size: int,
    info: bytes | None = None,
    piece_layer: bytes | None = None,
) -> P2PDescriptorV1:
    """Validate a descriptor and rederive it from the referenced file."""
    parsed = validate_p2p_descriptor(
        descriptor,
        asset_digest=asset_digest,
        size=size,
        info=info,
        piece_layer=piece_layer,
    )
    actual = derive_p2p_descriptor(path)
    if actual.asset_digest != asset_digest:
        raise P2PDescriptorError("asset bytes do not match the BLAKE3 identity")
    if actual.size != size:
        raise P2PDescriptorError("asset byte size does not match the descriptor binding")
    if actual.descriptor != parsed:
        raise P2PDescriptorError("asset bytes do not match the BitTorrent v2 descriptor")
    if piece_layer is not None and actual.piece_layer != piece_layer:
        raise P2PDescriptorError("asset bytes do not match the published piece layer")
    return parsed
