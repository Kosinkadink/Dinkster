from __future__ import annotations

import json
import os
import random
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from dinkster_assets import (
    P2P_FORMAT_POLICY_VERSION,
    P2P_PARTIAL_RETENTION_SECONDS,
    P2P_PIECE_LENGTH,
    AssetVault,
    P2PDescriptorError,
    P2PDescriptorResult,
    P2PStorageError,
    derive_p2p_descriptor,
    digest_bytes,
)
from dinkster_assets import p2p_storage as storage_module

from tests.platform_support import symlink_or_skip


def _safetensors(data: bytes = b"fixture-model-bytes") -> bytes:
    header = {
        "weight": {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [0, len(data)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(encoded)) + encoded + data


def _gguf() -> bytes:
    name = b"weight"
    header = b"".join(
        (
            b"GGUF",
            struct.pack("<I", 3),
            struct.pack("<Q", 1),
            struct.pack("<Q", 0),
            struct.pack("<Q", len(name)),
            name,
            struct.pack("<I", 1),
            struct.pack("<Q", 1),
            struct.pack("<I", 0),
            struct.pack("<Q", 0),
        )
    )
    header += b"\0" * (-len(header) % 32)
    return header + struct.pack("<f", 1.0) + b"\0" * 28


def _descriptor(vault: AssetVault, data: bytes) -> P2PDescriptorResult:
    source = vault.root.parent / f"descriptor-source-{digest_bytes(data)[7:]}"
    source.write_bytes(data)
    return derive_p2p_descriptor(source)


def _stage(vault: AssetVault, data: bytes) -> tuple[P2PDescriptorResult, str, Path]:
    descriptor = _descriptor(vault, data)
    digest = descriptor.asset_digest
    path = vault.p2p_staging_path(descriptor.descriptor, digest, len(data))
    path.write_bytes(data)
    return descriptor, digest, path


def test_random_piece_restart_resume_and_adopt_into_real_vault(tmp_path: Path) -> None:
    payload_size = 2 * P2P_PIECE_LENGTH + 257
    pattern = bytes(range(251))
    data = _safetensors((pattern * (payload_size // len(pattern) + 1))[:payload_size])
    source = tmp_path / "source.safetensors"
    source.write_bytes(data)
    descriptor = derive_p2p_descriptor(source)
    digest = descriptor.asset_digest
    vault = AssetVault(tmp_path / "vault")
    partial = vault.open_p2p_partial(descriptor.descriptor, digest, len(data))
    pieces = [
        (offset, data[offset : offset + P2P_PIECE_LENGTH])
        for offset in range(0, len(data), P2P_PIECE_LENGTH)
    ]
    random.Random(1168).shuffle(pieces)

    for offset, piece in pieces[: len(pieces) // 2]:
        partial.write_piece(offset, piece)
    assert not partial.complete

    restarted = AssetVault(tmp_path / "vault").open_p2p_partial(
        descriptor.descriptor, digest, len(data)
    )
    for offset, piece in pieces[len(pieces) // 2 :]:
        restarted.write_piece(offset, piece)
    assert restarted.complete
    staged_inode = restarted.path.stat().st_ino

    published = vault.adopt_staged_asset(
        descriptor.descriptor,
        digest,
        len(data),
        restarted.path,
        P2P_FORMAT_POLICY_VERSION,
    )

    assert published == vault.root / digest[7:9] / digest[7:]
    assert published.read_bytes() == data
    assert published.stat().st_ino == staged_inode
    assert not restarted.path.exists()
    assert not restarted.resume_path.exists()
    resolution = vault.resolve_asset(digest)
    assert resolution is not None
    assert resolution.path == published
    assert resolution.verification is not None


def test_reopened_partials_serialize_resume_range_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _safetensors(bytes(range(128)))
    vault = AssetVault(tmp_path / "vault")
    descriptor = _descriptor(vault, data)
    digest = descriptor.asset_digest
    first = vault.open_p2p_partial(descriptor.descriptor, digest, len(data))
    second = vault.open_p2p_partial(descriptor.descriptor, digest, len(data))
    real_load = storage_module._load_resume_state  # pyright: ignore[reportPrivateUsage]
    entered = 0
    entered_lock = threading.Lock()
    second_entered = threading.Event()

    def synchronized_load(*args: object) -> object:
        nonlocal entered
        state = real_load(*args)  # type: ignore[arg-type]
        with entered_lock:
            order = entered
            entered += 1
        if order == 0:
            second_entered.wait(0.1)
        else:
            second_entered.set()
        return state

    monkeypatch.setattr(storage_module, "_load_resume_state", synchronized_load)
    midpoint = len(data) // 2
    with ThreadPoolExecutor(max_workers=2) as executor:
        writes = (
            executor.submit(first.write_piece, 0, data[:midpoint]),
            executor.submit(second.write_piece, midpoint, data[midpoint:]),
        )
        for write in writes:
            write.result()

    assert first.completed_ranges == ((0, len(data)),)


def test_staging_layout_is_canonical_and_bound_to_descriptor(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    data = _safetensors()
    descriptor = _descriptor(vault, data)
    digest = descriptor.asset_digest
    assert vault.p2p_staging_path(descriptor.descriptor, digest, len(data)) == (
        vault.root.resolve() / ".p2p" / "staging" / descriptor.descriptor.info_hash / digest[7:]
    )
    malformed = descriptor.descriptor.to_wire()
    malformed["infoHash"] = "0" * 64
    with pytest.raises(P2PDescriptorError, match="infoHash"):
        vault.open_p2p_partial(malformed, digest, len(data))


def test_adoption_rejects_staging_path_for_another_info_hash(tmp_path: Path) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    forged_parent = staged.parent.with_name("0" * 64)
    forged_parent.mkdir()
    forged = forged_parent / staged.name
    staged.rename(forged)

    with pytest.raises(P2PStorageError, match="validated descriptor"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            forged,
            P2P_FORMAT_POLICY_VERSION,
        )
    assert forged.read_bytes() == data
    assert vault.resolve(digest) is None


@pytest.mark.parametrize(
    "path_factory, message",
    (
        (lambda vault, staged, outside: outside, "outside"),
        (
            lambda vault, staged, outside: staged.parent / "nested" / ".." / staged.name,
            "traversal",
        ),
        (lambda vault, staged, outside: staged.parent / ("2" * 64), "canonical"),
    ),
)
def test_adoption_rejects_external_traversal_and_wrong_digest_paths(
    tmp_path: Path,
    path_factory: object,
    message: str,
) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    outside = tmp_path / "outside"
    outside.write_bytes(data)
    if message == "canonical":
        candidate = staged.parent / ("2" * 64)
        candidate.write_bytes(data)
    else:
        candidate = path_factory(vault, staged, outside)  # type: ignore[operator]

    with pytest.raises(P2PStorageError, match=message):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            candidate,
            P2P_FORMAT_POLICY_VERSION,
        )


def test_adoption_rejects_symlink_directory_and_external_hard_link(tmp_path: Path) -> None:
    data = _safetensors()
    symlink_vault = AssetVault(tmp_path / "symlink-vault")
    descriptor = _descriptor(symlink_vault, data)
    digest = descriptor.asset_digest
    staged = symlink_vault.p2p_staging_path(descriptor.descriptor, digest, len(data))
    outside = tmp_path / "outside"
    outside.write_bytes(data)
    symlink_or_skip(staged, outside)
    with pytest.raises(P2PStorageError, match="regular non-symlink"):
        symlink_vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )

    directory_vault = AssetVault(tmp_path / "directory-vault")
    staged = directory_vault.p2p_staging_path(descriptor.descriptor, digest, len(data))
    staged.mkdir()
    with pytest.raises(P2PStorageError, match="regular non-symlink"):
        directory_vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )

    parent_symlink_vault = AssetVault(tmp_path / "parent-symlink-vault")
    staging_root = parent_symlink_vault.p2p_staging_root
    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (outside_directory / digest[7:]).write_bytes(data)
    info_hash = descriptor.descriptor.info_hash
    symlink_or_skip(staging_root / info_hash, outside_directory, target_is_directory=True)
    with pytest.raises(P2PStorageError, match="not confined"):
        parent_symlink_vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staging_root / info_hash / digest[7:],
            P2P_FORMAT_POLICY_VERSION,
        )

    hardlink_vault = AssetVault(tmp_path / "hardlink-vault")
    staged = hardlink_vault.p2p_staging_path(descriptor.descriptor, digest, len(data))
    os.link(outside, staged)
    with pytest.raises(P2PStorageError, match="hard links"):
        hardlink_vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )


def test_p2p_staging_rejects_symlink_vault_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual-vault"
    actual.mkdir()
    link = tmp_path / "vault-link"
    symlink_or_skip(link, actual, target_is_directory=True)

    with pytest.raises(P2PStorageError, match="root must be a regular non-symlink"):
        _ = AssetVault(link).p2p_staging_root


def test_cross_filesystem_staged_stat_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    staged_identity = (staged.stat().st_dev, staged.stat().st_ino)
    real_fstat = os.fstat

    def cross_filesystem_fstat(descriptor: int) -> os.stat_result:
        item = real_fstat(descriptor)
        if (item.st_dev, item.st_ino) != staged_identity:
            return item
        values = list(item)
        values[2] += 1
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", cross_filesystem_fstat)
    with pytest.raises(P2PStorageError, match="filesystem"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )


def test_adoption_rejects_mutation_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _safetensors(b"before")
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    real_hash = storage_module._hash_handle  # pyright: ignore[reportPrivateUsage]

    def hash_then_mutate(handle: object) -> str:
        result = real_hash(handle)  # type: ignore[arg-type]
        handle.seek(len(data) - 1)  # type: ignore[attr-defined]
        handle.write(b"X")  # type: ignore[attr-defined]
        handle.flush()  # type: ignore[attr-defined]
        return result

    monkeypatch.setattr(storage_module, "_hash_handle", hash_then_mutate)
    with pytest.raises(P2PStorageError, match="changed during verification"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )
    assert vault.resolve(digest) is None
    assert staged.exists()


def test_adoption_rejects_staging_directory_rebind_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _safetensors(b"verified source")
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    real_validate = storage_module._validate_safe_format  # pyright: ignore[reportPrivateUsage]
    rebind_denied = False

    def validate_then_rebind(handle: object, version: int) -> str:
        nonlocal rebind_denied
        result = real_validate(handle, version)  # type: ignore[arg-type]
        displaced = staged.parent.with_name(staged.parent.name + "-displaced")
        try:
            staged.parent.rename(displaced)
        except PermissionError:
            if os.name != "nt":
                raise
            rebind_denied = True
            return result
        outside = tmp_path / "attacker-directory"
        outside.mkdir()
        symlink_or_skip(staged.parent, outside, target_is_directory=True)
        return result

    monkeypatch.setattr(storage_module, "_validate_safe_format", validate_then_rebind)
    if os.name == "nt":
        published = vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )
        assert rebind_denied
        assert published == vault.resolve(digest)
        return
    with pytest.raises(P2PStorageError, match="directory was rebound"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )
    assert vault.resolve(digest) is None


def test_adoption_rejects_bad_size_format_digest_and_policy(tmp_path: Path) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    staged.write_bytes(data + b"X")
    with pytest.raises(P2PStorageError, match="size mismatch"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )
    staged.write_bytes(data)
    with pytest.raises(P2PStorageError, match="unsupported P2P format policy"):
        vault.adopt_staged_asset(descriptor.descriptor, digest, len(data), staged, 2)

    bad_format = b"pickle and executable code are not safe model formats"
    bad_descriptor, bad_digest, bad_staged = _stage(vault, bad_format)
    with pytest.raises(P2PStorageError, match="safe model format|header size"):
        vault.adopt_staged_asset(
            bad_descriptor.descriptor,
            bad_digest,
            len(bad_format),
            bad_staged,
            P2P_FORMAT_POLICY_VERSION,
        )

    other = data[:-1] + b"X"
    staged.write_bytes(other)
    with pytest.raises(P2PStorageError, match="bytes hash to"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(other),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )


def test_safe_format_policy_rejects_duplicate_safetensors_keys(tmp_path: Path) -> None:
    tensor = b'{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
    header = b'{"weight":' + tensor + b',"weight":' + tensor + b"}"
    data = struct.pack("<Q", len(header)) + header + b"x"
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)

    with pytest.raises(P2PStorageError, match="duplicate key"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )


def test_safe_format_policy_accepts_strict_gguf_and_rejects_trailing_bytes(
    tmp_path: Path,
) -> None:
    vault = AssetVault(tmp_path / "vault")
    data = _gguf()
    descriptor, digest, staged = _stage(vault, data)
    assert (
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        ).read_bytes()
        == data
    )

    malformed = data + b"trailing"
    malformed_descriptor, malformed_digest, malformed_staged = _stage(vault, malformed)
    with pytest.raises(P2PStorageError, match="size does not match"):
        vault.adopt_staged_asset(
            malformed_descriptor.descriptor,
            malformed_digest,
            len(malformed),
            malformed_staged,
            P2P_FORMAT_POLICY_VERSION,
        )


def test_existing_valid_canonical_object_wins_publication_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    target = vault.root / digest[7:9] / digest[7:]
    real_validate = storage_module._validate_safe_format  # pyright: ignore[reportPrivateUsage]
    published = False

    def validate_and_publish_competing_object(handle: object, version: int) -> str:
        nonlocal published
        result = real_validate(handle, version)  # type: ignore[arg-type]
        if not published:
            published = True
            with vault.writer(digest) as writer:
                writer.write(data)
                writer.commit()
        return result

    monkeypatch.setattr(
        storage_module,
        "_validate_safe_format",
        validate_and_publish_competing_object,
    )
    result = vault.adopt_staged_asset(
        descriptor.descriptor,
        digest,
        len(data),
        staged,
        P2P_FORMAT_POLICY_VERSION,
    )

    assert result == target.resolve()
    assert target.read_bytes() == data
    assert not staged.exists()


def test_corrupt_existing_canonical_object_is_never_replaced(tmp_path: Path) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    target = vault.root / digest[7:9] / digest[7:]
    target.parent.mkdir(parents=True)
    corrupt = b"Z" * len(data)
    target.write_bytes(corrupt)

    with pytest.raises(P2PStorageError):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )
    assert target.read_bytes() == corrupt
    assert staged.read_bytes() == data


def test_hardlinked_existing_canonical_object_is_never_trusted(tmp_path: Path) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    target = vault.root / digest[7:9] / digest[7:]
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    external = tmp_path / "external-hardlink"
    os.link(target, external)

    with pytest.raises(P2PStorageError, match="does not match"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )
    assert staged.read_bytes() == data
    assert target.read_bytes() == data
    assert external.read_bytes() == data


def test_partial_retention_purges_at_seven_days_and_keeps_active(tmp_path: Path) -> None:
    now = time.time()
    vault = AssetVault(tmp_path / "vault")
    old_data = _safetensors(b"old")
    old_descriptor = _descriptor(vault, old_data)
    old = vault.open_p2p_partial(
        old_descriptor.descriptor,
        old_descriptor.asset_digest,
        len(old_data),
    )
    old.write_piece(0, old_data[:10])
    active_data = _safetensors(b"active")
    active_descriptor = _descriptor(vault, active_data)
    active = vault.open_p2p_partial(
        active_descriptor.descriptor,
        active_descriptor.asset_digest,
        len(active_data),
    )
    active.write_piece(0, active_data[:10])
    cutoff_ns = int(now * 1_000_000_000) - P2P_PARTIAL_RETENTION_SECONDS * 1_000_000_000
    os.utime(old.path, ns=(cutoff_ns, cutoff_ns))
    os.utime(old.resume_path, ns=(cutoff_ns, cutoff_ns))

    purged = vault.purge_inactive_p2p_partials(now=now)

    assert purged.removed_partials == 1
    assert purged.reclaimed_logical_bytes > 0
    assert not old.path.exists()
    assert not old.resume_path.exists()
    assert active.path.exists()
    assert active.resume_path.exists()


def test_staging_quota_reports_actual_and_logical_bytes_without_following_symlinks(
    tmp_path: Path,
) -> None:
    vault = AssetVault(tmp_path / "vault")
    data = _safetensors(b"quota")
    asset = data + b"\0" * (8 * 1024 * 1024 - len(data))
    descriptor = _descriptor(vault, asset)
    partial = vault.open_p2p_partial(
        descriptor.descriptor,
        descriptor.asset_digest,
        len(asset),
    )
    partial.write_piece(0, data)
    external = tmp_path / "large-external"
    external.write_bytes(b"X" * 1024 * 1024)
    link = vault.p2p_staging_root / "external-link"
    symlink_or_skip(link, external)

    usage = vault.p2p_staging_usage()
    expected_logical = sum(
        path.lstat().st_size for path in vault.p2p_staging_root.rglob("*") if not path.is_dir()
    )

    assert usage.partials == 1
    assert usage.logical_bytes == expected_logical
    assert usage.logical_bytes < external.stat().st_size + partial.size
    assert usage.actual_bytes >= 0


def test_purge_does_not_follow_info_hash_directory_symlink(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / ("a" * 64)
    external.write_bytes(b"must remain")
    symlink_or_skip(vault.p2p_staging_root / ("4" * 64), outside, target_is_directory=True)

    vault.purge_inactive_p2p_partials(now=time.time() + 2 * P2P_PARTIAL_RETENTION_SECONDS)

    assert external.read_bytes() == b"must remain"


def test_adoption_rejects_rebound_canonical_shard(tmp_path: Path) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    outside = tmp_path / "outside-canonical"
    outside.mkdir()
    symlink_or_skip(vault.root / digest[7:9], outside, target_is_directory=True)

    with pytest.raises(P2PStorageError, match="not confined"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )
    assert not (outside / digest[7:]).exists()


def test_reopen_rejects_symlink_resume_state(tmp_path: Path) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor = _descriptor(vault, data)
    digest = descriptor.asset_digest
    partial = vault.open_p2p_partial(descriptor.descriptor, digest, len(data))
    external = tmp_path / "external-resume.json"
    external.write_text(partial.resume_path.read_text("utf-8"), "utf-8")
    partial.resume_path.unlink()
    symlink_or_skip(partial.resume_path, external)

    with pytest.raises(P2PStorageError, match="regular non-symlink"):
        vault.open_p2p_partial(descriptor.descriptor, digest, len(data))
    assert external.exists()


def test_no_copy_local_mapping_revokes_on_mutation_deletion_and_symlink(tmp_path: Path) -> None:
    data = _safetensors(b"local model")
    digest = digest_bytes(data)
    local = tmp_path / "models" / "model.safetensors"
    local.parent.mkdir()
    local.write_bytes(data)
    vault = AssetVault(tmp_path / "vault")

    mapping = vault.verify_p2p_local_file(
        digest,
        len(data),
        local.resolve(),
        P2P_FORMAT_POLICY_VERSION,
    )
    assert mapping.path == local.resolve()
    assert mapping.require_current() == local.resolve()
    assert vault.resolve(digest) is None

    verified = local.stat()
    local.write_bytes(data[:-1] + b"X")
    os.utime(local, ns=(verified.st_atime_ns, verified.st_mtime_ns))
    assert not mapping.is_current()
    with pytest.raises(P2PStorageError, match="stale"):
        mapping.require_current()
    local.unlink()
    assert not mapping.is_current()

    target = tmp_path / "target.safetensors"
    target.write_bytes(data)
    symlink_or_skip(local, target)
    with pytest.raises(P2PStorageError):
        vault.verify_p2p_local_file(
            digest,
            len(data),
            local.absolute(),
            P2P_FORMAT_POLICY_VERSION,
        )


@pytest.mark.skipif(os.name != "nt", reason="exercises the Windows file fingerprint")
@pytest.mark.parametrize("canonical_name", [False, True], ids=["short-name", "vault-name"])
def test_windows_local_mapping_poll_uses_fingerprint_without_rehashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, canonical_name: bool
) -> None:
    data = _safetensors(b"local model")
    digest = digest_bytes(data)
    local = tmp_path / (digest.split(":", 1)[1] if canonical_name else "model.safetensors")
    local.write_bytes(data)
    vault = AssetVault(tmp_path / "vault")
    mapping = vault.verify_p2p_local_file(
        digest,
        len(data),
        local.resolve(),
        P2P_FORMAT_POLICY_VERSION,
    )
    full_reads = 0

    def count_full_read(_handle: object) -> str:
        nonlocal full_reads
        full_reads += 1
        return digest

    monkeypatch.setattr(storage_module, "_hash_handle", count_full_read)
    unchanged = mapping.is_current()
    assert unchanged
    assert full_reads == 0

    verified = local.stat()
    local.write_bytes(data[:-1] + b"X")
    os.utime(local, ns=(verified.st_atime_ns, verified.st_mtime_ns))
    changed = mapping.is_current()
    assert not changed
    assert full_reads == 0


def test_corrupt_resume_state_fails_closed_without_truncating_partial(tmp_path: Path) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor = _descriptor(vault, data)
    digest = descriptor.asset_digest
    partial = vault.open_p2p_partial(descriptor.descriptor, digest, len(data))
    partial.write_piece(0, data[:10])
    before = partial.path.read_bytes()
    partial.resume_path.write_text("{}", "utf-8")

    with pytest.raises(P2PStorageError, match="fields are not canonical"):
        vault.open_p2p_partial(descriptor.descriptor, digest, len(data))
    assert partial.path.read_bytes() == before


def test_missing_resume_state_is_rebuilt_without_claiming_ranges(tmp_path: Path) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor = _descriptor(vault, data)
    digest = descriptor.asset_digest
    path = vault.p2p_staging_path(descriptor.descriptor, digest, len(data))
    path.write_bytes(b"")

    partial = vault.open_p2p_partial(descriptor.descriptor, digest, len(data))

    assert partial.path.stat().st_size == len(data)
    assert partial.completed_ranges == ()
    assert partial.resume_path.is_file()


@pytest.mark.skipif(os.name != "posix", reason="exercises POSIX replace while source is open")
def test_post_rename_identity_race_does_not_delete_competing_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _safetensors()
    vault = AssetVault(tmp_path / "vault")
    descriptor, digest, staged = _stage(vault, data)
    target = vault.root / digest[7:9] / digest[7:]
    competing = tmp_path / "competing"
    competing.write_bytes(data)
    real_rename = storage_module._rename_noreplace  # pyright: ignore[reportPrivateUsage]

    def rename_then_replace(*args: object) -> None:
        real_rename(*args)  # type: ignore[arg-type]
        os.replace(competing, target)

    monkeypatch.setattr(storage_module, "_rename_noreplace", rename_then_replace)

    with pytest.raises(P2PStorageError, match="rebound during publication"):
        vault.adopt_staged_asset(
            descriptor.descriptor,
            digest,
            len(data),
            staged,
            P2P_FORMAT_POLICY_VERSION,
        )
    assert target.read_bytes() == data
