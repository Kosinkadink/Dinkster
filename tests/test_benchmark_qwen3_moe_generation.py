"""Qwen3 MoE benchmark artifact validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import tools.benchmark_qwen3_moe_generation as benchmark


def test_pinned_shard_manifest_is_complete() -> None:
    assert len(benchmark.SHARDS) == 16
    assert sum(shard.size for shard in benchmark.SHARDS) == benchmark.MODEL_FILE_BYTES
    assert len({shard.name for shard in benchmark.SHARDS}) == 16
    assert len({shard.sha256 for shard in benchmark.SHARDS}) == 16


def test_shard_validation_requires_pinned_size_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"exact checkpoint shard"
    shard = benchmark.ArtifactShard(
        "model-00001-of-00001.safetensors",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(benchmark, "SHARDS", (shard,))
    path = tmp_path / shard.name
    path.write_bytes(payload)

    assert benchmark._verify_shards(tmp_path) == (path,)

    path.write_bytes(payload + b"!")
    with pytest.raises(SystemExit, match="size differs"):
        benchmark._verify_shards(tmp_path)


def test_shard_validation_rejects_missing_and_wrong_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = benchmark.ArtifactShard(
        "model-00001-of-00001.safetensors",
        1,
        hashlib.sha256(b"a").hexdigest(),
    )
    monkeypatch.setattr(benchmark, "SHARDS", (shard,))
    with pytest.raises(SystemExit, match="missing pinned"):
        benchmark._verify_shards(tmp_path)

    path = tmp_path / shard.name
    path.write_bytes(b"b")
    with pytest.raises(SystemExit, match="sha256 differs"):
        benchmark._verify_shards(tmp_path)
