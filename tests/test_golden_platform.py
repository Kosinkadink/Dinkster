from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.golden_platform import (
    GOLDEN_OFFLINE_ENV,
    GoldenIntegrityError,
    GoldenUnavailableError,
    GoldenVariantNotFoundError,
    fetch_platform_golden,
    platform_golden_path,
    platform_variant_output_path,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_evidence(
    evidence_root: Path,
    relative: Path,
    key: str,
    data: bytes,
    *,
    sha256: str | None = None,
) -> Path:
    source = evidence_root / "platform-goldens" / "files" / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    manifest = {
        "format": "dinkster-platform-goldens/1",
        "files": [
            {
                "path": relative.as_posix(),
                "platform": key,
                "sha256": sha256 or _sha256(data),
            }
        ],
    }
    manifest_path = evidence_root / "platform-goldens" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source


def _fixture_paths(tmp_path: Path, key: str) -> tuple[Path, Path, Path, Path]:
    dinkster_root = tmp_path / "Dinkster"
    baseline = dinkster_root / "tests" / "goldens" / "sample.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("baseline", encoding="utf-8")
    relative = Path("tests/goldens") / f"sample.{key}.json"
    return dinkster_root, baseline, tmp_path / "evidence", relative


def test_generators_write_platform_variants_to_evidence(tmp_path: Path) -> None:
    dinkster_root = tmp_path / "Dinkster"
    baseline = dinkster_root / "tests" / "goldens" / "sample.json"
    evidence_root = tmp_path / "dinkster-evidence"
    (evidence_root / "platform-goldens" / "files").mkdir(parents=True)
    output = platform_variant_output_path(
        baseline,
        "win32-py3.12-torch2.13",
        dinkster_root=dinkster_root,
        evidence_root=evidence_root,
    )
    assert output == (
        evidence_root
        / "platform-goldens"
        / "files"
        / "tests"
        / "goldens"
        / "sample.win32-py3.12-torch2.13.json"
    )
    assert not any(dinkster_root.rglob("sample.*.json"))


def test_generator_requires_evidence_layout(tmp_path: Path) -> None:
    dinkster_root = tmp_path / "Dinkster"
    baseline = dinkster_root / "tests" / "goldens" / "sample.json"
    with pytest.raises(RuntimeError, match="platform-goldens checkout not found"):
        platform_variant_output_path(
            baseline,
            "darwin-py3.12-torch2.13",
            dinkster_root=dinkster_root,
            evidence_root=tmp_path / "missing-evidence",
        )


def test_generator_uses_baseline_on_linux_and_evidence_on_other_platforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dinkster_root = tmp_path / "Dinkster"
    baseline = dinkster_root / "tests" / "goldens" / "sample.json"
    evidence_root = tmp_path / "dinkster-evidence"
    (evidence_root / "platform-goldens" / "files").mkdir(parents=True)
    monkeypatch.setattr("tools.golden_platform.sys.platform", "linux")
    assert (
        platform_golden_path(
            baseline,
            "2.13.0",
            dinkster_root=dinkster_root,
            evidence_root=evidence_root,
        )
        == baseline
    )
    monkeypatch.setattr("tools.golden_platform.sys.platform", "win32")
    assert platform_golden_path(
        baseline,
        "2.13.0",
        dinkster_root=dinkster_root,
        evidence_root=evidence_root,
    ).is_relative_to(evidence_root / "platform-goldens" / "files")


def test_fetches_verified_variant_and_reuses_cache(tmp_path: Path) -> None:
    key = "win32-py3.12.11-torch2.13.0+cpu"
    dinkster_root, baseline, evidence_root, relative = _fixture_paths(tmp_path, key)
    source = _write_evidence(evidence_root, relative, key, b"platform bytes")
    cache_root = dinkster_root / ".golden-cache"

    cached = fetch_platform_golden(
        baseline,
        key,
        dinkster_root=dinkster_root,
        evidence_root=evidence_root,
        cache_root=cache_root,
    )
    assert cached == cache_root / relative
    assert cached.read_bytes() == b"platform bytes"

    source.write_bytes(b"source changed after caching")
    assert (
        fetch_platform_golden(
            baseline,
            key,
            dinkster_root=dinkster_root,
            evidence_root=evidence_root,
            cache_root=cache_root,
        )
        == cached
    )


def test_corrupt_cache_is_replaced_from_verified_evidence(tmp_path: Path) -> None:
    key = "darwin-py3.12.11-torch2.13.0"
    dinkster_root, baseline, evidence_root, relative = _fixture_paths(tmp_path, key)
    _write_evidence(evidence_root, relative, key, b"verified")
    cache_root = dinkster_root / ".golden-cache"
    cached = cache_root / relative
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"corrupt")

    assert (
        fetch_platform_golden(
            baseline,
            key,
            dinkster_root=dinkster_root,
            evidence_root=evidence_root,
            cache_root=cache_root,
        ).read_bytes()
        == b"verified"
    )


def test_corrupt_evidence_is_rejected(tmp_path: Path) -> None:
    key = "win32-py3.12.11-torch2.13.0+cu130"
    dinkster_root, baseline, evidence_root, relative = _fixture_paths(tmp_path, key)
    _write_evidence(evidence_root, relative, key, b"corrupt", sha256=_sha256(b"expected"))

    with pytest.raises(GoldenIntegrityError, match="evidence golden sha256 mismatch"):
        fetch_platform_golden(
            baseline,
            key,
            dinkster_root=dinkster_root,
            evidence_root=evidence_root,
            cache_root=dinkster_root / ".golden-cache",
        )


def test_missing_variant_is_distinct_from_unavailable_evidence(tmp_path: Path) -> None:
    key = "win32-py3.12.11-torch2.13.0+cpu"
    dinkster_root, baseline, evidence_root, relative = _fixture_paths(tmp_path, key)
    _write_evidence(evidence_root, relative, "another-platform", b"other")
    with pytest.raises(GoldenVariantNotFoundError, match="no evidence golden"):
        fetch_platform_golden(
            baseline,
            key,
            dinkster_root=dinkster_root,
            evidence_root=evidence_root,
            cache_root=dinkster_root / ".golden-cache",
        )

    with pytest.raises(GoldenUnavailableError, match="evidence manifest not found"):
        fetch_platform_golden(
            baseline,
            key,
            dinkster_root=dinkster_root,
            evidence_root=tmp_path / "missing",
            cache_root=dinkster_root / ".golden-cache",
        )


def test_offline_mode_refuses_platform_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "win32-py3.12.11-torch2.13.0+cpu"
    dinkster_root, baseline, evidence_root, _ = _fixture_paths(tmp_path, key)
    monkeypatch.setenv(GOLDEN_OFFLINE_ENV, "1")
    with pytest.raises(
        GoldenUnavailableError, match="comparison uses baseline only and is skipped"
    ):
        fetch_platform_golden(
            baseline,
            key,
            dinkster_root=dinkster_root,
            evidence_root=evidence_root,
            cache_root=dinkster_root / ".golden-cache",
        )


def test_offline_mode_rejects_unknown_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "win32-py3.12.11-torch2.13.0+cpu"
    dinkster_root, baseline, evidence_root, _ = _fixture_paths(tmp_path, key)
    monkeypatch.setenv(GOLDEN_OFFLINE_ENV, "true")
    with pytest.raises(ValueError, match=f"{GOLDEN_OFFLINE_ENV} must be unset or 1"):
        fetch_platform_golden(
            baseline,
            key,
            dinkster_root=dinkster_root,
            evidence_root=evidence_root,
            cache_root=dinkster_root / ".golden-cache",
        )
