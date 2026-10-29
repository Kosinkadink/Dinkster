"""Loader behavior for the CPU-pinned executed-reference golden gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import golden_files
import pytest
import torch
from golden_files import (
    assert_reference_schedule,
    assert_reference_tensor,
    assert_reference_values,
    cpu_identity,
    load_platform_golden,
    platform_digest,
    platform_golden_path,
    reference_validation_enabled,
)


@pytest.fixture(autouse=True)
def canonical_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The gate, not sidecar selection, is under test; on non-Linux hosts the
    # real platform_golden_path would demand a platform-suffixed sidecar.
    def identity(path: Path, **_kwargs: object) -> Path:
        return path

    def missing_evidence(_path: Path, key: str) -> Path:
        raise golden_files.GoldenVariantNotFoundError(f"no evidence golden for {key}")

    monkeypatch.setattr(golden_files, "platform_golden_path", identity)
    monkeypatch.setattr(golden_files, "fetch_platform_golden", missing_evidence)


def _write(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document))
    return path


def test_missing_platform_tuple_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(pytest.skip.Exception, match="no executed-reference golden minted"):
        platform_golden_path(tmp_path / "golden.json")


def test_present_platform_tuple_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    canonical = tmp_path / "golden.json"
    sidecar = _write(
        tmp_path / f"golden.{golden_files.runtime_key()}.json",
        {"cases": []},
    )
    assert platform_golden_path(canonical) == sidecar


def test_missing_platform_tuple_uses_canonical_portable_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    canonical = tmp_path / "golden.json"
    assert platform_golden_path(canonical, allow_portable_fallback=True) == canonical


def test_linux_uses_canonical_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    canonical = tmp_path / "golden.json"
    assert platform_golden_path(canonical) == canonical


def test_missing_explicit_runtime_tuple_skips_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(
        pytest.skip.Exception,
        match="no executed-reference golden minted for py3.13.15-torch2.13.0\\+cpu",
    ):
        platform_golden_path(
            tmp_path / "wan21_pipeline_goldens.json",
            key="py3.13.15-torch2.13.0+cpu",
        )


def test_cpu_identity_is_nonempty_and_stable() -> None:
    identity = cpu_identity()
    assert isinstance(identity, str)
    assert identity
    assert identity == cpu_identity()


def test_matching_cpu_loads(tmp_path: Path) -> None:
    fixture = _write(
        tmp_path / "golden.json",
        {"_meta": {"cpu": cpu_identity()}, "cases": [1, 2]},
    )
    assert load_platform_golden(fixture)["cases"] == [1, 2]


def test_mismatched_cpu_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _write(
        tmp_path / "golden.json",
        {"_meta": {"cpu": "Minted-Elsewhere CPU @ 9.99GHz"}, "cases": []},
    )
    monkeypatch.setattr(golden_files, "cpu_identity", lambda: "Live Host CPU")
    with pytest.raises(pytest.skip.Exception, match="Minted-Elsewhere CPU"):
        load_platform_golden(fixture)


def test_portable_fallback_ignores_minted_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write(
        tmp_path / "golden.json",
        {"_meta": {"cpu": "Minted-Elsewhere CPU @ 9.99GHz"}, "cases": [1, 2]},
    )
    monkeypatch.setattr(golden_files, "cpu_identity", lambda: "Live Host CPU")
    assert load_platform_golden(fixture, allow_portable_fallback=True)["cases"] == [1, 2]


def test_missing_cpu_key_enforces(tmp_path: Path) -> None:
    fixture = _write(
        tmp_path / "golden.json",
        {"_meta": {"python": "Python 3.12.3"}, "cases": [3]},
    )
    assert load_platform_golden(fixture)["cases"] == [3]


def test_reference_cpu_used_when_meta_lacks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write(
        tmp_path / "golden.json",
        {
            "_meta": {"python": "Python 3.12.3"},
            "reference": {"cpu": "Minted-Elsewhere CPU @ 9.99GHz"},
            "cases": [],
        },
    )
    monkeypatch.setattr(golden_files, "cpu_identity", lambda: "Live Host CPU")
    with pytest.raises(pytest.skip.Exception):
        load_platform_golden(fixture)


def test_platform_digest_selects_exact_cpu_and_runtime_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(golden_files, "cpu_identity", lambda: "Intel(R) Core(TM) Ultra 7 270K Plus")
    assert platform_digest("sd15_cfg_one") == (
        "c6ba833bdd54fd01e09749cf35385c91213c2611eb0c69c128eb0e289872a6eb"
    )

    provenance = golden_files.runtime_provenance()
    monkeypatch.setattr(golden_files, "cpu_identity", lambda: "AMD Ryzen 9 5950X 16-Core Processor")
    monkeypatch.setattr(
        golden_files,
        "runtime_provenance",
        lambda: {**provenance, "python": "Python 3.12.13"},
    )
    assert platform_digest("sd15_cfg_one") == (
        "cf162ee434116de5e1f4a9c7b399d1cabb8629bb4f88d1addb61c80263a681e1"
    )
    assert platform_digest("sd15_inpaint") == (
        "8b3530802ec03a272a16537c7c10c74131c6c179ebe20a7e4409cec1e470ba0a"
    )

    monkeypatch.setattr(golden_files, "cpu_identity", lambda: "Unminted CPU")
    with pytest.raises(pytest.skip.Exception, match="no wiring digest minted"):
        platform_digest("sd15_cfg_one")


def test_reference_validation_defaults_to_portable_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(golden_files.REFERENCE_VALIDATION_ENV, raising=False)
    assert not reference_validation_enabled()
    assert_reference_tensor(torch.tensor([1.0]), torch.tensor([2.0]))
    assert_reference_schedule((2.0, 1.0, 0.0), (3.0, 2.0, 0.0))
    assert_reference_values((2.0, 1.0), (3.0, 2.0))


def test_reference_validation_requires_documented_value_and_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(golden_files.REFERENCE_VALIDATION_ENV, "true")
    with pytest.raises(RuntimeError, match="must be unset or 1"):
        reference_validation_enabled()
    monkeypatch.setenv(golden_files.REFERENCE_VALIDATION_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA-capable runtime"):
        reference_validation_enabled()


def test_reference_validation_enforces_exact_values_on_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(golden_files.REFERENCE_VALIDATION_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert reference_validation_enabled()
    with pytest.raises(AssertionError):
        assert_reference_tensor(torch.tensor([1.0]), torch.tensor([2.0]))
    with pytest.raises(AssertionError):
        assert_reference_schedule((2.0, 1.0, 0.0), (3.0, 1.5, 0.0))
    with pytest.raises(AssertionError):
        assert_reference_values((2.0, 1.0), (3.0, 1.5))


def test_portable_reference_checks_reject_invalid_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(golden_files.REFERENCE_VALIDATION_ENV, raising=False)
    with pytest.raises(AssertionError):
        assert_reference_tensor(torch.tensor([float("nan")]), torch.tensor([1.0]))
    with pytest.raises(AssertionError):
        assert_reference_tensor(torch.zeros(2), torch.tensor([-1.0, 1.0]))
    with pytest.raises(AssertionError):
        assert_reference_tensor(torch.tensor([-1e30, 1e30]), torch.tensor([-1.0, 1.0]))
    with pytest.raises(AssertionError):
        assert_reference_schedule((2.0, 2.5, 0.0), (3.0, 1.5, 0.0))
    with pytest.raises(AssertionError):
        assert_reference_schedule((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))
    with pytest.raises(AssertionError):
        assert_reference_values((1e30,), (1.0,))
