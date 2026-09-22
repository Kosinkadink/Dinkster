"""Engine identity and pack state share one atomic generation selection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_registry import InstallError, Lockfile

from dinkster.installer import EngineEnvironment, Installer


def environment(marker: str) -> EngineEnvironment:
    return EngineEnvironment(marker * 64, "c" * 64, "d" * 40, "linux-cu128", ("e" * 64,))


def test_engine_change_creates_generation_with_identical_pack_lock(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")
    first, _ = installer.apply(Lockfile(), environment=environment("a"))
    second, _ = installer.apply(Lockfile(), environment=environment("b"))
    assert (first, second) == (1, 2)
    assert installer.lockfile_of(first) == installer.lockfile_of(second) == Lockfile()
    assert installer.environment_of(first) == environment("a")
    assert installer.environment_of(second) == environment("b")
    assert installer.current_number() == second
    assert installer.apply(Lockfile(), environment=environment("b"))[0] == second


def test_pack_apply_inherits_engine_and_explicit_none_clears_it(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")
    first, _ = installer.apply(Lockfile(), environment=environment("a"))
    assert installer.apply(Lockfile())[0] == first
    second, _ = installer.apply(Lockfile(), environment=None)
    assert second == first + 1
    assert installer.environment_of(second) is None
    assert installer.environment_of(installer.rollback()) == environment("a")


def test_rollback_ignores_never_activated_generation(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")
    first, _ = installer.apply(Lockfile(), environment=environment("a"))
    staged, _ = installer.apply(Lockfile(), environment=environment("b"), activate=False)
    assert installer.current_number() == first
    third, _ = installer.apply(Lockfile(), environment=environment("f"))
    assert (first, staged, third) == (1, 2, 3)
    restored = installer.rollback()
    assert restored == 4
    assert installer.environment_of(restored) == environment("a")
    assert installer.environment_of(staged) == environment("b")


def test_activation_rejects_stale_staging_and_leaves_current(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")
    installer.apply(Lockfile(), environment=environment("a"))
    staged, _ = installer.apply(Lockfile(), environment=environment("b"), activate=False)
    third, _ = installer.apply(Lockfile(), environment=environment("f"))
    with pytest.raises(InstallError, match="stale"):
        installer.activate(staged)
    assert installer.current_number() == third


def test_first_staged_generation_activation_and_validation_failure(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")
    staged, _ = installer.apply(Lockfile(), environment=environment("b"), activate=False)
    assert installer.current_number() is None

    def refuse() -> None:
        raise InstallError("missing interpreter")

    with pytest.raises(InstallError, match="missing interpreter"):
        installer.activate(staged, validate=refuse)
    assert installer.current_number() is None
    installer.activate(staged)
    assert installer.current_number() == staged
    assert installer.environment_of(staged) == environment("b")
    with pytest.raises(InstallError, match="no earlier"):
        installer.rollback()


def test_staging_failure_does_not_record_or_activate(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")
    first, _ = installer.apply(Lockfile(), environment=environment("a"))
    record = (installer.root / "generations/1.json").read_bytes()

    def fail_download() -> EngineEnvironment:
        raise InstallError("checksum mismatch")

    with pytest.raises(InstallError, match="checksum mismatch"):
        installer.apply(Lockfile(), stage_environment=fail_download)
    assert installer.current_number() == first
    assert installer.generation_numbers() == (first,)
    assert (installer.root / "generations/1.json").read_bytes() == record


def test_stage_callback_selects_environment_before_activation(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")

    def stage() -> EngineEnvironment:
        assert installer.current_number() is None
        return environment("b")

    number, _ = installer.apply(Lockfile(), stage_environment=stage)
    assert installer.environment_of(number) == environment("b")
    assert installer.current_number() == number


def test_old_pack_record_remains_readable_and_rolls_back(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")
    (installer.root / "generations/1.json").write_text(Lockfile().record_json())
    (installer.root / "current").write_text("1\n")
    assert installer.environment_of(1) is None
    installer.apply(Lockfile(), environment=environment("a"))
    restored = installer.rollback()
    assert installer.environment_of(restored) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_id", "../outside"),
        ("manifest_sha256", "A" * 64),
        ("commit", "d" * 39),
        ("cell", "../../data"),
        ("objects", ("e" * 64, "e" * 64)),
        ("objects", ("../data",)),
    ],
)
def test_engine_identity_rejects_corrupt_content_paths(field: str, value: object) -> None:
    with pytest.raises(InstallError):
        replace(environment("a"), **{field: value})


def test_generation_corruption_is_not_treated_as_pack_only(tmp_path: Path) -> None:
    installer = Installer(tmp_path / "install")
    number, _ = installer.apply(Lockfile(), environment=environment("a"))
    path = installer.root / f"generations/{number}.json"
    record = json.loads(path.read_text())
    record["engine"]["objects"] = ["outside"]
    path.write_text(json.dumps(record))
    with pytest.raises(InstallError, match="SHA-256"):
        installer.environment_of(number)
    with pytest.raises(InstallError, match="SHA-256"):
        installer.apply(Lockfile())
