"""The host installer: an install root on disk (DESIGN M8).

What this proves: applying a lockfile stages artifacts content-addressed
and provisions venvs BEFORE the one-file pointer swap (a staging failure
leaves the previous generation active and untouched), rollback re-activates
prior content as a new generation, gc removes only unreferenced content,
and the current generation surfaces as compose-ready PackSpecs whose
manifests live in the immutable store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
from collections.abc import Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from dinkster_registry import (
    InstallError,
    LockedPack,
    Lockfile,
    PlanRecord,
    SnapshotRecord,
    build_artifact,
)
from dinkster_workers import DoctorReport, Finding, PackManifest
from dinkster_workers.provision import ProvisionError

import dinkster.installer as installer_module
from dinkster.installer import (
    Installer,
    VenvSpec,
    http_registry_fetcher,
    lock_git_pack,
    lock_local_pack,
)

MANIFEST_TEMPLATE = '[pack]\nname = "{name}"\n\n[pack.entry]\nnodes = "{name}_nodes:NODES"\n'


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def venv_metadata_glob(root: Path, pack: str, filename: str) -> list[Path]:
    scripts = "Scripts" if os.name == "nt" else "bin"
    return list((root / "venvs").glob(f"*/{pack}/{scripts}/{filename}"))


@pytest.fixture(autouse=True)
def _no_host_runtime_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every installer test hermetic: the default runtime probe
    shells out to the host's vendor status tool (nvidia-smi), which must
    never run under tests. Tests that WANT runtime facts inject their
    own probe or re-monkeypatch this name."""
    monkeypatch.setattr(installer_module, "detect_runtime", lambda _accelerator: ())
    monkeypatch.setattr(
        installer_module,
        "diagnose",
        lambda manifest, **_kwargs: DoctorReport(
            pack_name=Path(manifest).parent.name,
            manifest_path=str(manifest),
            findings=(),
        ),
    )


def write_pack(directory: Path, name: str, body: str = "NODES = []\n") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dinkster-pack.toml").write_text(MANIFEST_TEMPLATE.format(name=name))
    (directory / f"{name}_nodes.py").write_text(body)
    return directory


def fake_provision(manifest: PackManifest, venv_root: Path, spec: VenvSpec | None = None) -> Path:
    """Stand-in for ensure_pack_venv: same layout, no uv, same
    reuse-if-present behavior. Records the spec it was handed so
    snapshot-restore tests can assert exactness/constraints."""
    python = venv_python(venv_root / manifest.name)
    if python.exists():
        return python  # mirror ensure_pack_venv: staged venvs are reused
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!fake\n")
    if spec is not None and spec.exact is not None:
        content = "\n".join(spec.exact)
    elif spec is not None and spec.constraints:
        content = "RANGES-CONSTRAINED:" + ",".join(spec.constraints)
    else:
        content = "RANGES"
    (python.parent / "pins.txt").write_text(content)
    return python


def fake_freeze(python: Path) -> tuple[str, ...]:
    """Stand-in for freeze_venv: reads what fake_provision recorded, so a
    venv 'contains' exactly what it was provisioned with."""
    recorded = (python.parent / "pins.txt").read_text()
    if recorded.startswith("RANGES"):
        return ("numpy==1.0", "torch==2.0")  # what 'range resolution' found
    return tuple(recorded.splitlines()) if recorded else ()


def fake_group_provision(
    manifests: Sequence[PackManifest],
    group_name: str,
    venv_root: Path,
    spec: VenvSpec | None = None,
) -> Path:
    python = venv_python(venv_root / group_name)
    if python.exists():
        return python
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!fake-group\n")
    pins = spec.exact if spec is not None and spec.exact is not None else ("shared==1.0",)
    (python.parent / "pins.txt").write_text("\n".join(pins))
    (python.parent / "members.txt").write_text(
        "\n".join(sorted(manifest.name for manifest in manifests))
    )
    return python


def make_installer(tmp_path: Path) -> Installer:
    return Installer(
        tmp_path / "root",
        provision=fake_provision,
        group_provision=fake_group_provision,
        freeze=fake_freeze,
    )


def local_lockfile(installer: Installer, *packs: Path) -> Lockfile:
    return Lockfile.of([lock_local_pack(pack, installer.artifacts_dir)[0] for pack in packs])


def write_hosting(installer: Installer, body: str) -> None:
    (installer.root / "hosting.toml").write_text(body)


def test_apply_activates_and_is_idempotent(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    assert installer.current_lockfile() is None
    assert installer.packs_for_serving() == []
    target = local_lockfile(installer, write_pack(tmp_path / "demo", "demo"))
    number, steps = installer.apply(target)
    assert number == 1
    assert [step.action for step in steps.steps] == ["add"]
    assert installer.current_number() == 1
    assert installer.current_lockfile() == target
    # identical content: no new generation, empty plan
    again, replan = installer.apply(target)
    assert again == 1
    assert replan.empty
    assert installer.generation_numbers() == (1,)


def test_packs_for_serving_reads_the_immutable_store(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    installer.apply(local_lockfile(installer, pack_dir))
    specs = installer.packs_for_serving()
    assert len(specs) == 1
    manifest = Path(specs[0].manifest)
    assert manifest.is_file()
    assert installer.root / "store" in manifest.parents  # not the source dir
    assert specs[0].python is not None
    assert Path(specs[0].python).read_text() == "#!fake\n"
    # editing the source pack after install changes nothing served
    (pack_dir / "demo_nodes.py").write_text("NODES = [1]\n")
    assert (manifest.parent / "demo_nodes.py").read_text() == "NODES = []\n"
    # Lockfile provenance reaches the spec's packs-table entry: the digest
    # is the real pin; a local install has no release version, so none is
    # surfaced (the 0.0.0 sentinel never masquerades as a pin).
    entry = installer.current_lockfile().packs[0]  # type: ignore[union-attr]
    info = specs[0].packs["demo"]  # type: ignore[index]
    assert info.artifact_digest == entry.artifact_digest
    assert info.source.startswith("local:")
    assert info.publisher == "local"
    assert info.version == ""
    # A published (registry) entry surfaces its real release version.
    registry_entry = replace(entry, version="1.2.3", publisher="alice", source="registry")
    installer.apply(Lockfile.of([registry_entry]))
    served = installer.packs_for_serving()[0].packs["demo"]  # type: ignore[index]
    assert served.version == "1.2.3"
    assert served.publisher == "alice"
    assert served.source == "registry"
    assert served.artifact_digest == entry.artifact_digest


def test_packs_for_serving_carries_generation_worker_groups(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    alpha = write_pack(tmp_path / "alpha", "alpha")
    beta = write_pack(tmp_path / "beta", "beta")
    installer.apply(
        local_lockfile(installer, alpha, beta),
        hosting_groups=(("models", ("alpha", "beta")),),
    )
    specs = installer.packs_for_serving()
    assert {spec.worker_group for spec in specs} == {"models"}
    manifests = tuple(Path(spec.manifest) for spec in specs)
    assert all(spec.group_manifests == manifests for spec in specs)


def test_upgrade_creates_generation_and_rollback_restores(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    installer.apply(local_lockfile(installer, pack_dir))
    (pack_dir / "demo_nodes.py").write_text("NODES = [2]\n")
    second = local_lockfile(installer, pack_dir)
    number, steps = installer.apply(second)
    assert number == 2
    assert [step.action for step in steps.steps] == ["reinstall"]  # local: same version
    served = Path(installer.packs_for_serving()[0].manifest).parent
    assert (served / "demo_nodes.py").read_text() == "NODES = [2]\n"
    # rollback: generation 1's content, as NEW generation 3
    assert installer.rollback() == 3
    assert installer.generation_numbers() == (1, 2, 3)
    assert installer.current_lockfile() == installer.lockfile_of(1)
    served = Path(installer.packs_for_serving()[0].manifest).parent
    assert (served / "demo_nodes.py").read_text() == "NODES = []\n"


def test_rollback_requires_history(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    with pytest.raises(InstallError, match="nothing to roll back"):
        installer.rollback()
    installer.apply(local_lockfile(installer, write_pack(tmp_path / "demo", "demo")))
    with pytest.raises(InstallError, match="no earlier generation"):
        installer.rollback()


def test_staging_failure_leaves_current_generation_active(tmp_path: Path) -> None:
    """Activation is all-or-nothing: a failing pack aborts BEFORE the
    pointer swap, and the running installation is untouched."""
    installer = make_installer(tmp_path)
    first = local_lockfile(installer, write_pack(tmp_path / "demo", "demo"))
    installer.apply(first)

    def failing_provision(
        manifest: PackManifest, venv_root: Path, spec: VenvSpec | None = None
    ) -> Path:
        if manifest.name == "other":
            raise RuntimeError("dependency resolution failed")
        return fake_provision(manifest, venv_root, spec)

    breaking = Installer(installer.root, provision=failing_provision)
    target = local_lockfile(
        breaking,
        write_pack(tmp_path / "demo", "demo"),
        write_pack(tmp_path / "other", "other"),
    )
    with pytest.raises(RuntimeError, match="dependency resolution"):
        breaking.apply(target)
    assert breaking.current_number() == 1
    assert breaking.current_lockfile() == first
    assert breaking.generation_numbers() == (1,)


def test_absent_hosting_policy_preserves_per_pack_layout_and_identity(
    tmp_path: Path,
) -> None:
    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    before = target.record_json()
    installer.apply(target)
    assert target.record_json() == before
    for entry in target.packs:
        digest = entry.artifact_digest.partition(":")[2]
        assert venv_python(installer.root / "venvs" / digest / entry.pack).is_file()
    assert installer.snapshot().venv_groups == ()


def test_group_staging_uses_one_aggregate_venv_and_serves_every_member_from_it(
    tmp_path: Path,
) -> None:
    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    write_hosting(installer, '[venv-groups]\nmodels = ["beta", "alpha"]\n')
    installer.apply(target)
    payload = "".join(
        f"{entry.pack}\0{entry.artifact_digest}\n"
        for entry in sorted(target.packs, key=lambda item: item.pack)
    )
    aggregate = hashlib.sha256(payload.encode()).hexdigest()
    group = installer.root / "venvs" / aggregate / "models"
    assert (venv_python(group).parent / "members.txt").read_text() == "alpha\nbeta"
    assert not any(
        (installer.root / "venvs" / entry.artifact_digest.partition(":")[2]).exists()
        for entry in target.packs
    )
    specs = installer.packs_for_serving()
    assert {spec.python for spec in specs} == {str(venv_python(group))}
    installed = installer.current_lockfile()
    assert installed is not None
    assert target.record_json() == installed.record_json()


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ('[venv-groups]\nsolo = ["alpha"]\n', "at least 2"),
        ('[venv-groups]\nbad = ["alpha", "ghost"]\n', "not in the lockfile"),
        (
            '[venv-groups]\none = ["alpha", "beta"]\ntwo = ["beta", "gamma"]\n',
            "belongs to both",
        ),
    ],
)
def test_invalid_hosting_groups_refuse_before_activation(
    tmp_path: Path, policy: str, message: str
) -> None:
    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
        write_pack(tmp_path / "gamma", "gamma"),
    )
    write_hosting(installer, policy)
    with pytest.raises(InstallError, match=message):
        installer.apply(target)
    assert installer.current_number() is None


def test_group_resolution_conflict_leaves_current_generation_active(
    tmp_path: Path,
) -> None:
    installer = make_installer(tmp_path)
    first = local_lockfile(installer, write_pack(tmp_path / "alpha", "alpha"))
    installer.apply(first)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    write_hosting(installer, '[venv-groups]\nmodels = ["alpha", "beta"]\n')

    def conflict(
        _manifests: Sequence[PackManifest],
        _name: str,
        _root: Path,
        _spec: VenvSpec | None,
    ) -> Path:
        raise ProvisionError("requirements are unsatisfiable")

    failing = Installer(
        installer.root,
        provision=fake_provision,
        group_provision=conflict,
        freeze=fake_freeze,
    )
    with pytest.raises(InstallError, match="requirements are unsatisfiable"):
        failing.apply(target)
    assert failing.current_lockfile() == first
    assert failing.generation_numbers() == (1,)


def test_install_doctor_refusal_leaves_generation_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = make_installer(tmp_path)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    first = local_lockfile(installer, pack_dir)
    installer.apply(first)
    (pack_dir / "demo_nodes.py").write_text("BROKEN = True\n")
    target = local_lockfile(installer, pack_dir)
    failing = DoctorReport(
        pack_name="demo",
        manifest_path="dinkster-pack.toml",
        findings=(Finding("error", "entry.unresolvable", "broken import"),),
    )
    monkeypatch.setattr(installer_module, "diagnose", lambda *_args, **_kwargs: failing)

    with pytest.raises(InstallError, match="entry.unresolvable"):
        installer.apply(target, venvs=False)
    assert installer.current_number() == 1
    assert installer.current_lockfile() == first


def test_install_doctor_override_warns_and_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installer = make_installer(tmp_path)
    target = local_lockfile(installer, write_pack(tmp_path / "demo", "demo"))
    failing = DoctorReport(
        pack_name="demo",
        manifest_path="dinkster-pack.toml",
        findings=(Finding("error", "entry.unresolvable", "broken import"),),
    )
    monkeypatch.setattr(installer_module, "diagnose", lambda *_args, **_kwargs: failing)

    number, _ = installer.apply(target, venvs=False, allow_doctor_findings=True)
    assert number == 1
    assert "WARNING: allowing doctor findings" in capsys.readouterr().out


def test_git_sources_are_gated_and_registry_sources_are_exempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = make_installer(tmp_path)
    local = local_lockfile(installer, write_pack(tmp_path / "demo", "demo")).packs[0]
    seen: list[Path | str | None] = []

    def failing(_manifest: Path, *, interpreter: Path | str | None = None) -> DoctorReport:
        seen.append(interpreter)
        return DoctorReport(
            pack_name="demo",
            manifest_path="dinkster-pack.toml",
            findings=(Finding("error", "doctor.test", "refused"),),
        )

    monkeypatch.setattr(installer_module, "diagnose", failing)
    git_entry = replace(local, source="git:https://example.invalid/demo@abc")
    with pytest.raises(InstallError, match="doctor.test"):
        installer.apply(Lockfile.of([git_entry]), venvs=False)
    assert seen == [sys.executable]

    seen.clear()
    registry_entry = replace(local, source="registry:public", publisher="alice")
    number, _ = installer.apply(Lockfile.of([registry_entry]), venvs=False)
    assert number == 1
    assert seen == [sys.executable]


def test_doctor_gate_uses_staged_or_fallback_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path | str | None] = []

    def healthy(manifest: Path, *, interpreter: Path | str | None = None) -> DoctorReport:
        seen.append(interpreter)
        return DoctorReport(Path(manifest).parent.name, str(manifest), ())

    monkeypatch.setattr(installer_module, "diagnose", healthy)
    with_venv = make_installer(tmp_path / "with")
    target = local_lockfile(with_venv, write_pack(tmp_path / "demo", "demo"))
    with_venv.apply(target)
    assert seen[-1] == with_venv.venv_python(target.packs[0])

    without_venv = make_installer(tmp_path / "without")
    target = local_lockfile(without_venv, write_pack(tmp_path / "other", "other"))
    without_venv.apply(target, venvs=False)
    assert seen[-1] == sys.executable


def test_group_doctor_probes_each_member_with_the_shared_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, Path | str | None]] = []

    def healthy(manifest: Path, *, interpreter: Path | str | None = None) -> DoctorReport:
        pack_name = installer_module.load_manifest(manifest).name
        seen.append((pack_name, interpreter))
        return DoctorReport(pack_name, str(manifest), ())

    monkeypatch.setattr(installer_module, "diagnose", healthy)
    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    write_hosting(installer, '[venv-groups]\nmodels = ["alpha", "beta"]\n')
    installer.apply(target)
    assert [name for name, _ in seen] == ["alpha", "beta"]
    assert len({str(interpreter) for _, interpreter in seen}) == 1


def test_apply_requires_staged_or_provided_artifacts(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(pack_dir, tmp_path / "elsewhere")
    target = Lockfile.of([entry])
    with pytest.raises(InstallError, match="no artifact was provided"):
        installer.apply(target)
    assert installer.current_number() is None
    # explicit artifact mapping works from any location
    number, _ = installer.apply(target, {entry.artifact_digest: archive})
    assert number == 1


def test_lock_local_pack_digest_is_identity(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(pack_dir, installer.artifacts_dir)
    assert entry.pack == "demo"
    assert entry.publisher == "local"
    assert entry.version == "0.0.0"
    assert entry.claims == ("demo",)
    assert entry.source == f"local:{pack_dir.resolve()}"
    assert archive.is_file()
    # the archive is content-addressed next to the digest
    assert archive.stem in entry.artifact_digest
    # re-archiving identical content is a no-op identity-wise
    entry_again, archive_again = lock_local_pack(pack_dir, installer.artifacts_dir)
    assert entry_again.artifact_digest == entry.artifact_digest
    assert archive_again == archive
    # the digest matches a from-scratch build of the same tree
    rebuilt = build_artifact(pack_dir, tmp_path / "rebuild.zip")
    assert rebuilt == entry.artifact_digest


def test_gc_removes_only_unreferenced_content(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    first = local_lockfile(installer, pack_dir)
    installer.apply(first)
    (pack_dir / "demo_nodes.py").write_text("NODES = [2]\n")
    installer.apply(local_lockfile(installer, pack_dir))
    # both generations exist, so both digests stay referenced
    assert installer.gc() == ()
    # orphan content: an artifact never locked by any generation
    orphan_dir = installer.root / "store" / "feedface"
    orphan_dir.mkdir()
    (installer.artifacts_dir / "feedface.zip").write_bytes(b"zzz")
    removed = installer.gc()
    assert "store/feedface" in removed
    assert "artifacts/feedface.zip" in removed
    assert installer.packs_for_serving()  # current generation intact


def test_corrupt_pointer_and_missing_generation_fail_loudly(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    with pytest.raises(InstallError, match="no generation 5"):
        installer.lockfile_of(5)
    (installer.root / "current").write_text("banana\n")
    with pytest.raises(InstallError, match="corrupt current-generation pointer"):
        installer.current_number()


# ---------------------------------------------------------------------------
# snapshots: lockfile + per-venv freezes as one restorable record
# ---------------------------------------------------------------------------


def test_snapshot_captures_lockfile_and_per_venv_freezes(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "demo", "demo"),
        write_pack(tmp_path / "other", "other"),
    )
    installer.apply(target)
    record = installer.snapshot()
    assert record.lockfile.record_digest() == target.record_digest()
    # every staged venv frozen, per pack (fake range resolution result)
    assert record.pins_for("demo") == ("numpy==1.0", "torch==2.0")
    assert record.pins_for("other") == ("numpy==1.0", "torch==2.0")
    assert record.python and record.platform and record.dinkster  # host scope recorded
    # decode-what-you-wrote: the file round-trips to the same record
    assert SnapshotRecord.from_record_json(record.record_json()) == record


def test_group_snapshot_freezes_once_records_identical_pins_and_provenance(
    tmp_path: Path,
) -> None:
    freezes: list[Path] = []

    def counting_freeze(python: Path) -> tuple[str, ...]:
        freezes.append(python)
        return ("shared==1.0",)

    installer = Installer(
        tmp_path / "root",
        provision=fake_provision,
        group_provision=fake_group_provision,
        freeze=counting_freeze,
    )
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    write_hosting(installer, '[venv-groups]\nmodels = ["beta", "alpha"]\n')
    installer.apply(target)
    record = installer.snapshot()
    assert len(freezes) == 1
    assert record.pins_for("alpha") == record.pins_for("beta") == ("shared==1.0",)
    assert record.venv_groups == (("models", ("alpha", "beta")),)
    document = json.loads(record.record_json())
    assert document["venvGroups"] == {"models": ["alpha", "beta"]}
    del document["venvGroups"]
    assert SnapshotRecord.from_record_json(json.dumps(document)).venv_groups == ()


def test_removing_group_policy_restages_per_pack_and_gc_keeps_live_collects_dead_groups(
    tmp_path: Path,
) -> None:
    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    identity = target.record_json()
    write_hosting(installer, '[venv-groups]\nmodels = ["alpha", "beta"]\n')
    installer.apply(target)
    group_python = installer.venv_python(target.packs[0])
    assert group_python is not None
    aggregate_dir = group_python.parents[2]
    assert installer.gc() == ()

    (installer.root / "hosting.toml").unlink()
    installer.apply(target)
    assert target.record_json() == identity
    assert installer.venv_python(target.packs[0]) != group_python
    dead_group = aggregate_dir / "dead-group"
    dead_group.mkdir()
    removed = installer.gc()
    assert f"venvs/{aggregate_dir.name}/dead-group" in removed
    assert aggregate_dir.exists()  # generation 1 still references the original group
    assert not dead_group.exists()


def test_restore_replays_only_an_identical_current_group_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster.manager import _matching_restore_groups, _validated_plan_groups

    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    recorded = (("models", ("alpha", "beta")),)
    write_hosting(installer, '[venv-groups]\nmodels = ["alpha", "beta"]\n')
    assert _matching_restore_groups(installer, target, recorded) == recorded
    installer.apply(target, hosting_groups=recorded)
    assert installer.current_groups() == recorded

    write_hosting(installer, '[venv-groups]\nrenamed = ["alpha", "beta"]\n')
    fallback = _matching_restore_groups(installer, target, recorded)
    assert fallback == ()
    with pytest.raises(InstallError, match="plan is stale"):
        _validated_plan_groups(installer, target, recorded)
    installer.apply(target, hosting_groups=fallback)
    assert installer.current_groups() == ()
    assert len({installer.venv_python(entry) for entry in target.packs}) == 2
    assert "falling back to per-pack provisioning" in capsys.readouterr().out


def test_pack_identity_is_byte_identical_solo_and_grouped(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    installer.apply(target)
    solo = [
        (Path(spec.manifest).read_bytes(), spec.packs) for spec in installer.packs_for_serving()
    ]
    write_hosting(installer, '[venv-groups]\nmodels = ["alpha", "beta"]\n')
    installer.apply(target)
    grouped = [
        (Path(spec.manifest).read_bytes(), spec.packs) for spec in installer.packs_for_serving()
    ]
    assert grouped == solo
    assert installer.current_lockfile() == target


def test_manager_applies_a_policy_only_topology_change(tmp_path: Path) -> None:
    from dinkster.manager import _finish

    installer = make_installer(tmp_path)
    target = local_lockfile(
        installer,
        write_pack(tmp_path / "alpha", "alpha"),
        write_pack(tmp_path / "beta", "beta"),
    )
    installer.apply(target)
    write_hosting(installer, '[venv-groups]\nmodels = ["alpha", "beta"]\n')
    args = argparse.Namespace(
        no_venv=False,
        plan_file=None,
        allow_doctor_findings=False,
        yes=True,
    )
    _finish(args, installer, target)
    assert installer.current_number() == 2
    assert installer.current_groups() == (("models", ("alpha", "beta")),)


def test_snapshot_records_unstaged_venvs_as_unpinned(tmp_path: Path) -> None:
    """--no-venv installs have no venv to freeze; the snapshot says so
    honestly (absent from venvs) instead of inventing pins."""
    installer = make_installer(tmp_path)
    installer.apply(local_lockfile(installer, write_pack(tmp_path / "demo", "demo")), venvs=False)
    record = installer.snapshot()
    assert record.pins_for("demo") is None


def test_snapshot_requires_an_installation(tmp_path: Path) -> None:
    with pytest.raises(InstallError, match="nothing to snapshot"):
        make_installer(tmp_path).snapshot()


def test_snapshot_records_accelerator_scope(tmp_path: Path) -> None:
    """The snapshot's host scope includes the accelerator the venvs were
    provisioned for - what cross-scope restore compares against."""
    installer = Installer(
        tmp_path / "root", provision=fake_provision, freeze=fake_freeze, accelerator="cuda"
    )
    installer.apply(local_lockfile(installer, write_pack(tmp_path / "demo", "demo")))
    record = installer.snapshot()
    assert record.accelerator == "cuda"
    assert SnapshotRecord.from_record_json(record.record_json()).accelerator == "cuda"


def test_snapshot_records_runtime_facts_from_injected_probe(tmp_path: Path) -> None:
    """Snapshot scope includes advisory runtime facts via the injectable
    probe - deterministic here, the vendor status tool in production.
    Hosts where nothing is detectable record nothing (no fake facts)."""
    installer = Installer(
        tmp_path / "root",
        provision=fake_provision,
        freeze=fake_freeze,
        accelerator="cuda",
        runtime_probe=lambda accelerator: (("cuda", "12.4"), ("driver", "550.54.14")),
    )
    installer.apply(local_lockfile(installer, write_pack(tmp_path / "demo", "demo")))
    record = installer.snapshot()
    assert record.runtime == (("cuda", "12.4"), ("driver", "550.54.14"))
    assert SnapshotRecord.from_record_json(record.record_json()).runtime == record.runtime
    # default probe is faked to () by the autouse fixture: not recorded
    installer_bare = Installer(
        tmp_path / "bare", provision=fake_provision, freeze=fake_freeze, accelerator="cuda"
    )
    installer_bare.apply(local_lockfile(installer_bare, write_pack(tmp_path / "demo2", "demo2")))
    assert installer_bare.snapshot().runtime == ()


def test_restore_provisions_missing_venvs_from_snapshot_pins(tmp_path: Path) -> None:
    """The reproduction property: applying with a snapshot's venv_pins
    hands the exact recorded dist list to provisioning, not the
    manifest's ranges."""
    installer = make_installer(tmp_path)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(pack_dir, installer.artifacts_dir)
    target = Lockfile.of([entry])
    pins = ("numpy==1.26.4", "torch==2.5.1")
    installer.apply(
        target, {entry.artifact_digest: archive}, venv_specs={"demo": VenvSpec(exact=pins)}
    )
    python = installer.venv_python(entry)
    assert python is not None
    assert fake_freeze(python) == pins  # provisioned from pins, not ranges


def test_snapshot_hashes_annotates_freezes_through_the_injected_hasher(tmp_path: Path) -> None:
    """snapshot(hashes=True) routes each venv's freeze through the pin
    hasher (the one index-touching part of capture, injectable); the
    default capture stays offline and bare. The annotated record still
    round-trips through the snapshot format (validated at decode)."""

    def fake_hasher(pins: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(f"{pin} --hash=sha256:aaaa" for pin in pins))

    installer = Installer(
        tmp_path / "root", provision=fake_provision, freeze=fake_freeze, hasher=fake_hasher
    )
    installer.apply(local_lockfile(installer, write_pack(tmp_path / "demo", "demo")))
    record = installer.snapshot(hashes=True)
    assert record.pins_for("demo") == (
        "numpy==1.0 --hash=sha256:aaaa",
        "torch==2.0 --hash=sha256:aaaa",
    )
    assert SnapshotRecord.from_record_json(record.record_json()) == record
    assert installer.snapshot().pins_for("demo") == ("numpy==1.0", "torch==2.0")


def test_venv_drift_ignores_hash_annotations(tmp_path: Path) -> None:
    """Drift is version identity: a freeze reports name==version, so a
    hash-annotated snapshot pin matching the staged version is no drift -
    and a real version difference still is."""
    installer = make_installer(tmp_path)
    target = local_lockfile(installer, write_pack(tmp_path / "demo", "demo"))
    installer.apply(target)  # range-resolved: numpy==1.0, torch==2.0
    entry = target.packs[0]
    annotated = ("numpy==1.0 --hash=sha256:aaaa", "torch==2.0")
    assert installer.venv_drift(entry, annotated) == ()
    drift = installer.venv_drift(entry, ("numpy==1.0 --hash=sha256:aaaa", "torch==9.9"))
    assert drift == ("torch==2.0", "torch==9.9")


def test_venv_drift_reports_but_never_rebuilds(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    target = local_lockfile(installer, pack_dir)
    installer.apply(target)  # range-resolved: numpy==1.0, torch==2.0
    entry = target.packs[0]
    assert installer.venv_drift(entry, ("numpy==1.0", "torch==2.0")) == ()
    drift = installer.venv_drift(entry, ("numpy==1.0", "torch==9.9"))
    assert drift == ("torch==2.0", "torch==9.9")  # symmetric difference
    # re-applying with pins does NOT rebuild the shared staged venv
    installer.apply(Lockfile(), venvs=False)  # move away
    installer.apply(target, venv_specs={"demo": VenvSpec(exact=("numpy==1.0", "torch==9.9"))})
    python = installer.venv_python(entry)
    assert python is not None
    assert fake_freeze(python) == ("numpy==1.0", "torch==2.0")  # untouched


# ---------------------------------------------------------------------------
# dinkster-pack CLI (the shell surface over the same transaction)
# ---------------------------------------------------------------------------


def run_cli(*argv: str) -> None:
    import sys

    from dinkster import manager

    old = sys.argv
    sys.argv = ["dinkster-pack", *argv]
    try:
        manager.main()
    finally:
        sys.argv = old


def test_cli_install_status_remove_rollback_gc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    other = write_pack(tmp_path / "other", "other")

    run_cli("--root", root, "install", "--no-venv", "--yes", str(demo), str(other))
    out = capsys.readouterr().out
    assert "plan (empty root -> new generation" in out
    assert "add: demo" in out and "add: other" in out
    assert "isolation:" in out  # the per-pack process cost is in the plan
    assert "activated generation 1" in out

    run_cli("--root", root, "status")
    out = capsys.readouterr().out
    assert "current generation: 1" in out
    assert "demo 0.0.0" in out and "other 0.0.0" in out

    # identical re-install: empty plan, no prompt, no new generation
    run_cli("--root", root, "install", "--no-venv", str(demo))
    assert "nothing to do" in capsys.readouterr().out

    run_cli("--root", root, "remove", "--no-venv", "--yes", "other")
    out = capsys.readouterr().out
    assert "remove: other (was 0.0.0)" in out and "activated generation 2" in out

    run_cli("--root", root, "rollback", "--no-venv", "--yes")
    out = capsys.readouterr().out
    assert "add: other" in out  # rollback plans like any mutation
    assert "activated generation 3" in out
    installer = Installer(Path(root))
    assert installer.current_lockfile() == installer.lockfile_of(1)

    run_cli("--root", root, "gc")
    assert "0 unreferenced" in capsys.readouterr().out  # history still references all


def test_cli_refuses_unconfirmed_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plan/apply hard requirement: noninteractive mutation without an
    explicit opt-in refuses loudly - and still shows the plan first, so
    the refusal message is reviewable, not a dead end."""
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    with pytest.raises(SystemExit):
        run_cli("--root", root, "install", "--no-venv", str(demo))
    captured = capsys.readouterr()
    assert "plan (empty root -> new generation" in captured.out
    assert "refusing to mutate without confirmation" in captured.err
    assert Installer(Path(root)).current_number() is None  # nothing changed


def test_cli_interactive_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive plans apply on yes and do nothing on anything else."""
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    run_cli("--root", root, "install", "--no-venv", str(demo))
    assert "plan not applied; nothing changed" in capsys.readouterr().out
    assert Installer(Path(root)).current_number() is None

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    run_cli("--root", root, "install", "--no-venv", str(demo))
    assert "activated generation 1" in capsys.readouterr().out
    assert Installer(Path(root)).current_number() == 1


def test_cli_plan_file_and_apply(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--plan writes the reviewable record WITHOUT mutating; apply
    executes exactly that plan; and once the installation moves past the
    plan's base, the same file is stale and refused - replanned, never
    partially applied."""
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    plan_file = tmp_path / "install.plan.json"

    run_cli("--root", root, "install", "--no-venv", "--plan", str(plan_file), str(demo))
    out = capsys.readouterr().out
    assert "add: demo" in out and f"plan written to {plan_file}" in out
    installer = Installer(Path(root))
    assert installer.current_number() is None  # planning never mutates
    # The plan-time archive is staged content-addressed, inert until applied.
    assert list(installer.artifacts_dir.glob("*.zip"))

    run_cli("--root", root, "apply", str(plan_file))
    out = capsys.readouterr().out
    assert "add: demo" in out and "activated generation 1" in out
    assert installer.current_number() == 1

    # Same file again: the base (empty root) no longer matches.
    with pytest.raises(SystemExit):
        run_cli("--root", root, "apply", str(plan_file))
    assert "plan is stale" in capsys.readouterr().err
    assert installer.generation_numbers() == (1,)


def test_plan_apply_honors_recorded_doctor_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    plan_file = tmp_path / "allow.plan.json"
    run_cli(
        "--root",
        root,
        "install",
        "--no-venv",
        "--allow-doctor-findings",
        "--plan",
        str(plan_file),
        str(demo),
    )
    record = PlanRecord.from_record_json(plan_file.read_text())
    assert record.allow_doctor_findings
    failing = DoctorReport(
        "demo",
        "dinkster-pack.toml",
        (Finding("error", "doctor.plan", "planned override"),),
    )
    monkeypatch.setattr(installer_module, "diagnose", lambda *_args, **_kwargs: failing)
    capsys.readouterr()

    run_cli("--root", root, "apply", str(plan_file))
    out = capsys.readouterr().out
    assert "WARNING: allowing doctor findings" in out
    assert "activated generation 1" in out


def test_cli_apply_refuses_when_artifacts_reclaimed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An abandoned plan's artifacts are gc-able (they are referenced by
    no generation); applying the plan afterwards refuses UP FRONT, naming
    the packs, instead of failing mid-stage."""
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    plan_file = tmp_path / "install.plan.json"
    run_cli("--root", root, "install", "--no-venv", "--plan", str(plan_file), str(demo))
    capsys.readouterr()

    run_cli("--root", root, "gc", "--yes")
    out = capsys.readouterr().out
    assert "would remove: artifacts/" in out and "1 unreferenced item(s) removed" in out

    with pytest.raises(SystemExit):
        run_cli("--root", root, "apply", str(plan_file))
    err = capsys.readouterr().err
    assert "no longer available" in err and "demo" in err


def test_cli_gc_requires_confirmation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    plan_file = tmp_path / "plan.json"
    run_cli("--root", root, "install", "--no-venv", "--plan", str(plan_file), str(demo))
    capsys.readouterr()

    with pytest.raises(SystemExit):
        run_cli("--root", root, "gc")
    captured = capsys.readouterr()
    assert "would remove: artifacts/" in captured.out
    assert "refusing to mutate" in captured.err
    assert list(Installer(Path(root)).artifacts_dir.glob("*.zip"))  # still there


def test_cli_snapshot_and_restore(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """snapshot is read-only capture; restore is the same plan/apply
    discipline as every mutation and reproduces the captured lockfile."""
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    other = write_pack(tmp_path / "other", "other")
    snap = tmp_path / "env.snapshot.json"

    run_cli("--root", root, "install", "--no-venv", "--yes", str(demo), str(other))
    capsys.readouterr()
    run_cli("--root", root, "snapshot", str(snap))
    out = capsys.readouterr().out
    assert "snapshot: 2 pack(s)" in out and f"snapshot written to {snap}" in out
    assert "unpinned (no venv staged): demo, other" in out  # --no-venv honesty
    installer = Installer(Path(root))
    assert installer.generation_numbers() == (1,)  # capture mutated nothing

    # drift the installation, then restore the snapshot
    run_cli("--root", root, "remove", "--no-venv", "--yes", "other")
    capsys.readouterr()
    run_cli("--root", root, "restore", "--no-venv", "--yes", str(snap))
    out = capsys.readouterr().out
    assert "add: other" in out and "activated generation 3" in out
    assert installer.current_lockfile() == installer.lockfile_of(1)

    # restoring what already matches plans nothing and mutates nothing
    run_cli("--root", root, "restore", "--no-venv", str(snap))
    assert "already matches" in capsys.readouterr().out
    assert installer.generation_numbers() == (1, 2, 3)

    # unconfirmed noninteractive restore refuses like any mutation
    run_cli("--root", root, "remove", "--no-venv", "--yes", "other")
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run_cli("--root", root, "restore", "--no-venv", str(snap))
    assert "refusing to mutate" in capsys.readouterr().err


def test_cli_plan_shows_platform_and_accelerator_compat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plan shows the selected accelerator, the extra requirements it
    pulls in, and a LOUD (but advisory - the user decides) warning when a
    pack declares platforms excluding this host. The written plan record
    pins the accelerator so apply installs what was reviewed."""
    root = str(tmp_path / "root")
    pack_dir = tmp_path / "compat"
    pack_dir.mkdir()
    foreign = "darwin" if sys.platform != "darwin" else "linux"
    (pack_dir / "dinkster-pack.toml").write_text(
        f'[pack]\nname = "compat"\nplatforms = ["{foreign}"]\nrequires = ["numpy"]\n'
        '[pack.entry]\nnodes = "compat_nodes:NODES"\n'
        '[pack.extra-requires]\ncuda = ["torch==2.5.1", "nvidia-ml-py"]\n'
    )
    (pack_dir / "compat_nodes.py").write_text("NODES = []\n")
    plan_file = tmp_path / "plan.json"

    run_cli(
        "--root",
        root,
        "--accelerator",
        "cuda",
        "install",
        "--no-venv",
        "--plan",
        str(plan_file),
        str(pack_dir),
    )
    out = capsys.readouterr().out
    assert "accelerator: cuda" in out
    assert "+2 cuda requirement(s)" in out
    assert f"WARNING: compat declares platforms {foreign}" in out
    assert f"this host is {sys.platform}" in out
    assert PlanRecord.from_record_json(plan_file.read_text()).accelerator == "cuda"

    # the matching accelerator without extras shows no extras and no warning
    run_cli(
        "--root",
        root,
        "--accelerator",
        "cpu",
        "install",
        "--no-venv",
        "--plan",
        str(tmp_path / "plan2.json"),
        str(pack_dir),
    )
    out = capsys.readouterr().out
    assert "accelerator: cpu" in out and "requirement(s)" not in out


def test_cli_restore_across_scope_partitions_pins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-scope restore honesty (DESIGN M8, cross-platform): restoring
    under a DIFFERENT accelerator partitions the freeze by the data in
    each pin - local-version-label pins (torch==2.0+cu124) are
    vendor-specific and dropped, label-free pins ride along as version
    constraints while ranges plus the destination accelerator decide what
    installs. --fresh-pins drops everything (the escape hatch). A
    matching scope still reproduces exactly. Runtime facts ride along as
    ADVISORY narration and never change any of those decisions."""

    def fake_ensure(
        manifest: PackManifest,
        *,
        venv_root: Path,
        workspace_packages: object = (),
        pinned: tuple[str, ...] | None = None,
        constraints: tuple[str, ...] = (),
        accelerator: str | None = None,
    ) -> Path:
        python = venv_python(venv_root / manifest.name)
        if python.exists():
            return python
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("#!fake\n")
        if pinned is not None:
            content = "\n".join(pinned)
        elif constraints:
            content = f"RANGES:{accelerator}|CONSTRAINTS:{','.join(constraints)}"
        else:
            content = f"RANGES:{accelerator}"
        (python.parent / "pins.txt").write_text(content)
        return python

    def fake_venv_freeze(python: Path, *, uv: str = "uv") -> tuple[str, ...]:
        recorded = (python.parent / "pins.txt").read_text()
        if recorded.startswith("RANGES"):
            return ("numpy==1.0", "torch==2.0+cu124")
        return tuple(recorded.splitlines())

    monkeypatch.setattr(installer_module, "ensure_pack_venv", fake_ensure)
    monkeypatch.setattr(installer_module, "freeze_venv", fake_venv_freeze)
    monkeypatch.setattr(
        installer_module,
        "detect_runtime",
        lambda accelerator: (
            (("cuda", "12.4"), ("driver", "550.54")) if accelerator == "cuda" else ()
        ),
    )

    root = tmp_path / "root"
    demo = write_pack(tmp_path / "demo", "demo")
    snap = tmp_path / "env.snapshot.json"

    run_cli("--root", str(root), "--accelerator", "cuda", "install", "--yes", str(demo))
    capsys.readouterr()
    run_cli("--root", str(root), "--accelerator", "cuda", "snapshot", str(snap))
    out = capsys.readouterr().out
    assert "accelerator cuda" in out
    assert "runtime (advisory): cuda 12.4, driver 550.54" in out
    recorded = SnapshotRecord.from_record_json(snap.read_text())
    assert recorded.accelerator == "cuda"
    assert recorded.runtime == (("cuda", "12.4"), ("driver", "550.54"))

    def venv_pins_content() -> str:
        pins_files = venv_metadata_glob(root, "demo", "pins.txt")
        assert len(pins_files) == 1
        return pins_files[0].read_text()

    # simulate a machine that has the artifacts but no venvs
    run_cli("--root", str(root), "remove", "--no-venv", "--yes", "demo")
    capsys.readouterr()
    shutil.rmtree(root / "venvs")

    # mismatched accelerator: partitioned - the label-free pin becomes a
    # constraint, the +cu124 pin drops, ranges + rocm extras decide
    run_cli("--root", str(root), "--accelerator", "rocm", "restore", "--yes", str(snap))
    out = capsys.readouterr().out
    assert "SCOPE MISMATCH - exact pins will NOT be reused" in out
    assert "accelerator: snapshot cuda, host rocm" in out
    assert "starting point, not an exact" in out
    # runtime facts undetectable on the rocm host: narrated as unknown
    assert "runtime differs (advisory only; does not change pin reuse):" in out
    assert "cuda: snapshot 12.4, host unknown" in out
    assert "1 portable pin(s) kept as version constraints" in out
    assert "1 environment-specific pin(s) dropped (torch)" in out
    assert venv_pins_content() == "RANGES:rocm|CONSTRAINTS:numpy==1.0"

    # --fresh-pins: the escape hatch drops everything, no constraints
    run_cli("--root", str(root), "remove", "--no-venv", "--yes", "demo")
    capsys.readouterr()
    shutil.rmtree(root / "venvs")
    run_cli(
        "--root", str(root), "--accelerator", "rocm", "restore", "--fresh-pins", "--yes", str(snap)
    )
    out = capsys.readouterr().out
    assert "--fresh-pins: ALL snapshot pins dropped" in out
    assert "2 snapshot pin(s) dropped (--fresh-pins)" in out
    assert venv_pins_content() == "RANGES:rocm"

    # matching scope: exact pins reused, faithful reproduction
    run_cli("--root", str(root), "remove", "--no-venv", "--yes", "demo")
    capsys.readouterr()
    shutil.rmtree(root / "venvs")
    run_cli("--root", str(root), "--accelerator", "cuda", "restore", "--yes", str(snap))
    out = capsys.readouterr().out
    assert "SCOPE MISMATCH" not in out
    assert "runtime differs" not in out  # identical facts: nothing to say
    assert "will provision from 2 snapshot pin(s)" in out
    assert venv_pins_content() == "numpy==1.0\ntorch==2.0+cu124"

    # runtime DRIFT on a matching scope (driver upgraded since capture):
    # narrated loudly, but exact pins are still reused verbatim - the
    # advisory facts never become a pin-reuse input
    run_cli("--root", str(root), "remove", "--no-venv", "--yes", "demo")
    capsys.readouterr()
    shutil.rmtree(root / "venvs")
    monkeypatch.setattr(
        installer_module,
        "detect_runtime",
        lambda accelerator: (("cuda", "12.4"), ("driver", "999.99")),
    )
    run_cli("--root", str(root), "--accelerator", "cuda", "restore", "--yes", str(snap))
    out = capsys.readouterr().out
    assert "SCOPE MISMATCH" not in out
    assert "runtime differs (advisory only; does not change pin reuse):" in out
    assert "driver: snapshot 550.54, host 999.99" in out
    assert "will provision from 2 snapshot pin(s)" in out
    assert venv_pins_content() == "numpy==1.0\ntorch==2.0+cu124"


def test_cli_snapshot_hashes_and_hash_verified_restore(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolution lock end to end: 'snapshot --hashes' annotates
    portable pins with artifact hashes (vendor builds honestly bare), a
    same-scope restore hands the annotated pins to provisioning verbatim
    (uv verifies them at install), and a cross-scope restore's demotion
    to version constraints strips the annotations - artifact identity
    drops with the exactness it belongs to."""

    def fake_ensure(
        manifest: PackManifest,
        *,
        venv_root: Path,
        workspace_packages: object = (),
        pinned: tuple[str, ...] | None = None,
        constraints: tuple[str, ...] = (),
        accelerator: str | None = None,
    ) -> Path:
        python = venv_python(venv_root / manifest.name)
        if python.exists():
            return python
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("#!fake\n")
        if pinned is not None:
            content = "\n".join(pinned)
        elif constraints:
            content = f"RANGES:{accelerator}|CONSTRAINTS:{','.join(constraints)}"
        else:
            content = f"RANGES:{accelerator}"
        (python.parent / "pins.txt").write_text(content)
        return python

    def fake_venv_freeze(python: Path, *, uv: str = "uv") -> tuple[str, ...]:
        recorded = (python.parent / "pins.txt").read_text()
        if recorded.startswith("RANGES"):
            return ("numpy==1.0", "torch==2.0+cu124")
        # a real freeze reports name==version only, never annotations
        return tuple(sorted(line.split()[0] for line in recorded.splitlines()))

    def fake_hash_pins(pins: Sequence[str], *, uv: str = "uv") -> tuple[str, ...]:
        return tuple(
            sorted(
                pin if "+" in pin.partition("==")[2] else f"{pin} --hash=sha256:aaaa"
                for pin in pins
            )
        )

    monkeypatch.setattr(installer_module, "ensure_pack_venv", fake_ensure)
    monkeypatch.setattr(installer_module, "freeze_venv", fake_venv_freeze)
    monkeypatch.setattr(installer_module, "hash_pins", fake_hash_pins)

    root = tmp_path / "root"
    demo = write_pack(tmp_path / "demo", "demo")
    snap = tmp_path / "env.snapshot.json"

    run_cli("--root", str(root), "--accelerator", "cuda", "install", "--yes", str(demo))
    capsys.readouterr()
    run_cli("--root", str(root), "--accelerator", "cuda", "snapshot", "--hashes", str(snap))
    out = capsys.readouterr().out
    assert "hashes: 1 pin(s) hash-annotated" in out
    assert "1 environment-specific pin(s) stay bare" in out
    recorded = SnapshotRecord.from_record_json(snap.read_text())
    assert recorded.pins_for("demo") == ("numpy==1.0 --hash=sha256:aaaa", "torch==2.0+cu124")

    def venv_pins_content() -> str:
        pins_files = venv_metadata_glob(root, "demo", "pins.txt")
        assert len(pins_files) == 1
        return pins_files[0].read_text()

    # same scope: annotated pins reach provisioning verbatim, said out loud
    run_cli("--root", str(root), "remove", "--no-venv", "--yes", "demo")
    capsys.readouterr()
    shutil.rmtree(root / "venvs")
    run_cli("--root", str(root), "--accelerator", "cuda", "restore", "--yes", str(snap))
    out = capsys.readouterr().out
    assert "will provision from 2 snapshot pin(s) (1 hash-verified)" in out
    assert venv_pins_content() == "numpy==1.0 --hash=sha256:aaaa\ntorch==2.0+cu124"
    # an annotated snapshot pin matching the staged version is no drift
    run_cli("--root", str(root), "--accelerator", "cuda", "restore", "--yes", str(snap))
    out = capsys.readouterr().out
    assert "drifts" not in out

    # cross scope: the portable pin demotes to a BARE version constraint
    run_cli("--root", str(root), "remove", "--no-venv", "--yes", "demo")
    capsys.readouterr()
    shutil.rmtree(root / "venvs")
    run_cli("--root", str(root), "--accelerator", "rocm", "restore", "--yes", str(snap))
    out = capsys.readouterr().out
    assert "1 portable pin(s) kept as version constraints" in out
    assert "1 environment-specific pin(s) dropped (torch)" in out
    assert venv_pins_content() == "RANGES:rocm|CONSTRAINTS:numpy==1.0"


def test_cli_restore_reacquires_missing_artifacts_from_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A snapshot restored on a machine WITHOUT the pack bytes re-fetches
    them from the recorded provenance (here a local: source), shows that
    in the plan, and digest-verifies what it fetched - never a silent
    substitution."""
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    snap = tmp_path / "env.snapshot.json"
    run_cli("--root", root, "install", "--no-venv", "--yes", str(demo))
    capsys.readouterr()
    run_cli("--root", root, "snapshot", str(snap))
    capsys.readouterr()

    fresh = str(tmp_path / "fresh-root")
    run_cli("--root", fresh, "restore", "--no-venv", "--yes", str(snap))
    out = capsys.readouterr().out
    assert f"re-acquire demo from local:{demo.resolve()}" in out
    assert "re-acquired demo: digest verified" in out
    assert "activated generation 1" in out

    # source tree changed since capture: acquisition refuses loudly
    (demo / "demo_nodes.py").write_text("NODES = [999]\n")
    with pytest.raises(SystemExit):
        run_cli("--root", str(tmp_path / "root2"), "restore", "--no-venv", "--yes", str(snap))
    err = capsys.readouterr().err
    assert "refusing to substitute different bytes" in err

    # a source nothing can re-fetch (registry downloads not built):
    # refused up front, before any plan chatter
    document = json.loads(snap.read_text())
    for entry in document["lockfile"]["packs"]:
        entry["source"] = "registry"
    snap.write_text(json.dumps(document))
    with pytest.raises(SystemExit):
        run_cli("--root", str(tmp_path / "root3"), "restore", "--no-venv", "--yes", str(snap))
    err = capsys.readouterr().err
    assert "not available locally" in err and "demo" in err
    assert "cannot be re-acquired" in err


def test_cli_restore_plan_file_defers_to_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """restore --plan FILE writes a PlanRecord carrying acquire + the
    venv specs the plan was reviewed with and mutates NOTHING;
    'dinkster-pack apply FILE' later re-acquires the bytes digest-verified
    and provisions exactly those specs - same-scope restores record the
    snapshot's exact pins, cross-scope restores record the portable
    constraints, and --fresh-pins/--no-venv record no specs at all. A
    stale base or changed source bytes refuse at apply, before anything
    activates."""

    def fake_ensure(
        manifest: PackManifest,
        *,
        venv_root: Path,
        workspace_packages: object = (),
        pinned: tuple[str, ...] | None = None,
        constraints: tuple[str, ...] = (),
        accelerator: str | None = None,
    ) -> Path:
        python = venv_python(venv_root / manifest.name)
        if python.exists():
            return python
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("#!fake\n")
        if pinned is not None:
            content = "\n".join(pinned)
        elif constraints:
            content = f"RANGES:{accelerator}|CONSTRAINTS:{','.join(constraints)}"
        else:
            content = f"RANGES:{accelerator}"
        (python.parent / "pins.txt").write_text(content)
        return python

    def fake_venv_freeze(python: Path, *, uv: str = "uv") -> tuple[str, ...]:
        recorded = (python.parent / "pins.txt").read_text()
        if recorded.startswith("RANGES"):
            return ("numpy==1.0", "torch==2.0+cu124")
        return tuple(recorded.splitlines())

    monkeypatch.setattr(installer_module, "ensure_pack_venv", fake_ensure)
    monkeypatch.setattr(installer_module, "freeze_venv", fake_venv_freeze)

    source_root = tmp_path / "source-root"
    demo = write_pack(tmp_path / "demo", "demo")
    snap = tmp_path / "env.snapshot.json"
    run_cli("--root", str(source_root), "--accelerator", "cuda", "install", "--yes", str(demo))
    run_cli("--root", str(source_root), "--accelerator", "cuda", "snapshot", str(snap))
    capsys.readouterr()

    def venv_pins_content(root: Path) -> str:
        pins_files = venv_metadata_glob(root, "demo", "pins.txt")
        assert len(pins_files) == 1
        return pins_files[0].read_text()

    # same scope: the plan records the exact freeze; nothing is touched
    fresh = tmp_path / "fresh-root"
    plan_file = tmp_path / "plan.json"
    run_cli(
        "--root",
        str(fresh),
        "--accelerator",
        "cuda",
        "restore",
        "--plan",
        str(plan_file),
        str(snap),
    )
    out = capsys.readouterr().out
    assert f"re-acquire demo from local:{demo.resolve()}" in out
    assert f"plan written to {plan_file}" in out
    record = PlanRecord.from_record_json(plan_file.read_text())
    assert record.acquire and not record.derive_claims
    assert record.venvs and record.accelerator == "cuda"
    assert record.venv_specs == (("demo", VenvSpec(exact=("numpy==1.0", "torch==2.0+cu124"))),)
    fresh_installer = Installer(fresh)
    assert fresh_installer.current_lockfile() is None  # plan phase stayed read-only
    assert not list(fresh_installer.artifacts_dir.glob("*.zip"))
    assert not list((fresh / "venvs").iterdir())  # scaffold only, nothing staged

    run_cli("--root", str(fresh), "apply", str(plan_file))
    out = capsys.readouterr().out
    assert "acquired demo: digest verified" in out
    assert "activated generation 1" in out
    assert venv_pins_content(fresh) == "numpy==1.0\ntorch==2.0+cu124"

    # cross-scope: the plan records the portable constraints and the
    # DESTINATION accelerator; apply resolves ranges under exactly those
    rocm_root = tmp_path / "rocm-root"
    rocm_plan = tmp_path / "rocm-plan.json"
    run_cli(
        "--root",
        str(rocm_root),
        "--accelerator",
        "rocm",
        "restore",
        "--plan",
        str(rocm_plan),
        str(snap),
    )
    out = capsys.readouterr().out
    assert "SCOPE MISMATCH" in out
    record = PlanRecord.from_record_json(rocm_plan.read_text())
    assert record.accelerator == "rocm"
    assert record.venv_specs == (("demo", VenvSpec(constraints=("numpy==1.0",))),)
    run_cli("--root", str(rocm_root), "apply", str(rocm_plan))
    capsys.readouterr()
    assert venv_pins_content(rocm_root) == "RANGES:rocm|CONSTRAINTS:numpy==1.0"

    # --no-venv and --fresh-pins record no specs (nothing was reviewed)
    novenv_plan = tmp_path / "novenv-plan.json"
    run_cli(
        "--root",
        str(tmp_path / "novenv-root"),
        "--accelerator",
        "cuda",
        "restore",
        "--no-venv",
        "--plan",
        str(novenv_plan),
        str(snap),
    )
    record = PlanRecord.from_record_json(novenv_plan.read_text())
    assert not record.venvs and record.venv_specs == ()
    fresh_plan = tmp_path / "fresh-pins-plan.json"
    run_cli(
        "--root",
        str(tmp_path / "fresh-pins-root"),
        "--accelerator",
        "cuda",
        "restore",
        "--fresh-pins",
        "--plan",
        str(fresh_plan),
        str(snap),
    )
    capsys.readouterr()
    record = PlanRecord.from_record_json(fresh_plan.read_text())
    assert record.venvs and record.venv_specs == ()

    # base moved after planning: apply refuses BEFORE acquiring anything
    stale_root = tmp_path / "stale-root"
    stale_plan = tmp_path / "stale-plan.json"
    run_cli(
        "--root",
        str(stale_root),
        "--accelerator",
        "cuda",
        "restore",
        "--plan",
        str(stale_plan),
        str(snap),
    )
    other = write_pack(tmp_path / "other", "other")
    run_cli("--root", str(stale_root), "install", "--no-venv", "--yes", str(other))
    capsys.readouterr()
    zips_before = sorted(Installer(stale_root).artifacts_dir.glob("*.zip"))
    with pytest.raises(SystemExit):
        run_cli("--root", str(stale_root), "apply", str(stale_plan))
    assert "plan is stale" in capsys.readouterr().err
    assert sorted(Installer(stale_root).artifacts_dir.glob("*.zip")) == zips_before

    # source bytes changed after planning: apply refuses the digest
    # mismatch and activates nothing
    drift_root = tmp_path / "drift-root"
    drift_plan = tmp_path / "drift-plan.json"
    run_cli(
        "--root",
        str(drift_root),
        "--accelerator",
        "cuda",
        "restore",
        "--plan",
        str(drift_plan),
        str(snap),
    )
    capsys.readouterr()
    (demo / "demo_nodes.py").write_text("NODES = [999]\n")
    with pytest.raises(SystemExit):
        run_cli("--root", str(drift_root), "apply", str(drift_plan))
    assert "refusing to substitute different bytes" in capsys.readouterr().err
    assert Installer(drift_root).current_lockfile() is None


def _stamped_workflow(path: Path, packs: dict[str, dict[str, str]], dinkster: str = "") -> Path:
    """A minimal workflow document carrying an environment stamp - the
    shape a saving client writes from the live packs table."""
    environment: dict[str, object] = {"packs": packs}
    if dinkster:
        environment["dinkster"] = {"version": dinkster}
    path.write_text(json.dumps({"graphs": {}, "environment": environment}))
    return path


def test_cli_reproduce_rebuilds_from_environment_stamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """dinkster-pack reproduce: a workflow's environment stamp - elsewhere
    an advisory record - becomes an explicit install target. Bytes are
    re-acquired from stamped provenance digest-verified, claims come
    from the acquired manifests, and the new generation contains
    EXACTLY the stamped packs (extras are removed, shown in the plan)."""
    source_root = tmp_path / "source-root"
    demo = write_pack(tmp_path / "demo", "demo")
    other = write_pack(tmp_path / "other", "other")
    run_cli("--root", str(source_root), "install", "--no-venv", "--yes", str(demo), str(other))
    capsys.readouterr()
    lockfile = Installer(source_root).current_lockfile()
    assert lockfile is not None
    entry = lockfile.get("demo")
    assert entry is not None
    workflow = _stamped_workflow(
        tmp_path / "workflow.json",
        {"demo": {"artifactDigest": entry.artifact_digest, "source": entry.source}},
        dinkster="9.9.9",
    )

    # fresh machine: bytes re-acquired from provenance, then installed
    fresh = tmp_path / "fresh-root"
    run_cli("--root", str(fresh), "reproduce", "--no-venv", "--yes", str(workflow))
    out = capsys.readouterr().out
    assert "note: workflow was stamped by dinkster 9.9.9" in out
    assert f"acquire demo from {entry.source}" in out
    assert "acquired demo: digest verified" in out
    assert "environment stamps pin pack bytes, not python dists" in out
    assert "add: demo" in out and "activated generation 1" in out
    reproduced = Installer(fresh).current_lockfile()
    assert reproduced is not None and len(reproduced.packs) == 1
    rebuilt = reproduced.get("demo")
    assert rebuilt is not None
    assert rebuilt.artifact_digest == entry.artifact_digest
    assert rebuilt.claims  # derived from the acquired manifest, never empty

    # existing root with an extra pack: the plan says so, then removes it
    run_cli("--root", str(source_root), "reproduce", "--no-venv", "--yes", str(workflow))
    out = capsys.readouterr().out
    assert "remove: other" in out and "activated generation" in out
    after = Installer(source_root).current_lockfile()
    assert after is not None and [e.pack for e in after.packs] == ["demo"]

    # already matching: nothing to do, no new generation
    run_cli("--root", str(fresh), "reproduce", "--no-venv", "--yes", str(workflow))
    assert "nothing to do" in capsys.readouterr().out


def test_cli_reproduce_plan_is_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without confirmation, reproduce fetches NOTHING: the plan narrates
    the acquisition it would perform, refuses to mutate noninteractively,
    and leaves the target root without artifacts or a generation."""
    source_root = tmp_path / "source-root"
    demo = write_pack(tmp_path / "demo", "demo")
    run_cli("--root", str(source_root), "install", "--no-venv", "--yes", str(demo))
    capsys.readouterr()
    lockfile = Installer(source_root).current_lockfile()
    assert lockfile is not None
    entry = lockfile.get("demo")
    assert entry is not None
    workflow = _stamped_workflow(
        tmp_path / "workflow.json",
        {"demo": {"artifactDigest": entry.artifact_digest, "source": entry.source}},
    )

    fresh = tmp_path / "fresh-root"
    with pytest.raises(SystemExit):
        run_cli("--root", str(fresh), "reproduce", "--no-venv", str(workflow))
    captured = capsys.readouterr()
    assert f"acquire demo from {entry.source}" in captured.out
    assert "add: demo" in captured.out
    assert "refusing to mutate without confirmation" in captured.err
    fresh_installer = Installer(fresh)
    assert fresh_installer.current_lockfile() is None
    assert not list(fresh_installer.artifacts_dir.glob("*.zip"))


def test_cli_reproduce_plan_file_defers_acquisition_to_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """reproduce --plan FILE writes a PlanRecord marked acquire +
    deriveClaims and mutates nothing; 'dinkster-pack apply FILE' later
    performs the acquisition digest-verified and derives claims from the
    acquired manifests - the plan file is the confirmation."""
    source_root = tmp_path / "source-root"
    demo = write_pack(tmp_path / "demo", "demo")
    run_cli("--root", str(source_root), "install", "--no-venv", "--yes", str(demo))
    capsys.readouterr()
    lockfile = Installer(source_root).current_lockfile()
    assert lockfile is not None
    entry = lockfile.get("demo")
    assert entry is not None
    workflow = _stamped_workflow(
        tmp_path / "workflow.json",
        {"demo": {"artifactDigest": entry.artifact_digest, "source": entry.source}},
    )

    fresh = tmp_path / "fresh-root"
    plan_file = tmp_path / "plan.json"
    run_cli("--root", str(fresh), "reproduce", "--no-venv", "--plan", str(plan_file), str(workflow))
    out = capsys.readouterr().out
    assert f"plan written to {plan_file}" in out
    record = PlanRecord.from_record_json(plan_file.read_text())
    assert record.acquire and record.derive_claims
    fresh_installer = Installer(fresh)
    assert fresh_installer.current_lockfile() is None  # plan phase stayed read-only
    assert not list(fresh_installer.artifacts_dir.glob("*.zip"))

    run_cli("--root", str(fresh), "apply", str(plan_file))
    out = capsys.readouterr().out
    assert f"acquire demo from {entry.source}" in out
    assert "acquired demo: digest verified" in out
    assert "activated generation 1" in out
    applied = Installer(fresh).current_lockfile()
    assert applied is not None
    rebuilt = applied.get("demo")
    assert rebuilt is not None
    assert rebuilt.artifact_digest == entry.artifact_digest
    assert rebuilt.claims == entry.claims  # manifest-derived, not placeholders

    # a mislabeled stamp rides the plan file but still refuses at apply,
    # before anything activates
    mislabeled = _stamped_workflow(
        tmp_path / "mislabeled.json",
        {"impostor": {"artifactDigest": entry.artifact_digest, "source": entry.source}},
    )
    bad_plan = tmp_path / "bad-plan.json"
    bad_root = tmp_path / "bad-root"
    run_cli(
        "--root", str(bad_root), "reproduce", "--no-venv", "--plan", str(bad_plan), str(mislabeled)
    )
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run_cli("--root", str(bad_root), "apply", str(bad_plan))
    assert "mislabeled stamp" in capsys.readouterr().err
    assert Installer(bad_root).current_lockfile() is None


def test_cli_reproduce_refuses_bad_stamps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(tmp_path / "root")
    demo = write_pack(tmp_path / "demo", "demo")
    run_cli("--root", root, "install", "--no-venv", "--yes", str(demo))
    capsys.readouterr()
    lockfile = Installer(Path(root)).current_lockfile()
    assert lockfile is not None
    entry = lockfile.get("demo")
    assert entry is not None

    # no stamp at all
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"graphs": {}}))
    with pytest.raises(SystemExit):
        run_cli("--root", root, "reproduce", "--no-venv", "--yes", str(bare))
    assert "no environment stamp" in capsys.readouterr().err

    # an unpinned stamp entry refuses by default, --skip-unpinned proceeds
    mixed = _stamped_workflow(
        tmp_path / "mixed.json",
        {
            "demo": {"artifactDigest": entry.artifact_digest, "source": entry.source},
            "devpack": {"source": "local:/dev/devpack"},
        },
    )
    with pytest.raises(SystemExit):
        run_cli("--root", root, "reproduce", "--no-venv", "--yes", str(mixed))
    err = capsys.readouterr().err
    assert "no digest pin for: devpack" in err and "--skip-unpinned" in err
    run_cli("--root", root, "reproduce", "--no-venv", "--yes", "--skip-unpinned", str(mixed))
    out = capsys.readouterr().out
    assert "skipping devpack" in out

    # a stamp that pins nothing has nothing to reproduce
    with pytest.raises(SystemExit):
        run_cli(
            "--root",
            root,
            "reproduce",
            "--no-venv",
            "--yes",
            "--skip-unpinned",
            str(_stamped_workflow(tmp_path / "none.json", {"devpack": {}})),
        )
    assert "pins no packs" in capsys.readouterr().err

    # unavailable bytes with un-refetchable provenance
    ghost = _stamped_workflow(
        tmp_path / "ghost.json",
        {"ghost": {"artifactDigest": "sha256:" + "0" * 64, "source": "registry"}},
    )
    with pytest.raises(SystemExit):
        run_cli("--root", root, "reproduce", "--no-venv", "--yes", str(ghost))
    err = capsys.readouterr().err
    assert "not available locally" in err and "cannot be re-fetched" in err

    # a stamp that pairs a pack id with some OTHER pack's digest
    mislabeled = _stamped_workflow(
        tmp_path / "mislabeled.json",
        {"impostor": {"artifactDigest": entry.artifact_digest, "source": entry.source}},
    )
    with pytest.raises(SystemExit):
        run_cli("--root", root, "reproduce", "--no-venv", "--yes", str(mislabeled))
    assert "mislabeled stamp" in capsys.readouterr().err

    # malformed digest strings are named before anything is fetched
    invalid = _stamped_workflow(
        tmp_path / "invalid.json", {"demo": {"artifactDigest": "sha256:nope"}}
    )
    with pytest.raises(SystemExit):
        run_cli("--root", root, "reproduce", "--no-venv", "--yes", str(invalid))
    assert "stamped digest for pack 'demo'" in capsys.readouterr().err


def test_cli_errors_exit_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = str(tmp_path / "root")
    with pytest.raises(SystemExit):
        run_cli("--root", "", "status")
    assert "no install root" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_cli("--root", root, "remove", "--no-venv", "ghost")
    assert "nothing is installed" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_cli("--root", root, "install", "--no-venv", str(tmp_path / "nowhere"))
    assert "not found" in capsys.readouterr().err


def test_publish_preflight_refuses_and_no_preflight_submits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster import manager
    from dinkster.registries import PublishVerdict

    pack = write_pack(tmp_path / "demo", "demo")
    report = DoctorReport(
        "demo",
        str(pack / "dinkster-pack.toml"),
        (Finding("error", "entry.unresolvable", "broken import"),),
    )
    monkeypatch.setattr(manager, "diagnose", lambda *_args, **_kwargs: report)
    submitted: list[str] = []

    def publish(_registry, _archive, digest: str, _version: str) -> PublishVerdict:
        submitted.append(digest)
        return PublishVerdict("accepted", ())

    monkeypatch.setattr(manager, "publish_release", publish)
    with pytest.raises(SystemExit) as refusal:
        run_cli(
            "--registry",
            "https://registry.invalid",
            "publish",
            "--version",
            "1.2.3",
            str(pack),
        )
    assert refusal.value.code == 1
    assert "entry.unresolvable: broken import" in capsys.readouterr().out
    assert submitted == []

    run_cli(
        "--registry",
        "https://registry.invalid",
        "publish",
        "--version",
        "1.2.3",
        "--no-preflight",
        str(pack),
    )
    assert len(submitted) == 1
    assert f"published demo 1.2.3 {submitted[0]}" in capsys.readouterr().out


def test_publish_admission_rejection_surfaces_registry_findings_verbatim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dinkster import manager
    from dinkster.registries import PublishFinding, PublishVerdict

    pack = write_pack(tmp_path / "demo", "demo")
    monkeypatch.setattr(
        manager,
        "publish_release",
        lambda *_args, **_kwargs: PublishVerdict(
            "rejected",
            (
                PublishFinding("registry.doctor-failed", "pack import failed"),
                PublishFinding("registry.namespace-denied", "claim is not granted"),
            ),
        ),
    )
    with pytest.raises(SystemExit) as refusal:
        run_cli(
            "--registry",
            "https://registry.invalid",
            "publish",
            "--version",
            "1.0.0",
            "--no-preflight",
            str(pack),
        )
    assert refusal.value.code == 1
    assert capsys.readouterr().out.splitlines() == [
        "registry.doctor-failed: pack import failed",
        "registry.namespace-denied: claim is not granted",
    ]


def test_publish_usage_error_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pack = write_pack(tmp_path / "demo", "demo")
    with pytest.raises(SystemExit) as usage:
        run_cli(
            "--registry",
            "https://registry.invalid",
            "publish",
            "--version",
            "not-a-version",
            str(pack),
        )
    assert usage.value.code == 2
    assert "version 'not-a-version'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# git install sources (same artifact path, provenance-pinned commits)
# ---------------------------------------------------------------------------


def make_git_repo(directory: Path, name: str = "demo") -> Path:
    """A throwaway fixture repository with one committed pack."""
    write_pack(directory, name)
    _git(directory, "init", "-q")
    _commit(directory, "init")
    return directory


def _git(repo: Path, *args: str) -> str:
    import subprocess

    env = {
        **os.environ,
        # fixture-repo identity only; never touches any real repo config
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def test_lock_git_pack_pins_the_commit(tmp_path: Path) -> None:
    repo = make_git_repo(tmp_path / "repo")
    head = _git(repo, "rev-parse", "HEAD")
    url = f"file://{repo}"
    entry, archive = lock_git_pack(url, tmp_path / "artifacts")
    assert entry.pack == "demo"
    assert entry.publisher == "local"
    assert entry.source == f"git:{url}@{head}"
    assert archive.is_file()
    # the clone archives to the same digest as the worktree itself
    # (.git is excluded from artifacts, so the trees are identical)
    rebuilt = build_artifact(repo, tmp_path / "worktree.zip")
    assert rebuilt == entry.artifact_digest


def test_lock_git_pack_ref_selects_history(tmp_path: Path) -> None:
    repo = make_git_repo(tmp_path / "repo")
    _git(repo, "tag", "v1")
    (repo / "demo_nodes.py").write_text("NODES = [2]\n")
    _commit(repo, "change")
    url = f"file://{repo}"
    at_tag, _ = lock_git_pack(url, tmp_path / "artifacts", ref="v1")
    at_head, _ = lock_git_pack(url, tmp_path / "artifacts")
    assert at_tag.artifact_digest != at_head.artifact_digest
    assert at_tag.source.endswith(_git(repo, "rev-parse", "v1"))


def test_lock_git_pack_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(InstallError, match="git clone failed"):
        lock_git_pack(f"file://{tmp_path}/nowhere", tmp_path / "artifacts")
    repo = make_git_repo(tmp_path / "repo")
    with pytest.raises(InstallError, match="git checkout failed"):
        lock_git_pack(f"file://{repo}", tmp_path / "artifacts", ref="no-such-ref")


# ---------------------------------------------------------------------------
# artifact re-acquisition from recorded provenance (digest-verified)
# ---------------------------------------------------------------------------


def test_acquire_refetches_local_source_bytes(tmp_path: Path) -> None:
    """A gc'd/absent artifact is rebuilt from its recorded local: path
    and admitted only because it hashes to the recorded digest.
    Idempotent: present bytes are never re-fetched."""
    installer = make_installer(tmp_path)
    demo = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(demo, installer.artifacts_dir)
    archive.unlink()
    assert not installer.artifact_available(entry)
    installer.acquire(entry)
    assert installer.artifact_available(entry) and archive.is_file()
    installer.acquire(entry)  # no-op, no error
    # nothing but content-addressed zips in the artifacts dir (no
    # leftover scratch)
    assert all(child.suffix == ".zip" for child in installer.artifacts_dir.iterdir())


def test_acquire_refuses_changed_or_vanished_local_source(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    demo = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(demo, installer.artifacts_dir)
    archive.unlink()
    (demo / "demo_nodes.py").write_text("NODES = [999]\n")
    with pytest.raises(InstallError, match="refusing to substitute different bytes"):
        installer.acquire(entry)
    # the mismatched candidate was discarded, not admitted
    assert not installer.artifact_available(entry)
    assert all(child.suffix == ".zip" for child in installer.artifacts_dir.iterdir())
    shutil.rmtree(demo)
    with pytest.raises(InstallError, match="no longer exists"):
        installer.acquire(entry)


def test_acquire_refetches_git_source_at_the_recorded_commit(tmp_path: Path) -> None:
    """git: provenance pins the exact commit, so re-acquisition succeeds
    even after the branch moves on - the recorded commit is checked out,
    not HEAD."""
    installer = make_installer(tmp_path)
    repo = make_git_repo(tmp_path / "repo")
    entry, archive = lock_git_pack(f"file://{repo}", installer.artifacts_dir)
    (repo / "demo_nodes.py").write_text("NODES = [2]\n")
    _commit(repo, "moved on")
    archive.unlink()
    installer.acquire(entry)
    assert installer.artifact_available(entry) and archive.is_file()


def test_acquire_refuses_unacquirable_sources(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    demo = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(demo, installer.artifacts_dir)
    archive.unlink()
    # registry sources are acquirable only when an endpoint is configured
    registry_entry = replace(entry, source="registry")
    assert not installer.acquirable(registry_entry)
    with pytest.raises(InstallError, match="no registry endpoint is configured"):
        installer.acquire(registry_entry)
    unknown = replace(entry, source="ftp:nowhere")
    assert not installer.acquirable(unknown)
    with pytest.raises(InstallError, match="not\\s+re-acquirable"):
        installer.acquire(unknown)
    malformed = replace(entry, source="git:no-commit-here")
    with pytest.raises(InstallError, match="git:<url>@<commit>"):
        installer.acquire(malformed)


def test_acquire_registry_source_downloads_digest_verified(tmp_path: Path) -> None:
    """Registry acquisition is a content-addressed download: the fetcher
    only moves bytes, and the entry's recorded digest decides admission.
    Exact bytes are admitted (once - present bytes never re-fetch); wrong
    bytes or a failed download refuse and leave the store untouched."""
    staging = make_installer(tmp_path)
    demo = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(demo, staging.artifacts_dir)
    data = archive.read_bytes()
    archive.unlink()
    release = replace(entry, source="registry", version="1.0.0")

    calls: list[str] = []

    def fetch(requested: LockedPack) -> bytes:
        calls.append(requested.artifact_digest)
        return data

    installer = Installer(
        tmp_path / "root", provision=fake_provision, freeze=fake_freeze, registry_fetch=fetch
    )
    assert installer.acquirable(release)
    assert installer.acquirable(replace(release, source="registry:main"))
    installer.acquire(release)
    assert installer.artifact_available(release)
    assert calls == [release.artifact_digest]
    installer.acquire(release)  # idempotent: present bytes never re-fetch
    assert calls == [release.artifact_digest]

    # a lying registry: bytes that hash differently are refused, store untouched
    lying = Installer(
        tmp_path / "lying-root",
        provision=fake_provision,
        freeze=fake_freeze,
        registry_fetch=lambda _entry: b"not the recorded bytes",
    )
    with pytest.raises(InstallError, match="refusing to\\s+substitute different bytes"):
        lying.acquire(release)
    assert not list(lying.artifacts_dir.glob("*.zip"))

    # a failed download propagates and admits nothing
    def broken(_entry: LockedPack) -> bytes:
        raise InstallError("registry download of demo from nowhere failed: down")

    offline = Installer(
        tmp_path / "offline-root",
        provision=fake_provision,
        freeze=fake_freeze,
        registry_fetch=broken,
    )
    with pytest.raises(InstallError, match="failed: down"):
        offline.acquire(release)
    assert not list(offline.artifacts_dir.glob("*.zip"))


def _serve_artifacts(
    files: dict[str, bytes], requests: list[tuple[str, str | None]]
) -> ThreadingHTTPServer:
    """A registry stand-in: plain HTTP GET of hosted paths, recording
    (path, Authorization header) per request. Caller must shutdown()."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            blob = files.get(self.path)
            if blob is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, format: str, *args: object) -> None:
            pass  # keep test output clean

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_http_registry_fetcher_is_content_addressed(tmp_path: Path) -> None:
    """The real fetcher asks for exactly /artifacts/<hex>.zip - the digest
    is the whole request, so there is nothing to resolve and nothing a
    server can substitute that verification would not catch. Failures
    (404, refused connection) surface as InstallError; a token rides as a
    bearer header; non-http endpoints refuse at construction."""
    installer = make_installer(tmp_path)
    demo = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(demo, installer.artifacts_dir)
    data = archive.read_bytes()
    hex_digest = entry.artifact_digest.partition(":")[2]
    release = replace(entry, source="registry", version="1.0.0")

    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts({f"/artifacts/{hex_digest}.zip": data}, requests)
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        fetch = http_registry_fetcher(endpoint + "/")  # trailing slash normalized
        assert fetch(release) == data
        assert requests == [(f"/artifacts/{hex_digest}.zip", None)]

        authed = http_registry_fetcher(endpoint, token="s3cret")
        assert authed(release) == data
        assert requests[-1] == (f"/artifacts/{hex_digest}.zip", "Bearer s3cret")

        missing = replace(release, artifact_digest="sha256:" + "0" * 64)
        with pytest.raises(InstallError, match="registry download of demo .* failed"):
            fetch(missing)
    finally:
        server.shutdown()

    with pytest.raises(InstallError, match="must be an http\\(s\\) URL"):
        http_registry_fetcher("file:///etc")
    with pytest.raises(InstallError, match="failed"):
        http_registry_fetcher("http://127.0.0.1:1")(release)  # connection refused


def test_cli_restore_downloads_registry_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: a snapshot whose entries record a registry source
    restores on a machine without the bytes by downloading them from
    --registry (digest-verified into the artifact store). Without an
    endpoint the plan refuses up front naming the fix; a server serving
    different bytes refuses at acquisition and activates nothing."""
    source_root = tmp_path / "source-root"
    demo = write_pack(tmp_path / "demo", "demo")
    snap = tmp_path / "env.snapshot.json"
    run_cli("--root", str(source_root), "install", "--no-venv", "--yes", str(demo))
    run_cli("--root", str(source_root), "snapshot", str(snap))
    capsys.readouterr()
    document = json.loads(snap.read_text())
    [entry_record] = document["lockfile"]["packs"]
    entry_record["source"] = "registry"
    snap.write_text(json.dumps(document))
    hex_digest = entry_record["artifactDigest"].partition(":")[2]
    data = (source_root / "artifacts" / f"{hex_digest}.zip").read_bytes()

    # no endpoint configured: refused up front, naming the way forward
    with pytest.raises(SystemExit):
        run_cli("--root", str(tmp_path / "nowhere"), "restore", "--no-venv", "--yes", str(snap))
    err = capsys.readouterr().err
    assert "cannot be re-acquired" in err and "--registry" in err

    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts({f"/artifacts/{hex_digest}.zip": data}, requests)
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        fresh = tmp_path / "fresh-root"
        run_cli(
            "--root",
            str(fresh),
            "--registry",
            endpoint,
            "restore",
            "--no-venv",
            "--yes",
            str(snap),
        )
        out = capsys.readouterr().out
        assert "re-acquire demo from registry" in out
        assert "re-acquired demo: digest verified" in out
        assert "activated generation 1" in out
        assert requests == [(f"/artifacts/{hex_digest}.zip", None)]

        # a lying server: different bytes refuse, nothing activates
        lying = _serve_artifacts({f"/artifacts/{hex_digest}.zip": b"different bytes entirely"}, [])
        try:
            bad_root = tmp_path / "bad-root"
            with pytest.raises(SystemExit):
                run_cli(
                    "--root",
                    str(bad_root),
                    "--registry",
                    f"http://127.0.0.1:{lying.server_port}",
                    "restore",
                    "--no-venv",
                    "--yes",
                    str(snap),
                )
            assert "refusing to" in capsys.readouterr().err
            assert Installer(bad_root).current_lockfile() is None
        finally:
            lying.shutdown()
    finally:
        server.shutdown()


def test_cli_git_install_and_update(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = str(tmp_path / "root")
    repo = make_git_repo(tmp_path / "repo")
    url = f"file://{repo}"

    run_cli("--root", root, "install", "--no-venv", "--yes", f"git+{url}")
    out = capsys.readouterr().out
    assert "add: demo" in out and "activated generation 1" in out
    installer = Installer(Path(root))
    lockfile = installer.current_lockfile()
    assert lockfile is not None
    assert lockfile.packs[0].source.startswith(f"git:{url}@")

    # nothing changed upstream: update is a no-op, no new generation
    run_cli("--root", root, "update", "--no-venv")
    assert "nothing to do" in capsys.readouterr().out
    assert installer.generation_numbers() == (1,)

    # upstream moves: update re-locks the new commit as generation 2
    (repo / "demo_nodes.py").write_text("NODES = [3]\n")
    new_head = _commit(repo, "upstream change")
    run_cli("--root", root, "update", "--no-venv", "--yes")
    out = capsys.readouterr().out
    assert "reinstall: demo" in out and "activated generation 2" in out
    lockfile = installer.current_lockfile()
    assert lockfile is not None
    assert lockfile.packs[0].source == f"git:{url}@{new_head}"
    served = Path(installer.packs_for_serving()[0].manifest).parent
    assert (served / "demo_nodes.py").read_text() == "NODES = [3]\n"

    with pytest.raises(SystemExit):
        run_cli("--root", root, "update", "--no-venv", "ghost")
    assert "not installed" in capsys.readouterr().err


def test_git_spec_parsing() -> None:
    from dinkster.manager import _parse_git_spec

    assert _parse_git_spec("git+https://host/a/b.git") == ("https://host/a/b.git", None)
    assert _parse_git_spec("git+https://host/a/b.git@v1.2") == ("https://host/a/b.git", "v1.2")
    assert _parse_git_spec("git+file:///tmp/repo@main") == ("file:///tmp/repo", "main")
    # in-URL @ never parses as a ref
    assert _parse_git_spec("git+ssh://git@host/a/b.git") == ("ssh://git@host/a/b.git", None)
    assert _parse_git_spec("git+ssh://git@host/a/b.git@dev") == ("ssh://git@host/a/b.git", "dev")


# ---------------------------------------------------------------------------
# Named registries: config, selection, routing, and pack@version resolution
# ---------------------------------------------------------------------------


def write_registries(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_registries_config_loads_and_refuses_loudly(tmp_path: Path) -> None:
    """The registries table parses strictly: canonical names only, http(s)
    endpoints, at most one default, no unknown keys - config is
    configuration, never best-effort."""
    from dinkster.registries import load_registries

    config = load_registries(
        write_registries(
            tmp_path / "registries.toml",
            '[registries.public]\nendpoint = "https://reg.example.org/"\ndefault = true\n\n'
            '[registries.corp]\nendpoint = "http://reg.corp.internal"\n'
            'token-env = "CORP_TOKEN"\n',
        )
    )
    assert config.names == ("public", "corp")
    public = config.default_registry()
    assert public is not None and public.name == "public"
    assert public.endpoint == "https://reg.example.org"  # trailing slash normalized
    corp = config.get("corp")
    assert corp is not None and corp.token_env == "CORP_TOKEN"
    assert corp.source == "registry:corp"
    assert config.get("Corp") is corp  # canonical lookup, one grammar

    for body, complaint in (
        ('[registries.corp]\nendpoint = "ftp://x"\n', "http\\(s\\)"),
        ('[registries.corp]\ntoken-env = "T"\n', "requires an endpoint"),
        ('[registries.Corp]\nendpoint = "http://x"\n', "lowercase"),  # one name grammar
        (
            # distinct TOML keys, one canonical identity
            '[registries."corp.x"]\nendpoint = "http://a"\n\n'
            '[registries.corp-x]\nendpoint = "http://b"\n',
            "names 'corp-x' twice",
        ),
        (
            '[registries.a]\nendpoint = "http://a"\ndefault = true\n\n'
            '[registries.b]\nendpoint = "http://b"\ndefault = true\n',
            "multiple defaults",
        ),
        ('[registries.a]\nendpoint = "http://a"\nurl = "http://b"\n', "unknown keys"),
        ("[other]\nx = 1\n", "unknown top-level"),
        ("not toml [", "not valid TOML"),
    ):
        with pytest.raises(InstallError, match=complaint):
            load_registries(write_registries(tmp_path / "bad.toml", body))

    with pytest.raises(InstallError, match="cannot read"):
        load_registries(tmp_path / "missing.toml")


def test_select_registry_never_searches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selection names exactly one registry: the selector, or the one
    default. Multiple registries without a default refuse loudly naming
    the choices - never an ordered fallback."""
    from dinkster.registries import (
        AD_HOC_TOKEN_ENV,
        RegistryConfig,
        load_registries,
        select_registry,
    )

    config = load_registries(
        write_registries(
            tmp_path / "registries.toml",
            '[registries.a]\nendpoint = "http://a.example"\n\n'
            '[registries.b]\nendpoint = "http://b.example"\n',
        )
    )
    with pytest.raises(InstallError, match="none is the default") as excinfo:
        select_registry(config, "")
    assert "a, b" in str(excinfo.value)  # the refusal names the choices
    assert select_registry(config, "b").endpoint == "http://b.example"
    with pytest.raises(InstallError, match="unknown registry 'ghost'.*a, b"):
        select_registry(config, "ghost")

    # a lone registry is the default without a flag
    [solo] = load_registries(
        write_registries(
            tmp_path / "solo.toml", '[registries.only]\nendpoint = "http://solo.example"\n'
        )
    ).registries
    assert select_registry(RegistryConfig((solo,)), "").name == "only"

    # ad-hoc URL: credential rides $DINKSTER_REGISTRY_TOKEN, provenance is the URL
    monkeypatch.setenv(AD_HOC_TOKEN_ENV, "adhoc-secret")
    ad_hoc = select_registry(RegistryConfig(), "http://direct.example/")
    assert ad_hoc.endpoint == "http://direct.example"
    assert ad_hoc.token() == "adhoc-secret"
    assert ad_hoc.source == "registry:http://direct.example"

    with pytest.raises(InstallError, match="no registry is configured"):
        select_registry(RegistryConfig(), "")


def test_parse_registry_spec_grammar() -> None:
    """pack@version parses only when both halves do - paths and git URLs
    never trip it, and names canonicalize."""
    from dinkster.registries import parse_registry_spec

    assert parse_registry_spec("img-tools@1.0.0") == ("img-tools", "1.0.0")
    assert parse_registry_spec("img.tools@2.10.3") == ("img-tools", "2.10.3")  # canonicalized
    for not_a_spec in (
        "demo",  # no version
        "demo@1.0",  # not major.minor.patch
        "demo@v1.0.0",  # not a strict version
        "Demo@1.0.0",  # the name grammar is lowercase
        "./demo@1.0.0",  # explicit path escape hatch
        "packs/demo@1.0.0",  # a path
        "c:\\packs\\demo@1.0.0",  # a windows path
        "@1.0.0",  # no name
    ):
        assert parse_registry_spec(not_a_spec) is None, not_a_spec


def _release_record(entry: LockedPack, **overrides: object) -> bytes:
    record: dict[str, object] = {
        "pack": entry.pack,
        "version": entry.version,
        "artifactDigest": entry.artifact_digest,
        "publisher": entry.publisher,
        "claims": list(entry.claims),
        "nodeTypes": [f"{entry.pack}.node"],
    }
    record.update(overrides)
    return json.dumps(record).encode()


def test_resolve_release_is_exact_or_refusal(tmp_path: Path) -> None:
    """Resolution asks one registry for the exact (pack, version) and
    cross-checks the answer: a 404 names the miss, a substituted or
    malformed record refuses - the index can confess or refuse, never
    redirect."""
    from dinkster.registries import NamedRegistry, resolve_release

    installer = make_installer(tmp_path)
    demo = write_pack(tmp_path / "demo", "demo")
    entry, _ = lock_local_pack(demo, installer.artifacts_dir)
    release = replace(entry, version="1.0.0", publisher="acme")
    good = "/index/packs/demo/versions/1.0.0"

    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts(
        {
            good: _release_record(release),
            "/index/packs/demo/versions/2.0.0": _release_record(release, version="9.9.9"),
            "/index/packs/demo/versions/3.0.0": _release_record(
                release, version="3.0.0", artifactDigest="md5:1"
            ),
            "/index/packs/demo/versions/4.0.0": b"not json",
            "/index/packs/demo/versions/5.0.0": _release_record(release, publisher=7),
        },
        requests,
    )
    try:
        registry = NamedRegistry(
            name="corp", endpoint=f"http://127.0.0.1:{server.server_port}", token_env="RESOLVE_T"
        )
        resolved = resolve_release(registry, "Demo", "1.0.0")
        assert resolved.pack == "demo" and resolved.version == "1.0.0"
        assert resolved.artifact_digest == entry.artifact_digest
        assert resolved.publisher == "acme"
        assert resolved.source == "registry:corp"
        assert requests[-1] == (good, None)  # token env unset: no header

        with pytest.raises(InstallError, match="publishes no release demo@0.0.1"):
            resolve_release(registry, "demo", "0.0.1")
        with pytest.raises(InstallError, match="refusing the substitution"):
            resolve_release(registry, "demo", "2.0.0")
        with pytest.raises(InstallError, match="artifact digest"):
            resolve_release(registry, "demo", "3.0.0")
        with pytest.raises(InstallError, match="invalid JSON"):
            resolve_release(registry, "demo", "4.0.0")
        with pytest.raises(InstallError, match="malformed release record"):
            resolve_release(registry, "demo", "5.0.0")
    finally:
        server.shutdown()


def test_browse_packs_pages_the_index_and_refuses_malformed(tmp_path: Path) -> None:
    """Browse asks ONE registry's pack index, parses rows strictly, and
    surfaces the query-bound keyset cursor; a malformed answer refuses
    instead of best-effort parsing."""
    from dinkster.registries import NamedRegistry, browse_packs

    page_one = {
        "packs": [
            {"pack": "img-tools", "publisher": "acme", "latestVersion": "2.0.0", "versions": 2}
        ],
        "cursor": "abc",
    }
    page_two = {
        "packs": [
            {"pack": "vid-tools", "publisher": "acme", "latestVersion": "1.0.0", "versions": 1}
        ]
    }
    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts(
        {
            "/index/packs?q=tools&limit=1": json.dumps(page_one).encode(),
            "/index/packs?q=tools&limit=1&cursor=abc": json.dumps(page_two).encode(),
            "/index/packs?limit=50": json.dumps({"packs": "nope"}).encode(),
        },
        requests,
    )
    try:
        registry = NamedRegistry(
            name="corp", endpoint=f"http://127.0.0.1:{server.server_port}", token_env="BROWSE_T"
        )
        first = browse_packs(registry, query="tools", limit=1)
        assert [entry.pack for entry in first.packs] == ["img-tools"]
        assert first.packs[0].latest_version == "2.0.0"
        assert first.packs[0].versions == 2
        assert first.cursor == "abc"

        second = browse_packs(registry, query="tools", limit=1, cursor=first.cursor)
        assert [entry.pack for entry in second.packs] == ["vid-tools"]
        assert second.cursor == ""  # listing complete

        with pytest.raises(InstallError, match="without a packs list"):
            browse_packs(registry)
        with pytest.raises(InstallError, match="failed: HTTP 404"):
            browse_packs(registry, query="missing-page")
    finally:
        server.shutdown()


def test_browse_templates_lists_the_remote_catalog(tmp_path: Path) -> None:
    """The template catalog browse: filters ride the query string, rows
    parse strictly (optional description/tags default), malformed rows
    refuse."""
    from dinkster.registries import NamedRegistry, browse_templates

    catalog = {
        "templates": [
            {
                "pack": "img-tools",
                "version": "2.0.0",
                "id": "starter",
                "name": "Starter",
                "description": "a first workflow",
                "tags": ["intro"],
                "digest": "blake3:" + "0" * 64,
            },
            {
                "pack": "img-tools",
                "version": "2.0.0",
                "id": "bare",
                "name": "Bare",
                "digest": "blake3:" + "1" * 64,
            },
        ]
    }
    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts(
        {
            "/index/templates?q=starter&tag=intro&pack=img-tools&limit=50": json.dumps(
                catalog
            ).encode(),
            "/index/templates?limit=50": json.dumps(
                {"templates": [{"pack": "img-tools"}]}
            ).encode(),
        },
        requests,
    )
    try:
        registry = NamedRegistry(
            name="corp", endpoint=f"http://127.0.0.1:{server.server_port}", token_env="BROWSE_T"
        )
        page = browse_templates(registry, query="starter", tag="intro", pack="img-tools")
        assert [entry.id for entry in page.templates] == ["starter", "bare"]
        assert page.templates[0].tags == ("intro",)
        assert page.templates[0].description == "a first workflow"
        assert page.templates[1].tags == ()  # optional fields default
        assert page.templates[1].description == ""
        assert page.cursor == ""

        with pytest.raises(InstallError, match="malformed row"):
            browse_templates(registry)
    finally:
        server.shutdown()


def test_routing_fetcher_dispatches_on_recorded_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fetcher, routed by each entry's source: bare 'registry' means
    the selected/default registry, registry:<name> the configured one
    (its credential riding along), registry:<url> that endpoint;
    unrouteable sources refuse naming the fix."""
    from dinkster.registries import load_registries, routing_registry_fetcher

    installer = make_installer(tmp_path)
    demo = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(demo, installer.artifacts_dir)
    data = archive.read_bytes()
    hex_digest = entry.artifact_digest.partition(":")[2]

    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts({f"/artifacts/{hex_digest}.zip": data}, requests)
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("CORP_TOKEN", "corp-secret")
        config = load_registries(
            write_registries(
                tmp_path / "registries.toml",
                f'[registries.corp]\nendpoint = "{endpoint}"\n'
                f'token-env = "CORP_TOKEN"\ndefault = true\n',
            )
        )
        fetch = routing_registry_fetcher(config, None)

        assert fetch(replace(entry, source="registry:corp")) == data
        assert requests[-1][1] == "Bearer corp-secret"
        assert fetch(replace(entry, source="registry")) == data  # default
        assert fetch(replace(entry, source=f"registry:{endpoint}")) == data
        assert requests[-1][1] == "Bearer corp-secret"  # endpoint matches corp

        with pytest.raises(InstallError, match="registry 'ghost'.*not configured.*corp"):
            fetch(replace(entry, source="registry:ghost"))
    finally:
        server.shutdown()

    from dinkster.registries import RegistryConfig

    with pytest.raises(InstallError, match="names no specific registry"):
        routing_registry_fetcher(RegistryConfig(), None)(replace(entry, source="registry"))


def test_cli_install_resolves_pack_version_from_named_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: 'dinkster-pack install demo@1.0.0' resolves the digest
    from the configured default registry's index, records
    registry:<name> provenance, and apply downloads + verifies the exact
    bytes. An unknown version refuses naming the miss; with several
    registries and no default, resolution refuses naming the choices."""
    scratch = make_installer(tmp_path / "scratch")
    demo = write_pack(tmp_path / "demo", "demo")
    entry, archive = lock_local_pack(demo, scratch.artifacts_dir)
    data = archive.read_bytes()
    release = replace(entry, version="1.0.0", publisher="acme")
    hex_digest = entry.artifact_digest.partition(":")[2]

    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts(
        {
            "/index/packs/demo/versions/1.0.0": _release_record(release),
            f"/artifacts/{hex_digest}.zip": data,
        },
        requests,
    )
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        root = tmp_path / "root"
        write_registries(
            root / "registries.toml",
            f'[registries.corp]\nendpoint = "{endpoint}"\n'
            f'token-env = "CORP_TOKEN"\ndefault = true\n',
        )
        monkeypatch.setenv("CORP_TOKEN", "corp-secret")

        run_cli("--root", str(root), "install", "--no-venv", "--yes", "demo@1.0.0")
        out = capsys.readouterr().out
        assert "add: demo" in out and "activated generation 1" in out
        lockfile = Installer(root).current_lockfile()
        assert lockfile is not None
        [locked] = lockfile.packs
        assert locked.source == "registry:corp"
        assert locked.version == "1.0.0" and locked.publisher == "acme"
        assert locked.artifact_digest == entry.artifact_digest
        # both the index lookup and the artifact download carried the credential
        assert [(path, auth) for path, auth in requests] == [
            ("/index/packs/demo/versions/1.0.0", "Bearer corp-secret"),
            (f"/artifacts/{hex_digest}.zip", "Bearer corp-secret"),
        ]

        with pytest.raises(SystemExit):
            run_cli("--root", str(root), "install", "--no-venv", "--yes", "demo@9.9.9")
        assert "publishes no release demo@9.9.9" in capsys.readouterr().err

        # several registries, no default: refuse naming the choices
        write_registries(
            root / "registries.toml",
            f'[registries.corp]\nendpoint = "{endpoint}"\n\n'
            f'[registries.other]\nendpoint = "http://other.example"\n',
        )
        with pytest.raises(SystemExit):
            run_cli("--root", str(root), "install", "--no-venv", "--yes", "demo@1.0.0")
        err = capsys.readouterr().err
        assert "none is the default" in err and "corp, other" in err
    finally:
        server.shutdown()


# -- shared content store ----------------------------------------------


def make_shared_installer(
    tmp_path: Path, name: str, shared: Path, *, accelerator: str = "cpu"
) -> Installer:
    return Installer(
        tmp_path / name,
        provision=fake_provision,
        freeze=fake_freeze,
        accelerator=accelerator,
        shared_store=shared,
    )


def hex_of(lockfile: Lockfile, index: int = 0) -> str:
    return lockfile.packs[index].artifact_digest.partition(":")[2]


def test_shared_store_two_roots_share_one_copy(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    first = make_shared_installer(tmp_path, "root-a", shared)
    second = make_shared_installer(tmp_path, "root-b", shared)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    target_a = local_lockfile(first, pack_dir)
    first.apply(target_a)
    second.apply(local_lockfile(second, pack_dir))
    digest = hex_of(target_a)
    # one copy of the content, under the shared store - never under a root
    assert (shared / "store" / digest / "dinkster-pack.toml").is_file()
    assert (shared / "artifacts" / f"{digest}.zip").is_file()
    assert not (first.root / "store").exists()
    assert not (second.root / "store").exists()
    # per-root state stays per root
    assert first.current_number() == 1
    assert second.current_number() == 1
    assert (first.root / "generations" / "1.json").is_file()
    assert (second.root / "generations" / "1.json").is_file()
    # serving manifests come from the shared store (managed classification
    # via store_root keeps working for live activation)
    spec = first.packs_for_serving()[0]
    assert Path(spec.manifest).is_relative_to(first.store_root)
    assert first.store_root == shared / "store"


def test_shared_store_pointer_persists_and_refuses_switch(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    make_shared_installer(tmp_path, "root-a", shared)
    # later constructions - dinkster-serve passes no flag - pick up the pointer
    resumed = Installer(tmp_path / "root-a", provision=fake_provision, freeze=fake_freeze)
    assert resumed.shared_store == shared.resolve()
    # a DIFFERENT store is a migration, not a flag: refuse loudly
    with pytest.raises(InstallError, match="refusing to silently switch"):
        make_shared_installer(tmp_path, "root-a", tmp_path / "other-store")


def test_shared_venvs_key_by_accelerator(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    cpu = make_shared_installer(tmp_path, "root-cpu", shared, accelerator="cpu")
    cuda = make_shared_installer(tmp_path, "root-cuda", shared, accelerator="cuda")
    pack_dir = write_pack(tmp_path / "demo", "demo")
    target = local_lockfile(cpu, pack_dir)
    cpu.apply(target)
    cuda.apply(local_lockfile(cuda, pack_dir))
    digest = hex_of(target)
    # same bytes, different accelerator resolution, different venv
    assert venv_python(shared / "venvs" / "cpu" / digest / "demo").is_file()
    assert venv_python(shared / "venvs" / "cuda" / digest / "demo").is_file()


def test_shared_gc_unions_references_across_registered_roots(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    first = make_shared_installer(tmp_path, "root-a", shared)
    second = make_shared_installer(tmp_path, "root-b", shared)
    only_b = local_lockfile(second, write_pack(tmp_path / "solo", "solo"))
    second.apply(only_b)
    # orphans: content no generation in ANY root references
    (shared / "store" / "feedface").mkdir()
    (shared / "artifacts" / "feedface.zip").write_bytes(b"zzz")
    (shared / "venvs" / "cpu" / "feedface").mkdir(parents=True)
    removed = first.gc()  # run from the root that references NOTHING
    assert set(removed) == {
        "store/feedface",
        "artifacts/feedface.zip",
        "venvs/cpu/feedface",
    }
    # root B's content survived a gc run from root A
    digest = hex_of(only_b)
    assert (shared / "store" / digest / "dinkster-pack.toml").is_file()
    assert second.packs_for_serving()


def test_shared_gc_counts_old_generations_of_other_roots(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    first = make_shared_installer(tmp_path, "root-a", shared)
    second = make_shared_installer(tmp_path, "root-b", shared)
    pack_dir = write_pack(tmp_path / "demo", "demo")
    old = local_lockfile(second, pack_dir)
    second.apply(old)
    (pack_dir / "demo_nodes.py").write_text("NODES = [2]\n")
    second.apply(local_lockfile(second, pack_dir))
    # B's generation 1 content is rollback material - A's gc must keep it
    assert first.gc() == ()
    assert (shared / "store" / hex_of(old)).is_dir()


def test_shared_gc_refuses_when_a_registered_root_is_missing(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    first = make_shared_installer(tmp_path, "root-a", shared)
    second = make_shared_installer(tmp_path, "root-b", shared)
    target = local_lockfile(second, write_pack(tmp_path / "demo", "demo"))
    second.apply(target)
    shutil.rmtree(second.root)  # unmounted? deleted? gc cannot tell
    with pytest.raises(InstallError, match="registered install root"):
        first.gc()
    with pytest.raises(InstallError, match="registered install root"):
        first.gc_candidates()
    # nothing was deleted
    assert (shared / "store" / hex_of(target)).is_dir()
    # forgetting the root is explicit: delete its registry entry, then gc
    roots_dir = shared / "roots"
    entries = [entry for entry in roots_dir.iterdir() if entry.is_file()]
    for entry in entries:
        if entry.read_text().strip() == str(second.root.resolve()):
            entry.unlink()
    removed = first.gc()
    assert f"store/{hex_of(target)}" in removed


def test_private_root_layout_is_unchanged(tmp_path: Path) -> None:
    installer = make_installer(tmp_path)
    target = local_lockfile(installer, write_pack(tmp_path / "demo", "demo"))
    installer.apply(target)
    digest = hex_of(target)
    assert installer.shared_store is None
    assert (installer.root / "store" / digest / "dinkster-pack.toml").is_file()
    assert venv_python(installer.root / "venvs" / digest / "demo").is_file()
    assert not (installer.root / "shared-store").exists()


def test_cli_shared_store_flag_and_gc_across_roots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = tmp_path / "shared"
    pack_dir = write_pack(tmp_path / "demo", "demo")
    run_cli(
        "--root",
        str(tmp_path / "root-a"),
        "--shared-store",
        str(shared),
        "--accelerator",
        "cpu",
        "install",
        "--no-venv",
        "--yes",
        str(pack_dir),
    )
    capsys.readouterr()
    # root B joins the same store; the flag is remembered in each root
    run_cli(
        "--root",
        str(tmp_path / "root-b"),
        "--shared-store",
        str(shared),
        "--accelerator",
        "cpu",
        "status",
    )
    capsys.readouterr()
    # gc from root B (which references nothing) keeps A's content
    run_cli("--root", str(tmp_path / "root-b"), "--accelerator", "cpu", "gc", "--yes")
    out = capsys.readouterr().out
    assert "shared store:" in out
    assert "0 unreferenced item(s) removed" in out
    store_dirs = [child for child in (shared / "store").iterdir() if child.is_dir()]
    assert len(store_dirs) == 1
