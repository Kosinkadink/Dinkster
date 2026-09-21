"""dinkster-pack: manage an install root from the shell.

The thin CLI over :class:`dinkster.installer.Installer` - what a manager UI
will eventually call, usable today. Every mutation is a new generation;
``dinkster-serve --install-root`` picks up whatever generation is current at
startup.

PLAN/APPLY discipline (hard requirement, user-directed 2026-07): every
mutating command first prints the exact plan - per pack the action,
versions, digest pin, provenance, and whether a NEW venv would be staged,
plus the isolation cost the user pays (one worker process per pack;
~300-450 MiB VRAM per idle GPU-touched process, measured) - and mutates
only on explicit confirmation: an interactive yes, ``--yes``, or applying
a previously written plan file (``--plan FILE`` writes it read-only;
``dinkster-pack apply FILE`` executes exactly that plan). Noninteractive use
without one of those refuses loudly - planning never silently collapses
into mutation. A plan file is applied only if the installation still
digests to the base it was computed against; a stale plan is rejected and
must be replanned, never partially applied.

Resolving a source (git clone, local archive) happens at PLAN time and
writes only content-addressed artifacts - inert until a generation
references them, reclaimable by gc if the plan is abandoned - so an
applied plan installs the exact bytes the user reviewed, not a re-resolve
that may have drifted.

Install sources: local pack directories, git URLs
(``git+<url>[@<ref>]``, pip's convention) - both archived through the
same deterministic ``build_artifact`` path registry downloads are
verified against, both locking the exact content digest - and
``pack@version``, resolved against exactly ONE registry's index (the
``--registry`` choice or the configured default; never a search across
registries - see ``dinkster.registries``) into a digest-pinned entry whose
bytes download and verify at apply. ``update``
re-resolves each installed pack from its recorded provenance (git clones
the default branch head, local re-archives the directory) and applies the
result as one new generation. Registry-sourced entries are ACQUIRABLE
(``--registry`` names a configured registry or an endpoint URL;
``registries.toml`` maps names to endpoints + per-registry credential
env vars; downloads are content-addressed by the pinned digest and
verified before admission), so restore/reproduce/apply can re-fetch
them from the registry each entry's provenance records.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from dinkster_registry import (
    ArtifactError,
    InstallError,
    InstallPlan,
    LockedPack,
    Lockfile,
    PlanRecord,
    SnapshotRecord,
    plan,
    validate_artifact_digest,
    validate_version,
)
from dinkster_registry.artifact import build_artifact
from dinkster_schema import canonical_name
from dinkster_workers import (
    KNOWN_ACCELERATORS,
    AcceleratorError,
    ManifestError,
    bare_pin,
    load_manifest,
    partition_portable,
    resolve_accelerator,
)
from dinkster_workers.doctor import diagnose, prepare_catalog, render_text

from . import __version__ as dinkster_version
from .compose import CompositionError, PackSpec, resolve_manifest_path
from .installer import (
    LOCAL_PUBLISHER,
    LOCAL_VERSION,
    Installer,
    VenvSpec,
    lock_git_pack,
    lock_local_pack,
)
from .pack_archive import PackArchiveError, build_pack_archive
from .registries import (
    DEFAULT_REGISTRIES_FILENAME,
    RegistryConfig,
    RegistryPublishError,
    browse_packs,
    browse_templates,
    load_registries,
    parse_registry_spec,
    publish_release,
    resolve_release,
    routing_registry_fetcher,
    select_registry,
)
from .registry_declaration import (
    RegistryDeclarationError,
    load_registry_contributions,
    registry_declaration_toml,
    write_registry_declaration,
)

GIT_PREFIX = "git+"

ISOLATION_NOTE = (
    "one isolated worker process per pack; an idle GPU-touched process "
    "holds ~300-450 MiB VRAM for its lifetime"
)


def _parse_git_spec(spec: str) -> tuple[str, str | None]:
    """``git+<url>[@<ref>]`` -> (url, ref). The ref separator is a final
    ``@`` whose suffix has no ``/`` - so ``ssh://git@host/repo`` and other
    in-URL ``@``s never parse as refs."""
    raw = spec[len(GIT_PREFIX) :]
    url, sep, ref = raw.rpartition("@")
    if sep and ref and "/" not in ref and ":" not in ref:
        return url, ref
    return raw, None


def _registry_config(args: argparse.Namespace) -> RegistryConfig:
    """The named-registries table: an explicit --registries path must
    exist and parse; the default location (<root>/registries.toml) is
    read only when present."""
    if args.registries:
        return load_registries(Path(args.registries))
    if args.root:
        default_path = Path(args.root) / DEFAULT_REGISTRIES_FILENAME
        if default_path.is_file():
            return load_registries(default_path)
    return RegistryConfig()


def _installer(args: argparse.Namespace, *, accelerator: str | None = None) -> Installer:
    if not args.root:
        raise InstallError("no install root: pass --root or set $DINKSTER_INSTALL_ROOT")
    workspace = [Path(entry) for entry in args.workspace_package]
    selected = accelerator if accelerator else resolve_accelerator(args.accelerator)
    config = _registry_config(args)
    chosen = select_registry(config, args.registry) if args.registry else None
    registry_fetch = (
        routing_registry_fetcher(config, chosen)
        if chosen is not None or config.registries
        else None
    )
    return Installer(
        Path(args.root),
        workspace_packages=workspace,
        accelerator=selected,
        registry_fetch=registry_fetch,
        shared_store=Path(args.shared_store) if args.shared_store else None,
    )


def _print_plan(installer: Installer, steps: InstallPlan, *, venvs: bool) -> None:
    """The reviewable value: per pack the action, versions, digest pin,
    provenance, venv work, and platform/accelerator compatibility; then
    the isolation cost - the real prices the user pays, visible BEFORE
    anything runs, never decided silently."""
    current_number = installer.current_number()
    base = f"generation {current_number}" if current_number is not None else "empty root"
    print(f"plan ({base} -> new generation, accelerator: {installer.accelerator}):")
    isolated = 0
    warnings: list[str] = []
    for step in steps.steps:
        if step.to is None:
            print(f"{step.action}: {step.pack} (was {step.from_version})")
            continue
        entry = step.to
        isolated += 1
        detail = entry.version
        if step.from_version is not None and step.from_version != entry.version:
            detail = f"{step.from_version} -> {entry.version}"
        parts = [detail, entry.artifact_digest[:19], f"[{entry.source}]"]
        if venvs:
            parts.append("venv: reuse" if installer.venv_staged(entry) else "venv: new")
        manifest = installer.manifest_of(entry)
        if manifest is not None:
            extra = manifest.requires_for(installer.accelerator)[len(manifest.requires) :]
            if extra:
                parts.append(f"+{len(extra)} {installer.accelerator} requirement(s)")
            if manifest.platforms and sys.platform not in manifest.platforms:
                warnings.append(
                    f"WARNING: {entry.pack} declares platforms "
                    f"{', '.join(manifest.platforms)} - this host is {sys.platform}. "
                    f"The author does not claim it works here; installing anyway "
                    f"is your call."
                )
        print(f"{step.action}: {step.pack} " + "  ".join(parts))
    for warning in warnings:
        print(warning)
    if isolated:
        print(f"isolation: {ISOLATION_NOTE}")


def _confirmed(args: argparse.Namespace) -> bool:
    """Explicit confirmation, or a loud refusal - never a silent default.
    Interactive: ask. Noninteractive: only --yes (or a plan file applied
    later) may mutate; anything else refuses with the ways forward."""
    if args.yes:
        return True
    if sys.stdin.isatty():
        return input("apply this plan? [y/N] ").strip().lower() in ("y", "yes")
    hint = "re-run with --yes"
    if getattr(args, "plan_file", None) is not None:
        hint += ", or write the plan with --plan FILE and run 'dinkster-pack apply FILE'"
    raise InstallError(f"refusing to mutate without confirmation: {hint}")


def _matching_restore_groups(
    installer: Installer,
    target: Lockfile,
    recorded: Sequence[tuple[str, tuple[str, ...]]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Snapshot groups still declared identically by current policy."""
    current = dict(installer.hosting_groups(target))
    matching: list[tuple[str, tuple[str, ...]]] = []
    for name, members in recorded:
        if current.get(name) == members:
            matching.append((name, members))
        else:
            print(
                f"venv group {name}: current hosting.toml does not declare the "
                f"identical member set; falling back to per-pack provisioning"
            )
    if not recorded and current:
        print(
            "snapshot records no venv groups; falling back to per-pack "
            "provisioning instead of inventing current-policy groups"
        )
    return tuple(matching)


def _validated_plan_groups(
    installer: Installer,
    target: Lockfile,
    recorded: Sequence[tuple[str, tuple[str, ...]]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Refuse a reviewed grouped plan when current policy no longer permits it."""
    current = dict(installer.hosting_groups(target))
    for name, members in recorded:
        if current.get(name) != members:
            raise InstallError(
                f"plan is stale: hosting.toml no longer declares venv group "
                f"{name!r} with the reviewed member set; re-plan"
            )
    return tuple(recorded)


def _validated_plan_topology(
    installer: Installer,
    target: Lockfile,
    groups: Sequence[tuple[str, tuple[str, ...]]],
    in_process: Sequence[str],
    runtime_pins: Mapping[str, str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Refuse when live policy/runtime no longer match the reviewed topology."""
    validated = _validated_plan_groups(installer, target, groups)
    policy = installer.hosting_topology(target)
    if policy.groups != tuple(groups):
        raise InstallError("plan is stale: hosting.toml venv-group topology changed; re-plan")
    if policy.in_process != tuple(in_process):
        raise InstallError("plan is stale: hosting.toml in-process placement changed; re-plan")
    if in_process and installer.serving_runtime_pins() != dict(runtime_pins):
        raise InstallError("plan is stale: serving runtime changed since review; re-plan")
    return validated


def _finish(
    args: argparse.Namespace,
    installer: Installer,
    target: Lockfile,
    *,
    hosting_groups: Sequence[tuple[str, tuple[str, ...]]] | None = None,
    hosting_in_process: Sequence[str] | None = None,
    hosting_runtime_pins: Mapping[str, str] | None = None,
    generation_topology: bool = False,
) -> None:
    """The shared plan/apply tail of every mutating command. Entries
    whose bytes are not local yet (registry-resolved pins carry only the
    digest) are acquired between confirmation and apply, digest-verified;
    unacquirable ones refuse at plan time, naming the fix."""
    current = installer.current_lockfile()
    steps = plan(current if current is not None else Lockfile(), target)
    venvs = not args.no_venv
    if hosting_groups is None and hosting_in_process is None:
        policy = installer.hosting_topology(target)
        groups = policy.groups if venvs else ()
        in_process = policy.in_process
        runtime_pins = installer.serving_runtime_pins() if in_process else {}
    else:
        groups = tuple(hosting_groups or ()) if venvs else ()
        in_process = tuple(hosting_in_process or ())
        runtime_pins = dict(hosting_runtime_pins or {})
    topology_change = (
        groups != installer.current_groups()
        or in_process != installer.current_in_process()
        or runtime_pins != installer.current_runtime_pins()
    )
    if steps.empty and not topology_change:
        print("nothing to do; the installation already matches")
        return
    missing = [entry for entry in target.packs if not installer.artifact_available(entry)]
    stuck = sorted(entry.pack for entry in missing if not installer.acquirable(entry))
    if stuck:
        raise InstallError(
            f"artifacts are not available locally for: {', '.join(stuck)}, "
            f"and their recorded provenance cannot be fetched (registry "
            f"sources need --registry or $DINKSTER_REGISTRY)"
        )
    if steps.empty:
        print("plan: restage the current lockfile with the declared venv topology")
    else:
        _print_plan(installer, steps, venvs=venvs)
    recorded_groups = groups if groups or installer.current_groups() else None
    if args.plan_file:
        record = PlanRecord(
            target=target,
            base=current.record_digest() if current is not None else None,
            venvs=venvs,
            accelerator=installer.accelerator,
            acquire=bool(missing),
            allow_doctor_findings=args.allow_doctor_findings,
            venv_groups=recorded_groups,
            in_process=in_process,
            runtime_pins=tuple(sorted(runtime_pins.items())),
            generation_topology=generation_topology,
        )
        path = Path(args.plan_file)
        path.write_text(record.record_json() + "\n")
        print(f"plan written to {path}; apply with: dinkster-pack apply {path}")
        return
    if not _confirmed(args):
        print("plan not applied; nothing changed")
        return
    for entry in missing:
        print(
            f"acquire {entry.pack} from {entry.source} "
            f"(must hash to {entry.artifact_digest[:19]}...)"
        )
        installer.acquire(entry)
        print(f"acquired {entry.pack}: digest verified")
    number, _ = installer.apply(
        target,
        venvs=venvs,
        allow_doctor_findings=args.allow_doctor_findings,
        hosting_groups=groups,
        hosting_in_process=in_process,
        hosting_runtime_pins=runtime_pins,
    )
    print(f"activated generation {number}")


def _cmd_install(args: argparse.Namespace) -> None:
    installer = _installer(args)
    current = installer.current_lockfile()
    entries = {entry.pack: entry for entry in (current.packs if current is not None else ())}
    for raw in args.packs:
        spec = None if raw.startswith(GIT_PREFIX) else parse_registry_spec(raw)
        if raw.startswith(GIT_PREFIX):
            url, ref = _parse_git_spec(raw)
            entry, _ = lock_git_pack(url, installer.artifacts_dir, ref=ref)
        elif spec is not None:
            # pack@version: resolved against exactly ONE registry - the
            # named/default one - never a search across registries.
            registry = select_registry(_registry_config(args), args.registry)
            entry = resolve_release(registry, *spec)
        else:
            pack_dir = resolve_manifest_path(raw).parent
            entry, _ = lock_local_pack(pack_dir, installer.artifacts_dir)
        entries[entry.pack] = entry
    _finish(args, installer, Lockfile.of(list(entries.values())))


def _cmd_remove(args: argparse.Namespace) -> None:
    installer = _installer(args)
    current = installer.current_lockfile()
    if current is None:
        raise InstallError("nothing is installed")
    remaining = list(current.packs)
    for name in args.packs:
        entry = current.get(name)
        if entry is None:
            raise InstallError(f"pack {name!r} is not installed")
        remaining.remove(entry)
    _finish(args, installer, Lockfile.of(remaining))


def _cmd_update(args: argparse.Namespace) -> None:
    installer = _installer(args)
    current = installer.current_lockfile()
    if current is None:
        raise InstallError("nothing is installed")
    only = {canonical_name(name) for name in args.packs}
    for name in only:
        if current.get(name) is None:
            raise InstallError(f"pack {name!r} is not installed")
    entries = []
    for entry in current.packs:
        if only and entry.pack not in only:
            entries.append(entry)
        elif entry.source.startswith("git:"):
            url = entry.source[len("git:") :].rpartition("@")[0]
            refreshed, _ = lock_git_pack(url, installer.artifacts_dir)
            entries.append(refreshed)
        elif entry.source.startswith("local:"):
            pack_dir = Path(entry.source[len("local:") :])
            refreshed, _ = lock_local_pack(pack_dir, installer.artifacts_dir)
            entries.append(refreshed)
        else:
            entries.append(entry)  # registry sources: resolution needs the service
    _finish(args, installer, Lockfile.of(entries))


def _cmd_publish(args: argparse.Namespace) -> None:
    """Preflight, archive, upload, and submit one pack for admission."""
    problem = validate_version(args.version)
    if problem is not None:
        raise RegistryPublishError(f"version {args.version!r} {problem}")
    manifest_path = resolve_manifest_path(args.pack)
    manifest = load_manifest(manifest_path)
    if not args.no_preflight:
        report = diagnose(manifest_path)
        if not report.ok:
            print(render_text(report))
            raise SystemExit(1)
    registry = select_registry(_registry_config(args), args.registry)
    with tempfile.TemporaryDirectory(prefix="dinkster-publish-") as scratch:
        archive = Path(scratch) / "artifact.zip"
        try:
            digest = build_artifact(manifest_path.parent, archive)
        except ArtifactError as exc:
            raise RegistryPublishError(str(exc)) from exc
        verdict = publish_release(registry, archive, digest, args.version)
    for finding in verdict.findings:
        print(f"{finding.code}: {finding.message}")
    if verdict.state == "rejected":
        raise SystemExit(1)
    print(f"published {manifest.name} {args.version} {digest}")


def _claims_from_manifests(installer: Installer, target: Lockfile) -> Lockfile:
    """Replace each entry's claims with the ones its pinned bytes'
    manifest declares, and revalidate the whole lockfile - claims come
    from the exact bytes, never from the planning input (a workflow's
    environment stamp or a plan file). Refuses mislabeled entries (a
    name pinned to some other pack's bytes). Every artifact must be
    available; callers acquire first."""
    entries: list[LockedPack] = []
    for entry in target.packs:
        manifest = installer.manifest_of(entry)
        if manifest is None:
            raise InstallError(f"artifact for {entry.pack} vanished before apply; re-run")
        if canonical_name(manifest.name) != entry.pack:
            raise InstallError(
                f"the stamp names {entry.pack!r} but the pinned bytes are pack "
                f"{canonical_name(manifest.name)!r}; refusing a mislabeled stamp"
            )
        entries.append(
            replace(entry, claims=tuple(canonical_name(claim) for claim in manifest.namespaces))
        )
    return Lockfile.of(entries)


def _cmd_apply(args: argparse.Namespace) -> None:
    """Execute a previously written plan - the noninteractive opt-in. The
    plan file IS the confirmation; what apply enforces is that it still
    describes reality: the base must match the current installation (a
    stale plan is rejected and replanned, never partially applied) and
    every pinned artifact must still be present - unless the record says
    ``acquire``, in which case missing artifacts are re-fetched from
    their recorded provenance, digest-verified, as part of the reviewed
    plan. The plan's recorded accelerator wins over the apply-time
    selection - what runs is what was reviewed."""
    path = Path(args.plan_file)
    if not path.is_file():
        raise InstallError(f"no plan file at {path}")
    record = PlanRecord.from_record_json(path.read_text())
    installer = _installer(args, accelerator=record.accelerator or None)
    current = installer.current_lockfile()
    if not record.matches_base(current):
        found = current.record_digest()[:19] if current is not None else "empty root"
        expected = record.base[:19] if record.base is not None else "empty root"
        raise InstallError(
            f"plan is stale: it was computed against {expected} but the "
            f"installation is now {found}; re-plan and review again"
        )
    missing = [entry for entry in record.target.packs if not installer.artifact_available(entry)]
    if missing and not record.acquire:
        names = ", ".join(sorted(entry.pack for entry in missing))
        raise InstallError(
            f"plan artifacts are no longer available for: {names} "
            f"(gc may have reclaimed them); re-plan to re-resolve the sources"
        )
    stuck = sorted(entry.pack for entry in missing if not installer.acquirable(entry))
    if stuck:
        raise InstallError(
            f"plan artifacts are not available locally for: {', '.join(stuck)}, "
            f"and their recorded provenance cannot be re-fetched (registry "
            f"sources need --registry or $DINKSTER_REGISTRY); re-plan to "
            f"re-resolve the sources"
        )
    steps = plan(current if current is not None else Lockfile(), record.target)
    recorded_groups = record.venv_groups if record.venv_groups is not None else ()
    groups = (
        tuple(recorded_groups)
        if record.generation_topology
        else _validated_plan_topology(
            installer,
            record.target,
            recorded_groups,
            record.in_process,
            dict(record.runtime_pins),
        )
    )
    topology_change = (
        (record.venvs and installer.current_groups() != groups)
        or installer.current_in_process() != record.in_process
        or installer.current_runtime_pins() != dict(record.runtime_pins)
    )
    if steps.empty and not topology_change:
        print("nothing to do; the installation already matches")
        return
    if steps.empty:
        print("plan: restage the current lockfile with the reviewed venv topology")
    else:
        _print_plan(installer, steps, venvs=record.venvs)
    for entry in missing:
        print(
            f"acquire {entry.pack} from {entry.source} "
            f"(must hash to {entry.artifact_digest[:19]}...)"
        )
        installer.acquire(entry)
        print(f"acquired {entry.pack}: digest verified")
    target = record.target
    if record.derive_claims:
        target = _claims_from_manifests(installer, target)
    number, _ = installer.apply(
        target,
        venvs=record.venvs,
        venv_specs=dict(record.venv_specs),
        allow_doctor_findings=record.allow_doctor_findings,
        hosting_groups=groups,
        hosting_in_process=record.in_process,
        hosting_runtime_pins=dict(record.runtime_pins),
    )
    print(f"activated generation {number}")


def _cmd_snapshot(args: argparse.Namespace) -> None:
    """Capture the environment as a complete record - the lockfile plus
    per-venv freezes plus host scope. Read-only: the Comfy-Desktop lesson
    made native."""
    installer = _installer(args)
    record = installer.snapshot(hashes=bool(getattr(args, "hashes", False)))
    pinned = {pack for pack, _ in record.venvs}
    unpinned = [entry.pack for entry in record.lockfile.packs if entry.pack not in pinned]
    print(
        f"snapshot: {len(record.lockfile.packs)} pack(s), "
        f"{len(pinned)} venv(s) pinned, python {record.python}, "
        f"{record.platform}, accelerator {record.accelerator}"
    )
    if getattr(args, "hashes", False):
        annotated = sum(1 for _, pins in record.venvs for pin in pins if " " in pin)
        bare = sum(1 for _, pins in record.venvs for pin in pins if " " not in pin)
        print(
            f"hashes: {annotated} pin(s) hash-annotated"
            + (
                f"; {bare} environment-specific pin(s) stay bare "
                f"(vendor-index builds the index cannot vouch for)"
                if bare
                else ""
            )
        )
    if record.runtime:
        facts = ", ".join(f"{key} {value}" for key, value in record.runtime)
        print(f"runtime (advisory): {facts}")
    if unpinned:
        print(f"unpinned (no venv staged): {', '.join(sorted(unpinned))}")
    path = Path(args.out)
    path.write_text(record.record_json() + "\n")
    print(f"snapshot written to {path}; restore with: dinkster-pack restore {path}")


def _cmd_restore(args: argparse.Namespace) -> None:
    """Re-create a snapshotted environment through the same plan/apply
    discipline as every mutation. Missing pack bytes are re-acquired
    from their recorded provenance after confirmation, digest-verified;
    --plan writes a PlanRecord carrying acquire + the computed venv
    specs, so a deferred 'dinkster-pack apply' provisions exactly what was
    reviewed. Venvs MISSING on this machine are provisioned from the
    snapshot's exact pins, venvs already staged are shared
    content-addressed state and are verified + reported, never silently
    rebuilt.

    SCOPE RULE (cross-platform/accelerator honesty): exact pins are
    reused only when the snapshot's recorded platform and accelerator
    match this host. On mismatch pins are PARTITIONED by the one signal
    that is data in the pin itself, never a name heuristic: a PEP 440
    local version label (torch==2.5.1+cu124) marks a vendor-specific
    build that cannot travel and is dropped; label-free pins are
    portable VERSIONS and ride along as pip constraints - manifest
    ranges plus this host's accelerator requirements still decide WHAT
    installs, constraints bind versions of whatever resolution pulls,
    and a constraint nothing pulls (a source-scope vendor runtime
    wheel) is inert. The restore still says loudly that it is a
    starting point, not an exact reproduction, and --fresh-pins drops
    everything when a kept constraint cannot resolve on this host."""
    installer = _installer(args)
    path = Path(args.snapshot_file)
    if not path.is_file():
        raise InstallError(f"no snapshot file at {path}")
    record = SnapshotRecord.from_record_json(path.read_text())
    replay_groups = _matching_restore_groups(installer, record.lockfile, record.venv_groups)
    host = f"{sys.platform}-{platform.machine()}"
    mismatches: list[str] = []
    if record.platform and record.platform != host:
        mismatches.append(f"platform: snapshot {record.platform}, host {host}")
    if record.accelerator and record.accelerator != installer.accelerator:
        mismatches.append(
            f"accelerator: snapshot {record.accelerator}, host {installer.accelerator}"
        )
    use_pins = not mismatches
    fresh_pins = bool(getattr(args, "fresh_pins", False))
    if mismatches:
        print("SCOPE MISMATCH - exact pins will NOT be reused:")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        if fresh_pins:
            print(
                f"--fresh-pins: ALL snapshot pins dropped; missing venvs "
                f"re-resolve manifest ranges plus {installer.accelerator} "
                f"requirements. This restore is a starting point, not an "
                f"exact reproduction."
            )
        else:
            print(
                f"environment-specific pins (PEP 440 local version labels, "
                f"e.g. +cu124) are dropped; portable pins ride along as "
                f"version constraints while manifest ranges plus "
                f"{installer.accelerator} requirements decide what installs. "
                f"This restore is a starting point, not an exact "
                f"reproduction. If a kept constraint cannot resolve on this "
                f"host, re-run with --fresh-pins."
            )
    elif fresh_pins:
        use_pins = False
        print(
            "--fresh-pins: snapshot pins ignored; missing venvs re-resolve "
            f"manifest ranges plus {installer.accelerator} requirements"
        )
    elif not record.accelerator:
        print(
            "note: snapshot does not record its accelerator (older format); "
            f"pins will be reused as-is on this {installer.accelerator} host"
        )
    # Runtime facts are ADVISORY narration only: a driver/toolkit bump is
    # not a scope, so a difference is said out loud but never changes
    # use_pins or the portable/environment-specific partition.
    if record.runtime:
        host_runtime = dict(installer.runtime_facts())
        drifted = [
            f"{key}: snapshot {value}, host {host_runtime.get(key, 'unknown')}"
            for key, value in record.runtime
            if host_runtime.get(key) != value
        ]
        if drifted:
            print("runtime differs (advisory only; does not change pin reuse):")
            for line in drifted:
                print(f"  {line}")
    missing = [entry for entry in record.lockfile.packs if not installer.artifact_available(entry)]
    stuck = sorted(entry.pack for entry in missing if not installer.acquirable(entry))
    if stuck:
        raise InstallError(
            f"snapshot pack bytes are not available locally for: "
            f"{', '.join(stuck)}, and their recorded provenance cannot be "
            f"re-acquired (registry sources need --registry or "
            f"$DINKSTER_REGISTRY) - stage "
            f"those artifacts first"
        )
    # Re-acquisition is part of the reviewed plan: fetch happens only
    # after confirmation, and apply admits only bytes that hash to the
    # recorded digest - never a silent substitution.
    for entry in missing:
        print(
            f"re-acquire {entry.pack} from {entry.source} "
            f"(must hash to {entry.artifact_digest[:19]}...)"
        )
    current = installer.current_lockfile()
    steps = plan(current if current is not None else Lockfile(), record.lockfile)
    venvs = not args.no_venv
    if steps.empty:
        print("lockfile already matches the snapshot")
    else:
        _print_plan(installer, steps, venvs=venvs)
    specs: dict[str, VenvSpec] = {}
    if venvs:
        for entry in record.lockfile.packs:
            pins = record.pins_for(entry.pack)
            if pins is None:
                print(f"venv {entry.pack}: unpinned in snapshot; ranges will resolve fresh")
                continue
            if use_pins:
                specs[entry.pack] = VenvSpec(exact=pins)
                drift = installer.venv_drift(entry, pins)
                if drift is None:
                    hashed = sum(1 for pin in pins if " " in pin)
                    print(
                        f"venv {entry.pack}: will provision from {len(pins)} "
                        f"snapshot pin(s)" + (f" ({hashed} hash-verified)" if hashed else "")
                    )
                elif drift:
                    print(
                        f"venv {entry.pack}: already staged but drifts from the "
                        f"snapshot in {len(drift)} dist(s) (e.g. {drift[0]}); "
                        f"shared venvs are not rebuilt - gc unreferenced content "
                        f"and restore again for an exact match"
                    )
                continue
            if fresh_pins:
                print(
                    f"venv {entry.pack}: {len(pins)} snapshot pin(s) dropped "
                    f"(--fresh-pins); ranges + {installer.accelerator} "
                    f"requirements will resolve fresh"
                )
                continue
            portable, environment_specific = partition_portable(pins)
            # Constraints bind VERSIONS of whatever resolution pulls;
            # hash annotations are artifact identity and drop with the
            # exactness they belong to.
            specs[entry.pack] = VenvSpec(constraints=tuple(bare_pin(pin) for pin in portable))
            dropped = ", ".join(bare_pin(pin).partition("==")[0] for pin in environment_specific)
            print(
                f"venv {entry.pack}: {len(portable)} portable pin(s) kept as "
                f"version constraints; {len(environment_specific)} "
                f"environment-specific pin(s) dropped"
                + (f" ({dropped})" if dropped else "")
                + f"; ranges + {installer.accelerator} requirements decide "
                f"what installs"
            )
    topology_change = (
        (venvs and installer.current_groups() != replay_groups)
        or installer.current_in_process() != record.in_process
        or installer.current_runtime_pins() != dict(record.runtime_pins)
    )
    if steps.empty and not topology_change:
        return
    if steps.empty:
        print("venv topology differs; the current lockfile will be restaged")
    if args.plan_file:
        plan_record = PlanRecord(
            target=record.lockfile,
            base=current.record_digest() if current is not None else None,
            venvs=venvs,
            accelerator=installer.accelerator,
            acquire=True,
            venv_specs=tuple(specs.items()),
            allow_doctor_findings=args.allow_doctor_findings,
            venv_groups=replay_groups,
            in_process=record.in_process,
            runtime_pins=record.runtime_pins,
            generation_topology=True,
        )
        plan_path = Path(args.plan_file)
        plan_path.write_text(plan_record.record_json() + "\n")
        print(f"plan written to {plan_path}; apply with: dinkster-pack apply {plan_path}")
        return
    if not _confirmed(args):
        print("plan not applied; nothing changed")
        return
    for entry in missing:
        installer.acquire(entry)
        print(f"re-acquired {entry.pack}: digest verified")
    number, _ = installer.apply(
        record.lockfile,
        venvs=venvs,
        venv_specs=specs,
        allow_doctor_findings=args.allow_doctor_findings,
        hosting_groups=replay_groups,
        hosting_in_process=record.in_process,
        hosting_runtime_pins=dict(record.runtime_pins),
    )
    print(f"activated generation {number}")


def _environment_stamp(text: str) -> tuple[str, dict[str, dict[str, str]]]:
    """Parse a workflow document's environment stamp into (stamped dinkster
    version, packId -> provenance fields). The stamp is written by a
    saving client from the live packs table; provenance fields are
    omitted-when-unknown strings (omission MEANS unpinned). Refuses
    loudly when the document carries no stamp - an unstamped workflow
    records nothing to reproduce from."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InstallError(f"workflow file is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise InstallError("workflow file must be a JSON object")
    environment = cast("dict[str, object]", document).get("environment")
    if environment is None:
        raise InstallError(
            "workflow has no environment stamp; nothing records what to "
            "reproduce (stamps are written on save by a stamping client)"
        )
    if not isinstance(environment, dict):
        raise InstallError("workflow 'environment' must be an object")
    stamp = cast("dict[str, object]", environment)
    packs_raw = stamp.get("packs")
    if not isinstance(packs_raw, dict) or not packs_raw:
        raise InstallError("environment stamp records no packs")
    packs: dict[str, dict[str, str]] = {}
    for pack_id, fields_raw in cast("dict[str, object]", packs_raw).items():
        if not isinstance(fields_raw, dict):
            raise InstallError(f"environment stamp for pack {pack_id!r} must be an object")
        fields = cast("dict[str, object]", fields_raw)
        entry: dict[str, str] = {}
        for key in ("version", "artifactDigest", "source", "publisher"):
            value = fields.get(key)
            if value is not None and not isinstance(value, str):
                raise InstallError(
                    f"environment stamp for pack {pack_id!r}: {key!r} must be a string"
                )
            if value:
                entry[key] = value
        packs[pack_id] = entry
    stamped_dinkster = ""
    dinkster_raw = stamp.get("dinkster")
    if isinstance(dinkster_raw, dict):
        version = cast("dict[str, object]", dinkster_raw).get("version")
        if isinstance(version, str):
            stamped_dinkster = version
    return stamped_dinkster, packs


def _cmd_reproduce(args: argparse.Namespace) -> None:
    """Rebuild an installation from a workflow's environment stamp.
    The stamp is an advisory RECORD everywhere
    else; here - and only here - it becomes an explicit user-requested
    install target containing exactly the stamped packs. Strictly
    plan/apply with a READ-ONLY plan phase: nothing is fetched, archived,
    or written before confirmation - re-acquisition from stamped
    provenance is narrated in the plan and happens on apply, admitting
    only bytes that hash to the recorded digest (restore's convention).
    --plan writes a PlanRecord with acquire + deriveClaims set, so a
    deferred 'dinkster-pack apply' performs the same fetch-and-derive.
    The workflow never carries authority: namespace claims come from the
    acquired manifests, never from workflow JSON, and a stamp whose
    pinned bytes are some other pack's refuses as mislabeled. Stamp
    entries WITHOUT a digest pin were unpinned at save time and refuse
    loudly (--skip-unpinned proceeds without them); recorded bytes that
    are unavailable and un-refetchable refuse loudly too - newer bytes
    are NEVER silently substituted."""
    installer = _installer(args)
    path = Path(args.workflow_file)
    if not path.is_file():
        raise InstallError(f"no workflow file at {path}")
    stamped_dinkster, stamped_packs = _environment_stamp(path.read_text())
    if stamped_dinkster and stamped_dinkster != dinkster_version:
        print(
            f"note: workflow was stamped by dinkster {stamped_dinkster}; this is "
            f"dinkster {dinkster_version} (advisory)"
        )
    unpinned = sorted(pid for pid, f in stamped_packs.items() if "artifactDigest" not in f)
    if unpinned:
        if not args.skip_unpinned:
            raise InstallError(
                f"the stamp records no digest pin for: {', '.join(unpinned)}; "
                f"these were unpinned when the workflow was saved and cannot "
                f"be reproduced exactly - re-run with --skip-unpinned to "
                f"reproduce the pinned packs without them"
            )
        for pack_id in unpinned:
            print(f"skipping {pack_id}: stamped without a digest pin (--skip-unpinned)")
    provisional: list[LockedPack] = []
    for pack_id, fields in sorted(stamped_packs.items()):
        digest = fields.get("artifactDigest")
        if digest is None:
            continue
        problem = validate_artifact_digest(digest)
        if problem is not None:
            raise InstallError(f"stamped digest for pack {pack_id!r} {problem}")
        name = canonical_name(pack_id)
        provisional.append(
            LockedPack(
                pack=name,
                version=fields.get("version", LOCAL_VERSION),
                artifact_digest=digest,
                publisher=fields.get("publisher", LOCAL_PUBLISHER),
                # The pack name itself is always a claim; the FULL claim
                # set is derived from the acquired manifest at apply -
                # workflow JSON never gets to claim namespaces.
                claims=(name,),
                source=fields.get("source", ""),
            )
        )
    if not provisional:
        raise InstallError("the environment stamp pins no packs; nothing to reproduce")
    missing = [entry for entry in provisional if not installer.artifact_available(entry)]
    stuck = sorted(entry.pack for entry in missing if not installer.acquirable(entry))
    if stuck:
        raise InstallError(
            f"recorded bytes are not available locally for: {', '.join(stuck)}, "
            f"and their stamped provenance cannot be re-fetched (registry "
            f"sources need --registry or $DINKSTER_REGISTRY) - stage those "
            f"artifacts first; "
            f"newer bytes are never silently substituted"
        )
    target = Lockfile.of(provisional)
    # Acquisition is part of the reviewed plan: fetch happens only after
    # confirmation, and apply admits only bytes that hash to the recorded
    # digest - never a silent substitution.
    for entry in missing:
        print(
            f"acquire {entry.pack} from {entry.source} "
            f"(must hash to {entry.artifact_digest[:19]}...)"
        )
    current = installer.current_lockfile()
    steps = plan(current if current is not None else Lockfile(), target)
    venvs = not args.no_venv
    hosting = installer.hosting_topology(target)
    hosting_groups = hosting.groups if venvs else ()
    hosting_in_process = hosting.in_process
    hosting_runtime_pins = installer.serving_runtime_pins() if hosting_in_process else {}
    topology_change = (
        hosting_groups != installer.current_groups()
        or hosting_in_process != installer.current_in_process()
        or hosting_runtime_pins != installer.current_runtime_pins()
    )
    if steps.empty and not topology_change:
        print("nothing to do; the installation already matches")
        return
    _print_plan(installer, steps, venvs=venvs)
    print(
        "environment stamps pin pack bytes, not python dists; venvs resolve "
        "manifest ranges fresh (snapshot restore is the exact-pin path); "
        "claims are derived from each pack's manifest and validated before "
        "activation"
    )
    if args.plan_file:
        record = PlanRecord(
            target=target,
            base=current.record_digest() if current is not None else None,
            venvs=venvs,
            accelerator=installer.accelerator,
            acquire=True,
            derive_claims=True,
            allow_doctor_findings=args.allow_doctor_findings,
            venv_groups=hosting_groups or None,
            in_process=hosting_in_process,
            runtime_pins=tuple(sorted(hosting_runtime_pins.items())),
        )
        plan_path = Path(args.plan_file)
        plan_path.write_text(record.record_json() + "\n")
        print(f"plan written to {plan_path}; apply with: dinkster-pack apply {plan_path}")
        return
    if not _confirmed(args):
        print("plan not applied; nothing changed")
        return
    for entry in missing:
        installer.acquire(entry)
        print(f"acquired {entry.pack}: digest verified")
    number, _ = installer.apply(
        _claims_from_manifests(installer, target),
        venvs=venvs,
        allow_doctor_findings=args.allow_doctor_findings,
        hosting_groups=hosting_groups,
        hosting_in_process=hosting_in_process,
        hosting_runtime_pins=hosting_runtime_pins,
    )
    print(f"activated generation {number}")


def _cmd_search(args: argparse.Namespace) -> None:
    """Browse ONE registry's pack index. Discovery only: installing what
    a row names still goes through resolve + digest verification."""
    registry = select_registry(_registry_config(args), args.registry)
    page = browse_packs(registry, query=args.query, limit=args.limit, cursor=args.cursor)
    if not page.packs:
        what = f"match {args.query!r}" if args.query else "are published"
        print(f"no packs {what} on {registry.label}")
        return
    for entry in page.packs:
        plural = "" if entry.versions == 1 else "s"
        print(
            f"{entry.pack}@{entry.latest_version}  "
            f"publisher {entry.publisher}  ({entry.versions} version{plural})"
        )
    if page.cursor:
        print(f"more results: rerun with --cursor {page.cursor}")


def _cmd_templates(args: argparse.Namespace) -> None:
    """Browse ONE registry's template catalog: starter workflows from
    each pack's latest release, before anything is installed."""
    registry = select_registry(_registry_config(args), args.registry)
    page = browse_templates(
        registry,
        query=args.query,
        tag=args.tag,
        pack=args.pack,
        limit=args.limit,
        cursor=args.cursor,
    )
    if not page.templates:
        print(f"no templates match on {registry.label}")
        return
    for entry in page.templates:
        tags = f"  [{', '.join(entry.tags)}]" if entry.tags else ""
        description = f"  - {entry.description}" if entry.description else ""
        print(
            f"{entry.pack}/{entry.id}  {entry.name} "
            f"({entry.pack}@{entry.version}){tags}{description}"
        )
    if page.cursor:
        print(f"more results: rerun with --cursor {page.cursor}")


def _cmd_status(args: argparse.Namespace) -> None:
    installer = _installer(args)
    number = installer.current_number()
    if number is None:
        print("nothing installed")
        return
    print(f"current generation: {number} (history: {list(installer.generation_numbers())})")
    for entry in installer.lockfile_of(number).packs:
        print(f"  {entry.pack} {entry.version} {entry.artifact_digest[:19]} [{entry.source}]")


def _cmd_rollback(args: argparse.Namespace) -> None:
    installer = _installer(args)
    current = installer.current_number()
    if current is None:
        raise InstallError("nothing is installed; nothing to roll back")
    previous = [n for n in installer.generation_numbers() if n < current]
    if not previous:
        raise InstallError("no earlier generation exists to roll back to")
    previous_number = previous[-1]
    _finish(
        args,
        installer,
        installer.lockfile_of(previous_number),
        hosting_groups=installer.generation_groups(previous_number),
        hosting_in_process=installer.generation_in_process(previous_number),
        hosting_runtime_pins=installer.generation_runtime_pins(previous_number),
        generation_topology=True,
    )


def _cmd_gc(args: argparse.Namespace) -> None:
    installer = _installer(args)
    candidates = installer.gc_candidates()
    if installer.shared_store is not None:
        print(
            f"shared store: {installer.shared_store} "
            f"(references counted across every registered install root)"
        )
    if not candidates:
        print("0 unreferenced item(s) removed")
        return
    for item in candidates:
        print(f"would remove: {item}")
    if not _confirmed(args):
        print("nothing removed")
        return
    removed = installer.gc()
    print(f"{len(removed)} unreferenced item(s) removed")


def _cmd_archive(args: argparse.Namespace) -> None:
    print(build_pack_archive(args.pack_root, args.output))


def _cmd_registry_declaration(args: argparse.Namespace) -> None:
    manifest_path, contributions = load_registry_contributions(args.pack_root)
    declaration = registry_declaration_toml(contributions)
    if args.write:
        write_registry_declaration(manifest_path, declaration)
    print(declaration)


def _prepared_pack_specs(args: argparse.Namespace) -> Iterator[PackSpec]:
    from .comfy_compose import comfy_compat_specs
    from .compose import default_pack_specs, training_pack_specs
    from .serve import _default_pack_venv_root, _prepare_default_pack, _with_remote_config

    specs = (
        (*default_pack_specs(), *comfy_compat_specs())
        if args.defaults
        else _installer(args).packs_for_serving()
    )
    default_count = len(specs)
    if args.defaults and args.library_root:
        specs = (*specs, *training_pack_specs(Path(args.library_root) / "training.sqlite"))
    for index, spec in enumerate(specs):
        if args.defaults and "dinkster-nodes-remote" in (spec.packs or {}):
            spec = _with_remote_config(
                spec,
                catalog_base=args.remote_catalog_base,
                gateway_base=args.remote_gateway_base,
            )
        if args.defaults and index < default_count and not spec.in_process:
            spec = _prepare_default_pack(
                spec,
                venv_root=_default_pack_venv_root(args.library_root),
                accelerator=resolve_accelerator(args.accelerator),
            )
        yield spec


def _cmd_doctor(args: argparse.Namespace) -> None:
    failed = False
    for spec in _prepared_pack_specs(args):
        report = diagnose(spec.manifest, interpreter=spec.python, environment=spec.env)
        print(render_text(report))
        failed |= not report.ok
    if failed:
        raise InstallError("doctor reported unhealthy packs; repair findings and retry")


def _cmd_prepare_catalogs(args: argparse.Namespace) -> None:
    failed = False
    for spec in _prepared_pack_specs(args):
        report = prepare_catalog(spec.manifest, interpreter=spec.python, environment=spec.env)
        if report.ok:
            print(f"Prepared runtime catalog: {report.pack_name} ({len(report.node_types)} nodes)")
        else:
            print(render_text(report))
            failed = True
    if failed:
        raise InstallError("runtime catalog preparation failed; repair findings and retry")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dinkster-pack",
        description="Manage a Dinkster install root (generations of installed packs)",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("DINKSTER_INSTALL_ROOT", ""),
        metavar="PATH",
        help="install root directory (default: $DINKSTER_INSTALL_ROOT)",
    )
    parser.add_argument(
        "--shared-store",
        default=os.environ.get("DINKSTER_SHARED_STORE", ""),
        metavar="PATH",
        help="shared content store several install roots reference for "
        "artifacts/store/venvs (default: $DINKSTER_SHARED_STORE). Recorded in "
        "the root on first use, so later invocations - and dinkster-serve - "
        "need no flag; pointing an existing root at a DIFFERENT store is "
        "refused (that is a migration, not a flag)",
    )
    parser.add_argument(
        "--workspace-package",
        action="append",
        default=[],
        metavar="DIR",
        help="local dinkster package directory to install into pack venvs instead "
        "of resolving from the index (dev workspaces; repeatable)",
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", *KNOWN_ACCELERATORS),
        default="auto",
        help="accelerator to provision pack venvs for (selects each pack's "
        "[pack.extra-requires] list); auto consults $DINKSTER_ACCELERATOR then "
        "host detection (default: auto)",
    )
    parser.add_argument(
        "--registry",
        default=os.environ.get("DINKSTER_REGISTRY", ""),
        metavar="NAME_OR_URL",
        help="the registry for pack@version resolution and artifact "
        "downloads: a name from the registries config, or an http(s) "
        "endpoint URL (default: $DINKSTER_REGISTRY; URL auth token, if any, "
        "from $DINKSTER_REGISTRY_TOKEN)",
    )
    parser.add_argument(
        "--registries",
        default=os.environ.get("DINKSTER_REGISTRIES", ""),
        metavar="FILE",
        help="named-registries config (TOML: [registries.<name>] with "
        "endpoint, optional token-env and default; default: "
        "$DINKSTER_REGISTRIES, else <root>/registries.toml when present)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    archive = commands.add_parser(
        "archive",
        help="build a deterministic registry-compatible pack archive",
    )
    archive.add_argument("pack_root", metavar="PACK_ROOT", help="pack directory to archive")
    archive.add_argument("--output", required=True, metavar="FILE.zip", help="archive to write")
    archive.set_defaults(handler=_cmd_archive)

    registry_declaration = commands.add_parser(
        "registry-declaration",
        help="print the registry providers materialized by a pack's inference entry",
    )
    registry_declaration.add_argument(
        "pack_root", metavar="PACK_ROOT", help="pack directory or dinkster-pack.toml"
    )
    registry_declaration.add_argument(
        "--write",
        action="store_true",
        help="update [pack.provides.registry] in dinkster-pack.toml",
    )
    registry_declaration.set_defaults(handler=_cmd_registry_declaration)

    for name, help_text, handler in (
        ("doctor", "lint installed packs and refresh schema catalogs", _cmd_doctor),
        (
            "prepare-catalogs",
            "prepare runtime catalogs for trusted installed packs",
            _cmd_prepare_catalogs,
        ),
    ):
        check = commands.add_parser(name, help=help_text)
        check.add_argument(
            "--defaults", action="store_true", help="use the installed default suite"
        )
        check.add_argument(
            "--library-root",
            default=os.environ.get("DINKSTER_LIBRARY_ROOT", "dinkster-library"),
            metavar="PATH",
            help="serving library root used for default-pack interpreters",
        )
        check.add_argument(
            "--remote-catalog-base", help="remote catalog URL, as used by dinkster-serve"
        )
        check.add_argument(
            "--remote-gateway-base", help="remote gateway URL, as used by dinkster-serve"
        )
        check.set_defaults(handler=handler)

    def add_plan_apply_arguments(command: argparse.ArgumentParser) -> None:
        """Every mutating command carries the plan/apply controls."""
        command.add_argument("--no-venv", action="store_true", help="skip venv provisioning")
        command.add_argument(
            "--allow-doctor-findings",
            action="store_true",
            help="warn instead of refusing when staged local/git packs fail doctor",
        )
        command.add_argument(
            "--yes",
            "-y",
            action="store_true",
            help="apply the printed plan without asking (the noninteractive opt-in)",
        )
        command.add_argument(
            "--plan",
            dest="plan_file",
            metavar="FILE",
            default="",
            help="write the plan to FILE and exit without applying; "
            "execute it later with 'dinkster-pack apply FILE'",
        )

    install = commands.add_parser("install", help="install packs as a new generation")
    install.add_argument(
        "packs",
        nargs="+",
        metavar="SOURCE",
        help="pack directory, manifest path, git+<url>[@<ref>], or "
        "pack@version resolved from the --registry/default registry "
        "(a local directory named like a spec: prefix with ./)",
    )
    add_plan_apply_arguments(install)
    install.set_defaults(handler=_cmd_install)

    update = commands.add_parser(
        "update",
        help="re-resolve installed packs from their recorded provenance "
        "(git: default branch head; local: the source directory)",
    )
    update.add_argument("packs", nargs="*", metavar="NAME", help="only these packs (default: all)")
    add_plan_apply_arguments(update)
    update.set_defaults(handler=_cmd_update)

    publish = commands.add_parser(
        "publish",
        help="doctor, upload, and submit a pack to one registry",
    )
    publish.add_argument("pack", metavar="PACK_DIR_OR_MANIFEST")
    publish.add_argument("--version", required=True, metavar="X.Y.Z")
    publish.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip the local doctor run (registry admission remains authoritative)",
    )
    publish.set_defaults(handler=_cmd_publish)

    remove = commands.add_parser("remove", help="remove installed packs as a new generation")
    remove.add_argument("packs", nargs="+", metavar="NAME", help="installed pack name")
    add_plan_apply_arguments(remove)
    remove.set_defaults(handler=_cmd_remove)

    apply_cmd = commands.add_parser(
        "apply",
        help="execute a plan file written by --plan; rejected if the "
        "installation changed since the plan was computed",
    )
    apply_cmd.add_argument("plan_file", metavar="FILE", help="plan file from --plan")
    apply_cmd.set_defaults(handler=_cmd_apply)

    snapshot = commands.add_parser(
        "snapshot",
        help="capture the environment (lockfile + per-venv dependency "
        "freezes + host scope) to a file; read-only",
    )
    snapshot.add_argument("out", metavar="FILE", help="where to write the snapshot")
    snapshot.add_argument(
        "--hashes",
        action="store_true",
        help="annotate portable pins with their artifact hashes (queries "
        "the package index; exact restores then verify downloaded "
        "artifacts against the record)",
    )
    snapshot.set_defaults(handler=_cmd_snapshot)

    restore = commands.add_parser(
        "restore",
        help="re-create a snapshotted environment (plan/apply; missing "
        "venvs provision from the snapshot's exact pins)",
    )
    restore.add_argument("snapshot_file", metavar="FILE", help="snapshot file from 'snapshot'")
    restore.add_argument(
        "--fresh-pins",
        action="store_true",
        help="ignore ALL snapshot pins and resolve manifest ranges fresh "
        "(the escape hatch when a cross-scope restore's kept version "
        "constraint cannot resolve on this host)",
    )
    add_plan_apply_arguments(restore)
    restore.set_defaults(handler=_cmd_restore)

    reproduce = commands.add_parser(
        "reproduce",
        help="rebuild an installation from a workflow's environment stamp "
        "(plan/apply; the new generation contains exactly the stamped packs)",
    )
    reproduce.add_argument(
        "workflow_file",
        metavar="WORKFLOW",
        help="workflow document carrying an environment stamp",
    )
    reproduce.add_argument(
        "--skip-unpinned",
        action="store_true",
        help="reproduce only digest-pinned packs, skipping stamp entries "
        "recorded without a pin (packs that were unpinned at save time)",
    )
    add_plan_apply_arguments(reproduce)
    reproduce.set_defaults(handler=_cmd_reproduce)

    def add_browse_arguments(command: argparse.ArgumentParser) -> None:
        """Every browse command speaks the keyset listing contract."""
        command.add_argument("--limit", type=int, default=50, help="rows per page (default: 50)")
        command.add_argument(
            "--cursor",
            default="",
            metavar="CURSOR",
            help="resume a previous listing from its printed cursor (bound to the same query)",
        )

    search = commands.add_parser(
        "search", help="browse a registry's pack index (--registry or the default)"
    )
    search.add_argument(
        "query",
        nargs="?",
        default="",
        help="substring over pack names and probed node types (optional)",
    )
    add_browse_arguments(search)
    search.set_defaults(handler=_cmd_search)

    templates = commands.add_parser(
        "templates",
        help="browse a registry's template catalog (latest release per pack)",
    )
    templates.add_argument(
        "query",
        nargs="?",
        default="",
        help="substring over template id/name/description/tags (optional)",
    )
    templates.add_argument("--tag", default="", help="exact tag filter")
    templates.add_argument(
        "--pack", default="", metavar="NAME", help="only templates from this pack"
    )
    add_browse_arguments(templates)
    templates.set_defaults(handler=_cmd_templates)

    status = commands.add_parser("status", help="show the current generation")
    status.set_defaults(handler=_cmd_status)

    rollback = commands.add_parser("rollback", help="re-activate the previous generation")
    add_plan_apply_arguments(rollback)
    rollback.set_defaults(handler=_cmd_rollback)

    gc = commands.add_parser("gc", help="delete content no generation references")
    gc.add_argument(
        "--yes", "-y", action="store_true", help="delete without asking (noninteractive opt-in)"
    )
    gc.set_defaults(handler=_cmd_gc)

    args = parser.parse_args()
    try:
        args.handler(args)
    except RegistryPublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except (
        InstallError,
        ArtifactError,
        PackArchiveError,
        RegistryDeclarationError,
        CompositionError,
        ManifestError,
        AcceleratorError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
