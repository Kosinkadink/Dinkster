"""The manager-side install model: lockfile, plan, atomic activation.

The pure half of the ComfyUI-Manager successor. The recorded failures it
answers: install-by-unpinned-git-clone (nothing reproducible), conflicts
discovered at user runtime after bytes already landed, and updates that
mutate a live installation under running jobs. The model gives the host
three guarantees to build on:

- **A lockfile is the installation.** ``Lockfile`` pins every installed
  pack to an exact artifact digest; its canonical serialization digests
  stably, so "what is installed" is one content-addressed value - two
  machines with the same lockfile digest have the same installation.
  Construction validates what must never reach a machine: duplicate
  canonical pack names, and namespace claims that conflict across
  publishers (same-publisher nesting - the suite shape - is legal, the
  same rule the registry's grant table enforces). The lockfile does NOT
  enforce reserved-root trust: whether a pack claiming ``std`` may
  compose is host trust policy at load, not lockfile shape.
- **Plans are diffs, not scripts.** ``plan`` computes the deterministic
  step list between two lockfiles (add/upgrade/downgrade/reinstall/
  remove, sorted by pack). Staging those steps (venv builds via
  ``ensure_pack_venv``) is host machinery with side effects; the plan
  itself is a value the user can review before anything runs.
- **Activation is a generation, never a mutation.** ``GenerationLedger``
  models the nixos-style pointer swap: activating a lockfile appends a
  new generation; history never rewrites (rollback re-activates old
  content AS A NEW generation). Running jobs pin the generation they
  started on, so an update can never mutate an installation under a job -
  a generation is prunable only when it is not current and no job pins
  it.

Install provenance (``LockedPack.source``) is metadata, never identity:
a git install and a registry install of one pack are one canonical name
(the manager's one-pack-two-entries failure stays
unrepresentable).

The transitive Python-dependency story is split across records: the
artifact's ``requires`` resolve inside the pack venv at stage time, a
snapshot freezes what resolution chose per venv, and pins may carry
``--hash=<algo>:<digest>`` annotations (the resolution lock's
artifact-identity half - provisioning verifies annotated pins against
the downloaded artifacts). Deliberately still out, documented not
implied: marker-annotated universal resolution (one lock serving
platforms whose dependency SETS differ), update channels/notifications,
and yank/security-revocation propagation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from blake3 import blake3
from dinkster_schema import canonical_name, claims_conflict, validate_name

from .model import (
    ARTIFACT_DIGEST_PREFIX,
    Release,
    Version,
    canonical_json,
    validate_artifact_digest,
    validate_version,
)

LOCKFILE_FORMAT = "dinkster.lock/1"


class InstallError(Exception):
    """An install-model invariant was violated."""


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockedPack:
    """One installed pack, pinned to exact bytes."""

    pack: str
    """Canonical pack name - the sole identity, whatever the source."""
    version: str
    artifact_digest: str
    publisher: str
    """Canonical publisher id (attribution + the claim-conflict rule)."""
    claims: tuple[str, ...]
    """Canonical namespace claims, for pre-install collision checking."""
    source: str = "registry"
    """Install provenance ('registry', 'git:<repo_id>', 'local:<path>') -
    updatable-from metadata, never identity."""

    @classmethod
    def from_release(cls, release: Release, source: str = "registry") -> LockedPack:
        return cls(
            pack=release.pack,
            version=release.version,
            artifact_digest=release.artifact_digest,
            publisher=release.publisher,
            claims=release.claims,
            source=source,
        )


def _require_canonical(label: str, name: str) -> None:
    problem = validate_name(name)
    if problem is not None:
        raise InstallError(f"{label} {name!r} {problem}")
    if canonical_name(name) != name:
        raise InstallError(
            f"{label} {name!r} is not canonical (expected {canonical_name(name)!r}); "
            f"lockfiles store canonical forms only"
        )


@dataclass(frozen=True)
class Lockfile:
    """The installation as a value. Use ``Lockfile.of`` to build one."""

    packs: tuple[LockedPack, ...] = ()

    @classmethod
    def of(cls, packs: tuple[LockedPack, ...] | list[LockedPack]) -> Lockfile:
        """Validate and normalize: sorted by pack name, no duplicate
        identities, no cross-publisher claim conflicts."""
        ordered = tuple(sorted(packs, key=lambda entry: entry.pack))
        for entry in ordered:
            _require_canonical("pack name", entry.pack)
            _require_canonical("publisher id", entry.publisher)
            # No minimum claim count: a pure executor pack (empty [pack]
            # namespaces, non-empty executes) locks no claims. Pack-name
            # uniqueness is enforced below independently of claims.
            for claim in entry.claims:
                _require_canonical(f"claim of pack {entry.pack!r}", claim)
            problem = validate_version(entry.version)
            if problem is not None:
                raise InstallError(f"pack {entry.pack!r} version {entry.version!r} {problem}")
            digest_problem = validate_artifact_digest(entry.artifact_digest)
            if digest_problem is not None:
                raise InstallError(
                    f"pack {entry.pack!r} digest {entry.artifact_digest!r} {digest_problem}"
                )
        for index, entry in enumerate(ordered):
            for earlier in ordered[:index]:
                if earlier.pack == entry.pack:
                    raise InstallError(
                        f"two entries lock pack {entry.pack!r}; one canonical "
                        f"name is one pack, whatever the source"
                    )
                if earlier.publisher == entry.publisher:
                    continue  # same-publisher nesting is the legal suite shape
                for a in earlier.claims:
                    for b in entry.claims:
                        if claims_conflict(a, b):
                            raise InstallError(
                                f"packs {earlier.pack!r} and {entry.pack!r} claim "
                                f"conflicting namespaces ({a!r} vs {b!r}) across "
                                f"publishers; installing both would make node "
                                f"ownership load-order dependent"
                            )
        return cls(ordered)

    def get(self, pack: str) -> LockedPack | None:
        canonical = canonical_name(pack)
        for entry in self.packs:
            if entry.pack == canonical:
                return entry
        return None

    @classmethod
    def from_record_json(cls, text: str) -> Lockfile:
        """Decode a persisted lockfile, re-running every ``of`` validation -
        a hand-edited or corrupted lockfile fails loudly at load, never at
        activation."""
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InstallError(f"lockfile is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise InstallError("lockfile must be a JSON object")
        record = cast("dict[str, object]", document)
        if record.get("format") != LOCKFILE_FORMAT:
            raise InstallError(
                f"unsupported lockfile format {record.get('format')!r}; "
                f"expected {LOCKFILE_FORMAT!r}"
            )
        packs_raw = record.get("packs")
        if not isinstance(packs_raw, list):
            raise InstallError("lockfile 'packs' must be a list")
        entries: list[LockedPack] = []
        for item in cast("list[object]", packs_raw):
            if not isinstance(item, dict):
                raise InstallError("lockfile pack entries must be JSON objects")
            entry = cast("dict[str, object]", item)
            fields: dict[str, str] = {}
            for key in ("pack", "version", "artifactDigest", "publisher", "source"):
                value = entry.get(key)
                if not isinstance(value, str):
                    raise InstallError(f"lockfile pack entry field {key!r} must be a string")
                fields[key] = value
            claims_raw = entry.get("claims")
            if not isinstance(claims_raw, list) or not all(
                isinstance(claim, str) for claim in cast("list[object]", claims_raw)
            ):
                raise InstallError("lockfile pack entry 'claims' must be a list of strings")
            entries.append(
                LockedPack(
                    pack=fields["pack"],
                    version=fields["version"],
                    artifact_digest=fields["artifactDigest"],
                    publisher=fields["publisher"],
                    claims=tuple(cast("list[str]", claims_raw)),
                    source=fields["source"],
                )
            )
        return cls.of(entries)

    def record_json(self) -> str:
        return canonical_json(
            {
                "format": LOCKFILE_FORMAT,
                "packs": [
                    {
                        "pack": entry.pack,
                        "version": entry.version,
                        "artifactDigest": entry.artifact_digest,
                        "publisher": entry.publisher,
                        "claims": list(entry.claims),
                        "source": entry.source,
                    }
                    for entry in self.packs
                ],
            }
        )

    def record_digest(self) -> str:
        """The installation's content address: same digest, same install."""
        return ARTIFACT_DIGEST_PREFIX + blake3(self.record_json().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Install plans (diffs between lockfiles, reviewable before side effects)
# ---------------------------------------------------------------------------

StepAction = Literal["add", "upgrade", "downgrade", "reinstall", "remove"]


@dataclass(frozen=True)
class InstallStep:
    action: StepAction
    pack: str
    to: LockedPack | None
    """The target entry; None only for remove."""
    from_version: str | None = None
    """The currently installed version this step moves away from."""


@dataclass(frozen=True)
class InstallPlan:
    steps: tuple[InstallStep, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.steps


PLAN_FORMAT = "dinkster.plan/1"


@dataclass(frozen=True)
class VenvSpec:
    """How restore shapes one venv's provisioning.

    ``exact`` reproduces a same-scope freeze verbatim (range resolution
    skipped entirely). ``constraints`` are a cross-scope snapshot's
    PORTABLE pins: ranges + destination accelerator requirements still
    drive what gets installed, and constraints bind the version of
    whatever resolution pulls - a constraint nothing pulls is inert.
    Both empty/None = fresh range resolution."""

    exact: tuple[str, ...] | None = None
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanRecord:
    """A persisted plan: the reviewable value between 'plan' and 'apply'
    (plan/apply discipline - mutation only ever runs on
    explicit confirmation of an exact plan).

    Single source of truth: the record stores the BASE (the lockfile
    digest the plan was computed against; None for an empty root) and the
    TARGET lockfile - the step list is always re-derived via :func:`plan`,
    never stored, so a record can never disagree with itself. Staleness is
    structural: apply recomputes the current installation's digest and
    refuses a mismatch (the environment changed since planning - replan,
    never partially apply)."""

    target: Lockfile
    base: str | None = None
    """record_digest() of the installation the plan diffs against; None
    means the plan expects an empty root."""
    venvs: bool = True
    accelerator: str = ""
    """Accelerator the plan was reviewed under (selects each pack's
    ``[pack.extra-requires]`` list at provisioning). Recorded so apply
    installs the dependency set the user actually saw, not whatever the
    apply-time host resolves. Empty = not recorded (older plan files);
    apply falls back to its own selection. Additive to dinkster.plan/1."""
    acquire: bool = False
    """Apply may re-fetch missing pinned artifacts from each entry's
    recorded ``source`` (digest-verified, never a substitution) instead
    of refusing. Set by planners whose targets come from provenance
    records (reproduce) - the fetch is part of the reviewed plan.
    Additive to dinkster.plan/1; absent decodes as False (refuse missing,
    the original behavior)."""
    derive_claims: bool = False
    """The target's claims are pack-name placeholders: apply re-derives
    each entry's claims from its pinned bytes' manifest and revalidates
    the lockfile before activation - the planning input (a workflow's
    environment stamp) never gets to claim namespaces. Additive to
    dinkster.plan/1; absent decodes as False (claims are already real)."""
    venv_specs: tuple[tuple[str, VenvSpec], ...] = ()
    """Per-pack venv shaping the plan was reviewed with (restore's exact
    snapshot pins, or a cross-scope restore's portable constraints) -
    recorded so apply provisions the venvs the user actually saw, not a
    fresh range resolution. Sorted by pack; every pack must be locked by
    the target. Additive to dinkster.plan/1; absent decodes as empty (fresh
    range resolution, the original behavior)."""
    allow_doctor_findings: bool = False
    """Whether apply may activate local/git packs whose staged doctor
    probe reports errors. Additive to dinkster.plan/1; absent decodes as
    False so older plans retain refusal-by-default."""
    venv_groups: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    """Topology reviewed with this plan. None is the legacy per-pack
    topology; an explicit tuple carries declared or snapshot groups."""
    in_process: tuple[str, ...] = ()
    """Packs reviewed for host-process placement. Additive to
    ``dinkster.plan/1``; absent in older plans means no in-process packs."""
    runtime_pins: tuple[tuple[str, str], ...] = ()
    """Exact serving-runtime baseline for generation-topology replay."""
    generation_topology: bool = False
    """The topology came from an immutable generation (rollback), not
    live hosting policy, so apply replays it without policy validation."""

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.venv_specs, key=lambda item: item[0]))
        object.__setattr__(self, "venv_specs", ordered)
        locked = {entry.pack for entry in self.target.packs}
        for pack, _ in ordered:
            if pack not in locked:
                raise InstallError(
                    f"plan records a venv spec for pack {pack!r} that the target does not lock"
                )
        if self.venv_groups is not None:
            groups = tuple(sorted(self.venv_groups))
            object.__setattr__(self, "venv_groups", groups)
            grouped: set[str] = set()
            specs = dict(ordered)
            for name, members in groups:
                if len(members) < 2 or any(member not in locked for member in members):
                    raise InstallError(f"plan venv group {name!r} has invalid members")
                if grouped.intersection(members):
                    raise InstallError("plan records duplicate venv-group membership")
                grouped.update(members)
                member_specs = [specs.get(member) for member in members]
                if any(spec != member_specs[0] for spec in member_specs[1:]):
                    raise InstallError(
                        f"plan venv group {name!r} has contradictory per-member exact pins"
                    )
        normalized_in_process = tuple(sorted(self.in_process))
        object.__setattr__(self, "in_process", normalized_in_process)
        object.__setattr__(self, "runtime_pins", tuple(sorted(self.runtime_pins)))
        if len(set(normalized_in_process)) != len(normalized_in_process):
            raise InstallError("plan records duplicate in-process membership")
        unknown_in_process = sorted(set(normalized_in_process) - locked)
        if unknown_in_process:
            raise InstallError(
                "plan records in-process packs not locked by the target: "
                + ", ".join(unknown_in_process)
            )
        grouped = {member for _, members in (self.venv_groups or ()) for member in members}
        contradictions = sorted(grouped.intersection(normalized_in_process))
        if contradictions:
            raise InstallError(
                "plan records packs in both a venv group and in-process: "
                + ", ".join(contradictions)
            )
        pin_names = [name for name, _ in self.runtime_pins]
        if len(set(pin_names)) != len(pin_names):
            raise InstallError("plan records duplicate runtime pin names")
        if normalized_in_process and set(pin_names) != {"torch", "dinkster-aimdo"}:
            raise InstallError(
                "plan records in-process packs without exact torch and dinkster-aimdo pins"
            )
        if self.runtime_pins and not normalized_in_process:
            raise InstallError("plan records runtime pins without in-process packs")

    def matches_base(self, current: Lockfile | None) -> bool:
        return self.base == (current.record_digest() if current is not None else None)

    @classmethod
    def from_record_json(cls, text: str) -> PlanRecord:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InstallError(f"plan file is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise InstallError("plan file must be a JSON object")
        record = cast("dict[str, object]", document)
        if record.get("format") != PLAN_FORMAT:
            raise InstallError(
                f"unsupported plan format {record.get('format')!r}; expected {PLAN_FORMAT!r}"
            )
        base = record.get("base")
        if base is not None and not isinstance(base, str):
            raise InstallError("plan 'base' must be a string digest or null")
        if isinstance(base, str) and validate_artifact_digest(base) is not None:
            raise InstallError(f"plan 'base' {base!r} is not a digest")
        venvs = record.get("venvs")
        if not isinstance(venvs, bool):
            raise InstallError("plan 'venvs' must be a boolean")
        accelerator = record.get("accelerator", "")
        if not isinstance(accelerator, str):
            raise InstallError("plan 'accelerator' must be a string")
        # These flags are additive to dinkster.plan/1:
        # absent in older plan files and decoded as False.
        flags: dict[str, bool] = {}
        for key in ("acquire", "deriveClaims", "allowDoctorFindings"):
            value = record.get(key, False)
            if not isinstance(value, bool):
                raise InstallError(f"plan {key!r} must be a boolean")
            flags[key] = value
        generation_topology = record.get("generationTopology", False)
        if not isinstance(generation_topology, bool):
            raise InstallError("plan 'generationTopology' must be a boolean")
        target_raw = record.get("target")
        if not isinstance(target_raw, dict):
            raise InstallError("plan 'target' must be a lockfile object")
        target = Lockfile.from_record_json(json.dumps(target_raw))
        # "venvSpecs" is additive like the flags: absent decodes as empty.
        specs_raw = record.get("venvSpecs")
        venv_specs: list[tuple[str, VenvSpec]] = []
        if specs_raw is not None:
            if not isinstance(specs_raw, dict):
                raise InstallError("plan 'venvSpecs' must be an object of pack -> spec")
            for pack, spec_raw in cast("dict[str, object]", specs_raw).items():
                if not isinstance(spec_raw, dict):
                    raise InstallError(f"plan venv spec for pack {pack!r} must be an object")
                spec_fields = cast("dict[str, object]", spec_raw)
                exact_raw = spec_fields.get("exact")
                exact = (
                    _decode_pins(pack, exact_raw, label="plan venv exact")
                    if exact_raw is not None
                    else None
                )
                constraints_raw = spec_fields.get("constraints")
                constraints = (
                    _decode_pins(
                        pack,
                        constraints_raw,
                        label="plan venv constraint",
                        allow_hashes=False,
                    )
                    if constraints_raw is not None
                    else ()
                )
                venv_specs.append((pack, VenvSpec(exact=exact, constraints=constraints)))
        groups_raw = record.get("venvGroups")
        venv_groups: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        if groups_raw is not None:
            if not isinstance(groups_raw, dict):
                raise InstallError("plan 'venvGroups' must be an object")
            decoded_groups: list[tuple[str, tuple[str, ...]]] = []
            for name, members_raw in cast("dict[str, object]", groups_raw).items():
                if not isinstance(members_raw, list):
                    raise InstallError(f"plan venv group {name!r} must be a list of strings")
                members_object = cast("list[object]", members_raw)
                if not all(isinstance(member, str) for member in members_object):
                    raise InstallError(f"plan venv group {name!r} must be a list of strings")
                members = cast("list[str]", members_object)
                decoded_groups.append((name, tuple(sorted(members))))
            venv_groups = tuple(decoded_groups)
        in_process_raw = record.get("inProcess", [])
        if not isinstance(in_process_raw, list) or not all(
            isinstance(pack, str) for pack in cast("list[object]", in_process_raw)
        ):
            raise InstallError("plan 'inProcess' must be a list of strings")
        runtime_pins_raw = record.get("runtimePins", {})
        if not isinstance(runtime_pins_raw, dict) or not all(
            isinstance(name, str) and isinstance(version, str)
            for name, version in cast("dict[object, object]", runtime_pins_raw).items()
        ):
            raise InstallError("plan 'runtimePins' must be an object of strings")
        return cls(
            target=target,
            base=base,
            venvs=venvs,
            accelerator=accelerator,
            acquire=flags["acquire"],
            derive_claims=flags["deriveClaims"],
            venv_specs=tuple(venv_specs),
            allow_doctor_findings=flags["allowDoctorFindings"],
            venv_groups=venv_groups,
            in_process=tuple(cast("list[str]", in_process_raw)),
            runtime_pins=tuple(cast("dict[str, str]", runtime_pins_raw).items()),
            generation_topology=generation_topology,
        )

    def record_json(self) -> str:
        document: dict[str, object] = {
            "format": PLAN_FORMAT,
            "base": self.base,
            "target": json.loads(self.target.record_json()),
            "venvs": self.venvs,
            "accelerator": self.accelerator,
        }
        # Omitted when False/empty, never written as false/{} - older
        # readers and records stay byte-identical for the behaviors they
        # know.
        if self.acquire:
            document["acquire"] = True
        if self.derive_claims:
            document["deriveClaims"] = True
        if self.allow_doctor_findings:
            document["allowDoctorFindings"] = True
        if self.venv_specs:
            specs_document: dict[str, object] = {}
            for pack, spec in self.venv_specs:
                spec_document: dict[str, object] = {}
                if spec.exact is not None:
                    spec_document["exact"] = list(spec.exact)
                if spec.constraints:
                    spec_document["constraints"] = list(spec.constraints)
                specs_document[pack] = spec_document
            document["venvSpecs"] = specs_document
        if self.venv_groups is not None:
            document["venvGroups"] = {name: list(members) for name, members in self.venv_groups}
        if self.in_process:
            document["inProcess"] = list(self.in_process)
        if self.runtime_pins:
            document["runtimePins"] = dict(self.runtime_pins)
        if self.generation_topology:
            document["generationTopology"] = True
        return canonical_json(document)


SNAPSHOT_FORMAT = "dinkster.snapshot/1"

_HASH_ANNOTATION = "--hash="

# An exact pin, structurally: a distribution name (PEP 503 shape) and
# ONE '==' to a concrete version built only from version characters.
# These strings end up in requirements files, where anything looser is
# ACTIVE SYNTAX - direct URLs (name?x==1), compact PEP 508 markers
# (name==1;python_version<'0'), extras, comments, backslash
# continuations, option lines - so the grammar is a full-match
# whitelist, not a blacklist of known smugglings.
_EXACT_PIN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"==(?:[0-9]+!)?[0-9A-Za-z]+(?:[.+][0-9A-Za-z]+)*"
)

_HASH_DIGEST = re.compile(r"[A-Za-z0-9_]+:[0-9a-fA-F]+")


def _decode_pins(
    pack: str, raw: object, *, label: str = "snapshot venv", allow_hashes: bool = True
) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise InstallError(f"{label} pins for pack {pack!r} must be a list of strings")
    pins: list[str] = []
    for pin in cast("list[object]", raw):
        if not isinstance(pin, str):
            raise InstallError(f"{label} pins for pack {pack!r} must be a list of strings")
        tokens = pin.split()
        if not tokens or _EXACT_PIN.fullmatch(tokens[0]) is None:
            raise InstallError(
                f"{label} pin {pin!r} for pack {pack!r} is not an exact 'name==version' pin"
            )
        # A pin may carry hash annotations (the resolution lock's
        # artifact-identity half): '--hash=<algo>:<hex>' tokens after
        # the exact pin, and nothing else - anything unrecognized is a
        # corrupted or hand-edited record, refused at load. Constraint
        # pins (allow_hashes=False) are version-only by contract:
        # cross-scope demotion strips artifact identity WITH the
        # exactness it belongs to, so a constraint carrying hashes is
        # itself a corrupt record, refused rather than stripped.
        for token in tokens[1:]:
            if not allow_hashes:
                raise InstallError(
                    f"{label} pin {pin!r} for pack {pack!r} carries an "
                    f"annotation {token!r}; constraints are version-only"
                )
            if (
                not token.startswith(_HASH_ANNOTATION)
                or _HASH_DIGEST.fullmatch(token.removeprefix(_HASH_ANNOTATION)) is None
            ):
                raise InstallError(
                    f"{label} pin {pin!r} for pack {pack!r} carries an "
                    f"unrecognized annotation {token!r}; only "
                    f"'--hash=<algo>:<hex digest>' may follow the exact pin"
                )
        pins.append(" ".join(tokens))
    return tuple(sorted(pins))


@dataclass(frozen=True)
class SnapshotRecord:
    """A complete environment record: the lockfile PLUS what range
    resolution decided per pack venv, plus the host facts that scope it
    (snapshots).

    The lockfile alone pins pack bytes exactly but venv provisioning
    resolves ``requires`` RANGES - the same lockfile staged months apart
    yields different torch/numpy. A snapshot closes that gap: per pack a
    sorted ``name==version`` freeze of its venv. Per-VENV, not global -
    the multiprocess model gives every pack its own interpreter, so two
    packs pinning different torch versions is a representable fact here,
    never a conflict to merge.

    A pack ABSENT from ``venvs`` is honestly unpinned (its venv was never
    staged or was provisioned with --no-venv); restore falls back to
    range resolution for it, visibly - no fake pins. ``python`` and
    ``platform`` are advisory scope: pins are wheel-platform-specific, so
    a snapshot reproduces faithfully on a like platform and is a
    starting point elsewhere, and restore says which.

    Pins may carry ``--hash=<algo>:<digest>`` annotations (captured with
    ``snapshot --hashes``): artifact identity next to version identity,
    verified by provisioning on exact replay. Annotations record EVERY
    artifact of the version, so they hold on any like-scope host; a
    cross-scope restore demotes pins to version constraints and the
    annotations drop with the exactness they belong to."""

    lockfile: Lockfile
    venvs: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """(pack name, sorted pins) pairs, sorted by pack name; every name
    must be locked by ``lockfile``."""
    python: str = ""
    """Host interpreter version the venvs were built with (advisory)."""
    platform: str = ""
    """Host platform tag, e.g. ``linux-x86_64`` (advisory)."""
    dinkster: str = ""
    """Dinkster version that captured the snapshot (advisory)."""
    accelerator: str = ""
    """Accelerator family the venvs were provisioned for (``cuda``/
    ``rocm``/``xpu``/``mps``/``cpu``; empty = not recorded, pre-scope
    snapshots). Advisory like ``platform`` - but pins carry vendor
    runtime wheels (nvidia-*, ROCm builds), so restore treats a
    DIFFERENT recorded accelerator as a scope mismatch: the freeze is
    not reused wholesale, ranges re-resolve for the destination."""
    runtime: tuple[tuple[str, str], ...] = ()
    """Accelerator runtime/toolchain facts at capture time, as sorted
    (key, value) pairs - e.g. (("cuda", "12.4"), ("driver", "550.54")) or
    (("rocm", "6.2.0"),). PURELY advisory (additive to
    ``dinkster.snapshot/1``; absent decodes as not-recorded): restore may
    narrate a difference, it never changes pin reuse - a driver bump is
    not a scope, and claiming otherwise would overstate what pins
    encode."""
    venv_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Capture-time hosting provenance as (group, sorted members) pairs.
    Additive; absent in older snapshots means no recorded groups."""
    in_process: tuple[str, ...] = ()
    """Capture-time host-process placement. Additive; absent means none."""
    runtime_pins: tuple[tuple[str, str], ...] = ()
    """Exact serving-runtime baseline for captured in-process packs."""

    def __post_init__(self) -> None:
        locked = {entry.pack for entry in self.lockfile.packs}
        for pack, _ in self.venvs:
            if pack not in locked:
                raise InstallError(
                    f"snapshot pins a venv for pack {pack!r} that the lockfile does not lock"
                )
        grouped: set[str] = set()
        for name, members in self.venv_groups:
            if len(members) < 2 or any(member not in locked for member in members):
                raise InstallError(f"snapshot venv group {name!r} has invalid members")
            if grouped.intersection(members):
                raise InstallError("snapshot records duplicate venv-group membership")
            grouped.update(members)
        if len(set(self.in_process)) != len(self.in_process):
            raise InstallError("snapshot records duplicate in-process membership")
        unknown_in_process = sorted(set(self.in_process) - locked)
        if unknown_in_process:
            raise InstallError(
                "snapshot records in-process packs not locked by the lockfile: "
                + ", ".join(unknown_in_process)
            )
        contradictions = sorted(grouped.intersection(self.in_process))
        if contradictions:
            raise InstallError(
                "snapshot records packs in both a venv group and in-process: "
                + ", ".join(contradictions)
            )
        pin_names = [name for name, _ in self.runtime_pins]
        if len(set(pin_names)) != len(pin_names):
            raise InstallError("snapshot records duplicate runtime pin names")
        if self.in_process and set(pin_names) != {"torch", "dinkster-aimdo"}:
            raise InstallError(
                "snapshot records in-process packs without exact torch and dinkster-aimdo pins"
            )
        if self.runtime_pins and not self.in_process:
            raise InstallError("snapshot records runtime pins without in-process packs")

    def pins_for(self, pack: str) -> tuple[str, ...] | None:
        """The exact dist list for ``pack``'s venv, or None when the
        snapshot recorded it unpinned."""
        for name, pins in self.venvs:
            if name == pack:
                return pins
        return None

    @classmethod
    def of(
        cls,
        lockfile: Lockfile,
        venvs: Mapping[str, Sequence[str]],
        *,
        python: str = "",
        platform: str = "",
        dinkster: str = "",
        accelerator: str = "",
        runtime: Mapping[str, str] | None = None,
        venv_groups: Mapping[str, Sequence[str]] | None = None,
        in_process: Sequence[str] = (),
        runtime_pins: Mapping[str, str] | None = None,
    ) -> SnapshotRecord:
        ordered = tuple((pack, tuple(sorted(venvs[pack]))) for pack in sorted(venvs))
        return cls(
            lockfile=lockfile,
            venvs=ordered,
            python=python,
            platform=platform,
            dinkster=dinkster,
            accelerator=accelerator,
            runtime=tuple(sorted(runtime.items())) if runtime else (),
            venv_groups=(
                tuple(
                    (name, tuple(sorted(members))) for name, members in sorted(venv_groups.items())
                )
                if venv_groups
                else ()
            ),
            in_process=tuple(sorted(in_process)),
            runtime_pins=tuple(sorted((runtime_pins or {}).items())),
        )

    @classmethod
    def from_record_json(cls, text: str) -> SnapshotRecord:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InstallError(f"snapshot file is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise InstallError("snapshot file must be a JSON object")
        record = cast("dict[str, object]", document)
        if record.get("format") != SNAPSHOT_FORMAT:
            raise InstallError(
                f"unsupported snapshot format {record.get('format')!r}; "
                f"expected {SNAPSHOT_FORMAT!r}"
            )
        lockfile_raw = record.get("lockfile")
        if not isinstance(lockfile_raw, dict):
            raise InstallError("snapshot 'lockfile' must be a lockfile object")
        lockfile = Lockfile.from_record_json(json.dumps(lockfile_raw))
        venvs_raw = record.get("venvs")
        if not isinstance(venvs_raw, dict):
            raise InstallError("snapshot 'venvs' must be an object of pack -> pins")
        venvs = {
            pack: _decode_pins(pack, pins)
            for pack, pins in cast("dict[str, object]", venvs_raw).items()
        }
        scope: dict[str, str] = {}
        # "accelerator" is additive to dinkster.snapshot/1: absent in older
        # snapshots and decoded as "" (not recorded), so existing files
        # keep loading unchanged.
        for key in ("python", "platform", "dinkster", "accelerator"):
            value = record.get(key, "")
            if not isinstance(value, str):
                raise InstallError(f"snapshot {key!r} must be a string")
            scope[key] = value
        # "runtime" is additive like "accelerator": absent in older
        # snapshots and decoded as not-recorded.
        runtime_raw = record.get("runtime")
        runtime: dict[str, str] = {}
        if runtime_raw is not None:
            if not isinstance(runtime_raw, dict):
                raise InstallError("snapshot 'runtime' must be an object of strings")
            for key, value in cast("dict[str, object]", runtime_raw).items():
                if not isinstance(value, str):
                    raise InstallError("snapshot 'runtime' must be an object of strings")
                runtime[key] = value
        groups_raw = record.get("venvGroups")
        groups: dict[str, tuple[str, ...]] = {}
        if groups_raw is not None:
            if not isinstance(groups_raw, dict):
                raise InstallError("snapshot 'venvGroups' must be an object")
            for name, members_raw in cast("dict[str, object]", groups_raw).items():
                if not isinstance(members_raw, list):
                    raise InstallError(f"snapshot venv group {name!r} must be a list of strings")
                members_object = cast("list[object]", members_raw)
                if not all(isinstance(member, str) for member in members_object):
                    raise InstallError(f"snapshot venv group {name!r} must be a list of strings")
                members = cast("list[str]", members_object)
                groups[name] = tuple(members)
        in_process_raw = record.get("inProcess", [])
        if not isinstance(in_process_raw, list) or not all(
            isinstance(pack, str) for pack in cast("list[object]", in_process_raw)
        ):
            raise InstallError("snapshot 'inProcess' must be a list of strings")
        runtime_pins_raw = record.get("runtimePins", {})
        if not isinstance(runtime_pins_raw, dict) or not all(
            isinstance(name, str) and isinstance(version, str)
            for name, version in cast("dict[object, object]", runtime_pins_raw).items()
        ):
            raise InstallError("snapshot 'runtimePins' must be an object of strings")
        return cls.of(
            lockfile,
            venvs,
            runtime=runtime,
            venv_groups=groups,
            in_process=cast("list[str]", in_process_raw),
            runtime_pins=cast("dict[str, str]", runtime_pins_raw),
            **scope,
        )

    def record_json(self) -> str:
        document: dict[str, object] = {
            "format": SNAPSHOT_FORMAT,
            "dinkster": self.dinkster,
            "python": self.python,
            "platform": self.platform,
            "accelerator": self.accelerator,
            "lockfile": json.loads(self.lockfile.record_json()),
            "venvs": {pack: list(pins) for pack, pins in self.venvs},
        }
        if self.runtime:  # omitted when not recorded, never an empty object
            document["runtime"] = dict(self.runtime)
        if self.venv_groups:
            document["venvGroups"] = {name: list(members) for name, members in self.venv_groups}
        if self.in_process:
            document["inProcess"] = list(self.in_process)
        if self.runtime_pins:
            document["runtimePins"] = dict(self.runtime_pins)
        return canonical_json(document)


def plan(current: Lockfile, target: Lockfile) -> InstallPlan:
    """The deterministic diff from ``current`` to ``target``, sorted by
    pack name. Pure - staging and activation are the host's, later."""
    steps: list[InstallStep] = []
    names = sorted({entry.pack for entry in current.packs} | {entry.pack for entry in target.packs})
    for name in names:
        have = current.get(name)
        want = target.get(name)
        if have is None and want is not None:
            steps.append(InstallStep("add", name, want))
        elif have is not None and want is None:
            steps.append(InstallStep("remove", name, None, from_version=have.version))
        elif have is not None and want is not None:
            if have.artifact_digest == want.artifact_digest:
                continue
            have_version = Version.parse(have.version)
            want_version = Version.parse(want.version)
            if want_version > have_version:
                action: StepAction = "upgrade"
            elif want_version < have_version:
                action = "downgrade"
            else:
                # Same version, different bytes: impossible for registry
                # releases (immutable) but legitimate for git/local dev
                # sources - named explicitly, never silently applied.
                action = "reinstall"
            steps.append(InstallStep(action, name, want, from_version=have.version))
    return InstallPlan(tuple(steps))


# ---------------------------------------------------------------------------
# Generations (activation is a pointer swap, never a mutation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    number: int
    lockfile: Lockfile

    @property
    def lockfile_digest(self) -> str:
        return self.lockfile.record_digest()


class GenerationLedger:
    """Activation history with job pins: activations only append
    (rollback included); ``prune`` reclaims old, unpinned generations.

    The model behind atomic activation: the host stages new pack venvs
    off to the side, then calls ``activate`` - one pointer swap. Old
    generations stay materialized until every job pinned to them ends,
    so a running job's installation is immutable for the job's lifetime.
    Rollback is ``activate`` with an old generation's lockfile: history
    appends, never rewrites.
    """

    def __init__(self) -> None:
        self._generations: list[Generation] = []
        self._pins: dict[str, int] = {}
        self._next_number = 1
        """Monotonic - pruning never frees a number for reuse, so a
        generation number is a stable identity for the ledger's lifetime."""

    @property
    def current(self) -> Generation | None:
        return self._generations[-1] if self._generations else None

    def generations(self) -> tuple[Generation, ...]:
        return tuple(self._generations)

    def activate(self, lockfile: Lockfile) -> Generation:
        """Make ``lockfile`` current. Idempotent for identical content;
        otherwise appends a new generation."""
        digest = lockfile.record_digest()
        current = self.current
        if current is not None and current.lockfile_digest == digest:
            return current
        generation = Generation(self._next_number, lockfile)
        self._next_number += 1
        self._generations.append(generation)
        return generation

    def rollback(self) -> Generation:
        """Re-activate the previous generation's content as a NEW
        generation - the audit trail keeps the failed activation."""
        if len(self._generations) < 2:
            raise InstallError("nothing to roll back to; only one generation exists")
        return self.activate(self._generations[-2].lockfile)

    def pin(self, job_id: str) -> Generation:
        """Pin the current generation for a starting job and return it.
        Idempotent: an already-pinned job gets the generation it started
        on - never the new current - so a job cannot migrate between
        installations mid-run."""
        current = self.current
        if current is None:
            raise InstallError("no generation has been activated; nothing to pin")
        existing = self._pins.get(job_id)
        if existing is not None:
            held = next((g for g in self._generations if g.number == existing), None)
            if held is None:  # unreachable: prune refuses pinned generations
                raise InstallError(f"job {job_id!r} pins missing generation {existing}")
            return held
        self._pins[job_id] = current.number
        return current

    def release(self, job_id: str) -> None:
        if job_id not in self._pins:
            raise InstallError(f"job {job_id!r} holds no generation pin")
        del self._pins[job_id]

    def prunable(self) -> tuple[Generation, ...]:
        """Generations safe to reclaim: not current, not pinned by any job."""
        current = self.current
        pinned = set(self._pins.values())
        return tuple(
            generation
            for generation in self._generations
            if generation is not current and generation.number not in pinned
        )

    def prune(self, number: int) -> Generation:
        """Drop one generation from the ledger; refuses the current one
        and anything a job still pins."""
        target = next((g for g in self._generations if g.number == number), None)
        if target is None:
            raise InstallError(f"no generation {number} exists")
        if target is self.current:
            raise InstallError(f"generation {number} is current and cannot be pruned")
        holders = sorted(job for job, pin in self._pins.items() if pin == number)
        if holders:
            raise InstallError(
                f"generation {number} is pinned by running jobs: {', '.join(holders)}"
            )
        self._generations.remove(target)
        return target


__all__ = [
    "LOCKFILE_FORMAT",
    "PLAN_FORMAT",
    "SNAPSHOT_FORMAT",
    "Generation",
    "GenerationLedger",
    "InstallError",
    "InstallPlan",
    "InstallStep",
    "LockedPack",
    "Lockfile",
    "PlanRecord",
    "SnapshotRecord",
    "StepAction",
    "plan",
]
