"""Pack launch configuration shared by generation preparation and composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from dinkster_security.policy import Artifact as SecurityArtifact
from dinkster_server import PackInfo, validate_comfy_args
from dinkster_values import TypeRegistry


@dataclass(frozen=True)
class PackSpec:
    """How one pack worker launches and attributes, beyond its manifest.

    The plain-``Path`` case (host interpreter, manifest-name attribution)
    needs none of this; a spec exists for packs like the ComfyUI compat
    workers, which run on a foreign interpreter with an environment
    contract and attribute one worker's nodes to several provenance
    entries (``comfy.<pack>`` per legacy pack).

    ``packs`` are the provenance-table entries this worker contributes and
    ``attribute`` maps each announced node type onto one of them; the
    defaults are the manifest's own name/presentation and constant
    attribution, i.e. exactly the plain-path behavior. Attributing to an
    id outside ``packs`` is a CompositionError - provenance is host
    configuration and fails loudly, never a silent fallback.
    """

    manifest: Path | str
    python: str | None = None
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    remote_token_directory: Path | None = None
    """Dedicated gateway-only directory, bound instead of a replaceable token inode."""
    aimdo: str = "off"
    vram_budgets: Mapping[str, int] = field(default_factory=dict)
    reserve_vram: int | None = None
    comfy_args: tuple[str, ...] = ()
    runtime_settings: bool = False
    """Refresh worker policy and ComfyUI argv from effective settings on reload."""
    require_catalog: bool = False
    """Refuse missing or stale installed declarations instead of importing at boot."""
    security_artifacts: tuple[SecurityArtifact, ...] = ()
    """Host-selected immutable release and dependency identities checked before execution."""
    start_timeout: float = 60.0
    packs: Mapping[str, PackInfo] | None = None
    attribute: Callable[[str], str] | None = None
    trust_reserved: bool = False
    """Allow this pack's manifest to claim reserved namespace roots
    (``std``, ``comfy``, ``core``, ``dinkster``). The local analog of the
    registry's grant table: manifests only claim, and a reserved claim
    composes only because the HOST vouched for the pack here - first-party
    compat/std wiring sets it, ordinary ``--pack`` entries never do."""
    asset_roots: Mapping[str, Path] = field(default_factory=dict)
    """Artifact roots for packaged asset acquisition, per pack-table id.
    Plain packs default to the manifest directory for the manifest's own
    id; specs that carry their own ``packs`` table (the compat workers'
    legacy entries) supply roots here for entries whose declarations name
    packaged files. An entry without a root still composes - its packaged
    sources just report "not present" at acquisition while remote leads
    keep working."""
    host_types: Callable[[TypeRegistry], None] | None = None
    """Host-side type registrations that accompany this worker's surface.

    A foreign-interpreter worker's value types normally register only
    inside the worker; the host still carries, caches and relays what it
    cannot load (DESIGN 3.2). When the HOST must load or render such
    values - the compat workers' comfy.IMAGE previews - the spec declares
    those registrations here and add_pack applies them to the composed
    registry at its commit point. Must be idempotent (guard with
    ``type_id in registry``): two specs may share one hook, reload
    re-applies specs, and the registry is additive for the process
    lifetime - remove_pack never unregisters a type."""
    extension_config: Mapping[str, object] = field(default_factory=dict)
    """Host-resolved behavior configuration for this pack's extension.

    Float values are accepted at this producer boundary but canonicalized to
    strings before entering BehaviorValue, whose vocabulary deliberately has
    no float. Other non-RPC-clean values are refused before publication.
    """
    execution_config: Mapping[str, str] = field(default_factory=dict)
    """Non-secret settings that rotate this pack's execution cache identity."""
    node_unavailability: Mapping[str, str] = field(default_factory=dict)
    """Sealed admission refusals for nodes that require resident objects."""
    worker_group: str | None = None
    """Host-only H2 process group name, absent for a solo worker."""
    group_manifests: tuple[Path, ...] = ()
    """All manifests in worker_group, supplied by generation topology."""
    in_process: bool = False
    """Host-only H3 placement; pack code loads in the serving process."""
    optional_execution: tuple[str, ...] = ()
    """Schema-only types retained as metadata when no provider is configured.

    These remain absent from the executable catalog until a provider composes.
    Other schema-only types still require a provider for a complete generation.
    """
    runtime_pins: Mapping[str, str] = field(default_factory=dict)
    """Exact torch/comfy-aimdo baseline recorded by the install generation."""
    asset_vault_write: bool = False
    """Allow this trusted isolated pack to ingest verified assets into the host vault."""
    replica_cuda_indices: tuple[int, ...] = ()
    """Parent-visible CUDA indices for homogeneous process replicas."""
    single_job_cuda_indices: tuple[int, ...] = ()
    """Parent-visible CUDA indices for one process-isolated sampling workgroup."""
    single_job_mode: str = "auto"
    part: str | None = None
    """Execution selector within one published pack; absent selects its main entry."""

    def __post_init__(self) -> None:
        if self.aimdo not in ("off", "auto", "on"):
            raise ValueError(f"PackSpec aimdo must be 'off', 'auto', or 'on', got {self.aimdo!r}")
        if self.replica_cuda_indices and (
            len(self.replica_cuda_indices) < 2
            or any(type(index) is not int or index < 0 for index in self.replica_cuda_indices)
            or len(set(self.replica_cuda_indices)) != len(self.replica_cuda_indices)
        ):
            raise ValueError(
                "PackSpec replica_cuda_indices must contain at least two unique non-negative ints"
            )
        if self.single_job_cuda_indices and (
            len(self.single_job_cuda_indices) < 2
            or any(type(index) is not int or index < 0 for index in self.single_job_cuda_indices)
            or len(set(self.single_job_cuda_indices)) != len(self.single_job_cuda_indices)
        ):
            raise ValueError(
                "PackSpec single_job_cuda_indices must contain at least two unique "
                "non-negative ints"
            )
        if self.replica_cuda_indices and self.single_job_cuda_indices:
            raise ValueError("replica and single-job CUDA indices are mutually exclusive")
        if self.in_process and self.asset_vault_write:
            raise ValueError("an in-process PackSpec cannot request asset vault write access")
        if self.single_job_mode not in ("auto", "guidance", "sequence"):
            raise ValueError("PackSpec single_job_mode is invalid")
        budgets: dict[str, int] = {}
        for residency, nbytes in self.vram_budgets.items():
            prefix = "vram:cuda:"
            index = residency.removeprefix(prefix)
            if not residency.startswith(prefix) or not index.isdigit():
                raise ValueError(
                    f"PackSpec vram_budgets keys must be vram:cuda:N, got {residency!r}"
                )
            value = int(nbytes)
            if value < 0:
                raise ValueError("PackSpec vram_budgets values must be non-negative")
            budgets[residency] = value
        object.__setattr__(self, "vram_budgets", MappingProxyType(budgets))
        if self.reserve_vram is not None and self.reserve_vram < 0:
            raise ValueError("PackSpec reserve_vram must be non-negative")
        object.__setattr__(
            self,
            "comfy_args",
            validate_comfy_args(self.comfy_args, require_tuple=True),
        )
        object.__setattr__(self, "runtime_pins", MappingProxyType(dict(self.runtime_pins)))
        object.__setattr__(
            self, "node_unavailability", MappingProxyType(dict(self.node_unavailability))
        )
        execution_config = dict(self.execution_config)
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in execution_config.items()
        ):
            raise ValueError("PackSpec execution_config must map non-empty strings to strings")
        object.__setattr__(self, "execution_config", MappingProxyType(execution_config))
