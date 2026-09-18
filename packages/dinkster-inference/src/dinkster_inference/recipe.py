"""Declarative native-runtime reconstruction recipes and patch overlays.

Recipes contain only frozen, RPC-clean data. Asset content keeps Dinkster's
existing ``blake3:<hex>`` identity; patch structure uses sha256 over a
canonical JSON description because it identifies decoded behavior rather
than an asset payload. Filesystem paths, live tensors, modules, resolvers,
and callables never enter these values.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from dinkster_protocol import AttentionPolicy, AttentionRouteToken, resolve_attention_runtime_status

from .lora import DecodedPatch, PatchTarget

_ASSET_DIGEST_RE = re.compile(r"^blake3:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_PROVIDER_ID_RE = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
_DISTRIBUTION_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_DEPENDENCY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
LOCAL_SAFETENSORS_PROVIDER = "dinkster.asset.local_safetensors.v1"

PlainValue: TypeAlias = str | int | bool | None


@dataclass(frozen=True)
class ProviderPin:
    """Exact package provenance for a future internal storage provider."""

    provider_id: str
    distribution: str
    version: str
    source_revision: str
    wheel_sha256: str

    def __post_init__(self) -> None:
        values = cast(
            "tuple[tuple[str, object, re.Pattern[str]], ...]",
            (
                ("provider_id", self.provider_id, _PROVIDER_ID_RE),
                ("distribution", self.distribution, _DISTRIBUTION_RE),
                ("version", self.version, _VERSION_RE),
                ("source_revision", self.source_revision, _SOURCE_REVISION_RE),
                ("wheel_sha256", self.wheel_sha256, _SHA256_RE),
            ),
        )
        for name, value, pattern in values:
            if not isinstance(value, str):
                raise TypeError(f"provider pin {name} must be a string")
            if pattern.fullmatch(value) is None:
                raise ValueError(f"provider pin {name} must be non-empty and canonical")
        if any(segment != str(int(segment)) for segment in self.version.split(".")):
            raise ValueError("provider pin version must be non-empty and canonical")

    def to_wire(self) -> str:
        """Return the canonical JSON spelling of this provenance value."""
        return json.dumps(
            {
                "distribution": self.distribution,
                "providerId": self.provider_id,
                "sourceRevision": self.source_revision,
                "version": self.version,
                "wheelSha256": self.wheel_sha256,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_wire(cls, wire: str, *, expected: ProviderPin) -> ProviderPin:
        """Strictly decode one canonical pin and reject provenance drift."""
        if not isinstance(cast("object", wire), str):
            raise TypeError("provider pin wire must be a string")
        if not isinstance(cast("object", expected), ProviderPin):
            raise TypeError("expected provider pin must be a ProviderPin")
        try:
            document_obj = json.loads(wire)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("invalid provider pin wire") from exc
        if not isinstance(document_obj, dict):
            raise ValueError("provider pin wire must contain exact fields")
        document = cast("dict[object, object]", document_obj)
        if set(document) != {
            "distribution",
            "providerId",
            "sourceRevision",
            "version",
            "wheelSha256",
        }:
            raise ValueError("provider pin wire must contain exact fields")
        raw_values = (
            document["providerId"],
            document["distribution"],
            document["version"],
            document["sourceRevision"],
            document["wheelSha256"],
        )
        if not all(isinstance(value, str) for value in raw_values):
            raise ValueError("provider pin wire fields must be strings")
        provider_id, distribution, version, source_revision, wheel_sha256 = cast(
            "tuple[str, str, str, str, str]", raw_values
        )
        pin = cls(
            provider_id=provider_id,
            distribution=distribution,
            version=version,
            source_revision=source_revision,
            wheel_sha256=wheel_sha256,
        )
        if pin.to_wire() != wire:
            raise ValueError("provider pin wire is not canonical provider pin wire")
        if pin != expected:
            raise ValueError("provider pin wire does not match expected provider pin")
        return pin


GGUF_PROVIDER_PIN = ProviderPin(
    provider_id="ggml.gguf-py.v0",
    distribution="gguf",
    version="0.19.0",
    source_revision="a290ce626663dae1d54f70bce3ca6d8f67aab62f",
    wheel_sha256="70bcd10edfe697fb2dad6e40af2234b9d8ece9a41a99761405121ebda1c3c1cd",
)


def canonical_float(value: float, *, name: str = "value") -> str:
    """Return BehaviorValue's stable, float-free spelling."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return format(value, ".17g")


def _require_pairs(name: str, values: tuple[tuple[str, PlainValue], ...]) -> None:
    values_obj = cast("object", values)
    if not isinstance(values_obj, tuple):
        raise TypeError(f"{name} must be a tuple")
    keys: list[str] = []
    for key, value in cast("tuple[tuple[object, object], ...]", values_obj):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if value is not None and type(value) not in (str, int, bool):
            raise TypeError(f"{name} values must be RPC-clean plain data")
        keys.append(key)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError(f"{name} keys must be sorted and unique")


@dataclass(frozen=True)
class WeightSourceRef:
    """Path-free reference to one safetensors asset.

    ``provider_config`` is intentionally empty for the v1 local provider:
    the worker's existing AssetResolver configuration owns digest-to-path
    resolution. Metadata is retained so a worker can reconstruct an
    AssetRef without inventing file identity.
    """

    digest: str
    name: str
    size: int
    media_type: str = "application/octet-stream"
    virtual_path: str = ""
    provider: str = LOCAL_SAFETENSORS_PROVIDER
    provider_config: tuple[tuple[str, PlainValue], ...] = ()

    def __post_init__(self) -> None:
        text_values = cast(
            "tuple[object, ...]",
            (
                self.digest,
                self.name,
                self.media_type,
                self.virtual_path,
                self.provider,
            ),
        )
        if not all(isinstance(value, str) for value in text_values):
            raise TypeError("weight source text fields must be strings")
        if _ASSET_DIGEST_RE.fullmatch(self.digest) is None:
            raise ValueError("weight source digest must be a canonical blake3 asset digest")
        if type(self.size) is not int:
            raise TypeError("weight source size must be an integer")
        if self.size < 0:
            raise ValueError("weight source size must be non-negative")
        if self.provider != LOCAL_SAFETENSORS_PROVIDER:
            raise ValueError(f"unsupported weight source provider {self.provider!r}")
        _require_pairs("provider_config", self.provider_config)
        if self.provider_config:
            raise ValueError("the v1 local safetensors provider has no per-source config")


@dataclass(frozen=True)
class WeightSourceBinding:
    """One named input slot in a reconstruction recipe."""

    role: str
    source: WeightSourceRef

    def __post_init__(self) -> None:
        role = cast("object", self.role)
        if not isinstance(role, str) or not role:
            raise ValueError("weight source role must be non-empty")
        if not isinstance(cast("object", self.source), WeightSourceRef):
            raise TypeError("weight source binding must contain a WeightSourceRef")


@dataclass(frozen=True)
class ProviderPatchRef:
    """RPC-clean input to one worker-local extension patch provider."""

    provider_id: str
    parameters: tuple[tuple[str, PlainValue], ...]

    def __post_init__(self) -> None:
        provider_id = cast("object", self.provider_id)
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("patch provider id must be non-empty")
        _require_pairs("patch provider parameters", self.parameters)


@dataclass(frozen=True)
class OverlayPatch:
    """One decoded patch routed to a concrete runtime component."""

    component: str
    target: PatchTarget
    decoded: DecodedPatch | ProviderPatchRef

    def __post_init__(self) -> None:
        component = cast("object", self.component)
        if not isinstance(component, str) or not component:
            raise ValueError("overlay patch component must be non-empty")
        if not isinstance(cast("object", self.target), PatchTarget):
            raise TypeError("overlay patch target must be a PatchTarget")
        if not isinstance(cast("object", self.decoded), DecodedPatch | ProviderPatchRef):
            raise TypeError("overlay patch decoded value is unsupported")


def _canonical_value(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in cast("tuple[object, ...]", value)]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            "fields": {
                field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)
            },
        }
    raise TypeError(f"patch overlay canonicalization cannot encode {type(value).__name__}")


def _patch_sort_key(patch: OverlayPatch) -> tuple[object, ...]:
    offset = patch.target.offset
    offset_key = (-1, -1, -1) if offset is None else (offset.dim, offset.start, offset.length)
    decoded = json.dumps(
        _canonical_value(patch.decoded),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (patch.component, patch.target.key, *offset_key, decoded)


def canonical_patch_overlay(overlay: PatchOverlay) -> str:
    """Canonical decoded behavior used by ``structural_digest``."""
    document = {
        "dialect": overlay.dialect,
        "keyMap": overlay.key_map,
        "patches": [_canonical_value(patch) for patch in overlay.patches],
        "provider": overlay.source.provider,
        "sourceDigest": overlay.source.digest,
        "strengthClip": overlay.strength_clip,
        "strengthModel": overlay.strength_model,
        "version": 1,
    }
    return json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class PatchOverlay:
    """One normalized, deterministically identified weight-patch source."""

    source: WeightSourceRef
    dialect: str
    key_map: str
    strength_model: str
    strength_clip: str
    patches: tuple[OverlayPatch, ...]
    structural_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.source), WeightSourceRef):
            raise TypeError("patch overlay source must be a WeightSourceRef")
        dialect = cast("object", self.dialect)
        key_map = cast("object", self.key_map)
        if (
            not isinstance(dialect, str)
            or not dialect
            or not isinstance(key_map, str)
            or not key_map
        ):
            raise ValueError("patch overlay dialect and key_map must be non-empty")
        patches_obj = cast("object", self.patches)
        if not isinstance(patches_obj, tuple) or not all(
            isinstance(patch, OverlayPatch) for patch in cast("tuple[object, ...]", patches_obj)
        ):
            raise TypeError("patch overlay patches must be a tuple of OverlayPatch")
        if not isinstance(cast("object", self.structural_digest), str):
            raise TypeError("patch overlay structural_digest must be a string")
        for name, value in (
            ("strength_model", self.strength_model),
            ("strength_clip", self.strength_clip),
        ):
            value_obj = cast("object", value)
            if not isinstance(value_obj, str):
                raise TypeError(f"{name} must use a canonical string")
            try:
                parsed = float(value_obj)
            except ValueError:
                raise ValueError(f"{name} must be a canonical finite float") from None
            if canonical_float(parsed, name=name) != value:
                raise ValueError(f"{name} is not canonically encoded")
        ordered = tuple(sorted(self.patches, key=_patch_sort_key))
        if ordered != self.patches:
            raise ValueError("overlay patches must be in canonical order")
        target_keys = [
            (
                patch.component,
                patch.target.key,
                None
                if patch.target.offset is None
                else (
                    patch.target.offset.dim,
                    patch.target.offset.start,
                    patch.target.offset.length,
                ),
            )
            for patch in self.patches
        ]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("overlay patch targets must be unique")
        computed = hashlib.sha256(canonical_patch_overlay(self).encode("utf-8")).hexdigest()
        if self.structural_digest and self.structural_digest != computed:
            raise ValueError("patch overlay structural_digest does not match its data")
        object.__setattr__(self, "structural_digest", computed)

    @classmethod
    def from_decoded(
        cls,
        *,
        source: WeightSourceRef,
        dialect: str,
        key_map: str,
        strength_model: float,
        strength_clip: float,
        patches: tuple[OverlayPatch, ...],
    ) -> PatchOverlay:
        return cls(
            source=source,
            dialect=dialect,
            key_map=key_map,
            strength_model=canonical_float(strength_model, name="strength_model"),
            strength_clip=canonical_float(strength_clip, name="strength_clip"),
            patches=tuple(sorted(patches, key=_patch_sort_key)),
        )


def patch_overlay_stack_digest(overlays: tuple[PatchOverlay, ...]) -> str | None:
    """Stable order-sensitive identity for a declared overlay stack."""
    if not overlays:
        return None
    canonical = json.dumps(
        [overlay.structural_digest for overlay in overlays],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


ATTACHMENT_CLONE_MODES = ("share", "copy", "rebuild", "refuse")


@dataclass(frozen=True)
class AttachmentDeclaration:
    """Serializable contract for extension-owned state on a model handle."""

    name: str
    clone: str
    device: str
    rebuild_data: tuple[tuple[str, PlainValue], ...] | None = None
    rebuild_entry_point: str | None = None

    def __post_init__(self) -> None:
        name = cast("object", self.name)
        clone = cast("object", self.clone)
        device = cast("object", self.device)
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(clone, str)
            or not isinstance(device, str)
            or not device
        ):
            raise ValueError("attachment name and device must be non-empty")
        if self.clone not in ATTACHMENT_CLONE_MODES:
            raise ValueError(f"attachment clone mode must be one of {ATTACHMENT_CLONE_MODES}")
        if self.rebuild_data is not None:
            _require_pairs("attachment rebuild_data", self.rebuild_data)
        entry = self.rebuild_entry_point
        if entry is not None and (
            entry.count(":") != 1
            or not all(entry.split(":"))
            or any(char.isspace() for char in entry)
        ):
            raise ValueError("attachment rebuild_entry_point must be module:attr")

    @property
    def rebuildable(self) -> bool:
        return self.rebuild_data is not None or self.rebuild_entry_point is not None


@dataclass(frozen=True)
class RuntimeKnobs:
    """Execution decisions already carried by native runtime identity."""

    diffusion_dtype: str
    text_dtype: str
    vae_dtype: str
    fp8_matmul: bool
    registry_token: str | None = None
    extension_behavior_hash: str | None = None
    embedding_binding_digest: str | None = None
    attention_policy: AttentionPolicy = "auto"
    attention_route_token: AttentionRouteToken | None = None
    runtime_facts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dtype_names = cast(
            "tuple[object, ...]",
            (self.diffusion_dtype, self.text_dtype, self.vae_dtype),
        )
        if not all(isinstance(value, str) and value for value in dtype_names):
            raise ValueError("runtime dtype names must be non-empty")
        if type(self.fp8_matmul) is not bool:
            raise TypeError("fp8_matmul must be a boolean")
        registry_token = cast("object", self.registry_token)
        if registry_token is not None and not isinstance(registry_token, str):
            raise TypeError("registry_token must be a string or None")
        digest = cast("object", self.extension_behavior_hash)
        if digest is not None and (
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError("extension_behavior_hash must be a lowercase sha256 hex digest")
        embedding_digest = cast("object", self.embedding_binding_digest)
        if embedding_digest is not None and (
            not isinstance(embedding_digest, str) or _SHA256_RE.fullmatch(embedding_digest) is None
        ):
            raise ValueError("embedding_binding_digest must be a lowercase sha256 hex digest")
        facts = cast("object", self.runtime_facts)
        if not isinstance(facts, tuple) or any(
            not isinstance(fact, str) or not fact for fact in cast("tuple[object, ...]", facts)
        ):
            raise ValueError("runtime_facts must be a tuple of non-empty strings")
        resolve_attention_runtime_status(self.attention_policy, self.attention_route_token)


DependencyScope = Literal["model", "contribution", "invocation", "conditional"]
DependencyCloneMode = Literal["with-parent", "shared"]
ChildT = TypeVar("ChildT")


@dataclass(frozen=True)
class DependencyRef(Generic[ChildT]):
    """One declarative child edge in a reconstruction dependency graph."""

    child_id: str
    child: ChildT
    residency_group: str
    scope: DependencyScope
    clone_mode: DependencyCloneMode
    accounting_owner: str

    def __post_init__(self) -> None:
        for name, value in (
            ("child_id", self.child_id),
            ("residency_group", self.residency_group),
            ("accounting_owner", self.accounting_owner),
        ):
            if not isinstance(cast("object", value), str):
                raise TypeError(f"dependency {name} must be a string")
            if _DEPENDENCY_ID_RE.fullmatch(value) is None:
                raise ValueError(f"dependency {name} must be a canonical id")
        if self.child_id == "parent":
            raise ValueError("dependency child_id 'parent' is reserved")
        if self.scope not in ("model", "contribution", "invocation", "conditional"):
            raise ValueError("dependency scope is unsupported")
        if self.clone_mode not in ("with-parent", "shared"):
            raise ValueError("dependency clone_mode is unsupported")


@dataclass(frozen=True)
class ReconstructionRecipe:
    """Complete declarative identity and materialization recipe for a handle."""

    sources: tuple[WeightSourceBinding, ...]
    family_id: str
    component_identity: tuple[str, ...]
    knobs: RuntimeKnobs
    overlays: tuple[PatchOverlay, ...] = ()
    attachments: tuple[AttachmentDeclaration, ...] = ()
    dependencies: tuple[DependencyRef[ReconstructionRecipe], ...] = ()

    def __post_init__(self) -> None:
        family_id = cast("object", self.family_id)
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("recipe family_id must be non-empty")
        values = (
            ("sources", self.sources, WeightSourceBinding),
            ("overlays", self.overlays, PatchOverlay),
            ("attachments", self.attachments, AttachmentDeclaration),
            ("dependencies", self.dependencies, DependencyRef),
        )
        for name, value, item_type in values:
            value_obj = cast("object", value)
            if not isinstance(value_obj, tuple) or not all(
                isinstance(item, item_type) for item in cast("tuple[object, ...]", value_obj)
            ):
                raise TypeError(f"recipe {name} must be a tuple of {item_type.__name__}")
        components_obj = cast("object", self.component_identity)
        if not isinstance(components_obj, tuple) or not all(
            isinstance(component, str) for component in cast("tuple[object, ...]", components_obj)
        ):
            raise TypeError("recipe component_identity must be a tuple of strings")
        if not isinstance(cast("object", self.knobs), RuntimeKnobs):
            raise TypeError("recipe knobs must be RuntimeKnobs")
        roles = [binding.role for binding in self.sources]
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("recipe weight source roles must be non-empty and unique")
        if tuple(roles) != tuple(sorted(roles)):
            raise ValueError("recipe weight sources must be ordered by role")
        if not self.component_identity or self.component_identity[0] != (
            f"family={self.family_id}"
        ):
            raise ValueError("recipe component identity must begin with its family")
        attachment_names = [attachment.name for attachment in self.attachments]
        if len(attachment_names) != len(set(attachment_names)):
            raise ValueError("recipe attachment names must be unique")
        if not all(
            isinstance(cast("object", dependency.child), ReconstructionRecipe)
            for dependency in self.dependencies
        ):
            raise TypeError("recipe dependency children must be ReconstructionRecipe values")
        child_ids = [dependency.child_id for dependency in self.dependencies]
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("recipe dependency child_ids must be unique")
        child_groups = {
            dependency.child_id: dependency.residency_group for dependency in self.dependencies
        }
        group_owners: dict[str, str] = {}
        for dependency in self.dependencies:
            owner = dependency.accounting_owner
            if owner != "parent" and owner not in child_groups:
                raise ValueError("recipe dependency accounting_owner is unknown")
            if owner != "parent" and child_groups[owner] != dependency.residency_group:
                raise ValueError(
                    "recipe dependency accounting_owner must be in the same residency_group"
                )
            previous_owner = group_owners.setdefault(dependency.residency_group, owner)
            if previous_owner != owner:
                raise ValueError(
                    "recipe dependencies in a residency_group must share accounting_owner"
                )
        self._refuse_dependency_cycles(set())

    def _refuse_dependency_cycles(self, active: set[int]) -> None:
        recipe_id = id(self)
        if recipe_id in active:
            raise ValueError("recipe dependency graph must be acyclic")
        active.add(recipe_id)
        for dependency in self.dependencies:
            dependency.child._refuse_dependency_cycles(active)
        active.remove(recipe_id)

    @property
    def runtime_identity(self) -> str:
        from .identity import build_runtime_identity_from_facts

        return build_runtime_identity_from_facts(
            self.family_id,
            self.component_identity,
            diffusion_dtype=self.knobs.diffusion_dtype,
            text_dtype=self.knobs.text_dtype,
            vae_dtype=self.knobs.vae_dtype,
            fp8_matmul=self.knobs.fp8_matmul,
            registry_token=self.knobs.registry_token,
            extension_behavior_hash=self.knobs.extension_behavior_hash,
            embedding_binding_digest=self.knobs.embedding_binding_digest,
            attention_policy=self.knobs.attention_policy,
            attention_route_token=self.knobs.attention_route_token,
            runtime_facts=self.knobs.runtime_facts,
            patch_overlay_digests=tuple(overlay.structural_digest for overlay in self.overlays),
            dependency_facts=tuple(
                (
                    dependency.child_id,
                    dependency.residency_group,
                    dependency.scope,
                    dependency.clone_mode,
                    dependency.accounting_owner,
                    dependency.child.runtime_identity,
                )
                for dependency in self.dependencies
            ),
        )

    @property
    def patch_stack_digest(self) -> str | None:
        return patch_overlay_stack_digest(self.overlays)

    def append_overlays(self, overlays_delta: tuple[PatchOverlay, ...]) -> ReconstructionRecipe:
        overlays_obj = cast("object", overlays_delta)
        if not isinstance(overlays_obj, tuple):
            raise TypeError("overlays_delta must be a tuple")
        if not all(
            isinstance(overlay, PatchOverlay)
            for overlay in cast("tuple[object, ...]", overlays_obj)
        ):
            raise TypeError("overlays_delta must contain PatchOverlay values")
        return replace(self, overlays=self.overlays + overlays_delta)

    def append_dependencies(
        self,
        dependencies_delta: tuple[DependencyRef[ReconstructionRecipe], ...],
    ) -> ReconstructionRecipe:
        """Return a validated ordered dependency append without mutation."""
        dependencies_obj = cast("object", dependencies_delta)
        if not isinstance(dependencies_obj, tuple):
            raise TypeError("dependencies_delta must be a tuple")
        if not all(
            isinstance(dependency, DependencyRef)
            for dependency in cast("tuple[object, ...]", dependencies_obj)
        ):
            raise TypeError("dependencies_delta must contain DependencyRef values")
        return replace(
            self,
            dependencies=self.dependencies + dependencies_delta,
        )


__all__ = [
    "ATTACHMENT_CLONE_MODES",
    "DependencyCloneMode",
    "DependencyRef",
    "DependencyScope",
    "LOCAL_SAFETENSORS_PROVIDER",
    "AttachmentDeclaration",
    "OverlayPatch",
    "PatchOverlay",
    "PlainValue",
    "ProviderPatchRef",
    "ReconstructionRecipe",
    "RuntimeKnobs",
    "WeightSourceBinding",
    "WeightSourceRef",
    "canonical_float",
    "canonical_patch_overlay",
    "patch_overlay_stack_digest",
]
