"""Worker-local sampler extension materialization.

The parent writes import recipes and RPC-clean declarations to a local catalog
keyed by the pinned extension snapshot digest. Only that opaque key crosses
the worker RPC. The sampling worker imports callables into its own interpreter,
builds the existing Registry, and validates the resulting declarations in both
directions before the registry can execute.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dinkster_protocol import (
    GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
    GRAPH_COMPILERS_SURFACE,
    GUIDANCE_ATTENTION_SURFACE,
    GUIDANCE_PLAN_AUGMENTATION_SURFACE,
    GUIDANCE_SURFACES,
    GraphCompilerRegistrySnapshot,
    GuidanceRegistrySnapshot,
    KeyedContribution,
    SamplerRegistrySnapshot,
)

from .attention import (
    AttentionContribution,
    attention_declarations,
    check_attention_pins,
)
from .component_registry import ComponentDescriptor
from .families import ModelFamily
from .graph_compilers import (
    GraphCompilerDescriptor,
    InferenceGraphCompileError,
    execute_graph_compilers,
    graph_compiler_declaration_metadata,
)
from .guidance import GuidanceContribution
from .registries import InferenceRegistries, builtin_registries
from .registries import merge as merge_registries
from .registry import Registry
from .runtime import AssemblyRegistration
from .sampling import OptionSpec, SamplerDescriptor, SchedulerDescriptor

INFERENCE_SAMPLERS_SURFACE = "inference.samplers"
INFERENCE_SCHEDULERS_SURFACE = "inference.schedulers"
INFERENCE_FAMILIES_SURFACE = "inference.families"
INFERENCE_COMPONENTS_SURFACE = "inference.components"
INFERENCE_ASSEMBLIES_SURFACE = "inference.assemblies"
SAMPLER_CATALOG_ENV = "DINKSTER_INFERENCE_CATALOG"
_CATALOG_FORMAT = "dinkster.sampler-catalog-v1"


@dataclass(frozen=True)
class SamplerContribution:
    """Explicit result of one pack's inference-scope entry point."""

    samplers: tuple[SamplerDescriptor[Any], ...]

    def __post_init__(self) -> None:
        samplers = cast("object", self.samplers)
        if not isinstance(samplers, tuple) or not samplers:
            raise TypeError("samplers must be a non-empty tuple")
        if not all(
            isinstance(sampler, SamplerDescriptor)
            for sampler in cast("tuple[object, ...]", samplers)
        ):
            raise TypeError("samplers must contain SamplerDescriptor values")


@dataclass(frozen=True)
class InferenceContribution:
    """Additive result of one pack's single inference entry point."""

    samplers: tuple[SamplerDescriptor[Any], ...] = ()
    schedulers: tuple[SchedulerDescriptor, ...] = ()
    guidance: GuidanceContribution[Any] | None = None
    graph_compilers: tuple[GraphCompilerDescriptor, ...] = ()
    families: tuple[ModelFamily, ...] = ()
    components: tuple[ComponentDescriptor, ...] = ()
    assemblies: tuple[AssemblyRegistration, ...] = ()
    attention: AttentionContribution[Any] | None = None

    def __post_init__(self) -> None:
        if (
            not self.samplers
            and not self.schedulers
            and self.guidance is None
            and not self.graph_compilers
            and not self.families
            and not self.components
            and not self.assemblies
            and self.attention is None
        ):
            raise ValueError("inference contribution must be nonempty")
        samplers = cast("object", self.samplers)
        if not isinstance(samplers, tuple) or not all(
            isinstance(item, SamplerDescriptor) for item in cast("tuple[object, ...]", samplers)
        ):
            raise TypeError("samplers must contain SamplerDescriptor values")
        schedulers = cast("object", self.schedulers)
        if not isinstance(schedulers, tuple) or not all(
            isinstance(item, SchedulerDescriptor) for item in cast("tuple[object, ...]", schedulers)
        ):
            raise TypeError("schedulers must contain SchedulerDescriptor values")
        compilers = cast("object", self.graph_compilers)
        if not isinstance(compilers, tuple) or not all(
            isinstance(item, GraphCompilerDescriptor)
            for item in cast("tuple[object, ...]", compilers)
        ):
            raise TypeError("graph_compilers must contain GraphCompilerDescriptor values")
        GraphCompilerRegistrySnapshot(
            tuple(graph_compiler_declaration(item) for item in self.graph_compilers)
        )
        for name, values, expected in (
            ("families", self.families, ModelFamily),
            ("components", self.components, ComponentDescriptor),
            ("assemblies", self.assemblies, AssemblyRegistration),
        ):
            if not isinstance(cast("object", values), tuple) or not all(
                isinstance(item, expected) for item in cast("tuple[object, ...]", values)
            ):
                raise TypeError(f"{name} must contain {expected.__name__} values")
        raw_attention = cast("object", self.attention)
        if raw_attention is not None and not isinstance(raw_attention, AttentionContribution):
            raise TypeError("attention must be an AttentionContribution or None")


@dataclass(frozen=True)
class MaterializedInferenceGeneration:
    registries: InferenceRegistries
    sampler_snapshot: SamplerRegistrySnapshot
    guidance_snapshot: GuidanceRegistrySnapshot
    extensions: tuple[tuple[str, tuple[KeyedContribution, ...]], ...]
    module_prefixes: tuple[str, ...]
    guidance_contributions: tuple[tuple[str, GuidanceContribution[Any]], ...]
    graph_compiler_snapshot: GraphCompilerRegistrySnapshot
    graph_compilers: tuple[GraphCompilerDescriptor, ...]
    attention_contributions: tuple[tuple[str, AttentionContribution[Any]], ...] = ()


_GUIDANCE_SURFACES = GUIDANCE_SURFACES
_PLAN_AUGMENTATION_SURFACE = GUIDANCE_PLAN_AUGMENTATION_SURFACE
_GUIDANCE_ATTENTION_SURFACE = GUIDANCE_ATTENTION_SURFACE


def guidance_declarations(
    contribution: GuidanceContribution[Any], *, attention_order: int = 0
) -> tuple[KeyedContribution, ...]:
    """Project worker-local callbacks to canonical RPC-clean declarations.

    Plan-augmentation and attention-kind declarations project onto their
    canonical guidance surfaces, so their behavior stays visible to the
    extension's declared identity, the registry snapshot, and behavior
    identity. Both remain model-owned at runtime: a family consumes them
    only by positively opting in during model evaluation.

    The legacy attention descriptor has no ``order`` field; its projected
    ``order`` metadata is ``attention_order``, its position in declaration
    order across the generation, so the canonical snapshot preserves the
    order the rewrites execute in instead of re-sorting ties by id.
    """
    declarations: list[KeyedContribution] = []
    for surface, descriptors in (
        (_GUIDANCE_SURFACES[0], contribution.evaluation_wrappers),
        (_GUIDANCE_SURFACES[1], contribution.pre_cfg),
        (_GUIDANCE_SURFACES[3], contribution.post_cfg),
        (_PLAN_AUGMENTATION_SURFACE, contribution.plan_augmentations),
    ):
        for descriptor in sorted(descriptors, key=lambda item: (item.order, item.id)):
            declarations.append(
                KeyedContribution(
                    surface,
                    descriptor.id,
                    behavior_metadata=tuple(
                        sorted(
                            (
                                ("contractVersion", 1),
                                ("order", descriptor.order),
                                ("requiresUncond", descriptor.requires_uncond),
                                *descriptor.behavior_metadata,
                            )
                        )
                    ),
                )
            )
    if contribution.strategy is not None:
        descriptor = contribution.strategy
        declarations.append(
            KeyedContribution(
                _GUIDANCE_SURFACES[2],
                descriptor.id,
                behavior_metadata=tuple(
                    sorted(
                        (
                            ("contractVersion", 1),
                            ("participation", descriptor.participation.value),
                            ("requiresUncond", descriptor.requires_uncond),
                            *descriptor.behavior_metadata,
                        )
                    )
                ),
            )
        )
    if contribution.attention is not None:
        descriptor = contribution.attention
        declarations.append(
            KeyedContribution(
                _GUIDANCE_ATTENTION_SURFACE,
                descriptor.id,
                behavior_metadata=tuple(
                    sorted(
                        (
                            ("contractVersion", 1),
                            ("order", attention_order),
                            ("requiresUncond", descriptor.requires_uncond),
                            *descriptor.behavior_metadata,
                        )
                    )
                ),
            )
        )
    surface_order = {surface: index for index, surface in enumerate(_GUIDANCE_SURFACES)}
    return tuple(
        sorted(
            declarations,
            key=lambda item: (
                surface_order[item.surface_id],
                dict(item.behavior_metadata).get("order", 0),
                item.id,
            ),
        )
    )


def graph_compiler_declaration(descriptor: GraphCompilerDescriptor) -> KeyedContribution:
    """Project a worker-local compiler onto its RPC-clean identity facts."""
    return KeyedContribution(
        GRAPH_COMPILERS_SURFACE,
        descriptor.id,
        behavior_metadata=graph_compiler_declaration_metadata(descriptor),
    )


@dataclass(frozen=True)
class SamplerExtensionEntry:
    """Process-local recipe for one importable inference entry point."""

    extension_id: str
    entry_point: str

    def __post_init__(self) -> None:
        if not self.extension_id:
            raise ValueError("extension_id must be non-empty")
        if (
            self.entry_point.count(":") != 1
            or not all(self.entry_point.split(":"))
            or any(char.isspace() for char in self.entry_point)
        ):
            raise ValueError("entry_point must be a 'module:attr' string")


@dataclass(frozen=True)
class MaterializedSamplerRegistry:
    """Callable registry plus its exact RPC-clean declaration evidence."""

    registry: Registry[SamplerDescriptor[Any]]
    snapshot: SamplerRegistrySnapshot
    extensions: tuple[tuple[str, tuple[KeyedContribution, ...]], ...]

    @property
    def extension_ids(self) -> tuple[str, ...]:
        return tuple(extension_id for extension_id, _ in self.extensions)


def _option_wire(option: OptionSpec) -> dict[str, object]:
    def number(value: float | None) -> str | None:
        return None if value is None else format(value, ".17g")

    default = option.default
    if isinstance(default, float):
        default = format(default, ".17g")
    return {
        "name": option.name,
        "kind": option.kind.value,
        "default": default,
        "minimum": number(option.minimum),
        "maximum": number(option.maximum),
        "choices": list(option.choices),
    }


def sampler_declaration(descriptor: SamplerDescriptor[Any]) -> KeyedContribution:
    """Project a callable descriptor onto its behavior-affecting data."""
    option_schema = json.dumps(
        [_option_wire(option) for option in descriptor.options],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return KeyedContribution(
        surface_id=INFERENCE_SAMPLERS_SURFACE,
        id=descriptor.id,
        aliases=descriptor.aliases,
        behavior_metadata=(
            ("contextAware", descriptor.context_aware),
            ("discardPenultimate", descriptor.discard_penultimate),
            ("needsUncond", descriptor.needs_uncond),
            ("noise", descriptor.noise.value),
            ("options", option_schema),
            ("requiresSnrOffset", descriptor.requires_snr_offset),
        ),
    )


def scheduler_declaration(descriptor: SchedulerDescriptor) -> KeyedContribution:
    """Project a callable scheduler onto its pack-owned identity."""
    return KeyedContribution(
        surface_id=INFERENCE_SCHEDULERS_SURFACE,
        id=descriptor.id,
        aliases=descriptor.aliases,
        behavior_metadata=(("displayName", descriptor.display_name),),
    )


def family_declaration(family: ModelFamily) -> KeyedContribution:
    """Project a family registration onto its pack-owned identity facts."""
    return KeyedContribution(
        surface_id=INFERENCE_FAMILIES_SURFACE,
        id=family.id,
        aliases=family.aliases,
        behavior_metadata=(
            ("denoiser", family.denoiser),
            ("displayName", family.display_name),
            ("latentCodec", family.latent_codec),
            ("loader", family.loader),
            ("specificity", family.specificity),
            ("textEncoder", family.text_encoder),
        ),
    )


def component_declaration(descriptor: ComponentDescriptor) -> KeyedContribution:
    """Project a component registration onto its worker execution paths."""
    return KeyedContribution(
        surface_id=INFERENCE_COMPONENTS_SURFACE,
        id=descriptor.id,
        aliases=descriptor.aliases,
        behavior_metadata=(
            ("loader", descriptor.loader),
            ("runtimeClass", descriptor.runtime_class),
        ),
    )


def assembly_declaration(registration: AssemblyRegistration) -> KeyedContribution:
    """Project an assembly registration onto its worker loader path."""
    return KeyedContribution(
        surface_id=INFERENCE_ASSEMBLIES_SURFACE,
        id=registration.id,
        aliases=registration.aliases,
        behavior_metadata=(("loader", registration.load),),
    )


def builtin_sampler_snapshot() -> SamplerRegistrySnapshot:
    """The RPC-clean builtin baseline in registry insertion order."""
    return SamplerRegistrySnapshot(
        tuple(sampler_declaration(descriptor) for descriptor in builtin_registries().samplers)
    )


def _declaration_to_wire(declaration: KeyedContribution) -> dict[str, object]:
    return {
        "surfaceId": declaration.surface_id,
        "id": declaration.id,
        "aliases": list(declaration.aliases),
        "behaviorMetadata": [list(item) for item in declaration.behavior_metadata],
    }


def _declaration_from_wire(raw: object) -> KeyedContribution:
    if not isinstance(raw, Mapping):
        raise RuntimeError("sampler catalog declaration must be an object")
    body = cast("Mapping[str, object]", raw)
    aliases_raw = body.get("aliases", ())
    metadata_raw = body.get("behaviorMetadata", ())
    if not isinstance(aliases_raw, list) or not isinstance(metadata_raw, list):
        raise RuntimeError("sampler catalog declaration has malformed tuples")
    aliases = cast("list[object]", aliases_raw)
    metadata_items = cast("list[object]", metadata_raw)
    metadata: list[tuple[str, str | int | bool | None]] = []
    for raw_item in metadata_items:
        if not isinstance(raw_item, list):
            raise RuntimeError("sampler catalog behavior metadata is malformed")
        item = cast("list[object]", raw_item)
        if len(item) != 2 or not isinstance(item[0], str):
            raise RuntimeError("sampler catalog behavior metadata is malformed")
        value = item[1]
        if value is not None and type(value) not in (str, int, bool):
            raise RuntimeError("sampler catalog behavior metadata is not RPC-clean")
        metadata.append((item[0], cast("str | int | bool | None", value)))
    return KeyedContribution(
        surface_id=str(body.get("surfaceId", "")),
        id=str(body.get("id", "")),
        aliases=tuple(str(alias) for alias in aliases),
        behavior_metadata=tuple(metadata),
    )


def write_sampler_catalog(
    path: Path,
    key: str,
    entries: Sequence[SamplerExtensionEntry],
    expected: SamplerRegistrySnapshot | None = None,
    *,
    expected_extensions: Sequence[tuple[str, tuple[KeyedContribution, ...]]] | None = None,
) -> None:
    """Atomically append one immutable generation to the local catalog."""
    if not key:
        raise ValueError("sampler catalog key must be non-empty")
    normalized = tuple(sorted(entries, key=lambda entry: entry.extension_id))
    if len({entry.extension_id for entry in normalized}) != len(normalized):
        raise ValueError("sampler extension ids must be unique")
    document: dict[str, object] = {"format": _CATALOG_FORMAT, "records": {}}
    if path.exists():
        loaded_raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(loaded_raw, dict):
            raise RuntimeError(f"invalid sampler catalog at {path}")
        loaded = cast("dict[str, object]", loaded_raw)
        if loaded.get("format") != _CATALOG_FORMAT:
            raise RuntimeError(f"invalid sampler catalog at {path}")
        document = loaded
    records_raw = document.get("records")
    if not isinstance(records_raw, dict):
        raise RuntimeError(f"invalid sampler catalog records at {path}")
    records = cast("dict[str, object]", records_raw)
    record: dict[str, object] = {
        "extensions": [
            {"id": entry.extension_id, "entryPoint": entry.entry_point} for entry in normalized
        ]
    }
    if expected is not None:
        record["expected"] = [
            _declaration_to_wire(declaration) for declaration in expected.samplers
        ]
    if expected_extensions is not None:
        record["expectedExtensions"] = [
            {
                "id": extension_id,
                "contributions": [_declaration_to_wire(item) for item in declarations],
            }
            for extension_id, declarations in expected_extensions
        ]
    existing = records.get(key)
    if existing is not None and existing != record:
        raise RuntimeError(f"sampler catalog key {key!r} already names different data")
    records[key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def remove_sampler_catalog_record(path: Path, key: str) -> None:
    """Atomically remove one transient staging record if it still exists."""
    if not path.exists():
        return
    loaded_raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded_raw, dict):
        raise RuntimeError(f"invalid sampler catalog at {path}")
    loaded = cast("dict[str, object]", loaded_raw)
    if loaded.get("format") != _CATALOG_FORMAT:
        raise RuntimeError(f"invalid sampler catalog at {path}")
    records_raw = loaded.get("records")
    if not isinstance(records_raw, dict):
        raise RuntimeError(f"invalid sampler catalog records at {path}")
    records = cast("dict[str, object]", records_raw)
    if records.pop(key, None) is None:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(loaded, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_record(key: str, path: Path) -> tuple[tuple[SamplerExtensionEntry, ...], object, object]:
    loaded_raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded_raw, dict):
        raise RuntimeError(f"invalid sampler catalog at {path}")
    loaded = cast("dict[str, object]", loaded_raw)
    if loaded.get("format") != _CATALOG_FORMAT:
        raise RuntimeError(f"invalid sampler catalog at {path}")
    records_raw = loaded.get("records")
    if not isinstance(records_raw, dict):
        raise RuntimeError(f"sampler catalog has no generation {key!r}")
    records = cast("dict[str, object]", records_raw)
    record_raw = records.get(key)
    if not isinstance(record_raw, dict):
        raise RuntimeError(f"sampler catalog has no generation {key!r}")
    record = cast("dict[str, object]", record_raw)
    entries_raw = record.get("extensions")
    if not isinstance(entries_raw, list):
        raise RuntimeError(f"sampler catalog generation {key!r} has no extensions")
    entries: list[SamplerExtensionEntry] = []
    for raw in cast("list[object]", entries_raw):
        if not isinstance(raw, dict):
            raise RuntimeError("sampler catalog extension entry must be an object")
        body = cast("dict[str, object]", raw)
        entries.append(
            SamplerExtensionEntry(
                extension_id=str(body.get("id", "")),
                entry_point=str(body.get("entryPoint", "")),
            )
        )
    return tuple(entries), record.get("expected"), record.get("expectedExtensions")


def _resolve_contribution(entry: SamplerExtensionEntry) -> InferenceContribution:
    module_name, attr_name = entry.entry_point.split(":", 1)
    for loaded_name in tuple(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(module_name + "."):
            del sys.modules[loaded_name]
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    register = getattr(module, attr_name, None)
    if not callable(register):
        raise RuntimeError(
            f"inference entry {entry.entry_point!r} for {entry.extension_id!r} is not callable"
        )
    contribution = register()
    if isinstance(contribution, SamplerContribution):
        return InferenceContribution(samplers=contribution.samplers)
    if not isinstance(contribution, InferenceContribution):
        raise RuntimeError(
            f"inference entry {entry.entry_point!r} for {entry.extension_id!r} "
            "must return SamplerContribution or InferenceContribution"
        )
    return contribution


_inference_cache: OrderedDict[str, MaterializedInferenceGeneration] = OrderedDict()
_candidate_cache: dict[str, MaterializedInferenceGeneration] = {}
_cache_lock = threading.RLock()


def materialize_sampler_registry(
    key: str, *, catalog_path: Path | None = None
) -> MaterializedSamplerRegistry:
    """Build and bidirectionally validate one worker-local callable registry."""
    generation = materialize_inference_generation(key, catalog_path=catalog_path)
    return MaterializedSamplerRegistry(
        registry=generation.registries.samplers,
        snapshot=generation.sampler_snapshot,
        extensions=tuple(
            (
                extension_id,
                tuple(
                    item for item in declarations if item.surface_id == INFERENCE_SAMPLERS_SURFACE
                ),
            )
            for extension_id, declarations in generation.extensions
        ),
    )


def _materialize_inference_generation(
    key: str, *, catalog_path: Path | None = None, check_pins: bool = True
) -> MaterializedInferenceGeneration:
    """Import each inference entry once and project every inference surface."""
    with _cache_lock:
        cacheable = not key.startswith("candidate:")
        if cacheable:
            cached = _inference_cache.get(key)
            if cached is not None:
                if check_pins:
                    # The cached generation may have been materialized with
                    # check_pins disabled, so enforce pins before handing the
                    # callbacks out under the default checked contract.
                    for extension_id, contribution in cached.attention_contributions:
                        check_attention_pins(extension_id, contribution)
                return cached
        if catalog_path is None:
            raw_path = os.environ.get(SAMPLER_CATALOG_ENV)
            if not raw_path:
                raise RuntimeError(f"${SAMPLER_CATALOG_ENV} is not configured")
            catalog_path = Path(raw_path)
        entries, expected_raw, expected_extensions_raw = _read_record(key, catalog_path)
        contributions: list[InferenceContribution] = []
        extension_declarations: list[tuple[str, tuple[KeyedContribution, ...]]] = []
        guidance: list[KeyedContribution] = []
        prefixes: list[str] = []
        materialized_guidance: list[tuple[str, GuidanceContribution[Any]]] = []
        graph_compilers: list[GraphCompilerDescriptor] = []
        materialized_attention: list[tuple[str, AttentionContribution[Any]]] = []
        attention_declared: dict[str, tuple[str, str]] = {}
        backend_families: dict[str, tuple[str, str]] = {}
        attention_guidance_count = 0
        for entry in entries:
            contribution = _resolve_contribution(entry)
            contributions.append(contribution)
            sampler_declarations = tuple(
                sampler_declaration(descriptor) for descriptor in contribution.samplers
            )
            scheduler_declarations = tuple(
                scheduler_declaration(descriptor) for descriptor in contribution.schedulers
            )
            attention_order = attention_guidance_count
            if contribution.guidance is not None and contribution.guidance.attention is not None:
                attention_guidance_count += 1
            produced_guidance = (
                ()
                if contribution.guidance is None
                else guidance_declarations(
                    contribution.guidance,
                    attention_order=attention_order,
                )
            )
            produced_attention: tuple[KeyedContribution, ...] = ()
            if contribution.attention is not None:
                if check_pins:
                    check_attention_pins(entry.extension_id, contribution.attention)
                produced_attention = attention_declarations(contribution.attention)
                for declaration in produced_attention:
                    owner = attention_declared.get(declaration.id)
                    if owner is not None and owner[0] != entry.extension_id:
                        raise RuntimeError(
                            f"attention descriptor {declaration.id!r} is declared by "
                            f"extensions {owner[0]!r} (surface {owner[1]!r}) and "
                            f"{entry.extension_id!r} (surface "
                            f"{declaration.surface_id!r}): attention descriptor ids "
                            "must be globally unique"
                        )
                    attention_declared[declaration.id] = (
                        entry.extension_id,
                        declaration.surface_id,
                    )
                for backend in contribution.attention.backends:
                    conflict = backend_families.get(backend.family)
                    if conflict is not None:
                        raise RuntimeError(
                            f"attention backend family {backend.family!r} is exclusively "
                            f"claimed by extension {conflict[0]!r} descriptor {conflict[1]!r}; "
                            f"extension {entry.extension_id!r} descriptor {backend.id!r} "
                            "conflicts: each model family admits exactly one attention "
                            "backend"
                        )
                    backend_families[backend.family] = (entry.extension_id, backend.id)
                materialized_attention.append((entry.extension_id, contribution.attention))
            produced_compilers = tuple(
                graph_compiler_declaration(descriptor)
                for descriptor in sorted(
                    contribution.graph_compilers, key=lambda item: (item.order, item.id)
                )
            )
            produced_families = tuple(
                family_declaration(family) for family in contribution.families
            )
            produced_components = tuple(
                component_declaration(descriptor) for descriptor in contribution.components
            )
            produced_assemblies = tuple(
                assembly_declaration(registration) for registration in contribution.assemblies
            )
            declarations = (
                sampler_declarations
                + scheduler_declarations
                + produced_guidance
                + produced_compilers
                + produced_families
                + produced_components
                + produced_assemblies
                + produced_attention
            )
            guidance.extend(produced_guidance)
            graph_compilers.extend(contribution.graph_compilers)
            if contribution.guidance is not None:
                materialized_guidance.append((entry.extension_id, contribution.guidance))
            prefixes.append(entry.entry_point.split(":", 1)[0])
            extension_declarations.append((entry.extension_id, declarations))
        registries = merge_registries(builtin_registries(), contributions)
        strategies = tuple(
            (extension, contribution.strategy.id)
            for extension, contribution in materialized_guidance
            if contribution.strategy is not None
        )
        if len(strategies) > 1:
            owners = "; ".join(
                f"extension={extension} contribution={strategy_id}"
                for extension, strategy_id in strategies
            )
            raise RuntimeError(f"multiple guidance strategies were materialized: {owners}")
        snapshot = SamplerRegistrySnapshot(
            tuple(sampler_declaration(descriptor) for descriptor in registries.samplers)
        )
        if expected_raw is not None:
            if not isinstance(expected_raw, list):
                raise RuntimeError("sampler catalog expected declarations are malformed")
            expected = SamplerRegistrySnapshot(
                tuple(_declaration_from_wire(item) for item in cast("list[object]", expected_raw))
            )
            if snapshot != expected:
                produced = tuple(declaration.id for declaration in snapshot.samplers)
                declared = tuple(declaration.id for declaration in expected.samplers)
                raise RuntimeError(
                    "sampler registry declaration mismatch: "
                    f"worker produced {produced}, snapshot declared {declared}"
                )
        if expected_extensions_raw is not None:
            if not isinstance(expected_extensions_raw, list):
                raise RuntimeError("inference catalog expected extensions are malformed")
            expected_extensions: list[tuple[str, tuple[KeyedContribution, ...]]] = []
            for raw_extension in cast("list[object]", expected_extensions_raw):
                if not isinstance(raw_extension, Mapping):
                    raise RuntimeError("inference catalog expected extension is malformed")
                body = cast("Mapping[str, object]", raw_extension)
                raw_items = body.get("contributions")
                if not isinstance(body.get("id"), str) or not isinstance(raw_items, list):
                    raise RuntimeError("inference catalog expected extension fields are malformed")
                expected_extensions.append(
                    (
                        cast("str", body["id"]),
                        tuple(
                            _declaration_from_wire(item) for item in cast("list[object]", raw_items)
                        ),
                    )
                )
            if tuple(extension_declarations) != tuple(expected_extensions):
                raise RuntimeError(
                    "inference extension declarations changed during materialization"
                )
        materialized = MaterializedInferenceGeneration(
            registries=registries,
            sampler_snapshot=snapshot,
            guidance_snapshot=GuidanceRegistrySnapshot(
                tuple(
                    sorted(
                        guidance,
                        key=lambda item: (
                            _GUIDANCE_SURFACES.index(item.surface_id),
                            dict(item.behavior_metadata).get("order", 0),
                            item.id,
                        ),
                    )
                )
            ),
            extensions=tuple(extension_declarations),
            module_prefixes=tuple(prefixes),
            guidance_contributions=tuple(materialized_guidance),
            graph_compiler_snapshot=GraphCompilerRegistrySnapshot(
                tuple(
                    graph_compiler_declaration(descriptor)
                    for descriptor in sorted(
                        graph_compilers, key=lambda item: (item.order, item.id)
                    )
                )
            ),
            graph_compilers=tuple(sorted(graph_compilers, key=lambda item: (item.order, item.id))),
            attention_contributions=tuple(materialized_attention),
        )
        if cacheable:
            _inference_cache[key] = materialized
        else:
            _candidate_cache[key] = materialized
        return materialized


def materialize_inference_generation(
    key: str, *, catalog_path: Path | None = None, check_pins: bool = True
) -> MaterializedInferenceGeneration:
    """Materialize atomically, removing every attempted import on failure.

    ``check_pins`` defaults to the runtime rule: attention pins are checked
    against this interpreter's installed distributions before execution.
    The doctor probe passes ``check_pins=False`` because it only lists
    declared attention points from the serve-side process and never
    executes them - pin enforcement belongs to the inference worker that
    materializes for execution.
    """
    resolved_path = catalog_path
    if resolved_path is None:
        raw_path = os.environ.get(SAMPLER_CATALOG_ENV)
        if not raw_path:
            raise RuntimeError(f"${SAMPLER_CATALOG_ENV} is not configured")
        resolved_path = Path(raw_path)
    entries, _, _ = _read_record(key, resolved_path)
    prefixes = tuple(entry.entry_point.split(":", 1)[0] for entry in entries)
    try:
        return _materialize_inference_generation(
            key, catalog_path=resolved_path, check_pins=check_pins
        )
    except BaseException:
        for prefix in prefixes:
            for name in tuple(sys.modules):
                if name == prefix or name.startswith(prefix + "."):
                    del sys.modules[name]
        importlib.invalidate_caches()
        raise


def _release_modules(generation: MaterializedInferenceGeneration) -> None:
    for prefix in generation.module_prefixes:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
    importlib.invalidate_caches()


def release_inference_generation(key: str) -> None:
    """Drop one unpinned materialized generation and its imported modules."""
    with _cache_lock:
        generation = _inference_cache.pop(key, None) or _candidate_cache.pop(key, None)
        if generation is not None:
            _release_modules(generation)


def compile_inference_graph(
    generation_key: str,
    graph_wire: Mapping[str, object],
    targets: Sequence[str],
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> Mapping[str, object]:
    """Compile against an already-materialized exact generation."""
    with _cache_lock:
        generation = _inference_cache.get(generation_key) or _candidate_cache.get(generation_key)
        if generation is None:
            raise InferenceGraphCompileError(
                GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
                f"inference generation {generation_key!r} is not materialized",
            )
        compilers = generation.graph_compilers
    return execute_graph_compilers(
        generation_key,
        graph_wire,
        targets,
        compilers,
        cancelled=cancelled,
    )


def registry_choice_values(declarations: Sequence[KeyedContribution]) -> tuple[str, ...]:
    """Compatibility dropdown values for keyed registry declarations."""
    return tuple(
        declaration.aliases[0] if declaration.aliases else declaration.id
        for declaration in declarations
    )


def sampler_choice_values(snapshot: SamplerRegistrySnapshot) -> tuple[str, ...]:
    """Native dropdown values, derived solely from the effective registry."""
    return registry_choice_values(snapshot.samplers)


__all__ = [
    "INFERENCE_ASSEMBLIES_SURFACE",
    "INFERENCE_COMPONENTS_SURFACE",
    "INFERENCE_FAMILIES_SURFACE",
    "INFERENCE_SAMPLERS_SURFACE",
    "INFERENCE_SCHEDULERS_SURFACE",
    "KeyedContribution",
    "SAMPLER_CATALOG_ENV",
    "MaterializedSamplerRegistry",
    "MaterializedInferenceGeneration",
    "InferenceContribution",
    "SamplerContribution",
    "SamplerExtensionEntry",
    "builtin_sampler_snapshot",
    "materialize_sampler_registry",
    "materialize_inference_generation",
    "release_inference_generation",
    "guidance_declarations",
    "attention_declarations",
    "graph_compiler_declaration",
    "compile_inference_graph",
    "registry_choice_values",
    "remove_sampler_catalog_record",
    "sampler_choice_values",
    "sampler_declaration",
    "scheduler_declaration",
    "family_declaration",
    "component_declaration",
    "assembly_declaration",
    "write_sampler_catalog",
]
