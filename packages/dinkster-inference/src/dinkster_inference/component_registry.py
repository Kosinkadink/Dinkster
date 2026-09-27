"""Component architecture registrations shared by planning and execution.

Execution symbols are resolved only by workers. Inspecting an artifact or
selecting its execution provider never imports torch.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dinkster_protocol import AttentionPolicy, AttentionRouteToken

from .assembly import AssemblyError, ComponentPlan
from .devices import BFLOAT16, FLOAT32, DType
from .families import ModelFamily
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .quantization import QuantizationError, quantization_error_cause
from .recipe import ReconstructionRecipe, RuntimeKnobs, WeightSourceBinding, WeightSourceRef
from .registry import Registry
from .sampling import SamplingDescriptor
from .weights import WeightSource

if TYPE_CHECKING:
    from .component_checkpoint import ComponentCheckpointPlan


@dataclass(frozen=True)
class ComponentDescriptor:
    """One architecture's component contracts, not an admission allowlist.

    The detector returns all components present, including components of a
    combined checkpoint. Loaders consume those same plans and key maps.
    Runtime classes live in the execution package and are imported lazily.
    Combined assembly calls detectors with ``bind_asset_identity=False``;
    geometry plans must not depend on whether the header carries asset identity.
    """

    family: ModelFamily
    detector: Callable[..., tuple[tuple[str, Any], ...]]
    roles: tuple[str, ...]
    text_encoder_roles: tuple[str, ...]
    codec_roles: tuple[str, ...]
    loader: str
    runtime_class: str
    default_diffusion_dtype: DType = BFLOAT16
    default_text_dtype: DType = BFLOAT16
    vae_dtypes: tuple[DType, ...] = (BFLOAT16, FLOAT32)
    role_default_dtypes: tuple[tuple[str, DType], ...] = ()
    text_loader_hints: tuple[str, ...] = ()
    requires_text_recipe: bool = False
    runtime_variants: tuple[str, ...] = ()
    model_role: str = "diffusion"
    aliases: tuple[str, ...] = ()
    requires_runtime_versions: bool = False
    pool_model: bool = False
    runtime_with_family: bool = False
    tokenizer_attribute: str | None = None
    aimdo_roles: tuple[str, ...] = ()
    fixed_promotion_roles: tuple[str, ...] = ()
    shared_conditioning_families: tuple[str, ...] = ()
    canonicalize_runtime_facts: bool = False
    prepare_conditioning: str = "materialize_basic_conditioning"
    conditioning_roles: tuple[str, ...] | None = None
    conditioning_format: str = "rows"
    frame_rate_conditioning: bool = False
    allow_unbound_conditioning: bool = False
    release_conditioning: bool = False
    execution_options: Callable[[Any, float | None, tuple[Any, ...]], dict[str, Any]] | None = None
    execution_resolver: str | None = None
    runtime_factory: str | None = None
    fp8_matmul: Callable[[str, Any], bool] | None = None
    plan_family: Callable[[Any], ModelFamily] | None = None
    checkpoint_loader: str | None = None
    checkpoint_validator: Callable[[ComponentCheckpointPlan], object] | None = None
    checkpoint_source_aliases: tuple[tuple[str, str], ...] = ()
    component_realizer: str | None = None
    checkpoint_text_factory: str | None = None
    checkpoint_codec_factory: str | None = None
    codec_adapter: str | None = None
    native_encode_text: str | None = None
    native_decode: str | None = None
    native_encode: str | None = None
    native_load: str | None = None

    @property
    def id(self) -> str:
        return self.family.id

    @property
    def sigma_space(self) -> SamplingDescriptor:
        return self.family.sampling

    def family_for(self, planned: Any) -> str:
        if self.plan_family is not None:
            return self.plan_family(component_plans(planned)[0]).id
        return getattr(planned, "family_id", self.id)

    def rebind_attention_recipe(
        self,
        recipe: ReconstructionRecipe,
        runtime_versions: Mapping[str, str] | None = None,
    ) -> ReconstructionRecipe:
        """Refresh family-owned identity facts after an attention route change."""
        del runtime_versions
        return recipe

    def knobs(
        self,
        role: str,
        plans: tuple[ComponentPlan[Any], ...],
        compute_dtype: str,
        *,
        attention_policy: AttentionPolicy = "auto",
        attention_route_token: AttentionRouteToken | None = None,
    ) -> RuntimeKnobs:
        engine = self.family.engine
        has_attention = engine.attention_backend(role) is not None and (
            attention_route_token is not None or not engine.attention_requires_route
        )
        return RuntimeKnobs(
            diffusion_dtype=compute_dtype if role == self.model_role else "unloaded",
            text_dtype=compute_dtype if role in self.text_encoder_roles else "unloaded",
            vae_dtype=compute_dtype if role in self.codec_roles else "unloaded",
            fp8_matmul=(False if self.fp8_matmul is None else self.fp8_matmul(role, plans[0])),
            runtime_facts=(
                plans[0].runtime_facts
                if len(plans) == 1 and not self.canonicalize_runtime_facts
                else tuple(sorted({fact for plan in plans for fact in plan.runtime_facts}))
            ),
            attention_policy=attention_policy if has_attention else "auto",
            attention_route_token=attention_route_token if has_attention else None,
        )

    def recipe(
        self,
        source: WeightSourceRef,
        loaded: Any,
        compute_dtype: str,
        *,
        attention_policy: AttentionPolicy = "auto",
        attention_route_token: AttentionRouteToken | None = None,
    ) -> ReconstructionRecipe:
        plans = (
            loaded.identity_components if hasattr(loaded, "identity_components") else (loaded.plan,)
        )
        family_id = self.family_for(loaded)
        return ReconstructionRecipe(
            sources=(WeightSourceBinding(loaded.role, source),),
            family_id=family_id,
            component_identity=runtime_component_identity(family_id, plans),
            knobs=self.knobs(
                loaded.role,
                plans,
                compute_dtype,
                attention_policy=attention_policy,
                attention_route_token=attention_route_token,
            ),
        )

    def component_identity(
        self,
        role: str,
        planned: Any,
        compute_dtype: str,
        *,
        attention_policy: AttentionPolicy = "auto",
        attention_route_token: AttentionRouteToken | None = None,
        runtime_versions: Mapping[str, str] | None = None,
    ) -> str:
        del runtime_versions
        plans = component_plans(planned)
        family_id = self.family_for(planned)
        knobs = self.knobs(
            role,
            plans,
            compute_dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        )
        return build_runtime_identity_from_facts(
            family_id,
            runtime_component_identity(family_id, plans),
            diffusion_dtype=knobs.diffusion_dtype,
            text_dtype=knobs.text_dtype,
            vae_dtype=knobs.vae_dtype,
            fp8_matmul=knobs.fp8_matmul,
            runtime_facts=knobs.runtime_facts,
            attention_policy=knobs.attention_policy,
            attention_route_token=knobs.attention_route_token,
        )

    def __post_init__(self) -> None:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("component roles must be unique")
        if not set((*self.text_encoder_roles, *self.codec_roles)) <= set(self.roles):
            raise ValueError("text and codec roles must be component roles")


@dataclass(frozen=True)
class DetectedComponents:
    descriptor: ComponentDescriptor
    components: tuple[tuple[str, Any], ...]

    def plan_for(self, role: str) -> Any | None:
        return dict(self.components).get(role)


class AmbiguousComponentError(ValueError):
    """Detected components need a more specific architecture or text recipe."""


class ComponentRegistry(Registry[ComponentDescriptor]):
    """Detect without assigning an unknown asset to an unrelated family."""

    def detect(
        self,
        source: WeightSource,
        path: Path,
        *,
        bind_asset_identity: bool = True,
        diagnostics: list[tuple[str, AssemblyError | QuantizationError]] | None = None,
    ) -> tuple[DetectedComponents, ...]:
        matches: list[DetectedComponents] = []
        quantization_error: QuantizationError | None = None
        for descriptor in self:
            try:
                components = (
                    descriptor.detector(source, path)
                    if bind_asset_identity
                    else descriptor.detector(source, path, bind_asset_identity=False)
                )
            except QuantizationError as error:
                # A nonmatching probe can scope a valid quantized component incorrectly.
                quantization_error = quantization_error or error
                if diagnostics is not None:
                    diagnostics.append((descriptor.id, error))
                continue
            except AssemblyError as error:
                quantization_error = quantization_error or quantization_error_cause(error)
                if diagnostics is not None:
                    diagnostics.append((descriptor.id, error))
                continue
            if components:
                matches.append(DetectedComponents(descriptor, components))
        if not matches:
            if quantization_error is not None:
                raise quantization_error
            keys = source.keys()
            logging.getLogger(__name__).warning(
                "No matching component architecture; detected %d tensor keys (%s)",
                len(keys),
                ", ".join(keys[:3]),
            )
        return tuple(matches)

    def select(
        self,
        source: WeightSource,
        path: Path,
        kind: str,
        *,
        family_id: str | None = None,
    ) -> tuple[ComponentDescriptor, str, Any]:
        matches = self.detect(source, path)
        if not matches:
            keys = source.keys()
            raise ValueError(
                f"{kind} loading: no matching component architecture; "
                f"detected {len(keys)} tensor keys ({', '.join(keys[:3])})"
            )
        return self.select_detected(matches, kind, family_id=family_id)

    def select_detected(
        self,
        matches: tuple[DetectedComponents, ...],
        kind: str,
        *,
        family_id: str | None = None,
    ) -> tuple[ComponentDescriptor, str, Any]:
        """Select a fixed-profile component, not an unresolved text architecture."""
        candidates: list[tuple[ComponentDescriptor, str, Any]] = []
        for match in matches:
            descriptor = match.descriptor
            roles = (
                (descriptor.model_role,)
                if kind == "model"
                else descriptor.text_encoder_roles
                if kind == "text"
                else descriptor.codec_roles
                if kind == "codec"
                else (kind,)
                if kind in descriptor.roles
                else ()
            )
            candidates.extend(
                (descriptor, role, plan) for role, plan in match.components if role in roles
            )
        recipe_required = kind == "text" and any(
            descriptor.requires_text_recipe for descriptor, _role, _plan in candidates
        )
        if recipe_required:
            candidates = [
                candidate for candidate in candidates if not candidate[0].requires_text_recipe
            ]
        preferred = [
            candidate
            for candidate in candidates
            if family_id is not None
            and (
                self.get(family_id) is candidate[0]
                or (
                    kind == "text"
                    and family_id.removeprefix("dinkster.") in candidate[0].text_loader_hints
                )
            )
        ]
        ranked = sorted(
            preferred or candidates,
            key=lambda candidate: candidate[0].family.specificity,
            reverse=True,
        )
        if ranked and (
            len(ranked) == 1 or ranked[0][0].family.specificity > ranked[1][0].family.specificity
        ):
            return ranked[0]
        detected = (
            ", ".join(
                f"{match.descriptor.family.display_name} {role}"
                for match in matches
                for role, _plan in match.components
            )
            or "no recognized components"
        )
        if ranked:
            raise AmbiguousComponentError(
                f"{kind} loading: ambiguous components; detected {detected}"
            )
        if recipe_required:
            raise AmbiguousComponentError(
                f"{kind} loading: explicit text recipe required; detected {detected}"
            )
        raise ValueError(f"{kind} loading: no matching component architecture; detected {detected}")


def execution_symbol(reference: str) -> Any:
    """Resolve a registered execution symbol only when the worker needs it."""
    module, separator, name = reference.partition(":")
    if not separator:
        module, name = "dinkster_inference_torch", reference
    return getattr(importlib.import_module(module), name)


def build_component_runtime(
    descriptor: ComponentDescriptor,
    loaded: Any,
    identity: str,
    dtype: Any,
    **options: Any,
) -> Any:
    """Bind the descriptor's sampling behavior to its already-realized model."""
    if descriptor.runtime_factory is not None:
        return execution_symbol(descriptor.runtime_factory)(loaded, identity, dtype, **options)
    runtime_type = execution_symbol(descriptor.runtime_class)
    if descriptor.runtime_with_family:
        return runtime_type(loaded.module, descriptor.family, runtime_identity=identity, **options)
    return runtime_type(loaded.module, runtime_identity=identity, compute_dtype=dtype, **options)


def component_plans(planned: Any) -> tuple[ComponentPlan[Any], ...]:
    """Normalize standalone and composite architecture plans without changing their maps."""
    if isinstance(planned, ComponentPlan):
        return (cast("ComponentPlan[Any]", planned),)
    if hasattr(planned, "identity_components"):
        return planned.identity_components
    if hasattr(planned, "plan"):
        return component_plans(planned.plan)
    return (planned.component,)
