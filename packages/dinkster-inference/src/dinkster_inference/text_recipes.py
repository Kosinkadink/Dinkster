"""Text encoding semantics over already detected component plans."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from dinkster_protocol import AttentionPolicy, AttentionRouteToken

from .assembly import ComponentPlan
from .component_registry import DetectedComponents, component_plans
from .identity import runtime_component_identity
from .prompt_tokens import CLIP_G_PROFILE, CLIP_L_PROFILE, TokenizerProfile
from .recipe import ReconstructionRecipe, RuntimeKnobs, WeightSourceBinding, WeightSourceRef
from .registry import Registry
from .t5_spm import T5_XXL_FLUX_PROFILE

DetectedTextSources = tuple[tuple[DetectedComponents, ...], ...]


class UnresolvedTextRecipe(ValueError):
    """No text recipe matches; existing provider resolution may continue."""


@dataclass(frozen=True)
class TextEncodingProfile:
    """Tokenizer packing and representations consumed by a text recipe."""

    tokenizer: TokenizerProfile
    hidden_layer: int | None = None
    layer_norm_hidden_state: bool = True
    projected_pooled: bool = True
    attention_masked: bool = False
    zero_out_masked: bool = False


@dataclass(frozen=True)
class TextRecipeComponent:
    """None leaves tokenizer packing policy to the recipe-specific runtime/config."""

    source_index: int
    role: str
    plan: ComponentPlan[Any]
    profile: TextEncodingProfile | None


@dataclass(frozen=True)
class TextRecipeBinding:
    id: str
    family_id: str
    components: tuple[TextRecipeComponent, ...]
    loader: str
    runtime_class: str
    composer: str | None
    composition_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        roles = tuple(component.role for component in self.components)
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("text recipe component roles must be nonempty and unique")
        if sorted(self.composition_roles) != sorted(roles):
            raise ValueError("text recipe composition must consume every component once")
        if self.composer is None and len(roles) != 1:
            raise ValueError("multi-component text recipes require a composer")

    @property
    def component_identity(self) -> tuple[str, ...]:
        return runtime_component_identity(
            self.family_id, tuple(component.plan for component in self.components)
        )

    def knobs(
        self,
        compute_dtype: str,
        *,
        attention_policy: AttentionPolicy = "auto",
        attention_route_token: AttentionRouteToken | None = None,
        embedding_binding_digest: str | None = None,
    ) -> RuntimeKnobs:
        facts = [
            f"text_recipe={self.id}",
            f"text_composer={self.composer}",
            "text_composition_roles=" + json.dumps(self.composition_roles),
        ]
        for component in self.components:
            facts.append(
                "text_component="
                + json.dumps(
                    (
                        component.source_index,
                        component.role,
                        None if component.profile is None else asdict(component.profile),
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            facts.extend(component.plan.runtime_facts)
        return RuntimeKnobs(
            diffusion_dtype="unloaded",
            text_dtype=compute_dtype,
            vae_dtype="unloaded",
            fp8_matmul=False,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            embedding_binding_digest=embedding_binding_digest,
            runtime_facts=tuple(facts),
        )

    def recipe(
        self,
        source_refs: tuple[WeightSourceRef, ...],
        compute_dtype: str,
        *,
        attention_policy: AttentionPolicy = "auto",
        attention_route_token: AttentionRouteToken | None = None,
        embedding_binding_digest: str | None = None,
    ) -> ReconstructionRecipe:
        indices = {component.source_index for component in self.components}
        if indices != set(range(len(source_refs))):
            raise ValueError("text recipe must retain every source in submitted order")
        return ReconstructionRecipe(
            sources=tuple(
                WeightSourceBinding(f"text_source_{index:04d}", source)
                for index, source in enumerate(source_refs)
            ),
            family_id=self.family_id,
            component_identity=self.component_identity,
            knobs=self.knobs(
                compute_dtype,
                attention_policy=attention_policy,
                attention_route_token=attention_route_token,
                embedding_binding_digest=embedding_binding_digest,
            ),
        )


@dataclass(frozen=True)
class TextRecipeDescriptor:
    id: str
    bind: Callable[[DetectedTextSources], TextRecipeBinding]
    aliases: tuple[str, ...] = ()


def _component(
    sources: DetectedTextSources, role: str, profile: TextEncodingProfile
) -> TextRecipeComponent:
    candidates: list[TextRecipeComponent] = []
    for source_index, matches in enumerate(sources):
        for match in matches:
            if role not in match.descriptor.text_encoder_roles:
                continue
            for detected_role, planned in match.components:
                if detected_role != role:
                    continue
                for plan in component_plans(planned):
                    candidate = TextRecipeComponent(source_index, role, plan, profile)
                    if candidate not in candidates:
                        candidates.append(candidate)
    if len(candidates) != 1:
        raise UnresolvedTextRecipe(f"text recipe needs one unambiguous {role} component")
    return candidates[0]


def _classic_binding(
    id: str,
    family_id: str,
    sources: DetectedTextSources,
    profiles: tuple[tuple[str, TextEncodingProfile], ...],
    composer: str | None,
    composition_roles: tuple[str, ...],
) -> TextRecipeBinding:
    components = tuple(_component(sources, role, profile) for role, profile in profiles)
    if {component.source_index for component in components} != set(range(len(sources))):
        raise UnresolvedTextRecipe("text recipe would discard an input source")
    return TextRecipeBinding(
        id,
        family_id,
        components,
        "dinkster_inference_torch.text_recipes:assemble_text_recipe",
        "dinkster_inference_torch.text_recipes:TextRecipeRuntime",
        composer,
        composition_roles,
    )


def _sdxl(sources: DetectedTextSources) -> TextRecipeBinding:
    return _classic_binding(
        "dinkster.text_sdxl",
        "dinkster.sdxl",
        sources,
        (
            ("clip_l", TextEncodingProfile(CLIP_L_PROFILE, -2, False)),
            ("clip_g", TextEncodingProfile(CLIP_G_PROFILE, -2, False)),
        ),
        "dinkster_inference_torch.clip_text:compose_sdxl_conditioning",
        ("clip_l", "clip_g"),
    )


def _flux(sources: DetectedTextSources) -> TextRecipeBinding:
    roles = {role for matches in sources for match in matches for role, _ in match.components}
    profiles = [("t5xxl", TextEncodingProfile(T5_XXL_FLUX_PROFILE))]
    composer = "dinkster_inference_torch.t5_text:compose_flux_t5_conditioning"
    composition_roles: tuple[str, ...] = ("t5xxl",)
    if "clip_l" in roles:
        profiles.insert(
            0,
            ("clip_l", TextEncodingProfile(CLIP_L_PROFILE, projected_pooled=False)),
        )
        composer = "dinkster_inference_torch.t5_text:compose_flux_conditioning"
        composition_roles = ("t5xxl", "clip_l")
    return _classic_binding(
        "dinkster.text_flux",
        "dinkster.flux_dev",
        sources,
        tuple(profiles),
        composer,
        composition_roles,
    )


def _stable_diffusion(sources: DetectedTextSources) -> TextRecipeBinding:
    roles = {role for matches in sources for match in matches for role, _ in match.components}
    if "clip_l" in roles and "clip_g" in roles:
        return _sdxl(sources)
    if "clip_g" in roles:
        return _classic_binding(
            "dinkster.text_sdxl_refiner",
            "dinkster.sdxl_refiner",
            sources,
            (("clip_g", TextEncodingProfile(CLIP_G_PROFILE, -2, False)),),
            None,
            ("clip_g",),
        )
    return _classic_binding(
        "dinkster.text_sd15",
        "dinkster.sd15",
        sources,
        (("clip_l", TextEncodingProfile(CLIP_L_PROFILE, projected_pooled=False)),),
        None,
        ("clip_l",),
    )


def default_text_recipe_registry() -> Registry[TextRecipeDescriptor]:
    from .ace15_text import ACE15_TEXT_RECIPE
    from .hunyuan_image_text import HUNYUAN_IMAGE_TEXT_RECIPE
    from .hunyuan_video_text import HUNYUAN_VIDEO_TEXT_RECIPE
    from .ltx_text_recipe import LTX_TEXT_RECIPE
    from .newbie_text import NEWBIE_TEXT_RECIPE

    registry: Registry[TextRecipeDescriptor] = Registry()
    registry.register(TextRecipeDescriptor("dinkster.text_sdxl", _sdxl, ("sdxl",)))
    registry.register(TextRecipeDescriptor("dinkster.text_flux", _flux, ("flux",)))
    registry.register(
        TextRecipeDescriptor("dinkster.text_sd15", _stable_diffusion, ("stable_diffusion",))
    )
    registry.register(TextRecipeDescriptor("dinkster.text_sdxl_refiner", _stable_diffusion))
    registry.register(LTX_TEXT_RECIPE)
    registry.register(HUNYUAN_VIDEO_TEXT_RECIPE)
    registry.register(HUNYUAN_IMAGE_TEXT_RECIPE)
    registry.register(NEWBIE_TEXT_RECIPE)
    registry.register(ACE15_TEXT_RECIPE)
    return registry


def resolve_text_recipe(
    detected_sources: DetectedTextSources,
    requested_type: str,
    *,
    registry: Registry[TextRecipeDescriptor] | None = None,
) -> TextRecipeBinding:
    """Select encoding behavior without detecting weights or guessing another recipe."""
    if registry is None:
        registry = default_text_recipe_registry()
    detected = (
        ", ".join(
            f"source {index}: {match.descriptor.id}/{role}"
            for index, matches in enumerate(detected_sources)
            for match in matches
            for role, _plan in match.components
        )
        or "no recognized components"
    )
    descriptor = registry.get(requested_type)
    if descriptor is None:
        raise UnresolvedTextRecipe(f"unknown text recipe {requested_type!r}; detected {detected}")
    try:
        return descriptor.bind(detected_sources)
    except UnresolvedTextRecipe as error:
        raise UnresolvedTextRecipe(
            f"text recipe {requested_type!r}: {error}; detected {detected}"
        ) from error
