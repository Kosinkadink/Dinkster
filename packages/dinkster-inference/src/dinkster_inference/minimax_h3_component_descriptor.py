"""H3 artifact roles and provider-bound identities for generic component loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from dinkster_protocol import AttentionPolicy, AttentionRouteToken

from .assembly import NativePlanningContext
from .component_registry import ComponentDescriptor
from .minimax_h3_assembly import (
    MINIMAX_H3_SPLIT_COMMON_ROLES,
    MiniMaxH3CommonComponentRole,
    plan_minimax_h3_common_component,
    plan_minimax_h3_model_assembly,
)
from .minimax_h3_dit import (
    MiniMaxH3DiTRole,
    minimax_h3_dit_component_identity,
    minimax_h3_dit_provider_facts,
    minimax_h3_dit_runtime_identity,
)
from .recipe import ReconstructionRecipe, RuntimeKnobs, WeightSourceBinding, WeightSourceRef
from .weights import AssetIdentifiedSource, WeightSource


@dataclass(frozen=True)
class H3ComponentCandidate:
    source: WeightSource
    path: Path
    artifact_role: str
    asset_digest: str
    asset_size: int


def h3_component_candidate(
    source: WeightSource,
    path: Path,
    artifact_role: str,
) -> H3ComponentCandidate:
    if not isinstance(source, AssetIdentifiedSource):
        raise ValueError("MiniMax H3 component source must carry asset identity")
    digest, size = source.asset_digest, source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise ValueError("MiniMax H3 component source must carry asset identity")
    return H3ComponentCandidate(source, path, artifact_role, digest, size)


def _h3_dit_role(path: Path) -> MiniMaxH3DiTRole:
    """Use FL2VA for architecture-identical DiTs unless the artifact declares REF2VA."""
    name = path.name.casefold()
    if "ref2va" in name:
        return "ref2va-dit"
    return "fl2va-dit"


def detect_h3_components(
    source: WeightSource, path: Path, *, bind_asset_identity: bool = True
) -> tuple[tuple[str, Any], ...]:
    if not bind_asset_identity or not isinstance(source, AssetIdentifiedSource):
        return ()
    detected: list[tuple[str, Any]] = []
    context = NativePlanningContext(torch_version="detection", dinkster_kitchen_version="detection")
    dit_role = _h3_dit_role(path)
    try:
        plan_minimax_h3_model_assembly(
            source,
            role=dit_role,
            path=path,
            context=context,
        )
    except (TypeError, ValueError):
        pass
    else:
        detected.append(("diffusion", h3_component_candidate(source, path, dit_role)))
    for role in MINIMAX_H3_SPLIT_COMMON_ROLES:
        try:
            plan_minimax_h3_common_component(source, role=role, path=path, context=context)
        except (TypeError, ValueError):
            continue
        detected.append((role, h3_component_candidate(source, path, role)))
    return tuple(detected)


class H3ComponentDescriptor(ComponentDescriptor):
    def rebind_attention_recipe(
        self,
        recipe: ReconstructionRecipe,
        runtime_versions: Mapping[str, str] | None = None,
    ) -> ReconstructionRecipe:
        token = recipe.knobs.attention_route_token
        if token is None or recipe.family_id != self.id:
            return recipe
        if token.version == 3:
            raise RuntimeError(
                f"required attention policy {token.requested_policy!r} is unavailable "
                "on the selected worker"
            )
        artifact_role = cast(
            "MiniMaxH3DiTRole",
            next(
                fact.removeprefix("artifact_role=")
                for fact in recipe.component_identity
                if fact.startswith("artifact_role=")
            ),
        )
        providers = dict(token.provider_versions if runtime_versions is None else runtime_versions)
        effective_policy = cast(
            "AttentionPolicy",
            next(route.primary for route in token.routes if route.role == "flux"),
        )
        runtime_facts = minimax_h3_dit_provider_facts(
            artifact_role,
            quantized=any(fact.startswith("int8_provider=") for fact in recipe.knobs.runtime_facts),
            torch_version=providers["torch"],
            dinkster_kitchen_version=providers.get("dinkster-kitchen"),
            attention_policy=effective_policy,
        )
        return replace(
            recipe,
            knobs=replace(recipe.knobs, runtime_facts=runtime_facts),
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
        if runtime_versions is None:
            raise ValueError("MiniMax H3 component identity requires worker runtime versions")
        candidate = cast("H3ComponentCandidate", planned)
        source = candidate.source
        if role == self.model_role:
            artifact_role = cast("MiniMaxH3DiTRole", candidate.artifact_role)
            plan = plan_minimax_h3_model_assembly(
                source,
                role=artifact_role,
                path=candidate.path,
                context=NativePlanningContext(
                    torch_version=runtime_versions.get("torch", ""),
                    dinkster_kitchen_version=runtime_versions.get("dinkster-kitchen", ""),
                ),
                attention_policy=attention_policy,
            )
            return minimax_h3_dit_runtime_identity(
                asset_digest=candidate.asset_digest,
                asset_size=candidate.asset_size,
                role=artifact_role,
                diffusion_dtype=compute_dtype,
                attention_policy=attention_policy,
                attention_route_token=attention_route_token,
                runtime_facts=plan.diffusion.identity_facts,
            )
        plan = plan_minimax_h3_common_component(
            source,
            role=cast("MiniMaxH3CommonComponentRole", candidate.artifact_role),
            path=candidate.path,
            context=NativePlanningContext(
                torch_version=runtime_versions.get("torch", ""),
                dinkster_kitchen_version=runtime_versions.get("dinkster-kitchen", ""),
            ),
        )
        return super().component_identity(role, plan, compute_dtype)

    def recipe(
        self,
        source: WeightSourceRef,
        loaded: Any,
        compute_dtype: str,
        *,
        attention_policy: AttentionPolicy = "auto",
        attention_route_token: AttentionRouteToken | None = None,
    ) -> ReconstructionRecipe:
        if loaded.role != self.model_role:
            return super().recipe(source, loaded, compute_dtype)
        runtime = loaded.runtime
        return ReconstructionRecipe(
            sources=(WeightSourceBinding(self.model_role, source),),
            family_id=self.id,
            component_identity=minimax_h3_dit_component_identity(
                source.digest, source.size, runtime.model_role
            ),
            knobs=RuntimeKnobs(
                diffusion_dtype=compute_dtype,
                text_dtype="unloaded",
                vae_dtype="unloaded",
                fp8_matmul=False,
                runtime_facts=runtime.runtime_facts,
                attention_policy=attention_policy if attention_route_token is not None else "auto",
                attention_route_token=attention_route_token,
            ),
        )
