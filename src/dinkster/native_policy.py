"""Host-side native execution dispatch policy.

The policy sees only immutable composition facts supplied by the composer:
an owner arm name and arm-to-cache-tag mapping. It deliberately knows
nothing about workers or topology records. Checkpoint inspection is
torch-free and happens before engine cache lookup.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Literal, cast

from dinkster_assets import (
    AssetError,
    AssetIntegrityError,
    EmbeddingNameIndex,
    open_verified,
    require_digest,
    verified_local_path,
)
from dinkster_engine import ExecutionSelection
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    DType,
    MalformedSafetensors,
    NativeRefusalError,
    Trellis2FlowConfig,
    build_runtime_identity,
    build_runtime_identity_from_facts,
    compose_execution,
    default_diffusion_dtype,
    default_text_dtype,
    default_vae_dtype,
    load_safetensors_header,
    plan_native,
)
from dinkster_inference.assembly import (
    ComponentPlan,
    LTXAVAudioCodecPlan,
    LTXAVStandaloneComponentPlan,
    LTXAVStandaloneComponentRole,
    Trellis2FlowRole,
    Trellis2SplitModelPlan,
    identify_ltxav_text_source,
)
from dinkster_inference.component_registry import DetectedComponents
from dinkster_inference.ltxav_component import (
    LTXAVAudioCodecAssemblyError,
    LTXAVComponentAssemblyError,
    ltxav_audio_codec_runtime_identity,
    ltxav_component_runtime_identity,
    plan_ltxav_split_audio_codec,
    plan_ltxav_split_component,
)
from dinkster_inference.minimax_h3_component_descriptor import (
    H3ComponentCandidate,
    h3_component_candidate,
)
from dinkster_inference.minimax_h3_dit import MiniMaxH3DiTRole
from dinkster_inference.runtime import uses_classic_embedding_bindings
from dinkster_inference.trellis2_assembly import (
    Trellis2ArtifactRole,
    Trellis2AssemblyError,
    Trellis2PlannedArtifact,
    plan_trellis2_artifact,
    plan_trellis2_flow_artifact,
    trellis2_artifact_runtime_identity,
    trellis2_split_model_runtime_identity,
)
from dinkster_native.legacy_sources import (
    classify_weight_source,
    discover_converted_sidecar,
)
from dinkster_protocol import AttentionPolicy, AttentionRouteToken
from dinkster_values import Value, value_resource_provenance_refs

if TYPE_CHECKING:
    from dinkster_inference.text_recipes import TextRecipeBinding
    from dinkster_schema import NodeSchema

_LOAD_CHECKPOINT = "dinkster.load_checkpoint"
_LOAD_CHECKPOINT_STACK = "dinkster.load_checkpoint_stack"
_LOAD_MODEL_PROFILE = "dinkster.load_model_profile"
_LOAD_CLIP = "dinkster.load_clip"
_LOAD_DUAL_CLIP = "dinkster.load_dual_clip"
_LOAD_DIFFUSION_MODEL = "dinkster.load_diffusion_model"
_LOAD_LATENT_UPSCALE_MODEL = "dinkster.load_latent_upscale_model"
_LOAD_DIFFUSION_COMPONENTS = "dinkster.load_diffusion_components"
_LOAD_LTXAV_AUDIO_VAE = "dinkster.load_ltxav_audio_vae"
_LOAD_LTXAV_TEXT_ENCODER = "dinkster.load_ltxav_text_encoder"
_LOAD_VAE = "dinkster.load_vae"
_LOAD_VISION = "dinkster.load_vision"
_TRELLIS2_SPLIT_FLOW_ROLES: tuple[Trellis2FlowRole, ...] = (
    "structure",
    "shape",
    "shape-512",
    "texture",
    "texture-512",
)
_MINIMAX_H3_DIT_ROLES: tuple[MiniMaxH3DiTRole, ...] = ("fl2va-dit", "ref2va-dit")
_DTYPES = {dtype.name: dtype for dtype in (FLOAT16, BFLOAT16, FLOAT32)}


def resolve_dtype_policy(
    family_id: str,
    policy: Mapping[str, str],
    compute_dtypes: frozenset[str] = frozenset(_DTYPES),
) -> tuple[DType, DType, DType]:
    """Resolve independent serve selectors to concrete inference dtypes."""
    text_default = default_text_dtype(family_id)
    supported = frozenset(dtype for name, dtype in _DTYPES.items() if name in compute_dtypes)
    if family_id == "dinkster.minimax_h3":
        diffusion_default = BFLOAT16
    else:
        diffusion_default = default_diffusion_dtype(family_id)
    defaults = (diffusion_default, text_default, default_vae_dtype(family_id, supported))
    modes = (policy["diffusion"], policy["textEncoder"], policy["vae"])
    # Text falls back to float32 before float16: T5-class RMS variance
    # overflows at float16 while bfloat16/float32 keep the exponent range.
    fallbacks = (
        (FLOAT16, BFLOAT16, FLOAT32),
        (FLOAT32, FLOAT16),
        (),
    )
    try:
        resolved = []
        for mode, default, fallback in zip(modes, defaults, fallbacks, strict=True):
            if mode != "auto":
                resolved.append(_DTYPES[mode])
                continue
            candidates = (default, *fallback)
            resolved.append(next(dtype for dtype in candidates if dtype.name in compute_dtypes))
        return resolved[0], resolved[1], resolved[2]
    except (KeyError, StopIteration) as exc:
        if not compute_dtypes:
            raise ValueError("native compute dtype capability set is empty") from None
        raise ValueError(f"unsupported dtype policy value: {exc.args[0]!r}") from None


def _attention_facts(
    routes: Mapping[str, AttentionRouteToken | None], arm: str
) -> tuple[AttentionPolicy, AttentionRouteToken | None]:
    token = routes.get(arm)
    return ("auto", None) if token is None else (token.requested_policy, token)


def select_resident_producer(
    inputs: Mapping[str, Value],
    cache_tags: Mapping[str, str],
    attention_routes: Mapping[str, AttentionRouteToken | None],
) -> ExecutionSelection | None:
    producer_arms: set[str] = set()
    for value in inputs.values():
        for resource_id, _owner, producer_arm, present in value_resource_provenance_refs(value):
            if not present:
                continue
            if not isinstance(producer_arm, str):
                raise RuntimeError(
                    f"resident resource {resource_id!r} has malformed producer arm stamp"
                )
            if producer_arm not in cache_tags:
                raise RuntimeError(
                    f"resident resource {resource_id!r} names unknown producer arm {producer_arm!r}"
                )
            producer_arms.add(producer_arm)
    if not producer_arms:
        return None
    if len(producer_arms) != 1:
        raise RuntimeError(
            "resident inputs have conflicting producer arms: " + ", ".join(sorted(producer_arms))
        )
    target = next(iter(producer_arms))
    policy, token = _attention_facts(attention_routes, target)
    return ExecutionSelection(
        target=target,
        cache_tag=cache_tags[target],
        attention_policy=policy,
        attention_route_token=token,
    )


def _embedding_binding_digest() -> str | None:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT")
    return None if not snapshot else EmbeddingNameIndex(snapshot).binding_digest


@dataclass(frozen=True)
class NativePolicyDiagnostic:
    """One expected native-policy fallback, suitable for host logging."""

    kind: Literal["convertible", "missing", "refused", "unsupported"]
    digest: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _ProbeVerdict:
    kind: Literal["native", "refused", "unsupported", "missing", "convertible"]
    cache_tag: str | None = None
    fp8_matmul: bool = False
    diffusion_dtype: str | None = None
    text_dtype: str | None = None
    vae_dtype: str | None = None
    reasons: tuple[str, ...] = ()
    source_path: Path | None = None

    @property
    def terminal(self) -> bool:
        return self.kind not in ("missing", "convertible")


@dataclass(frozen=True)
class _LTXAVAudioCodecProbe:
    planned: LTXAVAudioCodecPlan | None = None


@dataclass(frozen=True)
class _Trellis2ArtifactProbe:
    role: Trellis2ArtifactRole | None
    planned: Trellis2PlannedArtifact | None = None


class _LocatedProbeError(Exception):
    """Internal marker preserving the original post-locate exception."""

    def __init__(self, error: Exception) -> None:
        self.error = error


class NativeDispatchPolicy:
    """Select native checkpoint execution and preserve body affinity."""

    def __init__(
        self,
        locate: Callable[[str], Path | None],
        on_diagnostic: Callable[[NativePolicyDiagnostic], None],
        fp8_matmul: Callable[[], bool] = lambda: False,
        dtype_policy: Callable[[], Mapping[str, str]] = lambda: {
            "diffusion": "auto",
            "textEncoder": "auto",
            "vae": "auto",
        },
        compute_dtypes: Callable[[], frozenset[str]] = lambda: frozenset(_DTYPES),
        schedule_conversion: (
            Callable[[Path, str], Awaitable[tuple[str, str | None]]] | None
        ) = None,
        minimax_h3_runtime_versions: Callable[[], Mapping[str, str]] | None = None,
        schemas: Callable[[], Mapping[str, NodeSchema]] = lambda: {},
    ) -> None:
        self._locate = locate
        self._on_diagnostic = on_diagnostic
        self._fp8_matmul = fp8_matmul
        self._dtype_policy = dtype_policy
        self._compute_dtypes = compute_dtypes
        self._schedule_conversion = schedule_conversion
        self._minimax_h3_runtime_versions = minimax_h3_runtime_versions
        self._schemas = schemas
        self._next_native_lane = 0
        self._run_native_lanes: dict[str, str] = {}
        self._memo: dict[tuple[object, ...], _ProbeVerdict] = {}
        self._inflight: dict[tuple[object, ...], asyncio.Task[_ProbeVerdict]] = {}
        self._ltxav_component_probes: dict[
            tuple[str, LTXAVStandaloneComponentRole], LTXAVStandaloneComponentPlan | None
        ] = {}
        self._ltxav_component_probes_lock = Lock()
        self._ltxav_audio_codec_probes: dict[str, _LTXAVAudioCodecProbe] = {}
        self._ltxav_audio_codec_probes_lock = Lock()
        self._trellis2_artifact_probes: dict[str, _Trellis2ArtifactProbe] = {}
        self._trellis2_artifact_probes_lock = Lock()
        self._trellis2_flow_probes: dict[
            tuple[str, Trellis2FlowRole], ComponentPlan[object] | None
        ] = {}
        self._trellis2_flow_probes_lock = Lock()
        self._minimax_h3_dit_probes: dict[
            tuple[str, MiniMaxH3DiTRole], H3ComponentCandidate | None
        ] = {}
        self._minimax_h3_dit_probes_lock = Lock()
        self._ltxav_text_probes: dict[str, Literal["gemma3_12b", "gemma4_12b", False]] = {}
        self._ltxav_text_probes_lock = Lock()
        self._component_probes: dict[
            tuple[str, tuple[str, ...]], tuple[DetectedComponents, ...]
        ] = {}
        self._component_probes_lock = Lock()
        self._conversions: dict[str, asyncio.Task[None]] = {}
        self._conversion_refusals: dict[str, _ProbeVerdict] = {}
        self._missing: set[str] = set()
        self._convertible: set[str] = set()

    async def select(
        self,
        node_type: str,
        inputs: Mapping[str, Value],
        arms: tuple[str, Mapping[str, str]],
        *,
        run_id: str | None = None,
        extension_behavior_hash: str | None = None,
        attention_routes: Mapping[str, AttentionRouteToken | None] | None = None,
    ) -> ExecutionSelection | None:
        """Return an explicit arm selection, or leave owner planning alone."""
        owner_arm, cache_tags = arms
        attention_routes = {} if attention_routes is None else attention_routes
        native_arms = tuple(arm for arm in cache_tags if arm.partition(":")[0].endswith("@native"))
        owner_attention = _attention_facts(attention_routes, owner_arm)
        schema = self._schemas().get(node_type)
        if schema is not None and schema.dispatch_affinity == "native":
            if not native_arms:
                return None
            native_arm = self._choose_native_lane(native_arms, run_id)
            native_attention = _attention_facts(attention_routes, native_arm)
            return ExecutionSelection(
                target=native_arm,
                cache_tag=cache_tags[native_arm],
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            )
        if node_type in (_LOAD_LTXAV_TEXT_ENCODER, _LOAD_LTXAV_AUDIO_VAE):
            if not native_arms:
                raise RuntimeError(f"{node_type} requires an available native arm")
            native_arm = self._choose_native_lane(native_arms, run_id)
            native_attention = _attention_facts(attention_routes, native_arm)
            return await self._select_ltxav_loader(
                node_type,
                inputs,
                native_arm,
                native_attention,
            )
        if node_type == _LOAD_LATENT_UPSCALE_MODEL:
            native_arm = self._choose_native_lane(native_arms, run_id) if native_arms else None
            native_attention = (
                _attention_facts(attention_routes, native_arm)
                if native_arm is not None
                else owner_attention
            )
            return await self._select_ltxav_latent_upscaler(
                inputs,
                owner_arm,
                native_arm,
                cache_tags,
                native_attention,
                owner_attention,
            )
        if node_type in (_LOAD_CLIP, _LOAD_DUAL_CLIP, _LOAD_VAE):
            native_arm = self._choose_native_lane(native_arms, run_id) if native_arms else None
            native_attention = (
                _attention_facts(attention_routes, native_arm)
                if native_arm is not None
                else owner_attention
            )
            if node_type == _LOAD_DUAL_CLIP:
                return await self._select_dual_clip(
                    inputs,
                    native_arm,
                    native_attention,
                )
            return await self._select_component(
                node_type,
                inputs,
                owner_arm,
                native_arm,
                cache_tags,
                native_attention,
                owner_attention,
            )
        if node_type == _LOAD_VISION:
            native_arm = self._choose_native_lane(native_arms, run_id) if native_arms else None
            native_attention = (
                _attention_facts(attention_routes, native_arm)
                if native_arm is not None
                else owner_attention
            )
            return await self._select_trellis2_vision(
                inputs,
                owner_arm,
                native_arm,
                cache_tags,
                native_attention,
                owner_attention,
            )
        if node_type == _LOAD_MODEL_PROFILE:
            return await self._select_model_profile(
                inputs,
                owner_arm,
                native_arms,
                cache_tags,
                run_id,
                extension_behavior_hash,
                attention_routes,
                owner_attention,
            )
        if node_type == _LOAD_DIFFUSION_MODEL:
            native_only_provider = not native_arms
            if native_only_provider:
                native_arms = (owner_arm,)
            native_arm = self._choose_native_lane(native_arms, run_id)
            native_attention = _attention_facts(attention_routes, native_arm)
            selection = await self._select_diffusion_model(
                inputs,
                owner_arm,
                native_arm,
                cache_tags,
                native_attention,
                owner_attention,
            )
            if native_only_provider and selection.cache_tag == cache_tags[owner_arm]:
                raise RuntimeError(
                    "dinkster.load_diffusion_model has no execution provider for an asset "
                    "that native admission did not recognize"
                )
            return selection
        if node_type == _LOAD_DIFFUSION_COMPONENTS:
            native_arm = self._choose_native_lane(native_arms, run_id) if native_arms else owner_arm
            native_attention = _attention_facts(attention_routes, native_arm)
            return await self._select_diffusion_components(
                inputs,
                native_arm,
                native_attention,
            )
        if node_type in (_LOAD_CHECKPOINT, _LOAD_CHECKPOINT_STACK):
            native_only_provider = not native_arms
            if native_only_provider:
                native_arms = (owner_arm,)
            embedding_binding_digest = _embedding_binding_digest()
            native_arm = self._choose_native_lane(native_arms, run_id)
            native_attention = _attention_facts(attention_routes, native_arm)
            return await self._select_checkpoint(
                node_type,
                inputs,
                owner_arm,
                native_arm,
                cache_tags,
                extension_behavior_hash,
                embedding_binding_digest,
                native_attention,
                owner_attention,
                require_native=native_only_provider,
            )
        return select_resident_producer(inputs, cache_tags, attention_routes)

    async def _select_model_profile(
        self,
        inputs: Mapping[str, Value],
        owner_arm: str,
        native_arms: tuple[str, ...],
        cache_tags: Mapping[str, str],
        run_id: str | None,
        extension_behavior_hash: str | None,
        attention_routes: Mapping[str, AttentionRouteToken | None],
        owner_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        checkpoint = inputs.get("checkpoint")
        entries = inputs.get("entries")
        if checkpoint is None or entries is None:
            raise RuntimeError("dinkster.load_model_profile requires checkpoint and entries inputs")
        digest = checkpoint.meta.get("digest")
        if not isinstance(digest, str):
            raise RuntimeError(
                "dinkster.load_model_profile checkpoint has no canonical asset digest"
            )
        try:
            require_digest(digest)
        except AssetError as error:
            raise RuntimeError(
                "dinkster.load_model_profile checkpoint has no canonical asset digest"
            ) from error
        stored = entries.payload.load()
        if not isinstance(stored, str):
            raise RuntimeError("dinkster.load_model_profile entries must be stored JSON text")
        candidate = self._locate(digest)
        if candidate is None:
            raise RuntimeError("dinkster.load_model_profile checkpoint is not locally materialized")
        from dinkster_inference import load_model_output_profile

        try:
            with open_verified(candidate, digest) as verified:
                profile = load_model_output_profile(
                    candidate,
                    asset_digest=digest,
                    asset_size=os.fstat(verified.fileno()).st_size,
                    stored=stored,
                    handle=verified,
                )
        except ValueError as error:
            raise RuntimeError(str(error)) from error

        native_only_provider = not native_arms
        selected_arms = native_arms or (owner_arm,)
        native_arm = self._choose_native_lane(selected_arms, run_id)
        native_attention = _attention_facts(attention_routes, native_arm)
        if profile.kind == "checkpoint":
            return await self._select_checkpoint(
                _LOAD_MODEL_PROFILE,
                {"checkpoint": checkpoint},
                owner_arm,
                native_arm,
                cache_tags,
                extension_behavior_hash,
                _embedding_binding_digest(),
                native_attention,
                owner_attention,
                require_native=native_only_provider,
            )
        return await self._select_diffusion_model(
            {"diffusion_model": checkpoint},
            owner_arm,
            native_arm,
            cache_tags,
            native_attention,
            owner_attention,
        )

    def _choose_native_lane(self, native_arms: tuple[str, ...], run_id: str | None = None) -> str:
        """Commit the run's native lane.

        Loads must commit the lane before probing: probe cache identities
        embed the selected arm's attention route token (device SM, provider
        versions), so an identity computed against one arm is refused by any
        other arm on heterogeneous replicas.
        """
        if not native_arms:
            raise ValueError("native lane selection requires at least one arm")
        if run_id is not None:
            selected = self._run_native_lanes.get(run_id)
            if selected in native_arms:
                return selected
        selected = native_arms[self._next_native_lane % len(native_arms)]
        self._next_native_lane += 1
        if run_id is not None:
            self._run_native_lanes[run_id] = selected
        return selected

    def release_run(self, run_id: str) -> None:
        self._run_native_lanes.pop(run_id, None)

    async def _select_checkpoint(
        self,
        node_type: str,
        inputs: Mapping[str, Value],
        owner_arm: str,
        native_arm: str,
        cache_tags: Mapping[str, str],
        extension_behavior_hash: str | None,
        embedding_binding_digest: str | None,
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
        owner_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
        *,
        require_native: bool,
    ) -> ExecutionSelection:
        checkpoint = inputs.get("checkpoint")
        if checkpoint is None:
            raise RuntimeError("dinkster.load_checkpoint is missing its checkpoint asset input")
        digest = checkpoint.meta.get("digest")
        logical_name = checkpoint.meta.get("name")
        if not isinstance(digest, str):
            raise RuntimeError(
                "dinkster.load_checkpoint checkpoint asset has missing or malformed digest metadata"
            )
        try:
            require_digest(digest)
        except AssetError as exc:
            raise RuntimeError(
                "dinkster.load_checkpoint checkpoint asset has missing or malformed digest metadata"
            ) from exc
        if not isinstance(logical_name, str):
            logical_name = ""

        fp8_matmul = self._fp8_matmul()
        memo_key = (
            digest,
            logical_name,
            fp8_matmul,
            tuple(sorted(self._dtype_policy().items())),
            extension_behavior_hash,
            embedding_binding_digest,
            *native_attention,
        )
        verdict = self._conversion_refusals.get(digest) or self._memo.get(memo_key)
        if verdict is None:
            verdict = await self._shared_probe(
                digest,
                logical_name,
                fp8_matmul,
                extension_behavior_hash,
                embedding_binding_digest,
                *native_attention,
            )
        if verdict.kind == "native":
            assert verdict.cache_tag is not None
            return ExecutionSelection(
                target=native_arm,
                cache_tag=verdict.cache_tag,
                fp8_matmul=verdict.fp8_matmul,
                diffusion_dtype=verdict.diffusion_dtype,
                text_dtype=verdict.text_dtype,
                vae_dtype=verdict.vae_dtype,
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            )
        if verdict.kind == "missing":
            if digest not in self._missing:
                self._missing.add(digest)
                self._emit("missing", digest, verdict.reasons)
        elif verdict.kind == "convertible":
            if digest not in self._convertible:
                self._convertible.add(digest)
                self._emit("convertible", digest, verdict.reasons)
            if verdict.source_path is not None:
                self._start_conversion(digest, verdict.source_path, logical_name)
        if require_native:
            raise RuntimeError(f"{node_type} cannot load checkpoint: " + "; ".join(verdict.reasons))
        return ExecutionSelection(
            target=owner_arm,
            cache_tag=cache_tags[owner_arm],
            attention_policy=owner_attention[0],
            attention_route_token=owner_attention[1],
        )

    def _probe_ltxav_text_encoder_transaction(
        self, digest: str
    ) -> Literal["gemma3_12b", "gemma4_12b"] | None:
        """Classify a local asset as the LTX-2 Gemma text encoder by weight
        geometry. The verdict is memoized per digest; a locate miss is never
        memoized because the asset may materialize later."""
        with self._ltxav_text_probes_lock:
            if digest in self._ltxav_text_probes:
                cached = self._ltxav_text_probes[digest]
                return None if cached is False else cached
        candidate = self._locate(digest)
        if candidate is None:
            return None
        try:
            size = candidate.stat().st_size
            path = verified_local_path(candidate, digest)
        except (AssetIntegrityError, OSError) as error:
            raise _LocatedProbeError(error) from error
        verdict: Literal["gemma3_12b", "gemma4_12b", False] = False
        try:
            source = load_safetensors_header(path, asset_digest=digest, asset_size=size)
        except MalformedSafetensors:
            source = None
        except OSError as error:
            raise _LocatedProbeError(error) from error
        if source is not None:
            verdict = identify_ltxav_text_source(source) or False
        with self._ltxav_text_probes_lock:
            self._ltxav_text_probes[digest] = verdict
        return None if verdict is False else verdict

    async def _select_ltxav_loader(
        self,
        node_type: str,
        inputs: Mapping[str, Value],
        native_arm: str,
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        def digest_for(input_id: str) -> str:
            value = inputs.get(input_id)
            if value is None:
                raise RuntimeError(f"{node_type} is missing its {input_id} asset input")
            digest = value.meta.get("digest")
            if not isinstance(digest, str):
                raise RuntimeError(
                    f"{node_type} {input_id} asset has missing or malformed digest metadata"
                )
            try:
                require_digest(digest)
            except AssetError as error:
                raise RuntimeError(
                    f"{node_type} {input_id} asset has missing or malformed digest metadata"
                ) from error
            return digest

        checkpoint_digest = digest_for("ckpt_name")
        if node_type == _LOAD_LTXAV_AUDIO_VAE:
            try:
                probe = await asyncio.to_thread(
                    self._probe_ltxav_audio_codec_transaction,
                    checkpoint_digest,
                )
            except _LocatedProbeError as wrapped:
                raise wrapped.error from None
            if probe.planned is None:
                raise RuntimeError(
                    "dinkster.load_ltxav_audio_vae requires an LTX-2 checkpoint with an audio codec"
                )
            return ExecutionSelection(
                target=native_arm,
                cache_tag=ltxav_audio_codec_runtime_identity(probe.planned),
                diffusion_dtype="unloaded",
                text_dtype="unloaded",
                vae_dtype="float32",
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            )

        text_digest = digest_for("text_encoder")
        try:
            text_role = await asyncio.to_thread(
                self._probe_ltxav_text_encoder_transaction,
                text_digest,
            )
            if text_role is None:
                raise RuntimeError(
                    "dinkster.load_ltxav_text_encoder requires an exact supported LTX-2 Gemma asset"
                )
            gemma, projection = await asyncio.gather(
                asyncio.to_thread(
                    self._probe_ltxav_component_transaction,
                    text_digest,
                    text_role,
                ),
                asyncio.to_thread(
                    self._probe_ltxav_component_transaction,
                    checkpoint_digest,
                    "text_projection",
                ),
            )
            if projection is None:
                projection = await asyncio.to_thread(
                    self._probe_ltxav_component_transaction,
                    text_digest,
                    "text_projection",
                )
        except _LocatedProbeError as wrapped:
            raise wrapped.error from None
        if gemma is None:
            raise RuntimeError(
                f"dinkster.load_ltxav_text_encoder requires an exact LTX-2 {text_role} asset"
            )
        if projection is None:
            raise RuntimeError("dinkster.load_ltxav_text_encoder requires an LTX-2 text projection")
        text_dtype = resolve_dtype_policy(
            "dinkster.ltxav", self._dtype_policy(), self._compute_dtypes()
        )[1]
        projection_kind = projection.component.config
        if (text_role == "gemma4_12b") != (projection_kind == "dual_linear_gemma4"):
            raise RuntimeError(
                "dinkster.load_ltxav_text_encoder Gemma and checkpoint projection profiles differ"
            )
        components = {
            text_role: ltxav_component_runtime_identity(
                gemma,
                text_dtype,
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            ),
            "text_projection": ltxav_component_runtime_identity(
                projection,
                text_dtype,
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            ),
        }
        if projection_kind == "single_linear":
            try:
                connectors = await asyncio.to_thread(
                    self._probe_ltxav_component_transaction,
                    checkpoint_digest,
                    "connectors",
                )
            except _LocatedProbeError as wrapped:
                raise wrapped.error from None
            if connectors is None:
                raise RuntimeError("LTX-2 19B text loading requires the checkpoint connector pair")
            components["connectors"] = ltxav_component_runtime_identity(
                connectors,
                text_dtype,
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            )
        composition = compose_execution("dinkster.ltxav", components)
        return ExecutionSelection(
            target=native_arm,
            cache_tag=composition.execution_identity,
            diffusion_dtype="unloaded",
            text_dtype=text_dtype.name,
            vae_dtype="unloaded",
            attention_policy=native_attention[0],
            attention_route_token=native_attention[1],
        )

    async def _select_ltxav_latent_upscaler(
        self,
        inputs: Mapping[str, Value],
        owner_arm: str,
        native_arm: str | None,
        cache_tags: Mapping[str, str],
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
        owner_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        model = inputs.get("model_name")
        if model is None:
            raise RuntimeError(
                "dinkster.load_latent_upscale_model is missing its model_name asset input"
            )
        digest = model.meta.get("digest")
        if not isinstance(digest, str):
            raise RuntimeError(
                "dinkster.load_latent_upscale_model asset has missing or malformed digest metadata"
            )
        try:
            require_digest(digest)
        except AssetError as error:
            raise RuntimeError(
                "dinkster.load_latent_upscale_model asset has missing or malformed digest metadata"
            ) from error
        try:
            planned = await asyncio.to_thread(
                self._probe_ltxav_component_transaction,
                digest,
                "latent_upscaler",
            )
        except _LocatedProbeError as wrapped:
            raise wrapped.error from None
        if planned is None or native_arm is None:
            return ExecutionSelection(
                target=owner_arm,
                cache_tag=cache_tags[owner_arm],
                attention_policy=owner_attention[0],
                attention_route_token=owner_attention[1],
            )
        dtype = resolve_dtype_policy(
            "dinkster.ltxav", self._dtype_policy(), self._compute_dtypes()
        )[2]
        return ExecutionSelection(
            target=native_arm,
            cache_tag=ltxav_component_runtime_identity(
                planned,
                dtype,
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            ),
            diffusion_dtype="unloaded",
            text_dtype="unloaded",
            vae_dtype=dtype.name,
            attention_policy=native_attention[0],
            attention_route_token=native_attention[1],
        )

    async def _select_component(
        self,
        node_type: str,
        inputs: Mapping[str, Value],
        owner_arm: str,
        native_arm: str | None,
        cache_tags: Mapping[str, str],
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
        owner_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        input_name = "text_encoder" if node_type == _LOAD_CLIP else "vae"
        value = inputs.get(input_name)
        if node_type == _LOAD_VAE:
            pixel_value = inputs.get("pixel_space")
            pixel_space = False if pixel_value is None else pixel_value.payload.load()
            if type(pixel_space) is not bool:
                raise RuntimeError("dinkster.load_vae pixel_space input is malformed")
            if pixel_space:
                if value is not None:
                    raise RuntimeError(
                        "dinkster.load_vae pixel_space and vae are mutually exclusive"
                    )
                if native_arm is None:
                    raise RuntimeError(
                        "dinkster.load_vae pixel_space requires an available native arm"
                    )
                dtype = default_vae_dtype("dinkster.chroma_radiance")
                identity = build_runtime_identity_from_facts(
                    "dinkster.chroma_radiance",
                    ("family=dinkster.chroma_radiance", "component=pixel_space"),
                    diffusion_dtype="unloaded",
                    text_dtype="unloaded",
                    vae_dtype=dtype.name,
                    fp8_matmul=False,
                )
                return ExecutionSelection(
                    target=native_arm,
                    cache_tag=identity,
                    diffusion_dtype="unloaded",
                    text_dtype="unloaded",
                    vae_dtype=dtype.name,
                    attention_policy=native_attention[0],
                    attention_route_token=native_attention[1],
                )
        if value is None:
            raise RuntimeError(f"{node_type} is missing its {input_name} asset input")
        digest = value.meta.get("digest")
        if not isinstance(digest, str):
            raise RuntimeError(f"{node_type} asset has missing or malformed digest metadata")
        try:
            require_digest(digest)
        except AssetError as exc:
            raise RuntimeError(
                f"{node_type} asset has missing or malformed digest metadata"
            ) from exc
        clip_type = "stable_diffusion"
        if node_type == _LOAD_CLIP:
            type_value = inputs.get("type")
            clip_type = "stable_diffusion" if type_value is None else type_value.payload.load()
            if not isinstance(clip_type, str):
                raise RuntimeError("dinkster.load_clip type is malformed")
        try:
            matches = await asyncio.to_thread(self._probe_components_transaction, digest)
        except _LocatedProbeError as wrapped:
            raise wrapped.error from None
        if not matches:
            return ExecutionSelection(
                target=owner_arm,
                cache_tag=cache_tags[owner_arm],
                attention_policy=owner_attention[0],
                attention_route_token=owner_attention[1],
            )
        from dinkster_inference.component_catalog import default_component_registry
        from dinkster_inference.component_registry import AmbiguousComponentError

        kind = "text" if node_type == _LOAD_CLIP else "codec"
        if kind == "text":
            from dinkster_inference.text_recipes import UnresolvedTextRecipe, resolve_text_recipe

            try:
                binding = resolve_text_recipe((matches,), clip_type)
            except UnresolvedTextRecipe as error:
                fixed = tuple(
                    match for match in matches if not match.descriptor.requires_text_recipe
                )
                if not fixed:
                    self._emit("unsupported", digest, (str(error),))
                    return ExecutionSelection(
                        target=owner_arm,
                        cache_tag=cache_tags[owner_arm],
                        attention_policy=owner_attention[0],
                        attention_route_token=owner_attention[1],
                    )
                matches = fixed
            else:
                if native_arm is None:
                    raise RuntimeError(
                        f"{node_type} requires an available native arm for {binding.id}"
                    )
                return self._select_text_recipe(binding, native_arm, native_attention)
        try:
            descriptor, role, plan = default_component_registry().select_detected(
                matches, kind, family_id=f"dinkster.{clip_type}" if kind == "text" else None
            )
        except ValueError as error:
            if kind == "text" and isinstance(error, AmbiguousComponentError):
                self._emit(
                    "unsupported",
                    digest,
                    (f"native text profile {clip_type!r} is unresolved: {error}",),
                )
                return ExecutionSelection(
                    target=owner_arm,
                    cache_tag=cache_tags[owner_arm],
                    attention_policy=owner_attention[0],
                    attention_route_token=owner_attention[1],
                )
            raise RuntimeError(f"{node_type}: {error}") from error
        family = descriptor.family_for(plan)
        if native_arm is None:
            raise RuntimeError(
                f"{node_type} requires an available native arm for {family} role {role!r}"
            )
        dtype_policy = self._dtype_policy()
        dtypes = resolve_dtype_policy(family, dtype_policy, self._compute_dtypes())
        dtype = dtypes[1] if node_type == _LOAD_CLIP else dtypes[2]
        if dtype_policy["textEncoder" if kind == "text" else "vae"] == "auto":
            dtype = dict(descriptor.role_default_dtypes).get(role, dtype)
        runtime_versions = None
        if descriptor.requires_runtime_versions:
            if self._minimax_h3_runtime_versions is None:
                raise RuntimeError(
                    f"{descriptor.family.display_name} runtime versions are unavailable"
                )
            runtime_versions = self._minimax_h3_runtime_versions()
        identity = descriptor.component_identity(
            role,
            plan,
            dtype.name,
            attention_policy=native_attention[0],
            attention_route_token=native_attention[1],
            runtime_versions=runtime_versions,
        )
        return ExecutionSelection(
            target=native_arm,
            cache_tag=identity,
            diffusion_dtype="unloaded",
            text_dtype=dtype.name if node_type == _LOAD_CLIP else "unloaded",
            vae_dtype=dtype.name if node_type == _LOAD_VAE else "unloaded",
            attention_policy=native_attention[0],
            attention_route_token=native_attention[1],
        )

    async def _select_dual_clip(
        self,
        inputs: Mapping[str, Value],
        native_arm: str | None,
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        digests: list[str] = []
        for input_name in ("text_encoder1", "text_encoder2"):
            value = inputs.get(input_name)
            if value is None:
                raise RuntimeError(f"{_LOAD_DUAL_CLIP} is missing its {input_name} asset input")
            digest = value.meta.get("digest")
            if not isinstance(digest, str):
                raise RuntimeError(
                    f"{_LOAD_DUAL_CLIP} {input_name} asset has missing or malformed digest metadata"
                )
            try:
                require_digest(digest)
            except AssetError as error:
                raise RuntimeError(
                    f"{_LOAD_DUAL_CLIP} {input_name} asset has missing or malformed digest metadata"
                ) from error
            digests.append(digest)

        type_value = inputs.get("type")
        requested_type = "sdxl" if type_value is None else type_value.payload.load()
        if not isinstance(requested_type, str):
            raise RuntimeError(f"{_LOAD_DUAL_CLIP} type is malformed")

        detected_sources: list[tuple[DetectedComponents, ...]] = []
        for digest in digests:
            try:
                matches = await asyncio.to_thread(self._probe_components_transaction, digest)
            except _LocatedProbeError as wrapped:
                raise wrapped.error from None
            detected_sources.append(matches)

        from dinkster_inference.text_recipes import UnresolvedTextRecipe, resolve_text_recipe

        try:
            binding = resolve_text_recipe(tuple(detected_sources), requested_type)
        except UnresolvedTextRecipe as error:
            reasons = (str(error),)
            for digest in digests:
                self._emit("unsupported", digest, reasons)
            raise RuntimeError(
                f"{_LOAD_DUAL_CLIP} has no native text recipe for the ordered sources: {error}"
            ) from error
        if native_arm is None:
            raise RuntimeError(
                f"{_LOAD_DUAL_CLIP} requires an available native arm for {binding.id}"
            )
        return self._select_text_recipe(binding, native_arm, native_attention)

    def _select_text_recipe(
        self,
        binding: TextRecipeBinding,
        native_arm: str,
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        from dinkster_inference.clip_text import ClipTextConfig

        dtype = resolve_dtype_policy(
            binding.family_id, self._dtype_policy(), self._compute_dtypes()
        )[1]
        knobs = binding.knobs(
            dtype.name,
            attention_policy=native_attention[0],
            attention_route_token=native_attention[1],
            embedding_binding_digest=(
                _embedding_binding_digest()
                if any(
                    isinstance(component.plan.config, ClipTextConfig)
                    for component in binding.components
                )
                else None
            ),
        )
        identity = build_runtime_identity_from_facts(
            binding.family_id,
            binding.component_identity,
            diffusion_dtype=knobs.diffusion_dtype,
            text_dtype=knobs.text_dtype,
            vae_dtype=knobs.vae_dtype,
            fp8_matmul=knobs.fp8_matmul,
            attention_policy=knobs.attention_policy,
            attention_route_token=knobs.attention_route_token,
            embedding_binding_digest=knobs.embedding_binding_digest,
            runtime_facts=knobs.runtime_facts,
        )
        return ExecutionSelection(
            target=native_arm,
            cache_tag=identity,
            diffusion_dtype=knobs.diffusion_dtype,
            text_dtype=knobs.text_dtype,
            vae_dtype=knobs.vae_dtype,
            attention_policy=native_attention[0],
            attention_route_token=native_attention[1],
        )

    async def _select_trellis2_vision(
        self,
        inputs: Mapping[str, Value],
        owner_arm: str,
        native_arm: str | None,
        cache_tags: Mapping[str, str],
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
        owner_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        value = inputs.get("vision_encoder")
        if value is None:
            raise RuntimeError("dinkster.load_vision is missing its vision_encoder asset input")
        digest = value.meta.get("digest")
        if not isinstance(digest, str):
            raise RuntimeError(
                "dinkster.load_vision asset has missing or malformed digest metadata"
            )
        try:
            require_digest(digest)
        except AssetError as error:
            raise RuntimeError(
                "dinkster.load_vision asset has missing or malformed digest metadata"
            ) from error
        try:
            probe = await asyncio.to_thread(self._probe_trellis2_artifact_transaction, digest)
        except _LocatedProbeError as wrapped:
            raise wrapped.error from None
        if probe.role is None:
            return ExecutionSelection(
                target=owner_arm,
                cache_tag=cache_tags[owner_arm],
                attention_policy=owner_attention[0],
                attention_route_token=owner_attention[1],
            )
        if probe.role != "vision":
            raise RuntimeError(
                "dinkster.load_vision TRELLIS.2 role mismatch: "
                f"expected 'vision', got {probe.role!r}"
            )
        if native_arm is None:
            raise RuntimeError(
                "dinkster.load_vision requires an available native arm for TRELLIS.2"
            )
        planned = probe.planned
        assert planned is not None
        dtype = resolve_dtype_policy(
            planned.family_id, self._dtype_policy(), self._compute_dtypes()
        )[1]
        return ExecutionSelection(
            target=native_arm,
            cache_tag=trellis2_artifact_runtime_identity(planned, dtype),
            diffusion_dtype="unloaded",
            text_dtype=dtype.name,
            vae_dtype="unloaded",
            attention_policy=native_attention[0],
            attention_route_token=native_attention[1],
        )

    def _probe_ltxav_component_transaction(
        self,
        digest: str,
        role: LTXAVStandaloneComponentRole,
    ) -> LTXAVStandaloneComponentPlan | None:
        """Plan one LTX-2 role from an independently selected asset."""

        key = (digest, role)
        with self._ltxav_component_probes_lock:
            if key in self._ltxav_component_probes:
                return self._ltxav_component_probes[key]
        candidate = self._locate(digest)
        if candidate is None:
            return None
        try:
            size = candidate.stat().st_size
            path = verified_local_path(candidate, digest)
        except (AssetIntegrityError, OSError) as error:
            raise _LocatedProbeError(error) from error
        planned = None
        try:
            source = load_safetensors_header(path, asset_digest=digest, asset_size=size)
        except MalformedSafetensors:
            source = None
        except OSError as error:
            raise _LocatedProbeError(error) from error
        if source is not None:
            try:
                planned = plan_ltxav_split_component(source, role=role, path=path)
            except LTXAVComponentAssemblyError:
                pass
        with self._ltxav_component_probes_lock:
            self._ltxav_component_probes[key] = planned
        return planned

    def _probe_ltxav_audio_codec_transaction(self, digest: str) -> _LTXAVAudioCodecProbe:
        """Classify a local asset as one exact standalone LTX-2 audio codec."""

        with self._ltxav_audio_codec_probes_lock:
            probe = self._ltxav_audio_codec_probes.get(digest)
        if probe is not None:
            return probe
        candidate = self._locate(digest)
        if candidate is None:
            return _LTXAVAudioCodecProbe()
        try:
            size = candidate.stat().st_size
            path = verified_local_path(candidate, digest)
        except (AssetIntegrityError, OSError) as error:
            raise _LocatedProbeError(error) from error
        probe = _LTXAVAudioCodecProbe()
        try:
            source = load_safetensors_header(path, asset_digest=digest, asset_size=size)
        except MalformedSafetensors:
            source = None
        except OSError as error:
            raise _LocatedProbeError(error) from error
        if source is not None:
            try:
                planned = plan_ltxav_split_audio_codec(source, path=path)
            except LTXAVAudioCodecAssemblyError:
                pass
            else:
                probe = _LTXAVAudioCodecProbe(planned)
        with self._ltxav_audio_codec_probes_lock:
            self._ltxav_audio_codec_probes[digest] = probe
        return probe

    def _probe_trellis2_artifact_transaction(self, digest: str) -> _Trellis2ArtifactProbe:
        """Classify one local asset as an exact TRELLIS.2 artifact."""

        with self._trellis2_artifact_probes_lock:
            probe = self._trellis2_artifact_probes.get(digest)
        if probe is not None:
            return probe
        candidate = self._locate(digest)
        if candidate is None:
            return _Trellis2ArtifactProbe(None)
        try:
            size = candidate.stat().st_size
            path = verified_local_path(candidate, digest)
        except (AssetIntegrityError, OSError) as error:
            raise _LocatedProbeError(error) from error
        probe = _Trellis2ArtifactProbe(None)
        try:
            source = load_safetensors_header(path, asset_digest=digest, asset_size=size)
        except MalformedSafetensors:
            source = None
        except OSError as error:
            raise _LocatedProbeError(error) from error
        if source is not None:
            roles: tuple[Trellis2ArtifactRole, ...] = (
                "diffusion",
                "vision",
                "structure-decoder",
                "shape-decoder",
                "texture-decoder",
            )
            for role in roles:
                try:
                    planned = plan_trellis2_artifact(source, role=role, path=path)
                except Trellis2AssemblyError:
                    continue
                probe = _Trellis2ArtifactProbe(role, planned)
                break
        with self._trellis2_artifact_probes_lock:
            self._trellis2_artifact_probes[digest] = probe
        return probe

    def _probe_trellis2_flow_transaction(
        self,
        digest: str,
        role: Trellis2FlowRole,
    ) -> ComponentPlan[object] | None:
        """Plan one explicitly named split flow artifact."""

        key = (digest, role)
        with self._trellis2_flow_probes_lock:
            cached = self._trellis2_flow_probes.get(key)
            known = key in self._trellis2_flow_probes
        if known:
            return cached
        candidate = self._locate(digest)
        if candidate is None:
            return None
        try:
            size = candidate.stat().st_size
            path = verified_local_path(candidate, digest)
        except (AssetIntegrityError, OSError) as error:
            raise _LocatedProbeError(error) from error
        try:
            source = load_safetensors_header(path, asset_digest=digest, asset_size=size)
            planned = plan_trellis2_flow_artifact(source, role=role, path=path)
        except (MalformedSafetensors, Trellis2AssemblyError):
            planned = None
        except OSError as error:
            raise _LocatedProbeError(error) from error
        with self._trellis2_flow_probes_lock:
            self._trellis2_flow_probes[key] = planned
        return planned

    def _probe_minimax_h3_dit_transaction(
        self,
        digest: str,
        role: MiniMaxH3DiTRole,
    ) -> H3ComponentCandidate | None:
        key = (digest, role)
        with self._minimax_h3_dit_probes_lock:
            cached = self._minimax_h3_dit_probes.get(key)
            known = key in self._minimax_h3_dit_probes
        if known:
            return cached
        candidate = self._locate(digest)
        if candidate is None:
            return None
        try:
            size = candidate.stat().st_size
            path = verified_local_path(candidate, digest)
            source = load_safetensors_header(path, asset_digest=digest, asset_size=size)
            planned = h3_component_candidate(source, path, role)
        except MalformedSafetensors:
            planned = None
        except (AssetIntegrityError, OSError) as error:
            raise _LocatedProbeError(error) from error
        with self._minimax_h3_dit_probes_lock:
            self._minimax_h3_dit_probes[key] = planned
        return planned

    async def _select_diffusion_components(
        self,
        inputs: Mapping[str, Value],
        native_arm: str,
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        members: dict[str, dict[str, Value]] = {}
        for input_id, value in inputs.items():
            family, separator, suffix = input_id.partition(".")
            if family != "components" or not separator:
                continue
            member, separator, field = suffix.partition(".")
            if not separator or field not in ("component", "role"):
                raise RuntimeError(f"invalid diffusion component input {input_id!r}")
            members.setdefault(member, {})[field] = value
        selected: list[tuple[str, str]] = []
        for member, fields in members.items():
            if set(fields) != {"component", "role"}:
                raise RuntimeError(f"diffusion component {member!r} requires an asset and role")
            raw_role = fields["role"].payload.load()
            if not isinstance(raw_role, str):
                raise RuntimeError(f"diffusion component role must be a string, got {raw_role!r}")
            if any(role == raw_role for role, _digest in selected):
                raise RuntimeError(f"duplicate diffusion component role {raw_role!r}")
            digest = fields["component"].meta.get("digest")
            if not isinstance(digest, str):
                raise RuntimeError(f"diffusion component {raw_role!r} has no asset digest")
            require_digest(digest)
            selected.append((raw_role, digest))
        if len(selected) == 1 and selected[0][0] in _MINIMAX_H3_DIT_ROLES:
            raw_role, digest = selected[0]
            role = cast("MiniMaxH3DiTRole", raw_role)
            try:
                planned = await asyncio.to_thread(
                    self._probe_minimax_h3_dit_transaction,
                    digest,
                    role,
                )
            except _LocatedProbeError as wrapped:
                raise wrapped.error from None
            if planned is None:
                raise RuntimeError(f"diffusion component {role!r} is not a MiniMax H3 DiT")
            if self._minimax_h3_runtime_versions is None:
                raise RuntimeError("MiniMax H3 runtime versions are unavailable")
            from dinkster_inference.component_catalog import default_component_registry

            descriptor = default_component_registry().get("dinkster.minimax_h3")
            assert descriptor is not None
            dtype = resolve_dtype_policy(
                descriptor.id, self._dtype_policy(), self._compute_dtypes()
            )[0]
            try:
                identity = descriptor.component_identity(
                    descriptor.model_role,
                    planned,
                    dtype.name,
                    attention_policy=native_attention[0],
                    attention_route_token=native_attention[1],
                    runtime_versions=self._minimax_h3_runtime_versions(),
                )
            except ValueError as error:
                raise RuntimeError(f"diffusion component {role!r}: {error}") from error
            return ExecutionSelection(
                target=native_arm,
                cache_tag=identity,
                diffusion_dtype=dtype.name,
                text_dtype="unloaded",
                vae_dtype="unloaded",
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            )
        by_role: dict[Trellis2FlowRole, ComponentPlan[Trellis2FlowConfig]] = {}
        for raw_role, digest in selected:
            if raw_role not in _TRELLIS2_SPLIT_FLOW_ROLES:
                raise RuntimeError(f"unsupported diffusion component role {raw_role!r}")
            role = cast("Trellis2FlowRole", raw_role)
            try:
                planned = await asyncio.to_thread(
                    self._probe_trellis2_flow_transaction, digest, role
                )
            except _LocatedProbeError as wrapped:
                raise wrapped.error from None
            if planned is None:
                raise RuntimeError(f"diffusion component {role!r} is not an exact TRELLIS.2 flow")
            by_role[role] = cast("ComponentPlan[Trellis2FlowConfig]", planned)
        missing = tuple(role for role in _TRELLIS2_SPLIT_FLOW_ROLES if role not in by_role)
        if missing:
            raise RuntimeError("missing diffusion component roles: " + ", ".join(missing))
        plan = Trellis2SplitModelPlan(
            structure=by_role["structure"],
            shape=by_role["shape"],
            shape_512=by_role["shape-512"],
            texture=by_role["texture"],
            texture_512=by_role["texture-512"],
        )
        dtype = resolve_dtype_policy(
            "dinkster.trellis2", self._dtype_policy(), self._compute_dtypes()
        )[0]
        return ExecutionSelection(
            target=native_arm,
            cache_tag=trellis2_split_model_runtime_identity(plan, dtype),
            diffusion_dtype=dtype.name,
            text_dtype="unloaded",
            vae_dtype="unloaded",
            attention_policy=native_attention[0],
            attention_route_token=native_attention[1],
        )

    def _probe_components_transaction(self, digest: str) -> tuple[DetectedComponents, ...]:
        from dinkster_inference.component_catalog import default_component_registry

        registry = default_component_registry()
        key = (digest, registry.ids())
        with self._component_probes_lock:
            cached = self._component_probes.get(key)
        if cached is not None:
            return cached
        candidate = self._locate(digest)
        if candidate is None:
            return ()
        try:
            size = candidate.stat().st_size
            path = verified_local_path(candidate, digest)
            source = load_safetensors_header(path, asset_digest=digest, asset_size=size)
        except MalformedSafetensors:
            matches = ()
        except (AssetIntegrityError, OSError) as error:
            raise _LocatedProbeError(error) from error
        else:
            matches = registry.detect(source, path)
        with self._component_probes_lock:
            self._component_probes[key] = matches
        return matches

    async def _select_diffusion_model(
        self,
        inputs: Mapping[str, Value],
        owner_arm: str,
        native_arm: str | None,
        cache_tags: Mapping[str, str],
        native_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
        owner_attention: tuple[AttentionPolicy, AttentionRouteToken | None],
    ) -> ExecutionSelection:
        model = inputs.get("diffusion_model")
        if model is None:
            raise RuntimeError(
                "dinkster.load_diffusion_model is missing its diffusion_model asset input"
            )
        digest = model.meta.get("digest")
        if not isinstance(digest, str):
            raise RuntimeError(
                "dinkster.load_diffusion_model asset has missing or malformed digest metadata"
            )
        try:
            require_digest(digest)
        except AssetError as exc:
            raise RuntimeError(
                "dinkster.load_diffusion_model asset has missing or malformed digest metadata"
            ) from exc
        if native_arm is None:
            return ExecutionSelection(
                target=owner_arm,
                cache_tag=cache_tags[owner_arm],
                attention_policy=owner_attention[0],
                attention_route_token=owner_attention[1],
            )
        try:
            matches = await asyncio.to_thread(self._probe_components_transaction, digest)
        except _LocatedProbeError as wrapped:
            raise wrapped.error from None
        models = tuple(
            match for match in matches if match.plan_for(match.descriptor.model_role) is not None
        )
        if models:
            from dinkster_inference.component_catalog import default_component_registry

            try:
                descriptor, role, planned = default_component_registry().select_detected(
                    models, "model"
                )
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            dtype = resolve_dtype_policy(
                descriptor.family_for(planned), self._dtype_policy(), self._compute_dtypes()
            )[0]
            runtime_versions = None
            if descriptor.requires_runtime_versions:
                if self._minimax_h3_runtime_versions is None:
                    raise RuntimeError(
                        f"{descriptor.family.display_name} runtime versions are unavailable"
                    )
                runtime_versions = self._minimax_h3_runtime_versions()
            return ExecutionSelection(
                target=native_arm,
                cache_tag=descriptor.component_identity(
                    role,
                    planned,
                    dtype.name,
                    attention_policy=native_attention[0],
                    attention_route_token=native_attention[1],
                    runtime_versions=runtime_versions,
                ),
                diffusion_dtype=dtype.name,
                text_dtype="unloaded",
                vae_dtype="unloaded",
                attention_policy=native_attention[0],
                attention_route_token=native_attention[1],
            )
        if matches:
            detected = ", ".join(
                f"{match.descriptor.family.display_name} {role}"
                for match in matches
                for role, _plan in match.components
            )
            raise RuntimeError(f"dinkster.load_diffusion_model role mismatch: detected {detected}")
        return ExecutionSelection(
            target=owner_arm,
            cache_tag=cache_tags[owner_arm],
            attention_policy=owner_attention[0],
            attention_route_token=owner_attention[1],
        )

    async def _shared_probe(
        self,
        digest: str,
        logical_name: str,
        fp8_matmul: bool,
        extension_behavior_hash: str | None,
        embedding_binding_digest: str | None,
        attention_policy: AttentionPolicy,
        attention_route_token: AttentionRouteToken | None,
    ) -> _ProbeVerdict:
        key = (
            digest,
            logical_name,
            fp8_matmul,
            tuple(sorted(self._dtype_policy().items())),
            extension_behavior_hash,
            embedding_binding_digest,
            attention_policy,
            attention_route_token,
        )
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(
                self._run_probe(
                    digest,
                    logical_name,
                    fp8_matmul,
                    extension_behavior_hash,
                    embedding_binding_digest,
                    attention_policy,
                    attention_route_token,
                )
            )
            self._inflight[key] = task
        return await asyncio.shield(task)

    async def _run_probe(
        self,
        digest: str,
        logical_name: str,
        fp8_matmul: bool,
        extension_behavior_hash: str | None,
        embedding_binding_digest: str | None,
        attention_policy: AttentionPolicy,
        attention_route_token: AttentionRouteToken | None,
    ) -> _ProbeVerdict:
        key = (
            digest,
            logical_name,
            fp8_matmul,
            tuple(sorted(self._dtype_policy().items())),
            extension_behavior_hash,
            embedding_binding_digest,
            attention_policy,
            attention_route_token,
        )
        task = asyncio.current_task()
        try:
            try:
                verdict = await asyncio.to_thread(
                    self._probe_transaction,
                    digest,
                    fp8_matmul,
                    extension_behavior_hash,
                    logical_name,
                    embedding_binding_digest,
                    attention_policy,
                    attention_route_token,
                )
            except _LocatedProbeError as wrapped:
                self._missing.discard(digest)
                raise wrapped.error from None
            if verdict.kind not in ("missing", "convertible"):
                self._missing.discard(digest)
                self._convertible.discard(digest)
            if verdict.terminal:
                self._memo[key] = verdict
                if verdict.kind in ("refused", "unsupported"):
                    self._emit(verdict.kind, digest, verdict.reasons)
            return verdict
        finally:
            if self._inflight.get(key) is task:
                del self._inflight[key]

    def _start_conversion(self, digest: str, path: Path, logical_name: str) -> None:
        if self._schedule_conversion is None or digest in self._conversions:
            return
        task = asyncio.create_task(self._run_conversion(digest, path, logical_name))
        self._conversions[digest] = task

    async def _run_conversion(self, digest: str, path: Path, logical_name: str) -> None:
        task = asyncio.current_task()
        try:
            assert self._schedule_conversion is not None
            try:
                status, reason = await self._schedule_conversion(path, logical_name)
            except Exception:  # noqa: BLE001 - every scheduler failure is retryable transport
                return
            if status == "refused":
                reasons = (reason or "legacy checkpoint conversion refused",)
                verdict = _ProbeVerdict("unsupported", reasons=reasons)
                self._conversion_refusals[digest] = verdict
                self._emit("unsupported", digest, reasons)
            elif status not in ("success", "transport-failure"):
                return
        finally:
            if self._conversions.get(digest) is task:
                del self._conversions[digest]

    def _probe_transaction(
        self,
        digest: str,
        fp8_matmul: bool = False,
        extension_behavior_hash: str | None = None,
        logical_name: str = "",
        embedding_binding_digest: str | None = None,
        attention_policy: AttentionPolicy = "auto",
        attention_route_token: AttentionRouteToken | None = None,
    ) -> _ProbeVerdict:
        """The complete blocking locate/verify/read/plan/identity transaction."""
        path = self._locate(digest)
        if path is None:
            return _ProbeVerdict(
                "missing", reasons=("checkpoint asset is not locally materialized",)
            )
        try:
            verified = verified_local_path(path, digest)
            try:
                source = load_safetensors_header(
                    verified,
                    asset_digest=digest,
                    asset_size=verified.stat().st_size,
                )
            except MalformedSafetensors as exc:
                classification = classify_weight_source(verified, logical_name)
                if classification != "legacy-convertible":
                    return _ProbeVerdict("unsupported", reasons=(str(exc),))
                sidecar = discover_converted_sidecar(verified)
                if sidecar is None:
                    return _ProbeVerdict(
                        "convertible",
                        reasons=("legacy checkpoint is awaiting safe conversion",),
                        source_path=verified,
                    )
                source = load_safetensors_header(sidecar)
            try:
                plan = plan_native(source)
            except NativeRefusalError as exc:
                return _ProbeVerdict("refused", reasons=exc.reasons)
            if fp8_matmul and any(
                dtype.name == "float8_e5m2"
                for component in plan.identity_components
                if component is not None
                for dtype in component.dtypes.values()
            ):
                return _ProbeVerdict(
                    "refused",
                    reasons=("fp8 matmul does not support float8_e5m2 checkpoint storage",),
                )
            cache_tag = build_runtime_identity(
                plan.family.id,
                plan.identity_components,
                diffusion_dtype=(
                    dtypes := resolve_dtype_policy(
                        plan.family.id, self._dtype_policy(), self._compute_dtypes()
                    )
                )[0],
                text_dtype=dtypes[1],
                vae_dtype=dtypes[2],
                fp8_matmul=fp8_matmul,
                registry_token=(
                    f"sha256:{extension_behavior_hash}"
                    if extension_behavior_hash is not None
                    else None
                ),
                extension_behavior_hash=extension_behavior_hash,
                embedding_binding_digest=(
                    embedding_binding_digest if uses_classic_embedding_bindings(plan) else None
                ),
                attention_policy=attention_policy,
                attention_route_token=attention_route_token,
            )
            return _ProbeVerdict(
                "native",
                cache_tag=cache_tag,
                fp8_matmul=fp8_matmul,
                diffusion_dtype=dtypes[0].name,
                text_dtype=dtypes[1].name,
                vae_dtype=dtypes[2].name,
            )
        except (AssetIntegrityError, OSError) as exc:
            raise _LocatedProbeError(exc) from exc

    def _emit(
        self,
        kind: Literal["convertible", "missing", "refused", "unsupported"],
        digest: str,
        reasons: tuple[str, ...],
    ) -> None:
        self._on_diagnostic(NativePolicyDiagnostic(kind, digest, reasons))


__all__ = ["NativeDispatchPolicy", "NativePolicyDiagnostic", "resolve_dtype_policy"]
