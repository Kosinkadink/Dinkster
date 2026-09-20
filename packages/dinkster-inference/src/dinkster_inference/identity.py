"""Torch-free native runtime cache identity."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Collection, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from dinkster_protocol import (
    AttentionPolicy,
    AttentionRouteToken,
    attention_route_token_to_wire,
    resolve_attention_runtime_status,
)

from .assembly import ComponentPlan
from .devices import BFLOAT16, FLOAT16, FLOAT32, DType
from .quantization import SUPPORTED_QUANT_FORMATS
from .weights import LinearToConv2D

if TYPE_CHECKING:
    from .registries import InferenceRegistries


def runtime_component_identity(
    family_id: str, components: tuple[ComponentPlan[Any] | None, ...]
) -> tuple[str, ...]:
    """A plan's behavior-affecting facts, canonically ordered.

    Structural identity, not weight identity: model keys, dtypes,
    quantization, planned tensor transforms, defaulted-absent keys,
    and each component's config - the facts that pick the modules
    and their math. Paths and source-key spellings are excluded on
    purpose (quant artifact names included - a combined checkpoint
    prefixes them, a split file does not, and both execute
    identically): the same components read from a moved file or a
    split-vs-combined layout must hash identically, and the
    dispatcher hashes weight content separately (asset identity).
    Payload-borne quant facts (the ``.comfy_quant`` spelling's
    format / full_precision_matrix_mult, which live in tensor bytes
    the header cannot see) ride that asset identity: their line says
    ``format=payload``. Transform lines appear only when a component
    plans transforms, so pre-transform identity strings (all Flux
    plans) are unchanged. ``None`` components (text-encoder slots
    the family does not wire) contribute nothing - the family id
    already pins the wired slot set."""
    lines = [f"family={family_id}"]
    for component in components:
        if component is None:
            continue
        lines.append(f"component={component.component}")
        lines.append(f"config={component.config!r}")
        for key in sorted(component.keys):
            lines.append(f"key={key} dtype={component.dtypes[key].name}")
        for layer in sorted(component.quant):
            quant = component.quant[layer]
            line = (
                f"quant={layer}"
                f" format={quant.format or 'payload'}"
                f" input_scale={quant.input_scale is not None}"
                f" full_precision_matmul={quant.full_precision_matmul}"
            )
            if quant.format == "nvfp4":
                line += (
                    f" weight_scale_2={quant.weight_scale_2 is not None}"
                    f" pre_quant_scale={quant.pre_quant_scale is not None}"
                )
            elif quant.format == "int8_tensorwise":
                parameters = ",".join(
                    f"{name}={quant.parameters[name]!r}" for name in sorted(quant.parameters)
                )
                line += f" parameters={parameters}"
            elif quant.format is not None and quant.format not in SUPPORTED_QUANT_FORMATS:
                parameters = ",".join(
                    f"{name}={quant.parameters[name]!r}" for name in sorted(quant.parameters)
                )
                payloads = ",".join(sorted(quant.payloads))
                line += (
                    f" logical_shape={quant.logical_shape!r}"
                    f" parameters={parameters} payloads={payloads}"
                )
                line += " executable=False"
            lines.append(line)
        for key in sorted(component.transforms):
            transform = component.transforms[key]
            # Diffusers stores these same semantic 1x1 conv weights as
            # rank-2 linears. Source layout is excluded from structural
            # identity exactly like source key spelling; asset identity
            # separately addresses the bytes.
            if not isinstance(transform, LinearToConv2D):
                lines.append(f"transform={key} {transform!r}")
        for key in sorted(component.absent):
            lines.append(f"absent={key}")
        for fact in component.identity_facts:
            lines.append(f"fact={fact}")
    return tuple(lines)


def _require_sha256(name: str, digest: str) -> None:
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a lowercase sha256 hex digest")


def extend_runtime_identity(base_identity: str, runtime_facts: Sequence[str]) -> str:
    """Derive an execution-topology identity from an artifact runtime identity."""

    parts = base_identity.split(":")
    if len(parts) != 3 or parts[0] != "native" or not parts[1]:
        raise ValueError("base runtime identity is malformed")
    _require_sha256("base runtime identity digest", parts[2])
    facts = tuple(runtime_facts)
    if not facts or any(not fact for fact in facts):
        raise ValueError("runtime identity facts must be non-empty strings")
    hasher = hashlib.sha256()
    hasher.update(base_identity.encode())
    hasher.update(b"\n")
    for fact in facts:
        hasher.update(fact.encode())
        hasher.update(b"\n")
    return ":".join((*parts[:2], hasher.hexdigest()))


def build_runtime_identity_from_facts(
    family_id: str,
    component_identity: Sequence[str],
    *,
    diffusion_dtype: str,
    text_dtype: str,
    vae_dtype: str,
    fp8_matmul: bool,
    registry_token: str | None = None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    dependency_facts: Sequence[tuple[str, str, str, str, str, str]] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    runtime_facts: Sequence[str] = (),
) -> str:
    """Build identity from the path-free facts retained by a recipe."""
    component_lines = tuple(component_identity)
    if not component_lines or component_lines[0] != f"family={family_id}":
        raise ValueError("component identity must begin with the selected family")
    hasher = hashlib.sha256()
    knobs = [
        f"diffusion_dtype={diffusion_dtype}",
        f"text_dtype={text_dtype}",
        f"vae_dtype={vae_dtype}",
        f"fp8_matmul={fp8_matmul}",
        f"registries={registry_token if registry_token is not None else 'builtin'}",
    ]
    attention = resolve_attention_runtime_status(attention_policy, attention_route_token)
    if attention.authenticated:
        assert attention_route_token is not None
        encoded_token = json.dumps(
            attention_route_token_to_wire(attention_route_token),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        knobs.extend(
            (
                f"attention_authenticated={attention.authenticated}",
                f"attention_route_token={encoded_token}",
                f"attention_policy={attention.requested_policy}",
                f"attention_adapter={attention.adapter_contract_revision}",
                f"attention_device={attention.device_kind}",
                f"attention_sm={attention.device_sm}",
                f"attention_sdpa_torch={attention.sdpa_torch_runtime}",
            )
        )
        knobs.extend(
            f"attention_provider={name}:{version}" for name, version in attention.provider_versions
        )
        knobs.extend(
            f"attention_route={route.role}:{route.primary}:{route.fallback or '-'}"
            for route in attention.routes
        )
    if extension_behavior_hash is not None:
        _require_sha256("extension_behavior_hash", extension_behavior_hash)
        knobs.append(f"extensions={extension_behavior_hash}")
    if embedding_binding_digest is not None:
        _require_sha256("embedding_binding_digest", embedding_binding_digest)
        knobs.append(f"embeddings={embedding_binding_digest}")
    if patch_overlay_digests:
        for index, digest in enumerate(patch_overlay_digests):
            _require_sha256("patch overlay digest", digest)
            knobs.append(f"patch_overlay[{index}]={digest}")
    if dependency_facts:
        for index, (
            child_id,
            residency_group,
            scope,
            clone_mode,
            accounting_owner,
            child_runtime_identity,
        ) in enumerate(dependency_facts):
            knobs.append(
                f"dependency[{index}] child_id={child_id}"
                f" residency_group={residency_group} scope={scope}"
                f" clone_mode={clone_mode} accounting_owner={accounting_owner}"
                f" child_runtime_identity={child_runtime_identity}"
            )
    for line in knobs + list(component_lines):
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    identity = f"native:{family_id}:{hasher.hexdigest()}"
    return extend_runtime_identity(identity, runtime_facts) if runtime_facts else identity


def build_runtime_identity(
    family_id: str,
    components: tuple[ComponentPlan[Any] | None, ...],
    *,
    diffusion_dtype: DType,
    text_dtype: DType,
    vae_dtype: DType,
    fp8_matmul: bool,
    registry_token: str | None = None,
    extension_behavior_hash: str | None = None,
    patch_overlay_digests: Sequence[str] | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    runtime_facts: Sequence[str] = (),
) -> str:
    """Build the native cache tag from a plan and execution knobs.

    The optional extension behavior hash is appended only when supplied, so
    callers without an extension snapshot retain the pre-S0 identity bytes.
    """
    component_runtime_facts = tuple(
        fact
        for component in components
        if component is not None
        for fact in component.runtime_facts
    )
    return build_runtime_identity_from_facts(
        family_id,
        runtime_component_identity(family_id, components),
        diffusion_dtype=diffusion_dtype.name,
        text_dtype=text_dtype.name,
        vae_dtype=vae_dtype.name,
        fp8_matmul=fp8_matmul,
        registry_token=registry_token,
        extension_behavior_hash=extension_behavior_hash,
        patch_overlay_digests=patch_overlay_digests,
        embedding_binding_digest=embedding_binding_digest,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
        runtime_facts=(*component_runtime_facts, *runtime_facts),
    )


@lru_cache(maxsize=8)
def _cached_default_inference_registries(
    _component_factory: Callable[[], object],
) -> InferenceRegistries:
    """Cache defaults until the component registry provider changes."""
    from .registries import builtin_registries

    return builtin_registries()


def _default_inference_registries() -> InferenceRegistries:
    from . import component_catalog

    return _cached_default_inference_registries(component_catalog.default_component_registry)


def _report_dtype_default(family_id: str, component: str, dtype: DType) -> DType:
    if _default_inference_registries().families.get(family_id) is None:
        logging.getLogger(__name__).warning(
            "Checkpoint label %r has no %s dtype specialization; defaulting to %s",
            family_id,
            component,
            dtype.name,
        )
    return dtype


def default_diffusion_dtype(family_id: str) -> DType:
    """Return the family's native diffusion default.

    The execution backend resolves text and VAE defaults separately; a
    dispatch host computing the cache tag must mirror all resolved defaults.
    """
    registries = _default_inference_registries()
    descriptor = registries.components.get(family_id)
    if descriptor is not None:
        return descriptor.default_diffusion_dtype
    family = registries.families.get(family_id)
    if family is not None:
        return family.engine.diffusion_dtype
    return _report_dtype_default(family_id, "diffusion", BFLOAT16)


def default_text_dtype(family_id: str) -> DType:
    """Return the family's reference text-encoder compute dtype.

    Architecture registrations own component defaults. Classic checkpoint
    defaults follow their reference text tower.
    """
    registries = _default_inference_registries()
    descriptor = registries.components.get(family_id)
    if descriptor is not None:
        return descriptor.default_text_dtype
    family = registries.families.get(family_id)
    if family is not None:
        return family.engine.text_dtype
    return _report_dtype_default(family_id, "text", BFLOAT16)


def default_vae_dtype(
    family_id: str,
    compute_dtypes: Collection[DType] = (FLOAT16, BFLOAT16, FLOAT32),
) -> DType:
    """Select the family's first device-supported VAE compute dtype.

    The preference lists mirror ComfyUI's VAE ``working_dtypes`` at the
    pinned revision. SD-era and other KL/LTX VAEs intentionally omit
    float16 because their activations can overflow to black or non-finite
    output.
    """
    registries = _default_inference_registries()
    descriptor = registries.components.get(family_id)
    family = None
    if descriptor is not None:
        preferences = descriptor.vae_dtypes
    else:
        family = registries.families.get(family_id)
        preferences = family.engine.vae_dtypes if family is not None else (BFLOAT16, FLOAT32)
    try:
        dtype = next(dtype for dtype in preferences if dtype in compute_dtypes)
    except StopIteration:
        names = ", ".join(sorted(dtype.name for dtype in compute_dtypes)) or "none"
        raise ValueError(
            f"family {family_id!r} has no VAE dtype supported by device set: {names}"
        ) from None
    return (
        dtype
        if descriptor is not None or family is not None
        else _report_dtype_default(family_id, "VAE", dtype)
    )


__all__ = [
    "build_runtime_identity",
    "build_runtime_identity_from_facts",
    "default_diffusion_dtype",
    "default_text_dtype",
    "default_vae_dtype",
    "extend_runtime_identity",
    "runtime_component_identity",
]
