"""Process-owned attention routing for fork-backed sampling."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import torch
from dinkster_protocol import (
    ATTENTION_POLICIES,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    derive_attention_route_token,
)

_ADAPTER_CONTRACT = "dinkster.attention-kernel.v1"
_POLICY_FUNCTIONS = {
    "sdpa": "pytorch",
    "flash": "flash",
    "xformers": "xformers",
    "sage": "sage",
    "sage3": "sage3",
    "dinkster_kitchen_int8": "comfy_kitchen_int8",
}
_PROVIDER_DISTRIBUTIONS = {
    "dinkster_kitchen_int8": ("dinkster-kitchen", "comfy-kitchen"),
    "flash": ("flash-attn", "flash-attn"),
    "sage": ("sageattention", "sageattention"),
    "sage3": ("sageattention", "sageattention"),
    "xformers": ("xformers", "xformers"),
}


@dataclass(frozen=True, slots=True)
class _DistributedConfig:
    rank: int
    world_size: int
    mode: str
    rendezvous: str
    token: str


_attempt_lock = threading.Lock()
_active_attempt: str | None = None
_process_group_config: _DistributedConfig | None = None
_attention_call_index = 0


def _package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


class AttentionRuntime:
    """Own one worker process's named fork attention registry."""

    def __init__(self) -> None:
        attention = cast("Any", importlib.import_module("dinkster_comfy.ldm.modules.attention"))
        self.registry = cast(
            "dict[str, Callable[..., Any]]", attention.create_attention_function_registry()
        )
        available = tuple(
            policy
            for policy in ATTENTION_POLICIES
            if policy != "auto"
            and policy in _POLICY_FUNCTIONS
            and attention.get_attention_function(
                _POLICY_FUNCTIONS[policy], None, registry=self.registry
            )
            is not None
        )
        providers: dict[str, str] = {"torch": torch.__version__}
        for policy in available:
            provider = _PROVIDER_DISTRIBUTIONS.get(policy)
            if provider is not None:
                providers[provider[0]] = _package_version(provider[1])
        if torch.version.hip is not None:
            device_kind = "rocm"
            device_sm = None
            providers["hip"] = torch.version.hip
        elif torch.cuda.is_available():
            device_kind = "cuda"
            major, minor = torch.cuda.get_device_capability()
            device_sm = major * 10 + minor
        else:
            device_kind = "cpu"
            device_sm = None
        self.capabilities = AttentionCapabilityEvidence(
            version=1,
            device_kind=device_kind,
            device_sm=device_sm,
            sdpa_torch_runtime=torch.__version__.split("+")[0],
            adapter_contract_revision=_ADAPTER_CONTRACT,
            available_policies=cast("tuple[Any, ...]", available),
            provider_versions=tuple(sorted(providers.items())),
        )
        self.route_token = derive_attention_route_token(self.capabilities, AttentionPolicyConfig())

    def resolve(self, token: AttentionRouteToken, role: str = "unet") -> Callable[..., Any]:
        attention = cast("Any", importlib.import_module("dinkster_comfy.ldm.modules.attention"))
        route = next(route for route in token.routes if route.role == role)
        for policy in (route.primary, route.fallback):
            if policy is None:
                continue
            name = _POLICY_FUNCTIONS.get(policy)
            selected = (
                None
                if name is None
                else attention.get_attention_function(name, None, registry=self.registry)
            )
            if selected is not None:
                return cast("Callable[..., Any]", selected)
        raise RuntimeError(f"worker attention route for {role!r} is unavailable")

    def for_model(self, token: AttentionRouteToken) -> Callable[..., Any]:
        selected = self.resolve(token)
        return _DistributedAttention(selected) if _active_attempt is not None else selected


def create_attention_runtime() -> AttentionRuntime:
    return AttentionRuntime()


def activate_distributed_attention(group: str, attempt: int) -> None:
    global _active_attempt, _attention_call_index
    if not group or type(attempt) is not int or attempt <= 0:
        raise ValueError("distributed attention attempt correlation is invalid")
    correlation = f"{group}:{attempt}"
    with _attempt_lock:
        if _active_attempt is not None:
            raise RuntimeError("distributed attention attempt is already active")
        _active_attempt = correlation
        _attention_call_index = 0


def release_distributed_attention(group: str, attempt: int) -> None:
    global _active_attempt, _process_group_config
    correlation = f"{group}:{attempt}"
    with _attempt_lock:
        if _active_attempt != correlation:
            raise RuntimeError("distributed attention release has foreign correlation")
        config = _process_group_config
        try:
            if config is not None and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        finally:
            _process_group_config = None
            _active_attempt = None
    if config is not None and config.rank == 0:
        Path(config.rendezvous.removeprefix("file://")).unlink(missing_ok=True)


def _distributed_config() -> _DistributedConfig:
    if _active_attempt is None:
        raise RuntimeError("distributed attention has no active workgroup attempt")
    try:
        rank = int(os.environ["DINKSTER_SINGLE_JOB_RANK"])
        world_size = int(os.environ["DINKSTER_SINGLE_JOB_WORLD_SIZE"])
        mode = os.environ["DINKSTER_SINGLE_JOB_MULTI_GPU_MODE"]
        rendezvous_base = os.environ["DINKSTER_SINGLE_JOB_RENDEZVOUS"]
        token_base = os.environ["DINKSTER_SINGLE_JOB_TOKEN"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError("single-job rank environment is malformed") from exc
    if (
        world_size < 2
        or not 0 <= rank < world_size
        or mode not in ("auto", "sequence")
        or not rendezvous_base.startswith("file://")
        or len(token_base) != 32
    ):
        raise RuntimeError("single-job distributed attention configuration is unsupported")
    digest = hashlib.sha256(f"{token_base}:{_active_attempt}".encode()).hexdigest()
    return _DistributedConfig(
        rank,
        world_size,
        mode,
        f"{rendezvous_base}.{digest}",
        digest[:32],
    )


def _ensure_process_group() -> _DistributedConfig:
    global _process_group_config
    config = _distributed_config()
    with _attempt_lock:
        if _process_group_config is not None:
            if _process_group_config != config:
                raise RuntimeError("distributed attention process-group configuration changed")
            return config
        if not torch.cuda.is_available():
            raise RuntimeError("single-job distributed attention requires CUDA")
        if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
            raise RuntimeError("single-job distributed attention requires torch.distributed NCCL")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            "nccl",
            init_method=config.rendezvous,
            rank=config.rank,
            world_size=config.world_size,
            timeout=timedelta(minutes=5),
            group_name=config.token,
        )
        _process_group_config = config
    return config


def _slice_heads(tensor: torch.Tensor, heads: int, start: int, stop: int) -> torch.Tensor:
    if tensor.ndim == 4:
        if tensor.shape[1] != heads:
            raise RuntimeError("distributed attention requires equal query, key, and value heads")
        return tensor[:, start:stop]
    if tensor.ndim != 3 or tensor.shape[-1] % heads:
        raise RuntimeError("distributed attention received malformed flattened heads")
    width = tensor.shape[-1] // heads
    return tensor[..., start * width : stop * width]


def _slice_mask(
    mask: torch.Tensor | None, heads: int, start: int, stop: int
) -> torch.Tensor | None:
    if mask is None or mask.ndim < 4 or mask.shape[1] == 1:
        return mask
    if mask.shape[1] != heads:
        raise RuntimeError("distributed attention mask has incompatible heads")
    return mask[:, start:stop]


def _fence_call(config: _DistributedConfig, q: torch.Tensor, heads: int) -> None:
    global _attention_call_index
    control = torch.tensor(
        (_attention_call_index, heads, q.ndim, *q.shape),
        dtype=torch.int64,
        device=q.device,
    )
    peers = [torch.empty_like(control) for _ in range(config.world_size)]
    torch.distributed.all_gather(peers, control)
    if any(not torch.equal(peer, control) for peer in peers):
        raise RuntimeError("distributed attention ranks reached different calls")
    _attention_call_index += 1


class _DistributedAttention:
    def __init__(self, selected: Callable[..., Any]) -> None:
        self.selected = selected
        self.container_function = self._containers

    def _containers(self, q: Any, k: Any, v: Any, *args: Any, **kwargs: Any) -> Any:
        return self(q.take(), k.take(), v.take(), *args, **kwargs)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
        attn_precision: torch.dtype | None = None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        config = _ensure_process_group()
        if kwargs.get("enable_gqa", False):
            raise RuntimeError("distributed attention does not support grouped-query attention")
        if heads % config.world_size:
            raise RuntimeError("attention heads must divide evenly across single-job ranks")
        _fence_call(config, q, heads)
        local_heads = heads // config.world_size
        start = config.rank * local_heads
        stop = start + local_heads
        local_q = _slice_heads(q, heads, start, stop)
        local_k = _slice_heads(k, heads, start, stop)
        local_v = _slice_heads(v, heads, start, stop)
        local_mask = _slice_mask(mask, heads, start, stop)
        failure: BaseException | None = None
        local_output: torch.Tensor | None = None
        try:
            local_output = self.selected(
                local_q,
                local_k,
                local_v,
                local_heads,
                mask=local_mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        except BaseException as exc:
            failure = exc
        failed = torch.tensor(int(failure is not None), dtype=torch.int32, device=q.device)
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failure is not None:
            raise failure
        if bool(failed.item()):
            raise RuntimeError("peer distributed attention rank failed")
        if not isinstance(local_output, torch.Tensor):
            raise RuntimeError("attention provider returned a non-tensor output")
        outputs = [torch.empty_like(local_output) for _ in range(config.world_size)]
        torch.distributed.all_gather(outputs, local_output.contiguous())
        dimension = 1 if skip_output_reshape else -1
        return torch.cat(outputs, dim=dimension)


__all__ = [
    "AttentionRuntime",
    "activate_distributed_attention",
    "create_attention_runtime",
    "release_distributed_attention",
]
