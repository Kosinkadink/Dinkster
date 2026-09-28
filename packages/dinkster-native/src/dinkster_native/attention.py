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

from dinkster_protocol import (
    ATTENTION_POLICIES,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    derive_attention_route_token,
)
from dinkster_workers import current_execution_context

torch = cast("Any", importlib.import_module("torch"))

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
        attention = cast("Any", importlib.import_module("dinkster_inference.ldm.modules.attention"))
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
        attention = cast("Any", importlib.import_module("dinkster_inference.ldm.modules.attention"))
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
        if _active_attempt is not None:
            config = _distributed_config()
            if config.mode in ("auto", "sequence"):
                return _UlyssesAttention(selected)
        return selected

    def configure_sparse_model(self, model: Any, config: dict[str, Any]) -> None:
        if _active_attempt is not None and _distributed_config().mode in ("auto", "sequence"):
            raise RuntimeError("Ulysses sequence mode does not support sparse attention")
        backend = self.registry["comfy_kitchen_sol_chunked"]
        sparse = _SparseH3Attention(backend, config)
        minimax = importlib.import_module("dinkster_comfy.ldm.minimax.model")
        diffusion_model = model.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, minimax.MiniMaxH3Model):
            raise RuntimeError("fork sparse attention requires a MiniMax H3 model")
        for block_index, block in enumerate(diffusion_model.blocks):
            model.set_model_patch_replace(
                sparse.block_patch(block, block_index), "dit", "double_block", block_index
            )

    def distributed_active(self) -> bool:
        return _active_attempt is not None


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
        context = current_execution_context()
        execution = None if context is None else context.single_job_multi_gpu_execution
        rank = (
            execution.rank if execution is not None else int(os.environ["DINKSTER_SINGLE_JOB_RANK"])
        )
        world_size = (
            execution.world_size
            if execution is not None
            else int(os.environ["DINKSTER_SINGLE_JOB_WORLD_SIZE"])
        )
        mode = (
            execution.mode
            if execution is not None
            else os.environ["DINKSTER_SINGLE_JOB_MULTI_GPU_MODE"]
        )
        rendezvous_base = os.environ["DINKSTER_SINGLE_JOB_RENDEZVOUS"]
        token_base = os.environ["DINKSTER_SINGLE_JOB_TOKEN"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError("single-job rank environment is malformed") from exc
    if (
        world_size < 2
        or not 0 <= rank < world_size
        or mode not in ("auto", "guidance", "sequence", "window")
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


def _fence_call(config: _DistributedConfig, q: Any, heads: int) -> None:
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


def _head_to_sequence(tensor: Any, config: _DistributedConfig) -> Any:
    local_heads = tensor.shape[1] // config.world_size
    outgoing = [
        tensor[:, rank * local_heads : (rank + 1) * local_heads].contiguous()
        for rank in range(config.world_size)
    ]
    incoming = [torch.empty_like(outgoing[0]) for _ in range(config.world_size)]
    torch.distributed.all_to_all(incoming, outgoing)
    return torch.cat(incoming, dim=2)


def _sequence_to_head(tensor: Any, config: _DistributedConfig) -> Any:
    if tensor.shape[2] % config.world_size:
        raise RuntimeError("attention sequence must divide evenly across single-job ranks")
    outgoing = [value.contiguous() for value in tensor.chunk(config.world_size, dim=2)]
    incoming = [torch.empty_like(outgoing[0]) for _ in range(config.world_size)]
    torch.distributed.all_to_all(incoming, outgoing)
    return torch.cat(incoming, dim=1)


class _UlyssesAttention:
    def __init__(self, selected: Callable[..., Any]) -> None:
        self.selected = selected
        self.container_function = self._containers

    def _containers(self, q: Any, k: Any, v: Any, *args: Any, **kwargs: Any) -> Any:
        return self(q.take(), k.take(), v.take(), *args, **kwargs)

    def __call__(
        self,
        q: Any,
        k: Any,
        v: Any,
        heads: int,
        mask: Any | None = None,
        attn_precision: Any | None = None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
        **kwargs: Any,
    ) -> Any:
        transformer_options = kwargs.get("transformer_options", {})
        if not transformer_options.get("dinkster_sequence_sharded", False):
            return self.selected(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        config = _ensure_process_group()
        if config.world_size != 2:
            raise RuntimeError("Ulysses sequence mode supports exactly two ranks")
        if not skip_reshape or q.ndim != 4:
            raise RuntimeError("Ulysses sequence mode requires separated attention heads")
        if mask is not None:
            raise RuntimeError("Ulysses sequence mode does not support an attention mask")
        if kwargs.get("enable_gqa", False):
            raise RuntimeError("Ulysses sequence mode does not support grouped-query attention")
        if heads % config.world_size:
            raise RuntimeError("attention heads must divide evenly across single-job ranks")
        valid_sequence = transformer_options.get("dinkster_sequence_valid")
        if (
            type(valid_sequence) is not int
            or not 0 < valid_sequence <= q.shape[2] * config.world_size
        ):
            raise RuntimeError("Ulysses sequence metadata is invalid")
        _fence_call(config, q, heads)
        local_heads = heads // config.world_size
        failure: BaseException | None = None
        local_output: Any | None = None
        try:
            local_q = _head_to_sequence(q, config)
            local_k = _head_to_sequence(k, config)
            local_v = _head_to_sequence(v, config)
            key_mask = None
            if valid_sequence < local_k.shape[2]:
                key_mask = torch.arange(local_k.shape[2], device=q.device) < valid_sequence
                key_mask = key_mask.reshape(1, 1, -1)
            output = self.selected(
                local_q,
                local_k,
                local_v,
                local_heads,
                mask=key_mask,
                attn_precision=attn_precision,
                skip_reshape=True,
                skip_output_reshape=True,
                **kwargs,
            )
            if (
                not isinstance(output, torch.Tensor)
                or output.ndim != 4
                or output.shape[1] != local_heads
            ):
                raise RuntimeError("attention provider returned malformed Ulysses output")
            local_output = _sequence_to_head(output, config)
        except BaseException as exc:
            failure = exc
        failed = torch.tensor(int(failure is not None), dtype=torch.int32, device=q.device)
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failure is not None:
            raise failure
        if bool(failed.item()):
            raise RuntimeError("peer distributed attention rank failed")
        assert local_output is not None
        if skip_output_reshape:
            return local_output
        return local_output.transpose(1, 2).reshape(
            local_output.shape[0], local_output.shape[2], heads * local_output.shape[3]
        )


class _SparseH3Attention:
    def __init__(
        self,
        backend: Callable[..., Any],
        config: dict[str, Any],
    ) -> None:
        self.backend = backend
        self.config = config
        self.pooled: dict[tuple[int, int, tuple[object, ...]], tuple[Any, Any]] = {}

    def _eligible(
        self, attention: Any, hidden: Any, rope_freqs: Any, options: dict[str, Any], block: int
    ) -> bool:
        layout = options.get("minimax_h3_layout")
        if layout is None or rope_freqs is None:
            return False
        if block in self.config["dense_blocks"] or hidden.shape[0] < self.config["min_tokens"]:
            return False
        if hidden.device.type != "cuda" or hidden.dtype != torch.bfloat16:
            return False
        kitchen = importlib.import_module("comfy_kitchen")
        if not kitchen.sol_attn_is_available(hidden.device):
            return False
        current = options.get("sigmas")
        if current is None:
            return False
        sigma = float(current.flatten()[0])
        return (
            attention.head_dim == 128
            and self.config["sigma_start"] >= sigma >= self.config["sigma_end"]
        )

    def _attention(
        self, attention: Any, hidden: Any, rope_freqs: Any, options: dict[str, Any], block: int
    ) -> Any:
        layout = options["minimax_h3_layout"]
        video_start = next(start for start, _stop, kind in layout.segments if kind == "video")
        sink_blocks = [0, (video_start + 63) // 64]
        sink_queries = [0, 0]
        if self.config["sink_conditioning"] == "off":
            sink_blocks = [0, 0]
        elif self.config["sink_conditioning"] == "exact_kv_and_rows":
            audio_start = next(start for start, _stop, kind in layout.segments if kind == "audio")
            sink_queries = [audio_start // 64, sink_blocks[1]]
        pool_key = (block, hidden.shape[0], tuple(options.get("uuids", ())))
        pooled = self.pooled.get(pool_key)
        output, key_mean, value_scale = self.backend(
            hidden,
            attention.qkv_proj,
            attention.out_proj,
            attention.q_norm,
            attention.k_norm,
            attention.heads,
            rope_freqs,
            kmean=None if pooled is None else pooled[0],
            vscale=None if pooled is None else pooled[1],
            tau=self.config["tau"],
            topk_ratio=self.config["keep_percent"] / 100.0,
            sink_blocks=sink_blocks,
            sink_q=sink_queries,
            token_aug=self.config["extra_tokens"],
        )
        self.pooled[pool_key] = (key_mean, value_scale)
        return output

    def block_patch(self, block: Any, block_index: int) -> Callable[..., Any]:
        def attention(
            hidden: Any,
            rope_freqs: Any = None,
            transformer_options: dict[str, Any] | None = None,
        ) -> Any:
            if transformer_options is None:
                transformer_options = {}
            return self._attention(block.attn, hidden, rope_freqs, transformer_options, block_index)

        def patch(args: dict[str, Any], extra: dict[str, Any]) -> Any:
            if self._eligible(
                block.attn,
                args["img"],
                args["rope_freqs"],
                args["transformer_options"],
                block_index,
            ):
                args = {**args, "attention": attention}
            return extra["original_block"](args)

        return patch


__all__ = [
    "AttentionRuntime",
    "activate_distributed_attention",
    "create_attention_runtime",
    "release_distributed_attention",
]
