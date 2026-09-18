"""Streaming Qwen generation over transactional paged KV sessions."""

from __future__ import annotations

import codecs
import secrets
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self, cast

import torch
from dinkster_inference import (
    ANIMA_QWEN3_06B_CONFIG,
    QWEN3_30B_A3B_CONFIG,
    GenerationEvent,
    GenerationFinishReason,
    GenerationProviderCapabilities,
    GenerationRequest,
    GenerationResult,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSessionBusyError,
    GenerationSessionClosedError,
    GenerationSessionHandle,
    GenerationStats,
    GenerationTerminalEvent,
    GenerationTokenEvent,
)

from .model_prefetch import close_prefetch_queue, make_prefetch_queue
from .operations import bound_compute_device, bound_compute_dtype
from .paged_kv import (
    PagedKVCache,
    PagedKVGeometry,
    PagedKVSessionBusyError,
    PagedKVSessionLease,
    PagedKVSessionMissingError,
)
from .qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel
from .qwen_layer_placement import resolve_qwen_layer_placement
from .qwen_paged_kv import QwenPagedKVCache, QwenPagedKVLease
from .qwen_text import QwenBlock, QwenTextModel
from .residency import ResidencyMechanism

_QwenFrequencies = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_QwenFrequencyCache = _QwenFrequencies | tuple[_QwenFrequencies, ...]

_CAPABILITIES = GenerationProviderCapabilities(
    frozenset(GenerationSamplerKind),
    sessions=True,
    token_ids=True,
    ordered_sampler_chain=True,
)


class QwenGenerationTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode_bytes(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> bytes: ...


@dataclass(frozen=True, slots=True)
class _SessionState:
    token_ids: tuple[int, ...]


class QwenGenerationProvider:
    """Native Qwen provider with explicit session ownership."""

    def __init__(
        self,
        model: QwenTextModel | Qwen3MoeForCausalLM,
        tokenizer: QwenGenerationTokenizer,
        model_identity: str,
        *,
        provider_id: str = "dinkster.qwen",
        eos_token_ids: tuple[int, ...] = (151643, 151645),
        block_tokens: int = 256,
        max_device_blocks: int = 128,
        max_host_blocks: int = 0,
    ) -> None:
        moe_model = (
            model
            if type(model) is Qwen3MoeForCausalLM and model.config == QWEN3_30B_A3B_CONFIG
            else None
        )
        if model.config == ANIMA_QWEN3_06B_CONFIG:
            backbone = cast(QwenTextModel, model)
        elif moe_model is not None:
            backbone = moe_model.model
        else:
            raise ValueError(
                "Qwen generation requires the exact Anima Qwen3-0.6B or Qwen3-30B-A3B profile"
            )
        if type(model_identity) is not str or not model_identity:
            raise ValueError("Qwen generation model identity must be non-empty")
        if type(provider_id) is not str or not provider_id:
            raise ValueError("Qwen generation provider ID must be non-empty")
        if type(eos_token_ids) is not tuple or any(
            type(token_id) is not int or not 0 <= token_id < model.config.vocab_size
            for token_id in eos_token_ids
        ):
            raise ValueError("Qwen EOS token IDs must be a tuple within the model vocabulary")
        if len(eos_token_ids) != len(set(eos_token_ids)):
            raise ValueError("Qwen EOS token IDs must be unique")

        placement = resolve_qwen_layer_placement(cast(QwenTextModel, backbone))
        device = bound_compute_device(backbone.embed_tokens) or backbone.embed_tokens.weight.device
        dtype = bound_compute_dtype(backbone.embed_tokens) or backbone.embed_tokens.weight.dtype
        if device.type == "meta":
            raise ValueError("Qwen generation requires materialized model placement")
        execution_dtypes = {dtype}
        if type(backbone) is QwenTextModel:
            execution_dtypes.update(
                bound_compute_dtype(cast(QwenBlock, layer).input_layernorm)
                or cast(QwenBlock, layer).input_layernorm.weight.dtype
                for layer in backbone.layers
            )
            execution_dtypes.add(bound_compute_dtype(backbone.norm) or backbone.norm.weight.dtype)
        elif moe_model is not None:
            execution_devices: set[torch.device] = set()
            execution_dtypes.clear()
            for module in moe_model.modules():
                tensors = tuple(module.parameters(recurse=False)) + tuple(
                    module.buffers(recurse=False)
                )
                if not tensors:
                    continue
                compute_device = bound_compute_device(module)
                compute_dtype = bound_compute_dtype(module)
                execution_devices.update(compute_device or tensor.device for tensor in tensors)
                execution_dtypes.update(compute_dtype or tensor.dtype for tensor in tensors)
            if execution_devices != {device}:
                raise ValueError("Qwen3 MoE generation requires homogeneous model placement")
        if len(execution_dtypes) != 1:
            raise ValueError("Qwen layer placement requires one homogeneous compute dtype")
        self._model: QwenTextModel | Qwen3MoeModel = backbone
        self._moe_model = moe_model
        self._tokenizer = tokenizer
        self._model_identity = model_identity
        self._id = provider_id
        self._eos_token_ids = frozenset(eos_token_ids)
        self._device = device
        self._dtype = dtype
        self._placement = placement
        self._layer_devices = tuple(
            placement.device_for_layer(index) for index in range(len(backbone.layers))
        )
        cache_id = f"{provider_id}:{model_identity}"
        self._cache = (
            PagedKVCache(
                cache_id,
                model_identity,
                PagedKVGeometry(
                    block_tokens,
                    backbone.config.num_hidden_layers,
                    backbone.config.num_key_value_heads,
                    backbone.config.head_dim,
                    dtype,
                ),
                load_device=device,
                max_device_blocks=max_device_blocks,
                max_host_blocks=max_host_blocks,
            )
            if len(placement.ranges) == 1
            else QwenPagedKVCache(
                cache_id,
                model_identity,
                placement,
                block_tokens=block_tokens,
                kv_heads=backbone.config.num_key_value_heads,
                head_dim=backbone.config.head_dim,
                dtype=dtype,
                max_device_blocks=max_device_blocks,
                max_host_blocks=max_host_blocks,
            )
        )
        self._lock = threading.RLock()
        self._sessions: dict[str, _SessionState] = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> GenerationProviderCapabilities:
        return _CAPABILITIES

    @property
    def cache(self) -> PagedKVCache | QwenPagedKVCache:
        return self._cache

    @property
    def cache_residency_mechanisms(self) -> tuple[ResidencyMechanism, ...]:
        """Per-device KV mechanisms for the shared residency manager."""
        if isinstance(self._cache, PagedKVCache):
            return (self._cache,)
        return self._cache.residency_mechanisms

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _QwenGenerationStream:
        if request.provider_id != self._id:
            raise ValueError("generation request targets a different provider")
        if request.model_identity != self._model_identity:
            raise ValueError("generation request targets a different model identity")
        if request.messages:
            raise ValueError("Qwen generation accepts raw prompts, not chat messages")
        if request.prompt is None:
            raise ValueError("Qwen generation requires a raw prompt")
        prompt_ids = tuple(self._tokenizer.encode(request.prompt))
        if not prompt_ids:
            raise ValueError("Qwen generation requires at least one prompt token")
        if any(
            type(token_id) is not int or not 0 <= token_id < self._model.config.vocab_size
            for token_id in prompt_ids
        ):
            raise ValueError("Qwen tokenizer IDs must be integers within the model vocabulary")
        if any(
            token_id >= self._model.config.vocab_size for token_id in request.stop.stop_token_ids
        ):
            raise ValueError("generation stop token ID is outside the model vocabulary")

        with self._lock:
            provisional = request.open_session
            ephemeral = request.session is None and not provisional
            handle = request.session
            if handle is not None:
                self._validate_handle(handle)
                state = self._sessions.get(handle.session_id)
                if state is None:
                    raise GenerationSessionClosedError("generation session is closed or unknown")
                session_id = handle.session_id
            else:
                session_id = self._create_cache_session()
                state = _SessionState(())
                if provisional:
                    handle = GenerationSessionHandle(
                        self._id,
                        self._model_identity,
                        session_id,
                    )
            try:
                lease = self._cache.pin(session_id)
            except PagedKVSessionBusyError as error:
                raise GenerationSessionBusyError(
                    "generation session already has an active request"
                ) from error
            except PagedKVSessionMissingError as error:
                self._sessions.pop(session_id, None)
                raise GenerationSessionClosedError(
                    "generation session is closed or unknown"
                ) from error

        try:
            return _QwenGenerationStream(
                self,
                request,
                cancelled,
                prompt_ids,
                state,
                session_id,
                lease,
                handle,
                provisional=provisional,
                ephemeral=ephemeral,
            )
        except BaseException:
            with self._lock:
                lease.close()
                if provisional or ephemeral:
                    self._cache.close_session(session_id)
            raise

    def close_session(self, session: GenerationSessionHandle) -> None:
        self._validate_handle(session)
        with self._lock:
            try:
                self._cache.close_session(session.session_id)
            except PagedKVSessionBusyError as error:
                raise GenerationSessionBusyError(
                    "active generation session cannot be closed"
                ) from error
            self._sessions.pop(session.session_id, None)

    def fork_session(
        self,
        session: GenerationSessionHandle,
        *,
        token_count: int | None = None,
    ) -> GenerationSessionHandle:
        self._validate_handle(session)
        with self._lock:
            source = self._sessions.get(session.session_id)
            if source is None:
                raise GenerationSessionClosedError("generation session is closed or unknown")
            target_count = len(source.token_ids) if token_count is None else token_count
            if type(target_count) is not int or not 0 <= target_count <= len(source.token_ids):
                raise ValueError("fork token count must be within the committed session")
            target_id = self._new_session_id()
            try:
                self._cache.fork_session(
                    session.session_id,
                    target_id,
                    token_count=target_count,
                )
            except PagedKVSessionBusyError as error:
                raise GenerationSessionBusyError(
                    "active generation session cannot be forked"
                ) from error
            except PagedKVSessionMissingError as error:
                self._sessions.pop(session.session_id, None)
                raise GenerationSessionClosedError(
                    "generation session is closed or unknown"
                ) from error
            self._sessions[target_id] = _SessionState(source.token_ids[:target_count])
            return GenerationSessionHandle(self._id, self._model_identity, target_id)

    def _finish(self, stream: _QwenGenerationStream, *, commit: bool) -> None:
        with self._lock:
            if commit:
                stream._lease.commit()  # pyright: ignore[reportPrivateUsage]
                if stream._handle is not None:  # pyright: ignore[reportPrivateUsage]
                    self._sessions[stream._session_id] = _SessionState(  # pyright: ignore[reportPrivateUsage]
                        tuple(stream._history)  # pyright: ignore[reportPrivateUsage]
                    )
                return
            stream._lease.close()  # pyright: ignore[reportPrivateUsage]
            if stream._provisional or stream._ephemeral:  # pyright: ignore[reportPrivateUsage]
                self._cache.close_session(stream._session_id)  # pyright: ignore[reportPrivateUsage]

    def _create_cache_session(self) -> str:
        while True:
            session_id = self._new_session_id()
            try:
                self._cache.create_session(session_id)
            except ValueError:
                continue
            return session_id

    def _new_session_id(self) -> str:
        while True:
            session_id = secrets.token_hex(16)
            if session_id not in self._sessions:
                return session_id

    def _validate_handle(self, session: GenerationSessionHandle) -> None:
        if type(session) is not GenerationSessionHandle:
            raise TypeError("generation session must be a GenerationSessionHandle")
        if session.provider_id != self._id or session.model_identity != self._model_identity:
            raise GenerationSessionClosedError("generation session belongs to another provider")

    def _forward_logits(
        self,
        ids: torch.Tensor,
        cache_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        cache_position: int,
        frequencies: _QwenFrequencyCache,
        prefetch: bool,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if self._moe_model is not None:
            hidden, new_key_values = cast(Qwen3MoeModel, self._model).forward_causal(
                ids,
                cache_key_values,
                cache_position=cache_position,
                frequencies=cast(_QwenFrequencies, frequencies),
                prefetch=prefetch,
            )
            logits = self._moe_model.lm_head(hidden[:, -1:])
        else:
            dense = cast(QwenTextModel, self._model)
            hidden, new_key_values = dense.forward_causal(
                ids,
                cache_key_values,
                cache_position=cache_position,
                frequencies=frequencies,
                prefetch=prefetch,
            )
            logits = dense.logits(hidden[:, -1:])
        return logits[:, -1], new_key_values


class _QwenGenerationStream(Iterator[GenerationEvent]):
    def __init__(
        self,
        provider: QwenGenerationProvider,
        request: GenerationRequest,
        cancelled: Callable[[], bool],
        prompt_ids: tuple[int, ...],
        state: _SessionState,
        session_id: str,
        lease: PagedKVSessionLease | QwenPagedKVLease,
        handle: GenerationSessionHandle | None,
        *,
        provisional: bool,
        ephemeral: bool,
    ) -> None:
        base_tokens = lease.token_count
        if base_tokens != len(state.token_ids):
            raise RuntimeError("generation session token history and KV length disagree")
        capacity = provider._model.config.max_position_embeddings  # pyright: ignore[reportPrivateUsage]
        available = capacity - base_tokens - len(prompt_ids)
        if available < 1:
            raise ValueError("Qwen prompt and session leave no generation position available")

        self._provider = provider
        self._request = request
        self._cancelled = cancelled
        self._prompt_ids = prompt_ids
        self._session_id = session_id
        self._lease = lease
        self._handle = handle
        self._provisional = provisional
        self._ephemeral = ephemeral
        self._base_tokens = base_tokens
        self._limit = min(request.stop.max_new_tokens, available)
        self._history = [*state.token_ids, *prompt_ids]
        self._history_counts = Counter(self._history)
        self._generated: list[int] = []
        self._cache_key: torch.Tensor | None = None
        self._cache_value: torch.Tensor | None = None
        self._cache_layers: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        self._frequency_cache: _QwenFrequencyCache | None = None
        self._cache_position = base_tokens
        self._pending_token: torch.Tensor | None = None
        self._prefetch: bool | None = None
        self._prefill_offset = 0
        self._prefilled = False
        self._pending_finish: GenerationFinishReason | None = None
        self._final_text: str | None = None
        self._decoded_text = ""
        self._emitted_text = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._decoder_finalized = False
        self._started = time.perf_counter()
        self._first_token_at: float | None = None
        self._prefill_time = 0.0
        self._decode_time = 0.0
        self._finished = False
        self._generator = _make_generator(provider._device, request.seed)  # pyright: ignore[reportPrivateUsage]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> GenerationEvent:
        if self._finished:
            raise StopIteration
        try:
            if self._cancelled():
                return self._cancelled_terminal()
            if self._pending_finish is not None:
                return self._successful_terminal()
            return self._next_token()
        except BaseException:
            self._abort()
            raise

    def close(self) -> None:
        self._abort()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _next_token(self) -> GenerationTokenEvent:
        if not self._prefilled:
            event = self._advance_prefill(len(self._prompt_ids))
            if event is None:
                raise AssertionError("complete Qwen prefill did not produce a token")
            return event
        return self._advance_decode()

    def _advance_prefill(self, max_tokens: int) -> GenerationTokenEvent | None:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Qwen prefill chunk size must be a positive exact integer")
        if self._prefilled:
            raise RuntimeError("Qwen generation stream is already prefilled")
        if not self._cache_layers:
            self._allocate_working_cache()
        end = min(len(self._prompt_ids), self._prefill_offset + max_tokens)
        chunk = self._prompt_ids[self._prefill_offset : end]
        started = time.perf_counter()
        logits = self._run_model(chunk)
        self._synchronize()
        self._prefill_time += time.perf_counter() - started
        self._prefill_offset = end
        if end < len(self._prompt_ids):
            return None
        self._prefilled = True
        return self._accept_logits(logits)

    def _advance_decode(self) -> GenerationTokenEvent:
        if not self._prefilled or not self._generated:
            raise RuntimeError("Qwen generation stream has no token to decode")
        started = time.perf_counter()
        logits = self._run_model((self._generated[-1],))
        self._synchronize()
        self._decode_time += time.perf_counter() - started
        return self._accept_logits(logits)

    def _accept_logits(self, logits: torch.Tensor) -> GenerationTokenEvent:
        token_id, token = _sample_token(
            logits,
            self._request.sampler,
            self._history_counts,
            self._generator,
        )
        self._pending_token = token
        self._generated.append(token_id)
        self._history.append(token_id)
        self._history_counts[token_id] += 1
        self._decoded_text += self._decoder.decode(
            self._provider._tokenizer.decode_bytes((token_id,)),  # pyright: ignore[reportPrivateUsage]
            final=False,
        )
        if self._first_token_at is None:
            self._first_token_at = time.perf_counter()

        finish = self._token_finish(token_id)
        stop_index = _first_stop_index(self._decoded_text, self._request.stop.stop_texts)
        if finish is None and stop_index is not None:
            finish = GenerationFinishReason.STOP
        if finish is None and len(self._generated) >= self._limit:
            finish = GenerationFinishReason.LENGTH
        if finish is not None:
            self._finalize_decoder()
            if stop_index is None:
                stop_index = _first_stop_index(
                    self._decoded_text,
                    self._request.stop.stop_texts,
                )
            self._final_text = (
                self._decoded_text if stop_index is None else self._decoded_text[:stop_index]
            )
            stable = self._final_text
            self._pending_finish = finish
        else:
            hold = _stop_prefix_length(self._decoded_text, self._request.stop.stop_texts)
            stable = self._decoded_text if hold == 0 else self._decoded_text[:-hold]
        delta = stable[len(self._emitted_text) :]
        self._emitted_text = stable
        return GenerationTokenEvent(len(self._generated) - 1, delta, token_id)

    def _successful_terminal(
        self,
        *,
        model_flushed: bool = False,
        flush_elapsed: float = 0.0,
    ) -> GenerationTerminalEvent:
        if self._handle is not None and not model_flushed:
            final_decode_started = time.perf_counter()
            self._run_model((self._generated[-1],))
            self._synchronize()
            flush_elapsed = time.perf_counter() - final_decode_started
        self._decode_time += flush_elapsed
        self._provider._finish(self, commit=self._handle is not None)  # pyright: ignore[reportPrivateUsage]
        self._release_working_cache()
        self._finished = True
        return GenerationTerminalEvent(
            GenerationResult(
                cast(str, self._final_text),
                cast(GenerationFinishReason, self._pending_finish),
                self._stats(),
                token_ids=tuple(self._generated),
                continuation=self._handle,
            )
        )

    def _cancelled_terminal(self) -> GenerationTerminalEvent:
        self._finalize_decoder()
        self._synchronize()
        self._provider._finish(self, commit=False)  # pyright: ignore[reportPrivateUsage]
        self._release_working_cache()
        self._finished = True
        return GenerationTerminalEvent(
            GenerationResult(
                self._decoded_text,
                GenerationFinishReason.CANCELLED,
                self._stats(),
                token_ids=tuple(self._generated),
                continuation=self._request.session,
            )
        )

    def _allocate_working_cache(self) -> None:
        provider = self._provider
        config = provider._model.config  # pyright: ignore[reportPrivateUsage]
        total = self._base_tokens + len(self._prompt_ids) + self._limit
        shape = (1, config.num_key_value_heads, total, config.head_dim)
        if len(set(provider._layer_devices)) == 1:  # pyright: ignore[reportPrivateUsage]
            cache_key = torch.empty(
                (config.num_hidden_layers, *shape),
                dtype=provider._dtype,  # pyright: ignore[reportPrivateUsage]
                device=provider._layer_devices[0],  # pyright: ignore[reportPrivateUsage]
            )
            cache_value = torch.empty_like(cache_key)
            cache_layers = tuple(
                (cache_key[layer], cache_value[layer]) for layer in range(config.num_hidden_layers)
            )
            self._cache_key = cache_key
            self._cache_value = cache_value
        else:
            cache_layers = tuple(
                (
                    torch.empty(
                        shape,
                        dtype=provider._dtype,  # pyright: ignore[reportPrivateUsage]
                        device=device,
                    ),
                    torch.empty(
                        shape,
                        dtype=provider._dtype,  # pyright: ignore[reportPrivateUsage]
                        device=device,
                    ),
                )
                for device in provider._layer_devices  # pyright: ignore[reportPrivateUsage]
            )
        frequency_cache = _allocate_qwen_frequency_cache(provider, total)
        self._bind_working_cache(cache_layers, frequency_cache)

    def _bind_working_cache(
        self,
        cache_layers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        frequency_cache: _QwenFrequencyCache,
    ) -> None:
        provider = self._provider
        config = provider._model.config  # pyright: ignore[reportPrivateUsage]
        if len(cache_layers) != config.num_hidden_layers:
            raise ValueError("Qwen working cache must contain one pair per model layer")
        capacity = self._base_tokens + len(self._prompt_ids) + self._limit
        expected = (1, config.num_key_value_heads, capacity, config.head_dim)
        if any(
            key.shape[:2] != expected[:2]
            or key.shape[2] < capacity
            or key.shape[3:] != expected[3:]
            or value.shape != key.shape
            or key.dtype != provider._dtype  # pyright: ignore[reportPrivateUsage]
            or value.dtype != provider._dtype  # pyright: ignore[reportPrivateUsage]
            or key.device != device
            or value.device != device
            for (key, value), device in zip(
                cache_layers,
                provider._layer_devices,  # pyright: ignore[reportPrivateUsage]
                strict=True,
            )
        ):
            raise ValueError("Qwen working cache geometry, dtype, or device is incompatible")
        self._cache_layers = cache_layers
        self._frequency_cache = frequency_cache
        if not self._base_tokens:
            return
        with torch.no_grad():
            for layer in range(config.num_hidden_layers):
                offset = 0
                for view in self._lease.block_views(layer):
                    end = offset + view.token_count
                    cache_layers[layer][0][0, :, offset:end].copy_(view.key.transpose(0, 1))
                    cache_layers[layer][1][0, :, offset:end].copy_(view.value.transpose(0, 1))
                    offset = end
                if offset != self._base_tokens:
                    raise RuntimeError("paged KV views do not cover the committed session")

    def _run_model(self, token_ids: tuple[int, ...]) -> torch.Tensor:
        provider = self._provider
        prefetch = self._resolve_prefetch()
        length = len(token_ids)
        if len(token_ids) == 1 and self._pending_token is not None:
            ids = self._pending_token.view(1, 1)
            self._pending_token = None
        else:
            ids = torch.tensor(
                (token_ids,),
                dtype=torch.long,
                device=provider._device,  # pyright: ignore[reportPrivateUsage]
            )
        if self._frequency_cache is None:
            raise RuntimeError("Qwen generation frequencies were not allocated")
        end = self._cache_position + length
        frequencies = _slice_qwen_frequency_cache(
            self._frequency_cache,
            self._cache_position,
            end,
        )
        with torch.inference_mode():
            logits, new_key_values = provider._forward_logits(  # pyright: ignore[reportPrivateUsage]
                ids,
                self._cache_layers,
                cache_position=self._cache_position,
                frequencies=frequencies,
                prefetch=prefetch,
            )
        self._accept_model_output(new_key_values, length)
        return logits[0]

    def _resolve_prefetch(self) -> bool:
        if self._prefetch is None:
            provider = self._provider
            prefetch_probe = make_prefetch_queue(provider._model.layers)  # pyright: ignore[reportPrivateUsage]
            self._prefetch = prefetch_probe is not None
            close_prefetch_queue(prefetch_probe)
        return self._prefetch

    def _accept_model_output(
        self,
        new_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        length: int,
    ) -> None:
        if self._handle is not None:
            if isinstance(self._lease, PagedKVSessionLease):
                key = torch.stack([layer_key[0].transpose(0, 1) for layer_key, _ in new_key_values])
                value = torch.stack(
                    [layer_value[0].transpose(0, 1) for _, layer_value in new_key_values]
                )
                self._lease.append(key, value)
            else:
                self._lease.append_layers(new_key_values)
        self._cache_position += length

    def _token_finish(self, token_id: int) -> GenerationFinishReason | None:
        if token_id in self._provider._eos_token_ids:  # pyright: ignore[reportPrivateUsage]
            return GenerationFinishReason.EOS
        if token_id in self._request.stop.stop_token_ids:
            return GenerationFinishReason.STOP
        return None

    def _finalize_decoder(self) -> None:
        if self._decoder_finalized:
            return
        self._decoded_text += self._decoder.decode(b"", final=True)
        self._decoder_finalized = True

    def _stats(self) -> GenerationStats:
        total = time.perf_counter() - self._started
        first = None if self._first_token_at is None else self._first_token_at - self._started
        return GenerationStats(
            total,
            prompt_tokens=len(self._prompt_ids),
            generated_tokens=len(self._generated),
            time_to_first_token_s=first,
            prefill_time_s=self._prefill_time,
            decode_time_s=self._decode_time,
        )

    def _synchronize(self) -> None:
        devices = dict.fromkeys(self._provider._placement.devices)  # pyright: ignore[reportPrivateUsage]
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()

    def _abort(self) -> None:
        if self._finished:
            return
        try:
            self._provider._finish(self, commit=False)  # pyright: ignore[reportPrivateUsage]
        finally:
            self._release_working_cache()
            self._finished = True

    def _release_working_cache(self) -> None:
        self._cache_layers = ()
        self._cache_key = None
        self._cache_value = None
        self._frequency_cache = None
        self._pending_token = None


def _allocate_qwen_frequency_cache(
    provider: QwenGenerationProvider,
    length: int,
) -> _QwenFrequencyCache:
    devices = provider._layer_devices  # pyright: ignore[reportPrivateUsage]
    if len(set(devices)) == 1:
        return provider._model.causal_frequencies(  # pyright: ignore[reportPrivateUsage]
            length,
            device=devices[0],
        )
    by_device = {
        device: provider._model.causal_frequencies(  # pyright: ignore[reportPrivateUsage]
            length,
            device=device,
        )
        for device in dict.fromkeys(devices)
    }
    return tuple(by_device[device] for device in devices)


def _slice_qwen_frequency_cache(
    cache: _QwenFrequencyCache,
    start: int,
    stop: int,
) -> _QwenFrequencyCache:
    if isinstance(cache[0], torch.Tensor):
        frequencies = cast(_QwenFrequencies, cache)
        return (
            frequencies[0][:, :, start:stop],
            frequencies[1][:, :, start:stop],
            frequencies[2][:, :, start:stop],
        )
    return tuple(
        (
            frequencies[0][:, :, start:stop],
            frequencies[1][:, :, start:stop],
            frequencies[2][:, :, start:stop],
        )
        for frequencies in cast(tuple[_QwenFrequencies, ...], cache)
    )


def _make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def _sample_token(
    source: torch.Tensor,
    chain: GenerationSamplerChain,
    history: Sequence[int] | Counter[int],
    generator: torch.Generator | None,
) -> tuple[int, torch.Tensor]:
    if len(chain.stages) == 1:
        selector = chain.stages[0].kind
        if selector is GenerationSamplerKind.GREEDY:
            token = torch.argmax(source).reshape(1)
            return int(token.item()), token
        if selector is GenerationSamplerKind.MULTINOMIAL:
            probabilities = torch.softmax(source.float(), dim=-1)
            token = torch.multinomial(probabilities, 1, generator=generator)
            return int(token.item()), token

    logits = source.to(dtype=torch.float32, copy=True)
    counts: Counter[int] | None = history if isinstance(history, Counter) else None
    for stage in chain.stages:
        kind = stage.kind
        if kind in (
            GenerationSamplerKind.REPETITION_PENALTY,
            GenerationSamplerKind.PRESENCE_PENALTY,
            GenerationSamplerKind.FREQUENCY_PENALTY,
        ):
            if counts is None:
                counts = Counter(history)
            if counts:
                token_ids = torch.tensor(tuple(counts), dtype=torch.long, device=logits.device)
                selected = logits[token_ids]
                value = cast(float, stage.value)
                if kind is GenerationSamplerKind.REPETITION_PENALTY:
                    selected = torch.where(selected < 0, selected * value, selected / value)
                elif kind is GenerationSamplerKind.PRESENCE_PENALTY:
                    selected = selected - value
                else:
                    frequencies = torch.tensor(
                        tuple(counts.values()),
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    selected = selected - frequencies * value
                logits[token_ids] = selected
        elif kind is GenerationSamplerKind.TEMPERATURE:
            logits.div_(cast(float, stage.value))
        elif kind is GenerationSamplerKind.TOP_K:
            count = min(cast(int, stage.value), logits.shape[-1])
            threshold = torch.topk(logits, count).values[-1]
            logits.masked_fill_(logits < threshold, -torch.inf)
        elif kind is GenerationSamplerKind.TOP_P:
            logits = _top_p(logits, cast(float, stage.value))
        elif kind is GenerationSamplerKind.MIN_P:
            probabilities = torch.softmax(logits, dim=-1)
            threshold = probabilities.max() * cast(float, stage.value)
            logits.masked_fill_(probabilities < threshold, -torch.inf)
        elif kind is GenerationSamplerKind.TYPICAL_P:
            logits = _typical_p(logits, cast(float, stage.value))
        elif kind is GenerationSamplerKind.GREEDY:
            token = torch.argmax(logits).reshape(1)
            return int(token.item()), token
        elif kind is GenerationSamplerKind.MULTINOMIAL:
            probabilities = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probabilities, 1, generator=generator)
            return int(token.item()), token
    raise AssertionError("validated sampler chain has no selector")


def _top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, order = torch.sort(logits, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    mask = torch.zeros_like(remove)
    mask.scatter_(0, order, remove)
    return logits.masked_fill(mask, -torch.inf)


def _typical_p(logits: torch.Tensor, typical_p: float) -> torch.Tensor:
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).nan_to_num().sum()
    deviation = (-log_probabilities - entropy).abs()
    order = torch.argsort(deviation)
    cumulative = probabilities[order].cumsum(dim=-1)
    remove = cumulative > typical_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    mask = torch.zeros_like(remove)
    mask.scatter_(0, order, remove)
    return logits.masked_fill(mask, -torch.inf)


def _first_stop_index(text: str, stops: Sequence[str]) -> int | None:
    indexes = tuple(index for stop in stops if (index := text.find(stop)) >= 0)
    return min(indexes) if indexes else None


def _stop_prefix_length(text: str, stops: Sequence[str]) -> int:
    longest = 0
    for stop in stops:
        for length in range(min(len(text), len(stop) - 1), longest, -1):
            if text.endswith(stop[:length]):
                longest = length
                break
    return longest


__all__ = [
    "QwenGenerationProvider",
    "QwenGenerationTokenizer",
]
