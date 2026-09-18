"""Wan Animate2 pose-branch transformer and execution-scoped pose cache."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from dinkster_inference import WAN21_ANIMATE2_14B, Wan21Config

from . import pinned_host
from .attention import AttentionKernel, select_attention
from .flux import apply_rope
from .memory import MemoryPolicy, get_free_memory
from .operations import INITLESS, Operations
from .ops import cast_weight
from .wan21_model import (
    Wan21Model,
    WanAttentionBlock,
    WanI2VCrossAttention,
    WanSelfAttention,
    _repeat_time_rows,  # pyright: ignore[reportPrivateUsage]
    sinusoidal_embedding_1d,
)

PoseCacheDType = Literal["default", "int8", "int4"]
_DEFAULT_ATTENTION = select_attention("flux").kernel
_ACCELERATOR_CACHE_RESERVE = MemoryPolicy().minimum_inference_memory()


@dataclass(slots=True)
class _PoseCacheCorrections:
    indices: torch.Tensor
    values: torch.Tensor


@dataclass(slots=True)
class _PoseCacheEntry:
    tensor: torch.Tensor
    params: Any | None
    corrections: _PoseCacheCorrections | None
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype


@dataclass(slots=True)
class _PoseCacheSlot:
    keys: tuple[torch.Tensor, ...]
    blocks: dict[int, _PoseCacheEntry]


@dataclass(slots=True)
class _StagingPair:
    host: tuple[torch.Tensor, torch.Tensor]
    device: tuple[torch.Tensor, torch.Tensor]
    ready: tuple[torch.cuda.Event, torch.cuda.Event]
    pinned_bytes: int
    memory_bytes: int


class _PoseCacheRestoreError(RuntimeError):
    pass


class PoseBranchCache:
    """Cache one pose-branch input per transformer block and pose sequence."""

    CONVROT_GROUP_SIZE = 256
    _RECONSTRUCTION_MAX_ABS = {"int8": 0.02, "int4": 0.35}

    def __init__(
        self,
        store_device: torch.device | str = "cpu",
        dtype: PoseCacheDType = "default",
        memory_limit_bytes: int | None = None,
    ) -> None:
        if dtype not in ("default", "int8", "int4"):
            raise ValueError("Animate2 cache dtype must be default, int8, or int4")
        if memory_limit_bytes is not None and (
            type(memory_limit_bytes) is not int or memory_limit_bytes < 1
        ):
            raise ValueError("Animate2 cache memory limit must be a positive integer")
        self.store_device = torch.device(store_device)
        self.dtype: PoseCacheDType = dtype
        self.memory_limit_bytes = memory_limit_bytes
        self._slots: list[_PoseCacheSlot] = []
        self._slot: _PoseCacheSlot | None = None
        self._selected = False
        self._pending: dict[int, tuple[torch.Tensor, torch.cuda.Stream | None]] = {}
        self._streams: dict[str, torch.cuda.Stream] = {}
        self._staging: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.dtype], _StagingPair
        ] = {}
        self.pin_active = True

    @staticmethod
    def _minimum_free_bytes(device: torch.device) -> int:
        if device.type in ("cpu", "mps"):
            return pinned_host.AVAILABLE_RAM_FLOOR
        return _ACCELERATOR_CACHE_RESERVE

    def _has_capacity(self, device: torch.device, size: int) -> bool:
        return get_free_memory(device).free_total >= size + self._minimum_free_bytes(device)

    def _has_store_capacity(self, size: int) -> bool:
        return self._has_capacity(self.store_device, size)

    def _synchronize_pending(self) -> None:
        for _, stream in self._pending.values():
            if stream is not None:
                stream.synchronize()
        self._pending.clear()

    def select(
        self,
        pose_latents: torch.Tensor,
        pose_context: torch.Tensor | None = None,
        pose_vision: torch.Tensor | None = None,
    ) -> None:
        self._synchronize_pending()
        self._selected = True
        keys = tuple(
            value[:1] for value in (pose_latents, pose_context, pose_vision) if value is not None
        )
        for slot in self._slots:
            if len(slot.keys) == len(keys) and all(
                stored.shape == key.shape and torch.equal(stored, key.to(stored.device))
                for stored, key in zip(slot.keys, keys, strict=True)
            ):
                self._slots.remove(slot)
                self._slots.append(slot)
                self._slot = slot
                return
        key_bytes = sum(self._tensor_bytes(key) for key in keys)
        estimate = max(
            key_bytes,
            int(max((self._slot_bytes(slot) for slot in self._slots), default=0) * 1.5),
        )
        while self._slots and (
            not self._has_store_capacity(estimate)
            or (
                self.memory_limit_bytes is not None
                and self.memory_bytes() + estimate > self.memory_limit_bytes
            )
        ):
            self._slots.pop(0)
        if (
            self.memory_limit_bytes is not None
            and self.memory_bytes() + key_bytes > self.memory_limit_bytes
        ) or not self._has_store_capacity(key_bytes):
            self._slot = None
            return
        self._slot = _PoseCacheSlot(
            keys=tuple(key.detach().to(self.store_device, copy=True) for key in keys),
            blocks={},
        )
        self._slots.append(self._slot)

    def filled(self, block_count: int) -> bool:
        return self._slot is not None and len(self._slot.blocks) == block_count

    def put(self, index: int, pose_input: torch.Tensor) -> None:
        slot = self._slot
        if slot is None:
            if self._selected:
                return
            raise RuntimeError("Animate2 pose cache has no selected slot")
        tensor = pose_input[:1].detach()
        original_shape = tuple(tensor.shape)
        original_dtype = tensor.dtype
        params: Any | None = None
        corrections: _PoseCacheCorrections | None = None
        if self.dtype != "default":
            from comfy_kitchen.tensor import (  # pyright: ignore[reportMissingTypeStubs]
                TensorCoreConvRotW4A4Layout,
                TensorWiseINT8Layout,
            )

            if not self._has_capacity(tensor.device, self._tensor_bytes(tensor) * 8):
                return
            source = tensor
            flat = tensor.reshape(-1, tensor.shape[-1])
            group_size = self.CONVROT_GROUP_SIZE
            while group_size > 4 and tensor.shape[-1] % group_size:
                group_size //= 4
            if self.dtype == "int4":
                tensor, params = TensorCoreConvRotW4A4Layout.quantize(
                    flat,
                    convrot_groupsize=group_size,
                )
            else:
                tensor, params = TensorWiseINT8Layout.quantize(
                    flat,
                    is_weight=True,
                    per_channel=True,
                    convrot=True,
                    convrot_groupsize=group_size,
                )
            layout = TensorCoreConvRotW4A4Layout if self.dtype == "int4" else TensorWiseINT8Layout
            reconstructed = layout.dequantize(tensor, params).reshape(original_shape)
            error = reconstructed.float().sub_(source).abs_().reshape(-1)
            indices = torch.nonzero(
                error > self._RECONSTRUCTION_MAX_ABS[self.dtype], as_tuple=False
            ).flatten()
            if indices.numel():
                values = source.reshape(-1).index_select(0, indices)
                corrections = _PoseCacheCorrections(indices.to(torch.int32), values)
        old = slot.blocks.get(index)
        old_bytes = 0 if old is None else self._entry_bytes(old)
        new_bytes = (
            self._tensor_bytes(tensor)
            + self._params_bytes(params)
            + self._corrections_bytes(corrections)
        )
        if (
            self.memory_limit_bytes is not None
            and self.memory_bytes() - old_bytes + new_bytes > self.memory_limit_bytes
        ) or not self._has_store_capacity(new_bytes):
            return
        tensor = tensor.to(self.store_device, copy=True)
        if params is not None:
            params = params.to_device(self.store_device)
        if corrections is not None:
            corrections = _PoseCacheCorrections(
                corrections.indices.to(self.store_device, copy=True),
                corrections.values.to(self.store_device, copy=True),
            )
        slot.blocks[index] = _PoseCacheEntry(
            tensor,
            params,
            corrections,
            original_shape,
            original_dtype,
        )

    @staticmethod
    def _dtype_bytes(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    @classmethod
    def _restored_bytes(cls, entry: _PoseCacheEntry, dtype: torch.dtype) -> int:
        return math.prod(entry.original_shape) * cls._dtype_bytes(dtype)

    @classmethod
    def _restore_working_bytes(
        cls,
        entry: _PoseCacheEntry,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> int:
        desired_bytes = cls._restored_bytes(entry, dtype)
        working_bytes = 0
        if entry.params is not None:
            working_bytes += cls._restored_bytes(entry, entry.original_dtype)
            working_bytes += cls._params_bytes(entry.params)
            if entry.corrections is not None:
                working_bytes += cls._corrections_bytes(entry.corrections)
                working_bytes += entry.corrections.indices.numel() * cls._dtype_bytes(torch.int64)
            if entry.original_dtype != dtype:
                working_bytes += desired_bytes
        elif entry.tensor.device != device and not (
            entry.tensor.device.type == "cpu" and device.type == "cuda"
        ):
            working_bytes += desired_bytes
        elif entry.tensor.device == device and entry.tensor.dtype != dtype:
            working_bytes += desired_bytes
        if batch_size > 1:
            working_bytes += desired_bytes * batch_size
        return working_bytes

    @staticmethod
    def _staging_key(
        tensor: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[str, tuple[int, ...], torch.dtype, torch.dtype]:
        return (str(device), tuple(tensor.shape), tensor.dtype, dtype)

    @classmethod
    def _staging_bytes(cls, tensor: torch.Tensor, dtype: torch.dtype) -> tuple[int, int]:
        host_bytes = cls._tensor_bytes(tensor) * 2
        device_bytes = tensor.numel() * cls._dtype_bytes(dtype) * 2
        return host_bytes, device_bytes

    def prepare_restore(
        self,
        block_count: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> bool:
        slot = self._slot
        if (
            slot is None
            or set(slot.blocks) != set(range(block_count))
            or type(batch_size) is not int
            or batch_size < 1
        ):
            return False
        staging: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.dtype],
            tuple[torch.Tensor, torch.dtype],
        ] = {}
        host_bytes = 0
        device_bytes = 0
        working_bytes = 0
        for entry in slot.blocks.values():
            working_bytes = max(
                working_bytes,
                self._restore_working_bytes(entry, device, dtype, batch_size),
            )
            if entry.tensor.device.type != "cpu" or device.type != "cuda":
                continue
            transfer_dtype = entry.tensor.dtype if entry.params is not None else dtype
            key = self._staging_key(entry.tensor, device, transfer_dtype)
            if key in self._staging or key in staging:
                continue
            staging[key] = (entry.tensor, transfer_dtype)
            pair_host_bytes, pair_device_bytes = self._staging_bytes(entry.tensor, transfer_dtype)
            host_bytes += pair_host_bytes
            device_bytes += pair_device_bytes
        retained_bytes = host_bytes + device_bytes
        if (
            self.memory_limit_bytes is not None
            and self.memory_bytes() + retained_bytes > self.memory_limit_bytes
        ):
            return False
        if host_bytes and not self._has_capacity(torch.device("cpu"), host_bytes):
            return False
        if not self._has_capacity(device, device_bytes + working_bytes):
            return False
        existing_keys = set(self._staging)
        for tensor, transfer_dtype in staging.values():
            if self._staging_pair(tensor, device, transfer_dtype, working_bytes) is None:
                for key in set(self._staging) - existing_keys:
                    self._drop_staging(key)
                return False
        return True

    def _staging_pair(
        self,
        tensor: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        working_bytes: int,
    ) -> _StagingPair | None:
        key = self._staging_key(tensor, device, dtype)
        pair = self._staging.get(key)
        if pair is not None:
            return pair
        host_bytes, device_bytes = self._staging_bytes(tensor, dtype)
        memory_bytes = host_bytes + device_bytes
        if (
            self.memory_limit_bytes is not None
            and self.memory_bytes() + memory_bytes > self.memory_limit_bytes
        ):
            return None
        if not self._has_capacity(torch.device("cpu"), host_bytes) or not self._has_capacity(
            device, device_bytes + working_bytes
        ):
            return None
        pinned_bytes = host_bytes
        use_pinned = pinned_host.reserve_storage(self, pinned_bytes)
        if not use_pinned:
            pinned_host.discard_owner_if_empty(self)
        if use_pinned and (
            not pinned_host.ensure_pin_budget(pinned_bytes)
            or not pinned_host.ensure_pin_registerable(pinned_bytes)
        ):
            pinned_host.account_storage(self, -pinned_bytes)
            pinned_host.discard_owner_if_empty(self)
            use_pinned = False
        try:
            host = (
                torch.empty_like(tensor, device="cpu", pin_memory=use_pinned),
                torch.empty_like(tensor, device="cpu", pin_memory=use_pinned),
            )
        except RuntimeError:
            if not use_pinned:
                raise
            pinned_host.account_storage(self, -pinned_bytes)
            pinned_host.discard_owner_if_empty(self)
            use_pinned = False
            host = (
                torch.empty_like(tensor, device="cpu"),
                torch.empty_like(tensor, device="cpu"),
            )
        if use_pinned:
            pinned_host.account(pinned_bytes)
        created: _StagingPair | None = None
        try:
            created = _StagingPair(
                host=host,
                device=(
                    torch.empty(tensor.shape, dtype=dtype, device=device),
                    torch.empty(tensor.shape, dtype=dtype, device=device),
                ),
                ready=(torch.cuda.Event(), torch.cuda.Event()),
                pinned_bytes=pinned_bytes if use_pinned else 0,
                memory_bytes=memory_bytes,
            )
        except torch.OutOfMemoryError:
            return None
        finally:
            if created is None and use_pinned:
                pinned_host.account(-pinned_bytes)
                pinned_host.account_storage(self, -pinned_bytes)
                pinned_host.discard_owner_if_empty(self)
        assert created is not None
        self._staging[key] = created
        return created

    def prefetch(self, index: int, device: torch.device, dtype: torch.dtype) -> bool:
        slot = self._slot
        if slot is None or index not in slot.blocks or index in self._pending:
            return index in self._pending
        entry = slot.blocks[index]
        transfer_dtype = entry.tensor.dtype if entry.params is not None else dtype
        if entry.tensor.device == device:
            self._pending[index] = (entry.tensor.to(dtype=transfer_dtype), None)
            return True
        if entry.tensor.device.type != "cpu" or device.type != "cuda":
            transfer_bytes = entry.tensor.numel() * self._dtype_bytes(transfer_dtype)
            if not self._has_capacity(device, transfer_bytes):
                return False
            self._pending[index] = (
                entry.tensor.to(device=device, dtype=transfer_dtype),
                None,
            )
            return True
        pair = self._staging_pair(
            entry.tensor,
            device,
            transfer_dtype,
            self._restore_working_bytes(entry, device, dtype, 1),
        )
        if pair is None:
            return False
        stream = self._streams.setdefault(str(device), torch.cuda.Stream(device=device))
        buffer_index = index % 2
        pair.ready[buffer_index].synchronize()
        pair.host[buffer_index].copy_(entry.tensor)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            pair.device[buffer_index].copy_(pair.host[buffer_index], non_blocking=True)
            pair.ready[buffer_index].record(stream)
        self._pending[index] = (pair.device[buffer_index], stream)
        return True

    def take(
        self,
        index: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        slot = self._slot
        if slot is None:
            raise RuntimeError("Animate2 pose cache has no selected slot")
        entry = slot.blocks[index]
        working_bytes = self._restore_working_bytes(entry, device, dtype, batch_size)
        if not self._has_capacity(device, working_bytes):
            raise _PoseCacheRestoreError(
                "Animate2 pose cache cannot restore within the memory reserve"
            )
        if index not in self._pending and not self.prefetch(index, device, dtype):
            raise _PoseCacheRestoreError(
                "Animate2 pose cache cannot restore within the memory bounds"
            )
        tensor, stream = self._pending.pop(index)
        if stream is not None:
            torch.cuda.current_stream(device).wait_stream(stream)
        if entry.params is not None:
            from comfy_kitchen.tensor import (  # pyright: ignore[reportMissingTypeStubs]
                TensorCoreConvRotW4A4Layout,
                TensorWiseINT8Layout,
            )

            params = entry.params.to_device(tensor.device)
            layout = TensorCoreConvRotW4A4Layout if self.dtype == "int4" else TensorWiseINT8Layout
            tensor = layout.dequantize(tensor, params).reshape(entry.original_shape)
            if entry.corrections is not None:
                indices = entry.corrections.indices.to(device=tensor.device, dtype=torch.int64)
                values = entry.corrections.values.to(tensor.device)
                tensor.reshape(-1).index_copy_(0, indices, values)
            tensor = tensor.to(dtype)
        if batch_size == 1:
            return tensor
        return tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1))

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    @classmethod
    def _params_bytes(cls, params: Any | None) -> int:
        return 0 if params is None else cls._tensor_bytes(params.scale)

    @classmethod
    def _corrections_bytes(cls, corrections: _PoseCacheCorrections | None) -> int:
        if corrections is None:
            return 0
        return cls._tensor_bytes(corrections.indices) + cls._tensor_bytes(corrections.values)

    @classmethod
    def _entry_bytes(cls, entry: _PoseCacheEntry) -> int:
        return (
            cls._tensor_bytes(entry.tensor)
            + cls._params_bytes(entry.params)
            + cls._corrections_bytes(entry.corrections)
        )

    @classmethod
    def _slot_bytes(cls, slot: _PoseCacheSlot) -> int:
        return sum(cls._tensor_bytes(key) for key in slot.keys) + sum(
            cls._entry_bytes(entry) for entry in slot.blocks.values()
        )

    def memory_bytes(self) -> int:
        return sum(self._slot_bytes(slot) for slot in self._slots) + sum(
            pair.memory_bytes for pair in self._staging.values()
        )

    def _drop_staging(
        self,
        key: tuple[str, tuple[int, ...], torch.dtype, torch.dtype],
    ) -> None:
        pair = self._staging.pop(key)
        for ready in pair.ready:
            ready.synchronize()
        if pair.pinned_bytes:
            pinned_host.account(-pair.pinned_bytes)
            pinned_host.account_storage(self, -pair.pinned_bytes)
            pinned_host.discard_owner_if_empty(self)

    def _free_staging(self, size: int) -> int:
        freed = 0
        for key, pair in tuple(self._staging.items()):
            if pair.pinned_bytes == 0:
                continue
            pinned_bytes = pair.pinned_bytes
            self._drop_staging(key)
            freed += pinned_bytes
            if freed >= size:
                break
        pinned_host.discard_owner_if_empty(self)
        return freed

    def free_pins(self, size: int) -> int:
        return 0 if self.pin_active else self._free_staging(size)

    def free_registrations(self, size: int) -> int:
        return 0 if self.pin_active else self._free_staging(size)

    def free(self) -> None:
        self._synchronize_pending()
        for stream in self._streams.values():
            stream.synchronize()
        self._free_staging(1 << 63)
        self._slots.clear()
        self._slot = None
        self._selected = False
        self._staging.clear()
        self._streams.clear()
        self.pin_active = False


class WanAnimate2SelfAttention(WanSelfAttention):
    def _qkv(
        self, x: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence = x.shape[:2]
        query = self.norm_q(self.q(x)).view(batch, sequence, self.num_heads, self.head_dim)
        key = self.norm_k(self.k(x)).view(batch, sequence, self.num_heads, self.head_dim)
        query, key = apply_rope(query, key, freqs)
        value = self.v(x).view(batch, sequence, self.num_heads, self.head_dim)
        return query, key, value

    def pose_kv(self, x: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, key, value = self._qkv(x, freqs)
        return key, value

    def forward_pose(
        self, x: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self._qkv(x, freqs)
        attended = self._attention_kernel(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        output = self.o(attended.transpose(1, 2).flatten(2))
        return output, key, value

    def forward_generation(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        pose_key: torch.Tensor | None,
        pose_value: torch.Tensor | None,
        *,
        frame_count: int,
        frame_rows: int,
        buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        reference_strength: float,
    ) -> torch.Tensor:
        query, key, value = self._qkv(x, freqs)
        if reference_strength != 1.0:
            value[:, :frame_rows] *= reference_strength
        if pose_key is None or pose_value is None:
            attended = self._attention_kernel(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
            )
            return self.o(attended.transpose(1, 2).flatten(2))
        if buffers is None:
            raise RuntimeError("Animate2 pose attention requires reusable buffers")
        key_buffer, value_buffer, output = buffers
        sequence = key.shape[1]
        key_buffer[:, :sequence] = key
        value_buffer[:, :sequence] = value
        for frame in range(frame_count):
            frame_slice = slice(frame * frame_rows, (frame + 1) * frame_rows)
            frame_query = query[:, frame_slice].transpose(1, 2)
            if frame == 0:
                attended = self._attention_kernel(
                    frame_query,
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                )
            else:
                pose_slice = slice((frame - 1) * frame_rows, frame * frame_rows)
                key_buffer[:, sequence:] = pose_key[:, pose_slice]
                value_buffer[:, sequence:] = pose_value[:, pose_slice]
                attended = self._attention_kernel(
                    frame_query,
                    key_buffer.transpose(1, 2),
                    value_buffer.transpose(1, 2),
                )
            output[:, frame_slice] = attended.transpose(1, 2).flatten(2)
        return self.o(output)


class WanAnimate2Block(WanAttentionBlock):
    self_attn: WanAnimate2SelfAttention

    def __init__(
        self,
        config: Wan21Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
            _self_attention_type=WanAnimate2SelfAttention,
        )

    def _cross_attention_feed_forward(
        self,
        x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        context: torch.Tensor,
        image_rows: int | None,
    ) -> torch.Tensor:
        cross_attention = cast("WanI2VCrossAttention", self.cross_attn)
        x = x + cross_attention(self.norm3(x), context, image_rows)
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[3], x),
            self.norm2(x),
            1 + _repeat_time_rows(modulation[4], x),
        )
        return torch.addcmul(
            x,
            self.ffn(normalized),
            _repeat_time_rows(modulation[5], x),
        )

    def _forward_pose_owned(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        image_rows: int | None,
        modulation_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modulation = (modulation_state.unsqueeze(0) + time).unbind(2)
        x = x.contiguous()
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[0], x),
            self.norm1(x),
            1 + _repeat_time_rows(modulation[1], x),
        )
        update, key, value = self.self_attn.forward_pose(normalized, freqs)
        x = torch.addcmul(x, update, _repeat_time_rows(modulation[2], x))
        return self._cross_attention_feed_forward(x, modulation, context, image_rows), key, value

    def forward_pose(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        image_rows: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        binding = self._offloaded_residency()
        if binding is None:
            modulation = cast_weight(self.modulation, device=x.device, dtype=x.dtype)
            return self._forward_pose_owned(x, time, freqs, context, image_rows, modulation)
        with binding.lease() as lease:
            return self._forward_pose_owned(
                x,
                time,
                freqs,
                context,
                image_rows,
                lease.get("modulation", dtype=x.dtype),
            )

    def _pose_kv_owned(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        modulation_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modulation = (modulation_state.unsqueeze(0) + time).unbind(2)
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[0], x),
            self.norm1(x.contiguous()),
            1 + _repeat_time_rows(modulation[1], x),
        )
        return self.self_attn.pose_kv(normalized, freqs)

    def pose_kv(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        binding = self._offloaded_residency()
        if binding is None:
            modulation = cast_weight(self.modulation, device=x.device, dtype=x.dtype)
            return self._pose_kv_owned(x, time, freqs, modulation)
        with binding.lease() as lease:
            return self._pose_kv_owned(
                x,
                time,
                freqs,
                lease.get("modulation", dtype=x.dtype),
            )

    def _forward_generation_owned(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        image_rows: int | None,
        pose_key: torch.Tensor | None,
        pose_value: torch.Tensor | None,
        *,
        frame_count: int,
        frame_rows: int,
        buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        reference_strength: float,
        modulation_state: torch.Tensor,
    ) -> torch.Tensor:
        modulation = (modulation_state.unsqueeze(0) + time).unbind(2)
        x = x.contiguous()
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[0], x),
            self.norm1(x),
            1 + _repeat_time_rows(modulation[1], x),
        )
        update = self.self_attn.forward_generation(
            normalized,
            freqs,
            pose_key,
            pose_value,
            frame_count=frame_count,
            frame_rows=frame_rows,
            buffers=buffers,
            reference_strength=reference_strength,
        )
        x = torch.addcmul(x, update, _repeat_time_rows(modulation[2], x))
        return self._cross_attention_feed_forward(x, modulation, context, image_rows)

    def forward_generation(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        image_rows: int | None,
        pose_key: torch.Tensor | None,
        pose_value: torch.Tensor | None,
        *,
        frame_count: int,
        frame_rows: int,
        buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        reference_strength: float,
    ) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            modulation = cast_weight(self.modulation, device=x.device, dtype=x.dtype)
            return self._forward_generation_owned(
                x,
                time,
                freqs,
                context,
                image_rows,
                pose_key,
                pose_value,
                frame_count=frame_count,
                frame_rows=frame_rows,
                buffers=buffers,
                reference_strength=reference_strength,
                modulation_state=modulation,
            )
        with binding.lease() as lease:
            return self._forward_generation_owned(
                x,
                time,
                freqs,
                context,
                image_rows,
                pose_key,
                pose_value,
                frame_count=frame_count,
                frame_rows=frame_rows,
                buffers=buffers,
                reference_strength=reference_strength,
                modulation_state=lease.get("modulation", dtype=x.dtype),
            )


class WanAnimate2Model(Wan21Model):
    """Wan 2.1 I2V geometry with Animate2's lockstep pose branch."""

    def __init__(
        self,
        config: Wan21Config = WAN21_ANIMATE2_14B,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        if config.model_variant != "animate2":
            raise ValueError("WanAnimate2Model requires an Animate2 config")
        super().__init__(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
            _block_type=WanAnimate2Block,
        )

    def _pose_rope(
        self,
        shape: tuple[int, int, int],
        x: torch.Tensor,
        *,
        generation_width: int,
    ) -> torch.Tensor:
        time, height, width = shape
        ids = torch.zeros((time, height, width, 3), device=x.device, dtype=x.dtype)
        ids[..., 0] += torch.linspace(1, time, time, device=x.device, dtype=x.dtype)[:, None, None]
        ids[..., 1] += torch.linspace(0, height - 1, height, device=x.device, dtype=x.dtype)[
            None, :, None
        ]
        ids[..., 2] += torch.linspace(
            generation_width,
            generation_width + width - 1,
            width,
            device=x.device,
            dtype=x.dtype,
        )[None, None, :]
        return self.rope_embedder(ids.reshape(1, -1, 3)).movedim(1, 2)

    @staticmethod
    def _validate_strength(name: str, value: float) -> None:
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative float")

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        vision: torch.Tensor | None = None,
        *,
        pose_latents: torch.Tensor | None = None,
        pose_context: torch.Tensor | None = None,
        pose_vision: torch.Tensor | None = None,
        pose_strength: float = 1.0,
        reference_strength: float = 1.0,
        pose_cache: PoseBranchCache | None = None,
    ) -> torch.Tensor:
        self._validate(
            x,
            timesteps,
            context,
            vision,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self._validate_strength("pose_strength", pose_strength)
        self._validate_strength("reference_strength", reference_strength)
        if pose_latents is None:
            if pose_context is not None or pose_vision is not None or pose_cache is not None:
                raise ValueError("Animate2 pose context, vision, and cache require pose latents")
        else:
            if (
                type(pose_latents) is not torch.Tensor
                or not pose_latents.is_floating_point()
                or pose_latents.layout != torch.strided
            ):
                raise TypeError("pose_latents must be an exact strided floating torch.Tensor")
            expected = (
                x.shape[0],
                self.config.out_channels,
                x.shape[2] - 1,
                x.shape[3],
                x.shape[4],
            )
            if tuple(pose_latents.shape) != expected:
                raise ValueError(f"pose_latents must have shape {expected}")
        for name, value, width in (
            ("pose_context", pose_context, self.config.text_dim),
            ("pose_vision", pose_vision, 1280),
        ):
            if value is None:
                continue
            if (
                type(value) is not torch.Tensor
                or not value.is_floating_point()
                or value.layout != torch.strided
            ):
                raise TypeError(f"{name} must be an exact strided floating torch.Tensor")
            if (
                value.ndim != 3
                or value.shape[0] != x.shape[0]
                or value.shape[1] == 0
                or value.shape[2] != width
            ):
                raise ValueError(
                    f"{name} must have shape ({x.shape[0]}, rows, {width}) with at least one row"
                )

        original_shape = x.shape[2:]
        pad_h = (-x.shape[3]) % self.config.patch_size[1]
        pad_w = (-x.shape[4]) % self.config.patch_size[2]
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="circular")
            if pose_latents is not None:
                pose_latents = F.pad(
                    pose_latents,
                    (0, pad_w, 0, pad_h, 0, 0),
                    mode="circular",
                )

        x = self.patch_embedding(x.float()).to(x.dtype)
        grid = (x.shape[2], x.shape[3], x.shape[4])
        frame_count, grid_height, grid_width = grid
        frame_rows = grid_height * grid_width
        freqs = self._rope(grid, x)
        x = x.flatten(2).transpose(1, 2)

        apply_pose = pose_latents is not None
        effective_pose_context = context if pose_context is None else pose_context
        effective_pose_vision = vision if pose_vision is None else pose_vision
        cache_inputs = (pose_latents, effective_pose_context, effective_pose_vision)
        active_pose_cache = pose_cache
        if active_pose_cache is not None and not all(
            value is None or value.shape[0] == 1 or torch.equal(value, value[:1].expand_as(value))
            for value in cache_inputs
        ):
            active_pose_cache = None
        cached = False
        if apply_pose and active_pose_cache is not None:
            assert pose_latents is not None
            active_pose_cache.select(
                pose_latents,
                effective_pose_context,
                effective_pose_vision,
            )
            cached = active_pose_cache.filled(
                len(self.blocks)
            ) and active_pose_cache.prepare_restore(len(self.blocks), x.device, x.dtype, x.shape[0])
        pose_input: torch.Tensor | None = None
        pose_freqs: torch.Tensor | None = None
        if apply_pose:
            assert pose_latents is not None
            pose_freqs = self._pose_rope(
                (frame_count - 1, grid_height, grid_width),
                x,
                generation_width=grid_width,
            )

        time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.time_freq_dim, timesteps.flatten()).to(x.dtype)
        ).reshape(timesteps.shape[0], -1, self.config.hidden_size)
        projected_time = self.time_projection(time).unflatten(2, (6, self.config.hidden_size))
        pose_time: torch.Tensor | None = None
        if apply_pose:
            pose_time_embedding = self.time_embedding(
                sinusoidal_embedding_1d(
                    self.config.time_freq_dim,
                    torch.ones_like(timesteps.flatten()),
                ).to(x.dtype)
            ).reshape(timesteps.shape[0], -1, self.config.hidden_size)
            pose_time = self.time_projection(pose_time_embedding).unflatten(
                2, (6, self.config.hidden_size)
            )

        generation_context = self.text_embedding(context)
        generation_image_rows: int | None = None
        if vision is not None:
            assert self.img_emb is not None
            projected_vision = self.img_emb(vision)
            generation_image_rows = projected_vision.shape[1]
            generation_context = torch.cat((projected_vision, generation_context), dim=1)

        pose_projected_context: torch.Tensor | None = None
        pose_image_rows: int | None = None

        def initialize_pose_branch() -> None:
            nonlocal pose_image_rows, pose_input, pose_projected_context
            assert pose_latents is not None
            projected_pose = self.patch_embedding(
                torch.cat(
                    (
                        pose_latents,
                        torch.ones_like(pose_latents[:, :4]),
                        pose_latents,
                    ),
                    dim=1,
                ).float()
            ).to(x.dtype)
            pose_input = projected_pose.flatten(2).transpose(1, 2)
            pose_projected_context = self.text_embedding(effective_pose_context)
            if effective_pose_vision is not None:
                assert self.img_emb is not None
                projected_pose_vision = self.img_emb(effective_pose_vision)
                pose_image_rows = projected_pose_vision.shape[1]
                assert pose_projected_context is not None
                pose_projected_context = torch.cat(
                    (projected_pose_vision, pose_projected_context), dim=1
                )

        if apply_pose and not cached:
            initialize_pose_branch()

        buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if apply_pose:
            head_width = self.config.hidden_size // self.config.num_heads
            buffers = (
                x.new_empty(
                    x.shape[0],
                    x.shape[1] + frame_rows,
                    self.config.num_heads,
                    head_width,
                ),
                x.new_empty(
                    x.shape[0],
                    x.shape[1] + frame_rows,
                    self.config.num_heads,
                    head_width,
                ),
                x.new_empty(x.shape),
            )

        for index, raw_block in enumerate(self.blocks):
            block = cast("WanAnimate2Block", raw_block)
            pose_key: torch.Tensor | None = None
            pose_value: torch.Tensor | None = None
            if apply_pose:
                assert pose_time is not None
                assert pose_freqs is not None
                if cached:
                    assert active_pose_cache is not None
                    cached_input: torch.Tensor | None = None
                    try:
                        cached_input = active_pose_cache.take(index, x.device, x.dtype, x.shape[0])
                        if index + 1 < len(self.blocks) and not active_pose_cache.prefetch(
                            index + 1, x.device, x.dtype
                        ):
                            raise _PoseCacheRestoreError(
                                "Animate2 pose cache cannot prefetch within the memory bounds"
                            )
                    except (_PoseCacheRestoreError, torch.OutOfMemoryError):
                        # The active traceback references the take/prefetch
                        # frames' staging tensors until the handler exits, so
                        # the cache release and pose recompute run after it.
                        cached_input = None
                    if cached_input is None:
                        active_pose_cache.free()
                        active_pose_cache = None
                        cached = False
                        initialize_pose_branch()
                        assert pose_input is not None
                        assert pose_projected_context is not None
                        for previous_raw_block in self.blocks[:index]:
                            previous_block = cast("WanAnimate2Block", previous_raw_block)
                            pose_input, _, _ = previous_block.forward_pose(
                                pose_input,
                                pose_time,
                                pose_freqs,
                                pose_projected_context,
                                pose_image_rows,
                            )
                    else:
                        pose_key, pose_value = block.pose_kv(cached_input, pose_time, pose_freqs)
                if not cached:
                    assert pose_input is not None
                    assert pose_projected_context is not None
                    if active_pose_cache is not None:
                        active_pose_cache.put(index, pose_input)
                    pose_input, pose_key, pose_value = block.forward_pose(
                        pose_input,
                        pose_time,
                        pose_freqs,
                        pose_projected_context,
                        pose_image_rows,
                    )
            if pose_value is not None and pose_strength != 1.0:
                pose_value = pose_value * pose_strength
            x = block.forward_generation(
                x,
                projected_time,
                freqs,
                generation_context,
                generation_image_rows,
                pose_key,
                pose_value,
                frame_count=frame_count,
                frame_rows=frame_rows,
                buffers=buffers,
                reference_strength=reference_strength,
            )

        x = self.head(x, time)
        batch = x.shape[0]
        patch_t, patch_h, patch_w = self.config.patch_size
        x = x.view(
            batch,
            *grid,
            patch_t,
            patch_h,
            patch_w,
            self.config.out_channels,
        )
        x = torch.einsum("bthwpqrc->bctphqwr", x)
        x = x.reshape(
            batch,
            self.config.out_channels,
            grid[0] * patch_t,
            grid[1] * patch_h,
            grid[2] * patch_w,
        )
        return x[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]


__all__ = [
    "PoseBranchCache",
    "PoseCacheDType",
    "WanAnimate2Block",
    "WanAnimate2Model",
    "WanAnimate2SelfAttention",
]
