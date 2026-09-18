"""Direct head-partitioned attention execution across explicit devices."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Self, cast

import torch
from dinkster_inference import LaneAssignment, plan_lanes

from .attention import AttentionKernel


class MultiDeviceAttentionError(ValueError):
    """The multi-device kernel cannot execute the requested call."""


class MultiDeviceAttentionClosedError(MultiDeviceAttentionError):
    """The multi-device kernel has been closed."""


def _snapshot_devices(devices: Sequence[torch.device]) -> tuple[torch.device, ...]:
    raw_snapshot = tuple(cast(Sequence[object], devices))
    if len(raw_snapshot) < 2:
        raise MultiDeviceAttentionError("at least two explicit devices are required")
    snapshot: list[torch.device] = []
    for device in raw_snapshot:
        if not isinstance(device, torch.device):
            raise TypeError("devices must contain only torch.device values")
        index = cast(int | None, device.index)
        if device.type not in ("cpu", "cuda") or index is None:
            raise MultiDeviceAttentionError(
                "each device must be an explicitly indexed CPU or CUDA device"
            )
        snapshot.append(device)
    if len(set(snapshot)) != len(snapshot):
        raise MultiDeviceAttentionError("devices must be unique")
    return tuple(snapshot)


def _validate_call(
    q: object,
    k: object,
    v: object,
    *,
    mask: object,
    causal: object,
    scale: object,
    enable_gqa: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise MultiDeviceAttentionError(f"{name} must have rank 4 [B, H, S, D]")
    if mask is not None:
        raise MultiDeviceAttentionError("mask is not supported")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if causal:
        raise MultiDeviceAttentionError("causal attention is not supported")
    if scale is not None:
        raise MultiDeviceAttentionError("explicit scale is not supported")
    if not isinstance(enable_gqa, bool):
        raise TypeError("enable_gqa must be a bool")
    if enable_gqa:
        raise MultiDeviceAttentionError("GQA is not supported")

    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    if not q.is_floating_point() or not k.is_floating_point() or not v.is_floating_point():
        raise MultiDeviceAttentionError("q, k, and v must use floating-point dtypes")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise MultiDeviceAttentionError("q, k, and v must share one dtype")
    if q.device != k.device or q.device != v.device:
        raise MultiDeviceAttentionError("q, k, and v must share one semantic-owner device")
    if q.device.type not in ("cpu", "cuda"):
        raise MultiDeviceAttentionError("the semantic-owner device must be CPU or CUDA")
    if q.shape[0] <= 0 or q.shape[1] <= 0:
        raise MultiDeviceAttentionError("q, k, and v must have positive batch and head counts")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise MultiDeviceAttentionError("q, k, and v must share one batch size")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise MultiDeviceAttentionError("q, k, and v must share one head count")
    if q.shape[1] < 2:
        raise MultiDeviceAttentionError("at least two heads are required")
    if q.shape[3] != k.shape[3]:
        raise MultiDeviceAttentionError("q and k head dimensions must match")
    if k.shape[2] != v.shape[2]:
        raise MultiDeviceAttentionError("k and v sequence lengths must match")
    return q, k, v


class MultiDeviceAttentionKernel:
    """Partition one rank-4 attention call by heads over explicit devices."""

    def __init__(
        self,
        kernel: AttentionKernel,
        devices: Sequence[torch.device],
    ) -> None:
        if not callable(kernel):
            raise TypeError("kernel must be callable")
        self._kernel = kernel
        self._devices = _snapshot_devices(devices)
        self._executor = ThreadPoolExecutor(
            max_workers=len(self._devices),
            thread_name_prefix="dinkster-multidevice-attention",
        )
        self._state_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._cuda_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._worker_state = threading.local()
        self._closed_event = threading.Event()
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self._worker_state, "active", False):
            raise MultiDeviceAttentionError("cannot close the kernel from one of its lanes")
        with self._state_lock:
            if self._closed:
                closed_event = self._closed_event
                owns_shutdown = False
            else:
                self._closed = True
                closed_event = self._closed_event
                owns_shutdown = True
        if not owns_shutdown:
            closed_event.wait()
            return
        try:
            self._executor.shutdown(wait=True)
        finally:
            closed_event.set()

    def _run_lane(
        self,
        assignment: LaneAssignment[int],
        device: torch.device,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grad_enabled: bool,
        input_ready: torch.cuda.Event | None,
    ) -> tuple[
        torch.Tensor,
        torch.cuda.Event | None,
        torch.cuda.Stream | None,
    ]:
        self._worker_state.active = True
        try:
            start = assignment.items[0].index
            stop = assignment.items[-1].index + 1
            stream: torch.cuda.Stream | None = None
            with torch.set_grad_enabled(grad_enabled):
                if device.type == "cuda":
                    with self._stream_lock:
                        stream = self._cuda_streams.get(device)
                        if stream is None:
                            stream = torch.cuda.Stream(device=device)
                            self._cuda_streams[device] = stream
                    with torch.cuda.device(device), torch.cuda.stream(stream):
                        if input_ready is not None:
                            stream.wait_event(input_ready)
                        lane_q = q[:, start:stop].to(device, non_blocking=True)
                        lane_k = k[:, start:stop].to(device, non_blocking=True)
                        lane_v = v[:, start:stop].to(device, non_blocking=True)
                        output_object = cast(object, self._kernel(lane_q, lane_k, lane_v))
                else:
                    lane_q = q[:, start:stop].to(device)
                    lane_k = k[:, start:stop].to(device)
                    lane_v = v[:, start:stop].to(device)
                    output_object = cast(object, self._kernel(lane_q, lane_k, lane_v))
                expected_shape = (q.shape[0], stop - start, q.shape[2], v.shape[3])
                if not isinstance(output_object, torch.Tensor):
                    raise MultiDeviceAttentionError(
                        f"lane {assignment.index} kernel must return a torch.Tensor"
                    )
                output = output_object
                if tuple(output.shape) != expected_shape:
                    raise MultiDeviceAttentionError(
                        f"lane {assignment.index} returned shape {tuple(output.shape)}, "
                        f"expected {expected_shape}"
                    )
                if output.dtype != q.dtype:
                    raise MultiDeviceAttentionError(
                        f"lane {assignment.index} returned dtype {output.dtype}, expected {q.dtype}"
                    )
                if output.device.type != device.type or (
                    device.type == "cuda" and output.device != device
                ):
                    raise MultiDeviceAttentionError(
                        f"lane {assignment.index} returned device {output.device}, "
                        f"expected {device}"
                    )
                if device.type == "cuda":
                    complete = torch.cuda.Event()
                    assert stream is not None
                    complete.record(stream)
                    return output, complete, stream
                return output, None, None
        finally:
            del self._worker_state.active

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if not self._call_lock.acquire(blocking=False):
            raise MultiDeviceAttentionError("kernel is already executing a call")
        try:
            with self._state_lock:
                if self._closed:
                    raise MultiDeviceAttentionClosedError("kernel is closed")
            q, k, v = _validate_call(
                q,
                k,
                v,
                mask=mask,
                causal=causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
            assignments = plan_lanes(tuple(range(q.shape[1])), len(self._devices))
            owner = q.device
            grad_enabled = torch.is_grad_enabled()
            input_ready: torch.cuda.Event | None = None
            owner_stream: torch.cuda.Stream | None = None
            if owner.type == "cuda":
                owner_stream = torch.cuda.current_stream(owner)
                input_ready = torch.cuda.Event()
                input_ready.record(owner_stream)
            futures: list[
                tuple[
                    int,
                    Future[
                        tuple[
                            torch.Tensor,
                            torch.cuda.Event | None,
                            torch.cuda.Stream | None,
                        ]
                    ],
                ]
            ] = []
            submission_failure: tuple[int, BaseException] | None = None
            with self._state_lock:
                if self._closed:
                    raise MultiDeviceAttentionClosedError("kernel is closed")
                for assignment in assignments:
                    try:
                        future = self._executor.submit(
                            self._run_lane,
                            assignment,
                            self._devices[assignment.index],
                            q,
                            k,
                            v,
                            grad_enabled,
                            input_ready,
                        )
                    except BaseException as error:
                        submission_failure = (assignment.index, error)
                        break
                    futures.append((assignment.index, future))

            lane_outcomes: list[
                tuple[
                    torch.Tensor,
                    torch.cuda.Event | None,
                    torch.cuda.Stream | None,
                ]
            ] = []
            failures: list[tuple[int, BaseException]] = []
            for lane_index, future in futures:
                try:
                    lane_outcomes.append(future.result())
                except BaseException as error:
                    failures.append((lane_index, error))
            if submission_failure is not None:
                failures.append(submission_failure)
            if failures:
                failures.sort(key=lambda item: item[0])
                _, primary = failures[0]
                for lane_index, error in failures[1:]:
                    primary.add_note(f"lane {lane_index} also failed: {error!r}")
                raise primary
            results: list[torch.Tensor] = []
            lane_results: list[tuple[torch.Tensor, torch.cuda.Stream]] = []
            try:
                for result, completion, creation_stream in lane_outcomes:
                    if completion is not None:
                        if owner_stream is None:
                            completion.synchronize()
                        else:
                            owner_stream.wait_event(completion)
                    results.append(result.to(owner, non_blocking=owner_stream is not None))
                    if creation_stream is not None:
                        lane_results.append((result, creation_stream))
                output = torch.cat(results, dim=1)
                expected_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[3])
                if tuple(output.shape) != expected_shape:
                    raise MultiDeviceAttentionError(
                        f"gather returned shape {tuple(output.shape)}, expected {expected_shape}"
                    )
                if output.dtype != q.dtype or output.device != owner:
                    raise MultiDeviceAttentionError(
                        "gather must preserve the input dtype and semantic-owner device"
                    )
                return output
            finally:
                if owner_stream is not None and lane_results:
                    consumed = torch.cuda.Event()
                    consumed.record(owner_stream)
                    for _, creation_stream in lane_results:
                        creation_stream.wait_event(consumed)
        finally:
            self._call_lock.release()


__all__ = [
    "MultiDeviceAttentionClosedError",
    "MultiDeviceAttentionError",
    "MultiDeviceAttentionKernel",
]
