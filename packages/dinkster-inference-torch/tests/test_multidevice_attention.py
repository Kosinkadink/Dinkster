"""Direct one-conditioning multi-device attention execution proofs."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import FluxConfig
from dinkster_inference_torch import AttentionKernel, Flux, select_attention
from dinkster_inference_torch.multidevice_attention import (
    MultiDeviceAttentionClosedError,
    MultiDeviceAttentionError,
    MultiDeviceAttentionKernel,
)

CPU_DEVICES = (torch.device("cpu", 0), torch.device("cpu", 1))


def tensors(
    *,
    batch: int = 2,
    heads: int = 4,
    q_seq: int = 3,
    kv_seq: int = 5,
    qk_dim: int = 6,
    v_dim: int = 7,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    return (
        torch.randn(batch, heads, q_seq, qk_dim, generator=generator, requires_grad=requires_grad),
        torch.randn(batch, heads, kv_seq, qk_dim, generator=generator, requires_grad=requires_grad),
        torch.randn(batch, heads, kv_seq, v_dim, generator=generator, requires_grad=requires_grad),
    )


class RecordingKernel:
    def __init__(self, kernel: AttentionKernel | None = None) -> None:
        self.kernel = kernel
        self.calls: list[tuple[torch.device, tuple[int, ...], bool]] = []
        self.lock = threading.Lock()

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
        assert mask is None
        assert causal is False
        assert scale is None
        assert enable_gqa is False
        with self.lock:
            self.calls.append((q.device, tuple(q.shape), torch.is_grad_enabled()))
        if self.kernel is not None:
            return self.kernel(
                q,
                k,
                v,
                mask=mask,
                causal=causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


def test_constructor_snapshots_devices_and_strictly_refuses_invalid_inputs() -> None:
    devices = list(CPU_DEVICES)
    delegate = RecordingKernel()
    kernel = MultiDeviceAttentionKernel(delegate, devices)
    devices[:] = [torch.device("meta", 0)]
    try:
        output = kernel(*tensors(heads=2))
        assert output.device.type == "cpu"
        assert [shape[1] for _, shape, _ in delegate.calls] == [1, 1]
    finally:
        kernel.close()

    invalid_devices: tuple[object, ...] = (
        (),
        (CPU_DEVICES[0],),
        (CPU_DEVICES[0], CPU_DEVICES[0]),
        ("cpu:0", CPU_DEVICES[1]),
        (torch.device("cpu"), CPU_DEVICES[1]),
        (torch.device("meta", 0), CPU_DEVICES[1]),
    )
    for invalid in invalid_devices:
        with pytest.raises((TypeError, MultiDeviceAttentionError)):
            MultiDeviceAttentionKernel(delegate, cast(Any, invalid))


def test_constructor_does_not_probe_or_allocate_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("construction must not probe or allocate")

    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch, "zeros", unexpected)
    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(torch.cuda, "device_count", unexpected)
    MultiDeviceAttentionKernel(RecordingKernel(), CPU_DEVICES).close()


@pytest.mark.parametrize(
    ("mutation", "kwargs"),
    [
        ("non_tensor", {}),
        ("rank", {}),
        ("integer", {}),
        ("dtype", {}),
        ("device", {}),
        ("meta_owner", {}),
        ("batch", {}),
        ("zero_batch", {}),
        ("heads", {}),
        ("zero_heads", {}),
        ("one_head", {}),
        ("head_dim", {}),
        ("sequence", {}),
        ("mask", {"mask": torch.ones(1)}),
        ("causal", {"causal": True}),
        ("scale", {"scale": 0.5}),
        ("gqa", {"enable_gqa": True}),
    ],
)
def test_refusals_happen_before_lane_submission(
    monkeypatch: pytest.MonkeyPatch, mutation: str, kwargs: dict[str, object]
) -> None:
    import dinkster_inference_torch.multidevice_attention as module

    CountingExecutor.instances.clear()
    monkeypatch.setattr(module, "ThreadPoolExecutor", CountingExecutor)
    delegate = RecordingKernel()
    q, k, v = tensors()
    if mutation == "non_tensor":
        q = cast(torch.Tensor, object())
    elif mutation == "rank":
        q = q[0]
    elif mutation == "integer":
        q = q.to(torch.int64)
    elif mutation == "dtype":
        q = q.double()
    elif mutation == "device":
        q = q.to("meta")
    elif mutation == "meta_owner":
        q, k, v = q.to("meta"), k.to("meta"), v.to("meta")
    elif mutation == "batch":
        q = q[:1]
    elif mutation == "zero_batch":
        q, k, v = q[:0], k[:0], v[:0]
    elif mutation == "heads":
        k = k[:, :-1]
    elif mutation == "zero_heads":
        q, k, v = q[:, :0], k[:, :0], v[:, :0]
    elif mutation == "one_head":
        q, k, v = q[:, :1], k[:, :1], v[:, :1]
    elif mutation == "head_dim":
        k = k[..., :-1]
    elif mutation == "sequence":
        v = v[:, :, :-1]

    with MultiDeviceAttentionKernel(cast(AttentionKernel, delegate), CPU_DEVICES) as kernel:
        with pytest.raises((TypeError, MultiDeviceAttentionError)):
            kernel(q, k, v, **cast(Any, kwargs))
    assert delegate.calls == []
    assert CountingExecutor.instances[0].futures == []


@pytest.mark.parametrize("invalid_output", ("foreign", "shape", "dtype", "device"))
def test_invalid_lane_results_refuse_after_all_lanes_settle(invalid_output: str) -> None:
    barrier = threading.Barrier(2)
    completed = 0
    lock = threading.Lock()

    def delegate(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        nonlocal completed
        barrier.wait(timeout=5)
        with lock:
            completed += 1
        if invalid_output == "foreign":
            return cast(torch.Tensor, object())
        if invalid_output == "shape":
            return q[:, :, :-1]
        if invalid_output == "dtype":
            return q.double()
        return q.to("meta")

    q = torch.randn(1, 2, 2, 3)
    with MultiDeviceAttentionKernel(cast(AttentionKernel, delegate), CPU_DEVICES) as kernel:
        with pytest.raises(MultiDeviceAttentionError):
            kernel(q, q, q)
    assert completed == 2


def test_lane_failure_prevents_gathering_successful_results() -> None:
    gather_attempts: list[tuple[object, ...]] = []

    class GatherTrackingTensor(torch.Tensor):
        @staticmethod
        def wrap(value: torch.Tensor) -> GatherTrackingTensor:
            return torch.Tensor._make_subclass(GatherTrackingTensor, value, value.requires_grad)

        def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            gather_attempts.append(args)
            return super().to(*args, **kwargs)

    def delegate(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        lane = int(q[0, 0, 0, 0].item()) // 2
        if lane == 1:
            raise LookupError("lane 1 failed")
        return GatherTrackingTensor.wrap(q)

    q = torch.stack([torch.full((1, 1, 1), float(head)) for head in range(4)], dim=1)
    with MultiDeviceAttentionKernel(cast(AttentionKernel, delegate), CPU_DEVICES) as kernel:
        with pytest.raises(LookupError, match="lane 1 failed"):
            kernel(q, q, q)

    assert gather_attempts == []


def test_calls_existing_planner_and_routes_balanced_contiguous_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference_torch.multidevice_attention as module

    real_plan = module.plan_lanes
    planned: list[tuple[tuple[int, ...], int]] = []

    def spy(items: tuple[int, ...], lane_count: int):  # noqa: ANN202 - private test seam
        planned.append((tuple(items), lane_count))
        return real_plan(items, lane_count)

    monkeypatch.setattr(module, "plan_lanes", spy)
    delegate = RecordingKernel()
    with MultiDeviceAttentionKernel(delegate, CPU_DEVICES) as kernel:
        kernel(*tensors(heads=5))

    assert planned == [((0, 1, 2, 3, 4), 2)]
    assert sorted(shape[1] for _, shape, _ in delegate.calls) == [2, 3]


def test_two_lanes_start_before_either_finishes_and_gather_global_order() -> None:
    barrier = threading.Barrier(3)
    release = threading.Event()
    completed: list[int] = []
    lock = threading.Lock()

    def delegate(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        lane = int(q[0, 0, 0, 0].item())
        barrier.wait(timeout=5)
        assert release.wait(timeout=5)
        if lane == 0:
            time.sleep(0.05)
        with lock:
            completed.append(lane)
        return q + 100

    q = torch.stack([torch.full((1, 2, 1), float(head)) for head in range(4)], dim=1)
    k = q.clone()
    v = q.clone()
    with MultiDeviceAttentionKernel(cast(AttentionKernel, delegate), CPU_DEVICES) as kernel:
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(kernel, q, k, v)
            barrier.wait(timeout=5)
            assert completed == []
            release.set()
            output = future.result(timeout=5)

    assert completed == [2, 0]
    torch.testing.assert_close(output, q + 100)
    assert output.device == q.device


class CountingFuture:
    def __init__(self, delegate: Future[torch.Tensor]) -> None:
        self.delegate = delegate
        self.result_calls = 0

    def result(self) -> torch.Tensor:
        self.result_calls += 1
        return self.delegate.result()


class CountingExecutor:
    instances: list[CountingExecutor] = []

    def __init__(self, max_workers: int, *, thread_name_prefix: str) -> None:
        self.delegate = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )
        self.futures: list[CountingFuture] = []
        self.shutdown_calls = 0
        self.__class__.instances.append(self)

    def submit(self, function: Any, *args: object) -> CountingFuture:
        future = CountingFuture(self.delegate.submit(function, *args))
        self.futures.append(future)
        return future

    def shutdown(self, *, wait: bool) -> None:
        self.shutdown_calls += 1
        self.delegate.shutdown(wait=wait)


@pytest.mark.parametrize("failed_lane", (None, 0, 1))
def test_every_submitted_lane_settles_exactly_once_and_kernel_stays_reusable(
    monkeypatch: pytest.MonkeyPatch, failed_lane: int | None
) -> None:
    import dinkster_inference_torch.multidevice_attention as module

    CountingExecutor.instances.clear()
    monkeypatch.setattr(module, "ThreadPoolExecutor", CountingExecutor)
    barrier = threading.Barrier(2)
    call_count = 0
    call_lock = threading.Lock()

    def delegate(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        nonlocal call_count
        lane = int(q[0, 0, 0, 0].item()) // 2
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call <= 2:
            barrier.wait(timeout=5)
            if lane == failed_lane:
                raise LookupError(f"lane {lane} failed")
        return q

    q = torch.stack([torch.full((1, 1, 1), float(head)) for head in range(4)], dim=1)
    kernel = MultiDeviceAttentionKernel(cast(AttentionKernel, delegate), CPU_DEVICES)
    if failed_lane is None:
        torch.testing.assert_close(kernel(q, q, q), q)
    else:
        with pytest.raises(LookupError, match=f"lane {failed_lane} failed"):
            kernel(q, q, q)
        kernel(q, q, q)
    kernel.close()
    kernel.close()

    executor = CountingExecutor.instances[0]
    assert all(future.result_calls == 1 for future in executor.futures)
    assert executor.shutdown_calls == 1


def test_deterministic_lane_failure_and_concurrent_call_refusal() -> None:
    barrier = threading.Barrier(3)
    release = threading.Event()

    def delegate(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        lane = int(q[0, 0, 0, 0].item()) // 2
        barrier.wait(timeout=5)
        assert release.wait(timeout=5)
        raise LookupError(f"lane {lane} failed")

    q = torch.stack([torch.full((1, 1, 1), float(head)) for head in range(4)], dim=1)
    with MultiDeviceAttentionKernel(cast(AttentionKernel, delegate), CPU_DEVICES) as kernel:
        with ThreadPoolExecutor(max_workers=1) as caller:
            running = caller.submit(kernel, q, q, q)
            barrier.wait(timeout=5)
            with pytest.raises(MultiDeviceAttentionError, match="already executing"):
                kernel(q, q, q)
            release.set()
            with pytest.raises(LookupError, match="lane 0 failed") as raised:
                running.result(timeout=5)
    assert raised.value.__notes__ == ["lane 1 also failed: LookupError('lane 1 failed')"]


def test_close_waits_for_started_work_is_idempotent_and_post_close_refuses() -> None:
    started = threading.Event()
    release = threading.Event()

    def delegate(
        q: torch.Tensor,
        _k: torch.Tensor,
        v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        started.set()
        assert release.wait(timeout=5)
        return v[:, :, : q.shape[2]]

    q, k, v = tensors(heads=2)
    kernel = MultiDeviceAttentionKernel(cast(AttentionKernel, delegate), CPU_DEVICES)
    with ThreadPoolExecutor(max_workers=2) as callers:
        call = callers.submit(kernel, q, k, v)
        assert started.wait(timeout=5)
        closing = callers.submit(kernel.close)
        time.sleep(0.05)
        assert not closing.done()
        release.set()
        call.result(timeout=5)
        closing.result(timeout=5)
    kernel.close()
    with pytest.raises(MultiDeviceAttentionClosedError):
        kernel(q, k, v)
    assert not any(
        thread.name.startswith("dinkster-multidevice-attention") for thread in threading.enumerate()
    )


def test_lane_local_close_refuses_without_corrupting_external_close() -> None:
    barrier = threading.Barrier(2)
    completed = 0
    lock = threading.Lock()
    kernel: MultiDeviceAttentionKernel | None = None

    def delegate(
        _q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        nonlocal completed
        barrier.wait(timeout=5)
        try:
            assert kernel is not None
            kernel.close()
        finally:
            with lock:
                completed += 1
        raise AssertionError("lane-local close must refuse")

    q, k, v = tensors(heads=2)
    kernel = MultiDeviceAttentionKernel(cast(AttentionKernel, delegate), CPU_DEVICES)
    with pytest.raises(MultiDeviceAttentionError, match="cannot close"):
        kernel(q, k, v)
    assert completed == 2
    kernel.close()
    assert not any(
        thread.name.startswith("dinkster-multidevice-attention") for thread in threading.enumerate()
    )


@pytest.mark.parametrize("grad_enabled", (True, False))
def test_worker_threads_preserve_caller_grad_mode(grad_enabled: bool) -> None:
    delegate = RecordingKernel()
    with MultiDeviceAttentionKernel(delegate, CPU_DEVICES) as kernel:
        with torch.set_grad_enabled(grad_enabled):
            kernel(*tensors(heads=2, requires_grad=grad_enabled))
    assert [enabled for _, _, enabled in delegate.calls] == [grad_enabled, grad_enabled]


def test_backward_crosses_transfer_kernel_and_gather() -> None:
    q, k, v = tensors(heads=4, requires_grad=True)
    with MultiDeviceAttentionKernel(select_attention("flux", "sdpa").kernel, CPU_DEVICES) as kernel:
        output = kernel(q, k, v)
        output.square().sum().backward()
    assert q.grad is not None and torch.count_nonzero(q.grad)
    assert k.grad is not None and torch.count_nonzero(k.grad)
    assert v.grad is not None and torch.count_nonzero(v.grad)


def test_reduced_flux_consumes_direct_kernel_for_every_attention_call() -> None:
    config = FluxConfig(
        in_channels=2,
        out_channels=2,
        vec_in_dim=4,
        context_in_dim=4,
        hidden_size=12,
        depth=1,
        depth_single_blocks=1,
        num_heads=2,
        axes_dim=(2, 2, 2),
        patch_size=1,
        guidance_embed=False,
    )
    delegate = RecordingKernel()
    with MultiDeviceAttentionKernel(delegate, CPU_DEVICES) as kernel:
        model = Flux(config, attention_kernel=kernel)
        output = model(
            torch.randn(1, 2, 2, 2),
            torch.tensor([0.5]),
            torch.randn(1, 3, 4),
            torch.randn(1, 4),
        )
    assert output.shape == (1, 2, 2, 2)
    assert len(delegate.calls) == 4
    assert all(shape[1] == 1 for _, shape, _ in delegate.calls)


def test_source_stays_in_direct_attention_lane() -> None:
    repo = Path(__file__).parents[3]
    source = (
        repo
        / "packages/dinkster-inference-torch/src/dinkster_inference_torch/multidevice_attention.py"
    )
    text = source.read_text()
    forbidden = ("dinkster_protocol", "CFG", "workgroup", "process", "session", "reservation")
    assert all(token not in text for token in forbidden)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_cuda_two_devices_match_owner_sdpa_and_backward() -> None:
    devices = (torch.device("cuda", 0), torch.device("cuda", 1))
    q, k, v = (
        tensor.to(devices[0]).detach().requires_grad_(True)
        for tensor in tensors(batch=1, heads=4, qk_dim=8, v_dim=8)
    )
    baseline_q = q.detach().clone().requires_grad_(True)
    baseline_k = k.detach().clone().requires_grad_(True)
    baseline_v = v.detach().clone().requires_grad_(True)
    selected_sdpa = select_attention("flux", "sdpa").kernel
    delegate = RecordingKernel(selected_sdpa)
    with MultiDeviceAttentionKernel(delegate, devices) as kernel:
        actual = kernel(q, k, v)
    expected = selected_sdpa(baseline_q, baseline_k, baseline_v)
    # Head partitioning can select a different SDPA reduction schedule on each device.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(q.grad, baseline_q.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k.grad, baseline_k.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(v.grad, baseline_v.grad, rtol=1e-5, atol=1e-6)
    assert {device for device, _, _ in delegate.calls} == set(devices)
    assert actual.device == devices[0]
    assert actual.shape == (1, 4, 3, 8)
    assert actual.dtype == torch.float32
