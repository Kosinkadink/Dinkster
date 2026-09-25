"""CPU proof for the typed attention selector and its kernels."""

from __future__ import annotations

import gc
import importlib.metadata
import subprocess
import sys
import threading
import weakref
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from dinkster_inference import (
    AttentionModifierSchedule,
    SamplingParameterCurve,
    SamplingTimelineSchedule,
    realize_sampling_timeline,
)
from dinkster_inference.sampling_timeline import use_realized_sampling_timeline
from dinkster_inference_torch import (
    ATTENTION_ADAPTER_CONTRACT,
    SAGE2_PROVIDER,
    SOL_ATTENTION_PROVIDER,
    AttentionKernel,
    AttentionPolicy,
    AttentionRole,
    AttentionSelectionError,
    AttentionTensorLease,
    AttentionValidationError,
    QkvConsumingAttentionKernel,
    attention_provider_identity,
    select_attention,
)
from dinkster_inference_torch import attention as attention_module
from dinkster_protocol import AttentionCapabilityEvidence, AttentionRoute, AttentionRouteToken
from torch.nn.attention import SDPBackend, sdpa_kernel

ROLES: tuple[AttentionRole, ...] = ("unet", "flux", "vae", "clip", "t5", "qwen")


def _capabilities(*, device_kind: str = "cpu", sage: bool = False) -> AttentionCapabilityEvidence:
    version = str(torch.__version__)
    providers = (("torch", version),)
    policies: tuple[AttentionPolicy, ...] = ("sdpa",)
    if sage:
        policies = (*policies, "sage")
        providers = (("sageattention", "2.2.0"), *providers)
    if device_kind == "rocm":
        providers = (("hip", "7.0"), *providers)
    return AttentionCapabilityEvidence(
        version=1,
        available_policies=policies,
        provider_versions=providers,
        adapter_contract_revision=ATTENTION_ADAPTER_CONTRACT,
        device_kind=device_kind,
        device_sm=120 if device_kind in ("cuda", "rocm") else None,
        sdpa_torch_runtime=version.split("+")[0],
    )


def _ignore_device(_device: torch.device) -> None:
    pass


def _bounded_steps_one(_q: torch.Tensor, _k: torch.Tensor) -> int:
    return 1


def _bounded_steps_two(_q: torch.Tensor, _k: torch.Tensor) -> int:
    return 2


def _bounded_steps_three(_q: torch.Tensor, _k: torch.Tensor) -> int:
    return 3


def _bounded_steps_max(_q: torch.Tensor, _k: torch.Tensor) -> int:
    return 128


def test_only_attention_adapter_calls_torch_sdpa_directly() -> None:
    source = Path(__file__).parents[1] / "src" / "dinkster_inference_torch"
    direct = {
        path.name: path.read_text().count("scaled_dot_product_attention(")
        for path in source.glob("*.py")
        if "scaled_dot_product_attention(" in path.read_text()
    }
    assert direct == {"attention.py": 1}


def tensors(
    *, batch: int = 2, q_heads: int = 3, kv_heads: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(123)
    kv_heads = q_heads if kv_heads is None else kv_heads
    return (
        torch.randn(batch, q_heads, 4, 5),
        torch.randn(batch, kv_heads, 6, 5),
        torch.randn(batch, kv_heads, 6, 7),
    )


def kernel() -> AttentionKernel:
    return select_attention("unet", "sdpa").kernel


def assert_matches_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> None:
    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    actual = kernel()(
        q,
        k,
        v,
        mask=mask,
        causal=causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    torch.testing.assert_close(actual, expected)


def test_noncausal_matches_direct_sdpa() -> None:
    assert_matches_direct(*tensors())


def test_causal_matches_direct_sdpa() -> None:
    q, k, v = tensors()
    assert_matches_direct(q, k, v, causal=True)


def test_boolean_mask_matches_direct_sdpa() -> None:
    q, k, v = tensors()
    mask = torch.tensor([[[[True, True, False, True, False, True]]]], dtype=torch.bool)
    assert_matches_direct(q, k, v, mask=mask)


def test_additive_mask_matches_direct_sdpa() -> None:
    q, k, v = tensors()
    mask = torch.tensor([[[[0.0, -torch.inf, 0.0, 0.0, -torch.inf, 0.0]]]])
    assert_matches_direct(q, k, v, mask=mask)


def test_explicit_scale_matches_direct_sdpa() -> None:
    q, k, v = tensors()
    assert_matches_direct(q, k, v, scale=0.125)
    assert_matches_direct(q, k, v, scale=2)


def test_gqa_matches_direct_sdpa() -> None:
    q, k, v = tensors(q_heads=4, kv_heads=2)
    assert_matches_direct(q, k, v, enable_gqa=True)


def sdpa_dtype_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[torch.dtype, torch.dtype | None]]:
    recorded: list[tuple[torch.dtype, torch.dtype | None]] = []
    real_sdpa = F.scaled_dot_product_attention

    def recording_sdpa(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        mask = kwargs.get("attn_mask")
        recorded.append((q.dtype, None if mask is None else mask.dtype))
        return real_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
    return recorded


def test_math_sdp_reduced_precision_reduction_enabled_at_import() -> None:
    assert attention_module._FP16_BF16_REDUCTION_MATH_SDP is True  # pyright: ignore[reportPrivateUsage]
    assert torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed() is True


def test_math_sdp_reduction_enable_applies_flag() -> None:
    enable = attention_module._enable_fp16_bf16_reduction_math_sdp  # pyright: ignore[reportPrivateUsage]
    calls: list[bool] = []
    assert enable(calls.append) is True
    assert calls == [True]


def test_math_sdp_reduction_enable_tolerates_missing_knob() -> None:
    enable = attention_module._enable_fp16_bf16_reduction_math_sdp  # pyright: ignore[reportPrivateUsage]

    def raising(_: bool) -> None:
        raise AttributeError("torch build without the math SDP knob")

    assert enable(raising) is False


def test_forced_upcast_runs_fp16_attention_in_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "_FORCE_FP16_ATTENTION_UPCAST", True)
    q, k, v = (t.to(torch.float16) for t in tensors())
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(torch.float16)
    recorded = sdpa_dtype_recorder(monkeypatch)
    actual = kernel()(q, k, v)
    assert recorded == [(torch.float32, None)]
    assert actual.dtype == torch.float16
    assert torch.equal(actual, expected)


def test_forced_upcast_widens_additive_fp16_mask_and_keeps_boolean_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_FORCE_FP16_ATTENTION_UPCAST", True)
    recorded = sdpa_dtype_recorder(monkeypatch)
    q, k, v = (t.to(torch.float16) for t in tensors())
    additive = torch.tensor([[[[0.0, -torch.inf, 0.0, 0.0, -torch.inf, 0.0]]]], dtype=torch.float16)
    boolean = torch.tensor([[[[True, True, False, True, False, True]]]], dtype=torch.bool)
    assert kernel()(q, k, v, mask=additive).dtype == torch.float16
    assert kernel()(q, k, v, mask=boolean).dtype == torch.float16
    assert recorded == [(torch.float32, torch.float32), (torch.float32, torch.bool)]


def test_forced_upcast_leaves_other_dtypes_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "_FORCE_FP16_ATTENTION_UPCAST", True)
    recorded = sdpa_dtype_recorder(monkeypatch)
    for dtype in (torch.bfloat16, torch.float32):
        q, k, v = (t.to(dtype) for t in tensors())
        assert kernel()(q, k, v).dtype == dtype
    assert recorded == [(torch.bfloat16, None), (torch.float32, None)]


def test_inactive_upcast_preserves_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "_FORCE_FP16_ATTENTION_UPCAST", False)
    recorded = sdpa_dtype_recorder(monkeypatch)
    q, k, v = (t.to(torch.float16) for t in tensors())
    assert kernel()(q, k, v).dtype == torch.float16
    assert recorded == [(torch.float16, None)]


def test_forced_upcast_applies_to_every_batch_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "_FORCE_FP16_ATTENTION_UPCAST", True)
    monkeypatch.setattr(attention_module, "SDP_BATCH_LIMIT", 1)
    q, k, v = (t.to(torch.float16) for t in tensors(batch=2))
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(torch.float16)
    recorded = sdpa_dtype_recorder(monkeypatch)
    actual = kernel()(q, k, v)
    assert recorded == [(torch.float32, None), (torch.float32, None)]
    assert actual.dtype == torch.float16
    assert torch.equal(actual, expected)


def test_large_attention_uses_comfy_backend_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[object], bool]] = []

    class RecordingContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def recording_factory(backends: list[object], *, set_priority: bool) -> RecordingContext:
        calls.append((backends, set_priority))
        return RecordingContext()

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", recording_factory)
    q, k, v = tensors(batch=2**12)
    assert_matches_direct(q, k, v)
    assert len(calls) == 1 and calls[0][1] is True
    assert tuple(cast("Any", backend).name for backend in calls[0][0]) == (
        "FLASH_ATTENTION",
        "CUDNN_ATTENTION",
        "EFFICIENT_ATTENTION",
        "MATH",
    )


def test_large_attention_context_reuses_backend_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class RecordingContext:
        def __enter__(self) -> None:
            calls.append("enter")

        def __exit__(self, *args: object) -> None:
            calls.append("exit")

    def recording_factory(*_args: object, **_kwargs: object) -> RecordingContext:
        calls.append("create")
        return RecordingContext()

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", recording_factory)
    q, k, v = tensors(batch=2**12)
    selected = kernel()
    with attention_module.attention_kernel_context(selected, q.numel(), device=q.device):
        selected(q, k, v)
        selected(q, k, v)

    assert calls == ["create", "enter", "exit"]


def test_priority_context_restores_cpu_order() -> None:
    before = torch._C._get_sdp_priority_order()  # pyright: ignore[reportPrivateUsage]
    expected = [
        int(backend)
        for backend in attention_module._SDPA_BACKEND_PRIORITY  # pyright: ignore[reportPrivateUsage]
    ]
    q, _, _ = tensors(batch=2**12)
    with attention_module.attention_kernel_context(kernel(), q.numel(), device=q.device):
        active = torch._C._get_sdp_priority_order()  # pyright: ignore[reportPrivateUsage]
        assert active[: len(expected)] == expected
    assert torch._C._get_sdp_priority_order() == before  # pyright: ignore[reportPrivateUsage]


def test_priority_initialization_reapplies_declared_order_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    fake = cast(torch.Tensor, SimpleNamespace(device=torch.device("cuda")))

    def choose(*args: object, **kwargs: object) -> int:
        calls.append(("choose", (args, kwargs)))
        return 3

    def get_order() -> list[int]:
        calls.append(("get", None))
        return [3, 1, 2, 0, 4]

    def set_order(order: list[int]) -> None:
        calls.append(("set", order))

    def fail_allocation(*args: object, **kwargs: object) -> torch.Tensor:
        pytest.fail("priority initialization must not allocate")

    monkeypatch.setattr(attention_module, "_sdpa_cuda_priority_initialized", False)
    monkeypatch.setattr(torch, "_fused_sdp_choice", choose)
    monkeypatch.setattr(torch, "empty", fail_allocation)
    monkeypatch.setattr(
        torch._C,  # pyright: ignore[reportPrivateUsage]
        "_get_sdp_priority_order",
        get_order,
    )
    monkeypatch.setattr(
        torch._C,  # pyright: ignore[reportPrivateUsage]
        "_set_sdp_priority_order",
        set_order,
    )

    initialize = attention_module._initialize_cuda_sdpa_priority  # pyright: ignore[reportPrivateUsage]
    initialize(fake, fake, fake, mask=None, causal=True, scale=0.25, enable_gqa=True)
    initialize(fake, fake, fake, mask=None, causal=True, scale=0.25, enable_gqa=True)

    assert calls[0] == (
        "choose",
        (
            (fake, fake, fake),
            {
                "attn_mask": None,
                "dropout_p": 0.0,
                "is_causal": True,
                "scale": 0.25,
                "enable_gqa": True,
            },
        ),
    )
    assert calls[1:] == [("get", None), ("set", [1, 3, 2, 0, 4])]


def test_priority_context_restores_complete_caller_order(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[object] = ["efficient", "math", "flash", "cudnn"]
    calls: list[str] = []

    class RecordingContext:
        def __init__(self, backends: list[object], *, set_priority: bool) -> None:
            self.backends = backends
            self.set_priority = set_priority

        def __enter__(self) -> None:
            assert self.set_priority
            calls.append("priority-enter")
            order[:] = self.backends

        def __exit__(self, *args: object) -> None:
            calls.append("priority-exit")

    def recording_factory(backends: list[object], *, set_priority: bool) -> RecordingContext:
        return RecordingContext(backends, set_priority=set_priority)

    def get_priority() -> list[object]:
        calls.append("get")
        return list(order)

    def set_priority(value: list[object]) -> None:
        calls.append("set")
        order[:] = value

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", recording_factory)
    monkeypatch.setattr(
        attention_module.torch._C,  # pyright: ignore[reportPrivateUsage]
        "_get_sdp_priority_order",
        get_priority,
    )
    monkeypatch.setattr(
        attention_module.torch._C,  # pyright: ignore[reportPrivateUsage]
        "_set_sdp_priority_order",
        set_priority,
    )
    with attention_module._sdpa_priority_context():  # pyright: ignore[reportPrivateUsage]
        calls.append("body")
        assert order == list(attention_module._SDPA_BACKEND_PRIORITY)  # pyright: ignore[reportPrivateUsage]

    assert order == ["efficient", "math", "flash", "cudnn"]
    assert calls == ["get", "priority-enter", "body", "priority-exit", "set"]


def test_priority_context_serializes_process_global_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def recording_factory(*args: object, **kwargs: object) -> nullcontext[None]:
        return nullcontext()

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", recording_factory)
    first_inside = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_inside = threading.Event()
    failures: list[BaseException] = []

    def use_context(name: str) -> None:
        try:
            if name == "second":
                second_started.set()
            with attention_module._sdpa_priority_context():  # pyright: ignore[reportPrivateUsage]
                if name == "first":
                    first_inside.set()
                    assert release_first.wait(5)
                else:
                    second_inside.set()
        except BaseException as error:  # pragma: no cover - surfaced below
            failures.append(error)

    first = threading.Thread(target=use_context, args=("first",))
    second = threading.Thread(target=use_context, args=("second",))
    first.start()
    assert first_inside.wait(5)
    second.start()
    assert second_started.wait(5)
    try:
        assert not second_inside.wait(0.1)
    finally:
        release_first.set()
        first.join(5)
        second.join(5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_inside.is_set()
    assert failures == []


def test_priority_context_serializes_small_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def recording_factory(*args: object, **kwargs: object) -> nullcontext[None]:
        return nullcontext()

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", recording_factory)
    dispatched = threading.Event()
    failures: list[BaseException] = []

    def recording_sdpa(q: torch.Tensor, *_args: object, **_kwargs: object) -> torch.Tensor:
        dispatched.set()
        return q

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    selected = kernel()
    q, k, v = tensors()

    def run_small() -> None:
        try:
            selected(q, k, v)
        except BaseException as error:  # pragma: no cover - surfaced below
            failures.append(error)

    with attention_module.attention_kernel_context(
        selected,
        attention_module.SDP_PRIORITY_MIN_ELEMENTS,
        device=q.device,
    ):
        worker = threading.Thread(target=run_small)
        worker.start()
        assert not dispatched.wait(0.1)
    worker.join(5)
    assert not worker.is_alive()
    assert dispatched.is_set()
    assert failures == []


def test_compiled_small_attention_serializes_with_priority_context() -> None:
    selected = kernel()
    q, k, v = (value.requires_grad_() for value in tensors())
    reference_q, reference_k, reference_v = (
        value.detach().clone().requires_grad_() for value in (q, k, v)
    )
    compiled = torch.compile(selected, fullgraph=True)
    actual = compiled(q, k, v)
    expected = selected(reference_q, reference_k, reference_v)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for value, reference in zip(
        (q, k, v),
        (reference_q, reference_k, reference_v),
        strict=True,
    ):
        assert value.grad is not None and reference.grad is not None
        torch.testing.assert_close(value.grad, reference.grad)

    started = threading.Event()
    completed = threading.Event()
    failures: list[BaseException] = []

    def run_small() -> None:
        started.set()
        try:
            compiled(q, k, v)
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)
        finally:
            completed.set()

    with sdpa_kernel([]):
        with attention_module.attention_kernel_context(
            selected,
            attention_module.SDP_PRIORITY_MIN_ELEMENTS,
            device=q.device,
        ):
            worker = threading.Thread(target=run_small)
            worker.start()
            assert started.wait(5)
            assert not completed.wait(0.1)
        worker.join(5)
    assert not worker.is_alive()
    assert completed.is_set()
    assert len(failures) == 1
    assert "No viable backend" in str(failures[0])


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    ((torch.float32, torch.bfloat16), (torch.float64, torch.float64)),
)
def test_compiled_small_attention_preserves_autocast(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    selected = kernel()
    q, k, v = (value.to(input_dtype).requires_grad_() for value in tensors())
    reference_q, reference_k, reference_v = (
        value.detach().clone().requires_grad_() for value in (q, k, v)
    )
    compiled = torch.compile(selected, fullgraph=True)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = compiled(q, k, v)
        expected = selected(reference_q, reference_k, reference_v)
    assert actual.dtype == expected.dtype == output_dtype
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for value, reference in zip(
        (q, k, v),
        (reference_q, reference_k, reference_v),
        strict=True,
    ):
        assert value.grad is not None and reference.grad is not None
        torch.testing.assert_close(value.grad, reference.grad)


@pytest.mark.parametrize("mask_requires_grad", (False, True))
def test_compiled_small_attention_preserves_mask_gradient_requirement(
    mask_requires_grad: bool,
) -> None:
    selected = kernel()
    q, k, v = (torch.randn(1, 2, 4, 8, requires_grad=True) for _ in range(3))
    mask = torch.zeros(1, 1, 4, 4, requires_grad=mask_requires_grad)
    reference_q, reference_k, reference_v = (
        value.detach().clone().requires_grad_() for value in (q, k, v)
    )
    reference_mask = mask.detach().clone().requires_grad_(mask_requires_grad)

    compiled = torch.compile(selected, fullgraph=True)
    backend = nullcontext() if mask_requires_grad else sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    with backend:
        actual = compiled(q, k, v, mask=mask)
        expected = selected(
            reference_q,
            reference_k,
            reference_v,
            mask=reference_mask,
        )
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    for value, reference in zip(
        (q, k, v),
        (reference_q, reference_k, reference_v),
        strict=True,
    ):
        assert value.grad is not None and reference.grad is not None
        torch.testing.assert_close(value.grad, reference.grad)
    assert (mask.grad is not None) is mask_requires_grad
    assert (reference_mask.grad is not None) is mask_requires_grad
    if mask_requires_grad:
        torch.testing.assert_close(mask.grad, reference_mask.grad)


def test_compiled_small_attention_layout_survives_backend_change() -> None:
    selected = kernel()
    q, k, v = (torch.randn(1, 4, 2, 8).transpose(1, 2) for _ in range(3))
    compiled = torch.compile(selected, fullgraph=True)
    compiled(q, k, v)

    with sdpa_kernel(SDPBackend.MATH):
        actual = compiled(q, k, v)
        expected = selected(q, k, v)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)


def test_compiled_small_attention_backward_uses_forward_state() -> None:
    selected = kernel()
    q, k, v = (value.requires_grad_() for value in tensors())
    reference_q, reference_k, reference_v = (
        value.detach().clone().requires_grad_() for value in (q, k, v)
    )
    actual = torch.compile(selected, fullgraph=True)(q, k, v)
    expected = selected(reference_q, reference_k, reference_v)

    with sdpa_kernel([]):
        actual.sum().backward()
        expected.sum().backward()
    for value, reference in zip(
        (q, k, v),
        (reference_q, reference_k, reference_v),
        strict=True,
    ):
        assert value.grad is not None and reference.grad is not None
        torch.testing.assert_close(value.grad, reference.grad)


@pytest.mark.parametrize(
    ("native_supported", "expected"),
    ((False, (4, 4, False)), (True, (2, 2, True))),
)
def test_large_masked_gqa_uses_capability_gated_fallback(
    monkeypatch: pytest.MonkeyPatch,
    native_supported: bool,
    expected: tuple[int, int, bool],
) -> None:
    observed: list[tuple[int, int, bool]] = []

    class RecordingContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def recording_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        observed.append((k.shape[1], v.shape[1], kwargs["enable_gqa"]))
        return torch.zeros_like(q)

    def recording_factory(*_args: object, **_kwargs: object) -> RecordingContext:
        return RecordingContext()

    def supports_native_gqa(*_args: object, **_kwargs: object) -> bool:
        return native_supported

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", recording_factory)
    monkeypatch.setattr(attention_module, "_supports_native_masked_gqa", supports_native_gqa)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
    q = torch.randn(1, 4, 4096, 32)
    k = torch.randn(1, 2, 16, 32)
    v = torch.randn_like(k)
    mask = torch.ones(1, 1, 4096, 16, dtype=torch.bool)

    select_attention("qwen").kernel(q, k, v, mask=mask, enable_gqa=True)

    assert observed == [expected]


def test_hip_builds_always_repeat_masked_kv_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_params(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("HIP builds must not consult the native GQA eligibility queries")

    monkeypatch.setattr(torch.version, "hip", "7.1.4")
    monkeypatch.setattr(torch.backends.cuda, "SDPAParams", unexpected_params)
    fake_q = cast("Any", type("FakeQ", (), {"device": torch.device("cuda")})())
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    assert (
        attention_module._supports_native_masked_gqa(  # pyright: ignore[reportPrivateUsage]
            fake_q, fake_q, fake_q, mask, causal=False
        )
        is False
    )


def test_nvidia_builds_consult_native_masked_kv_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consulted: list[bool] = []

    def recording_params(*_args: object, **_kwargs: object) -> object:
        consulted.append(True)
        raise TypeError("fake q is not a tensor")

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cuda, "SDPAParams", recording_params)
    fake_q = cast("Any", type("FakeQ", (), {"device": torch.device("cuda")})())
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    assert (
        attention_module._supports_native_masked_gqa(  # pyright: ignore[reportPrivateUsage]
            fake_q, fake_q, fake_q, mask, causal=False
        )
        is False
    )
    assert consulted == [True]


def test_intel_devices_never_enter_backend_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_factory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("XPU calls must retain torch's default SDPA dispatch")

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", unexpected_factory)
    huge = attention_module.SDP_PRIORITY_MIN_ELEMENTS * 4
    eligible = attention_module._priority_context_eligible  # pyright: ignore[reportPrivateUsage]
    assert eligible(torch.device("xpu"), huge) is False
    assert eligible(torch.device("cpu"), huge) is True
    assert eligible(torch.device("cpu"), attention_module.SDP_PRIORITY_MIN_ELEMENTS - 1) is False

    with attention_module.attention_kernel_context(kernel(), huge, device=torch.device("xpu")):
        pass


def test_kernel_context_with_eligible_device_keeps_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RecordingContext:
        def __enter__(self) -> None:
            calls.append("enter")

        def __exit__(self, *args: object) -> None:
            calls.append("exit")

    def recording_factory(*_args: object, **_kwargs: object) -> RecordingContext:
        calls.append("create")
        return RecordingContext()

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", recording_factory)
    huge = attention_module.SDP_PRIORITY_MIN_ELEMENTS * 4
    with attention_module.attention_kernel_context(kernel(), huge, device=torch.device("cpu")):
        pass
    assert calls == ["create", "enter", "exit"]


def test_small_attention_retains_torch_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("small attention must retain torch's default routing")

    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", unexpected_factory)
    assert_matches_direct(*tensors())


def test_missing_priority_api_falls_back_to_torch_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "_SDPA_KERNEL_FACTORY", None)
    q, k, v = tensors(batch=2**12)
    assert_matches_direct(q, k, v)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "v_shape"),
    [
        ((0, 2, 3, 4), (0, 2, 5, 4), (0, 2, 5, 6)),
        ((1, 0, 3, 4), (1, 0, 5, 4), (1, 0, 5, 6)),
        ((1, 2, 0, 4), (1, 2, 5, 4), (1, 2, 5, 6)),
        ((1, 2, 3, 4), (1, 2, 0, 4), (1, 2, 0, 6)),
        ((1, 2, 3, 0), (1, 2, 5, 0), (1, 2, 5, 6)),
    ],
)
def test_empty_geometry_matches_torch(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
) -> None:
    q = torch.randn(q_shape)
    k = torch.randn(k_shape)
    v = torch.randn(v_shape)
    assert_matches_direct(q, k, v)


def test_real_batch_limit_plus_one_matches_one_direct_call() -> None:
    shape = (2**15 + 1, 1, 1, 1)
    q = torch.linspace(-1.0, 1.0, shape[0]).reshape(shape)
    k = torch.ones(shape)
    v = torch.linspace(1.0, -1.0, shape[0]).reshape(shape)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    actual = kernel()(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("role", ROLES)
def test_every_role_selects_truthful_explicit_sdpa(role: AttentionRole) -> None:
    selection = select_attention(role, "sdpa")
    assert isinstance(selection.kernel, AttentionKernel)
    assert selection.status.requested_policy == "sdpa"
    assert selection.status.role == role
    assert selection.status.primary == "sdpa"
    assert selection.status.fallback is None
    assert selection.status.reason
    assert selection.status.authenticated is False
    assert selection.status.provider_versions == ()
    assert selection.status.adapter_contract == ATTENTION_ADAPTER_CONTRACT
    assert selection.status.device_kind == "unknown"
    assert selection.status.device_sm is None
    assert selection.status.sdpa_torch_runtime == "unknown"


def test_auto_selection_uses_capability_route_and_truthful_vae_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _capabilities()
    monkeypatch.setattr(attention_module, "discover_attention_capabilities", lambda: evidence)
    for role in ROLES:
        selection = select_attention(role, "auto")
        assert selection.status.requested_policy == "auto"
        assert selection.status.primary == "sdpa"
        assert selection.status.fallback == ("bounded" if role == "vae" else None)
        assert selection.status.device_kind == "cpu"
        assert selection.status.sdpa_torch_runtime == evidence.sdpa_torch_runtime


def test_auto_selection_keeps_sage_explicit_and_uses_rocm_bounded_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_sage_available(monkeypatch)
    sage = _capabilities(device_kind="cuda", sage=True)
    monkeypatch.setattr(attention_module, "discover_attention_capabilities", lambda: sage)
    selected = select_attention("unet", "auto")
    assert selected.kernel is attention_module._SDPA  # pyright: ignore[reportPrivateUsage]
    assert selected.status.primary == "sdpa"
    assert selected.status.fallback is None

    selected = select_attention("unet", "sage")
    assert selected.kernel is attention_module._SAGE2  # pyright: ignore[reportPrivateUsage]
    assert selected.status.primary == "sage"

    rocm = _capabilities(device_kind="rocm")
    monkeypatch.setattr(attention_module, "discover_attention_capabilities", lambda: rocm)
    selected = select_attention("vae", "auto")
    assert selected.kernel is attention_module._BOUNDED  # pyright: ignore[reportPrivateUsage]
    assert selected.status.primary == "bounded"
    assert selected.status.fallback is None
    assert attention_provider_identity(selected.status) == (
        "torch-bounded-attention-v1",
        None,
    )


def test_kitchen_load_disables_triton_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    disabled: list[str] = []
    kitchen = SimpleNamespace(registry=SimpleNamespace(disable=disabled.append))
    monkeypatch.setattr(attention_module.importlib, "import_module", lambda _name: kitchen)
    for name in (
        "_KITCHEN_AVAILABLE",
        "_KITCHEN_ATTENTION",
        "_KITCHEN_PREQUANTIZE",
        "_KITCHEN_FROM_PREQUANTIZED",
        "_KITCHEN_SOL_ATTENTION",
        "_KITCHEN_LIST_BACKENDS",
    ):
        monkeypatch.setattr(attention_module, name, attention_module._KITCHEN_UNPROBED)  # pyright: ignore[reportPrivateUsage]

    attention_module._load_kitchen_apis()  # pyright: ignore[reportPrivateUsage]

    assert disabled == ["triton"]


@pytest.mark.parametrize(
    "case", ("plain", "boolean-mask", "additive-mask", "causal", "scale", "gqa")
)
def test_bounded_attention_matches_direct_sdpa_across_query_chunks(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    q, k, v = tensors(batch=1, q_heads=4, kv_heads=2) if case == "gqa" else tensors(batch=1)
    mask: torch.Tensor | None = None
    causal = case == "causal"
    scale = 0.37 if case == "scale" else None
    enable_gqa = case == "gqa"
    if case == "boolean-mask":
        mask = torch.ones((1, 1, q.shape[2], k.shape[2]), dtype=torch.bool)
        mask[..., 0, -1] = False
    elif case == "additive-mask":
        mask = torch.zeros((1, 1, q.shape[2], k.shape[2]), dtype=q.dtype)
        mask[..., 1, 0] = -100.0
    monkeypatch.setattr(attention_module, "_initial_bounded_attention_steps", _bounded_steps_three)
    expected = kernel()(q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa)
    actual = attention_module._BOUNDED(  # pyright: ignore[reportPrivateUsage]
        q,
        k,
        v,
        mask=mask,
        causal=causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_vae_sdpa_falls_back_only_on_recognized_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    q, k, v = tensors(batch=1)
    calls = 0

    def oom_once(
        chunk_q: torch.Tensor,
        _k: torch.Tensor,
        chunk_v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise torch.OutOfMemoryError("expected test OOM")
        return torch.full((*chunk_q.shape[:-1], chunk_v.shape[-1]), 7.0, dtype=chunk_q.dtype)

    monkeypatch.setattr(attention_module, "_SDPA", oom_once)
    monkeypatch.setattr(attention_module, "soft_empty_cache", _ignore_device)
    result = attention_module._VAE_SDPA(q, k, v)  # pyright: ignore[reportPrivateUsage]
    assert calls == 2
    assert torch.equal(result, torch.full_like(result, 7.0))

    non_oom_calls = 0
    cleanup_calls: list[torch.device] = []

    def non_oom(*_args: object, **_kwargs: object) -> torch.Tensor:
        nonlocal non_oom_calls
        non_oom_calls += 1
        raise RuntimeError("not an OOM")

    monkeypatch.setattr(attention_module, "_SDPA", non_oom)
    monkeypatch.setattr(attention_module, "soft_empty_cache", cleanup_calls.append)
    with pytest.raises(RuntimeError, match="not an OOM"):
        attention_module._VAE_SDPA(q, k, v)  # pyright: ignore[reportPrivateUsage]
    assert non_oom_calls == 1
    assert cleanup_calls == []


def test_bounded_attention_retries_once_then_doubles_before_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = torch.randn(1, 1, 8, 3)
    k = torch.randn(1, 1, 5, 3)
    v = torch.randn(1, 1, 5, 3)
    query_lengths: list[int] = []

    def fail_twice(
        chunk_q: torch.Tensor, _k: torch.Tensor, _v: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        query_lengths.append(chunk_q.shape[2])
        if len(query_lengths) <= 2:
            raise torch.OutOfMemoryError("expected test OOM")
        return chunk_q * 2.0

    monkeypatch.setattr(attention_module, "_SDPA", fail_twice)
    monkeypatch.setattr(attention_module, "soft_empty_cache", _ignore_device)
    monkeypatch.setattr(attention_module, "_initial_bounded_attention_steps", _bounded_steps_one)
    result = attention_module._BOUNDED(q, k, v)  # pyright: ignore[reportPrivateUsage]
    assert query_lengths == [8, 8, 4, 4]
    assert torch.equal(result, q * 2.0)


def test_bounded_attention_never_retries_after_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = torch.randn(1, 1, 8, 3)
    k = torch.randn(1, 1, 5, 3)
    v = torch.randn(1, 1, 5, 3)
    calls = 0

    def partial_then_oom(
        chunk_q: torch.Tensor, _k: torch.Tensor, _v: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("expected test OOM")
        return chunk_q

    monkeypatch.setattr(attention_module, "_SDPA", partial_then_oom)
    monkeypatch.setattr(attention_module, "soft_empty_cache", _ignore_device)
    monkeypatch.setattr(attention_module, "_initial_bounded_attention_steps", _bounded_steps_two)
    with pytest.raises(torch.OutOfMemoryError):
        attention_module._BOUNDED(q, k, v)  # pyright: ignore[reportPrivateUsage]
    assert calls == 2


def test_bounded_attention_has_finite_retry_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    q = torch.randn(1, 1, 128, 3)
    k = torch.randn(1, 1, 5, 3)
    v = torch.randn(1, 1, 5, 3)
    calls = 0

    def always_oom(*_args: object, **_kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise torch.OutOfMemoryError("expected test OOM")

    monkeypatch.setattr(attention_module, "_SDPA", always_oom)
    monkeypatch.setattr(attention_module, "soft_empty_cache", _ignore_device)
    monkeypatch.setattr(attention_module, "_initial_bounded_attention_steps", _bounded_steps_max)
    with pytest.raises(torch.OutOfMemoryError):
        attention_module._BOUNDED(q, k, v)  # pyright: ignore[reportPrivateUsage]
    assert calls == 2


def test_adapter_contract_records_prioritized_sdpa() -> None:
    assert ATTENTION_ADAPTER_CONTRACT == "dinkster.attention-kernel.v2"


@pytest.mark.parametrize("policy", ("flash", "xformers", "sage3"))
def test_named_optional_policies_use_sdpa_with_diagnostic(
    policy: AttentionPolicy,
    caplog: pytest.LogCaptureFixture,
) -> None:
    selection = select_attention("flux", policy)
    assert selection.status.requested_policy == policy
    assert selection.status.primary == "sdpa"
    assert selection.status.fallback is None
    assert f"no {policy} provider adapter is implemented" in caplog.text
    q, k, v = tensors()
    assert torch.equal(selection.kernel(q, k, v), kernel()(q, k, v))


def test_selection_is_frozen_and_kernel_has_no_inspection_surface() -> None:
    selection = select_attention("qwen", "auto")

    def mutate_frozen(target: object, attribute: str, value: object) -> None:
        setattr(target, attribute, value)

    with pytest.raises(FrozenInstanceError):
        mutate_frozen(selection.status, "primary", "flash")
    with pytest.raises(FrozenInstanceError):
        mutate_frozen(selection, "kernel", None)
    for attribute in ("name", "policy", "role", "status", "provider"):
        assert not hasattr(selection.kernel, attribute)


def test_unknown_role_and_policy_refuse_at_construction() -> None:
    with pytest.raises(AttentionSelectionError, match="unknown attention role"):
        select_attention(cast(AttentionRole, "other"), "auto")
    with pytest.raises(AttentionSelectionError, match="unknown attention policy"):
        select_attention("unet", cast(AttentionPolicy, "other"))


def test_invalid_rank_refuses_before_torch() -> None:
    q, k, v = tensors()
    with pytest.raises(AttentionValidationError, match="q must have rank 4"):
        kernel()(q[0], k, v)


def test_non_tensor_qkv_and_mask_refuse_with_typed_errors() -> None:
    q, k, v = tensors()
    not_tensor = cast(torch.Tensor, object())
    with pytest.raises(AttentionValidationError, match="q must be a torch.Tensor"):
        kernel()(not_tensor, k, v)
    with pytest.raises(AttentionValidationError, match="k must be a torch.Tensor"):
        kernel()(q, not_tensor, v)
    with pytest.raises(AttentionValidationError, match="v must be a torch.Tensor"):
        kernel()(q, k, not_tensor)
    with pytest.raises(AttentionValidationError, match="mask must be a torch.Tensor"):
        kernel()(q, k, v, mask=not_tensor)


def test_control_flags_require_actual_bools() -> None:
    q, k, v = tensors()
    with pytest.raises(AttentionValidationError, match="causal must be a bool"):
        kernel()(q, k, v, causal=cast(bool, 1))
    with pytest.raises(AttentionValidationError, match="enable_gqa must be a bool"):
        kernel()(q, k, v, enable_gqa=cast(bool, 0))


@pytest.mark.parametrize("scale", (True, float("nan"), float("inf"), float("-inf"), "1"))
def test_scale_requires_finite_real_number_excluding_bool(scale: object) -> None:
    q, k, v = tensors()
    with pytest.raises(AttentionValidationError, match="finite int or float"):
        kernel()(q, k, v, scale=cast(float, scale))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("dtype", "share one dtype"),
        ("device", "share one device"),
        ("batch", "share one batch size"),
        ("head_dim", "head dimensions must match"),
        ("sequence", "sequence lengths must match"),
        ("kv_heads", "head counts must match"),
        ("q_heads", "when GQA is disabled"),
    ],
)
def test_invalid_geometry_refuses(case: str, message: str) -> None:
    q, k, v = tensors()
    if case == "dtype":
        q = q.double()
    elif case == "device":
        q = q.to("meta")
    elif case == "batch":
        q = q[:1]
    elif case == "head_dim":
        q = q[..., :4]
    elif case == "sequence":
        k = k[:, :, :-1]
    elif case == "kv_heads":
        k = k[:, :2]
    else:
        q = q[:, :2]
    with pytest.raises(AttentionValidationError, match=message):
        kernel()(q, k, v)


def test_gqa_requires_even_nonzero_kv_head_factor() -> None:
    q, k, v = tensors(q_heads=4, kv_heads=3)
    with pytest.raises(AttentionValidationError, match="GQA requires"):
        kernel()(q, k, v, enable_gqa=True)


def test_mask_contract_refuses_causal_dtype_and_shape_conflicts() -> None:
    q, k, v = tensors()
    with pytest.raises(AttentionValidationError, match="cannot also receive"):
        kernel()(q, k, v, mask=torch.ones(4, 6, dtype=torch.bool), causal=True)
    with pytest.raises(AttentionValidationError, match="boolean or additive"):
        kernel()(q, k, v, mask=torch.ones(4, 6, dtype=torch.int64))
    with pytest.raises(AttentionValidationError, match="not broadcastable"):
        kernel()(q, k, v, mask=torch.ones(5, 6, dtype=torch.bool))


def test_integer_qkv_refuse_without_coercion() -> None:
    q = torch.ones((1, 1, 1, 1), dtype=torch.int64)
    with pytest.raises(AttentionValidationError, match="floating-point"):
        kernel()(q, q, q)


def _kitchen_token(
    *,
    contract: str = ATTENTION_ADAPTER_CONTRACT,
    torch_version: str | None = None,
) -> AttentionRouteToken:
    kitchen_version = importlib.metadata.version("dinkster-kitchen")
    torch_version = str(torch.__version__) if torch_version is None else torch_version
    return AttentionRouteToken(
        1,
        tuple(AttentionRoute(role, "dinkster_kitchen_int8", "sdpa") for role in ROLES),
        (("dinkster-kitchen", kitchen_version), ("torch", torch_version)),
        contract,
        "cuda",
        89,
        torch_version.split("+")[0],
        "dinkster_kitchen_int8",
    )


def _capabilities_for_token(token: AttentionRouteToken) -> AttentionCapabilityEvidence:
    return AttentionCapabilityEvidence(
        version=1,
        available_policies=("sdpa", "dinkster_kitchen_int8"),
        provider_versions=token.provider_versions,
        adapter_contract_revision=token.adapter_contract_revision,
        device_kind=token.device_kind,
        device_sm=token.device_sm,
        sdpa_torch_runtime=token.sdpa_torch_runtime,
    )


@pytest.mark.parametrize("role", ROLES)
def test_kitchen_policy_selects_kitchen_for_every_role(
    role: AttentionRole, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    selection = select_attention(role, "dinkster_kitchen_int8")
    assert selection.status.requested_policy == "dinkster_kitchen_int8"
    assert selection.status.primary == "dinkster_kitchen_int8"
    assert selection.kernel is attention_module._COMFY_KITCHEN_INT8  # pyright: ignore[reportPrivateUsage]
    assert isinstance(selection.kernel, QkvConsumingAttentionKernel)
    supported_dtypes = selection.kernel.partition_compatibility.supported_dtypes
    assert tuple(dtype.name for dtype in supported_dtypes) == (
        "float32",
        "float16",
        "bfloat16",
    )
    assert selection.status.fallback == "sdpa"
    assert selection.status.reason


def test_kitchen_policy_falls_back_when_kernel_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: False)
    token = attention_module.discover_attention_route_token("dinkster_kitchen_int8")
    selection = attention_module.resolve_role_attention("flux", "dinkster_kitchen_int8", token)
    assert token.version == 3
    assert selection.status.authenticated
    assert selection.status.primary == "sdpa"
    assert "does not support this machine" in caplog.text


def test_package_import_without_kitchen_int8_apis_keeps_sdpa_available() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import dinkster_kitchen

missing = (
    "int8_attention_is_available",
    "int8_attention",
    "prequantize_int8_attention",
    "int8_attention_from_prequantized",
)
for name in missing:
    if hasattr(dinkster_kitchen, name):
        delattr(dinkster_kitchen, name)

from dinkster_inference_torch import select_attention

assert select_attention("flux").status.primary == "sdpa"
assert select_attention("flux", "sdpa").status.primary == "sdpa"
selection = select_attention("flux", "dinkster_kitchen_int8")
assert selection.status.primary == "sdpa"
assert "installed dinkster-kitchen is missing required APIs" in selection.status.reason
assert all(name in selection.status.reason for name in missing)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_kitchen_kernel_falls_back_for_non_cuda_inputs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    kitchen = select_attention("unet", "dinkster_kitchen_int8").kernel
    q, k, v = tensors()
    expected = kernel()(q, k, v)
    assert torch.equal(kitchen(q, k, v), expected)
    consuming = cast(QkvConsumingAttentionKernel, kitchen)
    result = consuming.consume(
        AttentionTensorLease(q), AttentionTensorLease(k), AttentionTensorLease(v)
    )
    assert torch.equal(result, expected)
    assert "unavailable on cpu; using SDPA" in caplog.text


def test_kitchen_kernel_validates_rank_before_wide_head_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    kitchen = select_attention("unet", "dinkster_kitchen_int8").kernel
    scalar = torch.tensor(1.0)
    # The wide-head fallback inspects the head dimension, so generic
    # validation must run first: a rank-0 tensor raises the adapter
    # contract's error, not IndexError from the shape probe.
    with pytest.raises(AttentionValidationError, match="q must have rank 4"):
        kitchen(scalar, scalar, scalar)
    consuming = cast(QkvConsumingAttentionKernel, kitchen)
    with pytest.raises(AttentionValidationError, match="q must have rank 4"):
        consuming.consume(
            AttentionTensorLease(scalar.clone()),
            AttentionTensorLease(scalar.clone()),
            AttentionTensorLease(scalar.clone()),
        )


def test_kitchen_kernel_serves_causal_gqa_and_wide_heads_through_sdpa_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    kitchen = select_attention("qwen", "dinkster_kitchen_int8").kernel
    sdpa = select_attention("qwen", "sdpa").kernel
    torch.manual_seed(7)
    q = torch.randn(2, 4, 6, 5)
    k = torch.randn(2, 4, 6, 5)
    v = torch.randn(2, 4, 6, 7)
    torch.testing.assert_close(kitchen(q, k, v, causal=True), sdpa(q, k, v, causal=True))
    gq, gk, gv = tensors(q_heads=4, kv_heads=2)
    torch.testing.assert_close(
        kitchen(gq, gk, gv, enable_gqa=True), sdpa(gq, gk, gv, enable_gqa=True)
    )
    # KL/Wan VAE single-head attention has head_dim = channels (384-512),
    # above dinkster-kitchen's 256 limit; these tensors run on CPU, so a passing
    # equality proves the fallback branch fired before the CUDA device check.
    wq = torch.randn(1, 1, 6, 300)
    wk = torch.randn(1, 1, 6, 300)
    wv = torch.randn(1, 1, 6, 300)
    torch.testing.assert_close(kitchen(wq, wk, wv), sdpa(wq, wk, wv))
    consuming = cast(QkvConsumingAttentionKernel, kitchen)
    consumed = consuming.consume(
        AttentionTensorLease(q.clone()),
        AttentionTensorLease(k.clone()),
        AttentionTensorLease(v.clone()),
        causal=True,
    )
    torch.testing.assert_close(consumed, sdpa(q, k, v, causal=True))
    wide_consumed = consuming.consume(
        AttentionTensorLease(wq.clone()),
        AttentionTensorLease(wk.clone()),
        AttentionTensorLease(wv.clone()),
    )
    torch.testing.assert_close(wide_consumed, sdpa(wq, wk, wv))


def test_lease_yields_its_tensor_exactly_once() -> None:
    tensor = torch.randn(1, 1, 1, 1)
    lease = AttentionTensorLease(tensor)
    assert lease.take() is tensor
    with pytest.raises(AttentionValidationError, match="already taken"):
        lease.take()
    with pytest.raises(AttentionValidationError, match="requires a torch.Tensor"):
        AttentionTensorLease(cast(torch.Tensor, object()))


def test_kitchen_consume_releases_leases_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    kitchen = cast(
        QkvConsumingAttentionKernel, select_attention("unet", "dinkster_kitchen_int8").kernel
    )
    q, k, v = tensors()
    k = k[..., :-1]
    refs = [weakref.ref(tensor) for tensor in (q, k, v)]
    leases = (AttentionTensorLease(q), AttentionTensorLease(k), AttentionTensorLease(v))
    del q, k, v
    with pytest.raises(AttentionValidationError, match="head dimensions must match") as excinfo:
        kitchen.consume(*leases)
    # The propagating exception's traceback must not keep the leased
    # floating-point tensors alive after the kernel took them.
    assert excinfo.value is not None
    gc.collect()
    assert all(ref() is None for ref in refs)
    for lease in leases:
        with pytest.raises(AttentionValidationError, match="already taken"):
            lease.take()


def test_kitchen_policy_discovers_on_any_device_the_probe_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    kitchen_version = importlib.metadata.version("dinkster-kitchen")
    rocm_torch = SimpleNamespace(__version__="2.13.0+rocm6.4", version=SimpleNamespace(hip="6.4.0"))
    rocm = attention_module.discover_attention_route_token(
        "dinkster_kitchen_int8",
        torch_module=rocm_torch,
        device_kind="rocm",
        device_sm=110,
    )
    assert rocm.device_kind == "rocm"
    assert all(route.primary == "dinkster_kitchen_int8" for route in rocm.routes)
    assert all(route.fallback == "sdpa" for route in rocm.routes)
    assert dict(rocm.provider_versions) == {
        "torch": "2.13.0+rocm6.4",
        "hip": "6.4.0",
        "dinkster-kitchen": kitchen_version,
    }
    cpu_torch = SimpleNamespace(__version__="2.13.0+cpu", version=SimpleNamespace(hip=None))
    cpu = attention_module.discover_attention_route_token(
        "dinkster_kitchen_int8",
        torch_module=cpu_torch,
        device_kind="cpu",
    )
    assert cpu.device_kind == "cpu"
    assert all(route.primary == "dinkster_kitchen_int8" for route in cpu.routes)


def test_kitchen_kernel_runs_on_rocm_builds_without_build_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)

    def fake_hip_runtime(torch_module: object = torch) -> str:
        return "6.4.0"

    monkeypatch.setattr(attention_module, "_hip_runtime_version", fake_hip_runtime)
    kitchen = select_attention("unet", "dinkster_kitchen_int8").kernel
    q, k, v = tensors()
    # CPU tensors use SDPA even when the worker has a supported HIP kernel.
    expected = kernel()(q, k, v)
    assert torch.equal(kitchen(q, k, v), expected)
    consuming = cast(QkvConsumingAttentionKernel, kitchen)
    assert torch.equal(
        consuming.consume(
            AttentionTensorLease(q), AttentionTensorLease(k), AttentionTensorLease(v)
        ),
        expected,
    )


def test_resolve_role_attention_carries_authenticated_kitchen_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    token = _kitchen_token()
    monkeypatch.setattr(
        attention_module,
        "discover_attention_capabilities",
        lambda: _capabilities_for_token(token),
    )
    selection = attention_module.resolve_role_attention("flux", "dinkster_kitchen_int8", token)
    assert cast("Any", selection.kernel).active_kernel() is attention_module._COMFY_KITCHEN_INT8  # pyright: ignore[reportPrivateUsage]
    assert selection.status.primary == "dinkster_kitchen_int8"
    assert selection.status.authenticated is True
    assert selection.status.provider_versions == token.provider_versions
    assert selection.status.device_kind == token.device_kind
    vae = attention_module.resolve_role_attention("vae", "dinkster_kitchen_int8", token)
    assert vae.status.primary == "dinkster_kitchen_int8"
    assert vae.status.fallback == "sdpa"
    assert vae.status.authenticated is True


def test_resolve_role_attention_refuses_route_and_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    local_evidence = _kitchen_token()
    monkeypatch.setattr(
        attention_module,
        "discover_attention_capabilities",
        lambda: _capabilities_for_token(local_evidence),
    )
    foreign_machine = replace(local_evidence, device_sm=90)
    with pytest.raises(AttentionSelectionError, match="rediscovered runtime evidence"):
        attention_module.resolve_role_attention("flux", "dinkster_kitchen_int8", foreign_machine)
    foreign_provider = replace(
        local_evidence,
        provider_versions=(("dinkster-kitchen", "forged"), ("torch", str(torch.__version__))),
    )
    with pytest.raises(AttentionSelectionError, match="rediscovered runtime evidence"):
        attention_module.resolve_role_attention("flux", "dinkster_kitchen_int8", foreign_provider)
    sdpa_routes = tuple(AttentionRoute(role, "sdpa") for role in ROLES)
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        replace(local_evidence, routes=sdpa_routes)
    foreign = _kitchen_token(contract="dinkster.attention-kernel.v0")
    with pytest.raises(AttentionSelectionError, match="adapter contract does not match"):
        attention_module.resolve_role_attention("flux", "dinkster_kitchen_int8", foreign)


def test_discovery_with_kitchen_policy_binds_routes_and_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "sage2_attention_available", lambda: False)
    monkeypatch.setattr(attention_module, "_KITCHEN_SOL_ATTENTION", None)
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    fake_torch = SimpleNamespace(__version__="2.13.0+cu130", version=SimpleNamespace(hip=None))
    capabilities = attention_module.discover_attention_capabilities(
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert capabilities.available_policies == ("sdpa", "dinkster_kitchen_int8")
    assert dict(capabilities.provider_versions).keys() == {"torch", "dinkster-kitchen"}
    token = attention_module.discover_attention_route_token(
        "dinkster_kitchen_int8",
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert token == _kitchen_token(torch_version="2.13.0+cu130")
    auto = attention_module.discover_attention_route_token(
        "auto",
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert all(route.primary == "sdpa" for route in auto.routes)
    assert dict(auto.provider_versions).keys() == {"torch"}


def test_discovery_with_role_overrides_mints_version_two_per_role_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    fake_torch = SimpleNamespace(__version__="2.13.0+cu130", version=SimpleNamespace(hip=None))
    overrides: tuple[tuple[str, AttentionPolicy], ...] = (("flux", "dinkster_kitchen_int8"),)
    token = attention_module.discover_attention_route_token(
        "sdpa",
        requested_role_policies=overrides,
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert token.version == 2
    assert token.requested_policy == "sdpa"
    assert token.requested_role_policies == overrides
    routes = {route.role: route for route in token.routes}
    assert (routes["flux"].primary, routes["flux"].fallback) == ("dinkster_kitchen_int8", "sdpa")
    assert all(
        (route.primary, route.fallback) == ("sdpa", None)
        for role, route in routes.items()
        if role != "flux"
    )
    assert dict(token.provider_versions).keys() == {"torch", "dinkster-kitchen"}
    fallback = attention_module.discover_attention_route_token(
        "sdpa",
        requested_role_policies=(("flux", "flash"),),
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert fallback.version == 3
    assert all(route.primary == "sdpa" for route in fallback.routes)
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: False)
    fallback = attention_module.discover_attention_route_token(
        "sdpa",
        requested_role_policies=overrides,
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert fallback.version == 3
    assert all(route.primary == "sdpa" for route in fallback.routes)


def test_resolve_role_attention_honors_role_overrides_or_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: True)
    overrides: tuple[tuple[str, AttentionPolicy], ...] = (("flux", "dinkster_kitchen_int8"),)
    token = attention_module.discover_attention_route_token(
        "sdpa",
        requested_role_policies=overrides,
    )
    flux = attention_module.resolve_role_attention("flux", "sdpa", token)
    assert cast("Any", flux.kernel).active_kernel() is attention_module._COMFY_KITCHEN_INT8  # pyright: ignore[reportPrivateUsage]
    assert flux.status.requested_policy == "dinkster_kitchen_int8"
    assert flux.status.primary == "dinkster_kitchen_int8"
    assert flux.status.authenticated is True
    vae = attention_module.resolve_role_attention("vae", "sdpa", token)
    assert vae.status.requested_policy == "sdpa"
    assert vae.status.primary == "sdpa"
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: False)
    with pytest.raises(AttentionSelectionError, match="rediscovered runtime evidence"):
        attention_module.resolve_role_attention("flux", "sdpa", token)


class _FakeSolAttn:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] = []

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.calls.append((q, k, v, kwargs))
        return q + 1


def _sol_current_device_only(device: torch.device | None = None) -> bool:
    return device is None


def _sol_any_device(device: torch.device | None = None) -> bool:
    del device
    return True


def _sol_execute_always(*args: object, **kwargs: object) -> bool:
    del args, kwargs
    return True


def _sm80_capability(device: object = None) -> tuple[int, int]:
    del device
    return (8, 0)


def _no_hip_runtime(torch_module: object = torch) -> None:
    del torch_module


def _hip64_runtime(torch_module: object = torch) -> str:
    del torch_module
    return "6.4"


def _make_sol_available(monkeypatch: pytest.MonkeyPatch) -> _FakeSolAttn:
    fake = _FakeSolAttn()
    monkeypatch.setattr(attention_module, "_KITCHEN_SOL_ATTENTION", fake)
    monkeypatch.setattr(
        attention_module,
        "_KITCHEN_LIST_BACKENDS",
        lambda: {
            "cuda": {
                "available": True,
                "disabled": False,
                "capabilities": ("sol_attn",),
            }
        },
    )
    monkeypatch.setattr(attention_module, "_sol_device_supported", _sol_current_device_only)
    return fake


def _sol_token(
    *,
    contract: str = ATTENTION_ADAPTER_CONTRACT,
    torch_version: str | None = None,
) -> AttentionRouteToken:
    kitchen_version = importlib.metadata.version("dinkster-kitchen")
    torch_version = str(torch.__version__) if torch_version is None else torch_version
    return AttentionRouteToken(
        1,
        tuple(AttentionRoute(role, "sol", "sdpa") for role in ROLES),
        (("dinkster-kitchen", kitchen_version), ("torch", torch_version)),
        contract,
        "cuda",
        89,
        torch_version.split("+")[0],
        "sol",
    )


def _sol_capabilities_for_token(token: AttentionRouteToken) -> AttentionCapabilityEvidence:
    return AttentionCapabilityEvidence(
        version=1,
        available_policies=("sdpa", "sol"),
        provider_versions=token.provider_versions,
        adapter_contract_revision=token.adapter_contract_revision,
        device_kind=token.device_kind,
        device_sm=token.device_sm,
        sdpa_torch_runtime=token.sdpa_torch_runtime,
    )


@pytest.mark.parametrize("role", ROLES)
def test_sol_policy_selects_sol_for_every_role(
    role: AttentionRole, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_sol_available(monkeypatch)
    selection = select_attention(role, "sol")
    assert selection.status.requested_policy == "sol"
    assert selection.status.primary == "sol"
    assert selection.status.fallback == "sdpa"
    assert selection.status.reason
    assert selection.kernel is attention_module._SOL  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("sol_api", "backends", "message"),
    (
        (None, {}, "missing the Sol sparse attention API"),
        (object(), None, "missing the Sol sparse attention API"),
        (object(), {}, "does not support this machine"),
        (
            object(),
            {"cuda": {"available": False, "disabled": False, "capabilities": ("sol_attn",)}},
            "does not support this machine",
        ),
        (
            object(),
            {"cuda": {"available": True, "disabled": True, "capabilities": ("sol_attn",)}},
            "does not support this machine",
        ),
        (
            object(),
            {"cuda": {"available": True, "disabled": False, "capabilities": ()}},
            "does not support this machine",
        ),
    ),
)
def test_sol_policy_falls_back_for_missing_api_or_backend(
    sol_api: object | None,
    backends: object | None,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_KITCHEN_SOL_ATTENTION", sol_api)
    monkeypatch.setattr(
        attention_module,
        "_KITCHEN_LIST_BACKENDS",
        None if backends is None else lambda: backends,
    )
    monkeypatch.setattr(attention_module, "_sol_device_supported", _sol_any_device)
    selection = select_attention("flux", "sol")
    assert selection.status.primary == "sdpa"
    assert message in selection.status.reason


def test_sol_capability_probe_requires_nvidia_sm80(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=_sm80_capability,
    )
    monkeypatch.setattr(torch, "cuda", cuda)
    monkeypatch.setattr(attention_module, "_hip_runtime_version", _no_hip_runtime)
    assert attention_module._sol_device_supported()  # pyright: ignore[reportPrivateUsage]
    cuda.get_device_capability = _sm75_capability
    assert not attention_module._sol_device_supported()  # pyright: ignore[reportPrivateUsage]
    assert not attention_module._sol_device_supported(  # pyright: ignore[reportPrivateUsage]
        torch.device("cpu")
    )
    monkeypatch.setattr(attention_module, "_hip_runtime_version", _hip64_runtime)
    assert not attention_module._sol_device_supported()  # pyright: ignore[reportPrivateUsage]


def test_sol_kernel_dispatches_kitchen_layout_and_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_sol_available(monkeypatch)
    monkeypatch.setattr(attention_module, "_sol_can_execute", _sol_execute_always)
    q = torch.randn(2, 3, 4, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    result = select_attention("flux", "sol").kernel(q, k, v, scale=0.125)
    assert torch.equal(result, q + 1)
    called_q, called_k, called_v, kwargs = fake.calls[0]
    assert called_q.shape == called_k.shape == called_v.shape == (2, 4, 3, 128)
    torch.testing.assert_close(called_q, q.transpose(1, 2))
    torch.testing.assert_close(called_k, k.transpose(1, 2))
    torch.testing.assert_close(called_v, v.transpose(1, 2))
    assert kwargs == {"tau": 1.0, "scale": 0.125, "tail": True}


def test_scheduled_attention_switches_centrally_for_repeated_calls() -> None:
    class Candidate:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, q: torch.Tensor, _k: torch.Tensor, _v: torch.Tensor, **_kwargs: object):
            self.calls += 1
            return q + 7

    candidate = Candidate()
    scheduled = attention_module.schedule_aware_attention_kernel(
        "sage", cast(AttentionKernel, candidate)
    )
    static_q = tensors(batch=1, q_heads=1)[0]
    torch.testing.assert_close(scheduled(static_q, static_q, static_q), static_q + 7)
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule("sage", 0.5, 1.0),
        (2.0, 1.0, 0.0),
    )
    q, k, v = tensors(batch=1, q_heads=1)
    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        torch.testing.assert_close(scheduled(q, k, v), kernel()(q, k, v))
        activate(1)
        first = scheduled(q, k, v)
        second = scheduled(q, k, v)

    torch.testing.assert_close(first, q + 7)
    torch.testing.assert_close(second, q + 7)
    assert candidate.calls == 3


def test_scheduled_attention_preserves_consuming_kernel_contract() -> None:
    class Candidate:
        def __call__(
            self,
            q: torch.Tensor,
            _k: torch.Tensor,
            _v: torch.Tensor,
            **_kwargs: object,
        ) -> torch.Tensor:
            return q + 1

        def consume(
            self,
            q: AttentionTensorLease,
            k: AttentionTensorLease,
            v: AttentionTensorLease,
            **_kwargs: object,
        ) -> torch.Tensor:
            result = q.take() + 2
            k.take()
            v.take()
            return result

    scheduled = attention_module.schedule_aware_attention_kernel(
        "dinkster_kitchen_int8",
        cast(AttentionKernel, Candidate()),
    )
    assert isinstance(scheduled, QkvConsumingAttentionKernel)
    q, k, v = tensors(batch=1, q_heads=1)
    result = scheduled.consume(
        AttentionTensorLease(q),
        AttentionTensorLease(k),
        AttentionTensorLease(v),
    )
    torch.testing.assert_close(result, q + 2)


def test_scheduled_attention_rejects_unimplemented_modifier_centrally() -> None:
    scheduled = attention_module.schedule_aware_attention_kernel("sdpa", kernel())
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule(
            "sage",
            0.0,
            1.0,
            attention_modifiers=(AttentionModifierSchedule("skip_softmax", 0.0, 1.0),),
        ),
        (1.0, 0.0),
    )
    q, k, v = tensors(batch=1, q_heads=1)

    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        with pytest.raises(AttentionSelectionError, match="unsupported attention modifiers"):
            scheduled(q, k, v)


def test_scheduled_attention_rejects_provider_not_authenticated_by_model() -> None:
    scheduled = attention_module.schedule_aware_attention_kernel("sdpa", kernel())
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule("sage", 0.0, 1.0),
        (1.0, 0.0),
    )
    q, k, v = tensors(batch=1, q_heads=1)

    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        with pytest.raises(AttentionSelectionError, match="authenticated attention provider"):
            scheduled(q, k, v)


def test_scheduled_sol_uses_realized_tau_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_sol_available(monkeypatch)
    monkeypatch.setattr(attention_module, "_sol_can_execute", _sol_execute_always)
    scheduled = attention_module.schedule_aware_attention_kernel(
        "sol",
        attention_module._SOL,  # pyright: ignore[reportPrivateUsage]
    )
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule(
            "sol",
            0.0,
            1.0,
            SamplingParameterCurve(((0.0, 2.5), (1.0, 3.5))),
        ),
        (1.0, 0.0),
    )
    q = torch.randn(1, 2, 4, 128, dtype=torch.bfloat16)
    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        scheduled(q, q, q)

    assert fake.calls[0][3]["tau"] == 2.5


@dataclass(frozen=True, slots=True)
class _PackedAttentionFacts:
    sequence_length: int
    conditioning_prefix_length: int | None


def test_scheduled_sol_binds_conditioning_sink_from_packed_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_sol_available(monkeypatch)
    monkeypatch.setattr(attention_module, "_sol_can_execute", _sol_execute_always)
    scheduled = attention_module.schedule_aware_attention_kernel(
        "sol",
        attention_module._SOL,  # pyright: ignore[reportPrivateUsage]
    )
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule(
            "sol",
            0.0,
            1.0,
            attention_modifiers=(AttentionModifierSchedule("sol_conditioning_exact_kv", 0.0, 1.0),),
        ),
        (1.0, 0.0),
    )
    q = torch.randn(1, 2, 130, 128, dtype=torch.bfloat16)

    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        bound = attention_module.bind_packed_attention_kernel(
            scheduled, _PackedAttentionFacts(130, 65)
        )
        actual = bound(q, q, q)

    options = fake.calls[0][3]
    assert options["sink_blocks"] == [0, 2]
    assert torch.equal(actual, q + 1)


def test_scheduled_sol_conditioning_sink_fails_closed_without_matching_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_sol_available(monkeypatch)
    monkeypatch.setattr(attention_module, "_sol_can_execute", _sol_execute_always)
    scheduled = attention_module.schedule_aware_attention_kernel(
        "sol",
        attention_module._SOL,  # pyright: ignore[reportPrivateUsage]
    )
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule(
            "sol",
            0.0,
            1.0,
            attention_modifiers=(AttentionModifierSchedule("sol_conditioning_exact_kv", 0.0, 1.0),),
        ),
        (1.0, 0.0),
    )
    q = torch.randn(1, 2, 128, 128, dtype=torch.bfloat16)

    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        with pytest.raises(AttentionSelectionError, match="packed layout facts"):
            scheduled(q, q, q)
        excluded = attention_module.exclude_packed_attention_modifiers(scheduled)
        excluded(q, q, q)
        assert "sink_blocks" not in fake.calls[0][3]
        assert "sink_q" not in fake.calls[0][3]
        with pytest.raises(AttentionSelectionError, match="conditioning prefix"):
            attention_module.bind_packed_attention_kernel(
                scheduled, _PackedAttentionFacts(128, None)
            )
        bound = attention_module.bind_packed_attention_kernel(
            scheduled, _PackedAttentionFacts(129, 64)
        )
        with pytest.raises(AttentionSelectionError, match="does not match"):
            bound(q, q, q)
        with pytest.raises(AttentionSelectionError, match="central schedule-aware"):
            attention_module.bind_packed_attention_kernel(
                attention_module._SOL,  # pyright: ignore[reportPrivateUsage]
                _PackedAttentionFacts(128, 64),
            )


def test_sol_layout_copies_only_when_alignment_contract_requires_it() -> None:
    aligned = torch.randn(2, 3, 4, 128, dtype=torch.bfloat16)
    aligned_layout = attention_module._sol_layout(aligned)  # pyright: ignore[reportPrivateUsage]
    assert aligned_layout.untyped_storage().data_ptr() == aligned.untyped_storage().data_ptr()
    misaligned = torch.randn(2, 3, 4, 129, dtype=torch.bfloat16)[..., 1:]
    assert misaligned.shape[-1] == 128
    copied = attention_module._sol_layout(misaligned)  # pyright: ignore[reportPrivateUsage]
    assert copied.is_contiguous()
    assert copied.untyped_storage().data_ptr() != misaligned.untyped_storage().data_ptr()
    torch.testing.assert_close(copied, misaligned.transpose(1, 2))


@pytest.mark.parametrize(
    "case",
    ("mask", "causal", "gqa", "dtype", "head_dim", "cross", "empty", "device"),
)
def test_sol_kernel_preserves_unsupported_calls_through_sdpa(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _make_sol_available(monkeypatch)
    shape = (1, 4, 3, 128)
    q = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)
    kwargs: dict[str, Any] = {}
    if case == "mask":
        kwargs["mask"] = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    elif case == "causal":
        kwargs["causal"] = True
    elif case == "gqa":
        k = k[:, :2]
        v = v[:, :2]
        kwargs["enable_gqa"] = True
    elif case == "dtype":
        q, k, v = (tensor.to(torch.float64) for tensor in (q, k, v))
    elif case == "head_dim":
        q, k, v = (tensor[..., :64] for tensor in (q, k, v))
    elif case == "cross":
        k = torch.randn(1, 4, 5, 128)
        v = torch.randn_like(k)
    elif case == "empty":
        q, k, v = (tensor[:, :, :0] for tensor in (q, k, v))
    expected = select_attention("flux", "sdpa").kernel(q, k, v, **kwargs)
    actual = select_attention("flux", "sol").kernel(q, k, v, **kwargs)
    torch.testing.assert_close(actual, expected)
    assert fake.calls == []


@pytest.mark.parametrize("grad_member", ("q", "k", "v"))
def test_sol_validator_refuses_gradient_recording(grad_member: str) -> None:
    values = {
        name: torch.randn(1, 1, 2, 128, requires_grad=name == grad_member)
        for name in ("q", "k", "v")
    }
    with pytest.raises(AttentionValidationError, match="cannot record gradients"):
        attention_module._validate_sol_invocation(  # pyright: ignore[reportPrivateUsage]
            values["q"], values["k"], values["v"]
        )


def test_discovery_with_sol_binds_one_kitchen_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_sol_available(monkeypatch)
    monkeypatch.setattr(attention_module, "sage2_attention_available", lambda: False)
    monkeypatch.setattr(attention_module, "dinkster_kitchen_int8_available", lambda: True)
    fake_torch = SimpleNamespace(__version__="2.13.0+cu130", version=SimpleNamespace(hip=None))
    capabilities = attention_module.discover_attention_capabilities(
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert capabilities.available_policies == ("sdpa", "sol", "dinkster_kitchen_int8")
    assert tuple(name for name, _ in capabilities.provider_versions).count("dinkster-kitchen") == 1
    token = attention_module.discover_attention_route_token(
        "sol",
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=89,
    )
    assert token == _sol_token(torch_version="2.13.0+cu130")

    unsupported = attention_module.discover_attention_capabilities(
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=75,
    )
    assert "sol" not in unsupported.available_policies
    rocm_torch = SimpleNamespace(__version__="2.13.0+rocm6.4", version=SimpleNamespace(hip="6.4"))
    rocm = attention_module.discover_attention_capabilities(
        torch_module=rocm_torch,
        device_kind="rocm",
        device_sm=110,
    )
    assert "sol" not in rocm.available_policies


def test_resolve_role_attention_carries_authenticated_sol_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_sol_available(monkeypatch)
    token = _sol_token()
    monkeypatch.setattr(
        attention_module,
        "discover_attention_capabilities",
        lambda: _sol_capabilities_for_token(token),
    )
    selection = attention_module.resolve_role_attention("flux", "sol", token)
    assert cast("Any", selection.kernel).active_kernel() is attention_module._SOL  # pyright: ignore[reportPrivateUsage]
    assert selection.status.authenticated is True
    assert selection.status.provider_versions == token.provider_versions
    assert selection.status.device_sm == 89


def test_sol_provider_identity_binds_installed_kitchen_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_sol_available(monkeypatch)
    status = select_attention("flux", "sol").status
    assert attention_provider_identity(status) == (
        SOL_ATTENTION_PROVIDER,
        importlib.metadata.version("dinkster-kitchen"),
    )


class _FakeSageAttn:
    """Recording stand-in for the optional sageattention entry point."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]] = []

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.calls.append((q, k, v, kwargs))
        return torch.zeros_like(q)


def _sage_arm_always_enabled(sm: int) -> bool:
    del sm
    return True


def _sage_arm_never_enabled(sm: int) -> bool:
    del sm
    return False


def _make_sage_available(monkeypatch: pytest.MonkeyPatch) -> _FakeSageAttn:
    fake = _FakeSageAttn()
    monkeypatch.setattr(attention_module, "_SAGE_ATTENTION", fake)
    monkeypatch.setattr(attention_module, "sage2_distribution_version", lambda: "2.2.0")
    monkeypatch.setattr(attention_module, "_SAGE_DEVICE_SUPPORTED", lambda: True)
    monkeypatch.setattr(attention_module, "_sage2_arm_enabled", _sage_arm_always_enabled)
    return fake


def _sage_fake_invocation_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    del q, k, v
    return True


def _sm120_capability(device: object = None) -> tuple[int, int]:
    del device
    return (12, 0)


def _sm75_capability(device: object = None) -> tuple[int, int]:
    del device
    return (7, 5)


def _sage_token(
    *,
    contract: str = ATTENTION_ADAPTER_CONTRACT,
    torch_version: str | None = None,
) -> AttentionRouteToken:
    torch_version = str(torch.__version__) if torch_version is None else torch_version
    return AttentionRouteToken(
        1,
        tuple(AttentionRoute(role, "sage", "sdpa") for role in ROLES),
        (("sageattention", "2.2.0"), ("torch", torch_version)),
        contract,
        "cuda",
        120,
        torch_version.split("+")[0],
        "sage",
    )


def _sage_capabilities_for_token(token: AttentionRouteToken) -> AttentionCapabilityEvidence:
    return AttentionCapabilityEvidence(
        version=1,
        available_policies=("sdpa", "sage"),
        provider_versions=token.provider_versions,
        adapter_contract_revision=token.adapter_contract_revision,
        device_kind=token.device_kind,
        device_sm=token.device_sm,
        sdpa_torch_runtime=token.sdpa_torch_runtime,
    )


@pytest.mark.parametrize("role", ROLES)
def test_sage_policy_selects_sage_for_every_role(
    role: AttentionRole, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_sage_available(monkeypatch)
    selection = select_attention(role, "sage")
    assert selection.status.requested_policy == "sage"
    assert selection.status.primary == "sage"
    assert selection.kernel is attention_module._SAGE2  # pyright: ignore[reportPrivateUsage]
    assert selection.status.fallback == "sdpa"
    assert selection.status.reason


@pytest.mark.parametrize(
    ("managed_import", "installed", "expected", "lookup"),
    (
        (
            True,
            {"dinkster-kitchen": "2.2.0.post1"},
            "2.2.0.post1",
            "dinkster-kitchen",
        ),
        (
            False,
            {"sageattention": "2.2.0"},
            "2.2.0",
            "sageattention",
        ),
        (
            False,
            {"dinkster-kitchen": "2.2.0.post1", "sageattention": "2.2.0"},
            "2.2.0",
            "sageattention",
        ),
        (True, {}, None, "dinkster-kitchen"),
        (True, {"sageattention": "2.2.0"}, None, "dinkster-kitchen"),
        (False, {}, None, "sageattention"),
    ),
)
def test_sage_distribution_version_matches_imported_distribution(
    monkeypatch: pytest.MonkeyPatch,
    managed_import: bool,
    installed: dict[str, str],
    expected: str | None,
    lookup: str,
) -> None:
    requested: list[str] = []

    def distribution_version(name: str) -> str:
        requested.append(name)
        try:
            return installed[name]
        except KeyError as error:
            raise importlib.metadata.PackageNotFoundError(name) from error

    package = SimpleNamespace(__distribution__="dinkster-kitchen") if managed_import else None
    monkeypatch.setattr(attention_module, "_SAGE_PACKAGE", package)
    monkeypatch.setattr(attention_module, "_distribution_version", distribution_version)
    assert attention_module.sage2_distribution_version() == expected
    assert requested == [lookup]


def test_sage_policy_falls_back_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_module, "_SAGE_ATTENTION", None)
    assert "not installed" in select_attention("flux", "sage").status.reason
    _make_sage_available(monkeypatch)
    monkeypatch.setattr(attention_module, "sage2_distribution_version", lambda: None)
    assert "not installed" in select_attention("flux", "sage").status.reason
    _make_sage_available(monkeypatch)
    monkeypatch.setattr(attention_module, "_SAGE_DEVICE_SUPPORTED", lambda: False)
    assert "does not support this machine" in select_attention("flux", "sage").status.reason
    token = attention_module.discover_attention_route_token("sage")
    assert token.version == 3
    assert all(route.primary == "sdpa" for route in token.routes)


def test_sage_kernel_serves_inadmissible_invocations_through_sdpa_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_sage_available(monkeypatch)
    sage = select_attention("qwen", "sage").kernel
    sdpa = select_attention("qwen", "sdpa").kernel
    torch.manual_seed(11)
    q = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    v = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    mask = torch.ones(4, 6, dtype=torch.bool)
    # Masked invocations must never reach sageattn: its public entry point
    # has no mask parameter, so executing there would silently drop the mask.
    torch.testing.assert_close(sage(q, k, v, mask=mask), sdpa(q, k, v, mask=mask))
    fq, fk, fv = q.float(), k.float(), v.float()
    torch.testing.assert_close(sage(fq, fk, fv), sdpa(fq, fk, fv))
    wq = torch.randn(1, 1, 6, 300, dtype=torch.bfloat16)
    wk = torch.randn(1, 1, 6, 300, dtype=torch.bfloat16)
    wv = torch.randn(1, 1, 6, 300, dtype=torch.bfloat16)
    torch.testing.assert_close(sage(wq, wk, wv), sdpa(wq, wk, wv))
    # Upstream pads q, k, and v to one shared head dim, so a narrower v
    # cannot execute there.
    nv = torch.randn(2, 3, 6, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(sage(q, k, nv), sdpa(q, k, nv))
    # Upstream applies its causal mask only to equal query/key lengths.
    torch.testing.assert_close(sage(q, k, v, causal=True), sdpa(q, k, v, causal=True))
    assert fake.calls == []


def test_sage_kernel_falls_back_for_non_cuda_inputs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = _make_sage_available(monkeypatch)
    sage = select_attention("unet", "sage").kernel
    q = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    v = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    assert torch.equal(sage(q, k, v), kernel()(q, k, v))
    assert fake.calls == []
    assert "unavailable on cpu; using SDPA" in caplog.text


def test_sage_kernel_dispatches_upstream_call_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_sage_available(monkeypatch)
    monkeypatch.setattr(
        attention_module, "_sage2_invocation_supported", _sage_fake_invocation_supported
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", _sm120_capability)
    sage = select_attention("unet", "sage").kernel
    strided = torch.randn(2, 3, 4, 128, dtype=torch.bfloat16)[..., ::2]
    k = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    v = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    assert strided.stride(-1) != 1
    result = sage(strided, k, v, causal=True, scale=0.5)
    assert result.shape == strided.shape
    (called_q, called_k, called_v, kwargs) = fake.calls[0]
    # Upstream asserts contiguous last dims, so the adapter must fix them up.
    assert called_q.stride(-1) == 1
    assert called_k.stride(-1) == 1
    assert called_v.stride(-1) == 1
    torch.testing.assert_close(called_q, strided.contiguous())
    assert kwargs == {"tensor_layout": "HND", "is_causal": True, "sm_scale": 0.5}
    q = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    cross_k = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    cross_v = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    sage(q, cross_k, cross_v)
    (_, _, _, default_kwargs) = fake.calls[1]
    assert default_kwargs == {"tensor_layout": "HND", "is_causal": False, "sm_scale": None}


def test_sage_kernel_falls_back_on_unsupported_device_sm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_sage_available(monkeypatch)
    monkeypatch.setattr(
        attention_module, "_sage2_invocation_supported", _sage_fake_invocation_supported
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", _sm75_capability)
    sage = select_attention("unet", "sage").kernel
    sdpa = select_attention("unet", "sdpa").kernel
    q = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    v = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(sage(q, k, v), sdpa(q, k, v))
    assert fake.calls == []


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "v_shape"),
    [
        ((0, 2, 3, 64), (0, 2, 5, 64), (0, 2, 5, 64)),
        ((1, 0, 3, 64), (1, 0, 5, 64), (1, 0, 5, 64)),
        ((1, 2, 0, 64), (1, 2, 5, 64), (1, 2, 5, 64)),
        ((1, 2, 3, 64), (1, 2, 0, 64), (1, 2, 0, 64)),
        ((1, 2, 3, 0), (1, 2, 5, 0), (1, 2, 5, 0)),
    ],
)
def test_sage_kernel_serves_empty_geometry_through_sdpa_fallback(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_sage_available(monkeypatch)
    sage = select_attention("unet", "sage").kernel
    sdpa = select_attention("unet", "sdpa").kernel
    q = torch.randn(q_shape, dtype=torch.bfloat16)
    k = torch.randn(k_shape, dtype=torch.bfloat16)
    v = torch.randn(v_shape, dtype=torch.bfloat16)
    torch.testing.assert_close(sage(q, k, v), sdpa(q, k, v))
    assert fake.calls == []


def test_sage_kernel_falls_back_when_compiled_arm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_sage_available(monkeypatch)
    monkeypatch.setattr(
        attention_module, "_sage2_invocation_supported", _sage_fake_invocation_supported
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", _sm120_capability)
    monkeypatch.setattr(attention_module, "_sage2_arm_enabled", _sage_arm_never_enabled)
    sage = select_attention("unet", "sage").kernel
    sdpa = select_attention("unet", "sdpa").kernel
    q = torch.randn(2, 3, 4, 64, dtype=torch.bfloat16)
    k = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    v = torch.randn(2, 3, 6, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(sage(q, k, v), sdpa(q, k, v))
    assert fake.calls == []


def test_sage_arm_probe_requires_compiled_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    # Upstream guards each compiled arm behind a try/except import flag, so
    # the probe must confirm the arm the current SM dispatches into.
    core = SimpleNamespace(SM80_ENABLED=True, SM89_ENABLED=False, SM90_ENABLED=True)
    monkeypatch.setattr(attention_module, "_SAGE_CORE", core)
    assert attention_module._sage2_arm_enabled(80)  # pyright: ignore[reportPrivateUsage]
    assert attention_module._sage2_arm_enabled(86)  # pyright: ignore[reportPrivateUsage]
    assert not attention_module._sage2_arm_enabled(89)  # pyright: ignore[reportPrivateUsage]
    assert attention_module._sage2_arm_enabled(90)  # pyright: ignore[reportPrivateUsage]
    assert not attention_module._sage2_arm_enabled(120)  # pyright: ignore[reportPrivateUsage]
    assert not attention_module._sage2_arm_enabled(75)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(attention_module, "_SAGE_CORE", None)
    assert not attention_module._sage2_arm_enabled(86)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("grad_member", ["q", "k", "v"])
def test_sage_validator_refuses_gradient_recording(grad_member: str) -> None:
    tensors = {
        name: torch.randn(1, 1, 2, 8, dtype=torch.bfloat16, requires_grad=(name == grad_member))
        for name in ("q", "k", "v")
    }
    with pytest.raises(AttentionValidationError, match="cannot record gradients"):
        attention_module._sage2_invocation_supported(  # pyright: ignore[reportPrivateUsage]
            tensors["q"], tensors["k"], tensors["v"]
        )


def test_discovery_with_sage_policy_binds_routes_and_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_sage_available(monkeypatch)
    monkeypatch.setattr(attention_module, "_KITCHEN_SOL_ATTENTION", None)
    monkeypatch.setattr(attention_module, "_KITCHEN_AVAILABLE", lambda: False)
    fake_torch = SimpleNamespace(__version__="2.13.0+cu130", version=SimpleNamespace(hip=None))
    capabilities = attention_module.discover_attention_capabilities(
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=120,
    )
    assert capabilities.available_policies == ("sdpa", "sage")
    assert dict(capabilities.provider_versions) == {
        "torch": "2.13.0+cu130",
        "sageattention": "2.2.0",
    }
    token = attention_module.discover_attention_route_token(
        "sage",
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=120,
    )
    assert token == _sage_token(torch_version="2.13.0+cu130")
    auto = attention_module.discover_attention_route_token(
        "auto",
        torch_module=fake_torch,
        device_kind="cuda",
        device_sm=120,
    )
    auto_routes = {route.role: route for route in auto.routes}
    assert auto_routes["vae"] == AttentionRoute("vae", "sdpa", "bounded")
    assert all(
        (route.primary, route.fallback) == ("sdpa", None)
        for role, route in auto_routes.items()
        if role != "vae"
    )
    assert auto.provider_versions == (("torch", "2.13.0+cu130"),)


def test_resolve_role_attention_carries_authenticated_sage_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_sage_available(monkeypatch)
    token = _sage_token()
    monkeypatch.setattr(
        attention_module,
        "discover_attention_capabilities",
        lambda: _sage_capabilities_for_token(token),
    )
    selection = attention_module.resolve_role_attention("flux", "sage", token)
    assert cast("Any", selection.kernel).active_kernel() is attention_module._SAGE2  # pyright: ignore[reportPrivateUsage]
    assert selection.status.primary == "sage"
    assert selection.status.fallback == "sdpa"
    assert selection.status.authenticated is True
    assert selection.status.provider_versions == token.provider_versions
    assert selection.status.device_sm == 120


def test_sage_provider_identity_binds_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_sage_available(monkeypatch)
    status = select_attention("unet", "sage").status
    assert attention_provider_identity(status) == (SAGE2_PROVIDER, "2.2.0")
    monkeypatch.setattr(attention_module, "sage2_distribution_version", lambda: None)
    with pytest.raises(AttentionSelectionError, match="version is unavailable"):
        attention_provider_identity(status)
