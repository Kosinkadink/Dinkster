"""Stage 4c slice 1: cast-at-use (ops.cast_weight / ops.DeferredPatch).

Pins the reference cast_bias_weight pipeline order (comfy/ops.py
@ 947c2749) over explicit inputs: move at storage dtype -> cast to
compute dtype (dequantizing fp8-scaled storage directly into it) ->
deferred functions last, on an owned buffer. Everything here runs on
CPU; nothing fakes a device move. CUDA/Triton validation of the same
pipeline (device moves, kitchen CUDA backend dispatch, inductor
compile, multi-GPU) lives in test_gpu.py, capability-gated.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import cast as type_cast

import pytest
import torch
from dinkster_inference.patches import AdapterPatch, DiffPatch, PatchEntry, SetPatch
from dinkster_inference_torch import (
    INITLESS,
    CastOperations,
    DeferredPatch,
    PreparedPatchSource,
    apply_patches,
    cast_weight,
    quantize_fp8_scaled,
)
from dinkster_inference_torch import quant_linear as quant_linear_mod


def make_fp8(scale_val: float = 0.5, orig: torch.dtype = torch.float32):
    source = torch.randn(8, 6, generator=torch.Generator().manual_seed(77)) * scale_val
    return quantize_fp8_scaled(source.clone().to(orig), torch.float8_e4m3fn)


# ------------------------------------------------------- plain tensors


def test_bare_cast_same_dtype_returns_input() -> None:
    """No functions, no dtype change, no device move -> the stored
    tensor itself comes back (the reference returns the module weight
    uncopied); callers must treat it as read-only."""
    stored = torch.randn(4, 4)
    out = cast_weight(stored, dtype=torch.float32)
    assert out is stored


def test_cast_to_compute_dtype() -> None:
    stored = torch.randn(4, 4)
    out = cast_weight(stored, dtype=torch.float16)
    assert out.dtype == torch.float16
    assert torch.equal(out, stored.to(torch.float16))


def test_explicit_same_device_move_is_not_a_copy() -> None:
    """device= pointing at the tensor's own device without functions
    stays copy-free (reference cast_to returns the input when device
    and dtype already match)."""
    stored = torch.randn(4, 4)
    out = cast_weight(stored, dtype=torch.float32, device=stored.device)
    assert out is stored


def test_functions_receive_owned_buffer() -> None:
    """A weight function may mutate in place (the reference forces
    copy=True on the move when functions exist); the stored tensor
    must never see the mutation - even in the no-op cast case."""
    stored = torch.randn(4, 4)
    snapshot = stored.clone()

    def mutate(w: torch.Tensor) -> torch.Tensor:
        return w.add_(1.0)

    out = cast_weight(stored, dtype=torch.float32, functions=[mutate])
    assert torch.equal(stored, snapshot)
    assert torch.equal(out, snapshot + 1.0)


def test_functions_run_in_order_on_compute_dtype() -> None:
    stored = torch.randn(4, 4)
    seen: list[torch.dtype] = []

    def add_one(w: torch.Tensor) -> torch.Tensor:
        seen.append(w.dtype)
        return w + 1.0

    def double(w: torch.Tensor) -> torch.Tensor:
        seen.append(w.dtype)
        return w * 2.0

    out = cast_weight(stored, dtype=torch.float16, functions=[add_one, double])
    # cast happens BEFORE the functions; both see compute dtype
    assert seen == [torch.float16, torch.float16]
    expected = (stored.to(torch.float16) + 1.0) * 2.0
    assert torch.equal(out, expected)


# --------------------------------------------------- fp8-scaled storage


def test_fp8_cast_equals_dequantize() -> None:
    stored = make_fp8()
    out = cast_weight(stored, dtype=torch.float16)
    assert out.dtype == torch.float16
    assert torch.equal(out, stored.dequantize(torch.float16))


def test_fp8_dequantizes_directly_into_compute_dtype() -> None:
    """The dequantize product is formed AT the compute dtype
    (qdata.to(dtype) * scale.to(dtype)), never via a detour through
    orig_dtype - the reference casts the wrapper straight to the
    requested dtype."""
    stored = make_fp8(orig=torch.float32)
    out = cast_weight(stored, dtype=torch.bfloat16)
    direct = stored.qdata.to(torch.bfloat16) * stored.scale.to(torch.bfloat16)
    assert torch.equal(out, direct)


def test_fp8_cast_output_is_owned() -> None:
    """The dequantize product is always a fresh buffer: mutating it
    never touches the stored qdata/scale."""
    stored = make_fp8()
    q_snapshot = stored.qdata.clone()
    out = cast_weight(stored, dtype=torch.float32)
    out.add_(5.0)
    assert torch.equal(stored.qdata.float(), q_snapshot.float())


# ------------------------------------------------------- DeferredPatch


def diff_entry(
    shape: tuple[int, ...], fill: float, strength: float = 1.0
) -> PatchEntry[torch.Tensor]:
    return PatchEntry(DiffPatch(torch.full(shape, fill)), strength=strength)


def test_deferred_patch_pins_intermediate_to_weight_dtype() -> None:
    """LowVramPatch semantics: patches applied at cast time run at the
    weight's own compute dtype, NOT the fp32 intermediate the
    patched-at-load path uses."""
    stored = torch.randn(4, 4)
    entries = (diff_entry((4, 4), 0.25, strength=0.8),)
    patch = DeferredPatch("k", entries)
    out = cast_weight(stored, dtype=torch.float16, functions=[patch])
    expected = apply_patches(
        stored.to(torch.float16),
        list(entries),
        key="k",
        intermediate_dtype=torch.float16,
    )
    assert out.dtype == torch.float16
    assert torch.equal(out, expected)


def test_deferred_patch_on_fp8_storage() -> None:
    """Low-VRAM cast of quantized storage: dequantize to compute
    dtype, then patch there - the stored fp8 weight is untouched."""
    stored = make_fp8()
    q_snapshot = stored.qdata.clone()
    entries = (diff_entry((8, 6), 0.5),)
    out = cast_weight(
        stored,
        dtype=torch.float16,
        functions=[DeferredPatch("k", entries)],
    )
    expected = apply_patches(
        stored.dequantize(torch.float16),
        list(entries),
        key="k",
        intermediate_dtype=torch.float16,
    )
    assert torch.equal(out, expected)
    assert torch.equal(stored.qdata.float(), q_snapshot.float())


def test_multiple_deferred_patches_apply_in_order() -> None:
    """Two deferred keys stack like the reference's weight_function
    list: later entries see the earlier entries' output."""
    stored = torch.zeros(4, 4)
    set_patch = DeferredPatch("a", (PatchEntry(SetPatch(torch.full((4, 4), 3.0))),))
    add_patch = DeferredPatch("b", (diff_entry((4, 4), 1.0),))
    out = cast_weight(stored, dtype=torch.float32, functions=[set_patch, add_patch])
    assert torch.equal(out, torch.full((4, 4), 4.0))
    # reversed order gives set-last semantics
    out2 = cast_weight(stored, dtype=torch.float32, functions=[add_patch, set_patch])
    assert torch.equal(out2, torch.full((4, 4), 3.0))
    assert torch.equal(stored, torch.zeros(4, 4))


def test_prepared_patch_source_alignment_storage_views_and_single_use() -> None:
    half = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    full = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    deferred = DeferredPatch(
        "k",
        (PatchEntry(DiffPatch(half)), PatchEntry(DiffPatch(full))),
    )
    prepared = PreparedPatchSource(deferred, alignment=1024)
    assert prepared.memory_required() == 2048
    destination = torch.full((2048,), 0xA5, dtype=torch.uint8)
    entries = prepared.prepare(destination, non_blocking=False)
    first = entries[0].value
    second = entries[1].value
    assert isinstance(first, DiffPatch) and isinstance(second, DiffPatch)
    assert first.value.dtype == half.dtype and second.value.dtype == full.dtype
    assert torch.equal(first.value.view(torch.uint8), half.view(torch.uint8))
    assert torch.equal(second.value.view(torch.uint8), full.view(torch.uint8))
    assert first.value.data_ptr() == destination.data_ptr()
    assert second.value.data_ptr() == destination.data_ptr() + 1024
    original_first = deferred.entries[0].value
    original_second = deferred.entries[1].value
    assert isinstance(original_first, DiffPatch)
    assert isinstance(original_second, DiffPatch)
    assert original_first.value is half
    assert original_second.value is full

    prepared.commit(entries)
    with pytest.raises(RuntimeError, match="already committed"):
        prepared.commit(entries)
    actual = prepared(torch.zeros(3, dtype=torch.float16))
    assert torch.equal(actual, half + full.to(torch.float16))
    assert prepared._prepared is None  # pyright: ignore[reportPrivateUsage]


def test_prepared_patch_source_missing_adapter_warns_once_and_degrades(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class LegacyAdapter:
        def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
            return base

        def calculate(
            self,
            weight: torch.Tensor,
            *,
            strength: float,
            function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ) -> torch.Tensor:
            del strength, function
            return weight

    adapter = LegacyAdapter()
    deferred = DeferredPatch(
        "legacy",
        (
            PatchEntry(AdapterPatch(adapter)),
            PatchEntry(DiffPatch(torch.ones(2))),
        ),
    )
    prepared = PreparedPatchSource(deferred)
    assert prepared.memory_required() == 1024
    assert prepared.memory_required() == 1024
    entries = prepared.prepare(torch.empty(1024, dtype=torch.uint8), non_blocking=False)
    value = entries[0].value
    assert isinstance(value, AdapterPatch) and value.adapter is adapter
    diagnostics = [
        record for record in caplog.records if "dinkster.patch_payload_unstaged" in record.message
    ]
    assert len(diagnostics) == 1


# --------------------------------------------- cast-at-use Operations
#
# CastOperations = the reference's manual_cast @ 947c2749 with the
# compute dtype fixed at bind time: storage dtype and compute dtype
# decouple (fp16-stored text encoders run at fp32 compute, the only
# shape in which T5-XXL does not overflow). State-dict layout must
# stay identical to the stock layers.


def _cast_ops_pairs() -> list[tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]]:
    """(initless, cast-at-use, input) triples covering every factory,
    with identical deterministic weights."""
    ops = CastOperations(torch.float32)
    gen = torch.Generator().manual_seed(9)

    def pair(name: str):
        stock = getattr(INITLESS, name)
        cast = getattr(ops, name)
        return stock, cast

    triples: list[tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]] = []

    def add(stock: torch.nn.Module, cast: torch.nn.Module, x: torch.Tensor) -> None:
        state = {k: torch.randn(v.shape, generator=gen) for k, v in stock.state_dict().items()}
        stock.load_state_dict(state, assign=True)
        cast.load_state_dict(state, assign=True)
        triples.append((stock, cast, x))

    s, c = pair("linear")
    add(s(6, 4), c(6, 4), torch.randn(3, 6, generator=gen))
    s, c = pair("conv2d")
    add(
        s(2, 3, 3, stride=1, padding=1),
        c(2, 3, 3, stride=1, padding=1),
        torch.randn(1, 2, 8, 8, generator=gen),
    )
    s, c = pair("conv1d")
    add(
        s(4, 6, 3, stride=2, padding=2, dilation=2, groups=2),
        c(4, 6, 3, stride=2, padding=2, dilation=2, groups=2),
        torch.randn(2, 4, 13, generator=gen),
    )
    s, c = pair("conv_transpose1d")
    add(
        s(4, 6, 3, stride=2, padding=1, output_padding=1, groups=2, dilation=2),
        c(4, 6, 3, stride=2, padding=1, output_padding=1, groups=2, dilation=2),
        torch.randn(2, 4, 7, generator=gen),
    )
    s, c = pair("conv3d")
    add(
        s(
            4,
            6,
            (2, 3, 3),
            stride=(1, 2, 1),
            padding=(1, 1, 2),
            dilation=(1, 2, 1),
            groups=2,
        ),
        c(
            4,
            6,
            (2, 3, 3),
            stride=(1, 2, 1),
            padding=(1, 1, 2),
            dilation=(1, 2, 1),
            groups=2,
        ),
        torch.randn(2, 4, 5, 9, 8, generator=gen),
    )
    s, c = pair("group_norm")
    add(
        s(8, num_groups=4),
        c(8, num_groups=4),
        torch.randn(2, 8, 5, 5, generator=gen),
    )
    s, c = pair("layer_norm")
    add(s(16), c(16), torch.randn(4, 16, generator=gen))
    s, c = pair("embedding")
    add(s(10, 8), c(10, 8), torch.arange(6).reshape(2, 3))
    s, c = pair("rms_norm")
    add(s(16, eps=1e-6), c(16, eps=1e-6), torch.randn(4, 16, generator=gen))
    return triples


def test_cast_operations_match_stock_layers_at_storage_dtype() -> None:
    """fp32 storage at fp32 compute: cast_weight is a no-op passthrough,
    so every factory's module must equal its stock counterpart bit for
    bit on the same weights."""
    for stock, cast, x in _cast_ops_pairs():
        assert torch.equal(cast(x), stock(x))


def test_cast_operations_state_dict_layout_matches_stock() -> None:
    for stock, cast, _ in _cast_ops_pairs():
        stock_sd = stock.state_dict()
        cast_sd = cast.state_dict()
        assert set(stock_sd) == set(cast_sd)
        for key, value in cast_sd.items():
            assert value.shape == stock_sd[key].shape


def test_cast_operations_decouple_storage_from_compute() -> None:
    """fp16-stored weights, fp32 compute: outputs come back at the
    compute dtype and match the fp32 forward over the fp16-rounded
    weights exactly (the cast IS the fp32 forward's weight)."""
    for stock, cast, x in _cast_ops_pairs():
        half_state = {k: v.to(torch.float16) for k, v in cast.state_dict().items()}
        cast.load_state_dict(half_state, assign=True)
        stock.load_state_dict(
            {k: v.to(torch.float32) for k, v in half_state.items()},
            assign=True,
        )
        if x.dtype == torch.int64:  # embedding ids
            got = cast(x)
        else:
            got = cast(x.to(torch.float32))
        assert got.dtype == torch.float32
        assert torch.equal(got, stock(x))


def test_cast_embedding_gathers_rows_before_compute_cast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = CastOperations(torch.float32).embedding(128, 16)
    weight = torch.randn(128, 16, generator=torch.Generator().manual_seed(19)).to(torch.float16)
    embedding.load_state_dict({"weight": weight}, assign=True)
    token_ids = torch.tensor([[7, 3, 7]])
    seen: list[torch.Tensor] = []
    original = torch.nn.functional.embedding

    def tracked(input: torch.Tensor, stored: torch.Tensor, *args: Any) -> torch.Tensor:
        seen.append(stored)
        return original(input, stored, *args)

    monkeypatch.setattr(torch.nn.functional, "embedding", tracked)
    with torch.no_grad():
        output = embedding(token_ids)

    assert seen == [embedding.weight]
    assert seen[0].dtype == torch.float16
    assert output.dtype == torch.float32
    assert torch.equal(output, original(token_ids, weight).to(torch.float32))


def test_cast_embedding_accumulates_trainable_repeated_rows_at_compute_dtype() -> None:
    embedding = CastOperations(torch.float32).embedding(1, 1)
    embedding.load_state_dict({"weight": torch.ones(1, 1, dtype=torch.float16)}, assign=True)
    token_ids = torch.zeros(70_000, dtype=torch.long)

    (embedding(token_ids) * 1e-5).sum().backward()

    assert embedding.weight.grad is not None
    assert embedding.weight.grad.item() == pytest.approx(0.7002, abs=5e-4)


def test_cast_operations_gradients_reach_stored_parameters() -> None:
    """The cast is differentiable: grads flow back to the fp16-stored
    parameters (training over low-precision storage)."""
    ops = CastOperations(torch.float32)
    linear = ops.linear(6, 4)
    linear.load_state_dict(
        {
            k: torch.randn(v.shape, generator=torch.Generator().manual_seed(3))
            .to(torch.float16)
            .requires_grad_()
            for k, v in linear.state_dict().items()
        },
        assign=True,
    )
    out = linear(torch.randn(2, 6))
    out.square().mean().backward()
    weight = linear.weight
    assert weight.grad is not None
    assert weight.grad.dtype == torch.float16
    assert bool(torch.isfinite(weight.grad).all())


def _convolution_cases(
    operations: Any,
    *,
    bias: bool = True,
) -> list[tuple[torch.nn.Module, type[torch.nn.Module], dict[str, object]]]:
    return [
        (
            operations.conv1d(
                4,
                6,
                3,
                stride=2,
                padding=2,
                dilation=2,
                groups=2,
                bias=bias,
            ),
            torch.nn.Conv1d,
            {
                "stride": (2,),
                "padding": (2,),
                "dilation": (2,),
                "groups": 2,
            },
        ),
        (
            operations.conv_transpose1d(
                4,
                6,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=2,
                bias=bias,
                dilation=2,
            ),
            torch.nn.ConvTranspose1d,
            {
                "stride": (2,),
                "padding": (1,),
                "output_padding": (1,),
                "dilation": (2,),
                "groups": 2,
            },
        ),
        (
            operations.conv3d(
                4,
                6,
                (2, 3, 3),
                stride=(1, 2, 1),
                padding=(1, 1, 2),
                dilation=(1, 2, 1),
                groups=2,
                bias=bias,
            ),
            torch.nn.Conv3d,
            {
                "stride": (1, 2, 1),
                "padding": (1, 1, 2),
                "dilation": (1, 2, 1),
                "groups": 2,
            },
        ),
    ]


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
def test_convolution_factories_match_stock_geometry_and_state_layout(
    cast_at_use: bool,
    bias: bool,
) -> None:
    operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    for module, stock_type, geometry in _convolution_cases(operations, bias=bias):
        assert isinstance(module, stock_type)
        for name, expected in geometry.items():
            assert getattr(module, name) == expected
        state = module.state_dict()
        assert set(state) == ({"weight", "bias"} if bias else {"weight"})
        assert state["weight"].shape == (
            (4, 3, 3)
            if stock_type is torch.nn.ConvTranspose1d
            else ((6, 2, 2, 3, 3) if stock_type is torch.nn.Conv3d else (6, 2, 3))
        )
        if bias:
            assert state["bias"].shape == (6,)


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
def test_convolution_factories_match_stock_forward_and_gradients(
    cast_at_use: bool,
) -> None:
    operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    inputs = (
        torch.randn(2, 4, 13),
        torch.randn(2, 4, 7),
        torch.randn(2, 4, 5, 9, 8),
    )
    for (module, stock_type, _), input_value in zip(
        _convolution_cases(operations), inputs, strict=True
    ):
        generator = torch.Generator().manual_seed(107)
        storage_dtype = torch.float16 if cast_at_use else torch.float32
        state = {
            key: torch.randn(value.shape, generator=generator).to(storage_dtype)
            for key, value in module.state_dict().items()
        }
        module.load_state_dict(state, assign=True)
        stock = stock_type(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True,
            **(
                {"output_padding": module.output_padding}
                if isinstance(module, torch.nn.ConvTranspose1d)
                else {}
            ),
        )
        stock.load_state_dict({key: value.float() for key, value in state.items()}, assign=True)
        actual_input = input_value.clone().requires_grad_()
        expected_input = input_value.clone().requires_grad_()
        if isinstance(module, torch.nn.ConvTranspose1d):
            actual = module(actual_input, output_size=[15])
            expected = stock(expected_input, output_size=[15])
        else:
            actual = module(actual_input)
            expected = stock(expected_input)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        assert actual_input.grad is not None and expected_input.grad is not None
        torch.testing.assert_close(actual_input.grad, expected_input.grad)
        for (name, parameter), expected_parameter in zip(
            module.named_parameters(), stock.parameters(), strict=True
        ):
            assert parameter.grad is not None, name
            assert expected_parameter.grad is not None, name
            torch.testing.assert_close(
                parameter.grad.float(),
                expected_parameter.grad,
                atol=1e-3 if cast_at_use else 0.0,
                rtol=1e-3 if cast_at_use else 0.0,
            )
            assert parameter.dtype == storage_dtype


def test_convolution_factories_skip_parameter_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parameter initialization must be skipped")

    monkeypatch.setattr(torch.nn.init, "kaiming_uniform_", refuse)
    for operations in (INITLESS, CastOperations(torch.float32)):
        assert len(_convolution_cases(operations)) == 3


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
def test_convolution_factories_delegate_malformed_geometry_to_torch(
    cast_at_use: bool,
) -> None:
    operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    for factory in (
        operations.conv1d,
        operations.conv_transpose1d,
        operations.conv3d,
    ):
        with pytest.raises(ValueError, match="divisible by groups"):
            factory(3, 6, 3, groups=2)


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
def test_conv2d_factory_preserves_full_geometry_and_forward(cast_at_use: bool) -> None:
    operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    module = operations.conv2d(
        4,
        6,
        (3, 2),
        stride=(2, 1),
        padding=(2, 1),
        dilation=(2, 1),
        groups=2,
        padding_mode="replicate",
    )
    stock = torch.nn.Conv2d(
        4,
        6,
        (3, 2),
        stride=(2, 1),
        padding=(2, 1),
        dilation=(2, 1),
        groups=2,
        padding_mode="replicate",
    )
    generator = torch.Generator().manual_seed(409)
    storage_dtype = torch.float16 if cast_at_use else torch.float32
    state = {
        key: torch.randn(value.shape, generator=generator).to(storage_dtype)
        for key, value in module.state_dict().items()
    }
    module.load_state_dict(state, assign=True)
    stock.load_state_dict({key: value.float() for key, value in state.items()}, assign=True)
    input_value = torch.randn(2, 4, 11, 13, generator=generator)

    assert module.kernel_size == stock.kernel_size == (3, 2)
    assert module.stride == stock.stride == (2, 1)
    assert module.padding == stock.padding == (2, 1)
    assert module.dilation == stock.dilation == (2, 1)
    assert module.groups == stock.groups == 2
    assert module.padding_mode == stock.padding_mode == "replicate"
    assert torch.equal(module(input_value), stock(input_value))


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
def test_conv_transpose2d_factory_preserves_output_size_semantics(cast_at_use: bool) -> None:
    operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    module = operations.conv_transpose2d(
        4,
        6,
        (3, 2),
        stride=(2, 3),
        padding=(1, 0),
        groups=2,
        dilation=(1, 2),
    )
    stock = torch.nn.ConvTranspose2d(
        4,
        6,
        (3, 2),
        stride=(2, 3),
        padding=(1, 0),
        groups=2,
        dilation=(1, 2),
    )
    generator = torch.Generator().manual_seed(410)
    storage_dtype = torch.float16 if cast_at_use else torch.float32
    state = {
        key: torch.randn(value.shape, generator=generator).to(storage_dtype)
        for key, value in module.state_dict().items()
    }
    module.load_state_dict(state, assign=True)
    stock.load_state_dict({key: value.float() for key, value in state.items()}, assign=True)
    input_value = torch.randn(2, 4, 5, 4, generator=generator)

    assert module.state_dict()["weight"].shape == (4, 3, 3, 2)
    assert torch.equal(
        module(input_value, output_size=[10, 13]),
        stock(input_value, output_size=[10, 13]),
    )


@pytest.mark.parametrize("cast_at_use", [False, True], ids=["initless", "cast"])
@pytest.mark.parametrize("affine", [False, True], ids=["no_affine", "affine"])
@pytest.mark.parametrize("track_running_stats", [False, True], ids=["batch_stats", "running"])
def test_batch_norm2d_factory_preserves_eval_variants(
    cast_at_use: bool,
    affine: bool,
    track_running_stats: bool,
) -> None:
    operations = CastOperations(torch.float32) if cast_at_use else INITLESS
    module = operations.batch_norm2d(
        4,
        eps=2e-4,
        momentum=None,
        affine=affine,
        track_running_stats=track_running_stats,
    ).eval()
    stock = torch.nn.BatchNorm2d(
        4,
        eps=2e-4,
        momentum=None,
        affine=affine,
        track_running_stats=track_running_stats,
    ).eval()
    generator = torch.Generator().manual_seed(411)
    storage_dtype = torch.float16 if cast_at_use else torch.float32
    state = {
        key: (
            torch.randint(0, 10, value.shape, generator=generator)
            if not value.is_floating_point()
            else torch.randn(value.shape, generator=generator).to(storage_dtype)
        )
        for key, value in module.state_dict().items()
    }
    if track_running_stats:
        state["running_var"] = state["running_var"].abs().add(0.5)
    module.load_state_dict(state, assign=True)
    stock.load_state_dict(
        {
            key: value if not value.is_floating_point() else value.float()
            for key, value in state.items()
        },
        assign=True,
    )
    input_value = torch.randn(2, 4, 7, 9, generator=generator)

    assert torch.equal(module(input_value), stock(input_value))
    module.train()
    with pytest.raises(RuntimeError, match="eval mode only"):
        module(input_value)


def test_plain_fp8_matmul_synthesizes_scale_one_call_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quant_linear_mod, "_kitchen_probed", True)
    monkeypatch.setattr(quant_linear_mod, "_kitchen_scaled_mm_v2", None)
    seen: dict[str, object] = {}

    def eager(input: torch.Tensor, weight: torch.Tensor, **kwargs: object) -> torch.Tensor:
        seen.update(input=input, weight=weight, **kwargs)
        return torch.zeros((input.shape[0], weight.shape[1]), dtype=torch.float32)

    monkeypatch.setattr(torch, "_scaled_mm", eager)
    linear = CastOperations(torch.float32).linear(4, 3, bias=False)
    linear.load_state_dict(
        {"weight": torch.randn(3, 4).to(torch.float8_e4m3fn)},
        assign=True,
    )
    bind = type_cast(Any, linear).bind_fp8_matmul
    bind(True)
    with torch.no_grad():
        result = linear(torch.randn(2, 4))
    assert result.shape == (2, 3)
    routed_input = seen["input"]
    routed_weight = seen["weight"]
    input_scale = seen["scale_a"]
    weight_scale = seen["scale_b"]
    assert isinstance(routed_input, torch.Tensor)
    assert isinstance(routed_weight, torch.Tensor)
    assert isinstance(input_scale, torch.Tensor)
    assert isinstance(weight_scale, torch.Tensor)
    assert routed_input.shape == (2, 4)
    assert routed_weight.shape == (4, 3)
    assert input_scale.shape == weight_scale.shape == ()
    assert input_scale.dtype == weight_scale.dtype == torch.float32
    assert input_scale.item() == weight_scale.item() == 1.0
    assert input_scale is not weight_scale


def test_plain_fp8_matmul_preserves_e5m2_and_autograd_refusals() -> None:
    linear = CastOperations(torch.float32).linear(4, 3, bias=False)
    linear.load_state_dict(
        {"weight": torch.randn(3, 4).to(torch.float8_e5m2)},
        assign=True,
    )
    bind = type_cast(Any, linear).bind_fp8_matmul
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        bind(True)

    linear.load_state_dict(
        {"weight": torch.randn(3, 4).to(torch.float8_e4m3fn)},
        assign=True,
    )
    bind(True)
    with pytest.raises(RuntimeError, match="inference-only"):
        linear(torch.randn(2, 4, requires_grad=True))
    # CastOperations keeps its ordinary Parameters trainable; even a
    # constant input therefore has a weight/bias autograd consumer.
    with pytest.raises(RuntimeError, match="inference-only"):
        linear(torch.randn(2, 4))


def test_cast_operations_compile_fullgraph() -> None:
    """The compile discipline the module docstring pins: the compute
    dtype is bind-time state, so a cast-at-use module must compile
    fullgraph with no per-forward Python policy."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable on this build")
    ops = CastOperations(torch.float32)
    linear = ops.linear(6, 4)
    linear.load_state_dict(
        {
            k: torch.randn(v.shape, generator=torch.Generator().manual_seed(4)).to(torch.float16)
            for k, v in linear.state_dict().items()
        },
        assign=True,
    )
    compiled = torch.compile(linear, backend="eager", fullgraph=True)
    x = torch.randn(2, 6)
    assert torch.equal(compiled(x), linear(x))


# ------------------------------------------------------- torch.compile


def test_compiled_consumer_parity() -> None:
    """The declared compile boundary: cast_weight (patching, storage
    mutation) runs eager; the compute that CONSUMES its output
    compiles cleanly (fullgraph, no breaks) and matches eager
    numerics. The CUDA/inductor/triton runs of the same boundary are
    in test_gpu.py (main thread + worker thread)."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable on this build")

    def consumer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, w)

    compiled = torch.compile(consumer, backend="eager", fullgraph=True)

    stored = make_fp8()
    entries = (diff_entry((8, 6), 0.125, strength=0.5),)
    weight = cast_weight(
        stored,
        dtype=torch.float32,
        functions=[DeferredPatch("k", entries)],
    )
    x = torch.randn(3, 6)
    assert torch.equal(compiled(x, weight), consumer(x, weight))
