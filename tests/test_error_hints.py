from __future__ import annotations

from dinkster_protocol import ErrorHint, InvocationResult, NodeError
from dinkster_values import TypeRegistry
from dinkster_workers.boundary import ValueCodec, encode_result, error_from_wire
from dinkster_workers.error_hints import _hints_for, hints_for


def test_tensor_shape_mismatch_interpreter_extracts_dimensions() -> None:
    hints = hints_for(
        RuntimeError(
            "The size of tensor a (64) must match the size of tensor b (32) "
            "at non-singleton dimension 2"
        )
    )

    assert hints == (
        ErrorHint(
            code="tensor-shape-mismatch",
            message="Tensor sizes 64 and 32 do not match at dimension 2.",
        ),
    )


def test_tensor_shape_mismatch_interpreter_suggests_layout_permutation() -> None:
    hints = hints_for(
        RuntimeError(
            "size mismatch for input: got torch.Size([1, 64, 64, 4]) "
            "and expected torch.Size([1, 4, 64, 64])"
        )
    )

    assert hints[0].code == "tensor-shape-mismatch"
    assert hints[0].message == "Tensor shapes [1, 64, 64, 4] and [1, 4, 64, 64] do not match."
    assert hints[0].suggestion is not None
    assert "channels-first versus channels-last" in hints[0].suggestion


def test_dtype_mismatch_interpreter() -> None:
    hints = hints_for(RuntimeError("expected scalar type Half but found Float"))

    assert hints[0].code == "dtype-mismatch"
    assert hints[0].message == "Input dtype Float does not match required dtype Half."
    assert "weight-dtype options" in (hints[0].suggestion or "")


def test_device_mismatch_interpreter() -> None:
    hints = hints_for(
        RuntimeError(
            "Expected all tensors to be on the same device, but found at least two devices"
        )
    )

    assert hints[0].code == "device-mismatch"
    assert hints[0].suggestion == "Check that the model and inputs are on the same device."


def test_cuda_oom_interpreter_matches_structural_type_name() -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    hints = hints_for(OutOfMemoryError("allocation failed"))

    assert hints[0].code == "cuda-oom"
    assert "Lower the resolution or batch size" in (hints[0].suggestion or "")


def test_unrecognized_runtime_error_has_no_hints() -> None:
    assert hints_for(RuntimeError("an unrelated node failure")) == ()
    assert hints_for(RuntimeError("size mismatch for payload length")) == ()


def test_interpreter_failure_is_dropped_without_losing_other_hints() -> None:
    expected = ErrorHint("test-hint", "still returned")

    def raises(_exc: BaseException) -> tuple[ErrorHint, ...]:
        raise RuntimeError("broken interpreter")

    def succeeds(_exc: BaseException) -> tuple[ErrorHint, ...]:
        return (expected,)

    assert _hints_for(RuntimeError("failure"), (raises, succeeds)) == (expected,)


def test_hints_are_capped_at_three() -> None:
    def produces_four(_exc: BaseException) -> tuple[ErrorHint, ...]:
        return tuple(ErrorHint(f"hint-{index}", "message") for index in range(4))

    assert len(_hints_for(RuntimeError("failure"), (produces_four,))) == 3


def test_error_hints_survive_worker_boundary_encoding() -> None:
    error = NodeError(
        "node",
        "test.node",
        "failed",
        hints=(
            ErrorHint("without-suggestion", "first"),
            ErrorHint("with-suggestion", "second", "try this"),
        ),
    )
    header, blobs, segments = encode_result(
        ValueCodec(TypeRegistry()), InvocationResult(error=error), "invocation", 1.0
    )

    assert blobs == []
    assert segments == []
    wire_error = header["error"]
    assert isinstance(wire_error, dict)
    assert "suggestion" not in wire_error["hints"][0]
    assert error_from_wire(wire_error) == error
