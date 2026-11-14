from __future__ import annotations

from dinkster_protocol import ErrorHint
from dinkster_workers.error_hints import hints_for


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
