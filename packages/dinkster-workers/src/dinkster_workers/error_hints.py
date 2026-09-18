"""Conservative interpretations of common node execution failures."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from dinkster_protocol import ErrorHint

Interpreter = Callable[[BaseException], tuple[ErrorHint, ...]]

_SIZE_TENSOR = re.compile(
    r"size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\) "
    r"at non-singleton dimension (\d+)",
    re.IGNORECASE,
)
_MATMUL = re.compile(
    r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)",
    re.IGNORECASE,
)
_CHANNELS = re.compile(
    r"expected input .* to have (\d+) channels, but got (\d+) channels",
    re.IGNORECASE,
)
_SHAPE = re.compile(r"(?:torch\.)?Size\(\[([\d, ]+)\]\)|\[([\d, ]+)\]")
_SCALAR_TYPE = re.compile(
    r"expected scalar type ([A-Za-z0-9_.]+) but found ([A-Za-z0-9_.]+)",
    re.IGNORECASE,
)
_INPUT_WEIGHT_TYPE = re.compile(
    r"Input type \(([^)]+)\) and weight type \(([^)]+)\) should be the same",
    re.IGNORECASE,
)


def _has_type(exc: BaseException, name: str) -> bool:
    return name in [kind.__name__ for kind in type(exc).__mro__]


def _shapes(message: str) -> tuple[tuple[int, ...], ...]:
    found: list[tuple[int, ...]] = []
    for match in _SHAPE.finditer(message):
        raw = match.group(1) or match.group(2)
        found.append(tuple(int(part.strip()) for part in raw.split(",") if part.strip()))
    return tuple(found)


def _looks_like_layout_permutation(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    if left == right or len(left) != len(right) or sorted(left) != sorted(right):
        return False
    return any(
        value <= 4 and left.index(value) != right.index(value) for value in set(left) & set(right)
    )


def _tensor_shape_mismatch(exc: BaseException) -> tuple[ErrorHint, ...]:
    if not _has_type(exc, "RuntimeError"):
        return ()
    message = str(exc)
    size_match = _SIZE_TENSOR.search(message)
    matmul_match = _MATMUL.search(message)
    channel_match = _CHANNELS.search(message)
    shapes = _shapes(message)
    if not (
        size_match
        or matmul_match
        or channel_match
        or ("size mismatch for " in message.lower() and shapes)
    ):
        return ()

    suggestion = None
    if len(shapes) >= 2 and _looks_like_layout_permutation(shapes[0], shapes[1]):
        suggestion = (
            "Check for a channels-first versus channels-last layout mismatch and transpose "
            "the input to the layout expected by the model."
        )
    if len(shapes) >= 2:
        detail = f"Tensor shapes {list(shapes[0])} and {list(shapes[1])} do not match."
    elif size_match:
        detail = (
            f"Tensor sizes {size_match.group(1)} and {size_match.group(2)} do not match "
            f"at dimension {size_match.group(3)}."
        )
    elif matmul_match:
        detail = (
            f"Matrix shapes {matmul_match.group(1)}x{matmul_match.group(2)} and "
            f"{matmul_match.group(3)}x{matmul_match.group(4)} cannot be multiplied."
        )
    elif channel_match:
        detail = (
            f"Input has {channel_match.group(2)} channels, but "
            f"{channel_match.group(1)} channels were expected."
        )
    else:
        detail = "Tensor shapes or dimensions do not match."
    return (ErrorHint("tensor-shape-mismatch", detail, suggestion),)


def _dtype_mismatch(exc: BaseException) -> tuple[ErrorHint, ...]:
    if not _has_type(exc, "RuntimeError"):
        return ()
    message = str(exc)
    scalar_match = _SCALAR_TYPE.search(message)
    if scalar_match is not None:
        # "expected scalar type X but found Y": X is required, Y is the input.
        required, actual = scalar_match.group(1), scalar_match.group(2)
    else:
        pair_match = _INPUT_WEIGHT_TYPE.search(message)
        if pair_match is None:
            return ()
        # "Input type (X) and weight type (Y)": X is the input, Y is required.
        actual, required = pair_match.group(1), pair_match.group(2)
    return (
        ErrorHint(
            "dtype-mismatch",
            f"Input dtype {actual} does not match required dtype {required}.",
            "Cast the inputs to the expected dtype or check model precision "
            "and weight-dtype options.",
        ),
    )


def _device_mismatch(exc: BaseException) -> tuple[ErrorHint, ...]:
    if not _has_type(exc, "RuntimeError"):
        return ()
    message = str(exc).lower()
    if "expected all tensors to be on the same device" not in message and (
        "found at least two devices" not in message
    ):
        return ()
    return (
        ErrorHint(
            "device-mismatch",
            "Tensors required by this operation are on different devices.",
            "Check that the model and inputs are on the same device.",
        ),
    )


def _cuda_oom(exc: BaseException) -> tuple[ErrorHint, ...]:
    if not _has_type(exc, "OutOfMemoryError") and "cuda out of memory" not in str(exc).lower():
        return ()
    return (
        ErrorHint(
            "cuda-oom",
            "CUDA ran out of memory while executing this node.",
            "Lower the resolution or batch size, or adjust memory and offload settings.",
        ),
    )


INTERPRETERS: tuple[Interpreter, ...] = (
    _tensor_shape_mismatch,
    _dtype_mismatch,
    _device_mismatch,
    _cuda_oom,
)


def _hints_for(exc: BaseException, interpreters: Sequence[Interpreter]) -> tuple[ErrorHint, ...]:
    hints: list[ErrorHint] = []
    for interpreter in interpreters:
        try:
            hints.extend(interpreter(exc))
        except Exception:  # noqa: BLE001 - hints must never break error reporting
            continue
        if len(hints) >= 3:
            break
    return tuple(hints[:3])


def hints_for(exc: BaseException) -> tuple[ErrorHint, ...]:
    """Return up to three confident interpretations without ever raising."""
    return _hints_for(exc, INTERPRETERS)
