"""Quantized-checkpoint classification: split quant artifacts from
architecture, torch-free.

A quantized safetensors checkpoint carries the same architecture keys
as a plain one plus quantization artifacts, in one of three spellings
the reference accepts (comfy/utils.py convert_old_quants,
detect_layer_quantization @ b78cec87):

- NEW metadata: the header's ``_quantization_metadata`` entry is JSON
  ``{"layers": {layer: {"format": ..., ...}}}``; each listed layer
  ships ``{layer}.weight_scale`` and optionally ``{layer}.input_scale``
  beside its fp8 ``{layer}.weight``.
- PER-LAYER config keys: ``{layer}.comfy_quant`` uint8 tensors holding
  the same per-layer JSON (what _quantized_weight_state_dict writes
  when a quantized model is saved). The layer config lives in payload
  bytes, so its format is decoded at LOAD time; the header still says
  which layers are quantized and where their scales are.
- LEGACY scaled-fp8: a ``scaled_fp8`` marker tensor plus
  ``{layer}.scale_weight`` / ``{layer}.scale_input`` keys. The marker's
  dtype names the fp8 dtype (float32 means float8_e4m3fn) and a
  2-element marker means full-precision matmul; ``scale_input`` values
  equal to 1.0 are dropped at load (convert_old_quants @ b78cec87 -
  the VALUE is payload, so the drop happens in the executing layer).

Plain-fp8 checkpoints (fp8 storage dtype, no scales - e.g. the classic
flux1-dev-fp8 combined file) are NOT quantization in this sense: fp8
is just another storage dtype, handled by cast-at-use. This module
only classifies scale-carrying layouts.

:func:`split_quantization` runs BEFORE config detection: detectors
(detect_flux_config et al.) are strict about unexpected keys, so the
scale/marker/config keys must be segregated first. The architecture
half keeps the stored weight geometries unchanged (per-tensor fp8 is
unpacked; shapes survive quantization). Packed formats replace the
physical weight geometry with the recoverable logical geometry while
retaining every physical payload key in :class:`LayerQuant`.

Scope pin: this module types the exact current Comfy/Kitchen vocabulary.
Per-tensor ``float8_e4m3fn``/``float8_e5m2``, NVFP4, and INT8
tensorwise (including ConvRot) are executable. MXFP8, ConvRot W4A4,
and asymmetric W4A8 remain identity-significant contracts which the
assembly planner refuses before runtime/provider selection.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from .weights import TensorGeometry

#: Exact current Comfy/Kitchen quantization vocabulary. Membership means
#: the checkpoint contract can be typed; it does not imply execution.
KNOWN_QUANT_FORMATS = frozenset(
    {
        "float8_e4m3fn",
        "float8_e5m2",
        "nvfp4",
        "mxfp8",
        "int8_tensorwise",
        "convrot_w4a4",
        "asym_w4a8_int8",
    }
)

#: Formats whose runtime implementation already exists. Keep this name and
#: value stable: callers use it as an execution-support allowlist.
SUPPORTED_QUANT_FORMATS = frozenset({"float8_e4m3fn", "float8_e5m2", "nvfp4", "int8_tensorwise"})

#: Typed checkpoint contracts which are deliberately not runtime eligible.
UNSUPPORTED_QUANT_FORMATS = KNOWN_QUANT_FORMATS - SUPPORTED_QUANT_FORMATS

#: Marker dtypes -> fp8 dtype names for the legacy ``scaled_fp8``
#: tensor (convert_old_quants @ b78cec87: float32 means e4m3fn,
#: anything else names itself).
_LEGACY_MARKER_DTYPES = frozenset({"float8_e4m3fn", "float8_e5m2"})


class QuantizationError(ValueError):
    """A quantized layout this port cannot classify: an unsupported
    format, a malformed metadata entry, or scale keys that do not line
    up with their weights. The message names the offender."""


def quantization_error_cause(error: BaseException) -> QuantizationError | None:
    """Recover quantization failures wrapped with component planning context."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, QuantizationError):
            return cause
        cause = cause.__cause__
    return None


def _empty_payloads() -> Mapping[str, str]:
    return {}


def _empty_parameters() -> Mapping[str, int | str | bool]:
    return {}


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object member {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class LayerQuant:
    """One quantized layer's artifact map, in SOURCE keys (whatever
    prefix stripping produced them), normalized across the three
    spellings so the executing layer never re-derives renames.

    ``format`` is None only for the per-layer-config spelling, where
    the JSON lives in payload bytes (``config`` names the
    ``.comfy_quant`` key to decode at load). ``full_precision_matmul``
    mirrors the reference's full_precision_matrix_mult flag: dequantize
    and matmul at compute dtype, never quantize the input."""

    layer: str
    format: str | None
    weight: str
    weight_scale: str
    input_scale: str | None = None
    config: str | None = None
    full_precision_matmul: bool = False
    weight_scale_2: str | None = None
    pre_quant_scale: str | None = None
    payloads: Mapping[str, str] = field(default_factory=_empty_payloads)
    parameters: Mapping[str, int | str | bool] = field(default_factory=_empty_parameters)
    logical_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.format is None and self.config is None:
            raise ValueError(f"layer {self.layer!r}: format and config cannot both be unknown")
        if self.format is not None and self.format not in KNOWN_QUANT_FORMATS:
            raise ValueError(f"layer {self.layer!r}: unknown format {self.format!r}")
        payloads = dict(self.payloads)
        parameters = dict(self.parameters)
        if any(not key or not value for key, value in payloads.items()):
            raise ValueError(f"layer {self.layer!r}: payload names and keys must not be empty")
        if any(not key for key in parameters):
            raise ValueError(f"layer {self.layer!r}: parameter names must not be empty")
        object.__setattr__(self, "payloads", MappingProxyType(payloads))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    @property
    def executable(self) -> bool:
        """Whether Dinkster has a runtime implementation for this format."""
        return self.format is None or self.format in SUPPORTED_QUANT_FORMATS


@dataclass(frozen=True)
class QuantSplit:
    """A header split into detector-ready architecture geometries and
    per-layer quantization descriptors. ``layers`` empty means nothing
    scale-quantized (plain fp8 storage still shows in the architecture
    dtypes)."""

    architecture: Mapping[str, TensorGeometry]
    layers: Mapping[str, LayerQuant]


def _require_scalar_scale(geometries: Mapping[str, TensorGeometry], key: str) -> None:
    geometry = geometries[key]
    if geometry.numel != 1 or geometry.dtype.name != "float32":
        raise QuantizationError(
            f"{key}: expected a single float32 scale, found"
            f" {geometry.dtype.name} with shape {geometry.shape}"
        )


def _require_weight_dtype(geometries: Mapping[str, TensorGeometry], weight: str, fmt: str) -> None:
    """The stored qdata must BE the format's fp8 dtype. The reference
    silently value-converts (``weight.to(storage_t)`` in
    _load_quantized_module @ b78cec87); a mismatch means a mislabeled
    checkpoint, so Dinkster refuses instead of converting."""
    stored = geometries[weight].dtype.name
    if stored != fmt:
        raise QuantizationError(
            f"{weight}: stored dtype {stored} does not match the declared quantization format {fmt}"
        )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _nvfp4_layer(
    geometries: Mapping[str, TensorGeometry],
    layer: str,
    conf: Mapping[str, object],
    *,
    config_key: str | None = None,
) -> tuple[LayerQuant, TensorGeometry, set[str]]:
    weight = _layer_weight(geometries, layer, "_quantization_metadata")
    stored = geometries[weight]
    if len(stored.shape) != 2 or stored.dtype.name != "uint8":
        raise QuantizationError(
            f"{weight}: NVFP4 weight must be rank-2 uint8 packed storage,"
            f" found {stored.dtype.name} with shape {stored.shape}"
        )
    logical = (stored.shape[0], stored.shape[1] * 2)
    if any(dimension % 16 for dimension in logical):
        raise QuantizationError(f"{weight}: NVFP4 logical shape {logical} must be 16-aligned")

    weight_scale = f"{layer}.weight_scale"
    weight_scale_2 = f"{layer}.weight_scale_2"
    input_scale: str | None = f"{layer}.input_scale"
    for key in (weight_scale, weight_scale_2):
        if key not in geometries:
            raise QuantizationError(f"layer {layer!r}: missing {key!r}")
    if input_scale not in geometries:
        input_scale = None
    for key in (weight_scale_2,) if input_scale is None else (weight_scale_2, input_scale):
        scalar = geometries[key]
        if scalar.shape != () or scalar.dtype.name != "float32":
            raise QuantizationError(
                f"{key}: NVFP4 scale must be scalar float32, found"
                f" {scalar.dtype.name} with shape {scalar.shape}"
            )

    block = geometries[weight_scale]
    expected_block = (
        _round_up(logical[0], 128),
        _round_up(logical[1] // 16, 4),
    )
    if block.dtype.name != "float8_e4m3fn" or block.shape != expected_block:
        raise QuantizationError(
            f"{weight_scale}: NVFP4 block scale must be float8_e4m3fn"
            f" with shape {expected_block}, found {block.dtype.name}"
            f" with shape {block.shape}"
        )

    pre_quant_scale: str | None = f"{layer}.pre_quant_scale"
    if pre_quant_scale in geometries:
        pre = geometries[pre_quant_scale]
        if pre.dtype.kind != "float" or pre.shape != (logical[1],):
            raise QuantizationError(
                f"{pre_quant_scale}: NVFP4 pre-quant scale must be a"
                f" floating rank-1 tensor of length {logical[1]}, found"
                f" {pre.dtype.name} with shape {pre.shape}"
            )
    else:
        pre_quant_scale = None

    allowed_conf = {"format", "full_precision_matrix_mult"}
    unexpected_conf = sorted(set(conf) - allowed_conf)
    if unexpected_conf:
        raise QuantizationError(
            f"layer {layer!r}: unexpected NVFP4 metadata parameters: {', '.join(unexpected_conf)}"
        )
    full_precision = conf.get("full_precision_matrix_mult", False)
    if not isinstance(full_precision, bool):
        raise QuantizationError(
            f"layer {layer!r}: NVFP4 full_precision_matrix_mult must be a bool,"
            f" got {type(full_precision).__name__}"
        )
    consumed = {weight_scale, weight_scale_2}
    if input_scale is not None:
        consumed.add(input_scale)
    if pre_quant_scale is not None:
        consumed.add(pre_quant_scale)
    return (
        LayerQuant(
            layer=layer,
            format="nvfp4",
            weight=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            weight_scale_2=weight_scale_2,
            pre_quant_scale=pre_quant_scale,
            config=config_key,
            full_precision_matmul=full_precision,
        ),
        TensorGeometry(logical, stored.dtype),
        consumed,
    )


def _strict_parameters(
    layer: str,
    fmt: str,
    conf: Mapping[str, object],
    *,
    defaults: Mapping[str, int | str | bool],
) -> tuple[dict[str, int | str | bool], bool]:
    """Normalize exact current top-level/``params`` format options.

    Comfy accepts both locations for the three formats which have options.
    Dinkster additionally rejects contradictory duplicates and unknown members so
    malformed identity-bearing configuration cannot become runtime policy.
    """
    allowed = set(defaults)
    allowed_top = {"format", "full_precision_matrix_mult", "params", *allowed}
    unexpected = sorted(set(conf) - allowed_top)
    if unexpected:
        raise QuantizationError(
            f"layer {layer!r}: unexpected {fmt} metadata parameters: {', '.join(unexpected)}"
        )
    raw_params = conf.get("params", {})
    if not isinstance(raw_params, dict):
        raise QuantizationError(f"layer {layer!r}: {fmt} params must be an object")
    nested = cast("dict[str, object]", raw_params)
    unexpected_nested = sorted(set(nested) - allowed)
    if unexpected_nested:
        raise QuantizationError(
            f"layer {layer!r}: unexpected {fmt} params: {', '.join(unexpected_nested)}"
        )
    result = dict(defaults)
    for name in defaults:
        top = conf.get(name)
        inner = nested.get(name)
        if name in conf and name in nested and top != inner:
            raise QuantizationError(f"layer {layer!r}: contradictory {fmt} parameter {name!r}")
        if name in conf:
            result[name] = cast("int | str | bool", top)
        elif name in nested:
            result[name] = cast("int | str | bool", inner)
    full_precision = conf.get("full_precision_matrix_mult", False)
    if not isinstance(full_precision, bool):
        raise QuantizationError(f"layer {layer!r}: {fmt} full_precision_matrix_mult must be a bool")
    return result, full_precision


def _require_payload(
    geometries: Mapping[str, TensorGeometry], layer: str, suffix: str
) -> tuple[str, TensorGeometry]:
    key = f"{layer}.{suffix}"
    if key not in geometries:
        raise QuantizationError(f"layer {layer!r}: missing {key!r}")
    return key, geometries[key]


def _require_geometry(
    key: str,
    geometry: TensorGeometry,
    *,
    dtype: str | tuple[str, ...],
    shape: tuple[int, ...],
    contract: str,
) -> None:
    dtypes = (dtype,) if isinstance(dtype, str) else dtype
    if geometry.dtype.name not in dtypes or geometry.shape != shape:
        raise QuantizationError(
            f"{key}: {contract} must be {' or '.join(dtypes)} with shape {shape},"
            f" found {geometry.dtype.name} with shape {geometry.shape}"
        )


def _require_int8_scale_geometry(
    key: str,
    geometry: TensorGeometry,
    *,
    rows: int,
    convrot: bool,
) -> None:
    allowed_shapes = ((rows, 1),) if convrot else ((), (rows, 1))
    if geometry.dtype.name != "float32" or geometry.shape not in allowed_shapes:
        shapes = " or ".join(str(shape) for shape in allowed_shapes)
        raise QuantizationError(
            f"{key}: int8_tensorwise weight scale must be float32 with shape {shapes},"
            f" found {geometry.dtype.name} with shape {geometry.shape}"
        )


def _power_of_four(value: int) -> bool:
    if value < 4:
        return False
    while value % 4 == 0:
        value //= 4
    return value == 1


def _typed_unsupported_layer(
    geometries: Mapping[str, TensorGeometry],
    layer: str,
    conf: Mapping[str, object],
    *,
    config_key: str | None = None,
) -> tuple[LayerQuant, TensorGeometry, set[str]]:
    """Validate one known but non-executable Kitchen payload exactly."""
    fmt = cast(str, conf["format"])
    weight = _layer_weight(geometries, layer, config_key or "_quantization_metadata")
    stored = geometries[weight]
    payloads: dict[str, str] = {}

    if fmt == "mxfp8":
        parameters, full_precision = _strict_parameters(layer, fmt, conf, defaults={})
        if len(stored.shape) != 2 or stored.dtype.name != "float8_e4m3fn":
            raise QuantizationError(
                f"{weight}: MXFP8 weight must be rank-2 float8_e4m3fn padded storage"
            )
        rows, columns = stored.shape
        if rows % 32 or columns % 32:
            raise QuantizationError(
                f"{weight}: MXFP8 storage shape {stored.shape} must be 32-padded"
            )
        scale_key, scale = _require_payload(geometries, layer, "weight_scale")
        expected = (_round_up(rows, 128), _round_up(columns // 32, 4))
        _require_geometry(
            scale_key,
            scale,
            dtype=("float8_e8m0", "uint8"),
            shape=expected,
            contract="MXFP8 block scale",
        )
        payloads["weight_scale"] = scale_key
        input_scale_key = f"{layer}.input_scale"
        if input_scale_key in geometries:
            _require_geometry(
                input_scale_key,
                geometries[input_scale_key],
                dtype="float32",
                shape=(),
                contract="MXFP8 input scale",
            )
            payloads["input_scale"] = input_scale_key
        logical = stored.shape

    elif fmt == "int8_tensorwise":
        parameters, full_precision = _strict_parameters(
            layer,
            fmt,
            conf,
            defaults={"convrot": False, "convrot_groupsize": 256},
        )
        convrot = parameters["convrot"]
        group = parameters["convrot_groupsize"]
        if not isinstance(convrot, bool):
            raise QuantizationError(f"layer {layer!r}: int8_tensorwise convrot must be a bool")
        if isinstance(group, bool) or not isinstance(group, int) or not _power_of_four(group):
            raise QuantizationError(
                f"layer {layer!r}: int8_tensorwise convrot_groupsize must be a power of 4 >= 4"
            )
        if len(stored.shape) != 2 or stored.dtype.name != "int8":
            raise QuantizationError(f"{weight}: int8_tensorwise weight must be rank-2 int8")
        rows, columns = stored.shape
        if convrot and columns % group:
            raise QuantizationError(
                f"{weight}: ConvRot width {columns} must be divisible by convrot_groupsize {group}"
            )
        scale_key, scale = _require_payload(geometries, layer, "weight_scale")
        _require_int8_scale_geometry(
            scale_key,
            scale,
            rows=rows,
            convrot=convrot,
        )
        payloads["weight_scale"] = scale_key
        logical = stored.shape

    elif fmt == "convrot_w4a4":
        parameters, full_precision = _strict_parameters(
            layer,
            fmt,
            conf,
            defaults={"convrot_groupsize": 256, "linear_dtype": "int4"},
        )
        group = parameters["convrot_groupsize"]
        linear_dtype = parameters["linear_dtype"]
        if isinstance(group, bool) or not isinstance(group, int) or not _power_of_four(group):
            raise QuantizationError(
                f"layer {layer!r}: convrot_w4a4 convrot_groupsize must be a power of 4 >= 4"
            )
        if linear_dtype not in {"int4", "int8"}:
            raise QuantizationError(
                f"layer {layer!r}: convrot_w4a4 linear_dtype must be 'int4' or 'int8'"
            )
        if len(stored.shape) != 2 or stored.dtype.name != "int8":
            raise QuantizationError(
                f"{weight}: convrot_w4a4 weight must be rank-2 int8 packed storage"
            )
        rows, packed_columns = stored.shape
        logical = (rows, packed_columns * 2)
        if logical[1] % 64 or logical[1] % group:
            raise QuantizationError(
                f"{weight}: convrot_w4a4 logical width {logical[1]} must be divisible"
                f" by 64 and convrot_groupsize {group}"
            )
        scale_key, scale = _require_payload(geometries, layer, "weight_scale")
        _require_geometry(
            scale_key,
            scale,
            dtype="float32",
            shape=(rows,),
            contract="convrot_w4a4 row scale",
        )
        payloads["weight_scale"] = scale_key

    elif fmt == "asym_w4a8_int8":
        parameters, full_precision = _strict_parameters(
            layer,
            fmt,
            conf,
            defaults={"group_size": 16, "convrot_groupsize": 256},
        )
        group = parameters["group_size"]
        convrot_group = parameters["convrot_groupsize"]
        if (
            isinstance(group, bool)
            or not isinstance(group, int)
            or group < 4
            or (16 % group != 0 and group % 16 != 0)
        ):
            raise QuantizationError(
                f"layer {layer!r}: asym_w4a8_int8 group_size must be >= 4"
                " and divide 16 or be a multiple of 16"
            )
        if (
            isinstance(convrot_group, bool)
            or not isinstance(convrot_group, int)
            or not _power_of_four(convrot_group)
        ):
            raise QuantizationError(
                f"layer {layer!r}: asym_w4a8_int8 convrot_groupsize must be a power of 4 >= 4"
            )
        if len(stored.shape) != 2 or stored.dtype.name != "int8":
            raise QuantizationError(
                f"{weight}: asym_w4a8_int8 weight must be rank-2 int8 packed storage"
            )
        rows, packed_columns = stored.shape
        logical = (rows, packed_columns * 2)
        if logical[1] % 16 or logical[1] % group or logical[1] % convrot_group:
            raise QuantizationError(
                f"{weight}: asym_w4a8_int8 logical width {logical[1]} is inconsistent"
                f" with group_size {group} and convrot_groupsize {convrot_group}"
            )
        rel_key, rel = _require_payload(geometries, layer, "weight_s_rel")
        channel_key, channel = _require_payload(geometries, layer, "weight_s_channel")
        _require_geometry(
            rel_key,
            rel,
            dtype=("float32", "float8_e4m3fn"),
            shape=(rows, logical[1] // group),
            contract="asym_w4a8_int8 relative scale",
        )
        _require_geometry(
            channel_key,
            channel,
            dtype="float32",
            shape=(rows,),
            contract="asym_w4a8_int8 channel scale",
        )
        payloads.update(weight_s_rel=rel_key, weight_s_channel=channel_key)
        correction_key = f"{layer}.weight_correction"
        if correction_key in geometries:
            correction = geometries[correction_key]
            correction_shape = (logical[1] // group, rows)
            if correction.dtype.kind != "float" or correction.shape != correction_shape:
                raise QuantizationError(
                    f"{correction_key}: asym_w4a8_int8 weight_correction must be"
                    f" floating with shape {correction_shape}, found"
                    f" {correction.dtype.name} with shape {correction.shape}"
                )
            payloads["weight_correction"] = correction_key
        codebook_key = f"{layer}.weight_codebook"
        if codebook_key in geometries:
            _require_geometry(
                codebook_key,
                geometries[codebook_key],
                dtype="float32",
                shape=(16,),
                contract="asym_w4a8_int8 weight_codebook",
            )
            payloads["weight_codebook"] = codebook_key
    else:
        raise AssertionError(f"unhandled typed quantization format {fmt!r}")

    consumed = set(payloads.values())
    return (
        LayerQuant(
            layer=layer,
            format=fmt,
            weight=weight,
            weight_scale=payloads.get("weight_scale", ""),
            input_scale=payloads.get("input_scale"),
            config=config_key,
            full_precision_matmul=full_precision,
            payloads=payloads,
            parameters=parameters,
            logical_shape=logical,
        ),
        TensorGeometry(logical, stored.dtype),
        consumed,
    )


def _layer_weight(geometries: Mapping[str, TensorGeometry], layer: str, source: str) -> str:
    weight = f"{layer}.weight"
    if weight not in geometries:
        raise QuantizationError(
            f"{source} names layer {layer!r} but {weight!r} is not in the checkpoint"
        )
    return weight


def _parse_metadata_layers(metadata_json: str) -> dict[str, Mapping[str, object]]:
    """Decode and type-check the ``_quantization_metadata`` JSON into
    a layer-name -> config mapping; scoping happens at the caller."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object member {key!r}")
            result[key] = value
        return result

    try:
        decoded: object = json.loads(metadata_json, object_pairs_hook=unique_object)
        entries = decoded["layers"]  # type: ignore[index]
    except (ValueError, TypeError, KeyError) as exc:
        raise QuantizationError(f"malformed _quantization_metadata: {exc}") from exc
    if not isinstance(entries, dict):
        raise QuantizationError(
            "_quantization_metadata layers must be an object, got"
            f" {type(cast(object, entries)).__name__}"
        )
    parsed: dict[str, Mapping[str, object]] = {}
    for raw_layer, raw_conf in cast("dict[object, object]", entries).items():
        if not isinstance(raw_layer, str):
            raise QuantizationError(
                "_quantization_metadata layer names must be strings,"
                f" got {type(raw_layer).__name__}"
            )
        if not isinstance(raw_conf, dict):
            raise QuantizationError(
                f"layer {raw_layer!r}: config must be an object, got {type(raw_conf).__name__}"
            )
        parsed[raw_layer] = cast("dict[str, object]", raw_conf)
    return parsed


def _split_metadata(
    geometries: Mapping[str, TensorGeometry],
    entries: Mapping[str, Mapping[str, object]],
) -> QuantSplit:
    layers: dict[str, LayerQuant] = {}
    # Explicit metadata wins over payload configs. Consume those lower
    # precedence declarations so they cannot become false orphans.
    consumed: set[str] = {key for key in geometries if key.endswith(".comfy_quant")}
    logical_weights: dict[str, TensorGeometry] = {}
    for layer, conf in entries.items():
        fmt = conf.get("format")
        if not isinstance(fmt, str) or fmt not in KNOWN_QUANT_FORMATS:
            raise QuantizationError(f"layer {layer!r}: unknown quantization format {fmt!r}")
        if fmt == "nvfp4":
            entry, logical_weight, layer_consumed = _nvfp4_layer(geometries, layer, conf)
            layers[layer] = entry
            logical_weights[entry.weight] = logical_weight
            consumed.update(layer_consumed)
            continue
        if fmt == "int8_tensorwise" or fmt in UNSUPPORTED_QUANT_FORMATS:
            entry, logical_weight, layer_consumed = _typed_unsupported_layer(
                geometries, layer, conf
            )
            layers[layer] = entry
            logical_weights[entry.weight] = logical_weight
            consumed.update(layer_consumed)
            continue
        weight = _layer_weight(geometries, layer, "_quantization_metadata")
        _require_weight_dtype(geometries, weight, fmt)
        weight_scale = f"{layer}.weight_scale"
        if weight_scale not in geometries:
            raise QuantizationError(f"layer {layer!r}: missing {weight_scale!r}")
        _require_scalar_scale(geometries, weight_scale)
        input_scale: str | None = f"{layer}.input_scale"
        if input_scale in geometries:
            _require_scalar_scale(geometries, input_scale)
        else:
            input_scale = None
        layers[layer] = LayerQuant(
            layer=layer,
            format=fmt,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            full_precision_matmul=bool(conf.get("full_precision_matrix_mult", False)),
        )
        consumed.add(weight_scale)
        if input_scale is not None:
            consumed.add(input_scale)

    architecture, orphans = _strip(geometries, consumed)
    if orphans:
        raise QuantizationError(
            "scale keys without a _quantization_metadata entry: " + ", ".join(sorted(orphans)[:6])
        )
    architecture.update(logical_weights)
    return QuantSplit(architecture=architecture, layers=layers)


def _split_config_keys(
    geometries: Mapping[str, TensorGeometry],
    payload_reader: Callable[[str], bytes],
    prefix: str,
) -> QuantSplit:
    layers: dict[str, LayerQuant] = {}
    consumed: set[str] = set()
    logical_weights: dict[str, TensorGeometry] = {}
    for key in geometries:
        if not key.endswith(".comfy_quant"):
            continue
        layer = key[: -len(".comfy_quant")]
        geometry = geometries[key]
        if geometry.dtype.name != "uint8" or len(geometry.shape) != 1:
            raise QuantizationError(f"{key}: configuration must be rank-1 uint8")
        if geometry.numel > 65_536:
            raise QuantizationError(f"{key}: configuration exceeds the 65536-byte cap")
        try:
            decoded = json.loads(
                payload_reader(prefix + key).decode("utf-8"),
                object_pairs_hook=_unique_pairs,
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise QuantizationError(f"{key}: malformed configuration: {error}") from error
        if not isinstance(decoded, dict):
            raise QuantizationError(f"{key}: configuration must be an object")
        conf = cast("Mapping[str, object]", decoded)
        fmt = conf.get("format")
        if not isinstance(fmt, str):
            raise QuantizationError(f"{key}: configuration format must be a string")
        if fmt == "nvfp4":
            entry, logical, layer_consumed = _nvfp4_layer(geometries, layer, conf, config_key=key)
            layers[layer] = entry
            logical_weights[entry.weight] = logical
            consumed.update(layer_consumed)
            consumed.add(key)
            continue
        if fmt == "int8_tensorwise" or fmt in UNSUPPORTED_QUANT_FORMATS:
            entry, logical, layer_consumed = _typed_unsupported_layer(
                geometries, layer, conf, config_key=key
            )
            layers[layer] = entry
            logical_weights[entry.weight] = logical
            consumed.update(layer_consumed)
            consumed.add(key)
            continue
        if fmt not in {"float8_e4m3fn", "float8_e5m2"}:
            raise QuantizationError(f"layer {layer!r}: unknown quantization format {fmt!r}")
        weight = _layer_weight(geometries, layer, key)
        _require_weight_dtype(geometries, weight, fmt)
        weight_scale = f"{layer}.weight_scale"
        if weight_scale not in geometries:
            raise QuantizationError(f"layer {layer!r}: missing {weight_scale!r}")
        _require_scalar_scale(geometries, weight_scale)
        input_scale: str | None = f"{layer}.input_scale"
        if input_scale in geometries:
            _require_scalar_scale(geometries, input_scale)
        else:
            input_scale = None
        layers[layer] = LayerQuant(
            layer=layer,
            # FP8 payload facts have always ridden the separately hashed
            # asset bytes. Preserve that identity contract while the loader
            # resolves and applies the validated payload at execution time.
            format=None,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            config=key,
        )
        consumed.update({key, weight_scale})
        if input_scale is not None:
            consumed.add(input_scale)

    architecture, orphans = _strip(geometries, consumed)
    if orphans:
        raise QuantizationError(
            "scale keys without a .comfy_quant entry: " + ", ".join(sorted(orphans)[:6])
        )
    architecture.update(logical_weights)
    return QuantSplit(architecture=architecture, layers=layers)


def _split_unresolved_config_keys(
    geometries: Mapping[str, TensorGeometry],
) -> QuantSplit:
    """Preserve the header-only fp8 plan when no payload seam exists.

    Packed NVFP4 needs its explicit payload format before detection;
    ordinary fp8 keeps the existing load-time config resolution path.
    """
    packed = sorted(key for key in geometries if key.endswith(".weight_scale_2"))
    if packed:
        layer = packed[0][: -len(".weight_scale_2")]
        raise QuantizationError(f"{layer}.comfy_quant: configuration payload is unavailable")
    layers: dict[str, LayerQuant] = {}
    consumed: set[str] = set()
    for key, geometry in geometries.items():
        if not key.endswith(".comfy_quant"):
            continue
        if geometry.dtype.name != "uint8" or len(geometry.shape) != 1:
            raise QuantizationError(f"{key}: configuration must be rank-1 uint8")
        layer = key[: -len(".comfy_quant")]
        weight = _layer_weight(geometries, layer, key)
        if geometries[weight].dtype.name not in _LEGACY_MARKER_DTYPES:
            raise QuantizationError(
                f"{key}: configuration payload is unavailable and stored weight"
                f" dtype {geometries[weight].dtype.name!r} is not an unresolved FP8 contract"
            )
        weight_scale = f"{layer}.weight_scale"
        if weight_scale not in geometries:
            raise QuantizationError(f"layer {layer!r}: missing {weight_scale!r}")
        _require_scalar_scale(geometries, weight_scale)
        input_scale: str | None = f"{layer}.input_scale"
        if input_scale in geometries:
            _require_scalar_scale(geometries, input_scale)
        else:
            input_scale = None
        layers[layer] = LayerQuant(
            layer=layer,
            format=None,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            config=key,
        )
        consumed.update({key, weight_scale})
        if input_scale is not None:
            consumed.add(input_scale)
    architecture, orphans = _strip(geometries, consumed)
    if orphans:
        raise QuantizationError(
            "scale keys without a .comfy_quant entry: " + ", ".join(sorted(orphans)[:6])
        )
    return QuantSplit(architecture=architecture, layers=layers)


def _split_legacy(
    geometries: Mapping[str, TensorGeometry],
) -> QuantSplit:
    marker = geometries["scaled_fp8"]
    # Deliberate divergence: the reference derives scaled_fp8_dtype
    # from the marker but never uses it - convert_old_quants hardcodes
    # format "float8_e4m3fn" for every legacy layer, mislabeling e5m2
    # legacy checkpoints (docs/comfyui-issues/
    # convert-old-quants-ignores-marker-dtype.md). Dinkster honors the
    # marker dtype.
    dtype = marker.dtype.name
    if dtype == "float32":
        dtype = "float8_e4m3fn"
    if dtype not in _LEGACY_MARKER_DTYPES:
        raise QuantizationError(
            f"scaled_fp8 marker dtype {marker.dtype.name!r} names an"
            " unported quantization dtype (ROADMAP: Native inference)"
        )
    # A 2-element marker flags full-precision matmul
    # (convert_old_quants @ b78cec87).
    full_precision = marker.numel == 2

    layers: dict[str, LayerQuant] = {}
    consumed: set[str] = {"scaled_fp8"}
    for key in geometries:
        if not key.endswith(".scale_weight"):
            continue
        layer = key[: -len(".scale_weight")]
        weight = _layer_weight(geometries, layer, key)
        _require_weight_dtype(geometries, weight, dtype)
        _require_scalar_scale(geometries, key)
        input_scale: str | None = f"{layer}.scale_input"
        if input_scale in geometries:
            _require_scalar_scale(geometries, input_scale)
            consumed.add(input_scale)
        else:
            input_scale = None
        layers[layer] = LayerQuant(
            layer=layer,
            format=dtype,
            weight=weight,
            weight_scale=key,
            input_scale=input_scale,
            full_precision_matmul=full_precision,
        )
        consumed.add(key)

    architecture, orphans = _strip(geometries, consumed)
    if orphans:
        raise QuantizationError(
            "scale_input keys without a matching scale_weight: " + ", ".join(sorted(orphans)[:6])
        )
    return QuantSplit(architecture=architecture, layers=layers)


def _strip(
    geometries: Mapping[str, TensorGeometry], consumed: set[str]
) -> tuple[dict[str, TensorGeometry], list[str]]:
    """Remove consumed artifact keys; report stragglers that LOOK like
    quantization artifacts but were not claimed (a scale for a layer
    the metadata never mentioned would otherwise reach a detector and
    produce a misleading unexpected-key refusal)."""
    architecture: dict[str, TensorGeometry] = {}
    orphans: list[str] = []
    suffixes = (
        ".weight_scale",
        ".weight_scale_2",
        ".weight_s_rel",
        ".weight_s_channel",
        ".weight_correction",
        ".weight_codebook",
        ".input_scale",
        ".pre_quant_scale",
        ".scale_weight",
        ".scale_input",
        ".comfy_quant",
    )
    for key, geometry in geometries.items():
        if key in consumed:
            continue
        if key == "scaled_fp8" or key.endswith(suffixes):
            orphans.append(key)
            continue
        architecture[key] = geometry
    return architecture, orphans


def split_quantization(
    geometries: Mapping[str, TensorGeometry],
    metadata: Mapping[str, str] | None = None,
    *,
    prefix: str = "",
    payload_reader: Callable[[str], bytes] | None = None,
) -> QuantSplit:
    """Classify a component-scoped header into architecture geometries
    plus per-layer quantization.

    ``prefix`` scopes a multi-component header to one component
    (mirroring convert_old_quants' model_prefix @ b78cec87): only keys
    under it are considered and every returned key - architecture,
    layer names, artifact keys - comes back STRIPPED. Combined
    checkpoints park quantization per component: the legacy marker
    lives at ``{prefix}scaled_fp8`` and ``_quantization_metadata``
    names layers by their full prefixed keys (comfy/sd.py text-encoder
    scaled_fp8 scan @ b78cec87), so classification must happen at
    component scope, never on the whole file at once.

    Spelling precedence matches the reference loader
    (comfy/utils.py convert_old_quants @ b78cec87): explicit
    ``_quantization_metadata`` wins, then per-layer ``.comfy_quant``
    keys, then the legacy ``scaled_fp8`` marker. Metadata that names
    no layer under ``prefix`` does not claim the component - the other
    spellings are still honored, matching the reference's per-prefix
    text-encoder conversion (comfy/sd.py @ b78cec87). A header with no
    spelling passes through untouched; stray artifact-suffix keys with
    no spelling refuse loudly rather than leak into a detector.

    Payload configurations are read only through the bounded reader;
    packed geometry supplies NVFP4's stable aligned logical shape."""
    if prefix:
        geometries = {
            key[len(prefix) :]: geometry
            for key, geometry in geometries.items()
            if key.startswith(prefix)
        }
    if metadata and "_quantization_metadata" in metadata:
        entries = _parse_metadata_layers(metadata["_quantization_metadata"])
        if prefix:
            entries = {
                layer[len(prefix) :]: conf
                for layer, conf in entries.items()
                if layer.startswith(prefix)
            }
        if entries:
            return _split_metadata(geometries, entries)
    if any(key.endswith(".comfy_quant") for key in geometries):
        if payload_reader is None:
            return _split_unresolved_config_keys(geometries)
        return _split_config_keys(geometries, payload_reader, prefix)
    if "scaled_fp8" in geometries:
        return _split_legacy(geometries)
    architecture, orphans = _strip(geometries, set())
    if orphans:
        raise QuantizationError(
            "quantization artifacts with no recognized spelling: " + ", ".join(sorted(orphans)[:6])
        )
    return QuantSplit(architecture=architecture, layers={})


__all__ = [
    "KNOWN_QUANT_FORMATS",
    "SUPPORTED_QUANT_FORMATS",
    "UNSUPPORTED_QUANT_FORMATS",
    "LayerQuant",
    "QuantSplit",
    "QuantizationError",
    "split_quantization",
]
