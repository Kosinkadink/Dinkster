"""Shared conditioning carrier and payload operations for native nodes."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from .native_arm_core import Any, Mapping, Sequence, cast, importlib, replace

# Conditioning value ops mirror the pinned ComfyUI nodes (nodes.py at
# b78cec87): payload math runs in the payload's own dtype through NumPy, so
# the default arm stays torch-free. BF16 payloads are refused because NumPy
# cannot reproduce bf16 arithmetic bit-exactly.
_CONDITIONING_FLOAT_WIRE_DTYPES = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
_CONDITIONING_WIRE_BY_ITEMSIZE = {2: "F16", 4: "F32", 8: "F64"}


def _conditioning_carrier(value: object, name: str) -> Any:
    inference = importlib.import_module("dinkster_inference")
    if type(value) is not inference.ConditioningCarrier:
        raise TypeError(f"{name} must come from a Dinkster conditioning node")
    return cast("Any", value)


def _conditioning_op_float(value: object, name: str, low: float, high: float) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    number = float(cast("int | float", value))
    if not low <= number <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
    return number


def _float_payload_array(np_mod: Any, binding: Any, name: str) -> Any:
    dtype = _CONDITIONING_FLOAT_WIRE_DTYPES.get(binding.dtype)
    if dtype is None:
        raise ValueError(
            f"{name} has a {binding.dtype} payload; conditioning math supports F16, F32, and F64"
        )
    return np_mod.frombuffer(binding.data, dtype=np_mod.dtype(dtype)).reshape(binding.shape)


def _scalar_scaled(np_mod: Any, array: Any, scalar: float) -> Any:
    # Torch CPU half kernels keep the python scalar in float32 opmath and
    # round once per element; NEP50 would first demote the scalar to float16
    # and drift one ulp from the pin.
    if array.dtype == np_mod.float16:
        product = array.astype(np_mod.float32) * np_mod.float32(scalar)
        return product.astype(np_mod.float16)
    return array * scalar


def _float_payload_binding(
    np_mod: Any, inference: Any, reference_id: str, array: Any, space: str
) -> Any:
    data = np_mod.ascontiguousarray(array)
    wire = (
        _CONDITIONING_WIRE_BY_ITEMSIZE.get(int(data.dtype.itemsize))
        if data.dtype.kind == "f"
        else None
    )
    if wire is None:
        raise ValueError(f"unsupported conditioning payload dtype: {data.dtype}")
    if data.dtype.byteorder == ">":
        data = data.astype(data.dtype.newbyteorder("<"))
    return inference.PayloadBinding(
        reference_id, tuple(int(dim) for dim in data.shape), wire, space, data.tobytes()
    )


def _reachable_conditioning_ids(inference: Any, conditioning: Any) -> set[str]:
    ids: set[str] = set()

    def scan(value: object) -> None:
        if type(value) is inference.PayloadReference:
            ids.add(cast("Any", value).id)
        elif isinstance(value, tuple):
            for item in cast("tuple[object, ...]", value):
                scan(item)
        elif isinstance(value, Mapping):
            for item in cast("Mapping[object, object]", value).values():
                scan(item)

    for record in conditioning.records:
        for _, payload in record.channels:
            ids.add(payload.reference.id)
        if record.mask is not None:
            ids.add(record.mask.payload.id)
        if record.scale_vector is not None:
            ids.add(record.scale_vector.values.reference.id)
        for _, metadata in record.extension_metadata:
            scan(metadata)
    return ids


def _rebound_conditioning_carrier(
    inference: Any, records: Sequence[Any], bindings: Sequence[Any]
) -> Any:
    """Rebuild a carrier, keeping only the bindings the records still reach."""
    conditioning = inference.ConditioningSet(tuple(records))
    reachable = _reachable_conditioning_ids(inference, conditioning)
    kept: dict[str, Any] = {}
    for binding in bindings:
        if binding.reference_id in reachable and binding.reference_id not in kept:
            kept[binding.reference_id] = binding
    return inference.make_conditioning_carrier(conditioning, tuple(kept.values()))


def _combined_conditioning(inference: Any, carriers: Sequence[Any]) -> Any:
    records = tuple(record for carrier in carriers for record in carrier.conditioning.records)
    bindings = tuple(binding for carrier in carriers for binding in carrier.bindings)
    return _rebound_conditioning_carrier(inference, records, bindings)


def _from_side_text(inference: Any, from_carrier: Any) -> tuple[Any, Any]:
    """First from-side record's channels; extra records are ignored at the pin."""
    if not from_carrier.conditioning.records:
        raise ValueError("conditioning_from must contain at least one record")
    channels = dict(from_carrier.conditioning.records[0].channels)
    text = channels.get(inference.ConditioningChannel.TEXT)
    if text is None:
        raise ValueError("conditioning_from must carry a text payload")
    return text, channels.get(inference.ConditioningChannel.POOLED)


def _averaged_conditioning(
    inference: Any, to_carrier: Any, from_carrier: Any, strength: float
) -> Any:
    np_mod = cast("Any", importlib.import_module("numpy"))
    text_from, pooled_from = _from_side_text(inference, from_carrier)
    bindings = {
        binding.reference_id: binding for binding in (*to_carrier.bindings, *from_carrier.bindings)
    }
    cond_from = _float_payload_array(np_mod, bindings[text_from.reference.id], "conditioning_from")
    if cond_from.ndim != 3:
        raise ValueError("conditioning_from text payload must be [batch, tokens, features]")
    pooled_from_array = (
        _float_payload_array(np_mod, bindings[pooled_from.reference.id], "conditioning_from")
        if pooled_from is not None
        else None
    )
    records: list[Any] = []
    new_bindings: list[Any] = []

    def rebound(tag: str, index: int, array: Any, space: str) -> Any:
        reference_id = f"compat-conditioning-average:{index}:{tag}"
        binding = _float_payload_binding(np_mod, inference, reference_id, array, space)
        new_bindings.append(binding)
        return inference.PayloadDescriptor(
            inference.PayloadReference(reference_id), binding.shape, binding.dtype, binding.space
        )

    for index, record in enumerate(to_carrier.conditioning.records):
        channels = dict(record.channels)
        text_to = channels.get(inference.ConditioningChannel.TEXT)
        if text_to is None:
            raise ValueError("conditioning_to must carry a text payload")
        t1 = _float_payload_array(np_mod, bindings[text_to.reference.id], "conditioning_to")
        if t1.ndim != 3:
            raise ValueError("conditioning_to text payload must be [batch, tokens, features]")
        t0 = cond_from[:, : t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            # The pin pads with float32 zeros, promoting shorter non-f32
            # from-side payloads exactly as torch.cat does.
            pad = np_mod.zeros((1, t1.shape[1] - t0.shape[1], t1.shape[2]), dtype=np_mod.float32)
            t0 = np_mod.concatenate((t0, pad), axis=1)
        blended = _scalar_scaled(np_mod, t1, strength) + _scalar_scaled(np_mod, t0, 1.0 - strength)
        replaced = {
            inference.ConditioningChannel.TEXT: rebound("text", index, blended, text_to.space)
        }
        pooled_to = channels.get(inference.ConditioningChannel.POOLED)
        if pooled_from_array is not None:
            # A to-record without a pooled payload inherits the from-side one
            # before blending, matching the pin's .get default.
            if pooled_to is not None:
                base = _float_payload_array(
                    np_mod, bindings[pooled_to.reference.id], "conditioning_to"
                )
                space = pooled_to.space
            else:
                base = pooled_from_array
                space = cast("Any", pooled_from).space
            pooled = _scalar_scaled(np_mod, base, strength) + _scalar_scaled(
                np_mod, pooled_from_array, 1.0 - strength
            )
            replaced[inference.ConditioningChannel.POOLED] = rebound("pooled", index, pooled, space)
        new_channels = [
            (channel, replaced.get(channel, payload)) for channel, payload in record.channels
        ]
        if pooled_to is None and inference.ConditioningChannel.POOLED in replaced:
            new_channels.append(
                (
                    inference.ConditioningChannel.POOLED,
                    replaced[inference.ConditioningChannel.POOLED],
                )
            )
        records.append(replace(record, channels=tuple(new_channels)))
    return _rebound_conditioning_carrier(
        inference, records, (*to_carrier.bindings, *from_carrier.bindings, *new_bindings)
    )


def _concatenated_conditioning(inference: Any, to_carrier: Any, from_carrier: Any) -> Any:
    np_mod = cast("Any", importlib.import_module("numpy"))
    text_from, _ = _from_side_text(inference, from_carrier)
    bindings = {
        binding.reference_id: binding for binding in (*to_carrier.bindings, *from_carrier.bindings)
    }
    cond_from = _float_payload_array(np_mod, bindings[text_from.reference.id], "conditioning_from")
    records: list[Any] = []
    new_bindings: list[Any] = []
    for index, record in enumerate(to_carrier.conditioning.records):
        channels = dict(record.channels)
        text_to = channels.get(inference.ConditioningChannel.TEXT)
        if text_to is None:
            raise ValueError("conditioning_to must carry a text payload")
        t1 = _float_payload_array(np_mod, bindings[text_to.reference.id], "conditioning_to")
        if (
            t1.ndim < 2
            or t1.ndim != cond_from.ndim
            or t1.shape[:1] != cond_from.shape[:1]
            or t1.shape[2:] != cond_from.shape[2:]
        ):
            raise ValueError(
                f"conditioning payload shapes {tuple(t1.shape)} and "
                f"{tuple(cond_from.shape)} cannot be joined along the token axis"
            )
        joined = np_mod.concatenate((t1, cond_from), axis=1)
        reference_id = f"compat-conditioning-concat:{index}:text"
        binding = _float_payload_binding(np_mod, inference, reference_id, joined, text_to.space)
        new_bindings.append(binding)
        descriptor = inference.PayloadDescriptor(
            inference.PayloadReference(reference_id), binding.shape, binding.dtype, binding.space
        )
        new_channels = tuple(
            (channel, descriptor if channel is inference.ConditioningChannel.TEXT else payload)
            for channel, payload in record.channels
        )
        records.append(replace(record, channels=new_channels))
    return _rebound_conditioning_carrier(
        inference, records, (*to_carrier.bindings, *from_carrier.bindings, *new_bindings)
    )


def _scaled_conditioning(inference: Any, carrier: Any, multiplier: float) -> Any:
    np_mod = cast("Any", importlib.import_module("numpy"))
    bindings = {binding.reference_id: binding for binding in carrier.bindings}
    scaled_channels = (
        inference.ConditioningChannel.TEXT,
        inference.ConditioningChannel.POOLED,
    )
    records: list[Any] = []
    new_bindings: list[Any] = []
    for index, record in enumerate(carrier.conditioning.records):
        new_channels: list[Any] = []
        for channel, payload in record.channels:
            if channel not in scaled_channels:
                new_channels.append((channel, payload))
                continue
            array = _float_payload_array(np_mod, bindings[payload.reference.id], "conditioning")
            reference_id = f"compat-conditioning-scale:{index}:{channel.value}"
            binding = _float_payload_binding(
                np_mod,
                inference,
                reference_id,
                _scalar_scaled(np_mod, array, multiplier),
                payload.space,
            )
            new_bindings.append(binding)
            new_channels.append(
                (
                    channel,
                    inference.PayloadDescriptor(
                        inference.PayloadReference(reference_id),
                        binding.shape,
                        binding.dtype,
                        binding.space,
                    ),
                )
            )
        records.append(replace(record, channels=tuple(new_channels)))
    return _rebound_conditioning_carrier(inference, records, (*carrier.bindings, *new_bindings))


def _zeroed_conditioning(inference: Any, carrier: Any) -> Any:
    bindings = {binding.reference_id: binding for binding in carrier.bindings}
    zeroed_channels = (
        inference.ConditioningChannel.TEXT,
        inference.ConditioningChannel.POOLED,
    )
    records: list[Any] = []
    new_bindings: list[Any] = []

    def zeroed(payload: Any, index: int, tag: str) -> Any:
        binding = bindings[payload.reference.id]
        reference_id = f"compat-conditioning-zero:{index}:{tag}"
        new_bindings.append(
            inference.PayloadBinding(
                reference_id, binding.shape, binding.dtype, binding.space, bytes(len(binding.data))
            )
        )
        return inference.PayloadDescriptor(
            inference.PayloadReference(reference_id), binding.shape, binding.dtype, binding.space
        )

    for index, record in enumerate(carrier.conditioning.records):
        new_channels = tuple(
            (
                channel,
                zeroed(payload, index, channel.value) if channel in zeroed_channels else payload,
            )
            for channel, payload in record.channels
        )
        scale_vector = record.scale_vector
        if scale_vector is not None:
            # Mirrors the pin zeroing conditioning_scale alongside the prompt.
            scale_vector = replace(scale_vector, values=zeroed(scale_vector.values, index, "scale"))
        records.append(replace(record, channels=new_channels, scale_vector=scale_vector))
    return _rebound_conditioning_carrier(inference, records, (*carrier.bindings, *new_bindings))
