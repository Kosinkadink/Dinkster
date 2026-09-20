"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from .native_arm_core import (
    Any,
    KSampler,
    Mapping,
    Node,
    NodeSchema,
    Sequence,
    _CustomSigmasValue,
    _LatentOperationValue,
    _torch,
    cast,
    importlib,
    log,
    math,
)
from .native_arm_latent_utils import (
    _check_bounds,
    _check_choice,
    _composite_masked_tensor,
    _plain_latent,
    _reshape_latent_to,
)
from .native_arm_runtime import (
    _native_model,
)
from .nodes_guidance import (
    _guidance_transform_factory,
    _model_with_guidance_transform,
)
from .nodes_provider import (
    _generation_provider_schema,
)
from .nodes_vae_seedvr2 import (
    _LATENT_RESIZE_METHODS,
)


class GenerationLatentCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.combine")

    @classmethod
    def execute(cls, *, samples1: object, samples2: object, operation: str) -> Mapping[str, object]:
        _check_choice("operation", operation, ("add", "subtract"))
        torch = _torch()
        out, s1 = _plain_latent(samples1, torch, "samples1")
        _, s2 = _plain_latent(samples2, torch, "samples2")
        s2 = _reshape_latent_to(s1.shape, s2)
        out["samples"] = s1 + s2 if operation == "add" else s1 - s2
        return cls.outputs(latent=out)


class GenerationLatentMix(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.mix")

    @classmethod
    def execute(
        cls, *, samples1: object, samples2: object, operation: str, factor: float
    ) -> Mapping[str, object]:
        _check_choice("operation", operation, ("interpolate", "blend"))
        _check_bounds(("factor", factor, 0.0, 1.0))
        torch = _torch()
        out, s1 = _plain_latent(samples1, torch, "samples1")
        _, s2 = _plain_latent(samples2, torch, "samples2")
        if operation == "interpolate":
            s2 = _reshape_latent_to(s1.shape, s2)
            m1 = torch.linalg.vector_norm(s1, dim=(1))
            m2 = torch.linalg.vector_norm(s2, dim=(1))
            n1 = torch.nan_to_num(s1 / m1)
            n2 = torch.nan_to_num(s2 / m2)
            t = n1 * factor + n2 * (1.0 - factor)
            mt = torch.linalg.vector_norm(t, dim=(1))
            st = torch.nan_to_num(t / mt)
            out["samples"] = st * (m1 * factor + m2 * (1.0 - factor))
        else:
            if s1.shape != s2.shape:
                s2 = importlib.import_module("dinkster_inference_torch.resize").common_upscale(
                    s2, s1.shape[3], s1.shape[2], "bicubic", crop="center"
                )
            out["samples"] = s1 * factor + s2 * (1 - factor)
        return cls.outputs(latent=out)


class GenerationLatentMultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.multiply")

    @classmethod
    def execute(cls, *, samples: object, multiplier: float) -> Mapping[str, object]:
        _check_bounds(("multiplier", multiplier, -10.0, 10.0))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        out["samples"] = s1 * multiplier
        return cls.outputs(latent=out)


class GenerationLatentRotate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.rotate")

    @classmethod
    def execute(cls, *, samples: object, angle: str) -> Mapping[str, object]:
        _check_choice("angle", angle, ("none", "90", "180", "270"))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        rotate_by = {"none": 0, "90": 1, "180": 2, "270": 3}[angle]
        out["samples"] = torch.rot90(s1, k=rotate_by, dims=[3, 2])
        return cls.outputs(latent=out)


class GenerationLatentFlip(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.flip")

    @classmethod
    def execute(cls, *, samples: object, axis: str) -> Mapping[str, object]:
        _check_choice("axis", axis, ("vertical", "horizontal"))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        out["samples"] = torch.flip(s1, dims=[2] if axis == "vertical" else [3])
        return cls.outputs(latent=out)


class GenerationLatentCrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.crop")

    @classmethod
    def execute(
        cls, *, samples: object, width: int, height: int, x: int, y: int
    ) -> Mapping[str, object]:
        _check_bounds(
            ("width", width, 64, 16_384),
            ("height", height, 64, 16_384),
            ("x", x, 0, 16_384),
            ("y", y, 0, 16_384),
        )
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        x = x // 8
        y = y // 8
        # Clamp the origin so the crop keeps at least 8 latent cells (64 px).
        if x > (s_in.shape[3] - 8):
            x = s_in.shape[3] - 8
        if y > (s_in.shape[2] - 8):
            y = s_in.shape[2] - 8
        new_height = height // 8
        new_width = width // 8
        out["samples"] = s_in[:, :, y : y + new_height, x : x + new_width]
        return cls.outputs(latent=out)


class GenerationLatentResize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.resize")

    @classmethod
    def execute(
        cls, *, samples: object, method: str, width: int, height: int, crop: str
    ) -> Mapping[str, object]:
        _check_choice("method", method, _LATENT_RESIZE_METHODS)
        _check_choice("crop", crop, ("disabled", "center"))
        _check_bounds(("width", width, 0, 16_384), ("height", height, 0, 16_384))
        if width == 0 and height == 0:
            return cls.outputs(latent=samples)
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        if width == 0:
            height = max(64, height)
            width = max(64, round(s_in.shape[-1] * height / s_in.shape[-2]))
        elif height == 0:
            width = max(64, width)
            height = max(64, round(s_in.shape[-2] * width / s_in.shape[-1]))
        else:
            width = max(64, width)
            height = max(64, height)
        out["samples"] = importlib.import_module("dinkster_inference_torch.resize").common_upscale(
            s_in, width // 8, height // 8, method, crop
        )
        return cls.outputs(latent=out)


class GenerationLatentResizeBy(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.resize_by")

    @classmethod
    def execute(cls, *, samples: object, method: str, scale_by: float) -> Mapping[str, object]:
        _check_choice("method", method, _LATENT_RESIZE_METHODS)
        _check_bounds(("scale_by", scale_by, 0.01, 8.0))
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        width = round(s_in.shape[-1] * scale_by)
        height = round(s_in.shape[-2] * scale_by)
        out["samples"] = importlib.import_module("dinkster_inference_torch.resize").common_upscale(
            s_in, width, height, method, "disabled"
        )
        return cls.outputs(latent=out)


class GenerationLatentComposite(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.composite")

    @classmethod
    def execute(
        cls, *, destination: object, source: object, x: int, y: int, feather: int
    ) -> Mapping[str, object]:
        _check_bounds(
            ("x", x, 0, 16_384),
            ("y", y, 0, 16_384),
            ("feather", feather, 0, 16_384),
        )
        torch = _torch()
        out, dest = _plain_latent(destination, torch, "destination")
        _, src = _plain_latent(source, torch, "source")
        x = x // 8
        y = y // 8
        feather = feather // 8
        s = dest.clone()
        if feather == 0:
            s[:, :, y : y + src.shape[2], x : x + src.shape[3]] = src[
                :, :, : dest.shape[2] - y, : dest.shape[3] - x
            ]
        else:
            src = src[:, :, : dest.shape[2] - y, : dest.shape[3] - x]
            mask = torch.ones_like(src)
            for t in range(feather):
                if y != 0:
                    mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)
                if y + src.shape[2] < dest.shape[2]:
                    mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (1.0 / feather) * (
                        t + 1
                    )
                if x != 0:
                    mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
                if x + src.shape[3] < dest.shape[3]:
                    mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (1.0 / feather) * (
                        t + 1
                    )
            rev_mask = torch.ones_like(mask) - mask
            s[:, :, y : y + src.shape[2], x : x + src.shape[3]] = (
                src * mask + s[:, :, y : y + src.shape[2], x : x + src.shape[3]] * rev_mask
            )
        out["samples"] = s
        return cls.outputs(latent=out)


class GenerationLatentCompositeMasked(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.composite_masked")

    @classmethod
    def execute(
        cls,
        *,
        destination: object,
        source: object,
        x: int,
        y: int,
        resize_source: bool,
        mask: object = None,
    ) -> Mapping[str, object]:
        _check_bounds(("x", x, 0, 16_384), ("y", y, 0, 16_384))
        torch = _torch()
        out, dest = _plain_latent(destination, torch, "destination")
        _, src = _plain_latent(source, torch, "source")
        if mask is not None and type(mask) is not torch.Tensor:
            raise TypeError("mask must be an exact torch.Tensor")
        out["samples"] = _composite_masked_tensor(
            dest.clone(), src, x, y, mask, 8, resize_source, torch
        )
        return cls.outputs(latent=out)


class GenerationLatentConcat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.concat")

    @classmethod
    def execute(cls, *, samples1: object, samples2: object, dim: str) -> Mapping[str, object]:
        _check_choice("dim", dim, ("x", "-x", "y", "-y", "t", "-t"))
        torch = _torch()
        out, s1 = _plain_latent(samples1, torch, "samples1")
        _, s2 = _plain_latent(samples2, torch, "samples2")
        s2 = importlib.import_module("dinkster_inference_torch.resize").repeat_to_batch_size(
            s2, s1.shape[0]
        )
        ordered = (s2, s1) if "-" in dim else (s1, s2)
        axis = -1 if "x" in dim else -2 if "y" in dim else -3
        out["samples"] = torch.cat(ordered, dim=axis)
        return cls.outputs(latent=out)


class GenerationLatentCut(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.cut")

    @classmethod
    def execute(cls, *, samples: object, dim: str, index: int, amount: int) -> Mapping[str, object]:
        _check_choice("dim", dim, ("x", "y", "t"))
        _check_bounds(("index", index, -16_384, 16_384), ("amount", amount, 1, 16_384))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        axis = s1.ndim - 1 if dim == "x" else s1.ndim - 2 if dim == "y" else s1.ndim - 3
        if index >= 0:
            index = min(index, s1.shape[axis] - 1)
            amount = min(s1.shape[axis] - index, amount)
        else:
            index = max(index, -s1.shape[axis])
            amount = min(-index, amount)
        out["samples"] = torch.narrow(s1, axis, index, amount)
        return cls.outputs(latent=out)


class GenerationLatentCutToBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.cut_to_batch")

    @classmethod
    def execute(cls, *, samples: object, dim: str, slice_size: int) -> Mapping[str, object]:
        _check_choice("dim", dim, ("t", "x", "y"))
        _check_bounds(("slice_size", slice_size, 1, 16_384))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        axis = s1.ndim - 1 if dim == "x" else s1.ndim - 2 if dim == "y" else s1.ndim - 3
        if axis < 2:
            # The axis coincides with batch or channels (t on a 4D latent):
            # the latent passes through unchanged.
            return cls.outputs(latent=samples)
        s = s1.movedim(axis, 1)
        if s.shape[1] < slice_size:
            slice_size = s.shape[1]
        elif s.shape[1] % slice_size != 0:
            s = s[:, : math.floor(s.shape[1] / slice_size) * slice_size]
        new_shape = [-1, slice_size] + list(s.shape[2:])
        out["samples"] = s.reshape(new_shape).movedim(1, axis)
        return cls.outputs(latent=out)


class GenerationLatentFromBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.from_batch")

    @classmethod
    def execute(cls, *, samples: object, batch_index: int, length: int) -> Mapping[str, object]:
        _check_bounds(
            ("batch_index", batch_index, -16_384, 16_384),
            ("length", length, 1, 64),
        )
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        if batch_index < 0:
            batch_index = s_in.shape[0] + batch_index
        batch_index = max(0, min(s_in.shape[0] - 1, batch_index))
        length = min(s_in.shape[0] - batch_index, length)
        out["samples"] = s_in[batch_index : batch_index + length].clone()
        if "noise_mask" in out:
            masks = out["noise_mask"]
            if masks.shape[0] == 1:
                out["noise_mask"] = masks.clone()
            else:
                if masks.shape[0] < s_in.shape[0]:
                    masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[
                        : s_in.shape[0]
                    ]
                out["noise_mask"] = masks[batch_index : batch_index + length].clone()
        if "batch_index" not in out:
            out["batch_index"] = list(range(batch_index, batch_index + length))
        else:
            out["batch_index"] = out["batch_index"][batch_index : batch_index + length]
        return cls.outputs(latent=out)


class GenerationLatentRepeat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.repeat")

    @classmethod
    def execute(cls, *, samples: object, amount: int) -> Mapping[str, object]:
        _check_bounds(("amount", amount, 1, 64))
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        out["samples"] = s_in.repeat((amount,) + (1,) * (s_in.ndim - 1))
        if "noise_mask" in out and out["noise_mask"].shape[0] > 1:
            mask = out["noise_mask"]
            out["noise_mask"] = mask.repeat((amount,) + (1,) * (mask.ndim - 1))
        if "batch_index" in out:
            batch_index = out["batch_index"]
            offset = max(batch_index) - min(batch_index) + 1
            out["batch_index"] = batch_index + [
                x + i * offset for i in range(1, amount) for x in batch_index
            ]
        return cls.outputs(latent=out)


class GenerationLatentSeedBehavior(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.seed_behavior")

    @classmethod
    def execute(cls, *, samples: object, behavior: str) -> Mapping[str, object]:
        _check_choice("behavior", behavior, ("random", "fixed"))
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        if behavior == "random":
            out.pop("batch_index", None)
        else:
            out["batch_index"] = [out.get("batch_index", [0])[0]] * s_in.shape[0]
        return cls.outputs(latent=out)


class GenerationLatentBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.batch")

    @classmethod
    def execute(cls, *, latents: Mapping[str, object]) -> Mapping[str, object]:
        torch = _torch()
        members = list(latents.values())
        if not members:
            raise ValueError("latents requires at least one member")
        out, first = _plain_latent(members[0], torch, "latents member 1")
        tensors: list[Any] = []
        batch_index: list[int] = []
        for position, member in enumerate(members, start=1):
            mapping, s_in = _plain_latent(member, torch, f"latents member {position}")
            tensors.append(_reshape_latent_to(first.shape, s_in, repeat_batch=False))
            batch_index.extend(mapping.get("batch_index", list(range(s_in.shape[0]))))
        out["samples"] = torch.cat(tensors, dim=0)
        out["batch_index"] = batch_index
        return cls.outputs(latent=out)


def _rebatch_entry(mapping: Mapping[Any, Any], samples: Any, offset: int, torch: Any) -> Any:
    """Port of LatentRebatch.get_batch at b78cec87 (nodes_rebatch.py)."""
    shape = samples.shape
    mask = mapping.get("noise_mask")
    if mask is None:
        mask = torch.ones((shape[0], 1, shape[2] * 8, shape[3] * 8), device="cpu")
    if mask.shape[0] < samples.shape[0]:
        mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[: shape[0]]
    batch_inds = mapping.get("batch_index", [x + offset for x in range(shape[0])])
    return samples, mask, batch_inds


def _rebatch_slices(indexable: Any, num: int, batch_size: int) -> Any:
    """Port of LatentRebatch.get_slices at b78cec87."""
    slices = [indexable[i * batch_size : (i + 1) * batch_size] for i in range(num)]
    if num * batch_size < len(indexable):
        return slices, indexable[num * batch_size :]
    return slices, None


def _rebatch_slice_batch(batch: Any, num: int, batch_size: int) -> Any:
    """Port of LatentRebatch.slice_batch at b78cec87."""
    result = [_rebatch_slices(x, num, batch_size) for x in batch]
    return list(zip(*result, strict=True))


def _rebatch_cat(batch1: Any, batch2: Any, torch: Any) -> Any:
    """Port of LatentRebatch.cat_batch at b78cec87."""
    if batch1[0] is None:
        return batch2
    return [
        torch.cat((b1, b2)) if torch.is_tensor(b1) else b1 + b2
        for b1, b2 in zip(batch1, batch2, strict=True)
    ]


class GenerationLatentRebatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.rebatch")

    @classmethod
    def execute(cls, *, latents: object, batch_size: int) -> Mapping[str, object]:
        _check_bounds(("batch_size", batch_size, 1, 4_096))
        torch = _torch()
        if not isinstance(latents, Sequence) or isinstance(latents, (str, bytes)):
            raise TypeError("latents must be a list of LATENT mappings")
        entries = [
            _plain_latent(item, torch, f"latents entry {position}")
            for position, item in enumerate(cast("Sequence[object]", latents), start=1)
        ]
        output_list: list[dict[Any, Any]] = []
        current = (None, None, None)
        processed = 0
        for mapping, samples in entries:
            next_batch = _rebatch_entry(mapping, samples, processed, torch)
            processed += len(next_batch[2])
            if current[0] is None:
                current = next_batch
            elif (
                next_batch[0].shape[-1] != current[0].shape[-1]
                or next_batch[0].shape[-2] != current[0].shape[-2]
            ):
                sliced, _ = _rebatch_slice_batch(current, 1, batch_size)
                output_list.append(
                    {
                        "samples": sliced[0][0],
                        "noise_mask": sliced[1][0],
                        "batch_index": sliced[2][0],
                    }
                )
                current = next_batch
            else:
                current = _rebatch_cat(current, next_batch, torch)
            if current[0].shape[0] > batch_size:
                num = current[0].shape[0] // batch_size
                sliced, remainder = _rebatch_slice_batch(current, num, batch_size)
                for i in range(num):
                    output_list.append(
                        {
                            "samples": sliced[0][i],
                            "noise_mask": sliced[1][i],
                            "batch_index": sliced[2][i],
                        }
                    )
                current = remainder
        if current[0] is not None:
            sliced, _ = _rebatch_slice_batch(current, 1, batch_size)
            output_list.append(
                {
                    "samples": sliced[0][0],
                    "noise_mask": sliced[1][0],
                    "batch_index": sliced[2][0],
                }
            )
        for item in output_list:
            if item["noise_mask"].mean().item() == 1.0:
                del item["noise_mask"]
        return cls.outputs(latents=output_list)


class GenerationLatentSetNoiseMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.set_noise_mask")

    @classmethod
    def execute(cls, *, samples: object, mask: object) -> Mapping[str, object]:
        torch = _torch()
        out, _ = _plain_latent(samples, torch, "samples")
        if type(mask) is not torch.Tensor:
            raise TypeError("mask must be an exact torch.Tensor")
        mask_tensor = cast("Any", mask)
        out["noise_mask"] = mask_tensor.reshape(
            (-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])
        )
        return cls.outputs(latent=out)


class GenerationLatentReplaceFrames(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.replace_frames")

    @classmethod
    def execute(
        cls, *, destination: object, index: int, source: object = None
    ) -> Mapping[str, object]:
        _check_bounds(("index", index, -16_384, 16_384))
        if source is None:
            return cls.outputs(latent=destination)
        torch = _torch()
        _, dest = _plain_latent(destination, torch, "destination")
        out, src = _plain_latent(source, torch, "source")
        dest_frames = dest.shape[2]
        source_frames = src.shape[2]
        if index < 0:
            index = dest_frames + index
        if index > dest_frames:
            log.warning(
                "index %s is out of bounds for destination latent with %s frames",
                index,
                dest_frames,
            )
            return cls.outputs(latent=destination)
        if index + source_frames > dest_frames:
            log.warning(
                "source latent with %s frames at index %s does not fit destination "
                "latent with %s frames",
                source_frames,
                index,
                dest_frames,
            )
            return cls.outputs(latent=destination)
        merged = dest.clone()
        merged[:, :, index : index + source_frames] = src
        out["samples"] = merged
        return cls.outputs(latent=out)


def _apply_latent_operation(operation: _LatentOperationValue, latent: Any) -> Any:
    """Applies the torch-layer port of the LatentOperation* closures at b78cec87."""
    transforms = importlib.import_module("dinkster_inference_torch.guidance_transforms")
    return transforms.apply_latent_operation(operation.kind, operation.params, latent)


class GenerationLatentApplyOperation(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.apply_operation")

    @classmethod
    def execute(cls, *, samples: object, operation: object) -> Mapping[str, object]:
        if type(operation) is not _LatentOperationValue:
            raise TypeError("operation must be produced by a latent operation node")
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        out["samples"] = _apply_latent_operation(operation, s_in)
        return cls.outputs(latent=out)


class GenerationLatentOperationTonemapReinhard(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.operation_tonemap_reinhard")

    @classmethod
    def execute(cls, *, multiplier: float) -> Mapping[str, object]:
        _check_bounds(("multiplier", multiplier, 0.0, 100.0))
        return cls.outputs(
            operation=_LatentOperationValue(
                "tonemap_reinhard", (("multiplier", float(multiplier)),)
            )
        )


class GenerationLatentOperationSharpen(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.operation_sharpen")

    @classmethod
    def execute(cls, *, sharpen_radius: int, sigma: float, alpha: float) -> Mapping[str, object]:
        _check_bounds(
            ("sharpen_radius", sharpen_radius, 1, 31),
            ("sigma", sigma, 0.1, 10.0),
            ("alpha", alpha, 0.0, 5.0),
        )
        return cls.outputs(
            operation=_LatentOperationValue(
                "sharpen",
                (
                    ("sharpen_radius", int(sharpen_radius)),
                    ("sigma", float(sigma)),
                    ("alpha", float(alpha)),
                ),
            )
        )


class GenerationLatentApplyOperationCFG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.apply_operation_cfg")

    @classmethod
    def execute(cls, *, model: object, operation: object) -> Mapping[str, object]:
        if type(operation) is not _LatentOperationValue:
            raise TypeError("operation must be produced by a latent operation node")
        # The transform-chain position becomes the descriptor's order and a
        # distinct id, so the guidance registry's (order, id) sort reproduces
        # the reference's model_options append order for chained operations.
        _, _, _, _, _, transforms, _, _ = _native_model(model, "model")
        index = len(transforms)
        contribution = _guidance_transform_factory(
            "latent_operation",
            operation.kind,
            operation.params,
            descriptor_id=f"dinkster.latent-operation.{index}",
            order=index,
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model, "dinkster.latent.apply_operation_cfg", contribution
            )
        )


class GenerationLatentGenerateNoise(Node):
    """Port of GenerateNoise (KJNodes nodes/nodes.py @ 3f200542): one
    float32 CPU draw of the whole selected shape from a privately
    seeded generator (same stream as the reference's global
    torch.manual_seed), sigma/multiplier scaling, then optional
    normalize and constant-batch repeat, in the reference's order."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.generate_noise")

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        batch_size: int,
        seed: int,
        multiplier: float,
        constant_batch_noise: bool,
        normalize: bool,
        model: object | None = None,
        sigmas: object | None = None,
        latent_channels: str = "4",
        shape: str = "BCHW",
    ) -> Mapping[str, object]:
        _check_bounds(
            ("width", width, 16, 4_096),
            ("height", height, 16, 4_096),
            ("batch_size", batch_size, 1, 4_096),
            ("seed", seed, 0, KSampler.MAX_SEED),
            ("multiplier", multiplier, 0.0, 4_096.0),
        )
        _check_choice("latent_channels", latent_channels, ("4", "16"))
        _check_choice("shape", shape, ("BCHW", "BCTHW", "BTCHW"))
        torch = _torch()
        channels = int(latent_channels)
        if shape == "BCHW":
            size = [batch_size, channels, height // 8, width // 8]
        elif shape == "BCTHW":
            size = [1, channels, batch_size, height // 8, width // 8]
        else:
            size = [1, batch_size, channels, height // 8, width // 8]
        generator = torch.Generator("cpu")
        generator.manual_seed(seed)
        noise = torch.randn(
            size,
            dtype=torch.float32,
            layout=torch.strided,
            generator=generator,
            device="cpu",
        )
        if sigmas is not None:
            if type(sigmas) is not _CustomSigmasValue:
                raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
            if model is None:
                raise ValueError("sigma-scaled noise generation requires the model input")
            if not sigmas.values:
                raise ValueError("sigmas must contain at least one value")
            inference = importlib.import_module("dinkster_inference")
            # Overlays and transforms are accepted but irrelevant: only the
            # family's latent scale factor participates, and no overlay
            # changes the latent space.
            handle = _native_model(model, "model")[0]
            descriptor = handle.runtime.family.latent
            if type(descriptor) is not inference.LatentDescriptor:
                raise ValueError("sigma-scaled noise generation requires a plain-latent family")
            # Float32 tensor arithmetic, matching the reference bitwise.
            sigmas_tensor = torch.tensor(sigmas.values, dtype=torch.float32)
            noise *= (sigmas_tensor[0] - sigmas_tensor[-1]) / descriptor.scale_factor
        noise *= multiplier
        if normalize:
            noise = noise / noise.std()
        if constant_batch_noise:
            # The reference's noise[0].repeat(batch_size, 1, 1, 1) only
            # produces a batch for the 4D BCHW layout; on the 5D shapes it
            # scrambles channels, so those combinations refuse instead.
            if shape != "BCHW":
                raise ValueError("constant_batch_noise requires the BCHW shape")
            noise = noise[0].repeat(batch_size, 1, 1, 1)
        return cls.outputs(latent={"samples": noise})


class GenerationLatentInjectNoise(Node):
    """Port of InjectNoiseToLatent (KJNodes nodes/nodes.py @ 3f200542),
    including its metadata-dropping output: only the combined samples
    survive, exactly like the reference's fresh {"samples"} return."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.inject_noise")

    @classmethod
    def execute(
        cls,
        *,
        latents: object,
        strength: float,
        noise: object,
        normalize: bool,
        average: bool,
        mask: object | None = None,
        mix_randn_amount: float = 0.0,
        seed: int = 123,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("strength", strength, 0.0, 200.0),
            ("mix_randn_amount", mix_randn_amount, 0.0, 1_000.0),
            ("seed", seed, 0, KSampler.MAX_SEED),
        )
        torch = _torch()
        _, samples_in = _plain_latent(latents, torch, "latents")
        _, noise_in = _plain_latent(noise, torch, "noise")
        samples = samples_in.clone().cpu()
        noise_t = noise_in.clone().cpu()
        if average:
            noised = (samples + noise_t) / 2
        else:
            noised = samples + noise_t * strength
        if normalize:
            noised = noised / noised.std()
        if mask is not None:
            if type(mask) is not torch.Tensor:
                raise TypeError("mask must be an exact torch.Tensor")
            if noised.ndim != 4:
                raise ValueError("mask blending requires a rank-4 latent")
            mask_tensor = cast("Any", mask)
            blend = torch.nn.functional.interpolate(
                mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])),
                size=(noised.shape[2], noised.shape[3]),
                mode="bilinear",
            )
            blend = blend.expand((-1, noised.shape[1], -1, -1))
            if blend.shape[0] < noised.shape[0]:
                blend = blend.repeat((noised.shape[0] - 1) // blend.shape[0] + 1, 1, 1, 1)[
                    : noised.shape[0]
                ]
            noised = blend * noised + (1 - blend) * samples
        if mix_randn_amount > 0:
            generator = torch.Generator("cpu")
            generator.manual_seed(seed)
            rand_noise = torch.randn(
                noised.size(),
                dtype=noised.dtype,
                layout=noised.layout,
                generator=generator,
                device="cpu",
            )
            noised = noised + (mix_randn_amount * rand_noise)
        return cls.outputs(latent={"samples": noised})
