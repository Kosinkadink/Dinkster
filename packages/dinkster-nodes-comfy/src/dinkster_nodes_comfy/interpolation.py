"""Native FILM and RIFE frame interpolation."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    INTERP_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    report_progress,
)


class FrameInterpolate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FrameInterpolate",
            display_name="Run Frame Interpolation Model",
            category="video",
            search_terms=(
                "rife",
                "film",
                "frame interpolation",
                "slow motion",
                "interpolate frames",
                "vfi",
            ),
            inputs=(
                InputSpec("interp_model", TypeExpr.concrete(INTERP_MODEL_TYPE)),
                InputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "multiplier",
                    TypeExpr.concrete(CORE_INT),
                    default=2,
                    widget=NumberWidget(min=2, max=16),
                ),
            ),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls, *, interp_model: Any, images: Any, multiplier: int) -> Mapping[str, object]:
        images = materialize_image(images)
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        offload_device = management.intermediate_device()
        num_frames = images.shape[0]
        if num_frames < 2 or multiplier < 2:
            return cls.outputs(IMAGE=images)

        device = interp_model.load_device
        dtype = interp_model.model_dtype()
        model = interp_model.model
        activation_mem = model.memory_used_forward(images.shape, dtype)
        management.load_models_gpu([interp_model], memory_required=activation_mem)
        align = getattr(model, "pad_align", 1)
        height, width = images.shape[1:3]

        def prepare_frame(index: int) -> Any:
            frame = images[index : index + 1].movedim(-1, 1).to(dtype=dtype, device=device)
            if align > 1:
                dit = importlib.import_module("dinkster_inference.ldm.common_dit")
                frame = dit.pad_to_patch_size(frame, (align, align), padding_mode="reflect")
            return frame

        total_pairs = num_frames - 1
        num_interp = multiplier - 1
        total_steps = total_pairs * num_interp
        step = 0
        report_progress(step, total_steps, text="Frame interpolation")
        batch = num_interp
        times = [t / multiplier for t in range(1, multiplier)]
        out_dtype = management.intermediate_dtype()
        result = torch.empty(
            ((total_pairs * multiplier + 1), 3, height, width),
            dtype=out_dtype,
            device=offload_device,
        )
        result[0] = images[0].movedim(-1, 0).to(out_dtype)
        out_index = 1
        sample = prepare_frame(0)
        padded_height, padded_width = sample.shape[2:4]
        ts_full = torch.tensor(times, device=device, dtype=dtype).reshape(num_interp, 1, 1, 1)
        ts_full = ts_full.expand(-1, 1, padded_height, padded_width)
        del sample
        multi_fn = getattr(model, "forward_multi_timestep", None)
        cache: dict[str, Any] = {}
        previous = None
        for index in range(total_pairs):
            first = previous if previous is not None else prepare_frame(index)
            second = prepare_frame(index + 1)
            previous = second
            cache["img0"] = cache.pop("next") if "next" in cache else model.extract_features(first)
            cache["img1"] = model.extract_features(second)
            cache["next"] = cache["img1"]
            used_multi = False
            if multi_fn is not None:
                try:
                    mids = multi_fn(first, second, times, cache=cache)
                    result[out_index : out_index + num_interp] = mids[:, :, :height, :width].to(
                        out_dtype
                    )
                    out_index += num_interp
                    step += num_interp
                    report_progress(step, total_steps, text="Frame interpolation")
                    used_multi = True
                except management.OOM_EXCEPTION:
                    management.soft_empty_cache()
                    multi_fn = None
            if not used_multi:
                offset = 0
                while offset < num_interp:
                    count = min(batch, num_interp - offset)
                    try:
                        mids = model(
                            first.expand(count, -1, -1, -1),
                            second.expand(count, -1, -1, -1),
                            timestep=ts_full[offset : offset + count],
                            cache=cache,
                        )
                        result[out_index : out_index + count] = mids[:, :, :height, :width].to(
                            out_dtype
                        )
                        out_index += count
                        step += count
                        report_progress(step, total_steps, text="Frame interpolation")
                        offset += count
                    except management.OOM_EXCEPTION:
                        if batch <= 1:
                            raise
                        batch = max(1, batch // 2)
                        management.soft_empty_cache()
            result[out_index] = images[index + 1].movedim(-1, 0).to(out_dtype)
            out_index += 1
        return cls.outputs(IMAGE=result.movedim(1, -1).clamp_(0.0, 1.0))
