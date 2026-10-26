"""HiDream-O1 patch-grid smoothing through the inference model wrapper."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    MODEL_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class HiDreamO1PatchSeamSmoothing(Node):
    PATCH_SIZE = 32
    EDGE_FEATHER = 4
    SHIFTS_BY_PATTERN = {
        ("single_shift", 2): [(0, 0), (16, 16)],
        ("single_shift", 4): [(0, 0), (16, 0), (0, 16), (16, 16)],
        ("single_shift", 8): [
            (0, 0),
            (16, 0),
            (0, 16),
            (16, 16),
            (8, 8),
            (24, 8),
            (8, 24),
            (24, 24),
        ],
        ("symmetric", 2): [(-8, -8), (8, 8)],
        ("symmetric", 4): [(-8, -8), (8, -8), (-8, 8), (8, 8)],
        ("symmetric", 8): [
            (-12, -12),
            (4, -12),
            (-12, 4),
            (4, 4),
            (-4, -4),
            (12, -4),
            (-4, 12),
            (12, 12),
        ],
    }
    RAMP_LEVELS = {"2": [2], "4": [4], "ramp_2_4": [2, 4], "ramp_2_4_8": [2, 4, 8]}

    @staticmethod
    def _hann_tile(cy: int, cx: int, size: int = 32) -> Any:
        torch = importlib.import_module("torch")
        half = size // 2
        yy = torch.arange(size).view(size, 1)
        xx = torch.arange(size).view(1, size)
        dy = ((yy - cy + half) % size) - half
        dx = ((xx - cx + half) % size) - half
        return 0.25 * (1 + torch.cos(torch.pi * dy / half)) * (1 + torch.cos(torch.pi * dx / half))

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="HiDreamO1PatchSeamSmoothing",
            display_name="HiDream-O1 Patch Seam Smoothing",
            category="model/patch/hidream",
            description=(
                "Average the model output across multiple shifted patch-grid "
                "positions during the late portion of sampling. Cancels seams."
            ),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec(
                    "start_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.8,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "end_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "pattern",
                    TypeExpr.concrete(CORE_COMBO),
                    default="single_shift",
                    widget=ComboWidget(("single_shift", "symmetric")),
                ),
                InputSpec(
                    "passes",
                    TypeExpr.concrete(CORE_COMBO),
                    default="2",
                    widget=ComboWidget(("2", "4", "ramp_2_4", "ramp_2_4_8")),
                ),
                InputSpec(
                    "blend",
                    TypeExpr.concrete(CORE_COMBO),
                    default="average",
                    widget=ComboWidget(("average", "window", "median")),
                ),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        start_percent: float,
        end_percent: float,
        pattern: str,
        passes: str,
        blend: str,
        strength: float,
    ) -> Mapping[str, object]:
        if strength <= 0.0 or end_percent <= start_percent:
            return cls.outputs(MODEL=model)
        initialize_inference_process()
        torch = importlib.import_module("torch")
        extension = importlib.import_module("dinkster_inference.patcher_extension")
        patch_size = cls.PATCH_SIZE
        half = patch_size // 2
        shift_levels = [cls.SHIFTS_BY_PATTERN[(pattern, n)] for n in cls.RAMP_LEVELS[passes]]
        window_tile_levels: list[Any]
        if blend == "window":
            window_tile_levels = [
                torch.stack(
                    [
                        cls._hann_tile(
                            (half - sy) % patch_size, (half - sx) % patch_size, patch_size
                        )
                        for sy, sx in shifts
                    ],
                    dim=0,
                )
                for shifts in shift_levels
            ]
        else:
            window_tile_levels = [None] * len(shift_levels)
        patched = model.clone()
        sampling = patched.get_model_object("model_sampling")
        multiplier = float(sampling.multiplier)
        start_t = float(sampling.percent_to_sigma(start_percent)) * multiplier
        end_t = float(sampling.percent_to_sigma(end_percent)) * multiplier
        edge_ramp_cache: dict[Any, Any] = {}

        def get_edge_ramp(height: int, width: int, device: Any, dtype: Any) -> Any:
            key = (height, width, device, dtype)
            cached = edge_ramp_cache.get(key)
            if cached is not None:
                return cached
            feather = cls.EDGE_FEATHER
            ys = torch.minimum(
                torch.arange(height, device=device, dtype=torch.float32),
                (height - 1) - torch.arange(height, device=device, dtype=torch.float32),
            )
            xs = torch.minimum(
                torch.arange(width, device=device, dtype=torch.float32),
                (width - 1) - torch.arange(width, device=device, dtype=torch.float32),
            )
            y_mask = ((ys - patch_size) / feather).clamp(0, 1)
            x_mask = ((xs - patch_size) / feather).clamp(0, 1)
            ramp = (y_mask[:, None] * x_mask[None, :]).to(dtype)
            edge_ramp_cache[key] = ramp
            return ramp

        def smoothing_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
            x = args[0]
            timestep = float(args[1][0])
            pred = executor(*args, **kwargs)
            if not (end_t <= timestep <= start_t):
                return pred
            if len(shift_levels) == 1:
                level_idx = 0
            else:
                phase = (start_t - timestep) / max(start_t - end_t, 1e-8)
                level_idx = min(int(phase * len(shift_levels)), len(shift_levels) - 1)
            shifts = shift_levels[level_idx]
            window_tiles = window_tile_levels[level_idx]
            preds: list[Any] = []
            for sy, sx in shifts:
                if sy == 0 and sx == 0:
                    preds.append(pred)
                    continue
                rolled = torch.roll(x, shifts=(sy, sx), dims=(-2, -1))
                pred_rolled = executor(rolled, *args[1:], **kwargs)
                preds.append(torch.roll(pred_rolled, shifts=(-sy, -sx), dims=(-2, -1)))
            stacked = torch.stack(preds, dim=0)
            _, _, _, height, width = stacked.shape
            if blend == "window":
                count = stacked.shape[0]
                tiles = window_tiles.to(device=stacked.device, dtype=stacked.dtype)
                weights = tiles.repeat(1, height // patch_size, width // patch_size)[
                    :, :height, :width
                ]
                sum_weights = weights.sum(dim=0, keepdim=True)
                weights = torch.where(
                    sum_weights < 1e-3,
                    torch.full_like(weights, 1.0 / count),
                    weights / sum_weights.clamp(min=1e-8),
                )
                avg = (stacked * weights[:, None, None, :, :]).sum(dim=0)
            elif blend == "median":
                avg = torch.median(stacked, dim=0).values
            else:
                avg = stacked.mean(dim=0)
            mask = get_edge_ramp(height, width, pred.device, pred.dtype)
            return pred * (1.0 - mask * strength) + avg * (mask * strength)

        patched.add_wrapper_with_key(
            extension.WrappersMP.DIFFUSION_MODEL,
            "hidream_o1_patch_seam_smoothing",
            smoothing_wrapper,
        )
        return cls.outputs(MODEL=patched)
