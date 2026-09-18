"""Generate latent operation goldens from ComfyUI e20d433a.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/ComfyUI \
      /path/to/python tools/gen_latent_ops_goldens.py

The ComfyUI checkout must be clean and pinned to the commit below. Run the
generator twice and compare the printed sha256 before committing the fixture.

On Linux the canonical fixture is written; on other platforms a
platform-tuple-suffixed fixture is written with provenance the test loader
validates (torch.linspace and the interpolation kernels differ by ULPs
across torch builds, so exact-equality goldens are per-platform).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

BASELINE = "e20d433a4966dcc88fa5abbae6ace824cb78b263"
REPO = Path(__file__).resolve().parent.parent
OUT = (
    REPO
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "latent_ops_e20d433a.json"
)


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_record(tensor: Any) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "values": [float(value) for value in cpu.reshape(-1).tolist()],
    }


def build_goldens(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as _comfy_args  # pyright: ignore[reportMissingImports]

    # Golden math executes on CPU tensors; forcing CPU keeps the import working
    # on hosts whose torch build has no CUDA support.
    _comfy_args.cpu = True

    # nodes_post_processing must load before nodes_latent: the two modules
    # form an import cycle that only resolves in this order.
    import comfy_extras.nodes_post_processing  # noqa: F401  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]
    from comfy_extras import (  # pyright: ignore[reportMissingImports]
        nodes_latent,
        nodes_mask,
        nodes_post_processing,
        nodes_rebatch,
    )
    from nodes import (  # pyright: ignore[reportMissingImports]
        LatentBlend,
        LatentComposite,
        LatentCrop,
        LatentFlip,
        LatentFromBatch,
        LatentRotate,
        LatentUpscale,
        LatentUpscaleBy,
        RepeatLatentBatch,
        SetLatentNoiseMask,
    )

    def linspace(first: float, last: float, shape: tuple[int, ...]) -> Any:
        count = 1
        for size in shape:
            count *= size
        return torch.linspace(first, last, count, dtype=torch.float32).reshape(shape)

    sources = {
        "latent_a": linspace(-1.2, 1.35, (2, 4, 8, 6)),
        "latent_b": linspace(0.8, -1.05, (2, 4, 8, 6)),
        "latent_small": linspace(-0.6, 0.9, (1, 4, 4, 4)),
        "latent_wide": linspace(-1.0, 1.0, (1, 4, 16, 16)),
        "latent_wide_b": linspace(0.7, -1.15, (1, 4, 16, 16)),
        "latent_patch": linspace(1.0, -1.0, (1, 4, 8, 8)),
        "latent_video": linspace(-0.9, 1.1, (1, 4, 5, 4, 3)),
        "latent_video_b": linspace(0.5, -0.7, (1, 4, 5, 4, 3)),
        "mask_small": linspace(0.0, 1.0, (1, 6, 5)),
        "latent_batch3": linspace(-1.1, 1.2, (3, 4, 8, 6)),
        "latent_video_short": linspace(0.3, -0.4, (1, 4, 2, 4, 3)),
        "mask_batch2": linspace(0.1, 0.9, (2, 1, 64, 48)),
        "mask_one": linspace(0.2, 0.8, (1, 1, 64, 48)),
        "latent_c": linspace(0.55, -0.85, (2, 4, 8, 6)),
    }

    def latent(name: str) -> dict[str, Any]:
        return {"samples": sources[name].clone()}

    def v3(output: Any) -> Any:
        return output.result[0]["samples"]

    cases: dict[str, object] = {}

    cases["add"] = _tensor_record(
        v3(nodes_latent.LatentAdd.execute(latent("latent_a"), latent("latent_b")))
    )
    cases["add:reshape-repeat"] = _tensor_record(
        v3(nodes_latent.LatentAdd.execute(latent("latent_a"), latent("latent_small")))
    )
    cases["subtract"] = _tensor_record(
        v3(nodes_latent.LatentSubtract.execute(latent("latent_a"), latent("latent_b")))
    )
    # Upstream's per-position norms only broadcast for batch-1 latents, so
    # the interpolate cases pin batch-1 inputs.
    cases["interpolate"] = _tensor_record(
        v3(
            nodes_latent.LatentInterpolate.execute(
                latent("latent_wide"), latent("latent_wide_b"), 0.35
            )
        )
    )
    cases["interpolate:reshape"] = _tensor_record(
        v3(
            nodes_latent.LatentInterpolate.execute(
                latent("latent_wide"), latent("latent_small"), 0.7
            )
        )
    )
    cases["blend"] = _tensor_record(
        LatentBlend().blend(latent("latent_a"), latent("latent_b"), 0.65)[0]["samples"]
    )
    cases["blend:mismatched"] = _tensor_record(
        LatentBlend().blend(latent("latent_a"), latent("latent_small"), 0.4)[0]["samples"]
    )
    cases["multiply"] = _tensor_record(
        v3(nodes_latent.LatentMultiply.execute(latent("latent_a"), -1.75))
    )
    for rotation in ("none", "90 degrees", "180 degrees", "270 degrees"):
        cases[f"rotate:{rotation.split(' ')[0]}"] = _tensor_record(
            LatentRotate().rotate(latent("latent_a"), rotation)[0]["samples"]
        )
    cases["flip:vertical"] = _tensor_record(
        LatentFlip().flip(latent("latent_a"), "x-axis: vertically")[0]["samples"]
    )
    cases["flip:horizontal"] = _tensor_record(
        LatentFlip().flip(latent("latent_a"), "y-axis: horizontally")[0]["samples"]
    )
    cases["crop"] = _tensor_record(
        LatentCrop().crop(latent("latent_wide"), 64, 64, 32, 8)[0]["samples"]
    )
    cases["crop:clamped"] = _tensor_record(
        LatentCrop().crop(latent("latent_wide"), 64, 64, 120, 112)[0]["samples"]
    )
    for method in ("nearest-exact", "bilinear", "area", "bicubic", "bislerp"):
        cases[f"resize:{method}"] = _tensor_record(
            LatentUpscale().upscale(latent("latent_wide"), method, 96, 64, "disabled")[0]["samples"]
        )
    cases["resize:center"] = _tensor_record(
        LatentUpscale().upscale(latent("latent_wide"), "bilinear", 96, 64, "center")[0]["samples"]
    )
    cases["resize:width-zero"] = _tensor_record(
        LatentUpscale().upscale(latent("latent_wide"), "bilinear", 0, 96, "disabled")[0]["samples"]
    )
    cases["resize:height-zero"] = _tensor_record(
        LatentUpscale().upscale(latent("latent_wide"), "bilinear", 96, 0, "disabled")[0]["samples"]
    )
    cases["resize_by:bislerp-1.5x"] = _tensor_record(
        LatentUpscaleBy().upscale(latent("latent_wide"), "bislerp", 1.5)[0]["samples"]
    )
    cases["resize_by:nearest-half"] = _tensor_record(
        LatentUpscaleBy().upscale(latent("latent_wide"), "nearest-exact", 0.5)[0]["samples"]
    )
    cases["composite"] = _tensor_record(
        LatentComposite().composite(
            latent("latent_wide"), latent("latent_patch"), 32, 16, feather=0
        )[0]["samples"]
    )
    cases["composite:feather"] = _tensor_record(
        LatentComposite().composite(
            latent("latent_wide"), latent("latent_patch"), 32, 16, feather=16
        )[0]["samples"]
    )
    cases["composite_masked"] = _tensor_record(
        v3(
            nodes_mask.LatentCompositeMasked.execute(
                latent("latent_wide"), latent("latent_patch"), 8, 16, False
            )
        )
    )
    cases["composite_masked:mask"] = _tensor_record(
        v3(
            nodes_mask.LatentCompositeMasked.execute(
                latent("latent_wide"),
                latent("latent_patch"),
                8,
                16,
                False,
                sources["mask_small"].clone(),
            )
        )
    )
    cases["composite_masked:resize"] = _tensor_record(
        v3(
            nodes_mask.LatentCompositeMasked.execute(
                latent("latent_wide"),
                latent("latent_patch"),
                0,
                0,
                True,
                sources["mask_small"].clone(),
            )
        )
    )
    cases["concat:x"] = _tensor_record(
        v3(nodes_latent.LatentConcat.execute(latent("latent_a"), latent("latent_b"), "x"))
    )
    cases["concat:-x"] = _tensor_record(
        v3(nodes_latent.LatentConcat.execute(latent("latent_a"), latent("latent_b"), "-x"))
    )
    cases["concat:t"] = _tensor_record(
        v3(nodes_latent.LatentConcat.execute(latent("latent_video"), latent("latent_video_b"), "t"))
    )
    cases["concat:batch-repeat"] = _tensor_record(
        v3(nodes_latent.LatentConcat.execute(latent("latent_a"), latent("latent_patch"), "x"))
    )
    cases["cut:x"] = _tensor_record(
        v3(nodes_latent.LatentCut.execute(latent("latent_wide"), "x", 3, 4))
    )
    cases["cut:negative"] = _tensor_record(
        v3(nodes_latent.LatentCut.execute(latent("latent_wide"), "y", -5, 9))
    )
    cases["cut:t"] = _tensor_record(
        v3(nodes_latent.LatentCut.execute(latent("latent_video"), "t", 1, 2))
    )
    cases["cut:t-4d"] = _tensor_record(
        v3(nodes_latent.LatentCut.execute(latent("latent_a"), "t", 0, 1))
    )
    cases["cut_to_batch:t"] = _tensor_record(
        v3(nodes_latent.LatentCutToBatch.execute(latent("latent_video"), "t", 2))
    )
    cases["cut_to_batch:x"] = _tensor_record(
        v3(nodes_latent.LatentCutToBatch.execute(latent("latent_a"), "x", 3))
    )
    cases["cut_to_batch:passthrough"] = _tensor_record(
        v3(nodes_latent.LatentCutToBatch.execute(latent("latent_a"), "t", 2))
    )
    cases["cut_to_batch:oversize"] = _tensor_record(
        v3(nodes_latent.LatentCutToBatch.execute(latent("latent_a"), "x", 10))
    )

    def meta_latent(
        name: str, mask: str | None = None, batch_index: list[int] | None = None
    ) -> dict[str, Any]:
        value: dict[str, Any] = {"samples": sources[name].clone()}
        if mask is not None:
            value["noise_mask"] = sources[mask].clone()
        if batch_index is not None:
            value["batch_index"] = list(batch_index)
        return value

    def latent_record(value: dict[str, Any]) -> dict[str, object]:
        record: dict[str, object] = {"samples": _tensor_record(value["samples"])}
        if "noise_mask" in value:
            record["noise_mask"] = _tensor_record(value["noise_mask"])
        if "batch_index" in value:
            record["batch_index"] = [int(index) for index in value["batch_index"]]
        return record

    metadata_cases: dict[str, object] = {}

    metadata_cases["from_batch"] = latent_record(
        LatentFromBatch().frombatch(meta_latent("latent_a", mask="mask_batch2"), 1, 2)[0]
    )
    metadata_cases["from_batch:negative"] = latent_record(
        LatentFromBatch().frombatch(
            meta_latent("latent_a", mask="mask_one", batch_index=[5, 9]), -1, 1
        )[0]
    )
    metadata_cases["from_batch:mask-repeat"] = latent_record(
        LatentFromBatch().frombatch(meta_latent("latent_batch3", mask="mask_batch2"), 1, 2)[0]
    )
    metadata_cases["repeat"] = latent_record(
        RepeatLatentBatch().repeat(
            meta_latent("latent_a", mask="mask_batch2", batch_index=[3, 7]), 3
        )[0]
    )
    metadata_cases["repeat:mask-single"] = latent_record(
        RepeatLatentBatch().repeat(meta_latent("latent_a", mask="mask_one"), 2)[0]
    )
    metadata_cases["seed_behavior:fixed"] = latent_record(
        nodes_latent.LatentBatchSeedBehavior.execute(
            meta_latent("latent_a", batch_index=[4, 9]), "fixed"
        ).result[0]
    )
    metadata_cases["seed_behavior:fixed-default"] = latent_record(
        nodes_latent.LatentBatchSeedBehavior.execute(meta_latent("latent_a"), "fixed").result[0]
    )
    metadata_cases["seed_behavior:random"] = latent_record(
        nodes_latent.LatentBatchSeedBehavior.execute(
            meta_latent("latent_a", batch_index=[4, 9]), "random"
        ).result[0]
    )
    metadata_cases["batch"] = latent_record(
        nodes_latent.LatentBatch.execute(
            meta_latent("latent_a", batch_index=[2, 3]), meta_latent("latent_small")
        ).result[0]
    )
    metadata_cases["batch:multi"] = latent_record(
        nodes_post_processing.BatchLatentsNode.execute(
            {
                "latent_1": meta_latent("latent_a", batch_index=[2, 3]),
                "latent_2": meta_latent("latent_small"),
                "latent_3": meta_latent("latent_b"),
            }
        ).result[0]
    )
    metadata_cases["set_noise_mask"] = latent_record(
        SetLatentNoiseMask().set_mask(meta_latent("latent_a"), sources["mask_small"].clone())[0]
    )
    metadata_cases["replace_frames"] = latent_record(
        nodes_latent.ReplaceVideoLatentFrames.execute(
            meta_latent("latent_video", batch_index=[1]),
            2,
            meta_latent("latent_video_short", batch_index=[7]),
        ).result[0]
    )
    metadata_cases["replace_frames:negative"] = latent_record(
        nodes_latent.ReplaceVideoLatentFrames.execute(
            meta_latent("latent_video"), -2, meta_latent("latent_video_short")
        ).result[0]
    )
    metadata_cases["replace_frames:oob-start"] = latent_record(
        nodes_latent.ReplaceVideoLatentFrames.execute(
            meta_latent("latent_video", batch_index=[1]),
            6,
            meta_latent("latent_video_short"),
        ).result[0]
    )
    metadata_cases["replace_frames:oob-length"] = latent_record(
        nodes_latent.ReplaceVideoLatentFrames.execute(
            meta_latent("latent_video", batch_index=[1]),
            5,
            meta_latent("latent_video_short"),
        ).result[0]
    )
    metadata_cases["rebatch:merge"] = [
        latent_record(value)
        for value in nodes_rebatch.LatentRebatch.execute(
            [meta_latent("latent_a"), meta_latent("latent_b")], [3]
        ).result[0]
    ]
    metadata_cases["rebatch:masked"] = [
        latent_record(value)
        for value in nodes_rebatch.LatentRebatch.execute(
            [meta_latent("latent_a", mask="mask_batch2"), meta_latent("latent_b")], [4]
        ).result[0]
    ]
    metadata_cases["rebatch:mixed-dims"] = [
        latent_record(value)
        for value in nodes_rebatch.LatentRebatch.execute(
            [meta_latent("latent_a"), meta_latent("latent_small")], [4]
        ).result[0]
    ]
    metadata_cases["rebatch:split"] = [
        latent_record(value)
        for value in nodes_rebatch.LatentRebatch.execute(
            [meta_latent("latent_a", batch_index=[5, 6])], [1]
        ).result[0]
    ]

    def tonemap_operation(multiplier: float) -> Any:
        return nodes_latent.LatentOperationTonemapReinhard.execute(multiplier).result[0]

    def sharpen_operation(sharpen_radius: int, sigma: float, alpha: float) -> Any:
        return nodes_latent.LatentOperationSharpen.execute(sharpen_radius, sigma, alpha).result[0]

    operation_cases: dict[str, object] = {}

    operation_cases["apply:tonemap"] = _tensor_record(
        v3(nodes_latent.LatentApplyOperation.execute(latent("latent_a"), tonemap_operation(1.0)))
    )
    operation_cases["apply:tonemap-low"] = _tensor_record(
        v3(
            nodes_latent.LatentApplyOperation.execute(
                latent("latent_wide"), tonemap_operation(0.35)
            )
        )
    )
    operation_cases["apply:tonemap-high"] = _tensor_record(
        v3(
            nodes_latent.LatentApplyOperation.execute(
                latent("latent_batch3"), tonemap_operation(2.5)
            )
        )
    )
    operation_cases["apply:tonemap-video"] = _tensor_record(
        v3(
            nodes_latent.LatentApplyOperation.execute(
                latent("latent_video"), tonemap_operation(1.0)
            )
        )
    )
    operation_cases["apply:sharpen"] = _tensor_record(
        v3(
            nodes_latent.LatentApplyOperation.execute(
                latent("latent_wide"), sharpen_operation(9, 1.0, 0.1)
            )
        )
    )
    operation_cases["apply:sharpen-small"] = _tensor_record(
        v3(
            nodes_latent.LatentApplyOperation.execute(
                latent("latent_a"), sharpen_operation(3, 0.5, 0.25)
            )
        )
    )
    operation_cases["apply:sharpen-radius1"] = _tensor_record(
        v3(
            nodes_latent.LatentApplyOperation.execute(
                latent("latent_small"), sharpen_operation(1, 1.0, 0.5)
            )
        )
    )
    operation_cases["apply:tonemap-metadata"] = latent_record(
        nodes_latent.LatentApplyOperation.execute(
            meta_latent("latent_a", mask="mask_batch2", batch_index=[3, 7]),
            tonemap_operation(0.8),
        ).result[0]
    )

    from comfy.samplers import sampling_function  # pyright: ignore[reportMissingImports]

    class _RecordingModel:
        """Stands in for a ModelPatcher; records the pre-CFG callbacks a node sets."""

        def __init__(self) -> None:
            self.pre: list[Any] = []

        def clone(self) -> _RecordingModel:
            return self

        def set_model_sampler_pre_cfg_function(self, fn: Any, **_: Any) -> None:
            self.pre.append(fn)

    def cfg_case(operations: list[Any], cfg: float) -> dict[str, object]:
        recorder = _RecordingModel()
        model: Any = recorder
        for operation in operations:
            model = nodes_latent.LatentApplyOperationCFG.execute(model, operation).result[0]

        x = sources["latent_a"].clone()
        cond = sources["latent_b"].clone()
        uncond = sources["latent_c"].clone()

        # Upstream calc_cond_batch returns zeros for the skipped uncond lane
        # at cfg==1; the stub reproduces that so both branches of the node's
        # pre-CFG function are exercised by the recorded callbacks.
        def evaluate(args: dict[str, Any]) -> list[Any]:
            return [
                cond.clone(),
                uncond.clone() if args["conds"][1] is not None else torch.zeros_like(x),
            ]

        options: dict[str, Any] = {
            "sampler_calc_cond_batch_function": evaluate,
            "sampler_pre_cfg_function": list(recorder.pre),
        }
        sigma = torch.tensor([0.625] * x.shape[0], dtype=torch.float32)
        return _tensor_record(sampling_function(None, x, sigma, "u", "c", cfg, options))

    cfg_cases: dict[str, object] = {}

    cfg_cases["cfg:tonemap"] = cfg_case([tonemap_operation(0.8)], 3.5)
    cfg_cases["cfg:tonemap-cfg1"] = cfg_case([tonemap_operation(0.8)], 1.0)
    cfg_cases["cfg:sharpen"] = cfg_case([sharpen_operation(5, 0.7, 0.3)], 3.5)
    cfg_cases["cfg:sharpen-cfg1"] = cfg_case([sharpen_operation(5, 0.7, 0.3)], 1.0)
    cfg_cases["cfg:tonemap-then-sharpen"] = cfg_case(
        [tonemap_operation(1.2), sharpen_operation(3, 0.5, 0.25)], 3.5
    )

    document: dict[str, object] = {
        "baseline": BASELINE,
        "sources": {name: _tensor_record(tensor) for name, tensor in sources.items()},
        "cases": cases,
        "metadata_cases": metadata_cases,
        "operation_cases": operation_cases,
        "cfg_cases": cfg_cases,
    }
    provenance = tuple_provenance(str(torch.__version__), pin_cpu=True)
    if provenance:
        document["_meta"] = provenance
    return document


def main() -> None:
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    document = build_goldens(comfy_root.resolve())
    import torch  # pyright: ignore[reportMissingImports]

    out = platform_golden_path(OUT, str(torch.__version__))
    content = (json.dumps(document, indent=2) + "\n").encode()
    out.write_bytes(content)
    print(f"{out}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
