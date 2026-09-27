"""Compare Dinkster's SD1.5 worker path with the pinned ComfyUI oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

COMFYUI_REVISION = "4ef23c34d950eecc37040a21ee1741a49d2e44b1"
PROMPT = "a red cube on a blue table"
SEED = 459
STEPS = 5
CFG = 7.0
WIDTH = 256
HEIGHT = 256


def _sha256_tensor(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _run_stock(checkpoint: Path, stock_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(stock_root))
    torch = cast("Any", importlib.import_module("torch"))
    sd = importlib.import_module("comfy.sd")
    sample = importlib.import_module("comfy.sample")
    model, clip, vae, _ = sd.load_checkpoint_guess_config(
        str(checkpoint),
        output_vae=True,
        output_clip=True,
        embedding_directory=[],
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    latent = torch.zeros((1, 4, HEIGHT // 8, WIDTH // 8), device="cpu")
    records = []
    for _ in range(2):
        noise = sample.prepare_noise(latent, SEED, None)
        samples = sample.sample(
            model,
            noise,
            STEPS,
            CFG,
            "euler",
            "normal",
            positive,
            negative,
            latent,
            denoise=1.0,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=None,
            callback=None,
            disable_pbar=True,
            seed=SEED,
        )
        image = vae.decode(samples)
        records.append({"latent": _sha256_tensor(samples), "image": _sha256_tensor(image)})
    module_file = sd.__file__
    if module_file is None:
        raise RuntimeError("stock comfy.sd has no module file")
    return {
        "implementation": str(Path(module_file).resolve()),
        "torch": torch.__version__,
        "gpu": str(torch.cuda.get_device_properties(0).uuid),
        "records": records,
    }


class _FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _run_worker(checkpoint: Path) -> dict[str, object]:
    from dinkster_assets import AssetRef, digest_file
    from dinkster_native.fork_nodes import (
        GenerationClipTextEncode,
        GenerationKSampler,
        GenerationLoadCheckpoint,
        GenerationVAEDecode,
    )

    torch = cast("Any", importlib.import_module("torch"))
    sd = importlib.import_module("dinkster_comfy.sd")
    loaded = GenerationLoadCheckpoint.execute(
        checkpoint=AssetRef(
            digest=digest_file(checkpoint),
            name=checkpoint.name,
            size=checkpoint.stat().st_size,
            resolver=_FixedResolver(checkpoint),
        )
    )
    model, clip, vae = loaded["model"], loaded["clip"], loaded["vae"]
    positive = GenerationClipTextEncode.execute(text=PROMPT, clip=clip)["conditioning"]
    negative = GenerationClipTextEncode.execute(text="", clip=clip)["conditioning"]
    latent = {"samples": torch.zeros((1, 4, HEIGHT // 8, WIDTH // 8), device="cpu")}
    records = []
    for _ in range(2):
        sampled = cast(
            "dict[str, Any]",
            GenerationKSampler.execute(
                model=model,
                seed=SEED,
                steps=STEPS,
                cfg=CFG,
                sampler_name="euler",
                scheduler="normal",
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=1.0,
            )["latent"],
        )
        image = GenerationVAEDecode.execute(samples=sampled, vae=vae)["image"]
        records.append(
            {
                "latent": _sha256_tensor(sampled["samples"]),
                "image": _sha256_tensor(image),
            }
        )
    module_file = sd.__file__
    if module_file is None:
        raise RuntimeError("dinkster_comfy.sd has no module file")
    return {
        "implementation": str(Path(module_file).resolve()),
        "torch": torch.__version__,
        "gpu": str(torch.cuda.get_device_properties(0).uuid),
        "records": records,
    }


def _git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _child(command: str, checkpoint: Path, stock_root: Path | None) -> None:
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    if command == "stock":
        if stock_root is None:
            raise SystemExit("--stock-root is required for the stock child")
        result = _run_stock(checkpoint, stock_root)
    else:
        result = _run_worker(checkpoint)
    print(json.dumps(result, sort_keys=True))


def _run_child(command: str, checkpoint: Path, stock_root: Path | None) -> dict[str, Any]:
    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        command,
        "--checkpoint",
        str(checkpoint),
    ]
    if stock_root is not None:
        invocation.extend(("--stock-root", str(stock_root)))
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    return json.loads(completed.stdout.splitlines()[-1])


def _compare(checkpoint: Path, stock_root: Path) -> None:
    revision = _git_revision(stock_root)
    if revision != COMFYUI_REVISION:
        raise SystemExit(f"stock root is {revision}, expected pinned ComfyUI {COMFYUI_REVISION}")
    stock = _run_child("stock", checkpoint, stock_root)
    worker = _run_child("worker", checkpoint, None)
    fresh_worker = _run_child("worker", checkpoint, None)
    oracle = stock["records"][0]
    observed = [*worker["records"], *fresh_worker["records"]]
    if any(record != oracle for record in observed):
        raise SystemExit(
            json.dumps(
                {"status": "mismatch", "stock": stock, "worker": worker, "fresh": fresh_worker},
                indent=2,
                sort_keys=True,
            )
        )
    if worker["gpu"] != stock["gpu"] or fresh_worker["gpu"] != stock["gpu"]:
        raise SystemExit("stock and worker executions did not use the same physical GPU")
    print(
        json.dumps(
            {
                "status": "pass",
                "comfyui_revision": revision,
                "graph": {
                    "prompt": PROMPT,
                    "seed": SEED,
                    "steps": STEPS,
                    "cfg": CFG,
                    "sampler": "euler",
                    "scheduler": "normal",
                    "width": WIDTH,
                    "height": HEIGHT,
                },
                "stock": stock,
                "worker": worker,
                "fresh_worker": fresh_worker,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--child", choices=("stock", "worker"))
    args = parser.parse_args()
    if args.child:
        _child(args.child, args.checkpoint, args.stock_root)
        return
    if args.stock_root is None:
        parser.error("--stock-root is required")
    _compare(args.checkpoint.resolve(), args.stock_root.resolve())


if __name__ == "__main__":
    main()
