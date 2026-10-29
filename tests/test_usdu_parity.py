"""Executed parity for the Ultimate SD Upscale compatibility expansion."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_assets import AssetError, AssetRef, LocalAssetLibrary
from dinkster_compat_comfy import (
    COMFY_INPUT_ADAPTERS,
    UpscaleModelLoaderCarrier,
    make_load_checkpoint_adapter,
    make_load_image_adapter,
    make_model_asset_inputs_adapter,
    translate_prompt,
)
from dinkster_engine import Engine, EngineEvent
from dinkster_server import comfy_cpu_args, comfy_dtype_args
from PIL import Image

from dinkster.comfy_compose import comfy_compat_specs
from dinkster.compose import PackSpec, ServingComposer, default_pack_spec
from dinkster.native_policy import NativeDispatchPolicy, NativePolicyDiagnostic

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "tests" / "goldens" / "usdu_a5547db9.json"
GOLDEN_SHA256 = "bb211d4bc9126feaf226c9580c3d7584008e6a8fe244f92a246e37da4aeb27f1"
COMFYUI_COMMIT = "a1079ba16f2674734b065eb036fbfdddaa321a4d"
CHECKPOINT_ENV = "DINKSTER_USDU_TEST_CHECKPOINT"
UPSCALE_MODEL_ENV = "DINKSTER_USDU_TEST_UPSCALE_MODEL"
COMFYUI_ENV = "DINKSTER_USDU_TEST_COMFYUI"
TORCH_PYTHON_ENV = "DINKSTER_USDU_TEST_PYTHON"
UPSCALE_MANIFEST = ROOT / "packages" / "dinkster-vision-upscale" / "dinkster-pack.toml"

# These cases use standard SD1.5 conditioning. crop_cond leaves conditioning
# without control, gligen, area, mask, or reference-latent fields unchanged,
# and crop_model_cond patches only DiffSynth and ZImage models:
# https://github.com/ssitu/ComfyUI_UltimateSDUpscale/blob/a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df/usdu_utils.py
# https://github.com/ssitu/ComfyUI_UltimateSDUpscale/blob/a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df/crop_model_patch.py
# The linear case observed max/mean/seam/outside-blend drift of
# 1.960e-2/8.366e-4/1.631e-2/1.176e-2. Its Pillow masks are exact; one-level
# decoded-tile boundary crossings are amplified when the next tile encodes and
# samples the preceding composite. Limits provide 1.25x-1.32x headroom.
# The band-pass case observed 7.033e-2/1.705e-3/7.033e-2/1.176e-2. The third
# encode/sample pass amplifies the same crossings, and its bicubic-gradient
# approximation adds 3.861e-3 mask drift; its 4e-3 gate has 1.036x headroom.
# Output limits provide 1.27x-1.30x headroom without weakening the linear case.
USDU_PARITY_LIMITS = {
    "linear": {
        "max_abs": 0.025,
        "mean_abs": 0.0011,
        "seam_max_abs": 0.021,
        "seam_mean_abs": 0.0023,
        "outside_blend_max_abs": 0.015,
    },
    "band_pass": {
        "max_abs": 0.09,
        "mean_abs": 0.0022,
        "seam_max_abs": 0.09,
        "seam_mean_abs": 0.0054,
        "outside_blend_max_abs": 0.015,
    },
}
USDU_MASK_DELTA_LIMITS = {"linear": 0.0, "band_pass": 0.004}


def _golden() -> dict[str, object]:
    data = GOLDEN_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == GOLDEN_SHA256
    return cast("dict[str, object]", json.loads(data))


def _decode(record: object) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload["uint8Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(shape)


def _float_pixels(record: object) -> np.ndarray:
    return _decode(record).astype(np.float32) / 255.0


async def _cached_array(
    engine: Engine,
    events: list[EngineEvent],
    node_id: str,
    output_id: str,
) -> np.ndarray:
    event = next(
        (
            item
            for item in reversed(events)
            if item.node_id == node_id and item.kind in {"node_finished", "node_cached"}
        ),
        None,
    )
    assert event is not None, node_id
    entry = await engine.cache.get(cast("str", event.detail["cache_key"]))
    assert entry is not None
    return np.asarray(entry[output_id].resolve(), dtype=np.float32)


def _reference_source_canvas(job: Mapping[str, object]) -> np.ndarray:
    before = _float_pixels(job["before"])
    source = _float_pixels(job["sample"])
    x1, y1, x2, y2 = cast("list[int]", job["crop"])
    assert source.shape == (y2 - y1, x2 - x1, before.shape[2])
    canvas = np.zeros_like(before)
    canvas[y1:y2, x1:x2] = source
    return canvas


def _composite_bound(
    native_before: np.ndarray,
    native_source: np.ndarray,
    native_mask: np.ndarray,
    native_after: np.ndarray,
    reference_job: Mapping[str, object],
) -> dict[str, float]:
    reference_before = _float_pixels(reference_job["before"])
    reference_after = _float_pixels(reference_job["after"])
    reference_mask = _float_pixels(reference_job["mask"])
    reference_source = _reference_source_canvas(reference_job)
    x1, y1, x2, y2 = cast("list[int]", reference_job["crop"])
    native_source_canvas = np.zeros_like(native_before)
    native_source_canvas[y1:y2, x1:x2] = native_source
    native_alpha = native_mask[..., None]
    reference_alpha = reference_mask[..., None]

    native_formula = native_before * (1.0 - native_alpha) + native_source_canvas * native_alpha
    reference_formula = reference_before * (1.0 - reference_alpha) + (
        reference_source * reference_alpha
    )
    native_rounding = np.abs(native_after - native_formula)
    reference_rounding = np.abs(reference_after - reference_formula)
    assert float(native_rounding.max()) <= 2e-7

    inherited = np.abs(native_before - reference_before) * (1.0 - native_alpha)
    sample = np.abs(native_source_canvas - reference_source) * native_alpha
    mask_amplification = np.abs(native_alpha - reference_alpha) * np.abs(
        reference_source - reference_before
    )
    observed = np.abs(native_after - reference_after)
    explained = inherited + sample + mask_amplification + reference_rounding
    assert np.all(observed <= explained + 3e-7)
    direct_mask_effect = np.abs(
        (reference_before * (1.0 - native_alpha) + reference_source * native_alpha)
        - reference_formula
    )
    assert np.allclose(direct_mask_effect, mask_amplification, rtol=0.0, atol=2e-7)
    return {
        "inherited_max_abs": float(inherited.max()),
        "mask_delta_max_abs": float(np.abs(native_alpha - reference_alpha).max()),
        "mask_amplification_max_abs": float(mask_amplification.max()),
        "observed_max_abs": float(observed.max()),
        "reference_rounding_max_abs": float(reference_rounding.max()),
        "sample_max_abs": float(sample.max()),
    }


def _configured_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is not configured")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{variable} does not exist: {path}")
    return path


def _check_artifact(path: Path, record: Mapping[str, object], label: str) -> None:
    assert path.is_file(), label
    assert path.stat().st_size == record["bytes"], label
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], label


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _asset_resolver(library: LocalAssetLibrary, category: str) -> Callable[[str], AssetRef | None]:
    def resolve(name: str) -> AssetRef | None:
        try:
            return library.ref(f"models/{category}/{name}")
        except AssetError:
            return None

    return resolve


def _prompt(
    parameters: Mapping[str, object],
    choices: Mapping[str, object],
    *,
    checkpoint_name: str,
    upscale_model_name: str,
) -> dict[str, object]:
    usdu_inputs = {
        key: value
        for key, value in parameters.items()
        if key not in {"positive_text", "negative_text"}
    }
    usdu_inputs.update(
        {
            "image": ["source", 0],
            "model": ["checkpoint", 0],
            "positive": ["positive", 0],
            "negative": ["negative", 0],
            "vae": ["checkpoint", 2],
            "upscale_model": ["upscale_loader", 0],
            **choices,
        }
    )
    return {
        "source": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "checkpoint": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["checkpoint", 1], "text": parameters["positive_text"]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["checkpoint", 1], "text": parameters["negative_text"]},
        },
        "upscale_loader": {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": upscale_model_name},
        },
        "usdu": {"class_type": "UltimateSDUpscale", "inputs": usdu_inputs},
        "output": {"class_type": "dinkster.preview_image", "inputs": {"images": ["usdu", 0]}},
    }


async def _add_packs(
    composer: ServingComposer,
    *,
    comfyui: Path,
    torch_python: Path,
    asset_root: Path,
) -> None:
    for pack_id in ("dinkster-nodes-media-io", "dinkster-nodes-foundation", "dinkster-nodes-image"):
        await composer.add_pack(default_pack_spec(pack_id))
    upscale = PackSpec(
        UPSCALE_MANIFEST,
        python=str(torch_python),
        env={"DINKSTER_ASSET_ROOT": str(asset_root)},
    )
    await composer.add_pack(upscale)
    output_root = asset_root / "output"
    temp_root = asset_root / "temp"
    user_root = asset_root / "user"
    for path in (output_root, temp_root, user_root):
        path.mkdir()
    generation, compat = comfy_compat_specs(
        comfyui,
        python=str(torch_python),
        comfy_nodes=("CLIPTextEncode",),
        comfy_args=(
            *comfy_cpu_args(),
            *comfy_dtype_args({"diffusion": "float32", "textEncoder": "float32", "vae": "float32"}),
            "--input-directory",
            str(asset_root / "input"),
            "--output-directory",
            str(output_root),
            "--temp-directory",
            str(temp_root),
            "--user-directory",
            str(user_root),
        ),
    )
    await composer.add_pack(generation)
    await composer.add_pack(
        replace(
            compat,
            env={
                **compat.env,
                "DINKSTER_ASSET_ROOT": str(asset_root),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    )


def test_ultimate_sd_upscale_matches_the_pinned_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = _golden()
    environment = cast("dict[str, str]", golden["environment"])
    assert environment == {
        "comfyui": COMFYUI_COMMIT,
        "extension": "a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df",
        "numpy": "2.5.2",
        "pillow": "12.3.0",
        "spandrel": "0.4.2",
        "torch": "2.13.0+cpu",
        "upstream": "2322caa480535b1011a1f9c18126d85ea444f146",
    }
    artifacts = cast("dict[str, dict[str, object]]", golden["artifacts"])
    checkpoint = _configured_path(CHECKPOINT_ENV)
    upscale_model = _configured_path(UPSCALE_MODEL_ENV)
    comfyui = _configured_path(COMFYUI_ENV)
    torch_python = Path(os.environ.get(TORCH_PYTHON_ENV, ROOT / ".venv-torch/bin/python"))
    if not torch_python.is_file():
        pytest.fail(f"torch worker interpreter does not exist: {torch_python}")
    assert _git_head(comfyui) == COMFYUI_COMMIT
    _check_artifact(checkpoint, artifacts["checkpoint"], "checkpoint")
    _check_artifact(upscale_model, artifacts["upscale_model"], "upscale model")

    asset_root = tmp_path / "assets"
    for category, artifact in (("checkpoints", checkpoint), ("upscale_models", upscale_model)):
        destination = asset_root / category / artifact.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(artifact)
    source = _decode(golden["source"])
    input_path = asset_root / "input" / "source.png"
    input_path.parent.mkdir(parents=True)
    Image.fromarray(source, mode="RGB").save(input_path)
    library = LocalAssetLibrary(asset_root)
    library.scan()
    monkeypatch.setenv("DINKSTER_ASSET_ROOT", str(asset_root))
    checkpoint_ref = library.ref(f"models/checkpoints/{checkpoint.name}")
    diagnostics: list[NativePolicyDiagnostic] = []
    policy = NativeDispatchPolicy(
        lambda digest: checkpoint if digest == checkpoint_ref.digest else None,
        diagnostics.append,
        dtype_policy=lambda: {
            "diffusion": "float32",
            "textEncoder": "float32",
            "vae": "float32",
        },
    )

    async def scenario() -> None:
        composer = ServingComposer(native_policy=policy)
        try:
            await _add_packs(
                composer,
                comfyui=comfyui,
                torch_python=torch_python,
                asset_root=asset_root,
            )
            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            adapters = {
                **COMFY_INPUT_ADAPTERS,
                "dinkster.load_image": make_load_image_adapter(_asset_resolver(library, "input")),
                "dinkster.load_checkpoint": make_load_checkpoint_adapter(
                    _asset_resolver(library, "checkpoints")
                ),
                UpscaleModelLoaderCarrier.schema().node_type: make_model_asset_inputs_adapter(
                    {
                        "model_name": (
                            _asset_resolver(library, "upscale_models"),
                            "upscale_models",
                        )
                    }
                ),
            }
            parameters = cast("dict[str, object]", golden["parameters"])
            cases = cast("dict[str, dict[str, object]]", golden["cases"])
            reference_jobs = cast("dict[str, list[dict[str, object]]]", golden["jobs"])
            for name, case in cases.items():
                translation = translate_prompt(
                    _prompt(
                        parameters,
                        cast("dict[str, object]", case["choices"]),
                        checkpoint_name=checkpoint.name,
                        upscale_model_name=upscale_model.name,
                    ),
                    composer.composition.schemas,
                    input_adapters=adapters,
                )
                result = await engine.run(translation.graph, translation.targets)
                value = result.outputs["output"]["images"]
                actual = cast("np.ndarray[Any, Any]", value.resolve())[0]
                expected = _float_pixels(case["output"])
                assert actual.shape == expected.shape == (64, 128, 3)
                job_labels = [
                    "usdu__usdu_redraw[0]",
                    "usdu__usdu_redraw[1]",
                ]
                if name == "band_pass":
                    job_labels.append("usdu[0]")
                assert len(job_labels) == len(reference_jobs[name])
                native_before = (
                    await _cached_array(
                        engine,
                        events,
                        "usdu__usdu_canvas",
                        "image",
                    )
                )[0]
                blend_band = np.zeros(actual.shape[:2], dtype=np.bool_)
                decomposition: list[dict[str, float]] = []
                mask_max_abs = 0.0
                for label, reference_job in zip(job_labels, reference_jobs[name], strict=True):
                    native_mask = await _cached_array(engine, events, f"{label}/mask", "mask")
                    native_source = await _cached_array(engine, events, f"{label}/restore", "image")
                    native_after = await _cached_array(
                        engine, events, f"{label}/composite", "image"
                    )
                    reference_mask = _float_pixels(reference_job["mask"])
                    mask_delta = np.abs(native_mask[0] - reference_mask)
                    blend_band |= ((native_mask[0] > 0.0) & (native_mask[0] < 1.0)) | (
                        (reference_mask > 0.0) & (reference_mask < 1.0)
                    )
                    mask_max_abs = max(mask_max_abs, float(mask_delta.max()))
                    decomposition.append(
                        _composite_bound(
                            native_before,
                            native_source[0],
                            native_mask[0],
                            native_after[0],
                            reference_job,
                        )
                    )
                    native_before = native_after[0]
                assert np.allclose(actual, native_before, rtol=0.0, atol=0.0)
                assert np.array_equal(
                    _decode(case["output"]),
                    _decode(reference_jobs[name][-1]["after"]),
                )
                error = np.abs(actual - expected)
                seam_error = error[:, 48:80]
                outside_blend_error = error[~blend_band]
                metrics = {
                    "max_abs": float(error.max()),
                    "mean_abs": float(error.mean()),
                    "seam_max_abs": float(seam_error.max()),
                    "seam_mean_abs": float(seam_error.mean()),
                    "outside_blend_max_abs": float(outside_blend_error.max()),
                }
                print(
                    name,
                    json.dumps(
                        {
                            "decomposition": decomposition,
                            "mask_max_abs": mask_max_abs,
                            "metrics": metrics,
                        },
                        sort_keys=True,
                    ),
                )
                assert mask_max_abs <= USDU_MASK_DELTA_LIMITS[name]
                for metric, limit in USDU_PARITY_LIMITS[name].items():
                    assert metrics[metric] <= limit
            assert diagnostics == []
        finally:
            await composer.close()

    asyncio.run(scenario())
