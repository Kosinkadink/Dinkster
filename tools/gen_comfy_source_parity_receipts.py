"""Generate source-parity receipts from pinned reference evidence.

The selected ComfyUI implementations are read with ``git show`` from the exact
commits below. Framework types and deterministic filesystem paths are stubbed,
while the selected execute and video-input methods run with real Python,
NumPy, torch, Pillow, and PyAV. Pinned ecosystem receipts are projected from
their committed cross-process execution evidence. Group receipts compare
independently specified source interfaces with the maintained replacement
mapper. This generator owns the complete ``docs/comfy-confidence-receipts``
tree in the sibling ``dinkster-evidence`` checkout. Set
``DINKSTER_EVIDENCE_ROOT`` to use a different evidence checkout.

Run with the torch test interpreter, then run again with ``--check``:

    IMPACT_PACK_SOURCE=/path/to/special_samplers.py \
      IMPACT_PACK_CORE_SOURCE=/path/to/core.py \
      INSPIRE_PACK_SOURCE=/path/to/sampler_nodes.py \
      .venv-torch/bin/python tools/gen_comfy_source_parity_receipts.py
    IMPACT_PACK_SOURCE=/path/to/special_samplers.py \
      IMPACT_PACK_CORE_SOURCE=/path/to/core.py \
      INSPIRE_PACK_SOURCE=/path/to/sampler_nodes.py \
      .venv-torch/bin/python tools/gen_comfy_source_parity_receipts.py --check

Set ``COMFYUI_ROOT`` when the ComfyUI git checkout is not beside Dinkster or its
parent. The two Impact Pack paths and Inspire Pack path must name files from
their pinned commits and are verified by SHA-256 before execution.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import io
import itertools
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

import av
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

if __package__:
    from tools.evidence_paths import EVIDENCE_ROOT
else:
    from evidence_paths import EVIDENCE_ROOT

REPO = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
VIDEO_REFERENCE_COMMIT = "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
IMPACT_PACK_COMMIT = "429d0159ad429e64d2b3916e6e7be9c22d025c3c"
IMPACT_PACK_SOURCE_SHA256 = "32647b60dd169f953b04ebdacf9e2a59c3870ecf442f01f3f76d96906283989c"
IMPACT_PACK_CORE_SHA256 = "7989de178999904cacf440bd6c9fd6d250ff5f8df8a4a2dad7abe33d2dfacdd5"
INSPIRE_PACK_COMMIT = "6b2ca017a168bcdba5f22c258b3b86c5c76470ca"
INSPIRE_PACK_SOURCE_SHA256 = "153c5cbf334ca67176569625d10feac924bb289a8c18321bcaee0d9a99d18c51"
VISION_REFERENCE_COMMIT = "c67885b14556cf3e4e061862925282d403d09862"
RECEIPT_ROOT = EVIDENCE_ROOT / "docs" / "comfy-confidence-receipts"
DIRECT_GOLDEN = REPO / "tests" / "goldens" / "comfy_direct_operations_b78cec87.json"
INFERENCE_PARITY_RECORDS = Path(
    os.environ.get(
        "DINKSTER_INFERENCE_PARITY_RECORDS",
        EVIDENCE_ROOT / "inference-parity" / "records",
    )
)
TRELLIS2_COMFYUI_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"
TRELLIS2_WORKFLOW_COMMIT = "d3b4a9e89573162b005961865164c18c8ae2206b"
TRELLIS2_WORKFLOW_PATH = "templates/3d_pixal3d_trellis2_image_to_model.json"
TRELLIS2_EVIDENCE = INFERENCE_PARITY_RECORDS / "trellis2-1083" / "official-workflow.json"
TRELLIS2_BASELINE_RECORDS = frozenset(("ChromaRadianceOptions", "EmptyChromaRadianceLatentImage"))
CONTROL_AUX_EVIDENCE = INFERENCE_PARITY_RECORDS / "controlnet-aux-line-edge-1067"
CONTROL_AUX_COMFYUI_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"
CONTROL_AUX_COMMIT = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
CONTROL_AUX_INPUT_SHA256 = "28e2ffe0c96d7c7d44c45ff10c6754ef9741c638caa06c619774f82a0d4e12c5"
CONTROL_AUX_GPU_UUID = "5ac69527-f5f0-f6f0-1d46-24c6f401cdc6"
CONTROL_AUX_RESOLUTION = 512
CONTROL_AUX_RUNNER_SHA256 = "5e1e1d52dc097a491061835b24a22a8fdf4197a9e79abbd3138793aa231442f6"

for source_root in sorted(REPO.glob("packages/*/src")):
    sys.path.insert(0, str(source_root))
sys.path.insert(0, str(REPO / "tools"))

from comfy_confidence import canonical_bytes, create_receipt, write_receipt  # noqa: E402
from dinkster_api.v1 import render_video_original, save_video_stream  # noqa: E402
from dinkster_assets import AssetRef, digest_file  # noqa: E402
from dinkster_compat_comfy import native as compat_native  # noqa: E402
from dinkster_compat_comfy import native_arm  # noqa: E402
from dinkster_inference import (  # noqa: E402
    SEEDVR2_CODEC,
    CancellationToken,
    Conditioning,
    GuidanceCondition,
    GuidanceEvaluationRequest,
    GuidancePlanContext,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceRole,
    ProgressScope,
    SamplingExecutionContext,
)
from dinkster_inference_torch import (  # noqa: E402
    SeedVR2DiffusionRuntime,
    materialize_seedvr2_conditioning,
)
from dinkster_inference_torch.guidance import GuidanceExecutor, GuidanceRegistry  # noqa: E402
from dinkster_model_qwen_image.provider import (  # noqa: E402
    execute_empty_qwen_image_layered_latent,
)
from dinkster_model_wan.wandancer_audio import plan_wandancer_keyframe_list  # noqa: E402
from dinkster_nodes_foundation.basic import ConcatStrings  # noqa: E402
from dinkster_nodes_foundation.lists import ListElement  # noqa: E402
from dinkster_nodes_foundation.numeric import ValueSelect  # noqa: E402
from dinkster_nodes_foundation.primitives import BooleanPrimitive  # noqa: E402
from dinkster_nodes_foundation.string_ops import (  # noqa: E402
    StringLength,
    StringRegex,
    StringTest,
    StringTransform,
)
from dinkster_nodes_image.batch import ImageBatchCombine, ImageRebatch  # noqa: E402
from dinkster_nodes_image.compare import ImageCompare  # noqa: E402
from dinkster_nodes_image.geometry import ImageResize  # noqa: E402
from dinkster_nodes_media_io import (  # noqa: E402
    AssembleVideo,
    CropVideo,
    DisassembleVideo,
    LoadImage,
    LoadMask,
    LoadVideoValue,
    SaveImage,
    SaveVideoValue,
    TrimAudio,
    TrimVideo,
)
from dinkster_values import image_input  # noqa: E402
from generate_math_expression_vector import build_vector, load_reference  # noqa: E402


class _NodeOutput(tuple[object, ...]):
    ui: object

    def __new__(cls, *values: object, ui: object = None) -> _NodeOutput:
        output = tuple.__new__(cls, values)
        output.ui = ui
        return output


class _ComfyNode:
    pass


class _IO:
    ComfyNode = _ComfyNode
    DynamicCombo = SimpleNamespace(Type=object)
    FolderType = SimpleNamespace(output="output")
    NodeOutput = _NodeOutput
    Video = SimpleNamespace(Type=object)
    VideoEdit = SimpleNamespace(Type=object)


class _ReferenceFolderPaths:
    input_path: Path | None = None
    output_directory: Path | None = None

    @classmethod
    def get_annotated_filepath(cls, _name: str) -> str:
        if cls.input_path is None:
            raise RuntimeError("reference input path is not configured")
        return os.fspath(cls.input_path)

    @classmethod
    def exists_annotated_filepath(cls, _name: str) -> bool:
        return cls.input_path is not None and cls.input_path.is_file()

    @classmethod
    def get_output_directory(cls) -> str:
        if cls.output_directory is None:
            raise RuntimeError("reference output directory is not configured")
        return os.fspath(cls.output_directory)


class _FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == digest_file(self.path) else None


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    previous = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, previous)


@contextmanager
def _patched_classmethod(
    owner: type[Any], name: str, value: Callable[..., object]
) -> Iterator[None]:
    previous = inspect.getattr_static(owner, name)
    setattr(owner, name, classmethod(value))
    try:
        yield
    finally:
        setattr(owner, name, previous)


def _comfy_root() -> Path:
    configured = os.environ.get("COMFYUI_ROOT")
    candidates = (
        Path(configured) if configured else None,
        REPO.parent / "ComfyUI",
        REPO.parent.parent / "ComfyUI",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / ".git").exists():
            return candidate.resolve()
    raise SystemExit("set COMFYUI_ROOT to the ComfyUI git checkout")


def _workflow_templates_root() -> Path:
    configured = os.environ.get("WORKFLOW_TEMPLATES_ROOT")
    candidates = (
        Path(configured) if configured else None,
        REPO.parent / "workflow_templates",
        REPO.parent.parent / "workflow_templates",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / ".git").exists():
            return candidate.resolve()
    raise SystemExit("set WORKFLOW_TEMPLATES_ROOT to the workflow_templates git checkout")


def _verified_source(environment_name: str, expected_sha256: str) -> str:
    configured = os.environ.get(environment_name)
    if configured is None:
        raise SystemExit(f"set {environment_name} to the pinned source file")
    path = Path(configured)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"cannot read {environment_name}={path}: {exc}") from exc
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(f"{environment_name} must have SHA-256 {expected_sha256}, got {digest}")
    return content.decode("utf-8")


def _git(comfy_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(comfy_root), *arguments),
        check=True,
        capture_output=True,
    ).stdout


def _last_decodable_audio_stream(container: object) -> object | None:
    streams = cast("Any", container).streams.audio
    return next((stream for stream in reversed(streams) if stream.codec_context is not None), None)


def _load_reference_api(
    comfy_root: Path, *, revision: str = REFERENCE_COMMIT
) -> tuple[object, object]:
    enum_stub = SimpleNamespace(AUTO="auto", MP4="mp4", MKV="mkv", WEBM="webm")
    classes = _load_reference_classes(
        comfy_root,
        "comfy_api/latest/_input_impl/video_types.py",
        ("VideoFromFile", "VideoFromComponents"),
        {
            "AudioInput": lambda value: value,
            "Fraction": Fraction,
            "InputContainer": object,
            "Optional": Optional,
            "VIDEO_COLOR_TRANSFERS": {"sRGB": "sRGB", "HDR": "HDR", "HDR PQ": "HDR PQ"},
            "VideoCodec": enum_stub,
            "VideoComponents": SimpleNamespace,
            "VideoContainer": enum_stub,
            "VideoInput": object,
            "av": av,
            "io": io,
            "itertools": itertools,
            "last_decodable_audio_stream": _last_decodable_audio_stream,
            "logging": logging,
            "np": np,
            "torch": torch,
        },
        revision=revision,
    )
    return (
        SimpleNamespace(
            VideoFromComponents=classes["VideoFromComponents"],
            VideoFromFile=classes["VideoFromFile"],
        ),
        SimpleNamespace(VideoComponents=SimpleNamespace),
    )


def _load_reference_classes(
    comfy_root: Path,
    source_path: str,
    names: Sequence[str],
    globals_: Mapping[str, object],
    *,
    revision: str = REFERENCE_COMMIT,
) -> dict[str, type[Any]]:
    source = _git(comfy_root, "show", f"{revision}:{source_path}").decode("utf-8")
    return _load_source_classes(source, source_path, names, globals_)


def _load_source_classes(
    source: str,
    source_path: str,
    names: Sequence[str],
    globals_: Mapping[str, object],
) -> dict[str, type[Any]]:
    tree = ast.parse(source, filename=source_path)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    found = {node.name for node in selected}
    if found != set(names):
        raise RuntimeError(f"{source_path} is missing source classes: {sorted(set(names) - found)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"io": _IO, **globals_}
    exec(compile(module, source_path, "exec"), namespace)
    return {name: cast("type[Any]", namespace[name]) for name in names}


def _load_reference_functions(
    comfy_root: Path,
    source_path: str,
    names: Sequence[str],
    globals_: Mapping[str, object],
) -> dict[str, Callable[..., object]]:
    source = _git(comfy_root, "show", f"{REFERENCE_COMMIT}:{source_path}").decode("utf-8")
    return _load_source_functions(source, source_path, names, globals_)


def _load_source_functions(
    source: str,
    source_path: str,
    names: Sequence[str],
    globals_: Mapping[str, object],
) -> dict[str, Callable[..., object]]:
    tree = ast.parse(source, filename=source_path)
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != set(names):
        raise RuntimeError(
            f"{source_path} is missing source functions: {sorted(set(names) - found)}"
        )
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, source_path, "exec"), namespace)
    return {name: cast("Callable[..., object]", namespace[name]) for name in names}


def _canonical_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _array_value(value: object) -> dict[str, object]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.name,
        "data": contiguous.reshape(-1).tolist(),
    }


def _array_descriptor(value: object) -> dict[str, object]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.name,
        "nonzero": int(np.count_nonzero(contiguous)),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _mapping_records(source_pack: str = "comfy-core") -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    paths = sorted((REPO / "packages").glob("*/comfy-aliases.json"))
    paths.extend(sorted((REPO / "packages").glob("*/comfy-groups.json")))
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for value in cast("list[dict[str, Any]]", document["records"]):
            source = cast("dict[str, object]", value["source"])
            if source["pack"] == source_pack:
                source_name = source.get("nodeClass", source.get("name"))
                if not isinstance(source_name, str):
                    raise RuntimeError(f"{path} contains a mapping without a source name")
                records[source_name] = value
    return records


def _group_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((REPO / "packages").glob("*/comfy-groups.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for value in cast("list[dict[str, Any]]", document["records"]):
            source = cast("dict[str, object]", value["source"])
            if source["pack"] == "comfy-core":
                records[cast("str", source["name"])] = value
    return records


def _source_input_defaults(source_pack: str) -> dict[str, dict[str, object]]:
    records = _mapping_records(source_pack)
    node_classes = {
        cast("str", cast("Mapping[str, object]", record["source"])["nodeType"]): node_class
        for node_class, record in records.items()
    }
    defaults: dict[str, dict[str, object]] = {}
    for path in sorted((REPO / "packages").glob("*/comfy-aliases.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for schema in cast("list[Mapping[str, Any]]", document["sourceSchemas"]):
            node_class = node_classes.get(cast("str", schema["nodeType"]))
            if node_class is None:
                continue
            values = {
                cast("str", item["id"]): item["default"]
                for item in cast("list[Mapping[str, object]]", schema["interface"])
                if item.get("role") == "input" and "default" in item
            }
            previous = defaults.setdefault(node_class, values)
            if previous != values:
                raise RuntimeError(f"conflicting source schemas for {source_pack}/{node_class}")
    if defaults.keys() != records.keys():
        missing = sorted(records.keys() - defaults.keys())
        raise RuntimeError(f"missing source schemas for {source_pack}: {missing}")
    return defaults


def _mapping_case(record: Mapping[str, Any]) -> Mapping[str, Any]:
    replacement = cast("Mapping[str, Any]", record["replacement"])
    cases = cast("list[Mapping[str, Any]]", replacement["cases"])
    if len(cases) != 1 or cases[0].get("to") != record["carrier"]:
        raise RuntimeError(f"expected one direct replacement case for {record['id']}")
    return cases[0]


def _mapped_input(
    record: Mapping[str, Any],
    target_name: str,
    source_inputs: Mapping[str, object],
) -> object:
    case = _mapping_case(record)
    inputs = cast("Mapping[str, Mapping[str, Any]]", case["inputs"])
    spec = inputs[target_name]
    kind = spec.get("kind")
    if kind == "constant":
        return spec["value"]
    if kind == "copy":
        return source_inputs[cast("str", spec["input"])]
    if kind == "value":
        value = source_inputs[cast("str", spec["input"])]
        transform = cast("Mapping[str, Any]", spec.get("transform"))
        if transform.get("kind") == "enumRename":
            return cast("Mapping[object, object]", transform["map"])[value]
        if transform.get("kind") == "scale":
            return cast(float, value) * transform["factor"] + transform.get("offset", 0)
    raise RuntimeError(f"unsupported {record['id']} mapping input {target_name}")


def _receipt_mapping(record: Mapping[str, Any]) -> dict[str, object]:
    source = cast("Mapping[str, object]", record["source"])
    confidence = cast("Mapping[str, object]", record["confidence"])
    source_pack = cast("str", source["pack"])
    registry_kind = "group" if str(record["id"]).startswith("comfy_group:") else "alias"
    source_name = source["name"] if registry_kind == "group" else source["nodeClass"]
    return {
        "registryId": record["id"],
        "mappingKind": record["mappingKind"],
        "tier": confidence["tier"],
        "source": {
            "pack": source["pack"],
            "name": source_name,
            "revision": source["revision"],
            "referenceKind": "comfyui-pinned"
            if source_pack == "comfy-core"
            else "ecosystem-pinned",
        },
        "target": {
            "kind": "group" if registry_kind == "group" else "node",
            "id": record["carrier"],
        },
    }


def _record_digest(record: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(record)).hexdigest()


def _write_mapping_receipt(
    root: Path,
    record: Mapping[str, Any],
    *,
    slug: str,
    parameters: dict[str, object],
    reference: object,
    native: object,
    comparison: dict[str, object] | None = None,
) -> list[Path]:
    source = cast("Mapping[str, object]", record["source"])
    source_pack = cast("str", source["pack"])
    artifact_directory = root / "artifacts" / source_pack
    array = comparison is not None and comparison["dataKind"] != "value"
    suffix = "npy" if array else "json"
    reference_path = artifact_directory / f"{slug}.reference.{suffix}"
    native_path = artifact_directory / f"{slug}.native.{suffix}"
    if array:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        np.save(reference_path, np.asarray(reference), allow_pickle=False)
        np.save(native_path, np.asarray(native), allow_pickle=False)
    else:
        _canonical_write(reference_path, reference)
        _canonical_write(native_path, native)
    receipt_path = root / source_pack / f"{slug}.receipt.json"
    receipt = create_receipt(
        case_id=f"{source_pack}/{slug}/source-parity",
        mapping=_receipt_mapping(record),
        parameters={"mappingDigest": _record_digest(record), **parameters},
        comparison=comparison or {"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=root,
        reference_path=reference_path.relative_to(root).as_posix(),
        native_path=native_path.relative_to(root).as_posix(),
    )
    if receipt["pass"] is not True:
        raise RuntimeError(f"source parity failed for {record['id']}")
    write_receipt(receipt_path, receipt)
    return [reference_path, native_path, receipt_path]


def _impact_schedule(scheduler: str, steps: int) -> tuple[float, ...]:
    scheduler = scheduler.removeprefix("dinkster.")
    factor = {"simple": 1.0, "normal": 1.25}[scheduler]
    return tuple(float(steps - index) * factor for index in range(steps)) + (0.0,)


def _impact_mask_value(mask: object | None) -> object:
    if mask is None:
        return None
    value = cast("torch.Tensor", mask)
    return _array_value(value.squeeze())


def _impact_trace_call(
    trace: list[dict[str, object]],
    *,
    provider: str,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    add_noise: bool,
    sigmas: Sequence[float],
    mask: object | None,
) -> None:
    trace.append(
        {
            "provider": provider,
            "cfg": cfg,
            "sampler": sampler.removeprefix("dinkster."),
            "scheduler": scheduler.removeprefix("dinkster."),
            "seed": seed,
            "addNoise": add_noise,
            "sigmas": [float(value) for value in sigmas],
            "mask": _impact_mask_value(mask),
        }
    )


def _impact_reference_classes(
    trace: list[dict[str, object]],
    composites: list[dict[str, object]],
) -> dict[str, type[Any]]:
    source = _verified_source("IMPACT_PACK_SOURCE", IMPACT_PACK_SOURCE_SHA256)
    core_source = _verified_source("IMPACT_PACK_CORE_SOURCE", IMPACT_PACK_CORE_SHA256)
    erosion_mask = _load_source_functions(
        core_source,
        "modules/impact/core.py",
        ("erosion_mask",),
        {
            "comfy": SimpleNamespace(
                model_management=SimpleNamespace(get_torch_device=lambda: torch.device("cpu"))
            ),
            "math": math,
            "torch": torch,
            "utils": SimpleNamespace(
                make_2d_mask=lambda mask: (
                    mask.squeeze(0).squeeze(0)
                    if len(mask.shape) == 4
                    else mask.squeeze(0)
                    if len(mask.shape) == 3
                    else mask
                )
            ),
        },
    )["erosion_mask"]

    class ReferenceSampler:
        def __init__(
            self,
            model: object,
            cfg: float,
            sampler_name: str,
            scheduler: str,
            positive: object,
            negative: object,
            **_kwargs: object,
        ) -> None:
            self.params = (model, cfg, sampler_name, scheduler, positive, negative)

        def sample_advanced(
            self,
            add_noise: bool,
            seed: int,
            steps: int,
            latent: Mapping[object, object],
            start_at_step: int,
            end_at_step: int,
            return_with_leftover_noise: bool,
            **_kwargs: object,
        ) -> dict[object, object]:
            del return_with_leftover_noise
            if start_at_step >= steps:
                return dict(latent)
            model, cfg, sampler, scheduler, _positive, _negative = self.params
            sigmas = _impact_schedule(cast("str", scheduler), steps)
            _impact_trace_call(
                trace,
                provider=cast("str", model),
                cfg=cast("float", cfg),
                sampler=cast("str", sampler),
                scheduler=cast("str", scheduler),
                seed=seed,
                add_noise=add_noise,
                sigmas=sigmas[start_at_step : end_at_step + 1],
                mask=latent.get("noise_mask"),
            )
            return dict(latent)

    class ReferenceNoise:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def generate_noise(self, _latent: object) -> ReferenceNoise:
            return self

    class ReferenceCompositor:
        def composite(
            self,
            destination: Mapping[object, object],
            _source: Mapping[object, object],
            _x: int,
            _y: int,
            _resize_source: bool,
            mask: object,
        ) -> tuple[dict[object, object]]:
            composites.append({"mask": _impact_mask_value(mask)})
            return (dict(destination),)

    core = SimpleNamespace(get_schedulers=lambda: ())
    nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={"LatentCompositeMasked": ReferenceCompositor})
    classes = _load_source_classes(
        source,
        "modules/impact/special_samplers.py",
        ("KSamplerAdvancedProvider", "RegionalPrompt", "RegionalSampler"),
        {
            "KSamplerAdvancedWrapper": ReferenceSampler,
            "Noise_RandomNoise": ReferenceNoise,
            "comfy": SimpleNamespace(
                sample=SimpleNamespace(fix_empty_latent_channels=lambda _model, value: value),
                samplers=SimpleNamespace(KSampler=SimpleNamespace(SAMPLERS=())),
            ),
            "core": core,
            "math": math,
            "nodes": nodes,
            "np": np,
            "separated_sample": lambda *args, **kwargs: (args, kwargs),
            "torch": torch,
        },
    )

    class ReferenceRegionalPrompt:
        def __init__(
            self,
            mask: torch.Tensor,
            sampler: object,
            **_kwargs: object,
        ) -> None:
            self.mask = mask
            self.sampler = sampler

        def get_mask_erosion(self, factor: int) -> object:
            return erosion_mask(self.mask, factor)

        @staticmethod
        def touch_noise(noise: object) -> object:
            return noise

    core.REGIONAL_PROMPT = ReferenceRegionalPrompt
    core.update_node_status = lambda *_args, **_kwargs: None
    return classes


def _impact_source_case(
    parameters: Mapping[str, object],
    mask: torch.Tensor,
    samples: Mapping[str, object],
) -> dict[str, object]:
    trace: list[dict[str, object]] = []
    composites: list[dict[str, object]] = []
    classes = _impact_reference_classes(trace, composites)
    base_pipe = ("base", None, None, "base-positive", "base-negative")
    region_pipe = ("region", None, None, "region-positive", "region-negative")
    base_sampler = classes["KSamplerAdvancedProvider"].doit(
        parameters["base_cfg"],
        parameters["base_sampler_name"],
        parameters["base_scheduler"],
        base_pipe,
    )[0]
    region_sampler = classes["KSamplerAdvancedProvider"].doit(
        parameters["region_cfg"],
        parameters["region_sampler_name"],
        parameters["region_scheduler"],
        region_pipe,
    )[0]
    regional_prompts = classes["RegionalPrompt"].doit(mask, region_sampler)[0]
    result = classes["RegionalSampler"].doit(
        parameters["seed"],
        0,
        "ignore",
        parameters["steps"],
        parameters["base_only_steps"],
        parameters["denoise"],
        samples,
        base_sampler,
        regional_prompts,
        parameters["overlap_factor"],
        parameters["restore_latent"],
        "DISABLE",
        "AUTO",
        0.3,
        unique_id="receipt",
    )[0]
    return {
        "trace": trace,
        "composites": composites,
        "result": {
            "metadata": result["metadata"],
            "samples": _array_value(result["samples"]),
        },
    }


def _impact_native_case(
    record: Mapping[str, Any],
    parameters: Mapping[str, object],
    mask: torch.Tensor,
    samples: Mapping[str, object],
) -> dict[str, object]:
    trace: list[dict[str, object]] = []
    composites: list[dict[str, object]] = []

    def provider(
        value: object,
        *,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        steps: int,
        **_kwargs: object,
    ) -> object:
        model = cast("tuple[object, ...]", value)[0]
        return SimpleNamespace(
            model=model,
            cfg=cfg,
            sampler=sampler_name,
            scheduler=scheduler,
            sigmas=_impact_schedule(scheduler, steps),
        )

    def run_pass(
        selected: Any,
        latent: Mapping[object, object],
        sigmas: tuple[float, ...],
        *,
        seed: int,
        add_noise: bool,
        mask: object | None,
    ) -> dict[object, object]:
        _impact_trace_call(
            trace,
            provider=cast("str", selected.model),
            cfg=cast("float", selected.cfg),
            sampler=cast("str", selected.sampler),
            scheduler=cast("str", selected.scheduler),
            seed=seed,
            add_noise=add_noise,
            sigmas=sigmas,
            mask=mask,
        )
        return dict(latent)

    def composite(
        destination: torch.Tensor,
        _source: torch.Tensor,
        _x: int,
        _y: int,
        region_mask: object,
        _multiplier: int,
        _resize_source: bool,
        _torch: object,
    ) -> torch.Tensor:
        composites.append({"mask": _impact_mask_value(region_mask)})
        return destination

    source_inputs = {
        **parameters,
        "base_basic_pipe": ("base", None, None, "base-positive", "base-negative"),
        "region_basic_pipe": ("region", None, None, "region-positive", "region-negative"),
        "mask": mask,
        "samples": samples,
    }
    mapped = {
        name: _mapped_input(record, name, source_inputs)
        for name in cast("Mapping[str, object]", _mapping_case(record)["inputs"])
    }
    with ExitStack() as patches:
        patches.enter_context(_patched(native_arm, "_impact_regional_provider", provider))
        patches.enter_context(_patched(native_arm, "_run_impact_regional_pass", run_pass))
        patches.enter_context(_patched(native_arm, "_composite_masked_tensor", composite))
        result = native_arm.GenerationImpactRegionalSampler.execute(**mapped)["latent"]
    typed_result = cast("Mapping[str, object]", result)
    return {
        "trace": trace,
        "composites": composites,
        "result": {
            "metadata": typed_result["metadata"],
            "samples": _array_value(typed_result["samples"]),
        },
    }


def _impact_regional_receipt(
    root: Path,
    record: Mapping[str, Any],
) -> list[Path]:
    fixtures = (
        {
            "id": "restore-overlap-base-only",
            "seed": 17,
            "steps": 3,
            "base_only_steps": 1,
            "denoise": 1.0,
            "overlap_factor": 2,
            "restore_latent": True,
            "base_cfg": 6.5,
            "base_sampler_name": "euler",
            "base_scheduler": "simple",
            "region_cfg": 4.0,
            "region_sampler_name": "heun",
            "region_scheduler": "normal",
            "mask": torch.tensor(
                [
                    [0.0, 0.2, 0.0, 0.0, 0.8, 1.0],
                    [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                    [0.0, 0.0, 0.4, 1.0, 1.0, 0.0],
                    [0.0, 0.1, 1.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "latentShape": (1, 4, 2, 3),
        },
        {
            "id": "no-restore-no-overlap-partial-denoise",
            "seed": 29,
            "steps": 2,
            "base_only_steps": 0,
            "denoise": 0.5,
            "overlap_factor": 0,
            "restore_latent": False,
            "base_cfg": 3.0,
            "base_sampler_name": "euler",
            "base_scheduler": "simple",
            "region_cfg": 7.0,
            "region_sampler_name": "heun",
            "region_scheduler": "normal",
            "mask": torch.tensor(
                [[0.0, 0.25, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.5, 0.0]],
                dtype=torch.float32,
            ),
            "latentShape": (1, 4, 3, 5),
        },
    )
    reference_cases = []
    native_cases = []
    receipt_parameters = []
    for fixture in fixtures:
        parameters = {
            key: value for key, value in fixture.items() if key not in {"mask", "latentShape"}
        }
        latent_shape = cast("tuple[int, ...]", fixture["latentShape"])
        samples = {"samples": torch.zeros(latent_shape), "metadata": "preserved"}
        mask = cast("torch.Tensor", fixture["mask"])
        reference_cases.append(
            {"id": fixture["id"], **_impact_source_case(parameters, mask, samples)}
        )
        native_cases.append(
            {"id": fixture["id"], **_impact_native_case(record, parameters, mask, samples)}
        )
        receipt_parameters.append(
            {
                **parameters,
                "mask": _array_value(mask),
                "latentShape": list(latent_shape),
            }
        )
    return _write_mapping_receipt(
        root,
        record,
        slug="regional-sampler-single-region",
        parameters={
            "sourceCommit": IMPACT_PACK_COMMIT,
            "sourceFiles": {
                "modules/impact/special_samplers.py": IMPACT_PACK_SOURCE_SHA256,
                "modules/impact/core.py": IMPACT_PACK_CORE_SHA256,
            },
            "cases": receipt_parameters,
            "scope": (
                "one-region provider orchestration, denoise interval selection, noise admission, "
                "source mask growth, restoration composition, and result metadata; controlled "
                "provider schedules isolate orchestration from separately receipted solver numerics"
            ),
        },
        reference={"format": "dinkster-impact-regional-trace/1", "cases": reference_cases},
        native={"format": "dinkster-impact-regional-trace/1", "cases": native_cases},
    )


def _inspire_scheduled_cfg_class() -> type[Any]:
    source = _verified_source("INSPIRE_PACK_SOURCE", INSPIRE_PACK_SOURCE_SHA256)
    functions = _load_source_functions(
        source,
        "inspire/sampler_nodes.py",
        (
            "exponential_interpolation",
            "logarithmic_interpolation",
            "cosine_interpolation",
        ),
        {"math": math},
    )

    class ReferenceCFGGuider:
        def __init__(self, model_patcher: object) -> None:
            self.model_patcher = model_patcher
            self.cfg = 1.0

        def predict_noise(
            self,
            _x: object,
            _timestep: object,
            _model_options: object = None,
            _seed: object = None,
        ) -> torch.Tensor:
            cond, uncond, trace = cast(
                "tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]",
                self.model_patcher,
            )
            lanes = ["positive"]
            if not math.isclose(self.cfg, 1.0):
                lanes.append("negative")
            cast("list[dict[str, object]]", trace).append({"cfg": self.cfg, "lanes": lanes})
            return cond if len(lanes) == 1 else uncond + (cond - uncond) * self.cfg

    return _load_source_classes(
        source,
        "inspire/sampler_nodes.py",
        ("Guider_scheduled",),
        {"CFGGuider": ReferenceCFGGuider, "math": math, **functions},
    )["Guider_scheduled"]


def _scheduled_cfg_tensors(family: str) -> tuple[torch.Tensor, torch.Tensor]:
    channels = {"sd-eps": 4, "flux-flow": 16}[family]
    values = torch.arange(channels * 4, dtype=torch.float32).reshape(1, channels, 2, 2)
    return values / 11.0, -values / 7.0 - 0.25


def _inspire_scheduled_cfg_source_case(
    source_class: type[Any], parameters: Mapping[str, object]
) -> dict[str, object]:
    cond, uncond = _scheduled_cfg_tensors(cast("str", parameters["family"]))
    sigmas = torch.tensor(parameters["sigmas"], dtype=torch.float32)
    evaluations = torch.tensor(parameters["evaluations"], dtype=torch.float32)
    trace: list[dict[str, object]] = []
    guider = source_class(
        (cond, uncond, trace),
        sigmas,
        parameters["from_cfg"],
        parameters["to_cfg"],
        parameters["schedule"],
    )
    outputs = [guider.predict_noise(None, sigma.reshape(1), {}, None) for sigma in evaluations]
    return {
        "trace": trace,
        "outputs": [_array_value(value) for value in outputs],
    }


def _inspire_scheduled_cfg_native_case(
    record: Mapping[str, Any], parameters: Mapping[str, object]
) -> dict[str, object]:
    cond, uncond = _scheduled_cfg_tensors(cast("str", parameters["family"]))
    sigma_tensor = torch.tensor(parameters["sigmas"], dtype=torch.float32)
    sigmas = native_arm._CustomSigmasValue(  # pyright: ignore[reportPrivateUsage]
        tuple(float(value) for value in sigma_tensor.tolist())
    )
    source_inputs = {
        "model": object(),
        "positive": object(),
        "negative": object(),
        "sigmas": sigmas,
        "from_cfg": parameters["from_cfg"],
        "to_cfg": parameters["to_cfg"],
        "schedule": parameters["schedule"],
    }
    mapped = {
        name: _mapped_input(record, name, source_inputs)
        for name in cast("Mapping[str, object]", _mapping_case(record)["inputs"])
    }
    result = native_arm.GenerationScheduledCFGGuider.execute(**mapped)
    guider = cast("Any", result["guider"])
    contribution = guider.transforms[0][1]
    scales: list[float] = []
    descriptor = contribution.scale[0]

    def record_scale(context: GuidancePlanContext[torch.Tensor]) -> float:
        value = descriptor.transform(context)
        scales.append(value)
        return value

    contribution = replace(
        contribution,
        scale=(replace(descriptor, transform=record_scale),),
    )
    executor = GuidanceExecutor(GuidanceRegistry((("scheduled-cfg-receipt", contribution),)))
    conditions = (
        GuidanceCondition("positive", GuidanceRole.CONDITIONAL, Conditioning(torch.empty(0))),
        GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, Conditioning(torch.empty(0))),
    )
    token = CancellationToken(lambda: False)
    trace: list[dict[str, object]] = []
    outputs = []
    evaluations = torch.tensor(parameters["evaluations"], dtype=torch.float32)
    for index, sigma in enumerate(evaluations):
        execution = SamplingExecutionContext(
            sigmas.values,
            index,
            index,
            float(sigma),
            1,
            token,
            ProgressScope(token),
            {},
        )
        context = GuidancePlanContext(
            cond,
            sigma.reshape(1),
            guider.cfg,
            conditions,
            False,
            execution,
        )

        def evaluate(
            request: GuidanceEvaluationRequest[torch.Tensor],
        ) -> GuidancePredictions[torch.Tensor]:
            trace.append(
                {
                    "cfg": scales[-1],
                    "lanes": [lane.id for lane in request.plan.lanes],
                }
            )
            values = {"positive": cond, "negative": uncond}
            return GuidancePredictions(
                tuple(
                    GuidancePrediction(
                        lane.id,
                        values[lane.id],
                        GuidancePredictionSource.MODEL,
                    )
                    for lane in request.plan.lanes
                )
            )

        output = executor.execute(context, evaluate).denoised
        outputs.append(output)
    return {
        "trace": trace,
        "outputs": [_array_value(value) for value in outputs],
    }


def _inspire_scheduled_cfg_receipt(root: Path, record: Mapping[str, Any]) -> list[Path]:
    source_class = _inspire_scheduled_cfg_class()
    schedules = (
        ("linear", 6.5, 1.0),
        ("log", 7.0, 2.0),
        ("cos", 5.5, 1.5),
        ("exp", 0.0, 4.0),
        ("exp", 4.0, 0.0),
        ("exp", 3.0, 6.0),
    )
    parameters = []
    for family in ("sd-eps", "flux-flow"):
        for schedule, from_cfg, to_cfg in schedules:
            parameters.append(
                {
                    "id": f"{family}-{schedule}-{from_cfg:g}-{to_cfg:g}",
                    "family": family,
                    "schedule": schedule,
                    "from_cfg": from_cfg,
                    "to_cfg": to_cfg,
                    "sigmas": [1.0, 0.7, 0.25, 0.0],
                    "evaluations": [1.0, 0.85, 0.25, 0.0],
                }
            )
    reference = [
        {"id": case["id"], **_inspire_scheduled_cfg_source_case(source_class, case)}
        for case in parameters
    ]
    native = [
        {"id": case["id"], **_inspire_scheduled_cfg_native_case(record, case)}
        for case in parameters
    ]
    return _write_mapping_receipt(
        root,
        record,
        slug="scheduled-cfg-guider",
        parameters={
            "sourceCommit": INSPIRE_PACK_COMMIT,
            "sourceFiles": {"inspire/sampler_nodes.py": INSPIRE_PACK_SOURCE_SHA256},
            "cases": parameters,
            "scope": (
                "all source interpolation branches, exact-sigma and sequential fallback lookup, "
                "CFG-one lane admission, and guided reduction on SD/EPS and Flux/flow tensors"
            ),
        },
        reference={"format": "dinkster-inspire-scheduled-cfg-trace/1", "cases": reference},
        native={"format": "dinkster-inspire-scheduled-cfg-trace/1", "cases": native},
    )


def _translate_group_case(
    record: Mapping[str, Any],
    source_inputs: Mapping[str, object],
) -> dict[str, object]:
    replacement = cast("Mapping[str, Any]", record["replacement"])
    selected: Mapping[str, Any] | None = None
    fallback: Mapping[str, Any] | None = None
    for case in cast("list[Mapping[str, Any]]", replacement["cases"]):
        when = cast("Mapping[str, Any] | None", case.get("when"))
        if when is None:
            fallback = case
        elif when.get("kind") != "valueEquals":
            raise RuntimeError(f"unsupported group predicate in {record['id']}")
        elif source_inputs[cast("str", when["input"])] == when["value"]:
            selected = case
            break
    selected = selected or fallback
    if selected is None:
        raise RuntimeError(f"no group replacement case matches {record['id']}")

    translated: dict[str, object] = {}
    for name, raw in cast("Mapping[str, Mapping[str, Any]]", selected["inputs"]).items():
        kind = raw.get("kind")
        if kind == "constant":
            translated[name] = raw["value"]
        elif kind == "copy":
            translated[name] = source_inputs[cast("str", raw["input"])]
        else:
            raise RuntimeError(f"unsupported group input mapping in {record['id']}: {name}")
    return {
        "inputs": translated,
        "outputs": selected["outputs"],
        "target": selected["to"],
    }


def _group_receipts(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> list[Path]:
    image = "fixture:image-batch"
    video = "fixture:video-batch"
    mask = "fixture:initial-mask"
    specifications: tuple[
        tuple[
            str,
            tuple[tuple[str, dict[str, object], dict[str, object]], ...],
            dict[str, str],
        ],
        ...,
    ] = (
        (
            "rtdetr-detect-fp16",
            (
                (
                    "all-classes",
                    {"image": image, "threshold": 0.25, "class_name": "all", "max_detections": 5},
                    {
                        "image": image,
                        "prompt": "",
                        "min_score": 0.25,
                        "max_results": 5,
                        "result_limit_mode": "slice-stop",
                        "provider": "dinkster-vision-rtdetr",
                    },
                ),
                (
                    "one-class",
                    {
                        "image": image,
                        "threshold": 0.4,
                        "class_name": "person",
                        "max_detections": 3,
                    },
                    {
                        "image": image,
                        "prompt": "person",
                        "min_score": 0.4,
                        "max_results": 3,
                        "result_limit_mode": "slice-stop",
                        "provider": "dinkster-vision-rtdetr",
                    },
                ),
            ),
            {"detections": "bboxes"},
        ),
        (
            "sam3-text-detection",
            (
                (
                    "literal-text",
                    {"image": image, "text": "person", "threshold": 0.35},
                    {
                        "image": image,
                        "prompt": "person",
                        "prompt_mode": "literal",
                        "min_score": 0.35,
                        "max_results": 1,
                        "provider": "dinkster-vision-sam31",
                    },
                ),
            ),
            {"detections": "bboxes"},
        ),
        (
            "sam3-video-track-initial-mask",
            (
                (
                    "initial-mask",
                    {"images": video, "initial_mask": mask},
                    {
                        "image": video,
                        "initial_masks": mask,
                        "provider": "dinkster-vision-sam31",
                    },
                ),
            ),
            {"combined": "masks"},
        ),
        (
            "remove-background-birefnet",
            (
                (
                    "image",
                    {"image": image},
                    {"image": image, "provider": "dinkster-vision-birefnet"},
                ),
            ),
            {"mask": "mask"},
        ),
    )

    outputs: list[Path] = []
    for source_name, cases, expected_outputs in specifications:
        record = records[source_name]
        reference_cases = []
        native_cases = []
        for case_id, source_inputs, expected_inputs in cases:
            reference_cases.append(
                {
                    "id": case_id,
                    "translation": {
                        "inputs": expected_inputs,
                        "outputs": expected_outputs,
                        "target": record["carrier"],
                    },
                }
            )
            native_cases.append(
                {
                    "id": case_id,
                    "translation": _translate_group_case(record, source_inputs),
                }
            )
        outputs.extend(
            _write_mapping_receipt(
                root,
                record,
                slug=source_name,
                parameters={
                    "cases": [
                        {"id": case_id, "sourceInputs": source_inputs}
                        for case_id, source_inputs, _expected_inputs in cases
                    ],
                    "scope": "group input, output, provider, and parameter translation",
                    "sourceCommit": VISION_REFERENCE_COMMIT,
                },
                reference={"cases": reference_cases},
                native={"cases": native_cases},
            )
        )
    return outputs


def _trellis2_official_receipts(
    root: Path,
    comfy_root: Path,
    workflow_templates_root: Path,
) -> list[Path]:
    resolved_comfy = (
        _git(comfy_root, "rev-parse", f"{TRELLIS2_COMFYUI_COMMIT}^{{commit}}").decode().strip()
    )
    if resolved_comfy != TRELLIS2_COMFYUI_COMMIT:
        raise RuntimeError(f"ComfyUI does not contain {TRELLIS2_COMFYUI_COMMIT}")
    resolved_workflow = (
        _git(
            workflow_templates_root,
            "rev-parse",
            f"{TRELLIS2_WORKFLOW_COMMIT}^{{commit}}",
        )
        .decode()
        .strip()
    )
    if resolved_workflow != TRELLIS2_WORKFLOW_COMMIT:
        raise RuntimeError(f"workflow_templates does not contain {TRELLIS2_WORKFLOW_COMMIT}")

    evidence_bytes = TRELLIS2_EVIDENCE.read_bytes()
    evidence = cast("dict[str, Any]", json.loads(evidence_bytes))
    if evidence.get("format") != "dinkster-trellis2-official-workflow-evidence/1":
        raise RuntimeError("TRELLIS2 official workflow evidence has an unknown format")
    references = cast("Mapping[str, Mapping[str, object]]", evidence.get("references"))
    if (
        references.get("comfyUI", {}).get("commit") != TRELLIS2_COMFYUI_COMMIT
        or references.get("workflowTemplates", {}).get("commit") != TRELLIS2_WORKFLOW_COMMIT
    ):
        raise RuntimeError("TRELLIS2 official workflow evidence has stale source revisions")

    workflow_bytes = _git(
        workflow_templates_root,
        "show",
        f"{TRELLIS2_WORKFLOW_COMMIT}:{TRELLIS2_WORKFLOW_PATH}",
    )
    workflow_descriptor = cast("Mapping[str, object]", evidence.get("sourceWorkflow"))
    if workflow_descriptor != {
        "path": TRELLIS2_WORKFLOW_PATH,
        "bytes": len(workflow_bytes),
        "sha256": hashlib.sha256(workflow_bytes).hexdigest(),
    }:
        raise RuntimeError("TRELLIS2 official workflow evidence has a stale source workflow")
    workflow = cast("Mapping[str, Any]", json.loads(workflow_bytes))
    source_nodes = {
        f"n{node['id']}": node
        for node in cast("Sequence[Mapping[str, object]]", workflow.get("nodes"))
    }

    bindings = cast("Mapping[str, Sequence[str]]", evidence.get("bindings"))
    records = {
        name: record
        for name, record in _mapping_records().items()
        if name in bindings and name not in TRELLIS2_BASELINE_RECORDS
    }
    if bindings.keys() != records.keys():
        missing = sorted(records.keys() - bindings.keys())
        extra = sorted(bindings.keys() - records.keys())
        raise RuntimeError(
            "TRELLIS2 official workflow evidence binding mismatch: "
            f"missing={missing}, extra={extra}"
        )
    runs = {
        cast("str", run["variant"]): run
        for run in cast("Sequence[Mapping[str, object]]", evidence.get("nativeRuns"))
    }
    if runs.keys() != {"pixal3d", "trellis2"}:
        raise RuntimeError("TRELLIS2 official workflow evidence must cover both variants")
    for variant, run in runs.items():
        source_document = run.get("sourceDocument")
        if (
            run.get("state") != "completed"
            or run.get("skipped") != 0
            or type(run.get("executed")) is not int
            or cast("int", run["executed"]) <= 0
            or type(run.get("cached")) is not int
            or not isinstance(source_document, str)
            or re.fullmatch(r"blake3:[0-9a-f]{64}", source_document) is None
            or not isinstance(run.get("documentSha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", cast("str", run["documentSha256"])) is None
            or type(run.get("documentBytes")) is not int
            or cast("int", run["documentBytes"]) <= 0
        ):
            raise RuntimeError(f"invalid TRELLIS2 official workflow run evidence: {variant}")

    parity = {
        cast("str", item["variant"]): item
        for item in cast("Sequence[Mapping[str, Any]]", evidence.get("numericalParity"))
    }
    if parity.keys() != runs.keys():
        raise RuntimeError("TRELLIS2 numerical parity evidence must cover both variants")
    for variant, item in parity.items():
        metrics = cast("Mapping[str, float]", item.get("metrics"))
        if (
            metrics.get("structureCosine", 0.0) < 0.999
            or metrics.get("occupancyIou", 0.0) < 0.98
            or metrics.get("normalizedChamferSquared", 1.0) > 1.0e-5
            or metrics.get("pbrCosine", 0.0) < 0.93
        ):
            raise RuntimeError(f"TRELLIS2 numerical parity evidence failed: {variant}")

    outputs: list[Path] = []
    evidence_descriptor = {
        "repository": "Kosinkadink/dinkster-evidence",
        "path": TRELLIS2_EVIDENCE.relative_to(INFERENCE_PARITY_RECORDS.parents[1]).as_posix(),
        "sha256": f"sha256:{hashlib.sha256(evidence_bytes).hexdigest()}",
    }
    for source_class, record in sorted(records.items()):
        binding = bindings[source_class]
        if len(binding) != 4:
            raise RuntimeError(f"invalid TRELLIS2 workflow binding: {source_class}")
        source_node_id, target_type, variant, disposition = binding
        source_node = source_nodes.get(source_node_id)
        run = runs.get(variant)
        if (
            source_node is None
            or source_node.get("type") != source_class
            or target_type != record["carrier"]
            or run is None
            or disposition not in {"executed", "cached"}
        ):
            raise RuntimeError(f"stale TRELLIS2 workflow binding: {source_class}")
        normalized = {
            "sourceClass": source_class,
            "sourceNodeId": source_node_id,
            "semanticNode": target_type,
            "state": "completed",
        }
        slug = "trellis2-" + re.sub(r"[^a-z0-9]+", "-", source_class.casefold()).strip("-")
        outputs.extend(
            _write_mapping_receipt(
                root,
                record,
                slug=slug,
                parameters={
                    "binding": {
                        "disposition": disposition,
                        "variant": variant,
                    },
                    "confidenceEvidence": cast("Mapping[str, object]", record["confidence"])[
                        "evidence"
                    ],
                    "evidence": evidence_descriptor,
                    "input": evidence["input"],
                    "nativeRun": run,
                    "numericalParity": list(parity.values()),
                    "sourceWorkflow": workflow_descriptor,
                },
                reference=normalized,
                native=dict(normalized),
            )
        )
    return outputs


def _write_numeric_array_receipt(
    root: Path,
    record: Mapping[str, Any],
    *,
    slug: str,
    parameters: dict[str, object],
    reference: np.ndarray,
    native: np.ndarray,
    data_kind: str,
) -> list[Path]:
    source = cast("Mapping[str, object]", record["source"])
    source_pack = cast("str", source["pack"])
    artifact_directory = root / "artifacts" / source_pack
    reference_path = artifact_directory / f"{slug}.reference.npy"
    native_path = artifact_directory / f"{slug}.native.npy"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    np.save(reference_path, np.ascontiguousarray(reference), allow_pickle=False)
    np.save(native_path, np.ascontiguousarray(native), allow_pickle=False)
    confidence = cast("Mapping[str, object]", record["confidence"])
    tolerances = cast("list[dict[str, object]]", confidence["tolerances"])
    receipt_path = root / source_pack / f"{slug}.receipt.json"
    receipt = create_receipt(
        case_id=f"{source_pack}/{slug}/source-parity",
        mapping=_receipt_mapping(record),
        parameters={"mappingDigest": _record_digest(record), **parameters},
        comparison={
            "comparator": "numeric-array/1",
            "dataKind": data_kind,
            "tolerances": tolerances,
        },
        artifact_root=root,
        reference_path=reference_path.relative_to(root).as_posix(),
        native_path=native_path.relative_to(root).as_posix(),
    )
    if receipt["pass"] is not True:
        raise RuntimeError(f"source parity failed for {record['id']}")
    write_receipt(receipt_path, receipt)
    return [reference_path, native_path, receipt_path]


_CONTROL_AUX_CASES: dict[str, tuple[str, ...]] = {
    "AnimeLineArtPreprocessor": ("lineart-anime",),
    "AnyLineArtPreprocessor_aux": (
        "anyline-standard",
        "anyline-realistic",
        "anyline-anime",
        "anyline-manga",
    ),
    "FakeScribblePreprocessor": ("hed-scribble", "hed-scribble-unsafe"),
    "HEDPreprocessor": ("hed-safe", "hed-soft"),
    "LineArtPreprocessor": ("lineart-realistic", "lineart-realistic-coarse"),
    "M-LSDPreprocessor": ("mlsd",),
    "Manga2Anime_LineArt_Preprocessor": ("lineart-manga",),
    "TEEDPreprocessor": ("teed", "teed-unquantized"),
}
_CONTROL_AUX_TARGET_CLASSES = {
    "dinkster.preprocess.anyline": "AnyLinePreprocessor",
    "dinkster.preprocess.lineart_anime": "AnimeLineartPreprocessor",
    "dinkster.preprocess.lineart_manga": "MangaLineartPreprocessor",
    "dinkster.preprocess.lineart_realistic": "RealisticLineartPreprocessor",
    "dinkster.preprocess.mlsd": "MLSDPreprocessor",
    "dinkster.preprocess.model_edges": "ModelEdgePreprocessor",
    "dinkster.preprocess.teed": "TEEDPreprocessor",
}


def _native_input_defaults(node: type[Any]) -> dict[str, object]:
    return {
        name: parameter.default
        for name, parameter in inspect.signature(node.execute).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _case_table(
    assignment: ast.AnnAssign,
    *,
    string_node: bool,
) -> dict[str, tuple[str, dict[str, object]]]:
    if not isinstance(assignment.value, ast.Dict):
        raise RuntimeError("control auxiliary case table must be a dictionary")
    result: dict[str, tuple[str, dict[str, object]]] = {}
    for key_node, value_node in zip(assignment.value.keys, assignment.value.values, strict=True):
        if key_node is None or not isinstance(value_node, ast.Tuple) or len(value_node.elts) != 2:
            raise RuntimeError("control auxiliary case table has an unsupported entry")
        key = ast.literal_eval(key_node)
        node = ast.literal_eval(value_node.elts[0]) if string_node else value_node.elts[0]
        parameters_node = value_node.elts[1]
        if not isinstance(parameters_node, ast.Dict):
            raise RuntimeError("control auxiliary case parameters must be a dictionary")
        parameters = {
            ast.literal_eval(name): (
                "dinkster-vision-hed"
                if isinstance(value, ast.Name) and value.id == "provider"
                else ast.literal_eval(value)
            )
            for name, value in zip(parameters_node.keys, parameters_node.values, strict=True)
            if name is not None
        }
        if (
            not isinstance(key, str)
            or not isinstance(parameters, dict)
            or not all(isinstance(name, str) for name in parameters)
            or (not string_node and not isinstance(node, ast.Name))
            or (string_node and not isinstance(node, str))
        ):
            raise RuntimeError("control auxiliary case table is not canonical")
        result[key] = (
            cast("str", node) if string_node else cast("ast.Name", node).id,
            cast("dict[str, object]", parameters),
        )
    return result


def _control_aux_runner_cases() -> tuple[
    dict[str, tuple[str, dict[str, object]]],
    dict[str, tuple[str, dict[str, object]]],
]:
    runner = REPO / "tools" / "run_control_aux_line_edge_parity.py"
    encoded = runner.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != CONTROL_AUX_RUNNER_SHA256:
        raise RuntimeError("control auxiliary evidence runner differs from its pinned source")
    tree = ast.parse(encoded, filename=os.fspath(runner))
    reference_assignment = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "REFERENCE_CASES"
        ),
        None,
    )
    setup = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_setup_dinkster"
        ),
        None,
    )
    native_assignment = (
        next(
            (
                node
                for node in ast.walk(setup)
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "cases"
            ),
            None,
        )
        if setup is not None
        else None
    )
    if reference_assignment is None or native_assignment is None:
        raise RuntimeError("control auxiliary evidence runner is missing its case tables")
    return (
        _case_table(reference_assignment, string_node=True),
        _case_table(native_assignment, string_node=False),
    )


def _project_control_aux_inputs(
    record: Mapping[str, Any],
    source_parameters: Mapping[str, object],
) -> tuple[int, dict[str, object]]:
    replacement = cast("Mapping[str, Any]", record["replacement"])
    cases = cast("list[Mapping[str, Any]]", replacement["cases"])
    selected_index: int | None = None
    selected: Mapping[str, Any] | None = None
    for index, candidate in enumerate(cases):
        when = cast("Mapping[str, Any] | None", candidate.get("when"))
        if when is not None and when.get("kind") != "valueEquals":
            raise RuntimeError(f"unsupported control auxiliary mapping condition: {when}")
        if when is None or source_parameters.get(cast("str", when["input"])) == when.get("value"):
            selected_index = index
            selected = candidate
            break
    if selected is None or selected_index is None:
        raise RuntimeError(f"control auxiliary mapping has no case for {source_parameters}")
    if selected.get("to") != record["carrier"]:
        raise RuntimeError(f"control auxiliary mapping target differs from carrier: {record['id']}")
    projected: dict[str, object] = {}
    for target, source in cast("Mapping[str, Mapping[str, Any]]", selected["inputs"]).items():
        if target == "image":
            if source != {"kind": "copy", "input": "image"}:
                raise RuntimeError("control auxiliary image mapping must copy the source image")
            continue
        kind = source.get("kind")
        if kind == "constant":
            projected[target] = source["value"]
            continue
        source_name = cast("str", source.get("input"))
        if source_name not in source_parameters:
            raise RuntimeError(f"control auxiliary mapping reads unavailable input: {source_name}")
        if kind == "copy":
            projected[target] = source_parameters[source_name]
        elif kind == "value":
            value = source_parameters[source_name]
            transform = cast("Mapping[str, Any]", source.get("transform"))
            if transform.get("kind") != "enumRename":
                raise RuntimeError(f"unsupported control auxiliary transform: {transform}")
            projected[target] = cast("Mapping[object, object]", transform["map"])[value]
        else:
            raise RuntimeError(f"unsupported control auxiliary mapping input: {source}")
    return selected_index, projected


def _control_aux_checksums() -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in (CONTROL_AUX_EVIDENCE / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    return checksums


def _control_aux_result(
    implementation: str,
    case: str,
    checksums: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object], Mapping[str, object]]:
    prefix = "reference" if implementation == "reference" else "dinkster"
    relative = f"normal-512/{prefix}-{case}.json"
    path = CONTROL_AUX_EVIDENCE / relative
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if checksums.get(relative) != digest:
        raise RuntimeError(f"control auxiliary evidence checksum mismatch: {relative}")
    document = cast("dict[str, Any]", json.loads(encoded))
    source = cast("Mapping[str, object]", document.get("source"))
    environment = cast("Mapping[str, object]", document.get("environment"))
    expected_source = {
        "comfyuiCommit": CONTROL_AUX_COMFYUI_COMMIT,
        "controlnetAuxCommit": CONTROL_AUX_COMMIT,
        "inputSha256": CONTROL_AUX_INPUT_SHA256,
    }
    if (
        document.get("implementation") != implementation
        or document.get("case") != case
        or document.get("resolution") != CONTROL_AUX_RESOLUTION
        or document.get("error") is not None
        or document.get("outputBitStable") is not True
        or environment.get("gpuUuid") != CONTROL_AUX_GPU_UUID
        or any(source.get(key) != value for key, value in expected_source.items())
    ):
        raise RuntimeError(f"invalid control auxiliary evidence: {relative}")
    sample_hashes = cast("list[str]", document.get("outputSampleFloat32Sha256"))
    output_hash = cast("str", document.get("outputFloat32Sha256"))
    if not sample_hashes or set(sample_hashes) != {output_hash}:
        raise RuntimeError(f"unstable control auxiliary evidence: {relative}")
    result = {
        "case": case,
        "outputDtype": document.get("outputDtype"),
        "outputFloat32Sha256": output_hash,
        "outputShape": document.get("outputShape"),
        "repeatMaxAbsDiff": document.get("repeatMaxAbsDiff"),
    }
    evidence = {"path": relative, "sha256": f"sha256:{digest}"}
    return result, evidence, source


def _control_aux_receipts(root: Path) -> list[Path]:
    records = _mapping_records("comfyui_controlnet_aux")
    if not _CONTROL_AUX_CASES.keys() <= records.keys():
        missing = sorted(_CONTROL_AUX_CASES.keys() - records.keys())
        raise RuntimeError(f"missing maintained control auxiliary mappings: {missing}")
    checksums = _control_aux_checksums()
    reference_cases, native_cases = _control_aux_runner_cases()
    source_defaults = _source_input_defaults("comfyui_controlnet_aux")
    from dinkster_nodes_vision.hed import nodes as control_aux_nodes

    outputs: list[Path] = []
    for node_class, cases in _CONTROL_AUX_CASES.items():
        record = records[node_class]
        reference_results: list[dict[str, object]] = []
        native_results: list[dict[str, object]] = []
        evidence: list[dict[str, object]] = []
        bindings: list[dict[str, object]] = []
        model_artifacts: Mapping[str, object] | None = None
        selected_mapping_cases: set[int] = set()
        for case in cases:
            reference_node, reference_parameters = reference_cases[case]
            native_class, native_parameters = native_cases[case]
            expected_native_class = _CONTROL_AUX_TARGET_CLASSES.get(cast("str", record["carrier"]))
            if reference_node != node_class or native_class != expected_native_class:
                raise RuntimeError(f"control auxiliary evidence target mismatch: {case}")
            effective_source_parameters = {
                **source_defaults[node_class],
                **reference_parameters,
                "resolution": CONTROL_AUX_RESOLUTION,
            }
            selected_index, projected_parameters = _project_control_aux_inputs(
                record, effective_source_parameters
            )
            node = cast("type[Any]", getattr(control_aux_nodes, native_class))
            expected_parameters = {
                **_native_input_defaults(node),
                **native_parameters,
                "resolution": CONTROL_AUX_RESOLUTION,
            }
            if projected_parameters != expected_parameters:
                raise RuntimeError(
                    f"control auxiliary evidence parameters do not match mapping: {case}"
                )
            selected_mapping_cases.add(selected_index)
            bindings.append(
                {
                    "case": case,
                    "mappingCase": selected_index,
                    "sourceInputs": effective_source_parameters,
                    "sourceNode": reference_node,
                    "targetInputs": projected_parameters,
                    "targetNode": cast("str", record["carrier"]),
                }
            )
            reference, reference_evidence, reference_source = _control_aux_result(
                "reference", case, checksums
            )
            native, native_evidence, native_source = _control_aux_result(
                "dinkster", case, checksums
            )
            if reference_source.get("artifacts") != native_source.get("artifacts"):
                raise RuntimeError(f"control auxiliary artifacts differ between arms: {case}")
            if model_artifacts is None:
                model_artifacts = cast("Mapping[str, object]", reference_source.get("artifacts"))
            elif model_artifacts != reference_source.get("artifacts"):
                raise RuntimeError(f"control auxiliary artifacts differ between cases: {case}")
            reference_results.append(reference)
            native_results.append(native)
            evidence.extend((reference_evidence, native_evidence))
        replacement = cast("Mapping[str, Any]", record["replacement"])
        mapping_case_count = len(cast("list[object]", replacement["cases"]))
        if selected_mapping_cases != set(range(mapping_case_count)):
            raise RuntimeError(f"control auxiliary evidence misses mapping cases: {node_class}")
        slug = re.sub(r"[^a-z0-9]+", "-", node_class.casefold()).strip("-")
        outputs.extend(
            _write_mapping_receipt(
                root,
                record,
                slug=slug,
                parameters={
                    "bindings": bindings,
                    "cases": list(cases),
                    "comfyUIRevision": CONTROL_AUX_COMFYUI_COMMIT,
                    "evidence": evidence,
                    "evidenceRepository": "Kosinkadink/dinkster-evidence",
                    "evidenceRoot": CONTROL_AUX_EVIDENCE.relative_to(
                        INFERENCE_PARITY_RECORDS.parents[1]
                    ).as_posix(),
                    "gpuUuid": CONTROL_AUX_GPU_UUID,
                    "inputSha256": CONTROL_AUX_INPUT_SHA256,
                    "modelArtifacts": dict(model_artifacts or {}),
                    "runnerSha256": f"sha256:{CONTROL_AUX_RUNNER_SHA256}",
                },
                reference={"results": reference_results},
                native={"results": native_results},
            )
        )
    return outputs


_TRANSFORM_DEFAULTS: dict[str, object] = {
    "value": "",
    "replacement": "",
    "start": 0,
    "end": (1 << 53) - 1,
    "count": 0,
    "fill": " ",
    "unit": "characters",
    "side": "start",
}


def _transform(text: str, operation: str, **updates: object) -> object:
    arguments = {**_TRANSFORM_DEFAULTS, "text": text, "operation": operation, **updates}
    return StringTransform.execute(**arguments)["text"]


def _string_test(text: str, query: str, operation: str, case_sensitive: bool) -> object:
    return StringTest.execute(
        text=text,
        query=query,
        operation=operation,
        case_mode="sensitive" if case_sensitive else "lower_both",
    )["result"]


def _regex(
    *,
    text: str,
    pattern: str,
    operation: str,
    replacement: str = "",
    case_insensitive: bool,
    multiline: bool,
    dotall: bool,
    count: int = 0,
) -> Mapping[str, object]:
    return StringRegex.execute(
        text=text,
        pattern=pattern,
        operation=operation,
        replacement=replacement,
        group_index=1,
        count=count,
        case_mode="unicode_ignorecase" if case_insensitive else "sensitive",
        multiline=multiline,
        dotall=dotall,
    )


def _case_payload(
    source_class: type[Any],
    cases: Sequence[tuple[str, dict[str, object]]],
    native: Callable[[dict[str, object]], object],
) -> tuple[dict[str, object], dict[str, object]]:
    reference_cases = []
    native_cases = []
    for case_id, inputs in cases:
        reference_cases.append({"id": case_id, "output": source_class.execute(**inputs)[0]})
        native_cases.append({"id": case_id, "output": native(inputs)})
    return {"cases": reference_cases}, {"cases": native_cases}


def _string_receipts(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, type[Any]],
) -> list[Path]:
    outputs: list[Path] = []
    specifications: list[
        tuple[
            str,
            str,
            Sequence[tuple[str, dict[str, object]]],
            Callable[[dict[str, object]], object],
        ]
    ] = [
        (
            "StringConcatenate",
            "string-concatenate",
            (
                ("empty-delimiter", {"string_a": "alpha", "string_b": "beta", "delimiter": ""}),
                (
                    "unicode",
                    {"string_a": "caf\u00e9", "string_b": "\u6771\u4eac", "delimiter": " | "},
                ),
            ),
            lambda value: ConcatStrings.execute(
                a=cast("str", value["string_a"]),
                b=cast("str", value["string_b"]),
                separator=cast("str", value["delimiter"]),
            )["text"],
        ),
        (
            "StringSubstring",
            "string-substring",
            (
                ("middle", {"string": "012345", "start": 1, "end": 5}),
                ("negative", {"string": "abcdef", "start": -4, "end": -1}),
            ),
            lambda value: _transform(
                cast("str", value["string"]),
                "slice",
                start=value["start"],
                end=value["end"],
            ),
        ),
        (
            "StringLength",
            "string-length",
            (
                ("empty", {"string": ""}),
                ("unicode-codepoints", {"string": "A\u030a\U0001f642"}),
            ),
            lambda value: StringLength.execute(text=cast("str", value["string"]))["length"],
        ),
        (
            "CaseConverter",
            "case-converter",
            tuple(
                (mode.lower().replace(" ", "-"), {"string": "hELLO wORLD", "mode": mode})
                for mode in ("UPPERCASE", "lowercase", "Capitalize", "Title Case")
            ),
            lambda value: _transform(
                cast("str", value["string"]),
                {
                    "UPPERCASE": "upper",
                    "lowercase": "lower",
                    "Capitalize": "capitalize",
                    "Title Case": "title",
                }[cast("str", value["mode"])],
            ),
        ),
        (
            "StringTrim",
            "string-trim",
            tuple(
                (mode.lower(), {"string": " \t line \n", "mode": mode})
                for mode in ("Both", "Left", "Right")
            ),
            lambda value: _transform(
                cast("str", value["string"]),
                {"Both": "trim", "Left": "trim_left", "Right": "trim_right"}[
                    cast("str", value["mode"])
                ],
            ),
        ),
        (
            "StringReplace",
            "string-replace",
            (
                ("literal", {"string": "one two one", "find": "one", "replace": "1"}),
                ("empty-find", {"string": "ab", "find": "", "replace": "-"}),
            ),
            lambda value: _transform(
                cast("str", value["string"]),
                "replace_literal",
                value=value["find"],
                replacement=value["replace"],
            ),
        ),
        (
            "StringContains",
            "string-contains",
            (
                (
                    "sensitive",
                    {"string": "AlphaBeta", "substring": "Beta", "case_sensitive": True},
                ),
                (
                    "insensitive",
                    {"string": "AlphaBeta", "substring": "beta", "case_sensitive": False},
                ),
            ),
            lambda value: _string_test(
                cast("str", value["string"]),
                cast("str", value["substring"]),
                "contains",
                cast("bool", value["case_sensitive"]),
            ),
        ),
        (
            "StringCompare",
            "string-compare",
            (
                (
                    "equal-insensitive",
                    {
                        "string_a": "Alpha",
                        "string_b": "alpha",
                        "mode": "Equal",
                        "case_sensitive": False,
                    },
                ),
                (
                    "starts",
                    {
                        "string_a": "alphabet",
                        "string_b": "alpha",
                        "mode": "Starts With",
                        "case_sensitive": True,
                    },
                ),
                (
                    "ends-false",
                    {
                        "string_a": "alphabet",
                        "string_b": "ALPHA",
                        "mode": "Ends With",
                        "case_sensitive": False,
                    },
                ),
            ),
            lambda value: _string_test(
                cast("str", value["string_a"]),
                cast("str", value["string_b"]),
                {"Equal": "equals", "Starts With": "starts_with", "Ends With": "ends_with"}[
                    cast("str", value["mode"])
                ],
                cast("bool", value["case_sensitive"]),
            ),
        ),
        (
            "RegexMatch",
            "regex-match",
            (
                (
                    "insensitive",
                    {
                        "string": "Alpha",
                        "regex_pattern": "^alpha$",
                        "case_insensitive": True,
                        "multiline": False,
                        "dotall": False,
                    },
                ),
                (
                    "multiline",
                    {
                        "string": "x\nstart",
                        "regex_pattern": "^start$",
                        "case_insensitive": False,
                        "multiline": True,
                        "dotall": False,
                    },
                ),
                (
                    "invalid",
                    {
                        "string": "text",
                        "regex_pattern": "[",
                        "case_insensitive": False,
                        "multiline": False,
                        "dotall": False,
                    },
                ),
            ),
            lambda value: _regex(
                text=cast("str", value["string"]),
                pattern=cast("str", value["regex_pattern"]),
                operation="search",
                case_insensitive=cast("bool", value["case_insensitive"]),
                multiline=cast("bool", value["multiline"]),
                dotall=cast("bool", value["dotall"]),
            )["matched"],
        ),
        (
            "RegexReplace",
            "regex-replace",
            (
                (
                    "counted",
                    {
                        "string": "A a a",
                        "regex_pattern": "a",
                        "replace": "x",
                        "case_insensitive": True,
                        "multiline": False,
                        "dotall": False,
                        "count": 2,
                    },
                ),
                (
                    "dotall",
                    {
                        "string": "a\nb",
                        "regex_pattern": "a.*b",
                        "replace": "joined",
                        "case_insensitive": False,
                        "multiline": False,
                        "dotall": True,
                        "count": 0,
                    },
                ),
            ),
            lambda value: _regex(
                text=cast("str", value["string"]),
                pattern=cast("str", value["regex_pattern"]),
                operation="replace",
                replacement=cast("str", value["replace"]),
                case_insensitive=cast("bool", value["case_insensitive"]),
                multiline=cast("bool", value["multiline"]),
                dotall=cast("bool", value["dotall"]),
                count=cast("int", value["count"]),
            )["text"],
        ),
    ]
    for name, slug, cases, native in specifications:
        reference, native_payload = _case_payload(source[name], cases, native)
        outputs.extend(
            _write_mapping_receipt(
                root,
                records[name],
                slug=slug,
                parameters={
                    "cases": [{"id": case_id, "inputs": inputs} for case_id, inputs in cases]
                },
                reference=reference,
                native=native_payload,
            )
        )
    return outputs


def _switch_receipt(root: Path, record: Mapping[str, Any], source_class: type[Any]) -> list[Path]:
    cases = (
        ("false", {"switch": False, "on_false": "left", "on_true": "right"}),
        ("true", {"switch": True, "on_false": 1, "on_true": 2}),
        ("lazy-true", {"switch": True, "on_false": "left", "on_true": None}),
        ("lazy-false", {"switch": False, "on_false": None, "on_true": "right"}),
    )
    reference_cases = []
    native_cases = []
    for case_id, inputs in cases:
        reference_cases.append(
            {
                "id": case_id,
                "lazy": list(source_class.check_lazy_status(**inputs) or ()),
                "output": source_class.execute(**inputs)[0],
            }
        )
        native_cases.append(
            {
                "id": case_id,
                "lazy": list(
                    ValueSelect.check_lazy_status(
                        condition=cast("bool", inputs["switch"]),
                        on_false=inputs["on_false"],
                        on_true=inputs["on_true"],
                    )
                ),
                "output": ValueSelect.execute(
                    condition=cast("bool", inputs["switch"]),
                    on_false=inputs["on_false"],
                    on_true=inputs["on_true"],
                )["value"],
            }
        )
    return _write_mapping_receipt(
        root,
        record,
        slug="comfy-switch",
        parameters={"cases": [{"id": case_id, "inputs": inputs} for case_id, inputs in cases]},
        reference={"cases": reference_cases},
        native={"cases": native_cases},
    )


def _rebatch_receipt(root: Path, record: Mapping[str, Any], source_class: type[Any]) -> list[Path]:
    arrays = [
        np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1),
        np.arange(8, 20, dtype=np.float32).reshape(3, 2, 2, 1),
    ]
    source = source_class.execute([torch.from_numpy(value) for value in arrays], [3])[0]
    batch_size = cast("int", ListElement.execute(list=[3], index=0)["item"])
    native = ImageRebatch.execute(images=arrays, batch_size=batch_size)["images"]
    return _write_mapping_receipt(
        root,
        record,
        slug="rebatch-images",
        parameters={"batchSize": 3, "inputShapes": [list(value.shape) for value in arrays]},
        reference={"batches": [_array_value(value) for value in source]},
        native={"batches": [_array_value(value) for value in cast("Sequence[object]", native)]},
    )


def _math_expression_receipt(
    root: Path,
    record: Mapping[str, Any],
    comfy_root: Path,
) -> list[Path]:
    reference_vector = build_vector(load_reference(comfy_root))
    native_vector = build_vector()

    def source_cases(vector: Mapping[str, object]) -> list[dict[str, object]]:
        return [
            {
                "id": case["id"],
                "outputs": case["outputs"],
            }
            for case in cast("list[dict[str, object]]", vector["cases"])
            if case["compatibility"] == "ComfyMathExpression"
        ]

    parameters = [
        {
            "id": case["id"],
            "expression": case["expression"],
            "inputs": case["inputs"],
        }
        for case in cast("list[dict[str, object]]", reference_vector["cases"])
        if case["compatibility"] == "ComfyMathExpression"
    ]
    return _write_mapping_receipt(
        root,
        record,
        slug="math-expression",
        parameters={"cases": parameters},
        reference={"cases": source_cases(reference_vector)},
        native={"cases": source_cases(native_vector)},
    )


def _asset_ref(path: Path) -> AssetRef:
    return AssetRef(
        digest=digest_file(path),
        name=path.name,
        size=path.stat().st_size,
        media_type="image/png",
        resolver=_FixedResolver(path),
    )


def _image_receipts(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, type[Any]],
) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="dinkster-image-receipts-") as directory:
        workspace = Path(directory)
        rgba = np.array(
            [
                [[10, 20, 30, 0], [40, 50, 60, 64], [70, 80, 90, 128]],
                [[100, 110, 120, 192], [130, 140, 150, 255], [160, 170, 180, 32]],
            ],
            dtype=np.uint8,
        )
        rgba_path = workspace / "rgba.png"
        rgba_image = Image.fromarray(rgba, mode="RGBA")
        pnginfo = PngInfo()
        pnginfo.add_text("prompt", json.dumps({"node": "receipt"}))
        rgba_image.save(rgba_path, pnginfo=pnginfo)

        opaque_path = workspace / "opaque.png"
        Image.new("RGB", (3, 2), (7, 31, 211)).save(opaque_path)
        fixtures = (
            ("rgba", rgba_path, ["prompt"]),
            ("opaque-fallback", opaque_path, []),
        )

        reference_load_cases = []
        native_load_cases = []
        source_load = source["LoadImage"]()
        for case_id, path, _metadata_keys in fixtures:
            _ReferenceFolderPaths.input_path = path
            reference_image, reference_mask = source_load.load_image(path.name)
            native_image = _mapped_input(records["LoadImage"], "image", {"image": _asset_ref(path)})
            native_result = LoadImage.execute(image=cast("AssetRef", native_image))
            reference_load_cases.append(
                {
                    "id": case_id,
                    "fileSha256": source["LoadImage"].IS_CHANGED(path.name),
                    "image": _array_descriptor(reference_image),
                    "mask": _array_descriptor(reference_mask),
                }
            )
            native_load_cases.append(
                {
                    "id": case_id,
                    "fileSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "image": _array_descriptor(image_input(native_result["image"])),
                    "mask": _array_descriptor(image_input(native_result["mask"])),
                }
            )

        outputs = _write_mapping_receipt(
            root,
            records["LoadImage"],
            slug="load-image",
            parameters={
                "cases": [
                    {
                        "id": case_id,
                        "file": path.name,
                        "metadataKeys": metadata_keys,
                    }
                    for case_id, path, metadata_keys in fixtures
                ],
                "scope": (
                    "still-image filesystem decode, embedded metadata input, alpha polarity, "
                    "and opaque fallback"
                ),
            },
            reference={"cases": reference_load_cases},
            native={"cases": native_load_cases},
        )

        channels = ("alpha", "red", "green", "blue")
        _ReferenceFolderPaths.input_path = rgba_path
        source_mask = source["LoadImageMask"]()
        reference_mask_cases = []
        native_mask_cases = []
        for channel in channels:
            reference_mask = source_mask.load_image_mask(rgba_path.name, channel)[0]
            source_inputs = {"channel": channel, "image": _asset_ref(rgba_path)}
            native_mask = LoadMask.execute(
                mask=cast(
                    "AssetRef", _mapped_input(records["LoadImageMask"], "mask", source_inputs)
                ),
                channel=cast(
                    "str", _mapped_input(records["LoadImageMask"], "channel", source_inputs)
                ),
                mask_polarity=cast(
                    "str", _mapped_input(records["LoadImageMask"], "mask_polarity", source_inputs)
                ),
            )["mask"]
            reference_mask_cases.append(
                {"channel": channel, "mask": _array_descriptor(reference_mask)}
            )
            native_mask_cases.append(
                {"channel": channel, "mask": _array_descriptor(image_input(native_mask))}
            )
        outputs.extend(
            _write_mapping_receipt(
                root,
                records["LoadImageMask"],
                slug="load-image-mask",
                parameters={
                    "channels": list(channels),
                    "scope": "RGBA channel extraction with mapped alpha polarity",
                },
                reference={"cases": reference_mask_cases},
                native={"cases": native_mask_cases},
            )
        )
        return outputs


def _saved_png_payload(root: Path, pass_through: object) -> dict[str, object]:
    files = sorted(root.rglob("*.png"))
    summaries = []
    for index, path in enumerate(files):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            metadata = {
                key: json.loads(cast("str", image.info[key])) for key in ("prompt", "workflow")
            }
        summaries.append(
            {
                "batchIndex": index,
                "extension": path.suffix,
                "metadata": metadata,
                "pixels": _array_value(pixels),
                "subfolder": path.parent.relative_to(root).as_posix(),
            }
        )
    return {
        "files": summaries,
        "passThrough": _array_value(pass_through),
        "uniqueFilenames": len({path.name for path in files}) == len(files),
    }


def _save_image_receipt(
    root: Path,
    record: Mapping[str, Any],
    source_class: type[Any],
) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="dinkster-save-image-receipt-") as directory:
        workspace = Path(directory)
        reference_root = workspace / "reference"
        native_root = workspace / "native"
        reference_root.mkdir()
        native_root.mkdir()
        images = np.array(
            [
                [[[0.0, 0.5, 1.0], [1.25, -0.25, 0.25]]],
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            ],
            dtype=np.float32,
        )
        prompt = {"3": {"class_type": "KSampler"}}
        workflow = {"nodes": [{"type": "SaveImage"}]}
        filename_prefix = "nested/receipt"

        _ReferenceFolderPaths.output_directory = reference_root
        reference_result = source_class().save_images(
            torch.from_numpy(images),
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo={"workflow": workflow},
        )
        reference_payload = _saved_png_payload(
            reference_root,
            cast("Mapping[str, object]", reference_result)["result"][0],  # type: ignore[index]
        )

        source_inputs = {"filename_prefix": filename_prefix, "images": images}
        case = _mapping_case(record)
        nodes = cast("Mapping[str, Mapping[str, Any]]", case["nodes"])
        save_target_values = cast("Mapping[str, Any]", nodes["save_target"]["values"]["target"])
        target = dict(save_target_values)
        target["prefix"] = _mapped_input(record, "save_target:prefix", source_inputs)
        snapshot = workspace / "mounts.json"
        snapshot.write_text(
            json.dumps(
                {
                    "mounts": [
                        {
                            "id": target["mount"],
                            "root": os.fspath(native_root),
                            "mode": "readwrite",
                        }
                    ]
                }
            ),
            encoding="ascii",
        )
        previous_snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT")
        os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = os.fspath(snapshot)
        try:
            native_result = SaveImage.execute(
                images=_mapped_input(record, "images", source_inputs),
                target=target,
                format=cast("str", _mapped_input(record, "format", source_inputs)),
                compression=cast("int", _mapped_input(record, "compression", source_inputs)),
                metadata_json=json.dumps({"prompt": prompt, "workflow": workflow}),
            )
        finally:
            if previous_snapshot is None:
                os.environ.pop("DINKSTER_MOUNTS_SNAPSHOT", None)
            else:
                os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = previous_snapshot
        native_payload = _saved_png_payload(native_root, native_result["images"])

    return _write_mapping_receipt(
        root,
        record,
        slug="save-image",
        parameters={
            "batchSize": 2,
            "compression": _mapped_input(record, "compression", source_inputs),
            "format": _mapped_input(record, "format", source_inputs),
            "metadataKeys": ["prompt", "workflow"],
            "nativeOnlyInput": (
                "metadata_json is supplied directly to exercise PNG metadata writes; translation "
                "does not infer ComfyUI hidden prompt metadata"
            ),
            "scope": (
                "PNG filesystem writes, batch order, uint8 truncation, metadata, "
                "and image passthrough"
            ),
            "targetPrefix": target["prefix"],
            "uncompared": [
                "ComfyUI and Dinkster use intentionally different collision-suffix "
                "filename spellings"
            ],
        },
        reference=reference_payload,
        native=native_payload,
    )


def _trim_audio_receipt(
    root: Path,
    record: Mapping[str, Any],
    source_class: type[Any],
) -> list[Path]:
    waveform = np.arange(60, dtype=np.float32).reshape(1, 2, 30) / 10.0
    cases = (
        ("positive-rounding", 0.25, 0.45),
        ("negative-from-end", -0.8, 0.5),
        ("clamped-start", -9.0, 0.3),
    )
    reference_cases = []
    native_cases = []
    for case_id, start, duration in cases:
        source_audio = {"waveform": torch.from_numpy(waveform), "sample_rate": 10}
        native_audio = {"waveform": waveform, "sample_rate": 10}
        source_inputs = {
            "audio": native_audio,
            "duration": duration,
            "start_index": start,
        }
        reference = source_class.execute(
            source_audio,
            start,
            duration,
        )[0]
        native = TrimAudio.execute(
            audio=_mapped_input(record, "audio", source_inputs),
            start=cast("float", _mapped_input(record, "start", source_inputs)),
            duration=cast("float", _mapped_input(record, "duration", source_inputs)),
        )["audio"]
        reference_cases.append(
            {
                "id": case_id,
                "sampleRate": reference["sample_rate"],
                "waveform": _array_value(reference["waveform"]),
            }
        )
        native_cases.append(
            {
                "id": case_id,
                "sampleRate": native["sample_rate"],
                "waveform": _array_value(native["waveform"]),
            }
        )
    return _write_mapping_receipt(
        root,
        record,
        slug="trim-audio-duration",
        parameters={
            "cases": [
                {"duration": duration, "id": case_id, "start": start}
                for case_id, start, duration in cases
            ],
            "inputShape": list(waveform.shape),
            "sampleRate": 10,
            "scope": "frame rounding, negative-from-end indexing, clamping, and stereo slicing",
        },
        reference={"cases": reference_cases},
        native={"cases": native_cases},
    )


def _primitive_boolean_receipt(
    root: Path,
    record: Mapping[str, Any],
    source_class: type[Any],
) -> list[Path]:
    cases = []
    for value in (False, True):
        cases.append(
            {
                "input": value,
                "reference": source_class.execute(value)[0],
                "native": BooleanPrimitive.execute(value=value)["value"],
            }
        )
    return _write_mapping_receipt(
        root,
        record,
        slug="primitive-boolean",
        parameters={"inputs": [False, True]},
        reference={
            "cases": [{"input": case["input"], "output": case["reference"]} for case in cases]
        },
        native={"cases": [{"input": case["input"], "output": case["native"]} for case in cases]},
    )


def _resize_image_mask_receipt(
    root: Path,
    record: Mapping[str, Any],
    comfy_root: Path,
) -> list[Path]:
    utilities = _load_reference_functions(
        comfy_root,
        "comfy/utils.py",
        ("lanczos", "common_upscale"),
        {"Image": Image, "np": np, "torch": torch},
    )
    functions = _load_reference_functions(
        comfy_root,
        "comfy_extras/nodes_post_processing.py",
        (
            "is_image",
            "init_image_mask_input",
            "finalize_image_mask_input",
            "scale_by",
            "scale_dimensions",
            "scale_longer_dimension",
            "scale_shorter_dimension",
            "scale_total_pixels",
            "scale_match_size",
            "scale_to_multiple_cover",
        ),
        {
            "comfy": SimpleNamespace(
                utils=SimpleNamespace(common_upscale=utilities["common_upscale"])
            ),
            "math": math,
            "torch": torch,
        },
    )
    image = np.linspace(0.0, 1.0, 1 * 5 * 7 * 3, dtype=np.float32).reshape(1, 5, 7, 3)
    mask = np.linspace(0.0, 1.0, 1 * 5 * 7, dtype=np.float32).reshape(1, 5, 7)
    match = np.zeros((1, 8, 11), dtype=np.float32)
    target_megapixels = 88 / (1024 * 1024)
    methods = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")
    selections = (
        "scale dimensions",
        "scale by multiplier",
        "scale longer dimension",
        "scale shorter dimension",
        "scale width",
        "scale height",
        "scale total pixels",
        "match size",
        "scale to multiple",
    )

    def reference_output(value: np.ndarray, selection: str, method: str) -> np.ndarray:
        tensor = torch.from_numpy(value)
        if selection == "scale dimensions":
            output = functions["scale_dimensions"](tensor, 11, 8, method, "disabled")
        elif selection == "scale by multiplier":
            output = functions["scale_by"](tensor, 1.6, method)
        elif selection == "scale longer dimension":
            output = functions["scale_longer_dimension"](tensor, 11, method)
        elif selection == "scale shorter dimension":
            output = functions["scale_shorter_dimension"](tensor, 8, method)
        elif selection == "scale width":
            output = functions["scale_dimensions"](tensor, 11, 0, method)
        elif selection == "scale height":
            output = functions["scale_dimensions"](tensor, 0, 8, method)
        elif selection == "scale total pixels":
            output = functions["scale_total_pixels"](tensor, target_megapixels, method)
        elif selection == "match size":
            output = functions["scale_match_size"](
                tensor, torch.from_numpy(match), method, "disabled"
            )
        else:
            output = functions["scale_to_multiple_cover"](tensor, 4, method)
        return cast("torch.Tensor", output).detach().numpy()

    def native_output(value: np.ndarray, selection: str, method: str) -> np.ndarray:
        arguments: dict[str, object] = {"image": value, "interpolation": method}
        if selection == "scale dimensions":
            arguments.update(target="dimensions", width=11, height=8)
        elif selection == "scale by multiplier":
            arguments.update(target="factor", factor=1.6)
        elif selection == "scale longer dimension":
            arguments.update(target="longest", size=11)
        elif selection == "scale shorter dimension":
            arguments.update(target="shortest", size=8)
        elif selection == "scale width":
            arguments.update(target="width", width=11)
        elif selection == "scale height":
            arguments.update(target="height", height=8)
        elif selection == "scale total pixels":
            arguments.update(
                target="total_pixels",
                megapixels=target_megapixels,
                resolution_steps=1,
            )
        elif selection == "match size":
            arguments.update(target="match", reference=match)
        else:
            arguments.update(target="multiple_cover", multiple_of=4)
        return np.asarray(cast("Mapping[str, object]", ImageResize.execute(**arguments))["image"])

    reference_outputs = []
    native_outputs = []
    output_shapes: dict[str, list[list[int]]] = {}
    maximum_differences: dict[str, float] = {}
    for method in methods:
        method_reference = []
        method_native = []
        method_shapes = []
        for value in (image, mask):
            for selection in selections:
                reference = reference_output(value, selection, method)
                native = native_output(value, selection, method)
                if reference.shape != native.shape:
                    raise RuntimeError(
                        f"ResizeImageMaskNode {selection} shape mismatch: "
                        f"{reference.shape} != {native.shape}"
                    )
                method_shapes.append(list(reference.shape))
                method_reference.append(reference.reshape(-1))
                method_native.append(native.reshape(-1))
        reference_vector = np.concatenate(method_reference)
        native_vector = np.concatenate(method_native)
        reference_outputs.append(reference_vector)
        native_outputs.append(native_vector)
        output_shapes[method] = method_shapes
        maximum_differences[method] = float(np.max(np.abs(reference_vector - native_vector)))
    return _write_numeric_array_receipt(
        root,
        record,
        slug="resize-image-mask-dynamic-selections",
        parameters={
            "inputShapes": {"IMAGE": list(image.shape), "MASK": list(mask.shape)},
            "interpolation": list(methods),
            "maximumAbsoluteDifference": maximum_differences,
            "outputShapes": output_shapes,
            "scope": "all nine dynamic selections for IMAGE and MASK",
            "selections": list(selections),
        },
        reference=np.stack(reference_outputs),
        native=np.stack(native_outputs),
        data_kind="image",
    )


def _image_batch_pad_receipts(
    root: Path, records: Mapping[str, Mapping[str, Any]], comfy_root: Path
) -> list[Path]:
    utilities = _load_reference_functions(
        comfy_root,
        "comfy/utils.py",
        ("lanczos", "common_upscale"),
        {"Image": Image, "np": np, "torch": torch},
    )
    namespace = {
        "torch": torch,
        "IO": _IO,
        "comfy": SimpleNamespace(utils=SimpleNamespace(common_upscale=utilities["common_upscale"])),
    }
    batch_images = _load_reference_functions(
        comfy_root, "comfy_extras/nodes_post_processing.py", ("batch_images",), namespace
    )["batch_images"]
    pad_image = _load_reference_classes(
        comfy_root, "comfy_extras/nodes_images.py", ("ResizeAndPadImage",), namespace
    )["ResizeAndPadImage"]
    rgb = torch.arange(45, dtype=torch.float32).reshape(1, 3, 5, 3) / 44
    rgba = torch.arange(240, dtype=torch.float32).reshape(1, 6, 10, 4) / 239
    native_batch = ImageBatchCombine.execute(
        images={"1": rgb.numpy(), "2": rgba.numpy()},
        shape_policy="resize_to_first",
        channel_policy="pad_with_one",
    )["image"]
    outputs = _write_numeric_array_receipt(
        root,
        records["BatchImagesNode"],
        slug="batch-images",
        parameters={"inputShapes": [list(rgb.shape), list(rgba.shape)]},
        reference=batch_images([rgb, rgba]).numpy(),
        native=cast("np.ndarray", native_batch),
        data_kind="image",
    )
    reference, native = [], []
    methods = ("area", "bicubic", "nearest-exact", "bilinear", "lanczos")
    for method in methods:
        for color in ("black", "white"):
            reference.append(pad_image.execute(rgb, 4, 4, color, method)[0].numpy())
            native.append(
                ImageResize.execute(
                    image=rgb.numpy(),
                    width=4,
                    height=4,
                    mode="pad",
                    fit_rounding="floor",
                    interpolation=method,
                    pad_value=float(color == "white"),
                )["image"]
            )
    outputs.extend(
        _write_numeric_array_receipt(
            root,
            records["ResizeAndPadImage"],
            slug="resize-and-pad-image",
            parameters={
                "interpolation": list(methods),
                "paddingColors": ["black", "white"],
                "width": 4,
                "height": 4,
            },
            reference=np.stack(reference),
            native=np.stack(native),
            data_kind="image",
        )
    )
    return outputs


def _unet_loader_receipt(
    root: Path,
    record: Mapping[str, Any],
    comfy_root: Path,
) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="dinkster-unet-receipt-") as directory:
        path = Path(directory) / "diffusion.safetensors"
        path.write_bytes(b"source-parity-model")
        asset = _asset_ref(path)
        reference_calls: list[dict[str, object]] = []
        native_calls: list[dict[str, object]] = []

        def normalized(options: Mapping[str, object]) -> dict[str, object]:
            return {
                name: str(value).removeprefix("torch.") if name == "dtype" else value
                for name, value in options.items()
            }

        source_sd = SimpleNamespace(
            load_diffusion_model=lambda loaded_path, *, model_options: (
                reference_calls.append(
                    {"path": Path(loaded_path).name, "options": normalized(model_options)}
                )
                or object()
            )
        )
        source_class = _load_reference_classes(
            comfy_root,
            "nodes.py",
            ("UNETLoader",),
            {
                "comfy": SimpleNamespace(sd=source_sd),
                "folder_paths": SimpleNamespace(
                    get_full_path_or_raise=lambda _kind, _name: os.fspath(path)
                ),
                "torch": torch,
            },
        )["UNETLoader"]
        native_sd = SimpleNamespace(
            load_diffusion_model=lambda loaded_path, *, model_options: (
                native_calls.append(
                    {"path": Path(loaded_path).name, "options": normalized(model_options)}
                )
                or object()
            )
        )
        real_import = importlib.import_module

        def native_import(name: str) -> object:
            if name == "comfy.sd":
                return native_sd
            return real_import(name)

        pool = SimpleNamespace(label=lambda *_args: None)
        with (
            _patched(importlib, "import_module", native_import),
            _patched(compat_native, "_drop_path_reload_factories", lambda _model: None),
            _patched(compat_native, "default_pool", lambda: pool),
        ):
            for dtype in ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"):
                source_class().load_unet(path.name, dtype)
                compat_native.LoadDiffusionModel.execute(
                    diffusion_model=asset,
                    weight_dtype=dtype,
                )
    return _write_mapping_receipt(
        root,
        record,
        slug="unet-loader",
        parameters={"weightDtypes": list(compat_native.LoadDiffusionModel.WEIGHT_DTYPES)},
        reference={"calls": reference_calls},
        native={"calls": native_calls},
    )


def _vae_loader_receipt(
    root: Path,
    record: Mapping[str, Any],
    comfy_root: Path,
) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="dinkster-vae-receipt-") as directory:
        path = Path(directory) / "seedvr2-vae.safetensors"
        header = json.dumps(
            {
                "__metadata__": {"format": "seedvr2"},
                "decoder.weight": {
                    "data_offsets": [0, 0],
                    "dtype": "F32",
                    "shape": [0],
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header)
        asset = _asset_ref(path)
        state = {"decoder.weight": "sentinel"}
        metadata = {"format": "seedvr2"}
        reference_calls: list[dict[str, object]] = []
        native_calls: list[dict[str, object]] = []

        def load(calls: list[dict[str, object]]) -> Callable[..., object]:
            def load_torch_file(
                loaded_path: str,
                *,
                return_metadata: bool = False,
            ) -> object:
                calls.append(
                    {
                        "operation": "load_torch_file",
                        "path": Path(loaded_path).name,
                        "returnMetadata": return_metadata,
                    }
                )
                return (state, metadata) if return_metadata else state

            return load_torch_file

        def vae_type(calls: list[dict[str, object]]) -> type[Any]:
            class StubVAE:
                def __init__(self, *, sd: Mapping[str, object], metadata: object) -> None:
                    calls.append(
                        {
                            "metadata": metadata,
                            "operation": "construct_vae",
                            "stateKeys": sorted(sd),
                        }
                    )
                    self.patcher = SimpleNamespace(cached_patcher_init=None)

                def throw_exception_if_invalid(self) -> None:
                    calls.append({"operation": "validate_vae"})

            return StubVAE

        source_sd = SimpleNamespace(
            VAE=vae_type(reference_calls),
            load_vae_patcher=object(),
        )
        source_class = _load_reference_classes(
            comfy_root,
            "nodes.py",
            ("VAELoader",),
            {
                "comfy": SimpleNamespace(
                    sd=source_sd,
                    utils=SimpleNamespace(load_torch_file=load(reference_calls)),
                ),
                "folder_paths": SimpleNamespace(
                    get_full_path_or_raise=lambda _kind, _name: os.fspath(path)
                ),
                "os": os,
                "torch": torch,
            },
        )["VAELoader"]
        source_class().load_vae(path.name)

        native_modules = {
            "comfy.sd": SimpleNamespace(VAE=vae_type(native_calls)),
            "dinkster_inference_torch.checkpoint": SimpleNamespace(
                load_checkpoint_with_metadata=lambda loaded_path: load(native_calls)(
                    loaded_path, return_metadata=True
                )
            ),
        }
        real_import = importlib.import_module

        def native_import(name: str) -> object:
            if name in native_modules:
                return native_modules[name]
            return real_import(name)

        pool = SimpleNamespace(
            label=lambda *_args: None,
            label_source=lambda *_args, **_kwargs: None,
        )
        with (
            _patched(importlib, "import_module", native_import),
            _patched(compat_native, "default_pool", lambda: pool),
        ):
            compat_native.LoadVae.execute(vae=asset)
    return _write_mapping_receipt(
        root,
        record,
        slug="vae-loader",
        parameters={
            "scope": (
                "single-file state-dict loading, metadata propagation, and VAE validation; "
                "path-backed reload factories are excluded by the digest-backed asset contract"
            )
        },
        reference={"calls": reference_calls},
        native={"calls": native_calls},
    )


def _ksampler_receipt(
    root: Path,
    record: Mapping[str, Any],
    comfy_root: Path,
) -> list[Path]:
    case = _mapping_case(record)
    mapped_inputs = cast("Mapping[str, Mapping[str, Any]]", case["inputs"])
    sampler_map = cast("Mapping[str, str]", mapped_inputs["sampler_name"]["transform"]["map"])
    scheduler_map = cast("Mapping[str, str]", mapped_inputs["scheduler"]["transform"]["map"])
    for mapping in (sampler_map, scheduler_map):
        if mapping != {name: f"dinkster.{name}" for name in mapping}:
            raise RuntimeError("KSampler enum mapping must preserve every source id explicitly")
    source_calls: list[dict[str, object]] = []
    native_calls: list[dict[str, object]] = []

    def source_sample(*args: object, **kwargs: object) -> tuple[str]:
        names = (
            "model",
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "positive",
            "negative",
            "latent",
        )
        values = dict(zip(names, args, strict=True))
        source_calls.append(
            {
                "seed": values["seed"],
                "steps": values["steps"],
                "cfg": values["cfg"],
                "sampler_name": f"dinkster.{values['sampler_name']}",
                "scheduler": f"dinkster.{values['scheduler']}",
                "denoise": kwargs["denoise"],
            }
        )
        return ("sampled",)

    source_class = _load_reference_classes(
        comfy_root,
        "nodes.py",
        ("KSampler",),
        {"common_ksampler": source_sample},
    )["KSampler"]

    def native_sample(_cls: type[Any], **kwargs: object) -> Mapping[str, object]:
        native_calls.append(
            {
                name: kwargs[name]
                for name in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise")
            }
        )
        return {"latent": "sampled"}

    handle = SimpleNamespace(
        runtime=object(),
        recipe=SimpleNamespace(
            family_id="dinkster.seedvr2",
            sources=(SimpleNamespace(role="diffusion"),),
        ),
    )
    source_class().sample(
        object(), 7, 2, 1.0, "euler", "simple", object(), object(), object(), denoise=0.75
    )
    with (
        _patched(native_arm, "_require_provider_runtime", lambda *_args: handle),
        _patched_classmethod(native_arm.NativeKSampler, "execute", native_sample),
    ):
        native_arm.GenerationKSampler.execute(
            model=object(),
            seed=7,
            steps=2,
            cfg=1.0,
            sampler_name=sampler_map["euler"],
            scheduler=scheduler_map["simple"],
            positive=object(),
            negative=object(),
            latent_image=object(),
            denoise=0.75,
        )
    return _write_mapping_receipt(
        root,
        record,
        slug="ksampler",
        parameters={
            "samplerMap": dict(sampler_map),
            "schedulerMap": dict(scheduler_map),
            "scope": "production KSampler sugar dispatch into NativeKSampler",
        },
        reference={"calls": source_calls},
        native={"calls": native_calls},
    )


def _tiled_vae_receipts(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, type[Any]],
) -> list[Path]:
    encoded = torch.arange(1 * 16 * 2 * 4 * 6, dtype=torch.float32).reshape(1, 16, 2, 4, 6)
    decoded = torch.linspace(0.0, 1.0, 1 * 3 * 2 * 32 * 48).reshape(1, 3, 2, 32, 48)

    class SourceVAE:
        @staticmethod
        def temporal_compression_decode() -> int:
            return 4

        @staticmethod
        def spacial_compression_decode() -> int:
            return 8

        @staticmethod
        def encode_tiled(_pixels: object, **_geometry: object) -> torch.Tensor:
            return encoded.clone()

        @staticmethod
        def decode_tiled(_latent: object, **_geometry: object) -> torch.Tensor:
            return decoded.permute(0, 2, 3, 4, 1).clone()

    class NativeCodec:
        descriptor = SEEDVR2_CODEC
        load_device = torch.device("cpu")
        resource_identity = "source-parity-seedvr2-codec"

        @staticmethod
        def require_active() -> None:
            return None

        @staticmethod
        def stage():
            return nullcontext()

        @staticmethod
        def encode_content(_content: torch.Tensor) -> torch.Tensor:
            return encoded.clone()

        @staticmethod
        def decode_latent(_latent: torch.Tensor) -> torch.Tensor:
            return decoded.clone()

        @staticmethod
        def encode_content_tiled(
            _content: torch.Tensor, *, tile: tuple[int, ...], overlap: tuple[int, ...]
        ) -> torch.Tensor:
            if tile != (64, 512, 512) or overlap != (8, 128, 128):
                raise RuntimeError("native tiled VAE encode geometry drifted")
            return encoded.clone()

        @staticmethod
        def decode_latent_tiled(
            _latent: torch.Tensor, *, tile: tuple[int, ...], overlap: tuple[int, ...]
        ) -> torch.Tensor:
            if tile != (2, 64, 64) or overlap != (1, 16, 16):
                raise RuntimeError("native tiled VAE decode geometry drifted")
            return decoded.clone()

    source_vae = SourceVAE()
    native_vae = NativeCodec()
    pixels = torch.linspace(0.0, 1.0, 5 * 32 * 48 * 3).reshape(5, 32, 48, 3)
    reference_encoded = source["VAEEncodeTiled"]().encode(source_vae, pixels, 512, 128, 64, 8)[0][
        "samples"
    ]
    reference_decoded = source["VAEDecodeTiled"]().decode(
        source_vae, {"samples": encoded}, 512, 200, 8, 8
    )[0]
    native_encoded = cast(
        "Mapping[str, torch.Tensor]",
        native_arm.GenerationVAEEncodeTiled.execute(
            pixels=pixels,
            vae=native_vae,
            tile_size=512,
            overlap=128,
            temporal_size=64,
            temporal_overlap=8,
        )["latent"],
    )["samples"]
    native_decoded = native_arm.GenerationVAEDecodeTiled.execute(
        samples={"samples": encoded},
        vae=native_vae,
        tile_size=512,
        overlap=200,
        temporal_size=8,
        temporal_overlap=8,
    )["image"]
    outputs = _write_mapping_receipt(
        root,
        records["VAEEncodeTiled"],
        slug="vae-encode-tiled",
        parameters={"overlap": 128, "temporalOverlap": 8, "temporalSize": 64, "tileSize": 512},
        reference={"latent": _array_value(reference_encoded)},
        native={"latent": _array_value(native_encoded)},
    )
    outputs.extend(
        _write_mapping_receipt(
            root,
            records["VAEDecodeTiled"],
            slug="vae-decode-tiled",
            parameters={"overlap": 200, "temporalOverlap": 8, "temporalSize": 8, "tileSize": 512},
            reference={"image": _array_value(reference_decoded)},
            native={"image": _array_value(native_decoded)},
        )
    )
    return outputs


def _seedvr2_source_classes(comfy_root: Path) -> dict[str, type[Any]]:
    helpers = _load_reference_functions(
        comfy_root,
        "comfy_extras/nodes_seedvr.py",
        (
            "_resolve_seedvr2_diffusion_model",
            "div_pad",
            "cut_videos",
            "_seedvr2_input_shorter_edge",
            "_seedvr2_pad",
            "_seedvr2_chunk_crossfade_weights",
        ),
        {
            "_ATTR_MISSING": object(),
            "_SEEDVR2_INVALID_MODEL_MSG_PREFIX": "invalid SeedVR2 model",
            "io": _IO,
            "torch": torch,
        },
    )
    return _load_reference_classes(
        comfy_root,
        "comfy_extras/nodes_seedvr.py",
        (
            "SeedVR2Preprocess",
            "SeedVR2PostProcessing",
            "SeedVR2Conditioning",
            "SeedVR2TemporalChunk",
            "SeedVR2TemporalMerge",
        ),
        {
            **helpers,
            "SEEDVR2_LATENT_CHANNELS": 16,
            "io": _IO,
            "torch": torch,
        },
    )


def _seedvr2_native_branch(carrier: object) -> dict[str, object]:
    prepared = materialize_seedvr2_conditioning(cast("Any", carrier), device="cpu")
    return {
        "branch": prepared.branch,
        "condition": _array_value(prepared.embeddings),
    }


def _seedvr2_receipts(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, type[Any]],
) -> list[Path]:
    input_image = torch.linspace(0.0, 1.0, 2 * 17 * 18 * 4).reshape(2, 17, 18, 4)
    reference_preprocessed = source["SeedVR2Preprocess"].execute(input_image)[0]
    native_preprocessed = native_arm.GenerationSeedVR2Preprocess.execute(
        resized_images=input_image
    )["images"]
    outputs = _write_mapping_receipt(
        root,
        records["SeedVR2Preprocess"],
        slug="seedvr2-preprocess",
        parameters={"inputShape": list(input_image.shape)},
        reference={"images": _array_value(reference_preprocessed)},
        native={"images": _array_value(native_preprocessed)},
    )

    decoded = torch.linspace(0.0, 1.0, 3 * 19 * 21 * 3).reshape(3, 19, 21, 3)
    reference_postprocessed = source["SeedVR2PostProcessing"].execute(decoded, input_image, "none")[
        0
    ]
    native_postprocessed = native_arm.GenerationSeedVR2PostProcessing.execute(
        images=decoded,
        original_resized_images=input_image,
        color_correction_method="none",
    )["images"]
    outputs.extend(
        _write_mapping_receipt(
            root,
            records["SeedVR2PostProcessing"],
            slug="seedvr2-postprocess",
            parameters={"colorCorrectionMethod": "none", "restoresAlpha": True},
            reference={"images": _array_value(reference_postprocessed)},
            native={"images": _array_value(native_postprocessed)},
        )
    )

    identity = "native:dinkster.seedvr2:source-parity"
    runtime = SeedVR2DiffusionRuntime(
        cast("Any", torch.nn.Identity()),
        runtime_identity=identity,
        compute_dtype=torch.float32,
    )
    handle = SimpleNamespace(runtime=runtime, recipe=SimpleNamespace(runtime_identity=identity))
    latent = torch.linspace(-1.0, 1.0, 1 * 16 * 2 * 3 * 4).reshape(1, 16, 2, 3, 4)
    diffusion = SimpleNamespace(
        positive_conditioning=torch.tensor([1.0]),
        negative_conditioning=torch.tensor([-1.0]),
    )
    reference_conditioning = source["SeedVR2Conditioning"].execute(
        SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion)),
        {"samples": latent},
    )
    with (
        _patched(native_arm, "_application_chain_model", lambda model, _name: (model, ())),
        _patched(
            native_arm,
            "_native_model",
            lambda _model, _name: (handle, (), {}, None, None, (), None, {}),
        ),
    ):
        native_conditioning = native_arm.GenerationSeedVR2Conditioning.execute(
            model=handle,
            vae_conditioning={"samples": latent},
        )

    def reference_branch(index: int, branch: str) -> dict[str, object]:
        row = cast("Sequence[Sequence[object]]", reference_conditioning[index])[0]
        metadata = cast("Mapping[str, torch.Tensor]", row[1])
        return {"branch": branch, "condition": _array_value(metadata["condition"])}

    outputs.extend(
        _write_mapping_receipt(
            root,
            records["SeedVR2Conditioning"],
            slug="seedvr2-conditioning",
            parameters={"latentShape": list(latent.shape)},
            reference={
                "positive": reference_branch(0, "positive"),
                "negative": reference_branch(1, "negative"),
            },
            native={
                "positive": _seedvr2_native_branch(native_conditioning["positive"]),
                "negative": _seedvr2_native_branch(native_conditioning["negative"]),
            },
        )
    )

    full = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1, 1).expand(1, 16, 6, 1, 1)
    reference_chunked = source["SeedVR2TemporalChunk"].execute(
        {"samples": full, "noise_mask": torch.ones_like(full)},
        2,
        {"chunking_mode": "manual", "frames_per_chunk": 13},
    )
    native_chunked = native_arm.GenerationSeedVR2TemporalChunk.execute(
        latent={"samples": full, "noise_mask": torch.ones_like(full)},
        temporal_overlap=2,
        chunking_mode={"chunking_mode": "manual", "frames_per_chunk": 13},
    )
    reference_chunks = cast("Sequence[Mapping[str, torch.Tensor]]", reference_chunked[0])
    native_chunks = cast("Sequence[Mapping[str, torch.Tensor]]", native_chunked["latents"])

    def chunk_payload(values: Sequence[Mapping[str, torch.Tensor]]) -> list[dict[str, object]]:
        return [_array_value(value["samples"]) for value in values]

    outputs.extend(
        _write_mapping_receipt(
            root,
            records["SeedVR2TemporalChunk"],
            slug="seedvr2-temporal-chunk",
            parameters={"framesPerChunk": 13, "temporalOverlap": 2},
            reference={
                "chunks": chunk_payload(reference_chunks),
                "overlap": reference_chunked[1],
            },
            native={
                "chunks": chunk_payload(native_chunks),
                "overlap": native_chunked["temporal_overlap"],
            },
        )
    )
    reference_merged = source["SeedVR2TemporalMerge"].execute(reference_chunks, [2])[0]
    native_merged = native_arm.GenerationSeedVR2TemporalMerge.execute(
        latents=native_chunks,
        temporal_overlap=2,
    )["latent"]
    outputs.extend(
        _write_mapping_receipt(
            root,
            records["SeedVR2TemporalMerge"],
            slug="seedvr2-temporal-merge",
            parameters={"chunkCount": len(reference_chunks), "temporalOverlap": 2},
            reference={
                "latent": _array_value(cast("Mapping[str, object]", reference_merged)["samples"])
            },
            native={"latent": _array_value(cast("Mapping[str, object]", native_merged)["samples"])},
        )
    )
    return outputs


class _VideoContainer(StrEnum):
    AUTO = "auto"
    MP4 = "mp4"
    MKV = "mkv"
    WEBM = "webm"

    @classmethod
    def get_extension(cls, value: str) -> str:
        return value


class _VideoCodec(StrEnum):
    AUTO = "auto"
    H264 = "h264"
    AV1 = "av1"


def _write_video_fixture(path: Path) -> None:
    with av.open(os.fspath(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=3)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for index in range(3):
            pixels = np.full((16, 16, 3), index * 64, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 3)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _layer_receipts(root: Path, records: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    from dinkster_image_document.compat import from_comfy_layers
    from dinkster_nodes_image.compositor import (
        AddLayer,
        CreateLayeredImage,
        LayersFromBoundingBoxes,
    )
    from layer_comfy_aliases import REFERENCE, layer_alias_data

    golden_path = REPO / "tests/fixtures/layer_document_goldens.json"
    golden = json.loads(golden_path.read_text())
    if golden["reference"] != REFERENCE:
        raise RuntimeError("layer golden uses the wrong reference")
    schemas, _ = layer_alias_data()
    defaults = {
        schema.node_type: {item.id: item.default for item in schema.inputs} for schema in schemas
    }
    nodes = {
        "AddLayer": AddLayer,
        "LayersFromBoundingBoxes": LayersFromBoundingBoxes,
        "ImageCompositor": CreateLayeredImage,
    }
    paths = []
    for case in golden["aliasCases"]:
        name = case["node"]
        record = records[name]
        inputs = defaults[f"comfy.{name}"] | case["inputs"]
        inputs["layers"] = from_comfy_layers(inputs["layers"])
        for key in ("image", "mask"):
            if key in inputs and inputs[key] is not None:
                inputs[key] = np.asarray(inputs[key], dtype=np.float32)
        arguments = {
            key: _mapped_input(record, key, inputs) for key in _mapping_case(record)["inputs"]
        }
        result = nodes[name].execute(**arguments)
        if name != "ImageCompositor":
            result = CreateLayeredImage.execute(layers=result["layers"], color_space="linear")
        paths.extend(
            _write_mapping_receipt(
                root,
                record,
                slug=f"layers-{name.lower()}",
                parameters={"goldenSha256": hashlib.sha256(golden_path.read_bytes()).hexdigest()},
                reference=np.concatenate(
                    [np.asarray(case["image"]).ravel(), np.asarray(case["mask"]).ravel()]
                ),
                native=np.concatenate(
                    [
                        np.asarray(result["image"]).ravel(),
                        np.asarray(result["transparency_mask"]).ravel(),
                    ]
                ).astype(np.float64),
                comparison={
                    "comparator": "numeric-array/1",
                    "dataKind": "tensor",
                    "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 1 / 255}],
                },
            )
        )
    return paths


def _audio_alias_receipts(root: Path) -> list[Path]:
    from dinkster_values.audio_codec import audio_window, effective_audio_facts
    from gen_audio_alias_receipts import COMFY_PIN, CROP_PIN, VHS_PIN
    from pytest import MonkeyPatch

    sys.path.insert(0, str(REPO))
    from tests.test_audio_alias_replay import RECEIPTS, RECORDS, _audio, _run_case
    from tests.test_audio_io import _mount

    source = RECEIPTS["source"]
    if (source["comfyCommit"], source["cropCommit"], source["vhsCommit"]) != (
        COMFY_PIN,
        CROP_PIN,
        VHS_PIN,
    ):
        raise RuntimeError("audio alias golden uses the wrong source pins")
    golden = REPO / "tests/goldens/audio_aliases_b78cec87.json"
    paths: list[Path] = []

    def write(
        name: str, slug: str, reference: object, native: object, **parameters: object
    ) -> None:
        paths.extend(
            _write_mapping_receipt(
                root,
                RECORDS[name],
                slug=slug,
                parameters={
                    "goldenSha256": hashlib.sha256(golden.read_bytes()).hexdigest(),
                    **parameters,
                },
                reference=reference,
                native=native,
                comparison={"comparator": "exact-array/1", "dataKind": "tensor"},
            )
        )

    with tempfile.TemporaryDirectory(prefix="dinkster-audio-replay-") as directory:
        workspace = Path(directory)
        import base64

        wav = workspace / "input.wav"
        wav.write_bytes(base64.b64decode(RECEIPTS["loadWav"]))
        asset = _asset_ref(wav)
        for name in ("VHS_LoadAudio", "VHS_LoadAudioUpload"):
            references, native = [], []
            cases = [case for case in RECEIPTS["loadCases"] if case["nodeClass"] == name]
            for case in cases:
                inputs = {"duration": case["duration"]}
                if name == "VHS_LoadAudio":
                    inputs.update(audio_file=asset, seek_seconds=case["start"])
                else:
                    inputs.update(audio=asset, start_time=case["start"])
                outputs, _ = _run_case(name, inputs)
                facts = effective_audio_facts(outputs["audio"])
                native.append(
                    audio_window(outputs["audio"], 0, facts["frames"])["waveform"].ravel()
                )
                references.append(_audio(case["output"])["waveform"].ravel())
            write(
                name,
                name.lower(),
                np.concatenate(references),
                np.concatenate(native),
                windows=len(cases),
            )

        case = RECEIPTS["cropCases"][0]
        outputs, _ = _run_case(
            "AudioCrop",
            {
                "audio": _audio(RECEIPTS["cropInput"]),
                "start_time": case["start_time"],
                "end_time": case["end_time"],
            },
        )
        facts = effective_audio_facts(outputs["audio"])
        write(
            "AudioCrop",
            "audio-crop",
            _audio(case["output"])["waveform"],
            audio_window(outputs["audio"], 0, facts["frames"])["waveform"],
            start=case["start_time"],
            end=case["end_time"],
        )

        with MonkeyPatch.context() as patch:
            _mount(workspace, patch)
            case = next(case for case in RECEIPTS["saveCases"] if case["format"] == "flac")
            _, outputs = _run_case(
                "SaveAudioAdvanced",
                {
                    "audio": _audio(RECEIPTS["saveInput"]),
                    "format": "flac",
                },
            )
            from dinkster_nodes_media_io import LoadAudio

            native = []
            for asset in outputs["audios"]:
                audio = LoadAudio.execute(audio=asset)["audio"]
                facts = effective_audio_facts(audio)
                native.append(audio_window(audio, 0, facts["frames"])["waveform"])
            write(
                "SaveAudioAdvanced",
                "save-audio-advanced",
                np.concatenate([_audio(output)["waveform"] for output in case["outputs"]]),
                np.concatenate(native),
                format="flac",
                batches=len(native),
            )
    return paths


def _current_media_receipts(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    comfy_root: Path,
) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="dinkster-current-media-receipts-") as directory:
        workspace = Path(directory)
        video_path = workspace / "fixture.mp4"
        _write_video_fixture(video_path)
        source_root = workspace / "source"
        native_root = workspace / "native"
        source_root.mkdir()
        native_root.mkdir()
        folder_paths = SimpleNamespace(
            exists_annotated_filepath=lambda _name: True,
            get_annotated_filepath=lambda _name: os.fspath(video_path),
            get_output_directory=lambda: os.fspath(source_root),
            get_save_image_path=lambda prefix, output, _width, _height: (
                output,
                Path(prefix).name,
                1,
                "",
                prefix,
            ),
        )
        input_impl = SimpleNamespace(
            VideoFromFile=lambda path: SimpleNamespace(get_stream_source=lambda: path)
        )
        video_source = _load_reference_classes(
            comfy_root,
            "comfy_extras/nodes_video.py",
            ("LoadVideo", "SaveVideo", "SaveWEBM", "VideoSlice"),
            {
                "Input": SimpleNamespace(Video=object),
                "InputImpl": input_impl,
                "Optional": Optional,
                "Types": SimpleNamespace(VideoCodec=_VideoCodec, VideoContainer=_VideoContainer),
                "args": SimpleNamespace(disable_metadata=True),
                "av": av,
                "Fraction": Fraction,
                "json": json,
                "torch": torch,
                "folder_paths": folder_paths,
                "io": _IO,
                "os": os,
                "preview_input_video": lambda _file, _source: None,
                "ui": SimpleNamespace(
                    PreviewVideo=lambda value: value,
                    SavedResult=lambda *value: value,
                ),
            },
            revision=VIDEO_REFERENCE_COMMIT,
        )
        source_load = video_source["LoadVideo"].execute(video_path.name)[0]
        native_load = LoadVideoValue.execute(video=_asset_ref(video_path))["video"]
        outputs = _write_mapping_receipt(
            root,
            records["LoadVideo"],
            slug="load-video-value",
            parameters={"container": "mp4"},
            reference={
                "sha256": hashlib.sha256(
                    Path(source_load.get_stream_source()).read_bytes()
                ).hexdigest()
            },
            native={"sha256": hashlib.sha256(render_video_original(native_load)).hexdigest()},
        )
        reference_api, _ = _load_reference_api(comfy_root, revision=VIDEO_REFERENCE_COMMIT)
        edit_globals: dict[str, object] = {"Input": SimpleNamespace(Video=object), "io": _IO}
        edit_source = _git(
            comfy_root, "show", f"{VIDEO_REFERENCE_COMMIT}:comfy_extras/nodes_video.py"
        ).decode("utf-8")
        edit_globals.update(
            _load_source_functions(
                edit_source,
                "nodes_video.py",
                ("apply_video_trim", "apply_video_crop"),
                edit_globals,
            )
        )
        edit_globals["save_video_preview"] = lambda _video: None
        edit_nodes = _load_source_classes(
            edit_source, "nodes_video.py", ("VideoTrim", "VideoCrop"), edit_globals
        )
        for name, native_node, port in (
            ("VideoTrim", TrimVideo, "trim"),
            ("VideoCrop", CropVideo, "crop"),
        ):
            widgets = [{}, {port: {}}]
            source_results, native_results = [], []
            for widget in widgets:
                source_clip = cast(Any, reference_api).VideoFromFile(str(video_path))
                reference = edit_nodes[name].execute(
                    source_clip, widget, **({"strict_duration": False} if port == "trim" else {})
                )[0]
                native = native_node.execute(
                    video=native_load,
                    video_edit=_mapped_input(records[name], "video_edit", {port: widget}),
                )["video"]
                source_results.append(
                    {
                        "sha256": hashlib.sha256(
                            Path(reference.get_stream_source()).read_bytes()
                        ).hexdigest()
                    }
                )
                native_results.append(
                    {"sha256": hashlib.sha256(render_video_original(native)).hexdigest()}
                )
            outputs.extend(
                _write_mapping_receipt(
                    root,
                    records[name],
                    slug=f"video-{port}-empty-widget",
                    parameters={"widgets": widgets, "previewDisposition": "not-compared"},
                    reference=source_results,
                    native=native_results,
                )
            )

        class SourceVideo:
            """Exercise node routing with a copy-only source, not reference encoder parity."""

            @staticmethod
            def get_dimensions() -> tuple[int, int]:
                return 16, 16

            @staticmethod
            def save_to(path: str, **_options: object) -> None:
                Path(path).write_bytes(video_path.read_bytes())

            def as_trimmed(self, start: float, duration: float, *, strict_duration: bool):
                if (start, duration, strict_duration) != (0.0, 0.0, False):
                    raise RuntimeError("unexpected source trim parameters")
                return self

        source_video = SourceVideo()
        source_save = video_source["SaveVideo"]
        source_save.hidden = SimpleNamespace(extra_pnginfo=None, prompt=None)
        source_save.execute(source_video, "video/receipt", "auto", {"codec": "auto"})
        snapshot = workspace / "mounts.json"
        snapshot.write_text(
            json.dumps(
                {"mounts": [{"id": "receipt", "root": os.fspath(native_root), "mode": "readwrite"}]}
            ),
            encoding="ascii",
        )
        previous_snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT")
        os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = os.fspath(snapshot)
        try:
            SaveVideoValue.execute(
                video=native_load,
                target={"mount": "receipt", "prefix": "video/receipt"},
            )
        finally:
            if previous_snapshot is None:
                os.environ.pop("DINKSTER_MOUNTS_SNAPSHOT", None)
            else:
                os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = previous_snapshot
        source_saved = next(source_root.rglob("*.mp4")).read_bytes()
        native_saved = next(native_root.rglob("*.mp4")).read_bytes()
        outputs.extend(
            _write_mapping_receipt(
                root,
                records["SaveVideo"],
                slug="save-video-value",
                parameters={"codec": "auto", "format": "auto"},
                reference={"sha256": hashlib.sha256(source_saved).hexdigest()},
                native={"sha256": hashlib.sha256(native_saved).hexdigest()},
            )
        )

        source_webm = video_source["SaveWEBM"]
        source_webm.hidden = SimpleNamespace(extra_pnginfo=None, prompt=None)

        def webm_summary(path: Path, passthrough: object) -> dict[str, object]:
            with av.open(os.fspath(path)) as container:
                stream = container.streams.video[0]
                return {
                    "codec": stream.codec_context.codec.canonical_name,
                    "pix_fmt": stream.codec_context.pix_fmt,
                    "dimensions": [stream.width, stream.height],
                    "fps": str(stream.average_rate),
                    "frames": sum(1 for _ in container.decode(video=0)),
                    "passthrough": _array_descriptor(passthrough),
                }

        images = np.zeros((3, 64, 64, 3), dtype=np.float32)
        images[1, :, :, 0] = 1
        images[2, :, :, 1] = 1
        webm_references, webm_native = [], []
        for case in records["SaveWEBM"]["replacement"]["cases"]:
            record = {**records["SaveWEBM"], "replacement": {"cases": [case]}}
            assembly_values = case["nodes"]["assemble"]["values"]
            codec = "av1" if assembly_values["bit_depth"] == "10" else "vp9"
            prefix = f"receipt-{codec}"
            source_inputs = {"images": images, "fps": 3.0, "codec": codec, "crf": 32.0}
            returned = source_webm.execute(torch.from_numpy(images), codec, 3.0, prefix, 32.0)[0]
            webm_references.append(webm_summary(source_root / f"{prefix}_00001_.webm", returned))
            native_images = _mapped_input(record, "images:images", source_inputs)
            assembled = AssembleVideo.execute(
                images=native_images,
                fps=cast(float, _mapped_input(record, "assemble:fps", source_inputs)),
                **assembly_values,
            )["video"]
            output = io.BytesIO()
            save_video_stream(
                assembled,
                output,
                container=cast(str, _mapped_input(record, "container", source_inputs)),
                codec=cast(str, _mapped_input(record, "codec", source_inputs)),
                crf=int(cast(float, _mapped_input(record, "crf_value:value", source_inputs))),
            )
            native_path = native_root / f"{prefix}.webm"
            native_path.write_bytes(output.getvalue())
            webm_native.append(webm_summary(native_path, native_images))
        outputs.extend(
            _write_mapping_receipt(
                root,
                records["SaveWEBM"],
                slug="save-webm-stream-facts",
                parameters={
                    "codecs": ["av1", "vp9"],
                    "crf": 32,
                    "fps": 3,
                    "scope": "encoded stream facts, frame count and unchanged IMAGE output",
                    "uncompared": ["pixel conversion", "alpha", "hidden metadata", "preview"],
                },
                reference=webm_references,
                native=webm_native,
            )
        )

        source_trimmed = video_source["VideoSlice"].execute(source_video, 0.0, 0.0, False)[0]
        native_trimmed = TrimVideo.execute(
            video=native_load,
            start_time=0.0,
            duration=0.0,
            strict_duration=False,
        )["video"]
        trimmed_bytes = io.BytesIO()
        save_video_stream(native_trimmed, trimmed_bytes)
        outputs.extend(
            _write_mapping_receipt(
                root,
                records["Video Slice"],
                slug="video-slice-zero-window",
                parameters={"duration": 0.0, "startTime": 0.0, "strictDuration": False},
                reference={
                    "sha256": hashlib.sha256(
                        video_path.read_bytes() if source_trimmed is source_video else b""
                    ).hexdigest()
                },
                native={"sha256": hashlib.sha256(trimmed_bytes.getvalue()).hexdigest()},
            )
        )

        reference_previews: list[dict[str, object]] = []

        class SourcePreview:
            def save_images(self, images: torch.Tensor, prefix: str) -> dict[str, object]:
                reference_previews.append({"prefix": prefix, "images": _array_value(images)})
                return {"ui": {"images": []}}

        compare = _load_reference_classes(
            comfy_root,
            "comfy_extras/nodes_image_compare.py",
            ("ImageCompare",),
            {"IO": _IO, "nodes": SimpleNamespace(PreviewImage=SourcePreview)},
        )["ImageCompare"]
        a = torch.linspace(0.0, 1.0, 1 * 2 * 3 * 3).reshape(1, 2, 3, 3)
        b = torch.flip(a, (2,))
        native_previews = []
        for _case_id, image_a, image_b in (
            ("both", a, b),
            ("a", a, None),
            ("b", None, b),
            ("neither", None, None),
        ):
            compare.execute(image_a, image_b, None)
            native_result = ImageCompare.execute(
                image_a=None if image_a is None else image_a.numpy(),
                image_b=None if image_b is None else image_b.numpy(),
            )
            for label, image in (("a", image_a), ("b", image_b)):
                if image is not None:
                    native = native_result[f"image_{label}"]
                    native_previews.append(
                        {"prefix": f"comfy.compare.{label}", "images": _array_value(native)}
                    )
        outputs.extend(
            _write_mapping_receipt(
                root,
                records["ImageCompare"],
                slug="image-compare-previews",
                parameters={"cases": ["both", "a", "b", "neither"], "noInputDisposition": "empty"},
                reference={"previews": reference_previews},
                native={"previews": native_previews},
            )
        )
        return outputs


def _video_component_operation_golden(source: Mapping[str, type[Any]]) -> dict[str, object]:
    frame_count = 6
    frames = np.zeros((frame_count, 64, 64, 3), dtype=np.float32)
    for index in range(frame_count):
        frames[index, ..., 0] = index / (frame_count - 1)
        frames[index, ..., 1] = 0.25
        frames[index, ..., 2] = 0.75
    sample_rate = 8000
    timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * 440 * timeline, dtype=np.float32))[None, None, :]
    source_audio = {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}
    source_video = source["CreateVideo"].execute(
        torch.from_numpy(frames),
        6.0,
        source_audio,
        "auto",
        "sRGB",
    )[0]
    source_components = source["GetVideoComponents"].execute(source_video)
    if not torch.equal(cast("torch.Tensor", source_components[0]), torch.from_numpy(frames)):
        raise RuntimeError("ComfyUI source video components changed the input frames")
    source_component_audio = cast("Mapping[str, object]", source_components[1])
    if not torch.equal(
        cast("torch.Tensor", source_component_audio["waveform"]), torch.from_numpy(waveform)
    ):
        raise RuntimeError("ComfyUI source video components changed the input audio")

    native_video = cast(
        "Mapping[str, object]",
        AssembleVideo.execute(
            images=frames,
            fps=6.0,
            audio={"waveform": waveform, "sample_rate": sample_rate},
        )["video"],
    )
    native_components = DisassembleVideo.execute(video=native_video)
    decoded_frames = cast("np.ndarray", native_components["images"])
    decoded_audio = cast("Mapping[str, object]", native_components["audio"])
    decoded_waveform = cast("np.ndarray", decoded_audio["waveform"])
    max_frame_abs = float(np.max(np.abs(decoded_frames - frames)))
    max_audio_abs = float(np.max(np.abs(decoded_waveform - waveform)))
    container = cast("Mapping[str, object]", native_video["probe"])["container"]
    if (
        container is not None
        or decoded_frames.shape != frames.shape
        or native_components["frame_count"] != frame_count
        or native_components["fps"] != 6.0
        or native_components["duration"] != 1.0
        or decoded_audio["sample_rate"] != sample_rate
        or decoded_waveform.shape != waveform.shape
        or native_components["bit_depth"] != source_components[3]
        or native_components["color_space"] != source_components[4]
        or max_frame_abs != 0
        or max_audio_abs != 0
    ):
        raise RuntimeError(
            "native video container/component operation exceeded its declared bounds"
        )
    return {
        "scope": "operation-only-not-mapping-parity",
        "reason": (
            "ComfyUI and Dinkster preserve these lazy source components exactly; this operation "
            "check does not establish translation coverage or encoder parity."
        ),
        "sourceNode": "CreateVideo + GetVideoComponents",
        "sourcePath": "comfy_extras/nodes_video.py",
        "nativeNode": "dinkster.video.assemble + dinkster.video.disassemble",
        "cases": [
            {
                "inputs": {
                    "audioFixture": "float32-mono-half-scale-440hz-sine",
                    "audioSamples": sample_rate,
                    "fps": 6.0,
                    "frameFixture": "float32-rgb-frame-index-and-constant-channels",
                    "frameShape": list(frames.shape),
                    "sampleRate": sample_rate,
                },
                "sourceOutputs": {
                    "audio": _array_descriptor(source_component_audio["waveform"]),
                    "bitDepth": source_components[3],
                    "colorSpace": source_components[4],
                    "fps": source_components[2],
                    "images": _array_descriptor(source_components[0]),
                    "sampleRate": source_component_audio["sample_rate"],
                },
                "nativeObserved": {
                    "audioShape": list(decoded_waveform.shape),
                    "container": container,
                    "duration": native_components["duration"],
                    "fps": native_components["fps"],
                    "frameShape": list(decoded_frames.shape),
                    "sampleRate": decoded_audio["sample_rate"],
                    "withinDeclaredBounds": {
                        "audio": max_audio_abs == 0,
                        "frames": max_frame_abs == 0,
                    },
                },
                "verifiedBounds": {"maxAudioAbs": 0.0, "maxFrameAbs": 0.0},
            }
        ],
    }


class _SourceControlNet:
    def __init__(self) -> None:
        self.hint: torch.Tensor | None = None
        self.strength: float | None = None
        self.window = (0.0, 1.0)
        self.previous: _SourceControlNet | None = None
        self.control_type: tuple[int, ...] | None = None

    def copy(self) -> _SourceControlNet:
        copied = _SourceControlNet()
        copied.hint = self.hint
        copied.strength = self.strength
        copied.window = self.window
        copied.previous = self.previous
        copied.control_type = self.control_type
        return copied

    def set_cond_hint(
        self,
        hint: torch.Tensor,
        strength: float,
        window: tuple[float, float] = (0.0, 1.0),
        **_kwargs: object,
    ) -> _SourceControlNet:
        self.hint = hint
        self.strength = float(strength)
        self.window = window
        return self

    def set_previous_controlnet(self, previous: _SourceControlNet | None) -> None:
        self.previous = previous

    def set_extra_arg(self, name: str, value: list[int]) -> None:
        if name != "control_type":
            raise RuntimeError(f"unexpected source ControlNet argument {name}")
        self.control_type = tuple(value)


class _ReceiptControlPool:
    @staticmethod
    def rid_for(value: object) -> str:
        return cast("Any", value).resource_digest


def _source_control_trace(control: _SourceControlNet | None) -> object:
    if control is None:
        return None
    return {
        "hint": None if control.hint is None else _array_descriptor(control.hint),
        "strength": control.strength,
        "window": list(control.window),
        "previous": _source_control_trace(control.previous),
    }


def _native_control_trace(application: object | None, hints: Mapping[str, torch.Tensor]) -> object:
    if application is None:
        return None
    value = cast("Any", application)
    return {
        "hint": _array_descriptor(hints[value.child_id]),
        "strength": value.strength,
        "window": list(value.window.to_percent_pair()),
        "previous": _native_control_trace(value.previous, hints),
    }


def _source_conditioning_trace(conditioning: object) -> list[dict[str, object]]:
    return [
        {
            "control": _source_control_trace(cast("Any", entry)[1].get("control")),
            "applyToUncond": cast("Any", entry)[1].get("control_apply_to_uncond"),
        }
        for entry in cast("Any", conditioning)
    ]


def _native_conditioning_trace(conditioning: object) -> list[dict[str, object]]:
    controlled = native_arm._controlled_conditioning(conditioning)
    if controlled is None:
        raise RuntimeError("native ControlNet wrapper did not return controlled conditioning")
    hints = {
        entry.child_id: torch.frombuffer(bytearray(entry.hint.data), dtype=torch.float32).reshape(
            entry.hint.shape
        )
        for entry in controlled.binding.entries
    }
    return [
        {
            "control": _native_control_trace(controlled.binding.application, hints),
            "applyToUncond": controlled.binding.apply_to_uncond,
        }
    ]


def _controlnet_loader_trace(
    source_class: type[Any], source_calls: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    from dinkster_inference import component_catalog, component_registry

    source_calls.clear()
    source_output = source_class().load_controlnet("fixture-control-lora.safetensors")[0]
    descriptor = SimpleNamespace(
        id="dinkster.sdxl_control_lora",
        requires_base=True,
        hint_channels=3,
        default_diffusion_dtype=SimpleNamespace(name="float16"),
        checkpoint_loader=None,
        loader="unused",
    )
    plan = SimpleNamespace(source_layout="canonical", component=SimpleNamespace(config=object()))
    selection_calls: list[dict[str, object]] = []

    class Registry:
        def __iter__(self) -> Iterator[object]:
            return iter((descriptor,))

        @staticmethod
        def select(source: object, path: Path, role: str) -> tuple[object, str, object]:
            selection_calls.append(
                {
                    "path": path.as_posix(),
                    "role": role,
                    "selection": descriptor.id.removeprefix("dinkster."),
                }
            )
            return descriptor, role, plan

    with tempfile.TemporaryDirectory(prefix="dinkster-controlnet-loader-receipt-") as directory:
        path = Path(directory) / "fixture-control-lora.safetensors"
        path.write_bytes(b"receipt fixture")
        asset = AssetRef(
            digest=digest_file(path),
            name=path.name,
            size=path.stat().st_size,
            resolver=_FixedResolver(path),
        )
        inference = importlib.import_module("dinkster_inference")
        with ExitStack() as stack:
            stack.enter_context(
                _patched(component_catalog, "default_component_registry", lambda: Registry())
            )
            stack.enter_context(
                _patched(
                    component_registry,
                    "component_plans",
                    lambda _plan: (SimpleNamespace(config=SimpleNamespace(hint_channels=3)),),
                )
            )
            stack.enter_context(
                _patched(inference, "load_safetensors_header", lambda *_args, **_kwargs: object())
            )
            stack.enter_context(
                _patched(
                    native_arm,
                    "resolve_weight_source",
                    lambda _path, **_kwargs: Path(
                        "models/controlnet/fixture-control-lora.safetensors"
                    ),
                )
            )
            native_output = native_arm.GenerationLoadControlNet.execute(control_net_name=asset)[
                "control_net"
            ]
    source = {
        "requestedName": "fixture-control-lora.safetensors",
        "calls": source_calls,
        "returnedControl": isinstance(source_output, _SourceControlNet),
    }
    native = {
        "requestedName": "fixture-control-lora.safetensors",
        "calls": selection_calls,
        "returnedControl": isinstance(native_output, native_arm._NativeControlNetResource),
    }
    return source, native


def _load_controlnet_reference(
    comfy_root: Path,
) -> tuple[dict[str, type[Any]], list[dict[str, object]]]:
    loader_calls: list[dict[str, object]] = []

    def load_controlnet(path: str) -> _SourceControlNet:
        loader_calls.append({"path": path, "role": "controlnet", "selection": "sdxl_control_lora"})
        return _SourceControlNet()

    classes = _load_reference_classes(
        comfy_root,
        "nodes.py",
        ("ControlNetLoader", "ControlNetApply", "ControlNetApplyAdvanced"),
        {
            "comfy": SimpleNamespace(controlnet=SimpleNamespace(load_controlnet=load_controlnet)),
            "folder_paths": SimpleNamespace(
                get_full_path_or_raise=lambda _kind, name: f"models/controlnet/{name}"
            ),
        },
    )
    classes.update(
        _load_reference_classes(
            comfy_root,
            "comfy_extras/nodes_controlnet.py",
            ("SetUnionControlNetType",),
            {
                "UNION_CONTROLNET_TYPES": {
                    "openpose": 0,
                    "depth": 1,
                    "hed/pidi/scribble/ted": 2,
                    "canny/lineart/anime_lineart/mlsd": 3,
                    "normal": 4,
                    "segment": 5,
                    "tile": 6,
                    "repaint": 7,
                }
            },
        )
    )
    return classes, loader_calls


def _controlnet_receipts(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, type[Any]],
    loader_calls: list[dict[str, object]],
) -> list[Path]:
    image = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) / 23.0
    source_base = _SourceControlNet()
    native_base = native_arm._NativeControlNetResource(
        None,
        "blake3:" + "1" * 64,
        "a" * 64,
        "canonical",
        descriptor=SimpleNamespace(id="dinkster.sdxl_control_lora"),
    )
    conditioning = [[torch.zeros((1, 2, 3)), {"tag": "preserved"}]]
    pool = _ReceiptControlPool()
    inference = importlib.import_module("dinkster_inference")
    carrier = inference.make_conditioning_carrier(inference.ConditioningSet(()), ())
    with _patched(native_arm, "default_pool", lambda: pool):
        loader_source, loader_native = _controlnet_loader_trace(
            source["ControlNetLoader"], loader_calls
        )
        outputs = _write_mapping_receipt(
            root,
            records["ControlNetLoader"],
            slug="controlnet-loader",
            parameters={
                "scope": (
                    "pinned wrapper path resolution and mocked architecture selection; no model "
                    "weights are loaded or executed"
                )
            },
            reference=loader_source,
            native=loader_native,
        )

        source_zero = source["ControlNetApply"]().apply_controlnet(
            conditioning, source_base, image, 0.0
        )[0]
        native_zero = native_arm.GenerationApplyControlNet.execute(
            conditioning=carrier, control_net=native_base, image=image, strength=0.0
        )["conditioning"]
        source_applied = source["ControlNetApply"]().apply_controlnet(
            conditioning, source_base, image, 0.75
        )[0]
        native_applied = native_arm.GenerationApplyControlNet.execute(
            conditioning=carrier, control_net=native_base, image=image, strength=0.75
        )["conditioning"]
        outputs.extend(
            _write_mapping_receipt(
                root,
                records["ControlNetApply"],
                slug="controlnet-apply",
                parameters={
                    "scope": (
                        "deprecated wrapper strength-zero identity, copied conditioning metadata, "
                        "positive application strength, hint layout, and unconditional-copy flag"
                    )
                },
                reference={
                    "zeroStrengthIdentity": source_zero is conditioning,
                    "inputUnchanged": "control" not in conditioning[0][1],
                    "baseControlUnchanged": source_base.hint is None,
                    "conditioning": _source_conditioning_trace(source_applied),
                },
                native={
                    "zeroStrengthIdentity": native_zero is carrier,
                    "inputUnchanged": native_arm._controlled_conditioning(carrier) is None,
                    "baseControlUnchanged": native_base.mode is None,
                    "conditioning": _native_conditioning_trace(native_applied),
                },
            )
        )

        positive = [[torch.ones((1, 2, 3)), {"tag": "positive"}]]
        negative = [[torch.full((1, 2, 3), -1.0), {"tag": "negative"}]]
        source_first = source["ControlNetApplyAdvanced"]().apply_controlnet(
            positive, negative, source_base, image, 1.0, 0.0, 0.75
        )
        source_second = source["ControlNetApplyAdvanced"]().apply_controlnet(
            source_first[0], source_first[1], source_base, image, 0.5, 0.25, 1.0
        )
        native_first = native_arm.GenerationApplyControlNetAdvanced.execute(
            positive=carrier,
            negative=carrier,
            control_net=native_base,
            image=image,
            strength=1.0,
            start_percent=0.0,
            end_percent=0.75,
        )
        native_second = native_arm.GenerationApplyControlNetAdvanced.execute(
            positive=native_first["positive"],
            negative=native_first["negative"],
            control_net=native_base,
            image=image,
            strength=0.5,
            start_percent=0.25,
            end_percent=1.0,
        )
        source_advanced_zero = source["ControlNetApplyAdvanced"]().apply_controlnet(
            positive, negative, source_base, image, 0.0, 0.1, 0.9
        )
        native_advanced_zero = native_arm.GenerationApplyControlNetAdvanced.execute(
            positive=carrier,
            negative=carrier,
            control_net=native_base,
            image=image,
            strength=0.0,
            start_percent=0.1,
            end_percent=0.9,
        )
        outputs.extend(
            _write_mapping_receipt(
                root,
                records["ControlNetApplyAdvanced"],
                slug="controlnet-apply-advanced",
                parameters={
                    "scope": (
                        "positive and negative wrapper application, immutable two-control "
                        "chaining, strength-zero identity, hint layout, strengths, and "
                        "start/end windows"
                    )
                },
                reference={
                    "zeroStrengthIdentity": [
                        source_advanced_zero[0] is positive,
                        source_advanced_zero[1] is negative,
                    ],
                    "inputsUnchanged": [
                        "control" not in positive[0][1],
                        "control" not in negative[0][1],
                    ],
                    "baseControlUnchanged": source_base.hint is None,
                    "positive": _source_conditioning_trace(source_second[0]),
                    "negative": _source_conditioning_trace(source_second[1]),
                },
                native={
                    "zeroStrengthIdentity": [
                        native_advanced_zero["positive"] is carrier,
                        native_advanced_zero["negative"] is carrier,
                    ],
                    "inputsUnchanged": [
                        native_arm._controlled_conditioning(carrier) is None,
                        native_arm._controlled_conditioning(carrier) is None,
                    ],
                    "baseControlUnchanged": native_base.mode is None,
                    "positive": _native_conditioning_trace(native_second["positive"]),
                    "negative": _native_conditioning_trace(native_second["negative"]),
                },
            )
        )

        source_union_cases = []
        native_union_cases = []
        for selector in ("auto", "canny/lineart/anime_lineart/mlsd"):
            source_union = source["SetUnionControlNetType"].execute(source_base, selector)[0]
            native_union = native_arm.GenerationSetControlNetUnionType.execute(
                control_net=native_base, type=selector
            )["control_net"]
            native_mode = cast("Any", native_union).mode
            source_union_cases.append(
                {
                    "selector": selector,
                    "modeIndex": None
                    if source_union.control_type == ()
                    else cast("tuple[int, ...]", source_union.control_type)[0],
                    "inputUnchanged": source_base.control_type is None,
                }
            )
            native_union_cases.append(
                {
                    "selector": selector,
                    "modeIndex": None
                    if native_mode is None
                    else inference.SD_CONTROL_MODE_INDEX[native_mode.token],
                    "inputUnchanged": native_base.mode is None,
                }
            )
        outputs.extend(
            _write_mapping_receipt(
                root,
                records["SetUnionControlNetType"],
                slug="set-union-controlnet-type",
                parameters={
                    "scope": (
                        "immutable wrapper selection for automatic mode and the explicit grouped "
                        "Canny mode; no Union model weights are loaded or executed"
                    )
                },
                reference={"cases": source_union_cases},
                native={"cases": native_union_cases},
            )
        )
    return outputs


def _direct_operation_goldens(
    qwen_class: type[Any],
    wandancer_classes: Mapping[str, type[Any]],
    video_classes: Mapping[str, type[Any]],
) -> dict[str, object]:
    qwen_cases = []
    for inputs in (
        {"width": 640, "height": 480, "layers": 3, "batch_size": 2},
        {"width": 32, "height": 16, "layers": 0, "batch_size": 1},
    ):
        latent = qwen_class.execute(**inputs)[0]["samples"]
        reference_output = _array_descriptor(latent)
        native_latent = execute_empty_qwen_image_layered_latent(**inputs)["latent"]
        native_output = _array_descriptor(cast("Mapping[str, object]", native_latent)["samples"])
        if native_output != reference_output:
            raise RuntimeError("Qwen layered-latent source parity failed")
        qwen_cases.append({"inputs": inputs, "output": reference_output})

    images = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2, 1)
    waveform = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    wandancer_cases = []
    for inputs in (
        {"segment_length": 4, "num_segments": 2, "sample_rate": 30},
        {"segment_length": 4, "num_segments": 3, "sample_rate": 30},
    ):
        result = wandancer_classes["WanDancerPadKeyframesList"].execute(
            images,
            inputs["segment_length"],
            inputs["num_segments"],
            {"waveform": waveform, "sample_rate": inputs["sample_rate"]},
        )
        keyframes, masks, audio_segments = result
        reference_outputs = [
            {
                "keyframes": _array_descriptor(keyframes[index]),
                "mask": _array_descriptor(masks[index]),
                "waveform": _array_descriptor(audio_segments[index]["waveform"]),
                "sampleRate": audio_segments[index]["sample_rate"],
            }
            for index in range(len(keyframes))
        ]
        native_segments = plan_wandancer_keyframe_list(
            images.numpy(),
            inputs["segment_length"],
            inputs["num_segments"],
            waveform.numpy(),
            inputs["sample_rate"],
        )
        native_outputs = [
            {
                "keyframes": _array_descriptor(segment.keyframes),
                "mask": _array_descriptor(segment.mask),
                "waveform": _array_descriptor(segment.audio_waveform),
                "sampleRate": segment.sample_rate,
            }
            for segment in native_segments
        ]
        if native_outputs != reference_outputs:
            raise RuntimeError("WanDancer keyframe-list source parity failed")
        wandancer_cases.append(
            {
                "inputs": {
                    **inputs,
                    "images": _array_value(images),
                    "waveform": _array_value(waveform),
                },
                "outputs": reference_outputs,
            }
        )
    return {
        "format": "dinkster-comfy-direct-operation-golden/1",
        "referenceCommit": REFERENCE_COMMIT,
        "operations": [
            {
                "scope": "operation-only-not-family-parity",
                "sourceNode": "EmptyQwenImageLayeredLatentImage",
                "sourcePath": "comfy_extras/nodes_qwen.py",
                "nativeNode": "dinkster.empty_qwen_image_layered_latent",
                "cases": qwen_cases,
            },
            {
                "scope": "operation-only-not-family-parity",
                "sourceNode": "WanDancerPadKeyframesList",
                "sourcePath": "comfy_extras/nodes_wandancer.py",
                "nativeNode": "dinkster.wandancer.pad_keyframes_list",
                "cases": wandancer_cases,
            },
            _video_component_operation_golden(video_classes),
        ],
    }


def _generate(comfy_root: Path, receipt_root: Path, direct_golden: Path) -> list[Path]:
    resolved = _git(comfy_root, "rev-parse", f"{REFERENCE_COMMIT}^{{commit}}").decode().strip()
    if resolved != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI does not contain {REFERENCE_COMMIT}")
    vision_resolved = (
        _git(comfy_root, "rev-parse", f"{VISION_REFERENCE_COMMIT}^{{commit}}").decode().strip()
    )
    if vision_resolved != VISION_REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI does not contain {VISION_REFERENCE_COMMIT}")
    reference_input_impl, reference_types = _load_reference_api(comfy_root)
    reference_folder_functions = _load_reference_functions(
        comfy_root,
        "folder_paths.py",
        ("is_within_directory", "get_save_image_path"),
        {"logging": logging, "os": os, "time": time},
    )
    reference_folder_paths = SimpleNamespace(
        exists_annotated_filepath=_ReferenceFolderPaths.exists_annotated_filepath,
        get_annotated_filepath=_ReferenceFolderPaths.get_annotated_filepath,
        get_output_directory=_ReferenceFolderPaths.get_output_directory,
        get_save_image_path=reference_folder_functions["get_save_image_path"],
    )

    strings = _load_reference_classes(
        comfy_root,
        "comfy_extras/nodes_string.py",
        (
            "StringConcatenate",
            "StringSubstring",
            "StringLength",
            "CaseConverter",
            "StringTrim",
            "StringReplace",
            "StringContains",
            "StringCompare",
            "RegexMatch",
            "RegexReplace",
        ),
        {"re": re},
    )
    switch = _load_reference_classes(
        comfy_root, "comfy_extras/nodes_logic.py", ("SwitchNode",), {}
    )["SwitchNode"]
    rebatch = _load_reference_classes(
        comfy_root, "comfy_extras/nodes_rebatch.py", ("ImageRebatch",), {"torch": torch}
    )["ImageRebatch"]
    qwen = _load_reference_classes(
        comfy_root,
        "comfy_extras/nodes_qwen.py",
        ("EmptyQwenImageLayeredLatentImage",),
        {
            "torch": torch,
            "comfy": SimpleNamespace(
                model_management=SimpleNamespace(intermediate_device=lambda: torch.device("cpu"))
            ),
        },
    )["EmptyQwenImageLayeredLatentImage"]
    wandancer = _load_reference_classes(
        comfy_root,
        "comfy_extras/nodes_wandancer.py",
        ("WanDancerPadKeyframes", "WanDancerPadKeyframesList"),
        {"math": math, "torch": torch},
    )
    image_io = _load_reference_classes(
        comfy_root,
        "nodes.py",
        ("SaveImage", "LoadImage", "LoadImageMask"),
        {
            "Image": Image,
            "ImageOps": ImageOps,
            "ImageSequence": ImageSequence,
            "InputImpl": reference_input_impl,
            "PngInfo": PngInfo,
            "args": SimpleNamespace(disable_metadata=False),
            "comfy": SimpleNamespace(
                model_management=SimpleNamespace(
                    intermediate_device=lambda: torch.device("cpu"),
                    intermediate_dtype=lambda: torch.float32,
                )
            ),
            "folder_paths": reference_folder_paths,
            "hashlib": hashlib,
            "json": json,
            "node_helpers": SimpleNamespace(pillow=lambda function, *args: function(*args)),
            "np": np,
            "os": os,
            "torch": torch,
        },
    )
    trim_audio = _load_reference_classes(
        comfy_root,
        "comfy_extras/nodes_audio.py",
        ("TrimAudioDuration",),
        {"IO": _IO},
    )["TrimAudioDuration"]
    primitive_boolean = _load_reference_classes(
        comfy_root,
        "comfy_extras/nodes_primitive.py",
        ("Boolean",),
        {},
    )["Boolean"]
    tiled_vae = _load_reference_classes(
        comfy_root,
        "nodes.py",
        ("VAEDecodeTiled", "VAEEncodeTiled"),
        {"torch": torch},
    )
    seedvr2 = _seedvr2_source_classes(comfy_root)
    video = _load_reference_classes(
        comfy_root,
        "comfy_extras/nodes_video.py",
        ("CreateVideo", "GetVideoComponents"),
        {
            "Fraction": Fraction,
            "Input": SimpleNamespace(Image=object, Audio=object, Video=object),
            "InputImpl": reference_input_impl,
            "Optional": Optional,
            "Types": reference_types,
        },
    )
    controlnet, control_loader_calls = _load_controlnet_reference(comfy_root)

    records = _mapping_records()
    group_records = _group_records()
    required_records = set(strings) | {
        "ComfyMathExpression",
        "ComfySwitchNode",
        "ControlNetApply",
        "ControlNetApplyAdvanced",
        "ControlNetLoader",
        "CreateVideo",
        "GetVideoComponents",
        "ImageCompare",
        "KSampler",
        "LoadImage",
        "LoadImageMask",
        "LoadVideo",
        "PrimitiveBoolean",
        "RebatchImages",
        "ResizeImageMaskNode",
        "SaveImage",
        "SaveVideo",
        "SeedVR2Conditioning",
        "SeedVR2PostProcessing",
        "SeedVR2Preprocess",
        "SeedVR2TemporalChunk",
        "SeedVR2TemporalMerge",
        "SetUnionControlNetType",
        "TrimAudioDuration",
        "UNETLoader",
        "VAEDecodeTiled",
        "VAEEncodeTiled",
        "VAELoader",
        "Video Slice",
    }
    if not required_records <= records.keys():
        missing = sorted(required_records - records.keys())
        raise RuntimeError(f"missing maintained mappings: {missing}")
    required_group_records = {
        "remove-background-birefnet",
        "rtdetr-detect-fp16",
        "sam3-text-detection",
        "sam3-video-track-initial-mask",
    }
    if not required_group_records <= group_records.keys():
        missing = sorted(required_group_records - group_records.keys())
        raise RuntimeError(f"missing maintained group mappings: {missing}")
    outputs = _string_receipts(receipt_root, records, strings)
    outputs.extend(_switch_receipt(receipt_root, records["ComfySwitchNode"], switch))
    outputs.extend(_rebatch_receipt(receipt_root, records["RebatchImages"], rebatch))
    outputs.extend(
        _math_expression_receipt(
            receipt_root,
            records["ComfyMathExpression"],
            comfy_root,
        )
    )
    outputs.extend(_image_receipts(receipt_root, records, image_io))
    outputs.extend(_image_batch_pad_receipts(receipt_root, records, comfy_root))
    outputs.extend(_save_image_receipt(receipt_root, records["SaveImage"], image_io["SaveImage"]))
    outputs.extend(_trim_audio_receipt(receipt_root, records["TrimAudioDuration"], trim_audio))
    outputs.extend(_group_receipts(receipt_root, group_records))
    outputs.extend(
        _primitive_boolean_receipt(
            receipt_root,
            records["PrimitiveBoolean"],
            primitive_boolean,
        )
    )
    outputs.extend(
        _resize_image_mask_receipt(
            receipt_root,
            records["ResizeImageMaskNode"],
            comfy_root,
        )
    )
    outputs.extend(_unet_loader_receipt(receipt_root, records["UNETLoader"], comfy_root))
    outputs.extend(_vae_loader_receipt(receipt_root, records["VAELoader"], comfy_root))
    outputs.extend(_ksampler_receipt(receipt_root, records["KSampler"], comfy_root))
    outputs.extend(_tiled_vae_receipts(receipt_root, records, tiled_vae))
    outputs.extend(_seedvr2_receipts(receipt_root, records, seedvr2))
    outputs.extend(_controlnet_receipts(receipt_root, records, controlnet, control_loader_calls))
    outputs.extend(_current_media_receipts(receipt_root, records, comfy_root))
    outputs.extend(_layer_receipts(receipt_root, records))
    outputs.extend(_audio_alias_receipts(receipt_root))
    outputs.extend(_control_aux_receipts(receipt_root))
    impact_records = _mapping_records("comfyui-impact-pack")
    impact_record = impact_records.get("regional-sampler-single-region")
    if impact_record is None:
        raise RuntimeError("missing maintained Impact regional sampling group")
    outputs.extend(_impact_regional_receipt(receipt_root, impact_record))
    inspire_records = _mapping_records("comfyui-inspire-pack")
    inspire_record = inspire_records.get("ScheduledCFGGuider //Inspire")
    if inspire_record is None:
        raise RuntimeError("missing maintained Inspire Pack ScheduledCFGGuider mapping")
    outputs.extend(_inspire_scheduled_cfg_receipt(receipt_root, inspire_record))
    outputs.extend(
        _trellis2_official_receipts(
            receipt_root,
            comfy_root,
            _workflow_templates_root(),
        )
    )
    _canonical_write(direct_golden, _direct_operation_goldens(qwen, wandancer, video))
    outputs.append(direct_golden)
    return outputs


def _tree_digest(paths: Sequence[Path], roots: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    entries = [
        (next(path.relative_to(root) for root in roots if path.is_relative_to(root)), path)
        for path in paths
    ]
    for relative, path in sorted(entries, key=lambda value: value[0].as_posix()):
        digest.update(relative.as_posix().encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    comfy_root = _comfy_root()
    if not args.check:
        outputs = _generate(comfy_root, RECEIPT_ROOT, DIRECT_GOLDEN)
        print(_tree_digest(outputs, (RECEIPT_ROOT, DIRECT_GOLDEN.parent)))
        return

    with tempfile.TemporaryDirectory(prefix="dinkster-source-parity-") as directory:
        temporary = Path(directory)
        receipt_root = temporary / "receipts"
        direct_golden = temporary / DIRECT_GOLDEN.name
        generated = _generate(comfy_root, receipt_root, direct_golden)
        expected_receipts = {
            RECEIPT_ROOT / path.relative_to(receipt_root)
            for path in generated
            if path.is_relative_to(receipt_root)
        }
        committed_receipts = {path for path in RECEIPT_ROOT.rglob("*") if path.is_file()}
        if committed_receipts != expected_receipts:
            extra = sorted(committed_receipts - expected_receipts)
            missing = sorted(expected_receipts - committed_receipts)
            details = [
                *(f"extra {path.relative_to(REPO)}" for path in extra),
                *(f"missing {path.relative_to(REPO)}" for path in missing),
            ]
            raise SystemExit(
                "committed source parity receipt tree differs from generated outputs: "
                + ", ".join(details)
            )
        for path in generated:
            if path.is_relative_to(receipt_root):
                destination = RECEIPT_ROOT / path.relative_to(receipt_root)
            else:
                destination = DIRECT_GOLDEN
            if not destination.is_file() or destination.read_bytes() != path.read_bytes():
                raise SystemExit(f"generated source parity artifact is stale: {destination}")
        print(_tree_digest(generated, (receipt_root, temporary)))


if __name__ == "__main__":
    main()
