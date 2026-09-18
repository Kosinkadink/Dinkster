# pyright: reportPrivateUsage=false

from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_assets import AssetRef
from dinkster_inference import (
    FLOAT16,
    Trellis2ArtifactRole,
    Trellis2FlowRole,
    load_safetensors_header,
    plan_trellis2_artifact,
    trellis2_artifact_runtime_identity,
)
from dinkster_inference_torch import (
    enroll_component,
    load_trellis2_artifact,
    load_trellis2_flow_artifact,
    soft_empty_cache,
)
from dinkster_inference_torch.sparse import (
    make_sparse_support,
    pack_sparse_latent,
    unpack_sparse_latent,
)
from dinkster_inference_torch.trellis2_vae import _Sparse
from gpu_test_gate import require_gpu_tests_enabled

_GOLDEN = Path(__file__).parent / "goldens" / "trellis2_official_goldens.json"
_MODELS = Path(
    os.environ.get(
        "DINKSTER_TRELLIS2_SOURCE_MODELS",
        "/home/kosin/model-artifacts/dinkster-trellis2-1083/source",
    )
)
_ARTIFACTS = {
    "structure-flow": (
        "ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "blake3:c5ef46bf1f7995edda69cd8b4886e953d24ef0a4509b1ec748914520d077c27b",
    ),
    "shape-512": (
        "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "blake3:22b7a243cd96487116d2daa766bd016ef824c223a53f39c1b5c000dadf6bfb81",
    ),
    "shape-1024": (
        "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "blake3:09dec698caaa38c62c1cdfcbc639ff063541d4f67f7d0d4e5b9075034d8130e6",
    ),
    "texture-512": (
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        "blake3:c940162b669c106ea681f129b14bded3d74d703c4bec548152c505db9d3e7995",
    ),
    "texture-1024": (
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        "blake3:f1ac2f9a94c33735ccce8bd1a73235d4f9de944b1a49fe116ea3c16d4da70248",
    ),
    "structure-decoder": (
        "ss_dec_conv3d_16l8_fp16.safetensors",
        "blake3:a33f7a7e3edddf302e4cbbb48b7a909e588afac6393a6a610ded62c6c8dc04b3",
    ),
    "shape-decoder": (
        "shape_dec_next_dc_f16c32_fp16.safetensors",
        "blake3:740d2a57022f1a074b5d4b35c4b5f856864fb38a681eac9a4c2990b2d5100045",
    ),
    "texture-decoder": (
        "tex_dec_next_dc_f16c32_fp16.safetensors",
        "blake3:da99065a40faa7602f3fc60c7398806d07803d501ff6de9ce8b39c3f0b897991",
    ),
}
_FLOW_ROLES: dict[str, Trellis2FlowRole] = {
    "structure": "structure",
    "shape-512": "shape-512",
    "shape-1024": "shape",
    "texture-512": "texture-512",
    "texture-1024": "texture",
}
_FLOW_LIMITS = {
    # The source golden uses xFormers while the standard CUDA gate uses Dinkster's
    # selected optimized backend. A matched xFormers run made every other seam
    # bit-exact; these per-case limits cover the measured 30-block backend drift.
    "structure": (0.16, 0.014),
    "shape-512": (0.030, 0.0065),
    "shape-1024": (0.039, 0.0070),
    "texture-512": (0.036, 0.0062),
    "texture-1024": (0.034, 0.0071),
}


class _Resolver:
    def __init__(self, digest: str, path: Path) -> None:
        self.digest = digest
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == self.digest else None


def _tensor(record: dict[str, Any]) -> torch.Tensor:
    assert record["encoding"] == "base64+zlib"
    raw = zlib.decompress(base64.b64decode(cast(str, record["data"])))
    assert hashlib.sha256(raw).hexdigest() == record["raw_sha256"]
    dtype = getattr(torch, cast(str, record["dtype"]))
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(record["shape"]).clone()


def _asset(
    role: str,
    facts: dict[str, Any],
) -> tuple[Path, AssetRef]:
    name, digest = _ARTIFACTS[role]
    path = _MODELS / name
    assert path.stat().st_size == facts["byte_size"]
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == facts["sha256"]
    assert cast(str, facts["url"]).startswith("https://huggingface.co/")
    return path, AssetRef(
        digest,
        name,
        cast(int, facts["byte_size"]),
        resolver=_Resolver(digest, path),
    )


def _release(mechanism: Any, module: torch.nn.Module) -> None:
    mechanism.unload()
    del module
    gc.collect()
    soft_empty_cache(torch.device("cuda:0"))


def _error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual.float().cpu().sub(expected.float()).abs()
    return difference.max().item(), difference.square().mean().sqrt().item()


def _coordinate_index(coordinates: torch.Tensor) -> dict[tuple[int, int, int, int], int]:
    return {
        (int(row[0]), int(row[1]), int(row[2]), int(row[3])): index
        for index, row in enumerate(coordinates.cpu())
    }


def _check_flow(
    name: str,
    record: dict[str, Any],
    artifact_facts: dict[str, dict[str, Any]],
) -> None:
    artifact_role = cast(str, record["artifact_role"])
    path, asset = _asset(artifact_role, artifact_facts[artifact_role])
    loaded = load_trellis2_flow_artifact(
        path,
        asset=asset,
        expected_role=_FLOW_ROLES[name],
        compute_dtype=torch.bfloat16,
    )
    module = loaded.module.eval()
    mechanism = enroll_component(module, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    inputs = {
        key: _tensor(cast(dict[str, Any], value)).cuda()
        for key, value in cast(dict[str, Any], record["inputs"]).items()
    }
    expected = _tensor(cast(dict[str, Any], record["output"]))
    with torch.inference_mode():
        if name == "structure":
            latent = torch.nn.functional.pad(inputs["latent"], (0, 0, 0, 0, 0, 0, 0, 24))
            output = module(latent, inputs["timestep"], inputs["context"])
            assert type(output) is torch.Tensor
            actual = output[:, :8]
        else:
            resolution = 512 if name.endswith("512") else 1024
            coordinates = inputs["coordinates"]
            support = make_sparse_support(
                coordinates,
                (coordinates.shape[0],),
                resolution,
                (-0.5, -0.5, -0.5),
                (1 / resolution, 1 / resolution, 1 / resolution),
            )
            features = inputs["latent"]
            if name.startswith("texture"):
                features = torch.cat((features, inputs["shape"]), dim=-1)
            output = module(
                pack_sparse_latent(support, features),
                inputs["timestep"],
                inputs["context"],
            )
            _, actual = unpack_sparse_latent(output)
    maximum, root_mean_square = _error(actual, expected)
    maximum_limit, root_mean_square_limit = _FLOW_LIMITS[name]
    assert maximum <= maximum_limit
    assert root_mean_square <= root_mean_square_limit
    _release(mechanism, module)


def _load_decoder(
    role: Trellis2ArtifactRole,
    artifact_facts: dict[str, dict[str, Any]],
) -> tuple[torch.nn.Module, Any]:
    path, asset = _asset(role, artifact_facts[role])
    source = load_safetensors_header(
        path,
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    planned = plan_trellis2_artifact(source, role=role, path=path)
    identity = trellis2_artifact_runtime_identity(planned, FLOAT16)
    loaded = load_trellis2_artifact(
        path,
        asset=asset,
        expected_role=role,
        expected_identity=identity,
        compute_dtype=torch.float16,
    )
    module = loaded.module.eval()
    mechanism = enroll_component(module, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    return module, mechanism


def _check_decoders(
    records: dict[str, dict[str, Any]],
    artifact_facts: dict[str, dict[str, Any]],
) -> None:
    structure_record = records["structure"]
    structure, mechanism = _load_decoder("structure-decoder", artifact_facts)
    structure_input = _tensor(cast(dict[str, Any], structure_record["input"])).cuda()
    with torch.inference_mode():
        structure_actual = structure(structure_input)
    structure_expected = _tensor(cast(dict[str, Any], structure_record["output"]))
    maximum, root_mean_square = _error(structure_actual, structure_expected)
    assert maximum <= 0.24
    assert root_mean_square <= 0.036
    _release(mechanism, structure)

    shape_record = records["shape"]
    shape, mechanism = _load_decoder("shape-decoder", artifact_facts)
    shape_input = cast(dict[str, Any], shape_record["input"])
    coordinates = _tensor(cast(dict[str, Any], shape_input["coordinates"])).cuda()
    features = _tensor(cast(dict[str, Any], shape_input["features"])).cuda()
    support = make_sparse_support(
        coordinates,
        (coordinates.shape[0],),
        2,
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.5),
    )
    with torch.inference_mode():
        shape_actual, subdivisions = shape(pack_sparse_latent(support, features))
    shape_output = cast(dict[str, Any], shape_record["output"])
    expected_coordinates = _tensor(cast(dict[str, Any], shape_output["coordinates"]))
    expected_features = _tensor(cast(dict[str, Any], shape_output["features"]))
    actual_index = _coordinate_index(shape_actual.support.coordinates)
    expected_index = _coordinate_index(expected_coordinates)
    common = sorted(actual_index.keys() & expected_index.keys())
    union = actual_index.keys() | expected_index.keys()
    assert len(common) / len(union) >= 0.99
    aligned_actual = torch.stack(
        [shape_actual.feats[actual_index[coordinate]] for coordinate in common]
    )
    aligned_expected = torch.stack(
        [expected_features[expected_index[coordinate]] for coordinate in common]
    )
    difference = aligned_actual.float().cpu().sub(aligned_expected.float()).abs()
    # FlexGEMM and ComfyUI's bounded torch sparse convolution differ at a few
    # subdivision sign boundaries; the matched output keeps over 99% topology.
    assert difference.max().item() <= 20.0
    assert difference.mean().item() <= 0.05
    assert len(subdivisions) == 4
    _release(mechanism, shape)

    texture_record = records["texture"]
    texture, mechanism = _load_decoder("texture-decoder", artifact_facts)
    texture_input = cast(dict[str, Any], texture_record["input"])
    coordinates = _tensor(cast(dict[str, Any], texture_input["coordinates"])).cuda()
    features = _tensor(cast(dict[str, Any], texture_input["features"])).cuda()
    support = make_sparse_support(
        coordinates,
        (coordinates.shape[0],),
        2,
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.5),
    )
    guides: list[_Sparse] = []
    for index, raw_guide in enumerate(cast(list[dict[str, Any]], texture_record["guides"])):
        guide_coordinates = _tensor(raw_guide["coordinates"]).cuda()
        guide_features = _tensor(raw_guide["features"]).cuda()
        resolution = 2 ** (index + 1)
        guide_support = make_sparse_support(
            guide_coordinates,
            (guide_coordinates.shape[0],),
            resolution,
            (-0.5, -0.5, -0.5),
            (1 / resolution, 1 / resolution, 1 / resolution),
        )
        guides.append(_Sparse(guide_support, guide_features))
    with torch.inference_mode():
        texture_actual, _ = texture(
            pack_sparse_latent(support, features),
            tuple(guides),
        )
    texture_output = cast(dict[str, Any], texture_record["output"])
    assert torch.equal(
        texture_actual.support.coordinates.cpu(),
        _tensor(cast(dict[str, Any], texture_output["coordinates"])),
    )
    maximum, root_mean_square = _error(
        texture_actual.feats,
        _tensor(cast(dict[str, Any], texture_output["features"])),
    )
    assert maximum <= 0.008
    assert root_mean_square <= 0.0010
    _release(mechanism, texture)


@pytest.mark.skipif(
    not all((_MODELS / name).exists() for name, _digest in _ARTIFACTS.values()),
    reason="official TRELLIS.2 source artifact set is unavailable",
)
def test_official_trellis2_components_match_pinned_source_goldens() -> None:
    require_gpu_tests_enabled()
    golden = cast(dict[str, Any], json.loads(_GOLDEN.read_text()))
    assert golden["format"] == 1
    assert golden["reference"] == {
        "attention": "xformers memory-efficient attention",
        "commit": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
        "device": "NVIDIA GeForce RTX 4090",
        "repository": "https://github.com/microsoft/TRELLIS.2",
        "sparse_convolution": "FlexGEMM",
        "torch": "2.6.0+cu124",
        "xformers": None,
    }
    artifact_facts = cast(dict[str, dict[str, Any]], golden["artifacts"])
    assert set(artifact_facts) == set(_ARTIFACTS)
    flow_records = cast(dict[str, dict[str, Any]], golden["flows"])
    for name in ("structure", "shape-512", "shape-1024", "texture-512", "texture-1024"):
        _check_flow(name, flow_records[name], artifact_facts)
    _check_decoders(
        cast(dict[str, dict[str, Any]], golden["decoders"]),
        artifact_facts,
    )
