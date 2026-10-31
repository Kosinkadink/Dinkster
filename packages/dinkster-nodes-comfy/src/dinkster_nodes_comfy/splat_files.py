"""Serialize gaussian splats through the public 3D file asset boundary."""

from __future__ import annotations

import gzip
import importlib
import logging
import struct
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    FILE_3D_SPLAT_ANY_TYPE,
    SPLAT_TYPE,
    AssetRef,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    digest_bytes,
    materialize_splat,
    produced_output_scratch,
    render_splat_ply,
)

_SH_C0 = 0.28209479177387814


def _ksplat_bytes(value: Any, end: int) -> bytes:
    np = importlib.import_module("numpy")
    xyz = value.positions[0, :end].cpu().numpy().astype(np.float32)
    scale = value.scales[0, :end].cpu().numpy().astype(np.float32)
    rotation = value.rotations[0, :end].cpu().numpy().astype(np.float32)
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-12)
    rgb = np.clip(value.sh[0, :end, 0, :].cpu().numpy().astype(np.float32) * _SH_C0 + 0.5, 0, 1)
    opacity = value.opacities[0, :end].cpu().numpy().astype(np.float32).reshape(end, 1).clip(0, 1)
    rgba = np.round(np.concatenate([rgb, opacity], axis=1) * 255.0).astype(np.uint8)
    records = np.empty(end, dtype=[("f", "<f4", 10), ("c", "u1", 4)])
    records["f"] = np.concatenate([xyz, scale, rotation], axis=1).astype("<f4")
    records["c"] = rgba
    header = bytearray(4096)
    header[1] = 1
    struct.pack_into("<IIII", header, 4, 1, 1, end, end)
    section = bytearray(1024)
    struct.pack_into("<II", section, 0, end, end)
    struct.pack_into("<I", section, 28, end * 44)
    return bytes(header) + bytes(section) + bytes(records.tobytes())


def _spz_bytes(value: Any, end: int) -> bytes:
    np = importlib.import_module("numpy")
    xyz = value.positions[0, :end].cpu().numpy().astype(np.float32)
    signed = np.clip(np.round(xyz * (1 << 12)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
    unsigned = (signed & 0xFFFFFF).astype(np.uint32)
    positions = (
        np.stack([unsigned & 0xFF, (unsigned >> 8) & 0xFF, (unsigned >> 16) & 0xFF], axis=-1)
        .reshape(end, 9)
        .astype(np.uint8)
    )
    opacity = (
        np.round(value.opacities[0, :end].cpu().numpy().astype(np.float32).reshape(end) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    rgb = value.sh[0, :end, 0, :].cpu().numpy().astype(np.float32) * _SH_C0 + 0.5
    colors = np.round(((rgb - 0.5) / (_SH_C0 / 0.15) + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    log_scale = np.log(value.scales[0, :end].cpu().numpy().astype(np.float32).clip(min=1e-9))
    scales = np.round((log_scale + 10.0) * 16.0).clip(0, 255).astype(np.uint8)
    rotation = value.rotations[0, :end].cpu().numpy().astype(np.float32)
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-12)
    rotation[rotation[:, 0] < 0] *= -1.0
    rotations = np.round((rotation[:, 1:4] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    header = struct.pack("<IIIBBBB", 0x5053474E, 2, end, 0, 12, 0, 0)
    return gzip.compress(
        header
        + positions.tobytes()
        + opacity.tobytes()
        + colors.tobytes()
        + scales.tobytes()
        + rotations.tobytes()
    )


class SplatToFile3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SplatToFile3D",
            display_name="Create 3D File (from Splat)",
            search_terms=("gaussian to ply", "splat to file", "export gaussian"),
            category="3d/splat",
            description=(
                "Serialize a gaussian splat to a File3D object for Save / Preview 3D nodes. "
                "Supports one item per batch only."
            ),
            inputs=(
                InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),
                InputSpec(
                    "format",
                    TypeExpr.concrete(CORE_COMBO),
                    default="ply",
                    widget=ComboWidget(options=("ply", "ksplat", "spz")),
                ),
            ),
            outputs=(OutputSpec("model_3d", TypeExpr.concrete(FILE_3D_SPLAT_ANY_TYPE)),),
        )

    @classmethod
    def execute(cls, *, splat: object, format: str = "ply") -> Mapping[str, object]:
        if format not in ("ply", "ksplat", "spz"):
            raise ValueError(f"Unsupported splat format: {format!r}")
        value = materialize_splat(splat)
        if value.positions.shape[0] > 1:
            logging.warning(
                "SplatToFile3D supports one item per batch only. Got %d; using first.",
                value.positions.shape[0],
            )
        end = int(value.counts[0].item()) if value.counts is not None else value.positions.shape[1]
        if end == 0:
            raise ValueError("SplatToFile3D: gaussian is empty")
        data = (
            render_splat_ply(splat)
            if format == "ply"
            else _ksplat_bytes(value, end)
            if format == "ksplat"
            else _spz_bytes(value, end)
        )
        vault = produced_output_scratch()
        if vault is None:
            raise RuntimeError("SplatToFile3D requires a produced-output store")
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        source = AssetRef(
            digest, f"splat.{format}", len(data), "application/octet-stream", resolver=vault
        )
        return cls.outputs(model_3d={"format": format, "source": source})
