"""Native Gaussian splat inspection and transforms."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    SPLAT_TYPE,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_splat,
    report_progress,
)

from .camera_info import matrix_quaternion


def _quaternion_product(torch: Any, a: Any, b: Any) -> Any:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quaternion_matrix(torch: Any, quaternion: Any) -> Any:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


class TransformSplat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TransformSplat",
            display_name="Transform Splat",
            category="3d/splat",
            description=(
                "Translate, rotate, and scale a gaussian splat. "
                "Non-uniform scale also reshapes every individual splat, slower process."
            ),
            search_terms=("move splat", "rotate splat", "scale splat", "gaussian transform"),
            inputs=(InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),)
            + tuple(
                InputSpec(
                    f"{operation}_{axis}",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=default,
                    widget=NumberWidget(min=minimum, max=maximum, step=step),
                )
                for operation, default, minimum, maximum, step in (
                    ("translate", 0.0, -100.0, 100.0, 0.01),
                    ("rotate", 0.0, -360.0, 360.0, 1.0),
                    ("scale", 1.0, 0.01, 100.0, 0.01),
                )
                for axis in "xyz"
            ),
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        splat: object,
        translate_x: float,
        translate_y: float,
        translate_z: float,
        rotate_x: float,
        rotate_y: float,
        rotate_z: float,
        scale_x: float,
        scale_y: float,
        scale_z: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        value = materialize_splat(splat)
        pos = value.positions
        device, dtype = pos.device, pos.dtype
        angles = np.radians([rotate_x, rotate_y, rotate_z]) / 2.0
        c, s = np.cos(angles), np.sin(angles)
        # The source composes in float32 before converting to the input dtype/device.
        qx = torch.tensor([c[0], s[0], 0.0, 0.0], dtype=torch.float32)
        qy = torch.tensor([c[1], 0.0, s[1], 0.0], dtype=torch.float32)
        qz = torch.tensor([c[2], 0.0, 0.0, s[2]], dtype=torch.float32)
        rotation = _quaternion_product(torch, _quaternion_product(torch, qz, qy), qx).to(
            device=device, dtype=dtype
        )
        matrix = _quaternion_matrix(torch, rotation[None])[0]
        diagonal = torch.tensor([scale_x, scale_y, scale_z], dtype=dtype, device=device)
        transform = diagonal[:, None] * matrix
        translation = torch.tensor(
            [translate_x, translate_y, translate_z], dtype=dtype, device=device
        )
        positions = pos @ transform.T + translation
        if scale_x == scale_y == scale_z:
            scales = value.scales * scale_x
            rotations = _quaternion_product(
                torch, rotation.expand_as(value.rotations), value.rotations
            )
            rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            axes = _quaternion_matrix(torch, value.rotations.reshape(-1, 4))
            variances = value.scales.reshape(-1, 3).square()
            covariance = (axes * variances[:, None, :]) @ axes.transpose(-1, -2)
            covariance = transform @ covariance @ transform.T
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            eigenvectors = (
                eigenvectors
                * torch.where(torch.linalg.det(eigenvectors) < 0, -1.0, 1.0)[..., None, None]
            )
            scales = eigenvalues.clamp_min(0).sqrt().reshape(value.scales.shape)
            rotations = matrix_quaternion(torch, eigenvectors).reshape(value.rotations.shape)
        return cls.outputs(
            splat=SimpleNamespace(
                positions=positions,
                scales=scales,
                rotations=rotations,
                opacities=value.opacities,
                sh=value.sh,
                counts=value.counts,
            )
        )


class GetSplatCount(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GetSplatCount",
            display_name="Get Splat Count",
            category="3d/splat",
            description="Returns the number of splats summed across the batch.",
            search_terms=("splat count", "gaussian count", "number of splats", "splat info"),
            inputs=(InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
            outputs=(
                OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),
                OutputSpec("count", TypeExpr.concrete(CORE_INT)),
            ),
        )

    @classmethod
    def execute(cls, *, splat: object) -> Mapping[str, object]:
        value = materialize_splat(splat)
        count = sum(
            int(value.counts[index].item())
            if value.counts is not None
            else value.positions.shape[1]
            for index in range(value.positions.shape[0])
        )
        report_progress(1, 1, text=f"{count:,} splats")
        return cls.outputs(splat=splat, count=count)


class MergeSplat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MergeSplat",
            display_name="Merge Splats",
            category="3d/splat",
            description=(
                "Concatenate any number of gaussian splats into one. Unioning several decodes "
                "of the same latent at different seeds densifies the surface, this can improve "
                "surface quality when meshing."
            ),
            search_terms=("union splat", "densify gaussian", "combine splat", "merge gaussian"),
            input_families=(
                InputFamilySpec(
                    "splats",
                    (InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
                    min_members=2,
                    max_members=32,
                    member_prefix="splat",
                    optional_after_minimum=True,
                ),
            ),
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, splats: Mapping[str, object]) -> Mapping[str, object]:
        values = [materialize_splat(value) for value in splats.values() if value is not None]
        if not values:
            raise ValueError("MergeSplat: connect at least one splat.")
        torch = importlib.import_module("torch")
        batch = values[0].positions.shape[0]
        for value in values:
            if value.positions.shape[0] != batch:
                raise ValueError(
                    f"MergeSplat: batch size mismatch ({batch} vs {value.positions.shape[0]})."
                )
        max_degree = max(value.sh.shape[2] for value in values)
        fields: dict[str, list[Any]] = {
            name: [] for name in ("positions", "scales", "rotations", "opacities", "sh")
        }
        lengths: list[int] = []
        for index in range(batch):
            rows: dict[str, list[Any]] = {name: [] for name in fields}
            for value in values:
                end = (
                    int(value.counts[index].item())
                    if value.counts is not None
                    else value.positions.shape[1]
                )
                for name in fields:
                    row = getattr(value, name)[index, :end]
                    if name == "sh" and row.shape[1] < max_degree:
                        row = torch.cat(
                            [
                                row,
                                row.new_zeros(
                                    row.shape[0], max_degree - row.shape[1], row.shape[2]
                                ),
                            ],
                            dim=1,
                        )
                    rows[name].append(row)
            for name in fields:
                fields[name].append(torch.cat(rows[name]))
            lengths.append(fields["positions"][-1].shape[0])
        size = max(lengths)
        result: dict[str, Any] = {}
        for name, batch_rows in fields.items():
            padded = batch_rows[0].new_zeros((batch, size, *batch_rows[0].shape[1:]))
            for index, row in enumerate(batch_rows):
                padded[index, : row.shape[0]] = row
            result[name] = padded
        result["counts"] = (
            torch.tensor(lengths, device=values[0].positions.device, dtype=torch.int64)
            if len(set(lengths)) > 1
            else None
        )
        return cls.outputs(splat=SimpleNamespace(**result))
