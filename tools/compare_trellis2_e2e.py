#!/usr/bin/env python3
"""Compare matched ComfyUI and Dinkster TRELLIS.2 execution tensors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from scipy.spatial import cKDTree  # pyright: ignore[reportMissingImports]


def numeric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    ref = reference.double().reshape(-1)
    cand = candidate.double().reshape(-1)
    delta = cand - ref
    denominator = float(torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand))
    cosine = float(torch.dot(ref, cand)) / denominator if denominator else 1.0
    return {
        "shape": list(reference.shape),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(delta.square()))) if delta.numel() else 0.0,
        "cosine": cosine,
    }


def occupancy(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.bool().reshape(-1)
    cand = candidate.bool().reshape(-1)
    if ref.shape != cand.shape:
        return {
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    intersection = int(torch.count_nonzero(ref & cand))
    union = int(torch.count_nonzero(ref | cand))
    return {
        "reference_occupied": int(torch.count_nonzero(ref)),
        "candidate_occupied": int(torch.count_nonzero(cand)),
        "intersection": intersection,
        "union": union,
        "iou": intersection / union if union else 1.0,
    }


def coordinate_keys(coordinates: torch.Tensor) -> np.ndarray:
    array = coordinates.numpy(force=True)
    array = np.ascontiguousarray(array.astype(np.int32, copy=False))
    dtype = np.dtype([(f"c{i}", np.int32) for i in range(array.shape[1])])
    return array.view(dtype).reshape(-1)


def sparse(
    reference_coordinates: torch.Tensor,
    reference_features: torch.Tensor,
    candidate_coordinates: torch.Tensor,
    candidate_features: torch.Tensor,
) -> dict[str, Any]:
    ref_keys = coordinate_keys(reference_coordinates)
    cand_keys = coordinate_keys(candidate_coordinates)
    common, ref_indices, cand_indices = np.intersect1d(
        ref_keys, cand_keys, assume_unique=True, return_indices=True
    )
    union = reference_coordinates.shape[0] + candidate_coordinates.shape[0] - common.shape[0]
    result: dict[str, Any] = {
        "reference_points": reference_coordinates.shape[0],
        "candidate_points": candidate_coordinates.shape[0],
        "common_points": common.shape[0],
        "union_points": union,
        "coordinate_iou": common.shape[0] / union if union else 1.0,
        "reference_coverage": common.shape[0] / reference_coordinates.shape[0]
        if reference_coordinates.shape[0]
        else 1.0,
        "candidate_coverage": common.shape[0] / candidate_coordinates.shape[0]
        if candidate_coordinates.shape[0]
        else 1.0,
    }
    if common.shape[0]:
        result["common_features"] = numeric(
            reference_features[torch.from_numpy(ref_indices)],
            candidate_features[torch.from_numpy(cand_indices)],
        )
    return result


def sampled_rows(value: torch.Tensor, limit: int = 200_000) -> torch.Tensor:
    if value.shape[0] <= limit:
        return value
    indices = torch.linspace(0, value.shape[0] - 1, limit, dtype=torch.int64)
    return value[indices]


def sparse_nearest(
    reference_coordinates: torch.Tensor,
    reference_features: torch.Tensor,
    candidate_coordinates: torch.Tensor,
    candidate_features: torch.Tensor,
) -> dict[str, Any]:
    ref = sampled_rows(torch.cat((reference_coordinates, reference_features), dim=1))
    cand = sampled_rows(torch.cat((candidate_coordinates, candidate_features), dim=1))
    coordinate_width = reference_coordinates.shape[1]
    ref_coordinates = ref[:, :coordinate_width].double().numpy(force=True)
    cand_coordinates = cand[:, :coordinate_width].double().numpy(force=True)
    ref_features = ref[:, coordinate_width:]
    cand_features = cand[:, coordinate_width:]
    ref_distances, ref_indices = cKDTree(cand_coordinates).query(ref_coordinates, workers=-1)
    cand_distances, cand_indices = cKDTree(ref_coordinates).query(cand_coordinates, workers=-1)
    return {
        "reference_sample": ref.shape[0],
        "candidate_sample": cand.shape[0],
        "reference_to_candidate_coordinate_mean": float(np.mean(ref_distances)),
        "reference_to_candidate_coordinate_p95": float(np.percentile(ref_distances, 95)),
        "candidate_to_reference_coordinate_mean": float(np.mean(cand_distances)),
        "candidate_to_reference_coordinate_p95": float(np.percentile(cand_distances, 95)),
        "reference_to_candidate_features": numeric(
            ref_features, cand_features[torch.from_numpy(ref_indices)]
        ),
        "candidate_to_reference_features": numeric(
            cand_features, ref_features[torch.from_numpy(cand_indices)]
        ),
    }


def sampled_vertices(vertices: torch.Tensor, limit: int = 200_000) -> np.ndarray:
    array = vertices.double().numpy(force=True)
    if array.shape[0] <= limit:
        return array
    indices = np.linspace(0, array.shape[0] - 1, limit, dtype=np.int64)
    return array[indices]


def mesh(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = sampled_vertices(reference)
    cand = sampled_vertices(candidate)
    ref_to_cand = cKDTree(cand).query(ref, workers=-1)[0]
    cand_to_ref = cKDTree(ref).query(cand, workers=-1)[0]
    all_vertices = np.concatenate((ref, cand), axis=0)
    scale = float(np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0)))
    chamfer = float((np.mean(ref_to_cand**2) + np.mean(cand_to_ref**2)) / 2.0)
    return {
        "reference_vertices": reference.shape[0],
        "candidate_vertices": candidate.shape[0],
        "reference_sample": ref.shape[0],
        "candidate_sample": cand.shape[0],
        "bbox_reference": [ref.min(axis=0).tolist(), ref.max(axis=0).tolist()],
        "bbox_candidate": [cand.min(axis=0).tolist(), cand.max(axis=0).tolist()],
        "chamfer_squared": chamfer,
        "normalized_chamfer_squared": chamfer / (scale * scale) if scale else 0.0,
        "reference_to_candidate_mean": float(np.mean(ref_to_cand)),
        "reference_to_candidate_p95": float(np.percentile(ref_to_cand, 95)),
        "candidate_to_reference_mean": float(np.mean(cand_to_ref)),
        "candidate_to_reference_p95": float(np.percentile(cand_to_ref, 95)),
    }


def faces(vertices: torch.Tensor, value: torch.Tensor) -> dict[str, Any]:
    valid = value.numel() == 0 or (int(value.min()) >= 0 and int(value.max()) < vertices.shape[0])
    return {
        "count": value.shape[0],
        "valid_indices": valid,
        "degenerate": int(
            torch.count_nonzero(
                (value[:, 0] == value[:, 1])
                | (value[:, 1] == value[:, 2])
                | (value[:, 0] == value[:, 2])
            )
        ),
    }


def load(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(
        isinstance(item, torch.Tensor) for item in value.values()
    ):
        raise TypeError(f"{path} is not a tensor evidence dictionary")
    return value


def finite(result: object) -> bool:
    if isinstance(result, dict):
        return all(finite(value) for value in result.values())
    if isinstance(result, list):
        return all(finite(value) for value in result)
    return not isinstance(result, float) or math.isfinite(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = load(args.reference)
    candidate = load(args.candidate)
    if reference.keys() != candidate.keys():
        raise RuntimeError("reference and candidate tensor sets differ")

    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "conditioning_512": numeric(reference["conditioning_512"], candidate["conditioning_512"]),
        "conditioning_1024": numeric(
            reference["conditioning_1024"], candidate["conditioning_1024"]
        ),
        "structure": numeric(reference["structure"], candidate["structure"]),
        "occupancy": occupancy(reference["occupancy"], candidate["occupancy"]),
        "shape": sparse(
            reference["shape_coordinates"],
            reference["shape_features"],
            candidate["shape_coordinates"],
            candidate["shape_features"],
        ),
        "mesh": mesh(reference["mesh_vertices"], candidate["mesh_vertices"]),
        "reference_faces": faces(reference["mesh_vertices"], reference["mesh_faces"]),
        "candidate_faces": faces(candidate["mesh_vertices"], candidate["mesh_faces"]),
        "texture": sparse(
            reference["texture_coordinates"],
            reference["texture_features"],
            candidate["texture_coordinates"],
            candidate["texture_features"],
        ),
        "pbr": sparse(
            reference["pbr_coordinates"],
            reference["pbr_features"],
            candidate["pbr_coordinates"],
            candidate["pbr_features"],
        ),
        "pbr_nearest": sparse_nearest(
            reference["pbr_coordinates"],
            reference["pbr_features"],
            candidate["pbr_coordinates"],
            candidate["pbr_features"],
        ),
    }
    if not finite(result):
        raise RuntimeError("comparison produced a non-finite metric")
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
