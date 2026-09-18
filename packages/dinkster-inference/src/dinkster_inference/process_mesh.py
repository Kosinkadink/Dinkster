"""Torch-free logical process meshes and physical rank placement."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = ["PlacementMap", "ProcessMesh", "ProcessMeshCoordinate", "ProcessMeshError"]


class ProcessMeshError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProcessMeshCoordinate:
    guidance: int
    tp: int
    sp_ulysses: int
    sp_ring: int

    def __post_init__(self) -> None:
        for name, value in (
            ("guidance", self.guidance),
            ("tp", self.tp),
            ("sp_ulysses", self.sp_ulysses),
            ("sp_ring", self.sp_ring),
        ):
            if type(value) is not int or value < 0:
                raise ProcessMeshError(f"{name} coordinate must be an exact int >= 0")


@dataclass(frozen=True, slots=True)
class ProcessMesh:
    """A logical row-major sampling mesh with Ring as its innermost axis."""

    guidance: int
    tp: int
    sp_ulysses: int
    sp_ring: int

    def __post_init__(self) -> None:
        for name, value in (
            ("guidance", self.guidance),
            ("tp", self.tp),
            ("sp_ulysses", self.sp_ulysses),
            ("sp_ring", self.sp_ring),
        ):
            if type(value) is not int or value < 1:
                raise ProcessMeshError(f"{name} axis degree must be an exact int >= 1")

    @property
    def world_size(self) -> int:
        return self.guidance * self.tp * self.sp_ulysses * self.sp_ring

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(range(self.world_size))

    def coordinates(self, rank: int) -> ProcessMeshCoordinate:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ProcessMeshError(f"rank must be an exact int in [0, {self.world_size})")
        guidance, remainder = divmod(rank, self.tp * self.sp_ulysses * self.sp_ring)
        tp, remainder = divmod(remainder, self.sp_ulysses * self.sp_ring)
        sp_ulysses, sp_ring = divmod(remainder, self.sp_ring)
        return ProcessMeshCoordinate(guidance, tp, sp_ulysses, sp_ring)

    def rank_of(self, coordinate: ProcessMeshCoordinate) -> int:
        if type(coordinate) is not ProcessMeshCoordinate:
            raise ProcessMeshError("coordinate must be an exact ProcessMeshCoordinate")
        for name, value, degree in (
            ("guidance", coordinate.guidance, self.guidance),
            ("tp", coordinate.tp, self.tp),
            ("sp_ulysses", coordinate.sp_ulysses, self.sp_ulysses),
            ("sp_ring", coordinate.sp_ring, self.sp_ring),
        ):
            if not 0 <= value < degree:
                raise ProcessMeshError(f"{name} coordinate must be in [0, {degree})")
        return (
            (coordinate.guidance * self.tp + coordinate.tp) * self.sp_ulysses
            + coordinate.sp_ulysses
        ) * self.sp_ring + coordinate.sp_ring

    def guidance_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.guidance == 1:
            return ()
        return tuple(
            tuple(
                self.rank_of(ProcessMeshCoordinate(guidance, tp, ulysses, ring))
                for guidance in range(self.guidance)
            )
            for tp in range(self.tp)
            for ulysses in range(self.sp_ulysses)
            for ring in range(self.sp_ring)
        )

    def tp_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.tp == 1:
            return ()
        return tuple(
            tuple(
                self.rank_of(ProcessMeshCoordinate(guidance, tp, ulysses, ring))
                for tp in range(self.tp)
            )
            for guidance in range(self.guidance)
            for ulysses in range(self.sp_ulysses)
            for ring in range(self.sp_ring)
        )

    def ulysses_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.sp_ulysses == 1:
            return ()
        return tuple(
            tuple(
                self.rank_of(ProcessMeshCoordinate(guidance, tp, ulysses, ring))
                for ulysses in range(self.sp_ulysses)
            )
            for guidance in range(self.guidance)
            for tp in range(self.tp)
            for ring in range(self.sp_ring)
        )

    def ring_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.sp_ring == 1:
            return ()
        return tuple(
            tuple(
                self.rank_of(ProcessMeshCoordinate(guidance, tp, ulysses, ring))
                for ring in range(self.sp_ring)
            )
            for guidance in range(self.guidance)
            for tp in range(self.tp)
            for ulysses in range(self.sp_ulysses)
        )

    def sequence_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.sp_ulysses * self.sp_ring == 1:
            return ()
        return tuple(
            tuple(
                self.rank_of(ProcessMeshCoordinate(guidance, tp, ulysses, ring))
                for ulysses in range(self.sp_ulysses)
                for ring in range(self.sp_ring)
            )
            for guidance in range(self.guidance)
            for tp in range(self.tp)
        )

    def model_parallel_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.tp * self.sp_ulysses * self.sp_ring == 1:
            return ()
        return tuple(
            tuple(
                self.rank_of(ProcessMeshCoordinate(guidance, tp, ulysses, ring))
                for tp in range(self.tp)
                for ulysses in range(self.sp_ulysses)
                for ring in range(self.sp_ring)
            )
            for guidance in range(self.guidance)
        )

    @property
    def subgroup_digest(self) -> str:
        families = (
            ("guidance", self.guidance_groups()),
            ("tp", self.tp_groups()),
            ("sp_ulysses", self.ulysses_groups()),
            ("sp_ring", self.ring_groups()),
            ("sequence", self.sequence_groups()),
            ("model_parallel", self.model_parallel_groups()),
        )
        serialized = ":".join(
            f"{name}={';'.join(','.join(map(str, group)) for group in groups)}"
            for name, groups in families
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def identity_facts(self) -> tuple[str, ...]:
        return (
            "process_mesh_layout=guidance-tp-sp_ulysses-sp_ring.v1",
            (
                f"process_mesh_axes=cfg{self.guidance}xtp{self.tp}"
                f"xu{self.sp_ulysses}xr{self.sp_ring}"
            ),
            f"process_mesh_world_size={self.world_size}",
            f"process_mesh_subgroup_digest={self.subgroup_digest}",
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(":".join(self.identity_facts()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PlacementMap:
    mesh: ProcessMesh
    physical_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.mesh) is not ProcessMesh:
            raise ProcessMeshError("mesh must be an exact ProcessMesh")
        if type(self.physical_ranks) is not tuple or any(
            type(rank) is not int for rank in self.physical_ranks
        ):
            raise ProcessMeshError("physical_ranks must be a tuple of exact ints")
        if len(self.physical_ranks) != self.mesh.world_size:
            raise ProcessMeshError("physical_ranks length must equal the mesh world size")
        if tuple(sorted(self.physical_ranks)) != self.mesh.ranks:
            raise ProcessMeshError(
                "physical_ranks must be a permutation of consecutive integers from zero"
            )

    @property
    def mesh_digest(self) -> str:
        return self.mesh.digest

    @classmethod
    def identity(cls, mesh: ProcessMesh) -> PlacementMap:
        if type(mesh) is not ProcessMesh:
            raise ProcessMeshError("mesh must be an exact ProcessMesh")
        return cls(mesh, mesh.ranks)

    def physical_rank(self, logical_rank: int) -> int:
        if type(logical_rank) is not int or not 0 <= logical_rank < len(self.physical_ranks):
            raise ProcessMeshError(
                f"logical_rank must be an exact int in [0, {len(self.physical_ranks)})"
            )
        return self.physical_ranks[logical_rank]

    def logical_rank(self, physical_rank: int) -> int:
        if type(physical_rank) is not int or not 0 <= physical_rank < len(self.physical_ranks):
            raise ProcessMeshError(
                f"physical_rank must be an exact int in [0, {len(self.physical_ranks)})"
            )
        return self.physical_ranks.index(physical_rank)

    def apply(self, logical_ranks: tuple[int, ...]) -> tuple[int, ...]:
        if type(logical_ranks) is not tuple:
            raise ProcessMeshError("logical_ranks must be a tuple")
        return tuple(self.physical_rank(rank) for rank in logical_ranks)

    def identity_facts(self) -> tuple[str, ...]:
        if self.physical_ranks == tuple(range(len(self.physical_ranks))):
            return ()
        return (f"process_mesh_rank_order={','.join(map(str, self.physical_ranks))}",)
