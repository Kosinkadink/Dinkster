"""Compatibility facade for guidance/Ulysses/Ring process meshes."""

from __future__ import annotations

from dataclasses import dataclass

from .process_mesh import ProcessMesh, ProcessMeshCoordinate, ProcessMeshError

__all__ = ["UspMesh", "UspMeshCoordinate", "UspMeshError"]


class UspMeshError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class UspMeshCoordinate:
    guidance: int
    ulysses: int
    ring: int

    def __post_init__(self) -> None:
        for name, value in (
            ("guidance", self.guidance),
            ("ulysses", self.ulysses),
            ("ring", self.ring),
        ):
            if type(value) is not int:
                raise UspMeshError(f"{name} coordinate must be an exact int")


@dataclass(frozen=True, slots=True)
class UspMesh:
    """A logical USP mesh with tensor parallelism fixed at degree one."""

    guidance: int
    ulysses: int
    ring: int

    def __post_init__(self) -> None:
        try:
            _ = self.process_mesh
        except ProcessMeshError as error:
            raise UspMeshError(str(error).replace("axis degree", "axis size")) from error

    @classmethod
    def build(cls, *, guidance: int, ulysses: int, ring: int) -> UspMesh:
        return cls(guidance, ulysses, ring)

    @property
    def process_mesh(self) -> ProcessMesh:
        return ProcessMesh(self.guidance, 1, self.ulysses, self.ring)

    @property
    def ranks(self) -> tuple[int, ...]:
        return self.process_mesh.ranks

    @property
    def world_size(self) -> int:
        return self.process_mesh.world_size

    def coordinates(self, rank: int) -> UspMeshCoordinate:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise UspMeshError(f"rank must be an exact int in [0, {self.world_size})")
        coordinate = self.process_mesh.coordinates(rank)
        return UspMeshCoordinate(coordinate.guidance, coordinate.sp_ulysses, coordinate.sp_ring)

    def rank_of(self, coordinate: UspMeshCoordinate) -> int:
        if type(coordinate) is not UspMeshCoordinate:
            raise UspMeshError("coordinate must be an exact UspMeshCoordinate")
        for name, value, size in (
            ("guidance", coordinate.guidance, self.guidance),
            ("ulysses", coordinate.ulysses, self.ulysses),
            ("ring", coordinate.ring, self.ring),
        ):
            if not 0 <= value < size:
                raise UspMeshError(f"{name} coordinate must be in [0, {size})")
        return self.process_mesh.rank_of(
            ProcessMeshCoordinate(coordinate.guidance, 0, coordinate.ulysses, coordinate.ring)
        )

    def ring_groups(self) -> tuple[tuple[int, ...], ...]:
        return self.process_mesh.ring_groups()

    def ulysses_groups(self) -> tuple[tuple[int, ...], ...]:
        return self.process_mesh.ulysses_groups()

    def guidance_groups(self) -> tuple[tuple[int, ...], ...]:
        return self.process_mesh.guidance_groups()

    def sequence_groups(self) -> tuple[tuple[int, ...], ...]:
        return self.process_mesh.sequence_groups()

    def identity_facts(self) -> tuple[str, ...]:
        return self.process_mesh.identity_facts()

    @property
    def digest(self) -> str:
        return self.process_mesh.digest
