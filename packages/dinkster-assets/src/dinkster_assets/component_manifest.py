"""Declarative component metadata for combined model assets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from .identity import AssetError
from .json_metadata import freeze_json, thaw_json
from .kind import KIND_MODEL_CHECKPOINT, require_asset_kind


@dataclass(frozen=True)
class AssetComponent:
    """One declared component contained in a combined model asset."""

    kind: str
    architecture: str = ""
    dtype: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        kind = cast("object", self.kind)
        if not isinstance(kind, str):
            raise AssetError("component kind must be a string")
        object.__setattr__(self, "kind", require_asset_kind(kind))
        for name in ("architecture", "dtype"):
            if not isinstance(getattr(self, name), str):
                raise AssetError(f"component {name} must be a string")
        metadata = cast("object", self.metadata)
        if not isinstance(metadata, Mapping):
            raise AssetError("component metadata must be an object")
        object.__setattr__(
            self,
            "metadata",
            cast(
                "Mapping[str, object]",
                freeze_json(cast("Mapping[object, object]", metadata), "component metadata"),
            ),
        )

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"kind": self.kind}
        if self.architecture:
            wire["architecture"] = self.architecture
        if self.dtype:
            wire["dtype"] = self.dtype
        if self.metadata:
            wire["metadata"] = thaw_json(self.metadata)
        return wire

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> AssetComponent:
        metadata = wire.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise AssetError("component metadata must be an object")
        return cls(
            kind=wire.get("kind", ""),  # type: ignore[arg-type]
            architecture=wire.get("architecture", ""),  # type: ignore[arg-type]
            dtype=wire.get("dtype", ""),  # type: ignore[arg-type]
            metadata=cast("Mapping[str, object]", metadata),
        )


@dataclass(frozen=True)
class AssetComponentManifest:
    """An ordered declaration of components contained in one asset."""

    components: tuple[AssetComponent, ...]

    def __post_init__(self) -> None:
        raw_components = cast("object", self.components)
        if not isinstance(raw_components, (list, tuple)):
            raise AssetError("component manifest must be a list")
        components = tuple(cast("list[object] | tuple[object, ...]", raw_components))
        if any(not isinstance(component, AssetComponent) for component in components):
            raise AssetError("component manifest must contain AssetComponent values")
        object.__setattr__(self, "components", cast("tuple[AssetComponent, ...]", components))

    def to_wire(self) -> list[dict[str, object]]:
        return [component.to_wire() for component in self.components]

    @classmethod
    def from_wire(cls, wire: object) -> AssetComponentManifest:
        if not isinstance(wire, Sequence) or isinstance(wire, (str, bytes)):
            raise AssetError("component manifest must be a list")
        components: list[AssetComponent] = []
        for entry in cast("Sequence[object]", wire):
            if not isinstance(entry, Mapping):
                raise AssetError("component manifest entries must be objects")
            components.append(AssetComponent.from_wire(cast("Mapping[str, object]", entry)))
        return cls(tuple(components))


def asset_kind_matches(
    asset_kind: str,
    accepted_kind: str,
    component_manifest: AssetComponentManifest | None = None,
) -> bool:
    """Match a direct kind or a checkpoint's declared component kind."""
    require_asset_kind(asset_kind)
    require_asset_kind(accepted_kind)
    if asset_kind == accepted_kind:
        return True
    return (
        asset_kind == KIND_MODEL_CHECKPOINT
        and component_manifest is not None
        and any(component.kind == accepted_kind for component in component_manifest.components)
    )
