"""Authored package-relative frontend modules and their effective identities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .pack_surfaces import validate_pack_event_name

FRONTEND_PRIVILEGES = (
    "schema-widget",
    "graph-editor-canvas",
    "app-workflow",
    "event-consumer",
)
FRONTEND_CONTRIBUTION_KINDS = (
    "widgetKind",
    "widgetView",
    "previewRenderer",
    "textEditorExtension",
    "menu",
    "command",
    "keybinding",
    "setting",
    "canvasLayer",
    "nodeDecoration",
    "linkDecoration",
    "canvasTool",
    "hostUi",
    "searchProvider",
    "workflowObserver",
    "workflowGuard",
    "eventConsumer",
    "workflowImporter",
)
FRONTEND_ASSET_PATH = "/api/extension-assets/{pack_id}/{digest}/{entry_id}.js"


@dataclass(frozen=True)
class FrontendContribution:
    id: str
    kind: str
    event: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.id)
        if self.kind not in FRONTEND_CONTRIBUTION_KINDS:
            raise ValueError("unknown frontend contribution kind")
        if self.kind == "eventConsumer":
            validate_pack_event_name(self.event)
        elif self.event is not None:
            raise ValueError("only eventConsumer contributions may name an event")

    def to_wire(self) -> dict[str, object]:
        return {"id": self.id, "kind": self.kind, **({"event": self.event} if self.event else {})}


@dataclass(frozen=True)
class FrontendModule:
    id: str
    module: str
    privileges: tuple[str, ...]
    contributions: tuple[FrontendContribution, ...]
    module_digest: str = ""

    def __post_init__(self) -> None:
        _identifier(self.id)
        if (
            not isinstance(cast(object, self.module), str)
            or not self.module.startswith("./")
            or not self.module.endswith(".js")
            or any(part in ("", ".", "..") for part in self.module[2:].split("/"))
            or not re.fullmatch(r"\./[A-Za-z0-9_./-]+", self.module)
        ):
            raise ValueError("frontend module must be a confined ./relative.js path")
        if (
            not isinstance(cast(object, self.privileges), tuple)
            or any(privilege not in FRONTEND_PRIVILEGES for privilege in self.privileges)
            or len(set(self.privileges)) != len(self.privileges)
        ):
            raise ValueError("frontend privileges must use the four-privilege vocabulary uniquely")
        raw = cast(object, self.contributions)
        if not isinstance(raw, tuple) or not all(
            isinstance(item, FrontendContribution) for item in cast(tuple[object, ...], raw)
        ):
            raise TypeError("frontend contributions must be a tuple")
        ids = [item.id for item in self.contributions]
        if len(set(ids)) != len(ids):
            raise ValueError("frontend contribution ids must be unique")
        if self.module_digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", self.module_digest):
            raise ValueError("frontend module digest must be sha256:<lowercase hex>")

    def to_wire(self, pack: str) -> dict[str, object]:
        if not self.module_digest:
            raise ValueError("effective frontend module requires a content digest")
        return {
            "id": self.id,
            "moduleUrl": FRONTEND_ASSET_PATH.format(
                pack_id=pack, digest=self.module_digest, entry_id=self.id
            ),
            "moduleDigest": self.module_digest,
            "authorizedPrivileges": list(self.privileges),
            "contributions": [item.to_wire() for item in self.contributions],
        }

    @classmethod
    def from_manifest(cls, raw: object) -> FrontendModule:
        if not isinstance(raw, Mapping) or set(cast(Mapping[object, object], raw)) != {
            "id",
            "module",
            "privileges",
            "contributions",
        }:
            raise ValueError("frontend module requires id, module, privileges, contributions")
        data = cast(Mapping[str, object], raw)
        privileges, contributions = data["privileges"], data["contributions"]
        if not isinstance(privileges, list) or not isinstance(contributions, list):
            raise ValueError("frontend privileges and contributions must be arrays")
        parsed: list[FrontendContribution] = []
        for raw_item in cast(list[object], contributions):
            if not isinstance(raw_item, Mapping) or set(
                cast(Mapping[object, object], raw_item)
            ) not in (
                {"id", "kind"},
                {"id", "kind", "event"},
            ):
                raise ValueError("invalid frontend contribution fields")
            item = cast(Mapping[str, object], raw_item)
            parsed.append(
                FrontendContribution(
                    cast(str, item["id"]),
                    cast(str, item["kind"]),
                    cast(str | None, item.get("event")),
                )
            )
        return cls(
            cast(str, data["id"]),
            cast(str, data["module"]),
            tuple(cast(list[str], privileges)),
            tuple(parsed),
        )


def validate_frontend_modules(raw: object, *, pack: str | None = None) -> None:
    if not isinstance(raw, tuple) or not all(
        isinstance(item, FrontendModule) for item in cast(tuple[object, ...], raw)
    ):
        raise TypeError("frontend_modules must be a tuple of FrontendModule values")
    modules = cast(tuple[FrontendModule, ...], raw)
    ids = [item.id for item in modules]
    contributions = [item.id for module in modules for item in module.contributions]
    if ids != sorted(set(ids)) or len(set(contributions)) != len(contributions):
        raise ValueError("frontend module ids must be sorted and unique; contribution ids unique")
    if pack is not None and any(not id_.startswith(pack + ".") for id_ in ids + contributions):
        raise ValueError("frontend entry and contribution ids must belong to the pack")


def _identifier(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", value):
        raise ValueError("frontend id must be dot-namespaced ASCII")
