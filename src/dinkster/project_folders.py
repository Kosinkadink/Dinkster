"""Durable project membership and ordering, separate from filesystem grants."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dinkster_assets import MountsError
from dinkster_assets.mounts import MOUNT_ID_PATTERN
from dinkster_values.asset_folders import AssetFolderContext


@dataclass(frozen=True)
class HiddenModelPreference:
    name: str
    kind: str
    identities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.kind.startswith("model/") or not self.identities:
            raise ValueError("hidden model requires name, model kind and identities")
        AssetFolderContext((), hidden_models=self.identities)

    def to_wire(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "identities": list(self.identities)}


class ProjectFolders:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._shared: list[str] = []
        self._projects: dict[str, list[str]] = {}
        self._excluded: dict[str, list[str]] = {}
        self._outputs: dict[str, str | None] = {}
        self._hidden_models: dict[str, list[HiddenModelPreference]] = {}
        self._initialized = path is not None and path.exists()
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text("utf-8"))
            if (
                not isinstance(raw, dict)
                or set(raw)
                not in (
                    {"version", "shared", "projects", "excluded", "outputs"},
                    {"version", "shared", "projects", "excluded", "outputs", "hiddenModels"},
                )
                or raw["version"] != 1
            ):
                raise ValueError("invalid project folders document")
            self._shared = self._mount_ids(raw["shared"])
            for key, target in (("projects", self._projects), ("excluded", self._excluded)):
                if not isinstance(raw[key], dict):
                    raise ValueError(f"{key} must be an object")
                for project, ids in raw[key].items():
                    self.validate_project(project)
                    target[project] = self._mount_ids(ids)
            if not isinstance(raw["outputs"], dict):
                raise ValueError("outputs must be an object")
            for project, mount_id in raw["outputs"].items():
                self.validate_project(project)
                if mount_id is not None:
                    self._mount_ids([mount_id])
                self._outputs[project] = mount_id
            hidden = raw.get("hiddenModels", {})
            if not isinstance(hidden, dict):
                raise ValueError("hidden models must be an object")
            for project, preferences in hidden.items():
                self.validate_project(project)
                if not isinstance(preferences, list):
                    raise ValueError("hidden models must be a list")
                parsed: list[HiddenModelPreference] = []
                for preference in preferences:
                    if (
                        not isinstance(preference, dict)
                        or set(preference) != {"name", "kind", "identities"}
                        or not isinstance(preference["name"], str)
                        or not isinstance(preference["kind"], str)
                        or not isinstance(preference["identities"], list)
                    ):
                        raise ValueError("invalid hidden model preference")
                    parsed.append(
                        HiddenModelPreference(
                            preference["name"], preference["kind"], tuple(preference["identities"])
                        )
                    )
                self._hidden_models[project] = parsed
            if "hiddenModels" in raw and not any(self._hidden_models.values()):
                self._persist(self._state())
        except (OSError, UnicodeError, ValueError, MountsError) as exc:
            raise MountsError(f"cannot load project asset folders: {exc}") from exc

    @staticmethod
    def validate_project(project: str) -> None:
        if (
            not isinstance(project, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", project) is None
        ):
            raise MountsError("project must be a lowercase project id of at most 64 characters")

    @staticmethod
    def _mount_ids(raw: object) -> list[str]:
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not MOUNT_ID_PATTERN.fullmatch(item) for item in raw
        ):
            raise MountsError("mounts must be a list of mount ids")
        if len(set(raw)) != len(raw):
            raise MountsError("mount ids must be unique")
        return list(raw)

    def ordered(
        self,
        project: str,
        available: tuple[str, ...],
        *,
        modes: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        self.validate_project(project)
        modes = modes or {}
        known = set(available)
        selected = self._projects.get(project, ())
        excluded = set(self._excluded.get(project, ()))
        defaults = (
            mount_id for mount_id in available if modes.get(mount_id) in {"always-on", "default-on"}
        )
        return tuple(
            dict.fromkeys(
                mount_id
                for mount_id in (*selected, *self._shared, *defaults)
                if mount_id in known
                and modes.get(mount_id) != "disabled"
                and (
                    modes.get(mount_id) == "always-on"
                    or mount_id not in excluded
                    and (modes.get(mount_id) != "opt-in" or mount_id in selected)
                )
            )
        )

    def output(self, project: str, fallback: str | None) -> str | None:
        self.validate_project(project)
        return self._outputs.get(project, fallback)

    def register(self, mount_id: str, *, project: str, shared: bool) -> None:
        self.validate_project(project)
        self._mount_ids([mount_id])
        state = self._state()
        self._shared = [item for item in self._shared if item != mount_id]
        for memberships in (self._projects, self._excluded):
            for owner, ids in memberships.items():
                memberships[owner] = [item for item in ids if item != mount_id]
        self._outputs = {key: value for key, value in self._outputs.items() if value != mount_id}
        if shared:
            self._shared.append(mount_id)
        else:
            mounts = self._projects.setdefault(project, [])
            mounts.append(mount_id)
        self._persist(state)

    def initialize(
        self,
        mount_ids: tuple[str, ...],
        *,
        shared_defaults: tuple[str, ...] = (),
        groups: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        """Seed shared grants, adding defaults only where no project decision exists."""
        state = self._state()
        if not self._initialized:
            self._shared = list(mount_ids)
        else:
            known = set(self._shared).union(*self._projects.values(), *self._excluded.values())
            self._shared.extend(mount_id for mount_id in shared_defaults if mount_id not in known)
        for project in set(self._projects) | set(self._excluded):
            selected = self._projects.get(project, [])
            excluded = self._excluded.get(project, [])
            for group in groups:
                if set(group).intersection(selected):
                    selected.extend(
                        mount_id
                        for mount_id in group
                        if mount_id in mount_ids and mount_id not in selected
                    )
                    excluded[:] = [mount_id for mount_id in excluded if mount_id not in group]
                elif set(group).intersection(excluded):
                    excluded.extend(mount_id for mount_id in group if mount_id not in excluded)
        if not self._initialized or self._state() != state:
            self._persist(state)
        self._initialized = True

    def update(
        self,
        project: str,
        mounts: object,
        *,
        available: tuple[str, ...],
        output: str | None,
        required: tuple[str, ...] = (),
        groups: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        self.validate_project(project)
        ordered = self._mount_ids(mounts)
        for group in groups:
            included = set(group).intersection(ordered)
            if included and included != set(group):
                raise MountsError("enable or disable the complete model location")
        if set(required) - set(ordered):
            raise MountsError("always-on model folders cannot be disabled for a project")
        if set(ordered) - set(available):
            raise MountsError("project refers to an unknown mount")
        if output is not None and output not in ordered:
            raise MountsError("project output must be included in its folders")
        state = self._state()
        self._projects[project] = ordered
        self._excluded[project] = [mount_id for mount_id in self._shared if mount_id not in ordered]
        self._outputs[project] = output
        self._persist(state)

    def hidden_models(self, project: str) -> tuple[str, ...]:
        self.validate_project(project)
        return tuple(
            dict.fromkeys(
                identity
                for row in self._hidden_models.get(project, ())
                for identity in row.identities
            )
        )

    def hidden_model_preferences(self, project: str) -> list[dict[str, object]]:
        self.validate_project(project)
        return [row.to_wire() for row in self._hidden_models.get(project, ())]

    def set_model_hidden(
        self, project: str, preference: HiddenModelPreference, hidden: bool
    ) -> None:
        self.validate_project(project)
        state = self._state()
        selected = [
            row
            for row in self._hidden_models.get(project, ())
            if not set(row.identities).intersection(preference.identities)
        ]
        if hidden:
            selected.append(preference)
        self._hidden_models[project] = selected
        self._persist(state)

    def output_projects(self, mount_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(project for project, output in self._outputs.items() if output == mount_id)
        )

    def _state(
        self,
    ) -> tuple[
        list[str],
        dict[str, list[str]],
        dict[str, list[str]],
        dict[str, str | None],
        dict[str, list[HiddenModelPreference]],
    ]:
        return (
            list(self._shared),
            {key: list(value) for key, value in self._projects.items()},
            {key: list(value) for key, value in self._excluded.items()},
            dict(self._outputs),
            {key: list(value) for key, value in self._hidden_models.items()},
        )

    def _persist(
        self,
        previous: tuple[
            list[str],
            dict[str, list[str]],
            dict[str, list[str]],
            dict[str, str | None],
            dict[str, list[HiddenModelPreference]],
        ],
    ) -> None:
        if self._path is None:
            return
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".project-folders-", dir=self._path.parent
            )
            hidden_models = {
                project: [row.to_wire() for row in rows]
                for project, rows in self._hidden_models.items()
                if rows
            }
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
                json.dump(
                    {
                        "version": 1,
                        "shared": self._shared,
                        "projects": self._projects,
                        "excluded": self._excluded,
                        "outputs": self._outputs,
                        # Releases through 0.0.14 reject even an empty unknown field.
                        **({"hiddenModels": hidden_models} if hidden_models else {}),
                    },
                    file,
                    sort_keys=True,
                    indent=2,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path)
        except OSError:
            (self._shared, self._projects, self._excluded, self._outputs, self._hidden_models) = (
                previous
            )
            raise
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
