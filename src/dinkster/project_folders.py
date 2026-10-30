"""Durable project membership and ordering, separate from filesystem grants."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from dinkster_assets import MountsError
from dinkster_assets.mounts import MOUNT_ID_PATTERN


class ProjectFolders:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._shared: list[str] = []
        self._projects: dict[str, list[str]] = {}
        self._excluded: dict[str, list[str]] = {}
        self._outputs: dict[str, str | None] = {}
        self._initialized = path is not None and path.exists()
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text("utf-8"))
            if (
                not isinstance(raw, dict)
                or set(raw) != {"version", "shared", "projects", "excluded", "outputs"}
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

    def ordered(self, project: str, available: tuple[str, ...]) -> tuple[str, ...]:
        self.validate_project(project)
        known = set(available)
        excluded = set(self._excluded.get(project, ()))
        return tuple(
            dict.fromkeys(
                mount_id
                for mount_id in (*self._projects.get(project, ()), *self._shared)
                if mount_id in known and mount_id not in excluded
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
        self, mount_ids: tuple[str, ...], *, shared_defaults: tuple[str, ...] = ()
    ) -> None:
        """Seed shared grants, adding defaults only where no project decision exists."""
        state = self._state()
        if not self._initialized:
            self._shared = list(mount_ids)
        else:
            known = set(self._shared).union(*self._projects.values(), *self._excluded.values())
            self._shared.extend(mount_id for mount_id in shared_defaults if mount_id not in known)
        if not self._initialized or self._state() != state:
            self._persist(state)
        self._initialized = True

    def update(
        self, project: str, mounts: object, *, available: tuple[str, ...], output: str | None
    ) -> None:
        self.validate_project(project)
        ordered = self._mount_ids(mounts)
        if set(ordered) - set(available):
            raise MountsError("project refers to an unknown mount")
        if output is not None and output not in ordered:
            raise MountsError("project output must be included in its folders")
        state = self._state()
        self._projects[project] = ordered
        self._excluded[project] = [mount_id for mount_id in self._shared if mount_id not in ordered]
        self._outputs[project] = output
        self._persist(state)

    def output_projects(self, mount_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(project for project, output in self._outputs.items() if output == mount_id)
        )

    def _state(
        self,
    ) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]], dict[str, str | None]]:
        return (
            list(self._shared),
            {key: list(value) for key, value in self._projects.items()},
            {key: list(value) for key, value in self._excluded.items()},
            dict(self._outputs),
        )

    def _persist(
        self,
        previous: tuple[
            list[str], dict[str, list[str]], dict[str, list[str]], dict[str, str | None]
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
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
                json.dump(
                    {
                        "version": 1,
                        "shared": self._shared,
                        "projects": self._projects,
                        "excluded": self._excluded,
                        "outputs": self._outputs,
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
            self._shared, self._projects, self._excluded, self._outputs = previous
            raise
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
