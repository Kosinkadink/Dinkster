"""Closed, serializable registry boundary for pure partner helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from types import MappingProxyType


class HelperRefusal(ValueError):
    pass


HelperFunction = Callable[[Mapping[str, object]], Mapping[str, object]]
_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


class HelperRegistry:
    def __init__(
        self,
        provider: str,
        entries: Mapping[str, Mapping[str, HelperFunction]],
    ) -> None:
        if not _TOKEN.fullmatch(provider):
            raise ValueError("helper provider must be a lowercase ASCII token")
        copied: dict[str, Mapping[str, HelperFunction]] = {}
        for helper_id, stages in entries.items():
            parts = helper_id.split(".")
            if len(parts) != 2 or parts[0] != provider or not _TOKEN.fullmatch(parts[1]):
                raise ValueError("helper id must use the registry provider and a lowercase token")
            if not stages:
                raise ValueError("helper stages must not be empty")
            stage_copy: dict[str, HelperFunction] = {}
            for stage, function in stages.items():
                if not _TOKEN.fullmatch(stage):
                    raise ValueError("helper stage must be a lowercase ASCII token")
                if not callable(function):
                    raise ValueError("helper entry must be callable")
                stage_copy[stage] = function
            copied[helper_id] = MappingProxyType(stage_copy)
        self._provider = provider
        self._entries = MappingProxyType(copied)

    @property
    def provider(self) -> str:
        return self._provider

    def manifest(self) -> str:
        return json.dumps(
            {
                "helpers": {
                    helper_id: sorted(stages) for helper_id, stages in sorted(self._entries.items())
                },
                "provider": self.provider,
                "version": 1,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def resolve(self, helper_id: str, stage: str) -> HelperFunction:
        provider = helper_id.partition(".")[0]
        if provider != self.provider:
            raise HelperRefusal(f"helper provider {provider!r} does not match {self.provider!r}")
        stages = self._entries.get(helper_id)
        if stages is None:
            raise HelperRefusal(f"unknown helper id {helper_id!r}")
        function = stages.get(stage)
        if function is None:
            raise HelperRefusal(f"unknown helper stage {stage!r} for {helper_id!r}")
        return function

    def invoke(
        self, helper_id: str, stage: str, payload: Mapping[str, object]
    ) -> Mapping[str, object]:
        return self.resolve(helper_id, stage)(payload)
