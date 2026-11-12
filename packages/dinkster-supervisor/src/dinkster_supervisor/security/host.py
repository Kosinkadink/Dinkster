"""Host-owned signed-feed transport and execution policy."""

from __future__ import annotations

import http.client
import logging
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .documents import SecurityError
from .policy import Artifact, Assessment, assess
from .store import SecurityStore

if TYPE_CHECKING or __package__ == "dinkster_supervisor.security":
    from ..limits import SECURITY_FEED_LIMIT_BYTES
else:
    try:
        from .limits import SECURITY_FEED_LIMIT_BYTES
    except ModuleNotFoundError:
        from dinkster_supervisor.limits import SECURITY_FEED_LIMIT_BYTES

logger = logging.getLogger(__name__)


class HostSettings(BaseModel):
    """Local policy; transport overrides never supply registry identity or trust."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    feed_urls: dict[str, HttpUrl] = Field(default_factory=dict)
    refresh_interval_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    request_timeout_seconds: float = Field(default=5, gt=0, allow_inf_nan=False)
    refuse_vulnerable: bool = False
    maximum_feed_age_seconds: int | None = Field(default=None, gt=0)


class SecurityHost:
    """Refresh authenticated checkpoints without changing trust, answers or software."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.store = SecurityStore(root / "security.sqlite3")
        self.settings = HostSettings()
        self.refresh_errors: dict[str, str] = {}
        self._reported: set[tuple[str, str, int, str, str]] = set()

    def refresh(self) -> None:
        """Try every trusted issuer independently; outage never erases its checkpoint."""
        config = self.root / "security.toml"
        self.settings = (
            HostSettings.model_validate(tomllib.loads(config.read_text(encoding="utf-8")))
            if config.is_file()
            else HostSettings()
        )
        registries = self.store.registries()
        if set(self.settings.feed_urls) - set(registries):
            raise SecurityError("security feed override has no independently provisioned trust")
        failures: dict[str, str] = {}
        for registry in registries:
            url = self.settings.feed_urls.get(registry)
            endpoint = str(url) if url is not None else registry.rstrip("/") + "/v1/security/blocks"
            try:
                endpoint = str(HttpUrl(endpoint))
                # No registry credential is sent to a public feed or mirror.
                with urllib.request.urlopen(  # noqa: S310
                    endpoint, timeout=self.settings.request_timeout_seconds
                ) as response:
                    data = response.read(SECURITY_FEED_LIMIT_BYTES + 1)
                if len(data) > SECURITY_FEED_LIMIT_BYTES:
                    raise SecurityError("security feed exceeds maximum document size")
                self.store.import_feed(registry, data)
            except (OSError, http.client.HTTPException, ValueError) as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    exc.close()
                # Do not log transport URLs or exception text that may contain credentials.
                failures[registry] = type(exc).__name__
                if self.refresh_errors.get(registry) != failures[registry]:
                    logger.warning("Security feed refresh failed for %s; retaining cache", registry)
        self.refresh_errors = failures

    def assess(self, artifacts: tuple[Artifact, ...], *, installing: bool = False) -> Assessment:
        """Use the shared policy, including signed issue age rather than download age."""
        seconds = self.settings.maximum_feed_age_seconds
        result = assess(
            self.store,
            artifacts,
            installing=installing,
            refuse_vulnerable=self.settings.refuse_vulnerable,
            maximum_feed_age=None if seconds is None else timedelta(seconds=seconds),
        )
        for item in result.findings:
            identity = (
                item.registry,
                str(item.notice.id),
                item.notice.revision,
                item.artifact.release.pack,
                item.decision,
            )
            if identity not in self._reported:
                logger.warning(
                    "Security %s: %s==%s, notice %s revision %d, decision %s, execution %s",
                    item.state,
                    item.artifact.release.pack,
                    item.artifact.release.version,
                    item.notice.id,
                    item.notice.revision,
                    item.decision,
                    "allowed" if item.allowed else "refused",
                )
                self._reported.add(identity)
        return result

    def status(self, artifacts: tuple[Artifact, ...]) -> dict[str, object]:
        """Describe accepted local evidence without refreshing or changing software."""
        result = self.assess(artifacts)
        provisioned = set(self.store.registries())
        registries = sorted(
            provisioned
            | {item.registry for item in artifacts}
            | {
                dependency.registry
                for item in artifacts
                for dependency in item.dependencies
                if dependency.registry
            }
        )
        feeds: list[dict[str, object]] = []
        now = datetime.now(UTC)
        for registry in registries:
            document, verified_at = (
                self.store.cached(registry) if registry in provisioned else (None, None)
            )
            feeds.append(
                {
                    "registry": registry,
                    "status": (
                        "not-provisioned"
                        if registry not in provisioned
                        else "not-received"
                        if document is None
                        else "expired"
                        if document.expires_at <= now
                        else "cached"
                    ),
                    "lastVerifiedAt": None if verified_at is None else verified_at.isoformat(),
                    "issuedAt": None if document is None else document.issued_at.isoformat(),
                    "expiresAt": None if document is None else document.expires_at.isoformat(),
                    "refreshError": self.refresh_errors.get(registry),
                    "ageRefused": registry in result.stale_registries,
                }
            )
        return {
            "allowed": result.allowed,
            "findings": [
                {
                    "pack": item.artifact.release.pack,
                    "version": item.artifact.release.version,
                    "artifactDigest": item.artifact.release.artifact_digest,
                    "registry": item.artifact.registry,
                    "state": item.state,
                    "notice": str(item.notice.id),
                    "noticeRegistry": item.registry,
                    "revision": item.notice.revision,
                    "details": item.notice.model_dump(mode="json", by_alias=True),
                    "decision": item.decision,
                    "allowed": item.allowed,
                    "lastVerifiedAt": (
                        None if item.verified_at is None else item.verified_at.isoformat()
                    ),
                    "recovery": {
                        "status": (
                            "allowed" if item.allowed and not result.stale_registries else "refused"
                        ),
                        "repair": "not-reviewed",
                        "action": (
                            "Review a compatible repair; approval cannot bypass this block."
                            if item.decision == "blocked"
                            else (
                                "Import or refresh an authentic feed within the explicit age limit."
                            )
                            if result.stale_registries
                            else (
                                "Review the explicit vulnerability-refusal policy; "
                                "answers do not override it."
                            )
                            if self.settings.refuse_vulnerable
                            else "Change the exact notice answer to allow or deny continuation."
                        ),
                        "automaticResume": False,
                    },
                }
                for item in result.findings
            ],
            "feeds": feeds,
            "staleRegistries": list(result.stale_registries),
            "refreshErrors": dict(self.refresh_errors),
            "policy": {
                "refuseVulnerable": self.settings.refuse_vulnerable,
                "maximumFeedAgeSeconds": self.settings.maximum_feed_age_seconds,
            },
        }
