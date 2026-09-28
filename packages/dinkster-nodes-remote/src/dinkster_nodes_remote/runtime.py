from __future__ import annotations

import asyncio
import contextlib
import email.utils
import ipaddress
import math
import os
import socket
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeGuard, cast
from urllib.parse import urlparse

import httpx
from dinkster_api.v1 import (
    ASSET_TYPE,
    MEBIBYTE,
    AssetError,
    AssetRef,
    AssetVault,
    Node,
    NodeSchema,
    TypeExpr,
    TypeRegistry,
    digest_bytes,
    pack_logger,
    register_asset_type,
    report_preview,
    report_progress,
    resolver_from_env,
)
from dinkster_workers import current_execution_context

from .catalog import CatalogClient, RemoteSchema, cache_path_from_env

_DEFAULT_GATEWAY_BASE = ""
_ERROR_MESSAGES = {
    "unauthenticated": "Sign in to use remote nodes.",
    "insufficient_credit": "The account does not have enough credit for this remote job.",
    "policy_denied": "The remote job was denied by policy.",
    "schema_version_unsupported": "The remote node schema version is no longer supported.",
    "schema_signature_mismatch": "The remote node schema changed before this job was submitted.",
    "node_paused": "This remote node is temporarily paused.",
    "inventory_exhausted": "This remote node has no available capacity right now.",
    "partner_unavailable": "The remote generation provider is temporarily unavailable.",
    "partner_failed": "The remote generation provider failed this job.",
    "rate_limited": "The remote gateway rate limit was reached.",
    "job_cancelled": "The remote job was cancelled.",
}

log = pack_logger("dinkster-nodes-remote")


def _is_string_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return False
    mapping = cast("Mapping[object, object]", value)
    return all(isinstance(key, str) for key in mapping)


def _positive_float(value: str, name: str, *, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    lower_ok = parsed >= 0 if allow_zero else parsed > 0
    if not math.isfinite(parsed) or not lower_ok:
        condition = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {condition} number")
    return parsed


def _non_negative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _http_url(value: str, name: str, *, allow_query: bool) -> str:
    if not value:
        return ""
    try:
        parsed = urlparse(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid HTTP(S) URL") from exc
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or (parsed.query and not allow_query)
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be a valid HTTP(S) URL")
    if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError(f"{name} must use HTTPS except for a loopback test service")
    return value


def _base_url(value: str, name: str) -> str:
    value = _http_url(value, name, allow_query=False)
    return value.rstrip("/")


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.casefold().rstrip(".")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        try:
            address = ipaddress.ip_address(socket.inet_aton(normalized))
        except OSError:
            return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


@dataclass(frozen=True)
class RemoteConfig:
    catalog_base: str
    gateway_base: str
    cache_path: Path | None
    token: str
    token_file: Path | None
    catalog_poll_interval: float
    image_poll_interval: float
    video_poll_interval: float
    request_timeout: float
    max_retries: int
    max_retry_after: float

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> RemoteConfig:
        env = os.environ if environment is None else environment
        catalog_raw = env.get("DINKSTER_REMOTE_CATALOG_BASE", _DEFAULT_GATEWAY_BASE)
        catalog_base = _base_url(catalog_raw, "DINKSTER_REMOTE_CATALOG_BASE")
        gateway_base = _base_url(
            env.get("DINKSTER_REMOTE_GATEWAY_BASE", catalog_base),
            "DINKSTER_REMOTE_GATEWAY_BASE",
        )
        token_file_raw = env.get("DINKSTER_REMOTE_AUTH_TOKEN_FILE", "")
        return cls(
            catalog_base=catalog_base,
            gateway_base=gateway_base,
            cache_path=cache_path_from_env(env),
            token="",
            token_file=Path(token_file_raw) if token_file_raw else None,
            catalog_poll_interval=_positive_float(
                env.get("DINKSTER_REMOTE_CATALOG_POLL_INTERVAL", "3600"),
                "DINKSTER_REMOTE_CATALOG_POLL_INTERVAL",
            ),
            image_poll_interval=_positive_float(
                env.get("DINKSTER_REMOTE_IMAGE_POLL_INTERVAL", "1"),
                "DINKSTER_REMOTE_IMAGE_POLL_INTERVAL",
            ),
            video_poll_interval=_positive_float(
                env.get("DINKSTER_REMOTE_VIDEO_POLL_INTERVAL", "5"),
                "DINKSTER_REMOTE_VIDEO_POLL_INTERVAL",
            ),
            request_timeout=_positive_float(
                env.get("DINKSTER_REMOTE_REQUEST_TIMEOUT", "60"),
                "DINKSTER_REMOTE_REQUEST_TIMEOUT",
            ),
            max_retries=_non_negative_int(
                env.get("DINKSTER_REMOTE_MAX_RETRIES", "3"),
                "DINKSTER_REMOTE_MAX_RETRIES",
            ),
            max_retry_after=_positive_float(
                env.get("DINKSTER_REMOTE_MAX_RETRY_AFTER", "30"),
                "DINKSTER_REMOTE_MAX_RETRY_AFTER",
                allow_zero=True,
            ),
        )

    def auth_token(self) -> str:
        if self.token_file is not None:
            try:
                token = self.token_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise GatewayError(
                    "unauthenticated",
                    "client",
                    False,
                    f"The remote session token could not be read: {exc}",
                ) from exc
            if token:
                return token
            raise GatewayError(
                "unauthenticated", "client", False, "The remote session token file is empty."
            )
        if self.token:
            return self.token
        raise GatewayError("unauthenticated", "client", False, "No remote session token is set.")


class GatewayError(RuntimeError):
    def __init__(
        self,
        code: str,
        source: str,
        retryable: bool,
        message: str,
        *,
        job_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.code = code
        self.source = source
        self.retryable = retryable
        self.gateway_message = message
        self.job_id = job_id
        self.retry_after = retry_after
        friendly = _ERROR_MESSAGES.get(code, "The remote gateway rejected this job.")
        detail = f" {message}" if message and message.rstrip(".") != friendly.rstrip(".") else ""
        retry = f" Retry after {retry_after:g} seconds." if retry_after is not None else ""
        super().__init__(f"{friendly}{detail}{retry} [code={code}, source={source}]")


class RemoteCancelled(RuntimeError):
    pass


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            moment = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return max(0.0, (moment - datetime.now(UTC)).total_seconds())


def _gateway_error_from_mapping(
    value: object,
    *,
    status_code: int = 500,
    retry_after: float | None = None,
) -> GatewayError:
    envelope: Mapping[str, object] = value if _is_string_mapping(value) else {}
    raw = envelope.get("error", envelope)
    error: Mapping[str, object] = raw if _is_string_mapping(raw) else {}
    fallback_codes = {
        401: "unauthenticated",
        402: "insufficient_credit",
        403: "policy_denied",
        429: "rate_limited",
    }
    code_value = error.get("code", fallback_codes.get(status_code, "gateway_error"))
    source_value = error.get("source", "gateway")
    retryable_value = error.get("retryable", status_code >= 500 or status_code == 429)
    message_value = error.get("message", f"HTTP {status_code}")
    job_value = error.get("jobId")
    return GatewayError(
        code_value if isinstance(code_value, str) and code_value else "gateway_error",
        source_value if isinstance(source_value, str) and source_value else "gateway",
        retryable_value if isinstance(retryable_value, bool) else False,
        message_value if isinstance(message_value, str) else f"HTTP {status_code}",
        job_id=job_value if isinstance(job_value, str) else None,
        retry_after=retry_after,
    )


def _gateway_error(response: httpx.Response) -> GatewayError:
    try:
        value: object = response.json()
    except ValueError:
        value = {}
    return _gateway_error_from_mapping(
        value,
        status_code=response.status_code,
        retry_after=_retry_after(response),
    )


def _cancelled() -> bool:
    context = current_execution_context()
    return context is not None and context.cancelled()


def _ensure_running() -> None:
    if _cancelled():
        raise RemoteCancelled("remote job cancelled by the client")


async def _sleep_running(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        _ensure_running()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.1))


def _json(response: httpx.Response, subject: str) -> Mapping[str, object]:
    try:
        value: object = response.json()
    except ValueError as exc:
        raise GatewayError(
            "malformed_response", "gateway", False, f"{subject} was not JSON"
        ) from exc
    if not _is_string_mapping(value):
        raise GatewayError(
            "malformed_response", "gateway", False, f"{subject} was not a JSON object"
        )
    return value


def _type_expressions(value: object) -> Iterator[TypeExpr]:
    if isinstance(value, TypeExpr):
        yield value
    elif isinstance(value, tuple):
        for item in cast("tuple[object, ...]", value):
            yield from _type_expressions(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for descriptor in fields(value):
            yield from _type_expressions(getattr(value, descriptor.name))


def _asset_target_atoms(expression: TypeExpr, *, inside_asset: bool = False) -> Iterator[str]:
    inside_asset = inside_asset or expression.kind == "asset"
    if inside_asset:
        yield from expression.types
    if expression.element is not None:
        yield from _asset_target_atoms(expression.element, inside_asset=inside_asset)


class RemoteRuntime:
    def __init__(self, config: RemoteConfig) -> None:
        self.config = config
        self.catalog = CatalogClient(
            config.catalog_base,
            config.cache_path,
            timeout=min(config.request_timeout, 5.0),
        )
        self.snapshot = self.catalog.startup()
        self.announced_epoch = self.snapshot.epoch
        self._force_reload = asyncio.Event()
        self._reload_pending = False

    def register_types(self, registry: TypeRegistry) -> None:
        if ASSET_TYPE not in registry:
            register_asset_type(registry, resolver_from_env())
        for remote in self.snapshot.schemas:
            for expression in _type_expressions(remote.schema):
                for atom in _asset_target_atoms(expression):
                    if atom not in registry:
                        registry.register(atom)

    async def wait_for_schema_reload(self) -> None:
        while True:
            if self._reload_pending:
                await asyncio.sleep(self.config.catalog_poll_interval)
                return
            forced = False
            try:
                await asyncio.wait_for(
                    self._force_reload.wait(), timeout=self.config.catalog_poll_interval
                )
                forced = True
                self._force_reload.clear()
            except TimeoutError:
                pass
            if forced:
                if await self.catalog.refresh(conditional=False) is not None:
                    self._reload_pending = True
                    return
                continue
            if not await self.catalog.changed(self.announced_epoch):
                continue
            snapshot = await self.catalog.refresh(conditional=True)
            if snapshot is not None and snapshot.epoch != self.announced_epoch:
                self._reload_pending = True
                return

    async def _refresh_after_schema_error(self) -> None:
        if await self.catalog.refresh(conditional=False) is not None:
            self._force_reload.set()

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        token: str,
        *,
        json_body: Mapping[str, object] | None = None,
        content: Callable[[], AsyncIterator[bytes]] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        accepted: Sequence[int] = (200,),
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}", **dict(extra_headers or {})}
        for attempt in range(self.config.max_retries + 1):
            _ensure_running()
            try:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    content=content() if content is not None else None,
                )
            except httpx.HTTPError as exc:
                if attempt >= self.config.max_retries:
                    raise GatewayError(
                        "gateway_unreachable",
                        "gateway",
                        True,
                        f"The remote gateway could not be reached: {exc}",
                    ) from exc
                await _sleep_running(0.25 * (2**attempt))
                continue
            if response.status_code in accepted:
                return response
            error = _gateway_error(response)
            if error.code in ("schema_version_unsupported", "schema_signature_mismatch"):
                await self._refresh_after_schema_error()
            if not error.retryable or attempt >= self.config.max_retries:
                raise error
            delay = error.retry_after
            if delay is None:
                delay = 0.25 * (2**attempt)
            elif delay > self.config.max_retry_after:
                raise error
            await _sleep_running(delay)
        raise AssertionError("request retry loop exhausted")

    @staticmethod
    async def _asset_content(asset: AssetRef) -> AsyncIterator[bytes]:
        handle = await asyncio.to_thread(asset.open)
        try:
            while True:
                _ensure_running()
                chunk = await asyncio.to_thread(handle.read, MEBIBYTE)
                if not chunk:
                    return
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)

    async def _upload_asset(self, client: httpx.AsyncClient, token: str, asset: AssetRef) -> None:
        response = await self._request(
            client,
            "POST",
            f"{self.config.gateway_base}/v1/uploads",
            token,
            content=lambda: self._asset_content(asset),
            extra_headers={
                "Content-Type": asset.media_type,
                "X-Dinkster-Digest": asset.digest,
            },
            accepted=(200, 201),
        )
        returned = _json(response, "upload response").get("digest")
        if returned != asset.digest:
            raise GatewayError(
                "upload_digest_mismatch",
                "gateway",
                False,
                f"Upload response digest {returned!r} did not match {asset.digest!r}.",
            )

    async def _prepare_inputs(
        self,
        client: httpx.AsyncClient,
        token: str,
        inputs: Mapping[str, object],
    ) -> dict[str, object]:
        uploaded: set[str] = set()

        async def prepare(value: object) -> object:
            if isinstance(value, AssetRef):
                if value.digest not in uploaded:
                    await self._upload_asset(client, token, value)
                    uploaded.add(value.digest)
                return {
                    "digest": value.digest,
                    "name": value.name,
                    "size": value.size,
                    "mediaType": value.media_type,
                }
            if _is_string_mapping(value):
                return {key: await prepare(child) for key, child in value.items()}
            if isinstance(value, Mapping):
                raise GatewayError(
                    "unsupported_input",
                    "client",
                    False,
                    "Remote input mappings need string keys.",
                )
            if isinstance(value, (list, tuple)):
                children = cast("Sequence[object]", value)
                return [await prepare(child) for child in children]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            raise GatewayError(
                "unsupported_input",
                "client",
                False,
                f"Remote input type {type(value).__name__} is not JSON or an AssetRef.",
            )

        return {name: await prepare(value) for name, value in inputs.items()}

    @staticmethod
    def _is_video(schema: NodeSchema) -> bool:
        choices = () if schema.output_descriptors is None else schema.output_descriptors.choices
        output_types = (
            *(output.type.runtime_type_id() for output in schema.outputs),
            *(family.type.runtime_type_id() for family in schema.output_families),
            *(choice.type.runtime_type_id() for choice in choices),
        )
        return any(
            type_id is not None and "video" in type_id.casefold() for type_id in output_types
        )

    def _download_identity(
        self, descriptor: Mapping[str, object], subject: str
    ) -> tuple[str, str, int]:
        url = descriptor.get("downloadUrl")
        if not isinstance(url, str):
            raise GatewayError(
                "malformed_output", "gateway", False, f"{subject} has no HTTP(S) downloadUrl."
            )
        try:
            url = _http_url(url, f"{subject} downloadUrl", allow_query=True)
        except ValueError as exc:
            raise GatewayError("malformed_output", "gateway", False, str(exc)) from exc
        download_url = urlparse(url)
        gateway_url = urlparse(self.config.gateway_base)
        if _is_loopback_host(download_url.hostname) and (
            download_url.scheme,
            download_url.hostname,
            download_url.port,
        ) != (gateway_url.scheme, gateway_url.hostname, gateway_url.port):
            raise GatewayError(
                "malformed_output",
                "gateway",
                False,
                f"{subject} has a loopback downloadUrl outside the configured test gateway.",
            )
        expected = descriptor.get("digest")
        size = descriptor.get("size")
        if not isinstance(expected, str) or type(size) is not int or size < 0:
            raise GatewayError(
                "malformed_output", "gateway", False, f"{subject} has no valid digest and size."
            )
        return url, expected, size

    async def _download_bytes(
        self,
        client: httpx.AsyncClient,
        descriptor: Mapping[str, object],
        *,
        subject: str,
    ) -> bytes:
        url, expected, expected_size = self._download_identity(descriptor, subject)
        for attempt in range(self.config.max_retries + 1):
            _ensure_running()
            try:
                data = bytearray()
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        _ensure_running()
                        data.extend(chunk)
                        if len(data) > expected_size:
                            raise AssetError(
                                f"{subject} exceeds its declared size of {expected_size} bytes"
                            )
            except httpx.HTTPError as exc:
                if attempt >= self.config.max_retries:
                    raise GatewayError(
                        "download_failed", "gateway", True, f"{subject} download failed: {exc}"
                    ) from exc
                await _sleep_running(0.25 * (2**attempt))
                continue
            if len(data) != expected_size:
                raise AssetError(
                    f"{subject} size mismatch: expected {expected_size}, "
                    f"downloaded {len(data)} bytes"
                )
            downloaded = bytes(data)
            actual = digest_bytes(downloaded)
            if actual != expected:
                raise AssetError(
                    f"remote output digest mismatch: expected {expected}, bytes hash to {actual}"
                )
            return downloaded
        raise AssertionError("download retry loop exhausted")

    async def _download_to_vault(
        self,
        client: httpx.AsyncClient,
        vault: AssetVault,
        descriptor: Mapping[str, object],
        *,
        subject: str,
    ) -> None:
        url, expected, expected_size = self._download_identity(descriptor, subject)
        for attempt in range(self.config.max_retries + 1):
            _ensure_running()
            try:
                size = 0
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with vault.writer(expected) as writer:
                        async for chunk in response.aiter_bytes():
                            _ensure_running()
                            size += len(chunk)
                            if size > expected_size:
                                raise AssetError(
                                    f"{subject} exceeds its declared size of {expected_size} bytes"
                                )
                            writer.write(chunk)
                        if size != expected_size:
                            raise AssetError(
                                f"{subject} size mismatch: expected {expected_size}, "
                                f"downloaded {size} bytes"
                            )
                        writer.commit()
            except httpx.HTTPError as exc:
                if attempt >= self.config.max_retries:
                    raise GatewayError(
                        "download_failed", "gateway", True, f"{subject} download failed: {exc}"
                    ) from exc
                await _sleep_running(0.25 * (2**attempt))
                continue
            return
        raise AssertionError("download retry loop exhausted")

    async def _forward_status(
        self,
        client: httpx.AsyncClient,
        job: Mapping[str, object],
        previous_progress: object,
        seen_previews: set[str],
    ) -> object:
        progress = job.get("progress")
        if progress != previous_progress and _is_string_mapping(progress):
            step = progress.get("step")
            total = progress.get("total")
            text = progress.get("text", "")
            if type(step) is int and type(total) is int and isinstance(text, str):
                report_progress(step, total, text=text)
        raw_previews = job.get("previews", job.get("preview"))
        previews: Sequence[object]
        if isinstance(raw_previews, Sequence) and not isinstance(raw_previews, (str, bytes)):
            previews = cast("Sequence[object]", raw_previews)
        elif _is_string_mapping(raw_previews):
            previews = (raw_previews,)
        else:
            previews = ()
        for raw in previews:
            if not _is_string_mapping(raw):
                continue
            descriptor = raw
            key = str(descriptor.get("digest", descriptor.get("downloadUrl", "")))
            if not key or key in seen_previews:
                continue
            data = await self._download_bytes(client, descriptor, subject="remote preview")
            seen_previews.add(key)
            report_preview(
                data,
                mime=str(descriptor.get("mediaType", "image/jpeg")),
                width=(
                    cast("int", descriptor["width"])
                    if type(descriptor.get("width")) is int
                    else None
                ),
                height=(
                    cast("int", descriptor["height"])
                    if type(descriptor.get("height")) is int
                    else None
                ),
                stream=(
                    cast("str", descriptor["stream"])
                    if isinstance(descriptor.get("stream"), str)
                    else None
                ),
                frame_index=(
                    cast("int", descriptor["frameIndex"])
                    if type(descriptor.get("frameIndex")) is int
                    else None
                ),
                frame_count=(
                    cast("int", descriptor["frameCount"])
                    if type(descriptor.get("frameCount")) is int
                    else None
                ),
                fps=(
                    float(cast("int | float", descriptor["fps"]))
                    if type(descriptor.get("fps")) in (int, float)
                    and math.isfinite(float(cast("int | float", descriptor["fps"])))
                    else None
                ),
            )
        return progress

    async def _ingest_output(
        self,
        client: httpx.AsyncClient,
        vault: AssetVault,
        value: object,
        name: str,
    ) -> object:
        if _is_string_mapping(value) and "digest" in value:
            descriptor = value
            digest = descriptor.get("digest")
            size = descriptor.get("size")
            media_type = descriptor.get("mediaType", "application/octet-stream")
            if (
                not isinstance(digest, str)
                or type(size) is not int
                or size < 0
                or not isinstance(media_type, str)
            ):
                raise GatewayError(
                    "malformed_output", "gateway", False, f"Remote output {name!r} is malformed."
                )
            await self._download_to_vault(
                client, vault, descriptor, subject=f"remote output {name!r}"
            )
            output_name = descriptor.get("name", name)
            return AssetRef(
                digest=digest,
                name=output_name if isinstance(output_name, str) else name,
                size=size,
                media_type=media_type,
                resolver=vault,
            )
        if _is_string_mapping(value):
            return {
                str(key): await self._ingest_output(client, vault, child, f"{name}.{key}")
                for key, child in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            children = cast("Sequence[object]", value)
            return [
                await self._ingest_output(client, vault, child, f"{name}.{index}")
                for index, child in enumerate(children)
            ]
        return value

    async def _cancel_job(self, client: httpx.AsyncClient, token: str, job_id: str) -> None:
        with contextlib.suppress(Exception):
            await client.delete(
                f"{self.config.gateway_base}/v1/jobs/{job_id}",
                headers={"Authorization": f"Bearer {token}"},
            )

    async def invoke(
        self,
        remote: RemoteSchema,
        inputs: Mapping[str, object],
    ) -> Mapping[str, object]:
        if not self.config.gateway_base:
            raise GatewayError(
                "gateway_unconfigured",
                "client",
                False,
                "No remote gateway base URL is configured.",
            )
        token = self.config.auth_token()
        vault_root = os.environ.get("DINKSTER_ASSET_VAULT", "")
        if not vault_root:
            raise GatewayError(
                "vault_unconfigured",
                "client",
                False,
                "Remote output ingestion requires a configured Dinkster asset vault.",
            )
        vault = AssetVault(vault_root)
        invocation_key = str(uuid.uuid4())
        job_id: str | None = None
        is_video = self._is_video(remote.schema)
        poll_interval = (
            self.config.video_poll_interval if is_video else self.config.image_poll_interval
        )
        previous_progress: object = None
        seen_previews: set[str] = set()
        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            try:
                prepared = await self._prepare_inputs(client, token, inputs)
                response = await self._request(
                    client,
                    "POST",
                    f"{self.config.gateway_base}/v1/jobs",
                    token,
                    json_body={
                        "nodeType": remote.schema.node_type,
                        "schemaVersion": remote.schema.version,
                        "schemaSignature": remote.signature,
                        "inputs": prepared,
                        "routing": {"mode": "pooled"},
                        "resultDelivery": "client",
                        "waitSeconds": 0 if is_video else 30,
                    },
                    extra_headers={"Idempotency-Key": invocation_key},
                    accepted=(200, 202),
                )
                document = _json(response, "job submission response")
                raw_job = document.get("job", document)
                if not _is_string_mapping(raw_job):
                    raise GatewayError(
                        "malformed_response", "gateway", False, "Job submission returned no job."
                    )
                job = raw_job
                while True:
                    identifier = job.get("jobId")
                    if isinstance(identifier, str) and identifier:
                        job_id = identifier
                    if job_id is None:
                        raise GatewayError(
                            "malformed_response", "gateway", False, "Remote job has no jobId."
                        )
                    previous_progress = await self._forward_status(
                        client, job, previous_progress, seen_previews
                    )
                    state = job.get("state")
                    if state == "succeeded":
                        outputs = job.get("outputs")
                        if not _is_string_mapping(outputs):
                            raise GatewayError(
                                "malformed_output",
                                "gateway",
                                False,
                                "Succeeded remote job returned no output mapping.",
                            )
                        return {
                            name: await self._ingest_output(client, vault, value, name)
                            for name, value in outputs.items()
                        }
                    if state == "failed":
                        error = _gateway_error_from_mapping(job.get("error", {}))
                        if error.code in (
                            "schema_version_unsupported",
                            "schema_signature_mismatch",
                        ):
                            await self._refresh_after_schema_error()
                        raise error
                    if state == "cancelled":
                        raise GatewayError(
                            "job_cancelled", "gateway", False, "The remote job was cancelled."
                        )
                    if state not in ("queued", "running"):
                        raise GatewayError(
                            "malformed_response",
                            "gateway",
                            False,
                            f"Remote job has unknown state {state!r}.",
                        )
                    await _sleep_running(poll_interval)
                    polled = await self._request(
                        client,
                        "GET",
                        f"{self.config.gateway_base}/v1/jobs/{job_id}",
                        token,
                    )
                    job = _json(polled, "job poll response")
            except (asyncio.CancelledError, RemoteCancelled):
                if job_id is not None:
                    task = asyncio.create_task(self._cancel_job(client, token, job_id))
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(task)
                raise


def build_node_classes(runtime: RemoteRuntime) -> list[type[Node]]:
    classes: list[type[Node]] = []

    for index, remote in enumerate(runtime.snapshot.schemas):

        class RemoteNode(Node):
            _remote = remote

            @classmethod
            def define_schema(cls) -> NodeSchema:
                return cls._remote.schema

            @classmethod
            async def execute(cls, **inputs: object) -> Mapping[str, object]:
                return await runtime.invoke(cls._remote, inputs)

        RemoteNode.__name__ = f"RemoteCatalogNode{index}"
        RemoteNode.__qualname__ = RemoteNode.__name__
        classes.append(RemoteNode)
    return classes


__all__ = [
    "GatewayError",
    "RemoteConfig",
    "RemoteRuntime",
    "build_node_classes",
]
