"""Shared async runtime for partner operation descriptors.

Constants and friendly messages are transcribed from comfy_api_nodes/util at
ComfyUI e651b7be. Network I/O sits behind Transport so CI never contacts a
provider. A production worker lazily creates one pooled httpx client.
"""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import ipaddress
import json
import math
import os
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import MappingProxyType, TracebackType
from typing import Any, Protocol, TypeVar, cast
from urllib.parse import urljoin, urlparse

from dinkster_api.v1 import pack_logger, report_progress

from .helper_registry import HelperRefusal, HelperRegistry
from .opspec import (
    BatchMapJoin,
    CheckInputs,
    Class3Spec,
    Cond,
    DownloadDecode,
    EncodeMedia,
    FixedField,
    FormatField,
    HelperCall,
    HttpSyncJson,
    InputBinding,
    LocalProgress,
    MaskPrepare,
    MediaConstraints,
    OpSpec,
    ProxyUpload,
    ResponseSelect,
    SubmitPoll,
    ValueConstruct,
)

RETRY_STATUSES = frozenset({408, 500, 502, 503, 504})
DEFAULT_RATE_LIMIT_RETRIES = 16
MAX_RETRY_AFTER_SECONDS = 150.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_DOWNLOAD_CAP = 1024 * 1024 * 1024
MAX_REDIRECTS = 3
DEFAULT_API_BASE = "https://api.comfy.org"
EGRESS_PROXY_ENV = "DINKSTER_EGRESS_PROXY"

_DENIED_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)
_LOG = pack_logger("partner")


class PartnerError(RuntimeError):
    pass


class MissingApiKeyError(PartnerError):
    pass


class TrustPolicyError(PartnerError):
    pass


class LocalNetworkError(PartnerError):
    pass


class ApiServerError(PartnerError):
    pass


class OperationCancelled(PartnerError):
    pass


class TransportNetworkError(OSError):
    """Transport-level connectivity failure eligible for retry/diagnosis."""


class TransportResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    async def json(self) -> object: ...

    def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


class Transport(Protocol):
    async def resolve(self, host: str, port: int) -> tuple[str, ...]: ...

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object] | None,
        timeout: float,
    ) -> TransportResponse: ...

    async def request_bytes(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> TransportResponse: ...

    async def internet_accessible(self) -> bool: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class RuntimeContext:
    transport: Transport
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    job_id: str = ""
    cancelled: Callable[[], bool] = lambda: False
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    progress: Callable[[int, int, str], None] = lambda step, total, text: report_progress(
        step, total, text=text
    )

    @classmethod
    def from_environment(cls, transport: Transport) -> RuntimeContext:
        return cls(
            transport=transport,
            api_key=os.environ.get("DINKSTER_COMFY_API_KEY", ""),
            api_base=os.environ.get("DINKSTER_COMFY_API_BASE", DEFAULT_API_BASE),
        )


T = TypeVar("T")


def _check_cancelled(ctx: RuntimeContext) -> None:
    if ctx.cancelled():
        raise OperationCancelled("Partner operation cancelled")


async def _await(ctx: RuntimeContext, pending: Awaitable[T]) -> T:
    _check_cancelled(ctx)
    value = await pending
    _check_cancelled(ctx)
    return value


async def _request(
    ctx: RuntimeContext,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    json_body: Mapping[str, object] | None,
    timeout: float,
) -> TransportResponse:
    """Cancellation-safe transport await that closes a just-arrived response."""
    _check_cancelled(ctx)
    response = await _await_interruptibly(
        ctx,
        ctx.transport.request(
            method,
            url,
            headers=headers,
            json_body=json_body,
            timeout=timeout,
        ),
    )
    if ctx.cancelled():
        await response.close()
        raise OperationCancelled("Partner operation cancelled")
    return response


async def _await_interruptibly(ctx: RuntimeContext, pending: Awaitable[T]) -> T:
    """Cancel an in-flight transport operation when local interruption is requested."""
    task = asyncio.ensure_future(pending)
    try:
        while not task.done():
            _check_cancelled(ctx)
            await asyncio.wait((task,), timeout=0.05)
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
            with suppress(BaseException):
                await task
        raise


async def _close(response: TransportResponse) -> None:
    """Cleanup must run even after cancellation has become visible."""
    await response.close()


def _retry_after(value: str | None, fallback: float, now: datetime | None = None) -> float:
    seconds: float | None = None
    if value is not None:
        stripped = value.strip()
        if stripped.isascii() and stripped.isdigit():
            seconds = float(stripped)
        elif stripped:
            try:
                parsed = parsedate_to_datetime(stripped)
            except (TypeError, ValueError, OverflowError):
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                current = now or datetime.now(UTC)
                seconds = max(0.0, (parsed - current).total_seconds())
    return fallback if seconds is None else min(seconds, MAX_RETRY_AFTER_SECONDS)


def _friendly_http_message(status: int, body: object) -> str:
    if status == 401:
        return "Unauthorized: Please login first to use this node."
    if status == 402:
        return "Payment Required: Please add credits to your account to use this node."
    if status == 409:
        return "There is a problem with your account. Please contact support@comfy.org."
    if status == 429:
        return (
            "Rate Limit Exceeded: The server returned 429 after all retry attempts. "
            "Please wait and try again."
        )
    if isinstance(body, dict):
        body_map = cast("dict[str, object]", body)
        error = body_map.get("error")
        if isinstance(error, dict):
            error_map = cast("dict[str, object]", error)
            raw_message = error_map.get("message")
            if not raw_message:
                return f"API Error: {json.dumps(body_map, sort_keys=True)}"
            message = str(raw_message)
            kind = error_map.get("type")
            return f"API Error: {message} (Type: {kind})" if kind else f"API Error: {message}"
        return f"API Error: {json.dumps(body_map, sort_keys=True)}"
    text = str(body)
    return f"API Error (raw): {text}" if len(text) <= 200 else f"API Error (status {status})"


def _comfy_headers(ctx: RuntimeContext) -> dict[str, str]:
    if not ctx.api_key:
        raise MissingApiKeyError("Unauthorized: Configure --comfy-api-key to use partner nodes.")
    headers = {
        "Accept": "application/json",
        "X-API-KEY": ctx.api_key,
        "Comfy-Usage-Source": "dinkster",
        "Comfy-Core-Version": "dinkster-0.0.1",
    }
    if ctx.job_id:
        headers["Comfy-Job-Id"] = ctx.job_id
    return headers


def _request_target(path: str, ctx: RuntimeContext) -> tuple[str, dict[str, str]]:
    parsed = urlparse(path)
    if parsed.scheme or parsed.netloc:
        return path, {"Accept": "application/json"}
    return (
        urljoin(ctx.api_base.rstrip("/") + "/", path.lstrip("/")),
        _comfy_headers(ctx),
    )


def _host_for_error(url: str) -> str:
    return urlparse(url).hostname or url


async def _validate_absolute_url(
    url: str, ctx: RuntimeContext, *, from_proxy: bool = False
) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.scheme != "https" or not host:
        raise TrustPolicyError(
            f"Rejected partner URL host {_host_for_error(url)!r}: HTTPS is required"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise TrustPolicyError(f"Rejected partner URL host {host!r}: invalid port") from exc
    if port not in (None, 443) and not from_proxy:
        raise TrustPolicyError(f"Rejected partner URL host {host!r}: only port 443 is allowed")
    resolved = await _await(ctx, ctx.transport.resolve(host, port or 443))
    if not resolved:
        raise TrustPolicyError(f"Rejected partner URL host {host!r}: DNS returned no addresses")
    for value in resolved:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise TrustPolicyError(
                f"Rejected partner URL host {host!r}: resolver returned invalid address"
            ) from exc
        if any(address in network for network in _DENIED_NETWORKS):
            raise TrustPolicyError(
                f"Rejected partner URL host {host!r}: resolved to a private or local address"
            )


async def _sleep(ctx: RuntimeContext, seconds: float) -> None:
    await _await(ctx, ctx.sleep(seconds))


async def _json_request(
    step: HttpSyncJson,
    inputs: Mapping[str, object],
    ctx: RuntimeContext,
    raw_inputs: Mapping[str, object] | None = None,
) -> object:
    path = step.path
    if step.path_input:
        choice = _select(inputs, tuple(step.path_input.split(".")))
        try:
            path = dict(step.paths)[str(choice)]
        except KeyError as exc:
            raise PartnerError(f"unknown request variant {choice!r}") from exc
    url, headers = _request_target(path, ctx)
    selected = urlparse(path)
    if selected.scheme or selected.netloc:
        await _validate_absolute_url(url, ctx)
    body = _construct_body(step.body, step.fixed, inputs, step.formatted, raw_inputs=raw_inputs)

    is_comfy_request = not bool(selected.scheme or selected.netloc)
    ordinary_attempts = 0
    rate_limit_attempts = 0
    delay = step.retry_delay
    rate_delay = step.retry_delay
    while True:
        response: TransportResponse | None = None
        try:
            response = await _request(
                ctx,
                step.method,
                url,
                headers=headers,
                json_body=body if step.method != "GET" else None,
                timeout=step.timeout,
            )
            payload = await _await_interruptibly(ctx, response.json())
            if response.status < 400:
                if is_comfy_request:
                    raw_credits = response.headers.get("X-Comfy-Credits-Used")
                    if raw_credits:
                        try:
                            credits = float(raw_credits)
                        except ValueError:
                            credits = -1.0
                        if math.isfinite(credits) and credits >= 0:
                            ctx.progress(1, 1, f"Credits used: {credits:g}")
                            _LOG.info("partner request used %g Comfy credits", credits)
                return payload
            wait: float | None = None
            if response.status == 429 and rate_limit_attempts < step.max_rate_limit_retries:
                rate_limit_attempts += 1
                wait = min(rate_delay, 30.0)
                rate_delay *= step.retry_backoff
            elif response.status in RETRY_STATUSES and ordinary_attempts < step.max_retries:
                ordinary_attempts += 1
                wait = delay
                delay *= step.retry_backoff
            if wait is None:
                raise PartnerError(_friendly_http_message(response.status, payload))
            wait = _retry_after(response.headers.get("Retry-After"), wait)
            await _close(response)
            response = None
            await _sleep(ctx, wait)
        except TransportNetworkError as exc:
            if ordinary_attempts < step.max_retries:
                ordinary_attempts += 1
                await _sleep(ctx, delay)
                delay *= step.retry_backoff
                continue
            accessible = await _await(ctx, ctx.transport.internet_accessible())
            if not accessible:
                raise LocalNetworkError(
                    "Unable to connect to the API server due to local network issues. "
                    "Please check your internet connection and try again."
                ) from exc
            raise ApiServerError(
                "The remote API service appears unreachable at this time."
            ) from exc
        finally:
            if response is not None:
                await _close(response)


def _assign_target(
    body: dict[str, object],
    target: str,
    value: object,
    tracked: list[tuple[tuple[str, ...], str]],
    *,
    literal: bool = False,
) -> None:
    parts = (target,) if literal else tuple(target.split("."))
    for prior_parts, prior_target in tracked:
        common = min(len(parts), len(prior_parts))
        if parts[:common] == prior_parts[:common]:
            raise PartnerError(f"request body conflict between {prior_target!r} and {target!r}")
    current = body
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            current[part] = value
        else:
            existing = current.get(part)
            if existing is None:
                nested: dict[str, object] = {}
                current[part] = nested
                current = nested
            elif isinstance(existing, dict):
                current = cast("dict[str, object]", existing)
            else:
                raise AssertionError("untracked request body conflict")
    tracked.append((parts, target))


def _construct_body(
    bindings: tuple[InputBinding, ...],
    fixed: tuple[FixedField, ...],
    inputs: Mapping[str, object],
    formatted: tuple[FormatField, ...] = (),
    *,
    raw_inputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {}
    tracked: list[tuple[tuple[str, ...], str]] = []
    for field in fixed:
        _assign_target(body, field.target, field.value, tracked)
    format_source = inputs if raw_inputs is None else raw_inputs
    for field in formatted:

        def replace_input(match: re.Match[str], target: str = field.target) -> str:
            input_id = match.group(1)
            value = format_source.get(input_id)
            if input_id not in format_source or value is None:
                raise PartnerError(
                    f"Missing input {input_id!r} for formatted request field {target!r}"
                )
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise PartnerError(
                    f"formatted request field {target!r} input {input_id!r} "
                    "must be a string or integer"
                )
            return str(value)

        rendered = re.sub(r"\{([^{}]+)\}", replace_input, field.template)
        _assign_target(body, field.target, rendered, tracked)
    for binding in bindings:
        if binding.input not in inputs:
            if binding.present_if and inputs.get(binding.present_if) is None:
                continue
            raise PartnerError(
                f"Missing input {binding.input!r} for request field {binding.target!r}"
            )
        value = inputs[binding.input]
        if binding.source_path:
            value = _select(value, binding.source_path)
        if binding.omit_if is not None and value == binding.omit_if:
            value = None
        elif binding.value_map is not None:
            if isinstance(value, bool):
                map_key = "true" if value else "false"
            elif isinstance(value, str):
                map_key = value
            else:
                raise PartnerError(
                    f"request field {binding.target!r} value_map binding {binding.input!r} "
                    "requires a string or boolean"
                )
            if map_key not in binding.value_map:
                raise PartnerError(
                    f"request field {binding.target!r} value {value!r} is missing from "
                    f"value_map binding {binding.input!r}"
                )
            value = binding.value_map[map_key]
        if binding.string_case == "lower" and value is not None:
            if not isinstance(value, str):
                raise PartnerError(f"request input {binding.input!r} must be a string for lower")
            value = value.lower()
        if binding.round_digits is not None and isinstance(value, (int, float)):
            value = round(value, binding.round_digits)
        if binding.present_if and inputs.get(binding.present_if) is None:
            continue
        if value is None and binding.omit_none:
            continue
        if binding.expand:
            if not isinstance(value, Mapping):
                raise PartnerError(f"Expanded request input {binding.input!r} must be a mapping")
            for key, item in cast("Mapping[object, object]", value).items():
                if not isinstance(key, str):
                    raise PartnerError("Expanded request input keys must be strings")
                _assign_target(body, key, item, tracked, literal=True)
            continue
        _assign_target(body, binding.target, value, tracked)
    return body


def _select(value: object, path: tuple[str | int, ...]) -> object:
    current = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list):
                raise PartnerError(f"response path index {part} is unavailable")
            items = cast("list[object]", current)
            if part >= len(items) or part < -len(items):
                raise PartnerError(f"response path index {part} is unavailable")
            current = items[part]
        else:
            if not isinstance(current, dict) or part not in current:
                raise PartnerError(f"response path field {part!r} is unavailable")
            current = cast("dict[str, object]", current)[part]
    return current


def _select_candidate(value: object, paths: tuple[tuple[str | int, ...], ...]) -> object:
    for path in paths:
        try:
            selected = _select(value, path)
        except PartnerError:
            continue
        if selected is not None and selected != []:
            return selected
    raise PartnerError(f"response candidate paths are unavailable: {paths!r}")


def _count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        values = cast("Mapping[object, object]", value).values()
    else:
        values = (value,)
    total = 0
    for item in values:
        if item is None:
            continue
        shape = getattr(item, "shape", ())
        total += int(shape[0]) if len(shape) == 4 else 1
    return total


def _condition(cond: Cond, inputs: Mapping[str, object]) -> bool:
    value = _condition_value(cond.input, inputs)

    if cond.op == "present":
        return value is not None
    if cond.op == "absent":
        return value is None
    if cond.op in ("eq", "ne"):
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise PartnerError(f"condition input {cond.input!r} must be scalar")
        return (value == cond.value) if cond.op == "eq" else (value != cond.value)
    if cond.op == "strip_min_len":
        if not isinstance(value, str):
            raise PartnerError(f"condition input {cond.input!r} must be a string")
        return len(value.strip()) >= cast("int", cond.value)
    count = _count(value)
    expected = cast("int", cond.value)
    return {
        "count_eq": count == expected,
        "count_le": count <= expected,
        "count_ge": count >= expected,
    }[cond.op]


def _condition_value(input_id: str, inputs: Mapping[str, object]) -> object:
    value = inputs.get(input_id)
    if input_id not in inputs:
        prefix = input_id + "."
        family = {
            key.removeprefix(prefix): item for key, item in inputs.items() if key.startswith(prefix)
        }
        if family:
            value = family
    return value


def _plain_json(value: object, label: str) -> object:
    def mutable(item: object) -> object:
        if isinstance(item, Mapping):
            mapping = cast("Mapping[object, object]", item)
            result: dict[str, object] = {}
            for key, child in mapping.items():
                if not isinstance(key, str):
                    raise TypeError("mapping keys must be str")
                result[key] = mutable(child)
            return result
        if isinstance(item, (list, tuple)):
            return [mutable(child) for child in cast("list[object] | tuple[object, ...]", item)]
        return item

    try:
        return json.loads(json.dumps(mutable(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PartnerError(f"{label} must contain only finite JSON values") from exc


def _helper_payload(
    call: HelperCall,
    inputs: Mapping[str, object],
    state: Mapping[str, object],
) -> Mapping[str, object]:
    payload = cast("dict[str, object]", _plain_json(dict(call.fixed), "helper fixed payload"))
    for binding in call.bindings:
        source = inputs if binding.source_kind == "input" else state
        if binding.mode == "family":
            prefix = binding.source + "."
            family = {}
            if binding.source in inputs:
                family[binding.source] = inputs[binding.source]
            family.update(
                {
                    key.removeprefix(prefix): value
                    for key, value in inputs.items()
                    if key.startswith(prefix)
                }
            )
            if not family:
                if binding.optional:
                    continue
                raise PartnerError(f"helper binding source {binding.source!r} is unavailable")
            value: object = family
        elif binding.mode == "present":
            value = binding.source in source and source[binding.source] is not None
        else:
            if binding.source not in source:
                if binding.optional:
                    continue
                raise PartnerError(f"helper binding source {binding.source!r} is unavailable")
            value = source[binding.source]
            if binding.source_path:
                try:
                    value = _select(value, binding.source_path)
                except PartnerError:
                    if binding.optional:
                        continue
                    raise
        payload[binding.name] = _plain_json(value, f"helper binding {binding.name!r}")
    return MappingProxyType(payload)


def _invoke_helper(
    call: HelperCall,
    inputs: Mapping[str, object],
    state: dict[str, object],
    registry: HelperRegistry,
) -> Mapping[str, object]:
    try:
        raw = cast(
            "object",
            registry.invoke(call.helper_id, call.stage, _helper_payload(call, inputs, state)),
        )
    except HelperRefusal as exc:
        raise PartnerError(str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise PartnerError(f"helper call {call.id!r} result must be a mapping")
    result = cast("dict[str, object]", _plain_json(dict(raw), f"helper call {call.id!r} result"))
    if set(result) != set(call.outputs):
        raise PartnerError(
            f"helper call {call.id!r} must return exactly declared outputs {call.outputs!r}"
        )
    for key, value in result.items():
        state[f"{call.id}.{key}"] = value
    return result


async def run_op(
    spec: OpSpec,
    inputs: Mapping[str, object],
    ctx: RuntimeContext,
    helper_registry: HelperRegistry | None = None,
) -> Mapping[str, object]:
    state: dict[str, object] = {}
    outputs: dict[str, object] = {}
    if spec.helper_calls:
        if helper_registry is None:
            raise PartnerError("helper-bearing OpSpec requires a helper registry")
        try:
            for call in spec.helper_calls:
                helper_registry.resolve(call.helper_id, call.stage)
        except HelperRefusal as exc:
            raise PartnerError(str(exc)) from exc
    for adapter in spec.adapters:
        _check_cancelled(ctx)
        if helper_registry is not None:
            for call in spec.helper_calls:
                if call.placement == "before" and call.anchor == adapter.id:
                    _invoke_helper(call, inputs, state, helper_registry)
        incomplete = (
            isinstance(adapter, SubmitPoll)
            and not adapter.source
            or isinstance(adapter, (ProxyUpload, EncodeMedia, MediaConstraints))
            and not adapter.input
            or isinstance(adapter, MaskPrepare)
            and (not adapter.mask_input or not adapter.image_input)
            or isinstance(adapter, DownloadDecode)
            and (
                not adapter.source
                or not (adapter.url_path or adapter.url_paths or adapter.items_path)
                or not adapter.output
            )
        )
        if incomplete:
            raise NotImplementedError(
                f"adapter {adapter.kind!r} is typed but not implemented in this partner slice"
            )
        if isinstance(adapter, CheckInputs):
            for check in adapter.checks:
                if all(_condition(cond, inputs) for cond in check.when) and not all(
                    _condition(cond, inputs) for cond in check.require
                ):
                    message = check.message
                    if "{count}" in message:
                        count_input = next(
                            cond.input for cond in check.require if cond.op.startswith("count_")
                        )
                        value = _condition_value(count_input, inputs)
                        message = message.format(count=_count(value))
                    raise PartnerError(message)
            state[adapter.id] = None
        elif isinstance(adapter, ValueConstruct):
            value = _construct_body(
                adapter.bindings,
                adapter.fixed,
                {**inputs, **state},
                adapter.formatted,
                raw_inputs=inputs,
            )
            state[adapter.id] = value
            outputs[adapter.output] = value
        elif isinstance(adapter, BatchMapJoin):
            joined: list[object] = []
            segments = adapter.segments
            if not segments:
                from .opspec import Segment

                segments = (Segment(adapter.source, wrap_key=adapter.wrap_key, mode="mapping"),)
            for segment in segments:
                source = state.get(segment.source, inputs.get(segment.source))
                if segment.mode.endswith("optional") and source is None:
                    continue
                if segment.mode.startswith("verbatim"):
                    if not isinstance(source, Mapping):
                        raise PartnerError(
                            f"batch_map_join segment {segment.source!r} must be a mapping "
                            "with string keys"
                        )
                    raw_source = cast("Mapping[object, object]", source)
                    if not all(isinstance(key, str) for key in raw_source):
                        raise PartnerError(
                            f"batch_map_join segment {segment.source!r} must be a mapping "
                            "with string keys"
                        )
                    joined.append(dict(cast("Mapping[str, object]", raw_source)))
                    continue
                segment_values: Iterable[tuple[object, object]]
                if segment.mode in ("mapping", "mapping_values"):
                    if not isinstance(source, Mapping):
                        raise PartnerError(
                            f"batch_map_join segment {segment.source!r} must be a mapping"
                        )
                    segment_values = cast("Mapping[object, object]", source).items()
                else:
                    segment_values = cast(
                        "Iterable[tuple[object, object]]", ((segment.source, source),)
                    )
                for key, value in segment_values:
                    if segment.mode in ("mapping", "mapping_values") and not isinstance(key, str):
                        raise PartnerError(f"batch_map_join key {key!r} must be a string")
                    if value is None:
                        if segment.mode == "single":
                            raise PartnerError(
                                f"batch_map_join segment {segment.source!r} must not be null"
                            )
                        continue
                    if not isinstance(value, str):
                        raise PartnerError(f"batch_map_join value for key {key!r} must be a string")
                    if segment.mode == "mapping_values":
                        joined.append(cast("Any", value))
                    else:
                        joined.append({**segment.fixed, cast("str", segment.wrap_key): value})
            count = len(joined)
            if adapter.min_items is not None and count < adapter.min_items:
                raise PartnerError(adapter.min_message.format(count=count))
            if adapter.max_items is not None and count > adapter.max_items:
                raise PartnerError(adapter.max_message.format(count=count))
            state[adapter.id] = joined
        elif isinstance(adapter, HttpSyncJson):
            state[adapter.id] = await _json_request(
                adapter, {**inputs, **state}, ctx, raw_inputs=inputs
            )
        elif isinstance(adapter, EncodeMedia):
            value = state.get(adapter.input, inputs.get(adapter.input))
            if value is None and adapter.batch_targets:
                prefix = adapter.input + "."
                value = {
                    key.removeprefix(prefix): item
                    for key, item in {**inputs, **state}.items()
                    if key.startswith(prefix)
                }
            if value is None and adapter.optional:
                state[adapter.id] = {} if adapter.batch_targets else None
            elif (
                adapter.optional
                and adapter.batch_targets
                and isinstance(value, Mapping)
                and not value
            ):
                state[adapter.id] = {}
            elif adapter.batch_targets:
                current_value = cast("object", value)
                if adapter.source_path:
                    current_value = _select(current_value, adapter.source_path)
                values = (
                    tuple(cast("Mapping[object, object]", current_value).values())
                    if isinstance(current_value, Mapping)
                    else (current_value,)
                )
                images: list[object] = []
                for item in values:
                    if item is None:
                        continue
                    array = _array(item)
                    images.extend(
                        [array] if array.ndim != 4 else [array[i] for i in range(array.shape[0])]
                    )
                if len(images) > len(adapter.batch_targets):
                    raise PartnerError(
                        f"media count exceeds maximum of {len(adapter.batch_targets)}"
                    )
                single = replace(adapter, batch_targets=(), source_path=())
                state[adapter.id] = {
                    adapter.batch_targets[index]: _encode_media(single, image)
                    for index, image in enumerate(images)
                }
            else:
                state[adapter.id] = _encode_media(adapter, cast("object", value))
        elif isinstance(adapter, MediaConstraints):
            value = state.get(adapter.input, inputs.get(adapter.input))
            if value is None and adapter.optional:
                state[adapter.id] = None
            elif adapter.batch:
                if not isinstance(value, Mapping):
                    raise PartnerError(
                        f"media_constraints input {adapter.input!r} must be a mapping"
                    )
                for key, item in cast("Mapping[object, object]", value).items():
                    if not isinstance(key, str):
                        raise PartnerError(
                            f"media_constraints input {adapter.input!r} keys must be strings"
                        )
                    try:
                        _validate_media(replace(adapter, batch=False), item)
                    except PartnerError as exc:
                        raise PartnerError(f"media_constraints item {key!r}: {exc}") from exc
                state[adapter.id] = value
            else:
                if value is None:
                    raise PartnerError(f"media_constraints input {adapter.input!r} is unavailable")
                _validate_media(adapter, value)
                state[adapter.id] = value
        elif isinstance(adapter, MaskPrepare):
            state[adapter.id] = _prepare_mask(
                inputs[adapter.mask_input], inputs[adapter.image_input]
            )
        elif isinstance(adapter, SubmitPoll):
            submitted = state.get(adapter.source, inputs.get(adapter.source))
            if submitted is None:
                raise PartnerError(f"poll source {adapter.source!r} is unavailable")
            state[adapter.id] = await _poll(adapter, submitted, ctx)
        elif isinstance(adapter, ProxyUpload):
            value = state.get(adapter.input, inputs.get(adapter.input))
            if value is None and adapter.optional:
                state[adapter.id] = None
            elif adapter.batch:
                if not isinstance(value, Mapping):
                    raise PartnerError("batch proxy_upload input must be a mapping")
                uploaded: dict[str, object] = {}
                for key, item in cast("Mapping[object, object]", value).items():
                    if not isinstance(key, str):
                        raise PartnerError("batch proxy_upload keys must be strings")
                    try:
                        uploaded[key] = await _upload(adapter, item, ctx)
                    except PartnerError as exc:
                        raise PartnerError(f"batch proxy_upload item {key!r}: {exc}") from exc
                state[adapter.id] = uploaded
            else:
                state[adapter.id] = await _upload(adapter, value, ctx)
        elif isinstance(adapter, DownloadDecode):
            source = state[adapter.source]
            if adapter.items_path:
                raw_items = _select(source, adapter.items_path)
                if not isinstance(raw_items, list):
                    raise PartnerError("download items_path must select a list")
                urls: list[object] = []
                for item in cast("list[object]", raw_items):
                    if item is None:
                        continue
                    try:
                        urls.append(_select(item, adapter.item_url_path))
                    except PartnerError:
                        continue
            elif adapter.url_paths:
                candidate = _select_candidate(source, adapter.url_paths)
                if adapter.item_url_path:
                    if not isinstance(candidate, list):
                        raise PartnerError("download candidate path must select a list")
                    urls = []
                    for item in cast("list[object]", candidate):
                        if item is None:
                            continue
                        try:
                            urls.append(_select(item, adapter.item_url_path))
                        except PartnerError:
                            continue
                else:
                    urls = [candidate]
            else:
                urls = [_select(source, adapter.url_path)]
            decoded_items: list[object] = []
            for item in urls:
                if item is None:
                    continue
                if not isinstance(item, str):
                    raise PartnerError("download URL must be a string")
                raw = await download_bytes(
                    item, ctx, media_family=adapter.media_family, byte_cap=adapter.byte_cap
                )
                decoded_items.append(_decode_media(raw, adapter.media_family))
            if not decoded_items:
                raise PartnerError("download produced no media")
            if adapter.media_family == "image":
                numpy = importlib.import_module("numpy")
                decoded = numpy.concatenate(decoded_items, axis=0)
            else:
                decoded = decoded_items[0]
            state[adapter.id] = decoded
            outputs[adapter.output] = decoded
        elif isinstance(adapter, ResponseSelect):
            selected = (
                _select_candidate(state[adapter.source], adapter.paths)
                if adapter.paths
                else _select(state[adapter.source], adapter.path)
            )
            state[adapter.id] = selected
            outputs[adapter.output] = selected
        elif isinstance(adapter, LocalProgress):
            ctx.progress(adapter.step, adapter.total, adapter.text)
            state[adapter.id] = None
        else:
            raise NotImplementedError(
                f"adapter {adapter.kind!r} is typed but not implemented in this partner slice"
            )
        if helper_registry is not None:
            for call in spec.helper_calls:
                if call.placement == "after" and call.anchor == adapter.id:
                    _invoke_helper(call, inputs, state, helper_registry)
    return outputs


async def run_class3_op(
    spec: Class3Spec,
    inputs: Mapping[str, object],
    ctx: RuntimeContext,
    helper_registry: HelperRegistry,
) -> Mapping[str, object]:
    try:
        helper_registry.resolve(spec.selector.helper_id, spec.selector.stage)
        for variant in spec.variants.values():
            for call in variant.helper_calls:
                helper_registry.resolve(call.helper_id, call.stage)
    except HelperRefusal as exc:
        raise PartnerError(str(exc)) from exc
    payload = _helper_payload(spec.selector, inputs, {})
    try:
        raw = cast(
            "object",
            helper_registry.invoke(spec.selector.helper_id, spec.selector.stage, payload),
        )
    except HelperRefusal as exc:
        raise PartnerError(str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise PartnerError("class-3 selector result must be a mapping")
    result = cast("dict[str, object]", _plain_json(dict(raw), "class-3 selector result"))
    if set(result) != {"variant", "values"}:
        raise PartnerError("class-3 selector must return exactly variant and values")
    variant = result["variant"]
    values = result["values"]
    if not isinstance(variant, str) or variant not in spec.variants:
        raise PartnerError(f"unknown class-3 variant {variant!r}")
    if not isinstance(values, dict) or set(values) != set(spec.selector.outputs):
        raise PartnerError("class-3 selector values must match declared outputs")
    enriched = dict(inputs)
    for key, value in cast("dict[str, object]", values).items():
        derived = f"{spec.selector.id}.{key}"
        if derived in enriched:
            raise PartnerError(f"class-3 selector output collides with input {derived!r}")
        enriched[derived] = value
    return await run_op(spec.variants[variant], enriched, ctx, helper_registry)


def _array(value: object):
    numpy = importlib.import_module("numpy")
    result = numpy.asarray(value)
    if result.dtype.kind not in "fui" or result.ndim not in (2, 3, 4):
        raise PartnerError("media must be a numeric HWC or BHWC array")
    return result


def _encode_media(step: EncodeMedia, value: object) -> bytes | str:
    if step.media_family == "audio":
        if not isinstance(value, Mapping) or set(cast("Mapping[object, object]", value)) != {
            "waveform",
            "sample_rate",
        }:
            raise PartnerError("audio must contain exactly waveform and sample_rate")
        audio_value = cast("Mapping[str, object]", value)
        sample_rate = audio_value["sample_rate"]
        waveform = audio_value["waveform"]
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise PartnerError("audio sample_rate must be a positive integer")
        numpy = importlib.import_module("numpy")
        if not isinstance(waveform, numpy.ndarray):
            raise PartnerError("audio waveform must be a numpy array")
        waveform_array = cast("Any", waveform)
        if waveform_array.dtype != numpy.float32:
            raise PartnerError("audio waveform must have dtype float32")
        if (
            waveform_array.ndim != 3
            or waveform_array.shape[0] < 1
            or waveform_array.shape[1] not in (1, 2)
        ):
            raise PartnerError("audio waveform must have shape [B,C,T] with one or two channels")
        av = importlib.import_module("av")
        stream = io.BytesIO()
        with av.open(stream, mode="w", format="mp3") as container:
            audio = container.add_stream("libmp3lame", rate=sample_rate)
            layout = "stereo" if waveform_array.shape[1] == 2 else "mono"
            for batch in waveform_array:
                frame = av.AudioFrame.from_ndarray(batch, format="fltp", layout=layout)
                frame.sample_rate = sample_rate
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
        return _format_encoded(step, stream.getvalue())
    if isinstance(value, bytes):
        return _format_encoded(step, value)
    if step.media_family == "video":
        if isinstance(value, Mapping):
            video_value = cast("Mapping[str, object]", value)
            if set(video_value) != {"container", "bytes"}:
                raise PartnerError("video must contain exactly container and bytes")
            container_name, raw = video_value["container"], video_value["bytes"]
            if container_name not in ("mp4", "webm") or not isinstance(raw, bytes):
                raise PartnerError("video container must be mp4 or webm with raw bytes")
            return _format_encoded(step, raw)
        av = importlib.import_module("av")
        numpy = importlib.import_module("numpy")
        frames = _array(value)
        if frames.ndim == 3:
            frames = frames[None, ...]
        if frames.dtype.kind == "f":
            frames = numpy.clip(frames, 0, 1) * 255
        frames = frames.astype("uint8")
        stream = io.BytesIO()
        with av.open(stream, mode="w", format="mp4") as container:
            video = container.add_stream("libx264", rate=24)
            video.width = int(frames.shape[2])
            video.height = int(frames.shape[1])
            video.pix_fmt = "yuv420p"
            for pixels in frames:
                frame = av.VideoFrame.from_ndarray(pixels[..., :3], format="rgb24")
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
        raw = stream.getvalue()
        return _format_encoded(step, raw)
    numpy = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    array = _array(value)
    if array.ndim == 4:
        array = array[0]
    if step.rgb:
        array = array[..., :3]
    if array.dtype.kind == "f":
        array = numpy.clip(array, 0, 1) * 255
    array = array.astype("uint8")
    image = image_module.fromarray(array)
    if step.max_pixels and image.width * image.height > step.max_pixels:
        scale = (step.max_pixels / (image.width * image.height)) ** 0.5
        width = max(2, int(image.width * scale))
        height = max(2, int(image.height * scale))
        width -= width % 2
        height -= height % 2
        image = image.resize((width, height), image_module.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format=step.format)
    raw = stream.getvalue()
    return _format_encoded(step, raw)


def _format_encoded(step: EncodeMedia, raw: bytes) -> bytes | str:
    if step.output == "bytes":
        return raw
    encoded = base64.b64encode(raw).decode("ascii")
    if step.output == "base64":
        return encoded
    mime = {
        ("image", "PNG"): "image/png",
        ("image", "JPEG"): "image/jpeg",
        ("video", "MP4"): "video/mp4",
        ("audio", "MP3"): "audio/mpeg",
    }[(step.media_family, step.format)]
    return f"data:{mime};base64,{encoded}"


def _decode_media(raw: bytes, family: str) -> object:
    if family == "audio":
        raise PartnerError("audio download decode is not implemented")
    if family == "video":
        if raw[4:12] in (b"ftypisom", b"ftypmp42") or raw[4:8] == b"ftyp":
            return {"container": "mp4", "bytes": raw}
        if raw.startswith(b"\x1aE\xdf\xa3"):
            return {"container": "webm", "bytes": raw}
        raise PartnerError("downloaded video has unknown container magic")
    if family != "image":
        return raw
    numpy = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    image = image_module.open(io.BytesIO(raw)).convert("RGB")
    return numpy.asarray(image, dtype=numpy.float32)[None, ...] / 255.0


def _validate_media(step: MediaConstraints, value: object) -> None:
    duration_requested = step.min_duration is not None or step.max_duration is not None
    dimensions_requested = (
        step.min_width != 1
        or step.min_height != 1
        or step.max_width is not None
        or step.max_height is not None
    )
    if duration_requested and step.duration_media == "audio":
        if not isinstance(value, Mapping):
            raise PartnerError(f"audio input {step.input!r} must contain waveform and sample_rate")
        audio = cast("Mapping[str, object]", value)
        if set(audio) != {"waveform", "sample_rate"}:
            raise PartnerError(f"audio input {step.input!r} must contain waveform and sample_rate")
        sample_rate = audio["sample_rate"]
        waveform = audio["waveform"]
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise PartnerError(f"audio input {step.input!r} sample_rate must be a positive integer")
        numpy = importlib.import_module("numpy")
        if not isinstance(waveform, numpy.ndarray):
            raise PartnerError(f"audio input {step.input!r} waveform must be a tensor")
        waveform_array = cast("Any", waveform)
        if waveform_array.ndim < 1:
            raise PartnerError(f"audio input {step.input!r} waveform must be a tensor")
        duration = waveform_array.shape[-1] / int(sample_rate)
        epsilon = 1.0 / int(sample_rate)
        if step.min_duration is not None and duration + epsilon < step.min_duration:
            raise PartnerError(
                f"Audio duration must be at least {step.min_duration}s, "
                f"got {duration + epsilon:.2f}s"
            )
        if step.max_duration is not None and duration - epsilon > step.max_duration:
            raise PartnerError(
                f"Audio duration must be at most {step.max_duration}s, "
                f"got {duration - epsilon:.2f}s"
            )
        return
    video_mapping = isinstance(value, Mapping) and set(cast("Mapping[object, object]", value)) == {
        "container",
        "bytes",
    }
    if duration_requested or video_mapping:
        if not video_mapping:
            raise PartnerError("video must contain exactly container and bytes")
        video = cast("Mapping[str, object]", value)
        if set(video) != {"container", "bytes"}:
            raise PartnerError("video must contain exactly container and bytes")
        container_name = video["container"]
        raw = video["bytes"]
        if container_name not in ("mp4", "webm") or not isinstance(raw, bytes):
            raise PartnerError("video container must be mp4 or webm with raw bytes")
        if step.max_bytes is not None and len(raw) > step.max_bytes:
            raise PartnerError(
                f"media size must be at most {step.max_bytes} bytes, got {len(raw)} bytes"
            )
        if not duration_requested and not dimensions_requested:
            return
        try:
            av = importlib.import_module("av")
            with av.open(io.BytesIO(raw), format=container_name) as container:
                streams = [stream for stream in container.streams if stream.type == "video"]
                if not streams:
                    raise PartnerError("video has no video stream")
                stream = streams[0]
                duration = None
                if duration_requested:
                    duration = (
                        float(stream.duration * stream.time_base)
                        if stream.duration is not None and stream.time_base is not None
                        else float(container.duration / av.time_base)
                        if container.duration is not None
                        else None
                    )
                width = stream.width if dimensions_requested else None
                height = stream.height if dimensions_requested else None
        except PartnerError:
            raise
        except Exception as exc:
            label = "duration" if duration_requested else "dimensions"
            raise PartnerError(f"unable to probe video {label}") from exc
        if duration_requested:
            if duration is None:
                raise PartnerError("video duration is indeterminable")
            # Pinned to upstream validate_video_duration's 0.0001-second tolerance.
            epsilon = 0.0001
            if step.min_duration is not None and duration + epsilon < step.min_duration:
                raise PartnerError(f"video duration must be at least {step.min_duration:g} seconds")
            if step.max_duration is not None and duration > step.max_duration + epsilon:
                raise PartnerError(f"video duration must be at most {step.max_duration:g} seconds")
        if dimensions_requested:
            if (
                not isinstance(width, int)
                or isinstance(width, bool)
                or width <= 0
                or not isinstance(height, int)
                or isinstance(height, bool)
                or height <= 0
            ):
                raise PartnerError("video dimensions are indeterminable")
            if width < step.min_width:
                raise PartnerError(
                    f"media width must be at least {step.min_width}px, got {width}px"
                )
            if height < step.min_height:
                raise PartnerError(
                    f"media height must be at least {step.min_height}px, got {height}px"
                )
            if step.max_width is not None and width > step.max_width:
                raise PartnerError(f"media width must be at most {step.max_width}px, got {width}px")
            if step.max_height is not None and height > step.max_height:
                raise PartnerError(
                    f"media height must be at most {step.max_height}px, got {height}px"
                )
        return
    if isinstance(value, str) and step.min_aspect_ratio is not None:
        maximum = step.max_aspect_ratio
        assert maximum is not None
        parts = value.split(":")
        try:
            width, height = (int(part) for part in parts)
        except (TypeError, ValueError) as exc:
            raise PartnerError("aspect ratio must use integer width:height values") from exc
        if len(parts) != 2 or width <= 0 or height <= 0:
            raise PartnerError("aspect ratio must use positive width:height values")
        ratio = width / height
        in_range = (
            step.min_aspect_ratio < ratio < maximum
            if step.aspect_strict
            else step.min_aspect_ratio <= ratio <= maximum
        )
        if not in_range:
            raise PartnerError(
                f"aspect ratio must be between {step.min_aspect_ratio:g} and "
                f"{maximum:g}, got {ratio:g}"
            )
        return
    array = _array(cast("object", value))
    height, width = array.shape[-3:-1] if array.ndim >= 3 else array.shape[-2:]
    count = array.shape[0] if array.ndim == 4 else 1
    if width < step.min_width:
        raise PartnerError(f"media width must be at least {step.min_width}px, got {width}px")
    if height < step.min_height:
        raise PartnerError(f"media height must be at least {step.min_height}px, got {height}px")
    if step.max_width is not None and width > step.max_width:
        raise PartnerError(f"media width must be at most {step.max_width}px, got {width}px")
    if step.max_height is not None and height > step.max_height:
        raise PartnerError(f"media height must be at most {step.max_height}px, got {height}px")
    if step.max_count is not None and count > step.max_count:
        raise PartnerError(f"media count must be at most {step.max_count}, got {count}")
    if step.max_bytes is not None and array.nbytes > step.max_bytes:
        raise PartnerError(
            f"media size must be at most {step.max_bytes} bytes, got {array.nbytes} bytes"
        )
    if step.min_aspect_ratio is not None:
        maximum = step.max_aspect_ratio
        assert maximum is not None
        ratio = width / height
        in_range = (
            step.min_aspect_ratio < ratio < maximum
            if step.aspect_strict
            else step.min_aspect_ratio <= ratio <= maximum
        )
        if not in_range:
            raise PartnerError(
                f"aspect ratio must be between {step.min_aspect_ratio:g} and "
                f"{maximum:g}, got {ratio:g}"
            )


def _prepare_mask(mask: object, image: object) -> object:
    numpy = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    target = _array(image)
    source = _array(mask)
    if source.ndim == 2:
        source = source[None, ...]
    elif source.ndim == 4 and source.shape[-1] in (1, 3, 4):
        source = source[..., 0]
    if source.ndim != 3:
        raise PartnerError("mask must be a numeric HW or BHW array")
    height, width = target.shape[-3:-1]
    channels: list[object] = []
    for item in source:
        pixels = (numpy.clip(item, 0, 1) * 255).astype("uint8")
        resized = image_module.fromarray(pixels).resize(
            (width, height), image_module.Resampling.NEAREST
        )
        channels.append(numpy.asarray(resized, dtype=numpy.float32) / 255.0)
    batch = numpy.stack(channels)
    return numpy.repeat(batch[..., None], 3, axis=-1)


def _normal_status(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


async def _poll(step: SubmitPoll, submitted: object, ctx: RuntimeContext) -> object:
    if step.path_template:
        raw_value = _select(submitted, step.path_value_path)
        if (
            not isinstance(raw_value, (str, int))
            or isinstance(raw_value, bool)
            or not re.fullmatch(r"(?=.*[A-Za-z0-9])[A-Za-z0-9._-]+", str(raw_value))
            or str(raw_value) in (".", "..")
        ):
            raise PartnerError("poll path template value contains unsafe characters")
        url = step.path_template.replace("{value}", str(raw_value))
        if urlparse(url).scheme or urlparse(url).netloc:
            raise PartnerError("poll path template must produce a relative comfy-auth path")
    else:
        url = _select(submitted, step.url_path)
    if not isinstance(url, str):
        raise PartnerError("polling URL must be a string")
    attempts = 0
    try:
        while attempts < step.max_attempts:
            poll_spec = HttpSyncJson(
                step.id + "_request",
                url,
                method="GET",
                timeout=step.timeout,
                max_retries=step.max_retries,
                retry_delay=step.retry_delay,
                retry_backoff=step.retry_backoff,
            )
            payload = await _json_request(poll_spec, {}, ctx)
            status = _normal_status(_select(payload, step.status_path))
            if step.allowed and status not in {_normal_status(v) for v in step.allowed}:
                raise PartnerError(f"Partner returned unknown status {status!r}")
            if status in {_normal_status(v) for v in step.completed}:
                return payload
            if status in {_normal_status(v) for v in step.failed}:
                raise PartnerError(f"Partner operation failed with status {status!r}")
            if status not in {_normal_status(v) for v in step.queued}:
                attempts += 1
            progress = (
                _select(cast("object", payload), step.progress_path)
                if isinstance(payload, dict) and step.progress_path[0] in payload
                else None
            )
            ctx.progress(
                attempts,
                step.max_attempts,
                f"Status: {status}" + (f" ({progress})" if progress is not None else ""),
            )
            await _sleep(ctx, step.interval)
    except OperationCancelled:
        if step.cancel_path:
            try:
                await _json_request(
                    HttpSyncJson(
                        step.id + "_cancel",
                        step.cancel_path,
                        timeout=10.0,
                        max_retries=0,
                        max_rate_limit_retries=0,
                    ),
                    {},
                    replace(ctx, cancelled=lambda: False),
                )
            except BaseException:
                pass
        raise
    raise PartnerError(f"Partner operation timed out after {step.max_attempts} attempts")


async def _upload(step: ProxyUpload, value: object, ctx: RuntimeContext) -> str:
    content_type = step.content_type
    if isinstance(value, Mapping):
        video_value = cast("Mapping[str, object]", value)
        if set(video_value) != {"container", "bytes"}:
            raise PartnerError("proxy_upload video must contain exactly container and bytes")
        container_name, value = video_value["container"], video_value["bytes"]
        if container_name not in ("mp4", "webm") or not isinstance(value, bytes):
            raise PartnerError("proxy_upload video container must be mp4 or webm with raw bytes")
        content_type = f"video/{container_name}"
    if not isinstance(value, bytes):
        raise PartnerError("proxy_upload input must be raw bytes")
    allocation = await _json_request(
        HttpSyncJson(
            step.id + "_allocate",
            step.allocation_path,
            body=(
                InputBinding("file_name", "file_name"),
                InputBinding("content_type", "content_type"),
            ),
        ),
        {"file_name": step.file_name, "content_type": content_type},
        ctx,
    )
    upload_url = _select(allocation, ("upload_url",))
    download_url = _select(allocation, ("download_url",))
    if not isinstance(upload_url, str) or not isinstance(download_url, str):
        raise PartnerError("upload allocation response is invalid")
    await _validate_absolute_url(upload_url, ctx, from_proxy=True)
    raw_request = getattr(ctx.transport, "request_bytes", None)
    if raw_request is None:
        raise PartnerError("transport does not support raw byte uploads")
    delay = 1.0
    retries = 0
    while True:
        response: TransportResponse | None = None
        try:
            response = await _await_interruptibly(
                ctx,
                raw_request(
                    "PUT",
                    upload_url,
                    headers={"Content-Type": content_type},
                    body=value,
                    timeout=3600.0,
                ),
            )
            current = cast("TransportResponse", response)
            _check_cancelled(ctx)
            if current.status < 400:
                return download_url
            if current.status not in {408, 429, 500, 502, 503, 504} or retries >= 3:
                raise PartnerError(f"upload failed with status {current.status}")
        except TransportNetworkError as exc:
            if retries >= 3:
                accessible = await _await(ctx, ctx.transport.internet_accessible())
                if not accessible:
                    raise LocalNetworkError(
                        "Unable to connect to the network. "
                        "Please check your connection and try again."
                    ) from exc
                raise ApiServerError(
                    "The upload service appears unreachable at this time."
                ) from exc
        finally:
            if response is not None:
                await response.close()
        retries += 1
        await _sleep(ctx, delay)
        delay *= 2


async def download_bytes(
    url: str,
    ctx: RuntimeContext,
    *,
    media_family: str,
    byte_cap: int = DEFAULT_DOWNLOAD_CAP,
    from_proxy: bool = False,
) -> bytes:
    if byte_cap <= 0 or byte_cap > DEFAULT_DOWNLOAD_CAP:
        raise ValueError(f"download byte cap must be between 1 and {DEFAULT_DOWNLOAD_CAP}")
    current = url
    redirects = 0
    while True:
        await _validate_absolute_url(current, ctx, from_proxy=from_proxy)
        response = await _request(
            ctx,
            "GET",
            current,
            headers={},
            json_body=None,
            timeout=3600.0,
        )
        try:
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location") or response.headers.get("location")
                if not location:
                    host = _host_for_error(current)
                    raise TrustPolicyError(
                        f"Rejected partner URL host {host!r}: redirect has no location"
                    )
                if redirects >= MAX_REDIRECTS:
                    host = _host_for_error(current)
                    raise TrustPolicyError(
                        f"Rejected partner URL host {host!r}: redirect limit exceeded"
                    )
                redirects += 1
                current = urljoin(current, location)
                from_proxy = False
                continue
            if response.status >= 400:
                payload = await _await_interruptibly(ctx, response.json())
                raise PartnerError(_friendly_http_message(response.status, payload))
            content_type = response.headers.get(
                "Content-Type", response.headers.get("content-type", "")
            )
            actual_family = content_type.partition(";")[0].strip().partition("/")[0].lower()
            if actual_family != media_family.lower():
                raise TrustPolicyError(
                    f"Rejected partner URL host {_host_for_error(current)!r}: content type "
                    f"{content_type!r} is not {media_family}"
                )
            content_length = response.headers.get("Content-Length") or response.headers.get(
                "content-length"
            )
            if content_length and int(content_length) > byte_cap:
                host = _host_for_error(current)
                raise TrustPolicyError(
                    f"Rejected partner URL host {host!r}: download exceeds byte cap"
                )
            result = bytearray()
            iterator = response.iter_bytes(DOWNLOAD_CHUNK_BYTES)
            while True:
                try:
                    chunk = await _await_interruptibly(ctx, anext(iterator))
                except StopAsyncIteration:
                    break
                result.extend(chunk)
                if len(result) > byte_cap:
                    host = _host_for_error(current)
                    raise TrustPolicyError(
                        f"Rejected partner URL host {host!r}: download exceeds byte cap"
                    )
            return bytes(result)
        finally:
            await _close(response)


class _HttpxRawResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> object: ...

    async def aread(self) -> bytes: ...

    def aiter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...


class _HttpxResponse:
    def __init__(self, response: _HttpxRawResponse) -> None:
        self._response = response
        self.status = int(response.status_code)
        self.headers = response.headers

    async def json(self) -> object:
        data = await self._response.aread()
        return json.loads(data)

    async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_bytes(chunk_size):
            yield bytes(chunk)

    async def close(self) -> None:
        await self._response.aclose()


class _HttpcoreResponse:
    def __init__(self, response: Any, context: Any, headers: Mapping[str, str]) -> None:
        self._response = response
        self._context = context
        self._closed = False
        self.status_code = int(response.status)
        self.headers = headers

    async def aread(self) -> bytes:
        return bytes(await self._response.aread())

    async def aiter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_stream():
            data = bytes(chunk)
            for offset in range(0, len(data), chunk_size):
                yield data[offset : offset + chunk_size]

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._context.__aexit__(None, None, None)


def _unix_proxy_backend(httpcore: Any, socket_path: str) -> Any:
    class UnixProxyBackend(httpcore.AsyncNetworkBackend):
        def __init__(self) -> None:
            self._backend = httpcore.AnyIOBackend()

        async def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Iterable[object] | None = None,
        ) -> Any:
            del host, port, local_address
            return await self._backend.connect_unix_socket(
                socket_path,
                timeout=timeout,
                socket_options=socket_options,
            )

        async def connect_unix_socket(
            self,
            path: str,
            timeout: float | None = None,
            socket_options: Iterable[object] | None = None,
        ) -> Any:
            return await self._backend.connect_unix_socket(
                path,
                timeout=timeout,
                socket_options=socket_options,
            )

        async def sleep(self, seconds: float) -> None:
            await self._backend.sleep(seconds)

    return UnixProxyBackend()


def _proxy_authority(host: str, port: int) -> str:
    rendered = host.encode("idna").decode("ascii")
    if ":" in rendered:
        rendered = f"[{rendered}]"
    return f"{rendered}:{port}"


async def _proxy_resolve(socket_path: str, host: str, port: int) -> tuple[str, ...]:
    if sys.platform == "win32":
        raise TransportNetworkError("egress proxy requires Unix sockets")
    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(10.0):
            reader, writer = await asyncio.open_unix_connection(socket_path)
            authority = _proxy_authority(host, port)
            request = (
                f"RESOLVE {authority} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n\r\n"
            )
            writer.write(request.encode("ascii"))
            await writer.drain()
            status_line = (await reader.readline()).decode("ascii").strip()
            try:
                _version, status_text, _reason = status_line.split(" ", 2)
                status = int(status_text)
            except (UnicodeDecodeError, ValueError) as exc:
                raise TransportNetworkError("invalid egress proxy response") from exc
            headers: dict[str, str] = {}
            while line := await reader.readline():
                if line == b"\r\n":
                    break
                name, separator, value = line.decode("ascii").partition(":")
                if not separator:
                    raise TransportNetworkError("invalid egress proxy response")
                headers[name.lower()] = value.strip()
            if status != 200:
                return ()
            try:
                length = int(headers.get("content-length", ""))
            except ValueError as exc:
                raise TransportNetworkError("invalid egress proxy response") from exc
            if not 0 <= length <= 64 * 1024:
                raise TransportNetworkError("invalid egress proxy response")
            payload = cast("object", json.loads(await reader.readexactly(length)))
            addresses = (
                cast("dict[str, object]", payload).get("addresses")
                if isinstance(payload, dict)
                else None
            )
            if not isinstance(addresses, list):
                raise TransportNetworkError("invalid egress proxy response")
            address_items = cast("list[object]", addresses)
            if not all(isinstance(value, str) for value in address_items):
                raise TransportNetworkError("invalid egress proxy response")
            return tuple(cast("list[str]", address_items))
    except TransportNetworkError:
        raise
    except (OSError, TimeoutError, UnicodeError, ValueError, asyncio.IncompleteReadError) as exc:
        raise TransportNetworkError(str(exc)) from exc
    finally:
        if writer is not None:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()


class HttpxTransport:
    """One connection-pooled HTTP transport for the lifetime of a worker."""

    def __init__(self) -> None:
        httpx = importlib.import_module("httpx")
        self._httpx = httpx
        self._proxy_path = os.environ.get(EGRESS_PROXY_ENV, "")
        self._client = None if self._proxy_path else httpx.AsyncClient(follow_redirects=False)
        self._proxy_pool: Any | None = None
        if self._proxy_path:
            httpcore = importlib.import_module("httpcore")
            self._proxy_pool = httpcore.AsyncHTTPProxy(
                proxy_url="http://dinkster-egress",
                network_backend=_unix_proxy_backend(httpcore, self._proxy_path),
            )

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        if self._proxy_path:
            return await _proxy_resolve(self._proxy_path, host, port)
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=0, proto=0)
        return tuple(sorted({str(info[4][0]) for info in infos}))

    async def _proxy_request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        content: bytes | None,
        timeout: float,
    ) -> TransportResponse:
        assert self._proxy_pool is not None
        extensions = {
            "timeout": {
                "connect": timeout,
                "read": timeout,
                "write": timeout,
                "pool": timeout,
            }
        }
        context = self._proxy_pool.stream(
            method,
            url,
            headers=list(headers.items()),
            content=content,
            extensions=extensions,
        )
        try:
            async with asyncio.timeout(timeout):
                response = await context.__aenter__()
        except Exception as exc:
            raise TransportNetworkError(str(exc)) from exc
        raw = _HttpcoreResponse(response, context, self._httpx.Headers(response.headers))
        return _HttpxResponse(cast("_HttpxRawResponse", raw))

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object] | None,
        timeout: float,
    ) -> TransportResponse:
        if self._proxy_pool is not None:
            request_headers = dict(headers)
            content = None
            if json_body is not None:
                request_headers.setdefault("Content-Type", "application/json")
                content = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            return await self._proxy_request(
                method,
                url,
                headers=request_headers,
                content=content,
                timeout=timeout,
            )
        assert self._client is not None
        try:
            request = self._client.build_request(
                method, url, headers=dict(headers), json=json_body, timeout=timeout
            )
            async with asyncio.timeout(timeout):
                response = await self._client.send(request, stream=True)
        except Exception as exc:
            raise TransportNetworkError(str(exc)) from exc
        return _HttpxResponse(cast("_HttpxRawResponse", response))

    async def request_bytes(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> TransportResponse:
        if self._proxy_pool is not None:
            return await self._proxy_request(
                method,
                url,
                headers=headers,
                content=body,
                timeout=timeout,
            )
        assert self._client is not None
        try:
            request = self._client.build_request(
                method, url, headers=dict(headers), content=body, timeout=timeout
            )
            async with asyncio.timeout(timeout):
                response = await self._client.send(request, stream=True)
        except Exception as exc:
            raise TransportNetworkError(str(exc)) from exc
        return _HttpxResponse(cast("_HttpxRawResponse", response))

    async def internet_accessible(self) -> bool:
        if self._proxy_pool is not None:
            response: TransportResponse | None = None
            try:
                response = await self.request(
                    "HEAD",
                    os.environ.get("DINKSTER_COMFY_API_BASE", DEFAULT_API_BASE),
                    headers={"Accept": "application/json"},
                    json_body=None,
                    timeout=5.0,
                )
                return response.status < 500
            except Exception:
                return False
            finally:
                if response is not None:
                    await response.close()
        client = self._client
        assert client is not None

        async def probe(url: str) -> bool:
            try:
                response = await client.get(url, timeout=5.0)
                return bool(response.status_code < 500)
            except Exception:
                return False

        results = await asyncio.gather(
            probe("https://www.google.com"), probe("https://www.baidu.com")
        )
        return any(results)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        if self._proxy_pool is not None:
            await self._proxy_pool.aclose()

    async def __aenter__(self) -> HttpxTransport:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()


_worker_transport: HttpxTransport | None = None


def worker_runtime_context() -> RuntimeContext:
    """Return a context backed by the worker's one pooled HTTP client."""
    global _worker_transport
    if _worker_transport is None:
        _worker_transport = HttpxTransport()
    return RuntimeContext.from_environment(_worker_transport)
