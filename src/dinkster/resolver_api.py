"""Management API for persistent declarative resolver-index subscriptions."""

from __future__ import annotations

import asyncio
import json
from typing import cast

from aiohttp import web
from dinkster_assets import (
    ResolverIndexError,
    ResolverSubscriptionError,
    ResolverSubscriptionStore,
)
from dinkster_server import STATE_KEY

__all__ = ["RESOLVER_INDEXES_KEY", "add_resolver_index_routes"]

RESOLVER_INDEXES_KEY: web.AppKey[ResolverSubscriptionStore] = web.AppKey(
    "dinkster_resolver_indexes"
)
RESOLVER_P2P_GRANTED_KEY: web.AppKey[bool] = web.AppKey("dinkster_resolver_p2p_granted")


def _json_error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


async def _json_body(request: web.Request) -> dict[str, object] | web.Response:
    try:
        raw = cast("object", await request.json())
    except json.JSONDecodeError as exc:
        return _json_error(400, f"invalid JSON: {exc}")
    if not isinstance(raw, dict):
        return _json_error(400, "request body must be an object")
    return cast("dict[str, object]", raw)


def _publish(app: web.Application) -> None:
    app[STATE_KEY].hub.publish(
        {"type": "resolver_indexes_changed"},
        client_id=None,
        droppable=False,
    )


async def handle_resolver_indexes_list(request: web.Request) -> web.Response:
    store = request.app[RESOLVER_INDEXES_KEY]
    subscriptions = await asyncio.to_thread(store.subscriptions)
    return web.json_response(
        {"subscriptions": [subscription.descriptor() for subscription in subscriptions]}
    )


async def handle_resolver_index_subscribe(request: web.Request) -> web.Response:
    body = await _json_body(request)
    if isinstance(body, web.Response):
        return body
    if set(body) != {"source"} or not isinstance(body.get("source"), str):
        return _json_error(400, "request body requires only a string 'source' field")
    store = request.app[RESOLVER_INDEXES_KEY]
    try:
        subscription = await asyncio.to_thread(store.subscribe, cast("str", body["source"]))
    except (ResolverIndexError, ResolverSubscriptionError) as exc:
        return _json_error(400, str(exc))
    _publish(request.app)
    return web.json_response(subscription.descriptor(), status=201)


async def handle_resolver_index_unsubscribe(request: web.Request) -> web.Response:
    store = request.app[RESOLVER_INDEXES_KEY]
    subscription_id = request.match_info["subscriptionId"]
    removed = await asyncio.to_thread(store.unsubscribe, subscription_id)
    if not removed:
        return _json_error(404, f"unknown resolver index subscription: {subscription_id}")
    _publish(request.app)
    return web.json_response({"removed": subscription_id})


async def handle_resolver_index_p2p_trust(request: web.Request) -> web.Response:
    if not request.app[RESOLVER_P2P_GRANTED_KEY]:
        return _json_error(403, "P2P trust settings are disabled by this host")
    body = await _json_body(request)
    if isinstance(body, web.Response):
        return body
    if set(body) != {"trustedForP2P", "licenseAuthoritative"} or any(
        type(body[field]) is not bool for field in body
    ):
        return _json_error(
            400,
            "request body requires boolean 'trustedForP2P' and 'licenseAuthoritative' fields",
        )
    store = request.app[RESOLVER_INDEXES_KEY]
    try:
        subscription = await asyncio.to_thread(
            store.set_p2p_trust,
            request.match_info["subscriptionId"],
            trusted_for_p2p=cast("bool", body["trustedForP2P"]),
            license_authoritative=cast("bool", body["licenseAuthoritative"]),
        )
    except ResolverSubscriptionError as exc:
        return _json_error(404, str(exc))
    _publish(request.app)
    return web.json_response(subscription.descriptor())


async def handle_resolver_indexes_refresh(request: web.Request) -> web.Response:
    body = await _json_body(request)
    if isinstance(body, web.Response):
        return body
    unknown = set(body) - {"id"}
    subscription_id = body.get("id")
    if unknown or (subscription_id is not None and not isinstance(subscription_id, str)):
        return _json_error(400, "request body accepts only an optional string 'id' field")
    store = request.app[RESOLVER_INDEXES_KEY]
    try:
        refreshed = await asyncio.to_thread(
            store.refresh,
            cast("str | None", subscription_id),
        )
    except ResolverSubscriptionError as exc:
        return _json_error(404, str(exc))
    if refreshed:
        _publish(request.app)
    return web.json_response({"refreshed": refreshed})


def add_resolver_index_routes(
    app: web.Application,
    store: ResolverSubscriptionStore,
    *,
    p2p_granted: bool = False,
    official_url: str | None = None,
    official_provider_id: str | None = None,
) -> None:
    app[RESOLVER_INDEXES_KEY] = store
    app[RESOLVER_P2P_GRANTED_KEY] = p2p_granted
    app.router.add_get("/api/assets/resolver-indexes", handle_resolver_indexes_list)
    app.router.add_post("/api/assets/resolver-indexes", handle_resolver_index_subscribe)
    app.router.add_post("/api/assets/resolver-indexes/refresh", handle_resolver_indexes_refresh)
    app.router.add_patch(
        "/api/assets/resolver-indexes/{subscriptionId}", handle_resolver_index_p2p_trust
    )
    app.router.add_delete(
        "/api/assets/resolver-indexes/{subscriptionId}", handle_resolver_index_unsubscribe
    )

    async def refresh_on_startup(app: web.Application) -> None:
        try:
            bootstrapped = await asyncio.to_thread(
                store.bootstrap_official, official_url, official_provider_id
            )
        except (OSError, ResolverIndexError, ResolverSubscriptionError) as exc:
            app.logger.warning("official resolver bootstrap refused: %s", exc)
        else:
            if bootstrapped is not None:
                _publish(app)
        try:
            refreshed = await asyncio.to_thread(store.refresh)
        except (OSError, ResolverSubscriptionError) as exc:
            app.logger.warning("resolver index startup refresh failed: %s", exc)
            return
        if refreshed:
            _publish(app)

    app.on_startup.append(refresh_on_startup)
