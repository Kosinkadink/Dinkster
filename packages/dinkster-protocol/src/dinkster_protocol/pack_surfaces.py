"""Bounded JSON contracts for host-owned pack routes and invocation events."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

PACK_ROUTES_SURFACE = "server.pack-routes"
PACK_EVENTS_SURFACE = "schema.pack-events"
PACK_JSON_MAX_BYTES = 64 * 1024
PACK_ROUTE_TIMEOUT = 10.0
PACK_ROUTE_PATH = "/api/extensions/{pack_id}/routes/{route_id}"


@dataclass(frozen=True)
class JsonField:
    """One required scalar field in a closed JSON object."""

    name: str
    type: str

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.name), str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", self.name
        ):
            raise ValueError("JSON field name must be an ASCII identifier")
        if self.type not in ("string", "integer", "number", "boolean"):
            raise ValueError("JSON field type must be string, integer, number, or boolean")


@dataclass(frozen=True)
class JsonObjectSchema:
    """A closed, immutable scalar-object schema; field order is canonical."""

    fields: tuple[JsonField, ...] = ()

    def __post_init__(self) -> None:
        raw = cast(object, self.fields)
        if not isinstance(raw, tuple) or not all(
            isinstance(item, JsonField) for item in cast(tuple[object, ...], raw)
        ):
            raise TypeError("fields must be a tuple of JsonField values")
        names = [item.name for item in self.fields]
        if names != sorted(set(names)):
            raise ValueError("JSON fields must be sorted by unique name")

    def validate(self, value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError("payload must be a JSON object")
        data = cast(Mapping[object, object], value)
        if set(data) != {item.name for item in self.fields}:
            raise ValueError("payload fields do not match the declared JSON schema")
        for item in self.fields:
            field = data[item.name]
            expected = {
                "string": (str,),
                "integer": (int,),
                "number": (int, float),
                "boolean": (bool,),
            }
            if type(field) not in expected[item.type]:
                raise ValueError(f"field {item.name!r} must be {item.type}")
            if type(field) is int and not -(2**53 - 1) <= field <= 2**53 - 1:
                raise ValueError(f"field {item.name!r} must be a JSON safe integer")
            if isinstance(field, float) and not math.isfinite(field):
                raise ValueError(f"field {item.name!r} must be finite")
        result = cast(dict[str, object], dict(data))
        if (
            len(json.dumps(result, ensure_ascii=True, allow_nan=False).encode())
            > PACK_JSON_MAX_BYTES
        ):
            raise ValueError("pack JSON payload exceeds 64 KiB")
        return result

    def to_wire(self) -> dict[str, str]:
        return {item.name: item.type for item in self.fields}

    @classmethod
    def from_wire(cls, raw: object) -> JsonObjectSchema:
        if not isinstance(raw, Mapping):
            raise ValueError("JSON schema must be an object of field names to scalar types")
        data = cast(Mapping[str, str], raw)
        return cls(
            tuple(
                sorted(
                    (JsonField(key, value) for key, value in data.items()),
                    key=lambda item: item.name,
                )
            )
        )


@dataclass(frozen=True)
class PackRoute:
    """A worker-local JSON handler, never a server framework callback.

    GET handlers accept an empty object. POST handlers accept the JSON body.
    Both return a declared JSON object and HTTP 200; the host owns errors,
    authentication, timeouts, headers, and the pack/route path namespace.
    """

    id: str
    method: str
    handler: str
    request: JsonObjectSchema = JsonObjectSchema()
    response: JsonObjectSchema = JsonObjectSchema()

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.id), str) or not re.fullmatch(
            r"[a-z][a-z0-9-]*", self.id
        ):
            raise ValueError("route id must be a lowercase ASCII slug")
        if self.method not in ("GET", "POST"):
            raise ValueError("pack route method must be GET or POST")
        if not isinstance(cast(object, self.handler), str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*",
            self.handler,
            re.ASCII,
        ):
            raise ValueError("route handler must be a module:attr reference")
        if not isinstance(cast(object, self.request), JsonObjectSchema) or not isinstance(
            cast(object, self.response), JsonObjectSchema
        ):
            raise TypeError("route request and response must be JsonObjectSchema values")
        if self.method == "GET" and self.request.fields:
            raise ValueError("GET pack routes must have an empty request schema")

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "method": self.method,
            "handler": self.handler,
            "request": self.request.to_wire(),
            "response": self.response.to_wire(),
        }

    @classmethod
    def from_wire(cls, raw: object) -> PackRoute:
        data = _fields(raw, {"id", "method", "handler", "request", "response"})
        return cls(
            cast(str, data["id"]),
            cast(str, data["method"]),
            cast(str, data["handler"]),
            JsonObjectSchema.from_wire(data["request"]),
            JsonObjectSchema.from_wire(data["response"]),
        )


def validate_pack_event_name(name: object) -> None:
    if not isinstance(name, str) or not re.fullmatch(
        r"[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+", name
    ):
        raise ValueError("pack event name must be dot-namespaced ASCII")


@dataclass(frozen=True)
class PackEvent:
    """A typed JSON event emitted during a node invocation via report_pack_event."""

    name: str
    payload: JsonObjectSchema = JsonObjectSchema()

    def __post_init__(self) -> None:
        validate_pack_event_name(self.name)
        if not isinstance(cast(object, self.payload), JsonObjectSchema):
            raise TypeError("event payload must be a JsonObjectSchema")

    def to_wire(self) -> dict[str, object]:
        return {
            "name": self.name,
            "payload": self.payload.to_wire(),
            "schemaVersion": 1,
            "scope": "execution",
            "delivery": "drop-oldest",
            "maxBytes": PACK_JSON_MAX_BYTES,
        }

    @classmethod
    def from_wire(cls, raw: object) -> PackEvent:
        if isinstance(raw, Mapping) and "schemaVersion" in raw:
            data = _fields(
                cast(object, raw),
                {"name", "payload", "schemaVersion", "scope", "delivery", "maxBytes"},
            )
            if (
                type(data["schemaVersion"]) is not int
                or data["schemaVersion"] != 1
                or data["scope"] != "execution"
                or data["delivery"] != "drop-oldest"
                or type(data["maxBytes"]) is not int
                or data["maxBytes"] != PACK_JSON_MAX_BYTES
            ):
                raise ValueError("unsupported pack event contract")
        else:
            data = _fields(cast(object, raw), {"name", "payload"})
        return cls(cast(str, data["name"]), JsonObjectSchema.from_wire(data["payload"]))


def _fields(raw: object, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or set(cast(Mapping[object, object], raw)) != fields:
        raise ValueError(f"declaration must have exactly these fields: {sorted(fields)}")
    return cast(Mapping[str, object], raw)


def validate_pack_surfaces(routes: object, events: object) -> None:
    for label, raw, cls in (("routes", routes, PackRoute), ("events", events, PackEvent)):
        if not isinstance(raw, tuple) or not all(
            isinstance(item, cls) for item in cast(tuple[object, ...], raw)
        ):
            raise TypeError(f"{label} must be a tuple of {cls.__name__} values")
    route_ids = [item.id for item in cast(tuple[PackRoute, ...], routes)]
    event_names = [item.name for item in cast(tuple[PackEvent, ...], events)]
    if route_ids != sorted(set(route_ids)) or event_names != sorted(set(event_names)):
        raise ValueError("routes and events must be sorted by unique id/name")


def pack_surfaces_from_wire(
    raw: Mapping[str, object],
) -> tuple[tuple[PackRoute, ...], tuple[PackEvent, ...]]:
    routes = raw.get("routes", [])
    events = raw.get("events", [])
    if not isinstance(routes, list) or not isinstance(events, list):
        raise ValueError("routes and events must be arrays")
    parsed_routes = tuple(PackRoute.from_wire(item) for item in cast(list[object], routes))
    parsed_events = tuple(PackEvent.from_wire(item) for item in cast(list[object], events))
    validate_pack_surfaces(parsed_routes, parsed_events)
    return parsed_routes, parsed_events


def pack_surfaces_to_wire(
    routes: tuple[PackRoute, ...], events: tuple[PackEvent, ...]
) -> dict[str, object]:
    # Absence preserves the canonical identity of extensions without these surfaces.
    return {
        **({"routes": [item.to_wire() for item in routes]} if routes else {}),
        **({"events": [item.to_wire() for item in events]} if events else {}),
    }


def report_pack_event(event: PackEvent, data: Mapping[str, object]) -> None:
    """Validate a declared JSON event and report it in the current invocation."""
    from dinkster_schema import report_event

    report_event(event.name, event.payload.validate(data))
