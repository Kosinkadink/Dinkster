"""Hosted service defaults and explicit local-service overrides."""

from __future__ import annotations

import pytest

from dinkster.services import resolve_services


def test_production_defaults_and_custom_catalog_routing() -> None:
    production = resolve_services()
    assert production.name == "production"
    assert production.identity_base == "https://id.dinkster.org"
    assert production.catalog_base == production.gateway_base == "https://gateway.dinkster.org"
    assert production.gateway_audience == "gateway"

    custom = resolve_services(catalog_base="http://127.0.0.1:8123")
    assert custom.name == "custom"
    assert custom.catalog_base == custom.gateway_base == "http://127.0.0.1:8123"
    separate = resolve_services(
        catalog_base="http://127.0.0.1:8123", gateway_base="http://127.0.0.1:9234"
    )
    assert separate.catalog_base == custom.catalog_base
    assert separate.gateway_base == "http://127.0.0.1:9234"
    assert (
        resolve_services(gateway_base=separate.gateway_base).catalog_base == production.catalog_base
    )
    assert resolve_services(catalog_base="").catalog_base == ""
    assert resolve_services(catalog_base="").gateway_base == ""
    assert resolve_services() == production


def test_unknown_services_cannot_be_hidden_by_custom_addresses() -> None:
    with pytest.raises(ValueError, match="allowed: production"):
        resolve_services("nightly", catalog_base="http://127.0.0.1:8123")
