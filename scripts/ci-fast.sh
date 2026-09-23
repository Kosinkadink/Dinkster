#!/usr/bin/env bash
set -euo pipefail

uv run --locked python tools/gen_extension_contribution_kinds.py --check
uv run --locked python scripts/check_extension_factories.py \
  --baseline-ref "origin/${GITHUB_BASE_REF:-main}"
uv run --locked python scripts/check_family_isinstance_gates.py
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
uv run --locked python -m pytest -q \
  tests/test_extension_contract_pack.py \
  tests/test_extension_factory_guard.py \
  tests/test_family_isinstance_guard.py \
  tests/test_family_registration_gates.py \
  tests/test_release_install.py \
  tests/test_schema.py \
  tests/test_schema_current_contracts.py \
  tests/test_values.py \
  tests/test_graph.py \
  tests/test_graph_wire.py
