#!/usr/bin/env bash
set -euo pipefail

# Dependency and build-backend provisioning belongs to setup, not the gate.
export UV_OFFLINE=1

if [ "${1:-all}" != tests ]; then
uv run --locked python tools/gen_extension_contribution_kinds.py --check
uv run --locked python scripts/check_extension_factories.py \
  --baseline-ref "origin/${GITHUB_BASE_REF:-main}"
uv run --locked python scripts/check_builtin_type_spellings.py
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright &
linux_types=$!
uv run --locked pyright --pythonplatform Windows &
windows_types=$!
check_status=0
wait "$linux_types" || check_status=$?
wait "$windows_types" || check_status=$?
if [ "$check_status" != 0 ]; then exit "$check_status"; fi
fi
if [ "${1:-all}" = checks ]; then exit 0; fi

uv run --locked python -m pytest -q
