#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-all}" != tests ]; then
uv run --no-sync python tools/gen_extension_contribution_kinds.py --check
uv run --no-sync python scripts/check_extension_factories.py \
  --baseline-ref "origin/${GITHUB_BASE_REF:-main}"
uv run --no-sync python scripts/check_builtin_type_spellings.py
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pyright &
linux_types=$!
uv run --no-sync pyright --pythonplatform Windows &
windows_types=$!
check_status=0
wait "$linux_types" || check_status=$?
wait "$windows_types" || check_status=$?
if [ "$check_status" != 0 ]; then exit "$check_status"; fi
fi
if [ "${1:-all}" = checks ]; then exit 0; fi

uv run --no-sync python -m pytest -q
