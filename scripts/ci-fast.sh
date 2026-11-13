#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-all}" != tests ]; then
python scripts/develop.py run -- python tools/gen_extension_contribution_kinds.py --check
python scripts/develop.py run -- python scripts/check_extension_factories.py \
  --baseline-ref "origin/${GITHUB_BASE_REF:-main}"
python scripts/develop.py run -- python scripts/check_builtin_type_spellings.py
python scripts/develop.py run -- ruff format --check .
python scripts/develop.py run -- ruff check .
python scripts/develop.py run -- pyright
python scripts/develop.py run -- pyright --pythonplatform Windows
fi
if [ "${1:-all}" = checks ]; then exit 0; fi

python scripts/develop.py run -- python -m pytest -q
