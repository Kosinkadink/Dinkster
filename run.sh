#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo 'error: uv is required: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
exec uv run --no-project --python 3.12 scripts/run.py "$@"
