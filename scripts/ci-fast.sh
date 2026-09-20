#!/usr/bin/env bash
set -euo pipefail

uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
uv run --locked python -m pytest -q \
  tests/test_schema.py \
  tests/test_values.py \
  tests/test_graph.py \
  tests/test_graph_wire.py
