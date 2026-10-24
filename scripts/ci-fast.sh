#!/usr/bin/env bash
set -euo pipefail

uv run --locked python tools/gen_extension_contribution_kinds.py --check
uv run --locked python scripts/check_extension_factories.py \
  --baseline-ref "origin/${GITHUB_BASE_REF:-main}"
uv run --locked python scripts/check_builtin_type_spellings.py
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
uv run --locked python -m pytest -q \
  tests/test_disabled_features.py \
  tests/test_settings.py \
  tests/test_extension_factory_guard.py \
  tests/test_builtin_type_spellings.py \
  tests/test_pinned_inference.py \
  tests/test_run_scripts.py \
  tests/test_launch.py \
  tests/test_release_install.py \
  tests/test_schema.py \
  tests/test_schema_current_contracts.py \
  tests/test_names.py \
  tests/test_comfy_controlnet_nodes.py \
  tests/test_validation_inputs.py \
  tests/test_values.py \
  tests/test_graph.py \
  tests/test_graph_wire.py \
  tests/test_vision_capability_schemas.py \
  tests/test_manifest.py::test_extension_contract_fixture_uses_current_public_pack_surfaces \
  tests/test_compose.py::test_compose_dev_pack_adds_scaffold_nodes \
  tests/test_serve.py::test_serve_rejects_removed_dev_argument \
  tests/test_serve.py::test_serve_plain_dev_pack_is_announced \
  tests/test_compat_comfy.py::test_compat_worker_type_ownership_preserves_image_dependency
