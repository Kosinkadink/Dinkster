#!/usr/bin/env bash
set -euo pipefail

# Dependency and build-backend provisioning belongs to setup, not the gate.
export UV_OFFLINE=1

uv run --locked python tools/gen_extension_contribution_kinds.py --check
uv run --locked python scripts/check_extension_factories.py \
  --baseline-ref "origin/${GITHUB_BASE_REF:-main}"
uv run --locked python scripts/check_builtin_type_spellings.py
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pyright
uv run --locked pyright --pythonplatform Windows
uv run --locked python -m pytest -q -m 'not network' \
  tests/test_disabled_features.py \
  tests/test_settings.py \
  tests/test_extension_factory_guard.py \
  tests/test_builtin_type_spellings.py \
  tests/test_pinned_inference.py \
  tests/test_pack_environment.py \
  tests/test_run_scripts.py \
  tests/test_launch.py \
  tests/test_environment_tools.py \
  tests/test_release_install.py \
  tests/test_installer.py::test_http_registry_fetcher_uses_verified_download_ticket \
  tests/test_installer.py::test_http_registry_fetcher_rejects_invalid_ticket_size \
  tests/test_installer.py::test_cli_restore_downloads_registry_sources \
  tests/test_installer.py::test_registry_resolves_newest_compatible_pack_across_all_pages \
  tests/test_installer.py::test_registry_selection_rechecks_exact_release_before_install \
  tests/test_installer.py::test_registry_compatible_selection_refuses_malformed_pages \
  tests/test_installer.py::test_registry_source_identity_comes_from_verified_archive \
  tests/test_installer.py::test_official_default_declaration_contains_only_ids_and_minimums \
  tests/test_installer.py::test_cli_prepares_compatible_registry_sources_without_generation_mutation \
  tests/test_installer.py::test_generation_source_staging_cli \
  tests/test_pack_archive.py::test_official_artifacts_preserve_sources_and_existing_pack_identities \
  tests/test_pack_archive.py::test_official_artifacts_reject_static_errors_in_built_zip \
  tests/test_pack_archive.py::test_official_publication_uses_selected_manifest_versions \
  tests/test_frontend_artifact.py \
  tests/test_schema.py \
  tests/test_schema_current_contracts.py \
  tests/test_names.py \
  tests/test_comfy_controlnet_nodes.py \
  tests/test_validation_inputs.py \
  tests/test_values.py \
  tests/test_graph.py \
  tests/test_graph_wire.py \
  tests/test_model3d_io.py::test_viewport_capture_survives_workers_and_persistent_cache \
  tests/test_vision_capability_schemas.py \
  tests/test_manifest.py::test_extension_contract_fixture_uses_current_public_pack_surfaces \
  tests/test_manifest.py::test_pack_prefix_reservation_depends_on_publisher \
  tests/test_manifest.py::test_official_pack_declares_its_version_and_engine_range \
  tests/test_manifest.py::test_pack_declares_independent_version_and_engine_compatibility \
  tests/test_manifest.py::test_pack_rejects_invalid_version_or_engine_declaration \
  tests/test_audio_lazy.py::test_native_comfy_audio_registration_preserves_first_authority \
  tests/test_audio_lazy.py::test_native_comfy_audio_registration_is_torch_free \
  tests/test_compose.py::test_compose_dev_pack_adds_scaffold_nodes \
  tests/test_compose.py::test_exact_node_claims_compose_reload_execute_and_reject_second_owner \
  tests/test_compose.py::test_asset_literals_use_host_and_worker_resolvers \
  tests/test_serve.py::test_serve_rejects_removed_dev_argument \
  tests/test_serve.py::test_serve_plain_dev_pack_is_announced \
  tests/test_compat_comfy.py::test_compat_worker_type_ownership_preserves_image_dependency
