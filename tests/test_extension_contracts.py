"""S0-A declarative extension and snapshot identity contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dinkster_protocol import (
    EXTENSION_SCOPES,
)
from dinkster_workers import ManifestError, load_manifest
from dinkster_workers.host import load_extension_contributions


def _manifest(tmp_path: Path, extension: str = "") -> Path:
    path = tmp_path / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "example"\n[pack.entry]\nnodes = "example.nodes:NODES"\n' + extension,
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("scope", EXTENSION_SCOPES)
def test_manifest_extension_entry_requires_matching_privilege(tmp_path: Path, scope: str) -> None:
    with pytest.raises(ManifestError, match=f"matching privileges: {scope}"):
        load_manifest(
            _manifest(
                tmp_path,
                f'[pack.extension]\n{scope} = "extension:{scope}"\n',
            )
        )


def test_pack_worker_never_resolves_inference_or_training_entries(tmp_path: Path) -> None:
    module_name = "must_not_import_worker_scoped_pack"
    sys.modules.pop(module_name, None)
    manifest = load_manifest(
        _manifest(
            tmp_path,
            "[pack.extension]\n"
            f'inference = "{module_name}:inference"\n'
            f'training = "{module_name}:training"\n'
            'privileges = ["inference", "training"]\n',
        )
    )
    assert load_extension_contributions(manifest) == ()
    assert module_name not in sys.modules
