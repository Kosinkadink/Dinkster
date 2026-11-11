"""Load explicitly supplied model artifacts through first-party selection nodes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dinkster_assets import AssetRef


def load_vae(asset: AssetRef) -> object:
    from dinkster_models.model_selections import VAELoader

    root = asset.local_path().resolve().parent
    with tempfile.TemporaryDirectory(prefix="dinkster-oracle-vae-") as directory:
        snapshot = Path(directory, "mounts.json")
        snapshot.write_text(
            json.dumps(
                {
                    "mounts": [
                        {
                            "id": "vae",
                            "root": str(root),
                            "kind": "model/vae",
                            "mode": "read",
                            "index": "",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        previous = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT")
        os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = str(snapshot)
        try:
            return VAELoader.execute(vae_name=asset)["VAE"]
        finally:
            if previous is None:
                os.environ.pop("DINKSTER_MOUNTS_SNAPSHOT", None)
            else:
                os.environ["DINKSTER_MOUNTS_SNAPSHOT"] = previous
