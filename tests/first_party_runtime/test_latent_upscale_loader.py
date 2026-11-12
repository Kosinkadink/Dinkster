"""Schema and loader dispatch contracts for latent-upscale assets."""

from __future__ import annotations

import pytest
from dinkster_models.latent_upscale_loader import LatentUpscaleModelLoader


def test_loader_rejects_unbound_non_asset_values() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LatentUpscaleModelLoader.execute(model_name=object())
