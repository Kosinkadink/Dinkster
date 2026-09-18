"""Pure source-authority validation for the official MiniMax H3 codecs."""

from __future__ import annotations

from typing import Any

from dinkster_inference.minimax_h3 import MINIMAX_H3_CONFIG

from tools.inference_parity.qwen_image_receipts import ArtifactReceiptError, load_manifest

_PROVIDER_REPOSITORY = "Comfy-Org/MiniMax-H3"
_PROVIDER_REVISION = "014cd40f7e177756c6b2473c0d93b1c89a790dd2"
_LICENSE_REPOSITORY = "MiniMaxAI/MiniMax-H3"
_LICENSE_REVISION = "9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6"
_TERRITORY_STATEMENT = (
    "Applicable Territory is worldwide excluding the Excluded Territories; "
    "Excluded Territories are the European Union, United Kingdom, Republic of Korea, "
    "and United States of America."
)
_MANIFEST_FIELDS = frozenset({"artifacts", "license", "provider", "schema", "scope"})
_PROVIDER_FIELDS = frozenset({"api_url", "model_card_url", "repository", "revision"})
_LICENSE_FIELDS = frozenset(
    {
        "acceptance_claimed",
        "bytes",
        "date",
        "name",
        "repository",
        "revision",
        "sha256",
        "source_url",
        "territory_statement",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "bytes",
        "component_id",
        "digest_authority",
        "provider_path",
        "role",
        "sha256",
        "source_url",
    }
)


def _expected_artifacts() -> dict[str, dict[str, object]]:
    return {
        "video-vae": {
            "bytes": 5207808496,
            "component_id": MINIMAX_H3_CONFIG.video_codec_id,
            "digest_authority": "provider_api.lfs.sha256",
            "provider_path": "vae/minimax_h3_video_vae_fp16.safetensors",
            "role": "video-vae",
            "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
            "source_url": (
                f"https://huggingface.co/{_PROVIDER_REPOSITORY}/resolve/"
                f"{_PROVIDER_REVISION}/vae/minimax_h3_video_vae_fp16.safetensors"
            ),
        },
        "audio-vae": {
            "bytes": 605254808,
            "component_id": MINIMAX_H3_CONFIG.audio_codec_id,
            "digest_authority": "provider_api.lfs.sha256",
            "provider_path": "vae/minimax_h3_audio_vae_fp32.safetensors",
            "role": "audio-vae",
            "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
            "source_url": (
                f"https://huggingface.co/{_PROVIDER_REPOSITORY}/resolve/"
                f"{_PROVIDER_REVISION}/vae/minimax_h3_audio_vae_fp32.safetensors"
            ),
        },
    }


def _exact_json(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return actual.keys() == expected.keys() and all(
            _exact_json(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return len(actual) == len(expected) and all(
            _exact_json(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_digest(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactReceiptError(f"{name} must be a 64-character lowercase SHA256")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed unless every source-authority fact is exact."""
    if (
        type(manifest) is not dict
        or set(manifest) != _MANIFEST_FIELDS
        or type(manifest.get("schema")) is not int
        or manifest["schema"] != 1
    ):
        raise ArtifactReceiptError("unsupported MiniMax H3 codec authority schema")
    expected_scope = [MINIMAX_H3_CONFIG.video_codec_id, MINIMAX_H3_CONFIG.audio_codec_id]
    if not _exact_json(manifest.get("scope"), expected_scope):
        raise ArtifactReceiptError("codec authority scope is not exact")

    provider = manifest.get("provider")
    expected_provider = {
        "api_url": (
            f"https://huggingface.co/api/models/{_PROVIDER_REPOSITORY}/"
            f"revision/{_PROVIDER_REVISION}?blobs=true"
        ),
        "model_card_url": (
            f"https://huggingface.co/{_PROVIDER_REPOSITORY}/raw/{_PROVIDER_REVISION}/README.md"
        ),
        "repository": _PROVIDER_REPOSITORY,
        "revision": _PROVIDER_REVISION,
    }
    if (
        not isinstance(provider, dict)
        or set(provider) != _PROVIDER_FIELDS
        or not _exact_json(provider, expected_provider)
    ):
        raise ArtifactReceiptError("provider is not the exact official immutable source")

    license_receipt = manifest.get("license")
    expected_license = {
        "acceptance_claimed": False,
        "bytes": 17604,
        "date": "2026-08-02",
        "name": "MiniMax H3 COMMUNITY LICENSE AGREEMENT",
        "repository": _LICENSE_REPOSITORY,
        "revision": _LICENSE_REVISION,
        "sha256": "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44",
        "source_url": (
            f"https://huggingface.co/{_LICENSE_REPOSITORY}/raw/{_LICENSE_REVISION}/LICENSE"
        ),
        "territory_statement": _TERRITORY_STATEMENT,
    }
    if not isinstance(license_receipt, dict) or set(license_receipt) != _LICENSE_FIELDS:
        raise ArtifactReceiptError("license authority must be an exact closed object")
    _require_digest(license_receipt.get("sha256"), "license sha256")
    if not _exact_json(license_receipt, expected_license):
        raise ArtifactReceiptError("license authority or restriction facts drifted")

    artifacts = manifest.get("artifacts")
    if type(artifacts) is not list or len(artifacts) != 2:
        raise ArtifactReceiptError("exactly two MiniMax H3 codec authorities are required")
    expected_artifacts = _expected_artifacts()
    roles: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            raise ArtifactReceiptError("codec authority must be an exact closed object")
        role = artifact.get("role")
        if not isinstance(role, str) or role not in expected_artifacts or role in roles:
            raise ArtifactReceiptError("codec authority roles must be exact and unique")
        roles.add(role)
        if type(artifact.get("bytes")) is not int or artifact["bytes"] <= 0:
            raise ArtifactReceiptError(f"codec authority {role} bytes must be a positive integer")
        _require_digest(artifact.get("sha256"), f"codec authority {role} sha256")
        if not _exact_json(artifact, expected_artifacts[role]):
            raise ArtifactReceiptError(f"codec authority {role} facts drifted")
    if roles != set(expected_artifacts):
        raise ArtifactReceiptError("codec authority roles do not form the exact scope")


__all__ = ["ArtifactReceiptError", "load_manifest", "validate_manifest"]
