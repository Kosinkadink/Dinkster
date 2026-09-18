"""Pure source-authority validation for the official MiniMax H3 runtime graph."""

from __future__ import annotations

from typing import Any

from tools.inference_parity.qwen_image_receipts import ArtifactReceiptError, load_manifest

_PROVIDER_REPOSITORY = "Comfy-Org/MiniMax-H3"
_PROVIDER_REVISION = "3f57e8291d2ef846f9a074b1b76d2767db434abe"
_LICENSE_REPOSITORY = "MiniMaxAI/MiniMax-H3"
_LICENSE_REVISION = "9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6"
_TERRITORY_STATEMENT = (
    "Applicable Territory is worldwide excluding the Excluded Territories; "
    "Excluded Territories are the European Union, United Kingdom, Republic of Korea, "
    "and United States of America."
)
_ARTIFACTS = {
    "fl2va-dit": (
        "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
        34038892334,
        "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5",
    ),
    "ref2va-dit": (
        "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
        34038894550,
        "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9",
    ),
    "qwen3vl-32b-conditioner": (
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        27141342152,
        "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
    ),
    "video-vae": (
        "vae/minimax_h3_video_vae_fp16.safetensors",
        5207808496,
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    ),
    "audio-vae": (
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        605254808,
        "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    ),
}
_ALTERNATE_DIT_ARTIFACTS = {
    "fl2va-dit": (
        "diffusion_models/minimax_h3_fl2va_bf16.safetensors",
        66280487368,
        "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182",
    ),
    "ref2va-dit": (
        "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
        66280487368,
        "e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026",
    ),
}
_FIELDS = frozenset({"alternate_dit_artifacts", "artifacts", "license", "provider", "schema"})
_ARTIFACT_FIELDS = frozenset({"bytes", "provider_path", "role", "sha256", "source_url"})


def _digest(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactReceiptError(f"{name} must be a 64-character lowercase SHA256")


def _exact(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed unless every immutable runtime source fact is exact."""
    if (
        type(manifest) is not dict
        or set(manifest) != _FIELDS
        or type(manifest.get("schema")) is not int
        or manifest["schema"] != 2
    ):
        raise ArtifactReceiptError("unsupported MiniMax H3 runtime authority schema")
    provider = manifest.get("provider")
    if not _exact(
        provider,
        {
            "api_url": (
                f"https://huggingface.co/api/models/{_PROVIDER_REPOSITORY}/"
                f"revision/{_PROVIDER_REVISION}?blobs=true"
            ),
            "gated": False,
            "model_card_url": (
                f"https://huggingface.co/{_PROVIDER_REPOSITORY}/raw/{_PROVIDER_REVISION}/README.md"
            ),
            "repository": _PROVIDER_REPOSITORY,
            "revision": _PROVIDER_REVISION,
        },
    ):
        raise ArtifactReceiptError("provider is not the exact immutable official source")
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
    if not _exact(license_receipt, expected_license):
        raise ArtifactReceiptError("license authority or restriction facts drifted")
    _digest(license_receipt.get("sha256") if isinstance(license_receipt, dict) else None, "license")

    artifacts = manifest.get("artifacts")
    if type(artifacts) is not list or len(artifacts) != len(_ARTIFACTS):
        raise ArtifactReceiptError("the exact five-component runtime graph is required")
    seen: set[str] = set()
    for artifact in artifacts:
        if type(artifact) is not dict or set(artifact) != _ARTIFACT_FIELDS:
            raise ArtifactReceiptError("runtime artifact authority must be an exact object")
        role = artifact.get("role")
        if not isinstance(role, str) or role not in _ARTIFACTS or role in seen:
            raise ArtifactReceiptError("runtime artifact roles must be exact and unique")
        seen.add(role)
        path, size, digest = _ARTIFACTS[role]
        expected = {
            "bytes": size,
            "provider_path": path,
            "role": role,
            "sha256": digest,
            "source_url": (
                f"https://huggingface.co/{_PROVIDER_REPOSITORY}/resolve/{_PROVIDER_REVISION}/{path}"
            ),
        }
        _digest(artifact.get("sha256"), f"artifact {role}")
        if not _exact(artifact, expected):
            raise ArtifactReceiptError(f"runtime artifact {role} facts drifted")
    if seen != set(_ARTIFACTS):
        raise ArtifactReceiptError("runtime artifact roles do not form the exact graph")

    alternates = manifest.get("alternate_dit_artifacts")
    if type(alternates) is not list or len(alternates) != len(_ALTERNATE_DIT_ARTIFACTS):
        raise ArtifactReceiptError("the exact BF16 DiT alternates are required")
    seen.clear()
    for artifact in alternates:
        if type(artifact) is not dict or set(artifact) != _ARTIFACT_FIELDS:
            raise ArtifactReceiptError("alternate DiT authority must be an exact object")
        role = artifact.get("role")
        if not isinstance(role, str) or role not in _ALTERNATE_DIT_ARTIFACTS or role in seen:
            raise ArtifactReceiptError("alternate DiT roles must be exact and unique")
        seen.add(role)
        path, size, digest = _ALTERNATE_DIT_ARTIFACTS[role]
        expected = {
            "bytes": size,
            "provider_path": path,
            "role": role,
            "sha256": digest,
            "source_url": (
                f"https://huggingface.co/{_PROVIDER_REPOSITORY}/resolve/{_PROVIDER_REVISION}/{path}"
            ),
        }
        _digest(artifact.get("sha256"), f"alternate artifact {role}")
        if not _exact(artifact, expected):
            raise ArtifactReceiptError(f"alternate DiT artifact {role} facts drifted")
    if seen != set(_ALTERNATE_DIT_ARTIFACTS):
        raise ArtifactReceiptError("alternate DiT roles do not form the exact pair")


__all__ = ["ArtifactReceiptError", "load_manifest", "validate_manifest"]
