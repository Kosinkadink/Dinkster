from __future__ import annotations

import io
import json
import struct

import pytest
from dinkster_assets import (
    AssetError,
    LatentAssetDescriptor,
    parse_latent_asset,
    valid_vae_hint,
)


def safetensors(header: dict[str, object], body: bytes) -> io.BytesIO:
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return io.BytesIO(struct.pack("<Q", len(encoded)) + encoded + body)


def test_plain_comfyui_profile_records_legacy_scaling() -> None:
    source = safetensors(
        {
            "__metadata__": {"format": "not-a-profile-selector", "prompt": "{}"},
            "latent_tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        b"data",
    )
    assert parse_latent_asset(source).legacy_scale is True


@pytest.mark.parametrize("marker_offset", [0, 4])
def test_stock_comfyui_marker_is_exact_and_metadata_format_is_irrelevant(
    marker_offset: int,
) -> None:
    source = safetensors(
        {
            "__metadata__": {"format": "arbitrary", "extra_pnginfo": "{}"},
            "latent_tensor": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
            "latent_format_version_0": {
                "dtype": "F32",
                "shape": [0],
                "data_offsets": [marker_offset, marker_offset],
            },
        },
        b"data",
    )
    descriptor = parse_latent_asset(source)
    assert descriptor.profile == "comfyui-single"
    assert descriptor.legacy_scale is False
    assert tuple(item.name for item in descriptor.tensors) == ("latent_tensor",)


def test_parsed_vae_hint_uses_the_shared_field_authority() -> None:
    digest = "blake3:" + "a" * 64

    def descriptor(fields: dict[str, object]) -> LatentAssetDescriptor:
        hint = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return parse_latent_asset(
            safetensors(
                {
                    "__metadata__": {"dinkster_vae_hint": hint},
                    "latent_tensor": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    },
                },
                b"data",
            )
        )

    valid = {
        "version": 1,
        "sourceDigest": digest,
        "sourceName": "vae.safetensors",
        "sourceLogicalId": "builtin-vae",
        "latentSpace": "dinkster.sdxl",
    }
    assert valid_vae_hint(descriptor(valid)) == json.dumps(
        valid, sort_keys=True, separators=(",", ":")
    )
    assert valid_vae_hint(descriptor({**valid, "version": True})) == ""
    assert valid_vae_hint(descriptor({**valid, "sourceName": "data:text/plain,vae"})) == ""
    assert valid_vae_hint(descriptor({**valid, "latentSpace": "arbitrary-family"})) == ""

    huge_hint = '{"sourceDigest":"' + digest + '","version":' + "9" * 5000 + "}"
    huge_descriptor = parse_latent_asset(
        safetensors(
            {
                "__metadata__": {"dinkster_vae_hint": huge_hint},
                "latent_tensor": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                },
            },
            b"data",
        )
    )
    assert valid_vae_hint(huge_descriptor) == ""

    nested_hint = '{"version":' + "[" * 3900 + "1" + "]" * 3900 + "}"
    nested_descriptor = parse_latent_asset(
        safetensors(
            {
                "__metadata__": {"dinkster_vae_hint": nested_hint},
                "latent_tensor": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                },
            },
            b"data",
        )
    )
    assert valid_vae_hint(nested_descriptor) == ""


def test_integer_digit_limit_is_normalized_for_header_and_native_schema() -> None:
    huge_integer = b"9" * 5000
    raw_header = b'{"value":' + huge_integer + b"}"
    raw_header += b" " * (-len(raw_header) % 8)
    with pytest.raises(AssetError, match="invalid safetensors header JSON"):
        parse_latent_asset(io.BytesIO(struct.pack("<Q", len(raw_header)) + raw_header))

    schema = (
        '{"format":"dinkster.latent","version":'
        + huge_integer.decode()
        + ',"structure":"single","streams":[]}'
    )
    source = safetensors(
        {
            "__metadata__": {"dinkster_latent_schema": schema},
            "dinkster_samples": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        b"data",
    )
    with pytest.raises(AssetError, match="invalid Dinkster latent schema JSON"):
        parse_latent_asset(source)


def test_bounded_nesting_is_normalized_for_header_and_native_schema() -> None:
    nested = b"[" * 10000 + b"0" + b"]" * 10000
    raw_header = nested
    raw_header += b" " * (-len(raw_header) % 8)
    with pytest.raises(AssetError, match="invalid safetensors header JSON"):
        parse_latent_asset(io.BytesIO(struct.pack("<Q", len(raw_header)) + raw_header))

    source = safetensors(
        {
            "__metadata__": {"dinkster_latent_schema": nested.decode()},
            "dinkster_samples": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        b"data",
    )
    with pytest.raises(AssetError, match="invalid Dinkster latent schema JSON"):
        parse_latent_asset(source)


def test_brackets_inside_metadata_text_do_not_count_as_json_nesting() -> None:
    source = safetensors(
        {
            "__metadata__": {"note": "[" * 10000 + '"quoted"' + "]" * 10000},
            "latent_tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        b"data",
    )
    descriptor = parse_latent_asset(source)
    assert descriptor.profile == "comfyui-single"
    assert descriptor.tensors[0].shape == (1,)


def _native_multi(
    streams: list[dict[str, object]],
    tensors: dict[str, object] | None = None,
    body: bytes = b"abcdefgh",
) -> io.BytesIO:
    schema = json.dumps(
        {
            "format": "dinkster.latent",
            "version": 1,
            "structure": "multi",
            "streams": streams,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    table = tensors or {
        "dinkster_stream_0000": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "dinkster_stream_0001": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }
    return safetensors({"__metadata__": {"dinkster_latent_schema": schema}, **table}, body)


def _stream(order: int, role: str, name: str) -> dict[str, object]:
    return {
        "order": order,
        "role": role,
        "tensor": name,
        "dtype": "F32",
        "shape": [1],
    }


def test_header_alignment_and_resource_bounds_refuse_before_body() -> None:
    with pytest.raises(AssetError, match="header"):
        parse_latent_asset(io.BytesIO(struct.pack("<Q", 7) + b"{}     "))
    with pytest.raises(AssetError, match="header"):
        parse_latent_asset(io.BytesIO(struct.pack("<Q", 4 * 1024 * 1024 + 8)))
