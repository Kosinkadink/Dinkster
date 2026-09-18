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
    validate_vae_hint_field,
)


def safetensors(header: dict[str, object], body: bytes) -> io.BytesIO:
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return io.BytesIO(struct.pack("<Q", len(encoded)) + encoded + body)


def test_native_single_descriptor_is_immutable_and_body_is_not_read() -> None:
    schema = json.dumps(
        {
            "format": "dinkster.latent",
            "version": 1,
            "structure": "single",
            "streams": [{"tensor": "dinkster_samples", "dtype": "F16", "shape": [1, 2]}],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    source = safetensors(
        {
            "__metadata__": {"dinkster_latent_schema": schema},
            "dinkster_samples": {"dtype": "F16", "shape": [1, 2], "data_offsets": [0, 4]},
        },
        b"body",
    )
    body_offset = len(source.getvalue()) - 4

    class HeaderOnly(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and size >= 0 and self.tell() + size <= body_offset
            return super().read(size)

    descriptor = parse_latent_asset(HeaderOnly(source.getvalue()))
    assert descriptor.profile == "dinkster-v1"
    assert descriptor.tensors[0].shape == (1, 2)
    with pytest.raises(AttributeError):
        descriptor.profile = "comfyui-single"  # type: ignore[misc]


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


@pytest.mark.parametrize(
    ("marker", "body", "error"),
    [
        ({"dtype": "F16", "shape": [0], "data_offsets": [4, 4]}, b"data", "marker"),
        ({"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}, b"dataxxxx", "marker"),
        ({"dtype": "F32", "shape": [0], "data_offsets": [2, 2]}, b"data", "contiguous"),
        ({"dtype": "F32", "shape": [0], "data_offsets": [8, 8]}, b"data", "contiguous"),
    ],
)
def test_comfyui_marker_dtype_shape_and_body_cursor_are_exact(
    marker: dict[str, object], body: bytes, error: str
) -> None:
    source = safetensors(
        {
            "latent_tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "latent_format_version_0": marker,
        },
        body,
    )
    with pytest.raises(AssetError, match=error):
        parse_latent_asset(source)


@pytest.mark.parametrize(
    "tensor",
    [
        {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]},
        {"dtype": "F32", "shape": [1.0], "data_offsets": [0, 4]},
        {"dtype": "F32", "shape": [1], "data_offsets": [False, 4]},
        {"dtype": "F32", "shape": [1], "data_offsets": [0.0, 4]},
    ],
)
def test_comfyui_shape_and_offsets_require_exact_python_integers(
    tensor: dict[str, object],
) -> None:
    with pytest.raises(AssetError):
        parse_latent_asset(safetensors({"latent_tensor": tensor}, b"data"))


@pytest.mark.parametrize("name", ["model.weight", "extra", "dinkster_samples"])
def test_schema_less_comfyui_profile_rejects_every_extra_tensor(name: str) -> None:
    source = safetensors(
        {
            "__metadata__": {"format": "pt"},
            "latent_tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            name: {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        },
        b"dataxxxx",
    )
    with pytest.raises(AssetError, match="schema-less"):
        parse_latent_asset(source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sourceName", "data:text/plain,vae"),
        ("sourceName", "file:vae.safetensors"),
        ("sourceName", "http://example.test/vae"),
        ("sourceName", "../vae.safetensors"),
        ("sourceName", "vae\\model.safetensors"),
        ("sourceLogicalId", "C:vae"),
        ("sourceLogicalId", "vae\x00id"),
        ("latentSpace", "sdxl"),
        ("latentSpace", "dinkster.SDXL"),
        ("latentSpace", "dinkster.sdxl/path"),
    ],
)
def test_vae_hint_field_validator_rejects_url_path_control_and_unregistered_ids(
    field: str, value: str
) -> None:
    with pytest.raises(AssetError):
        validate_vae_hint_field(field, value)


def test_vae_hint_field_validator_accepts_path_free_text_and_dinkster_ids() -> None:
    assert validate_vae_hint_field("sourceName", "vae.safetensors") == "vae.safetensors"
    assert validate_vae_hint_field("sourceLogicalId", "builtin-vae") == "builtin-vae"
    assert (
        validate_vae_hint_field("latentSpace", "dinkster.sdxl_refiner") == "dinkster.sdxl_refiner"
    )


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


def test_vae_hint_recursion_error_is_informational(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = parse_latent_asset(
        safetensors(
            {
                "__metadata__": {"dinkster_vae_hint": "{}"},
                "latent_tensor": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                },
            },
            b"data",
        )
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RecursionError("nested JSON")

    monkeypatch.setattr("dinkster_assets.latent_format.json.loads", fail)
    assert valid_vae_hint(descriptor) == ""


@pytest.mark.parametrize(
    "raw",
    [
        b'{"latent_tensor":{"dtype":"F32","dtype":"F16","shape":[1],"data_offsets":[0,4]}}',
        b'{"__metadata__":{"dinkster_latent_schema":"{\\"format\\":\\"dinkster.latent\\",\\"version\\":2,\\"structure\\":\\"single\\",\\"streams\\":[]}"}}',
    ],
)
def test_duplicate_keys_and_unknown_schema_refuse(raw: bytes) -> None:
    raw += b" " * (-len(raw) % 8)
    with pytest.raises(AssetError):
        parse_latent_asset(io.BytesIO(struct.pack("<Q", len(raw)) + raw))


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


@pytest.mark.parametrize("version", [True, False])
def test_native_schema_version_requires_an_exact_integer(version: bool) -> None:
    schema = json.dumps(
        {
            "format": "dinkster.latent",
            "version": version,
            "structure": "single",
            "streams": [{"tensor": "dinkster_samples", "dtype": "F32", "shape": [1]}],
        },
        separators=(",", ":"),
    )
    source = safetensors(
        {
            "__metadata__": {"dinkster_latent_schema": schema},
            "dinkster_samples": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        b"data",
    )
    with pytest.raises(AssetError, match="unknown Dinkster latent schema"):
        parse_latent_asset(source)


def test_native_single_redundant_shape_rejects_boolean_dimension() -> None:
    schema = json.dumps(
        {
            "format": "dinkster.latent",
            "version": 1,
            "structure": "single",
            "streams": [{"tensor": "dinkster_samples", "dtype": "F32", "shape": [True]}],
        },
        separators=(",", ":"),
    )
    source = safetensors(
        {
            "__metadata__": {"dinkster_latent_schema": schema},
            "dinkster_samples": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        b"data",
    )
    with pytest.raises(AssetError, match="does not equal"):
        parse_latent_asset(source)


def test_native_redundancy_and_extra_tensor_refuse() -> None:
    schema = json.dumps(
        {
            "format": "dinkster.latent",
            "version": 1,
            "structure": "single",
            "streams": [{"tensor": "dinkster_samples", "dtype": "F32", "shape": [1]}],
        },
        separators=(",", ":"),
    )
    source = safetensors(
        {
            "__metadata__": {"dinkster_latent_schema": schema},
            "dinkster_samples": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        },
        b"xx",
    )
    with pytest.raises(AssetError, match="does not equal"):
        parse_latent_asset(source)


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


def test_native_multi_role_order_and_names_are_exact() -> None:
    valid = [
        _stream(0, "video", "dinkster_stream_0000"),
        _stream(1, "audio", "dinkster_stream_0001"),
    ]
    descriptor = parse_latent_asset(_native_multi(valid))
    assert tuple(item.role for item in descriptor.tensors) == ("video", "audio")
    bad_cases = (
        [_stream(0, "video", "dinkster_stream_0000"), _stream(1, "video", "dinkster_stream_0001")],
        [_stream(0, "video", "dinkster_stream_0000"), _stream(2, "audio", "dinkster_stream_0001")],
        [_stream(0, "video", "dinkster_stream_0000"), _stream(1, "audio", "stream_1")],
        [_stream(0, "", "dinkster_stream_0000"), _stream(1, "audio", "dinkster_stream_0001")],
    )
    for streams in bad_cases:
        with pytest.raises(AssetError):
            parse_latent_asset(_native_multi(streams))


def test_native_multi_order_and_redundant_shape_reject_booleans() -> None:
    tensor: dict[str, object] = {
        "dinkster_stream_0000": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        }
    }
    boolean_order = [_stream(False, "video", "dinkster_stream_0000")]
    with pytest.raises(AssetError, match="order"):
        parse_latent_asset(_native_multi(boolean_order, tensor, b"abcd"))

    boolean_shape = [_stream(0, "video", "dinkster_stream_0000")]
    boolean_shape[0]["shape"] = [True]
    with pytest.raises(AssetError, match="does not equal"):
        parse_latent_asset(_native_multi(boolean_shape, tensor, b"abcd"))


def test_native_extra_tensor_and_offset_gap_refuse() -> None:
    streams = [_stream(0, "video", "dinkster_stream_0000")]
    extra: dict[str, object] = {
        "dinkster_stream_0000": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "extra": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }
    with pytest.raises(AssetError, match="unreferenced"):
        parse_latent_asset(_native_multi(streams, extra))
    gap: dict[str, object] = {
        "dinkster_stream_0000": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}
    }
    with pytest.raises(AssetError, match="contiguous"):
        parse_latent_asset(_native_multi(streams, gap))


def test_header_alignment_and_resource_bounds_refuse_before_body() -> None:
    with pytest.raises(AssetError, match="header"):
        parse_latent_asset(io.BytesIO(struct.pack("<Q", 7) + b"{}     "))
    with pytest.raises(AssetError, match="header"):
        parse_latent_asset(io.BytesIO(struct.pack("<Q", 4 * 1024 * 1024 + 8)))
