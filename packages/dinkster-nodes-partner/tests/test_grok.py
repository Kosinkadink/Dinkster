from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import MISSING, FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PACK_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACK_ROOT / "src"))

from dinkster_api.v1 import InputSpec  # noqa: E402
from dinkster_nodes_partner.grok import (  # noqa: E402
    GROK_CONTRACTS,
    GROK_NODES,
    GrokImageEditNode,
    GrokImageEditNodeV2,
    GrokImageNode,
    GrokVideoEditNode,
    GrokVideoExtendNode,
    GrokVideoNode,
    GrokVideoReferenceNode,
    InputUrlObject,
    VideoGenerationResponse,
    VideoStatusResponse,
)
from dinkster_nodes_partner.opspec import (  # noqa: E402
    CheckInputs,
    DownloadDecode,
    EncodeMedia,
    OpSpec,
    SubmitPoll,
)
from dinkster_nodes_partner.partner_runtime import (  # noqa: E402
    OperationCancelled,
    PartnerError,
    RuntimeContext,
    TrustPolicyError,
    run_op,
)

FIXTURES = PACK_ROOT / "tests" / "fixtures"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


class Response:
    def __init__(self, payload: object = None, content: bytes = b"", status: int = 200) -> None:
        self.status = status
        family = "video/mp4" if content.startswith(b"\x00\x00\x00") else "image/png"
        self.headers: dict[str, str] = {"Content-Type": family} if content else {}
        self.payload = {} if payload is None else payload
        self.content = content

    async def json(self) -> object:
        return self.payload

    async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        assert chunk_size == 1024 * 1024
        yield self.content

    async def close(self) -> None:
        pass


class Transport:
    """Complete injectable transport: JSON, raw PUT, DNS, and content bytes."""

    def __init__(self, *responses: Response) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, object]] = []
        self.resolutions: dict[str, tuple[str, ...]] = {}

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        return self.resolutions.get(host, ("203.0.113.8",))

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object] | None,
        timeout: float,
    ) -> Response:
        self.calls.append((method, url, json_body))
        return self.responses.pop(0)

    async def request_bytes(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> Response:
        self.calls.append((method, url, body))
        return self.responses.pop(0)

    async def internet_accessible(self) -> bool:
        return True

    async def close(self) -> None:
        pass


async def _zero_sleep(_: float) -> None:
    pass


def _run(spec: OpSpec, inputs: Mapping[str, object], transport: Transport, **kwargs: object):
    context = RuntimeContext(
        transport=transport,
        api_key="key",
        api_base="https://api.test",
        sleep=_zero_sleep,
        **kwargs,  # type: ignore[arg-type]
    )
    return asyncio.run(run_op(spec, inputs, context))


def _png(rgb: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), rgb).save(stream, "PNG")
    return stream.getvalue()


MP4 = b"\x00\x00\x00\x18ftypisom" + b"test-video"


def _video_responses(request_id: str = "r1", status: object = "complete") -> tuple[Response, ...]:
    return (
        Response({"request_id": request_id}),
        Response({"status": status, "video": {"url": "https://cdn.test/out.mp4"}}),
        Response(content=MP4),
    )


@pytest.mark.parametrize("node", GROK_NODES, ids=lambda node: node.__name__)
def test_grok_opspec_round_trip_and_required_pipeline(node: type) -> None:
    assert OpSpec.from_json(node.SPEC.to_json()) == node.SPEC
    assert isinstance(node.SPEC.adapters[0], CheckInputs)
    if node.define_schema().outputs[0].type.runtime_type_id() == "comfy.VIDEO":
        assert any(isinstance(adapter, SubmitPoll) for adapter in node.SPEC.adapters)
    assert isinstance(node.SPEC.adapters[-1], DownloadDecode)


def test_grok_image_encoders_pin_upstream_four_megapixel_limit() -> None:
    encoders = [
        adapter
        for node in GROK_NODES
        for adapter in node.SPEC.adapters
        if isinstance(adapter, EncodeMedia) and adapter.media_family == "image"
    ]
    assert len(encoders) == 4
    assert all(adapter.max_pixels == 2048 * 2048 for adapter in encoders)


def test_grok_contracts_are_frozen_and_match_pinned_fixture() -> None:
    fixture = _fixture("grok_contracts.json")
    assert fixture["provenance"]["commit"] == "e651b7bef55a5376343dcb1c0edb79f0142c985e"
    for name, model in GROK_CONTRACTS.items():
        assert [field.name for field in fields(model)] == fixture["models"][name]
    response = VideoGenerationResponse(request_id="request")
    with pytest.raises(FrozenInstanceError):
        response.request_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="url"):
        InputUrlObject(url="")
    assert VideoStatusResponse(status="mystery").status == "mystery"


def test_grok_contract_shapes_match_pinned_pydantic_requiredness_and_defaults() -> None:
    fixture = _fixture("grok_contracts.json")
    for name, model in GROK_CONTRACTS.items():
        shape: dict[str, list[object]] = {}
        for field in fields(model):
            required = field.default is MISSING
            value: list[object] = [str(field.type), required]
            if not required:
                value.append(field.default)
            shape[field.name] = value
        assert shape == fixture["model_shapes"][name]


@pytest.mark.parametrize("node", GROK_NODES, ids=lambda node: node.__name__)
def test_grok_schema_fixture_parity(node: type) -> None:
    expected = _fixture("grok_schemas.json")["nodes"][node.__name__]
    schema = node.define_schema()
    assert schema.node_type == expected["node_type"]
    assert schema.display_name == expected["display_name"]
    assert schema.category == expected["category"]
    assert list(schema.aliases) == expected["aliases"]
    assert schema.search_visibility == expected["search_visibility"]
    assert schema.node_type == schema.node_type.lower()
    assert [
        {
            "id": output.id,
            "type": output.type.runtime_type_id(),
        }
        for output in schema.outputs
    ] == expected["outputs"]


@pytest.mark.parametrize("node", GROK_NODES, ids=lambda node: node.__name__)
def test_grok_schema_all_top_level_inputs(node: type) -> None:
    expected = _fixture("grok_schemas.json")["nodes"][node.__name__]
    schema = node.define_schema()
    actual = [
        {
            "kind": type(item).__name__,
            "id": item.id,
            "type": item.type.runtime_type_id(),
            "required": item.required,
            "default": item.default,
        }
        for item in schema.inputs
    ]
    assert actual == expected["inputs"]


@pytest.mark.parametrize("node", GROK_NODES, ids=lambda node: node.__name__)
def test_grok_schema_all_nested_combo_inputs(node: type) -> None:
    expected = _fixture("grok_schemas.json")["nodes"][node.__name__]["combos"]
    schema = node.define_schema()

    def nested(item: object) -> dict[str, object]:
        result = {
            "id": item.id,
            "kind": type(item).__name__,
            "type": item.type.runtime_type_id(),
        }
        if isinstance(item, InputSpec):
            result.update(default=item.default, required=item.required)
        else:
            result.update(min_members=item.min_members, member_names=list(item.member_names))
        return result

    actual = [
        {
            "id": combo.id,
            "options": [
                {
                    "key": option.key,
                    "inputs": [nested(item) for item in option.inputs],
                }
                for option in combo.options
            ],
        }
        for combo in schema.combos
    ]
    assert actual == expected


@pytest.mark.parametrize(
    ("node", "expected"),
    tuple(
        zip(
            GROK_NODES,
            (
                "partner.grok.image",
                "partner.grok.image-edit",
                "partner.grok.image-edit-v2",
                "partner.grok.video",
                "partner.grok.video-reference",
                "partner.grok.video-edit",
                "partner.grok.video-extend",
            ),
            strict=True,
        )
    ),
)
def test_grok_mechanical_node_type(node: type, expected: str) -> None:
    assert node.define_schema().node_type == expected


@pytest.mark.parametrize("node", GROK_NODES, ids=lambda node: node.__name__)
def test_grok_video_poll_contract(node: type) -> None:
    polls = [adapter for adapter in node.SPEC.adapters if isinstance(adapter, SubmitPoll)]
    if node.define_schema().outputs[0].type.runtime_type_id() != "comfy.VIDEO":
        assert polls == []
        return
    poll = polls[0]
    # Relative descriptors are joined to the provider root as the exact
    # /proxy/xai/v1/videos/{value} request path.
    assert poll.path_template == "/proxy/xai/v1/videos/{value}"
    assert "none" in poll.completed
    assert poll.allowed == ()
    assert poll.queued == SubmitPoll("defaults").queued
    assert poll.failed == SubmitPoll("defaults").failed


@pytest.mark.parametrize("node", GROK_NODES, ids=lambda node: node.__name__)
def test_grok_specs_have_only_declared_public_outputs(node: type) -> None:
    output_ids = {output.id for output in node.define_schema().outputs}
    assert output_ids in ({"image"}, {"video"})
    assert all(vars(adapter).get("output") != "unused" for adapter in node.SPEC.adapters)


def test_grok_image_request_mapping_lowercase_multi_response_stacks() -> None:
    red, blue = _png((255, 0, 0)), _png((0, 0, 255))
    transport = Transport(
        Response({"data": [{"url": "https://cdn.test/red"}, {"url": "https://cdn.test/blue"}]}),
        Response(content=red),
        Response(content=blue),
    )
    result = _run(
        GrokImageNode.SPEC,
        {
            "model": "grok-imagine-image-quality",
            "prompt": "cat",
            "aspect_ratio": "16:9",
            "number_of_images": 2,
            "seed": 9,
            "resolution": "2K",
        },
        transport,
    )
    assert transport.calls[0] == (
        "POST",
        "https://api.test/proxy/xai/v1/images/generations",
        {
            "response_format": "url",
            "model": "grok-imagine-image-quality",
            "prompt": "cat",
            "aspect_ratio": "16:9",
            "n": 2,
            "seed": 9,
            "resolution": "2k",
        },
    )
    assert np.argmax(result["image"][0, 0, 0]) == 0
    assert np.argmax(result["image"][1, 0, 0]) == 2


def test_grok_legacy_edit_flattens_batch_data_urls_and_omits_auto_aspect() -> None:
    images = np.stack((np.full((2, 2, 3), (1, 0, 0)), np.full((2, 2, 3), (0, 1, 0))))
    transport = Transport(
        Response({"data": [{"url": "https://cdn.test/result"}]}), Response(content=_png((1, 2, 3)))
    )
    _run(
        GrokImageEditNode.SPEC,
        {
            "model": "grok-imagine-image-quality",
            "image": images,
            "prompt": "edit",
            "resolution": "1K",
            "number_of_images": 1,
            "seed": 3,
            "aspect_ratio": "auto",
        },
        transport,
    )
    body = transport.calls[0][2]
    assert isinstance(body, dict) and "aspect_ratio" not in body
    assert body.keys() == {
        "response_format",
        "model",
        "images",
        "prompt",
        "resolution",
        "n",
        "seed",
    }
    # Upstream client.py:247 sends each flattened image as a PNG data URL.
    decoded = [base64.b64decode(item["url"].split(",", 1)[1]) for item in body["images"]]
    assert [Image.open(io.BytesIO(item)).getpixel((0, 0)) for item in decoded] == [
        (1, 0, 0),
        (0, 1, 0),
    ]


def test_grok_v2_edit_flattens_nested_batches_and_pro_branch_omits_missing_aspect() -> None:
    one = np.full((1, 2, 2, 3), (1, 0, 0))
    two = np.full((1, 2, 2, 3), (0, 0, 1))
    transport = Transport(
        Response({"data": [{"url": "https://cdn.test/x"}]}), Response(content=_png((0, 0, 0)))
    )
    _run(
        GrokImageEditNodeV2.SPEC,
        {
            "model": "grok-imagine-image-quality",
            "model.images.image_1": one,
            "model.images.image_2": two,
            "model.resolution": "2K",
            "model.number_of_images": 1,
            "model.aspect_ratio": "auto",
            "prompt": "x",
            "seed": 2,
        },
        transport,
    )
    assert [next(iter(item)) for item in transport.calls[0][2]["images"]] == ["url", "url"]
    pro = Transport(
        Response({"data": [{"url": "https://cdn.test/y"}]}), Response(content=_png((0, 0, 0)))
    )
    _run(
        GrokImageEditNodeV2.SPEC,
        {
            "model": "grok-imagine-image-pro",
            "model.images.image_1": one,
            "model.resolution": "1K",
            "model.number_of_images": 1,
            "prompt": "pro",
            "seed": 1,
        },
        pro,
    )
    assert "aspect_ratio" not in pro.calls[0][2]


@pytest.mark.parametrize("with_image", (False, True))
def test_grok_video_request_mapping_optional_image_and_none_status_terminal(
    with_image: bool,
) -> None:
    inputs: dict[str, object] = {
        "model": "grok-imagine-video",
        "prompt": "move",
        "resolution": "480p",
        "aspect_ratio": "auto",
        "duration": 6,
        "seed": 4,
        "image": None,
    }
    if with_image:
        inputs["image"] = np.zeros((2, 2, 3))
    transport = Transport(*_video_responses(status=None))
    result = _run(GrokVideoNode.SPEC, inputs, transport)
    expected = {
        "model": "grok-imagine-video",
        "prompt": "move",
        "resolution": "480p",
        "duration": 6,
        "seed": 4,
    }
    if with_image:
        expected["image"] = {"url": transport.calls[0][2]["image"]["url"]}
    assert transport.calls[0][2] == expected
    assert transport.calls[1][1] == "https://api.test/proxy/xai/v1/videos/r1"
    assert result == {"video": {"container": "mp4", "bytes": MP4}}


def test_grok_reference_fanout_order_and_empty_list_pin() -> None:
    def allocation(n: int) -> Response:
        return Response(
            {
                "upload_url": f"https://upload.test/{n}",
                "download_url": f"https://cdn.test/{n}.png",
            }
        )

    transport = Transport(allocation(1), Response(), allocation(2), Response(), *_video_responses())
    _run(
        GrokVideoReferenceNode.SPEC,
        {
            "model": "grok-imagine-video",
            "model.reference_images.reference_1": np.zeros((2, 2, 3)),
            "model.reference_images.reference_2": np.ones((2, 2, 3)),
            "model.resolution": "480p",
            "model.duration": 6,
            "model.aspect_ratio": "16:9",
            "prompt": "refs",
            "seed": 1,
        },
        transport,
    )
    assert [call[:2] for call in transport.calls[:4]] == [
        ("POST", "https://api.test/customers/storage"),
        ("PUT", "https://upload.test/1"),
        ("POST", "https://api.test/customers/storage"),
        ("PUT", "https://upload.test/2"),
    ]
    assert transport.calls[4][2]["reference_images"] == [
        {"url": "https://cdn.test/1.png"},
        {"url": "https://cdn.test/2.png"},
    ]
    # Upstream nodes_grok.py:806-818 maps an empty dynamic family to [].
    empty = Transport(*_video_responses())
    _run(
        GrokVideoReferenceNode.SPEC,
        {
            "model": "grok-imagine-video",
            "model.resolution": "480p",
            "model.duration": 6,
            "model.aspect_ratio": "16:9",
            "prompt": "none",
            "seed": 0,
        },
        empty,
    )
    assert empty.calls[0][2]["reference_images"] == []


def _valid_mp4() -> bytes:
    import av

    stream = io.BytesIO()
    with av.open(stream, "w", format="mp4") as container:
        output = container.add_stream("libx264", rate=4)
        output.width = output.height = 16
        output.pix_fmt = "yuv420p"
        for _ in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in output.encode(frame):
                container.mux(packet)
        for packet in output.encode():
            container.mux(packet)
    return stream.getvalue()


@pytest.mark.parametrize(
    "node,path,inputs,expected",
    (
        (
            GrokVideoEditNode,
            "/proxy/xai/v1/videos/edits",
            {"model": "grok-imagine-video", "prompt": "edit", "seed": 7},
            {"model": "grok-imagine-video", "prompt": "edit", "seed": 7},
        ),
        (
            GrokVideoExtendNode,
            "/proxy/xai/v1/videos/extensions",
            {"model": "grok-imagine-video", "model.duration": 8, "prompt": "extend", "seed": 0},
            {"prompt": "extend", "duration": 8},
        ),
    ),
)
def test_grok_video_upload_execution_request_mapping(
    node: type, path: str, inputs: dict[str, object], expected: dict[str, object]
) -> None:
    raw = _valid_mp4()
    inputs = {**inputs, "video": {"container": "mp4", "bytes": raw}}
    transport = Transport(
        Response(
            {"upload_url": "https://upload.test/v", "download_url": "https://cdn.test/in.mp4"}
        ),
        Response(),
        *_video_responses(),
    )
    _run(node.SPEC, inputs, transport)
    assert transport.calls[1] == ("PUT", "https://upload.test/v", raw)
    assert transport.calls[2][1] == "https://api.test" + path
    assert transport.calls[2][2] == {**expected, "video": {"url": "https://cdn.test/in.mp4"}}


def test_grok_poll_queued_running_complete_failed_timeout_and_cancellation() -> None:
    base = {
        "model": "grok-imagine-video",
        "prompt": "x",
        "resolution": "480p",
        "aspect_ratio": "auto",
        "duration": 6,
        "seed": 0,
        "image": None,
    }
    transport = Transport(
        Response({"request_id": "q"}),
        Response({"status": "queued"}),
        Response({"status": "running"}),
        Response({"status": "complete", "video": {"url": "https://cdn.test/o"}}),
        Response(content=MP4),
    )
    assert _run(GrokVideoNode.SPEC, base, transport)["video"]["bytes"] == MP4
    failed = Transport(Response({"request_id": "f"}), Response({"status": "failed"}))
    with pytest.raises(PartnerError, match="failed"):
        _run(GrokVideoNode.SPEC, base, failed)
    poll = next(a for a in GrokVideoNode.SPEC.adapters if isinstance(a, SubmitPoll))
    from dataclasses import replace

    timeout_poll = replace(poll, max_attempts=1)
    timeout_spec = OpSpec(
        tuple(timeout_poll if a is poll else a for a in GrokVideoNode.SPEC.adapters)
    )
    with pytest.raises(PartnerError, match="timed out"):
        _run(
            timeout_spec,
            base,
            Transport(Response({"request_id": "t"}), Response({"status": "running"})),
        )
    with pytest.raises(OperationCancelled):
        _run(GrokVideoNode.SPEC, base, Transport(), cancelled=lambda: True)


def test_grok_private_ip_video_result_denies_before_download() -> None:
    transport = Transport(
        Response({"request_id": "x"}),
        Response({"status": "complete", "video": {"url": "https://private.test/x.mp4"}}),
    )
    transport.resolutions["private.test"] = ("127.0.0.1",)
    with pytest.raises(TrustPolicyError):
        _run(
            GrokVideoNode.SPEC,
            {
                "model": "grok-imagine-video",
                "prompt": "x",
                "resolution": "480p",
                "aspect_ratio": "auto",
                "duration": 6,
                "seed": 0,
                "image": None,
            },
            transport,
        )
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "node,inputs,error",
    (
        (
            GrokImageNode,
            {
                "prompt": " ",
                "model": "m",
                "aspect_ratio": "1:1",
                "number_of_images": 1,
                "seed": 0,
                "resolution": "1K",
            },
            "cannot be shorter",
        ),
        (GrokImageEditNode, {"prompt": "x", "model": "m", "image": None}, "At least one"),
        (
            GrokImageEditNode,
            {"prompt": "x", "model": "grok-imagine-image-pro", "image": np.zeros((2, 2, 2, 3))},
            "only 1",
        ),
        (
            GrokImageEditNode,
            {"prompt": "x", "model": "m", "image": np.zeros((4, 2, 2, 3))},
            "maximum of 3",
        ),
        (
            GrokImageEditNode,
            {"prompt": "x", "model": "m", "image": np.zeros((2, 2, 3)), "aspect_ratio": "16:9"},
            "Custom aspect",
        ),
        (
            GrokVideoNode,
            {"prompt": "x", "model": "grok-imagine-video-1.5", "image": None},
            "requires an input",
        ),
        (
            GrokVideoNode,
            {"prompt": "x", "model": "grok-imagine-video", "resolution": "1080p", "image": None},
            "1080p",
        ),
        (
            GrokVideoNode,
            {
                "prompt": "x",
                "model": "grok-imagine-video",
                "resolution": "480p",
                "image": np.zeros((2, 2, 2, 3)),
            },
            "Only one",
        ),
    ),
)
def test_every_checkinputs_grok_rule_exact_error_and_zero_transport_requests(
    node: type, inputs: dict[str, object], error: str
) -> None:
    transport = Transport()
    with pytest.raises(PartnerError, match=error):
        _run(node.SPEC, inputs, transport)
    assert transport.calls == []


def test_grok_fixture_exact_request_fields_and_fixed_values_each_node() -> None:
    fixture = _fixture("grok_contracts.json")["nodes"]
    for node in GROK_NODES:
        request = next(a for a in node.SPEC.adapters if type(a).__name__ == "HttpSyncJson")
        expected = fixture[node.__name__]
        assert request.path == expected["path"]
        assert [binding.target for binding in request.body] == expected["request_fields"]
        assert {field.target: field.value for field in request.fixed} == expected["fixed"]
