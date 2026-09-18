from __future__ import annotations

import asyncio
import io
import json
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import MISSING, FrozenInstanceError, fields, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dinkster_api.v1 import (
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    NumberWidget,
    StringWidget,
)
from PIL import Image

PACK_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACK_ROOT / "src"))

from dinkster_nodes_partner.opspec import (  # noqa: E402
    BatchMapJoin,
    Check,
    CheckInputs,
    Cond,
    DownloadDecode,
    EncodeMedia,
    HttpSyncJson,
    MediaConstraints,
    OpSpec,
    ProxyUpload,
    SubmitPoll,
)
from dinkster_nodes_partner.partner_runtime import (  # noqa: E402
    OperationCancelled,
    PartnerError,
    RuntimeContext,
    TrustPolicyError,
    run_op,
)
from dinkster_nodes_partner.wan import (  # noqa: E402
    WAN_CONTRACTS,
    WAN_NODES,
    HappyHorseReferenceVideoApi,
    HappyHorseTextToVideoApi,
    HappyHorseVideoEditApi,
    Image2ImageInputField,
    Image2ImageParametersField,
    Image2VideoParametersField,
    Reference2VideoParametersField,
    TaskCreationResponse,
    Text2VideoParametersField,
    Txt2ImageParametersField,
    Wan2ImageToVideoApi,
    Wan2ReferenceVideoApi,
    Wan2TextToVideoApi,
    Wan2VideoContinuationApi,
    Wan2VideoEditApi,
    Wan27ImageToVideoParametersField,
    Wan27ReferenceVideoParametersField,
    Wan27Text2VideoParametersField,
    Wan27VideoEditParametersField,
    WanImageToImageApi,
    WanImageToVideoApi,
    WanReferenceVideoApi,
    WanTextToImageApi,
    WanTextToVideoApi,
)

FIXTURES = PACK_ROOT / "tests" / "fixtures"
PINNED_COMFYUI_COMMIT = "e651b7bef55a5376343dcb1c0edb79f0142c985e"


class _Response:
    def __init__(self, payload: object = None, content: bytes = b"") -> None:
        self.status = 200
        self.payload = {} if payload is None else payload
        self.content = content
        self.headers = (
            {"Content-Type": "video/mp4" if b"ftyp" in content[:16] else "image/png"}
            if content
            else {}
        )

    async def json(self) -> object:
        return self.payload

    async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        assert chunk_size == 1024 * 1024
        yield self.content

    async def close(self) -> None:
        pass


class _RecorderTransport:
    """Deterministic transport recorder used by the executed wire proofs below."""

    def __init__(self, *responses: _Response) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, object]] = []

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        return ("203.0.113.9",)

    async def internet_accessible(self) -> bool:
        return True

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object] | None,
        timeout: float,
    ) -> _Response:
        self.calls.append((method, url, json_body))
        assert self.responses, f"unexpected request: {method} {url}"
        return self.responses.pop(0)

    async def request_bytes(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> _Response:
        self.calls.append((method, url, body))
        assert self.responses, f"unexpected byte request: {method} {url}"
        return self.responses.pop(0)

    async def close(self) -> None:
        pass


async def _no_sleep(_: float) -> None:
    pass


def _run(spec: OpSpec, inputs: Mapping[str, object], transport: _RecorderTransport) -> object:
    return asyncio.run(
        run_op(
            spec,
            inputs,
            RuntimeContext(
                transport=transport,
                api_key="key",
                api_base="https://api.test",
                sleep=_no_sleep,
            ),
        )
    )


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


def _video_success_responses(*, uploads: int = 0) -> list[_Response]:
    responses: list[_Response] = []
    for index in range(uploads):
        responses.extend(
            (
                _Response(
                    {
                        "upload_url": f"https://upload.test/{index}",
                        "download_url": f"https://cdn.test/input-{index}",
                    }
                ),
                _Response(),
            )
        )
    responses.extend(
        (
            _Response({"output": {"task_id": "video", "task_status": "PENDING"}}),
            _Response(
                {
                    "output": {
                        "task_id": "video",
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.test/out.mp4",
                    }
                }
            ),
            _Response(content=b"\0\0\0\x18ftypisom"),
        )
    )
    return responses


def _audio(seconds: float, rate: int = 10) -> dict[str, object]:
    return {
        "waveform": np.zeros((1, 1, round(seconds * rate)), dtype=np.float32),
        "sample_rate": rate,
    }


def _legacy_video_inputs(*, image: bool, audio: object = None) -> dict[str, object]:
    result: dict[str, object] = {
        "model": "wan2.5-i2v-preview" if image else "wan2.5-t2v-preview",
        "prompt": "move",
        "negative_prompt": "",
        "duration": 5,
        "audio": audio,
        "seed": 1,
        "generate_audio": False,
        "prompt_extend": True,
        "watermark": False,
        "shot_type": "single",
    }
    if image:
        result.update(image=np.zeros((1, 2, 2, 3), np.float32), resolution="720P")
    else:
        result["size"] = "720p: 1:1 (960x960)"
    return result


def _entry_shape(entry: InputSpec | InputFamilySpec) -> dict[str, object]:
    if isinstance(entry, InputFamilySpec):
        assert len(entry.template) == 1
        template = entry.template[0]
        assert isinstance(template, InputSpec)
        return {
            "id": entry.id,
            "kind": "InputFamilySpec",
            "max_members": entry.max_members,
            "member_names": list(entry.member_names) if entry.member_names is not None else None,
            "min_members": entry.min_members,
            "required": entry.required,
            "template": _entry_shape(template),
        }
    widget: dict[str, object] | None = None
    if isinstance(entry.widget, ComboWidget):
        widget = {
            "kind": "ComboWidget",
            "options": list(entry.widget.options),
            "refresh_button": entry.widget.refresh_button,
            "remote_route": entry.widget.remote_route,
        }
    elif isinstance(entry.widget, NumberWidget):
        widget = {
            "control_after_generate": entry.widget.control_after_generate,
            "kind": "NumberWidget",
            "max": entry.widget.max,
            "min": entry.widget.min,
            "step": entry.widget.step,
        }
    elif isinstance(entry.widget, StringWidget):
        widget = {"kind": "StringWidget", "multiline": entry.widget.multiline}
    return {
        "advanced": entry.advanced,
        "default": entry.default,
        "id": entry.id,
        "kind": "InputSpec",
        "required": entry.required,
        "type": entry.type.runtime_type_id(),
        "widget": widget,
    }


def _schema_shape(node: type) -> dict[str, object]:
    schema = node.define_schema()
    return {
        "aliases": list(schema.aliases),
        "category": schema.category,
        "combos": [
            {
                "id": combo.id,
                "options": [
                    {"inputs": [_entry_shape(item) for item in option.inputs], "key": option.key}
                    for option in combo.options
                ],
            }
            for combo in schema.combos
        ],
        "description": schema.description,
        "display_name": schema.display_name,
        "input_families": [_entry_shape(item) for item in schema.input_families],
        "inputs": [_entry_shape(item) for item in schema.inputs],
        "node_type": schema.node_type,
        "outputs": [
            {"id": output.id, "type": output.type.runtime_type_id()} for output in schema.outputs
        ],
        "search_visibility": schema.search_visibility,
    }


@pytest.mark.parametrize("node", WAN_NODES, ids=lambda node: node.__name__)
def test_wan_e1_e2_full_nested_schema_fixture_parity(node: type) -> None:
    fixture = _fixture("wan_schemas.json")
    assert fixture["provenance"]["commit"] == PINNED_COMFYUI_COMMIT  # type: ignore[index]
    assert _schema_shape(node) == fixture["nodes"][node.__name__]  # type: ignore[index]


def test_wan_provider_names_are_canonical_and_upstream_names_are_aliases() -> None:
    assert len(WAN_NODES) == 14
    for node in WAN_NODES:
        schema = node.define_schema()
        assert schema.node_type.startswith("partner.wan.")
        assert "Alibaba" not in schema.node_type
        assert "DashScope" not in schema.node_type
        assert schema.aliases == (node.__name__,)
        assert schema.category.endswith("/Wan")


def test_wan_34_contract_shapes_match_pinned_fixture_and_are_frozen() -> None:
    fixture = _fixture("wan_contracts.json")
    assert fixture["provenance"]["commit"] == PINNED_COMFYUI_COMMIT  # type: ignore[index]
    assert len(WAN_CONTRACTS) == 34
    assert set(WAN_CONTRACTS) == set(fixture["models"])  # type: ignore[arg-type]
    for name, model in WAN_CONTRACTS.items():
        shape: dict[str, list[object]] = {}
        for field in fields(model):
            required = field.default is MISSING
            value: list[object] = [str(field.type), required]
            if not required:
                value.append(field.default)
            shape[field.name] = value
        assert shape == fixture["model_shapes"][name]  # type: ignore[index]
    response = TaskCreationResponse(request_id="request", output=None)
    with pytest.raises(FrozenInstanceError):
        response.request_id = "changed"  # type: ignore[misc]


def test_wan_contract_constraints_match_pinned_pydantic_boundaries() -> None:
    fixture = _fixture("wan_contracts.json")
    assert len(fixture["constraints"]) == 16  # type: ignore[arg-type]

    for count in (1, 2):
        Image2ImageInputField(prompt="p", images=["x"] * count)
    for count in (0, 3):
        with pytest.raises(ValueError, match="images"):
            Image2ImageInputField(prompt="p", images=["x"] * count)

    seed_models = (
        lambda seed: Txt2ImageParametersField(size="x", seed=seed),
        lambda seed: Image2ImageParametersField(seed=seed),
        lambda seed: Text2VideoParametersField(size="x", seed=seed),
        lambda seed: Image2VideoParametersField(resolution="x", seed=seed),
        lambda seed: Reference2VideoParametersField(size="x", seed=seed),
        lambda seed: Wan27ReferenceVideoParametersField(resolution="x", seed=seed),
        lambda seed: Wan27ImageToVideoParametersField(resolution="x", seed=seed),
        lambda seed: Wan27VideoEditParametersField(resolution="x", seed=seed),
        lambda seed: Wan27Text2VideoParametersField(resolution="x", seed=seed),
    )
    for construct in seed_models:
        construct(0)
        construct(2147483647)
        for seed in (-1, 2147483648):
            with pytest.raises(ValueError, match="seed"):
                construct(seed)

    duration_models = (
        (lambda duration: Text2VideoParametersField(size="x", seed=0, duration=duration), 5),
        (
            lambda duration: Image2VideoParametersField(resolution="x", seed=0, duration=duration),
            5,
        ),
        (lambda duration: Reference2VideoParametersField(size="x", seed=0, duration=duration), 5),
        (
            lambda duration: Wan27ReferenceVideoParametersField(
                resolution="x", seed=0, duration=duration
            ),
            2,
        ),
        (
            lambda duration: Wan27ImageToVideoParametersField(
                resolution="x", seed=0, duration=duration
            ),
            2,
        ),
        (
            lambda duration: Wan27Text2VideoParametersField(
                resolution="x", seed=0, duration=duration
            ),
            2,
        ),
    )
    for construct, minimum in duration_models:
        construct(minimum)
        construct(15)
        for duration in (minimum - 1, 16):
            with pytest.raises(ValueError, match="duration"):
                construct(duration)

    # The pin deliberately has no duration bounds on video edit.
    Wan27VideoEditParametersField(resolution="x", seed=0, duration=-1)
    Wan27VideoEditParametersField(resolution="x", seed=0, duration=16)


@pytest.mark.parametrize("node", WAN_NODES, ids=lambda node: node.__name__)
def test_all_wan_specs_round_trip(node: type) -> None:
    assert OpSpec.from_json(node.SPEC.to_json()) == node.SPEC


def test_wan_poll_intervals_and_default_status_sets_match_pinned_client() -> None:
    expected_intervals = (3, 4, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7)
    defaults = SubmitPoll("defaults")
    for node, interval in zip(WAN_NODES, expected_intervals, strict=True):
        poll = next(adapter for adapter in node.SPEC.adapters if isinstance(adapter, SubmitPoll))
        assert poll.interval == interval
        assert (poll.completed, poll.failed, poll.queued) == (
            defaults.completed,
            defaults.failed,
            defaults.queued,
        )
        assert poll.allowed == ()
    for node in (WanTextToImageApi, WanImageToImageApi):
        download = next(
            adapter for adapter in node.SPEC.adapters if isinstance(adapter, DownloadDecode)
        )
        assert download.items_path == ()
        assert download.url_path == ("output", "results", 0, "url")


def test_all_14_wan_seed_widgets_randomize_with_independent_fixture_entries() -> None:
    fixture = _fixture("wan_schemas.json")["nodes"]
    seen_widgets: list[NumberWidget] = []
    for node in WAN_NODES:
        seed = next(entry for entry in node.define_schema().inputs if entry.id == "seed")
        assert isinstance(seed.widget, NumberWidget)
        assert seed.widget.control_after_generate == "randomize"
        assert (
            fixture[node.__name__]["inputs"][  # type: ignore[index]
                next(
                    index
                    for index, entry in enumerate(fixture[node.__name__]["inputs"])  # type: ignore[index]
                    if entry["id"] == "seed"
                )
            ]["widget"]["control_after_generate"]
            == "randomize"
        )
        seen_widgets.append(seed.widget)
    assert len(seen_widgets) == 14
    assert len({id(widget) for widget in seen_widgets}) == 14


def test_wan_k_image_cap_check_shape() -> None:
    check_inputs = WanImageToImageApi.SPEC.adapters[0]
    assert check_inputs == CheckInputs(
        "validate_images",
        (
            Check(
                "Expected 1 or 2 input images, but got {count}.",
                require=(Cond("image", "count_ge", 1), Cond("image", "count_le", 2)),
            ),
        ),
    )


def test_wan_m_exact_image_check_shape() -> None:
    check_inputs = WanImageToVideoApi.SPEC.adapters[0]
    assert isinstance(check_inputs, CheckInputs)
    assert check_inputs.checks[0] == Check(
        "Exactly one input image is required.",
        require=(Cond("image", "count_eq", 1),),
    )


@pytest.mark.parametrize("count", (0, 3))
def test_wan_k_i2i_rejects_out_of_range_counts_before_transport(count: int) -> None:
    transport = _RecorderTransport()
    with pytest.raises(PartnerError, match=rf"Expected 1 or 2 input images, but got {count}"):
        _run(
            WanImageToImageApi.SPEC,
            {
                "model": "wan2.5-i2i-preview",
                "image": np.zeros((count, 2, 2, 3), np.float32),
                "prompt": "paint",
                "negative_prompt": "",
                "seed": 7,
                "watermark": False,
            },
            transport,
        )
    assert transport.calls == []


@pytest.mark.parametrize("count", (1, 2))
def test_wan_k_i2i_executes_exact_bare_string_image_bodies_and_pixel_cap(count: int) -> None:
    png = io.BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(png, "PNG")
    transport = _RecorderTransport(
        _Response({"output": {"task_id": "i2i", "task_status": "PENDING"}}),
        _Response(
            {
                "output": {
                    "task_id": "i2i",
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://cdn.test/out.png"}],
                }
            }
        ),
        _Response(content=png.getvalue()),
    )
    _run(
        WanImageToImageApi.SPEC,
        {
            "model": "wan2.5-i2i-preview",
            "image": np.zeros((count, 2, 2, 3), np.float32),
            "prompt": "paint",
            "negative_prompt": "",
            "seed": 7,
            "watermark": False,
        },
        transport,
    )
    body = transport.calls[0][2]
    assert isinstance(body, dict)
    images = body["input"]["images"]  # type: ignore[index]
    assert isinstance(images, list) and len(images) == count
    assert all(
        isinstance(item, str) and item.startswith("data:image/png;base64,") for item in images
    )
    encoded = next(a for a in WanImageToImageApi.SPEC.adapters if isinstance(a, EncodeMedia))
    assert encoded.max_pixels == 4096 * 4096


def test_wan_m_i2v_exact_count_rejections_make_zero_transport_calls() -> None:
    base = {
        "model": "wan2.5-i2v-preview",
        "prompt": "move",
        "negative_prompt": "",
        "resolution": "720P",
        "duration": 5,
        "audio": None,
        "seed": 1,
        "generate_audio": False,
        "prompt_extend": True,
        "watermark": False,
        "shot_type": "single",
    }
    for count in (0, 3):
        transport = _RecorderTransport()
        with pytest.raises(PartnerError, match="Exactly one input image is required"):
            _run(
                WanImageToVideoApi.SPEC,
                {**base, "image": np.zeros((count, 2, 2, 3), np.float32)},
                transport,
            )
        assert transport.calls == []


def test_wan_h_optional_transport_chain_shapes_and_audio_key_omission_contract() -> None:
    # Pinned ComfyUI util/client.py:246-247 drops None-valued keys before transport.
    t2v = Wan2TextToVideoApi.SPEC.adapters
    assert [type(a) for a in t2v[1:4]] == [MediaConstraints, EncodeMedia, ProxyUpload]
    assert all(a.optional is True for a in t2v[1:4])  # type: ignore[union-attr]
    submit = next(a for a in t2v if isinstance(a, HttpSyncJson))
    audio = next(binding for binding in submit.body if binding.target == "input.audio_url")
    assert audio.omit_none is True

    i2v = Wan2ImageToVideoApi.SPEC.adapters
    uploads = [a for a in i2v if isinstance(a, ProxyUpload)]
    assert [(a.id, a.optional) for a in uploads] == [
        ("uploaded_first_frame", False),
        ("uploaded_last_frame", True),
        ("uploaded_audio", True),
    ]
    audio_encoder = next(a for a in i2v if isinstance(a, EncodeMedia) and a.media_family == "audio")
    assert (audio_encoder.format, audio_encoder.output, audio_encoder.optional) == (
        "MP3",
        "bytes",
        True,
    )
    media = next(a for a in i2v if isinstance(a, BatchMapJoin))
    assert [(s.source, s.fixed["type"]) for s in media.segments] == [
        ("uploaded_first_frame", "first_frame"),
        ("uploaded_last_frame", "last_frame"),
        ("uploaded_audio", "driving_audio"),
    ]

    continuation = Wan2VideoContinuationApi.SPEC.adapters
    continuation_uploads = [a for a in continuation if isinstance(a, ProxyUpload)]
    assert [(a.id, a.optional) for a in continuation_uploads] == [
        ("uploaded_first_clip", False),
        ("uploaded_last_frame", True),
    ]


@pytest.mark.parametrize(
    ("node_name", "spec", "inputs"),
    (
        ("WanTextToVideoApi", WanTextToVideoApi.SPEC, _legacy_video_inputs(image=False)),
        ("WanImageToVideoApi", WanImageToVideoApi.SPEC, _legacy_video_inputs(image=True)),
    ),
)
def test_wan_h3_h4_legacy_optional_audio_absent_executes_full_body(
    node_name: str, spec: OpSpec, inputs: dict[str, object]
) -> None:
    transport = _RecorderTransport(*_video_success_responses())
    _run(spec, inputs, transport)
    body = transport.calls[0][2]
    assert isinstance(body, dict)
    expected = _fixture("wan_requests.json")["omit_none_bodies"][node_name]
    assert _normalize_png_data_urls(body) == expected
    assert len(transport.calls) == 3


class _FakeAudioStream:
    def encode(self, frame: object = None) -> tuple[object, ...]:
        return ()


class _FakeAudioContainer:
    def __init__(self, stream: io.BytesIO) -> None:
        self.stream = stream

    def __enter__(self) -> _FakeAudioContainer:
        return self

    def __exit__(self, *args: object) -> None:
        self.stream.write(b"ID3\x04\x00\x00\x00\x00\x00\x00")

    def add_stream(self, codec: str, *, rate: int) -> _FakeAudioStream:
        assert codec == "libmp3lame"
        assert rate == 10
        return _FakeAudioStream()

    def mux(self, packet: object) -> None:
        raise AssertionError("fake encoder emits no packets")


class _FakeAudioFrame:
    sample_rate = 0

    @classmethod
    def from_ndarray(cls, value: np.ndarray, *, format: str, layout: str) -> _FakeAudioFrame:
        assert value.dtype == np.float32
        assert format == "fltp"
        assert layout == "mono"
        return cls()


@pytest.mark.parametrize(
    ("spec", "inputs"),
    (
        (WanTextToVideoApi.SPEC, _legacy_video_inputs(image=False, audio=_audio(3))),
        (WanImageToVideoApi.SPEC, _legacy_video_inputs(image=True, audio=_audio(3))),
    ),
)
def test_wan_h3_h4_legacy_optional_audio_present_executes_mp3_data_uri_body(
    spec: OpSpec, inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_av = SimpleNamespace(
        open=lambda stream, **kwargs: _FakeAudioContainer(stream), AudioFrame=_FakeAudioFrame
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)
    transport = _RecorderTransport(*_video_success_responses())
    _run(spec, inputs, transport)
    body = transport.calls[0][2]
    assert isinstance(body, dict)
    audio_url = body["input"]["audio_url"]  # type: ignore[index]
    assert isinstance(audio_url, str) and audio_url.startswith("data:audio/mpeg;base64,")
    assert audio_url == "data:audio/mpeg;base64,SUQzBAAAAAAAAA=="
    assert len(transport.calls) == 3


@pytest.mark.parametrize("image", (False, True), ids=("t2v", "i2v"))
def test_wan2_optional_audio_present_uploads_mp3_bytes_and_exact_body(
    image: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_av = SimpleNamespace(
        open=lambda stream, **kwargs: _FakeAudioContainer(stream), AudioFrame=_FakeAudioFrame
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)
    common: dict[str, object] = {
        "model": "wan2.7-i2v" if image else "wan2.7-t2v",
        "model.prompt": "go",
        "model.negative_prompt": "",
        "model.resolution": "720P",
        "model.duration": 5,
        "audio": _audio(3),
        "seed": 1,
        "prompt_extend": True,
        "watermark": False,
    }
    if image:
        common.update(first_frame=np.zeros((1, 2, 2, 3), np.float32), last_frame=None)
        spec = Wan2ImageToVideoApi.SPEC
        uploads = 2
    else:
        common["model.ratio"] = "16:9"
        spec = Wan2TextToVideoApi.SPEC
        uploads = 1
    transport = _RecorderTransport(*_video_success_responses(uploads=uploads))
    _run(spec, common, transport)
    assert transport.calls[(uploads - 1) * 2 + 1][2] == b"ID3\x04\x00\x00\x00\x00\x00\x00"
    body = transport.calls[uploads * 2][2]
    assert isinstance(body, dict)
    assert "negative_prompt" not in body["input"]  # type: ignore[operator]
    if image:
        assert body["input"]["media"] == [  # type: ignore[index]
            {"type": "first_frame", "url": "https://cdn.test/input-0"},
            {"type": "driving_audio", "url": "https://cdn.test/input-1"},
        ]
    else:
        assert body["input"]["audio_url"] == "https://cdn.test/input-0"  # type: ignore[index]


@pytest.mark.parametrize(
    ("spec", "inputs", "minimum", "maximum"),
    (
        (WanTextToVideoApi.SPEC, _legacy_video_inputs(image=False), 3.0, 29.0),
        (WanImageToVideoApi.SPEC, _legacy_video_inputs(image=True), 3.0, 29.0),
        (
            Wan2TextToVideoApi.SPEC,
            {
                "model": "wan2.7-t2v",
                "model.prompt": "go",
                "model.negative_prompt": "",
                "model.resolution": "720P",
                "model.ratio": "16:9",
                "model.duration": 5,
                "seed": 1,
                "prompt_extend": True,
                "watermark": False,
            },
            1.5,
            60.0,
        ),
        (
            Wan2ImageToVideoApi.SPEC,
            {
                "first_frame": b"png",
                "last_frame": None,
                "model": "wan2.7-i2v",
                "model.prompt": "go",
                "model.negative_prompt": "",
                "model.resolution": "720P",
                "model.duration": 5,
                "seed": 1,
                "prompt_extend": True,
                "watermark": False,
            },
            2.0,
            30.0,
        ),
    ),
)
def test_wan_i_all_audio_bounds_epsilon_exact_messages_and_malformed_before_transport(
    spec: OpSpec,
    inputs: dict[str, object],
    minimum: float,
    maximum: float,
) -> None:
    maximum_label = "29.0" if maximum == 29 else f"{maximum:g}"
    for seconds, message in (
        (minimum - 0.2, rf"Audio duration must be at least {minimum}s, got {minimum - 0.1:.2f}s"),
        (
            maximum + 0.2,
            rf"Audio duration must be at most {maximum_label}s, got {maximum + 0.1:.2f}s",
        ),
    ):
        transport = _RecorderTransport()
        with pytest.raises(PartnerError, match=message):
            _run(spec, {**inputs, "audio": _audio(seconds)}, transport)
        assert transport.calls == []
    # One sample is the provider's inclusive epsilon and must pass validation.
    for seconds in (minimum - 0.1, minimum, maximum, maximum + 0.1):
        transport = _RecorderTransport()
        try:
            _run(spec, {**inputs, "audio": _audio(seconds)}, transport)
        except (PartnerError, AssertionError, ValueError) as exc:
            assert "duration" not in str(exc)
    for malformed in ({}, {"waveform": np.zeros(2), "sample_rate": 0}):
        transport = _RecorderTransport()
        with pytest.raises(PartnerError):
            _run(spec, {**inputs, "audio": malformed}, transport)
        assert transport.calls == []


def _reference_inputs(images: Mapping[str, object]) -> dict[str, object]:
    return {
        "model": "happyhorse-1.1-r2v",
        "model.prompt": "character1 waves",
        "model.resolution": "720P",
        "model.ratio": "16:9",
        "model.duration": 5,
        "model.reference_images": images,
        "seed": 4,
        "watermark": False,
    }


def test_wan_j_happyhorse_full_node_mapped_images_order_and_inclusive_boundaries() -> None:
    images = {
        "image2": np.zeros((400, 1000, 3), np.float32),
        "image1": np.zeros((1000, 400, 3), np.float32),
    }
    transport = _RecorderTransport(*_video_success_responses(uploads=2))
    _run(HappyHorseReferenceVideoApi.SPEC, _reference_inputs(images), transport)
    body = transport.calls[4][2]
    assert isinstance(body, dict)
    assert body["input"]["media"] == [  # type: ignore[index]
        {"type": "reference_image", "url": "https://cdn.test/input-0"},
        {"type": "reference_image", "url": "https://cdn.test/input-1"},
    ]


@pytest.mark.parametrize(
    ("key", "shape", "message"),
    (
        ("small", (399, 400, 3), "height must be at least 400px"),
        ("narrow", (1001, 400, 3), "aspect ratio must be between 0.4 and 2.5"),
        ("wide", (400, 1001, 3), "aspect ratio must be between 0.4 and 2.5"),
    ),
)
def test_wan_j_happyhorse_failed_key_validation_precedes_upload(
    key: str, shape: tuple[int, int, int], message: str
) -> None:
    transport = _RecorderTransport()
    with pytest.raises(PartnerError, match=rf"item '{key}'.*{message}"):
        _run(
            HappyHorseReferenceVideoApi.SPEC,
            _reference_inputs({key: np.zeros(shape, np.float32)}),
            transport,
        )
    assert transport.calls == []


def test_wan_j_happyhorse_empty_exact_refusal_and_zero_upload() -> None:
    transport = _RecorderTransport()
    with pytest.raises(PartnerError, match="At least one reference image must be provided"):
        _run(HappyHorseReferenceVideoApi.SPEC, _reference_inputs({}), transport)
    assert transport.calls == []


class _FakeContainer:
    def __init__(self, duration: float | BaseException) -> None:
        self.value = duration

    def __enter__(self) -> _FakeContainer:
        if isinstance(self.value, BaseException):
            raise self.value
        self.streams = [SimpleNamespace(type="video", duration=self.value, time_base=Fraction(1))]
        self.duration = None
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _patch_av(monkeypatch: pytest.MonkeyPatch, durations: list[float | BaseException]) -> None:
    values = iter(durations)
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(open=lambda *_args, **_kwargs: _FakeContainer(next(values)), time_base=1),
    )


def _wan_reference_inputs(videos: Mapping[str, object]) -> dict[str, object]:
    return {
        "model": "wan2.6-r2v",
        "prompt": "character1 waves",
        "negative_prompt": "",
        "size": "720p: 1:1 (960x960)",
        "duration": 5,
        "seed": 2,
        "shot_type": "single",
        "watermark": False,
        "reference_videos": videos,
    }


def test_wan_l_reference_video_duration_epsilon_order_and_full_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_av(monkeypatch, [1.9999, 30.0001])
    videos = {
        "character2": {"container": "mp4", "bytes": b"two"},
        "character1": {"container": "mp4", "bytes": b"one"},
    }
    transport = _RecorderTransport(*_video_success_responses(uploads=2))
    _run(WanReferenceVideoApi.SPEC, _wan_reference_inputs(videos), transport)
    body = transport.calls[4][2]
    assert isinstance(body, dict)
    assert body["input"]["reference_video_urls"] == [  # type: ignore[index]
        "https://cdn.test/input-0",
        "https://cdn.test/input-1",
    ]


@pytest.mark.parametrize(
    ("duration", "message"),
    (
        (1.9998, "video duration must be at least 2 seconds"),
        (30.0002, "video duration must be at most 30 seconds"),
    ),
)
def test_wan_l_reference_video_first_failed_key_and_zero_upload(
    monkeypatch: pytest.MonkeyPatch, duration: float, message: str
) -> None:
    _patch_av(monkeypatch, [duration])
    transport = _RecorderTransport()
    with pytest.raises(PartnerError, match=rf"item 'bad'.*{message}"):
        _run(
            WanReferenceVideoApi.SPEC,
            _wan_reference_inputs({"bad": {"container": "mp4", "bytes": b"bad"}}),
            transport,
        )
    assert transport.calls == []


def test_wan_l_reference_video_loud_probe_failure_and_zero_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_av(monkeypatch, [RuntimeError("probe exploded")])
    transport = _RecorderTransport()
    with pytest.raises(PartnerError, match="item 'loud': unable to probe video duration"):
        _run(
            WanReferenceVideoApi.SPEC,
            _wan_reference_inputs({"loud": {"container": "mp4", "bytes": b"bad"}}),
            transport,
        )
    assert transport.calls == []


_SUBMIT_PATH = "/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis"
_IMAGE_SUBMIT_PATHS = {
    "WanTextToImageApi": "/proxy/wan/api/v1/services/aigc/text2image/image-synthesis",
    "WanImageToImageApi": "/proxy/wan/api/v1/services/aigc/image2image/image-synthesis",
}
_POLL_PATH = "/proxy/wan/api/v1/tasks/video"


def _matrix_inputs(node: type) -> dict[str, object]:
    image = np.zeros((1, 400, 400, 3), np.float32)
    video = {"container": "mp4", "bytes": b"video"}
    common: dict[str, object] = {
        "model": "matrix-model",
        "prompt": "matrix prompt",
        "negative_prompt": "matrix negative",
        "width": 640,
        "height": 480,
        "size": "720p: 1:1 (960x960)",
        "resolution": "720P",
        "duration": 5,
        "seed": 23,
        "prompt_extend": True,
        "watermark": False,
        "generate_audio": False,
        "shot_type": "single",
        "audio": None,
        "image": image,
        "first_frame": image,
        "last_frame": None,
        "first_clip": video,
        "video": video,
        "audio_setting": "auto",
        "model.prompt": "matrix prompt",
        "model.negative_prompt": "matrix negative",
        "model.resolution": "720P",
        "model.ratio": "16:9",
        "model.duration": "5" if node.__name__ == "Wan2VideoEditApi" else 5,
        "model.reference_images": {"image1": image[0]},
        "model.reference_videos": {"character1": video},
        "reference_videos": {"character1": video},
    }
    if node.__name__ == "HappyHorseImageToVideoApi":
        common["first_frame"] = image[0]
    return common


def _normalize_png_data_urls(value: object) -> object:
    if isinstance(value, str) and value.startswith("data:"):
        assert value.startswith("data:image/png;base64,")
        return "<PNG_DATA_URL>"
    if isinstance(value, list):
        return [_normalize_png_data_urls(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_png_data_urls(item) for key, item in value.items()}
    return value


def test_wan_image_download_uses_only_output_results_zero_url() -> None:
    png = io.BytesIO()
    Image.new("RGB", (2, 2)).save(png, "PNG")
    transport = _RecorderTransport(
        _Response({"output": {"task_id": "video", "task_status": "PENDING"}}),
        _Response(
            {
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [
                        {"url": "https://cdn.test/first.png"},
                        {"url": "https://cdn.test/must-not-download.png"},
                    ],
                }
            }
        ),
        _Response(content=png.getvalue()),
    )
    result = _run(WanTextToImageApi.SPEC, _matrix_inputs(WanTextToImageApi), transport)
    assert isinstance(result, dict) and isinstance(result["image"], np.ndarray)
    assert [call[1] for call in transport.calls] == [
        "https://api.test" + _IMAGE_SUBMIT_PATHS["WanTextToImageApi"],
        "https://api.test" + _POLL_PATH,
        "https://cdn.test/first.png",
    ]


@pytest.mark.parametrize(
    "node",
    (
        Wan2TextToVideoApi,
        Wan2VideoEditApi,
        Wan2ReferenceVideoApi,
        HappyHorseTextToVideoApi,
        HappyHorseVideoEditApi,
        HappyHorseReferenceVideoApi,
    ),
    ids=lambda node: node.__name__,
)
def test_wan_six_upstream_prompt_checks_reject_only_empty_and_execute_whitespace(
    node: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_transport = _RecorderTransport()
    with pytest.raises(
        PartnerError,
        match="^Field 'prompt' cannot be shorter than 1 characters; was 0 characters long\\.$",
    ):
        _run(node.SPEC, {**_matrix_inputs(node), "model.prompt": ""}, empty_transport)
    assert empty_transport.calls == []

    _patch_av(monkeypatch, [5.0] * 8)
    inputs = {**_matrix_inputs(node), "model.prompt": "   "}
    uploads = sum(
        1
        for adapter in node.SPEC.adapters
        if isinstance(adapter, ProxyUpload) and not adapter.optional
    )
    transport = _RecorderTransport(*_video_success_responses(uploads=uploads))
    _run(node.SPEC, inputs, transport)
    submit = next(call for call in transport.calls if call[1].endswith(_SUBMIT_PATH))
    assert submit[2]["input"]["prompt"] == "   "  # type: ignore[index]


@pytest.mark.parametrize("node", WAN_NODES, ids=lambda node: node.__name__)
def test_wan_all_14_request_families_execute_final_post_body_mapping(
    node: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_av(monkeypatch, [5.0] * 8)
    uploads = sum(
        1
        for adapter in node.SPEC.adapters
        if isinstance(adapter, ProxyUpload) and not adapter.optional
    )
    if node.__name__ in ("WanTextToImageApi", "WanImageToImageApi"):
        png = io.BytesIO()
        Image.new("RGB", (2, 2)).save(png, "PNG")
        responses = []
        for index in range(uploads):
            responses.extend(
                (
                    _Response(
                        {
                            "upload_url": f"https://upload.test/{index}",
                            "download_url": f"https://cdn.test/input-{index}",
                        }
                    ),
                    _Response(),
                )
            )
        responses.extend(
            (
                _Response({"output": {"task_id": "video", "task_status": "PENDING"}}),
                _Response(
                    {
                        "output": {
                            "task_status": "SUCCEEDED",
                            "results": [{"url": "https://cdn.test/out.png"}],
                        }
                    }
                ),
                _Response(content=png.getvalue()),
            )
        )
    else:
        responses = _video_success_responses(uploads=uploads)
    transport = _RecorderTransport(*responses)
    result = _run(node.SPEC, _matrix_inputs(node), transport)
    expected_submit_path = _IMAGE_SUBMIT_PATHS.get(node.__name__, _SUBMIT_PATH)
    submit_calls = [call for call in transport.calls if call[1].endswith(expected_submit_path)]
    assert len(submit_calls) == 1
    method, path, body = submit_calls[0]
    assert (method, path) == ("POST", "https://api.test" + expected_submit_path)
    assert isinstance(body, dict)
    requests = _fixture("wan_requests.json")["request_bodies"]
    assert _normalize_png_data_urls(body) == requests[node.__name__]
    assert isinstance(result, dict) and set(result) in ({"image"}, {"video"})
    assert ("GET", "https://api.test" + _POLL_PATH) == transport.calls[-2][:2]
    assert transport.calls[-1][:2] == ("GET", "https://cdn.test/out.mp4") or node.__name__ in (
        "WanTextToImageApi",
        "WanImageToImageApi",
    )


def _video_spec_with(
    *, poll: SubmitPoll | None = None, download: DownloadDecode | None = None
) -> OpSpec:
    adapters = tuple(
        poll
        if poll is not None and isinstance(item, SubmitPoll)
        else download
        if download is not None and isinstance(item, DownloadDecode)
        else item
        for item in WanTextToVideoApi.SPEC.adapters
    )
    return replace(WanTextToVideoApi.SPEC, adapters=adapters)


def test_wan_submit_poll_executes_queued_running_completed_paths_and_download() -> None:
    responses = [
        _Response({"output": {"task_id": "video", "task_status": "PENDING"}}),
        _Response({"output": {"task_status": "QUEUED"}}),
        _Response({"output": {"task_status": "RUNNING"}}),
        _Response(
            {"output": {"task_status": "SUCCEEDED", "video_url": "https://cdn.test/out.mp4"}}
        ),
        _Response(content=b"\0\0\0\x18ftypisom"),
    ]
    transport = _RecorderTransport(*responses)
    result = _run(WanTextToVideoApi.SPEC, _legacy_video_inputs(image=False), transport)
    assert [call[:2] for call in transport.calls] == [
        ("POST", "https://api.test" + _SUBMIT_PATH),
        ("GET", "https://api.test" + _POLL_PATH),
        ("GET", "https://api.test" + _POLL_PATH),
        ("GET", "https://api.test" + _POLL_PATH),
        ("GET", "https://cdn.test/out.mp4"),
    ]
    assert isinstance(result, dict) and isinstance(result["video"], dict)
    assert result["video"]["bytes"] == b"\0\0\0\x18ftypisom"


@pytest.mark.parametrize("status", ("FAILED", "CANCELED"))
def test_wan_submit_poll_executes_failure_statuses(status: str) -> None:
    transport = _RecorderTransport(
        _Response({"output": {"task_id": "video", "task_status": "PENDING"}}),
        _Response({"output": {"task_status": status}}),
    )
    with pytest.raises(PartnerError, match=rf"failed with status '{status.lower()}'"):
        _run(WanTextToVideoApi.SPEC, _legacy_video_inputs(image=False), transport)
    assert transport.calls[-1][:2] == ("GET", "https://api.test" + _POLL_PATH)


def test_wan_submit_poll_executes_timeout() -> None:
    original = next(a for a in WanTextToVideoApi.SPEC.adapters if isinstance(a, SubmitPoll))
    poll = replace(original, max_attempts=2, interval=0)
    transport = _RecorderTransport(
        _Response({"output": {"task_id": "video", "task_status": "PENDING"}}),
        _Response({"output": {"task_status": "RUNNING"}}),
        _Response({"output": {"task_status": "RUNNING"}}),
    )
    with pytest.raises(PartnerError, match="timed out after 2 attempts"):
        _run(_video_spec_with(poll=poll), _legacy_video_inputs(image=False), transport)


def test_wan_submit_poll_executes_cancellation() -> None:
    transport = _RecorderTransport(
        _Response({"output": {"task_id": "video", "task_status": "PENDING"}})
    )
    with pytest.raises(OperationCancelled):
        asyncio.run(
            run_op(
                WanTextToVideoApi.SPEC,
                _legacy_video_inputs(image=False),
                RuntimeContext(
                    transport=transport,
                    api_key="key",
                    api_base="https://api.test",
                    cancelled=lambda: bool(transport.calls),
                    sleep=_no_sleep,
                ),
            )
        )
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("url", "message"),
    (("http://cdn.test/out.mp4", "HTTPS"),),
)
def test_wan_full_spec_rejects_untrusted_absolute_download_urls(url: str, message: str) -> None:
    transport = _RecorderTransport(
        _Response({"output": {"task_id": "video", "task_status": "PENDING"}}),
        _Response({"output": {"task_status": "SUCCEEDED", "video_url": url}}),
    )
    with pytest.raises(TrustPolicyError, match=message):
        _run(WanTextToVideoApi.SPEC, _legacy_video_inputs(image=False), transport)
    assert len(transport.calls) == 2


def test_wan_full_spec_enforces_streamed_download_byte_cap() -> None:
    original = next(a for a in WanTextToVideoApi.SPEC.adapters if isinstance(a, DownloadDecode))
    transport = _RecorderTransport(
        _Response({"output": {"task_id": "video", "task_status": "PENDING"}}),
        _Response(
            {"output": {"task_status": "SUCCEEDED", "video_url": "https://cdn.test/out.mp4"}}
        ),
        _Response(content=b"\0\0\0\x18ftypisom"),
    )
    with pytest.raises(TrustPolicyError, match="download exceeds byte cap"):
        _run(
            _video_spec_with(download=replace(original, byte_cap=4)),
            _legacy_video_inputs(image=False),
            transport,
        )
