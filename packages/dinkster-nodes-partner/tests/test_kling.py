from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import MISSING, FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PACK_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACK_ROOT / "src"))

from dinkster_nodes_partner import kling as kling_module  # noqa: E402
from dinkster_nodes_partner.bfl import BFL_NODES  # noqa: E402
from dinkster_nodes_partner.grok import GROK_NODES  # noqa: E402
from dinkster_nodes_partner.helper_registry import HelperRefusal, HelperRegistry  # noqa: E402
from dinkster_nodes_partner.kling import (  # noqa: E402
    KLING_CONTRACTS,
    KLING_HELPERS,
    KLING_NODES,
    MODE_START_END_FRAME,
    MODE_TEXT2VIDEO,
    VOICES_CONFIG,
    KlingCameraControls,
    KlingVirtualTryOnNode,
    KlingVirtualTryOnResponse,
    TextToVideoWithAudio,
)
from dinkster_nodes_partner.opspec import (  # noqa: E402
    Class3Spec,
    DownloadDecode,
    HelperBinding,
    HelperCall,
    HttpSyncJson,
    InputBinding,
    MediaConstraints,
    OpSpec,
    ResponseSelect,
    SubmitPoll,
    ValueConstruct,
)
from dinkster_nodes_partner.partner_runtime import (  # noqa: E402
    OperationCancelled,
    PartnerError,
    RuntimeContext,
    TrustPolicyError,
    run_class3_op,
    run_op,
)
from dinkster_nodes_partner.wan import WAN_NODES  # noqa: E402

FIXTURES = PACK_ROOT / "tests" / "fixtures"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


class Response:
    def __init__(self, payload: object = None, content: bytes = b"") -> None:
        self.status = 200
        self.payload = {} if payload is None else payload
        self.content = content
        family = "video/mp4" if content.startswith(b"\x00\x00\x00") else "image/png"
        self.headers = {"Content-Type": family} if content else {}

    async def json(self) -> object:
        return self.payload

    async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        assert chunk_size == 1024 * 1024
        yield self.content

    async def close(self) -> None:
        pass


class Transport:
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


class KlingMatrixTransport:
    def __init__(self, video: bytes) -> None:
        self.video = video
        self.calls: list[tuple[str, str, object]] = []
        self.upload_count = 0

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        return ("203.0.113.8",)

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
        if url.endswith("/customers/storage"):
            index = self.upload_count
            self.upload_count += 1
            return Response(
                {
                    "upload_url": f"https://upload.test/{index}",
                    "download_url": f"https://cdn.test/input-{index}",
                }
            )
        if method == "POST":
            if "kling-3.0-turbo" in url:
                return Response({"code": 0, "data": {"id": "task-1", "status": "submitted"}})
            return Response({"code": 0, "data": {"task_id": "task-1"}})
        if method == "GET" and "/proxy/kling/tasks?" in url:
            return Response(
                {
                    "code": 0,
                    "data": [
                        {
                            "id": "task-1",
                            "status": "succeeded",
                            "outputs": [
                                {"type": "image", "url": "https://cdn.test/skip.png"},
                                {
                                    "type": "video",
                                    "id": "video-1",
                                    "url": "https://cdn.test/out.mp4",
                                    "duration": "5",
                                },
                            ],
                        }
                    ],
                }
            )
        if method == "GET" and "/proxy/kling/" in url:
            return Response(
                {
                    "code": 0,
                    "data": {
                        "task_id": "task-1",
                        "task_status": "succeed",
                        "task_result": {
                            "videos": [
                                {
                                    "id": "video-1",
                                    "url": "https://cdn.test/out.mp4",
                                    "duration": "5",
                                }
                            ],
                            "images": [{"index": 0, "url": "https://cdn.test/out.png"}],
                            "series_images": [{"index": 0, "url": "https://cdn.test/out.png"}],
                        },
                    },
                }
            )
        if method == "GET" and url.endswith(".mp4"):
            return Response(content=self.video)
        if method == "GET" and url.endswith(".png"):
            return Response(content=_png((1, 2, 3)))
        raise AssertionError(f"unexpected matrix request: {method} {url}")

    async def request_bytes(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> Response:
        self.calls.append((method, url, b"<bytes>"))
        return Response()

    async def internet_accessible(self) -> bool:
        return True

    async def close(self) -> None:
        pass


async def _zero_sleep(_: float) -> None:
    pass


def _run(
    spec: OpSpec,
    inputs: Mapping[str, object],
    transport: Transport,
    *,
    helper_registry: HelperRegistry | None = None,
    **kwargs: object,
) -> Mapping[str, object]:
    return asyncio.run(
        run_op(
            spec,
            inputs,
            RuntimeContext(
                transport=transport,
                api_key="key",
                api_base="https://api.test",
                sleep=_zero_sleep,
                **kwargs,  # type: ignore[arg-type]
            ),
            helper_registry,
        )
    )


def _png(rgb: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), rgb).save(stream, "PNG")
    return stream.getvalue()


def _matrix_video() -> bytes:
    av = pytest.importorskip("av")
    stream = io.BytesIO()
    with av.open(stream, mode="w", format="mp4") as container:
        video = container.add_stream("mpeg4", rate=1)
        video.width = 720
        video.height = 720
        video.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((720, 720, 3), np.uint8), format="rgb24")
        for _ in range(3):
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
    return stream.getvalue()


def _normalize_matrix_body(value: object) -> object:
    if isinstance(value, str) and len(value) > 100:
        return "<PNG_BASE64>"
    if isinstance(value, Mapping):
        return {str(key): _normalize_matrix_body(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_matrix_body(item) for item in value]
    return value


def _input_shape(item: object) -> dict[str, object]:
    result = {
        "id": item.id,
        "type": item.type.runtime_type_id(),
        "required": item.required,
        "default": item.default,
    }
    if item.widget is not None and hasattr(item.widget, "options"):
        result["options"] = list(item.widget.options)
    return result


def test_all_23_new_kling_nodes_and_six_class3_variants_execute_exact_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture("kling_requests.json")
    cases = fixture["cases"]
    assert isinstance(cases, dict) and len(cases) == 27
    image = np.zeros((1, 720, 720, 3), np.float32)
    video_bytes = _matrix_video()
    video = {"container": "mp4", "bytes": video_bytes}
    audio = {
        "waveform": np.zeros((1, 1, 24000), np.float32),
        "sample_rate": 8000,
    }
    base: dict[str, object] = {
        "prompt": "matrix prompt",
        "negative_prompt": "matrix negative",
        "cfg_scale": 0.5,
        "aspect_ratio": "16:9",
        "duration": "5",
        "mode": "std",
        "model_name": "kling-v2-master",
        "generate_audio": True,
        "seed": 7,
        "start_frame": image,
        "end_frame": image,
        "image": image,
        "image_left": image,
        "image_right": image,
        "reference_image": image,
        "reference_images": image,
        "first_frame": image,
        "video": video,
        "reference_video": video,
        "audio": audio,
        "sound_file": audio,
        "voice_language": "en",
        "text": "matrix text",
        "voice": "Melody",
        "voice_speed": 1.0,
        "video_id": "video-id",
        "effect_scene": "squish",
        "image_type": "subject",
        "image_fidelity": 0.5,
        "human_fidelity": 0.45,
        "n": 1,
        "resolution": "1080p",
        "series_amount": "disabled",
        "keep_original_sound": True,
        "character_orientation": "video",
        "model": "kling-v2-6",
    }
    overrides: dict[str, dict[str, object]] = {
        "KlingTextToVideoNode": {"mode": "pro mode / 5s duration / kling-v2-5-turbo"},
        "KlingCameraControlI2VNode": {"camera_control": {"type": "simple", "config": {"pan": 1}}},
        "KlingCameraControlT2VNode": {"camera_control": {"type": "simple", "config": {"pan": 1}}},
        "KlingStartEndFrameNode": {"mode": "pro mode / 5s duration / kling-v2-5-turbo"},
        "KlingImageGenerationNode": {"model_name": "kling-v3"},
        "KlingSingleImageVideoEffectNode": {"model_name": "kling-v1-6"},
        "KlingDualCharacterVideoEffectNode": {
            "effect_scene": "hug",
            "model_name": "kling-v1",
        },
        "OmniProTextToVideoNode": {
            "model_name": "kling-v3-omni",
            "duration": 5,
            "storyboards": "disabled",
        },
        "OmniProFirstLastFrameNode": {
            "model_name": "kling-v3-omni",
            "duration": 5,
            "storyboards": "disabled",
            "end_frame": None,
            "reference_images": None,
        },
        "OmniProImageToVideoNode": {
            "model_name": "kling-v3-omni",
            "duration": 5,
            "storyboards": "disabled",
        },
        "OmniProVideoToVideoNode": {
            "model_name": "kling-v3-omni",
            "duration": 3,
            "reference_images": None,
        },
        "OmniProEditVideoNode": {
            "model_name": "kling-v3-omni",
            "reference_images": None,
        },
        "OmniProImageNode": {
            "model_name": "kling-v3-omni",
            "reference_images": None,
        },
        "TextToVideoWithAudio": {"model_name": "kling-v2-6", "mode": "pro"},
        "ImageToVideoWithAudio": {"model_name": "kling-v2-6", "mode": "pro"},
        "MotionControl-video": {
            "character_orientation": "video",
            "model": "kling-v2-6",
        },
        "MotionControl-image": {
            "character_orientation": "image",
            "model": "kling-v2-6",
        },
        "KlingFirstLastFrameNode": {
            "duration": 5,
            "model": "kling-v3",
            "model.resolution": "1080p",
        },
        "KlingAvatarNode": {"mode": "std"},
    }
    for variant, model, frame in (
        ("v3-text", "kling-v3", None),
        ("v3-image", "kling-v3", image),
        ("turbo-text", "kling-3.0-turbo", None),
        ("turbo-image", "kling-3.0-turbo", image),
    ):
        overrides[f"KlingVideo-{variant}"] = {
            "start_frame": frame,
            "multi_shot": "disabled",
            "multi_shot.prompt": "matrix prompt",
            "multi_shot.negative_prompt": "matrix negative",
            "multi_shot.duration": 5,
            "model": model,
            "model.resolution": "720p" if model == "kling-3.0-turbo" else "1080p",
            "model.aspect_ratio": "16:9",
        }

    covered: set[str] = set()
    for case_name, expected_value in cases.items():
        expected = dict(expected_value)
        class_name = (
            "MotionControl"
            if case_name.startswith("MotionControl-")
            else "KlingVideoNode"
            if case_name.startswith("KlingVideo-")
            else case_name
        )
        node = getattr(kling_module, class_name)
        covered.add(class_name)
        inputs = {**base, **overrides.get(case_name, {})}
        transport = KlingMatrixTransport(video_bytes)
        ctx = RuntimeContext(
            transport=transport,
            api_key="key",
            api_base="https://api.test",
            sleep=_zero_sleep,
        )
        monkeypatch.setattr(kling_module, "worker_runtime_context", lambda ctx=ctx: ctx)
        result = asyncio.run(node.execute(**inputs))
        submits = [
            call
            for call in transport.calls
            if call[0] == "POST" and not call[1].endswith("/customers/storage")
        ]
        polls = [
            call[1] for call in transport.calls if call[0] == "GET" and "/proxy/kling/" in call[1]
        ]
        assert len(submits) == 1, case_name
        assert submits[0][1].removeprefix("https://api.test") == expected["path"]
        assert _normalize_matrix_body(submits[0][2]) == expected["body"]
        assert polls == ["https://api.test" + expected["poll"]]
        assert sorted(result) == expected["outputs"]
        if "video" in result:
            assert result["video"] == video
        if case_name in (
            "ImageToVideoWithAudio",
            "KlingVideo-v3-image",
            "KlingFirstLastFrameNode",
        ):
            expected_uploads = 2 if case_name == "KlingFirstLastFrameNode" else 1
            allocations = [
                call for call in transport.calls if call[1].endswith("/customers/storage")
            ]
            puts = [call for call in transport.calls if call[0] == "PUT"]
            assert len(allocations) == expected_uploads
            assert len(puts) == expected_uploads
            assert [call[1] for call in puts] == [
                f"https://upload.test/{index}" for index in range(expected_uploads)
            ]
            assert all(call[2] == b"<bytes>" for call in puts)
            assert max(transport.calls.index(call) for call in puts) < transport.calls.index(
                submits[0]
            )

    assert covered == {
        "KlingTextToVideoNode",
        "KlingImage2VideoNode",
        "KlingCameraControlI2VNode",
        "KlingCameraControlT2VNode",
        "KlingStartEndFrameNode",
        "KlingVideoExtendNode",
        "KlingLipSyncAudioToVideoNode",
        "KlingLipSyncTextToVideoNode",
        "KlingImageGenerationNode",
        "KlingSingleImageVideoEffectNode",
        "KlingDualCharacterVideoEffectNode",
        "OmniProTextToVideoNode",
        "OmniProFirstLastFrameNode",
        "OmniProImageToVideoNode",
        "OmniProVideoToVideoNode",
        "OmniProEditVideoNode",
        "OmniProImageNode",
        "TextToVideoWithAudio",
        "ImageToVideoWithAudio",
        "MotionControl",
        "KlingFirstLastFrameNode",
        "KlingAvatarNode",
        "KlingVideoNode",
    }
    assert {name for name in cases if name.startswith(("MotionControl-", "KlingVideo-"))} == {
        "MotionControl-image",
        "MotionControl-video",
        "KlingVideo-turbo-image",
        "KlingVideo-turbo-text",
        "KlingVideo-v3-image",
        "KlingVideo-v3-text",
    }


@pytest.mark.parametrize("node", KLING_NODES, ids=lambda node: node.__name__)
def test_kling_schema_fixture_parity_and_opspec_round_trip(node: type) -> None:
    expected = _fixture("kling_schemas.json")["nodes"][node.__name__]
    schema = node.define_schema()
    assert {
        "aliases": list(schema.aliases),
        "category": schema.category,
        "display_name": schema.display_name,
        "inputs": [_input_shape(item) for item in schema.inputs],
        "dynamic_combos": [
            {
                "id": combo.id,
                "required": combo.required,
                "default": combo.default,
                "options": [
                    {
                        "key": option.key,
                        "inputs": [_input_shape(item) for item in option.inputs],
                    }
                    for option in combo.options
                ],
            }
            for combo in schema.combos
        ],
        "node_type": schema.node_type,
        "outputs": [
            {"id": output.id, "type": output.type.runtime_type_id()} for output in schema.outputs
        ],
    } == expected
    if isinstance(node.SPEC, Class3Spec):
        assert Class3Spec.from_json(node.SPEC.to_json()) == node.SPEC
    else:
        assert OpSpec.from_json(node.SPEC.to_json()) == node.SPEC


def test_kling_corrected_registration_set_and_all_specs_round_trip() -> None:
    assert len(KLING_NODES) == 25
    assert len({node.define_schema().aliases[0] for node in KLING_NODES}) == 25
    for node in KLING_NODES:
        spec = node.SPEC
        if isinstance(spec, Class3Spec):
            assert Class3Spec.from_json(spec.to_json()) == spec
        else:
            assert OpSpec.from_json(spec.to_json()) == spec


def test_kling_independent_fixture_pins_every_submit_endpoint_family() -> None:
    expected = _fixture("kling_requests.json")["endpoint_families"]
    for node in KLING_NODES:
        alias = node.define_schema().aliases[0]
        if alias not in expected:
            continue
        pinned = expected[alias]
        specs = (
            tuple(node.SPEC.variants.values())
            if isinstance(node.SPEC, Class3Spec)
            else (node.SPEC,)
        )
        paths = {
            adapter.path
            for spec in specs
            for adapter in spec.adapters
            if isinstance(adapter, HttpSyncJson) and adapter.method == "POST"
        }
        required = {pinned} if isinstance(pinned, str) else set(pinned)
        assert required <= paths


def test_class3_helper_registry_manifest_dispatch_and_refusals_are_closed() -> None:
    assert KLING_HELPERS.manifest() == (
        '{"helpers":{"kling.audio-video":["image","text"],'
        '"kling.avatar":["prepare"],"kling.first-last-frame":["prepare"],'
        '"kling.image-generation":["prepare"],'
        '"kling.legacy-video":["extend","image","text"],'
        '"kling.lip-sync":["audio","text"],'
        '"kling.motion-control":["prepare"],"kling.omni-image":["prepare"],'
        '"kling.omni-video":["edit","first-last","images","text","video"],'
        '"kling.task":["creation","image-result","omni-creation",'
        '"omni-image-result","omni-video-result","turbo-creation",'
        '"turbo-video-result","video-result"],"kling.video":["prepare"],'
        '"kling.video-effect":["dual","single"]},"provider":"kling","version":1}'
    )
    registry = HelperRegistry("unit", {"unit.echo": {"prepare": lambda value: value}})
    assert registry.manifest() == (
        '{"helpers":{"unit.echo":["prepare"]},"provider":"unit","version":1}'
    )
    assert registry.invoke("unit.echo", "prepare", {"ok": True}) == {"ok": True}
    for helper_id, stage in (
        ("other.echo", "prepare"),
        ("unit.missing", "prepare"),
        ("unit.echo", "missing"),
    ):
        with pytest.raises(HelperRefusal):
            registry.invoke(helper_id, stage, {})

    fixed = {"nested": [1, {"ok": True}]}
    call = HelperCall(
        "prepare",
        "unit.echo",
        "prepare",
        "before",
        "submit",
        (HelperBinding("value", "value"),),
        fixed,
        ("result",),
    )
    fixed["nested"] = []
    assert call.fixed["nested"] == (1, {"ok": True})


def test_nested_frozen_helper_fixed_payload_executes_as_plain_json() -> None:
    registry = HelperRegistry(
        "unit",
        {"unit.echo": {"prepare": lambda payload: {"result": payload["nested"]}}},
    )
    spec = OpSpec(
        (ValueConstruct("submit", "ignored", ()),),
        helper_calls=(
            HelperCall(
                "prepare",
                "unit.echo",
                "prepare",
                "before",
                "submit",
                fixed={"nested": {"items": [1, {"ok": True}]}},
                outputs=("result",),
            ),
        ),
    )
    ctx = RuntimeContext(Transport(), "token")
    assert asyncio.run(run_op(spec, {}, ctx, registry))["ignored"] == {}


def test_helper_result_with_nested_non_string_mapping_key_refuses() -> None:
    registry = HelperRegistry(
        "unit",
        {"unit.echo": {"prepare": lambda payload: {"result": {("a", "b"): 1, None: 2, True: 3}}}},
    )
    spec = OpSpec(
        (ValueConstruct("submit", "ignored", ()),),
        helper_calls=(
            HelperCall(
                "prepare",
                "unit.echo",
                "prepare",
                "before",
                "submit",
                fixed={},
                outputs=("result",),
            ),
        ),
    )
    ctx = RuntimeContext(Transport(), "token")
    with pytest.raises(PartnerError, match="must contain only finite JSON values"):
        asyncio.run(run_op(spec, {}, ctx, registry))


def test_class3_selector_output_collision_is_refused_for_every_variant() -> None:
    selector = HelperCall("choose", "unit.choose", "prepare", "select", outputs=("body",))
    with pytest.raises(ValueError, match="collides with selector outputs"):
        Class3Spec(selector, {"bad": OpSpec((ValueConstruct("choose.body", "out", ()),))})


def test_class3_direct_binding_and_unknown_selector_state_refuse_at_freeze() -> None:
    with pytest.raises(ValueError, match="source_path must contain strings or integers"):
        HelperBinding("value", "source", source_path=(True,))  # type: ignore[arg-type]
    selector = HelperCall("select", "unit.choose", "prepare", "select", outputs=("known",))
    bad = OpSpec(
        (
            ValueConstruct(
                "body",
                "value",
                (InputBinding("missing", "select.missing"),),
            ),
        )
    )
    with pytest.raises(ValueError, match="references unknown selector outputs.*select.missing"):
        Class3Spec(selector, {name: bad for name in ("first", "middle", "last")})


def test_ordinary_helper_output_collision_with_adapter_id_refuses_at_freeze() -> None:
    helper = HelperCall(
        "prepare", "unit.echo", "prepare", "before", "prepare.value", outputs=("value",)
    )
    with pytest.raises(ValueError, match="adapter ids must not collide"):
        OpSpec((ValueConstruct("prepare.value", "value", ()),), helper_calls=(helper,))


@pytest.mark.parametrize(
    ("node_name", "inputs", "message"),
    (
        (
            "KlingImage2VideoNode",
            {"prompt": "", "negative_prompt": "", "model_name": "kling-v2-master", "mode": "std"},
            "Positive prompt is empty",
        ),
        (
            "KlingLipSyncTextToVideoNode",
            {"text": "x" * 121, "voice": "Melody", "voice_speed": 1.0},
            "Field 'Text cannot be longer than 120 characters; was 121 characters long.",
        ),
        (
            "KlingImageGenerationNode",
            {"prompt": "", "negative_prompt": "", "image_type": "subject"},
            "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
        ),
        (
            "OmniProFirstLastFrameNode",
            {
                "model_name": "kling-v3-omni",
                "prompt": "",
                "duration": 5,
                "resolution": "1080p",
                "generate_audio": False,
            },
            "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
        ),
        (
            "OmniProImageToVideoNode",
            {
                "model_name": "kling-v3-omni",
                "prompt": "",
                "duration": 5,
                "resolution": "1080p",
                "generate_audio": False,
            },
            "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
        ),
        (
            "OmniProImageNode",
            {
                "model_name": "kling-v3-omni",
                "prompt": "",
                "resolution": "1K",
                "series_amount": "disabled",
            },
            "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
        ),
        (
            "ImageToVideoWithAudio",
            {"prompt": "", "generate_audio": True},
            "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
        ),
        (
            "KlingFirstLastFrameNode",
            {"prompt": "", "model": "kling-v3", "model.resolution": "1080p"},
            "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
        ),
        (
            "KlingSingleImageVideoEffectNode",
            {"model_name": "kling-v1-6", "duration": "10"},
            "Input should be '5'",
        ),
    ),
)
def test_kling_local_refusals_precede_invalid_media_and_zero_transport(
    node_name: str, inputs: dict[str, object], message: str
) -> None:
    node = getattr(kling_module, node_name)
    transport = Transport()
    with pytest.raises(PartnerError, match=message.replace("'", "\\'")):
        _run(node.SPEC, inputs, transport, helper_registry=KLING_HELPERS)
    assert transport.calls == []


def test_omni_o1_duration_refusal_order_and_exact_result_messages() -> None:
    transport = Transport()
    inputs = {
        "model_name": "kling-video-o1",
        "prompt": "valid",
        "duration": 3,
        "resolution": "1080p",
        "generate_audio": False,
    }
    with pytest.raises(
        PartnerError,
        match=(
            "Duration is only supported for 5 or 10 seconds if there is no end frame or "
            "reference images"
        ),
    ):
        _run(
            kling_module.OmniProFirstLastFrameNode.SPEC,
            inputs,
            transport,
            helper_registry=KLING_HELPERS,
        )
    assert transport.calls == []


def test_omni_first_last_conflicts_precede_malformed_storyboard() -> None:
    malformed = {
        "storyboards": "1 storyboard",
        "storyboard_1_prompt": "",
        "storyboard_1_duration": 5,
    }
    common = {
        "model_name": "kling-v3-omni",
        "prompt": "valid",
        "duration": 5,
        "resolution": "1080p",
        "generate_audio": False,
        "storyboards": malformed,
    }
    for extras, message in (
        (
            {"end_frame_present": True, "reference_images_present": True},
            "The 'end_frame' input cannot be used simultaneously with 'reference_images'.",
        ),
        (
            {"end_frame_present": True, "reference_images_present": False},
            "The 'end_frame' input cannot be used simultaneously with storyboards.",
        ),
    ):
        with pytest.raises(HelperRefusal, match=message.replace("'", "\\'")):
            KLING_HELPERS.invoke("kling.omni-video", "first-last", {**common, **extras})


def test_image_generation_request_model_negative_prompt_limit_is_pretransport() -> None:
    transport = Transport()
    with pytest.raises(PartnerError, match="String should have at most 200 characters"):
        _run(
            kling_module.KlingImageGenerationNode.SPEC,
            {"prompt": "valid", "negative_prompt": "x" * 201, "image_type": "subject"},
            transport,
            helper_registry=KLING_HELPERS,
        )
    assert transport.calls == []


def test_kling_exact_missing_result_messages() -> None:
    with pytest.raises(
        HelperRefusal,
        match="Kling task task-9 succeeded but no video data found in response",
    ):
        KLING_HELPERS.invoke(
            "kling.task",
            "video-result",
            {"response": {"data": {"task_id": "task-9", "task_result": {"videos": []}}}},
        )
    with pytest.raises(
        HelperRefusal,
        match="Kling task task-8 succeeded but no image data found in response",
    ):
        KLING_HELPERS.invoke(
            "kling.task",
            "image-result",
            {"response": {"data": {"task_id": "task-8", "task_result": {"images": []}}}},
        )


@pytest.mark.parametrize(
    ("node_name", "count", "message"),
    (
        (
            "OmniProFirstLastFrameNode",
            7,
            "The maximum number of reference images allowed is 6.",
        ),
        ("OmniProImageToVideoNode", 8, "The maximum number of reference images is 7."),
        ("OmniProImageNode", 11, "The maximum number of reference images is 10."),
    ),
)
def test_omni_reference_count_refusals_are_exact_and_pretransport(
    node_name: str, count: int, message: str
) -> None:
    inputs: dict[str, object] = {
        "model_name": "kling-v3-omni",
        "prompt": "valid",
        "duration": 5,
        "resolution": "1080p",
        "generate_audio": False,
        "reference_images": np.zeros((count, 720, 720, 3), np.float32),
        "series_amount": "disabled",
    }
    transport = Transport()
    with pytest.raises(PartnerError, match=message.replace(".", r"\.")):
        _run(
            getattr(kling_module, node_name).SPEC,
            inputs,
            transport,
            helper_registry=KLING_HELPERS,
        )
    assert transport.calls == []


def test_omni_video_reference_count_is_exact_before_upload() -> None:
    transport = Transport()
    with pytest.raises(
        PartnerError,
        match="The maximum number of reference images allowed with a video input is 4",
    ):
        _run(
            kling_module.OmniProVideoToVideoNode.SPEC,
            {
                "prompt": "valid",
                "resolution": "1080p",
                "reference_video": {"container": "mp4", "bytes": _matrix_video()},
                "reference_images": np.zeros((5, 720, 720, 3), np.float32),
            },
            transport,
            helper_registry=KLING_HELPERS,
        )
    assert transport.calls == []


@pytest.mark.parametrize(
    ("node_name", "video_input"),
    (("OmniProVideoToVideoNode", "reference_video"), ("OmniProEditVideoNode", "video")),
)
def test_omni_video_and_edit_prompt_validation_is_exact_before_media(
    node_name: str, video_input: str
) -> None:
    transport = Transport()
    with pytest.raises(
        PartnerError,
        match=" Field 'prompt cannot be longer than 2500 characters; was 2501 characters long",
    ):
        _run(
            getattr(kling_module, node_name).SPEC,
            {"prompt": "x" * 2501, "resolution": "1080p", video_input: object()},
            transport,
            helper_registry=KLING_HELPERS,
        )
    assert transport.calls == []


def test_motion_control_prompt_validation_is_exact_before_media_for_both_variants() -> None:
    assert isinstance(kling_module.MotionControl.SPEC, Class3Spec)
    for orientation in ("image", "video"):
        transport = Transport()
        with pytest.raises(
            PartnerError,
            match=" Field 'prompt cannot be longer than 2500 characters; was 2501 characters long",
        ):
            asyncio.run(
                run_class3_op(
                    kling_module.MotionControl.SPEC,
                    {"prompt": "x" * 2501, "character_orientation": orientation},
                    RuntimeContext(transport=transport, api_key="key"),
                    KLING_HELPERS,
                )
            )
        assert transport.calls == []


@pytest.mark.parametrize("ratio", ["2:5", "5:2"])
def test_strict_aspect_string_boundaries_refuse_and_just_inside_passes(ratio: str) -> None:
    strict = OpSpec(
        (
            MediaConstraints(
                "aspect",
                "ratio",
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
        )
    )
    ctx = RuntimeContext(Transport(), "token")
    with pytest.raises(PartnerError, match="aspect ratio must be between"):
        asyncio.run(run_op(strict, {"ratio": ratio}, ctx))
    inside = "401:1000" if ratio == "2:5" else "2499:1000"
    asyncio.run(run_op(strict, {"ratio": inside}, ctx))


@pytest.mark.parametrize(("height", "width", "inside_width"), ((100, 40, 41), (40, 100, 99)))
def test_strict_aspect_array_boundaries_refuse_while_wan_inclusive_passes(
    height: int, width: int, inside_width: int
) -> None:
    strict = OpSpec(
        (
            MediaConstraints(
                "strict",
                "image",
                min_aspect_ratio=0.4,
                max_aspect_ratio=2.5,
                aspect_strict=True,
            ),
        )
    )
    inclusive = OpSpec(
        (MediaConstraints("inclusive", "image", min_aspect_ratio=0.4, max_aspect_ratio=2.5),)
    )
    boundary = np.zeros((1, height, width, 3), np.float32)
    inside = np.zeros((1, height, inside_width, 3), np.float32)
    with pytest.raises(PartnerError, match="aspect ratio must be between"):
        _run(strict, {"image": boundary}, Transport())
    assert _run(strict, {"image": inside}, Transport()) == {}
    assert _run(inclusive, {"image": boundary}, Transport()) == {}
    assert all("aspect_strict" not in node.SPEC.to_json() for node in WAN_NODES)


def test_aspect_strict_freeze_and_json_rules() -> None:
    with pytest.raises(ValueError, match="requires aspect ratio constraints"):
        MediaConstraints("bad", aspect_strict=True)
    with pytest.raises(ValueError, match="distinct bounds"):
        MediaConstraints("bad", min_aspect_ratio=1, max_aspect_ratio=1, aspect_strict=True)
    payload = MediaConstraints(
        "strict", min_aspect_ratio=0.4, max_aspect_ratio=2.5, aspect_strict=True
    )
    assert '"aspect_strict":true' in OpSpec((payload,)).to_json()
    assert "aspect_strict" not in OpSpec((MediaConstraints("default"),)).to_json()
    malformed = json.loads(OpSpec((payload,)).to_json())
    malformed["adapters"][0]["aspect_strict"] = 1
    with pytest.raises(ValueError, match="aspect_strict.*must be a boolean"):
        OpSpec.from_json(json.dumps(malformed))


def test_all_15_pinned_strict_kling_call_sites_are_explicit() -> None:
    def adapter(node: type, adapter_id: str, variant: str | None = None) -> MediaConstraints:
        spec = node.SPEC
        if variant is not None:
            assert isinstance(spec, Class3Spec)
            spec = spec.variants[variant]
        assert isinstance(spec, OpSpec)
        value = next(item for item in spec.adapters if item.id == adapter_id)
        assert isinstance(value, MediaConstraints)
        return value

    sites: dict[str, tuple[MediaConstraints, ...]] = {
        "validate_input_image": tuple(
            adapter(node, "validate_image")
            for node in (
                kling_module.KlingImage2VideoNode,
                kling_module.KlingCameraControlI2VNode,
                kling_module.KlingStartEndFrameNode,
            )
        ),
        "omni_first": (adapter(kling_module.OmniProFirstLastFrameNode, "validate_first"),),
        "omni_end": (adapter(kling_module.OmniProFirstLastFrameNode, "validate_end"),),
        "omni_first_last_references": (
            adapter(kling_module.OmniProFirstLastFrameNode, "validate_references"),
        ),
        "omni_image_to_video_references": (
            adapter(kling_module.OmniProImageToVideoNode, "validate_images"),
        ),
        "omni_video_references": (
            adapter(kling_module.OmniProVideoToVideoNode, "validate_references"),
        ),
        "omni_edit_references": (
            adapter(kling_module.OmniProEditVideoNode, "validate_references"),
        ),
        "omni_image_references": (adapter(kling_module.OmniProImageNode, "validate_references"),),
        "image_with_audio": (adapter(kling_module.ImageToVideoWithAudio, "validate"),),
        "motion_reference": tuple(
            adapter(kling_module.MotionControl, "validate_image", variant)
            for variant in ("image-orientation", "video-orientation")
        ),
        "turbo_start_frame": (adapter(kling_module.KlingVideoNode, "validate", "turbo-image"),),
        "v3_start_frame": (adapter(kling_module.KlingVideoNode, "validate", "v3-image"),),
        "first_last_first": (adapter(kling_module.KlingFirstLastFrameNode, "validate_first"),),
        "first_last_end": (adapter(kling_module.KlingFirstLastFrameNode, "validate_end"),),
        "avatar_image": (adapter(kling_module.KlingAvatarNode, "validate_image"),),
    }
    assert len(sites) == 15
    assert sum(len(expanded) for expanded in sites.values()) == 18
    assert all(step.aspect_strict for expanded in sites.values() for step in expanded)


def test_serialized_video_dimension_probe_width_only_and_both_axes() -> None:
    av = pytest.importorskip("av")
    stream = io.BytesIO()
    with av.open(stream, mode="w", format="mp4") as container:
        video = container.add_stream("mpeg4", rate=1)
        video.width = 720
        video.height = 100
        video.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((100, 720, 3), np.uint8), format="rgb24")
        for _ in range(2):
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
    value = {"container": "mp4", "bytes": stream.getvalue()}
    lip_constraint = next(
        item
        for item in kling_module.KlingLipSyncTextToVideoNode.SPEC.adapters
        if item.id == "validate_video"
    )
    assert isinstance(lip_constraint, MediaConstraints)
    width_only = OpSpec((lip_constraint,))
    assert _run(width_only, {"video": value}, Transport()) == {}
    both_axes = OpSpec((MediaConstraints("validate", "video", min_width=720, min_height=720),))
    with pytest.raises(PartnerError, match="height must be at least 720px"):
        _run(both_axes, {"video": value}, Transport())


def test_serialized_video_probe_matrix_no_stream_no_default_probe_and_one_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    av = pytest.importorskip("av")
    import dinkster_nodes_partner.partner_runtime as runtime

    raw = _matrix_video()
    real_import = runtime.importlib.import_module
    opens = 0

    class AvProxy:
        time_base = av.time_base

        @staticmethod
        def open(*args: object, **kwargs: object) -> object:
            nonlocal opens
            opens += 1
            return av.open(*args, **kwargs)

    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: AvProxy if name == "av" else real_import(name),
    )
    combined = OpSpec(
        (
            MediaConstraints(
                "combined",
                "video",
                min_width=720,
                min_height=720,
                min_duration=2,
                max_duration=4,
            ),
        )
    )
    assert _run(combined, {"video": {"container": "mp4", "bytes": raw}}, Transport()) == {}
    assert opens == 1

    no_probe = OpSpec((MediaConstraints("bytes", "video", max_bytes=100),))
    assert (
        _run(
            no_probe,
            {"video": {"container": "mp4", "bytes": b"not an mp4"}},
            Transport(),
        )
        == {}
    )
    assert opens == 1

    audio_only = io.BytesIO()
    with av.open(audio_only, mode="w", format="mp4") as container:
        stream = container.add_stream("aac", rate=8000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 8000), np.float32), format="flt", layout="mono"
        )
        frame.sample_rate = 8000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with pytest.raises(PartnerError, match="video has no video stream"):
        _run(
            combined,
            {"video": {"container": "mp4", "bytes": audio_only.getvalue()}},
            Transport(),
        )

    lip = next(
        item
        for item in kling_module.KlingLipSyncTextToVideoNode.SPEC.adapters
        if item.id == "validate_video"
    )
    assert isinstance(lip, MediaConstraints)
    assert (lip.min_width, lip.max_width, lip.min_height, lip.max_height) == (720, 1920, 1, None)


def test_kling_audio_text_exact_request_and_transport_success() -> None:
    av = pytest.importorskip("av")
    video_bytes = io.BytesIO()
    with av.open(video_bytes, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    transport = Transport(
        Response({"data": {"task_id": "task-1"}}),
        Response(
            {
                "data": {
                    "task_status": "succeed",
                    "task_result": {
                        "videos": [
                            {"id": "video-1", "url": "https://cdn.test/out.mp4", "duration": "5"}
                        ]
                    },
                }
            }
        ),
        Response(content=video_bytes.getvalue()),
    )
    result = _run(
        TextToVideoWithAudio.SPEC,
        {
            "model_name": "kling-v2-6",
            "prompt": "A fox runs",
            "mode": "pro",
            "aspect_ratio": "16:9",
            "duration": "5",
            "generate_audio": True,
        },
        transport,
        helper_registry=KLING_HELPERS,
    )
    assert transport.calls[0][2] == {
        "model_name": "kling-v2-6",
        "prompt": "A fox runs",
        "mode": "pro",
        "aspect_ratio": "16:9",
        "duration": "5",
        "sound": "on",
    }
    assert result["video"]["container"] == "mp4"


def test_kling_contract_shapes_match_pinned_fixture_and_are_frozen() -> None:
    fixture = _fixture("kling_contracts.json")
    assert fixture["provenance"]["commit"] == "e651b7bef55a5376343dcb1c0edb79f0142c985e"
    schemas = _fixture("kling_schemas.json")
    requests = _fixture("kling_requests.json")
    assert schemas["provenance"] == {
        "commit": "e651b7bef55a5376343dcb1c0edb79f0142c985e",
        "source": "comfy_api_nodes/nodes_kling.py",
    }
    assert requests["provenance"] == {
        "commit": "e651b7bef55a5376343dcb1c0edb79f0142c985e",
        "sources": [
            "comfy_api_nodes/nodes_kling.py",
            "comfy_api_nodes/apis/kling.py",
            "comfy_api_nodes/apis/__init__.py",
        ],
    }
    for name, model in KLING_CONTRACTS.items():
        shape: dict[str, list[object]] = {}
        for field in fields(model):
            required = field.default is MISSING
            value: list[object] = [str(field.type), required]
            if not required:
                value.append(field.default)
            shape[field.name] = value
        assert shape == fixture["model_shapes"][name]
    response = KlingVirtualTryOnResponse()
    with pytest.raises(FrozenInstanceError):
        response.code = 4  # type: ignore[misc]


def test_shared_grammar_defaults_keep_bfl_grok_and_kling_canonical_json_byte_identical() -> None:
    new_fields = {
        "InputBinding": {"value_map"},
        "ResponseSelect": {"paths"},
        "DownloadDecode": {"url_paths"},
        "BatchMapJoin": {"segments", "min_items", "max_items", "min_message", "max_message"},
        "HttpSyncJson": {"formatted"},
        "ValueConstruct": {"formatted"},
        "ProxyUpload": {"optional"},
        "MediaConstraints": {"optional", "duration_media", "batch", "aspect_strict"},
        "OpSpec": {"helper_calls"},
    }

    def legacy(value: object) -> object:
        if is_dataclass(value) and not isinstance(value, type):
            omitted = new_fields.get(type(value).__name__, set())
            return {
                field.name: legacy(getattr(value, field.name))
                for field in fields(value)
                if field.name not in omitted
            }
        if isinstance(value, Mapping):
            return {str(key): legacy(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [legacy(item) for item in value]
        return value

    for node in (*BFL_NODES, *GROK_NODES, KlingCameraControls, KlingVirtualTryOnNode):
        expected = json.dumps(legacy(node.SPEC), separators=(",", ":"), sort_keys=True)
        assert node.SPEC.to_json() == expected


def test_value_construct_builds_pinned_kling_camera_control_and_round_trips() -> None:
    result = _run(
        KlingCameraControls.SPEC,
        {
            "camera_control_type": "simple",
            "horizontal_movement": 1.0,
            "vertical_movement": 2.0,
            "pan": 3.0,
            "tilt": 4.0,
            "roll": 5.0,
            "zoom": 6.0,
        },
        Transport(),
    )
    assert result == {
        "camera_control": {
            "type": "simple",
            "config": {
                "horizontal": 1.0,
                "vertical": 2.0,
                "pan": 3.0,
                "roll": 5.0,
                "tilt": 4.0,
                "zoom": 6.0,
            },
        }
    }
    with pytest.raises(PartnerError, match="at least one.*non-zero"):
        _run(
            KlingCameraControls.SPEC,
            {
                "camera_control_type": "simple",
                "horizontal_movement": 0.0,
                "vertical_movement": 0.0,
                "pan": 0.0,
                "tilt": 0.0,
                "roll": 0.0,
                "zoom": 0.0,
            },
            Transport(),
        )


def test_kling_virtual_try_on_request_poll_and_multi_image_download() -> None:
    red, blue = _png((255, 0, 0)), _png((0, 0, 255))
    transport = Transport(
        Response({"data": {"task_id": "try-1"}}),
        Response({"data": {"task_status": "submitted"}}),
        Response({"data": {"task_status": "processing"}}),
        Response(
            {
                "data": {
                    "task_status": "succeed",
                    "task_result": {
                        "images": [
                            {"url": "https://cdn.test/red.png"},
                            {"url": "https://cdn.test/blue.png"},
                        ]
                    },
                }
            }
        ),
        Response(content=red),
        Response(content=blue),
    )
    images = np.stack(
        (np.full((2, 2, 3), (1, 0, 0), np.float32), np.full((2, 2, 3), (0, 1, 0), np.float32))
    )
    result = _run(
        KlingVirtualTryOnNode.SPEC,
        {
            "human_image": images[:1],
            "cloth_image": images[1:],
            "model_name": "kolors-virtual-try-on-v1-5",
        },
        transport,
    )
    body = transport.calls[0][2]
    assert isinstance(body, dict)
    assert body["model_name"] == "kolors-virtual-try-on-v1-5"
    for key in ("human_image", "cloth_image"):
        assert Image.open(io.BytesIO(base64.b64decode(str(body[key])))).size == (
            2,
            2,
        )
    assert transport.calls[1][1].endswith("/proxy/kling/v1/images/kolors-virtual-try-on/try-1")
    assert np.argmax(result["image"][0, 0, 0]) == 0
    assert np.argmax(result["image"][1, 0, 0]) == 2


def test_kling_virtual_try_on_downscales_both_inputs_to_pinned_four_megapixels() -> None:
    transport = Transport(
        Response({"data": {"task_id": "resize"}}),
        Response(
            {
                "data": {
                    "task_status": "succeed",
                    "task_result": {"images": [{"url": "https://cdn.test/out.png"}]},
                }
            }
        ),
        Response(content=_png((1, 2, 3))),
    )
    large = np.zeros((1, 2050, 2050, 3), np.float32)
    _run(
        KlingVirtualTryOnNode.SPEC,
        {
            "human_image": large,
            "cloth_image": large,
            "model_name": "kolors-virtual-try-on-v1",
        },
        transport,
    )
    body = transport.calls[0][2]
    assert isinstance(body, dict)
    for key in ("human_image", "cloth_image"):
        with Image.open(io.BytesIO(base64.b64decode(str(body[key])))) as image:
            assert image.width * image.height <= 2048 * 2048
            assert image.size != (2050, 2050)


def test_kling_poll_failure_timeout_cancellation_and_absolute_url_trust() -> None:
    inputs = {
        "human_image": np.zeros((1, 2, 2, 3), np.float32),
        "cloth_image": np.zeros((1, 2, 2, 3), np.float32),
        "model_name": "kolors-virtual-try-on-v1",
    }
    with pytest.raises(PartnerError, match="failed"):
        _run(
            KlingVirtualTryOnNode.SPEC,
            inputs,
            Transport(
                Response({"data": {"task_id": "failed"}}),
                Response({"data": {"task_status": "failed"}}),
            ),
        )
    poll = next(a for a in KlingVirtualTryOnNode.SPEC.adapters if isinstance(a, SubmitPoll))
    timeout_spec = OpSpec(
        tuple(
            replace(poll, max_attempts=1) if adapter is poll else adapter
            for adapter in KlingVirtualTryOnNode.SPEC.adapters
        )
    )
    with pytest.raises(PartnerError, match="timed out"):
        _run(
            timeout_spec,
            inputs,
            Transport(
                Response({"data": {"task_id": "timeout"}}),
                Response({"data": {"task_status": "processing"}}),
            ),
        )
    with pytest.raises(OperationCancelled):
        _run(KlingVirtualTryOnNode.SPEC, inputs, Transport(), cancelled=lambda: True)

    private = Transport(
        Response({"data": {"task_id": "private"}}),
        Response(
            {
                "data": {
                    "task_status": "succeed",
                    "task_result": {"images": [{"url": "https://private.test/out.png"}]},
                }
            }
        ),
    )
    private.resolutions["private.test"] = ("127.0.0.1",)
    with pytest.raises(TrustPolicyError):
        _run(KlingVirtualTryOnNode.SPEC, inputs, private)
    assert len(private.calls) == 2


def test_kling_turbo_query_lifecycle_type_filter_timeout_and_cancellation() -> None:
    inputs = {
        "generate_audio": True,
        "seed": 0,
        "start_frame": None,
        "multi_shot": "disabled",
        "multi_shot.prompt": "turbo prompt",
        "multi_shot.negative_prompt": "",
        "multi_shot.duration": 5,
        "model": "kling-3.0-turbo",
        "model.resolution": "720p",
        "model.aspect_ratio": "16:9",
    }

    def execute(spec: Class3Spec, transport: Transport, **kwargs: object) -> Mapping[str, object]:
        return asyncio.run(
            run_class3_op(
                spec,
                inputs,
                RuntimeContext(
                    transport=transport,
                    api_key="key",
                    api_base="https://api.test",
                    sleep=_zero_sleep,
                    **kwargs,  # type: ignore[arg-type]
                ),
                KLING_HELPERS,
            )
        )

    video = _matrix_video()
    transport = Transport(
        Response({"code": 0, "data": {"id": "turbo-1", "status": "submitted"}}),
        Response({"data": [{"id": "turbo-1", "status": "submitted"}]}),
        Response({"data": [{"id": "turbo-1", "status": "processing"}]}),
        Response(
            {
                "data": [
                    {
                        "id": "turbo-1",
                        "status": "succeeded",
                        "outputs": [
                            {"type": "image", "url": "https://cdn.test/skip.png"},
                            {"type": "video", "url": ""},
                            {"type": "video", "url": "https://cdn.test/turbo.mp4"},
                        ],
                    }
                ]
            }
        ),
        Response(content=video),
    )
    assert isinstance(kling_module.KlingVideoNode.SPEC, Class3Spec)
    result = execute(kling_module.KlingVideoNode.SPEC, transport)
    assert result["video"] == {"container": "mp4", "bytes": video}
    assert [call[1] for call in transport.calls if call[0] == "GET"][:3] == [
        "https://api.test/proxy/kling/tasks?task_ids=turbo-1"
    ] * 3

    failed = Transport(
        Response({"code": 0, "data": {"id": "turbo-2"}}),
        Response({"data": [{"id": "turbo-2", "status": "failed"}]}),
    )
    with pytest.raises(PartnerError, match="failed"):
        execute(kling_module.KlingVideoNode.SPEC, failed)

    variants = dict(kling_module.KlingVideoNode.SPEC.variants)
    turbo_text = variants["turbo-text"]
    poll = next(item for item in turbo_text.adapters if isinstance(item, SubmitPoll))
    variants["turbo-text"] = OpSpec(
        tuple(
            replace(item, max_attempts=1) if item is poll else item for item in turbo_text.adapters
        ),
        helper_calls=turbo_text.helper_calls,
    )
    timeout_spec = Class3Spec(kling_module.KlingVideoNode.SPEC.selector, variants)
    with pytest.raises(PartnerError, match="timed out"):
        execute(
            timeout_spec,
            Transport(
                Response({"code": 0, "data": {"id": "turbo-3"}}),
                Response({"data": [{"id": "turbo-3", "status": "processing"}]}),
            ),
        )

    cancelled = Transport()
    with pytest.raises(OperationCancelled):
        execute(kling_module.KlingVideoNode.SPEC, cancelled, cancelled=lambda: True)
    assert cancelled.calls == []


def test_kling_download_byte_cap_is_enforced_before_decode() -> None:
    download = next(a for a in KlingVirtualTryOnNode.SPEC.adapters if isinstance(a, DownloadDecode))
    spec = OpSpec(
        tuple(
            replace(download, byte_cap=3) if adapter is download else adapter
            for adapter in KlingVirtualTryOnNode.SPEC.adapters
        )
    )
    transport = Transport(
        Response({"data": {"task_id": "large"}}),
        Response(
            {
                "data": {
                    "task_status": "succeed",
                    "task_result": {"images": [{"url": "https://cdn.test/out.png"}]},
                }
            }
        ),
        Response(content=_png((1, 2, 3))),
    )
    with pytest.raises(TrustPolicyError, match="byte cap"):
        _run(
            spec,
            {
                "human_image": np.zeros((1, 2, 2, 3), np.float32),
                "cloth_image": np.zeros((1, 2, 2, 3), np.float32),
                "model_name": "kolors-virtual-try-on-v1",
            },
            transport,
        )


@pytest.mark.parametrize(
    ("name", "table"),
    (
        ("mode_text2video", MODE_TEXT2VIDEO),
        ("mode_start_end_frame", MODE_START_END_FRAME),
        ("voice", VOICES_CONFIG),
    ),
)
def test_kling_value_map_fanout_fixture_pins_exact_wire_types(
    name: str, table: Mapping[str, tuple[object, ...]]
) -> None:
    contracts = _fixture("kling_contracts.json")
    fixture = contracts["lookup_examples"][name]
    assert json.loads(json.dumps(table)) == contracts["lookup_tables"][name]
    selected = fixture["input"]
    expected = fixture["body"]
    targets = tuple(expected)
    spec = OpSpec(
        (
            ValueConstruct(
                "body",
                "body",
                tuple(
                    InputBinding(
                        target,
                        "choice",
                        value_map={key: values[index] for key, values in table.items()},
                    )
                    for index, target in enumerate(targets)
                ),
            ),
        )
    )
    assert _run(spec, {"choice": selected}, Transport())["body"] == expected
    assert OpSpec.from_json(spec.to_json()) == spec
    if "duration" in expected:
        assert isinstance(expected["duration"], str)


def test_kling_voice_names_decode_to_pinned_non_ascii_upstream_strings() -> None:
    expected = (
        "\u9633\u5149\u5c11\u5e74",
        "\u61c2\u4e8b\u5c0f\u5f1f",
        "\u8fd0\u52a8\u5c11\u5e74",
        "\u9752\u6625\u5c11\u5973",
        "\u6e29\u67d4\u5c0f\u59b9",
        "\u5143\u6c14\u5c11\u5973",
        "\u9633\u5149\u7537\u751f",
        "\u5e7d\u9ed8\u5c0f\u54e5",
        "\u6587\u827a\u5c0f\u54e5",
        "\u751c\u7f8e\u90bb\u5bb6",
        "\u6e29\u67d4\u59d0\u59d0",
        "\u804c\u573a\u5973\u9752",
        "\u6d3b\u6cfc\u7537\u7ae5",
        "\u4fcf\u76ae\u5973\u7ae5",
        "\u7a33\u91cd\u8001\u7238",
        "\u6e29\u67d4\u5988\u5988",
        "\u4e25\u8083\u4e0a\u53f8",
        "\u4f18\u96c5\u8d35\u5987",
        "\u6148\u7965\u7237\u7237",
        "\u5520\u53e8\u7237\u7237",
        "\u5520\u53e8\u5976\u5976",
        "\u548c\u853c\u5976\u5976",
        "\u4e1c\u5317\u8001\u94c1",
        "\u91cd\u5e86\u5c0f\u4f19",
        "\u56db\u5ddd\u59b9\u5b50",
        "\u6f6e\u6c55\u5927\u53d4",
        "\u53f0\u6e7e\u7537\u751f",
        "\u897f\u5b89\u638c\u67dc",
        "\u5929\u6d25\u59d0\u59d0",
        "\u65b0\u95fb\u64ad\u62a5\u7537",
        "\u8bd1\u5236\u7247\u7537",
        "\u6492\u5a07\u5973\u53cb",
        "\u5200\u7247\u70df\u55d3",
        "\u4e56\u5de7\u6b63\u592a",
    )
    assert tuple(VOICES_CONFIG)[27:] == expected


def test_kling_response_candidate_paths_pin_empty_list_as_miss() -> None:
    spec = OpSpec(
        (
            ValueConstruct(
                "response",
                "unused",
                (InputBinding("series_images", "series"), InputBinding("images", "images")),
            ),
            ResponseSelect(
                "select",
                "response",
                (),
                "selected",
                paths=(("series_images",), ("images",)),
            ),
        )
    )
    assert _run(spec, {"series": [], "images": ["fallback"]}, Transport()) == {
        "unused": {"series_images": [], "images": ["fallback"]},
        "selected": ["fallback"],
    }
