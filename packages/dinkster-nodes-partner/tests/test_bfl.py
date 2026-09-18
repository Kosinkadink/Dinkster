from __future__ import annotations

import asyncio
import io
import json
import sys
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import MISSING, FrozenInstanceError, fields
from pathlib import Path

import numpy
import pytest
from PIL import Image

PACK_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACK_ROOT / "src"))

import dinkster_nodes_partner.bfl as bfl_module  # noqa: E402
from dinkster_api.v1 import (  # noqa: E402
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    NumberWidget,
)
from dinkster_nodes_partner.bfl import (  # noqa: E402
    BFL_CONTRACTS,
    BFL_NODES,
    BFLGenerateResponse,
    BFLStatusResponse,
    Flux2ImageNode,
    Flux2ProImageNode,
    Flux2Request,
    FluxKontextProImageNode,
    FluxProExpandNode,
    FluxProUltraImageNode,
)
from dinkster_nodes_partner.opspec import OpSpec  # noqa: E402
from dinkster_nodes_partner.partner_runtime import RuntimeContext, run_op  # noqa: E402

FIXTURES = PACK_ROOT / "tests" / "fixtures"


class Response:
    def __init__(self, payload: object = None, *, content: bytes = b"") -> None:
        self.status = 200
        self.headers = {"Content-Type": "image/png"} if content else {}
        self.payload = payload
        self.content = content

    async def json(self) -> object:
        return self.payload

    async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        yield self.content

    async def close(self) -> None:
        pass


class Transport:
    def __init__(self, png: bytes) -> None:
        self.responses = [
            Response({"id": "job", "polling_url": "https://poll.bfl.test/job"}),
            Response(
                {
                    "id": "job",
                    "status": "Ready",
                    "result": {"sample": "https://cdn.bfl.test/out.png"},
                }
            ),
            Response(content=png),
        ]
        self.requests: list[tuple[str, str, Mapping[str, str], object]] = []

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        return ("203.0.113.2",)

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object] | None,
        timeout: float,
    ) -> Response:
        self.requests.append((method, url, headers, json_body))
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
        raise AssertionError("BFL nodes do not use proxy_upload")

    async def internet_accessible(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


def _image() -> numpy.ndarray:
    return numpy.full((1, 256, 256, 3), 0.5, dtype=numpy.float32)


def _inputs(node: type) -> dict[str, object]:
    schema = node.define_schema()
    values = {item.id: item.default for item in schema.inputs}
    for item in schema.inputs:
        if item.type.runtime_type_id() == "comfy.IMAGE":
            values[item.id] = _image()
        elif item.type.runtime_type_id() == "comfy.MASK":
            values[item.id] = numpy.ones((1, 256, 256), dtype=numpy.float32)
    if node is Flux2ImageNode:
        values.update(
            {
                "model": "Flux.2 [pro]",
                "model.width": 1024,
                "model.height": 768,
                "model.images.image_1": _image(),
            }
        )
    return values


@pytest.mark.parametrize("node", BFL_NODES, ids=lambda node: node.__name__)
def test_bfl_opspec_executes_exact_submit_poll_select_and_download(node: type) -> None:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(stream, "PNG")
    transport = Transport(stream.getvalue())
    result = asyncio.run(
        run_op(
            node.SPEC,
            _inputs(node),
            RuntimeContext(
                transport=transport, api_key="fixture", sleep=lambda _: asyncio.sleep(0)
            ),
        )
    )
    contract = _fixture("bfl_contracts.json")["nodes"][node.__name__]
    submit = transport.requests[0]
    expected_path = contract["path"]
    assert submit[0] == "POST"
    assert submit[1].endswith(expected_path)
    expected_fields = set(contract["request_fields"]) | set(contract["fixed"])
    assert set(submit[3]) == expected_fields
    assert {name: submit[3][name] for name in contract["fixed"]} == contract["fixed"]
    assert transport.requests[1][0:2] == ("GET", "https://poll.bfl.test/job")
    assert transport.requests[1][2] == {"Accept": "application/json"}
    assert transport.requests[2][1] == "https://cdn.bfl.test/out.png"
    assert numpy.asarray(result["image"]).shape == (1, 2, 2, 3)


@pytest.mark.parametrize("node", BFL_NODES, ids=lambda node: node.__name__)
def test_bfl_contract_fixture_and_opspec_round_trip(node: type) -> None:
    contracts = _fixture("bfl_contracts.json")
    assert contracts["provenance"]["commit"] == "e651b7bef55a5376343dcb1c0edb79f0142c985e"
    assert node.__name__ in contracts["nodes"]
    assert OpSpec.from_json(node.SPEC.to_json()) == node.SPEC


def test_bfl_frozen_contract_models_match_pinned_api_fixture_and_validate() -> None:
    contracts = _fixture("bfl_contracts.json")
    for name, model in BFL_CONTRACTS.items():
        assert {field.name for field in fields(model)} == set(contracts["models"][name])
        shape = {}
        for field in fields(model):
            required = field.default is MISSING
            description = [str(field.type), required]
            if not required:
                description.append(field.default)
            shape[field.name] = description
        assert shape == contracts["model_shapes"][name]

    response = BFLGenerateResponse("task", "https://poll.test/task")
    with pytest.raises(FrozenInstanceError):
        response.id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="polling_url"):
        BFLGenerateResponse("task", "")
    with pytest.raises(ValueError, match="multiples of 32"):
        Flux2Request(prompt="prompt", width=257)
    with pytest.raises(ValueError, match="unknown BFL status"):
        BFLStatusResponse("task", "Mystery")


def test_bfl_optional_media_rounding_and_channel_request_mapping() -> None:
    stream = io.BytesIO()
    Image.new("RGB", (1, 1)).save(stream, "PNG")

    ultra_transport = Transport(stream.getvalue())
    ultra_inputs = _inputs(FluxProUltraImageNode)
    ultra_inputs["image_prompt"] = None
    asyncio.run(
        run_op(
            FluxProUltraImageNode.SPEC,
            ultra_inputs,
            RuntimeContext(
                transport=ultra_transport,
                api_key="key",
                sleep=lambda _: asyncio.sleep(0),
            ),
        )
    )
    ultra_body = ultra_transport.requests[0][3]
    assert "image_prompt" not in ultra_body
    assert "image_prompt_strength" not in ultra_body

    kontext_transport = Transport(stream.getvalue())
    kontext_inputs = _inputs(FluxKontextProImageNode)
    rgba = numpy.zeros((1, 2, 2, 4), dtype=numpy.float32)
    rgba[..., 0] = 1
    rgba[..., 3] = 0.25
    kontext_inputs.update(input_image=rgba, guidance=3.06)
    asyncio.run(
        run_op(
            FluxKontextProImageNode.SPEC,
            kontext_inputs,
            RuntimeContext(
                transport=kontext_transport,
                api_key="key",
                sleep=lambda _: asyncio.sleep(0),
            ),
        )
    )
    kontext_body = kontext_transport.requests[0][3]
    assert kontext_body["guidance"] == 3.1
    encoded = Image.open(io.BytesIO(__import__("base64").b64decode(kontext_body["input_image"])))
    assert encoded.mode == "RGBA"

    expand_transport = Transport(stream.getvalue())
    expand_inputs = _inputs(FluxProExpandNode)
    expand_inputs["image"] = rgba
    asyncio.run(
        run_op(
            FluxProExpandNode.SPEC,
            expand_inputs,
            RuntimeContext(
                transport=expand_transport,
                api_key="key",
                sleep=lambda _: asyncio.sleep(0),
            ),
        )
    )
    expand_body = expand_transport.requests[0][3]
    expand_image = Image.open(io.BytesIO(__import__("base64").b64decode(expand_body["image"])))
    assert expand_image.mode == "RGBA"


def test_bfl_flux2_maps_all_numbered_references_and_dynamic_max_endpoint() -> None:
    stream = io.BytesIO()
    Image.new("RGB", (1, 1)).save(stream, "PNG")
    batch = numpy.zeros((9, 2, 2, 3), dtype=numpy.float32)

    deprecated_transport = Transport(stream.getvalue())
    deprecated_inputs = _inputs(Flux2ProImageNode)
    deprecated_inputs["images"] = batch
    asyncio.run(
        run_op(
            Flux2ProImageNode.SPEC,
            deprecated_inputs,
            RuntimeContext(
                transport=deprecated_transport,
                api_key="key",
                sleep=lambda _: asyncio.sleep(0),
            ),
        )
    )
    deprecated_body = deprecated_transport.requests[0][3]
    assert [key for key in deprecated_body if key.startswith("input_image")] == [
        "input_image",
        *(f"input_image_{index}" for index in range(2, 10)),
    ]

    dynamic_transport = Transport(stream.getvalue())
    dynamic_inputs = _inputs(Flux2ImageNode)
    dynamic_inputs.update(
        {
            "model": "Flux.2 [max]",
            "model.width": 512,
            "model.height": 256,
            "model.images.image_1": batch[:3],
            "model.images.image_2": None,
            "model.images.image_3": batch[3:8],
        }
    )
    asyncio.run(
        run_op(
            Flux2ImageNode.SPEC,
            dynamic_inputs,
            RuntimeContext(
                transport=dynamic_transport,
                api_key="key",
                sleep=lambda _: asyncio.sleep(0),
            ),
        )
    )
    assert dynamic_transport.requests[0][1].endswith("/proxy/bfl/flux-2-max/generate")
    dynamic_body = dynamic_transport.requests[0][3]
    assert (dynamic_body["width"], dynamic_body["height"]) == (512, 256)
    assert len([key for key in dynamic_body if key.startswith("input_image")]) == 8


def test_bfl_flux2_execute_accepts_flat_worker_kwargs_with_zero_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.BytesIO()
    Image.new("RGB", (1, 1)).save(stream, "PNG")
    transport = Transport(stream.getvalue())
    monkeypatch.setattr(
        bfl_module,
        "worker_runtime_context",
        lambda: RuntimeContext(
            transport=transport,
            api_key="key",
            sleep=lambda _: asyncio.sleep(0),
        ),
    )
    result = asyncio.run(
        Flux2ImageNode.execute(
            prompt="test",
            seed=1,
            model="Flux.2 [pro]",
            **{"model.width": 512, "model.height": 256},
        )
    )
    assert numpy.asarray(result["image"]).shape == (1, 1, 1, 3)
    assert transport.requests[0][1].endswith("/proxy/bfl/flux-2-pro/generate")
    assert not any(key.startswith("input_image") for key in transport.requests[0][3])


@pytest.mark.parametrize("node", BFL_NODES, ids=lambda node: node.__name__)
def test_bfl_schema_parity_including_nested_dynamic_options(node: type) -> None:
    fixture = _fixture("bfl_schemas.json")
    assert fixture["provenance"]["commit"] == "e651b7bef55a5376343dcb1c0edb79f0142c985e"
    schema = node.define_schema()
    actual_inputs = [
        {
            **({"advanced": True} if item.advanced else {}),
            "id": item.id,
            "type": item.type.runtime_type_id(),
            "default": item.default,
            "required": item.required,
        }
        for item in schema.inputs
    ]

    def nested_input(item) -> dict[str, object]:
        result = {
            "id": item.id,
            "kind": type(item).__name__,
            "type": item.type.runtime_type_id(),
        }
        if type(item).__name__ == "InputSpec":
            result.update(default=item.default, required=item.required)
        else:
            result.update(min_members=item.min_members, member_names=list(item.member_names))
        return result

    actual_combos = [
        {
            "id": combo.id,
            "options": [
                {
                    "key": option.key,
                    "inputs": [nested_input(item) for item in option.inputs],
                }
                for option in combo.options
            ],
        }
        for combo in schema.combos
    ]
    expected = fixture["nodes"][node.__name__]
    assert actual_inputs == expected["inputs"]
    assert actual_combos == expected["combos"]
    assert schema.aliases == (node.__name__,)
    assert schema.node_type.startswith("partner.bfl.")
    assert schema.node_type == schema.node_type.lower()

    if node is Flux2ImageNode:
        for option in schema.combos[0].options:
            width, height, family = option.inputs
            assert (width.id, width.default, width.required, width.type.runtime_type_id()) == (
                "width",
                1024,
                True,
                "core.int",
            )
            assert (height.id, height.default, height.required, height.type.runtime_type_id()) == (
                "height",
                768,
                True,
                "core.int",
            )
            assert family.member_names == tuple(f"image_{index}" for index in range(1, 9))
            assert family.type.runtime_type_id() == "comfy.IMAGE"


@pytest.mark.parametrize("node", BFL_NODES, ids=lambda node: node.__name__)
def test_bfl_number_widget_bounds_are_json_double_safe(node: type) -> None:
    """Upstream declares a 2^64-1 seed maximum on the shared seed widget,
    but an int beyond 2^53-1 is lossy through a JSON double; that widget
    clamps to the safe bound (a documented parity delta - Erase/Flux2 seeds
    keep their upstream 2^31-1 maximum, which is already safe) and no BFL
    widget bound exceeds it anywhere, nested dynamic-combo branch inputs
    included."""
    safe = 2**53 - 1
    assert bfl_module.SEED_WIDGET.max == safe

    def iter_number_widgets(entries: Iterable[object]) -> list[tuple[str, NumberWidget]]:
        found: list[tuple[str, NumberWidget]] = []
        for entry in entries:
            if isinstance(entry, InputSpec):
                if isinstance(entry.widget, NumberWidget):
                    found.append((entry.id, entry.widget))
            elif isinstance(entry, InputFamilySpec):
                found.extend(iter_number_widgets(entry.template))
            elif isinstance(entry, DynamicComboSpec):
                for option in entry.options:
                    found.extend(iter_number_widgets(option.inputs))
            else:
                raise AssertionError(f"unhandled schema entry kind: {entry!r}")
        return found

    schema = node.define_schema()
    widgets = iter_number_widgets(schema.inputs)
    widgets += iter_number_widgets(schema.input_families)
    widgets += iter_number_widgets(schema.combos)
    for item_id, widget in widgets:
        for bound in (widget.min, widget.max, widget.step):
            if isinstance(bound, int):
                assert abs(bound) <= safe, (node.__name__, item_id, bound)
