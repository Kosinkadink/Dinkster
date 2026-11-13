"""Compact storage is converted at typed invocation, never at transport."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import audio_input, image_input
from dinkster_protocol import Invocation
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, IsolatedWorker

from tests.async_scenarios import run_scenario

IMAGE = TypeExpr.concrete("test.image")
BF16 = np.dtype([("bfloat16", "<u2")])


def pixels(dtype: str) -> Any:
    if dtype == "bf16":
        return np.array([0, 0x3F80], dtype=np.uint16).reshape(1, 1, 2, 1).view(BF16)
    maximum = {"uint8": 255, "uint16": 65535}.get(dtype, 1)
    return np.array([0, maximum], dtype=dtype).reshape(1, 1, 2, 1)


def register_types(registry: TypeRegistry) -> None:
    # The fallback codec preserves arrays independently of media codec policy.
    registry.register("test.image", input_convert=image_input)
    registry.register("test.audio", input_convert=audio_input)
    registry.register("ASSET")
    registry.register_asset_decoder(
        "test.image", provider_id="test.storage-decode", decode=lambda obj: obj
    )
    registry.register_asset_decoder(
        "list<test.image>", provider_id="test.storage-list-decode", decode=lambda obj: obj
    )
    registry.register_batch_merge(
        "test.image",
        provider_id="test.storage-merge",
        merge=lambda arrays: np.concatenate(cast(Any, arrays)),
    )


class FloatInput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.float-input",
            inputs=(InputSpec("image", IMAGE),),
            outputs=(OutputSpec("dtype", TypeExpr.concrete("STRING")),),
        )

    @classmethod
    def execute(cls, image: Any) -> Mapping[str, object]:
        if isinstance(image, list):
            return cls.outputs(dtype=",".join(str(item.dtype) for item in image))
        if isinstance(image, dict):
            image = image["waveform"]
        return cls.outputs(dtype=str(image.dtype))


class StorageInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatInput.schema(),
            node_type="test.storage-input",
            inputs=(InputSpec("image", IMAGE, accepts_storage=True),),
        )


class FloatListInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatInput.schema(),
            node_type="test.float-list-input",
            inputs=(InputSpec("image", TypeExpr.list_of(IMAGE)),),
        )


class StorageListInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatListInput.schema(),
            node_type="test.storage-list-input",
            inputs=(InputSpec("image", TypeExpr.list_of(IMAGE), accepts_storage=True),),
        )


class FloatAudioInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatInput.schema(),
            node_type="test.float-audio-input",
            inputs=(InputSpec("image", TypeExpr.concrete("test.audio")),),
        )


class StorageAudioInput(FloatInput):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            FloatAudioInput.schema(),
            node_type="test.storage-audio-input",
            inputs=(InputSpec("image", TypeExpr.concrete("test.audio"), accepts_storage=True),),
        )


NODES = [
    FloatInput,
    StorageInput,
    FloatListInput,
    StorageListInput,
    FloatAudioInput,
    StorageAudioInput,
]


def registry() -> TypeRegistry:
    result = TypeRegistry()
    register_core_types(result)
    register_types(result)
    return result


@pytest.mark.parametrize("isolated", [False, True])
def test_invocation_storage_contract_crosses_worker_boundary(
    tmp_path: Path, isolated: bool
) -> None:
    async def scenario() -> None:
        types = registry()
        if isolated:
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "storage-test"\npublisher = "dinkster"\n'
                'type-namespaces = ["test"]\n[pack.entry]\n'
                'nodes = "tests.test_storage_inputs:NODES"\n'
                'types = "tests.test_storage_inputs:register_types"\n'
            )
            worker = IsolatedWorker(
                manifest, types, extra_env={"PYTHONPATH": str(Path(__file__).parents[1])}
            )
            await worker.start()
        else:
            worker = InProcessWorker({node.schema().node_type: node for node in NODES}, types)
        try:
            for node in NODES:
                schema = node.schema()
                spec = schema.inputs[0]
                audio = spec.type == TypeExpr.concrete("test.audio")
                listed = spec.type.kind == "list"
                dtypes = (
                    ("int16", "float32")
                    if audio
                    else ("uint8", "uint16", "float16", "bf16", "float32")
                )
                sources = (
                    ("test.audio",)
                    if audio
                    else (("list<test.image>",) if listed else ("test.image",))
                    if isolated
                    else (
                        ("list<test.image>", "asset<list<test.image>>", "list<asset<test.image>>")
                        if listed
                        else ("test.image", "asset<test.image>", "list<asset<test.image>>")
                    )
                )
                for dtype in dtypes:
                    for source_type in sources:
                        array = pixels(dtype)
                        obj = (
                            {"waveform": array, "sample_rate": 48000}
                            if audio
                            else ([array, array] if "list<" in source_type else array)
                        )
                        value = types.wrap(source_type, obj)
                        result = await worker.invoke(
                            Invocation(
                                "invocation", "node", schema.node_type, {"image": value}, schema
                            )
                        )
                        assert result.error is None, result.error
                        assert result.outputs is not None
                        expected = str(array.dtype) if spec.accepts_storage else "float32"
                        if listed:
                            expected = ",".join([expected, expected])
                        assert result.outputs["dtype"].resolve() == expected
        finally:
            if isinstance(worker, IsolatedWorker):
                await worker.close()

    run_scenario(scenario())
