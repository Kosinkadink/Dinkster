"""Text-generation reasoning boundaries and multimodal input contracts on CPU."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    IMAGE_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.text_generation import TextGenerate, TextGenerateLTX2Prompt

from dinkster.compose import PackSpec, ServingComposer


class Clip:
    def __init__(self, text: str = "answer", suffix: str = "", dictionary: bool = False) -> None:
        self.text = text
        self.suffix = suffix
        self.dictionary = dictionary
        self.prompt = ""
        self.media: dict[str, Any] = {}
        self.sampling: dict[str, Any] = {}
        self.tokenizer = SimpleNamespace(clip_name="gemma3")

    def tokenize(self, prompt: str, **kwargs: Any) -> Any:
        self.prompt = prompt
        self.media = kwargs
        tokens = [[("ignored", 1.0), (17, 1.0), (23, 0.5), (42, 1.0), (19, 1.0), (8, 1.0)]]
        return {"gemma": tokens} if self.dictionary else tokens

    def generate(self, tokens: Any, **kwargs: Any) -> str:
        self.sampling = kwargs
        return "generated"

    def decode(self, tokens: Any) -> str:
        if tokens == "generated":
            return self.text
        assert tokens == [23, 42, 19, 8]
        return self.suffix


@pytest.mark.parametrize(
    ("text", "prompt", "suffix", "expected"),
    [
        ("  ordinary answer  ", "question", "", ("ordinary answer", "")),
        ("<think> reason </think> answer ", "question", "", ("answer", "reason")),
        (" reason </think>answer</think>tail", "question", "", ("answer</think>tail", "reason")),
        (" <think> unfinished ", "question", "", ("", "unfinished")),
        (" unfinished ", "question <think> ", "", ("", "unfinished")),
        (" unfinished ", "question", "template <think> ", ("", "unfinished")),
    ],
)
@pytest.mark.parametrize("dictionary", (True, False))
def test_generation_separates_completed_and_truncated_reasoning(
    text: str, prompt: str, suffix: str, expected: tuple[str, str], dictionary: bool
) -> None:
    result = TextGenerate.execute(
        clip=Clip(text, suffix, dictionary), prompt=prompt, max_length=37, sampling_mode="off"
    )
    assert (result["generated_text"], result["thinking"]) == expected


def test_generation_materializes_media_and_preserves_sampling_values() -> None:
    clip = Clip()
    image = torch.arange(1 * 3 * 5 * 4, dtype=torch.float32).reshape(1, 3, 5, 4) / 64
    video = torch.ones((2, 5, 3, 3)) * 0.25
    waveform = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    result = TextGenerate.execute(
        clip=clip,
        prompt="question",
        max_length=37,
        sampling_mode="on",
        image=image.numpy(),
        video=video.numpy(),
        audio={"waveform": waveform.numpy(), "sample_rate": 16000},
        thinking=True,
        use_default_template=False,
        system_prompt="ignored",
        mtp="tree2s1",
        **{
            "sampling_mode.temperature": 0.31,
            "sampling_mode.top_k": 7,
            "sampling_mode.top_p": 0.82,
            "sampling_mode.min_p": 0.13,
            "sampling_mode.repetition_penalty": 1.2,
            "sampling_mode.seed": 29,
            "sampling_mode.presence_penalty": 0.47,
        },
    )
    assert result == {"generated_text": "answer", "thinking": ""}
    assert clip.prompt == "question"
    assert clip.media["skip_template"] is True
    assert clip.media["min_length"] == 1
    assert clip.media["thinking"] is True
    assert clip.media["system_prompt"] == ""
    torch.testing.assert_close(clip.media["image"], image, rtol=0, atol=0)
    torch.testing.assert_close(clip.media["video"], video, rtol=0, atol=0)
    torch.testing.assert_close(clip.media["audio"]["waveform"], waveform, rtol=0, atol=0)
    assert clip.media["audio"]["sample_rate"] == 16000
    assert clip.sampling == {
        "do_sample": True,
        "max_length": 37,
        "temperature": 0.31,
        "top_k": 7,
        "top_p": 0.82,
        "min_p": 0.13,
        "repetition_penalty": 1.2,
        "presence_penalty": 0.47,
        "seed": 29,
        "mtp": "tree2s1",
    }


@pytest.mark.parametrize(("mtp", "expected"), [("off", False), ("auto", True), ("3", "3")])
def test_generation_off_mode_uses_reference_fallbacks(mtp: str, expected: object) -> None:
    clip = Clip()
    TextGenerate.execute(
        clip=clip,
        prompt="question",
        max_length=37,
        sampling_mode={"sampling_mode": "off"},
        mtp=mtp,
        system_prompt="system",
    )
    assert clip.media["system_prompt"] == "system"
    assert clip.sampling == {
        "do_sample": False,
        "max_length": 37,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "seed": None,
        "mtp": expected,
    }


class TextGenerationSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="text-generation-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("clip", CLIP_TYPE),
                    ("image", IMAGE_TYPE),
                    ("video", IMAGE_TYPE),
                    ("audio", AUDIO_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            clip=Clip("<think>reason</think> answer"),
            image=torch.ones((1, 3, 5, 3)).numpy(),
            video=torch.zeros((2, 5, 3, 3)).numpy(),
            audio={"waveform": torch.zeros((1, 2, 8)).numpy(), "sample_rate": 16000},
        )


@pytest.mark.parametrize("in_process", (True,))
@pytest.mark.parametrize("node_type", ("TextGenerate", "TextGenerateLTX2Prompt"))
def test_generation_ordinary_manifest_hosts(
    tmp_path: Path, in_process: bool, node_type: str
) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(root / "tests"),
                        str(root / "src"),
                        os.environ.get("PYTHONPATH", ""),
                    )
                )
            }
        )
        try:
            fixture_module = f"text_generation_fixture_{node_type}"
            (tmp_path / f"{fixture_module}.py").write_text(
                "from test_text_generation_torch import TextGenerationSource\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (TextGenerationSource,)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "text-generation-fixture.toml"
            fixture.write_text(
                '[pack]\nname = "text-generation-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["text-generation-fixture"]\n'
                f'[pack.entry]\nnodes = "{fixture_module}:NODES"\n'
                f'types = "{fixture_module}:register_inference_types"\n',
                encoding="ascii",
            )
            native = root / "dinkster-pack.toml"
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="text-generation",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("text-generation-fixture.source"),
                    **{
                        mode: GraphNode(
                            node_type,
                            inputs={
                                "clip": Link("source", "clip"),
                                "prompt": "question",
                                "max_length": 37,
                                "image": Link("source", "image"),
                                "video": Link("source", "video"),
                                "audio": Link("source", "audio"),
                                **(
                                    {
                                        "sampling_mode.temperature": 0.31,
                                        "sampling_mode.top_k": 7,
                                        "sampling_mode.top_p": 0.82,
                                        "sampling_mode.min_p": 0.13,
                                        "sampling_mode.repetition_penalty": 1.2,
                                        "sampling_mode.seed": 29,
                                        "sampling_mode.presence_penalty": 0.47,
                                    }
                                    if mode == "on"
                                    else {}
                                ),
                            },
                            slot_variants={"sampling_mode": mode},
                        )
                        for mode in ("on", "off")
                    },
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["on", "off"])
                for mode in ("on", "off"):
                    assert result.outputs[mode]["generated_text"].resolve() == "answer"
                    assert result.outputs[mode]["thinking"].resolve() == "reason"
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("gemma4", "with_image", "digest"),
    [
        (False, False, "f00b22f47dad68358f5c2c7396c701db95095cf26dc3dbd6b5556eab04692071"),
        (False, True, "366efb00c104803e2276af92d6492456ef2b729e030b9112eb49faf5eea589b0"),
        (True, False, "36bb8e2b530a7489f5a3dd7e4d5c7b1fc037ec46e20ddedb587a9e61caf86520"),
        (True, True, "fa6f57a977ab952abb66cc91d42ab86ba152c885d8b55c05134559642d28577e"),
    ],
)
@pytest.mark.parametrize("thinking", (True, False))
def test_ltx_prompt_selects_exact_model_and_image_template(
    gemma4: bool, with_image: bool, digest: str, thinking: bool
) -> None:
    clip = Clip("<think>reason</think> caption")
    clip.tokenizer.clip_name = "ltx_gemma4" if gemma4 else "ltx_gemma3"
    image = torch.ones((1, 3, 5, 3)) if with_image else None
    result = TextGenerateLTX2Prompt.execute(
        clip=clip,
        prompt="A bird turns left",
        max_length=37,
        sampling_mode="off",
        image=image,
        thinking=thinking,
        system_prompt="  ",
    )
    assert result == {"generated_text": "caption", "thinking": "reason"}
    if gemma4:
        system, suffix = clip.prompt.removeprefix("<|turn>system\n").split("<turn|>\n", 1)
        if thinking:
            assert system.startswith("<|think|>\n")
            system = system.removeprefix("<|think|>\n")
        assert hashlib.sha256(system.encode()).hexdigest() == digest
        user = (
            "<|image><|image|><image|>\n\nUser Raw Input Prompt: A bird turns left."
            if with_image
            else "user prompt: A bird turns left"
        )
        assert suffix == (
            f"<|turn>user\n{user}<turn|>\n<|turn>model\n"
            + ("" if thinking else "<|channel>final\n")
        )
    else:
        system, suffix = clip.prompt.removeprefix("<start_of_turn>system\n").split(
            "<end_of_turn>\n", 1
        )
        assert hashlib.sha256(system.encode()).hexdigest() == digest
        media = "\n<image_soft_token>\n" if with_image else ""
        assert suffix == (
            f"<start_of_turn>user\n{media}\nUser Raw Input Prompt: A bird turns left."
            "<end_of_turn>\n<start_of_turn>model\n"
        )
    assert clip.media["system_prompt"] == ""
    assert clip.media["thinking"] is thinking
    assert clip.media["image"] is image


@pytest.mark.parametrize("gemma4", (True, False))
def test_ltx_custom_system_and_empty_generation_fallback(gemma4: bool) -> None:
    clip = Clip("<think>unfinished")
    clip.tokenizer.clip_name = "gemma4" if gemma4 else "gemma3"
    result = TextGenerateLTX2Prompt.execute(
        clip=clip,
        prompt="original prompt",
        max_length=37,
        sampling_mode="on",
        thinking=True,
        use_default_template=False,
        system_prompt="  custom system  ",
        **{"sampling_mode.seed": 43},
    )
    assert result == {"generated_text": "original prompt", "thinking": "unfinished"}
    assert clip.prompt.startswith(
        "<|turn>system\n<|think|>\ncustom system<turn|>\n"
        if gemma4
        else "<start_of_turn>system\ncustom system<end_of_turn>\n"
    )
    assert clip.media["skip_template"] is True
    assert clip.media["system_prompt"] == ""
    assert clip.sampling["seed"] == 43
