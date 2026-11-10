"""LTX prompt templates and empty-generation behavior on CPU."""

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_ltx.text_generation import TextGenerateLTX2Prompt


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
