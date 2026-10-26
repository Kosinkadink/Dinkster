"""Text generation through the loaded text encoder's native generation API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CLIP_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    materialize_audio,
    materialize_image,
)

_STRING = TypeExpr.concrete(CORE_STRING)


class TextGenerate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextGenerate",
            display_name="Generate Text",
            category="text",
            search_terms=("LLM", "gemma"),
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec(
                    "prompt",
                    _STRING,
                    default="",
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE), required=False),
                InputSpec(
                    "video",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc=(
                        "Video frames as image batch. Assumed to be 24 FPS; "
                        "subsampled to 1 FPS internally."
                    ),
                ),
                InputSpec("audio", TypeExpr.concrete(AUDIO_TYPE), required=False),
                InputSpec(
                    "max_length",
                    TypeExpr.concrete(CORE_INT),
                    default=512,
                    widget=NumberWidget(min=1, max=32768),
                ),
                InputSpec(
                    "thinking",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    required=False,
                    default=False,
                    doc="Operate in thinking mode if the model supports it.",
                ),
                InputSpec(
                    "use_default_template",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    required=False,
                    default=True,
                    advanced=True,
                    doc="Use the built in system prompt/template if the model has one.",
                ),
                InputSpec(
                    "mtp",
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default="auto",
                    widget=ComboWidget(
                        (
                            "auto",
                            "off",
                            "2",
                            "3",
                            "4",
                            "5",
                            "tree2s1",
                            "tree2s2",
                            "tree3s1",
                            "tree4s1m8",
                        )
                    ),
                    doc=(
                        "Speculative decoding with the checkpoint's multi-token-prediction head. "
                        "No effect without MTP weights. auto adapts the draft depth and breadth "
                        "to observed token matches and execution cost; 2-5 pins a constant draft "
                        "depth; tree modes pin both breadth and depth. Sampled output stays "
                        "correctly distributed but differs from non-MTP output for the same seed."
                    ),
                ),
                InputSpec(
                    "system_prompt",
                    _STRING,
                    required=False,
                    force_input=True,
                    widget=StringWidget(multiline=False),
                    doc=(
                        "Replaces the system prompt in the model's chat template. "
                        "Ignored when the default template is not used."
                    ),
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "sampling_mode",
                    (
                        DynamicComboOption(
                            "on",
                            (
                                InputSpec(
                                    "temperature",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=0.7,
                                    widget=NumberWidget(min=0.01, max=2.0, step=0.000001),
                                ),
                                InputSpec(
                                    "top_k",
                                    TypeExpr.concrete(CORE_INT),
                                    default=64,
                                    widget=NumberWidget(min=0, max=1000),
                                ),
                                InputSpec(
                                    "top_p",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=0.95,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "min_p",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=0.05,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                InputSpec(
                                    "repetition_penalty",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    default=1.05,
                                    widget=NumberWidget(min=0.0, max=5.0, step=0.01),
                                ),
                                InputSpec(
                                    "seed",
                                    TypeExpr.concrete(CORE_INT),
                                    default=0,
                                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                                ),
                                InputSpec(
                                    "presence_penalty",
                                    TypeExpr.concrete(CORE_FLOAT),
                                    required=False,
                                    default=0.0,
                                    widget=NumberWidget(min=0.0, max=5.0, step=0.01),
                                ),
                            ),
                        ),
                        DynamicComboOption("off"),
                    ),
                    display_name="Sampling Mode",
                ),
            ),
            input_order=(
                "clip",
                "prompt",
                "image",
                "video",
                "audio",
                "max_length",
                "sampling_mode",
                "thinking",
                "use_default_template",
                "mtp",
                "system_prompt",
            ),
            outputs=(
                OutputSpec("generated_text", _STRING, display_name="generated_text"),
                OutputSpec("thinking", _STRING, display_name="thinking"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        max_length: int,
        sampling_mode: str | Mapping[str, Any],
        image: object = None,
        thinking: bool = False,
        use_default_template: bool = True,
        video: object = None,
        audio: object = None,
        mtp: str = "auto",
        system_prompt: str = "",
        **inputs: Any,
    ) -> Mapping[str, object]:
        sampling = (
            dict(sampling_mode)
            if isinstance(sampling_mode, Mapping)
            else {"sampling_mode": sampling_mode}
        )
        sampling.update(
            {key.removeprefix("sampling_mode."): value for key, value in inputs.items()}
        )
        tokens = clip.tokenize(
            prompt,
            image=materialize_image(image) if image is not None else None,
            skip_template=not use_default_template,
            min_length=1,
            thinking=thinking,
            video=materialize_image(video) if video is not None else None,
            audio=materialize_audio(audio) if audio is not None else None,
            system_prompt=system_prompt if use_default_template else "",
        )
        generated_ids = clip.generate(
            tokens,
            do_sample=sampling.get("sampling_mode") == "on",
            max_length=max_length,
            temperature=sampling.get("temperature", 1.0),
            top_k=sampling.get("top_k", 50),
            top_p=sampling.get("top_p", 1.0),
            min_p=sampling.get("min_p", 0.0),
            repetition_penalty=sampling.get("repetition_penalty", 1.0),
            presence_penalty=sampling.get("presence_penalty", 0.0),
            seed=sampling.get("seed"),
            mtp={"off": False, "auto": True}.get(mtp, mtp),
        )
        generated_text = clip.decode(generated_ids)
        reasoning, separator, text = generated_text.partition("</think>")
        if not separator and not reasoning.lstrip().startswith("<think>"):
            if isinstance(tokens, dict):
                tokens = next(iter(cast("dict[str, Any]", tokens).values()))
            prompt_ids = [
                token[0]
                for token in cast("list[Any]", tokens[-1])
                if isinstance(token, tuple) and isinstance(token[0], int)
            ]
            if not prompt.rstrip().endswith("<think>") and not clip.decode(
                prompt_ids[-4:]
            ).rstrip().endswith("<think>"):
                reasoning, text = "", generated_text
        return cls.outputs(
            generated_text=text.strip(), thinking=reasoning.replace("<think>", "", 1).strip()
        )


LTX2_T2V_SYSTEM_PROMPT = """You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
#### Guidelines
- Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions, actions, camera movement, audio).
    - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
        - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
- Use active language: present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Maintain chronological flow: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested). Integrate sounds chronologically alongside actions. Be specific (e.g., "soft footsteps on tile"), not vague (e.g., "ambient sound is present").
- Speech (only when requested):
    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with voice characteristics (e.g., "The man says in an excited voice: 'You won't believe what I just saw!'").
    - Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
    - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

#### Important notes:
- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is requested.
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video generation.

#### Example
Input: "A woman at a coffee shop talking on the phone"
Output:
Style: realistic with cinematic lighting. In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone to her ear with the other. Ambient cafe sounds fill the space\u2014espresso machine hiss, quiet conversations, gentle clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd love to meet up this weekend. How about Saturday afternoon?' She laughs softly\u2014a genuine chuckle\u2014and shifts in her chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully, lowering the phone.
"""  # noqa: E501

LTX2_I2V_SYSTEM_PROMPT = """You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
You are a Creative Assistant writing concise, action-focused image-to-video prompts. Given an image (first frame) and user Raw Input Prompt, generate a prompt to guide video generation from that image.

#### Guidelines:
- Analyze the Image: Identify Subject, Setting, Elements, Style and Mood.
- Follow user Raw Input Prompt: Include all requested motion, actions, camera movements, audio, and details. If in conflict with the image, prioritize user request while maintaining visual consistency (describe transition from image to user's scene).
- Describe only changes from the image: Don't reiterate established visual details. Inaccurate descriptions may cause scene cuts.
- Active language: Use present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Chronological flow: Use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape throughout the prompt alongside actions\u2014NOT at the end. Align audio intensity with action tempo. Include natural background audio, ambient sounds, effects, speech or music (when requested). Be specific (e.g., "soft footsteps on tile") not vague (e.g., "ambient sound").
- Speech (only when requested): Provide exact words in quotes with character's visual/voice characteristics (e.g., "The tall man speaks in a low, gravelly voice"), language if not English and accent if relevant. If general conversation mentioned without text, generate contextual quoted dialogue. (i.e., "The man is talking" input -> the output should include exact spoken words, like: "The man is talking in an excited voice saying: 'You won't believe what I just saw!' His hands gesture expressively as he speaks, eyebrows raised with enthusiasm. The ambient sound of a quiet room underscores his animated speech.")
- Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit to avoid conflicts.
- Visual and audio only: Describe only what is seen and heard. NO smell, taste, or tactile sensations.
- Restrained language: Avoid dramatic terms. Use mild, natural, understated phrasing.

#### Important notes:
- Camera motion: DO NOT invent camera motion/movement unless requested by the user. Make sure to include camera motion only if specified in the input.
- Speech: DO NOT modify or alter the user's provided character dialogue in the prompt, unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Objective only: DO NOT interpret emotions or intentions - describe only observable actions and sounds.
- Format: DO NOT use phrases like "The scene opens with..." / "The video starts...". Start directly with Style (optional) and chronological scene description.
- Format: Never start output with punctuation marks or special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- Your performance is CRITICAL. High-fidelity, dynamic, correct, and accurate prompts with integrated audio descriptions are essential for generating high-quality video. Your goal is flawless execution of these rules.

#### Output Format (Strict):
- Single concise paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

#### Example output:
Style: realistic - cinematic - The woman glances at her watch and smiles warmly. She speaks in a cheerful, friendly voice, "I think we're right on time!" In the background, a caf\u00e9 barista prepares drinks at the counter. The barista calls out in a clear, upbeat tone, "Two cappuccinos ready!" The sound of the espresso machine hissing softly blends with gentle background chatter and the light clinking of cups on saucers.
"""  # noqa: E501

LTX24_T2V_SYSTEM_PROMPT = """You are given a user's short text-to-video request. Write a single, highly detailed audio-visual caption describing the video that best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens\u2026", "We see\u2026", "There is\u2026".

2. Objective, observable description only. Do not infer emotions or intentions \u2014 describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes \u2014 apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up)
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good \u2014 match the user if they specified it, otherwise choose the treatment that best presents the requested scene.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle).
   Express these as flowing prose: "a medium shot frames\u2026, captured from a front-facing angle as the camera slowly pans\u2026". Never as "medium shot, static camera \u2014".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially\u2026", "A moment later\u2026", "Simultaneously\u2026". Keep every stated action in motion.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless \u2014 include background elements, subtle movements, lighting, secondary sounds \u2014 detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150\u2013220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text \u2014 no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style): render the described scene with strong visual production value \u2014 cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") \u2014 describe how the exact requested scene LOOKS at its most visually striking, never adding new objects or actions. Keep everything else (framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
"""  # noqa: E501

LTX24_I2V_SYSTEM_PROMPT = """You are given a REFERENCE IMAGE (the exact first frame of the video) and a user's short image-to-video request. Write a single, highly detailed audio-visual caption describing the video that BEGINS from this exact reference image and best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

FIRST-FRAME / IMAGE GROUNDING (do this first): the opening of your caption must match the reference image exactly \u2014 same subject(s), identity, appearance, clothing, setting, lighting, and composition as shown. The video starts on this frame; describe it faithfully, then narrate chronologically as the user's requested action unfolds from it. Never contradict, replace, or invent things not consistent with the image. Single continuous take \u2014 no hard cuts.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens\u2026", "We see\u2026", "There is\u2026".

2. Objective, observable description only. Do not infer emotions or intentions \u2014 describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements \u2014 grounded in and consistent with the reference image. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes \u2014 apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up) \u2014 consistent with how the reference image is framed at the start.
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good \u2014 match the user if they specified it, otherwise choose the treatment that best presents the requested scene starting from this frame.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle) \u2014 matching the reference image's viewpoint at the opening.
   Express these as flowing prose: "a medium shot frames\u2026, captured from a front-facing angle as the camera slowly pans\u2026". Never as "medium shot, static camera \u2014".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially\u2026", "A moment later\u2026", "Simultaneously\u2026". Keep the user's requested motion/action central and in motion throughout.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless \u2014 include background elements, subtle movements, lighting, secondary sounds \u2014 detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150\u2013220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text \u2014 no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style or contradicting the reference image): render the described scene with strong visual production value \u2014 cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") \u2014 describe how the exact requested scene, starting from this frame, LOOKS at its most visually striking, never adding new objects or actions and never contradicting the first frame. Keep everything else (first-frame grounding, framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
"""  # noqa: E501


class TextGenerateLTX2Prompt(TextGenerate):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return replace(
            super().define_schema(),
            node_type="TextGenerateLTX2Prompt",
            display_name="Generate LTX2 Prompt",
            search_terms=("prompt enhance", "LLM", "gemma"),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        max_length: int,
        sampling_mode: str | Mapping[str, Any],
        image: object = None,
        thinking: bool = False,
        use_default_template: bool = True,
        video: object = None,
        audio: object = None,
        mtp: str = "auto",
        system_prompt: str = "",
        **inputs: Any,
    ) -> Mapping[str, object]:
        is_gemma4 = "gemma4" in getattr(clip.tokenizer, "clip_name", "")
        if is_gemma4:
            if image is not None:
                system = system_prompt.strip() or LTX24_I2V_SYSTEM_PROMPT.strip()
                user_text = f"User Raw Input Prompt: {prompt}."
            else:
                system = system_prompt.strip() or LTX24_T2V_SYSTEM_PROMPT.strip()
                user_text = f"user prompt: {prompt}"
            think_prefix = "<|think|>\n" if thinking else ""
            model_open = "" if thinking else "<|channel>final\n"
            media = "<|image><|image|><image|>\n\n" if image is not None else ""
            formatted_prompt = (
                f"<|turn>system\n{think_prefix}{system}<turn|>\n"
                f"<|turn>user\n{media}{user_text}<turn|>\n"
                f"<|turn>model\n{model_open}"
            )
        else:
            system = (
                system_prompt.strip()
                or (LTX2_I2V_SYSTEM_PROMPT if image is not None else LTX2_T2V_SYSTEM_PROMPT).strip()
            )
            media = "\n<image_soft_token>\n" if image is not None else ""
            formatted_prompt = (
                f"<start_of_turn>system\n{system}<end_of_turn>\n"
                f"<start_of_turn>user\n{media}\nUser Raw Input Prompt: {prompt}.<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        out = super().execute(
            clip=clip,
            prompt=formatted_prompt,
            max_length=max_length,
            sampling_mode=sampling_mode,
            image=image,
            thinking=thinking,
            use_default_template=use_default_template,
            video=video,
            audio=audio,
            mtp=mtp,
            **inputs,
        )
        return cls.outputs(generated_text=out["generated_text"] or prompt, thinking=out["thinking"])
