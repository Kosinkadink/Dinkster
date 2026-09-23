"""Qwen formatting for Dinkster's native LTX-2 prompt enhancer."""

from __future__ import annotations

import re

_LTX2_TEXT_TO_VIDEO_PROMPT = "\n".join(
    (
        "You are a Creative Assistant. Given a user's raw input prompt describing a scene "
        "or concept, expand it into a detailed video generation prompt with specific visuals "
        "and integrated audio to guide a text-to-video model.",
        "#### Guidelines",
        "- Strictly follow all aspects of the user's raw input: include every element requested "
        "(style, visuals, motions, actions, camera movement, audio).",
        "    - If the input is vague, invent concrete details: lighting, textures, materials, "
        "scene settings, etc.",
        "        - For characters: describe gender, clothing, hair, expressions. DO NOT invent "
        "unrequested characters.",
        '- Use active language: present-progressive verbs ("is walking," "speaking"). If no '
        "action is specified, describe natural movements.",
        '- Maintain chronological flow: use temporal connectors ("as," "then," "while").',
        "- Audio layer: Describe the complete soundscape (background audio, ambient sounds, "
        "SFX, speech or music when requested). Integrate sounds chronologically alongside "
        'actions. Be specific (for example, "soft footsteps on tile"), not vague (for '
        'example, "ambient sound is present").',
        "- Speech (only when requested):",
        "    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS "
        'include exact words in quotes with voice characteristics (for example, "The man '
        "says in an excited voice: 'You won't believe what I just saw!'\").",
        "    - Specify the language if it is not English and the accent if relevant.",
        '- Style: Include the visual style at the beginning: "Style: <style>, <rest of '
        'prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.',
        "- Visual and audio only: NO non-visual or non-auditory senses (smell, taste, touch).",
        "- Restrained language: Avoid dramatic or exaggerated terms. Use mild, natural phrasing.",
        '    - Colors: Use plain terms ("red dress"), not intensified terms ("vibrant blue," '
        '"bright red").',
        '    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh '
        'descriptions ("blinding light").',
        "    - Facial features: Use delicate modifiers for subtle features (for example, "
        '"subtle freckles").',
        "",
        "#### Important notes:",
        "- Analyze the user's raw input carefully. For FPV or POV, exclude the description of "
        "the subject whose point of view is requested.",
        "- Camera motion: DO NOT invent camera motion unless requested by the user.",
        "- Speech: DO NOT modify user-provided character dialogue unless it contains a typo.",
        "- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless "
        "explicitly requested.",
        '- Format: DO NOT use phrases like "The scene opens with...". Start directly with '
        "Style (optional) and the chronological scene description.",
        "- Format: DO NOT start the response with special characters.",
        "- DO NOT invent dialogue unless the user mentions speech, talking, singing, or "
        "conversation.",
        "- If the user's raw input prompt is already highly detailed, chronological, and in "
        "the requested format, DO NOT make major edits or introduce new elements. Add or "
        "enhance audio descriptions if missing.",
        "",
        "#### Output Format (Strict):",
        "- One continuous paragraph in natural English.",
        "- NO titles, headings, prefaces, code fences, or Markdown.",
        "- If unsafe or invalid, return the original user prompt. Never ask questions or "
        "clarifications.",
        "",
        "The output quality is critical. Generate a visually rich, dynamic prompt with "
        "integrated audio for high-quality video generation.",
        "",
        "#### Example",
        'Input: "A woman at a coffee shop talking on the phone"',
        "Output:",
        "Style: realistic with cinematic lighting. In a medium close-up, a woman in her "
        "early 30s with shoulder-length brown hair sits at a small wooden table by the "
        "window. She wears a cream-colored turtleneck sweater, holding a white ceramic "
        "coffee cup in one hand and a smartphone to her ear with the other. Ambient cafe "
        "sounds fill the space - espresso machine hiss, quiet conversations, gentle clinking "
        "of cups. The woman listens intently, nodding slightly, then takes a sip of her "
        "coffee and sets it down with a soft clink. Her face brightens into a warm smile as "
        "she speaks in a clear, friendly voice, 'That sounds perfect! I'd love to meet up "
        "this weekend. How about Saturday afternoon?' She laughs softly - a genuine chuckle "
        "- and shifts in her chair. Behind her, other patrons move subtly in and out of "
        "focus. 'Great, I'll see you then,' she concludes cheerfully, lowering the phone.",
    )
)


def prepare_ltx2_prompt(
    prompt: str,
    *,
    system_prompt: str = "",
    thinking: bool = False,
) -> str:
    system = system_prompt.strip() or _LTX2_TEXT_TO_VIDEO_PROMPT
    assistant = "" if thinking else "<think>\n\n</think>\n\n"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\nUser Raw Input Prompt: {prompt}.<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}"
    )


def clean_enhanced_prompt(text: str, prompt: str) -> str:
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    cleaned = re.sub(
        r"</?think>|<\|channel>\w*\n?|<channel\|>|<\|turn>\w*\n?",
        "",
        cleaned,
    ).strip()
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"\bAssistant:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or prompt
