"""Pure request planning and response projection for pinned Kling nodes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypeGuard

from .helper_registry import HelperRefusal, HelperRegistry


def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _require_text(value: object, name: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise HelperRefusal(f"Field {name!r} cannot be empty.")
    stripped = value.strip()
    if not allow_empty and not stripped:
        raise HelperRefusal(
            f"Field {name!r} cannot be shorter than 1 characters; was 0 characters long."
        )
    if len(stripped) > maximum:
        raise HelperRefusal(
            f" Field '{name} cannot be longer than {maximum} characters; "
            f"was {len(stripped)} characters long."
        )
    return value


def _legacy_text(payload: Mapping[str, object]) -> Mapping[str, object]:
    prompt = payload.get("prompt")
    negative = payload.get("negative_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise HelperRefusal("Positive prompt is empty")
    if len(prompt) > 2500:
        raise HelperRefusal(f"Positive prompt is too long: {len(prompt)} characters")
    if isinstance(negative, str) and len(negative) > 2500:
        raise HelperRefusal(f"Negative prompt is too long: {len(negative)} characters")
    mode = payload.get("mode")
    modes = payload.get("modes")
    if not isinstance(mode, str) or not _is_mapping(modes) or mode not in modes:
        raise HelperRefusal(f"unknown Kling mode {mode!r}")
    selected = modes[mode]
    if not _is_list(selected) or len(selected) != 3:
        raise HelperRefusal("invalid Kling mode table")
    return {"mode": selected[0], "duration": selected[1], "model_name": selected[2]}


def _legacy_image(payload: Mapping[str, object]) -> Mapping[str, object]:
    prompt = payload.get("prompt")
    negative = payload.get("negative_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise HelperRefusal("Positive prompt is empty")
    if len(prompt) > 500:
        raise HelperRefusal(f"Positive prompt is too long: {len(prompt)} characters")
    if isinstance(negative, str) and len(negative) > 500:
        raise HelperRefusal(f"Negative prompt is too long: {len(negative)} characters")
    mode = payload.get("mode")
    if mode == "std" and payload.get("model_name") == "kling-v2-5-turbo":
        mode = "pro"
    camera = payload.get("camera_control")
    if _is_mapping(camera):
        camera = {**camera, "type": "simple"}
    return {"mode": mode, "camera_control": camera}


def _legacy_extend(payload: Mapping[str, object]) -> Mapping[str, object]:
    _legacy_text({**payload, "modes": {"x": ["x", "x", "x"]}, "mode": "x"})
    return {
        "prompt": payload.get("prompt") or None,
        "negative_prompt": payload.get("negative_prompt") or None,
    }


def _single_effect(payload: Mapping[str, object]) -> Mapping[str, object]:
    if payload.get("duration") != "5":
        raise HelperRefusal("Input should be '5'")
    return {
        "input": {
            "model_name": payload.get("model_name"),
            "image": payload.get("image"),
            "duration": payload.get("duration"),
        }
    }


def _dual_effect(payload: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "input": {
            "model_name": payload.get("model_name"),
            "mode": payload.get("mode"),
            "images": payload.get("images"),
            "duration": payload.get("duration"),
        }
    }


def _lip_audio(payload: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "input": {
            "video_url": payload.get("video_url"),
            "mode": "audio2video",
            "voice_language": payload.get("voice_language"),
            "audio_type": "url",
            "audio_url": payload.get("audio_url"),
        }
    }


def _lip_text(payload: Mapping[str, object]) -> Mapping[str, object]:
    text = _require_text(payload.get("text"), "Text", 120, allow_empty=True)
    voices = payload.get("voices")
    voice = payload.get("voice")
    if not isinstance(voice, str) or not _is_mapping(voices) or voice not in voices:
        raise HelperRefusal(f"unknown voice {voice!r}")
    selected = voices[voice]
    if not _is_list(selected) or len(selected) != 2:
        raise HelperRefusal("invalid voice table")
    return {
        "input": {
            "video_url": payload.get("video_url"),
            "mode": "text2video",
            "text": text,
            "voice_language": selected[1],
            "voice_speed": payload.get("voice_speed"),
            "audio_type": "url",
            "voice_id": selected[0],
        }
    }


def _image_generation(payload: Mapping[str, object]) -> Mapping[str, object]:
    _require_text(payload.get("prompt"), "prompt", 500)
    negative_prompt = _require_text(
        payload.get("negative_prompt"), "negative_prompt", 500, allow_empty=True
    )
    if len(negative_prompt) > 200:
        raise HelperRefusal("String should have at most 200 characters")
    present = payload.get("image_present") is True
    return {
        "image_reference": payload.get("image_type") if present else None,
    }


def _normalize_prompt(prompt: object) -> str:
    if not isinstance(prompt, str):
        raise HelperRefusal("Field 'prompt' cannot be empty.")
    prompt = re.sub(
        r"(?<!\w)@image(?P<idx>\d*)(?!\w)",
        lambda match: f"<<<image_{match.group('idx') or '1'}>>>",
        prompt,
    )
    return re.sub(
        r"(?<!\w)@video(?P<idx>\d*)(?!\w)",
        lambda match: f"<<<video_{match.group('idx') or '1'}>>>",
        prompt,
    )


def _storyboards(
    value: object, duration: object
) -> tuple[bool | None, list[dict[str, object]] | None]:
    if value is None:
        return None, None
    if not _is_mapping(value) or value.get("storyboards") == "disabled":
        return None, None
    label = value.get("storyboards")
    if not isinstance(label, str) or not label[:1].isdigit():
        raise HelperRefusal("invalid storyboards selection")
    count = int(label.split()[0])
    items: list[dict[str, object]] = []
    for index in range(1, count + 1):
        prompt = _require_text(
            value.get(f"storyboard_{index}_prompt"), f"storyboard_{index}_prompt", 512
        )
        item_duration = value.get(f"storyboard_{index}_duration")
        items.append({"index": index, "prompt": prompt, "duration": str(item_duration)})
    total = sum(int(str(item["duration"])) for item in items)
    if duration is not None and total != duration:
        raise HelperRefusal(
            f"Total storyboard duration ({total}s) must equal the global duration ({duration}s)."
        )
    return True, items


def _mode(resolution: object) -> str:
    return "4k" if resolution == "4k" else "pro" if resolution == "1080p" else "std"


def _omni_common(
    payload: Mapping[str, object], *, normalize: bool, materialize_storyboards: bool = True
) -> dict[str, object]:
    model = payload.get("model_name")
    duration = payload.get("duration")
    resolution = payload.get("resolution", "1080p")
    generate_audio = payload.get("generate_audio") is True
    storyboard_value = payload.get("storyboards")
    stories_enabled = (
        _is_mapping(storyboard_value) and storyboard_value.get("storyboards") != "disabled"
    )
    if model == "kling-video-o1":
        if isinstance(duration, int) and duration > 10:
            raise HelperRefusal(
                "kling-video-o1 does not support durations greater than 10 seconds."
            )
        if generate_audio:
            raise HelperRefusal("kling-video-o1 does not support audio generation.")
        if resolution == "4k":
            raise HelperRefusal("kling-video-o1 does not support 4k resolution.")
        if stories_enabled:
            raise HelperRefusal("kling-video-o1 does not support storyboards.")
    prompt = _normalize_prompt(payload.get("prompt")) if normalize else payload.get("prompt")
    _require_text(prompt, "prompt", 2500, allow_empty=stories_enabled)
    stories, prompts = (
        _storyboards(storyboard_value, duration)
        if materialize_storyboards
        else (True if stories_enabled else None, None)
    )
    return {
        "prompt": prompt,
        "duration": str(duration) if duration is not None else None,
        "mode": _mode(resolution),
        "sound": "on" if generate_audio else "off",
        "multi_shot": stories,
        "multi_prompt": prompts,
        "shot_type": "customize" if stories else None,
    }


def _omni_text(payload: Mapping[str, object]) -> Mapping[str, object]:
    if payload.get("model_name") == "kling-video-o1" and payload.get("duration") not in (5, 10):
        raise HelperRefusal("kling-video-o1 only supports durations of 5 or 10 seconds.")
    return _omni_common(payload, normalize=False)


def _omni_first_last(payload: Mapping[str, object]) -> Mapping[str, object]:
    values = _omni_common(payload, normalize=True, materialize_storyboards=False)
    end_frame = payload.get("end_frame_present") is True
    references = payload.get("reference_images_present") is True
    if end_frame and references:
        raise HelperRefusal(
            "The 'end_frame' input cannot be used simultaneously with 'reference_images'."
        )
    if payload.get("end_frame_present") and values["multi_shot"]:
        raise HelperRefusal("The 'end_frame' input cannot be used simultaneously with storyboards.")
    if (
        payload.get("model_name") == "kling-video-o1"
        and payload.get("duration") not in (5, 10)
        and not end_frame
        and not references
    ):
        raise HelperRefusal(
            "Duration is only supported for 5 or 10 seconds if there is no end frame or "
            "reference images."
        )
    stories, prompts = _storyboards(payload.get("storyboards"), payload.get("duration"))
    values["multi_shot"] = stories
    values["multi_prompt"] = prompts
    values["shot_type"] = "customize" if stories else None
    values["image_list"] = payload.get("image_list")
    return values


def _omni_images(payload: Mapping[str, object]) -> Mapping[str, object]:
    values = _omni_common(payload, normalize=True)
    values["image_list"] = payload.get("list_reference_images")
    return values


def _omni_video(payload: Mapping[str, object]) -> Mapping[str, object]:
    images = payload.get("list_reference_images")
    prompt = _normalize_prompt(payload.get("prompt"))
    _require_text(prompt, "prompt", 2500)
    return {
        "prompt": prompt,
        "mode": _mode(payload.get("resolution")),
        "image_list": images if images else None,
        "video_list": payload.get("video_list"),
    }


def _omni_edit(payload: Mapping[str, object]) -> Mapping[str, object]:
    return _omni_video(payload)


def _omni_image(payload: Mapping[str, object]) -> Mapping[str, object]:
    model = payload.get("model_name")
    resolution = payload.get("resolution")
    series = payload.get("series_amount")
    if model == "kling-image-o1" and resolution == "4K":
        raise HelperRefusal("4K resolution is not supported for kling-image-o1 model.")
    prompt = _normalize_prompt(payload.get("prompt"))
    _require_text(prompt, "prompt", 2500)
    if model == "kling-image-o1" and series != "disabled":
        raise HelperRefusal("kling-image-o1 does not support series generation.")
    return {
        "prompt": prompt,
        "resolution": str(resolution).lower(),
        "result_type": "series" if series != "disabled" else None,
        "series_amount": int(str(series)) if series != "disabled" else None,
    }


def _audio_text(payload: Mapping[str, object]) -> Mapping[str, object]:
    _require_text(payload.get("prompt"), "prompt", 2500)
    return {"sound": "on" if payload.get("generate_audio") else "off"}


def _audio_image(payload: Mapping[str, object]) -> Mapping[str, object]:
    return _audio_text(payload)


def _motion(payload: Mapping[str, object]) -> Mapping[str, object]:
    _require_text(payload.get("prompt"), "prompt", 2500, allow_empty=True)
    orientation = payload.get("character_orientation")
    return {
        "variant": f"{orientation}-orientation",
        "values": {"max_duration": 10 if orientation == "image" else 30},
    }


def _video(payload: Mapping[str, object]) -> Mapping[str, object]:
    model = payload.get("model")
    multi = payload.get("multi_shot")
    if not _is_mapping(model) or not _is_mapping(multi):
        raise HelperRefusal("Kling model and multi_shot inputs must be objects")
    custom = multi.get("multi_shot") != "disabled"
    prompts, duration = None, multi.get("duration")
    if custom:
        _, prompts = _storyboards(
            {"storyboards": multi.get("multi_shot"), **multi},
            None,
        )
        duration = sum(int(str(item["duration"])) for item in prompts or [])
        if not 3 <= duration <= 15:
            raise HelperRefusal(
                f"Total storyboard duration ({duration}s) must be between 3 and 15 seconds."
            )
    else:
        _require_text(multi.get("prompt"), "prompt", 2500)
    turbo = model.get("model") == "kling-3.0-turbo"
    image = payload.get("start_frame_present") is True
    variant = ("turbo" if turbo else "v3") + ("-image" if image else "-text")
    prompt = multi.get("prompt")
    if turbo and custom:
        prompt = (
            "; ".join(
                f"shot {index}, {int(str(item['duration']))}, {item['prompt']}"
                for index, item in enumerate(prompts or [], 1)
            )
            + ";"
        )
    return {
        "variant": variant,
        "values": {
            "prompt": prompt,
            "negative_prompt": None if custom else multi.get("negative_prompt"),
            "duration": duration,
            "duration_string": str(duration),
            "resolution": model.get("resolution"),
            "aspect_ratio": model.get("aspect_ratio"),
            "model_name": model.get("model"),
            "mode": _mode(model.get("resolution")),
            "multi_shot": True if custom else None,
            "multi_prompt": prompts,
            "shot_type": "customize" if custom else None,
        },
    }


def _first_last(payload: Mapping[str, object]) -> Mapping[str, object]:
    _require_text(payload.get("prompt"), "prompt", 2500)
    model = payload.get("model")
    if not _is_mapping(model):
        raise HelperRefusal("Kling model input must be an object")
    return {"mode": _mode(model.get("resolution")), "model_name": model.get("model")}


def _avatar(payload: Mapping[str, object]) -> Mapping[str, object]:
    return {"prompt": payload.get("prompt") or None}


def _task_creation(payload: Mapping[str, object]) -> Mapping[str, object]:
    return {"task_id": {"value": _task_id(payload)}}


def _task_id(payload: Mapping[str, object]) -> object:
    response = payload.get("response")
    if not _is_mapping(response):
        raise HelperRefusal("Kling initial request failed")
    data = response.get("data")
    task_id = data.get("task_id") if _is_mapping(data) else None
    if not task_id:
        raise HelperRefusal(
            f"Kling initial request failed. Code: {response.get('code')}, "
            f"Message: {response.get('message')}, Data: {data}"
        )
    return task_id


def _video_result(payload: Mapping[str, object]) -> Mapping[str, object]:
    response = payload.get("response")
    data = response.get("data") if _is_mapping(response) else None
    result = data.get("task_result") if _is_mapping(data) else None
    videos = result.get("videos") if _is_mapping(result) else None
    if not _is_list(videos) or not videos:
        task_id = data.get("task_id") if _is_mapping(data) else None
        raise HelperRefusal(f"Kling task {task_id} succeeded but no video data found in response.")
    video = videos[0]
    if not _is_mapping(video):
        task_id = data.get("task_id") if _is_mapping(data) else None
        raise HelperRefusal(f"Kling task {task_id} succeeded but no video data found in response.")
    return {"video": dict(video)}


def _image_result(payload: Mapping[str, object]) -> Mapping[str, object]:
    response = payload.get("response")
    data = response.get("data") if _is_mapping(response) else None
    result = data.get("task_result") if _is_mapping(data) else None
    images = result.get("images") if _is_mapping(result) else None
    if not _is_list(images) or not images:
        task_id = data.get("task_id") if _is_mapping(data) else None
        raise HelperRefusal(f"Kling task {task_id} succeeded but no image data found in response.")
    return {"images": images}


def _omni_creation(payload: Mapping[str, object]) -> Mapping[str, object]:
    response = payload.get("response")
    if not _is_mapping(response) or response.get("code"):
        raise HelperRefusal(
            "Kling request failed. Code: "
            f"{response.get('code') if _is_mapping(response) else None}, "
            f"Message: {response.get('message') if _is_mapping(response) else None}, "
            f"Data: {response.get('data') if _is_mapping(response) else None}"
        )
    return {"task_id": {"value": _task_id(payload)}}


def _omni_video_result(payload: Mapping[str, object]) -> Mapping[str, object]:
    return _video_result(payload)


def _omni_image_result(payload: Mapping[str, object]) -> Mapping[str, object]:
    response = payload.get("response")
    data = response.get("data") if _is_mapping(response) else None
    result = data.get("task_result") if _is_mapping(data) else None
    images = None
    if _is_mapping(result):
        images = result.get("series_images") or result.get("images")
    if not _is_list(images) or not images:
        raise HelperRefusal("Kling task succeeded but no image data found in response.")
    return {"images": images}


def _turbo_creation(payload: Mapping[str, object]) -> Mapping[str, object]:
    response = payload.get("response")
    data = response.get("data") if _is_mapping(response) else None
    task_id = data.get("id") if _is_mapping(data) else None
    if not task_id:
        raise HelperRefusal(
            "Kling 3.0 Turbo create failed. Code: "
            f"{response.get('code') if _is_mapping(response) else None}, "
            f"Message: {response.get('message') if _is_mapping(response) else None}"
        )
    return {"task_id": {"value": task_id}}


def _turbo_video_result(payload: Mapping[str, object]) -> Mapping[str, object]:
    response = payload.get("response")
    data = response.get("data") if _is_mapping(response) else None
    task = data[0] if _is_list(data) and data else None
    outputs = task.get("outputs") if _is_mapping(task) else None
    if _is_list(outputs):
        for output in outputs:
            if _is_mapping(output) and output.get("type") == "video" and output.get("url"):
                return {"video": dict(output)}
    raise HelperRefusal("Kling 3.0 Turbo task finished without a video output")


KLING_HELPERS = HelperRegistry(
    "kling",
    {
        "kling.legacy-video": {
            "text": _legacy_text,
            "image": _legacy_image,
            "extend": _legacy_extend,
        },
        "kling.video-effect": {"single": _single_effect, "dual": _dual_effect},
        "kling.lip-sync": {"audio": _lip_audio, "text": _lip_text},
        "kling.image-generation": {"prepare": _image_generation},
        "kling.omni-video": {
            "text": _omni_text,
            "first-last": _omni_first_last,
            "images": _omni_images,
            "video": _omni_video,
            "edit": _omni_edit,
        },
        "kling.omni-image": {"prepare": _omni_image},
        "kling.audio-video": {"text": _audio_text, "image": _audio_image},
        "kling.motion-control": {"prepare": _motion},
        "kling.video": {"prepare": _video},
        "kling.first-last-frame": {"prepare": _first_last},
        "kling.avatar": {"prepare": _avatar},
        "kling.task": {
            "creation": _task_creation,
            "video-result": _video_result,
            "image-result": _image_result,
            "omni-creation": _omni_creation,
            "omni-video-result": _omni_video_result,
            "omni-image-result": _omni_image_result,
            "turbo-creation": _turbo_creation,
            "turbo-video-result": _turbo_video_result,
        },
    },
)
