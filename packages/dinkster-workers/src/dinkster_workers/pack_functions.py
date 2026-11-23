"""Run a node's declared ``@pack_function`` for a button's ``Call`` step, inside the pack's worker.

The worker accepts only a function that a ``Call`` step of the named node type
declares, with the same writes or file name, and checks the arguments against
the function's annotations. The function runs on a thread under its declared
time limit. When the limit passes or the presser cancels, the worker answers
at once, tells the function through ``CallContext.cancelled``, and drops
whatever the function returns later. A thread cannot be stopped safely, so it
is left to finish.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

from dinkster_protocol.pack_functions import (
    FAILED_MESSAGE,
    PACK_FUNCTION_NOTE_MAX_CHARS,
    PACK_FUNCTION_PROGRESS_INTERVAL,
    TIMEOUT_MESSAGE,
    PackFunctionRefused,
    PackFunctionRequest,
    check_answer,
    check_file,
    declared_call,
    user_message,
)
from dinkster_schema import NodeSchema
from dinkster_schema.body import (
    CallContext,
    pack_function_mark,
    takes_call_context,
)
from dinkster_schema.body import (
    _parameters as declared_parameters,  # pyright: ignore[reportPrivateUsage]
)

from .manifest import resolve_entry

logger = logging.getLogger(__name__)

Send = Callable[[dict[str, object], Sequence[bytes]], Awaitable[None]]


def _declared_function(
    schemas: Mapping[str, NodeSchema], request: PackFunctionRequest
) -> Callable[..., object]:
    call = declared_call(schemas.get(request.node_type), request)
    if call is None:
        raise PackFunctionRefused(
            "not-declared",
            FAILED_MESSAGE,
            f"{request.node_type} declares no Call of {request.function} with these writes",
        )
    function = call.function
    if isinstance(function, str):
        module, _, attribute = function.rpartition(".")
        function = cast("Callable[..., object]", resolve_entry(f"{module}:{attribute}"))
    if pack_function_mark(function) is None:
        raise PackFunctionRefused(
            "not-declared", FAILED_MESSAGE, f"{request.function} is not a @pack_function"
        )
    return function


def _argument(value: object, annotation: object, parameter: str) -> object:
    """A worked-out argument as the parameter's annotation takes it, or a refusal."""
    if annotation is float and type(value) is int:
        return float(value)
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        (item,) = getattr(annotation, "__args__", (object,))
        if not isinstance(value, list):
            raise ValueError(f"parameter {parameter} takes a list")
        return [_argument(entry, item, parameter) for entry in cast("list[object]", value)]
    if annotation in (int, float, bool, str) and type(value) is annotation:
        return value
    if isinstance(annotation, type) and annotation.__name__ == "AssetRef":
        raise ValueError(
            f"parameter {parameter}: handing an asset to a pack function is not built yet"
        )
    raise ValueError(f"parameter {parameter} takes {getattr(annotation, '__name__', annotation)}")


def _arguments(function: Callable[..., object], args: tuple[object, ...]) -> list[object]:
    parameters = declared_parameters(function, pack=True)
    required = sum(1 for _, _, has_default in parameters if not has_default)
    if not required <= len(args) <= len(parameters):
        raise ValueError(f"takes {required} to {len(parameters)} arguments, given {len(args)}")
    return [
        _argument(value, annotation, name)
        for value, (name, annotation, _) in zip(args, parameters, strict=False)
    ]


async def run_pack_function(
    header: Mapping[str, Any],
    schemas: Mapping[str, NodeSchema],
    send: Send,
    cancelled: threading.Event,
) -> None:
    """Answer one ``packFunction`` request with one ``packFunctionResult``."""
    request_id = str(header.get("requestId", ""))
    reply: dict[str, object] = {"type": "packFunctionResult", "requestId": request_id}
    blobs: list[bytes] = []
    notes: list[str] = []
    loop = asyncio.get_running_loop()
    last_progress = [0.0]

    def progress(fraction: float, text: str) -> None:
        now = time.monotonic()
        if cancelled.is_set() or (
            now - last_progress[0] < PACK_FUNCTION_PROGRESS_INTERVAL and fraction < 1
        ):
            return
        last_progress[0] = now
        event: dict[str, object] = {
            "type": "packFunctionProgress",
            "requestId": request_id,
            "fraction": fraction,
            "text": text[:200],
        }
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(send(event, ())))

    def note(text: str) -> None:
        if not cancelled.is_set():
            notes.append(text[:PACK_FUNCTION_NOTE_MAX_CHARS])

    request: PackFunctionRequest | None = None
    try:
        request = PackFunctionRequest.from_wire(header.get("call"))
        function = _declared_function(schemas, request)
        try:
            arguments = _arguments(function, request.args)
        except ValueError as exc:
            raise PackFunctionRefused("wrong-arguments", FAILED_MESSAGE, str(exc)) from None
        if takes_call_context(function):
            arguments.insert(0, CallContext(progress, note, cancelled.is_set))
        mark = pack_function_mark(function)
        assert mark is not None
        running = asyncio.ensure_future(asyncio.to_thread(function, *arguments))
        running.add_done_callback(_retrieve)
        try:
            async with asyncio.timeout(mark.time_limit):
                answer = await asyncio.shield(running)
                if inspect.isawaitable(answer):
                    answer = await cast("Awaitable[object]", answer)
        except TimeoutError:
            cancelled.set()
            raise PackFunctionRefused(
                "timeout", TIMEOUT_MESSAGE, f"no answer within {mark.time_limit:g} seconds"
            ) from None
        if cancelled.is_set():
            return
        if request.download is not None:
            blobs.append(check_file(answer))
        else:
            reply["answer"] = check_answer(answer, request.writes)
        reply["notes"] = notes
    except asyncio.CancelledError:
        cancelled.set()
        raise
    except PackFunctionRefused as exc:
        reply.update(code=exc.code, message=exc.message)
        logger.warning(
            "pack function %s refused (%s): %s",
            request.function if request else "?",
            exc.code,
            exc.detail,
        )
    except Exception as exc:
        reply.update(code="raised", message=user_message(exc))
        logger.warning(
            "pack function %s raised", request.function if request else "?", exc_info=True
        )
    await send(reply, blobs)


def _retrieve(future: asyncio.Future[object]) -> None:
    if not future.cancelled():
        future.exception()
