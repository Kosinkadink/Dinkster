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
import contextlib
import contextvars
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final, cast

from dinkster_assets import AssetRef
from dinkster_assets.privacy import check_secret_data, protect_secret_outputs
from dinkster_protocol.lookups import LookupCallError
from dinkster_protocol.pack_functions import (
    FAILED_MESSAGE,
    PACK_FUNCTION_NOTE_MAX_CHARS,
    PACK_FUNCTION_NOTES_MAX,
    PACK_FUNCTION_PROGRESS_INTERVAL,
    TIMEOUT_MESSAGE,
    PackFunctionRefused,
    PackFunctionRequest,
    check_answer,
    check_file,
    declared_call,
    secret_arguments,
    user_message,
)
from dinkster_schema import NodeSchema
from dinkster_schema.body import (
    CallContext,
    pack_function_mark,
    pack_function_parameters,
    takes_call_context,
)
from dinkster_schema.capture import install_stream_capture
from dinkster_schema.model import SECRET_NAME_PATTERN
from dinkster_schema.reporting import use_private_output

from .manifest import resolve_entry

logger = logging.getLogger(__name__)

Send = Callable[[dict[str, object], Sequence[bytes]], Awaitable[None]]

_THREADS: Final = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pack-function")
"""Pack functions run here, apart from the worker's shared pool: a function that
keeps running after Cancel or its time limit holds one of these threads only."""


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


def _argument(
    value: object, annotation: object, parameter: str, asset: Callable[[object], object]
) -> object:
    """A worked-out argument as the parameter's annotation takes it, or a refusal."""
    if annotation is AssetRef:
        if not isinstance(value, dict):
            raise ValueError(f"parameter {parameter} takes an asset")
        return asset(cast("dict[str, object]", value))
    if annotation is float and type(value) is int:
        try:
            return float(value)
        except OverflowError:
            raise ValueError(f"parameter {parameter}: {value} is too large for float") from None
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        (item,) = getattr(annotation, "__args__", (object,))
        if not isinstance(value, list):
            raise ValueError(f"parameter {parameter} takes a list")
        return [_argument(entry, item, parameter, asset) for entry in cast("list[object]", value)]
    if annotation in (int, float, bool, str) and type(value) is annotation:
        return value
    raise ValueError(f"parameter {parameter} takes {getattr(annotation, '__name__', annotation)}")


def _arguments(
    function: Callable[..., object], args: tuple[object, ...], asset: Callable[[object], object]
) -> list[object]:
    parameters = pack_function_parameters(function)
    required = sum(1 for _, _, has_default in parameters if not has_default)
    if not required <= len(args) <= len(parameters):
        raise ValueError(f"takes {required} to {len(parameters)} arguments, given {len(args)}")
    return [
        _argument(value, annotation, name, asset)
        for value, (name, annotation, _) in zip(args, parameters, strict=False)
    ]


async def run_pack_function(
    header: Mapping[str, Any],
    schemas: Mapping[str, NodeSchema],
    send: Send,
    cancelled: threading.Event,
    *,
    asset: Callable[[object], object],
    resolve_secret: Callable[[str, str], Awaitable[str | None]] | None = None,
) -> None:
    """Answer one ``packFunction`` request with one ``packFunctionResult``."""
    request_id = str(header.get("requestId", ""))
    reply: dict[str, object] = {"type": "packFunctionResult", "requestId": request_id}
    blobs: list[bytes] = []
    notes: list[str] = []
    loop = asyncio.get_running_loop()
    last_progress = [0.0]
    sending: set[asyncio.Future[None]] = set()
    private = False
    private_context = contextlib.ExitStack()

    def send_progress(event: dict[str, object]) -> None:
        sent = asyncio.ensure_future(send(event, ()))
        sending.add(sent)
        sent.add_done_callback(sending.discard)

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
            "text": "" if private else text[:200],
        }
        loop.call_soon_threadsafe(send_progress, event)

    def note(text: str) -> None:
        if not private and not cancelled.is_set() and len(notes) < PACK_FUNCTION_NOTES_MAX:
            notes.append(text[:PACK_FUNCTION_NOTE_MAX_CHARS])

    request: PackFunctionRequest | None = None
    running: asyncio.Future[object] | None = None
    try:
        request = PackFunctionRequest.from_wire(header.get("call"))
        positions = secret_arguments(schemas.get(request.node_type), request)
        private = bool(positions)
        install_stream_capture()
        private_context.enter_context(use_private_output(private))
        function = _declared_function(schemas, request)
        args = list(request.args)
        credentials: dict[str, str] = {}
        for position in sorted(positions):
            if resolve_secret is None or position >= len(args):
                raise PackFunctionRefused("failed", FAILED_MESSAGE)
            name = args[position]
            if type(name) is not str or SECRET_NAME_PATTERN.fullmatch(name) is None:
                raise PackFunctionRefused("failed", FAILED_MESSAGE)
            if name not in credentials:
                value = await resolve_secret(request_id, str(position))
                if type(value) is not str or not value:
                    raise PackFunctionRefused("failed", FAILED_MESSAGE)
                credentials[name] = value
                private_context.enter_context(protect_secret_outputs((value,)))
            args[position] = credentials[name]

        def prepare(answer: object) -> object:
            check_secret_data(answer)
            prepared = (
                check_file(answer)
                if request.download is not None
                else check_answer(answer, request.writes)
            )
            check_secret_data(prepared)
            return prepared

        def invoke() -> object:
            try:
                arguments = _arguments(function, tuple(args), asset)
            except (ValueError, LookupCallError) as exc:
                raise PackFunctionRefused("wrong-arguments", FAILED_MESSAGE, str(exc)) from None
            if cancelled.is_set():
                return None
            if takes_call_context(function):
                arguments.insert(0, CallContext(progress, note, cancelled.is_set))
            answer = function(*arguments)
            return answer if inspect.isawaitable(answer) else prepare(answer)

        mark = pack_function_mark(function)
        assert mark is not None
        running = loop.run_in_executor(_THREADS, contextvars.copy_context().run, invoke)
        running.add_done_callback(_retrieve)
        try:
            async with asyncio.timeout(mark.time_limit):
                answer = await asyncio.shield(running)
                if inspect.isawaitable(answer):
                    answer = prepare(await cast("Awaitable[object]", answer))
        except TimeoutError:
            cancelled.set()
            # A function still waiting for a thread never starts; a running one is told.
            running.cancel()
            raise PackFunctionRefused(
                "timeout", TIMEOUT_MESSAGE, f"no answer within {mark.time_limit:g} seconds"
            ) from None
        if cancelled.is_set():
            private_context.close()
            return
        if request.download is not None:
            blobs.append(cast("bytes", answer))
        else:
            reply["answer"] = answer
        reply["notes"] = notes
    except asyncio.CancelledError:
        cancelled.set()
        if running is not None:
            running.cancel()
        private_context.close()
        raise
    except PackFunctionRefused as exc:
        if private:
            reply.update(code="failed", message=FAILED_MESSAGE)
        else:
            reply.update(code=exc.code, message=exc.message)
            logger.warning(
                "pack function %s refused (%s): %s",
                request.function if request else "?",
                exc.code,
                exc.detail,
            )
    except Exception as exc:
        if private:
            reply.update(code="failed", message=FAILED_MESSAGE)
        else:
            reply.update(code="raised", message=user_message(exc))
            logger.warning(
                "pack function %s raised", request.function if request else "?", exc_info=True
            )
    try:
        await send(reply, blobs)
    except Exception:
        logger.warning("pack function answer could not be sent", exc_info=not private)
        await send(
            {
                "type": "packFunctionResult",
                "requestId": request_id,
                "code": "failed",
                "message": FAILED_MESSAGE,
            },
            (),
        )
    finally:
        private_context.close()


def _retrieve(future: asyncio.Future[object]) -> None:
    if not future.cancelled():
        future.exception()
