"""Non-TCP wakeup sockets for the contained Windows worker's proactor loop."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import socket
import sys
import tempfile
import uuid


def unix_socketpair() -> tuple[socket.socket, socket.socket]:
    """Use Winsock AF_UNIX, which CPython 3.12 does not expose on Windows."""
    if sys.platform != "win32":
        return socket.socketpair()
    winsock = ctypes.WinDLL("ws2_32", use_last_error=True)

    class Address(ctypes.Structure):
        _fields_ = [("family", ctypes.c_ushort), ("path", ctypes.c_char * 108)]

    winsock.socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    winsock.socket.restype = ctypes.c_size_t
    winsock.accept.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
    winsock.accept.restype = ctypes.c_size_t
    for function in (winsock.bind, winsock.connect):
        function.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
    winsock.listen.argtypes = [ctypes.c_size_t, ctypes.c_int]
    winsock.closesocket.argtypes = [ctypes.c_size_t]
    invalid = ctypes.c_size_t(-1).value
    # MXC permits AF_UNIX only in its private temp directory. Keep the leaf short.
    path = os.path.join(tempfile.gettempdir(), "s" + uuid.uuid4().hex[:6])
    encoded = os.fsencode(path)
    detail = (
        f"AF_UNIX wakeup path={path!r}, encoded length={len(encoded)} (limit 107); "
        f"TEMP={os.environ.get('TEMP')!r}, TMP={os.environ.get('TMP')!r}, "
        f"contained LOCALAPPDATA={os.environ.get('LOCALAPPDATA')!r}"
    )
    if len(encoded) > 107:
        raise OSError(f"Windows sandbox temp path is too long: {detail}")
    address = Address(1, encoded)
    handles: list[int] = []
    bound = False

    def check(ok: bool, operation: str) -> None:
        if not ok:
            raise OSError(
                winsock.WSAGetLastError(), f"Windows sandbox {operation} failed: {detail}"
            )

    try:
        server = winsock.socket(1, socket.SOCK_STREAM, 0)
        check(server != invalid, "socket")
        handles.append(server)
        check(winsock.bind(server, ctypes.byref(address), ctypes.sizeof(address)) == 0, "bind")
        bound = True
        check(winsock.listen(server, 1) == 0, "listen")
        client = winsock.socket(1, socket.SOCK_STREAM, 0)
        check(client != invalid, "socket")
        handles.append(client)
        check(
            winsock.connect(client, ctypes.byref(address), ctypes.sizeof(address)) == 0, "connect"
        )
        accepted = winsock.accept(server, None, None)
        check(accepted != invalid, "accept")
        handles.append(accepted)
        left = socket.socket(1, socket.SOCK_STREAM, 0, fileno=client)
        handles.remove(client)
        try:
            right = socket.socket(1, socket.SOCK_STREAM, 0, fileno=accepted)
        except BaseException:
            left.close()
            raise
        handles.remove(accepted)
        return left, right
    finally:
        for handle in handles:
            winsock.closesocket(handle)
        if bound:
            os.unlink(path)


if sys.platform == "win32":

    class ContainedLoop(asyncio.ProactorEventLoop):
        _internal_fds: int = 0

        def _make_self_pipe(self) -> None:
            self._ssock, self._csock = unix_socketpair()
            self._ssock.setblocking(False)
            self._csock.setblocking(False)
            self._internal_fds += 1

    class ContainedPolicy(asyncio.WindowsProactorEventLoopPolicy):
        _loop_factory = ContainedLoop


def install_worker_loop_policy() -> None:
    """Pack-owned HTTP threads also create loops; every worker loop must avoid TCP."""
    if sys.platform == "win32":
        # Pack libraries call asyncio.run() without accepting a loop factory.
        asyncio.set_event_loop_policy(ContainedPolicy())  # pyright: ignore[reportDeprecated]


def worker_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return ContainedLoop()
    return asyncio.new_event_loop()


if __name__ == "__main__":
    try:
        left, right = unix_socketpair()
        with left, right:
            left.settimeout(2)
            right.settimeout(2)
            left.sendall(b"w")
            if right.recv(1) != b"w":
                raise OSError("Windows sandbox wakeup did not carry the expected bytes")
            right.sendall(b"r")
            if left.recv(1) != b"r":
                raise OSError("Windows sandbox wakeup reply did not carry the expected bytes")
    except OSError as exc:
        print(json.dumps({"format": "dinkster.windows-wakeup/1", "ok": False, "detail": str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps({"format": "dinkster.windows-wakeup/1", "ok": True}))
