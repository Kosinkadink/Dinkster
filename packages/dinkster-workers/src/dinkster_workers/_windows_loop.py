"""Non-TCP wakeup sockets for the contained Windows worker's proactor loop."""

from __future__ import annotations

import asyncio
import ctypes
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
    if len(encoded) > 107:
        raise OSError("Windows sandbox temp path is too long for an AF_UNIX wakeup socket")
    address = Address(1, encoded)
    handles: list[int] = []
    bound = False

    def check(ok: bool) -> None:
        if not ok:
            raise OSError(winsock.WSAGetLastError(), "Windows sandbox AF_UNIX socket failed")

    try:
        server = winsock.socket(1, socket.SOCK_STREAM, 0)
        check(server != invalid)
        handles.append(server)
        check(winsock.bind(server, ctypes.byref(address), ctypes.sizeof(address)) == 0)
        bound = True
        check(winsock.listen(server, 1) == 0)
        client = winsock.socket(1, socket.SOCK_STREAM, 0)
        check(client != invalid)
        handles.append(client)
        check(winsock.connect(client, ctypes.byref(address), ctypes.sizeof(address)) == 0)
        accepted = winsock.accept(server, None, None)
        check(accepted != invalid)
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
        asyncio.set_event_loop_policy(ContainedPolicy())


def worker_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return ContainedLoop()
    return asyncio.new_event_loop()
