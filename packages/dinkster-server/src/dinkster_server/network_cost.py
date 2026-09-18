from __future__ import annotations

import importlib
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal, Protocol, cast

NetworkCost = Literal["metered", "unmetered", "unknown"]
CommandRunner = Callable[[tuple[str, ...]], str | None]


class NWPath(Protocol):
    @property
    def is_expensive(self) -> bool: ...

    @property
    def is_constrained(self) -> bool: ...


NWPathReader = Callable[[], NWPath | None]


class _NetworkModule(Protocol):
    def nw_path_monitor_create(self) -> object: ...

    def nw_path_monitor_set_update_handler(
        self, monitor: object, handler: Callable[[object], None]
    ) -> None: ...

    def nw_path_monitor_set_queue(self, monitor: object, queue: object) -> None: ...

    def nw_path_monitor_start(self, monitor: object) -> None: ...

    def nw_path_monitor_cancel(self, monitor: object) -> None: ...

    def nw_path_is_expensive(self, path: object) -> bool: ...

    def nw_path_is_constrained(self, path: object) -> bool: ...


class _DispatchModule(Protocol):
    DISPATCH_QUEUE_SERIAL: object

    def dispatch_queue_create(self, label: bytes, attribute: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _MacOSPath:
    is_expensive: bool
    is_constrained: bool


def _run(command: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def detect_linux_network_cost(runner: CommandRunner = _run) -> NetworkCost:
    output = runner(
        (
            "nmcli",
            "--terse",
            "--fields",
            "GENERAL.STATE,GENERAL.METERED",
            "device",
            "show",
        )
    )
    if output is None:
        return "unknown"
    values: set[str] = set()
    for block in output.split("\n\n"):
        fields = dict(
            line.split(":", 1) for line in block.splitlines() if ":" in line and line.strip()
        )
        if fields.get("GENERAL.STATE", "").split(maxsplit=1)[0] != "100":
            continue
        metered = fields.get("GENERAL.METERED")
        if metered is not None:
            values.add(metered.strip().lower())
    if values & {"yes", "yes (guessed)"}:
        return "metered"
    if values & {"no", "no (guessed)"}:
        return "unmetered"
    return "unknown"


_WINDOWS_COMMAND = (
    "powershell.exe",
    "-NoLogo",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "[Windows.Networking.Connectivity.NetworkInformation,Windows.Networking.Connectivity,"
    "ContentType=WindowsRuntime] | Out-Null; "
    "$p=[Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile(); "
    "if ($null -ne $p) { $p.GetConnectionCost().NetworkCostType.ToString() }",
)


def detect_windows_network_cost(runner: CommandRunner = _run) -> NetworkCost:
    output = runner(_WINDOWS_COMMAND)
    if output is None:
        return "unknown"
    value = output.strip().lower()
    if value in {"fixed", "variable"}:
        return "metered"
    if value == "unrestricted":
        return "unmetered"
    return "unknown"


def _module(name: str) -> ModuleType:
    return importlib.import_module(name)


def _read_macos_nwpath() -> NWPath | None:
    try:
        network = cast(_NetworkModule, cast(object, _module("Network")))
        dispatch = cast(_DispatchModule, cast(object, _module("dispatch")))
        monitor = network.nw_path_monitor_create()
        queue = dispatch.dispatch_queue_create(
            b"com.dinkster.network-cost", dispatch.DISPATCH_QUEUE_SERIAL
        )
    except Exception:
        return None

    ready = threading.Event()
    result: list[NWPath] = []

    def updated(path: object) -> None:
        try:
            result.append(
                _MacOSPath(
                    is_expensive=bool(network.nw_path_is_expensive(path)),
                    is_constrained=bool(network.nw_path_is_constrained(path)),
                )
            )
        except BaseException:
            pass
        finally:
            ready.set()

    started = False
    try:
        network.nw_path_monitor_set_update_handler(monitor, updated)
        network.nw_path_monitor_set_queue(monitor, queue)
        network.nw_path_monitor_start(monitor)
        started = True
        if not ready.wait(timeout=3):
            return None
        return result[0] if result else None
    except Exception:
        return None
    finally:
        if started:
            try:
                network.nw_path_monitor_cancel(monitor)
            except Exception:
                pass


def detect_macos_network_cost(reader: NWPathReader | None = None) -> NetworkCost:
    try:
        path = (reader or _read_macos_nwpath)()
        if path is None:
            return "unknown"
        return "metered" if path.is_expensive or path.is_constrained else "unmetered"
    except Exception:
        return "unknown"


def detect_network_cost(
    *,
    runner: CommandRunner = _run,
    nwpath_reader: NWPathReader | None = None,
    platform_name: str | None = None,
) -> NetworkCost:
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name.startswith("linux"):
        return detect_linux_network_cost(runner)
    if platform_name == "win32":
        return detect_windows_network_cost(runner)
    if platform_name == "darwin":
        return detect_macos_network_cost(nwpath_reader)
    return "unknown"
