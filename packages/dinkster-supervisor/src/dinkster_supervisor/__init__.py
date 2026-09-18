"""dinkster-supervisor: layer-0 process that owns the public port, spawns an
engine host behind it, and proxies once the engine is healthy. The
station (multi-install management) composes the same machinery: one
supervised engine port per configured install plus a management port."""

from .ingress import IngressMember, create_ingress_app
from .installs import (
    IngressDef,
    InstallDef,
    InstallsError,
    StationConfig,
    dump_installs,
    dump_station_config,
    load_installs,
    load_station_config,
    parse_installs,
    parse_station_config,
)
from .leases import LeaseConflictError, LeaseStore, LeaseStoreError
from .station import (
    ManagedInstall,
    Station,
    create_station_app,
    engine_command,
    start_station,
)
from .supervisor import (
    PROTOCOL_VERSION,
    STATUS_PATH,
    EngineLink,
    EngineProcess,
    create_supervisor_app,
)

__all__ = [
    "PROTOCOL_VERSION",
    "STATUS_PATH",
    "EngineLink",
    "EngineProcess",
    "InstallDef",
    "IngressDef",
    "IngressMember",
    "InstallsError",
    "LeaseConflictError",
    "LeaseStore",
    "LeaseStoreError",
    "ManagedInstall",
    "Station",
    "StationConfig",
    "create_ingress_app",
    "create_station_app",
    "create_supervisor_app",
    "dump_installs",
    "dump_station_config",
    "engine_command",
    "load_installs",
    "load_station_config",
    "parse_installs",
    "parse_station_config",
    "start_station",
]
