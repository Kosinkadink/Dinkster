"""Catalog-driven remote nodes served through the Dinkster gateway."""

from dinkster_api.v1 import TypeRegistry

from .runtime import RemoteConfig, RemoteRuntime, build_node_classes

REMOTE_RUNTIME = RemoteRuntime(RemoteConfig.from_env())
REMOTE_NODES = build_node_classes(REMOTE_RUNTIME)


def register_remote_types(registry: TypeRegistry) -> None:
    REMOTE_RUNTIME.register_types(registry)


async def wait_for_schema_reload() -> None:
    await REMOTE_RUNTIME.wait_for_schema_reload()


__all__ = ["REMOTE_NODES", "register_remote_types", "wait_for_schema_reload"]
