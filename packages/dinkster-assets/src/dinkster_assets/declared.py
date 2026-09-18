"""Worker-side declared-asset resolution: pack code reaching its own
``[[pack.assets]]`` declarations by pack-local id (DESIGN 3.12; the
controlnet-aux story's last leg).

Declaration, preflight, consent, and verified acquisition all live
host-side (pack_assets.py, acquire.py). What was missing is the read
end: node code that wants the bytes its manifest declared had to know
the digest and walk the raw resolver chain itself. :func:`declared_asset`
closes that: pack code names the declaration's stable pack-local id and
gets back a resolver-bound :class:`AssetRef` - or a loud AssetError that
says exactly which contract was not met. It NEVER downloads: execution
reads what consented acquisition already landed, nothing else.

Attribution is explicit and context-local. A host may install several
pack tables, but pack load and invocation run under exactly one active
pack context. Tables are never merged, so another pack's local ids are
not visible, and reads outside a known active context refuse loudly.

:func:`resolver_from_env` is the one place the worker-side store chain
is assembled from the environment contract (``DINKSTER_ASSET_VAULT``,
``DINKSTER_MOUNTS_SNAPSHOT``, ``DINKSTER_ASSET_ROOT``) - shared by the compat
pack's type registration and the worker host, so the chain's composition
can never drift between the two.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from .fetch import ChainResolver
from .identity import AssetError
from .library import IndexedAssetResolver
from .model import AssetRef, AssetResolver
from .mounts import MountSnapshotResolver
from .pack_assets import DeclaredAsset
from .vault import AssetVault

__all__ = [
    "clear_declared_assets",
    "declared_asset",
    "install_declared_assets",
    "resolver_from_env",
    "use_declared_asset_pack",
]

VAULT_ENV = "DINKSTER_ASSET_VAULT"
MOUNTS_ENV = "DINKSTER_MOUNTS_SNAPSHOT"
ROOT_ENV = "DINKSTER_ASSET_ROOT"


def produced_asset_vault(env: Mapping[str, str] | None = None) -> AssetVault | None:
    """Worker publications use writable scratch, not the sandbox's read-only input vault."""
    environment = os.environ if env is None else env
    scratch = environment.get("DINKSTER_PACK_SCRATCH")
    if scratch:
        return AssetVault(Path(scratch) / "produced-assets")
    root = environment.get(VAULT_ENV)
    return AssetVault(root) if root else None


def resolver_from_env(env: Mapping[str, str] | None = None) -> AssetResolver | None:
    """The process's asset store chain from its environment, cheapest
    first: verified vault CAS, then the live mounts snapshot, then the
    indexed library root. None when no store is configured (refs still
    flow; only reads need a store). Pass ``env`` explicitly in tests;
    production callers read the real environment."""
    if env is None:
        env = os.environ
    resolvers: list[AssetResolver] = []
    if env.get("DINKSTER_PACK_SCRATCH"):
        produced = produced_asset_vault(env)
        assert produced is not None
        resolvers.append(produced)
    vault_root = env.get(VAULT_ENV, "")
    if vault_root:
        resolvers.append(AssetVault(vault_root))
    snapshot = env.get(MOUNTS_ENV, "")
    if snapshot:
        # Re-reads the engine-published snapshot per resolve, so folders
        # granted after this process started materialize without restart.
        resolvers.append(MountSnapshotResolver(snapshot))
    root = env.get(ROOT_ENV, "")
    if root:
        try:
            resolvers.append(IndexedAssetResolver(root))
        except AssetError as error:
            # The indexed root is one advisory store among several, and
            # EVERY worker assembles this chain at startup now - a stale
            # or unscanned library must not stop a pack that never reads
            # assets from loading. Reads that needed it still fail loudly
            # ("not acquired"/"not materializable") at the read site.
            logging.getLogger("dinkster.assets").warning(
                "ignoring unreadable asset root %s: %s", root, error
            )
    if not resolvers:
        return None
    return resolvers[0] if len(resolvers) == 1 else ChainResolver(*resolvers)


@dataclass(frozen=True)
class _DeclaredTable:
    pack: str
    assets: Mapping[str, DeclaredAsset]
    resolver: AssetResolver | None


_installed: dict[str, _DeclaredTable] = {}
_active_pack: ContextVar[str | None] = ContextVar("dinkster_declared_asset_pack", default=None)


def install_declared_assets(
    pack: str,
    assets: Iterable[DeclaredAsset],
    resolver: AssetResolver | None,
) -> None:
    """Install one hosted pack's table before its entry code runs."""
    _installed[pack] = _DeclaredTable(
        pack=pack,
        assets={asset.id: asset for asset in assets},
        resolver=resolver,
    )


@contextmanager
def use_declared_asset_pack(pack: str) -> Generator[None]:
    """Attribute declared-asset reads to exactly one hosted pack."""
    if pack not in _installed:
        raise AssetError(f"unknown declared-asset pack context {pack!r}")
    token = _active_pack.set(pack)
    try:
        yield
    finally:
        _active_pack.reset(token)


def clear_declared_assets() -> None:
    """Remove the installed table (test hygiene; production workers never
    clear - the table lives as long as the process)."""
    _installed.clear()
    _active_pack.set(None)


def declared_asset(asset_id: str) -> AssetRef:
    """Resolve one of the hosting pack's ``[[pack.assets]]`` declarations
    to a resolver-bound :class:`AssetRef` whose bytes are readable NOW.

    Fails loudly - never downloads, never guesses:

    - no table installed: this code is not running inside a Dinkster worker
      (or the host predates the hook);
    - unknown id: the manifest does not declare it (declared ids are
      listed in the error - the id is the contract, typos surface here);
    - no store configured: the worker has nowhere to read bytes from;
    - not acquired: the declaration exists but consented acquisition has
      not landed the bytes on this machine - the error names the digest
      so it correlates with the preflight plan that offers it.
    """
    pack = _active_pack.get()
    if pack is None:
        raise AssetError(
            f"declared asset {asset_id!r} requested, but no declaration "
            "table is installed in this process - declared_asset() only "
            "works inside a Dinkster worker hosting the declaring pack"
        )
    table = _installed.get(pack)
    if table is None:
        raise AssetError(f"unknown declared-asset pack context {pack!r}")
    declared = table.assets.get(asset_id)
    if declared is None:
        known = ", ".join(sorted(table.assets)) or "none"
        raise AssetError(
            f"pack {table.pack!r} declares no asset {asset_id!r} in its "
            f"[[pack.assets]] (declared ids: {known})"
        )
    need = declared.need
    if table.resolver is None:
        raise AssetError(
            f"declared asset {asset_id!r} ({need.digest}) cannot be read: "
            "this worker has no asset store configured (no "
            f"{VAULT_ENV}/{MOUNTS_ENV}/{ROOT_ENV})"
        )
    path = table.resolver.resolve(need.digest)
    if path is None:
        raise AssetError(
            f"declared asset {asset_id!r} ({need.digest}) is not acquired "
            f"on this machine; pack {table.pack!r} declares where it can "
            "come from, but assets never download during execution - "
            "acquire it through job preflight consent (acquireAssets) or "
            "the assets API, then rerun"
        )
    size = need.size if need.size >= 0 else path.stat().st_size
    return AssetRef(
        digest=need.digest,
        name=need.name or asset_id,
        size=size,
        media_type=need.media_type or "application/octet-stream",
        resolver=table.resolver,
    )
