"""Peers as asset sources (DESIGN 3.12).

The client half of asset sharing is deliberately tiny: a peer's
``/assets/{digest}`` endpoint is just another place bytes might live, so
it plugs into the same provenance-guided fetch path as any mirror URL.
``peer_asset_sources`` builds the ``sources_for`` callback a
FetchingResolver wants, merging provenance leads (best-first: a canonical
mirror over a busy peer, or flip the order for LAN-first) with each peer's
digest URL. Verification stays where it always is - in the vault's
verifying writer - so a compromised or confused peer can waste bandwidth,
never plant bytes.

Trust: asset bytes are verified by digest, so these endpoints demand less
trust than cache sharing (no codec bytes are interpreted). The endpoint
list still describes machines you chose to talk to.
"""

from __future__ import annotations

from collections.abc import Sequence

from dinkster_assets import require_digest
from dinkster_assets.fetch import SourcesFor


def peer_asset_sources(
    endpoints: Sequence[str],
    *,
    provenance: SourcesFor | None = None,
    peers_first: bool = False,
) -> SourcesFor:
    """digest -> candidate URLs across provenance and peer instances.

    ``provenance`` is any digest->URLs callback (a ProvenanceStore's
    ``sources`` bound method fits). Peers come after provenance unless
    ``peers_first`` (the LAN-beats-internet deployment)."""
    cleaned = [endpoint.rstrip("/") for endpoint in endpoints]

    def sources_for(digest: str) -> list[str]:
        require_digest(digest)
        from_peers = [f"{endpoint}/assets/{digest}" for endpoint in cleaned]
        from_provenance = list(provenance(digest)) if provenance is not None else []
        ordered = from_peers + from_provenance if peers_first else from_provenance + from_peers
        seen = dict.fromkeys(ordered)
        return list(seen)

    return sources_for
