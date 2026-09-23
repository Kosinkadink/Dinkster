"""dinkster-collab: server-ordered collaborative document sessions.

A separate additive surface beside the job queue (platform plan): the
server orders forward JSON patches, makes them replayable, and holds a
snapshot checkpoint - it never applies or interprets a patch. See
sessions.py for the model and routes.py for the HTTP/WS contract.
"""

from .routes import SESSIONS_KEY, add_session_routes, install_session_extension
from .sessions import (
    PROTOCOL_VERSION,
    SESSION_ROLES,
    ActorPrincipalMismatchError,
    DocumentSession,
    InvalidSnapshotError,
    NoSessionOwnerError,
    ResyncRequiredError,
    SessionOp,
    SessionRoleError,
    SessionService,
    SnapshotRequiredError,
    StaleBaseError,
    UnknownSessionError,
    validate_patch,
)
from .snapshots import (
    DOCUMENT_KINDS,
    SnapshotValidatorRegistry,
    document_kind_wire,
    normalize_document_kind,
)
from .store import SessionStore

__all__ = [
    "DOCUMENT_KINDS",
    "PROTOCOL_VERSION",
    "SESSION_ROLES",
    "SESSIONS_KEY",
    "ActorPrincipalMismatchError",
    "DocumentSession",
    "InvalidSnapshotError",
    "NoSessionOwnerError",
    "ResyncRequiredError",
    "SessionOp",
    "SessionRoleError",
    "SessionService",
    "SessionStore",
    "SnapshotValidatorRegistry",
    "SnapshotRequiredError",
    "StaleBaseError",
    "UnknownSessionError",
    "add_session_routes",
    "document_kind_wire",
    "install_session_extension",
    "normalize_document_kind",
    "validate_patch",
]
