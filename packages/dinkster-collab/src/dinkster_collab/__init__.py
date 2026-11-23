"""dinkster-collab: server-ordered collaborative document sessions.

A separate additive surface beside the job queue (platform plan): the
server orders forward JSON patches, makes them replayable, and holds a
snapshot checkpoint; a host may have each patch judged before it is
ordered. See
sessions.py for the model and routes.py for the HTTP/WS contract.
"""

from .routes import SESSIONS_KEY, add_session_routes, install_session_extension
from .sessions import (
    DOCUMENT_KINDS,
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
    WriteCheck,
    WriteRefusedError,
    validate_patch,
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
    "SnapshotRequiredError",
    "StaleBaseError",
    "UnknownSessionError",
    "WriteCheck",
    "WriteRefusedError",
    "add_session_routes",
    "install_session_extension",
    "validate_patch",
]
