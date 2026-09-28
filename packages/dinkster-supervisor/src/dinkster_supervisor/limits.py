"""Supervisor ingress limits that protect the version-independent proxy process.

Buffered job submissions and accepted responses are bounded so routing facts
can be validated without unbounded memory. WebSocket frames and the fan-out
queue are bounded to protect event forwarding. ``client_max_size`` protects
supervisor-owned buffered endpoints; proxied uploads stream through
``request.content`` and are governed by the destination engine instead.
"""

MEBIBYTE = 1024 * 1024

INGRESS_CLIENT_MAX_SIZE_BYTES = 16 * MEBIBYTE
INGRESS_JOB_SUBMISSION_LIMIT_BYTES = 4 * MEBIBYTE
INGRESS_EVENT_FRAME_LIMIT_BYTES = 4 * MEBIBYTE
INGRESS_EVENT_QUEUE_LIMIT_BYTES = 8 * MEBIBYTE
