"""Making network calls that come back before anyone gives up.

IPv6 does not work on this connection, and Python tries it first. A host that
publishes both records costs a full connect timeout before falling back:
measured at 20.2 seconds against 0.20 with IPv6 skipped, on the same request
to the same server. That turned a weather lookup into something slower than
the model that asked for it.

The hosts that happen to be IPv4-only were fine, which is why this went
unnoticed until a dual-stack one arrived.

`ipv4_only()` filters the address list for the duration of a call. It patches
a module-level function, so the patch is counted: Telegram polls on its own
thread while skills make calls on another, and the first one to finish must
not restore the original under the other.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Iterator

_resolve = socket.getaddrinfo
_lock = threading.Lock()
_users = 0


def _ipv4(*args, **kwargs):
    resolved = _resolve(*args, **kwargs)
    only = [entry for entry in resolved if entry[0] == socket.AF_INET]
    # If a host is genuinely IPv6-only, filtering everything out would break
    # it entirely rather than making it slow.
    return only or resolved


@contextlib.contextmanager
def ipv4_only() -> Iterator[None]:
    """Skip IPv6 for calls made inside this block."""
    global _users
    with _lock:
        _users += 1
        socket.getaddrinfo = _ipv4
    try:
        yield
    finally:
        with _lock:
            _users -= 1
            if _users == 0:
                socket.getaddrinfo = _resolve
