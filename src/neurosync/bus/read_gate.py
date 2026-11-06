"""Host-bound, opt-in read/use gate for a *future* StateBus transport.

This is a source-level enforcement seam, not an identity provider.  The hosting
broker must authenticate the caller, mint ``HostBoundContext`` from that
authenticated connection (never from model-supplied JSON), keep the SQLite
database inaccessible to callers, and implement ``authorize`` against an
independent, current grants/rights store.  A callback returning True in a unit
test is not evidence that any of those components exist.

``StateLedger.project`` remains a low-level trusted-local API and is deliberately
not made private by this facade.  Anyone with direct access to that database or
the Python object can bypass this class.  A real effect broker also needs an
atomic authorization/effect transaction or a short-lived, bound effect lease:
the final policy check and a handler call are not atomic across external
processes.  Do not expose this as a production multi-tenant service as-is.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Callable


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,95}$")
_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,127}$")
MAX_READ_VALUE_BYTES = 8192
_MISSING = object()


class ReadAccessDenied(PermissionError):
    """Policy denied the operation before data or a tool effect was returned."""


@dataclass(frozen=True)
class HostBoundContext:
    """Identity/scope asserted by a separately authenticated host broker.

    Constructing this Python object does NOT authenticate the named principal.
    """

    principal_id: str
    target_id: str
    session_id: str

    def __post_init__(self) -> None:
        for label in ("principal_id", "target_id", "session_id"):
            value = getattr(self, label)
            if type(value) is not str or not _ID.fullmatch(value):
                raise ValueError(f"invalid host-bound {label}")


@dataclass(frozen=True)
class AccessCheck:
    """A request to host policy, not a producer's authorization claim."""

    context: HostBoundContext
    operation: str  # read-pre, read-post, use
    state_key: str
    value: Any = None


class HostBoundReadGate:
    """Check current host policy on each read and immediately before tool use.

    A receiving model chooses only a state key; it does not choose the bound
    principal, recipient, session, policy function or effect handler.  The
    hosting application's transport integration must enforce that separation.
    The handler registry is copied and fixed when this facade is constructed
    by its host.  This does not make a handler's code or captured state
    immutable; production brokers must separately pin and isolate those.
    """

    def __init__(self, ledger: Any, *, context: HostBoundContext,
                 authorize: Callable[[AccessCheck], bool],
                 approved_handlers: Mapping[str, Callable[[Any], Any]] | None = None) -> None:
        if not callable(getattr(ledger, "project", None)):
            raise TypeError("a local ledger projection interface is required")
        if not isinstance(context, HostBoundContext):
            raise TypeError("a host-bound context is required")
        if not callable(authorize):
            raise TypeError("an independent host-policy callback is required")
        if approved_handlers is not None and not isinstance(approved_handlers, Mapping):
            raise TypeError("host-approved handler registry must be a mapping")
        handlers: dict[str, Callable[[Any], Any]] = {}
        for key, handler in (approved_handlers or {}).items():
            self._state_key(key)
            if not callable(handler):
                raise TypeError("host-approved handler must be callable")
            handlers[key] = handler
        self._ledger = ledger
        self._context = context
        self._authorize = authorize
        self._handlers = MappingProxyType(handlers)

    @staticmethod
    def _state_key(key: str) -> str:
        if type(key) is not str or not _KEY.fullmatch(key):
            raise ReadAccessDenied("invalid state key")
        return key

    def _check(self, operation: str, key: str, value: Any = None) -> None:
        # Only the literal singleton True is authorization.  A producer's
        # nonempty string, mapping, error message or truthy object is not.
        try:
            allowed = self._authorize(AccessCheck(
                self._context, operation, key, deepcopy(value)))
        except Exception as exc:
            raise ReadAccessDenied("host policy unavailable or denied") from exc
        if allowed is not True:
            raise ReadAccessDenied("host policy denied")

    def read(self, state_key: str) -> Any:
        """Return only one scoped, bounded value after two host checks.

        The second check catches revocation while the local projection was
        being read; the source-only facade cannot solve distributed TOCTOU.
        """
        key = self._state_key(state_key)
        self._check("read-pre", key)
        projection = self._ledger.project(self._context.target_id,
                                          self._context.session_id)
        if not isinstance(projection, dict):
            raise ReadAccessDenied("invalid ledger projection")
        value = projection.get(key, _MISSING)
        if value is _MISSING:
            raise ReadAccessDenied("state unavailable")
        try:
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ReadAccessDenied("state value not bounded JSON") from exc
        if len(payload) > MAX_READ_VALUE_BYTES:
            raise ReadAccessDenied("state value too large")
        self._check("read-post", key, value)
        return deepcopy(value)

    def use(self, state_key: str) -> Any:
        """Select a prebound host handler and reauthorize before its effect.

        The caller can select only the state key, never supply executable code
        at invocation time.  An initial use-policy check precedes the registry
        lookup so unauthorized callers cannot enumerate installed handlers by
        error differences.  The host must mint the registry outside model
        control; cross-process revocation during execution still needs atomic
        effect custody, which this local facade does not implement.
        """
        key = self._state_key(state_key)
        self._check("use", key)
        handler = self._handlers.get(key)
        if handler is None:
            raise ReadAccessDenied("no host-approved handler for state key")
        value = self.read(key)
        self._check("use", key, value)
        return handler(deepcopy(value))
