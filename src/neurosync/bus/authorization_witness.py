"""In-process reference contract for an *external* authorization event witness.

This module is intentionally NOT connected to HostOwnedEffectBroker. A real
implementation would need independent, durable custody and authenticated,
linearizable compare-and-append across a process/host failure. The reference
below proves only deterministic event-ordering mechanics in one Python object.
Neither the hash chain nor the class itself authenticates history or time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
import threading
from typing import Literal, Protocol


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,95}$")
_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^sha1:[0-9a-f]{40}$|^sha256:[0-9a-f]{64}$")
# Above 2**53, binary64 cannot represent every adjacent integer second.
# Reject such integer inputs instead of silently rounding across expiry.
_MAX_EXACT_EPOCH_INT = 2 ** 53


class WitnessDenied(RuntimeError):
    """No witness authorization event was committed; caller must fail closed."""


@dataclass(frozen=True)
class WitnessCursor:
    sequence: int
    event_sha256: str

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("invalid witness sequence")
        if type(self.event_sha256) is not str or not _HEX64.fullmatch(self.event_sha256):
            raise ValueError("invalid witness digest")


@dataclass(frozen=True)
class WitnessGrant:
    authority_id: str
    generation: int
    principal_id: str
    target_id: str
    session_id: str
    grant_id: str
    state_key: str
    plan_sha256: str
    planned_state_sha256: str
    handler_artifact_sha256: str
    handler_source_git_oid: str
    expires_at_epoch: float

    def __post_init__(self):
        for value in (self.authority_id, self.principal_id, self.target_id,
                      self.session_id, self.grant_id):
            if type(value) is not str or not _IDENTIFIER.fullmatch(value):
                raise ValueError("invalid witness grant identifier")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("invalid authority generation")
        if type(self.state_key) is not str or not _KEY.fullmatch(self.state_key):
            raise ValueError("invalid witness state key")
        for value in (self.plan_sha256, self.planned_state_sha256,
                      self.handler_artifact_sha256):
            if type(value) is not str or not _HEX64.fullmatch(value):
                raise ValueError("invalid witness SHA-256 binding")
        if (type(self.handler_source_git_oid) is not str or
                not _GIT_OID.fullmatch(self.handler_source_git_oid)):
            raise ValueError("invalid host-asserted source OID")
        if type(self.expires_at_epoch) not in (float, int):
            raise ValueError("finite witness grant expiry required")
        if (type(self.expires_at_epoch) is int and
                abs(self.expires_at_epoch) > _MAX_EXACT_EPOCH_INT):
            raise ValueError("witness grant expiry exceeds exact integer clock range")
        try:
            expiry = float(self.expires_at_epoch)
        except (OverflowError, ValueError) as exc:
            raise ValueError("finite witness grant expiry required") from exc
        if not math.isfinite(expiry):
            raise ValueError("finite witness grant expiry required")
        # Dataclass equality considers 300 == 300.0, so canonical event
        # serialization must never distinguish those equal grant bindings.
        object.__setattr__(self, "expires_at_epoch", expiry)

    @property
    def scoped_id(self) -> tuple[str, str, str, str]:
        return self.principal_id, self.target_id, self.session_id, self.grant_id


@dataclass(frozen=True)
class WitnessEvent:
    kind: Literal["clock", "issue", "claim", "revoke"]
    cursor: WitnessCursor
    observed_epoch: float
    grant_id: str | None
    invocation_id: str | None
    # Complete canonical JSON includes predecessor digest and full grant
    # binding, allowing offline reconstruction of this reference hash chain.
    canonical_payload_json: str
    evidence_class: str = "IN_PROCESS_SOURCE_REFERENCE_NOT_DURABLE"


class AuthorizationWitness(Protocol):
    """Host-supplied interface: a missing/untrusted instance is not optional."""

    @property
    def tip(self) -> WitnessCursor: ...
    def observe_time(self, expected: WitnessCursor, at_epoch: float) -> WitnessEvent: ...
    def issue(self, expected: WitnessCursor, grant: WitnessGrant, *, at_epoch: float) -> WitnessEvent: ...
    def reserve_claim(self, expected: WitnessCursor, grant: WitnessGrant,
                      invocation_id: str, *, at_epoch: float) -> WitnessEvent: ...
    def revoke(self, expected: WitnessCursor, grant: WitnessGrant, *, at_epoch: float) -> WitnessEvent: ...


@dataclass
class _GrantState:
    grant: WitnessGrant
    claimed_invocation_id: str | None = None
    revoked: bool = False


class InProcessAuthorizationWitness:
    """Deterministic comparison model, NOT an independent monotonic witness.

    The caller's main-DB cursor must match the current witness tip *for every
    event*. Witness acceptance linearizes before the caller's DB commit. If
    that commit later aborts, the old DB cursor is deliberately rejected on
    its next operation. No automatic rewind, reset or reconciliation exists.
    """

    def __init__(self, authority_id: str, generation: int):
        if type(authority_id) is not str or not _IDENTIFIER.fullmatch(authority_id):
            raise ValueError("invalid authority identifier")
        if type(generation) is not int or generation < 1:
            raise ValueError("invalid authority generation")
        self._authority_id = authority_id
        self._generation = generation
        genesis = sha256(
            f"neurosync:authorization-witness:genesis:{authority_id}:{generation}".encode("utf-8")
        ).hexdigest()
        self._tip = WitnessCursor(0, genesis)
        self._last_observed_epoch: float | None = None
        self._grants: dict[tuple[str, str, str, str], _GrantState] = {}
        # Grant IDs are not reusable, even in a different principal/session
        # scope; otherwise a copied main-db row could become ambiguous.
        self._issued_grant_ids: set[str] = set()
        self._events: list[WitnessEvent] = []
        self._lock = threading.RLock()

    def __copy__(self):
        raise WitnessDenied("live witness cannot be cloned")

    def __deepcopy__(self, memo):
        raise WitnessDenied("live witness cannot be cloned")

    def __getstate__(self):
        raise WitnessDenied("live witness cannot be serialized or restored")

    @property
    def tip(self) -> WitnessCursor:
        with self._lock:
            return self._tip

    @property
    def events(self) -> tuple[WitnessEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def _check(self, expected: WitnessCursor, at_epoch: float) -> float:
        if type(expected) is not WitnessCursor or expected != self._tip:
            raise WitnessDenied("authority history cursor is stale or mismatched")
        if type(at_epoch) not in (int, float):
            raise WitnessDenied("invalid host-observed clock")
        if type(at_epoch) is int and abs(at_epoch) > _MAX_EXACT_EPOCH_INT:
            raise WitnessDenied("host-observed clock exceeds exact integer range")
        try:
            observed = float(at_epoch)
        except (OverflowError, ValueError) as exc:
            raise WitnessDenied("invalid host-observed clock") from exc
        if not math.isfinite(observed):
            raise WitnessDenied("invalid host-observed clock")
        if self._last_observed_epoch is not None and observed < self._last_observed_epoch:
            raise WitnessDenied("observed clock moved backward")
        return observed

    def _grant_state(self, grant: WitnessGrant) -> _GrantState:
        if not isinstance(grant, WitnessGrant):
            raise WitnessDenied("witness grant binding required")
        if (grant.authority_id != self._authority_id or
                grant.generation != self._generation):
            raise WitnessDenied("authority generation or identifier mismatch")
        row = self._grants.get(grant.scoped_id)
        if row is None or row.grant != grant:
            raise WitnessDenied("unknown or mismatched grant binding")
        return row

    def _append(self, kind: Literal["clock", "issue", "claim", "revoke"],
                at_epoch: float, grant: WitnessGrant | None = None,
                invocation_id: str | None = None) -> WitnessEvent:
        sequence = self._tip.sequence + 1
        payload = {
            "authority_id": self._authority_id,
            "generation": self._generation,
            "sequence": sequence,
            "previous_sha256": self._tip.event_sha256,
            "kind": kind,
            "observed_epoch": at_epoch,
            "grant": asdict(grant) if grant is not None else None,
            "invocation_id": invocation_id,
        }
        canonical = json.dumps(
            payload, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        event = WitnessEvent(kind, WitnessCursor(sequence, digest), at_epoch,
                             grant.grant_id if grant else None, invocation_id,
                             canonical)
        self._tip = event.cursor
        self._last_observed_epoch = at_epoch
        self._events.append(event)
        return event

    def observe_time(self, expected: WitnessCursor, at_epoch: float) -> WitnessEvent:
        with self._lock:
            observed = self._check(expected, at_epoch)
            return self._append("clock", observed)

    def issue(self, expected: WitnessCursor, grant: WitnessGrant, *,
              at_epoch: float) -> WitnessEvent:
        with self._lock:
            observed = self._check(expected, at_epoch)
            if not isinstance(grant, WitnessGrant):
                raise WitnessDenied("witness grant binding required")
            if (grant.authority_id != self._authority_id or
                    grant.generation != self._generation):
                raise WitnessDenied("authority generation or identifier mismatch")
            if grant.scoped_id in self._grants or grant.grant_id in self._issued_grant_ids:
                raise WitnessDenied("grant ID already used in witness history")
            if grant.expires_at_epoch <= observed:
                raise WitnessDenied("grant already expired at observed clock")
            event = self._append("issue", observed, grant)
            self._grants[grant.scoped_id] = _GrantState(grant)
            self._issued_grant_ids.add(grant.grant_id)
            return event

    def reserve_claim(self, expected: WitnessCursor, grant: WitnessGrant,
                      invocation_id: str, *, at_epoch: float) -> WitnessEvent:
        with self._lock:
            observed = self._check(expected, at_epoch)
            if type(invocation_id) is not str or not _IDENTIFIER.fullmatch(invocation_id):
                raise WitnessDenied("invalid invocation identifier")
            row = self._grant_state(grant)
            if row.revoked or row.claimed_invocation_id is not None:
                raise WitnessDenied("grant revoked or already consumed")
            if grant.expires_at_epoch <= observed:
                raise WitnessDenied("grant expired")
            event = self._append("claim", observed, grant, invocation_id)
            row.claimed_invocation_id = invocation_id
            return event

    def revoke(self, expected: WitnessCursor, grant: WitnessGrant, *,
               at_epoch: float) -> WitnessEvent:
        with self._lock:
            observed = self._check(expected, at_epoch)
            row = self._grant_state(grant)
            if row.revoked:
                raise WitnessDenied("grant already revoked")
            event = self._append("revoke", observed, grant)
            row.revoked = True
            return event


def require_witness(witness: AuthorizationWitness | None) -> AuthorizationWitness:
    """Optional policy wiring must fail closed, never revert to v2 execution."""
    if witness is None:
        raise WitnessDenied("v3 authorization witness unavailable")
    try:
        if type(getattr(witness, "tip", None)) is not WitnessCursor:
            raise WitnessDenied("authorization witness contract unavailable")
        for name in ("observe_time", "issue", "reserve_claim", "revoke"):
            if not callable(getattr(witness, name, None)):
                raise WitnessDenied("authorization witness contract unavailable")
    except (AttributeError, TypeError) as exc:
        raise WitnessDenied("authorization witness contract unavailable") from exc
    return witness
