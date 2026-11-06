"""Local host-custodied capability execution with use-time grant serialization.

This is a *single-authority SQLite prototype*, not a network authentication
service. A host must keep both the database and this broker behind an
independently authenticated transport boundary. A same-UID caller with direct
database or handler access can bypass every check here.

All grant mutations through this authority serialize with execution via a
SQLite ``BEGIN IMMEDIATE`` transaction held across the handler call. An
invocation ID is durably consumed *before* invoking the handler; this prevents
replay after process restart, at the cost of a fail-closed/ambiguous receipt
if a process dies after consumption and before recording its effect. External
effects cannot be transactionally rolled back by SQLite. This broker promises
at-most-one *handler attempt* per invocation ID, never exactly-once delivery.
Handled projection/serialization failures before the handler are recorded as
denied without invoking it; a real process crash can still leave `claimed`.

The older ``HostBoundReadGate.use`` remains a cooperative, non-atomic source
seam; protected effects should go through this separate broker instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from types import MappingProxyType
from typing import Any, Callable

from .read_gate import HostBoundContext, ReadAccessDenied, MAX_READ_VALUE_BYTES


IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,95}$")
STATE_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,127}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^sha1:[0-9a-f]{40}$|^sha256:[0-9a-f]{64}$")


def _identifier(value: str) -> str:
    if type(value) is not str or not IDENTIFIER.fullmatch(value):
        raise ReadAccessDenied("invalid invocation or grant identifier")
    return value


def _key(value: str) -> str:
    if type(value) is not str or not STATE_KEY.fullmatch(value):
        raise ReadAccessDenied("invalid state key")
    return value


def _encoded(value: Any, *, limit: int = MAX_READ_VALUE_BYTES) -> bytes:
    try:
        result = json.dumps(value, ensure_ascii=False, allow_nan=False,
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ReadAccessDenied("effect value is not bounded JSON") from exc
    if len(result) > limit:
        raise ReadAccessDenied("effect value exceeds host limit")
    return result


def _hash(value: Any) -> str:
    return sha256(_encoded(value)).hexdigest()


@dataclass(frozen=True)
class ApprovedEffect:
    """Host-installed handler with a *host asserted* artifact/source binding."""

    handler: Callable[[Any], Any]
    handler_artifact_sha256: str
    source_git_oid: str

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise TypeError("host handler must be callable")
        if type(self.handler_artifact_sha256) is not str or not HEX64.fullmatch(self.handler_artifact_sha256):
            raise ValueError("invalid handler artifact SHA-256")
        if type(self.source_git_oid) is not str or not GIT_OID.fullmatch(self.source_git_oid):
            raise ValueError("invalid source commit OID")


class EffectAttemptFailed(RuntimeError):
    """Handler failed; invocation remains consumed, even if effects happened."""


class HostOwnedEffectBroker:
    """Local durable grants, revocation, and replay-safe handler attempts.

    All mutators must use the *same* SQLite database; arbitrary direct SQL
    edits or separate grant systems are explicitly outside the safety claim.
    The host (not a model message) creates ``context`` and ``approved_handlers``.
    """

    def __init__(self, database_path: str | Path, ledger: Any, *,
                 context: HostBoundContext,
                 approved_handlers: Mapping[str, ApprovedEffect],
                 clock: Callable[[], float] = time.time) -> None:
        if not callable(getattr(ledger, "project", None)):
            raise TypeError("host ledger projection required")
        if not isinstance(context, HostBoundContext):
            raise TypeError("host context required")
        if not isinstance(approved_handlers, Mapping):
            raise TypeError("approved handler mapping required")
        if not callable(clock):
            raise TypeError("host clock required")
        if (not isinstance(database_path, (str, Path)) or
                not str(database_path).strip() or
                str(database_path) == ":memory:" or
                str(database_path).startswith("file:") or
                Path(database_path).is_dir()):
            raise ValueError("persistent local authority DB file path required")
        handlers: dict[str, ApprovedEffect] = {}
        for key, effect in approved_handlers.items():
            _key(key)
            if not isinstance(effect, ApprovedEffect):
                raise TypeError("unapproved handler descriptor")
            handlers[key] = effect
        self._handlers = MappingProxyType(handlers)
        self._ledger = ledger
        self._context = context
        self._clock = clock
        self._path = str(database_path)
        self._in_handler = threading.local()
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS grants (
                    grant_id TEXT NOT NULL, principal TEXT NOT NULL,
                    target TEXT NOT NULL, session TEXT NOT NULL,
                    state_key TEXT NOT NULL, expires REAL NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)),
                    PRIMARY KEY (principal,target,session,grant_id)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    invocation_id TEXT NOT NULL, grant_id TEXT NOT NULL,
                    principal TEXT NOT NULL, target TEXT NOT NULL,
                    session TEXT NOT NULL, state_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('claimed','denied','ok','raised')),
                    started REAL NOT NULL, finished REAL,
                    state_sha256 TEXT, result_sha256 TEXT,
                    handler_artifact_sha256 TEXT NOT NULL,
                    source_git_oid TEXT NOT NULL, receipt_sha256 TEXT,
                    PRIMARY KEY (principal,target,session,invocation_id)
                );
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self._path, timeout=10.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        try:
            yield db
        except Exception:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def _now(self) -> float:
        try:
            now = self._clock()
        except Exception as exc:
            raise ReadAccessDenied("host clock failed") from exc
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ReadAccessDenied("invalid host clock")
        return float(now)

    def _assert_not_handler(self) -> None:
        # The executing callback must not reenter grant mutation on this
        # instance while its transaction owns the SQLite write lock.
        if getattr(self._in_handler, "active", False):
            raise ReadAccessDenied("reentrant broker operation denied")

    def grant(self, grant_id: str, state_key: str, *, expires_at_epoch: float) -> None:
        """Host-only grant creation; cannot resurrect an existing grant ID."""
        self._assert_not_handler()
        grant_id, state_key = _identifier(grant_id), _key(state_key)
        if state_key not in self._handlers:
            raise ReadAccessDenied("no host-approved handler")
        if type(expires_at_epoch) not in (int, float) or not math.isfinite(expires_at_epoch):
            raise ValueError("finite grant expiry required")
        if expires_at_epoch <= self._now():
            raise ReadAccessDenied("grant already expired")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("INSERT INTO grants(grant_id,principal,target,session,state_key,expires) "
                           "VALUES(?,?,?,?,?,?)", (
                    grant_id, self._context.principal_id, self._context.target_id,
                    self._context.session_id, state_key, float(expires_at_epoch)))
                db.commit()
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise ReadAccessDenied("grant ID already used") from exc

    def revoke(self, grant_id: str) -> None:
        """Revocation linearizes at the SQLite writer transaction boundary."""
        self._assert_not_handler()
        grant_id = _identifier(grant_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            scope = (grant_id, self._context.principal_id,
                     self._context.target_id, self._context.session_id)
            row = db.execute("SELECT grant_id FROM grants WHERE grant_id=? "
                             "AND principal=? AND target=? AND session=?", scope).fetchone()
            if row is None:
                db.rollback()
                raise ReadAccessDenied("grant unavailable")
            db.execute("UPDATE grants SET revoked=1 WHERE grant_id=? "
                       "AND principal=? AND target=? AND session=?", scope)
            db.commit()

    def _granted(self, db: sqlite3.Connection, grant_id: str, key: str, now: float) -> bool:
        row = db.execute("SELECT * FROM grants WHERE grant_id=? AND principal=? "
                         "AND target=? AND session=?", (
                             grant_id, self._context.principal_id,
                             self._context.target_id, self._context.session_id)).fetchone()
        return bool(row is not None and row["principal"] == self._context.principal_id
                    and row["target"] == self._context.target_id
                    and row["session"] == self._context.session_id
                    and row["state_key"] == key and not row["revoked"]
                    and now < row["expires"])

    def _record(self, db: sqlite3.Connection, invocation_id: str,
                *, status: str, finished: float, state_hash: str | None = None,
                result_hash: str | None = None) -> dict[str, Any]:
        scope = (invocation_id, self._context.principal_id,
                 self._context.target_id, self._context.session_id)
        row = db.execute("SELECT * FROM attempts WHERE invocation_id=? "
                         "AND principal=? AND target=? AND session=?", scope).fetchone()
        receipt = {
            "version": 1, "invocation_id": invocation_id,
            "grant_id": row["grant_id"], "principal": row["principal"],
            "target": row["target"], "session": row["session"],
            "state_key": row["state_key"], "status": status,
            "started_at_epoch": row["started"], "finished_at_epoch": finished,
            "state_sha256": state_hash,
            "result_sha256": result_hash,
            "handler_artifact_sha256": row["handler_artifact_sha256"],
            "source_git_oid": row["source_git_oid"],
            "artifact_binding_kind": "host_asserted_unverified",
            "evidence_class": "HOST_EXECUTION_RECEIPT_NOT_INDEPENDENT_VERIFY",
        }
        receipt_hash = _hash(receipt)
        db.execute("UPDATE attempts SET status=?, finished=?, state_sha256=?, "
                   "result_sha256=?, receipt_sha256=? WHERE invocation_id=? "
                   "AND principal=? AND target=? AND session=?", (
                       status, finished, state_hash, result_hash, receipt_hash, *scope))
        return {**receipt, "receipt_sha256": receipt_hash}

    def execute(self, grant_id: str, invocation_id: str, state_key: str) -> tuple[Any, dict[str, Any]]:
        """One durable invocation, with current grant and effect serialized.

        The ledger and handler run under a SQLite *writer* transaction shared
        with every broker-owned grant mutator. The transaction does not undo
        external effects, and same-UID direct DB or handler use remains a
        bypass. Returns (result, same-trust host receipt).
        """
        self._assert_not_handler()
        grant_id, invocation_id, state_key = (
            _identifier(grant_id), _identifier(invocation_id), _key(state_key))
        # Persist a claim BEFORE touching any effect handler. On crash, the
        # claim survives as ambiguous and all retries of the ID fail closed.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute("SELECT 1 FROM attempts WHERE invocation_id=? AND principal=? "
                              "AND target=? AND session=?", (
                                  invocation_id, self._context.principal_id,
                                  self._context.target_id, self._context.session_id)).fetchone():
                    raise ReadAccessDenied("invocation replay denied")
                now = self._now()
                if not self._granted(db, grant_id, state_key, now):
                    raise ReadAccessDenied("current use grant denied")
                # Do not leak the installed handler registry to an ungranted
                # caller by returning different errors for known/unknown keys.
                approved = self._handlers.get(state_key)
                if approved is None:
                    raise ReadAccessDenied("current use grant denied")
                db.execute("INSERT INTO attempts(invocation_id,grant_id,principal,target,session,"
                           "state_key,status,started,handler_artifact_sha256,source_git_oid) "
                           "VALUES(?,?,?,?,? ,?,'claimed',?,?,?)", (
                               invocation_id, grant_id, self._context.principal_id,
                               self._context.target_id, self._context.session_id,
                               state_key, now, approved.handler_artifact_sha256,
                               approved.source_git_oid))
                db.commit()
            except Exception:
                db.rollback()
                raise

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if not self._granted(db, grant_id, state_key, self._now()):
                    self._record(db, invocation_id, status="denied", finished=self._now())
                    db.commit()
                    raise ReadAccessDenied("grant revoked or expired before effect")
                try:
                    projection = self._ledger.project(self._context.target_id,
                                                      self._context.session_id)
                except Exception as exc:
                    raise ReadAccessDenied("host ledger projection unavailable") from exc
                if type(projection) is not dict or state_key not in projection:
                    raise ReadAccessDenied("effect state unavailable")
                state = json.loads(_encoded(projection[state_key]).decode("utf-8"))
                state_hash = _hash(state)
                # Recheck immediately before handler invocation. A concurrent
                # revoke cannot commit while this transaction owns the lock.
                if not self._granted(db, grant_id, state_key, self._now()):
                    self._record(db, invocation_id, status="denied", finished=self._now(),
                                 state_hash=state_hash)
                    db.commit()
                    raise ReadAccessDenied("grant revoked or expired before effect")
                self._in_handler.active = True
                try:
                    result = approved.handler(state)
                    result_hash = _hash(result)
                    receipt = self._record(db, invocation_id, status="ok",
                                           finished=self._now(), state_hash=state_hash,
                                           result_hash=result_hash)
                    db.commit()
                    return result, receipt
                except Exception as exc:
                    self._record(db, invocation_id, status="raised",
                                 finished=self._now(), state_hash=state_hash)
                    db.commit()
                    raise EffectAttemptFailed("effect attempt failed; invocation consumed") from exc
                finally:
                    self._in_handler.active = False
            except (ReadAccessDenied, sqlite3.Error):
                if db.in_transaction:
                    # Reservation remains consumed even if this effect never
                    # began. There is no safe automatic retry after ambiguity.
                    self._record(db, invocation_id, status="denied", finished=self._now())
                    db.commit()
                raise

    def attempt(self, invocation_id: str) -> dict[str, Any] | None:
        """Read durable attempt status; ``claimed`` means crash ambiguity."""
        invocation_id = _identifier(invocation_id)
        with self._connect() as db:
            row = db.execute("SELECT status,receipt_sha256,grant_id,state_sha256,result_sha256 "
                             "FROM attempts WHERE invocation_id=? AND principal=? AND target=? "
                             "AND session=?", (
                                 invocation_id, self._context.principal_id,
                                 self._context.target_id, self._context.session_id)).fetchone()
        return dict(row) if row is not None else None
