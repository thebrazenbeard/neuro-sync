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
The durable ``failure_class`` distinguishes permission denials from
infrastructure failures. Counting every ``denied`` status as a successful
permission-governance test would contaminate an evaluator.

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
from uuid import uuid4

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


def _plan_encoded(value: Any) -> bytes:
    """Bound the *input traversal* before JSON serialization can allocate.

    This does not defend against hostile concurrent mutation of a Python
    object inside the same process; process isolation remains host duty.
    """
    nodes = 0
    active: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 4096 or depth > 32:
            raise ReadAccessDenied("plan exceeds bounded JSON complexity")
        if type(item) is str:
            if len(item) > MAX_READ_VALUE_BYTES:
                raise ReadAccessDenied("plan scalar exceeds host limit")
        elif item is None or type(item) is bool:
            return
        elif type(item) is int:
            if item.bit_length() > MAX_READ_VALUE_BYTES * 3:
                raise ReadAccessDenied("plan integer exceeds host limit")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ReadAccessDenied("plan value is not finite")
        elif type(item) in (list, dict):
            ident = id(item)
            if ident in active:
                raise ReadAccessDenied("plan contains cyclic JSON")
            if len(item) > 4096:
                raise ReadAccessDenied("plan exceeds bounded JSON complexity")
            active.add(ident)
            try:
                if type(item) is list:
                    for child in item:
                        visit(child, depth + 1)
                else:
                    for key, child in item.items():
                        if type(key) is not str or len(key) > MAX_READ_VALUE_BYTES:
                            raise ReadAccessDenied("plan has invalid object key")
                        visit(key, depth + 1)
                        visit(child, depth + 1)
            finally:
                active.remove(ident)
        else:
            raise ReadAccessDenied("plan value is not JSON")

    visit(value, 0)
    try:
        return _encoded(value)
    except RecursionError as exc:
        raise ReadAccessDenied("plan exceeds bounded JSON depth") from exc


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


@dataclass(frozen=True)
class ApprovedPlanEffect:
    """Host-installed effect that consumes the broker's copied, checked plan.

    Artifact/source provenance is host-asserted, not independently attested.
    """

    handler: Callable[[Any, Any], Any]
    handler_artifact_sha256: str
    source_git_oid: str

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise TypeError("host plan handler must be callable")
        if type(self.handler_artifact_sha256) is not str or not HEX64.fullmatch(self.handler_artifact_sha256):
            raise ValueError("invalid plan handler artifact SHA-256")
        if type(self.source_git_oid) is not str or not GIT_OID.fullmatch(self.source_git_oid):
            raise ValueError("invalid plan handler source OID")


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
                 clock: Callable[[], float] = time.time,
                 approved_plan_handlers: Mapping[str, ApprovedPlanEffect] | None = None) -> None:
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
        if approved_plan_handlers is None:
            approved_plan_handlers = {}
        if not isinstance(approved_plan_handlers, Mapping):
            raise TypeError("approved plan handler mapping required")
        plan_handlers: dict[str, ApprovedPlanEffect] = {}
        for key, effect in approved_plan_handlers.items():
            _key(key)
            if not isinstance(effect, ApprovedPlanEffect):
                raise TypeError("unapproved plan handler descriptor")
            plan_handlers[key] = effect
        self._plan_handlers = MappingProxyType(plan_handlers)
        self._ledger = ledger
        self._context = context
        self._clock = clock
        self._path = str(database_path)
        self._clock_path = str(Path(self._path).with_name(
            Path(self._path).name + ".clock.sqlite3"))
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
                -- Separate table keeps plan approvals invisible to older
                -- binaries which only know how to query legacy grants.
                CREATE TABLE IF NOT EXISTS plan_grants (
                    grant_id TEXT NOT NULL, principal TEXT NOT NULL,
                    target TEXT NOT NULL, session TEXT NOT NULL,
                    state_key TEXT NOT NULL, expires REAL NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)),
                    plan_sha256 TEXT NOT NULL,
                    planned_state_sha256 TEXT NOT NULL,
                    plan_handler_artifact_sha256 TEXT NOT NULL,
                    plan_handler_source_git_oid TEXT NOT NULL,
                    plan_consumed_invocation_id TEXT,
                    PRIMARY KEY (principal,target,session,grant_id)
                );
                -- v1 plan grants stay inert historical records in this code.
                -- A pre-clock binary can query plan_grants but cannot see v2.
                CREATE TABLE IF NOT EXISTS clock_guarded_plan_grants (
                    grant_id TEXT NOT NULL, principal TEXT NOT NULL,
                    target TEXT NOT NULL, session TEXT NOT NULL,
                    state_key TEXT NOT NULL, expires REAL NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)),
                    plan_sha256 TEXT NOT NULL,
                    planned_state_sha256 TEXT NOT NULL,
                    plan_handler_artifact_sha256 TEXT NOT NULL,
                    plan_handler_source_git_oid TEXT NOT NULL,
                    plan_consumed_invocation_id TEXT,
                    PRIMARY KEY (principal,target,session,grant_id)
                );
                -- Last OBSERVED host clock value for this whole authority DB.
                -- This cannot detect time lost before the first observation.
                CREATE TABLE IF NOT EXISTS plan_clock_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    last_observed_epoch REAL NOT NULL
                );
                -- A binding marker for the *separately committed* clock DB.
                -- The former in-main plan_clock_state is legacy metadata;
                -- never use it as the durable floor for new v2 effects.
                CREATE TABLE IF NOT EXISTS plan_clock_binding (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    binding_uuid TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('PENDING','READY'))
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    invocation_id TEXT NOT NULL, grant_id TEXT NOT NULL,
                    principal TEXT NOT NULL, target TEXT NOT NULL,
                    session TEXT NOT NULL, state_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('claimed','denied','ok','raised')),
                    failure_class TEXT CHECK(failure_class IS NULL OR failure_class IN
                        ('permission','infrastructure','handler')),
                    started REAL NOT NULL, finished REAL,
                    state_sha256 TEXT, result_sha256 TEXT,
                    handler_artifact_sha256 TEXT NOT NULL,
                    source_git_oid TEXT NOT NULL, receipt_sha256 TEXT,
                    PRIMARY KEY (principal,target,session,invocation_id)
                );
            """)
            # Existing databases may lack either independently introduced
            # nullable column. Serialize migrations for concurrent starts;
            # neither historical rows nor replay claims may be discarded.
            db.execute("BEGIN IMMEDIATE")
            try:
                for table, columns in {"attempts": {
                    "plan_sha256": "TEXT",
                    "failure_class": "TEXT CHECK(failure_class IS NULL OR failure_class IN "
                                     "('permission','infrastructure','handler'))",
                }}.items():
                    present = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
                    for name, kind in columns.items():
                        if name not in present:
                            db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
                db.commit()
            except Exception:
                db.rollback()
                raise
        self._init_clock_authority()

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

    @contextmanager
    def _clock_connect(self, *, create: bool = False):
        # READY always opens read/write without sqlite's default creation.
        # Loss of the sidecar must fail closed, never initialize a new floor.
        path = Path(self._clock_path).absolute().as_uri()
        mode = "rwc" if create else "rw"
        clock_db = sqlite3.connect(f"{path}?mode={mode}", uri=True,
                                   timeout=10.0, isolation_level=None)
        try:
            clock_db.row_factory = sqlite3.Row
            clock_db.execute("PRAGMA busy_timeout=10000")
            clock_db.execute("PRAGMA synchronous=FULL")
            yield clock_db
        except Exception:
            if clock_db.in_transaction:
                clock_db.rollback()
            raise
        finally:
            clock_db.close()

    def _init_clock_authority(self) -> None:
        """Pair two host-owned files before any v2 grant can be issued.

        If v2 grants exist without a marker, never create replacement time
        authority. A PENDING bootstrap with zero grants can be finished only
        against the matching sidecar or a genuinely absent fresh sidecar.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM plan_clock_binding WHERE singleton=1").fetchone()
            if row is None:
                if db.execute("SELECT 1 FROM clock_guarded_plan_grants LIMIT 1").fetchone():
                    db.rollback()
                    return
                db.execute("INSERT INTO plan_clock_binding(singleton,binding_uuid,status) "
                           "VALUES(1,?,'PENDING')", (uuid4().hex,))
            db.commit()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM plan_clock_binding WHERE singleton=1").fetchone()
            if row is None or row["status"] == "READY":
                db.commit()
                return
            if (row["status"] != "PENDING" or
                    db.execute("SELECT 1 FROM clock_guarded_plan_grants LIMIT 1").fetchone()):
                db.rollback()
                return
            try:
                fresh_sidecar = not Path(self._clock_path).exists()
                with self._clock_connect(create=fresh_sidecar) as sidecar:
                    sidecar.execute("BEGIN IMMEDIATE")
                    sidecar.execute("CREATE TABLE IF NOT EXISTS clock_state ("
                                    "singleton INTEGER PRIMARY KEY CHECK(singleton=1),"
                                    "binding_uuid TEXT NOT NULL,"
                                    "last_observed_epoch REAL)")
                    stored = sidecar.execute("SELECT * FROM clock_state WHERE singleton=1").fetchone()
                    if stored is None:
                        # Existing sidecar without a marker is not reusable.
                        if not fresh_sidecar:
                            raise ReadAccessDenied("existing clock sidecar has no binding")
                        sidecar.execute("INSERT INTO clock_state(singleton,binding_uuid) "
                                        "VALUES(1,?)", (row["binding_uuid"],))
                    elif stored["binding_uuid"] != row["binding_uuid"]:
                        raise ReadAccessDenied("clock sidecar binding mismatch")
                    sidecar.commit()
                db.execute("UPDATE plan_clock_binding SET status='READY' WHERE singleton=1")
                db.commit()
            except (sqlite3.Error, ReadAccessDenied):
                db.rollback()
                # A failed bootstrap leaves PENDING; no plan use is authorized.
                return

    def _now(self) -> float:
        try:
            now = self._clock()
        except Exception as exc:
            raise ReadAccessDenied("host clock failed") from exc
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ReadAccessDenied("invalid host clock")
        return float(now)

    def _plan_now(self, db: sqlite3.Connection) -> float:
        """Commit observed host time in a separate paired SQLite authority.

        The caller must already hold main BEGIN IMMEDIATE. Under that lock,
        acquire the sidecar writer and commit its high-water *independently*
        of the effect database. Handler errors, receipt errors, SQLite write
        failures or rollback of the main transaction cannot erase the sample.
        This is not independently attested time or protection against a host
        rolling back/replacing the sidecar itself.
        """
        if not db.in_transaction:
            raise ReadAccessDenied("main authority writer required")
        binding = db.execute("SELECT binding_uuid,status FROM plan_clock_binding "
                             "WHERE singleton=1").fetchone()
        if binding is None or binding["status"] != "READY":
            raise ReadAccessDenied("clock authority binding unavailable")
        try:
            with self._clock_connect() as sidecar:
                sidecar.execute("BEGIN IMMEDIATE")
                row = sidecar.execute("SELECT binding_uuid,last_observed_epoch "
                                      "FROM clock_state WHERE singleton=1").fetchone()
                if row is None or row["binding_uuid"] != binding["binding_uuid"]:
                    raise ReadAccessDenied("clock authority binding mismatch")
                now = self._now()
                previous = row["last_observed_epoch"]
                # NULL is legitimate only for an unused, freshly paired
                # authority. A floor missing after any v2 approval would
                # permit resurrection; malformed, nonfinite SQLite values
                # must never silently become the new comparison baseline.
                if previous is None:
                    if db.execute("SELECT 1 FROM clock_guarded_plan_grants LIMIT 1").fetchone():
                        raise ReadAccessDenied("clock high-water missing for existing plan grants")
                elif (type(previous) not in (float, int) or
                      not math.isfinite(previous)):
                    raise ReadAccessDenied("invalid persisted clock high-water")
                if previous is not None and now < previous:
                    raise ReadAccessDenied("host clock moved backward; plan authority denied")
                if previous is None or now > previous:
                    sidecar.execute("UPDATE clock_state SET last_observed_epoch=? "
                                    "WHERE singleton=1", (now,))
                sidecar.commit()
                return now
        except sqlite3.Error as exc:
            raise ReadAccessDenied("clock authority sidecar unavailable") from exc

    def _checkpoint_plan_clock(self) -> float:
        """Checkpoint once under main->sidecar lock ordering before callbacks.

        The sidecar commits independently; the ordinary main transaction is
        simply an ordering lock, not the owner of the observed high-water.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            now = self._plan_now(db)
            db.commit()
        return now

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
                for table in ("plan_grants", "clock_guarded_plan_grants"):
                    if db.execute(f"SELECT 1 FROM {table} WHERE grant_id=? AND principal=? "
                                  "AND target=? AND session=?", (
                                      grant_id, self._context.principal_id,
                                      self._context.target_id, self._context.session_id)).fetchone():
                        raise ReadAccessDenied("grant ID already used")
                db.execute("INSERT INTO grants(grant_id,principal,target,session,state_key,expires) "
                           "VALUES(?,?,?,?,?,?)", (
                    grant_id, self._context.principal_id, self._context.target_id,
                    self._context.session_id, state_key, float(expires_at_epoch)))
                db.commit()
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise ReadAccessDenied("grant ID already used") from exc

    def grant_plan(self, grant_id: str, state_key: str, *,
                   plan: Any, expires_at_epoch: float) -> None:
        """Host-only approval of exact bounded plan, state and installed handler.

        This does not authenticate a remote caller or make external ledger
        changes atomic with this authority database.
        """
        self._assert_not_handler()
        grant_id, state_key = _identifier(grant_id), _key(state_key)
        approved = self._plan_handlers.get(state_key)
        if approved is None:
            raise ReadAccessDenied("no host-approved plan handler")
        if type(expires_at_epoch) not in (int, float) or not math.isfinite(expires_at_epoch):
            raise ValueError("finite grant expiry required")
        # Canonical bytes are computed by the host broker. No supplied plan
        # hash is trusted, and mutations of the caller's object cannot modify
        # this approval's digest.
        plan_hash = sha256(_plan_encoded(plan)).hexdigest()
        self._checkpoint_plan_clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                try:
                    projection = self._ledger.project(self._context.target_id,
                                                      self._context.session_id)
                except Exception as exc:
                    raise ReadAccessDenied("host ledger projection unavailable") from exc
                if type(projection) is not dict or state_key not in projection:
                    raise ReadAccessDenied("effect state unavailable")
                state_hash = sha256(_plan_encoded(projection[state_key])).hexdigest()
                if self._plan_now(db) >= expires_at_epoch:
                    raise ReadAccessDenied("grant already expired")
                for table in ("grants", "plan_grants", "clock_guarded_plan_grants"):
                    if db.execute(f"SELECT 1 FROM {table} WHERE grant_id=? AND principal=? "
                                  "AND target=? AND session=?", (
                                      grant_id, self._context.principal_id,
                                      self._context.target_id, self._context.session_id)).fetchone():
                        raise ReadAccessDenied("grant ID already used")
                db.execute("INSERT INTO clock_guarded_plan_grants(grant_id,principal,target,session,state_key,expires,"
                           "plan_sha256,planned_state_sha256,plan_handler_artifact_sha256,"
                           "plan_handler_source_git_oid) VALUES(?,?,?,?,?,?,?,?,?,?)", (
                               grant_id, self._context.principal_id, self._context.target_id,
                               self._context.session_id, state_key, float(expires_at_epoch),
                               plan_hash, state_hash, approved.handler_artifact_sha256,
                               approved.source_git_oid))
                db.commit()
            except ReadAccessDenied:
                # No plan-grant mutation precedes a ReadAccessDenied. Preserve
                # the observed high-water even for a rejected grant attempt.
                if db.in_transaction:
                    db.commit()
                raise
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
            legacy = db.execute("SELECT 1 FROM grants WHERE grant_id=? "
                                "AND principal=? AND target=? AND session=?", scope).fetchone()
            planned = db.execute("SELECT 1 FROM plan_grants WHERE grant_id=? "
                                 "AND principal=? AND target=? AND session=?", scope).fetchone()
            clocked = db.execute("SELECT 1 FROM clock_guarded_plan_grants WHERE grant_id=? "
                                 "AND principal=? AND target=? AND session=?", scope).fetchone()
            if legacy is None and planned is None and clocked is None:
                db.rollback()
                raise ReadAccessDenied("grant unavailable")
            for table in ("grants", "plan_grants", "clock_guarded_plan_grants"):
                db.execute(f"UPDATE {table} SET revoked=1 WHERE grant_id=? "
                           "AND principal=? AND target=? AND session=?", scope)
            db.commit()

    def _granted(self, db: sqlite3.Connection, grant_id: str, key: str, now: float, *,
                 plan_hash: str | None = None,
                 plan_effect: ApprovedPlanEffect | None = None,
                 plan_invocation_id: str | None = None) -> bool:
        table = "grants" if plan_hash is None else "clock_guarded_plan_grants"
        row = db.execute(f"SELECT * FROM {table} WHERE grant_id=? AND principal=? "
                         "AND target=? AND session=?", (
                             grant_id, self._context.principal_id,
                             self._context.target_id, self._context.session_id)).fetchone()
        authorized = bool(row is not None and row["principal"] == self._context.principal_id
                    and row["target"] == self._context.target_id
                    and row["session"] == self._context.session_id
                    and row["state_key"] == key and not row["revoked"]
                    and now < row["expires"])
        if not authorized:
            return False
        if plan_hash is None:
            return True
        return bool(plan_effect is not None
                    and row["plan_sha256"] == plan_hash
                    and row["planned_state_sha256"] is not None
                    and row["plan_handler_artifact_sha256"] == plan_effect.handler_artifact_sha256
                    and row["plan_handler_source_git_oid"] == plan_effect.source_git_oid
                    and row["plan_consumed_invocation_id"] == plan_invocation_id)

    def _record(self, db: sqlite3.Connection, invocation_id: str,
                *, status: str, finished: float, state_hash: str | None = None,
                result_hash: str | None = None,
                failure_class: str | None = None) -> dict[str, Any]:
        permitted = {"ok": (None,), "denied": ("permission", "infrastructure"),
                     "raised": ("handler",)}
        if status not in permitted or failure_class not in permitted[status]:
            raise ValueError("invalid effect outcome/failure classification")
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
        # Preserve the prior successful-receipt schema/hash. Denial and handler
        # failure receipts gain an explicit non-success classification.
        if failure_class is not None:
            receipt["failure_class"] = failure_class
        if row["plan_sha256"] is not None:
            receipt["version"] = 2
            receipt["plan_sha256"] = row["plan_sha256"]
        receipt_hash = _hash(receipt)
        db.execute("UPDATE attempts SET status=?, finished=?, state_sha256=?, "
                   "result_sha256=?, receipt_sha256=?, failure_class=? WHERE invocation_id=? "
                   "AND principal=? AND target=? AND session=?", (
                       status, finished, state_hash, result_hash, receipt_hash,
                       failure_class, *scope))
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
                    self._record(db, invocation_id, status="denied", finished=self._now(),
                                 failure_class="permission")
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
                                 state_hash=state_hash, failure_class="permission")
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
                                 finished=self._now(), state_hash=state_hash,
                                 failure_class="handler")
                    db.commit()
                    raise EffectAttemptFailed("effect attempt failed; invocation consumed") from exc
                finally:
                    self._in_handler.active = False
            except (ReadAccessDenied, sqlite3.Error):
                if db.in_transaction:
                    # Reservation remains consumed even if this effect never
                    # began. There is no safe automatic retry after ambiguity.
                    self._record(db, invocation_id, status="denied", finished=self._now(),
                                 failure_class="infrastructure")
                    db.commit()
                raise

    def execute_plan(self, grant_id: str, invocation_id: str, state_key: str, *,
                     plan: Any) -> tuple[Any, dict[str, Any]]:
        """Attempt the exact approved plan against the unchanged projected state.

        The authoritative broker canonicalizes caller input and passes a fresh
        copy to an explicitly plan-aware handler. The durable claim and effect
        phases have the same crash/replay semantics as ``execute``. This is
        local host mediation, not a process identity/OS attestation boundary.
        """
        self._assert_not_handler()
        grant_id, invocation_id, state_key = (
            _identifier(grant_id), _identifier(invocation_id), _key(state_key))
        encoded_plan = _plan_encoded(plan)
        plan_hash = sha256(encoded_plan).hexdigest()
        # Detached from caller-owned objects; the handler gets this copy only.
        detached_plan = json.loads(encoded_plan.decode("utf-8"))
        self._checkpoint_plan_clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                now = self._plan_now(db)
                if db.execute("SELECT 1 FROM attempts WHERE invocation_id=? AND principal=? "
                              "AND target=? AND session=?", (
                                  invocation_id, self._context.principal_id,
                                  self._context.target_id, self._context.session_id)).fetchone():
                    raise ReadAccessDenied("invocation replay denied")
                approved = self._plan_handlers.get(state_key)
                if not self._granted(db, grant_id, state_key, now, plan_hash=plan_hash,
                                     plan_effect=approved):
                    raise ReadAccessDenied("current use grant denied")
                # A plan approval is single-use. Claim it atomically with the
                # durable attempt reservation under this same writer lock.
                updated = db.execute("UPDATE clock_guarded_plan_grants SET plan_consumed_invocation_id=? "
                           "WHERE grant_id=? AND principal=? AND target=? AND session=? "
                           "AND plan_consumed_invocation_id IS NULL", (
                               invocation_id, grant_id, self._context.principal_id,
                               self._context.target_id, self._context.session_id))
                if updated.rowcount != 1:
                    raise ReadAccessDenied("plan grant already consumed")
                db.execute("INSERT INTO attempts(invocation_id,grant_id,principal,target,session,"
                           "state_key,status,started,handler_artifact_sha256,source_git_oid,"
                           "plan_sha256) VALUES(?,?,?,?,?,?,'claimed',?,?,?,?)", (
                               invocation_id, grant_id, self._context.principal_id,
                               self._context.target_id, self._context.session_id,
                               state_key, now, approved.handler_artifact_sha256,
                               approved.source_git_oid, plan_hash))
                db.commit()
            except ReadAccessDenied:
                # Every ReadAccessDenied in this claim phase occurs before any
                # successful grant/attempt mutation. Clock checkpoint also
                # persists independently if a later main transaction fails.
                if db.in_transaction:
                    db.commit()
                raise
            except Exception:
                db.rollback()
                raise

        self._checkpoint_plan_clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                try:
                    projection = self._ledger.project(self._context.target_id,
                                                      self._context.session_id)
                except Exception as exc:
                    raise ReadAccessDenied("host ledger projection unavailable") from exc
                if type(projection) is not dict or state_key not in projection:
                    raise ReadAccessDenied("effect state unavailable")
                encoded_state = _plan_encoded(projection[state_key])
                state_hash = sha256(encoded_state).hexdigest()
                state = json.loads(encoded_state.decode("utf-8"))
                if not self._granted(db, grant_id, state_key, self._plan_now(db),
                                     plan_hash=plan_hash, plan_effect=approved,
                                     plan_invocation_id=invocation_id):
                    self._record(db, invocation_id, status="denied", finished=self._plan_now(db),
                                 state_hash=state_hash, failure_class="permission")
                    db.commit()
                    raise ReadAccessDenied("grant revoked or expired before effect")
                saved = db.execute("SELECT planned_state_sha256 FROM clock_guarded_plan_grants WHERE "
                                   "grant_id=? AND principal=? AND target=? AND session=?", (
                                       grant_id, self._context.principal_id,
                                       self._context.target_id, self._context.session_id)).fetchone()
                if saved is None or state_hash != saved["planned_state_sha256"]:
                    self._record(db, invocation_id, status="denied", finished=self._plan_now(db),
                                 state_hash=state_hash, failure_class="infrastructure")
                    db.commit()
                    raise ReadAccessDenied("approved plan state changed before effect")
                if not self._granted(db, grant_id, state_key, self._plan_now(db),
                                     plan_hash=plan_hash, plan_effect=approved,
                                     plan_invocation_id=invocation_id):
                    self._record(db, invocation_id, status="denied", finished=self._plan_now(db),
                                 state_hash=state_hash, failure_class="permission")
                    db.commit()
                    raise ReadAccessDenied("grant revoked or expired before effect")
                self._in_handler.active = True
                try:
                    result = approved.handler(state, detached_plan)
                    result_hash = _hash(result)
                    receipt = self._record(db, invocation_id, status="ok",
                                           finished=self._plan_now(db), state_hash=state_hash,
                                           result_hash=result_hash)
                    db.commit()
                    return result, receipt
                except Exception as exc:
                    self._record(db, invocation_id, status="raised",
                                 finished=self._plan_now(db), state_hash=state_hash,
                                 failure_class="handler")
                    db.commit()
                    raise EffectAttemptFailed("effect attempt failed; invocation consumed") from exc
                finally:
                    self._in_handler.active = False
            except (ReadAccessDenied, sqlite3.Error):
                if db.in_transaction:
                    self._record(db, invocation_id, status="denied", finished=self._plan_now(db),
                                 failure_class="infrastructure")
                    db.commit()
                raise

    def attempt(self, invocation_id: str) -> dict[str, Any] | None:
        """Read durable outcome/class; ``claimed`` means crash ambiguity.

        A metrics consumer must demand both ``status='denied'`` and
        ``failure_class='permission'`` for local permission-denial evidence.
        This still does not authenticate the broker's principal or host.
        """
        invocation_id = _identifier(invocation_id)
        with self._connect() as db:
            row = db.execute("SELECT status,failure_class,receipt_sha256,grant_id,"
                             "state_sha256,result_sha256,plan_sha256 "
                             "FROM attempts WHERE invocation_id=? AND principal=? AND target=? "
                             "AND session=?", (
                                 invocation_id, self._context.principal_id,
                                 self._context.target_id, self._context.session_id)).fetchone()
        return dict(row) if row is not None else None
