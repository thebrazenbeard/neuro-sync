"""SQLite StateBus PROPOSE -> VALIDATE -> COMMIT reference ledger.

SQLite transactions and a local hash chain detect many accidental changes;
they do NOT create immutable custody. Anyone able to modify the DB and its
local head anchor can rewrite or truncate history. External authenticated
authority, durable independently stored anchors, access control and backup
remain required before production use. No model-written 'verified' flag is
accepted as authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
import time
from typing import Callable

from .schema import (
    SchemaError, StateDelta, StateOp, ValidationReceipt,
    canonical_json, _validate_key,
)


GENESIS_HASH = "0" * 64
MAX_READ = 500
CLOCK_SKEW_SECONDS = 300


class AuthorizationError(PermissionError):
    pass


class ConflictError(ValueError):
    pass


class SequenceError(ConflictError):
    pass


class ExpiredError(ValueError):
    pass


class LedgerIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProposalReceipt:
    event_id: str
    content_hash: str
    status: str
    already_present: bool


@dataclass(frozen=True)
class CommitReceipt:
    event_id: str
    event_offset: int
    content_hash: str
    chain_hash: str
    key_revision: int
    already_committed: bool


@dataclass(frozen=True)
class CommittedEvent:
    event_offset: int
    key_revision: int
    delta: StateDelta
    chain_hash: str


def _chain_hash(prev_hash: str, event_json: str, validation_sha: str,
                event_offset: int, key_revision: int) -> str:
    body = canonical_json({
        "prev_hash": prev_hash,
        "event_json": event_json,
        "validation_sha256": validation_sha,
        "event_offset": event_offset,
        "key_revision": key_revision,
    })
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _authorized(callback: Callable | None, *args: object) -> None:
    if not callable(callback):
        raise AuthorizationError("a separately configured authority callback is required")
    try:
        decision = callback(*args)
    except Exception as exc:
        raise AuthorizationError("external authority callback failed") from exc
    if decision is not True:
        raise AuthorizationError("external authority rejected the action")


class StateLedger:
    """SQLite-backed local reference, with a trusted host-supplied clock hook.

    The `clock` hook is for deterministic tests. In a real runtime only the
    host, not a model/requester, may configure the clock or authority callbacks.
    SQLite file ownership alone is not cryptographic authority.
    """

    def __init__(self, db_path: str | Path, *, clock: Callable[[], float] | None = None):
        path = str(db_path)
        if not path or path == ":memory:":
            raise ValueError("StateLedger requires an explicit persistent SQLite file")
        self.db_path = path
        self._clock = clock or time.time
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS proposals (
                    event_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_key TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('proposed','validated','committed')),
                    validation_json TEXT,
                    commit_offset INTEGER,
                    UNIQUE(source_id, session_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_offset INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE REFERENCES proposals(event_id),
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_key TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    key_revision INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    validation_sha256 TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    chain_hash TEXT NOT NULL,
                    UNIQUE(source_id, session_id, sequence),
                    UNIQUE(target_id, session_id, payload_key, key_revision)
                );
                CREATE TABLE IF NOT EXISTS ledger_head (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    head_offset INTEGER NOT NULL,
                    head_hash TEXT NOT NULL
                );
                INSERT OR IGNORE INTO ledger_head(id,head_offset,head_hash)
                    VALUES(1,0,'""" + GENESIS_HASH + """');
                CREATE TRIGGER IF NOT EXISTS events_no_update
                    BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append-only events'); END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete
                    BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append-only events'); END;
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @contextmanager
    def _db(self):
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def _now(self) -> int:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("trusted clock returned invalid value")
        return int(value)

    def _check_window(self, delta: StateDelta) -> None:
        now = self._now()
        if delta.expires_at_epoch <= now:
            raise ExpiredError("StateDelta delivery deadline expired")
        if delta.issued_at_epoch > now + CLOCK_SKEW_SECONDS:
            raise ExpiredError("StateDelta issued too far in the future")

    def _load_proposal(self, db: sqlite3.Connection, event_id: str):
        row = db.execute(
            "SELECT event_id,content_hash,status,event_json,validation_json,commit_offset "
            "FROM proposals WHERE event_id=?", (event_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown proposal: {event_id}")
        return row

    def propose(
        self, delta: StateDelta, *,
        authorize_proposer: Callable[[StateDelta], bool] | None,
    ) -> ProposalReceipt:
        if not isinstance(delta, StateDelta):
            raise TypeError("StateDelta required")
        _authorized(authorize_proposer, delta)
        encoded = delta.to_json()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            found = db.execute(
                "SELECT content_hash,status FROM proposals WHERE event_id=?", (delta.event_id,)
            ).fetchone()
            if found:
                if found[0] != delta.content_hash:
                    raise ConflictError("event ID reused with a different payload/sequence")
                # Expired identical retries merely read the previously
                # accepted receipt; they never create a fresh event.
                return ProposalReceipt(delta.event_id, delta.content_hash, found[1], True)
            self._check_window(delta)
            occupied = db.execute(
                "SELECT event_id FROM proposals WHERE source_id=? AND session_id=? AND sequence=?",
                (delta.source_id, delta.session_id, delta.sequence),
            ).fetchone()
            if occupied:
                raise ConflictError("source/session sequence already proposed")
            db.execute(
                "INSERT INTO proposals(event_id,source_id,target_id,session_id,sequence,payload_key,"
                "content_hash,event_json,status) VALUES(?,?,?,?,?,?,?,?,?)",
                (delta.event_id, delta.source_id, delta.target_id, delta.session_id,
                 delta.sequence, delta.payload["key"], delta.content_hash, encoded, "proposed"),
            )
        return ProposalReceipt(delta.event_id, delta.content_hash, "proposed", False)

    def validate(
        self, event_id: str, receipt: ValidationReceipt, *,
        verify_validator: Callable[[StateDelta, ValidationReceipt], bool] | None,
    ) -> ProposalReceipt:
        if not isinstance(receipt, ValidationReceipt):
            raise TypeError("ValidationReceipt required")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._load_proposal(db, event_id)
            delta = StateDelta.from_json(row[3])
            if receipt.proposal_hash != delta.content_hash or receipt.evidence_sha256 != delta.evidence.sha256():
                raise ConflictError("validation receipt not bound to exact proposal/evidence")
            if receipt.verifier_id == delta.source_id:
                raise AuthorizationError("source cannot self-verify")
            if not delta.issued_at_epoch <= receipt.evaluated_at_epoch <= self._now() + CLOCK_SKEW_SECONDS:
                raise ConflictError("validation timestamp outside proposal/clock window")
            _authorized(verify_validator, delta, receipt)
            encoded = receipt.to_json()
            if row[2] in ("validated", "committed"):
                if row[4] != encoded:
                    raise ConflictError("proposal already validated using different receipt")
                return ProposalReceipt(event_id, delta.content_hash, row[2], True)
            self._check_window(delta)
            db.execute(
                "UPDATE proposals SET status='validated', validation_json=? WHERE event_id=? AND status='proposed'",
                (encoded, event_id),
            )
        return ProposalReceipt(event_id, delta.content_hash, "validated", False)

    def _commit_receipt(self, db: sqlite3.Connection, event_id: str, already: bool) -> CommitReceipt:
        row = db.execute(
            "SELECT event_offset,content_hash,chain_hash,key_revision FROM events WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise LedgerIntegrityError("committed proposal has no event")
        return CommitReceipt(event_id, row[0], row[1], row[2], row[3], already)

    def commit(
        self, event_id: str, *,
        verify_validator: Callable[[StateDelta, ValidationReceipt], bool] | None,
        authorize_commit: Callable[[StateDelta, ValidationReceipt], bool] | None,
    ) -> CommitReceipt:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._load_proposal(db, event_id)
            delta = StateDelta.from_json(row[3])
            if row[2] == "committed":
                return self._commit_receipt(db, event_id, True)  # crash-before-ack readback
            if row[2] != "validated" or not row[4]:
                raise ConflictError("proposal requires external validation before commit")
            self._check_window(delta)
            validation = ValidationReceipt.from_json(row[4])
            if validation.proposal_hash != delta.content_hash or validation.evidence_sha256 != delta.evidence.sha256():
                raise ConflictError("validation/proposal evidence binding changed")
            _authorized(verify_validator, delta, validation)
            # Separate current host-owned grant/role/revocation lookup is
            # required at COMMIT time. A once-valid verifier receipt is NOT a
            # standing permission to mutate after the owner's grant expires.
            _authorized(authorize_commit, delta, validation)

            last_seq = db.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM events WHERE source_id=? AND session_id=?",
                (delta.source_id, delta.session_id),
            ).fetchone()[0]
            if delta.sequence != last_seq + 1:
                raise SequenceError(f"expected source/session sequence {last_seq + 1}")

            key = delta.payload["key"]
            previous = db.execute(
                "SELECT source_id,operation,key_revision FROM events "
                "WHERE target_id=? AND session_id=? AND payload_key=? "
                "ORDER BY key_revision DESC LIMIT 1",
                (delta.target_id, delta.session_id, key),
            ).fetchone()
            latest_rev = previous[2] if previous else 0
            if delta.expected_revision != latest_rev:
                raise ConflictError("state key revision CAS mismatch")
            # Tombstones retain the key owner. Reassignment requires a future
            # separately authorized protocol, not implicit post-delete theft.
            if previous and previous[0] != delta.source_id:
                raise ConflictError("state key owned by a different source")
            if delta.operation is StateOp.DELETE and (previous is None or previous[1] == StateOp.DELETE.value):
                raise ConflictError("cannot delete a missing/already-deleted key")

            head_offset, prev_hash = db.execute(
                "SELECT head_offset,head_hash FROM ledger_head WHERE id=1"
            ).fetchone()
            actual_last = db.execute(
                "SELECT event_offset,chain_hash FROM events ORDER BY event_offset DESC LIMIT 1"
            ).fetchone()
            if (actual_last or (0, GENESIS_HASH)) != (head_offset, prev_hash):
                raise LedgerIntegrityError("local chain head mismatch")

            validation_sha = _sha_text(row[4])
            next_revision = latest_rev + 1
            seq = db.execute("SELECT seq FROM sqlite_sequence WHERE name='events'").fetchone()
            offset = max(head_offset, seq[0] if seq else 0) + 1
            chain_hash = _chain_hash(prev_hash, row[3], validation_sha, offset, next_revision)
            db.execute(
                "INSERT INTO events(event_offset,event_id,source_id,target_id,session_id,sequence,"
                "payload_key,operation,key_revision,content_hash,event_json,validation_sha256,"
                "prev_hash,chain_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (offset, event_id, delta.source_id, delta.target_id, delta.session_id,
                 delta.sequence, key, delta.operation.value, next_revision,
                 delta.content_hash, row[3], validation_sha, prev_hash, chain_hash),
            )
            db.execute("UPDATE ledger_head SET head_offset=?,head_hash=? WHERE id=1",
                       (offset, chain_hash))
            db.execute(
                "UPDATE proposals SET status='committed',commit_offset=? WHERE event_id=?",
                (offset, event_id),
            )
            return CommitReceipt(event_id, offset, delta.content_hash,
                                 chain_hash, next_revision, False)

    def verify_chain(self) -> bool:
        try:
            with self._db() as db:
                rows = db.execute(
                    "SELECT event_offset,event_id,content_hash,event_json,validation_sha256,"
                    "prev_hash,chain_hash,key_revision FROM events ORDER BY event_offset"
                ).fetchall()
                previous = GENESIS_HASH
                last_offset = 0
                for offset, event_id, content_hash, encoded, validation_sha, prev, chain, revision in rows:
                    delta = StateDelta.from_json(encoded)
                    if delta.event_id != event_id or delta.content_hash != content_hash:
                        return False
                    proposal = db.execute(
                        "SELECT validation_json,status FROM proposals WHERE event_id=?", (event_id,)
                    ).fetchone()
                    if proposal is None or proposal[1] != "committed" or not proposal[0]:
                        return False
                    receipt = ValidationReceipt.from_json(proposal[0])
                    if receipt.proposal_hash != content_hash or receipt.evidence_sha256 != delta.evidence.sha256():
                        return False
                    if _sha_text(proposal[0]) != validation_sha or prev != previous:
                        return False
                    if chain != _chain_hash(prev, encoded, validation_sha, offset, revision):
                        return False
                    previous = chain
                    last_offset = offset
                local_head = db.execute(
                    "SELECT head_offset,head_hash FROM ledger_head WHERE id=1"
                ).fetchone()
                return local_head == (last_offset, previous)
        except (sqlite3.Error, SchemaError, ValueError, KeyError):
            return False

    def _require_chain(self) -> None:
        if not self.verify_chain():
            raise LedgerIntegrityError("local ledger hash chain check failed")

    def get_revision(self, target_id: str, session_id: str, key: str) -> int:
        _validate_key(key)
        self._require_chain()
        with self._db() as db:
            return db.execute(
                "SELECT COALESCE(MAX(key_revision),0) FROM events "
                "WHERE target_id=? AND session_id=? AND payload_key=?",
                (target_id, session_id, key),
            ).fetchone()[0]

    def project(self, target_id: str, session_id: str) -> dict[str, object]:
        """Return a newly allocated materialized view; never write state."""
        self._require_chain()
        state: dict[str, object] = {}
        with self._db() as db:
            rows = db.execute(
                "SELECT event_json FROM events WHERE target_id=? AND session_id=? "
                "ORDER BY event_offset", (target_id, session_id),
            ).fetchall()
        for (encoded,) in rows:
            delta = StateDelta.from_json(encoded)
            payload = delta.payload
            if delta.operation is StateOp.SET:
                state[payload["key"]] = payload["value"]
            else:
                state.pop(payload["key"], None)
        return state

    def events(self, target_id: str, session_id: str, *, after_offset: int = 0,
               limit: int = 100) -> tuple[CommittedEvent, ...]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_READ:
            raise ValueError("limit out of range")
        if not isinstance(after_offset, int) or after_offset < 0:
            raise ValueError("after_offset out of range")
        self._require_chain()
        with self._db() as db:
            rows = db.execute(
                "SELECT event_offset,key_revision,event_json,chain_hash FROM events "
                "WHERE target_id=? AND session_id=? AND event_offset>? "
                "ORDER BY event_offset LIMIT ?",
                (target_id, session_id, after_offset, limit),
            ).fetchall()
        return tuple(CommittedEvent(offset, rev, StateDelta.from_json(encoded), chain)
                     for offset, rev, encoded, chain in rows)
