"""SOURCE-ONLY v3 broker composition model; never a qualified host boundary.

The in-process witness can be reconstructed with a matching genesis. This
class exists to test event ordering, cursor comparisons and legacy API
isolation, NOT to provide rollback resistance or production authorization.
Host deployment requires independent witness custody, authenticated provider
identity, anti-fork protection, trusted time and operator crash recovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import sqlite3
from typing import Any

from .authorization_witness import (
    AuthorizationWitness, WitnessCursor, WitnessDenied, WitnessGrant,
    require_witness,
)
from .effect_gate import (
    ApprovedEffect, ApprovedPlanEffect, EffectAttemptFailed,
    HostOwnedEffectBroker, _identifier, _key, _plan_encoded,
)
from .read_gate import HostBoundContext, ReadAccessDenied


class SourceOnlyWitnessedPlanBroker(HostOwnedEffectBroker):
    """An opt-in, explicitly untrusted v3 source integration exercise.

    ``required_keys`` is supplied by a host OUTSIDE the rollbackable database.
    It locks down all v1/v2 entrypoints for those keys even if old rows exist.
    If the host omits that policy (or runs older code), protection vanishes.

    ``explicit_source_only`` MUST be literal True. No provider is authenticated
    by this module; even a convincing witness instance may be a forgery.
    """

    def __init__(self, database_path, ledger, *, context: HostBoundContext,
                 approved_handlers: Mapping[str, ApprovedEffect],
                 approved_plan_handlers: Mapping[str, ApprovedPlanEffect],
                 required_keys: set[str] | frozenset[str], authority_id: str,
                 generation: int, witness: AuthorizationWitness | None,
                 explicit_source_only: bool, clock=None):
        if explicit_source_only is not True:
            raise ReadAccessDenied("v3 requires a qualified external witness; SOURCE ONLY")
        if type(required_keys) not in (set, frozenset) or not required_keys:
            raise ValueError("nonempty externally configured v3 required keys needed")
        self._required_keys = frozenset(_key(value) for value in required_keys)
        self._v3_authority_id = _identifier(authority_id)
        if type(generation) is not int or generation < 1:
            raise ValueError("invalid v3 authority generation")
        self._v3_generation = generation
        # A missing/malformed source provider must not even bootstrap a new
        # main SQLite/clock sidecar. This structural check is NOT provider
        # authentication; the explicit SOURCE-only mode remains mandatory.
        try:
            require_witness(witness)
        except WitnessDenied as exc:
            raise ReadAccessDenied("v3 witness unavailable or invalid before bootstrap") from exc
        self._v3_witness = witness
        kwargs = {"context": context, "approved_handlers": approved_handlers,
                  "approved_plan_handlers": approved_plan_handlers}
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(database_path, ledger, **kwargs)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS v3_witness_binding (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    authority_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    digest TEXT NOT NULL
                );
                -- An old binary has no query path to this separate table.
                CREATE TABLE IF NOT EXISTS v3_witness_plan_grants (
                    grant_id TEXT NOT NULL, principal TEXT NOT NULL,
                    target TEXT NOT NULL, session TEXT NOT NULL,
                    state_key TEXT NOT NULL, expires REAL NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)),
                    plan_sha256 TEXT NOT NULL,
                    planned_state_sha256 TEXT NOT NULL,
                    plan_handler_artifact_sha256 TEXT NOT NULL,
                    plan_handler_source_git_oid TEXT NOT NULL,
                    claimed_invocation_id TEXT,
                    PRIMARY KEY(principal,target,session,grant_id)
                );
            """)

    def _block_legacy(self, state_key: str):
        if state_key in self._required_keys:
            raise ReadAccessDenied("witness-required key denied by legacy broker API")

    def grant(self, grant_id, state_key, *, expires_at_epoch):
        self._block_legacy(state_key)
        return super().grant(grant_id, state_key, expires_at_epoch=expires_at_epoch)

    def grant_plan(self, grant_id, state_key, *, plan, expires_at_epoch):
        self._block_legacy(state_key)
        return super().grant_plan(grant_id, state_key, plan=plan,
                                  expires_at_epoch=expires_at_epoch)

    def execute(self, grant_id, invocation_id, state_key):
        self._block_legacy(state_key)
        return super().execute(grant_id, invocation_id, state_key)

    def execute_plan(self, grant_id, invocation_id, state_key, *, plan):
        self._block_legacy(state_key)
        return super().execute_plan(grant_id, invocation_id, state_key, plan=plan)

    def _require_key(self, state_key: str):
        if _key(state_key) not in self._required_keys:
            raise ReadAccessDenied("state key is not in host v3-required policy")

    def _witness_and_cursor(self, db: sqlite3.Connection):
        """Check exact MAIN cursor under writer lock before each mutation.

        No reset/reconciliation occurs after a witness-ahead failed DB commit.
        The bootstrap below is *not* independently non-rollbackable: a fresh
        same-generation in-process witness may accept a restored empty MAIN.
        """
        if not db.in_transaction:
            raise ReadAccessDenied("main authority writer required")
        try:
            witness = require_witness(self._v3_witness)
            tip = witness.tip
        except (WitnessDenied, AttributeError, TypeError) as exc:
            raise ReadAccessDenied("v3 witness unavailable or invalid") from exc
        binding = db.execute("SELECT * FROM v3_witness_binding WHERE singleton=1").fetchone()
        if binding is None:
            if (tip.sequence != 0 or db.execute(
                    "SELECT 1 FROM v3_witness_plan_grants LIMIT 1").fetchone()):
                raise ReadAccessDenied("v3 witness authority missing; recovery required")
            db.execute("INSERT INTO v3_witness_binding VALUES(1,?,?,?,?)", (
                self._v3_authority_id, self._v3_generation,
                tip.sequence, tip.event_sha256))
            return witness, tip
        if (binding["authority_id"] != self._v3_authority_id or
                binding["generation"] != self._v3_generation or
                binding["seq"] != tip.sequence or
                binding["digest"] != tip.event_sha256):
            raise ReadAccessDenied("v3 witness/main cursor mismatch; HALT")
        return witness, tip

    @staticmethod
    def _persist_tip(db: sqlite3.Connection, cursor: WitnessCursor):
        db.execute("UPDATE v3_witness_binding SET seq=?, digest=? WHERE singleton=1", (
            cursor.sequence, cursor.event_sha256))

    def _observe_v3_time(self):
        """Record each sampled epoch in witness+MAIN before grant decisions.

        Even a rejected/expired operation must advance the independent clock
        history, or restoring only the v2 sidecar could resurrect old grants.
        If witness append succeeds but MAIN commit fails, later operations
        reject the unmatched cursor instead of retrying or resetting it.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            witness, tip = self._witness_and_cursor(db)
            now = self._plan_now(db)
            try:
                event = witness.observe_time(tip, at_epoch=now)
            except WitnessDenied as exc:
                raise ReadAccessDenied("v3 witness denied clock observation") from exc
            self._persist_tip(db, event.cursor)
            db.commit()
        return now

    def _bound_grant(self, row: sqlite3.Row) -> WitnessGrant:
        return WitnessGrant(
            authority_id=self._v3_authority_id,
            generation=self._v3_generation,
            principal_id=row["principal"], target_id=row["target"],
            session_id=row["session"], grant_id=row["grant_id"],
            state_key=row["state_key"], plan_sha256=row["plan_sha256"],
            planned_state_sha256=row["planned_state_sha256"],
            handler_artifact_sha256=row["plan_handler_artifact_sha256"],
            handler_source_git_oid=row["plan_handler_source_git_oid"],
            expires_at_epoch=row["expires"],
        )

    def _v3_row(self, db, grant_id):
        return db.execute("SELECT * FROM v3_witness_plan_grants WHERE grant_id=? "
                          "AND principal=? AND target=? AND session=?", (
            grant_id, self._context.principal_id,
            self._context.target_id, self._context.session_id)).fetchone()

    def grant_witnessed_plan(self, grant_id: str, state_key: str, *,
                             plan: Any, expires_at_epoch: float):
        self._assert_not_handler()
        grant_id, state_key = _identifier(grant_id), _key(state_key)
        self._require_key(state_key)
        handler = self._plan_handlers.get(state_key)
        if handler is None:
            raise ReadAccessDenied("v3 plan handler is not approved")
        plan_sha = sha256(_plan_encoded(plan)).hexdigest()
        witnessed_now = self._observe_v3_time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            projection = self._ledger.project(self._context.target_id, self._context.session_id)
            if type(projection) is not dict or state_key not in projection:
                raise ReadAccessDenied("v3 state unavailable")
            state_sha = sha256(_plan_encoded(projection[state_key])).hexdigest()
            # This authorization decision must use the exact clock value
            # already appended to the witness. Resampling via _plan_now here
            # could observe an expiry which the witness never records.
            now = witnessed_now
            for table in ("grants", "plan_grants", "clock_guarded_plan_grants",
                          "v3_witness_plan_grants"):
                if db.execute(f"SELECT 1 FROM {table} WHERE grant_id=? AND principal=? "
                              "AND target=? AND session=?", (
                    grant_id, self._context.principal_id,
                    self._context.target_id, self._context.session_id,
                )).fetchone():
                    raise ReadAccessDenied("v3 grant id already used")
            witness, tip = self._witness_and_cursor(db)
            try:
                grant = WitnessGrant(
                    authority_id=self._v3_authority_id,
                    generation=self._v3_generation,
                    principal_id=self._context.principal_id,
                    target_id=self._context.target_id,
                    session_id=self._context.session_id,
                    grant_id=grant_id, state_key=state_key,
                    plan_sha256=plan_sha, planned_state_sha256=state_sha,
                    handler_artifact_sha256=handler.handler_artifact_sha256,
                    handler_source_git_oid=handler.source_git_oid,
                    expires_at_epoch=expires_at_epoch,
                )
                issued = witness.issue(tip, grant, at_epoch=now)
            except (WitnessDenied, ValueError) as exc:
                raise ReadAccessDenied("v3 witness denied grant") from exc
            db.execute("INSERT INTO v3_witness_plan_grants "
                       "(grant_id,principal,target,session,state_key,expires,"
                       "plan_sha256,planned_state_sha256,plan_handler_artifact_sha256,"
                       "plan_handler_source_git_oid) VALUES(?,?,?,?,?,?,?,?,?,?)", (
                grant_id, self._context.principal_id, self._context.target_id,
                self._context.session_id, state_key, grant.expires_at_epoch,
                plan_sha, state_sha, handler.handler_artifact_sha256,
                handler.source_git_oid,
            ))
            self._persist_tip(db, issued.cursor)
            db.commit()

    def revoke_witnessed_plan(self, grant_id: str):
        self._assert_not_handler()
        grant_id = _identifier(grant_id)
        witnessed_now = self._observe_v3_time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            witness, tip = self._witness_and_cursor(db)
            row = self._v3_row(db, grant_id)
            if row is None or row["revoked"] or row["state_key"] not in self._required_keys:
                raise ReadAccessDenied("v3 grant unavailable")
            now = witnessed_now
            try:
                event = witness.revoke(tip, self._bound_grant(row), at_epoch=now)
            except WitnessDenied as exc:
                raise ReadAccessDenied("v3 witness denied revoke") from exc
            db.execute("UPDATE v3_witness_plan_grants SET revoked=1 WHERE grant_id=? "
                       "AND principal=? AND target=? AND session=?", (
                grant_id, self._context.principal_id,
                self._context.target_id, self._context.session_id))
            self._persist_tip(db, event.cursor)
            db.commit()

    def execute_witnessed_plan(self, grant_id: str, invocation_id: str, state_key: str,
                               *, plan: Any):
        self._assert_not_handler()
        grant_id, invocation_id, state_key = (
            _identifier(grant_id), _identifier(invocation_id), _key(state_key))
        self._require_key(state_key)
        encoded = _plan_encoded(plan)
        plan_sha = sha256(encoded).hexdigest()
        detached_plan = json.loads(encoded.decode("utf-8"))
        witnessed_now = self._observe_v3_time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            witness, tip = self._witness_and_cursor(db)
            now = witnessed_now
            scope = (invocation_id, self._context.principal_id,
                     self._context.target_id, self._context.session_id)
            if db.execute("SELECT 1 FROM attempts WHERE invocation_id=? AND principal=? "
                          "AND target=? AND session=?", scope).fetchone():
                raise ReadAccessDenied("invocation replay denied")
            approved = self._plan_handlers.get(state_key)
            row = self._v3_row(db, grant_id)
            if (row is None or row["state_key"] != state_key or row["revoked"] or
                    row["claimed_invocation_id"] is not None or now >= row["expires"] or
                    approved is None or row["plan_sha256"] != plan_sha or
                    row["plan_handler_artifact_sha256"] != approved.handler_artifact_sha256 or
                    row["plan_handler_source_git_oid"] != approved.source_git_oid):
                raise ReadAccessDenied("v3 current grant denied")
            try:
                claimed = witness.reserve_claim(tip, self._bound_grant(row),
                                                 invocation_id, at_epoch=now)
            except WitnessDenied as exc:
                raise ReadAccessDenied("v3 witness denied claim") from exc
            db.execute("UPDATE v3_witness_plan_grants SET claimed_invocation_id=? "
                       "WHERE grant_id=? AND principal=? AND target=? AND session=?", (
                invocation_id, grant_id, self._context.principal_id,
                self._context.target_id, self._context.session_id))
            db.execute("INSERT INTO attempts(invocation_id,grant_id,principal,target,session,"
                       "state_key,status,started,handler_artifact_sha256,source_git_oid,"
                       "plan_sha256) VALUES(?,?,?,?,?,?,'claimed',?,?,?,?)", (
                invocation_id, grant_id, self._context.principal_id,
                self._context.target_id, self._context.session_id, state_key,
                now, approved.handler_artifact_sha256, approved.source_git_oid, plan_sha))
            self._persist_tip(db, claimed.cursor)
            db.commit()

        witnessed_now = self._observe_v3_time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._witness_and_cursor(db)
                row = self._v3_row(db, grant_id)
                projection = self._ledger.project(self._context.target_id,
                                                  self._context.session_id)
                if type(projection) is not dict or state_key not in projection:
                    raise ReadAccessDenied("v3 projected state unavailable")
                encoded_state = _plan_encoded(projection[state_key])
                state_sha = sha256(encoded_state).hexdigest()
                now = witnessed_now
                if (row is None or row["revoked"] or now >= row["expires"] or
                        row["claimed_invocation_id"] != invocation_id or
                        row["plan_sha256"] != plan_sha or
                        row["planned_state_sha256"] != state_sha):
                    raise ReadAccessDenied("v3 grant revoked, expired or state changed")
                # A projection callback may be a separate actor. It must not
                # advance witness history behind the broker and still have
                # its output reach an effect callback. This is only a narrow
                # source check, NOT an atomic lease over external custody.
                self._witness_and_cursor(db)
                self._in_handler.active = True
                try:
                    result = approved.handler(json.loads(encoded_state.decode("utf-8")),
                                              detached_plan)
                    result_sha = sha256(_plan_encoded(result)).hexdigest()
                    receipt = self._record(db, invocation_id, status="ok",
                                           finished=witnessed_now,
                                           state_hash=state_sha, result_hash=result_sha)
                    db.commit()
                    return result, receipt
                except Exception as exc:
                    self._record(db, invocation_id, status="raised",
                                 finished=witnessed_now, state_hash=state_sha)
                    db.commit()
                    raise EffectAttemptFailed("v3 effect attempt failed; claim consumed") from exc
                finally:
                    self._in_handler.active = False
            except ReadAccessDenied:
                if db.in_transaction:
                    self._record(db, invocation_id, status="denied",
                                 finished=witnessed_now)
                    db.commit()
                raise
