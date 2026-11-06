"""Plan-bound effect grants: source-level host mechanics, not OS authentication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from neurosync.bus.effect_gate import (
    ApprovedEffect, ApprovedPlanEffect, EffectAttemptFailed, HostOwnedEffectBroker,
)
from neurosync.bus.read_gate import HostBoundContext, ReadAccessDenied


KEY = "capabilities.synthetic"
ARTIFACT = "a" * 64
SOURCE = "sha1:" + "b" * 40


def independently_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class MutableProjection:
    def __init__(self):
        self.value = {"inputs": [2, 3], "origin": "fixture"}

    def project(self, target, session):
        return {KEY: self.value}


class PlanBoundEffectTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "authority.sqlite3"
        self.context = HostBoundContext("host", "receiver", "session")
        self.ledger = MutableProjection()
        self.now = 100.0
        self.effects = []
        self.plan = {"steps": [{"handle": "cap-one", "input": "initial"}],
                     "abstain": False}

    def broker(self, *, descriptor=None, context=None):
        if descriptor is None:
            def plan_handler(state, plan):
                self.effects.append((state, plan))
                return {"value": sum(state["inputs"]), "handle": plan["steps"][0]["handle"]}
            descriptor = ApprovedPlanEffect(plan_handler, ARTIFACT, SOURCE)
        return HostOwnedEffectBroker(
            self.path, self.ledger, context=context or self.context,
            approved_handlers={KEY: ApprovedEffect(lambda state: state, ARTIFACT, SOURCE)},
            approved_plan_handlers={KEY: descriptor}, clock=lambda: self.now,
        )

    def test_one_approved_invocation_replay_denied_with_durable_receipt(self):
        broker = self.broker()
        broker.grant_plan("grant", KEY, plan=self.plan, expires_at_epoch=200)
        result, receipt = broker.execute_plan("grant", "invoke", KEY, plan=dict(self.plan))
        self.assertEqual(result, {"value": 5, "handle": "cap-one"})
        self.assertEqual(receipt["version"], 2)
        self.assertEqual(receipt["plan_sha256"], independently_hash(self.plan))
        self.assertEqual(receipt["state_sha256"], independently_hash(self.ledger.value))
        self.assertEqual(receipt["status"], "ok")
        self.assertNotIn("failure_class", receipt)
        self.assertIsNone(broker.attempt("invoke")["failure_class"])
        self.assertEqual(self.broker().attempt("invoke")["plan_sha256"], receipt["plan_sha256"])
        self.assertEqual(len(self.effects), 1)
        # Handler sees frozen copies, never mutable caller-owned values.
        self.effects[0][0]["inputs"].append(99)
        self.effects[0][1]["steps"][0]["handle"] = "mutated-in-handler"
        self.assertEqual(self.ledger.value["inputs"], [2, 3])
        self.assertEqual(self.plan["steps"][0]["handle"], "cap-one")
        with self.assertRaisesRegex(ReadAccessDenied, "replay"):
            self.broker().execute_plan("grant", "invoke", KEY, plan=self.plan)
        self.assertEqual(len(self.effects), 1)

    def test_plan_grant_consumed_with_distinct_invocation_ids(self):
        broker = self.broker()
        broker.grant_plan("one-use", KEY, plan=self.plan, expires_at_epoch=200)
        _, first = broker.execute_plan("one-use", "invoke-one", KEY, plan=self.plan)
        self.assertEqual(first["status"], "ok")
        with self.assertRaises(ReadAccessDenied):
            broker.execute_plan("one-use", "invoke-two", KEY, plan=self.plan)
        self.assertIsNone(broker.attempt("invoke-two"))
        self.assertEqual(len(self.effects), 1)

    def test_wrong_plan_and_mutated_caller_plan_fail_before_handler(self):
        broker = self.broker()
        broker.grant_plan("grant", KEY, plan=self.plan, expires_at_epoch=200)
        alternate = {"steps": [{"handle": "cap-two", "input": "initial"}], "abstain": False}
        with self.assertRaises(ReadAccessDenied):
            broker.execute_plan("grant", "mismatch", KEY, plan=alternate)
        self.assertIsNone(broker.attempt("mismatch"))
        self.plan["steps"][0]["handle"] = "changed-since-approval"
        with self.assertRaises(ReadAccessDenied):
            broker.execute_plan("grant", "mutated", KEY, plan=self.plan)
        self.assertIsNone(broker.attempt("mutated"))
        self.assertEqual(self.effects, [])

    def test_canonical_reordered_plan_accepted_and_receipt_hash_recomputes(self):
        broker = self.broker()
        broker.grant_plan("grant", KEY, plan=self.plan, expires_at_epoch=200)
        reordered = {"abstain": False,
                     "steps": [{"input": "initial", "handle": "cap-one"}]}
        result, receipt = broker.execute_plan("grant", "canonical", KEY, plan=reordered)
        self.assertEqual(result["handle"], "cap-one")
        self.assertEqual(receipt["plan_sha256"], independently_hash(reordered))
        receipt_without_hash = dict(receipt)
        self.assertEqual(receipt_without_hash.pop("receipt_sha256"),
                         independently_hash(receipt_without_hash))

    def test_unbounded_or_non_json_plan_never_granted_or_executed(self):
        broker = self.broker()
        extremely_deep = []
        nested = extremely_deep
        for _ in range(15000):
            child = []
            nested.append(child)
            nested = child
        cycle = []
        cycle.append(cycle)
        for invalid in ({"n": float("nan")}, {"n": float("inf")},
                        {"cyclic": object()}, {"huge": "x" * 1_000_000},
                        extremely_deep, cycle, list(range(10_000))):
            with self.assertRaises(ReadAccessDenied):
                broker.grant_plan("bad", KEY, plan=invalid, expires_at_epoch=200)
        broker.grant_plan("valid", KEY, plan=self.plan, expires_at_epoch=200)
        for number, invalid in enumerate(({"n": float("nan")},
                                          {"n": float("-inf")}, {"opaque": object()})):
            with self.assertRaises(ReadAccessDenied):
                broker.execute_plan("valid", f"invalid-{number}", KEY, plan=invalid)
            self.assertIsNone(broker.attempt(f"invalid-{number}"))
        self.assertEqual(self.effects, [])

    def test_scope_components_and_reused_grant_id_stay_denied(self):
        broker = self.broker()
        broker.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=200)
        # A legacy broker binary has only `grants` and cannot see plan_grants.
        with sqlite3.connect(self.path) as db:
            visible = db.execute("SELECT 1 FROM grants WHERE grant_id='once'").fetchone()
            self.assertIsNone(visible)
            self.assertIsNotNone(db.execute(
                "SELECT 1 FROM plan_grants WHERE grant_id='once'").fetchone())
        with self.assertRaises(ReadAccessDenied):
            broker.grant("once", KEY, expires_at_epoch=200)
        with self.assertRaises(ReadAccessDenied):
            broker.execute("once", "legacy-rollback", KEY)
        broker.revoke("once")
        with self.assertRaises(ReadAccessDenied):
            broker.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=200)
        self.assertIsNone(broker.attempt("legacy-rollback"))
        broker.grant_plan("correct", KEY, plan=self.plan, expires_at_epoch=200)
        with self.assertRaises(ReadAccessDenied):
            broker.grant("correct", KEY, expires_at_epoch=200)
        for context in (HostBoundContext("host", "different", "session"),
                        HostBoundContext("host", "receiver", "different")):
            alien = self.broker(context=context)
            with self.assertRaises(ReadAccessDenied):
                alien.execute_plan("correct", "alien", KEY, plan=self.plan)
            self.assertIsNone(alien.attempt("alien"))
        self.assertEqual(self.effects, [])

    def test_no_legacy_mode_downgrade(self):
        broker = self.broker()
        broker.grant("legacy", KEY, expires_at_epoch=200)
        broker.grant_plan("plan", KEY, plan=self.plan, expires_at_epoch=200)
        with self.assertRaises(ReadAccessDenied):
            broker.execute("plan", "try-legacy", KEY)
        with self.assertRaises(ReadAccessDenied):
            broker.execute_plan("legacy", "try-plan", KEY, plan=self.plan)
        self.assertIsNone(broker.attempt("try-legacy"))
        self.assertIsNone(broker.attempt("try-plan"))
        self.assertEqual(self.effects, [])

    def test_changed_state_denied_with_consumed_claim(self):
        broker = self.broker()
        broker.grant_plan("plan", KEY, plan=self.plan, expires_at_epoch=200)
        self.ledger.value["inputs"] = [40, 60]
        with self.assertRaises(ReadAccessDenied):
            broker.execute_plan("plan", "stale", KEY, plan=self.plan)
        attempt = broker.attempt("stale")
        self.assertEqual(attempt["status"], "denied")
        self.assertEqual(attempt["failure_class"], "infrastructure")
        self.assertEqual(attempt["plan_sha256"], independently_hash(self.plan))
        self.assertEqual(self.effects, [])

    def test_pathological_state_denied_before_grant_and_consume(self):
        broker = self.broker()
        deep_state = []
        node = deep_state
        for _ in range(5000):
            nxt = []
            node.append(nxt)
            node = nxt
        self.ledger.value = deep_state
        with self.assertRaises(ReadAccessDenied):
            broker.grant_plan("deep", KEY, plan=self.plan, expires_at_epoch=200)
        self.ledger.value = {"inputs": [2, 3], "origin": "fixture"}
        broker.grant_plan("normal", KEY, plan=self.plan, expires_at_epoch=200)
        self.ledger.value = deep_state
        with self.assertRaises(ReadAccessDenied):
            broker.execute_plan("normal", "pathological", KEY, plan=self.plan)
        self.assertEqual(broker.attempt("pathological")["status"], "denied")
        self.assertEqual(broker.attempt("pathological")["failure_class"], "infrastructure")
        self.assertEqual(self.effects, [])

    def test_revoked_expired_scope_and_changed_handler_denied(self):
        broker = self.broker()
        broker.grant_plan("revoked", KEY, plan=self.plan, expires_at_epoch=200)
        broker.revoke("revoked")
        with self.assertRaises(ReadAccessDenied):
            broker.execute_plan("revoked", "revoke-invoke", KEY, plan=self.plan)
        broker.grant_plan("expires", KEY, plan=self.plan, expires_at_epoch=101)
        self.now = 101
        with self.assertRaises(ReadAccessDenied):
            broker.execute_plan("expires", "expires-invoke", KEY, plan=self.plan)
        self.now = 100
        broker.grant_plan("owner", KEY, plan=self.plan, expires_at_epoch=200)
        foreign = self.broker(context=HostBoundContext("foreign", "receiver", "session"))
        with self.assertRaises(ReadAccessDenied):
            foreign.execute_plan("owner", "foreign-invoke", KEY, plan=self.plan)
        changed = self.broker(descriptor=ApprovedPlanEffect(
            lambda state, plan: {"forged": True}, "c" * 64, SOURCE))
        with self.assertRaises(ReadAccessDenied):
            changed.execute_plan("owner", "changed-handler", KEY, plan=self.plan)
        self.assertEqual(self.effects, [])

    def test_effect_then_exception_is_consumed(self):
        def handler(state, plan):
            self.effects.append((state, plan))
            raise RuntimeError("effect may already have happened")
        effect = ApprovedPlanEffect(handler, ARTIFACT, SOURCE)
        broker = self.broker(descriptor=effect)
        broker.grant_plan("grant", KEY, plan=self.plan, expires_at_epoch=200)
        with self.assertRaises(EffectAttemptFailed):
            broker.execute_plan("grant", "invoke", KEY, plan=self.plan)
        self.assertEqual(broker.attempt("invoke")["status"], "raised")
        self.assertEqual(broker.attempt("invoke")["failure_class"], "handler")
        with self.assertRaises(ReadAccessDenied):
            self.broker(descriptor=effect).execute_plan("grant", "invoke", KEY, plan=self.plan)
        self.assertEqual(len(self.effects), 1)

    def test_post_claim_projection_outage_is_not_permission_denial(self):
        broker = self.broker()
        broker.grant_plan("grant", KEY, plan=self.plan, expires_at_epoch=200)
        def broken_project(_target, _session):
            raise RuntimeError("local ledger unavailable")
        self.ledger.project = broken_project
        with self.assertRaisesRegex(ReadAccessDenied, "projection unavailable"):
            broker.execute_plan("grant", "outage", KEY, plan=self.plan)
        attempt = broker.attempt("outage")
        self.assertEqual(attempt["status"], "denied")
        self.assertEqual(attempt["failure_class"], "infrastructure")
        self.assertEqual(attempt["plan_sha256"], independently_hash(self.plan))
        self.assertEqual(self.effects, [])

    def test_post_claim_expiry_is_permission_denial(self):
        broker = self.broker()
        broker.grant_plan("grant", KEY, plan=self.plan, expires_at_epoch=101)
        original_project = self.ledger.project
        def advancing_project(target, session):
            value = original_project(target, session)
            self.now = 101.0
            return value
        self.ledger.project = advancing_project
        with self.assertRaisesRegex(ReadAccessDenied, "revoked or expired"):
            broker.execute_plan("grant", "expired-after-claim", KEY, plan=self.plan)
        attempt = broker.attempt("expired-after-claim")
        self.assertEqual(attempt["status"], "denied")
        self.assertEqual(attempt["failure_class"], "permission")
        self.assertEqual(attempt["plan_sha256"], independently_hash(self.plan))
        self.assertEqual(self.effects, [])

    def test_writer_lock_blocks_revocation_until_handler_returns(self):
        entered = threading.Event()
        release = threading.Event()
        outcomes = []
        def handler(state, plan):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test handler timed out")
            self.effects.append((state, plan))
            return {"ok": True}
        effect = ApprovedPlanEffect(handler, ARTIFACT, SOURCE)
        broker = self.broker(descriptor=effect)
        broker.grant_plan("grant", KEY, plan=self.plan, expires_at_epoch=200)

        def invoke():
            try:
                outcomes.append(broker.execute_plan("grant", "attempt", KEY, plan=self.plan))
            except Exception as exc:
                outcomes.append(exc)

        effect_thread = threading.Thread(target=invoke)
        effect_thread.start()
        self.assertTrue(entered.wait(3), "handler never began")
        revoke_finished = threading.Event()
        revoke_started = threading.Event()
        def revoke():
            revoke_started.set()
            broker.revoke("grant")
            revoke_finished.set()
        revoker = threading.Thread(target=revoke)
        revoker.start()
        try:
            self.assertTrue(revoke_started.wait(3), "revoker never began")
            self.assertFalse(revoke_finished.wait(0.1), "revocation raced past writer lock")
        finally:
            release.set()
            effect_thread.join(timeout=5)
            revoker.join(timeout=5)
        self.assertFalse(effect_thread.is_alive())
        self.assertFalse(revoker.is_alive())
        self.assertTrue(revoke_finished.is_set())
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], tuple)
        self.assertEqual(outcomes[0][1]["status"], "ok")
        self.assertEqual(len(self.effects), 1)

    def test_parallel_constructor_migration_is_idempotent(self):
        gate = threading.Barrier(3)
        errors = []
        def construct():
            try:
                gate.wait(timeout=5)
                self.broker()
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=construct) for _ in range(2)]
        for thread in threads:
            thread.start()
        gate.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIsNotNone(self.broker())

    def test_legacy_database_migrates_without_destroying_history(self):
        # These are the exact legacy tables as created on main before plan mode.
        with sqlite3.connect(self.path) as db:
            db.executescript("""
                CREATE TABLE grants (
                    grant_id TEXT NOT NULL, principal TEXT NOT NULL,
                    target TEXT NOT NULL, session TEXT NOT NULL,
                    state_key TEXT NOT NULL, expires REAL NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)),
                    PRIMARY KEY (principal,target,session,grant_id)
                );
                CREATE TABLE attempts (
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
            db.execute("INSERT INTO grants(grant_id,principal,target,session,state_key,expires,revoked) "
                       "VALUES('old','host','receiver','session',?,200,1)", (KEY,))
            db.execute("INSERT INTO attempts(invocation_id,grant_id,principal,target,session,state_key,status,started,handler_artifact_sha256,source_git_oid) "
                       "VALUES('old-attempt','old','host','receiver','session',?,'claimed',100,?,?)",
                       (KEY, ARTIFACT, SOURCE))
        broker = self.broker()
        self.assertEqual(broker.attempt("old-attempt")["status"], "claimed")
        with self.assertRaises(ReadAccessDenied):
            broker.execute("old", "old-retry", KEY)
        broker.grant_plan("new", KEY, plan=self.plan, expires_at_epoch=200)
        result, receipt = self.broker().execute_plan("new", "new-call", KEY, plan=self.plan)
        self.assertEqual(result["value"], 5)
        self.assertEqual(receipt["version"], 2)
        self.assertEqual(self.broker().attempt("old-attempt")["status"], "claimed")
        self.assertEqual(self.broker().attempt("new-call")["status"], "ok")


if __name__ == "__main__":
    unittest.main()
