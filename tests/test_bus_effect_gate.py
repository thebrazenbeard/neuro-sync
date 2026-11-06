"""Brokers sharing a local host-owned DB serialize use and grant mutations.

These are same-host SQLite mechanics, *not* network principal authentication,
OS privilege isolation, exactly-once external effects, or model behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest

from neurosync.bus.effect_gate import (
    ApprovedEffect, EffectAttemptFailed, HostOwnedEffectBroker,
)
from neurosync.bus.read_gate import HostBoundContext, ReadAccessDenied


class Ledger:
    def __init__(self, value=None):
        self.value = {"ref": "cap.synthetic", "inputs": [2, 3]} if value is None else value
        self.reads = []

    def project(self, target, session):
        self.reads.append((target, session))
        return {"capabilities.synthetic": self.value}


class HostOwnedEffectBrokerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "authority.sqlite3"
        self.context = HostBoundContext("host-operator", "local-qwen", "sess1")
        self.now = 100.0
        self.ledger = Ledger()
        self.effects = []

    def broker(self, handler=None, *, context=None):
        if handler is None:
            def handler(state):
                self.effects.append(state)
                return {"sum": sum(state["inputs"])}
        return HostOwnedEffectBroker(
            self.path, self.ledger, context=context or self.context,
            approved_handlers={"capabilities.synthetic": ApprovedEffect(
                handler, "a" * 64, "sha1:" + "b" * 40)},
            clock=lambda: self.now,
        )

    def test_effect_checks_current_grant_and_binds_execution_receipt(self):
        broker = self.broker()
        broker.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)
        result, receipt = broker.execute("grant1", "invoke1", "capabilities.synthetic")
        self.assertEqual(result, {"sum": 5})
        self.assertEqual(self.effects, [{"ref": "cap.synthetic", "inputs": [2, 3]}])
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["handler_artifact_sha256"], "a" * 64)
        self.assertEqual(receipt["source_git_oid"], "sha1:" + "b" * 40)
        self.assertEqual(receipt["artifact_binding_kind"], "host_asserted_unverified")
        expected_hash = hashlib.sha256(json.dumps(
            self.effects[0], ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(receipt["state_sha256"], expected_hash)
        self.assertEqual(broker.attempt("invoke1")["receipt_sha256"], receipt["receipt_sha256"])
        self.assertEqual(self.ledger.reads, [("local-qwen", "sess1")])
        # Inputs passed to the callback are JSON copies rather than live ledger storage.
        self.effects[0]["inputs"].append(99)
        self.assertEqual(self.ledger.value["inputs"], [2, 3])

    def test_revoked_and_expired_grants_never_invoke_handler(self):
        broker = self.broker()
        broker.grant("revoked", "capabilities.synthetic", expires_at_epoch=200)
        broker.revoke("revoked")
        with self.assertRaisesRegex(ReadAccessDenied, "current use grant denied"):
            broker.execute("revoked", "once", "capabilities.synthetic")
        self.assertIsNone(broker.attempt("once"))
        broker.grant("short", "capabilities.synthetic", expires_at_epoch=101)
        self.now = 101.0  # exact expiry excludes use
        with self.assertRaisesRegex(ReadAccessDenied, "current use grant denied"):
            broker.execute("short", "twice", "capabilities.synthetic")
        self.assertEqual(self.effects, [])
        self.assertEqual(self.ledger.reads, [])

    def test_replay_and_grant_id_reuse_denied_across_restart(self):
        broker = self.broker()
        broker.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)
        broker.execute("grant1", "once", "capabilities.synthetic")
        restarted = self.broker()
        with self.assertRaisesRegex(ReadAccessDenied, "invocation replay denied"):
            restarted.execute("grant1", "once", "capabilities.synthetic")
        with self.assertRaisesRegex(ReadAccessDenied, "grant ID already used"):
            restarted.grant("grant1", "capabilities.synthetic", expires_at_epoch=300)
        restarted.revoke("grant1")
        with self.assertRaises(ReadAccessDenied):
            restarted.execute("grant1", "new-invocation", "capabilities.synthetic")
        self.assertEqual(len(self.effects), 1)

    def test_wrong_scoped_identity_never_inherits_grant(self):
        owner = self.broker()
        owner.grant("owner-grant", "capabilities.synthetic", expires_at_epoch=200)
        forged = self.broker(context=HostBoundContext("different", "local-qwen", "sess1"))
        with self.assertRaises(ReadAccessDenied):
            forged.execute("owner-grant", "forged-use", "capabilities.synthetic")
        self.assertEqual(self.effects, [])
        self.assertEqual(self.ledger.reads, [])

    def test_unauthorized_caller_cannot_enumerate_handler_registry(self):
        broker = self.broker()
        messages = []
        for key in ("capabilities.synthetic", "capabilities.unknown"):
            with self.assertRaises(ReadAccessDenied) as caught:
                broker.execute("no-such-grant", "distinct-" + key.split(".")[-1], key)
            messages.append(str(caught.exception))
        self.assertEqual(messages, ["current use grant denied"] * 2)
        self.assertEqual(self.effects, [])
        self.assertEqual(self.ledger.reads, [])

    def test_different_context_cannot_revoke_or_read_receipt_even_with_known_ids(self):
        alice = self.broker()
        alice.grant("shared-grant", "capabilities.synthetic", expires_at_epoch=200)
        alice.execute("shared-grant", "shared-invocation", "capabilities.synthetic")
        bob = self.broker(context=HostBoundContext("bob", "other-target", "session-b"))
        self.assertIsNone(bob.attempt("shared-invocation"))
        with self.assertRaisesRegex(ReadAccessDenied, "grant unavailable"):
            bob.revoke("shared-grant")
        # Identical caller-supplied IDs are scoped, not a cross-context DoS.
        bob.grant("shared-grant", "capabilities.synthetic", expires_at_epoch=200)
        bob_result, _ = bob.execute("shared-grant", "shared-invocation",
                                    "capabilities.synthetic")
        self.assertEqual(bob_result, {"sum": 5})
        self.assertIsNotNone(alice.attempt("shared-invocation"))
        self.assertIsNotNone(bob.attempt("shared-invocation"))
        alice.revoke("shared-grant")
        self.assertEqual(len(self.effects), 2)

    def test_handler_effect_then_raise_is_consumed_not_safely_retried(self):
        def does_effect_then_crashes(state):
            self.effects.append(state)
            raise RuntimeError("external effect happened before exception")
        broker = self.broker(does_effect_then_crashes)
        broker.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)
        with self.assertRaisesRegex(EffectAttemptFailed, "invocation consumed"):
            broker.execute("grant1", "uncertain", "capabilities.synthetic")
        self.assertEqual(len(self.effects), 1)
        self.assertEqual(broker.attempt("uncertain")["status"], "raised")
        with self.assertRaisesRegex(ReadAccessDenied, "replay denied"):
            self.broker(does_effect_then_crashes).execute(
                "grant1", "uncertain", "capabilities.synthetic")
        self.assertEqual(len(self.effects), 1)

    def test_claimed_invocation_persists_as_crash_ambiguity(self):
        broker = self.broker()
        broker.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)
        # Simulate the durable first phase having committed, followed by a
        # process crash before the second phase reached the effect handler.
        with broker._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO attempts(invocation_id,grant_id,principal,target,session,"
                       "state_key,status,started,handler_artifact_sha256,source_git_oid) "
                       "VALUES(?,?,?,?,?,?,'claimed',?,?,?)", (
                "ambiguous", "grant1", "host-operator", "local-qwen", "sess1",
                "capabilities.synthetic", 100, "a" * 64, "sha1:" + "b" * 40))
            db.commit()
        self.assertEqual(self.broker().attempt("ambiguous")["status"], "claimed")
        with self.assertRaises(ReadAccessDenied):
            self.broker().execute("grant1", "ambiguous", "capabilities.synthetic")
        self.assertEqual(self.effects, [])

    def test_grant_mutation_serializes_across_separate_broker_connections(self):
        started = threading.Event()
        release = threading.Event()
        revoke_started = threading.Event()
        outcomes = {}

        def long_handler(state):
            started.set()
            if not release.wait(timeout=4):
                raise RuntimeError("test failed to release handler")
            self.effects.append(state)
            return {"sum": 5}

        executor = self.broker(long_handler)
        revoker = self.broker(long_handler)
        executor.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)

        def run_effect():
            try:
                outcomes["effect"] = executor.execute("grant1", "once", "capabilities.synthetic")
            except Exception as exc:
                outcomes["effect_error"] = exc

        def run_revoke():
            revoke_started.set()
            try:
                revoker.revoke("grant1")
                outcomes["revoked"] = True
            except Exception as exc:
                outcomes["revoke_error"] = exc

        a = threading.Thread(target=run_effect)
        b = threading.Thread(target=run_revoke)
        a.start()
        self.assertTrue(started.wait(timeout=3))
        b.start()
        self.assertTrue(revoke_started.wait(timeout=3))
        try:
            time.sleep(0.05)
            self.assertNotIn("revoked", outcomes)  # broker writer lock held by effect
        finally:
            release.set()
            a.join(timeout=4)
            b.join(timeout=4)
        self.assertFalse(a.is_alive())
        self.assertFalse(b.is_alive())
        self.assertNotIn("effect_error", outcomes)
        self.assertNotIn("revoke_error", outcomes)
        self.assertEqual(outcomes["effect"][1]["status"], "ok")
        self.assertTrue(outcomes["revoked"])
        self.assertEqual(len(self.effects), 1)
        with self.assertRaises(ReadAccessDenied):
            executor.execute("grant1", "after-revoke", "capabilities.synthetic")

    def test_concurrent_duplicate_request_only_one_handler_attempt(self):
        broker_a = self.broker()
        broker_b = self.broker()
        broker_a.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)
        barrier = threading.Barrier(3)
        outcomes = []
        lock = threading.Lock()

        def worker(broker):
            barrier.wait(timeout=4)
            try:
                outcome = broker.execute("grant1", "same-invocation",
                                         "capabilities.synthetic")[1]["status"]
            except ReadAccessDenied as exc:
                outcome = str(exc)
            with lock:
                outcomes.append(outcome)

        workers = [threading.Thread(target=worker, args=(broker,))
                   for broker in (broker_a, broker_b)]
        for worker_thread in workers:
            worker_thread.start()
        barrier.wait(timeout=4)
        for worker_thread in workers:
            worker_thread.join(timeout=4)
            self.assertFalse(worker_thread.is_alive())
        self.assertCountEqual(outcomes, ["ok", "invocation replay denied"])
        self.assertEqual(len(self.effects), 1)

    def test_invalid_host_inputs_fail_closed(self):
        broker = self.broker()
        for identifier in ("", "x/x", "__evil", 1, True):
            with self.subTest(identifier=identifier):
                with self.assertRaises(ReadAccessDenied):
                    broker.grant(identifier, "capabilities.synthetic", expires_at_epoch=200)
        with self.assertRaises(ValueError):
            broker.grant("grant1", "capabilities.synthetic", expires_at_epoch=float("inf"))
        with self.assertRaises(ReadAccessDenied):
            broker.grant("grant1", "capabilities.synthetic", expires_at_epoch=99)

    def test_ephemeral_or_empty_database_path_rejected_not_claimed_durable(self):
        for path in ("", ":memory:", "file::memory:?cache=shared", self.tmp.name):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "persistent"):
                HostOwnedEffectBroker(
                    path, self.ledger, context=self.context,
                    approved_handlers={"capabilities.synthetic": ApprovedEffect(
                        lambda value: value, "a" * 64, "sha1:" + "b" * 40)})

    def test_handler_cannot_reenter_same_broker_grant_mutation(self):
        owner = None
        def handler(_value):
            with self.assertRaisesRegex(ReadAccessDenied, "reentrant"):
                owner.revoke("grant1")
            return {"ok": True}
        owner = self.broker(handler)
        owner.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)
        result, receipt = owner.execute("grant1", "once", "capabilities.synthetic")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(receipt["status"], "ok")
        self.assertIsNotNone(owner.attempt("once"))

    def test_bound_plan_rejects_swapped_intent_and_handler_after_grant(self):
        owner = self.broker()
        intended = "1" * 64
        swapped = "2" * 64
        owner.grant("bound", "capabilities.synthetic", expires_at_epoch=200,
                    plan_sha256=intended)

        for label, digest in (("omitted", None), ("swapped", swapped)):
            with self.subTest(label=label), self.assertRaises(ReadAccessDenied):
                owner.execute("bound", label, "capabilities.synthetic",
                              plan_sha256=digest)
            self.assertIsNone(owner.attempt(label))
        replacement = HostOwnedEffectBroker(
            self.path, self.ledger, context=self.context,
            approved_handlers={"capabilities.synthetic": ApprovedEffect(
                lambda _state: self.effects.append("unexpected"),
                "c" * 64, "sha1:" + "b" * 40)},
            clock=lambda: self.now)
        with self.assertRaises(ReadAccessDenied):
            replacement.execute("bound", "changed-handler", "capabilities.synthetic",
                                plan_sha256=intended)
        self.assertIsNone(owner.attempt("changed-handler"))
        self.assertEqual(self.effects, [])

        result, receipt = owner.execute("bound", "authorized", "capabilities.synthetic",
                                        plan_sha256=intended)
        self.assertEqual(result, {"sum": 5})
        self.assertEqual(receipt["plan_sha256"], intended)
        self.assertEqual(receipt["intent_binding_kind"], "host_asserted_plan_and_handler")
        self.assertEqual(owner.attempt("authorized")["plan_sha256"], intended)

    def test_existing_authority_migrates_without_regrant_or_replay_reset(self):
        # Construct the pre-plan-binding tables, as in a persisted authority
        # created by the old broker. A migration must not erase claims.
        with sqlite3.connect(self.path) as db:
            db.executescript("""
                CREATE TABLE grants (
                  grant_id TEXT NOT NULL, principal TEXT NOT NULL,
                  target TEXT NOT NULL, session TEXT NOT NULL,
                  state_key TEXT NOT NULL, expires REAL NOT NULL,
                  revoked INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY(principal,target,session,grant_id));
                CREATE TABLE attempts (
                  invocation_id TEXT NOT NULL, grant_id TEXT NOT NULL,
                  principal TEXT NOT NULL, target TEXT NOT NULL,
                  session TEXT NOT NULL, state_key TEXT NOT NULL,
                  status TEXT NOT NULL, started REAL NOT NULL, finished REAL,
                  state_sha256 TEXT, result_sha256 TEXT,
                  handler_artifact_sha256 TEXT NOT NULL,
                  source_git_oid TEXT NOT NULL, receipt_sha256 TEXT,
                  PRIMARY KEY(principal,target,session,invocation_id));
            """)
            db.execute("INSERT INTO grants(grant_id,principal,target,session,state_key,expires) "
                       "VALUES(?,?,?,?,?,?)", (
                           "old", "host-operator", "local-qwen", "sess1",
                           "capabilities.synthetic", 200))
            db.execute("INSERT INTO attempts(invocation_id,grant_id,principal,target,session,"
                       "state_key,status,started,handler_artifact_sha256,source_git_oid) "
                       "VALUES(?,?,?,?,?,?,'claimed',?,?,?)", (
                           "old-claim", "old", "host-operator", "local-qwen", "sess1",
                           "capabilities.synthetic", 100, "a" * 64, "sha1:" + "b" * 40))
        first = self.broker()
        second = self.broker()  # repeat migration is idempotent
        self.assertEqual(first.attempt("old-claim")["status"], "claimed")
        self.assertIsNone(first.attempt("old-claim")["plan_sha256"])
        with self.assertRaisesRegex(ReadAccessDenied, "replay denied"):
            second.execute("old", "old-claim", "capabilities.synthetic")
        second.grant("new-bound", "capabilities.synthetic", expires_at_epoch=200,
                     plan_sha256="9" * 64)
        result, receipt = first.execute("new-bound", "new-use", "capabilities.synthetic",
                                        plan_sha256="9" * 64)
        self.assertEqual(result, {"sum": 5})
        self.assertEqual(receipt["plan_sha256"], "9" * 64)
        self.assertEqual(len(self.effects), 1)

    def test_bound_plan_digest_validation_and_unbound_legacy_separation(self):
        owner = self.broker()
        for invalid in ("", "A" * 64, "a" * 63, 0, True, [], {}):
            with self.subTest(invalid=invalid), self.assertRaises(ReadAccessDenied):
                owner.grant("invalid", "capabilities.synthetic", expires_at_epoch=200,
                            plan_sha256=invalid)
        owner.grant("legacy", "capabilities.synthetic", expires_at_epoch=200)
        with self.assertRaises(ReadAccessDenied):
            owner.execute("legacy", "cannot-upgrade", "capabilities.synthetic",
                          plan_sha256="9" * 64)
        self.assertIsNone(owner.attempt("cannot-upgrade"))
        result, receipt = owner.execute("legacy", "legacy-use", "capabilities.synthetic")
        self.assertEqual(result, {"sum": 5})
        self.assertNotIn("plan_sha256", receipt)
        self.assertNotIn("intent_binding_kind", receipt)


if __name__ == "__main__":
    unittest.main()
