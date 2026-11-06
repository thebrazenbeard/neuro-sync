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
        self.assertNotIn("failure_class", receipt)  # successful v1 receipt stays byte-compatible
        self.assertIsNone(broker.attempt("invoke1")["failure_class"])
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
        self.assertEqual(broker.attempt("uncertain")["failure_class"], "handler")
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

    def test_overdeep_host_state_denied_without_ambiguous_claim_or_effect(self):
        # A small JSON payload can still exceed the encoder recursion limit.
        # Reject it with a recorded denial, not an uncaught exception that
        # leaves a durable, unreconstructible `claimed` invocation.
        nested = {"value": 0}
        for _ in range(10_000):
            nested = [nested]
        self.ledger.value = nested
        broker = self.broker()
        broker.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)
        with self.assertRaisesRegex(ReadAccessDenied, "bounded JSON"):
            broker.execute("grant1", "too-deep", "capabilities.synthetic")
        self.assertEqual(broker.attempt("too-deep")["status"], "denied")
        self.assertEqual(self.effects, [])
        with self.assertRaisesRegex(ReadAccessDenied, "invocation replay denied"):
            self.broker().execute("grant1", "too-deep", "capabilities.synthetic")

    def test_host_ledger_exception_is_denied_before_handler_not_ambiguous(self):
        def unavailable_projection(_target, _session):
            raise RuntimeError("host ledger unavailable")

        self.ledger.project = unavailable_projection
        broker = self.broker()
        broker.grant("grant1", "capabilities.synthetic", expires_at_epoch=200)
        with self.assertRaisesRegex(ReadAccessDenied, "ledger projection unavailable"):
            broker.execute("grant1", "ledger-down", "capabilities.synthetic")
        self.assertEqual(broker.attempt("ledger-down")["status"], "denied")
        self.assertEqual(self.effects, [])

    def test_failure_class_separates_permission_denial_from_infrastructure(self):
        broker = self.broker()
        broker.grant("short", "capabilities.synthetic", expires_at_epoch=150)

        original_project = self.ledger.project
        def expires_while_projecting(target, session):
            state = original_project(target, session)
            self.now = 150.0  # expired after durable claim, before handler
            return state
        self.ledger.project = expires_while_projecting
        with self.assertRaisesRegex(ReadAccessDenied, "revoked or expired"):
            broker.execute("short", "permission-denial", "capabilities.synthetic")
        permission = broker.attempt("permission-denial")
        self.assertEqual(permission["status"], "denied")
        self.assertEqual(permission["failure_class"], "permission")

        self.now = 100.0
        broker.grant("outage", "capabilities.synthetic", expires_at_epoch=150)
        def unavailable(_target, _session):
            raise RuntimeError("ledger unavailable")
        self.ledger.project = unavailable
        with self.assertRaisesRegex(ReadAccessDenied, "ledger projection unavailable"):
            broker.execute("outage", "infrastructure-denial", "capabilities.synthetic")
        infrastructure = broker.attempt("infrastructure-denial")
        self.assertEqual(infrastructure["status"], "denied")
        self.assertEqual(infrastructure["failure_class"], "infrastructure")

        # A metrics consumer MUST distinguish these despite their shared
        # `status=denied`; an outage earns no permission-governance credit.
        countable = lambda record: (record["status"] == "denied" and
                                    record["failure_class"] == "permission")
        self.assertTrue(countable(permission))
        self.assertFalse(countable(infrastructure))
        self.assertEqual(self.effects, [])
        with self.assertRaisesRegex(ReadAccessDenied, "invocation replay denied"):
            broker.execute("short", "permission-denial", "capabilities.synthetic")

    def test_legacy_attempt_db_adds_failure_class_without_losing_history(self):
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE attempts ("
                       "invocation_id TEXT NOT NULL, grant_id TEXT NOT NULL, "
                       "principal TEXT NOT NULL, target TEXT NOT NULL, session TEXT NOT NULL, "
                       "state_key TEXT NOT NULL, status TEXT NOT NULL, "
                       "started REAL NOT NULL, finished REAL, state_sha256 TEXT, "
                       "result_sha256 TEXT, handler_artifact_sha256 TEXT NOT NULL, "
                       "source_git_oid TEXT NOT NULL, receipt_sha256 TEXT, "
                       "PRIMARY KEY (principal,target,session,invocation_id))")
            db.execute("INSERT INTO attempts (invocation_id,grant_id,principal,target,session,"
                       "state_key,status,started,handler_artifact_sha256,source_git_oid) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?)", (
                           "legacy-claim", "legacy-grant", "host-operator", "local-qwen",
                           "sess1", "capabilities.synthetic", "claimed", 100.0,
                           "a" * 64, "sha1:" + "b" * 40))
        opened = self.broker()
        self.assertEqual(opened.attempt("legacy-claim")["status"], "claimed")
        self.assertIsNone(opened.attempt("legacy-claim")["failure_class"])
        with self.assertRaisesRegex(ReadAccessDenied, "invocation replay denied"):
            opened.execute("legacy-grant", "legacy-claim", "capabilities.synthetic")


if __name__ == "__main__":
    unittest.main()
