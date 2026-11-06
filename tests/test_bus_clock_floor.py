"""Persisted, observed clock high-water in the host's v2 plan authority.

These source-level tests cannot prove elapsed time that was never observed,
host clock authenticity, or security against a direct SQLite writer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from neurosync.bus.effect_gate import ApprovedPlanEffect, HostOwnedEffectBroker
from neurosync.bus.read_gate import HostBoundContext, ReadAccessDenied


KEY = "clock.test"
ARTIFACT = "a" * 64
SOURCE = "sha1:" + "b" * 40


class Ledger:
    def __init__(self):
        self.raise_on_project = False

    def project(self, target, session):
        if self.raise_on_project:
            raise RuntimeError("injected projection failure")
        return {KEY: {"values": [1, 2]}}


class ClockFloorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "authority.sqlite3"
        self.clock = [100.0]
        self.clock_sequence = []
        self.ledger = Ledger()
        self.calls = []
        self.handler_enter = None
        self.handler_release = None
        self.plan = {"steps": [{"handle": "cap-a", "input": "initial"}],
                     "abstain": False}
        self.context = HostBoundContext("host", "receiver", "session")

    def broker(self, *, context=None):
        def handler(state, plan):
            if self.handler_enter is not None:
                self.handler_enter.set()
                if not self.handler_release.wait(3):
                    raise RuntimeError("test handler wait timed out")
            self.calls.append((state, plan))
            return {"ok": True}
        return HostOwnedEffectBroker(
            self.path, self.ledger, context=context or self.context,
            approved_handlers={},
            approved_plan_handlers={KEY: ApprovedPlanEffect(handler, ARTIFACT, SOURCE)},
            clock=self.read_clock,
        )

    def read_clock(self):
        if self.clock_sequence:
            return self.clock_sequence.pop(0)
        return self.clock[0]

    def high_water(self):
        with sqlite3.connect(str(self.path) + ".clock.sqlite3") as db:
            row = db.execute(
                "SELECT last_observed_epoch FROM clock_state WHERE singleton=1"
            ).fetchone()
        return row[0] if row else None

    @property
    def sidecar_path(self):
        return Path(str(self.path) + ".clock.sqlite3")

    def test_observed_expired_denial_blocks_revival_after_restart(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=200)
        self.assertEqual(self.high_water(), 100)
        self.clock[0] = 250
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("once", "expired", KEY, plan=self.plan)
        self.assertIsNone(owner.attempt("expired"))
        self.assertEqual(self.high_water(), 250,
                         "denied first-phase grant check must persist observation")
        self.clock[0] = 150
        restarted = self.broker()
        with self.assertRaisesRegex(ReadAccessDenied, "clock.*backward"):
            restarted.execute_plan("once", "revived", KEY, plan=self.plan)
        self.assertIsNone(restarted.attempt("revived"))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.high_water(), 250)

    def test_rollback_blocks_new_plan_grant_after_reopen(self):
        owner = self.broker()
        owner.grant_plan("old", KEY, plan=self.plan, expires_at_epoch=200)
        self.clock[0] = 250
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("old", "expired", KEY, plan=self.plan)
        self.clock[0] = 150
        with self.assertRaisesRegex(ReadAccessDenied, "clock.*backward"):
            self.broker().grant_plan("new", KEY, plan=self.plan, expires_at_epoch=300)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM clock_guarded_plan_grants").fetchone()[0], 1)
        self.assertEqual(self.calls, [])

    def test_same_or_forward_time_allows_normal_one_use_plan(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=300)
        self.clock[0] = 101
        result, receipt = self.broker().execute_plan("once", "invoke", KEY, plan=self.plan)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(receipt["version"], 2)
        self.assertEqual(self.broker().attempt("invoke")["status"], "ok")
        self.assertGreaterEqual(self.high_water(), 101)
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("once", "invoke-again", KEY, plan=self.plan)
        self.assertEqual(len(self.calls), 1)

    def test_observation_is_database_wide_across_contexts(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=200)
        self.clock[0] = 250
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("once", "expired", KEY, plan=self.plan)
        self.clock[0] = 150
        another = self.broker(context=HostBoundContext("other", "receiver", "session2"))
        with self.assertRaisesRegex(ReadAccessDenied, "clock.*backward"):
            another.grant_plan("different", KEY, plan=self.plan, expires_at_epoch=300)
        self.assertEqual(self.high_water(), 250)

    def test_revocation_always_works_under_clock_rollback(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=200)
        self.clock[0] = 250
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("once", "expired", KEY, plan=self.plan)
        self.clock[0] = 150
        self.broker().revoke("once")
        with sqlite3.connect(self.path) as db:
            revoked = db.execute(
                "SELECT revoked FROM clock_guarded_plan_grants WHERE grant_id='once'"
            ).fetchone()[0]
        self.assertEqual(revoked, 1)

    def test_old_plan_grant_is_inert_and_blocks_identifier_reuse(self):
        owner = self.broker()
        digest = hashlib.sha256(json.dumps(self.plan, sort_keys=True, ensure_ascii=False,
                                            allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO plan_grants(grant_id,principal,target,session,state_key,expires,"
                "plan_sha256,planned_state_sha256,plan_handler_artifact_sha256,"
                "plan_handler_source_git_oid) VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("historic", "host", "receiver", "session", KEY, 300, digest,
                 "c" * 64, ARTIFACT, SOURCE),
            )
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("historic", "old-attempt", KEY, plan=self.plan)
        with self.assertRaises(ReadAccessDenied):
            owner.grant_plan("historic", KEY, plan=self.plan, expires_at_epoch=300)
        self.assertIsNone(owner.attempt("old-attempt"))
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM plan_grants WHERE grant_id='historic'").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM clock_guarded_plan_grants WHERE grant_id='historic'").fetchone()[0], 0)

    def test_clock_invalid_and_callback_error_fail_closed(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=200)
        for bad in (float("nan"), None, "100"):
            self.clock[0] = bad
            with self.assertRaises(ReadAccessDenied):
                owner.execute_plan("once", "invalid", KEY, plan=self.plan)
            self.assertIsNone(owner.attempt("invalid"))
        self.assertEqual(self.high_water(), 100)
        self.assertEqual(self.calls, [])

    def test_concurrent_observation_denial_is_ordered_by_writer_lock(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=200)
        finished_expiry = threading.Event()
        failures = []

        def observe_expiry():
            self.clock[0] = 250
            try:
                owner.execute_plan("once", "expired", KEY, plan=self.plan)
            except ReadAccessDenied:
                pass
            else:
                failures.append("expired effect executed")
            finally:
                finished_expiry.set()

        def observe_rollback():
            if not finished_expiry.wait(3):
                failures.append("expiry observation did not finish")
                return
            self.clock[0] = 150
            try:
                self.broker().execute_plan("once", "revived", KEY, plan=self.plan)
            except ReadAccessDenied:
                pass
            else:
                failures.append("rolled-back clock revived expired grant")

        first = threading.Thread(target=observe_expiry)
        second = threading.Thread(target=observe_rollback)
        first.start(); second.start()
        first.join(5); second.join(5)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(self.high_water(), 250)
        self.assertEqual(self.calls, [])

    def test_grant_projection_exception_cannot_erase_clock_checkpoint(self):
        owner = self.broker()
        owner.grant_plan("old", KEY, plan=self.plan, expires_at_epoch=200)
        self.clock[0] = 250
        self.ledger.raise_on_project = True
        with self.assertRaisesRegex(RuntimeError, "injected projection failure"):
            owner.grant_plan("failed", KEY, plan=self.plan, expires_at_epoch=300)
        self.assertEqual(self.high_water(), 250)
        self.ledger.raise_on_project = False
        self.clock[0] = 150
        with self.assertRaisesRegex(ReadAccessDenied, "clock.*backward"):
            self.broker().execute_plan("old", "revive", KEY, plan=self.plan)
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.broker().attempt("revive"))

    def test_effect_projection_exception_cannot_erase_clock_checkpoint(self):
        owner = self.broker()
        owner.grant_plan("old", KEY, plan=self.plan, expires_at_epoch=200)
        owner.grant_plan("other", KEY, plan=self.plan, expires_at_epoch=300)
        # Claim-phase checkpoint/authorization at 150, then phase-two
        # checkpoint at 250. Its fallible ledger callback must not roll back
        # the committed 250 observation.
        self.clock[0] = 250
        self.clock_sequence = [150, 150]
        self.ledger.raise_on_project = True
        with self.assertRaisesRegex(RuntimeError, "injected projection failure"):
            owner.execute_plan("other", "other-attempt", KEY, plan=self.plan)
        self.assertEqual(self.high_water(), 250)
        self.assertEqual(owner.attempt("other-attempt")["status"], "claimed")
        self.ledger.raise_on_project = False
        self.clock[0] = 180
        with self.assertRaisesRegex(ReadAccessDenied, "clock.*backward"):
            self.broker().execute_plan("old", "revive", KEY, plan=self.plan)
        self.assertEqual(self.calls, [])

    def test_receipt_clock_failure_after_effect_cannot_erase_checkpoint(self):
        owner = self.broker()
        owner.grant_plan("old", KEY, plan=self.plan, expires_at_epoch=200)
        owner.grant_plan("other", KEY, plan=self.plan, expires_at_epoch=300)
        # Claim and effect checkpoints/authorization sample 150 and 250;
        # receipt time is invalid after the handler has been invoked. The
        # external effect remains ambiguous, but old approval cannot revive.
        self.clock[0] = None
        self.clock_sequence = [150, 150, 250, 250, 250]
        with self.assertRaises(Exception):
            owner.execute_plan("other", "other-attempt", KEY, plan=self.plan)
        self.assertEqual(self.high_water(), 250)
        self.assertEqual(len(self.calls), 1,
                         "the other handler ran before receipt failure")
        self.clock[0] = 180
        self.clock_sequence = []
        with self.assertRaisesRegex(ReadAccessDenied, "clock.*backward"):
            self.broker().execute_plan("old", "revive", KEY, plan=self.plan)
        self.assertEqual(len(self.calls), 1)

    def test_clock_jump_between_checkpoint_and_locked_grant_check_denied(self):
        owner = self.broker()
        self.clock[0] = 250
        self.clock_sequence = [100]
        with self.assertRaises(ReadAccessDenied):
            owner.grant_plan("new", KEY, plan=self.plan, expires_at_epoch=200)
        self.assertEqual(self.high_water(), 250)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM clock_guarded_plan_grants"
            ).fetchone()[0], 0)

    def test_claim_sql_abort_does_not_erase_later_observed_epoch(self):
        owner = self.broker()
        owner.grant_plan("old", KEY, plan=self.plan, expires_at_epoch=275)
        owner.grant_plan("other", KEY, plan=self.plan, expires_at_epoch=400)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER reject_claim BEFORE UPDATE ON clock_guarded_plan_grants "
                       "WHEN OLD.grant_id='other' BEGIN "
                       "SELECT RAISE(ABORT,'injected claim write failure'); END")
        self.clock[0] = 300
        self.clock_sequence = [150]
        with self.assertRaises(sqlite3.IntegrityError):
            owner.execute_plan("other", "blocked", KEY, plan=self.plan)
        self.assertEqual(self.high_water(), 300,
                         "separately committed time survives main SQL abort")
        self.clock[0] = 260
        with self.assertRaisesRegex(ReadAccessDenied, "clock.*backward"):
            self.broker().execute_plan("old", "revive", KEY, plan=self.plan)
        self.assertEqual(self.calls, [])

    def test_later_in_effect_epoch_survives_receipt_exception(self):
        owner = self.broker()
        owner.grant_plan("old", KEY, plan=self.plan, expires_at_epoch=275)
        owner.grant_plan("other", KEY, plan=self.plan, expires_at_epoch=400)
        # First durable pre-effect sample 250, then two *newer* samples 300
        # within the main writer txn, followed by an invalid receipt clock.
        # The original same-DB implementation rolled back those two 300s.
        self.clock[0] = None
        self.clock_sequence = [150, 150, 250, 300, 300]
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("other", "other-attempt", KEY, plan=self.plan)
        self.assertEqual(self.high_water(), 300)
        self.assertEqual(len(self.calls), 1)
        self.clock[0] = 260
        with self.assertRaisesRegex(ReadAccessDenied, "clock.*backward"):
            self.broker().execute_plan("old", "revive", KEY, plan=self.plan)
        self.assertEqual(len(self.calls), 1)

    def test_ready_missing_sidecar_denies_plan_but_allows_revocation(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=300)
        self.sidecar_path.unlink()
        self.clock[0] = 150
        with self.assertRaisesRegex(ReadAccessDenied, "sidecar unavailable"):
            self.broker().execute_plan("once", "attempt", KEY, plan=self.plan)
        self.assertFalse(self.sidecar_path.exists(), "READY cannot recreate sidecar")
        self.broker().revoke("once")

    def test_ready_corrupt_or_mismatched_sidecar_fails_closed(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=300)
        with sqlite3.connect(self.sidecar_path) as db:
            db.execute("UPDATE clock_state SET binding_uuid='incorrect' WHERE singleton=1")
        self.clock[0] = 150
        with self.assertRaisesRegex(ReadAccessDenied, "binding mismatch"):
            self.broker().execute_plan("once", "bad-marker", KEY, plan=self.plan)
        self.broker().revoke("once")

        # A malformed existing READY sidecar is not silently replaced.
        self.sidecar_path.write_bytes(b"not a SQLite database")
        with self.assertRaises(ReadAccessDenied):
            self.broker().grant_plan("new", KEY, plan=self.plan, expires_at_epoch=400)
        self.assertEqual(self.sidecar_path.read_bytes(), b"not a SQLite database")

    def test_null_floor_with_existing_v2_grants_does_not_reinitialize(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=300)
        with sqlite3.connect(self.sidecar_path) as db:
            db.execute("UPDATE clock_state SET last_observed_epoch=NULL WHERE singleton=1")
        self.clock[0] = 150
        with self.assertRaisesRegex(ReadAccessDenied, "high-water missing"):
            self.broker().execute_plan("once", "attempt", KEY, plan=self.plan)
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.high_water())

    def test_nonfinite_or_nonnumeric_persisted_floor_fails_closed(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=300)
        for invalid in (float("-inf"), float("inf"), "not-a-clock"):
            with sqlite3.connect(self.sidecar_path) as db:
                db.execute("UPDATE clock_state SET last_observed_epoch=? WHERE singleton=1",
                           (invalid,))
            self.clock[0] = 150
            with self.assertRaisesRegex(ReadAccessDenied, "invalid persisted clock"):
                self.broker().execute_plan("once", "attempt", KEY, plan=self.plan)
            self.assertEqual(self.calls, [])

    def test_missing_binding_with_existing_v2_grant_cannot_bootstrap(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=300)
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM plan_clock_binding")
        self.clock[0] = 150
        with self.assertRaisesRegex(ReadAccessDenied, "binding unavailable"):
            self.broker().execute_plan("once", "attempt", KEY, plan=self.plan)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM plan_clock_binding").fetchone()[0], 0)

    def test_known_limitation_same_uuid_sidecar_snapshot_rollback_can_revive(self):
        """Document a genuine non-guarantee; a paired custody witness is needed."""
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=200)
        valid_previous_sidecar = self.sidecar_path.read_bytes()
        self.clock[0] = 250
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("once", "expired", KEY, plan=self.plan)
        self.assertEqual(self.high_water(), 250)
        self.sidecar_path.write_bytes(valid_previous_sidecar)
        self.clock[0] = 150
        result, _ = self.broker().execute_plan("once", "revived", KEY, plan=self.plan)
        self.assertEqual(result, {"ok": True}, "KNOWN LIMITATION: sidecar snapshot rollback")
        self.assertEqual(len(self.calls), 1)

    def test_main_snapshot_rollback_must_not_replay_successfully_consumed_plan(self):
        """An unchanged sidecar must not let restoring MAIN revive a used plan."""
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=300)
        # SQLite uses rollback journals for this broker. This is a bounded
        # synthetic backup of the host-owned main DB, with no services.
        main_before_effect = self.path.read_bytes()
        owner.execute_plan("once", "first", KEY, plan=self.plan)
        self.assertEqual(len(self.calls), 1)
        high_water_before_restore = self.high_water()
        self.path.write_bytes(main_before_effect)
        self.assertEqual(self.high_water(), high_water_before_restore)
        with self.assertRaises(ReadAccessDenied):
            self.broker().execute_plan("once", "replayed", KEY, plan=self.plan)
        self.assertEqual(len(self.calls), 1, "restored main must not repeat effect")

    def test_existing_claimed_plan_is_migrated_into_sidecar_replay_journal(self):
        owner = self.broker()
        owner.grant_plan("legacy-consumed", KEY, plan=self.plan, expires_at_epoch=300)
        owner.execute_plan("legacy-consumed", "old-invocation", KEY, plan=self.plan)
        # Emulate an intact pre-journal READY authority. Code deploy must
        # discover already consumed grants, without replaying their effects.
        with sqlite3.connect(self.sidecar_path) as db:
            db.execute("DROP TABLE consumed_plan_grants")
            db.execute("ALTER TABLE clock_state DROP COLUMN consumption_journal_ready")
        restored = self.broker()
        with sqlite3.connect(self.sidecar_path) as db:
            row = db.execute("SELECT invocation_id FROM consumed_plan_grants "
                             "WHERE grant_id='legacy-consumed'").fetchone()
        self.assertEqual(row, ("old-invocation",))
        with self.assertRaises(ReadAccessDenied):
            restored.execute_plan("legacy-consumed", "replayed", KEY, plan=self.plan)
        self.assertEqual(len(self.calls), 1)

    def test_ready_journal_removed_after_consumption_is_not_recreated(self):
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=300)
        main_before_effect = self.path.read_bytes()
        owner.execute_plan("once", "first", KEY, plan=self.plan)
        self.assertEqual(len(self.calls), 1)
        self.path.write_bytes(main_before_effect)
        with sqlite3.connect(self.sidecar_path) as db:
            db.execute("DROP TABLE consumed_plan_grants")
        # Journal had been initialized: losing it must not reinitialize from
        # a potentially rolled-back main authority snapshot.
        with self.assertRaises(ReadAccessDenied):
            self.broker().execute_plan("once", "replayed", KEY, plan=self.plan)
        self.assertEqual(len(self.calls), 1)

    def test_sidecar_replay_journal_loss_fails_before_effect(self):
        owner = self.broker()
        owner.grant_plan("planned", KEY, plan=self.plan, expires_at_epoch=300)
        with sqlite3.connect(self.sidecar_path) as db:
            db.execute("DROP TABLE consumed_plan_grants")
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("planned", "attempt", KEY, plan=self.plan)
        self.assertEqual(self.calls, [], "missing journal must not run handler")

    def test_pending_zero_grant_bootstrap_can_finish_without_old_sidecar(self):
        self.broker()  # freshly bound empty authority
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE plan_clock_binding SET status='PENDING' WHERE singleton=1")
        self.sidecar_path.unlink()
        recovered = self.broker()
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute(
                "SELECT status FROM plan_clock_binding WHERE singleton=1"
            ).fetchone()[0], "READY")
        recovered.grant_plan("fresh", KEY, plan=self.plan, expires_at_epoch=300)
        self.assertEqual(self.high_water(), 100)

    def test_effect_writer_and_other_clock_reader_serialize_main_first(self):
        self.handler_enter = threading.Event()
        self.handler_release = threading.Event()
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=self.plan, expires_at_epoch=550)
        failures = []
        completed = threading.Event()

        def execute_held_effect():
            try:
                owner.execute_plan("once", "held", KEY, plan=self.plan)
            except Exception as exc:
                failures.append(f"first: {exc!r}")

        def observe_from_other_broker():
            try:
                self.clock[0] = 500
                self.broker().grant_plan("other", KEY, plan=self.plan, expires_at_epoch=600)
            except Exception as exc:
                failures.append(f"second: {exc!r}")
            finally:
                completed.set()

        first = threading.Thread(target=execute_held_effect)
        second = threading.Thread(target=observe_from_other_broker)
        first.start()
        self.assertTrue(self.handler_enter.wait(3))
        second.start()
        self.assertFalse(completed.wait(.05), "second writer should wait for effect lock")
        self.handler_release.set()
        first.join(4); second.join(4)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(self.high_water(), 500)


if __name__ == "__main__":
    unittest.main()
