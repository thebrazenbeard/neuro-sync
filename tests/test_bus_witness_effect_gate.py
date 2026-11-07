"""Source-only v3 witnessed broker composition; no external custody qualification."""

from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from neurosync.bus.authorization_witness import InProcessAuthorizationWitness
from neurosync.bus.effect_gate import ApprovedEffect, ApprovedPlanEffect, HostOwnedEffectBroker
from neurosync.bus.read_gate import HostBoundContext, ReadAccessDenied
from neurosync.bus.witness_effect_gate import SourceOnlyWitnessedPlanBroker


KEY = "capabilities.synthetic"
ARTIFACT = "a" * 64
SOURCE = "sha1:" + "b" * 40


class Ledger:
    def __init__(self):
        self.value = {"inputs": [2, 3]}

    def project(self, target, session):
        return {KEY: self.value}


class WitnessedBrokerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "authority.sqlite3"
        self.ledger = Ledger()
        self.context = HostBoundContext("host", "receiver", "session")
        self.now = 100.0
        self.effects: list[tuple[object, object]] = []
        self.plan = {"steps": [{"handle": "cap", "input": "initial"}], "abstain": False}
        self.witness = InProcessAuthorizationWitness("authority-A", 1)

    def broker(self, witness=None, *, use_default_witness=True, context=None):
        if witness is None and use_default_witness:
            witness = self.witness

        def handler(state, plan):
            self.effects.append((state, plan))
            return {"value": sum(state["inputs"])}

        return SourceOnlyWitnessedPlanBroker(
            self.db_path, self.ledger, context=context or self.context,
            approved_handlers={KEY: ApprovedEffect(lambda state: state, ARTIFACT, SOURCE)},
            approved_plan_handlers={KEY: ApprovedPlanEffect(handler, ARTIFACT, SOURCE)},
            clock=lambda: self.now,
            required_keys={KEY},
            authority_id="authority-A", generation=1,
            witness=witness, explicit_source_only=True,
        )

    def test_source_only_v3_grant_claim_handler_once_and_legacy_denied(self):
        broker = self.broker()
        for operation in (
            lambda: broker.grant("old", KEY, expires_at_epoch=200),
            lambda: broker.grant_plan("old", KEY, plan=self.plan, expires_at_epoch=200),
            lambda: broker.execute("old", "legacy1", KEY),
            lambda: broker.execute_plan("old", "legacy2", KEY, plan=self.plan),
        ):
            with self.assertRaisesRegex(ReadAccessDenied, "witness"):
                operation()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        with sqlite3.connect(self.db_path) as db:
            self.assertIsNone(db.execute("SELECT 1 FROM grants WHERE grant_id='v3'").fetchone())
            self.assertIsNone(db.execute("SELECT 1 FROM plan_grants WHERE grant_id='v3'").fetchone())
            self.assertIsNone(db.execute("SELECT 1 FROM clock_guarded_plan_grants WHERE grant_id='v3'").fetchone())
            self.assertIsNotNone(db.execute("SELECT 1 FROM v3_witness_plan_grants WHERE grant_id='v3'").fetchone())
        result, receipt = broker.execute_witnessed_plan("v3", "invocation", KEY, plan=self.plan)
        self.assertEqual(result, {"value": 5})
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(len(self.effects), 1)
        with self.assertRaises(ReadAccessDenied):
            broker.execute_witnessed_plan("v3", "second", KEY, plan=self.plan)
        self.assertEqual(len(self.effects), 1)

    def test_unavailable_witness_and_snapshot_restore_fail_before_handler(self):
        broker = self.broker()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        with sqlite3.connect(self.db_path) as db:
            original = db.execute("SELECT seq,digest FROM v3_witness_binding WHERE singleton=1").fetchone()
        broker.execute_witnessed_plan("v3", "first", KEY, plan=self.plan)
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE v3_witness_binding SET seq=?,digest=? WHERE singleton=1", original)
            db.execute("UPDATE v3_witness_plan_grants SET claimed_invocation_id=NULL WHERE grant_id='v3'")
            db.commit()
        with self.assertRaises(ReadAccessDenied):
            broker.execute_witnessed_plan("v3", "replay", KEY, plan=self.plan)
        self.assertEqual(len(self.effects), 1)

    def test_main_db_snapshot_restore_with_live_witness_fails_closed(self):
        broker = self.broker()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        snapshot_path = self.db_path.with_name("old-main-only.sqlite3")
        with sqlite3.connect(self.db_path) as active, sqlite3.connect(snapshot_path) as backup:
            active.backup(backup)
        broker.execute_witnessed_plan("v3", "first", KEY, plan=self.plan)
        shutil.copyfile(snapshot_path, self.db_path)
        with self.assertRaisesRegex(ReadAccessDenied, "mismatch"):
            broker.execute_witnessed_plan("v3", "first", KEY, plan=self.plan)
        self.assertEqual(len(self.effects), 1)

    def test_expiry_denial_observed_by_witness_survives_clock_sidecar_restore(self):
        broker = self.broker()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        clock_path = Path(str(self.db_path) + ".clock.sqlite3")
        old_sidecar = clock_path.with_name("old-time-sidecar.sqlite3")
        shutil.copyfile(clock_path, old_sidecar)
        before = self.witness.tip.sequence
        self.now = 250
        with self.assertRaises(ReadAccessDenied):
            broker.execute_witnessed_plan("v3", "expired", KEY, plan=self.plan)
        self.assertGreater(self.witness.tip.sequence, before)
        # The hostile restore rewinds ONLY the rollbackable clock sidecar.
        # The separate live witness retains the later clock observation.
        shutil.copyfile(old_sidecar, clock_path)
        self.now = 150
        with self.assertRaisesRegex(ReadAccessDenied, "clock"):
            broker.execute_witnessed_plan("v3", "resurrected", KEY, plan=self.plan)
        self.assertEqual(self.effects, [])

    def test_phase_clock_resampling_cannot_hide_expiry_from_witness(self):
        broker = self.broker()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        clock_path = Path(str(self.db_path) + ".clock.sqlite3")
        old_sidecar = clock_path.with_name("old-clock-for-resample.sqlite3")
        shutil.copyfile(clock_path, old_sidecar)
        # One witnessed sample in claim phase, a later witnessed sample in
        # handler phase; no separate raw authority-time read in between.
        clock_values = iter([100.0, 250.0])
        broker._clock = lambda: next(clock_values)
        with self.assertRaises(ReadAccessDenied):
            broker.execute_witnessed_plan("v3", "expired-between-phases", KEY,
                                          plan=self.plan)
        self.assertEqual(self.witness.events[-1].kind, "clock")
        self.assertEqual(self.witness.events[-1].observed_epoch, 250.0)
        self.assertEqual(self.effects, [])
        shutil.copyfile(old_sidecar, clock_path)
        broker._clock = lambda: 150.0
        with self.assertRaisesRegex(ReadAccessDenied, "clock"):
            broker.execute_witnessed_plan("v3", "resurrected", KEY, plan=self.plan)
        self.assertEqual(self.effects, [])

    def test_claim_witness_succeeds_but_sql_commit_fails_freezes(self):
        broker = self.broker()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        after_issue = self.witness.tip
        original_persist = broker._persist_tip

        def fail_main_cursor_commit(db, cursor):
            if cursor.sequence > after_issue.sequence:
                raise sqlite3.OperationalError("synthetic main write failure")
            return original_persist(db, cursor)

        broker._persist_tip = fail_main_cursor_commit
        with self.assertRaises(sqlite3.OperationalError):
            broker.execute_witnessed_plan("v3", "unknown", KEY, plan=self.plan)
        self.assertEqual(self.effects, [])
        self.assertEqual(self.witness.tip.sequence, after_issue.sequence + 1)
        fresh = self.broker()
        with self.assertRaisesRegex(ReadAccessDenied, "mismatch"):
            fresh.execute_witnessed_plan("v3", "second", KEY, plan=self.plan)
        self.assertEqual(self.effects, [])

    def test_revoke_and_cross_context_reject_use(self):
        broker = self.broker()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        alien = self.broker(context=HostBoundContext("other", "receiver", "session"))
        with self.assertRaises(ReadAccessDenied):
            alien.execute_witnessed_plan("v3", "alien", KEY, plan=self.plan)
        broker.revoke_witnessed_plan("v3")
        self.assertEqual([event.kind for event in self.witness.events
                          if event.kind != "clock"], ["issue", "revoke"])
        with self.assertRaises(ReadAccessDenied):
            broker.execute_witnessed_plan("v3", "claimed", KEY, plan=self.plan)
        self.assertEqual(self.effects, [])

    def test_external_witness_advance_during_projection_denies_before_handler(self):
        broker = self.broker()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        project = self.ledger.project

        def intervening_projection(target, session):
            with sqlite3.connect(self.db_path) as db:
                db.row_factory = sqlite3.Row
                row = db.execute("SELECT * FROM v3_witness_plan_grants WHERE grant_id='v3'").fetchone()
            self.witness.revoke(self.witness.tip, broker._bound_grant(row), at_epoch=self.now)
            return project(target, session)

        self.ledger.project = intervening_projection
        with self.assertRaisesRegex(ReadAccessDenied, "mismatch"):
            broker.execute_witnessed_plan("v3", "invoke", KEY, plan=self.plan)
        self.assertEqual(self.effects, [])
        # Only the external witness tip advanced; normal v3 operations now
        # halt until an authorized and separately witnessed recovery exists.
        with self.assertRaisesRegex(ReadAccessDenied, "mismatch"):
            broker.revoke_witnessed_plan("v3")

    def test_missing_provider_blocks_every_v3_path(self):
        with self.assertRaisesRegex(ReadAccessDenied, "before bootstrap"):
            self.broker(use_default_witness=False)
        self.assertFalse(self.db_path.exists())
        self.assertFalse(Path(str(self.db_path) + ".clock.sqlite3").exists())
        class BrokenWitness:
            tip = "wrong"
        with self.assertRaisesRegex(ReadAccessDenied, "before bootstrap"):
            self.broker(witness=BrokenWitness())
        self.assertFalse(self.db_path.exists())
        broker = self.broker()
        broker._v3_witness = None  # Simulate provider loss AFTER bootstrap.
        for action in (
            lambda: broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200),
            lambda: broker.execute_witnessed_plan("v3", "invoke", KEY, plan=self.plan),
            lambda: broker.revoke_witnessed_plan("v3"),
        ):
            with self.assertRaisesRegex(ReadAccessDenied, "witness"):
                action()
        self.assertEqual(self.effects, [])

    def test_explicit_source_only_policy_is_not_authenticated_host_policy(self):
        with self.assertRaisesRegex(ReadAccessDenied, "SOURCE ONLY"):
            SourceOnlyWitnessedPlanBroker(
                self.db_path, self.ledger, context=self.context,
                approved_handlers={}, approved_plan_handlers={},
                required_keys={KEY}, authority_id="authority-A", generation=1,
                witness=self.witness, explicit_source_only=False)
        broker = self.broker()
        broker.grant_witnessed_plan("v3", KEY, plan=self.plan, expires_at_epoch=200)
        clone = InProcessAuthorizationWitness("authority-A", 1)
        self.assertEqual(clone.tip.sequence, 0)
        self.assertGreater(self.witness.tip.sequence, 1)
        # A different, unprotected legacy broker remains a bypass to any
        # preexisting V2 grant. A host must enforce policy *outside* this code.
        legacy = HostOwnedEffectBroker(
            self.db_path, self.ledger, context=self.context,
            approved_handlers={KEY: ApprovedEffect(lambda state: state, ARTIFACT, SOURCE)},
            approved_plan_handlers={}, clock=lambda: self.now)
        legacy.grant("legacy", KEY, expires_at_epoch=200)
        self.assertIsNotNone(legacy.execute("legacy", "legacy-call", KEY))
        with self.assertRaises(ReadAccessDenied):
            broker.execute("legacy", "blocked", KEY)


if __name__ == "__main__":
    unittest.main()
