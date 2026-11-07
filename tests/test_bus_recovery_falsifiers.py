"""Recovery *non-guarantees* of the source-only two-file effect authority.

These are executable falsifiers, not claims that restoring old database files
is safe. Every file is an isolated disposable SQLite fixture: no host DB,
model inference, external effect or production backup is touched.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from neurosync.bus.effect_gate import ApprovedPlanEffect, HostOwnedEffectBroker
from neurosync.bus.read_gate import HostBoundContext, ReadAccessDenied


KEY = "recovery.falsifier"
PLAN = {"steps": [{"handle": "cap-a", "input": "initial"}], "abstain": False}


class Ledger:
    def project(self, target, session):
        return {KEY: {"values": [1, 2]}}


class RecoveryFalsifierTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "authority.sqlite3"
        self.sidecar = Path(str(self.path) + ".clock.sqlite3")
        self.clock = [100.0]
        self.calls = []
        self.context = HostBoundContext("host", "receiver", "session")
        self.ledger = Ledger()

    def broker(self):
        def handler(state, plan):
            self.calls.append((state, plan))
            return {"ok": True}

        return HostOwnedEffectBroker(
            self.path,
            self.ledger,
            context=self.context,
            approved_handlers={},
            approved_plan_handlers={
                KEY: ApprovedPlanEffect(handler, "a" * 64, "sha1:" + "b" * 40)
            },
            clock=lambda: self.clock[0],
        )

    def high_water(self):
        with sqlite3.connect(self.sidecar) as db:
            row = db.execute(
                "SELECT last_observed_epoch FROM clock_state WHERE singleton=1"
            ).fetchone()
        return row[0] if row else None

    def test_known_limitation_full_paired_snapshot_revives_expired_grant(self):
        """KNOWN LIMITATION: valid historical pair bypasses observed time."""
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=PLAN, expires_at_epoch=200)
        old_main = self.path.read_bytes()
        old_sidecar = self.sidecar.read_bytes()

        self.clock[0] = 250
        with self.assertRaises(ReadAccessDenied):
            owner.execute_plan("once", "expired", KEY, plan=PLAN)
        self.assertEqual(self.high_water(), 250)
        self.assertEqual(self.calls, [])

        # Simulate a *whole-pair* historical restore, with no independent
        # witness surviving outside the rollback set. This is deliberately
        # an unsafe operation on a disposable fixture, never a recovery guide.
        self.path.write_bytes(old_main)
        self.sidecar.write_bytes(old_sidecar)
        self.clock[0] = 150
        result, receipt = self.broker().execute_plan(
            "once", "historical-restore", KEY, plan=PLAN
        )
        self.assertEqual(result, {"ok": True}, "KNOWN LIMITATION: paired rollback")
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(len(self.calls), 1)

    def test_known_limitation_main_only_restore_replays_consumed_invocation(self):
        """KNOWN LIMITATION: intact clock floor cannot prove consumption."""
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=PLAN, expires_at_epoch=300)
        old_main = self.path.read_bytes()

        self.clock[0] = 101
        first, receipt = owner.execute_plan("once", "invoke", KEY, plan=PLAN)
        self.assertEqual(first, {"ok": True})
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(len(self.calls), 1)
        retained_floor = self.high_water()
        self.assertEqual(retained_floor, 101)

        # Restoring ONLY main loses consumed flag *and* invocation receipt;
        # sidecar remains at 101, and forward wall time 102 is legitimate.
        self.path.write_bytes(old_main)
        self.assertEqual(self.high_water(), retained_floor)
        self.clock[0] = 102
        second, second_receipt = self.broker().execute_plan(
            "once", "invoke", KEY, plan=PLAN
        )
        self.assertEqual(second, {"ok": True}, "KNOWN LIMITATION: replay")
        self.assertEqual(second_receipt["status"], "ok")
        self.assertEqual(len(self.calls), 2, "same grant and invocation executed twice")

    def test_retained_main_revocation_survives_sidecar_only_restore(self):
        """CONTROL: old sidecar cannot erase revocation in retained main."""
        owner = self.broker()
        owner.grant_plan("once", KEY, plan=PLAN, expires_at_epoch=300)
        old_sidecar = self.sidecar.read_bytes()
        self.clock[0] = 150
        owner.grant_plan("other", KEY, plan=PLAN, expires_at_epoch=300)
        self.assertEqual(self.high_water(), 150)
        owner.revoke("once")

        self.sidecar.write_bytes(old_sidecar)
        self.clock[0] = 151
        with self.assertRaisesRegex(ReadAccessDenied, "current use grant denied"):
            self.broker().execute_plan("once", "revoked", KEY, plan=PLAN)
        self.assertEqual(self.calls, [])
        with sqlite3.connect(self.path) as db:
            revoked = db.execute(
                "SELECT revoked FROM clock_guarded_plan_grants WHERE grant_id='once'"
            ).fetchone()[0]
        self.assertEqual(revoked, 1)


if __name__ == "__main__":
    unittest.main()
