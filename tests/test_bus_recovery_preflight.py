"""Read-only recovery diagnostics; compatibility is deliberately not freshness."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neurosync.bus.effect_gate import ApprovedPlanEffect, HostOwnedEffectBroker
from neurosync.bus.recovery_preflight import inspect_clock_pair
from neurosync.bus.read_gate import HostBoundContext


KEY = "recovery.test"


class Ledger:
    def project(self, target, session):
        return {KEY: {"value": 1}}


class PairPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "authority.sqlite3"
        self.sidecar = Path(str(self.path) + ".clock.sqlite3")
        broker = HostOwnedEffectBroker(
            self.path, Ledger(), context=HostBoundContext("host", "receiver", "session"),
            approved_handlers={}, approved_plan_handlers={
                KEY: ApprovedPlanEffect(lambda state, plan: {"ok": True},
                                        "a" * 64, "sha1:" + "b" * 40)
            }, clock=lambda: 250.0)
        broker.grant_plan("one", KEY, plan={"steps": [], "abstain": True},
                          expires_at_epoch=500.0)

    def _change_floor(self, value):
        with sqlite3.connect(self.sidecar) as db:
            db.execute("UPDATE clock_state SET last_observed_epoch=? WHERE singleton=1",
                       (value,))

    def test_compatible_pair_is_not_claimed_fresh(self):
        report = inspect_clock_pair(self.path)
        self.assertEqual(report.classification, "STRUCTURALLY_COMPATIBLE_UNATTESTED")
        self.assertEqual(report.authority_effect, "NO_INTENTIONAL_AUTHORITY_DB_WRITE")
        self.assertEqual(report.last_observed_epoch, 250.0)
        self.assertTrue(report.has_v2_grants)
        self.assertIn("historical_same_uuid_restore_not_detectable", report.findings)
        self.assertEqual(json.loads(report.to_json())["last_observed_epoch"], 250)

    def test_valid_historical_sidecar_remains_undetectable(self):
        # Both files are structurally valid, but sidecar floor 100 could be
        # a malicious/accidental historical restore of an observed 250.
        self._change_floor(100.0)
        report = inspect_clock_pair(self.path)
        self.assertEqual(report.classification, "STRUCTURALLY_COMPATIBLE_UNATTESTED")
        self.assertEqual(report.last_observed_epoch, 100.0)
        self.assertIn("historical_same_uuid_restore_not_detectable", report.findings)

    def test_missing_sidecar_blocks_without_creating_files(self):
        self.sidecar.unlink()
        report = inspect_clock_pair(self.path)
        self.assertEqual(report.classification, "BLOCKED")
        self.assertFalse(self.sidecar.exists())

    def test_missing_main_blocks_without_creating_files(self):
        self.path.unlink()
        report = inspect_clock_pair(self.path)
        self.assertEqual(report.classification, "BLOCKED")
        self.assertFalse(self.path.exists())

    def test_mismatched_sidecar_binding_blocks(self):
        with sqlite3.connect(self.sidecar) as db:
            db.execute("UPDATE clock_state SET binding_uuid='wrong' WHERE singleton=1")
        report = inspect_clock_pair(self.path)
        self.assertEqual(report.classification, "BLOCKED")

    def test_incomplete_real_broker_schema_blocks(self):
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TABLE clock_guarded_plan_grants")
            db.execute("CREATE TABLE clock_guarded_plan_grants(grant_id TEXT)")
        report = inspect_clock_pair(self.path)
        self.assertEqual(report.classification, "BLOCKED")
        self.assertIn("required schema columns missing", " ".join(report.findings))

    def test_wal_mode_rejected_without_creating_sqlite_auxiliary_files(self):
        for component in (self.path, self.sidecar):
            with self.subTest(component=component.name):
                connection = sqlite3.connect(component)
                try:
                    self.assertEqual(connection.execute(
                        "PRAGMA journal_mode=WAL").fetchone()[0], "wal")
                finally:
                    connection.close()
                before = {path.name for path in self.path.parent.iterdir()}
                report = inspect_clock_pair(self.path)
                after = {path.name for path in self.path.parent.iterdir()}
                self.assertEqual(report.classification, "BLOCKED")
                self.assertEqual(after, before, "diagnostic created WAL/SHM files")
                connection = sqlite3.connect(component)
                try:
                    connection.execute("PRAGMA journal_mode=DELETE")
                finally:
                    connection.close()

    def test_concurrent_wal_switch_cannot_return_compatible(self):
        # Interleave a separate writer between the pre-open header check
        # and read-only SQLite connect. It may create auxiliaries; even then
        # the final status must not claim structural compatibility.
        raw_connect = sqlite3.connect
        switched = [False]

        def switch_before_read(*args, **kwargs):
            if not switched[0]:
                switched[0] = True
                writer = raw_connect(self.path)
                try:
                    writer.execute("PRAGMA journal_mode=WAL")
                finally:
                    writer.close()
            return raw_connect(*args, **kwargs)

        with patch("neurosync.bus.recovery_preflight.sqlite3.connect",
                   side_effect=switch_before_read):
            report = inspect_clock_pair(self.path)
        self.assertTrue(switched[0])
        self.assertEqual(report.classification, "BLOCKED")
        self.assertTrue(any("postread_journal_check_failed" in f
                            for f in report.findings), report.findings)

    def test_pending_main_binding_blocks(self):
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE plan_clock_binding SET status='PENDING' WHERE singleton=1")
        self.assertEqual(inspect_clock_pair(self.path).classification, "BLOCKED")

    def test_null_floor_with_v2_history_blocks(self):
        self._change_floor(None)
        self.assertEqual(inspect_clock_pair(self.path).classification, "BLOCKED")

    def test_nonfinite_or_nonnumeric_floor_blocks(self):
        for bad in (float("inf"), float("-inf"), "not-a-number"):
            with self.subTest(bad=bad):
                self._change_floor(bad)
                self.assertEqual(inspect_clock_pair(self.path).classification, "BLOCKED")

    def test_corrupt_sidecar_blocks_without_repair(self):
        self.sidecar.write_bytes(b"invalid SQLite")
        report = inspect_clock_pair(self.path)
        self.assertEqual(report.classification, "BLOCKED")
        self.assertEqual(self.sidecar.read_bytes(), b"invalid SQLite")

    def test_cli_is_read_only_and_exit_status_distinguishes_block(self):
        command = [sys.executable, "-m", "neurosync.bus.recovery_preflight", str(self.path)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["authority_effect"],
                         "NO_INTENTIONAL_AUTHORITY_DB_WRITE")
        self.sidecar.unlink()
        blocked = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(blocked.returncode, 2, blocked.stderr)
        self.assertEqual(json.loads(blocked.stdout)["classification"], "BLOCKED")
        self.assertFalse(self.sidecar.exists())


if __name__ == "__main__":
    unittest.main()
