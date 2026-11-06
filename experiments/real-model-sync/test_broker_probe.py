"""Integration checks for scripted typed-plan -> StateLedger -> effect broker.

No local/remote LLM is invoked; all tests use fresh temporary SQLite files.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


PATH = Path(__file__).with_name("broker_probe.py")
spec = importlib.util.spec_from_file_location("broker_probe_under_test", PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


class ActualEffectBrokerPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "broker-integration"
        self.effects: list[dict] = []
        self.report = probe.run_probe(self.root, handler_log=self.effects)

    def test_one_allowed_plan_records_one_durable_effect(self) -> None:
        self.assertEqual(self.effects, [{"result": 4, "steps": 1}])
        self.assertEqual(self.report["allowed"]["handler_result"], self.effects[0])
        self.assertEqual(self.report["allowed"]["receipt_status"], "ok")
        self.assertEqual(self.report["allowed"]["receipt_version"], 2)
        expected_plan = {
            "steps": [{"handle": probe._synthetic_case()["scripted_handle"],
                       "input": "initial"}], "abstain": False,
        }
        canonical = json.dumps(expected_plan, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, separators=(",", ":")).encode()
        expected_digest = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(self.report["allowed"]["receipt_plan_sha256"],
                         expected_digest)
        self.assertEqual(self.report["allowed"]["receipt_binding"],
                         "host_asserted_unverified")
        with sqlite3.connect(self.root / "authority.sqlite3") as connection:
            rows = connection.execute(
                "SELECT invocation_id, status FROM attempts ORDER BY invocation_id"
            ).fetchall()
        self.assertEqual(rows, [("allowed-use", "ok"), ("stale-use", "denied")])

    def test_revoke_after_typed_plan_never_runs_effect_handler(self) -> None:
        result = self.report["revoked_after_plan"]
        self.assertEqual(result["error"], "ReadAccessDenied")
        self.assertEqual(result["new_effects"], 0)
        # Revocation before claim is a denial, not a durable attempt receipt.
        self.assertIsNone(result["durable_attempt"])
        self.assertEqual(len(self.effects), 1)

    def test_exact_expiry_and_foreign_context_never_get_effects(self) -> None:
        expiry = self.report["expired_after_plan"]
        foreign = self.report["foreign_context"]
        self.assertEqual(expiry["error"], "ReadAccessDenied")
        self.assertEqual(expiry["new_effects"], 0)
        self.assertIsNone(expiry["durable_attempt"])
        self.assertEqual(foreign["error"], "ReadAccessDenied")
        self.assertIsNone(foreign["durable_attempt"])
        self.assertEqual(len(self.effects), 1)

    def test_fixture_clock_never_rewinds_between_negative_cases(self) -> None:
        # The expiry case deliberately observes the endpoint at t=1001;
        # later grant tests must not set the host authority back to t=1000.
        observations = self.report["synthetic_plan_clock_epochs"]
        self.assertEqual(observations, [probe.NOW, 1001.0, 1001.0])
        self.assertEqual(observations, sorted(observations))

    def test_durable_replay_rejected_after_broker_reopen(self) -> None:
        self.assertEqual(self.report["replayed_after_reopen"], {
            "error": "ReadAccessDenied", "effects_total": 1,
        })
        with sqlite3.connect(self.root / "authority.sqlite3") as connection:
            count = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        self.assertEqual(count, 2)

    def test_other_handle_under_same_state_key_is_denied(self) -> None:
        mismatch = self.report["unapproved_same_key_plan"]
        self.assertEqual(mismatch["error"], "ReadAccessDenied")
        self.assertEqual(mismatch["new_effects"], 0)
        self.assertIsNone(mismatch["durable_attempt"])
        self.assertEqual(self.report["grant_scope"],
                         "exact_plan_state_and_handler_one_use")
        self.assertEqual(self.effects, [{"result": 4, "steps": 1}])

    def test_one_use_grant_and_handler_artifact_enforced(self) -> None:
        self.assertEqual(self.report["plan_grant_second_use"]["error"],
                         "ReadAccessDenied")
        self.assertIsNone(self.report["plan_grant_second_use"]["durable_attempt"])
        changed = self.report["changed_handler_artifact"]
        self.assertEqual(changed["error"], "ReadAccessDenied")
        self.assertEqual(changed["new_effects"], 0)
        self.assertIsNone(changed["durable_attempt"])

    def test_state_revision_after_approval_consumes_claim_without_effect(self) -> None:
        stale = self.report["changed_committed_state"]
        self.assertEqual(stale["error"], "ReadAccessDenied")
        self.assertEqual(stale["new_effects"], 0)
        self.assertEqual(stale["durable_attempt_status"], "denied")
        self.assertEqual(len(self.effects), 1)

    def test_state_is_really_committed_in_stateledger(self) -> None:
        ledger = probe.StateLedger(self.root / "state.sqlite3", clock=lambda: probe.NOW)
        projected = ledger.project(probe.TARGET, probe.SESSION)
        self.assertEqual(set(projected), {probe.KEY})
        expected = probe._synthetic_case()["registry"]
        self.assertEqual(projected[probe.KEY]["operations"], expected)
        self.assertEqual(projected[probe.KEY]["revision_marker"], 2)
        self.assertEqual(len(expected), 13)

    def test_scripted_fixture_cannot_be_mistaken_for_real_model(self) -> None:
        self.assertIs(self.report["model_called"], False)
        self.assertEqual(self.report["evidence_class"],
                         "LOCAL_SCRIPTED_BROKER_FIXTURE_TEST")
        self.assertEqual(self.report["plan_source"],
                         "literal_scripted_typed_json_not_neural")
        self.assertEqual(self.report["plan_receipt_binding"],
                         "broker_verified_canonical_plan_sha256_host_asserted")
        self.assertTrue(any("same-UID" in limit for limit in self.report["limits"]))
        self.assertTrue(any("no durable attempt receipt" in limit
                            for limit in self.report["limits"]))

    def test_cli_receipt_is_json_with_no_model_endpoint(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PATH)], check=False, text=True,
            capture_output=True, timeout=5,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt["allowed"]["receipt_status"], "ok")
        self.assertFalse(receipt["model_called"])
        self.assertNotIn("endpoint", receipt)


if __name__ == "__main__":
    unittest.main()
