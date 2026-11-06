"""Integration checks for scripted typed-plan -> StateLedger -> effect broker.

No local/remote LLM is invoked; all tests use fresh temporary SQLite files.
"""

from __future__ import annotations

import importlib.util
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
        self.assertEqual(self.report["allowed"]["receipt_binding"],
                         "host_asserted_unverified")
        self.assertEqual(self.report["allowed"]["plan_sha256"],
                         self.report["scripted_plan_sha256"])
        self.assertEqual(self.report["allowed"]["intent_binding_kind"],
                         "host_asserted_plan_and_handler")
        with sqlite3.connect(self.root / "authority.sqlite3") as connection:
            rows = connection.execute(
                "SELECT invocation_id, status FROM attempts ORDER BY invocation_id"
            ).fetchall()
        self.assertEqual(rows, [("allowed-use", "ok")])

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

    def test_durable_replay_rejected_after_broker_reopen(self) -> None:
        self.assertEqual(self.report["replayed_after_reopen"], {
            "error": "ReadAccessDenied", "effects_total": 1,
        })
        with sqlite3.connect(self.root / "authority.sqlite3") as connection:
            count = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        self.assertEqual(count, 1)

    def test_bound_grant_denies_selected_plan_and_descriptor_substitution(self) -> None:
        """Plan or changed declared handler after approval cannot reuse grant.

        This is only source-level host asserted metadata. Actual replacement
        of code while reusing an asserted hash is outside this fixture.
        """
        ledger = probe.StateLedger(self.root / "state.sqlite3", clock=lambda: probe.NOW)
        changed_effects = []

        def changed_host_handler(state):
            self.assertIn(probe._synthetic_case()["handles"][1],
                          state["operations"])
            value = probe.pilot.apply_tool("add_seven", 2)
            changed_effects.append(value)
            return {"result": value, "steps": 1}

        replacement = probe.HostOwnedEffectBroker(
            self.root / "authority.sqlite3", ledger,
            context=probe.HostBoundContext(probe.PRINCIPAL, probe.TARGET, probe.SESSION),
            approved_handlers={probe.KEY: probe.ApprovedEffect(
                changed_host_handler, "c" * 64, "sha1:" + "b" * 40)},
            clock=lambda: probe.NOW,
        )
        case = probe._synthetic_case()
        changed_plan = json.dumps({"steps": [
            {"handle": case["handles"][1], "input": "initial"}], "abstain": False})
        other_plan_sha256 = probe.hashlib.sha256(changed_plan.encode()).hexdigest()
        with self.assertRaises(probe.ReadAccessDenied):
            replacement.execute("allowed", "changed-handler-use", probe.KEY,
                                plan_sha256=self.report["scripted_plan_sha256"])
        with self.assertRaises(probe.ReadAccessDenied):
            replacement.execute("allowed", "different-plan-use", probe.KEY,
                                plan_sha256=other_plan_sha256)
        with self.assertRaises(probe.ReadAccessDenied):
            replacement.execute("allowed", "omitted-plan-use", probe.KEY)
        self.assertEqual(changed_effects, [])
        for invocation in ("changed-handler-use", "different-plan-use", "omitted-plan-use"):
            self.assertIsNone(replacement.attempt(invocation))

    def test_state_is_really_committed_in_stateledger(self) -> None:
        ledger = probe.StateLedger(self.root / "state.sqlite3", clock=lambda: probe.NOW)
        projected = ledger.project(probe.TARGET, probe.SESSION)
        self.assertEqual(set(projected), {probe.KEY})
        expected = probe._synthetic_case()["registry"]
        self.assertEqual(projected[probe.KEY]["operations"], expected)
        self.assertEqual(len(expected), 13)

    def test_scripted_fixture_cannot_be_mistaken_for_real_model(self) -> None:
        self.assertIs(self.report["model_called"], False)
        self.assertEqual(self.report["evidence_class"],
                         "LOCAL_SCRIPTED_BROKER_FIXTURE_TEST")
        self.assertEqual(self.report["plan_source"],
                         "literal_scripted_typed_json_not_neural")
        self.assertEqual(self.report["plan_receipt_binding"],
                         "host_asserted_selected_plan_digest_and_handler_descriptor")
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
