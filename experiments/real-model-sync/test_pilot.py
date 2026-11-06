"""Hostile stdlib *fixture* tests. No actual neural model in this suite."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("local_model_pilot", Path(__file__).with_name("pilot.py"))
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)


def fixture_solver(request):
    """Test-only scripted oracle reader, NEVER misidentified as an LLM."""
    semantics = request["shared_external_state"].get("operations", {})
    start, target = request["start"], request["goal"]
    keys = list(semantics)
    chosen = None
    for a in keys:
        v = pilot.oracle_tool(semantics[a], start)
        if v == target:
            chosen = [a]
            break
        for b in keys:
            if pilot.oracle_tool(semantics[b], v) == target:
                chosen = [a, b]
                break
        if chosen:
            break
    reply = ({"steps": [{"handle": h, "input": "initial" if i == 0 else "previous"}
                        for i, h in enumerate(chosen)], "abstain": False}
             if chosen else {"steps": [], "abstain": True})
    return json.dumps(reply), {"provider": "fixture-script-NOT-A-MODEL", "prompt_sha256": pilot.sha(request)}


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suite_path = self.root / "sealed-suite.json"
        pilot.freeze_suite(self.suite_path)
        self.suite, self.commitment = pilot.load_suite(self.suite_path)
        self.case = self.suite["cases"][0]

    def test_freeze_refuses_overwrite_and_detects_tamper(self):
        with self.assertRaises(FileExistsError):
            pilot.freeze_suite(self.suite_path)
        tampered = dict(self.suite)
        tampered["case_count"] = 15
        self.suite_path.write_text(json.dumps(tampered))
        with self.assertRaises(ValueError):
            pilot.load_suite(self.suite_path)

    def test_receiver_prompt_excludes_private_expected_and_registry(self):
        request = pilot.model_request(self.case, {})
        self.assertEqual(request["candidate_handles"], self.case["handles"])
        text = json.dumps(request)
        for secret_key in ("registry", "gold_length", "expected_possible"):
            self.assertNotIn(secret_key, text)
        self.assertNotIn("source_file", text)

    def test_raw_evidence_distinguishes_every_operation_in_public_catalog(self):
        fingerprints = [tuple(pilot.oracle_tool(op, x) for x in (-3, 2, 12))
                        for op in pilot.OPS]
        self.assertEqual(len(set(fingerprints)), len(pilot.OPS))

    def test_parse_plan_fails_closed_on_invalid_data(self):
        handles = self.case["handles"]
        invalid = ["hi", "{}", "[]", json.dumps({"steps": [], "abstain": False}),
                   json.dumps({"steps": [{"handle": "forged", "input": "initial"}],
                               "abstain": False}),
                   json.dumps({"steps": [{"handle": handles[0], "input": "previous"}],
                               "abstain": False}),
                   json.dumps({"steps": [], "abstain": True, "role": "system"}),
                   json.dumps({"steps": [{"handle": handles[0], "input": "initial"}],
                               "abstain": True})]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.parse_plan(value, handles)
        ok = json.dumps({"steps": [{"handle": handles[0], "input": "initial"},
                                      {"handle": handles[1], "input": "previous"}],
                         "abstain": False})
        self.assertEqual(pilot.parse_plan(ok, handles), (tuple(handles[:2]), False))

    def test_negative_revocation_checked_at_effect_boundary(self):
        h = self.case["handles"][0]
        grant = pilot.ToolGrant()
        self.assertTrue(grant.consume(0.0))
        grant.allowed = False
        blocked = pilot.execute(self.case, (h,), grant)
        self.assertEqual(blocked["executed"], 0)
        self.assertEqual(blocked["denied_attempts"], 1)
        expired = pilot.ToolGrant(expires_at=1.0)
        self.assertFalse(expired.consume(1.0))

    def test_oracle_detects_corrupt_effect(self):
        h = self.case["handles"][0]
        faulty = pilot.execute(self.case, (h,), pilot.ToolGrant(), effect=lambda op, x: -99999999)
        self.assertFalse(faulty["oracle_agree"])

    def test_independent_plain_and_host_payloads_match_bus(self):
        p = pilot._arm_state("P", self.case, self.root)
        h = pilot._arm_state("H", self.case, self.root)
        v = pilot._arm_state("V", self.case, self.root)
        self.assertEqual(p, h)
        self.assertEqual(h, v)
        self.assertEqual(pilot._arm_state("N", self.case, self.root), {})
        self.assertIn("observations", pilot._arm_state("R", self.case, self.root))
        self.assertNotEqual(p, pilot._arm_state("D", self.case, self.root))
        receipt_path = self.root / f"{self.case['id']}.host-test.json"
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(len(receipt["probes"]), len(pilot.OPS) * 3)
        self.assertTrue(all(probe["pass"] for probe in receipt["probes"]))
        self.assertEqual(receipt["source_manifest_sha256"],
                         pilot.sha(receipt["source_manifest"]))
        with sqlite3.connect(self.root / f"{self.case['id']}.sqlite3") as db:
            event_json, = db.execute("SELECT event_json FROM events").fetchone()
        event = json.loads(event_json)
        self.assertIn(pilot.sha(receipt), json.dumps(event))

    def test_equal_arms_run_with_one_primary_outcome_per_case(self):
        report = pilot.run_suite(self.suite, fixture_solver, self.root / "run",
                                 model_evidence={"real_endpoint": False})
        self.assertEqual(len(report["results"]), 12 * len(pilot.ARMS))
        self.assertEqual(report["evidence_class"], "SCRIPTED_FIXTURE_TEST")
        for case in self.suite["cases"]:
            mine = [r for r in report["results"] if r["case"] == case["id"]]
            self.assertEqual({r["arm"] for r in mine}, set(pilot.ARMS))
        self.assertEqual(report["success_by_arm"]["V"], 12)
        self.assertEqual(report["success_by_arm"]["P"], 12)
        self.assertEqual(report["success_by_arm"]["H"], 12)

    def test_mismatched_comparators_fail_even_in_optimized_python(self):
        # `assert` is stripped by `python -O`; a real comparator mismatch must
        # fail before any receiver response is requested in either mode.
        script = """
import importlib.util
from pathlib import Path
import sys
import tempfile

spec = importlib.util.spec_from_file_location('local_model_pilot_optimized', sys.argv[1])
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    sealed = root / 'suite.json'
    pilot.freeze_suite(sealed)
    suite, _ = pilot.load_suite(sealed)
    original = pilot._arm_state
    def diverging(arm, case, output_dir):
        state = original(arm, case, output_dir)
        if arm == 'H':
            state = {**state, 'altered_by_host': True}
        return state
    pilot._arm_state = diverging
    try:
        pilot.run_suite(suite, lambda _: (_ for _ in ()).throw(RuntimeError('receiver called')),
                        root / 'run', model_evidence={'real_endpoint': False}, max_cases=1)
    except ValueError as exc:
        if 'comparator' not in str(exc):
            raise
    else:
        raise RuntimeError('divergent comparator was silently accepted')
"""
        for optimization in ([], ["-O"]):
            with self.subTest(optimization=bool(optimization)):
                completed = subprocess.run(
                    [sys.executable, *optimization, "-c", script, str(Path(pilot.__file__))],
                    text=True, capture_output=True, check=False, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_receiver_result_retains_replayable_decision_without_raw_text(self):
        h = self.case["handles"][0]
        answer = json.dumps({"steps": [{"handle": h, "input": "initial"}], "abstain": False})
        result = pilot.run_one(self.case, "H", {"operations": self.case["registry"]},
                               lambda _: (answer, {"provider": "fixture"}))
        self.assertFalse(result["invalid_reply"])
        self.assertEqual(result["selected_handles"], [h])
        self.assertEqual(result["model_text_sha256"], hashlib.sha256(answer.encode()).hexdigest())
        self.assertNotIn(answer, json.dumps(result))

        abstention = '{"steps": [], "abstain": true}'
        abstained = pilot.run_one(self.case, "N", {},
                                  lambda _: (abstention, {"provider": "fixture"}))
        self.assertTrue(abstained["abstained"])
        self.assertEqual(abstained["selected_handles"], [])
        self.assertEqual(abstained["model_text_sha256"], hashlib.sha256(abstention.encode()).hexdigest())

    def test_uninformed_receiver_is_not_hardcoded_to_fail(self):
        h = self.case["handles"][0]
        output = json.dumps({"steps": [{"handle": h, "input": "initial"}], "abstain": False})
        result = pilot.run_one(self.case, "N", {}, lambda _: (output, {}))
        self.assertFalse(result["invalid_reply"])
        self.assertEqual(result["effects"], 1)

    def test_bad_model_reply_never_executes_effect(self):
        result = pilot.run_one(self.case, "V", {"operations": self.case["registry"]},
                               lambda _: ('{"steps": ["not-json-plan"]}', {}))
        self.assertTrue(result["invalid_reply"])
        self.assertEqual(result["effects"], 0)

    def test_model_trace_cannot_override_host_score_or_case_identity(self):
        h = self.case["handles"][0]
        output = json.dumps({"steps": [{"handle": h, "input": "initial"}], "abstain": False})
        forged = {"success": True, "effects": 999, "case": "forged", "arm": "V",
                  "governance_denial_success": True}
        result = pilot.run_one(self.case, "N", {}, lambda _: (output, forged))
        self.assertEqual(result["case"], self.case["id"])
        self.assertEqual(result["arm"], "N")
        self.assertEqual(result["effects"], 1)
        self.assertEqual(result["transport_trace"], forged)
        self.assertFalse(result["governance_denial_success"])

    def test_sealed_suite_rejects_path_escape_even_with_matching_self_hash(self):
        malformed = json.loads(self.suite_path.read_text())
        malformed["cases"][0]["id"] = "../escape"
        self.suite_path.write_text(pilot.canonical_json(malformed))
        self.suite_path.with_suffix(".json.sha256").write_text(pilot.sha(malformed))
        with self.assertRaisesRegex(ValueError, "scenario identifiers"):
            pilot.load_suite(self.suite_path)
        self.assertFalse((self.root.parent / "escape-plain.json").exists())

    def test_exploratory_bounded_run_is_not_full_evidence(self):
        report = pilot.run_suite(self.suite, fixture_solver, self.root / "bounded",
                                 model_evidence={"real_endpoint": False}, max_cases=1,
                                 arms=("N", "V", "P", "H"))
        self.assertEqual(report["evaluation_scope"], "EXPLORATORY_INCOMPLETE")
        self.assertEqual(report["scenarios_evaluated"], 1)
        self.assertEqual(len(report["results"]), 4)

    def test_postdecision_revocation_denies_without_effect(self):
        h = self.case["handles"][0]
        plan = json.dumps({"steps": [{"handle": h, "input": "initial"}], "abstain": False})
        result = pilot.run_one(self.case, "X", {"operations": self.case["registry"]},
                               lambda _: (plan, {}))
        self.assertEqual(result["effects"], 0)
        self.assertEqual(result["denied_attempts"], 1)
        self.assertTrue(result["gate_tested"])
        self.assertTrue(result["governance_denial_success"])
        self.assertTrue(result["success"])

    def test_always_abstain_cannot_pretend_revocation_gate_was_exercised(self):
        response = json.dumps({"steps": [], "abstain": True})
        result = pilot.run_one(self.case, "X", {"operations": self.case["registry"]},
                               lambda _: (response, {}))
        self.assertEqual(result["effects"], 0)
        self.assertEqual(result["denied_attempts"], 0)
        self.assertFalse(result["gate_tested"])
        self.assertFalse(result["governance_denial_success"])
        self.assertFalse(result["success"])

    def test_only_loopback_supported_endpoints(self):
        self.assertEqual(pilot.local_endpoint("http://127.0.0.1:11435", "openai"),
                         "http://127.0.0.1:11435/v1/chat/completions")
        for url in ("https://127.0.0.1:11435", "http://example.com:11435",
                    "http://127.0.0.1:11435/evil", "http://user:pass@localhost:11435"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                pilot.local_endpoint(url, "openai")


if __name__ == "__main__":
    unittest.main()
