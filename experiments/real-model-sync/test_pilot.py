"""Hostile stdlib *fixture* tests. No actual neural model in this suite."""

import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("local_model_pilot", Path(__file__).with_name("pilot.py"))
pilot = importlib.util.module_from_spec(spec)
import sys
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

    def test_rehashed_forged_reachability_label_cannot_score_false_abstention(self):
        forged = json.loads(self.suite_path.read_text())
        # Attackers can rehash the suite, so a matching sidecar is not proof
        # that a label still matches the executable oracle's reachable set.
        impossible = next(case for case in forged["cases"] if not case["expected_possible"])
        impossible["expected_possible"] = True
        self.suite_path.write_text(pilot.canonical_json(forged) + "\n")
        self.suite_path.with_suffix(".json.sha256").write_text(pilot.sha(forged) + "\n")
        with self.assertRaisesRegex(ValueError, "reachability label"):
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

    def test_parse_plan_rejects_duplicate_json_keys_even_if_last_value_valid(self):
        handles = self.case["handles"]
        ambiguous = ('{"steps":[],"abstain":false,"abstain":true}')
        with self.assertRaises(ValueError):
            pilot.parse_plan(ambiguous, handles)
        ambiguous_step = ('{"steps":[{"handle":"' + handles[0] +
                          '","handle":"' + handles[1] +
                          '","input":"initial"}],"abstain":false}')
        with self.assertRaises(ValueError):
            pilot.parse_plan(ambiguous_step, handles)

    def test_subset_n_only_does_not_materialize_unselected_bus_or_file_arms(self):
        output = self.root / "n-only"
        report = pilot.run_suite(
            self.suite, fixture_solver, output,
            model_evidence={"real_endpoint": False}, max_cases=1, arms=("N",),
        )
        self.assertEqual(report["arms_evaluated"], ["N"])
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(list(output.iterdir()), [],
                         "N-only exploratory run must not create P/V authority fixtures")

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

    def test_compact_profile_reduces_bytes_without_leaking_or_changing_controls(self):
        state = {"operations": self.case["registry"]}
        task = pilot.model_request(self.case, state)
        compact = pilot._model_messages(task, "compact")
        legacy = pilot._model_messages(task, "legacy")
        self.assertLess(len(pilot.canonical_json(compact)), len(pilot.canonical_json(legacy)))
        delivered = json.loads(compact[1]["content"])
        self.assertEqual(delivered, {"start": self.case["start"],
                                     "goal": self.case["goal"],
                                     "handles": self.case["handles"], "state": state})
        self.assertEqual([m["role"] for m in compact], ["system", "user"])
        self.assertEqual([m["role"] for m in legacy], ["system", "user"])
        neutral_instruction = compact[0]["content"].lower()
        self.assertIn("you may abstain", neutral_instruction)
        for biasing_phrase in ("state only", "do not invent", "unknown handles",
                               "if unknown", "must abstain"):
            self.assertNotIn(biasing_phrase, neutral_instruction)
        for profile in (compact, legacy):
            text = pilot.canonical_json(profile)
            for secret in ("expected_possible", "gold_length", "registry"):
                self.assertNotIn(secret, text)

        # Every arm uses the same formatting and same handles; only the
        # authorized external-state slot differs (including distorted D).
        for arm, arm_state in (("N", {}),
                               ("R", {"observations": pilot._raw_probes(self.case)}),
                               ("V", state), ("P", state), ("H", state),
                               ("D", {"operations": {"opaque": "wrong"}}), ("X", state)):
            with self.subTest(arm=arm):
                delivered = json.loads(pilot._model_messages(
                    pilot.model_request(self.case, arm_state), "compact")[1]["content"])
                self.assertEqual(delivered["handles"], self.case["handles"])
                self.assertEqual(delivered["start"], self.case["start"])
                self.assertEqual(delivered["goal"], self.case["goal"])
                self.assertEqual(delivered["state"], arm_state)

    def test_request_budgets_apply_to_both_backends_and_report_latency(self):
        class Response:
            def __init__(self, raw):
                self.raw = raw
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, n): return self.raw

        for backend in ("ollama", "openai"):
            with self.subTest(backend=backend):
                captured = {}
                class Opener:
                    def open(self, request, timeout):
                        captured["timeout"] = timeout
                        captured["body"] = json.loads(request.data)
                        response = ({"message": {"content": '{"steps":[],"abstain":true}'}}
                                    if backend == "ollama" else {
                                        "choices": [{"message": {"content":
                                                     '{"steps":[],"abstain":true}'}}]})
                        return Response(json.dumps(response).encode())

                client = pilot.LocalModelClient("http://127.0.0.1:11435", backend,
                                                "fixture-no-model", timeout=5,
                                                max_tokens=64)
                client.opener = Opener()
                response, trace = client(pilot.model_request(self.case, {}))
                self.assertIn('"abstain":true', response)
                self.assertEqual(captured["timeout"], 5)
                self.assertEqual(trace["timeout_seconds"], 5)
                self.assertEqual(trace["max_tokens"], 64)
                self.assertEqual(trace["request_bytes"], len(
                    pilot.canonical_json(captured["body"]).encode()))
                self.assertGreaterEqual(trace["latency_ms"], 0)
                self.assertEqual(trace["outcome"], "received_text")
                options = captured["body"].get("options", captured["body"])
                self.assertEqual(options.get("num_predict", options.get("max_tokens")), 64)

    def test_timeout_receipt_preserves_host_control_scores(self):
        class TimeoutOpener:
            def open(self, request, timeout):
                raise TimeoutError("synthetic fixture timeout, NOT A REAL MODEL")

        client = pilot.LocalModelClient("http://127.0.0.1:11435", "openai",
                                        "fixture-no-model", timeout=0.5)
        client.opener = TimeoutOpener()
        result = pilot.run_one(self.case, "X", {"operations": self.case["registry"]}, client)
        self.assertEqual(result["error"], "TimeoutError")
        self.assertFalse(result["success"])
        self.assertFalse(result["gate_tested"])
        self.assertFalse(result["governance_denial_success"])
        self.assertEqual(result["effects"], 0)
        self.assertEqual(result["transport_trace"]["timeout_seconds"], 0.5)
        self.assertGreaterEqual(result["transport_trace"]["latency_ms"], 0)
        self.assertGreaterEqual(result["model_call_elapsed_ms"], 0)
        self.assertEqual(result["transport_trace"]["outcome"], "transport_or_parse_failure")

    def test_invalid_request_budget_rejected_before_network(self):
        for kwargs in ({"timeout": 0}, {"timeout": 61}, {"timeout": float("nan")},
                       {"max_tokens": 31}, {"max_tokens": 257},
                       {"max_tokens": True}, {"prompt_profile": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pilot.LocalModelClient("http://127.0.0.1:11435", "openai", "fixture", **kwargs)


if __name__ == "__main__":
    unittest.main()
