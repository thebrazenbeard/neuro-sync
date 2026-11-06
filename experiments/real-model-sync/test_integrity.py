"""Independent adversarial integrity tests for the synthetic receiver pilot.

These tests exercise the public pilot functions; they do not represent neural
model evaluations or independent model/host custody.
"""

from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "neuro_sync_integrity_target", Path(__file__).with_name("pilot.py")
)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)


def independently_reachable(case):
    """Check <=2 tool effects from registry without trusting a gold label."""
    start = case["start"]
    goal = case["goal"]
    operations = tuple(case["registry"].values())
    for first in operations:
        one = pilot.oracle_tool(first, start)
        if one == goal:
            return True
        if any(pilot.oracle_tool(second, one) == goal for second in operations):
            return True
    return False


class IndependentIntegrityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.suite_path = self.root / "sealed.json"
        pilot.freeze_suite(self.suite_path)
        self.suite, _ = pilot.load_suite(self.suite_path)

    def test_generated_suite_truth_matches_independent_reachability(self):
        self.assertEqual(len(self.suite["cases"]), 12)
        self.assertEqual(sum(not c["expected_possible"] for c in self.suite["cases"]), 2)
        for case in self.suite["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(independently_reachable(case), case["expected_possible"])

    def test_resealed_false_impossible_label_must_be_rejected(self):
        """Self-hash validation alone cannot be mistaken for semantic custody."""
        suite = json.loads(self.suite_path.read_text())
        case = next(c for c in suite["cases"] if c["expected_possible"])
        self.assertTrue(independently_reachable(case))
        case["expected_possible"] = False
        self.suite_path.write_text(pilot.canonical_json(suite), encoding="utf8")
        self.suite_path.with_suffix(".json.sha256").write_text(pilot.sha(suite), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "possible|reachab|inconsistent"):
            pilot.load_suite(self.suite_path)

    def test_spoofed_http_evidence_from_fixture_must_not_be_classified_as_http(self):
        """Caller-declared evidence must not upgrade a scripted adapter."""
        response = json.dumps({"steps": [], "abstain": True})
        # Rejecting a mismatched declaration is equally acceptable.
        try:
            report = pilot.run_suite(
                self.suite,
                lambda _: (response, {"provider": "scripted-fixture"}),
                self.root / "fake-claimed-http",
                model_evidence={"real_endpoint": True},
                max_cases=1,
                arms=("N",),
            )
        except ValueError:
            return
        self.assertNotEqual(report["evidence_class"], "LOCAL_HTTP_UNATTESTED")

    def test_all_abstain_baseline_does_not_falsely_qualify_revocation(self):
        response = json.dumps({"steps": [], "abstain": True})
        report = pilot.run_suite(
            self.suite,
            lambda _: (response, {"provider": "scripted-fixture"}),
            self.root / "abstain",
            model_evidence={"real_endpoint": False},
        )
        revoked = [r for r in report["results"] if r["arm"] == "X"]
        self.assertTrue(all(not r["gate_tested"] for r in revoked))
        self.assertTrue(all(not r["governance_denial_success"] for r in revoked))
        self.assertEqual(report["success_by_arm"]["X"], 0)
        # For every nonrevoked arm, only the two impossible cases score.
        for arm in set(pilot.ARMS) - {"X"}:
            self.assertEqual(report["success_by_arm"][arm], 2)

    def test_arm_order_counterbalances_positions(self):
        """Distinct case order should not systematically confound warmup/latency."""
        self.assertTrue(hasattr(pilot, "arm_order_for_case"), "public arm-order helper required")
        positions = {arm: Counter() for arm in pilot.ARMS}
        for case_index in range(12):
            order = pilot.arm_order_for_case(case_index, pilot.ARMS)
            self.assertEqual(set(order), set(pilot.ARMS))
            for pos, arm in enumerate(order):
                positions[arm][pos] += 1
        # An earlier seeded shuffle was biased. The public ex-ante cyclic
        # schedule must distribute every arm across positions fairly.
        for arm, counts in positions.items():
            with self.subTest(arm=arm):
                self.assertLessEqual(max(counts.get(i, 0) for i in range(7)) -
                                     min(counts.get(i, 0) for i in range(7)), 1)

    def test_overdeep_local_http_envelope_is_a_bounded_invalid_reply(self):
        client = pilot.LocalModelClient("http://127.0.0.1:11434", "ollama", "synthetic")
        bad_envelope = ("[" * 10_000 + "0" + "]" * 10_000).encode("utf8")

        class FakeOpener:
            def open(self, request, timeout):
                return io.BytesIO(bad_envelope)

        client.opener = FakeOpener()
        case = self.suite["cases"][0]
        result = pilot.run_one(case, "N", {}, client)
        self.assertTrue(result["invalid_reply"])
        self.assertEqual(result["effects"], 0)
        self.assertEqual(result["transport_trace"]["outcome"], "transport_or_parse_failure")

    def test_duplicate_http_envelope_keys_cannot_select_model_text(self):
        client = pilot.LocalModelClient("http://127.0.0.1:11434", "ollama", "synthetic")
        # Default json.loads silently chooses the last `message`, making a
        # hash-bound HTTP envelope ambiguous to independent parsers.
        contradictory = (b'{"message":{"content":"FIRST-IGNORED"},'
                         b'"message":{"content":"{\\"steps\\":[],\\"abstain\\":true}"}}')

        class FakeOpener:
            def open(self, request, timeout):
                return io.BytesIO(contradictory)

        client.opener = FakeOpener()
        case = self.suite["cases"][0]
        result = pilot.run_one(case, "N", {}, client)
        self.assertTrue(result["invalid_reply"])
        self.assertEqual(result["effects"], 0)

    def test_matched_comparator_cannot_be_disabled_by_python_optimization(self):
        script = """
import importlib.util
from pathlib import Path
import sys
import tempfile
spec = importlib.util.spec_from_file_location('neuro_pilot_opt', sys.argv[1])
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
        return {**state, 'altered_by_host': True} if arm == 'H' else state
    pilot._arm_state = diverging
    try:
        pilot.run_suite(suite, lambda _: (_ for _ in ()).throw(RuntimeError('receiver called')),
                        root / 'run', model_evidence={'real_endpoint': False}, max_cases=1)
    except ValueError as exc:
        if 'comparator' not in str(exc):
            raise
    else:
        raise RuntimeError('divergent comparator silently accepted')
"""
        for args in ([], ["-O"]):
            with self.subTest(optimized=bool(args)):
                result = subprocess.run([sys.executable, *args, "-c", script,
                                         str(Path(pilot.__file__))],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_bounded_decision_receipt_and_adversarial_reply_rejection(self):
        case = self.suite["cases"][0]
        h = case["handles"][0]
        valid = json.dumps({"steps": [{"handle": h, "input": "initial"}], "abstain": False})
        result = pilot.run_one(case, "H", {"operations": case["registry"]},
                               lambda _: (valid, {"provider": "synthetic"}))
        self.assertFalse(result["invalid_reply"])
        self.assertEqual(result["selected_handles"], [h])
        self.assertEqual(result["model_text_sha256"], hashlib.sha256(valid.encode()).hexdigest())
        self.assertNotIn(valid, json.dumps(result))
        invalid = [valid.encode(), valid + "\ud800", "[" * 10_000 + "0" + "]" * 10_000,
                   '{"steps":[],"abstain":false,"abstain":true}', "x" * 1_000_001]
        for reply in invalid:
            with self.subTest(reply_kind=type(reply).__name__, length=len(reply)):
                rejected = pilot.run_one(case, "H", {"operations": case["registry"]},
                                         lambda _: (reply, {}))
                self.assertTrue(rejected["invalid_reply"])
                self.assertEqual(rejected["effects"], 0)


if __name__ == "__main__":
    unittest.main()
