"""Independent adversarial integrity tests for the synthetic receiver pilot.

These tests exercise the public pilot functions; they do not represent neural
model evaluations or independent model/host custody.
"""

from collections import Counter
import importlib.util
import json
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
