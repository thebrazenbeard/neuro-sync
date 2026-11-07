"""Independent scenario-level scoring tests. No inference engines are called."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("analyze.py")


def load_module():
    spec = importlib.util.spec_from_file_location("neurosync_scenario_stats", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARMS = ("N", "R", "V", "P", "H", "D", "X")


def receipt(n=12, selected=ARMS):
    rows = []
    for i in range(n):
        case = f"case-{i:02d}-deadbeef"
        for arm in selected:
            success = arm in ("V", "P", "H")
            rows.append({"case": case, "arm": arm, "success": success,
                         "invalid_reply": False,
                         "effects": 1 if success else 0,
                         "denied_attempts": 0, "gate_tested": arm == "X",
                         "governance_denial_success": False})
    counts = {arm: sum(r["success"] for r in rows if r["arm"] == arm)
              for arm in selected}
    return {"experiment": "neuro-sync-local-receiver-v0",
            "evaluation_scope": ("FULL_LOCAL_PROCEDURE" if n == 12 and
                                  set(selected) == set(ARMS) else "EXPLORATORY_INCOMPLETE"),
            "scenarios_evaluated": n, "arms_evaluated": list(selected),
            "evidence_class": "SCRIPTED_FIXTURE_TEST",
            "model_evidence": {"real_endpoint": False},
            "unit": "scenario-arm, NOT individual tool calls",
            "results": rows, "success_by_arm": counts}


def current_pilot_v1_receipt(selected=ARMS, *, suite_size=12, max_cases=None):
    """Use actual pilot producer, without host/model/network access."""
    spec = importlib.util.spec_from_file_location(
        "neuro_sync_producer_for_stats", SCRIPT.parents[1] / "pilot.py"
    )
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        suite_path = root / "suite.json"
        pilot.freeze_suite(suite_path, count=suite_size)
        suite, _ = pilot.load_suite(suite_path)
        return pilot.run_suite(
            suite, lambda _: (json.dumps({"steps": [], "abstain": True}),
                              {"fixture": "always-abstain"}),
            root / "out", model_evidence={"real_endpoint": False}, arms=selected,
            max_cases=max_cases,
        )


class ScenarioStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = load_module()

    def test_one_case_smoke_remains_inconclusive(self):
        data = receipt(1, ("V",))
        data["results"][0]["success"] = False
        data["results"][0]["invalid_reply"] = True
        data["results"][0]["effects"] = 0
        data["success_by_arm"]["V"] = 0
        report = self.s.analyze(data)
        self.assertEqual(report["qualification"], "EXPLORATORY_ONLY")
        self.assertEqual(report["arms"]["V"]["invalid_replies"], 1)
        self.assertEqual(report["response_completeness"], "INCOMPLETE_MODEL_RESPONSES")
        self.assertEqual(report["paired_comparisons"], [])
        self.assertEqual(report["neural_transfer_established"], False)

    def test_no_state_difference_but_no_statebus_specific_advantage(self):
        report = self.s.analyze(receipt())
        self.assertEqual(report["qualification"], "SOURCE_ONLY_FULL_LOCAL_FIXTURE")
        paired = {(item["arm_a"], item["arm_b"]): item
                  for item in report["paired_comparisons"]}
        self.assertEqual(paired[("V", "P")]["paired_difference"], 0)
        self.assertEqual(paired[("V", "H")]["discordants"], 0)
        self.assertEqual(paired[("V", "P")]["exact_two_sided_p"], 1.0)
        self.assertEqual(paired[("V", "N")]["paired_difference"], 1.0)
        self.assertLess(paired[("V", "N")]["exact_two_sided_p"], .001)
        self.assertGreaterEqual(paired[("V", "N")]["holm_familywise_adjusted_p"],
                                paired[("V", "N")]["exact_two_sided_p"])
        self.assertLess(paired[("V", "N")]["holm_familywise_adjusted_p"], .01)
        self.assertFalse(report["statebus_specific_accuracy_advantage_established"])

    def test_paired_exact_one_discordant_is_not_significant(self):
        data = receipt(12)
        v = next(r for r in data["results"] if r["case"].startswith("case-00") and r["arm"] == "V")
        v["success"] = False
        data["success_by_arm"]["V"] -= 1
        report = self.s.analyze(data)
        pair = next(x for x in report["paired_comparisons"] if x["arm_a"] == "V" and x["arm_b"] == "P")
        self.assertEqual(pair["discordants"], 1)
        self.assertEqual(pair["exact_two_sided_p"], 1.0)
        self.assertEqual(pair["holm_familywise_adjusted_p"], 1.0)
        self.assertAlmostEqual(pair["paired_difference"], -1/12, places=5)

    def test_partial_arm_set_and_full_case_count_remains_exploratory(self):
        report = self.s.analyze(receipt(12, ("N", "V", "P", "H")))
        self.assertEqual(report["qualification"], "EXPLORATORY_ONLY")
        self.assertEqual(len(report["paired_comparisons"]), 3)

    def test_reject_duplicate_missing_arm_imbalanced_results(self):
        for mutate in (
            lambda d: d["results"].append(copy.deepcopy(d["results"][0])),
            lambda d: d["results"].pop(),
            lambda d: d["results"][0].update({"arm": "rogue"}),
            lambda d: d["results"][0].update({"case": "../escape"}),
            lambda d: d["results"][0].update({"success": 1}),
            lambda d: d["results"][0].update({"effects": -2}),
            lambda d: d["success_by_arm"].update({"V": 999}),
        ):
            with self.subTest(mutate=mutate.__code__.co_firstlineno):
                data = receipt(3, ("N", "V", "P", "H"))
                mutate(data)
                with self.assertRaises(ValueError):
                    self.s.analyze(data)

    def test_reject_revoked_arm_success_without_denial_attempt(self):
        data = receipt()
        x = next(r for r in data["results"] if r["arm"] == "X")
        x["success"] = True
        x["governance_denial_success"] = True
        data["success_by_arm"]["X"] = 1
        with self.assertRaises(ValueError):
            self.s.analyze(data)

    def test_cli_reads_receipt_without_writing_files(self):
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            path.write_text(json.dumps(receipt(1, ("V",))), encoding="utf8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = self.s.main([str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.getvalue())["qualification"], "EXPLORATORY_ONLY")
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), ["receipt.json"])

    def test_current_pilot_scripted_fixture_receipt_contract(self):
        """Exercise actual producer seam, not a duplicated receipt fixture only."""
        report = current_pilot_v1_receipt()
        summary = self.s.analyze(report)
        self.assertEqual(summary["qualification"], "SOURCE_ONLY_FULL_LOCAL_FIXTURE")
        self.assertEqual(summary["receipt_experiment_version"], "neuro-sync-local-receiver-v1")
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["analysis_protocol"], "V1_STRATIFIED")
        self.assertEqual(summary["arms"]["X"]["successes"], 0)
        self.assertEqual(summary["arms"]["V"]["successes"], 2)
        scores = summary["arms"]["V"]["stratified_task_scores"]
        self.assertEqual(scores["possible_cases"], 10)
        self.assertEqual(scores["possible_plans_solved"], 0)
        self.assertEqual(scores["possible_plan_success_rate"], 0)
        self.assertEqual(scores["impossible_cases"], 2)
        self.assertEqual(scores["impossible_correct_abstentions"], 2)
        self.assertEqual(scores["impossible_correct_abstention_rate"], 1)
        self.assertEqual(next(p for p in summary["paired_comparisons"]
                              if p["arm_b"] == "P")["scenarios"], 10)
        self.assertEqual(summary["neural_transfer_established"], False)

    def test_legacy_v0_stays_legacy_and_cannot_be_relabelled_v1(self):
        legacy = receipt()
        summary = self.s.analyze(legacy)
        self.assertEqual(summary["analysis_protocol"], "LEGACY_V0_UNSTRATIFIED")
        self.assertEqual(summary["schema_version"], 1)
        self.assertNotIn("stratified_task_scores", summary["arms"]["V"])
        legacy["experiment"] = "neuro-sync-local-receiver-v1"
        with self.assertRaisesRegex(ValueError, "v1"):
            self.s.analyze(legacy)

    def test_v1_rejects_corrupt_labels_aggregates_and_digest_bindings(self):
        valid = current_pilot_v1_receipt()
        def mislabel_first(d):
            d["case_oracle_labels"][0]["expected_possible"] = False
        def misbalance(d):
            one = next(e for e in d["case_oracle_labels"][2:]
                       if e["oracle_minimum_steps"] == 1)
            one["oracle_minimum_steps"] = 2
        def corrupt_aggregate(d):
            d["scoring_by_arm"]["V"]["impossible_correct_abstentions"] = 99
        def wrong_category(d):
            positive = next(e["case"] for e in d["case_oracle_labels"]
                            if e["expected_possible"])
            row = next(r for r in d["results"]
                       if r["case"] == positive and r["arm"] == "V")
            row["success"] = True  # abstention cannot solve a possible task
            d["success_by_arm"]["V"] += 1
            d["scoring_by_arm"]["V"]["possible_plans_solved"] += 1
        def wrong_impossible_category(d):
            impossible = next(e["case"] for e in d["case_oracle_labels"]
                              if not e["expected_possible"])
            row = next(r for r in d["results"]
                       if r["case"] == impossible and r["arm"] == "V")
            row["abstained"] = False
            row["effects"] = 1
            row["gate_tested"] = True
            row["oracle_agree"] = True  # fabricated tool action is still not a valid impossibility success
        def tamper_one_request(d):
            row = next(r for r in d["results"] if r["arm"] == "V")
            row["request_shared_state_sha256"] = "0" * 64
        def tamper_prepared(d):
            d["prepared_reference_state_sha256_by_case"][0]["state_sha256_by_arm"]["H"] = "0" * 64
        def tamper_no_state(d):
            row = next(r for r in d["results"] if r["arm"] == "N")
            row["request_shared_state_sha256"] = "1" * 64
        def fake_min_two_success_with_one_effect(d):
            case = next(e["case"] for e in d["case_oracle_labels"]
                        if e["oracle_minimum_steps"] == 2)
            row = next(r for r in d["results"]
                       if r["case"] == case and r["arm"] == "V")
            row.update(success=True, abstained=False, gate_tested=True,
                       effects=1, oracle_agree=True)
            d["success_by_arm"]["V"] += 1
            d["scoring_by_arm"]["V"]["possible_plans_solved"] += 1
        def fake_nonabstaining_without_tool(d):
            row = next(r for r in d["results"] if r["arm"] == "V" and not r["success"])
            row["abstained"] = False  # no effect and no gate, impossible pilot state
        def fake_failed_impossible_abstention(d):
            case = next(e["case"] for e in d["case_oracle_labels"]
                        if not e["expected_possible"])
            row = next(r for r in d["results"]
                       if r["case"] == case and r["arm"] == "V")
            row["success"] = False
            d["success_by_arm"]["V"] -= 1
            d["scoring_by_arm"]["V"]["impossible_correct_abstentions"] -= 1
        def fake_failed_revoked_effect_denial(d):
            row = next(r for r in d["results"] if r["arm"] == "X")
            row.update(abstained=False, gate_tested=True, effects=0,
                       denied_attempts=1, success=False,
                       governance_denial_success=False)
        for tamper in (mislabel_first, misbalance, corrupt_aggregate, wrong_category,
                       wrong_impossible_category, tamper_one_request, tamper_prepared,
                       tamper_no_state, fake_min_two_success_with_one_effect,
                       fake_nonabstaining_without_tool, fake_failed_impossible_abstention,
                       fake_failed_revoked_effect_denial):
            with self.subTest(tamper=tamper.__name__):
                item = copy.deepcopy(valid)
                tamper(item)
                with self.assertRaises(ValueError):
                    self.s.analyze(item)

    def test_v1_partial_arm_receipt_stays_exploratory(self):
        report = current_pilot_v1_receipt(("V",))
        summary = self.s.analyze(report)
        self.assertEqual(summary["qualification"], "EXPLORATORY_ONLY")
        self.assertEqual(summary["arms"]["V"]["stratified_task_scores"][
            "impossible_correct_abstentions"], 2)

    def test_v1_twelve_case_prefix_of_larger_suite_remains_exploratory(self):
        report = current_pilot_v1_receipt(suite_size=16, max_cases=12)
        summary = self.s.analyze(report)
        self.assertEqual(report["evaluation_scope"], "EXPLORATORY_INCOMPLETE")
        self.assertEqual(summary["qualification"], "EXPLORATORY_ONLY")
        self.assertEqual(summary["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
