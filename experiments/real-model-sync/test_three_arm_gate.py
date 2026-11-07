"""Offline assertions only: no local endpoint, model, host or bus calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


_spec = importlib.util.spec_from_file_location(
    "three_arm_gate", Path(__file__).with_name("three_arm_gate.py"))
gate = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(gate)


def make_receipt(arms=("V",), count=2, suite="a" * 64):
    meta = {
        "real_endpoint": True, "model_id": "fixture-identifier-not-a-real-model",
        "backend": "openai", "endpoint": "http://127.0.0.1:8080/v1/chat/completions",
        "source_file_sha256": gate._hash_file(gate.PILOT),
        "serving_process_file_binding_verified": False,
        "suite_self_commitment_sha256": suite,
        "request_config": {"timeout_seconds": 25, "max_tokens": 64,
                           "prompt_profile": "compact"},
    }
    rows = []
    reference_digests = []
    labels = []
    for i in range(count):
        case = f"case-{i:02d}-{suite[0] * 8}"
        minimum = (None if count == 12 and i >= 10 else
                   2 if i == 1 else
                   1 if count != 12 or i in (0, 2, 3, 4) else 2)
        labels.append({"case": case, "expected_possible": minimum is not None,
                       "oracle_minimum_steps": minimum})
        reference_digest = f"{i + 1:064x}"
        reference_digests.append({
            "case": case,
            "state_sha256_by_arm": {arm: reference_digest for arm in ("P", "H", "V")},
        })
        ordered = arms[i % len(arms):] + arms[:i % len(arms)]
        for arm in ordered:
            state_digest = gate.EMPTY_STATE_SHA256 if arm == "N" else reference_digest
            abstain = minimum is None and arm != "N"
            rows.append({"case": case, "arm": arm,
                         "success": arm != "N", "invalid_reply": False,
                         "abstained": abstain, "gate_tested": not abstain,
                         "effects": 0 if abstain else 1,
                         "denied_attempts": 0, "oracle_agree": True,
                         "model_call_elapsed_ms": 4000.0,
                         "request_shared_state_sha256": state_digest,
                         "transport_trace": {"outcome": "received_text",
                                             "prompt_sha256": "c" * 64,
                                             "shared_state_sha256": state_digest,
                                             "response_sha256": "d" * 64,
                                             "request_bytes": 1400,
                                             "timeout_seconds": 25,
                                             "max_tokens": 64,
                                             "prompt_profile": "compact"}})
    receipt = {
        "experiment": "neuro-sync-local-receiver-v1",
        "evidence_class": "LOCAL_HTTP_UNATTESTED",
        "evaluation_scope": "EXPLORATORY_INCOMPLETE",
        "model_evidence": meta, "results": rows,
        "case_oracle_labels": labels,
        "prepared_reference_state_sha256_by_case": reference_digests,
        "scenarios_evaluated": count, "arms_evaluated": list(arms),
        "success_by_arm": {arm: sum(row["success"] for row in rows if row["arm"] == arm)
                           for arm in arms},
    }
    refresh_scoring(receipt)
    return receipt


def refresh_scoring(receipt):
    labels = {entry["case"]: entry["expected_possible"]
              for entry in receipt["case_oracle_labels"]}
    scoring = {}
    for arm in receipt["arms_evaluated"]:
        if arm == "X":
            continue
        possible = [r for r in receipt["results"] if r["arm"] == arm and labels[r["case"]]]
        impossible = [r for r in receipt["results"] if r["arm"] == arm and not labels[r["case"]]]
        scoring[arm] = {
            "possible_cases": len(possible),
            "possible_plans_solved": sum(r["success"] for r in possible),
            "impossible_cases": len(impossible),
            "impossible_correct_abstentions": sum(
                r["success"] and r.get("abstained") is True for r in impossible),
        }
    receipt["scoring_by_arm"] = scoring


class ThreeArmReceiptGateTest(unittest.TestCase):
    def setUp(self):
        self.feasible = make_receipt()
        self.trial = make_receipt(("N", "P", "V"), 12, "b" * 64)

    def test_no_receipt_does_not_imply_work(self):
        self.assertEqual(gate.evaluate(None)["decision"], "NOT_RUN")

    def test_feasibility_preceding_trial_is_required(self):
        self.assertEqual(gate.evaluate(self.feasible)["decision"], "NOT_RUN")
        self.assertEqual(gate.evaluate(self.feasible)["feasibility"], "FEASIBLE_TO_ATTEMPT")

    def test_successes_are_descriptive_only(self):
        result = gate.evaluate(self.feasible, self.trial)
        self.assertEqual(result["decision"], "DESCRIPTIVE_ONLY")
        self.assertEqual(result["success_by_arm"], {"N": 0, "P": 12, "V": 12})
        self.assertIn("scripted", " ".join(result["limitations"]))

    def test_scripted_receipt_is_never_treated_as_real_model(self):
        self.feasible["evidence_class"] = "SCRIPTED_FIXTURE_TEST"
        result = gate.evaluate(self.feasible, self.trial)
        self.assertEqual(result["decision"], "NOT_FEASIBLE")
        self.assertIn("not_local_http_evidence", result["reasons"])

    def test_all_timeouts_have_no_valid_plan(self):
        for row in self.feasible["results"]:
            row.update({"invalid_reply": True, "gate_tested": False,
                        "success": False, "effects": 0,
                        "transport_trace": {"outcome": "transport_or_parse_failure",
                                            "prompt_sha256": "c" * 64,
                                            "shared_state_sha256": row["request_shared_state_sha256"],
                                            "request_bytes": 1400,
                                            "timeout_seconds": 25,
                                            "max_tokens": 64,
                                            "prompt_profile": "compact"}})
            row.pop("abstained")
        self.feasible["success_by_arm"]["V"] = 0
        refresh_scoring(self.feasible)
        result = gate.evaluate(self.feasible, self.trial)
        self.assertEqual(result["decision"], "NOT_FEASIBLE")
        self.assertEqual(result["valid_typed_nonabstaining"], 0)

    def test_abstention_does_not_pass_feasibility(self):
        self.feasible["results"][0]["abstained"] = True
        self.feasible["results"][0]["gate_tested"] = False
        self.feasible["results"][0]["effects"] = 0
        self.feasible["results"][0]["success"] = False
        self.feasible["success_by_arm"]["V"] = 1
        refresh_scoring(self.feasible)
        self.assertEqual(gate.evaluate(self.feasible)["valid_typed_nonabstaining"], 1)

    def test_one_arm_with_zero_valid_plans_is_not_feasible(self):
        for row in self.trial["results"]:
            if row["arm"] == "V":
                row.update({"invalid_reply": True, "gate_tested": False, "success": False})
        self.assertEqual(gate.evaluate(self.feasible, self.trial)["decision"], "NOT_FEASIBLE")

    def test_source_digest_mismatch_rejected(self):
        self.feasible["model_evidence"]["source_file_sha256"] = "0" * 64
        self.assertIn("pilot_source_bytes_differ", gate.evaluate(self.feasible)["reasons"])

    def test_configuration_mismatch_rejected(self):
        self.trial["model_evidence"]["request_config"]["max_tokens"] = 128
        self.assertIn("unmatched_generation_budget",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_reused_suite_rejected(self):
        self.trial["model_evidence"]["suite_self_commitment_sha256"] = "a" * 64
        self.assertIn("feasibility_and_trial_reuse_suite",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_missing_case_arm_rejected(self):
        self.trial["results"][-1]["case"] = "case-10-bbbbbbbb"
        self.assertEqual(gate.evaluate(self.feasible, self.trial)["decision"], "NOT_FEASIBLE")

    def test_p_to_v_budget_inequality_rejected(self):
        self.trial["model_evidence"]["request_config"]["timeout_seconds"] = 40
        self.assertEqual(gate.evaluate(self.feasible, self.trial)["decision"], "NOT_FEASIBLE")

    def test_transport_trace_required_for_model_produced_plan(self):
        self.feasible["results"][0]["transport_trace"] = {}
        self.assertEqual(gate.evaluate(self.feasible)["decision"], "NOT_FEASIBLE")

    def test_client_metadata_cannot_claim_process_binding(self):
        self.feasible["model_evidence"]["serving_process_file_binding_verified"] = True
        self.assertIn("unsupported_process_binding_claim", gate.evaluate(self.feasible)["reasons"])

    def test_model_file_change_between_stages_rejected(self):
        self.feasible["model_evidence"].update({
            "supplied_model_file_sha256_before": "d" * 64,
            "supplied_model_file_sha256_after": "d" * 64,
        })
        self.trial["model_evidence"].update({
            "supplied_model_file_sha256_before": "e" * 64,
            "supplied_model_file_sha256_after": "e" * 64,
        })
        self.assertIn("model_file_changed_between_stages",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_model_file_change_during_stage_rejected(self):
        self.feasible["model_evidence"].update({
            "supplied_model_file_sha256_before": "d" * 64,
            "supplied_model_file_sha256_after": "e" * 64,
        })
        self.assertIn("supplied_model_file_changed_or_unverifiable",
                      gate.evaluate(self.feasible)["reasons"])

    def test_backend_unhashable_rejected_without_exception(self):
        self.feasible["model_evidence"]["backend"] = ["openai"]
        self.assertIn("missing_model_runtime_id", gate.evaluate(self.feasible)["reasons"])

    def test_model_id_nonstring_truthy_rejected(self):
        self.feasible["model_evidence"]["model_id"] = {"fake": "model"}
        self.assertIn("missing_model_runtime_id", gate.evaluate(self.feasible)["reasons"])

    def test_invalid_reply_claiming_success_rejected(self):
        for row in self.trial["results"]:
            if row["arm"] == "P":
                row.update({"invalid_reply": True, "gate_tested": False,
                            "effects": 0, "success": True})
        self.assertIn("invalid_reply_claims_success_or_effect",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_per_request_generation_budget_mismatch_rejected(self):
        self.feasible["results"][0]["transport_trace"].update({
            "timeout_seconds": 120, "max_tokens": 2048, "prompt_profile": "legacy"})
        self.assertIn("unmatched_per_request_budget_or_trace",
                      gate.evaluate(self.feasible)["reasons"])

    def test_arm_sorted_rows_fail_cyclic_schedule(self):
        self.trial["results"].sort(key=lambda row: row["arm"])
        self.assertIn("non_counterbalanced_arm_order",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_reused_cases_with_different_suite_hash_rejected(self):
        for row in self.trial["results"]:
            if row["case"].startswith("case-00-"):
                row["case"] = "case-00-aaaaaaaa"
        self.assertIn("feasibility_and_trial_reuse_case_ids",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_truthy_oversized_model_id_rejected(self):
        self.feasible["model_evidence"]["model_id"] = "model" * 1000
        self.assertIn("missing_model_runtime_id", gate.evaluate(self.feasible)["reasons"])

    def test_abstention_cannot_claim_effect(self):
        self.feasible["results"][0].update({"abstained": True, "gate_tested": False})
        self.assertIn("abstention_claims_execution", gate.evaluate(self.feasible)["reasons"])

    def test_oversized_json_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "oversized.json"
            target.write_text(" " * 4_000_001, encoding="utf8")
            process = subprocess.run([sys.executable, str(gate.HERE / "three_arm_gate.py"),
                                      "--feasibility", str(target)],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["decision"], "NOT_FEASIBLE")

    def test_deep_json_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "deep.json"
            target.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf8")
            process = subprocess.run([sys.executable, str(gate.HERE / "three_arm_gate.py"),
                                      "--feasibility", str(target)],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["decision"], "NOT_FEASIBLE")

    def test_trial_endpoint_changed_rejected(self):
        self.trial["model_evidence"]["endpoint"] = "http://127.0.0.1:9999/v1/chat/completions"
        self.assertIn("changed_endpoint", gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_nonlocal_or_malformed_endpoint_rejected(self):
        for endpoint in ("https://example.com/v1/chat/completions", ["http://localhost:8080"],
                         "http://localhost:bad/v1/chat/completions"):
            with self.subTest(endpoint=endpoint):
                self.feasible["model_evidence"]["endpoint"] = endpoint
                self.assertIn("invalid_local_endpoint_claim",
                              gate.evaluate(self.feasible)["reasons"])

    def test_cli_failed_gate_returns_nonzero(self):
        self.feasible["model_evidence"]["backend"] = ["openai"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bad.json"
            target.write_text(json.dumps(self.feasible), encoding="utf8")
            process = subprocess.run([sys.executable, str(gate.HERE / "three_arm_gate.py"),
                                      "--feasibility", str(target)],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 1)
            self.assertEqual(json.loads(process.stdout)["decision"], "NOT_FEASIBLE")

    def test_missing_per_case_reference_hash_fails_closed(self):
        self.feasible.pop("prepared_reference_state_sha256_by_case")
        self.assertIn("missing_prepared_reference_state_digest",
                      gate.evaluate(self.feasible)["reasons"])

    def test_prepared_plain_host_bus_digest_divergence_fails_closed(self):
        self.feasible["prepared_reference_state_sha256_by_case"][0][
            "state_sha256_by_arm"]["H"] = "f" * 64
        self.assertIn("invalid_prepared_reference_state_digest",
                      gate.evaluate(self.feasible)["reasons"])

    def test_request_digest_must_match_prepared_reference(self):
        self.trial["results"][1]["request_shared_state_sha256"] = "f" * 64
        self.trial["results"][1]["transport_trace"]["shared_state_sha256"] = "f" * 64
        self.assertIn("request_prepared_reference_state_mismatch",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_transport_state_hash_cannot_disagree_with_request(self):
        self.feasible["results"][0]["transport_trace"]["shared_state_sha256"] = "f" * 64
        self.assertIn("unmatched_per_request_budget_or_trace",
                      gate.evaluate(self.feasible)["reasons"])

    def test_no_state_arm_cannot_report_hidden_reference(self):
        target = next(r for r in self.trial["results"] if r["arm"] == "N")
        target["request_shared_state_sha256"] = "e" * 64
        target["transport_trace"]["shared_state_sha256"] = "e" * 64
        self.assertIn("no_state_arm_contains_external_state",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_v0_legacy_receipt_cannot_claim_v1_experiment(self):
        self.trial["experiment"] = "neuro-sync-local-receiver-v0"
        self.assertIn("wrong_experiment_schema_version",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_case_label_balance_is_mandatory(self):
        self.trial["case_oracle_labels"][11].update({
            "oracle_minimum_steps": 2, "expected_possible": True,
        })
        self.assertIn("unexpected_trial_stratum_balance",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_cannot_conflate_possible_solution_with_impossible_abstention(self):
        impossible_row = next(r for r in self.trial["results"]
                              if r["case"].startswith("case-11-") and r["arm"] == "V")
        impossible_row.update({"abstained": False, "gate_tested": True, "effects": 1})
        self.assertIn("success_claims_wrong_solution_category",
                      gate.evaluate(self.feasible, self.trial)["reasons"])

    def test_separate_score_aggregate_cannot_be_fabricated(self):
        self.trial["scoring_by_arm"]["V"]["impossible_correct_abstentions"] = 5
        self.assertIn("scoring_aggregate_inconsistent",
                      gate.evaluate(self.feasible, self.trial)["reasons"])


if __name__ == "__main__":
    unittest.main()
