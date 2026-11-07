"""Offline contract tests; no model or external provider calls."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


FILE = Path(__file__).with_name("heldout_contract.py")
spec = importlib.util.spec_from_file_location("mini_vera_heldout", FILE)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixture() -> tuple[dict, dict, str]:
    cases = []
    for index, family in enumerate(sorted(module.FAMILIES)):
        cases.append({"case_id": f"case_{index:04d}", "family": family,
                      "task_sha256": digest("private task " + str(index)),
                      "accepted_exact_tokens": ["ALLOW"]})
    cases.append({"case_id": "case_extra", "family": "correction",
                  "task_sha256": digest("private extra task"),
                  "accepted_exact_tokens": ["ALLOW"]})
    manifests = {arm: digest(arm) for arm in module.ARMS}
    plan = {
        "schema": module.SCHEMA,
        "evidence_class": "OPERATOR_SUPPLIED_PLAN",
        "order_seed": 20261107,
        "model_seed": 317,
        "temperature": 0,
        "declared_common_weight_sha256": digest("common weights"),
        "reported_manifest_digests_by_arm": manifests,
        "cases": cases,
    }
    plan_digest = hashlib.sha256(module._canonical(plan)).hexdigest()
    rows = []
    task_by_id = {case["case_id"]: case["task_sha256"] for case in cases}
    for index, item in enumerate(module.schedule(
            [case["case_id"] for case in cases], plan["order_seed"])):
        # Candidate arms improve on baseline for this controlled fixture, not
        # a prediction that local Qwen actually does so.
        answer = "DENY" if item["arm"] == "C0_BARE" else "ALLOW"
        rows.append({**item, "order_index": index,
                     "declared_weight_sha256": plan["declared_common_weight_sha256"],
                     "reported_model_manifest_digest": manifests[item["arm"]],
                     "model_seed": 317, "temperature": 0,
                     "request_sha256": digest("request" + str(index)),
                     "task_sha256": task_by_id[item["case_id"]],
                     "response_text": answer,
                     "response_sha256": digest(answer),
                     "process_epoch_id": "process_before_1" if
                         item["epoch"] == "PRE_RESTART" else "process_after_2",
                     "host_observed_guard_findings": {
                         "unauthorized_effect": "CLEAR",
                         "false_effect_claim": "CLEAR", "privacy_leak": "CLEAR"}})
    observations = {"schema": module.SCHEMA,
                    "plan_sha256": plan_digest, "records": rows}
    return plan, observations, plan_digest


class HeldoutContractTests(unittest.TestCase):
    def test_complete_five_arm_comparison_is_provisional(self):
        p, o, h = fixture()
        result = module.evaluate(p, o, plan_sha256=h)
        self.assertEqual("PROVISIONAL_OBJECTIVE_TOKEN_SCORES",
                         result["claim_ceiling"])
        self.assertEqual(120, result["total_rows"])
        self.assertEqual(0, result["scores"]["C0_BARE"]["PRE_RESTART"]["correct"])
        self.assertEqual(12, result["scores"]["C2_GOVERNED"]["POST_RESTART"]["correct"])
        self.assertEqual("NOT_TESTED", result["model_weight_update"])
        self.assertEqual("NOT_ESTABLISHED", result["semantic_generalization"])

    def test_does_not_publish_answers_or_private_cases(self):
        p, o, h = fixture()
        private = "PRIVATE_RELATIONSHIP_MARKER_DO_NOT_EXPORT"
        p["cases"][0]["accepted_exact_tokens"] = [private]
        # Only the public-safe result is returned; private text stays local.
        result = module.evaluate(p, o, plan_sha256=h)
        self.assertNotIn(private, json.dumps(result))
        self.assertNotIn("response_text", json.dumps(result))

    def test_forged_or_misordered_observations_rejected(self):
        p, o, h = fixture()
        for variation in ("swapped", "wrong_digest", "missing", "wrong_model",
                          "same_restart", "wrong_seed", "duplicate",
                          "changed_heldout_task", "extra_field"):
            with self.subTest(variation=variation):
                altered = copy.deepcopy(o)
                rows = altered["records"]
                if variation == "swapped":
                    rows[0], rows[1] = rows[1], rows[0]
                elif variation == "wrong_digest":
                    rows[0]["response_sha256"] = digest("false receipt")
                elif variation == "missing":
                    rows.pop()
                elif variation == "wrong_model":
                    rows[0]["reported_model_manifest_digest"] = "0" * 64
                elif variation == "same_restart":
                    for row in rows:
                        row["process_epoch_id"] = "process_reused_0"
                elif variation == "wrong_seed":
                    rows[0]["model_seed"] = 1
                elif variation == "duplicate":
                    rows[1] = rows[0].copy()
                elif variation == "changed_heldout_task":
                    rows[0]["task_sha256"] = digest("altered task")
                elif variation == "extra_field":
                    rows[0]["unbound_task_hint"] = "change my score"
                with self.assertRaises(module.ContractError):
                    module.evaluate(p, altered, plan_sha256=h)

    def test_trusted_guard_findings_are_not_inferred_from_chat(self):
        p, o, h = fixture()
        o["records"][0]["host_observed_guard_findings"]["unauthorized_effect"] = (
            "VIOLATION")
        result = module.evaluate(p, o, plan_sha256=h)
        self.assertEqual("HOLD_VIOLATIONS", result["claim_ceiling"])
        self.assertEqual("unauthorized_effect",
                         result["hard_guard_violations"][0]["category"])
        o["records"][0]["host_observed_guard_findings"]["unauthorized_effect"] = (
            "UNASSESSED")
        result = module.evaluate(p, o, plan_sha256=h)
        self.assertEqual("HOLD_UNASSESSED_GUARDS", result["claim_ceiling"])
        self.assertEqual(1, result["unassessed_guard_fields"])
        del o["records"][0]["host_observed_guard_findings"]["privacy_leak"]
        with self.assertRaises(module.ContractError):
            module.evaluate(p, o, plan_sha256=h)

    def test_private_plan_binding_is_exact_bytes(self):
        p, o, h = fixture()
        with self.assertRaisesRegex(module.ContractError, "exact private"):
            module.evaluate(p, o, plan_sha256="0" * 64)
        p["cases"].pop()
        with self.assertRaisesRegex(module.ContractError, "12..50"):
            module.evaluate(p, o, plan_sha256=h)

    def test_plan_rejects_missing_separation_or_weight_binding(self):
        for change in ("missing_family", "identity_forced", "bad_digest",
                       "nonzero_temperature", "small_suite", "duplicate_task"):
            with self.subTest(change=change):
                p, _, _ = fixture()
                if change == "missing_family":
                    p["cases"][1]["family"] = "correction"
                elif change == "identity_forced":
                    p["evidence_class"] = "INDEPENDENTLY_VERIFIED"
                elif change == "bad_digest":
                    p["declared_common_weight_sha256"] = "not-a-weight"
                elif change == "nonzero_temperature":
                    p["temperature"] = 0.2
                elif change == "small_suite":
                    p["cases"] = p["cases"][:11]
                elif change == "duplicate_task":
                    p["cases"][1]["task_sha256"] = p["cases"][0]["task_sha256"]
                with self.assertRaises(module.ContractError):
                    module.validate_plan(p)

    def test_write_new_only_and_private_input_guard(self):
        p, o, h = fixture()
        result = module.evaluate(p, o, plan_sha256=h)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            out = folder / "safe-result.json"
            module._write_new(out, result)
            with self.assertRaises(FileExistsError):
                module._write_new(out, result)
            local = folder / "private-plan.json"
            local.write_bytes(module._canonical(p))
            read, seen_sha = module._read_private_json(local)
            self.assertEqual(p, read)
            self.assertEqual(h, seen_sha)
            symlink = folder / "link.json"
            try:
                symlink.symlink_to(local)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(module.ContractError):
                module._read_private_json(symlink)

    def test_duplicate_json_keys_and_wrong_external_precommit_rejected(self):
        p, o, h = fixture()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            duplicated = folder / "duped.json"
            duplicated.write_text('{"schema":"x","schema":"y"}',
                                  encoding="utf-8")
            with self.assertRaisesRegex(module.ContractError, "duplicate"):
                module._read_private_json(duplicated)
            plan = folder / "plan.json"
            responses = folder / "observations.json"
            plan.write_bytes(module._canonical(p))
            responses.write_bytes(module._canonical(o))
            self.assertEqual(2, module.main([
                "--private-plan", str(plan), "--private-observations",
                str(responses), "--precommitted-plan-sha256", "0" * 64]))
            self.assertEqual(h, hashlib.sha256(plan.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
