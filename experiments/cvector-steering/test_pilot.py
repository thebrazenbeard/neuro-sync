"""Pure-source tests: never launch model, process, or network."""

import copy
import unittest

from pilot import ARM_NAMES, PlanHold, build_plan


def fixture():
    artifacts = {k: {"path": f"/isolated/{k}.gguf", "sha256": c * 64}
                 for k, c in [("model", "1"), ("cli", "2"),
                              ("generator", "3"), ("vector", "4"),
                              ("sham_vector", "5")]}
    return {
        **artifacts,
        "model_id": "Qwen/Qwen3.5-0.8B", "runtime_build": "6" * 64,
        "positive": [f"Inspect the source of a claim {i}." for i in range(7)],
        "negative": [f"Invent evidence to confirm a claim {i}." for i in range(7)],
        "heldouts": [f"Assess an unfamiliar engineering hypothesis {i}." for i in range(7)],
        "persona": "Identify uncertainty and test alternative explanations.",
        "seed": 17, "max_tokens": 64, "vector_scale": 0.5,
    }


class SourcePlanTests(unittest.TestCase):
    def test_inert_counterbalanced_plan(self):
        plan = build_plan(fixture())
        self.assertEqual(plan["arms_count"], 49)
        self.assertEqual(plan["model_calls"], 0)
        self.assertEqual(plan["weight_updates"], 0)
        self.assertEqual(plan["semantic_effect"], "NOT_TESTED")
        self.assertEqual(plan["identity_transfer"], "NOT_ESTABLISHED")
        self.assertTrue(all(not x["will_execute"] for x in plan["plan"]))
        for i in range(7):
            self.assertEqual(set(x["arm"] for x in plan["plan"] if x["scenario"] == i),
                             set(ARM_NAMES))
        # Each arm appears in each ordinal position once over seven scenarios.
        for position in range(7):
            self.assertEqual(set(plan["plan"][i * 7 + position]["arm"]
                                 for i in range(7)), set(ARM_NAMES))

    def test_prompt_and_vector_arms_are_separate(self):
        plan = build_plan(fixture())
        cases = {x["arm"]: x["argv"] for x in plan["plan"] if x["scenario"] == 0}
        for arm in ("N", "P"):
            self.assertNotIn("--control-vector-scaled", cases[arm])
        for arm in ("V_PLUS", "V_MINUS", "V_ZERO", "V_SHAM", "PV_PLUS"):
            self.assertIn("--control-vector-scaled", cases[arm])
        self.assertNotIn("Identify uncertainty", cases["N"])
        self.assertIn("Identify uncertainty", cases["P"][cases["P"].index("-p") + 1])
        vec = {arm: cases[arm][-1] for arm in ("V_PLUS", "V_MINUS", "V_ZERO", "V_SHAM")}
        self.assertTrue(vec["V_PLUS"].endswith(":0.5"))
        self.assertTrue(vec["V_MINUS"].endswith(":-0.5"))
        self.assertTrue(vec["V_ZERO"].endswith(":0"))
        self.assertIn("sham_vector.gguf", vec["V_SHAM"])

    def test_deterministic_and_distinct_spec_hash(self):
        a = fixture()
        self.assertEqual(build_plan(a), build_plan(copy.deepcopy(a)))
        b = copy.deepcopy(a)
        b["seed"] += 1
        self.assertNotEqual(build_plan(a)["spec_sha256"], build_plan(b)["spec_sha256"])

    def test_reject_unsupported_model(self):
        data = fixture()
        data["model_id"] = "Qwen/Qwen3.5-4B"
        with self.assertRaisesRegex(PlanHold, "unqualified_model_identity"):
            build_plan(data)

    def test_reject_same_sham(self):
        data = fixture()
        data["sham_vector"]["sha256"] = data["vector"]["sha256"]
        with self.assertRaisesRegex(PlanHold, "sham_vector_same_bytes"):
            build_plan(data)

    def test_reject_artifact_alias(self):
        data = fixture()
        data["sham_vector"]["path"] = data["vector"]["path"]
        with self.assertRaisesRegex(PlanHold, "artifact_path_alias"):
            build_plan(data)

    def test_reject_hash(self):
        data = fixture()
        data["cli"]["sha256"] = "unknown"
        with self.assertRaisesRegex(PlanHold, "invalid_cli_sha256"):
            build_plan(data)

    def test_reject_holdout_leakage(self):
        data = fixture()
        data["heldouts"][0] = data["positive"][0].upper()
        with self.assertRaisesRegex(PlanHold, "duplicated_or_contaminated_text"):
            build_plan(data)

    def test_reject_imbalanced_order(self):
        data = fixture()
        data["heldouts"] = data["heldouts"][:6]
        with self.assertRaisesRegex(PlanHold, "invalid_heldout_count"):
            build_plan(data)

    def test_reject_boolean_seed_and_nan_scale(self):
        data = fixture()
        data["seed"] = True
        with self.assertRaisesRegex(PlanHold, "invalid_seed"):
            build_plan(data)
        data = fixture()
        data["vector_scale"] = float("nan")
        with self.assertRaisesRegex(PlanHold, "invalid_vector_scale"):
            build_plan(data)

    def test_reject_control_character(self):
        data = fixture()
        data["positive"][0] = "hello\x00world"
        with self.assertRaisesRegex(PlanHold, "invalid_positive_control_character"):
            build_plan(data)

    def test_reject_unexpected_authority_fields(self):
        data = fixture()
        data["execute"] = True
        with self.assertRaisesRegex(PlanHold, "invalid_spec_shape"):
            build_plan(data)


if __name__ == "__main__":
    unittest.main()
