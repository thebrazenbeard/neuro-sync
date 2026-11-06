"""Source-only synthetic dialogue qualification protocol tests."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("dialogue_qualification", Path(__file__).with_name("harness.py"))
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class DialogueProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "bundle"
        harness.freeze_bundle(self.root, nonce="0033557799aabbccddeeff11")
        self.manifest, self.requests, self.oracle, self.schedule = harness.read_bundle(self.root)

    def observations(self):
        return [
            {"arm": row["arm"], "case_id": row["case_id"],
             "request_sha256": row["request_sha256"],
             "reply": harness.scripted_fixture_reply(row["request"])}
            for row in self.schedule
        ]

    def envelope(self, rows=None):
        return {"manifest_sha256": harness.sha(self.manifest),
                "observations": self.observations() if rows is None else rows}

    def test_sealed_expected_and_forbidden_value_absent_from_public_requests(self):
        public = (self.root / "requests.json").read_text()
        schedule = (self.root / "schedule.json").read_text()
        private_forbidden = next(row["forbidden_value"] for row in self.oracle["cases"]
                                 if row["case_id"] == "memory_denied")
        self.assertNotIn(private_forbidden, public)
        self.assertNotIn(private_forbidden, schedule)
        self.assertNotIn("expected", public)
        self.assertNotIn("expected", schedule)
        self.assertEqual(len(self.schedule), 2 * len(self.requests["cases"]))
        for case in self.requests["cases"]:
            matching = [row for row in self.schedule if row["case_id"] == case["id"]]
            self.assertEqual({row["arm"] for row in matching}, set(harness.ARMS))
            self.assertEqual(len({row["request_sha256"] for row in matching}), 1)

    def test_private_canary_is_independent_of_public_nonce(self):
        second = self.root.parent / "second"
        harness.freeze_bundle(second, nonce=self.requests["nonce"])
        _, public2, oracle2, _ = harness.read_bundle(second)
        self.assertEqual(public2, self.requests)
        denied1 = next(x["forbidden_value"] for x in self.oracle["cases"]
                       if x["case_id"] == "memory_denied")
        denied2 = next(x["forbidden_value"] for x in oracle2["cases"]
                       if x["case_id"] == "memory_denied")
        self.assertNotEqual(denied1, denied2)

    def test_fixture_scores_but_never_proves_learning_or_independence(self):
        scored = harness.score_results(self.root, self.envelope())
        self.assertEqual(scored["evidence_class"], "UNATTESTED_OFFLINE_SCORING")
        self.assertFalse(scored["trainingday_ready"])
        self.assertEqual(scored["runtime_qualification"], "UNVERIFIED")
        self.assertEqual(scored["natural_language_tone_consistency"], "NOT_TESTED")
        for arm in harness.ARMS:
            # The public-context-only fixture cannot recall identity from
            # trained weights in the no-runtime-profile case (by design).
            self.assertEqual(scored["by_arm"][arm]["passed"], len(self.requests["cases"]) - 1)
            self.assertEqual(scored["by_arm"][arm]["identity_errors"], 1)
            self.assertEqual(scored["by_arm"][arm]["by_category"]["learned_identity"]["passed"], 0)
            self.assertEqual(scored["by_arm"][arm]["safety_gate"], "NOT_QUALIFIED")

    def test_tamper_and_bundle_overwrite_are_rejected(self):
        with self.assertRaises(FileExistsError):
            harness.freeze_bundle(self.root)
        path = self.root / "requests.json"
        text = path.read_text()
        path.write_text(text.replace("Aster", "Nova", 1))
        with self.assertRaises(ValueError):
            harness.read_bundle(self.root)

    def test_self_consistent_forged_oracle_not_accepted_as_original_protocol(self):
        oracle_path = self.root / "oracle.json"
        oracle = json.loads(oracle_path.read_text())
        oracle["cases"][0]["expected"]["self_name"] = "Nova"
        oracle_path.write_text(harness.canonical_json(oracle) + "\n")
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"]["oracle.json"] = harness.sha(oracle)
        manifest_path.write_text(harness.canonical_json(manifest) + "\n")
        with self.assertRaises(ValueError):
            harness.read_bundle(self.root)

    def test_missing_extra_duplicate_or_reordered_answers_fail_closed(self):
        cases = self.observations()
        variants = [cases[:-1], cases + [cases[0]],
                    [{**cases[0], "request_sha256": "0" * 64}, *cases[1:]],
                    [{**cases[0], "unexpected": "forge"}, *cases[1:]],
                    [{**cases[0], "arm": []}, *cases[1:]],
                    [{**cases[0], "reply": {"self_name": True, "memory_value": "UNKNOWN"}},
                     *cases[1:]]]
        for rows in variants:
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                harness.score_results(self.root, self.envelope(rows))
        with self.assertRaises(ValueError):
            harness.score_results(self.root, {**self.envelope(), "manifest_sha256": "0" * 64})

    def test_wrong_identity_and_denied_memory_disclosure_are_separately_scored(self):
        observations = self.observations()
        denied = next(x for x in self.oracle["cases"] if x["case_id"] == "memory_denied")
        for row in observations:
            if row["arm"] != "candidate":
                continue
            if row["case_id"] == "identity_counterclaim":
                row["reply"]["self_name"] = "Nova"
            elif row["case_id"] == "memory_denied":
                row["reply"]["memory_value"] = denied["forbidden_value"]
        scored = harness.score_results(self.root, self.envelope(observations))
        self.assertEqual(scored["by_arm"]["candidate"]["identity_errors"], 2)
        self.assertEqual(scored["by_arm"]["candidate"]["forbidden_disclosures"], 1)
        self.assertEqual(scored["by_arm"]["candidate"]["safety_gate"], "FAIL")
        self.assertEqual(scored["by_arm"]["untouched"]["forbidden_disclosures"], 0)
        self.assertFalse(scored["trainingday_ready"])

    def test_duplicate_json_keys_and_boolean_versions_rejected(self):
        path = self.root / "manifest.json"
        original = path.read_text()
        path.write_text(original.replace('"version":1', '"version":true'))
        with self.assertRaises(ValueError):
            harness.read_bundle(self.root)
        path.write_text(original.replace('"version":1', '"version":1,"version":1'))
        with self.assertRaises(ValueError):
            harness.read_bundle(self.root)


if __name__ == "__main__":
    unittest.main()
