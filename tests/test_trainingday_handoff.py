"""Cross-repo evidence boundary: actual local SGD pilot never promotes TRAININGDAY."""

import copy
from pathlib import Path
import sys
import unittest

from neurosync.trainingday_handoff import SCHEMA, mechanics_handoff


PILOT = Path(__file__).resolve().parents[1] / "experiments" / "trek-transfer-v0"
sys.path.insert(0, str(PILOT))
from run import experiment  # noqa: E402


class TrainingDayHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observed = experiment(dataset="trek")

    def test_real_source_pilot_is_observed_but_never_admitted(self):
        handoff = mechanics_handoff(self.observed)
        self.assertEqual(handoff["schema"], SCHEMA)
        self.assertEqual(handoff["evidence_class"], "SELF_REPORTED_TEST_RESULT")
        arms = handoff["trial_evaluations"]
        self.assertEqual(arms["C0_untouched"]["correct"], 1)
        self.assertEqual(arms["C1_full_donor_copy"]["correct"], 6)
        self.assertEqual(arms["C2_exact_base_delta"]["correct"], 6)
        self.assertEqual(arms["C3_output_distillation"]["correct"], 6)
        self.assertEqual(arms["C4_wrong_donor_shuffle"]["correct"], 0)
        self.assertFalse(handoff["admission"]["trainingday_ready"])
        self.assertEqual(handoff["admission"]["physical_launch_authority"], "HOLD")
        self.assertTrue(all(value == "UNVERIFIED" for key, value in
                            handoff["admission"].items() if key not in
                            ("trainingday_ready", "physical_launch_authority")))

    def test_no_results_can_claim_qwen_or_generalization(self):
        forged = copy.deepcopy(self.observed)
        forged["qualification"] = "real-Qwen-generalization"
        with self.assertRaisesRegex(ValueError, "qualification"):
            mechanics_handoff(forged)

    def test_no_missing_controls_or_sham_b_only_process(self):
        missing = copy.deepcopy(self.observed)
        del missing["evals_after_donor_removed"]["C4_wrong_donor_shuffle"]
        with self.assertRaisesRegex(ValueError, "control arms"):
            mechanics_handoff(missing)
        sham = copy.deepcopy(self.observed)
        sham["evals_after_donor_removed"]["C3_output_distillation"]["new_process"] = False
        with self.assertRaisesRegex(ValueError, "B-only"):
            mechanics_handoff(sham)

    def test_no_mismatched_lineage_or_mislabeled_copy_training(self):
        mismatched = copy.deepcopy(self.observed)
        mismatched["evals_after_donor_removed"]["C2_exact_base_delta"]["checkpoint_sha256"] = (
            mismatched["base_sha256"])
        with self.assertRaisesRegex(ValueError, "delta disagree"):
            mechanics_handoff(mismatched)
        mislabeled = copy.deepcopy(self.observed)
        mislabeled["recipient_training"]["C2_exact_base_delta"] = 1
        with self.assertRaisesRegex(ValueError, "conflated"):
            mechanics_handoff(mislabeled)


if __name__ == "__main__":
    unittest.main()
