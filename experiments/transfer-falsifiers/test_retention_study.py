"""Negative controls for learned-recipient weight transfer claims."""

import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("retention_study.py")
spec = importlib.util.spec_from_file_location("neuro_sync_transfer_falsifier", PATH)
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)


class RetentionFalsifierTests(unittest.TestCase):
    def test_partition_has_disjoint_taught_queries_but_shared_subject(self):
        donor, recipient = study.partitions()
        self.assertEqual((len(donor), len(recipient)), (4, 2))
        self.assertFalse({(f.subject, f.relation) for f in donor} &
                         {(f.subject, f.relation) for f in recipient})
        self.assertTrue({f.subject for f in donor} & {f.subject for f in recipient})

    def test_delta_rejects_knowledge_bearing_receiver_and_evaluation_is_fresh(self):
        result = study.trial(seed=11, epochs=400)
        self.assertTrue(result["exact_base_delta_rejected_for_learned_recipient"])
        self.assertEqual(result["delta_rejection_type"], "ValueError")
        self.assertNotEqual(result["recipient_sha256_after_own_sgd"], result["base_sha256"])
        self.assertNotEqual(result["donor_sha256_after_sgd"], result["base_sha256"])
        self.assertEqual(result["train_updates"], {
            "donor": 1600, "C0_untouched_base": 0,
            "B_retention_only": 800, "C1_full_donor_copy": 0,
            "C3_aligned_distillation": 1600, "C4_wrong_pairing": 1600,
            "C5_direct_label_supervision": 1600,
        })
        outcomes = result["recipient_only_process_evaluations"]
        self.assertEqual(len(outcomes), 6)
        for arm, measurements in outcomes.items():
            with self.subTest(arm=arm):
                self.assertTrue(measurements["donor_taught"]["new_process"])
                self.assertTrue(measurements["recipient_retention"]["new_process"])
                self.assertEqual(measurements["donor_taught"]["total"], 4)
                self.assertEqual(measurements["recipient_retention"]["total"], 2)
                self.assertEqual(measurements["donor_taught"]["checkpoint_sha256"],
                                 measurements["recipient_retention"]["checkpoint_sha256"])
        self.assertEqual(outcomes["B_retention_only"]["recipient_retention"]["checkpoint_sha256"],
                         result["recipient_sha256_after_own_sgd"])
        self.assertEqual(outcomes["C1_full_donor_copy"]["donor_taught"]["checkpoint_sha256"],
                         result["donor_sha256_after_sgd"])

    def test_selected_seed_exposes_copy_loss_and_partial_distillation_forgetting(self):
        # This is one preselected toy seed, not a population generalization test.
        result = study.trial(seed=23, epochs=400)
        outcomes = result["recipient_only_process_evaluations"]
        scores = {arm: (o["donor_taught"]["correct"],
                        o["recipient_retention"]["correct"])
                  for arm, o in outcomes.items()}
        self.assertEqual(scores["B_retention_only"], (0, 2))
        self.assertEqual(scores["C1_full_donor_copy"], (4, 0))
        self.assertEqual(scores["C3_distill_on_learned_B"], (4, 1))
        self.assertEqual(scores["C4_wrong_pairing_on_learned_B"], (0, 1))
        self.assertEqual(scores["C5_direct_labels_on_learned_B"], (4, 1))

    def test_three_seed_receipt_distinguishes_measured_from_claimed(self):
        result = study.study(epochs=400)
        self.assertEqual([x["seed"] for x in result["trials"]], [11, 23, 47])
        self.assertEqual(result["statistical_claim"],
                         "posthoc descriptive only; three chosen seeds cannot prove generalization")
        self.assertEqual(result["mean_correct_by_arm"]["C3_distill_on_learned_B"],
                         {"donor_taught": 4.0, "recipient_retention": 1.0})
        self.assertEqual(result["mean_correct_by_arm"]["C3_distill_on_learned_B"],
                         result["mean_correct_by_arm"]["C5_direct_labels_on_learned_B"])
        self.assertTrue(all(x["exact_base_delta_rejected_for_learned_recipient"]
                            for x in result["trials"]))

    def test_rejects_invalid_study_budget(self):
        with self.assertRaises(ValueError):
            study.trial(seed=-1)
        with self.assertRaises(ValueError):
            study.trial(epochs=0)
        with self.assertRaises(ValueError):
            study.study(seeds=(11, 11))


if __name__ == "__main__":
    unittest.main()
