"""No-network, no-dependency tests for the actual-SGD categorical pilot."""

import math
import unittest

from fixtures import SYNTHETIC_FACTS, TREK_FACTS
from learner import TinyLearner, delta_into_exact_base, train
from run import experiment


class LearnerTests(unittest.TestCase):
    def setUp(self):
        self.base = TinyLearner([f.subject for f in TREK_FACTS],
                                [f.relation for f in TREK_FACTS],
                                [f.answer for f in TREK_FACTS], seed=11)

    def test_untrained_clone_exact(self):
        a, b = self.base.clone(), self.base.clone()
        self.assertEqual(a.digest(), b.digest())
        self.assertEqual(a.digest(), self.base.digest())

    def test_train_is_actual_gradient_update_and_raises_accuracy(self):
        a = self.base.clone()
        targets = [[1.0 if answer == fact.answer else 0.0 for answer in a.answers]
                   for fact in TREK_FACTS]
        old = sum(a.predict(f.subject, f.relation) == f.answer for f in TREK_FACTS)
        receipts = train(a, TREK_FACTS, targets, epochs=700, lr=0.15)
        correct = sum(a.predict(f.subject, f.relation) == f.answer for f in TREK_FACTS)
        self.assertNotEqual(self.base.digest(), a.digest())
        self.assertEqual(receipts["updates"], 700 * len(TREK_FACTS))
        self.assertTrue(math.isfinite(receipts["last_loss"]))
        self.assertEqual(correct, len(TREK_FACTS))
        self.assertGreater(correct, old)

    def test_delta_rejects_modified_receiver(self):
        donor = self.base.clone()
        b = self.base.clone()
        b.weights["bias"][0] += 0.01
        with self.assertRaises(ValueError):
            delta_into_exact_base(self.base, donor, b)

    def test_public_fixtures_have_provenance_and_distinct_labels(self):
        self.assertEqual(len({(f.subject, f.relation) for f in TREK_FACTS}), len(TREK_FACTS))
        self.assertEqual(len({f.answer for f in TREK_FACTS}), len(TREK_FACTS))
        self.assertTrue(all(f.source_type == "official" and f.source_url.startswith("https://www.startrek.com/")
                            for f in TREK_FACTS))
        self.assertTrue(all(f.source_type == "synthetic_noncanon" and not f.source_url
                            for f in SYNTHETIC_FACTS))


class ExperimentalControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = experiment("trek", epochs=700)

    def test_initial_recipient_is_untouched_and_isolated(self):
        result = self.result
        self.assertTrue(result["initial_donor_equals_untouched_recipient"])
        self.assertEqual(result["base_sha256"],
                         result["evals_after_donor_removed"]["C0_untouched"]["checkpoint_sha256"])
        self.assertNotEqual(result["base_sha256"], result["donor_after_sgd_sha256"])
        self.assertEqual(result["recipient_training"]["C0_untouched"], 0)
        self.assertTrue(all(item["new_process"] for item in result["evals_after_donor_removed"].values()))

    def test_copy_and_delta_positive_controls(self):
        result = self.result["evals_after_donor_removed"]
        for arm in ("C1_full_donor_copy", "C2_exact_base_delta"):
            self.assertEqual(result[arm]["correct"], len(TREK_FACTS))
        self.assertEqual(result["C1_full_donor_copy"]["predictions"],
                         result["C2_exact_base_delta"]["predictions"])

    def test_no_copy_distillation_and_wrong_donor(self):
        evals = self.result["evals_after_donor_removed"]
        self.assertEqual(evals["C3_output_distillation"]["correct"], len(TREK_FACTS))
        self.assertLess(evals["C4_wrong_donor_shuffle"]["correct"],
                        evals["C3_output_distillation"]["correct"])
        self.assertNotEqual(evals["C3_output_distillation"]["checkpoint_sha256"],
                            evals["C1_full_donor_copy"]["checkpoint_sha256"])
        self.assertEqual(self.result["recipient_training"]["C3_output_distillation"],
                         700 * len(TREK_FACTS))

    def test_determinism_and_noncanonical_variant(self):
        second = experiment("trek", epochs=700)
        self.assertEqual(second["base_sha256"], self.result["base_sha256"])
        self.assertEqual(second["evals_after_donor_removed"],
                         self.result["evals_after_donor_removed"])
        noncanon = experiment("synthetic", epochs=700)
        self.assertEqual(noncanon["facts_source_types"], ["synthetic_noncanon"])
        self.assertEqual(noncanon["evals_after_donor_removed"]["C1_full_donor_copy"]["correct"],
                         len(SYNTHETIC_FACTS))


if __name__ == "__main__":
    unittest.main()
