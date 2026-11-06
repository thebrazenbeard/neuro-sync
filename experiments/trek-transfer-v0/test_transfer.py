"""No-network, no-dependency tests for the actual-SGD categorical pilot."""

import copy
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

    def test_checkpoint_rejects_malformed_tensor_shapes_and_metadata(self):
        original = self.base.checkpoint()
        self.assertEqual(TinyLearner.from_checkpoint(original).digest(), self.base.digest())

        def truncate_output(c):
            c["weights"]["output"].pop()

        def truncate_subject_width(c):
            c["weights"]["subject"][0].pop()

        def truncate_bias(c):
            c["weights"]["bias"].pop()

        def nonfinite_bias(c):
            c["weights"]["bias"][0] = float("nan")

        def boolean_weight(c):
            c["weights"]["output"][0][0] = True

        def duplicate_vocabulary(c):
            c["subjects"].append(c["subjects"][0])

        def reorder_vocabulary(c):
            c["answers"].reverse()

        def zero_hidden_width(c):
            c["hidden"] = 0

        def extreme_finite_weight(c):
            c["weights"]["output"][0][0] = 1e308

        for name, tamper in (
            ("truncated output rows", truncate_output),
            ("truncated input row", truncate_subject_width),
            ("truncated bias", truncate_bias),
            ("NaN bias", nonfinite_bias),
            ("boolean tensor value", boolean_weight),
            ("duplicated labels", duplicate_vocabulary),
            ("reordered labels", reorder_vocabulary),
            ("invalid hidden width", zero_hidden_width),
            ("finite but overflowing weight", extreme_finite_weight),
        ):
            with self.subTest(name=name):
                corrupted = copy.deepcopy(original)
                tamper(corrupted)
                with self.assertRaises(ValueError):
                    TinyLearner.from_checkpoint(corrupted)

    def test_transfer_rejects_incompatible_donor_hidden_width(self):
        for width in (self.base.hidden - 1, self.base.hidden + 1):
            with self.subTest(hidden=width):
                donor = TinyLearner(self.base.subjects, self.base.relations,
                                    self.base.answers, hidden=width, seed=11)
                with self.assertRaisesRegex(ValueError, "incompatible donor"):
                    delta_into_exact_base(self.base, donor, self.base.clone())

    def test_exact_base_delta_preserves_valid_boundary_weight_without_ulp_overshoot(self):
        base = self.base.clone()
        base.weights["output"][0][0] = -685011.8097196751
        self.assertEqual(base.clone().digest(), base.digest())
        donor = base.clone()
        donor.weights["output"][0][0] = 1_000_000.0
        self.assertEqual(donor.clone().digest(), donor.digest())
        transferred = delta_into_exact_base(base, donor, base.clone())
        self.assertEqual(transferred.weights["output"][0][0], 1_000_000.0)
        self.assertEqual(transferred.clone().digest(), donor.digest())

        signed_zero = base.clone()
        signed_zero.weights["output"][0][0] = -0.0
        self.assertEqual(delta_into_exact_base(base, signed_zero, base.clone()).digest(),
                         signed_zero.digest())

        invalid = donor.clone()
        invalid.weights["output"][0][0] = 1_000_000.01
        with self.assertRaisesRegex(ValueError, "invalid output weight"):
            delta_into_exact_base(base, invalid, base.clone())

    def test_training_rejects_nonprobability_targets_and_invalid_epochs(self):
        a = self.base.clone()
        before = a.digest()
        fake_distribution = [-1.0, 2.0] + [0.0] * (len(a.answers) - 2)
        with self.assertRaisesRegex(ValueError, "invalid target distribution"):
            a.sgd_step(TREK_FACTS[0].subject, TREK_FACTS[0].relation,
                       fake_distribution, 0.15)
        self.assertEqual(a.digest(), before)
        for bad_epochs in (-2, 1.5, True):
            with self.subTest(epochs=bad_epochs), self.assertRaisesRegex(ValueError, "epochs"):
                train(a, TREK_FACTS, [[1.0 / len(a.answers)] * len(a.answers)] * len(TREK_FACTS),
                      epochs=bad_epochs)
        for bad_lr in (-0.15, 0.0, float("nan"), float("inf")):
            with self.subTest(lr=bad_lr), self.assertRaisesRegex(ValueError, "learning rate"):
                train(a, TREK_FACTS, [[1.0 / len(a.answers)] * len(a.answers)] * len(TREK_FACTS),
                      epochs=1, lr=bad_lr)
            with self.subTest(sgd_lr=bad_lr), self.assertRaisesRegex(ValueError, "learning rate"):
                a.sgd_step(TREK_FACTS[0].subject, TREK_FACTS[0].relation,
                           [1.0 / len(a.answers)] * len(a.answers), bad_lr)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            train(a, [], [])

    def test_zero_epochs_preserves_existing_noop_contract(self):
        a = self.base.clone()
        prior = a.digest()
        receipt = train(a, TREK_FACTS, [[1.0 / len(a.answers)] * len(a.answers)] * len(TREK_FACTS),
                        epochs=0)
        self.assertEqual(receipt, {"updates": 0, "last_loss": None})
        self.assertEqual(a.digest(), prior)
        self.assertEqual(train(a, [], [], epochs=0)["updates"], 0)

    def test_constructor_and_loader_share_hidden_width_bounds(self):
        for hidden in (0, 4097, -1, True):
            with self.subTest(hidden=hidden), self.assertRaisesRegex(ValueError, "hidden width"):
                TinyLearner(self.base.subjects, self.base.relations,
                            self.base.answers, hidden=hidden)
        admitted = TinyLearner(self.base.subjects, self.base.relations,
                              self.base.answers, hidden=64)
        self.assertEqual(admitted.clone().digest(), admitted.digest())

    def test_overflowing_sgd_step_rolls_back_before_mutating_weights(self):
        a = self.base.clone()
        for index in range(len(a.weights["bias"])):
            a.weights["bias"][index] = 1_000_000.0
        # This is an accepted checkpoint; one ordinary target update would
        # increase a saturated logit beyond the loader's permitted domain.
        self.assertEqual(a.clone().digest(), a.digest())
        before = a.digest()
        target = [1.0] + [0.0] * (len(a.answers) - 1)
        with self.assertRaisesRegex(ValueError, "invalid output bias"):
            a.sgd_step(TREK_FACTS[0].subject, TREK_FACTS[0].relation, target, 0.15)
        self.assertEqual(a.digest(), before)
        self.assertEqual(a.clone().digest(), before)

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
