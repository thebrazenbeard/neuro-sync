"""Exploratory untaught-pair and recipient-retention controls (not LLM tests)."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from learner import TinyLearner, sha256, train
from untaught import experiment_untaught, receiver_only_eval
from fixtures import SYNTHETIC_FACTS


class UntaughtTransferTests(unittest.TestCase):
    def test_disjoint_taught_and_untaught_pairs_have_separate_scores(self):
        result = experiment_untaught("synthetic", model_seed=11, epochs=70)
        self.assertEqual(result["evidence_class"], "SOURCE_TEST_CATEGORICAL_UNTAUGHT")
        self.assertEqual(result["split"]["taught_count"], 4)
        self.assertEqual(result["split"]["untaught_count"], 2)
        self.assertEqual(len(result["split"]["commitment_sha256"]), 64)
        self.assertFalse(set(map(tuple, result["split"]["taught_pairs"])) &
                         set(map(tuple, result["split"]["untaught_pairs"])))
        for arm in ("C0_untouched", "C1_full_donor_copy", "C2_exact_base_delta",
                    "C3_output_distillation", "C4_wrong_donor_shuffle"):
            evaluation = result["strict_untouched_b"][arm]
            self.assertEqual(evaluation["taught"]["total"], 4)
            self.assertEqual(evaluation["untaught"]["total"], 2)
            self.assertTrue(evaluation["taught"]["new_process"])
            self.assertTrue(evaluation["untaught"]["new_process"])

    def test_retention_variant_is_separate_from_untouched_b(self):
        result = experiment_untaught("synthetic", epochs=80)
        prior = result["separate_prior_b_retention"]
        self.assertEqual(prior["prior_training_updates"], 2 * 80)
        self.assertEqual(prior["transfer_training_updates"], 4 * 80)
        self.assertEqual(prior["before_transfer"]["total"], 2)
        self.assertEqual(prior["after_transfer"]["total"], 2)
        self.assertEqual(prior["new_taught_after_transfer"]["total"], 4)
        self.assertNotEqual(prior["prior_recipient_sha256"], result["base_sha256"])
        self.assertEqual(result["strict_untouched_b_recipient_training"]["C0_untouched"], 0)

    def test_strict_donor_and_distillation_never_train_on_untaught_pairs(self):
        calls = []
        teacher_queries = []
        original_distribution = TinyLearner.distribution

        def recording_train(model, facts, targets, **kwargs):
            calls.append(tuple((f.subject, f.relation) for f in facts))
            return train(model, facts, targets, **kwargs)

        def recording_distribution(model, subject, relation):
            teacher_queries.append((subject, relation))
            return original_distribution(model, subject, relation)

        with patch("untaught.train", side_effect=recording_train), \
                patch.object(TinyLearner, "distribution", recording_distribution):
            result = experiment_untaught("synthetic", epochs=10)
        taught = tuple(map(tuple, result["split"]["taught_pairs"]))
        untaught = tuple(map(tuple, result["split"]["untaught_pairs"]))
        # Donor supervision, C3 distillation, C4 wrong-distribution control,
        # then separate B-prior and B-transfer runs; nothing scores while train runs.
        self.assertEqual(calls, [taught, taught, taught, untaught, taught])
        self.assertEqual(teacher_queries, list(taught))

    def test_invalid_setup_fails_without_training(self):
        with self.assertRaises(ValueError):
            experiment_untaught("unknown")
        with self.assertRaises(ValueError):
            experiment_untaught("synthetic", epochs=0)

    def test_receiver_eval_requires_bound_ordered_query_digest(self):
        facts = SYNTHETIC_FACTS[:2]
        model = TinyLearner([f.subject for f in SYNTHETIC_FACTS],
                            [f.relation for f in SYNTHETIC_FACTS],
                            [f.answer for f in SYNTHETIC_FACTS])
        forged = {"checkpoint_sha256": sha256(model.checkpoint()),
                  "predictions": [facts[0].answer, facts[1].answer]}
        with tempfile.TemporaryDirectory() as scratch:
            with patch("untaught.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(forged))):
                with self.assertRaisesRegex(ValueError, "schema"):
                    receiver_only_eval(model, facts, scratch)


if __name__ == "__main__":
    unittest.main()
