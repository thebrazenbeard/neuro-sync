"""Source-only transfer proposals cannot authorize TRAININGDAY physical runs."""

import copy
import unittest

from neurosync.trainingday_proposal import PROPOSAL_SCHEMA, propose_transfer


def digest(char):
    return char * 64


def proposal(method="exact-base-delta"):
    return {
        "schema": PROPOSAL_SCHEMA,
        "method": method,
        "lineage": {
            "donor_base_sha256": digest("1"),
            "donor_trained_sha256": digest("2"),
            "recipient_base_sha256": digest("1") if method == "exact-base-delta" else digest("3"),
            "donor_tokenizer_sha256": digest("4"),
            "recipient_tokenizer_sha256": digest("4") if method == "exact-base-delta" else digest("5"),
            "donor_config_sha256": digest("6"),
            "recipient_config_sha256": digest("6") if method == "exact-base-delta" else digest("7"),
            "adapter_config_sha256": digest("8") if method == "exact-base-delta" else None,
        },
        "input_artifacts": {
            "donor_trained": {"sha256": digest("2"), "size_bytes": 1048576},
            "recipient_base": {"sha256": digest("1") if method == "exact-base-delta" else digest("3"),
                               "size_bytes": 1048576},
            "train_manifest": {"sha256": digest("9"), "size_bytes": 1024},
            "sealed_holdout_manifest": {"sha256": digest("a"), "size_bytes": 2048},
            "shuffled_control_manifest": {"sha256": digest("b"), "size_bytes": 2048},
        },
        "data_rights": {
            "record_sha256": digest("c"),
            "rights_holder_label": "synthetic test steward",
            "usage_scope": "local-experiment-only",
        },
        "evaluation": {
            "donor_disconnected": True,
            "cold_recipient_reload": True,
            "baseline_arm": "untouched-recipient",
            "negative_control_arm": "shuffled-donor",
            "evaluator_custody": "independent-host-proposed",
        },
        "budget": {"max_external_cost_usd": 0, "max_gpu_minutes": 90},
    }


class TransferProposalTests(unittest.TestCase):
    def test_two_mechanisms_propose_but_cannot_authorize(self):
        for method in ("exact-base-delta", "receiver-native-distillation"):
            with self.subTest(method=method):
                accepted = propose_transfer(proposal(method))
                self.assertEqual(accepted["schema"], "NEUROSYNC_TRAININGDAY_TRANSFER_ENVELOPE_V1")
                self.assertEqual(accepted["method"], method)
                self.assertEqual(accepted["evidence_class"], "UNTRUSTED_TRANSFER_PROPOSAL")
                self.assertFalse(accepted["admission"]["trainingday_ready"])
                self.assertEqual(accepted["admission"]["physical_launch_authority"], "HOLD")
                self.assertEqual(accepted["admission"]["host_artifact_verification"], "UNVERIFIED")
                self.assertEqual(accepted["admission"]["protected_evaluator_custody"], "UNVERIFIED")
                self.assertRegex(accepted["proposal_sha256_untrusted"], r"^[a-f0-9]{64}$")

    def test_exact_base_requires_same_checkpoint_tokenizer_and_config(self):
        for field in ("recipient_base_sha256", "recipient_tokenizer_sha256",
                      "recipient_config_sha256"):
            altered = proposal()
            altered["lineage"][field] = digest("f")
            if field == "recipient_base_sha256":
                altered["input_artifacts"]["recipient_base"]["sha256"] = digest("f")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exact-base"):
                propose_transfer(altered)

    def test_manifest_integrity_and_split_separation(self):
        altered = proposal()
        altered["input_artifacts"]["donor_trained"]["sha256"] = digest("f")
        with self.assertRaisesRegex(ValueError, "lineage"):
            propose_transfer(altered)

        altered = proposal()
        altered["input_artifacts"]["sealed_holdout_manifest"] = copy.deepcopy(
            altered["input_artifacts"]["train_manifest"])
        with self.assertRaisesRegex(ValueError, "split"):
            propose_transfer(altered)

    def test_claimed_authority_and_bad_control_cannot_be_smuggled(self):
        altered = proposal()
        altered["host_authorized"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            propose_transfer(altered)
        altered = proposal()
        altered["evaluation"]["donor_disconnected"] = False
        with self.assertRaisesRegex(ValueError, "evaluation"):
            propose_transfer(altered)
        altered = proposal()
        altered["evaluation"]["evaluator_custody"] = "authorized"
        with self.assertRaisesRegex(ValueError, "custody"):
            propose_transfer(altered)

    def test_cost_limit_and_type_aliases(self):
        altered = proposal()
        altered["budget"]["max_external_cost_usd"] = 0.01
        with self.assertRaisesRegex(ValueError, "cost"):
            propose_transfer(altered)
        altered = proposal()
        altered["input_artifacts"]["train_manifest"]["size_bytes"] = True
        with self.assertRaisesRegex(ValueError, "artifact"):
            propose_transfer(altered)

    def test_no_zero_weight_path_promotion(self):
        altered = proposal()
        altered["method"] = "statebus-memory-share"
        with self.assertRaisesRegex(ValueError, "method"):
            propose_transfer(altered)


if __name__ == "__main__":
    unittest.main()
