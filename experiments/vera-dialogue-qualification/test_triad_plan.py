"""No-host source qualification for a three-arm Qwen adapter comparison."""

import importlib.util
from pathlib import Path
import unittest


root = Path(__file__).parent
spec = importlib.util.spec_from_file_location("neuro_triad_plan", root / "triad_plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def fixture():
    bundle = plan.expected_source_bundle()
    manifest = {"schema": "NEUROSYNC_BLIND_DIALOGUE_V2", "trial_id": "a" * 32,
                "arms": ["untouched", "candidate", "shuffled_control"],
                "requests_sha256": "4" * 64, "private_commitment_hmac_sha256": "5" * 64,
                "independent_custody": "UNVERIFIED", "trained_identity_proven": False}
    arms = {
        "untouched": {"role": "untouched-recipient", "base_weight_bytes_sha256": "6" * 64,
                      "adapter_weights_sha256": None, "adapter_config_sha256": None},
        "candidate": {"role": "candidate-adapter", "base_weight_bytes_sha256": "6" * 64,
                      "adapter_weights_sha256": "7" * 64, "adapter_config_sha256": "8" * 64},
        "shuffled_control": {"role": "shuffled-donor-control",
                             "base_weight_bytes_sha256": "6" * 64,
                             "adapter_weights_sha256": "9" * 64,
                             "adapter_config_sha256": "b" * 64},
    }
    intent = {"schema": "NEUROSYNC_QWEN35_PEFT_TRIAD_INTENT_V1",
              "holdout_manifest_sha256": plan.digest(manifest),
              "source_bundle_manifest_sha256": bundle["manifest_sha256"],
              "arms": arms}
    return manifest, bundle, intent


class TriadSourceQualificationTests(unittest.TestCase):
    def test_triad_is_declared_only_and_always_hard_hold(self):
        manifest, bundle, intent = fixture()
        outcome = plan.validate_triad_intent(manifest, bundle, intent)
        self.assertEqual(outcome["artifact_roles"]["untouched"], "untouched-recipient")
        self.assertEqual(outcome["artifact_roles"]["shuffled_control"], "shuffled-donor-control")
        self.assertFalse(outcome["base_weight_bytes_verified"])
        self.assertEqual(outcome["adapter_reload"], "UNVERIFIED")
        self.assertEqual(outcome["independent_holdout_custody"], "UNVERIFIED")
        self.assertEqual(outcome["physical_launch_authority"], "HOLD")
        self.assertFalse(outcome["trainingday_ready"])

    def test_cross_arm_base_or_control_adapter_alias_is_rejected(self):
        manifest, bundle, intent = fixture()
        intent["arms"]["candidate"]["base_weight_bytes_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            plan.validate_triad_intent(manifest, bundle, intent)
        manifest, bundle, intent = fixture()
        intent["arms"]["shuffled_control"]["adapter_weights_sha256"] = "7" * 64
        with self.assertRaises(ValueError):
            plan.validate_triad_intent(manifest, bundle, intent)

    def test_caller_must_not_label_source_metadata_as_full_model_bytes(self):
        _, bundle, _ = fixture()
        for metadata_sha in (bundle["manifest_sha256"],
                             *(shard["sha256_lfs_oid"] for shard in bundle["ordered_shards"])):
            with self.subTest(metadata_sha=metadata_sha[0]):
                manifest, bundle, intent = fixture()
                for role in intent["arms"].values():
                    role["base_weight_bytes_sha256"] = metadata_sha
                with self.assertRaises(ValueError):
                    plan.validate_triad_intent(manifest, bundle, intent)

    def test_wrong_trial_or_source_commitment_rejected(self):
        manifest, bundle, intent = fixture()
        intent["holdout_manifest_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            plan.validate_triad_intent(manifest, bundle, intent)
        manifest, bundle, intent = fixture()
        intent["source_bundle_manifest_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            plan.validate_triad_intent(manifest, bundle, intent)
        manifest, bundle, intent = fixture()
        manifest["trained_identity_proven"] = True
        with self.assertRaises(ValueError):
            plan.validate_triad_intent(manifest, bundle, intent)

    def test_typed_adapter_configs_cannot_alias_weight_or_metadata_digest(self):
        manifest, bundle, intent = fixture()
        illegal = (intent["arms"]["candidate"]["base_weight_bytes_sha256"],
                   bundle["manifest_sha256"], bundle["ordered_shards"][0]["sha256_lfs_oid"],
                   intent["arms"]["candidate"]["adapter_weights_sha256"],
                   intent["arms"]["shuffled_control"]["adapter_weights_sha256"])
        for candidate in illegal:
            with self.subTest(digest=candidate[:12]):
                manifest, bundle, intent = fixture()
                intent["arms"]["candidate"]["adapter_config_sha256"] = candidate
                with self.assertRaisesRegex(ValueError, "config"):
                    plan.validate_triad_intent(manifest, bundle, intent)
        # Identical adapter architecture/config between trained and shuffled
        # control is legitimate; weights must differ.
        manifest, bundle, intent = fixture()
        intent["arms"]["shuffled_control"]["adapter_config_sha256"] = (
            intent["arms"]["candidate"]["adapter_config_sha256"])
        self.assertFalse(plan.validate_triad_intent(manifest, bundle, intent)["trainingday_ready"])


if __name__ == "__main__":
    unittest.main()
