"""Synthetic cross-PR Path A binding, not a host or model test."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from neurosync.peft_proposal_binding import bind_peft_inspection_to_proposal


def d(char):
    return char * 64


def target_map_digest(inspection):
    payload = {
        "rank": inspection["rank"], "lora_alpha": inspection["lora_alpha"],
        "target_modules": inspection["target_modules"],
        "lora_pairs": inspection["lora_pairs"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def fixture():
    inspection = {
        "schema": "NEUROSYNC_PEFT_ARTIFACT_PREFLIGHT_V1",
        "candidate_only": True, "trainingday_ready": False,
        "evidence_class": "LOCAL_FILE_INSPECTION_SELF_REPORTED",
        "adapter_sha256": d("e"), "config_sha256": d("8"),
        "base_model_name_or_path": "example/Qwen4B", "base_revision": "f" * 40,
        "rank": 2, "lora_alpha": 4, "lora_scaling": 2.0,
        "target_modules": ["q_proj"],
        "lora_pairs": [{"module": "base_model.model.q_proj", "rank": 2,
                        "input_features": 3, "output_features": 4,
                        "dtype": "BF16", "target": "q_proj"}],
        "holds": ["no independent verification"],
    }
    intent = {
        "schema": "NEUROSYNC_TRANSFER_PROPOSAL_V1", "method": "exact-base-delta",
        "lineage": {
            "donor_base_sha256": d("1"), "donor_trained_sha256": d("2"),
            "recipient_base_sha256": d("1"), "donor_tokenizer_sha256": d("4"),
            "recipient_tokenizer_sha256": d("4"), "donor_config_sha256": d("6"),
            "recipient_config_sha256": d("6"), "adapter_config_sha256": d("8"),
        },
        "input_artifacts": {
            "donor_trained": {"sha256": d("2"), "size_bytes": 1024},
            "recipient_base": {"sha256": d("1"), "size_bytes": 1024},
            "train_manifest": {"sha256": d("9"), "size_bytes": 1024},
            "sealed_holdout_manifest": {"sha256": d("a"), "size_bytes": 1024},
            "shuffled_control_manifest": {"sha256": d("b"), "size_bytes": 1024},
        },
        "data_rights": {"record_sha256": d("c"),
                        "rights_holder_label": "synthetic", "usage_scope": "local-experiment-only"},
        "evaluation": {"donor_disconnected": True, "cold_recipient_reload": True,
                       "baseline_arm": "untouched-recipient",
                       "negative_control_arm": "shuffled-donor",
                       "evaluator_custody": "independent-host-proposed"},
        "budget": {"max_external_cost_usd": 0, "max_gpu_minutes": 60},
    }
    proposal = {
        "schema": "NEUROSYNC_TRAININGDAY_TRANSFER_ENVELOPE_V1",
        "method": "exact-base-delta",
        "proposal_sha256_untrusted": hashlib.sha256(json.dumps(
            intent, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest(),
        "evidence_class": "UNTRUSTED_TRANSFER_PROPOSAL",
        "proposed_artifact_digests": {
            key: value["sha256"] for key, value in intent["input_artifacts"].items()},
        "proposed_lineage": dict(intent["lineage"]),
        "proposed_rights_record_sha256": d("c"),
        "interpretation": "self-reported source-only intent",
        "admission": {
            "host_artifact_verification": "UNVERIFIED",
            "dataset_training_rights": "UNVERIFIED",
            "sealed_independent_evaluation": "UNVERIFIED",
            "protected_evaluator_custody": "UNVERIFIED",
            "optimizer_and_adapter_restore": "UNVERIFIED",
            "cold_recipient_reload": "UNVERIFIED",
            "serving_pid_model_binding": "UNVERIFIED",
            "physical_launch_authority": "HOLD",
            "trainingday_ready": False,
        },
    }
    binding = {
        "schema": "NEUROSYNC_PEFT_ARTIFACT_BINDING_CLAIM_V1",
        "donor_trained_checkpoint_sha256": d("2"),
        "adapter_weights_sha256": d("e"),
        "adapter_config_sha256": d("8"),
        "target_map_sha256": target_map_digest(inspection),
        "rights_record_sha256": d("c"),
    }
    return inspection, intent, proposal, binding


class BindingTests(unittest.TestCase):
    def test_matching_candidate_is_still_a_hold(self):
        inspected, intent, proposed, binding = fixture()
        result = bind_peft_inspection_to_proposal(inspected, intent, proposed, binding)
        self.assertEqual(result["schema"], "NEUROSYNC_PEFT_PROPOSAL_BINDING_V1")
        self.assertEqual(result["adapter_sha256_untrusted"], d("e"))
        self.assertEqual(result["donor_trained_checkpoint_sha256_untrusted"], d("2"))
        self.assertEqual(result["adapter_config_sha256_untrusted"], d("8"))
        self.assertFalse(result["trainingday_ready"])
        self.assertEqual(result["physical_launch_authority"], "HOLD")
        self.assertEqual(result["evidence_class"], "SELF_REPORTED_CROSS_CONSISTENCY_ONLY")

    def test_missing_or_wrong_adapter_weights_role_is_rejected(self):
        for role in ("adapter_weights_sha256", "donor_trained_checkpoint_sha256"):
            a, i, p, b = fixture()
            del b[role]
            with self.subTest(role=role), self.assertRaises(ValueError):
                bind_peft_inspection_to_proposal(a, i, p, b)

    def test_config_and_weight_hash_mismatches_are_rejected(self):
        for field in ("adapter_sha256", "config_sha256"):
            a, i, p, b = fixture()
            a[field] = d("f")
            with self.subTest(field=field), self.assertRaises(ValueError):
                bind_peft_inspection_to_proposal(a, i, p, b)

    def test_distinct_checkpoint_and_adapter_digests_cannot_alias(self):
        a, i, p, b = fixture()
        b["adapter_weights_sha256"] = b["donor_trained_checkpoint_sha256"]
        with self.assertRaises(ValueError):
            bind_peft_inspection_to_proposal(a, i, p, b)
        a, i, p, b = fixture()
        a["adapter_sha256"] = b["adapter_weights_sha256"] = i["lineage"]["recipient_base_sha256"]
        with self.assertRaises(ValueError):
            bind_peft_inspection_to_proposal(a, i, p, b)

    def test_same_module_names_but_different_shapes_fail_target_map_binding(self):
        a, i, p, b = fixture()
        a["lora_pairs"][0]["input_features"] = 99
        with self.assertRaises(ValueError):
            bind_peft_inspection_to_proposal(a, i, p, b)

    def test_cannot_bind_syntactically_impossible_inspector_target_map(self):
        a, i, p, b = fixture()
        a["target_modules"] = ["q_proj", "model.q_proj"]
        a["lora_pairs"].append({**a["lora_pairs"][0],
                                "module": "base_model.model.model.q_proj",
                                "target": "model.q_proj"})
        b["target_map_sha256"] = target_map_digest(a)
        with self.assertRaises(ValueError):
            bind_peft_inspection_to_proposal(a, i, p, b)

    def test_cannot_bind_forged_inspector_module_with_empty_segments(self):
        # The actual #38 tensor inspector requires nonempty dotted module
        # segments. A caller-supplied inspection dict cannot claim impossible
        # records merely by recomputing the self-reported target-map digest.
        for name in ("base_model..q_proj", ".base_model.q_proj"):
            a, i, p, b = fixture()
            a["lora_pairs"][0]["module"] = name
            b["target_map_sha256"] = target_map_digest(a)
            with self.subTest(name=name), self.assertRaises(ValueError):
                bind_peft_inspection_to_proposal(a, i, p, b)
        a, i, p, b = fixture()
        a["target_modules"] = ["q_proj\n"]
        a["lora_pairs"][0]["target"] = "q_proj\n"
        a["lora_pairs"][0]["module"] = "base_model.model.q_proj\n"
        b["target_map_sha256"] = target_map_digest(a)
        with self.assertRaises(ValueError):
            bind_peft_inspection_to_proposal(a, i, p, b)

    def test_exact_base_tokenizer_and_model_config_are_required(self):
        for field in ("recipient_base_sha256", "recipient_tokenizer_sha256",
                      "recipient_config_sha256"):
            a, i, p, b = fixture()
            i["lineage"][field] = d("f")
            p["proposed_lineage"][field] = d("f")
            p["proposal_sha256_untrusted"] = hashlib.sha256(json.dumps(
                i, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
            with self.subTest(field=field), self.assertRaises(ValueError):
                bind_peft_inspection_to_proposal(a, i, p, b)

    def test_wrong_rights_or_forged_envelope_digest_is_rejected(self):
        a, i, p, b = fixture()
        p["proposed_rights_record_sha256"] = d("f")
        with self.assertRaises(ValueError):
            bind_peft_inspection_to_proposal(a, i, p, b)
        a, i, p, b = fixture()
        p["proposal_sha256_untrusted"] = d("f")
        with self.assertRaises(ValueError):
            bind_peft_inspection_to_proposal(a, i, p, b)

    def test_cannot_join_distillation_or_promoted_approval(self):
        for mutated in ("method", "inspector_ready", "proposal_ready", "host_authority"):
            a, i, p, b = fixture()
            if mutated == "method":
                i["method"] = p["method"] = "receiver-native-distillation"
            elif mutated == "inspector_ready":
                a["trainingday_ready"] = True
            elif mutated == "proposal_ready":
                p["admission"]["trainingday_ready"] = True
            else:
                p["admission"]["physical_launch_authority"] = "ALLOW"
            with self.subTest(mutated=mutated), self.assertRaises(ValueError):
                bind_peft_inspection_to_proposal(a, i, p, b)

    def test_actual_inspector_and_proposal_modules_compose_when_integrated(self):
        # Draft #48 is deliberately standalone; this test only executes after
        # #37 and #38 arrive in the same source tree. The standalone branch
        # reports this integration test as SKIPPED, never as a verified join.
        try:
            from neurosync.peft_preflight import inspect_peft_lora_artifacts
            from neurosync.trainingday_proposal import propose_transfer
        except ImportError:
            self.skipTest("source-only PR #37/#38 modules not integrated")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
            header = {
                a: {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
                b: {"dtype": "BF16", "shape": [4, 2], "data_offsets": [12, 28]},
            }
            raw_header = json.dumps(header).encode()
            (root / "adapter_model.safetensors").write_bytes(
                struct.pack("<Q", len(raw_header)) + raw_header + b"\x00" * 28)
            (root / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "base_model_name_or_path": "example/Qwen4B",
                "revision": "f" * 40, "r": 2, "lora_alpha": 4,
                "target_modules": ["q_proj"], "bias": "none",
            }))
            observed = inspect_peft_lora_artifacts(root)
            _, intent, _, claim = fixture()
            intent["lineage"]["adapter_config_sha256"] = observed["config_sha256"]
            envelope = propose_transfer(intent)
            claim["adapter_weights_sha256"] = observed["adapter_sha256"]
            claim["adapter_config_sha256"] = observed["config_sha256"]
            claim["target_map_sha256"] = target_map_digest(observed)
            joined = bind_peft_inspection_to_proposal(observed, intent, envelope, claim)
            self.assertFalse(joined["trainingday_ready"])
            self.assertEqual(joined["physical_launch_authority"], "HOLD")
            observed["lora_pairs"][0]["input_features"] = 99
            with self.assertRaises(ValueError):
                bind_peft_inspection_to_proposal(observed, intent, envelope, claim)


if __name__ == "__main__":
    unittest.main()
