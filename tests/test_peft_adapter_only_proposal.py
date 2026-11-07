"""Adapter-only LoRA is a distinct Path A form, not a full trained checkpoint."""

import copy
import hashlib
import json
import unittest

from neurosync.peft_adapter_only_proposal import (
    ADAPTER_ONLY_SCHEMA, target_map_sha256, propose_adapter_only_transfer,
)


def sha(char):
    return char * 64


def intent():
    target_map = {
        "rank": 2,
        "lora_alpha": 4,
        "target_modules": ["q_proj"],
        "lora_pairs": [{
            "module": "base_model.model.layers.0.self_attn.q_proj",
            "rank": 2, "input_features": 3, "output_features": 4,
            "dtype": "BF16", "target": "q_proj",
        }],
    }
    return {
        "schema": ADAPTER_ONLY_SCHEMA,
        "method": "peft-adapter-only",
        "base": {
            "donor_base_weights_sha256": sha("1"),
            "recipient_base_weights_sha256": sha("1"),
            "donor_tokenizer_sha256": sha("2"),
            "recipient_tokenizer_sha256": sha("2"),
            "donor_config_sha256": sha("3"),
            "recipient_config_sha256": sha("3"),
            "base_revision": "e" * 40,
        },
        "adapter": {
            "weights_sha256": sha("4"),
            "config_sha256": sha("5"),
            "target_map": target_map,
            "target_map_sha256": target_map_sha256(target_map),
        },
        "input_artifacts": {
            name: {"sha256": digest, "size_bytes": size}
            for name, digest, size in (
                ("donor_base", sha("1"), 1000000),
                ("recipient_base", sha("1"), 1000000),
                ("adapter_weights", sha("4"), 70000),
                ("adapter_config", sha("5"), 500),
                ("train_manifest", sha("6"), 2000),
                ("sealed_holdout_manifest", sha("7"), 2000),
                ("shuffled_control_manifest", sha("8"), 2000),
            )
        },
        "training_provenance": {
            "source_revision_sha256": sha("9"),
            "optimizer_receipt_sha256": None,
            "status": "UNVERIFIED",
        },
        "data_rights": {
            "record_sha256": sha("a"),
            "rights_holder_label": "synthetic-test-steward",
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


class AdapterOnlyProposalTests(unittest.TestCase):
    def test_valid_adapter_only_has_no_changed_full_checkpoint_claim(self):
        proposed = propose_adapter_only_transfer(intent())
        self.assertEqual(proposed["evidence_class"], "UNTRUSTED_PEFT_ADAPTER_ONLY_PROPOSAL")
        self.assertFalse(proposed["admission"]["trainingday_ready"])
        self.assertEqual(proposed["admission"]["physical_launch_authority"], "HOLD")
        self.assertNotIn("donor_trained_checkpoint_sha256", json.dumps(proposed))
        self.assertEqual(proposed["proposed_artifact_digests"]["adapter_weights"], sha("4"))
        self.assertEqual(proposed["target_map_sha256_untrusted"], intent()["adapter"]["target_map_sha256"])

    def test_exact_unchanged_base_lineage_required(self):
        for key in ("recipient_base_weights_sha256", "recipient_tokenizer_sha256",
                    "recipient_config_sha256"):
            changed = intent()
            changed["base"][key] = sha("f")
            if key == "recipient_base_weights_sha256":
                changed["input_artifacts"]["recipient_base"]["sha256"] = sha("f")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exact base"):
                propose_adapter_only_transfer(changed)

    def test_adapter_roles_and_geometry_are_independently_bound(self):
        changed = intent()
        changed["input_artifacts"]["adapter_weights"]["sha256"] = sha("f")
        with self.assertRaisesRegex(ValueError, "artifact"):
            propose_adapter_only_transfer(changed)
        changed = intent()
        changed["adapter"]["weights_sha256"] = sha("1")
        changed["input_artifacts"]["adapter_weights"]["sha256"] = sha("1")
        with self.assertRaisesRegex(ValueError, "alias"):
            propose_adapter_only_transfer(changed)
        changed = intent()
        changed["adapter"]["target_map"]["lora_pairs"][0]["output_features"] += 1
        with self.assertRaisesRegex(ValueError, "target map digest"):
            propose_adapter_only_transfer(changed)

    def test_cross_role_aliases_cannot_masquerade_as_independent_artifacts(self):
        # Donor and recipient are allowed to share the *same role* (exact
        # frozen base), but adapter bytes/config cannot alias base,
        # tokenizer or model config: those represent different file types.
        for adapter_field in ("weights_sha256", "config_sha256"):
            for base_field in ("donor_base_weights_sha256", "donor_tokenizer_sha256",
                               "donor_config_sha256"):
                changed = intent()
                digest = changed["base"][base_field]
                changed["adapter"][adapter_field] = digest
                role = "adapter_weights" if adapter_field == "weights_sha256" else "adapter_config"
                changed["input_artifacts"][role]["sha256"] = digest
                with self.subTest(adapter=adapter_field, base=base_field), \
                        self.assertRaisesRegex(ValueError, "alias"):
                    propose_adapter_only_transfer(changed)

        changed = intent()
        changed["base"]["donor_tokenizer_sha256"] = sha("1")
        changed["base"]["recipient_tokenizer_sha256"] = sha("1")
        with self.assertRaisesRegex(ValueError, "alias"):
            propose_adapter_only_transfer(changed)

    def test_identical_digest_requires_identical_declared_byte_size(self):
        # Frozen donor/recipient base files are allowed identical digests; a
        # caller cannot describe the same bytes with contradictory lengths.
        changed = intent()
        changed["input_artifacts"]["recipient_base"]["size_bytes"] += 12345
        with self.assertRaisesRegex(ValueError, "size.*digest|digest.*size"):
            propose_adapter_only_transfer(changed)

    def test_target_map_digest_matches_documented_peft_join_encoding(self):
        target = intent()["adapter"]["target_map"]
        reference = hashlib.sha256(json.dumps(target, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
        self.assertEqual(target_map_sha256(target), reference)

    def test_self_consistent_forged_dotted_target_is_rejected(self):
        forged = intent()
        forged["adapter"]["target_map"]["lora_pairs"][0]["module"] = (
            "base_model..layers.0.self_attn.q_proj")
        # Malicious caller can recompute a *self-consistent* digest; the
        # source validator must reject the malformed module independently.
        forged["adapter"]["target_map_sha256"] = hashlib.sha256(json.dumps(
            forged["adapter"]["target_map"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        with self.assertRaisesRegex(ValueError, "target module"):
            propose_adapter_only_transfer(forged)

    def test_training_provenance_reference_remains_unverified(self):
        proposed = intent()
        proposed["training_provenance"]["status"] = "RECEIPT_REFERENCED_UNVERIFIED"
        proposed["training_provenance"]["optimizer_receipt_sha256"] = sha("b")
        output = propose_adapter_only_transfer(proposed)
        self.assertEqual(output["admission"]["optimizer_and_adapter_restore"], "UNVERIFIED")
        self.assertFalse(output["admission"]["trainingday_ready"])
        forged = copy.deepcopy(proposed)
        forged["training_provenance"]["status"] = "AUTHENTICATED"
        with self.assertRaisesRegex(ValueError, "provenance"):
            propose_adapter_only_transfer(forged)

    def test_no_authority_smuggling_or_cost_expansion(self):
        forged = intent()
        forged["host_authorized"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            propose_adapter_only_transfer(forged)
        forged = intent()
        forged["budget"]["max_external_cost_usd"] = 0.01
        with self.assertRaisesRegex(ValueError, "cost"):
            propose_adapter_only_transfer(forged)


if __name__ == "__main__":
    unittest.main()
