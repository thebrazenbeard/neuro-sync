"""Real synthetic local PEFT bytes through three source-only draft contracts.

This is a TEST of typed correlations, not an authenticated host receipt, model
training result, PEFT loader, or authorization to launch anything.
"""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from neurosync.peft_preflight import inspect_peft_lora_artifacts
from neurosync.peft_proposal_binding import bind_peft_inspection_to_proposal
from neurosync.trainingday_proposal import propose_transfer

from test_core_peft_proposal_binding import fixture, target_map_digest


class ActualBytesThroughBinderTests(unittest.TestCase):
    def test_file_hashes_remain_distinct_from_full_donor_checkpoint_even_if_claims_cohere(self):
        with tempfile.TemporaryDirectory(prefix="ns-peft-bridge-") as folder:
            root = Path(folder)
            a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
            header = {
                a: {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
                b: {"dtype": "BF16", "shape": [4, 2], "data_offsets": [12, 28]},
            }
            encoded = json.dumps(header).encode("utf8")
            weights = root / "adapter_model.safetensors"
            weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 28)
            config = root / "adapter_config.json"
            config.write_text(json.dumps({
                "peft_type": "LORA", "base_model_name_or_path": "example/Qwen4B",
                "revision": "f" * 40, "r": 2, "lora_alpha": 4,
                "target_modules": ["q_proj"], "bias": "none",
            }), encoding="utf8")

            inspected = inspect_peft_lora_artifacts(root)
            self.assertEqual(inspected["adapter_sha256"],
                             hashlib.sha256(weights.read_bytes()).hexdigest())
            self.assertEqual(inspected["config_sha256"],
                             hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertNotEqual(inspected["adapter_sha256"], inspected["config_sha256"])

            _, intent, _, claim = fixture()
            intent["lineage"]["adapter_config_sha256"] = inspected["config_sha256"]
            envelope = propose_transfer(intent)
            claim["adapter_weights_sha256"] = inspected["adapter_sha256"]
            claim["adapter_config_sha256"] = inspected["config_sha256"]
            claim["target_map_sha256"] = target_map_digest(inspected)
            result = bind_peft_inspection_to_proposal(inspected, intent, envelope, claim)
            self.assertEqual(result["adapter_sha256_untrusted"], inspected["adapter_sha256"])
            self.assertEqual(result["adapter_config_sha256_untrusted"], inspected["config_sha256"])
            self.assertEqual(result["donor_trained_checkpoint_sha256_untrusted"], "2" * 64)
            self.assertNotEqual(result["donor_trained_checkpoint_sha256_untrusted"],
                                result["adapter_sha256_untrusted"])
            self.assertFalse(result["trainingday_ready"])
            self.assertEqual(result["physical_launch_authority"], "HOLD")

            # A malicious caller coherently relabels the adapter as a *full*
            # checkpoint across intent, envelope and binding claim. This is a
            # semantic role violation, even though their declared hashes agree.
            intent["lineage"]["donor_trained_sha256"] = inspected["adapter_sha256"]
            intent["input_artifacts"]["donor_trained"]["sha256"] = inspected["adapter_sha256"]
            envelope = propose_transfer(intent)
            claim["donor_trained_checkpoint_sha256"] = inspected["adapter_sha256"]
            with self.assertRaisesRegex(ValueError, "roles must be distinct"):
                bind_peft_inspection_to_proposal(inspected, intent, envelope, claim)


if __name__ == "__main__":
    unittest.main()
