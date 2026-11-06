"""Read-only synthetic PEFT artifacts; no pretrained weights or host access."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from neurosync.peft_preflight import inspect_peft_lora_artifacts


class PEFTPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {
            "peft_type": "LORA", "base_model_name_or_path": "example/Qwen-4B",
            "revision": "f" * 40, "r": 2, "lora_alpha": 4,
            "target_modules": ["q_proj"], "bias": "none", "inference_mode": True,
        }
        self.a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        self.b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        self.tensors = {
            self.a: {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
            self.b: {"dtype": "BF16", "shape": [4, 2], "data_offsets": [12, 28]},
        }
        self.write_artifacts()

    def write_artifacts(self, *, header=None, config_text=None, data=None):
        h = json.dumps(self.tensors if header is None else header, separators=(",", ":")).encode()
        (self.root / "adapter_model.safetensors").write_bytes(
            struct.pack("<Q", len(h)) + h + (b"\x00" * 28 if data is None else data))
        (self.root / "adapter_config.json").write_text(
            json.dumps(self.config) if config_text is None else config_text, encoding="utf8")

    def test_exact_local_artifact_footprint_is_observed_not_authorized(self):
        result = inspect_peft_lora_artifacts(self.root)
        self.assertEqual(result["schema"], "NEUROSYNC_PEFT_ARTIFACT_PREFLIGHT_V1")
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["trainingday_ready"])
        self.assertEqual(len(result["adapter_sha256"]), 64)
        self.assertEqual(len(result["config_sha256"]), 64)
        self.assertEqual(result["base_model_name_or_path"], "example/Qwen-4B")
        self.assertEqual(result["base_revision"], "f" * 40)
        self.assertEqual(result["lora_pairs"], [{
            "module": "base_model.model.model.layers.0.self_attn.q_proj",
            "rank": 2, "input_features": 3, "output_features": 4,
            "dtype": "BF16", "target": "q_proj",
        }])
        self.assertIn("unverified base-model and tokenizer bytes", " ".join(result["holds"]))

    def test_refuses_hole_overlap_wrong_byte_count_and_tail(self):
        variants = []
        over = json.loads(json.dumps(self.tensors))
        over[self.b]["data_offsets"] = [11, 27]
        variants.append((over, b"\x00" * 28))
        hole = json.loads(json.dumps(self.tensors))
        hole[self.b]["data_offsets"] = [13, 29]
        variants.append((hole, b"\x00" * 29))
        wrong = json.loads(json.dumps(self.tensors))
        wrong[self.a]["shape"] = [2, 4]
        variants.append((wrong, b"\x00" * 28))
        variants.append((self.tensors, b"\x00" * 29))
        for header, data in variants:
            with self.subTest(header=header, size=len(data)):
                self.write_artifacts(header=header, data=data)
                with self.assertRaises(ValueError):
                    inspect_peft_lora_artifacts(self.root)

    def test_rejects_unpaired_or_mismatched_lora_rank(self):
        bad = json.loads(json.dumps(self.tensors))
        del bad[self.b]
        self.write_artifacts(header=bad, data=b"\x00" * 12)
        with self.assertRaises(ValueError):
            inspect_peft_lora_artifacts(self.root)
        bad = json.loads(json.dumps(self.tensors))
        bad[self.b]["shape"] = [2, 4]
        self.write_artifacts(header=bad, data=b"\x00" * 28)
        with self.assertRaises(ValueError):
            inspect_peft_lora_artifacts(self.root)

    def test_rejects_ambiguous_configs_and_headers(self):
        self.write_artifacts(config_text=json.dumps(self.config)[:-1] + ',"r":2}')
        with self.assertRaises(ValueError):
            inspect_peft_lora_artifacts(self.root)
        self.write_artifacts(config_text=json.dumps({**self.config, "r": True}))
        with self.assertRaises(ValueError):
            inspect_peft_lora_artifacts(self.root)
        self.write_artifacts(config_text=json.dumps({**self.config, "revision": None}))
        with self.assertRaises(ValueError):
            inspect_peft_lora_artifacts(self.root)
        self.write_artifacts()
        raw_header = json.dumps(self.tensors)
        dup_header = raw_header[:-1] + ',' + json.dumps(self.a) + ':' + json.dumps(self.tensors[self.a]) + '}'
        encoded = dup_header.encode()
        (self.root / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 28)
        with self.assertRaises(ValueError):
            inspect_peft_lora_artifacts(self.root)

    def test_rejects_unsupported_effectful_peft_variants(self):
        for update in ({"use_dora": True}, {"modules_to_save": ["lm_head"]},
                       {"rank_pattern": {"q_proj": 4}}, {"bias": "all"},
                       {"peft_type": "IA3"}, {"target_modules": ["k_proj"]},
                       {"target_modules": ["q_proj", "v_proj"]},
                       {"fan_in_fan_out": True}):
            with self.subTest(update=update):
                self.write_artifacts(config_text=json.dumps({**self.config, **update}))
                with self.assertRaises(ValueError):
                    inspect_peft_lora_artifacts(self.root)


if __name__ == "__main__":
    unittest.main()
