import json
import unittest
from unittest.mock import patch
from neurosync import peft_preflight as module

class ConfigContractTests(unittest.TestCase):
    def parse(self, **overrides):
        config = {"peft_type": "LORA", "base_model_name_or_path": "example/Qwen",
                  "revision": "a"*40, "r": 2, "lora_alpha": 4,
                  "target_modules": ["q_proj"], "bias": "none", **overrides}
        raw = json.dumps(config).encode("utf8")
        with patch.object(module, "_read_fd_range", return_value=raw):
            return module._read_config(-1, len(raw))

    def test_disabled_variant_requires_boolean_false(self):
        for key in ("use_dora", "use_rslora", "use_qalora", "use_bdlora"):
            for value in (None, 0, 0.0, "", [], {}, True):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        self.parse(**{key: value})

    def test_ordinary_default_variants_remain_supported(self):
        self.assertEqual(self.parse()["r"], 2)
        self.assertEqual(self.parse(use_dora=False, use_rslora=False,
                                    use_qalora=False, use_bdlora=False)["r"], 2)

    def test_numeric_validation_has_consistent_error_type(self):
        for key in ("lora_alpha", "lora_dropout"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.parse(**{key: 10**400})

    def test_valid_finite_numeric_config(self):
        for alpha in (4, 4.0, 0.25, 1e300):
            self.assertEqual(self.parse(lora_alpha=alpha, lora_dropout=0.1)["lora_alpha"], alpha)
