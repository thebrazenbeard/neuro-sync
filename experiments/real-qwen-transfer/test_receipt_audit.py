import copy
import math
import unittest
import receipt_audit

HEX = "a" * 64

def examples():
    train = {"source":"Qwen/Qwen3.5-0.8B-Base", "revision":"dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
             "steps":4, "loss_before":4.578, "loss_after":1.430,
             "lora_tensor_max_change":0.008, "adapter_files_sha256":{"adapter_model.safetensors":HEX}}
    recv = {"model":train["source"],"revision":train["revision"],"adapter_sha256":HEX,
            "base_losses":{"taught_exact":4.58,"unseen_paraphrase":4.69,"untaught_control":4.55},
            "adapter_losses":{"taught_exact":1.43,"unseen_paraphrase":1.89,"untaught_control":1.82},
            "training_steps_in_receiver":0,"generalization_established":False}
    return train,recv

class ConservativeReceipts(unittest.TestCase):
    def test_measured_adapter_effect_cannot_establish_generalization(self):
        a,b=examples()
        outcome=receipt_audit.classify_self_reported(a,b)
        self.assertEqual(outcome["classification"],"PERSISTENT_ADAPTER_EFFECT_UNBLINDED")
        self.assertTrue(outcome["untaught_target_also_improved"])
        self.assertFalse(outcome["generalization_proven"])
        self.assertFalse(outcome["neurosync_novelty_proven"])
    def test_mismatched_adapter_digest_fails(self):
        a,b=examples();b["adapter_sha256"]="b"*64
        with self.assertRaises(ValueError):
            receipt_audit.classify_self_reported(a,b)
    def test_incompatible_model_revision_fails(self):
        a,b=examples();b["revision"]="0"*40
        with self.assertRaises(ValueError):
            receipt_audit.classify_self_reported(a,b)
    def test_forged_generalization_claim_fails(self):
        a,b=examples();b["generalization_established"]=True
        with self.assertRaises(ValueError):
            receipt_audit.classify_self_reported(a,b)
    def test_bad_loss_or_boolean_coercion_fails(self):
        for value in (float("nan"),float("inf"),True,"1.2"):
            with self.subTest(value=str(value)):
                a,b=examples();b["adapter_losses"]["taught_exact"]=value
                with self.assertRaises(ValueError):
                    receipt_audit.classify_self_reported(a,b)
    def test_absent_positive_delta_fails(self):
        a,b=examples();a["lora_tensor_max_change"]=0
        with self.assertRaises(ValueError):
            receipt_audit.classify_self_reported(a,b)
    def test_taught_no_gain_classified_not_as_success(self):
        a,b=examples();b["adapter_losses"]["taught_exact"]=5
        r=receipt_audit.classify_self_reported(a,b)
        self.assertEqual(r["classification"],"NO_PERSISTENT_GAIN_OBSERVED")
        self.assertFalse(r["generalization_proven"])

if __name__=="__main__":
    unittest.main()
