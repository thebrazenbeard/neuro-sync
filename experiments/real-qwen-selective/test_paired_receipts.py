
import copy
import hashlib
import json
import unittest
from selective_protocol import generate_suites, shuffled_training, review_paired_receipts, VERSION

MODEL="Qwen/Qwen3.5-0.8B-Base"
REV="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

def rows(suite, known, unknown, start=0):
    counts={}
    out=[]
    for e in suite["examples"]:
        v=e["variant"]; i=counts.get(v,0);counts[v]=i+1
        correct=start <= i < start+(unknown if v=="heldout_unknown" else known)
        answer=e["answer"] if correct else next(c for c in suite["candidates"] if c!=e["answer"])
        out.append({"id":e["id"],"variant":v,
                    "scores":{c:-1.0 if c==answer else -10.0 for c in suite["candidates"]}})
    return out

def fixture():
    records=[]
    for seed in (109,110,111):
        train,suite=generate_suites(seed)
        record={"seed":seed}
        for arm in ("true","shuffled"):
            manifest=train if arm=="true" else shuffled_training(train)
            digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
            adapter=hashlib.sha256(f"{seed}:{arm}".encode()).hexdigest()
            record[arm]={
                "train":{"source":MODEL,"revision":REV,"protocol":VERSION,"seed":seed,"arm":arm,
                         "steps":40,"learning_rate":.001,"trainable_parameters":24,
                         "losses":[1.0]*40,"train_manifest_sha256":digest,
                         "adapter_files_sha256":{"adapter_model.safetensors":adapter,"adapter_config.json":"a"*64}},
                "eval":{"source_revision":REV,"seed":seed,"arm":arm,"adapter_sha256":adapter,
                        "screen_pass":True,"raw_rows":{"base":rows(suite,2,2),"adapted":rows(suite,8 if arm=="true" else 2,4)}}}
        records.append(record)
    return records

class PairedReviewTests(unittest.TestCase):
    def test_positive_is_diagnostic_only(self):
        result=review_paired_receipts(fixture())
        self.assertTrue(result["paired_local_screen"])
        self.assertFalse(result["generalization_proven"])
        self.assertFalse(result["independent_custody_proven"])
        self.assertFalse(result["neurosync_novelty_proven"])
        self.assertEqual([v["paraphrase_delta"] for v in result["seeds"].values()],[.75]*3)
    def test_equal_perfect_arms_are_not_selective(self):
        records=fixture()
        for rec in records:
            rec["shuffled"]["eval"]["raw_rows"]["adapted"]=copy.deepcopy(rec["true"]["eval"]["raw_rows"]["adapted"])
        self.assertFalse(review_paired_receipts(records)["paired_local_screen"])
    def test_negative_control_reversal_and_unknown_regression(self):
        for mode in ("reversal","unknown"):
            with self.subTest(mode=mode):
                records=fixture();_,suite=generate_suites(110)
                records[1]["true"]["eval"]["raw_rows"]["adapted"]=rows(suite,6 if mode=="reversal" else 8,4 if mode=="reversal" else 3)
                if mode=="reversal":records[1]["shuffled"]["eval"]["raw_rows"]["adapted"]=rows(suite,8,4)
                self.assertFalse(review_paired_receipts(records)["paired_local_screen"])
    def test_false_claimed_screen_is_ignored(self):
        records=fixture()
        for rec in records:rec["true"]["eval"]["screen_pass"]=False
        self.assertTrue(review_paired_receipts(records)["paired_local_screen"])
    def test_missing_duplicate_or_extra_arm(self):
        for mode in ("missing","duplicate","extra","float"):
            with self.subTest(mode=mode):
                records=fixture()
                if mode=="missing":records.pop()
                if mode=="duplicate":records[1]=copy.deepcopy(records[0])
                if mode=="extra":records[0]["third"]=None
                if mode=="float":records[0]["seed"]=109.0
                with self.assertRaises(ValueError):review_paired_receipts(records)
    def test_bad_training_fields(self):
        for key,value in (("steps",1),("learning_rate",.01),("seed",109.0),("protocol",None),("losses",[1]),("losses",[float("nan")]*40),("trainable_parameters",0),("train_manifest_sha256","0"*64)):
            with self.subTest(key=key,value=str(value)[:10]):
                records=fixture();records[0]["true"]["train"][key]=value
                with self.assertRaises(ValueError):review_paired_receipts(records)
    def test_equal_aggregate_but_mismatched_baseline(self):
        records=fixture();_,suite=generate_suites(109)
        records[0]["shuffled"]["eval"]["raw_rows"]["base"]=rows(suite,2,2,start=2)
        with self.assertRaisesRegex(ValueError,"baseline predictions differ"):review_paired_receipts(records)
    def test_same_baseline_rank_with_different_scores(self):
        records=fixture()
        for row in records[0]["shuffled"]["eval"]["raw_rows"]["base"]:
            row["scores"]={k:v*2 for k,v in row["scores"].items()}
        self.assertTrue(review_paired_receipts(records)["paired_local_screen"])
    def test_pair_parameter_and_adapter_binding(self):
        for mode in ("params","adapter","same_digest","map"):
            with self.subTest(mode=mode):
                records=fixture();rec=records[0]
                if mode=="params":rec["shuffled"]["train"]["trainable_parameters"]=25
                if mode=="adapter":rec["true"]["eval"]["adapter_sha256"]="0"*64
                if mode=="same_digest":
                    d=rec["true"]["eval"]["adapter_sha256"]
                    rec["shuffled"]["train"]["adapter_files_sha256"]["adapter_model.safetensors"]=d
                    rec["shuffled"]["eval"]["adapter_sha256"]=d
                if mode=="map":rec["true"]["train"]["adapter_files_sha256"]["../file"]="a"*64
                with self.assertRaises(ValueError):review_paired_receipts(records)
    def test_missing_score_row(self):
        records=fixture();records[0]["true"]["eval"]["raw_rows"]["adapted"].pop()
        with self.assertRaises(ValueError):review_paired_receipts(records)
