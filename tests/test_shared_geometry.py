
import copy
import unittest
from neurosync.peft_proposal_binding import _target_map, _canonical_sha, bind_peft_inspection_to_proposal
from neurosync.peft_adapter_only_proposal import target_map_sha256
from test_core_peft_proposal_binding import fixture, target_map_digest

def geom(name="base.layers.0.q_proj", rank=4):
    return dict(rank=rank,lora_alpha=8,target_modules=["q_proj"],
                lora_pairs=[dict(module=name,rank=rank,input_features=8,output_features=8,dtype="F32",target="q_proj")])
def inspection(g):return {**g,"lora_scaling":g["lora_alpha"]/g["rank"]}
class SharedGeometryTests(unittest.TestCase):
    def test_valid_v1_digest_preserved(self):
        g=geom()
        self.assertEqual(target_map_sha256(g),"c2b2103c9729428b3ec8f42a5f18575995384e04b5bc0c828216c9b1dc511c76")
        self.assertEqual(target_map_sha256(g),_canonical_sha(_target_map(inspection(g))))
    def test_both_forms_reject_malformed_modules_and_rank(self):
        for g in (geom("base..layers.0.q_proj"),geom(".base.layers.0.q_proj"),geom(rank=16385)):
            with self.subTest(geometry=g):
                with self.assertRaises(ValueError):target_map_sha256(g)
                with self.assertRaises(ValueError):_target_map(inspection(g))
    def test_actual_binder_refuses_coherently_rehashed_bad_geometry(self):
        for mode in ("double_dot","leading_dot","rank_over"):
            with self.subTest(mode=mode):
                obs,intent,envelope,claim=fixture()
                if mode=="double_dot":obs["lora_pairs"][0]["module"]="base..layers.0.q_proj"
                if mode=="leading_dot":obs["lora_pairs"][0]["module"]=".base.layers.0.q_proj"
                if mode=="rank_over":
                    obs["rank"]=16385;obs["lora_pairs"][0]["rank"]=16385
                    obs["lora_scaling"]=obs["lora_alpha"]/obs["rank"]
                claim["target_map_sha256"]=target_map_digest(obs)
                with self.assertRaises(ValueError):
                    bind_peft_inspection_to_proposal(obs,intent,envelope,claim)
    def test_scaling_still_bound(self):
        obs=inspection(geom());obs["lora_scaling"]=True
        with self.assertRaises(ValueError):_target_map(obs)
        obs["lora_scaling"]=999
        with self.assertRaises(ValueError):_target_map(obs)
    def test_normalization_preserves_input(self):
        g=geom();orig=copy.deepcopy(g);obs=inspection(g)
        _target_map(obs);target_map_sha256(g)
        self.assertEqual(g,orig)
