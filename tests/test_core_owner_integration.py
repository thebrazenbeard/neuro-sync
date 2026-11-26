
from dataclasses import replace
import json
import unittest
from neurosync import Tensor, ModelSnapshot, ScopedGrant, DeltaBundle, build_delta_bundle, apply_delta_bundle, rollback_transfer, snapshot_to_json, snapshot_from_json
class OwnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.a=ModelSnapshot("A","test","v1",(("w",Tensor.from_nested([-0.,2.])),))
        self.t=ModelSnapshot("A","test","v1",(("w",Tensor.from_nested([1.,3.])),))
        self.b=ModelSnapshot("B","test","v1",self.a.tensors)
        self.d=build_delta_bundle(self.a,self.t,receiver_id="B",transfer_id="t")
        c=dict(transfer_id="t",base_fingerprint=self.b.fingerprint(),manifest_sha256=self.d.manifest.sha256(),allowed_tensors=("w",),valid_from_epoch=1000,expires_epoch=2000)
        self.g=ScopedGrant(actor_id="A",counterpart_id="B",operation="export-delta",**c)
        self.h=ScopedGrant(actor_id="B",counterpart_id="A",operation="import-delta",**c)
    def test_zero_preserves_bits_and_receipt(self):
        for strength in (0.,-0.):
            with self.subTest(strength=strength.hex()):
                out,receipt=apply_delta_bundle(self.b,self.d,self.g,self.h,strength=strength,now_epoch=1500)
                self.assertEqual(out.fingerprint(),self.b.fingerprint())
                self.assertEqual(receipt.before_fingerprint,receipt.after_fingerprint)
                self.assertEqual(rollback_transfer(out,receipt).fingerprint(),self.b.fingerprint())
    def test_zero_still_requires_both_grants(self):
        for g,h in ((replace(self.g,enabled=False),self.h),(self.g,replace(self.h,enabled=False))):
            with self.assertRaises(PermissionError):
                apply_delta_bundle(self.b,self.d,g,h,strength=0.,now_epoch=1500)
    def test_zero_still_validates_manifest(self):
        d=DeltaBundle(self.d.manifest,(("w",Tensor.from_nested([1.,9.])),),self.d.targets)
        with self.assertRaisesRegex(ValueError,"manifest hash"):
            apply_delta_bundle(self.b,d,self.g,self.h,strength=0.,now_epoch=1500)
    def test_reject_unknown_snapshot_and_tensor_fields(self):
        for target in ("snapshot","tensor"):
            with self.subTest(target=target):
                decoded=json.loads(snapshot_to_json(self.b))
                if target=="snapshot":decoded["claimed_authorized"]=True
                else:decoded["tensors"][0]["rights_status"]="APPROVED"
                with self.assertRaisesRegex(ValueError,"unknown .* fields"):
                    snapshot_from_json(json.dumps(decoded))
    def test_missing_schema_and_bad_hex_are_typed_errors(self):
        decoded=json.loads(snapshot_to_json(self.b));del decoded["schema_id"]
        with self.assertRaisesRegex(ValueError,"missing required fields"):
            snapshot_from_json(json.dumps(decoded))
        decoded=json.loads(snapshot_to_json(self.b));decoded["tensors"][0]["values_hex"][0]=True
        with self.assertRaisesRegex(ValueError,"tensor hex values must be strings"):
            snapshot_from_json(json.dumps(decoded))
    def test_valid_signed_zero_roundtrip(self):
        out=snapshot_from_json(snapshot_to_json(self.b))
        self.assertEqual(out.fingerprint(),self.b.fingerprint())
        self.assertEqual(out.tensor_map()["w"].values[0].hex(),"-0x0.0p+0")
