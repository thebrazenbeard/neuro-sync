"""Stdlib tests for exact-base transfer and rollback (no model dependencies)."""

from dataclasses import replace
import os
import subprocess
import sys
import unittest

from neurosync import (
    SCHEMA_VERSION, RECEIPT_VERSION, Tensor, ModelSnapshot,
    DeltaBundle, ScopedGrant, build_delta_bundle, apply_delta_bundle,
    rollback_transfer, snapshot_to_json, snapshot_from_json,
)


def make_snapshot(model_id, *, head=None, untouched=None, schema="tokens-v1"):
    return ModelSnapshot(
        model_id=model_id, architecture_id="tiny-2-layer-v1", schema_id=schema,
        tensors=(
            ("encoder.weight", Tensor.from_nested([[0.25, 0.5], [0.75, 1.0]])
             if head is None else Tensor.from_nested(head)),
            ("frozen.bias", Tensor.from_nested([0.125, -0.125])
             if untouched is None else Tensor.from_nested(untouched)),
        ),
    )


def fixture():
    base_a = make_snapshot("donor-A")
    trained_a = make_snapshot("donor-A", head=[[0.5, 0.5], [0.75, 1.25]])
    base_b = make_snapshot("receiver-B")
    bundle = build_delta_bundle(base_a, trained_a, receiver_id="receiver-B", transfer_id="t-001")
    names = tuple(name for name, _ in bundle.deltas)
    common = dict(
        transfer_id="t-001", base_fingerprint=base_a.fingerprint(),
        manifest_sha256=bundle.manifest.sha256(), allowed_tensors=names,
        valid_from_epoch=1000, expires_epoch=2000,
    )
    export = ScopedGrant(actor_id="donor-A", counterpart_id="receiver-B",
                         operation="export-delta", **common)
    importer = ScopedGrant(actor_id="receiver-B", counterpart_id="donor-A",
                           operation="import-delta", **common)
    return base_a, trained_a, base_b, bundle, export, importer


class TensorAndFingerprintTests(unittest.TestCase):
    def test_flatten_rank_three_and_reject_ragged(self):
        value = Tensor.from_nested([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(value.shape, (2, 2, 2))
        self.assertEqual(value.values, tuple(float(i) for i in range(1, 9)))
        self.assertRaises(ValueError, Tensor.from_nested, [[1, 2], [3]])
        self.assertRaises(ValueError, Tensor.from_nested, [])
        self.assertRaises(ValueError, Tensor.from_nested, [float("inf")])
        self.assertRaises(TypeError, Tensor.from_nested, [True])
        self.assertRaises(ValueError, Tensor, (2, 2), (1.0, 2.0))

    def test_snapshot_immutable_copy_and_full_fingerprint(self):
        mutable = [[0.25, 0.5], [0.75, 1.0]]
        tensor = Tensor.from_nested(mutable)
        mutable[0][0] = 999.0
        self.assertEqual(tensor.values[0], 0.25)
        a = make_snapshot("A")
        b = make_snapshot("B")
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), make_snapshot("B", untouched=[0.0, 0.0]).fingerprint())
        self.assertNotEqual(a.fingerprint(), make_snapshot("B", schema="other-tokens").fingerprint())
        self.assertRaises(ValueError, ModelSnapshot, "B", "arch", "schema", (("x", tensor), ("x", tensor)))


class TransferTests(unittest.TestCase):
    def setUp(self):
        (self.a_base, self.a_trained, self.b, self.bundle,
         self.export, self.importer) = fixture()

    def apply(self, **kwargs):
        return apply_delta_bundle(
            self.b, self.bundle, self.export, self.importer,
            now_epoch=1500, **kwargs,
        )

    def test_bilateral_exact_base_and_donor_disconnect(self):
        self.assertEqual(self.a_base.fingerprint(), self.b.fingerprint())
        self.assertEqual(tuple(name for name, _ in self.bundle.deltas), ("encoder.weight",))
        result, receipt = self.apply(strength=1.0)
        self.assertEqual(result.model_id, "receiver-B")
        self.assertEqual(result.tensor_map()["encoder.weight"].values,
                         self.a_trained.tensor_map()["encoder.weight"].values)
        self.assertEqual(self.b.tensor_map()["encoder.weight"].values,
                         self.a_base.tensor_map()["encoder.weight"].values)
        self.assertEqual(receipt.version, RECEIPT_VERSION)
        self.assertEqual(receipt.before_fingerprint, self.b.fingerprint())
        self.assertEqual(receipt.after_fingerprint, result.fingerprint())
        del self.a_base, self.a_trained, self.bundle  # no donor input to receiver readback
        self.assertEqual(result.tensor_map()["encoder.weight"].values,
                         (0.5, 0.5, 0.75, 1.25))
        self.assertEqual(rollback_transfer(result, receipt).fingerprint(), self.b.fingerprint())

    def test_strength_and_zero_transfer(self):
        half, receipt = self.apply(strength=0.5)
        self.assertEqual(half.tensor_map()["encoder.weight"].values,
                         (0.375, 0.5, 0.75, 1.125))
        self.assertEqual(receipt.strength, 0.5)
        zero, _ = self.apply(strength=0.0)
        self.assertEqual(zero.fingerprint(), self.b.fingerprint())
        for invalid in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(value=invalid), self.assertRaises((TypeError, ValueError)):
                self.apply(strength=invalid)

    def test_both_grants_needed_and_exactly_scoped(self):
        variants = (
            (replace(self.export, enabled=False), self.importer),
            (replace(self.export, actor_id="other"), self.importer),
            (replace(self.export, counterpart_id="other"), self.importer),
            (replace(self.export, transfer_id="t-002"), self.importer),
            (replace(self.export, manifest_sha256="wrong"), self.importer),
            (replace(self.export, allowed_tensors=("frozen.bias",)), self.importer),
            (replace(self.export, rights_scope="all-models"), self.importer),
            (self.export, replace(self.importer, enabled=False)),
            (self.export, replace(self.importer, actor_id="other")),
            (self.export, replace(self.importer, operation="export-delta")),
            (self.export, replace(self.importer, base_fingerprint="wrong")),
        )
        for exp, imp in variants:
            with self.subTest(exp=exp, imp=imp), self.assertRaises(PermissionError):
                apply_delta_bundle(self.b, self.bundle, exp, imp, now_epoch=1500)
        for when in (999, 2000, float("nan")):
            with self.subTest(when=when), self.assertRaises((PermissionError, ValueError)):
                apply_delta_bundle(self.b, self.bundle, self.export, self.importer, now_epoch=when)

    def test_reject_base_schema_target_shape_and_manifest_tamper(self):
        wrong_untouched = make_snapshot("receiver-B", untouched=[0.0, -0.125])
        wrong_schema = make_snapshot("receiver-B", schema="other-tokens")
        wrong_id = make_snapshot("another-B")
        for receiver in (wrong_untouched, wrong_schema, wrong_id):
            with self.subTest(receiver=receiver.model_id), self.assertRaises(ValueError):
                apply_delta_bundle(receiver, self.bundle, self.export, self.importer, now_epoch=1500)
        wrong_shape = DeltaBundle(
            self.bundle.manifest,
            (("encoder.weight", Tensor.from_nested([0.1, 0.2, 0.3, 0.4])),),
        )
        with self.assertRaises(ValueError):
            apply_delta_bundle(self.b, wrong_shape, self.export, self.importer, now_epoch=1500)
        changed = DeltaBundle(
            self.bundle.manifest,
            (("encoder.weight", Tensor.from_nested([[0.2, 0.0], [0.0, 0.25]])),),
        )
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            apply_delta_bundle(self.b, changed, self.export, self.importer, now_epoch=1500)

    def test_rollback_rejects_receiver_drift_and_wrong_receipt(self):
        result, receipt = self.apply()
        drift = make_snapshot("receiver-B", head=[[0.5, 0.5], [0.75, 99.0]])
        with self.assertRaisesRegex(ValueError, "changed since transfer"):
            rollback_transfer(drift, receipt)
        with self.assertRaisesRegex(ValueError, "version"):
            rollback_transfer(result, replace(receipt, version=99))
        with self.assertRaisesRegex(ValueError, "before fingerprint"):
            rollback_transfer(result, replace(receipt, before_fingerprint="wrong"))
        with self.assertRaisesRegex(ValueError, "receiver mismatch"):
            rollback_transfer(make_snapshot("wrong"), receipt)

    def test_receiver_only_new_process_roundtrip(self):
        """Proves only state serialization, NOT fact transfer or model quality."""
        receiver, _ = self.apply()
        encoded = snapshot_to_json(receiver)
        self.assertNotIn("donor-A", encoded)
        self.assertEqual(snapshot_from_json(encoded).fingerprint(), receiver.fingerprint())
        script = (
            "import sys; from neurosync import snapshot_from_json; "
            "s=snapshot_from_json(sys.stdin.read()); "
            "print(s.model_id); print(s.fingerprint())"
        )
        process = subprocess.run(
            [sys.executable, "-c", script], input=encoded, text=True,
            capture_output=True, check=True, env=os.environ.copy(),
        )
        self.assertEqual(process.stdout.splitlines(),
                         ["receiver-B", receiver.fingerprint()])
        with self.assertRaises(ValueError):
            snapshot_from_json(encoded.replace('"version":1', '"version":99'))

    def test_no_changes_and_mismatched_donor_structure(self):
        with self.assertRaisesRegex(ValueError, "no weight changes"):
            build_delta_bundle(self.a_base, self.a_base, receiver_id="receiver-B", transfer_id="t-1")
        with self.assertRaisesRegex(ValueError, "identities"):
            build_delta_bundle(self.a_base, make_snapshot("other"), receiver_id="B", transfer_id="t-1")
        with self.assertRaisesRegex(ValueError, "names/shapes"):
            donor_change = make_snapshot("donor-A", head=[[1.0, 2.0, 3.0]])
            build_delta_bundle(self.a_base, donor_change, receiver_id="B", transfer_id="t-1")


if __name__ == "__main__":
    unittest.main()
