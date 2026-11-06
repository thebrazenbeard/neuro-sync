"""Actual SGD Trek factual pilot flowing through merged Neuro-Sync core APIs.

This is a six-item structured-association *mechanics* integration, not LLM
learning, held-out knowledge generalization, or secure rights authorization.
The public sealed/oracle questions are deliberately not imported or represented.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

from neurosync import (
    ModelSnapshot, ScopedGrant, Tensor, apply_delta_bundle,
    build_delta_bundle, rollback_transfer, snapshot_from_json, snapshot_to_json,
)


_TREK_DIR = Path(__file__).resolve().parents[1] / "experiments" / "trek-transfer-v0"
sys.path.insert(0, str(_TREK_DIR))

from fixtures import TREK_FACTS  # noqa: E402 - isolated experiment folder
from learner import TinyLearner  # noqa: E402
from learner import train as train_tiny  # noqa: E402
from run import evaluate_in_new_process  # noqa: E402


_WEIGHT_NAMES = ("bias", "output", "relation", "subject")
_ARCHITECTURE = "categorical-embedding-tanh-softmax-v0"


def _base_model(seed=11):
    return TinyLearner(
        [f.subject for f in TREK_FACTS],
        [f.relation for f in TREK_FACTS],
        [f.answer for f in TREK_FACTS],
        seed=seed,
    )


def _schema_from_checkpoint(checkpoint):
    """Embed FULL vocabulary/config metadata, not a misleading short label.

    The current core snapshots contain numeric tensors and a schema string;
    this string is therefore the standalone self-describing vocabulary binding.
    """
    metadata = {
        "architecture": checkpoint["architecture"],
        "subjects": checkpoint["subjects"],
        "relations": checkpoint["relations"],
        "answers": checkpoint["answers"],
        "hidden": checkpoint["hidden"],
    }
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _snapshot(model, model_id):
    checkpoint = model.checkpoint()
    return ModelSnapshot(
        model_id=model_id,
        architecture_id=_ARCHITECTURE,
        schema_id=_schema_from_checkpoint(checkpoint),
        tensors=tuple(
            (name, Tensor.from_nested(checkpoint["weights"][name]))
            for name in _WEIGHT_NAMES
        ),
    )


def _tensor_to_nested(tensor):
    if len(tensor.shape) == 1:
        return list(tensor.values)
    if len(tensor.shape) == 2:
        rows, cols = tensor.shape
        return [list(tensor.values[row * cols:(row + 1) * cols])
                for row in range(rows)]
    raise ValueError("TinyLearner only supports rank-one or rank-two weights")


def _restore_model(snapshot):
    """Reconstruct recipient solely from the complete persisted snapshot."""
    if snapshot.architecture_id != _ARCHITECTURE:
        raise ValueError("incompatible TinyLearner architecture")
    metadata = json.loads(snapshot.schema_id)
    if metadata.get("architecture") != _ARCHITECTURE:
        raise ValueError("incompatible vocabulary schema")
    weights = snapshot.tensor_map()
    if set(weights) != set(_WEIGHT_NAMES):
        raise ValueError("incomplete recipient weight inventory")
    checkpoint = dict(metadata)
    checkpoint["weights"] = {
        name: _tensor_to_nested(weights[name]) for name in _WEIGHT_NAMES
    }
    model = TinyLearner.from_checkpoint(checkpoint)
    # This checks schema/vocab ordering and every tensor shape and value.
    if _snapshot(model, snapshot.model_id).fingerprint() != snapshot.fingerprint():
        raise ValueError("reconstructed model differs from verified snapshot")
    return model


def _make_grants(donor_base, receiver_base, bundle):
    allowed = tuple(name for name, _ in bundle.deltas)
    assert allowed == _WEIGHT_NAMES, "all TinyLearner weights should be learned"
    common = dict(
        transfer_id=bundle.manifest.transfer_id,
        base_fingerprint=donor_base.fingerprint(),
        manifest_sha256=bundle.manifest.sha256(),
        allowed_tensors=allowed,
        valid_from_epoch=1000,
        expires_epoch=2000,
        rights_scope="local-test-only",  # caller-minted fixture, NOT verified consent
    )
    export = ScopedGrant(actor_id=donor_base.model_id,
                         counterpart_id=receiver_base.model_id,
                         operation="export-delta", **common)
    importer = ScopedGrant(actor_id=receiver_base.model_id,
                           counterpart_id=donor_base.model_id,
                           operation="import-delta", **common)
    return export, importer


class CoreTrekLearningIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _base_model()
        cls.donor = cls.base.clone()  # A before any training
        cls.receiver = cls.base.clone()  # B untouched through donor training
        cls.base_digest = cls.base.digest()
        assert cls.donor.digest() == cls.receiver.digest() == cls.base_digest

        cls.a_base = _snapshot(cls.donor, "trek-donor-A")
        cls.b_before = _snapshot(cls.receiver, "trek-receiver-B")
        expected = [
            [1.0 if answer == fact.answer else 0.0 for answer in cls.donor.answers]
            for fact in TREK_FACTS
        ]
        cls.training_receipt = train_tiny(cls.donor, TREK_FACTS, expected,
                                          epochs=700, lr=0.15, seed=77)
        cls.a_trained = _snapshot(cls.donor, "trek-donor-A")
        cls.bundle = build_delta_bundle(
            cls.a_base, cls.a_trained,
            receiver_id=cls.b_before.model_id,
            transfer_id="trek-integration-local-test-v0",
        )
        cls.export, cls.importer = _make_grants(cls.a_base, cls.b_before, cls.bundle)

    def _apply(self, receiver=None, export=None, importer=None):
        return apply_delta_bundle(
            self.b_before if receiver is None else receiver,
            self.bundle,
            self.export if export is None else export,
            self.importer if importer is None else importer,
            strength=1.0, now_epoch=1500,
        )

    def test_strict_initial_a_b_and_real_a_only_training(self):
        self.assertEqual(self.a_base.fingerprint(), self.b_before.fingerprint())
        self.assertEqual(self.receiver.digest(), self.base_digest)
        self.assertNotEqual(self.donor.digest(), self.base_digest)
        self.assertEqual(self.training_receipt["updates"], 4200)
        self.assertEqual(tuple(name for name, _ in self.a_base.tensors), _WEIGHT_NAMES)
        self.assertEqual(tuple(name for name, _ in self.bundle.deltas), _WEIGHT_NAMES)
        self.assertNotEqual(self.a_trained.fingerprint(), self.a_base.fingerprint())

    def test_exact_base_all_weight_transfer_to_independent_receiver(self):
        transferred, receipt = self._apply()
        self.assertEqual(receipt.before_fingerprint, self.b_before.fingerprint())
        self.assertEqual(receipt.after_fingerprint, transferred.fingerprint())
        self.assertEqual(transferred.model_id, "trek-receiver-B")
        self.assertEqual(transferred.fingerprint(), self.a_trained.fingerprint())
        self.assertEqual(transferred.tensor_map(), self.a_trained.tensor_map())
        self.assertEqual(self.b_before.fingerprint(), self.a_base.fingerprint())
        self.assertEqual(rollback_transfer(transferred, receipt).fingerprint(),
                         self.b_before.fingerprint())

    def test_receiver_only_fresh_process_before_and_after(self):
        transferred, _ = self._apply()
        # Use core's receiver-only serialization and reimport before translating
        # to the TinyLearner predictor's standalone checkpoint format.
        serialized = snapshot_to_json(transferred)
        self.assertNotIn("trek-donor-A", serialized)
        restored = snapshot_from_json(serialized)
        self.assertEqual(restored.fingerprint(), self.a_trained.fingerprint())
        before_model = _restore_model(self.b_before)
        after_model = _restore_model(restored)
        self.assertEqual(before_model.digest(), self.base_digest)
        with tempfile.TemporaryDirectory(prefix="core-trek-b-only-") as tmp:
            before = evaluate_in_new_process(before_model, TREK_FACTS, tmp)
            after = evaluate_in_new_process(after_model, TREK_FACTS, tmp)
        self.assertTrue(before["new_process"] and after["new_process"])
        self.assertEqual(before["correct"], 1)
        self.assertEqual(after["correct"], len(TREK_FACTS))
        self.assertNotEqual(before["checkpoint_sha256"], after["checkpoint_sha256"])
        self.assertEqual(after["predictions"], [f.answer for f in TREK_FACTS])

    def test_full_schema_and_vocab_mismatch_rejected(self):
        # Same numeric shapes and data, changed metadata: core must refuse.
        wrong_schema = ModelSnapshot(
            self.b_before.model_id, self.b_before.architecture_id,
            self.b_before.schema_id + "-wrong", self.b_before.tensors)
        with self.assertRaisesRegex(ValueError, "exact receiver base"):
            self._apply(receiver=wrong_schema)

        wrong_vocab_model = self.receiver.clone()
        wrong_vocab_checkpoint = wrong_vocab_model.checkpoint()
        wrong_vocab_checkpoint["answers"][-1] = "UNKNOWN_NEW_ANSWER"
        wrong_vocab_schema = _schema_from_checkpoint(wrong_vocab_checkpoint)
        wrong_vocab = ModelSnapshot(
            self.b_before.model_id, self.b_before.architecture_id,
            wrong_vocab_schema, self.b_before.tensors)
        self.assertEqual(wrong_vocab.tensor_map(), self.b_before.tensor_map())
        with self.assertRaisesRegex(ValueError, "exact receiver base"):
            self._apply(receiver=wrong_vocab)

        with self.assertRaisesRegex(ValueError, "incompatible vocabulary schema"):
            _restore_model(replace(self.b_before, schema_id='{"architecture":"wrong"}'))

    def test_bilateral_fixture_grants_are_required_not_authentication(self):
        with self.assertRaises(PermissionError):
            self._apply(export=replace(self.export, enabled=False))
        with self.assertRaises(PermissionError):
            self._apply(importer=replace(self.importer, enabled=False))
        with self.assertRaises(PermissionError):
            self._apply(export=replace(self.export, allowed_tensors=("bias",)))


if __name__ == "__main__":
    unittest.main()
