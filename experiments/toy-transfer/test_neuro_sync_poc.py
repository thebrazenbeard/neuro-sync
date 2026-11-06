"""CPU PyTorch tests; skip cleanly if torch is not installed."""

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch.nn import functional as F
    from neuro_sync_poc import (
        SOURCE_DIM, TARGET_DIM, NeuroSyncToy, load_checkpoint,
        make_synthetic_batch, save_checkpoint, train_steps,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch unavailable")
class DifferentiableTransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.model = NeuroSyncToy()
        self.x, self.labels = make_synthetic_batch(batch_size=16, seed=202)

    def test_heterogeneous_shapes_and_bounds(self):
        self.assertNotEqual(SOURCE_DIM, TARGET_DIM)
        state = self.model(self.x, alpha=0.75)
        self.assertEqual(tuple(state.projected_source.shape), (16, TARGET_DIM))
        self.assertEqual(tuple(state.logits.shape), (16, 2))
        self.assertGreaterEqual(float(state.effective_gate), 0.0)
        self.assertLessEqual(float(state.effective_gate), 0.75)
        with self.assertRaises(ValueError):
            self.model(self.x[:, :3])
        with self.assertRaises(ValueError):
            self.model(self.x, alpha=-0.1)
        with self.assertRaises(ValueError):
            self.model(self.x, alpha=1.1)

    def test_finite_backward_and_frozen_base(self):
        loss, parts = self.model.loss(self.x, self.labels, alpha=0.75)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(parts["task"]).item())
        self.assertTrue(torch.isfinite(parts["alignment"]).item())
        loss.backward()
        for parameter in self.model.trainable_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        for backbone in (self.model.source, self.model.target):
            for parameter in backbone.parameters():
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)

    def test_zero_transfer_is_exact_local_baseline(self):
        state = self.model(self.x, alpha=0.0)
        expected = self.model.baseline_logits(self.x)
        torch.testing.assert_close(state.logits, expected, rtol=0, atol=0)
        self.assertEqual(float(state.effective_gate), 0.0)
        loss, parts = self.model.loss(self.x, self.labels, alpha=0.0)
        torch.testing.assert_close(loss, F.cross_entropy(expected, self.labels), rtol=0, atol=0)
        self.assertEqual(float(parts["alignment"]), 0.0)
        loss.backward()
        # Zero coupling may create graph edges with exactly zero gradients.
        self.assertTrue(torch.count_nonzero(self.model.projector.weight.grad) == 0)
        self.assertEqual(float(self.model.gate_logit.grad), 0.0)

    def test_shuffled_transfer_breaks_pairing_not_target(self):
        paired = self.model(self.x, alpha=1.0, shuffle_source=False)
        shuffled = self.model(self.x, alpha=1.0, shuffle_source=True)
        torch.testing.assert_close(paired.target_hidden, shuffled.target_hidden, rtol=0, atol=0)
        torch.testing.assert_close(
            paired.projected_source.roll(shifts=1, dims=0),
            shuffled.projected_source, rtol=0, atol=0,
        )
        self.assertFalse(torch.equal(paired.projected_source, shuffled.projected_source))
        with self.assertRaises(ValueError):
            self.model(self.x[:1], shuffle_source=True)

    def test_training_preserves_frozen_bases(self):
        source_before = {k: v.clone() for k, v in self.model.source.state_dict().items()}
        target_before = {k: v.clone() for k, v in self.model.target.state_dict().items()}
        adapter_before = self.model.adapter.up.weight.detach().clone()
        losses = train_steps(self.model, self.x, self.labels, alpha=1.0, steps=3)
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(value == value for value in losses))
        for name, original in source_before.items():
            torch.testing.assert_close(self.model.source.state_dict()[name], original, rtol=0, atol=0)
        for name, original in target_before.items():
            torch.testing.assert_close(self.model.target.state_dict()[name], original, rtol=0, atol=0)
        self.assertFalse(torch.equal(adapter_before, self.model.adapter.up.weight))

    def test_save_restore_same_architecture(self):
        train_steps(self.model, self.x, self.labels, alpha=1.0, steps=2)
        self.model.eval()
        before = self.model(self.x, alpha=0.8).logits.detach()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "toy.pt"
            save_checkpoint(self.model, path)
            restored = load_checkpoint(path)
            restored.eval()
            after = restored(self.x, alpha=0.8).logits.detach()
        torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
