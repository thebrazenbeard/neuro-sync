"""Runnable with Python stdlib, including when PyTorch is unavailable."""

import unittest

from adapter_transplant import (
    SCHEMA,
    PermissionGrant,
    TransplantManifest,
    apply_delta,
    matrix_hash,
    matrix_vector,
    normalize_matrix,
    rollback,
)


class ExactBaseTransplantTests(unittest.TestCase):
    def setUp(self):
        self.base = [[0.2, 0.4], [0.6, 0.8]]
        self.donor_delta = [[0.1, -0.1], [0.0, 0.05]]
        self.manifest = TransplantManifest(
            schema=SCHEMA,
            target_id="receiver-B",
            base_sha256=matrix_hash(normalize_matrix(self.base)),
            delta_sha256=matrix_hash(normalize_matrix(self.donor_delta)),
            matrix_shape=(2, 2),
        )
        self.grant = PermissionGrant(target_id="receiver-B", allowed=True)

    def test_exact_base_apply_donor_disconnect_and_rollback(self):
        donor_before = [row[:] for row in self.donor_delta]
        applied = apply_delta(self.base, self.donor_delta, self.manifest, self.grant)
        for row, expected in zip(applied.after, ((0.3, 0.3), (0.6, 0.85))):
            for actual, nominal in zip(row, expected):
                self.assertAlmostEqual(actual, nominal)
        sample = (1.0, -2.0)
        output_after = matrix_vector(applied.after, sample)

        # Mutating/removing donor input after transfer has no influence on B.
        self.donor_delta[0][0] = 900.0
        del self.donor_delta
        self.assertEqual(output_after, matrix_vector(applied.after, sample))
        self.assertEqual(self.base, [[0.2, 0.4], [0.6, 0.8]])
        self.assertNotEqual(output_after, matrix_vector(self.base, sample))
        self.assertEqual(rollback(applied.after, applied), normalize_matrix(self.base))
        self.assertEqual(donor_before, [[0.1, -0.1], [0.0, 0.05]])

    def test_reject_unapproved_or_wrong_target_grant(self):
        for grant in (
            PermissionGrant("receiver-B", allowed=False),
            PermissionGrant("receiver-C", allowed=True),
            PermissionGrant("receiver-B", allowed=True, operation="delete"),
        ):
            with self.subTest(grant=grant), self.assertRaises(PermissionError):
                apply_delta(self.base, self.donor_delta, self.manifest, grant)

    def test_reject_wrong_receiver_base_hash(self):
        modified = [[0.2, 0.5], [0.6, 0.8]]
        with self.assertRaisesRegex(ValueError, "base hash"):
            apply_delta(modified, self.donor_delta, self.manifest, self.grant)

    def test_reject_shape_and_delta_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            apply_delta(self.base, [[0.1]], self.manifest, self.grant)
        with self.assertRaisesRegex(ValueError, "delta hash"):
            apply_delta(self.base, [[0.2, -0.1], [0.0, 0.05]], self.manifest, self.grant)

    def test_reject_tampered_rollback(self):
        receipt = apply_delta(self.base, self.donor_delta, self.manifest, self.grant)
        with self.assertRaisesRegex(ValueError, "changed"):
            rollback([[9.0, 8.0], [7.0, 6.0]], receipt)

    def test_reject_ragged_nonfinite(self):
        with self.assertRaises(ValueError):
            normalize_matrix([[1.0, 2.0], [3.0]])
        with self.assertRaises(ValueError):
            normalize_matrix([[float("nan")]])


if __name__ == "__main__":
    unittest.main()
