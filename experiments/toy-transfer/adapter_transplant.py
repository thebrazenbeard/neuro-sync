"""Pure-stdlib *mechanics-only* test of an exact-base weight-delta transplant.

This is not a secure authorization system, a model quality test, or a way to
transplant deltas between incompatible bases. A real deployment would need an
authenticated rights grant, trustworthy provenance and atomic durable storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence


Matrix = tuple[tuple[float, ...], ...]
SCHEMA = "toy-exact-base-delta-v1"


def normalize_matrix(rows: Sequence[Sequence[float]]) -> Matrix:
    if not rows or not rows[0]:
        raise ValueError("matrix cannot be empty")
    width = len(rows[0])
    result: list[tuple[float, ...]] = []
    for row in rows:
        if len(row) != width:
            raise ValueError("ragged matrix")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in row):
            raise TypeError("matrix entries must be numeric, not booleans")
        values = tuple(float(item) for item in row)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("matrix entries must be finite")
        result.append(values)
    return tuple(result)


def shape(matrix: Matrix) -> tuple[int, int]:
    return len(matrix), len(matrix[0])


def matrix_hash(matrix: Matrix) -> str:
    normalized = normalize_matrix(matrix)
    encoded = json.dumps(normalized, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TransplantManifest:
    schema: str
    target_id: str
    base_sha256: str
    delta_sha256: str
    matrix_shape: tuple[int, int]
    intended_scope: str = "local-test-only"


@dataclass(frozen=True)
class PermissionGrant:
    target_id: str
    allowed: bool
    operation: str = "apply-exact-base-delta"


@dataclass(frozen=True)
class AppliedTransplant:
    before: Matrix
    after: Matrix
    before_sha256: str
    after_sha256: str


def apply_delta(
    receiver_base: Sequence[Sequence[float]],
    donor_delta: Sequence[Sequence[float]],
    manifest: TransplantManifest,
    grant: PermissionGrant,
) -> AppliedTransplant:
    if manifest.schema != SCHEMA or manifest.intended_scope != "local-test-only":
        raise PermissionError("unsupported manifest schema or scope")
    if not grant.allowed or grant.operation != "apply-exact-base-delta":
        raise PermissionError("operation is not explicitly granted")
    if not manifest.target_id or grant.target_id != manifest.target_id:
        raise PermissionError("grant target does not match manifest")

    base = normalize_matrix(receiver_base)
    delta = normalize_matrix(donor_delta)
    if shape(base) != manifest.matrix_shape or shape(delta) != manifest.matrix_shape:
        raise ValueError("shape mismatch; no implicit cross-model projection")
    base_digest = matrix_hash(base)
    if base_digest != manifest.base_sha256:
        raise ValueError("receiver base hash mismatch")
    if matrix_hash(delta) != manifest.delta_sha256:
        raise ValueError("donor delta hash mismatch")

    after = normalize_matrix(
        [[original + change for original, change in zip(base_row, delta_row)]
         for base_row, delta_row in zip(base, delta)]
    )
    return AppliedTransplant(base, after, base_digest, matrix_hash(after))


def rollback(current: Sequence[Sequence[float]], receipt: AppliedTransplant) -> Matrix:
    """Reject rollback if the current receiver has drifted since transplant."""
    if matrix_hash(normalize_matrix(current)) != receipt.after_sha256:
        raise ValueError("receiver changed after transplant; unsafe to rollback")
    return receipt.before


def matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, ...]:
    matrix_copy = normalize_matrix(matrix)
    if len(vector) != shape(matrix_copy)[1]:
        raise ValueError("input feature width mismatch")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
        raise ValueError("input feature values must be finite")
    return tuple(sum(weight * value for weight, value in zip(row, vector)) for row in matrix_copy)
