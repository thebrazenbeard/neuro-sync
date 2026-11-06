"""Dependency-free exact-base tensor-delta transfer *mechanics*.

This is a reusable pure-Python reference implementation, not a secure rights
system or a live cross-model gradient bridge. Grants are explicit **fixtures**:
the caller can manufacture them, and no identity/signature is authenticated.
Never treat these objects as production authorization without a separate
trusted issuer, identity verification, revocation and durable state handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Sequence


SCHEMA_VERSION = 1
RECEIPT_VERSION = 1
_FIXTURE_RIGHTS = "local-test-only"


def _hash_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class Tensor:
    """Immutable float64-style flattened tensor with an exact rank/shape."""

    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        dims = tuple(self.shape)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in dims):
            raise ValueError("tensor dimensions must be positive integers")
        values = tuple(_finite_number(item, field="tensor value") for item in self.values)
        if len(values) != math.prod(dims):
            raise ValueError("tensor values do not match declared shape")
        object.__setattr__(self, "shape", dims)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_nested(cls, value: object) -> "Tensor":
        def flatten(part: object) -> tuple[tuple[int, ...], list[float]]:
            if isinstance(part, (list, tuple)):
                if not part:
                    raise ValueError("empty tensor dimension")
                first_shape, first_values = flatten(part[0])
                result = list(first_values)
                for item in part[1:]:
                    item_shape, item_values = flatten(item)
                    if item_shape != first_shape:
                        raise ValueError("ragged tensor")
                    result.extend(item_values)
                return (len(part),) + first_shape, result
            return (), [_finite_number(part, field="tensor value")]

        dims, values = flatten(value)
        return cls(dims, tuple(values))

    def sha256(self) -> str:
        return _hash_json({"shape": self.shape, "values_hex": [x.hex() for x in self.values]})


@dataclass(frozen=True)
class ModelSnapshot:
    """Model ID identifies a branch; fingerprint identifies the complete base.

    Two separately named copies of an identical untrained base deliberately
    have the SAME fingerprint. The fingerprint pins architecture, schema,
    tensor names, shapes and every value, but no tokenizer or external files.
    A caller must include such information in schema_id before relying on it.
    """

    model_id: str
    architecture_id: str
    schema_id: str
    tensors: tuple[tuple[str, Tensor], ...]

    def __post_init__(self) -> None:
        if any(not isinstance(x, str) or not x.strip() for x in
               (self.model_id, self.architecture_id, self.schema_id)):
            raise ValueError("model, architecture and schema IDs are required")
        records = tuple(self.tensors)
        if not records or any(not isinstance(p, tuple) or len(p) != 2 for p in records):
            raise ValueError("at least one named tensor is required")
        if any(not isinstance(name, str) or not name.strip() or not isinstance(t, Tensor)
               for name, t in records):
            raise ValueError("invalid named tensor")
        names = [name for name, _ in records]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tensor name")
        object.__setattr__(self, "tensors", tuple(sorted(records)))

    def tensor_map(self) -> dict[str, Tensor]:
        return dict(self.tensors)

    def fingerprint(self) -> str:
        return _hash_json({
            "schema_version": SCHEMA_VERSION,
            "architecture_id": self.architecture_id,
            "schema_id": self.schema_id,
            "tensors": [(name, tensor.sha256()) for name, tensor in self.tensors],
        })


@dataclass(frozen=True)
class DeltaManifest:
    version: int
    transfer_id: str
    donor_id: str
    receiver_id: str
    base_fingerprint: str
    architecture_id: str
    schema_id: str
    delta_hashes: tuple[tuple[str, str], ...]
    target_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError("unsupported delta manifest version")
        if any(not isinstance(item, str) or not item.strip() for item in (
            self.transfer_id, self.donor_id, self.receiver_id,
            self.base_fingerprint, self.architecture_id, self.schema_id,
        )):
            raise ValueError("manifest identities and fingerprints are required")
        hashes = tuple(self.delta_hashes)
        targets = tuple(self.target_hashes)
        if not hashes or any(not isinstance(x, tuple) or len(x) != 2 for x in hashes):
            raise ValueError("manifest needs named delta hashes")
        if len({name for name, _ in hashes}) != len(hashes):
            raise ValueError("duplicate manifest delta name")
        if sorted(name for name, _ in hashes) != sorted(name for name, _ in targets):
            raise ValueError("target tensor hashes must match delta names")
        object.__setattr__(self, "delta_hashes", tuple(sorted(hashes)))
        object.__setattr__(self, "target_hashes", tuple(sorted(targets)))

    def sha256(self) -> str:
        return _hash_json({
            "version": self.version, "transfer_id": self.transfer_id,
            "donor_id": self.donor_id, "receiver_id": self.receiver_id,
            "base_fingerprint": self.base_fingerprint,
            "architecture_id": self.architecture_id,
            "schema_id": self.schema_id,
            "delta_hashes": self.delta_hashes,
            "target_hashes": self.target_hashes,
        })


@dataclass(frozen=True)
class DeltaBundle:
    manifest: DeltaManifest
    deltas: tuple[tuple[str, Tensor], ...]
    targets: tuple[tuple[str, Tensor], ...]

    def __post_init__(self) -> None:
        records = tuple(self.deltas)
        if any(not isinstance(x, tuple) or len(x) != 2 for x in records):
            raise ValueError("invalid delta records")
        if len({name for name, _ in records}) != len(records):
            raise ValueError("duplicate delta name")
        if any(not isinstance(name, str) or not isinstance(tensor, Tensor)
               for name, tensor in records):
            raise ValueError("invalid delta tensor")
        targets = tuple(self.targets)
        if len({name for name, _ in targets}) != len(targets):
            raise ValueError("duplicate absolute target tensor name")
        if sorted(name for name, _ in targets) != sorted(name for name, _ in records):
            raise ValueError("absolute targets must match delta names")
        if any(not isinstance(name, str) or not isinstance(tensor, Tensor)
               for name, tensor in targets):
            raise ValueError("invalid absolute target tensor")
        object.__setattr__(self, "deltas", tuple(sorted(records)))
        object.__setattr__(self, "targets", tuple(sorted(targets)))


@dataclass(frozen=True)
class ScopedGrant:
    """Non-authenticated fixture; never a substitute for external consent."""

    actor_id: str
    counterpart_id: str
    transfer_id: str
    operation: str  # exactly export-delta or import-delta
    base_fingerprint: str
    manifest_sha256: str
    allowed_tensors: tuple[str, ...]
    valid_from_epoch: float
    expires_epoch: float
    rights_scope: str = _FIXTURE_RIGHTS
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_tensors", tuple(sorted(set(self.allowed_tensors))))
        if self.operation not in ("export-delta", "import-delta"):
            raise ValueError("unknown operation")
        if not self.allowed_tensors or not all(isinstance(x, str) and x for x in self.allowed_tensors):
            raise ValueError("grant must name allowed tensors")
        start = _finite_number(self.valid_from_epoch, field="valid_from_epoch")
        end = _finite_number(self.expires_epoch, field="expires_epoch")
        if end <= start:
            raise ValueError("grant expiry must follow start")
        object.__setattr__(self, "valid_from_epoch", start)
        object.__setattr__(self, "expires_epoch", end)


@dataclass(frozen=True)
class TransferReceipt:
    version: int
    transfer_id: str
    receiver_id: str
    manifest_sha256: str
    strength: float
    before_fingerprint: str
    after_fingerprint: str
    before: ModelSnapshot
    integrity_sha256: str


def _receipt_digest(
    *, version: int, transfer_id: str, receiver_id: str,
    manifest_sha256: str, strength: float,
    before_fingerprint: str, after_fingerprint: str,
) -> str:
    """Unkeyed corruption check only; an attacker can forge a new digest."""
    return _hash_json({
        "version": version, "transfer_id": transfer_id,
        "receiver_id": receiver_id, "manifest_sha256": manifest_sha256,
        "strength": strength, "before_fingerprint": before_fingerprint,
        "after_fingerprint": after_fingerprint,
    })


def build_delta_bundle(
    donor_base: ModelSnapshot,
    donor_trained: ModelSnapshot,
    *,
    receiver_id: str,
    transfer_id: str,
) -> DeltaBundle:
    """Extract changed tensors from one donor branch; no transfer occurs."""
    if donor_base.model_id != donor_trained.model_id:
        raise ValueError("donor base and trained model identities differ")
    if (donor_base.architecture_id, donor_base.schema_id) != (
        donor_trained.architecture_id, donor_trained.schema_id
    ):
        raise ValueError("donor architecture/schema changed")
    before, after = donor_base.tensor_map(), donor_trained.tensor_map()
    if before.keys() != after.keys() or any(before[name].shape != after[name].shape for name in before):
        raise ValueError("donor tensor names/shapes changed")
    changes: list[tuple[str, Tensor]] = []
    targets: list[tuple[str, Tensor]] = []
    for name in before:
        old, new = before[name], after[name]
        # Numeric equality hides signed-zero differences, but our canonical
        # tensor fingerprint (and λ=1 exact-target contract) does not.
        if old.sha256() != new.sha256():
            changes.append((name, Tensor(old.shape, tuple(b - a for a, b in zip(old.values, new.values)))))
            targets.append((name, Tensor(new.shape, tuple(new.values))))
    if not changes:
        raise ValueError("donor has no weight changes")
    manifest = DeltaManifest(
        version=SCHEMA_VERSION, transfer_id=transfer_id,
        donor_id=donor_base.model_id, receiver_id=receiver_id,
        base_fingerprint=donor_base.fingerprint(),
        architecture_id=donor_base.architecture_id,
        schema_id=donor_base.schema_id,
        delta_hashes=tuple((name, tensor.sha256()) for name, tensor in changes),
        target_hashes=tuple((name, tensor.sha256()) for name, tensor in targets),
    )
    return DeltaBundle(manifest, tuple(changes), tuple(targets))


def _check_grant(
    grant: ScopedGrant, *, actor: str, counterpart: str,
    operation: str, bundle: DeltaBundle, now_epoch: float,
) -> None:
    m = bundle.manifest
    if (not grant.enabled or grant.rights_scope != _FIXTURE_RIGHTS or
        grant.actor_id != actor or grant.counterpart_id != counterpart or
        grant.transfer_id != m.transfer_id or grant.operation != operation or
        grant.base_fingerprint != m.base_fingerprint or
        grant.manifest_sha256 != m.sha256() or
        not set(name for name, _ in bundle.deltas).issubset(grant.allowed_tensors) or
        not grant.valid_from_epoch <= now_epoch < grant.expires_epoch):
        raise PermissionError(f"invalid {operation} fixture grant")


def apply_delta_bundle(
    receiver: ModelSnapshot,
    bundle: DeltaBundle,
    export_grant: ScopedGrant,
    import_grant: ScopedGrant,
    *,
    strength: float = 1.0,
    now_epoch: float | None = None,
) -> tuple[ModelSnapshot, TransferReceipt]:
    """Apply a donor delta to an exact-base receiver COPY, with two grants.

    strength (λ) is in [0,1], but this bounds only the mixing coefficient;
    it does NOT bound the transferred norm, performance or training stability.
    """
    lam = _finite_number(strength, field="strength")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("strength must be within [0,1]")
    now = time.time() if now_epoch is None else _finite_number(now_epoch, field="now_epoch")
    m = bundle.manifest
    if receiver.model_id != m.receiver_id:
        raise ValueError("receiver identity mismatch")
    if (receiver.architecture_id, receiver.schema_id, receiver.fingerprint()) != (
        m.architecture_id, m.schema_id, m.base_fingerprint
    ):
        raise ValueError("exact receiver base fingerprint/schema mismatch")
    receiver_map = receiver.tensor_map()
    hashes = tuple(sorted((name, tensor.sha256()) for name, tensor in bundle.deltas))
    if hashes != m.delta_hashes:
        raise ValueError("delta bundle manifest hash mismatch")
    target_hashes = tuple(sorted((name, tensor.sha256()) for name, tensor in bundle.targets))
    if target_hashes != m.target_hashes:
        raise ValueError("absolute targets manifest hash mismatch")
    if any(name not in receiver_map or receiver_map[name].shape != tensor.shape
           for name, tensor in bundle.deltas):
        raise ValueError("delta tensor name/shape mismatch")
    if any(name not in receiver_map or receiver_map[name].shape != tensor.shape
           for name, tensor in bundle.targets):
        raise ValueError("absolute target tensor name/shape mismatch")
    # Individually correct hashes do not imply that a delta and an absolute
    # target describe the same update. Recompute their relation from the
    # receiver's verified exact base, including finite subtraction checks.
    # Hex hashes also distinguish signed zero, which float == does not.
    delta_map = dict(bundle.deltas)
    for name, target in bundle.targets:
        original = receiver_map[name]
        computed = Tensor(original.shape, tuple(
            trained - base for trained, base in zip(target.values, original.values)
        ))
        if computed.sha256() != delta_map[name].sha256():
            raise ValueError("inconsistent delta versus absolute target/base")
    _check_grant(export_grant, actor=m.donor_id, counterpart=m.receiver_id,
                 operation="export-delta", bundle=bundle, now_epoch=now)
    _check_grant(import_grant, actor=m.receiver_id, counterpart=m.donor_id,
                 operation="import-delta", bundle=bundle, now_epoch=now)

    changed = delta_map
    targets = dict(bundle.targets)
    copied: list[tuple[str, Tensor]] = []
    for name, original in receiver.tensors:
        if name in changed:
            # Floating subtraction can lose the exact trained value (e.g.
            # 1e20 -> 1.0). Full-strength transplant therefore copies the
            # separately hashed absolute target, not base + rounded delta.
            if lam == 1.0:
                values = targets[name].values
            else:
                values = tuple(base + lam * shift for base, shift in
                               zip(original.values, changed[name].values))
            copied.append((name, Tensor(original.shape, values)))
        else:
            copied.append((name, Tensor(original.shape, tuple(original.values))))
    result = ModelSnapshot(receiver.model_id, receiver.architecture_id,
                           receiver.schema_id, tuple(copied))
    receipt_fields = dict(
        version=RECEIPT_VERSION, transfer_id=m.transfer_id,
        receiver_id=receiver.model_id, manifest_sha256=m.sha256(),
        strength=lam, before_fingerprint=receiver.fingerprint(),
        after_fingerprint=result.fingerprint(),
    )
    receipt = TransferReceipt(
        **receipt_fields, before=receiver,
        integrity_sha256=_receipt_digest(**receipt_fields),
    )
    return result, receipt


def rollback_transfer(current: ModelSnapshot, receipt: TransferReceipt) -> ModelSnapshot:
    """Restore only the exact receiver/result pair; reject post-transfer drift.

    Receipt metadata has an unkeyed digest to reject accidental/casual edits,
    but it is NOT signed. A malicious party able to forge a receipt and digest
    bypasses this check; real systems need authenticated logs.
    """
    if receipt.version != RECEIPT_VERSION:
        raise ValueError("unsupported receipt version")
    fields = dict(
        version=receipt.version, transfer_id=receipt.transfer_id,
        receiver_id=receipt.receiver_id,
        manifest_sha256=receipt.manifest_sha256,
        strength=receipt.strength,
        before_fingerprint=receipt.before_fingerprint,
        after_fingerprint=receipt.after_fingerprint,
    )
    if _receipt_digest(**fields) != receipt.integrity_sha256:
        raise ValueError("receipt integrity digest mismatch")
    if current.model_id != receipt.receiver_id or receipt.before.model_id != receipt.receiver_id:
        raise ValueError("receipt receiver mismatch")
    if receipt.before.fingerprint() != receipt.before_fingerprint:
        raise ValueError("receipt before fingerprint mismatch")
    if current.fingerprint() != receipt.after_fingerprint:
        raise ValueError("receiver changed since transfer; rollback refused")
    return ModelSnapshot(receipt.before.model_id, receipt.before.architecture_id,
                         receipt.before.schema_id, tuple(receipt.before.tensors))


def snapshot_to_json(snapshot: ModelSnapshot) -> str:
    """Portable receiver-only text state, with hex float64 values.

    No donor source is included. Transport/storage access and encryption are
    caller responsibilities; this routine never writes to disk or network.
    """
    return json.dumps({
        "version": SCHEMA_VERSION,
        "model_id": snapshot.model_id,
        "architecture_id": snapshot.architecture_id,
        "schema_id": snapshot.schema_id,
        "tensors": [
            {"name": name, "shape": tensor.shape,
             "values_hex": [number.hex() for number in tensor.values]}
            for name, tensor in snapshot.tensors
        ],
    }, sort_keys=True, separators=(",", ":"))


def snapshot_from_json(encoded: str) -> ModelSnapshot:
    """Restore only the named receiver snapshot, not an executing model."""
    def unique_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> object:
        # Python accepts NaN/Infinity by default; these are not JSON numbers
        # under RFC 8259 and create cross-decoder disagreement.
        raise ValueError("nonstandard JSON numeric constant")

    decoded = json.loads(encoded, object_pairs_hook=unique_object_keys,
                         parse_constant=reject_nonstandard_constant)
    # `True == 1` in Python, but a JSON boolean is not an integer protocol
    # version. Reject ambiguous cross-language snapshots before constructing.
    if (not isinstance(decoded, dict) or
            type(decoded.get("version")) is not int or
            decoded["version"] != SCHEMA_VERSION):
        raise ValueError("unsupported snapshot version")
    if not isinstance(decoded.get("tensors"), list):
        raise ValueError("malformed snapshot tensor list")
    records = []
    for item in decoded["tensors"]:
        if not isinstance(item, dict) or not isinstance(item.get("values_hex"), list):
            raise ValueError("malformed snapshot tensor")
        tensor = Tensor(tuple(item["shape"]), tuple(float.fromhex(x) for x in item["values_hex"]))
        records.append((item["name"], tensor))
    return ModelSnapshot(
        decoded["model_id"], decoded["architecture_id"],
        decoded["schema_id"], tuple(records),
    )
