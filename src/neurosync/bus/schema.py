"""Versioned, source-only StateBus envelopes. No neural weights or execution.

Opaque capability/provenance references and evidence-class labels are DATA,
not authenticated authority. Only a separately configured trusted verifier
can promote a proposal. Canonical JSON here is a documented Python-specific
sorted-key encoding, not a claim of cross-language RFC 8785 compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
MAX_PAYLOAD_BYTES = 8192
MAX_ENVELOPE_BYTES = 16384
MAX_REFS = 16
MAX_DEPTH = 8
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,95}$")
_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DENIED_KEYS = ("secret", "credential", "password", "token", "private", "weight", "optimizer")


class SchemaError(ValueError):
    pass


class StateOp(str, Enum):
    SET = "set"
    DELETE = "delete"


class EvidenceClass(str, Enum):
    DOC = "DOC"
    SOURCE_INSPECTION = "SOURCE_INSPECTION"
    TEST = "TEST"
    BUILD = "BUILD"
    INSTALL = "INSTALL"
    RUNTIME_PROBE = "RUNTIME_PROBE"
    FORMAL_VERIFY = "FORMAL_VERIFY"


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise SchemaError(f"non-finite JSON number: {value}")


def _parse_json(raw: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_pairs_no_duplicates,
                          parse_constant=_reject_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaError("invalid JSON") from exc


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SchemaError("value is not canonical JSON data") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise SchemaError(f"invalid {field}")
    return value


def _epoch(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{field} must be a nonnegative integer epoch second")
    return value


def _json_value(value: Any, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise SchemaError("JSON value exceeds nesting depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError("non-finite payload number")
        return
    if isinstance(value, list):
        if len(value) > 512:
            raise SchemaError("payload array too long")
        for item in value:
            _json_value(item, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 512:
            raise SchemaError("payload object too large")
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError("payload object keys must be strings")
            _json_value(item, depth + 1)
        return
    raise SchemaError("unsupported payload type")


def _validate_key(key: Any) -> str:
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise SchemaError("invalid state key")
    if key.lower().startswith(_DENIED_KEYS):
        raise SchemaError("state key uses a restricted private/weight namespace")
    return key


def _validate_payload(op: StateOp, value: Any) -> str:
    if not isinstance(value, dict):
        raise SchemaError("operation payload must be a JSON object")
    expected = {"key", "value"} if op is StateOp.SET else {"key"}
    if set(value) != expected:
        raise SchemaError("operation payload fields mismatch")
    _validate_key(value["key"])
    _json_value(value)
    encoded = canonical_json(value)
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise SchemaError("payload byte limit exceeded")
    return encoded


def _refs(value: tuple[str, ...] | list[str], label: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise SchemaError(f"{label} must be a list/tuple, never scalar text")
    refs = tuple(value)
    if not refs or len(refs) > MAX_REFS or len(set(refs)) != len(refs):
        raise SchemaError(f"{label} must contain 1–{MAX_REFS} unique references")
    for ref in refs:
        _identifier(ref, label)
    return refs


@dataclass(frozen=True)
class EvidenceBinding:
    artifact_sha256: str
    source_head_sha256: str
    host_test_receipt_sha256: str
    evidence_class: EvidenceClass

    def __post_init__(self) -> None:
        for field in ("artifact_sha256", "source_head_sha256", "host_test_receipt_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise SchemaError(f"invalid {field}")
        try:
            object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        except ValueError as exc:
            raise SchemaError("unsupported evidence class") from exc

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "source_head_sha256": self.source_head_sha256,
            "host_test_receipt_sha256": self.host_test_receipt_sha256,
            "evidence_class": self.evidence_class.value,
        }

    def sha256(self) -> str:
        return _digest(self.as_dict())


@dataclass(frozen=True)
class StateDelta:
    version: int
    event_id: str
    source_id: str
    target_id: str
    session_id: str
    sequence: int
    expected_revision: int
    source_role: str
    operation: StateOp
    payload_json: str
    issued_at_epoch: int
    expires_at_epoch: int
    evidence: EvidenceBinding
    provenance_refs: tuple[str, ...]
    capability_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise SchemaError("unsupported StateDelta version")
        for field in ("event_id", "source_id", "target_id", "session_id"):
            _identifier(getattr(self, field), field)
        if self.source_role != "owner":
            raise SchemaError("unsupported claimed source role")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise SchemaError("sequence must be a positive integer")
        if isinstance(self.expected_revision, bool) or not isinstance(self.expected_revision, int) or self.expected_revision < 0:
            raise SchemaError("expected_revision must be a nonnegative integer")
        try:
            object.__setattr__(self, "operation", StateOp(self.operation))
        except ValueError as exc:
            raise SchemaError("unsupported StateBus operation") from exc
        if not isinstance(self.payload_json, str):
            raise SchemaError("payload_json must be text")
        payload = _parse_json(self.payload_json)
        object.__setattr__(self, "payload_json", _validate_payload(self.operation, payload))
        start = _epoch(self.issued_at_epoch, "issued_at_epoch")
        end = _epoch(self.expires_at_epoch, "expires_at_epoch")
        if end <= start:
            raise SchemaError("expires_at_epoch must follow issued_at_epoch")
        if not isinstance(self.evidence, EvidenceBinding):
            raise SchemaError("evidence binding required")
        object.__setattr__(self, "provenance_refs", _refs(self.provenance_refs, "provenance_refs"))
        object.__setattr__(self, "capability_refs", _refs(self.capability_refs, "capability_refs"))
        if len(self.to_json().encode("utf-8")) > MAX_ENVELOPE_BYTES:
            raise SchemaError("envelope byte limit exceeded")

    @classmethod
    def create(cls, *, event_id: str, source_id: str, target_id: str,
               session_id: str, sequence: int, expected_revision: int,
               operation: StateOp,
               payload: Mapping[str, Any], issued_at_epoch: int,
               expires_at_epoch: int, evidence: EvidenceBinding,
               provenance_refs: tuple[str, ...], capability_refs: tuple[str, ...],
               source_role: str = "owner") -> "StateDelta":
        op = StateOp(operation)
        return cls(SCHEMA_VERSION, event_id, source_id, target_id, session_id,
                   sequence, expected_revision, source_role, op, _validate_payload(op, dict(payload)),
                   issued_at_epoch, expires_at_epoch, evidence,
                   provenance_refs, capability_refs)

    @property
    def payload(self) -> dict[str, Any]:
        return _parse_json(self.payload_json)  # new deep copy; no mutable alias

    def _body(self) -> dict[str, Any]:
        return {
            "version": self.version, "event_id": self.event_id,
            "source_id": self.source_id, "target_id": self.target_id,
            "session_id": self.session_id, "sequence": self.sequence,
            "expected_revision": self.expected_revision,
            "source_role": self.source_role, "operation": self.operation.value,
            "payload": self.payload, "issued_at_epoch": self.issued_at_epoch,
            "expires_at_epoch": self.expires_at_epoch,
            "evidence": self.evidence.as_dict(),
            "provenance_refs": list(self.provenance_refs),
            "capability_refs": list(self.capability_refs),
        }

    @property
    def content_hash(self) -> str:
        return _digest(self._body())

    def to_json(self) -> str:
        return canonical_json({**self._body(), "content_hash": self.content_hash})

    @classmethod
    def from_json(cls, raw: str) -> "StateDelta":
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_ENVELOPE_BYTES:
            raise SchemaError("invalid or oversized envelope")
        data = _parse_json(raw)
        expected = {
            "version", "event_id", "source_id", "target_id", "session_id", "sequence",
            "expected_revision",
            "source_role", "operation", "payload", "issued_at_epoch", "expires_at_epoch",
            "evidence", "provenance_refs", "capability_refs", "content_hash",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise SchemaError("unknown/missing envelope field (self-asserted verified forbidden)")
        if not isinstance(data["provenance_refs"], list) or not isinstance(data["capability_refs"], list):
            raise SchemaError("provenance/capability refs must be JSON arrays")
        evidence = data["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {
            "artifact_sha256", "source_head_sha256", "host_test_receipt_sha256", "evidence_class",
        }:
            raise SchemaError("invalid evidence binding fields")
        obj = cls.create(
            event_id=data["event_id"], source_id=data["source_id"],
            target_id=data["target_id"], session_id=data["session_id"],
            sequence=data["sequence"], expected_revision=data["expected_revision"],
            operation=data["operation"],
            payload=data["payload"], issued_at_epoch=data["issued_at_epoch"],
            expires_at_epoch=data["expires_at_epoch"], evidence=EvidenceBinding(**evidence),
            provenance_refs=data["provenance_refs"],
            capability_refs=data["capability_refs"],
            source_role=data["source_role"],
        )
        if data["version"] != SCHEMA_VERSION or data["content_hash"] != obj.content_hash:
            raise SchemaError("StateDelta version or content hash mismatch")
        return obj


@dataclass(frozen=True)
class ValidationReceipt:
    """Verifier assertion record; authenticity comes ONLY from callback."""

    version: int
    validation_id: str
    proposal_hash: str
    evidence_sha256: str
    verifier_id: str
    verifier_role: str
    decision: str
    evaluated_at_epoch: int

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise SchemaError("unsupported validation receipt version")
        _identifier(self.validation_id, "validation_id")
        _identifier(self.verifier_id, "verifier_id")
        if self.verifier_role != "independent-verifier" or self.decision != "approve":
            raise SchemaError("invalid verifier role/decision")
        for field in ("proposal_hash", "evidence_sha256"):
            if not isinstance(getattr(self, field), str) or not _HEX64.fullmatch(getattr(self, field)):
                raise SchemaError(f"invalid {field}")
        _epoch(self.evaluated_at_epoch, "evaluated_at_epoch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "validation_id": self.validation_id,
            "proposal_hash": self.proposal_hash,
            "evidence_sha256": self.evidence_sha256,
            "verifier_id": self.verifier_id,
            "verifier_role": self.verifier_role, "decision": self.decision,
            "evaluated_at_epoch": self.evaluated_at_epoch,
        }

    def to_json(self) -> str:
        return canonical_json(self.as_dict())

    @classmethod
    def from_json(cls, raw: str) -> "ValidationReceipt":
        obj = _parse_json(raw)
        if not isinstance(obj, dict) or set(obj) != {
            "version", "validation_id", "proposal_hash", "evidence_sha256",
            "verifier_id", "verifier_role", "decision", "evaluated_at_epoch",
        }:
            raise SchemaError("invalid validation receipt fields")
        return cls(**obj)
