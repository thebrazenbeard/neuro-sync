"""A deliberately restricted, deterministic JSON wire profile.

This is *not* RFC 8785 JCS and is *not* the wider StateBus v0 envelope.
It provides one tiny, testable cross-runtime transport subset that avoids
JavaScript/Python disagreements about floating-point encodings, UTF-16 object
key sorting, and integer precision. A frame is untrusted data: neither its
content hash nor its ``kind`` grants authority, execution or state custody.

Only ASCII object keys, Unicode-scalar string values, integers within the
JavaScript safe-integer range, booleans, null, arrays and objects are allowed.
Floating point numbers are rejected *including* -0.0. The wire representation
preserves Unicode spelling exactly; it never normalizes composed/decomposed
strings, so those are intentionally different byte sequences and hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


WIRE_PROFILE = "statebus-restricted-json-v1"
MAX_WIRE_BYTES = 16384
MAX_DEPTH = 8
MAX_COLLECTION = 512
MAX_SAFE_INTEGER = (1 << 53) - 1
_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$", re.ASCII)
_KIND = re.compile(r"^[a-z][a-z0-9_-]{0,63}$", re.ASCII)
_UNSAFE_JS_KEYS = frozenset({"__proto__", "prototype", "constructor"})


class WireError(ValueError):
    """The transport data is unsafe, invalid or outside this wire profile."""


def _text(value: str) -> None:
    if not isinstance(value, str):
        raise WireError("text value required")
    # JSON strings with unpaired surrogates are not valid Unicode scalar text.
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise WireError("unpaired Unicode surrogate is not permitted")


def _validate(value: Any, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise WireError("maximum nesting depth exceeded")
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        _text(value)
        return
    if type(value) is int:
        if abs(value) > MAX_SAFE_INTEGER:
            raise WireError("integer outside portable 53-bit safe range")
        return
    if type(value) is list:
        if len(value) > MAX_COLLECTION:
            raise WireError("array is too long")
        for item in value:
            _validate(item, depth + 1)
        return
    if type(value) is dict:
        if len(value) > MAX_COLLECTION:
            raise WireError("object has too many keys")
        for key, item in value.items():
            if type(key) is not str or not _KEY.fullmatch(key) or key in _UNSAFE_JS_KEYS:
                raise WireError("nonportable or unsafe JSON object key")
            _validate(item, depth + 1)
        return
    # In particular, JSON has no NaN/Infinity, and Python float formatting is
    # not ECMAScript/JCS number serialization. Reject *all* floats for v1.
    raise WireError("unsupported JSON value (floats are excluded)")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WireError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_number(value: str) -> Any:
    raise WireError("floating point/nonfinite JSON numbers are excluded")


def _parse_integer(value: str) -> int:
    # Check lexical length *before* int() to avoid Python's configurable
    # 4300-digit conversion limit leaking an unexpected ValueError to callers.
    # 16 decimal digits is the maximum length of a positive safe integer.
    digits = value.removeprefix("-")
    if len(digits) > 16:
        raise WireError("integer exceeds portable safe length")
    result = int(value)
    if abs(result) > MAX_SAFE_INTEGER:
        raise WireError("integer outside portable safe range")
    return result


def _strict_load(blob: bytes) -> Any:
    if not isinstance(blob, bytes):
        raise WireError("wire input must be exact bytes")
    if not blob or len(blob) > MAX_WIRE_BYTES:
        raise WireError("wire byte length outside allowed bounds")
    try:
        raw = blob.decode("utf-8", errors="strict")
        obj = json.loads(raw, object_pairs_hook=_unique_pairs, parse_int=_parse_integer,
                         parse_float=_reject_number, parse_constant=_reject_number)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise WireError("invalid UTF-8 JSON") from exc
    _validate(obj)
    return obj


def canonical_bytes(value: Any) -> bytes:
    """Encode restricted JSON data deterministically; not a signature."""
    _validate(value)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_WIRE_BYTES:
        raise WireError("wire byte length exceeded")
    return encoded


def decode_canonical(blob: bytes) -> Any:
    """Decode only canonical bytes; refuse alternate but parseable JSON."""
    obj = _strict_load(blob)
    if canonical_bytes(obj) != blob:
        raise WireError("noncanonical or ambiguous JSON encoding")
    return obj


@dataclass(frozen=True)
class WireFrame:
    kind: str
    _wire_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        if type(self._wire_bytes) is not bytes:
            raise WireError("frame bytes must be immutable built-in bytes")
        if self.sha256 != hashlib.sha256(self._wire_bytes).hexdigest():
            raise WireError("frame digest does not match its stored bytes")
        value = decode_canonical(self._wire_bytes)
        if (not isinstance(value, dict)
                or set(value) != {"wire_profile", "kind", "payload"}
                or value["wire_profile"] != WIRE_PROFILE
                or not isinstance(self.kind, str)
                or not _KIND.fullmatch(self.kind)
                or value["kind"] != self.kind):
            raise WireError("frame metadata does not match stored bytes")

    @property
    def payload(self) -> Any:
        """A fresh parsed copy; mutation cannot desynchronize this frame's hash."""
        return decode_canonical(self._wire_bytes)["payload"]


def encode_frame(kind: str, payload: Any) -> bytes:
    """Frame data with an exact profile tag; no authentication is implied."""
    if not isinstance(kind, str) or not _KIND.fullmatch(kind):
        raise WireError("invalid frame kind")
    return canonical_bytes({"wire_profile": WIRE_PROFILE, "kind": kind, "payload": payload})


def decode_frame(blob: bytes, *, expected_kind: str | None = None) -> WireFrame:
    obj = decode_canonical(blob)
    if not isinstance(obj, dict) or set(obj) != {"wire_profile", "kind", "payload"}:
        raise WireError("wire frame fields mismatch")
    if obj["wire_profile"] != WIRE_PROFILE:
        raise WireError("unknown wire profile; no silent downgrade")
    kind = obj["kind"]
    if not isinstance(kind, str) or not _KIND.fullmatch(kind):
        raise WireError("invalid frame kind")
    if expected_kind is not None and kind != expected_kind:
        raise WireError("unexpected frame kind")
    return WireFrame(kind, blob, hashlib.sha256(blob).hexdigest())
