"""Bounded, transport-only cross-runtime wire-profile checks.

Node is optional. Its independent JSON serializer checks the *restricted*
v1 domain on an actual second runtime; it does not establish RFC 8785 JCS,
authenticated principals, or end-to-end StateBus interoperability.
"""

import hashlib
import json
import shutil
import subprocess
import unittest

from neurosync.bus.wire import (
    MAX_SAFE_INTEGER,
    WIRE_PROFILE,
    WireError,
    WireFrame,
    canonical_bytes,
    decode_canonical,
    decode_frame,
    encode_frame,
)


NODE_CANONICAL = r"""
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
function ordered(v) {
  if (Array.isArray(v)) return v.map(ordered);
  if (v !== null && typeof v === 'object') {
    const obj = {};
    for (const k of Object.keys(v).sort()) Object.defineProperty(obj, k, {
      value: ordered(v[k]), enumerable: true, writable: true, configurable: true,
    });
    return obj;
  }
  return v;
}
process.stdout.write(JSON.stringify(ordered(input)));
"""


class WireProfileTests(unittest.TestCase):
    def test_profile_tag_and_hash_of_actual_bytes(self):
        blob = encode_frame("capability_snapshot", {"version": 1, "ref": "fixture.ref"})
        self.assertEqual(blob, b'{"kind":"capability_snapshot","payload":{"ref":"fixture.ref","version":1},"wire_profile":"statebus-restricted-json-v1"}')
        frame = decode_frame(blob, expected_kind="capability_snapshot")
        self.assertEqual(frame.kind, "capability_snapshot")
        self.assertEqual(frame.payload, {"ref": "fixture.ref", "version": 1})
        self.assertEqual(frame.sha256, hashlib.sha256(blob).hexdigest())
        self.assertEqual(WIRE_PROFILE, "statebus-restricted-json-v1")

    def test_decoded_frame_hash_not_desynchronized_by_payload_mutation(self):
        blob = encode_frame("state_delta", {"value": {"nested": [1]}})
        frame = decode_frame(blob)
        purported_payload = frame.payload
        purported_payload["value"]["nested"].append(2)
        # The hash always refers to original immutable wire bytes. Every
        # payload access returns a fresh copy, not a mutable cached alias.
        self.assertEqual(frame.payload, {"value": {"nested": [1]}})
        self.assertEqual(frame.sha256, hashlib.sha256(blob).hexdigest())
        self.assertRaises(WireError, WireFrame, "state_delta", blob, "f" * 64)
        self.assertRaises(WireError, WireFrame, "other_kind", blob,
                          hashlib.sha256(blob).hexdigest())

    def test_reject_profile_downgrade_and_unknown_kind(self):
        blob = encode_frame("test_result", {"passed": True})
        self.assertRaises(WireError, decode_frame, blob, expected_kind="state_delta")
        self.assertRaises(WireError, encode_frame, "TEST_RESULT", {})
        data = decode_canonical(blob)
        data["wire_profile"] = "statebus-restricted-json-v0"
        self.assertRaises(WireError, decode_frame, canonical_bytes(data))

    def test_reject_extra_fields_and_wrong_frame_shape(self):
        blob = encode_frame("test_result", {"ok": True})
        data = decode_canonical(blob)
        data["claimed_verified"] = True
        self.assertRaises(WireError, decode_frame, canonical_bytes(data))
        self.assertRaises(WireError, decode_frame, canonical_bytes([]))

    def test_integer_range_is_exact_portable_subset(self):
        for value in (-MAX_SAFE_INTEGER, 0, MAX_SAFE_INTEGER):
            self.assertEqual(decode_canonical(canonical_bytes({"v": value})), {"v": value})
        for value in (-MAX_SAFE_INTEGER - 1, MAX_SAFE_INTEGER + 1):
            self.assertRaises(WireError, canonical_bytes, {"v": value})
            self.assertRaises(WireError, decode_canonical, json.dumps({"v": value}).encode())

    def test_oversize_integer_never_leaks_python_conversion_error(self):
        for digits in (5000, 12000):
            with self.subTest(digits=digits):
                for prefix in (b"", b"-"):
                    hostile = b'{"v":' + prefix + b"9" * digits + b"}"
                    self.assertLess(len(hostile), 16384)
                    self.assertRaises(WireError, decode_canonical, hostile)

    def test_reject_every_float_and_nonfinite_number(self):
        for number in (0.0, -0.0, 1.5, 1e20, float("nan"), float("inf")):
            with self.subTest(number=repr(number)):
                self.assertRaises(WireError, canonical_bytes, {"v": number})
        for raw in (b'{"v":0.0}', b'{"v":1e2}', b'{"v":NaN}', b'{"v":Infinity}'):
            self.assertRaises(WireError, decode_canonical, raw)

    def test_integer_negative_zero_encoding_cannot_alias_zero(self):
        self.assertEqual(decode_canonical(b'{"v":0}'), {"v": 0})
        self.assertRaises(WireError, decode_canonical, b'{"v":-0}')

    def test_reject_duplicate_keys_at_any_depth(self):
        for blob in (b'{"v":1,"v":2}', b'{"a":{"b":1,"b":2}}'):
            self.assertRaises(WireError, decode_canonical, blob)

    def test_reject_reordered_and_noncanonical_json(self):
        for blob in (b'{"z":1,"a":2}', b'{ "a":1}', b'{"a":1 }', b'{"a":1}\n'):
            self.assertRaises(WireError, decode_canonical, blob)

    def test_reject_non_utf8_and_surrogates(self):
        self.assertRaises(WireError, decode_canonical, b'\xff')
        self.assertRaises(WireError, canonical_bytes, {"a": "\ud800"})
        self.assertRaises(WireError, decode_canonical, b'{"a":"\\udfff"}')

    def test_reject_unsafe_keys_and_non_ascii_keys(self):
        for key in ("constructor", "prototype", "__proto__", "κλειδί", "x y", "", "1number"):
            self.assertRaises(WireError, canonical_bytes, {key: "value"})
        self.assertRaises(WireError, decode_canonical, b'{"constructor":0}')

    def test_reject_byte_limits_and_depth(self):
        self.assertRaises(WireError, canonical_bytes, {"a": "x" * 20000})
        self.assertRaises(WireError, decode_canonical, b"{" + b"x" * 20000)
        nested = 0
        for _ in range(9):
            nested = [nested]
        self.assertRaises(WireError, canonical_bytes, nested)
        # Some Python builds hit parser recursion limits before validation;
        # this must still be a typed refusal, not an uncaught RecursionError.
        hostile = b"[" * 1500 + b"0" + b"]" * 1500
        self.assertRaises(WireError, decode_canonical, hostile)

    def test_unknown_types_and_implicit_coercion(self):
        for value in ({"v": b"byte data"}, {"v": (1, 2)}, {"v": {1: "integer key"}}):
            self.assertRaises(WireError, canonical_bytes, value)
        self.assertRaises(WireError, decode_canonical, '{"v":1}')  # bytes only

    def test_custom_mapping_or_sequence_subclass_cannot_lie_to_validator(self):
        class MisleadingDict(dict):
            def items(self):
                return iter([("safe", 1)])

        # Do not assume that validation iteration and a JSON encoder's view
        # of a mutable/custom container are equivalent.
        self.assertRaises(WireError, canonical_bytes,
                          MisleadingDict({"credential": "unvalidated"}))
        self.assertRaises(WireError, canonical_bytes, [MisleadingDict({"value": 1})])

        class MutableList(list):
            pass

        self.assertRaises(WireError, canonical_bytes, MutableList([1]))

    def test_unicode_value_spelling_preserved_not_normalized(self):
        precomposed = canonical_bytes({"text": "é"})
        decomposed = canonical_bytes({"text": "e\u0301"})
        self.assertNotEqual(precomposed, decomposed)
        self.assertEqual(decode_canonical(precomposed)["text"], "é")
        self.assertEqual(decode_canonical(decomposed)["text"], "e\u0301")

    @unittest.skipUnless(shutil.which("node"), "Node.js not installed; cross-runtime fixture unverified")
    def test_python_and_node_restricted_wire_bytes(self):
        vectors = [
            {"z": 1, "a": {"k3": None, "k1": True, "k2": False}},
            {"a": [0, 1, -1, MAX_SAFE_INTEGER, -MAX_SAFE_INTEGER]},
            {"text": "é", "combining": "e\u0301", "astral": "✨🌌"},
            {"control": "a\n\t\b\f\r\\\"", "separator": "\u2028"},
            {"session-A": {"cap_ref": "cap-id", "sequence": 11}},
        ]
        for index, value in enumerate(vectors):
            with self.subTest(index=index):
                raw = json.dumps(value, ensure_ascii=True, allow_nan=False)
                result = subprocess.run(
                    ["node", "-e", NODE_CANONICAL], input=raw.encode("utf-8"),
                    capture_output=True, timeout=5, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                self.assertEqual(result.stdout, canonical_bytes(value))


if __name__ == "__main__":
    unittest.main()
