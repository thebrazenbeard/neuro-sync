"""Actual adversarial stdlib tests for the source-only zero-weight StateBus."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from neurosync.bus import (
    AuthorizationError, ConflictError, EvidenceBinding, EvidenceClass,
    ExpiredError, LedgerIntegrityError, SchemaError, SequenceError,
    StateDelta, StateLedger, StateOp, ValidationReceipt,
    canonical_json,
)


ALLOW_PROPOSER = lambda delta: True
ALLOW_VALIDATOR = lambda delta, receipt: True
DENY_VALIDATOR = lambda delta, receipt: False


def evidence():
    return EvidenceBinding(
        artifact_sha256="a" * 64,
        source_git_commit_oid="sha1:" + "d" * 40,
        source_tree_sha256="b" * 64,
        host_test_receipt_sha256="c" * 64,
        evidence_class=EvidenceClass.TEST,
    )


def delta(*, event_id="evt-1", source_id="node-A", target_id="node-B",
          session_id="session-A", sequence=1, expected_revision=0,
          key="facts.alpha", value="synthetic", operation=StateOp.SET,
          issued_at_epoch=1400, expires_at_epoch=2000):
    payload = {"key": key, "value": value} if operation is StateOp.SET else {"key": key}
    return StateDelta.create(
        event_id=event_id, source_id=source_id, target_id=target_id,
        session_id=session_id, sequence=sequence, expected_revision=expected_revision,
        operation=operation, payload=payload, issued_at_epoch=issued_at_epoch,
        expires_at_epoch=expires_at_epoch, evidence=evidence(),
        provenance_refs=("artifact-cas-a1",), capability_refs=("cap-fixture-c1",),
    )


def validation(item, *, verifier_id="verifier-C", validation_id=None):
    return ValidationReceipt(
        version=1,
        validation_id=validation_id or f"validation-{item.event_id}",
        proposal_hash=item.content_hash,
        evidence_sha256=item.evidence.sha256(),
        verifier_id=verifier_id, verifier_role="independent-verifier",
        decision="approve", evaluated_at_epoch=1500,
    )


class StateBusCoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "statebus.sqlite3"
        self.now = 1500
        self.ledger = StateLedger(self.path, clock=lambda: self.now)

    def propose(self, item):
        return self.ledger.propose(item, authorize_proposer=ALLOW_PROPOSER)

    def validate(self, item, receipt=None):
        return self.ledger.validate(
            item.event_id, receipt or validation(item), verify_validator=ALLOW_VALIDATOR,
        )

    def commit(self, item):
        return self.ledger.commit(item.event_id, verify_validator=ALLOW_VALIDATOR,
                                  authorize_commit=ALLOW_VALIDATOR)

    def complete(self, item):
        self.propose(item)
        self.validate(item)
        return self.commit(item)

    def test_schema_canonical_roundtrip_immutable_payload_and_claim_rejection(self):
        item = delta(value={"nested": [1, True, None]})
        encoded = item.to_json()
        self.assertEqual(StateDelta.from_json(encoded), item)
        self.assertEqual(StateDelta.from_json(encoded).content_hash, item.content_hash)
        payload = item.payload
        payload["value"]["nested"].append("mutation")
        self.assertEqual(item.payload["value"]["nested"], [1, True, None])

        self.assertRaises(SchemaError, StateDelta.from_json,
                          encoded.replace('"version":1', '"verified":true,"version":1'))
        self.assertRaises(SchemaError, StateDelta.from_json,
                          encoded.replace('"event_id":"evt-1"',
                                          '"event_id":"evt-1","event_id":"evt-1"'))
        self.assertRaises(SchemaError, StateDelta.from_json,
                          encoded.replace('"sequence":1', '"sequence":2'))
        # A self-consistent hash does NOT legitimize string-for-array refs.
        malformed = json.loads(encoded)
        malformed["capability_refs"] = "abc"
        body = {k: v for k, v in malformed.items() if k != "content_hash"}
        malformed["content_hash"] = hashlib.sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest()
        self.assertRaises(SchemaError, StateDelta.from_json, canonical_json(malformed))
        self.assertRaises(SchemaError, StateDelta.create,
                          event_id="e", source_id="a", target_id="b", session_id="s",
                          sequence=1, expected_revision=0, operation=StateOp.SET,
                          payload={"key": "facts.x", "value": "x"}, issued_at_epoch=1,
                          expires_at_epoch=2, evidence=evidence(),
                          provenance_refs="abc", capability_refs=("cap-fixture-c1",))
        self.assertRaises(SchemaError, delta, key="secret.api")
        self.assertRaises(SchemaError, delta, value="X" * 9000)
        self.assertRaises(SchemaError, delta, value=float("nan"))
        self.assertRaises(SchemaError, delta, expected_revision=-1)
        with self.assertRaises(SchemaError):
            EvidenceBinding(
                artifact_sha256="a" * 64,
                source_git_commit_oid="601907eb225ffb75d71ceb467af778156f486861",  # unqualified SHA1
                source_tree_sha256="b" * 64,
                host_test_receipt_sha256="c" * 64,
                evidence_class=EvidenceClass.TEST,
            )
        self.assertEqual(evidence().source_git_commit_oid, "sha1:" + "d" * 40)

    def test_propose_validate_commit_readback_and_restart(self):
        item = delta()
        proposed = self.propose(item)
        self.assertEqual((proposed.status, proposed.already_present), ("proposed", False))
        self.assertEqual(self.ledger.project("node-B", "session-A"), {})
        validated = self.validate(item)
        self.assertEqual(validated.status, "validated")
        self.assertEqual(self.ledger.project("node-B", "session-A"), {})
        committed = self.commit(item)
        self.assertEqual((committed.event_offset, committed.key_revision), (1, 1))
        self.assertEqual(self.ledger.project("node-B", "session-A"), {"facts.alpha": "synthetic"})
        self.assertEqual(self.ledger.get_revision("node-B", "session-A", "facts.alpha"), 1)
        self.assertTrue(self.ledger.verify_chain())
        self.assertEqual(len(self.ledger.events("node-B", "session-A")), 1)
        self.assertEqual(self.ledger.events("node-B", "session-A")[0].delta.content_hash, item.content_hash)

        fresh_process_like = StateLedger(self.path, clock=lambda: self.now)
        self.assertEqual(fresh_process_like.project("node-B", "session-A"), {"facts.alpha": "synthetic"})
        self.assertTrue(fresh_process_like.verify_chain())

    def test_authority_callback_required_self_verification_forbidden(self):
        item = delta()
        with self.assertRaises(AuthorizationError):
            self.ledger.propose(item, authorize_proposer=None)
        with self.assertRaises(AuthorizationError):
            self.ledger.propose(item, authorize_proposer=lambda _: "verified")
        self.propose(item)
        with self.assertRaises(AuthorizationError):
            self.ledger.validate(item.event_id, validation(item), verify_validator=None)
        with self.assertRaises(AuthorizationError):
            self.ledger.validate(item.event_id, validation(item), verify_validator=DENY_VALIDATOR)
        with self.assertRaises(AuthorizationError):
            self.ledger.validate(item.event_id, validation(item, verifier_id="node-A"),
                                 verify_validator=ALLOW_VALIDATOR)
        with self.assertRaises(ConflictError):
            self.ledger.commit(item.event_id, verify_validator=ALLOW_VALIDATOR,
                               authorize_commit=ALLOW_VALIDATOR)
        self.validate(item)
        with self.assertRaises(AuthorizationError):
            self.ledger.commit(item.event_id, verify_validator=DENY_VALIDATOR,
                               authorize_commit=ALLOW_VALIDATOR)
        self.assertEqual(self.ledger.project("node-B", "session-A"), {})
        self.commit(item)

    def test_binding_fake_receipt_or_changed_artifact_denied(self):
        item = delta()
        self.propose(item)
        forged = replace(validation(item), proposal_hash="f" * 64)
        with self.assertRaises(ConflictError):
            self.ledger.validate(item.event_id, forged, verify_validator=ALLOW_VALIDATOR)
        forged_evidence = replace(validation(item), evidence_sha256="f" * 64)
        with self.assertRaises(ConflictError):
            self.ledger.validate(item.event_id, forged_evidence, verify_validator=ALLOW_VALIDATOR)
        self.assertEqual(self.ledger.project("node-B", "session-A"), {})

    def test_commit_revocation_after_validation_is_denied(self):
        item = delta()
        self.propose(item)
        self.validate(item)
        with self.assertRaises(AuthorizationError):
            self.ledger.commit(item.event_id, verify_validator=ALLOW_VALIDATOR,
                               authorize_commit=lambda d, receipt: False)
        with self.assertRaises(AuthorizationError):
            self.ledger.commit(item.event_id, verify_validator=ALLOW_VALIDATOR,
                               authorize_commit=None)
        self.assertEqual(self.ledger.project("node-B", "session-A"), {})
        self.assertEqual(self.commit(item).event_offset, 1)

    def test_idempotency_same_event_new_sequence_and_crash_before_ack(self):
        item = delta()
        self.assertFalse(self.propose(item).already_present)
        self.assertTrue(self.propose(item).already_present)
        with self.assertRaises(ConflictError):
            self.propose(delta(sequence=2))  # same ID, new sequence
        with self.assertRaises(ConflictError):
            self.propose(delta(value="different"))  # same ID, different payload
        with self.assertRaises(ConflictError):
            self.propose(delta(event_id="different-id"))  # same source/session/seq
        self.validate(item)
        receipt = self.commit(item)
        retry = self.commit(item)  # simulate successful COMMIT, lost ACK, retry
        self.assertTrue(retry.already_committed)
        self.assertEqual((retry.event_offset, retry.chain_hash),
                         (receipt.event_offset, receipt.chain_hash))
        self.assertEqual(len(self.ledger.events("node-B", "session-A")), 1)

        # Expired duplicate receipts remain readable; no new effect is made.
        self.now = 2200
        self.assertTrue(self.propose(item).already_present)
        self.assertTrue(self.commit(item).already_committed)
        self.assertEqual(len(self.ledger.events("node-B", "session-A")), 1)

    def test_sequence_gap_then_ordered_commit_and_distinct_session(self):
        second = delta(event_id="evt-2", sequence=2, expected_revision=1,
                       value="second")
        self.propose(second)
        self.validate(second)
        with self.assertRaises(SequenceError):
            self.commit(second)
        first = delta()
        self.complete(first)
        self.assertEqual(self.commit(second).key_revision, 2)
        self.assertEqual(self.ledger.project("node-B", "session-A"), {"facts.alpha": "second"})
        other_session = delta(event_id="evt-other-session", session_id="session-other",
                              sequence=1, expected_revision=0, value="third")
        self.complete(other_session)
        self.assertEqual(self.ledger.get_revision("node-B", "session-A", "facts.alpha"), 2)
        self.assertEqual(self.ledger.get_revision("node-B", "session-other", "facts.alpha"), 1)
        self.assertEqual(self.ledger.project("node-B", "session-other"), {"facts.alpha": "third"})
        self.assertEqual(self.ledger.project("node-B", "session-A"), {"facts.alpha": "second"})

    def test_cross_session_state_and_revision_are_private(self):
        private_a = delta(event_id="secret-A", key="note", value="secret-for-session-A")
        self.complete(private_a)
        self.assertEqual(self.ledger.project("node-B", "session-A"),
                         {"note": "secret-for-session-A"})
        self.assertEqual(self.ledger.project("node-B", "session-B"), {})
        self.assertEqual(self.ledger.get_revision("node-B", "session-B", "note"), 0)
        self.assertEqual(self.ledger.events("node-B", "session-B"), ())
        private_b = delta(event_id="secret-B", session_id="session-B",
                          key="note", value="only-session-B", expected_revision=0)
        self.complete(private_b)
        self.assertEqual(self.ledger.project("node-B", "session-A"),
                         {"note": "secret-for-session-A"})
        self.assertEqual(self.ledger.project("node-B", "session-B"),
                         {"note": "only-session-B"})
        self.assertEqual(self.ledger.get_revision("node-B", "session-A", "note"), 1)
        self.assertEqual(self.ledger.get_revision("node-B", "session-B", "note"), 1)

    def test_key_owner_conflicts_cas_and_tombstone_does_not_reassign(self):
        self.complete(delta())
        foreign = delta(event_id="evt-foreign", source_id="node-C",
                        session_id="session-A", sequence=1, expected_revision=1)
        self.propose(foreign)
        self.validate(foreign)
        with self.assertRaisesRegex(ConflictError, "owned by"):
            self.commit(foreign)
        delete = delta(event_id="evt-delete", sequence=2, expected_revision=1,
                       operation=StateOp.DELETE)
        self.complete(delete)
        self.assertEqual(self.ledger.project("node-B", "session-A"), {})
        foreign_after_delete = delta(event_id="evt-foreign2", source_id="node-D",
                                     session_id="session-A", sequence=1,
                                     expected_revision=2)
        self.propose(foreign_after_delete)
        self.validate(foreign_after_delete)
        with self.assertRaisesRegex(ConflictError, "owned by"):
            self.commit(foreign_after_delete)
        # Original owner retains its key, even after a deletion tombstone.
        restored = delta(event_id="evt-restore", sequence=3, expected_revision=2)
        self.complete(restored)
        self.assertEqual(self.ledger.project("node-B", "session-A"), {"facts.alpha": "synthetic"})
        stale = delta(event_id="evt-stale", sequence=4, expected_revision=0)
        self.propose(stale)
        self.validate(stale)
        with self.assertRaisesRegex(ConflictError, "revision CAS"):
            self.commit(stale)

    def test_expiry_future_and_schema_rejection(self):
        self.assertRaises(ExpiredError, self.propose,
                          delta(expires_at_epoch=1499))
        self.assertRaises(ExpiredError, self.propose,
                          delta(issued_at_epoch=2000, expires_at_epoch=3000))
        item = delta(expires_at_epoch=1600)
        self.propose(item)
        self.now = 1700
        with self.assertRaises(ExpiredError):
            self.validate(item)
        with self.assertRaises(ConflictError):
            self.commit(item)

    def test_projection_copy_is_nonmutating_and_chain_detects_corruption(self):
        self.complete(delta(value={"n": [1, 2]}))
        projected = self.ledger.project("node-B", "session-A")
        projected["facts.alpha"]["n"].append(999)
        self.assertEqual(self.ledger.project("node-B", "session-A"), {"facts.alpha": {"n": [1, 2]}})
        with sqlite3.connect(self.path) as db:
            with self.assertRaises(sqlite3.DatabaseError):
                db.execute("UPDATE events SET content_hash=? WHERE event_offset=1", ("0" * 64,))
            db.execute("DROP TRIGGER events_no_update")
            db.execute("UPDATE events SET content_hash=? WHERE event_offset=1", ("0" * 64,))
        self.assertFalse(self.ledger.verify_chain())
        with self.assertRaises(LedgerIntegrityError):
            self.ledger.project("node-B", "session-A")

    def test_read_limits(self):
        with self.assertRaises(ValueError):
            self.ledger.events("node-B", "session-A", limit=501)
        with self.assertRaises(ValueError):
            self.ledger.events("node-B", "session-A", after_offset=-1)


if __name__ == "__main__":
    unittest.main()
