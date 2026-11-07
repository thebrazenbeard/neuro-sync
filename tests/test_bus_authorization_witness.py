"""Hostile source-level tests for the NON-DURABLE authorization witness reference."""

import copy
from dataclasses import replace
import concurrent.futures
from hashlib import sha256
import json
import math
import pickle
import sqlite3
import unittest

from neurosync.bus.authorization_witness import (
    InProcessAuthorizationWitness,
    WitnessCursor,
    WitnessDenied,
    WitnessGrant,
    require_witness,
)


def make_grant(**changes):
    baseline = WitnessGrant(
        authority_id="host-alpha",
        generation=7,
        principal_id="patrick",
        target_id="neuro-sync",
        session_id="session-A",
        grant_id="grant-001",
        state_key="effects.queue",
        plan_sha256="a" * 64,
        planned_state_sha256="b" * 64,
        handler_artifact_sha256="c" * 64,
        handler_source_git_oid="sha1:" + "d" * 40,
        expires_at_epoch=300.0,
    )
    return replace(baseline, **changes)


class WitnessReferenceTests(unittest.TestCase):
    def setUp(self):
        self.witness = InProcessAuthorizationWitness("host-alpha", 7)
        self.grant = make_grant()

    def issue(self):
        return self.witness.issue(self.witness.tip, self.grant, at_epoch=100)

    def test_missing_witness_denied_and_downgrade_not_silent(self):
        with self.assertRaisesRegex(WitnessDenied, "unavailable"):
            require_witness(None)
        with self.assertRaisesRegex(WitnessDenied, "contract"):
            require_witness(object())
        class BrokenProvider:
            tip = None
            observe_time = None
            issue = None
            reserve_claim = None
            revoke = None
        with self.assertRaisesRegex(WitnessDenied, "contract"):
            require_witness(BrokenProvider())
        class BrokenMethods:
            tip = self.witness.tip
            observe_time = None
            issue = None
            reserve_claim = None
            revoke = None
        with self.assertRaisesRegex(WitnessDenied, "contract"):
            require_witness(BrokenMethods())
        self.assertIs(require_witness(self.witness), self.witness)

    def test_claim_is_irreversible_before_any_effect(self):
        issued = self.issue()
        effect_count = 0

        def effect_handler(cursor, invocation):
            nonlocal effect_count
            self.witness.reserve_claim(
                cursor, self.grant, invocation, at_epoch=101
            )
            effect_count += 1

        effect_handler(issued.cursor, "invoke-001")
        with self.assertRaises(WitnessDenied):
            effect_handler(issued.cursor, "invoke-002")
        with self.assertRaisesRegex(WitnessDenied, "consumed"):
            effect_handler(self.witness.tip, "invoke-001")
        self.assertEqual(effect_count, 1)
        self.assertEqual([event.kind for event in self.witness.events], ["issue", "claim"])

    def test_restore_main_database_snapshot_fails_closed_against_consumed_witness(self):
        issued = self.issue()
        main = sqlite3.connect(":memory:")
        self.addCleanup(main.close)
        main.execute("CREATE TABLE authority (id INTEGER PRIMARY KEY, seq INTEGER, sha TEXT)")
        main.execute(
            "INSERT INTO authority VALUES (1, ?, ?)",
            (issued.cursor.sequence, issued.cursor.event_sha256),
        )
        main.commit()
        snapshot = tuple(main.execute("SELECT seq, sha FROM authority").fetchone())

        claimed = self.witness.reserve_claim(
            WitnessCursor(*snapshot), self.grant, "invoke-001", at_epoch=101
        )
        effects = 1  # Handler runs only after claim; this test simulates its side effect.
        main.execute(
            "UPDATE authority SET seq = ?, sha = ? WHERE id = 1",
            (claimed.cursor.sequence, claimed.cursor.event_sha256),
        )
        main.commit()

        # A complete DB restoration can revert only main custody, not this live witness.
        main.execute(
            "UPDATE authority SET seq = ?, sha = ? WHERE id = 1", snapshot
        )
        main.commit()
        restored = WitnessCursor(*main.execute(
            "SELECT seq, sha FROM authority"
        ).fetchone())
        with self.assertRaisesRegex(WitnessDenied, "stale"):
            self.witness.reserve_claim(
                restored, self.grant, "invoke-002", at_epoch=102
            )
        self.assertEqual(effects, 1)
        self.assertEqual(self.witness.tip, claimed.cursor)

    def test_witness_append_then_main_db_rollback_requires_operator_intervention(self):
        issued = self.issue()
        main_cursor = issued.cursor
        claim = self.witness.reserve_claim(
            main_cursor, self.grant, "invoke-crashed", at_epoch=101
        )
        # Main transaction rolled back or crashed; no auto-heal or witness rewind.
        self.assertNotEqual(main_cursor, claim.cursor)
        with self.assertRaisesRegex(WitnessDenied, "stale"):
            self.witness.observe_time(main_cursor, at_epoch=102)
        with self.assertRaisesRegex(WitnessDenied, "stale"):
            self.witness.revoke(main_cursor, self.grant, at_epoch=102)
        self.assertEqual(self.witness.tip, claim.cursor)

    def test_grant_bindings_reject_plan_state_handler_generation_and_context(self):
        self.issue()
        for field, replacement in (
            ("plan_sha256", "f" * 64),
            ("planned_state_sha256", "f" * 64),
            ("handler_artifact_sha256", "f" * 64),
            ("handler_source_git_oid", "sha1:" + "e" * 40),
            ("state_key", "other.queue"),
            ("generation", 8),
            ("authority_id", "host-beta"),
            ("principal_id", "someone-else"),
            ("target_id", "other-target"),
            ("session_id", "session-B"),
            ("grant_id", "grant-002"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(WitnessDenied):
                    self.witness.reserve_claim(
                        self.witness.tip,
                        replace(self.grant, **{field: replacement}),
                        "invoke-001", at_epoch=101,
                    )
        self.assertEqual(len(self.witness.events), 1)

    def test_revoked_grant_rejects_claim_even_at_latest_tip(self):
        issued = self.issue()
        revoked = self.witness.revoke(issued.cursor, self.grant, at_epoch=101)
        self.assertEqual(revoked.kind, "revoke")
        with self.assertRaisesRegex(WitnessDenied, "revoked"):
            self.witness.reserve_claim(
                revoked.cursor, self.grant, "invoke-001", at_epoch=102
            )
        with self.assertRaisesRegex(WitnessDenied, "stale"):
            self.witness.reserve_claim(
                issued.cursor, self.grant, "invoke-001", at_epoch=102
            )
        with self.assertRaises(WitnessDenied):
            self.witness.revoke(revoked.cursor, self.grant, at_epoch=102)

    def test_clock_failures_do_not_append_or_reset_tip(self):
        first = self.witness.observe_time(self.witness.tip, at_epoch=250)
        for invalid in (249, math.inf, math.nan, -math.inf,
                        2 ** 53 + 1, 10 ** 1000, "251", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(WitnessDenied):
                    self.witness.observe_time(first.cursor, at_epoch=invalid)
        self.assertEqual(self.witness.tip, first.cursor)
        self.assertEqual(len(self.witness.events), 1)
        second = self.witness.observe_time(first.cursor, at_epoch=251)
        self.assertEqual(second.cursor.sequence, 2)

    def test_grant_expiry_and_duplicate_issue(self):
        issued = self.issue()
        with self.assertRaisesRegex(WitnessDenied, "expired"):
            self.witness.reserve_claim(
                issued.cursor, self.grant, "invoke-001", at_epoch=301
            )
        with self.assertRaisesRegex(WitnessDenied, "already used"):
            self.witness.issue(issued.cursor, self.grant, at_epoch=120)
        with self.assertRaisesRegex(WitnessDenied, "already used"):
            self.witness.issue(
                issued.cursor,
                make_grant(principal_id="other-principal", session_id="session-B"),
                at_epoch=120,
            )
        with self.assertRaisesRegex(WitnessDenied, "already expired"):
            self.witness.issue(
                issued.cursor,
                make_grant(grant_id="grant-expired", expires_at_epoch=110),
                at_epoch=110,
            )
        self.assertEqual(self.witness.tip, issued.cursor)

    def test_invalid_input_types_and_authority(self):
        with self.assertRaises(ValueError):
            WitnessCursor(True, "a" * 64)
        with self.assertRaises(ValueError):
            WitnessCursor(1, "bad")
        with self.assertRaises(ValueError):
            make_grant(generation=True)
        with self.assertRaises(ValueError):
            make_grant(expires_at_epoch=math.nan)
        with self.assertRaises(ValueError):
            make_grant(expires_at_epoch=10 ** 1000)
        with self.assertRaises(ValueError):
            make_grant(expires_at_epoch=2 ** 53 + 1)
        with self.assertRaises(ValueError):
            make_grant(handler_source_git_oid="untrusted")
        with self.assertRaisesRegex(WitnessDenied, "authority"):
            self.witness.issue(
                self.witness.tip, make_grant(generation=8), at_epoch=100
            )
        self.assertFalse(self.witness.events)

    def test_precision_rounding_cannot_accept_claim_at_expiry(self):
        upper_exact = 2 ** 53
        exact_grant = make_grant(expires_at_epoch=upper_exact)
        issued = self.witness.issue(
            self.witness.tip, exact_grant, at_epoch=100
        )
        with self.assertRaisesRegex(WitnessDenied, "expired"):
            self.witness.reserve_claim(
                issued.cursor, exact_grant, "invoke-at-expiry",
                at_epoch=upper_exact,
            )
        # The adjacent integer is not representable in binary64.
        with self.assertRaisesRegex(WitnessDenied, "exact integer range"):
            self.witness.reserve_claim(
                issued.cursor, exact_grant, "invoke-rounded",
                at_epoch=upper_exact + 1,
            )
        self.assertEqual(self.witness.tip, issued.cursor)

    def test_equal_numeric_expiry_produces_identical_canonical_events(self):
        integer_grant = make_grant(expires_at_epoch=300)
        float_grant = make_grant(expires_at_epoch=300.0)
        self.assertEqual(integer_grant, float_grant)
        self.assertIs(type(integer_grant.expires_at_epoch), float)
        witness_a = InProcessAuthorizationWitness("host-alpha", 7)
        witness_b = InProcessAuthorizationWitness("host-alpha", 7)
        event_a = witness_a.issue(witness_a.tip, integer_grant, at_epoch=100)
        event_b = witness_b.issue(witness_b.tip, float_grant, at_epoch=100.0)
        self.assertEqual(event_a.canonical_payload_json, event_b.canonical_payload_json)
        self.assertEqual(event_a.cursor, event_b.cursor)

    def test_exported_events_reconstruct_chain_and_expose_tampering(self):
        issued = self.issue()
        claimed = self.witness.reserve_claim(
            issued.cursor, self.grant, "invoke-001", at_epoch=101
        )
        revoked = self.witness.revoke(claimed.cursor, self.grant, at_epoch=102)
        events = self.witness.events
        self.assertEqual(len(events), 3)

        # An offline reader can independently reconstruct every digest from
        # exported canonical payload bytes, including the full grant, and
        # independently check the predecessor chain.
        predecessor = sha256(
            b"neurosync:authorization-witness:genesis:host-alpha:7"
        ).hexdigest()
        for sequence, event in enumerate(events, start=1):
            payload = json.loads(event.canonical_payload_json)
            self.assertEqual(payload["sequence"], sequence)
            self.assertEqual(payload["previous_sha256"], predecessor)
            self.assertEqual(payload["kind"], event.kind)
            self.assertEqual(payload["observed_epoch"], event.observed_epoch)
            self.assertEqual(payload["grant"]["plan_sha256"], self.grant.plan_sha256)
            self.assertEqual(payload["grant"]["handler_source_git_oid"],
                             self.grant.handler_source_git_oid)
            self.assertEqual(
                json.dumps(payload, allow_nan=False, sort_keys=True,
                           separators=(",", ":")),
                event.canonical_payload_json,
            )
            self.assertEqual(
                sha256(event.canonical_payload_json.encode("utf-8")).hexdigest(),
                event.cursor.event_sha256,
            )
            predecessor = event.cursor.event_sha256
        self.assertEqual(predecessor, self.witness.tip.event_sha256)
        self.assertEqual(revoked.cursor, self.witness.tip)

        tampered = json.loads(events[1].canonical_payload_json)
        tampered["grant"]["handler_artifact_sha256"] = "f" * 64
        tampered_json = json.dumps(tampered, allow_nan=False, sort_keys=True,
                                  separators=(",", ":"))
        self.assertNotEqual(
            sha256(tampered_json.encode("utf-8")).hexdigest(),
            events[1].cursor.event_sha256,
        )
        predecessor_tampered = json.loads(events[2].canonical_payload_json)
        predecessor_tampered["previous_sha256"] = "0" * 64
        self.assertNotEqual(
            sha256(json.dumps(predecessor_tampered, allow_nan=False,
                              sort_keys=True, separators=(",", ":"))
                   .encode("utf-8")).hexdigest(),
            events[2].cursor.event_sha256,
        )

    def test_competing_claims_one_wins_with_compare_and_append(self):
        issued = self.issue()

        def compete(invocation):
            try:
                return self.witness.reserve_claim(
                    issued.cursor, self.grant, invocation, at_epoch=101
                )
            except WitnessDenied:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(compete, (
                "invoke-A", "invoke-B", "invoke-C", "invoke-D", "invoke-E", "invoke-F"
            )))
        winners = [result for result in results if result is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual([event.kind for event in self.witness.events],
                         ["issue", "claim"])

    def test_reset_or_cloning_live_object_disallowed_but_new_process_unsolved(self):
        issued = self.issue()
        claimed = self.witness.reserve_claim(
            issued.cursor, self.grant, "invoke-001", at_epoch=101
        )
        self.assertFalse(hasattr(self.witness, "reset"))
        self.assertFalse(hasattr(self.witness, "restore"))
        with self.assertRaisesRegex(WitnessDenied, "cloned"):
            copy.copy(self.witness)
        with self.assertRaisesRegex(WitnessDenied, "cloned"):
            copy.deepcopy(self.witness)
        with self.assertRaisesRegex(WitnessDenied, "serialized"):
            pickle.dumps(self.witness)
        with self.assertRaisesRegex(WitnessDenied, "stale"):
            self.witness.issue(issued.cursor, make_grant(grant_id="new"), at_epoch=102)
        self.assertEqual(self.witness.tip, claimed.cursor)

        # Important admitted flaw: a new process can instantiate another reference
        # with identical genesis. It does NOT establish independent custody.
        restarted = InProcessAuthorizationWitness("host-alpha", 7)
        self.assertEqual(restarted.tip.sequence, 0)
        self.assertNotEqual(restarted.tip, claimed.cursor)
        restarted.issue(restarted.tip, self.grant, at_epoch=100)
        restarted.reserve_claim(
            restarted.tip, self.grant, "invoke-001", at_epoch=101
        )
        self.assertEqual(restarted.tip, claimed.cursor)


if __name__ == "__main__":
    unittest.main()
