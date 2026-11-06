"""Empirical negative controls for the source-only zero-weight fixture."""

import importlib.util
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unittest

EXPERIMENT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("zero_weight_pilot", EXPERIMENT / "run.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class ZeroWeightPilotTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db = self.root / "state.sqlite3"
        self.a = self.root / "A.json"
        self.b = self.root / "B.json"
        pilot.model_artifact(self.a, node="node-A", style="annotated")
        pilot.model_artifact(self.b, node="node-B", style="terse")
        self.a_sha = pilot.canonical_file(self.a)
        self.b_sha = pilot.canonical_file(self.b)
        self.now = int(time.time())
        self.head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pilot.REPO, text=True).strip()
        self.task = {"source": "v=(7 + 11)*2"}

    def grant(self, *, session="study", ref=pilot.CAP_REF,
              valid_until=None, revoke_after_validate=False):
        return pilot.grant(
            pilot.StateLedger(self.db), self.a, source_head=self.head,
            session=session, recipient="shared-runtime",
            ref=ref, valid_until=valid_until or self.now + 300,
            now=self.now, revoke_after_validate=revoke_after_validate,
        )

    def receive(self, *, artifact=None, session="study", target="shared-runtime",
                task=None):
        return pilot.receiver_subprocess(
            self.db, artifact or self.b, task or self.task,
            session, target=target,
        )

    def test_no_state_baseline_cannot_use_tool(self):
        self.assertEqual(self.receive()["reason"], "missing-capability")
        self.assertEqual(self.receive(artifact=self.a)["reason"], "missing-capability")

    def test_propose_validate_commit_changes_capability_not_weights(self):
        item = self.grant()
        ledger = pilot.StateLedger(self.db)
        events = ledger.events("shared-runtime", "study")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].delta.content_hash, item.content_hash)
        self.assertEqual(ledger.get_revision("shared-runtime", "study", pilot.STATE_KEY), 1)
        self.assertTrue(ledger.verify_chain())
        self.assertEqual(self.b_sha, pilot.canonical_file(self.b))
        self.assertEqual(self.a_sha, pilot.canonical_file(self.a))

    def test_fresh_receiver_only_process_host_verified(self):
        self.grant()
        output = self.receive()
        self.assertTrue(output["ok"])
        self.assertEqual(output["artifact"], "v = (7 + 11) * 2")
        verdict = pilot.host_verify(self.task, output, self.b_sha)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["mock_adapter_artifact_sha256"], pilot.canonical_file(self.b))

    def test_heterogeneous_styles_read_exact_same_governed_namespace(self):
        self.grant()
        a, b = self.receive(artifact=self.a), self.receive(artifact=self.b)
        self.assertTrue(a["ok"] and b["ok"])
        self.assertEqual(a["artifact"], b["artifact"])
        self.assertNotEqual(a["display"], b["display"])
        self.assertEqual({a["style"], b["style"]}, {"annotated", "terse"})
        self.assertEqual(len(pilot.StateLedger(self.db).events("shared-runtime", "study")), 1)

    def test_task_blind_map_does_not_contain_input_or_answer(self):
        self.grant()
        projection = pilot.StateLedger(self.db).project("shared-runtime", "study")
        encoded = json.dumps(projection)
        self.assertEqual(set(projection[pilot.STATE_KEY]),
                         {"ref", "policy", "valid_until", "handler_sha256",
                          "donor_artifact_sha256"})
        self.assertNotIn(self.task["source"], encoded)
        self.assertNotIn("v = (7 + 11) * 2", encoded)

    def test_unseen_tasks_chosen_after_commit(self):
        self.grant()
        rng = random.Random(913)
        for _ in range(8):
            x, y = rng.randrange(20), rng.randrange(20)
            task = {"source": f"score=({x}+{y})*2"}
            result = self.receive(task=task)
            self.assertTrue(pilot.host_verify(task, result, self.b_sha)["passed"])

    def test_decoy_capability_ref_cannot_invoke_tool(self):
        with self.assertRaises(ValueError):
            self.grant(ref="cap-decoy-unregistered")
        self.assertEqual(self.receive()["reason"], "missing-capability")

    def test_a_absent_cannot_propose_and_b_stays_baseline(self):
        self.a.unlink()
        with self.assertRaises(FileNotFoundError):
            self.grant()
        self.assertEqual(self.receive()["reason"], "missing-capability")

    def test_forged_a_identity_rejected_before_proposal(self):
        pilot.model_artifact(self.a, node="node-X", style="annotated")
        with self.assertRaises(ValueError):
            self.grant()
        self.assertEqual(self.receive()["reason"], "missing-capability")

    def test_forged_or_unknown_source_git_oid_rejected(self):
        with self.assertRaises(ValueError):
            pilot.grant(pilot.StateLedger(self.db), self.a,
                        source_head="f" * 40, session="study", recipient="shared-runtime",
                        valid_until=self.now + 300, now=self.now)
        self.assertEqual(self.receive()["reason"], "missing-capability")

    def test_post_grant_handler_rebinding_denied_before_invocation(self):
        self.grant()
        calls = []
        original = pilot.HOST_CAPABILITIES[pilot.CAP_REF]
        try:
            pilot.HOST_CAPABILITIES[pilot.CAP_REF] = lambda source: calls.append(source) or "MALICIOUS"
            result = pilot.execute(self.db, self.b, self.task, session="study")
            self.assertEqual(result["reason"], "handler-mismatch")
            self.assertEqual(calls, [])
        finally:
            pilot.HOST_CAPABILITIES[pilot.CAP_REF] = original

    def test_stale_state_rejected_after_fixture_clock_advance(self):
        # A one-second TTL based on setUp's earlier timestamp races with CI
        # process startup and source-head verification before PROPOSE. Refresh
        # the clock immediately before grant and leave a bounded 4s window.
        self.now = int(time.time())
        deadline = self.now + 4
        self.grant(valid_until=deadline)
        self.assertTrue(pilot.execute(self.db, self.b, self.task, session="study", now=self.now)["ok"])
        # Public receiver subprocess only uses host wall time; it cannot
        # accept an untrusted CLI clock override to resurrect expired state.
        # Wait against REAL time; never patch the runtime's clock for this probe.
        time.sleep(max(0.0, deadline - time.time()) + 0.15)
        self.assertEqual(self.receive()["reason"], "stale-capability")

    def test_expiry_backdating_cli_option_is_not_accepted(self):
        self.grant()
        attack = subprocess.run(
            [sys.executable, str(EXPERIMENT / "run.py"), "receive",
             "--db", str(self.db), "--artifact", str(self.b),
             "--session", "study",
             "--now", str(self.now - 3600)],
            input=json.dumps(self.task), text=True, capture_output=True, timeout=5,
        )
        self.assertNotEqual(attack.returncode, 0)
        self.assertIn("unrecognized arguments", attack.stderr)

    def test_malformed_and_oversized_source_fail_closed_without_crash(self):
        self.grant()
        for source in ("def", "x=(", "x=1\n" * 400, "x=" + "(" * 300 + "2" + ")" * 300):
            with self.subTest(bytes=len(source)):
                result = self.receive(task={"source": source})
                self.assertEqual((result["ok"], result["reason"]), (False, "invalid-task"))

    def test_revocation_after_validation_prevents_commit(self):
        with self.assertRaises(PermissionError):
            self.grant(revoke_after_validate=True)
        self.assertEqual(pilot.StateLedger(self.db).project("shared-runtime", "study"), {})
        self.assertEqual(self.receive()["reason"], "missing-capability")

    def test_revocation_after_commit_denies_fresh_receiver_use(self):
        self.grant()
        self.assertTrue(self.receive()["ok"])
        pilot.revoke_use_grant(self.db, session="study", target="shared-runtime")
        self.assertEqual(self.receive()["reason"], "no-active-use-grant")

    def test_replay_cannot_reenable_revoked_use_grant(self):
        self.grant()
        pilot.revoke_use_grant(self.db, session="study", target="shared-runtime")
        self.grant()  # idempotent bus retry, not new host authority
        self.assertEqual(self.receive()["reason"], "no-active-use-grant")

    def test_missing_host_use_grant_is_fail_closed(self):
        self.grant()
        pilot.use_grant_path(self.db, session="study", target="shared-runtime").unlink()
        self.assertEqual(self.receive()["reason"], "no-active-use-grant")

    def test_session_and_target_isolation(self):
        self.grant(session="session-a")
        self.assertTrue(self.receive(session="session-a")["ok"])
        self.assertEqual(self.receive(session="session-b")["reason"], "missing-capability")
        self.assertEqual(self.receive(session="session-a", target="other-runtime")["reason"],
                         "missing-capability")

    def test_replayed_identical_event_has_one_effect(self):
        self.grant()
        self.grant()
        events = pilot.StateLedger(self.db).events("shared-runtime", "study")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].key_revision, 1)

    def test_external_host_checker_rejects_forged_model_claim(self):
        self.grant()
        genuine = self.receive()
        forged = {**genuine, "artifact": "v = 999", "verified": True}
        self.assertFalse(pilot.host_verify(self.task, forged, self.b_sha)["passed"])
        proc = subprocess.run(
            [sys.executable, str(EXPERIMENT / "run.py"), "verify",
             "--artifact", str(self.b)],
            input=json.dumps({"task": self.task, "answer": forged}),
            text=True, capture_output=True, check=True, timeout=5,
        )
        self.assertFalse(json.loads(proc.stdout)["passed"])

    def test_receiver_restart_stability_and_receipt_binding(self):
        self.grant()
        first = self.receive()
        second = self.receive()
        self.assertEqual(first, second)
        self.assertEqual(self.b_sha, pilot.canonical_file(self.b))
        verdict = pilot.host_verify(self.task, second, self.b_sha)
        self.assertEqual(verdict["input_sha256"], hashlib.sha256(
            pilot.canonical_json(self.task).encode()).hexdigest())
        self.assertEqual(verdict["source_tree_sha256"], pilot.source_manifest_sha256())

    def test_no_model_weights_or_injection_in_state(self):
        self.grant()
        delta = pilot.StateLedger(self.db).events("shared-runtime", "study")[0].delta
        raw = json.dumps(delta.payload)
        self.assertNotIn("weight", raw.lower())
        self.assertNotIn("source", raw.lower())
        self.assertNotIn("answer", raw.lower())
        self.assertEqual(delta.evidence.source_git_commit_oid, "sha1:" + self.head)
        self.assertEqual(delta.evidence.artifact_sha256, self.a_sha)


if __name__ == "__main__":
    unittest.main()
