"""Check the study's externally visible result and selected real mutations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("governance_differential_study", Path(__file__).with_name("study.py"))
study = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(study)


class GovernanceDifferentialTests(unittest.TestCase):
    def test_broker_policy_matches_for_all_three_state_sources(self):
        evidence = study.run()
        self.assertTrue(evidence["broker_policy_parity"])
        self.assertEqual(evidence["broker_controls"]["statebus"], evidence["broker_controls"]["file"])
        self.assertEqual(evidence["broker_controls"]["statebus"], evidence["broker_controls"]["governed_file"])
        self.assertEqual(evidence["broker_controls"]["statebus"]["handler_effect_count"], 1)

    def test_scope_probe_uses_a_fresh_grant_and_owner_can_still_use_it(self):
        """Probe scope before any invocation; ordinary grants permit reuse."""
        for arm in ("statebus", "file", "governed_file"):
            from tempfile import TemporaryDirectory
            with self.subTest(arm=arm), TemporaryDirectory() as temp:
                result = study._broker_controls(Path(temp), arm)
                self.assertTrue(result["owner_succeeded_after_foreign_probe"])

    def test_mutated_broker_ignoring_principal_is_detected_by_scope_probe(self):
        """Negative control: an intentionally unscoped broker must fail.

        Test-only fault injection makes the foreign caller use the owner's
        principal scope. Legacy grants are multi-use; only invocation IDs are
        single-use. Do not misidentify grant use as replay exhaustion.
        """
        from tempfile import TemporaryDirectory

        real_execute = study.HostOwnedEffectBroker.execute
        foreign_bypass_attempts = []

        def execute_without_principal_isolation(broker, grant_id, invocation_id, key):
            if broker._context.principal_id == "other-principal":
                foreign_bypass_attempts.append(invocation_id)
                original_context = broker._context
                broker._context = study.HostBoundContext(study.PRINCIPAL,
                                                         study.TARGET,
                                                         study.SESSION)
                try:
                    return real_execute(broker, grant_id, invocation_id, key)
                finally:
                    broker._context = original_context
            return real_execute(broker, grant_id, invocation_id, key)

        with TemporaryDirectory() as temp:
            with patch.object(study.HostOwnedEffectBroker, "execute",
                              execute_without_principal_isolation):
                # Under a broken principal check both calls can succeed;
                # the study's explicit scope check must expose that defect.
                result = study._broker_controls(Path(temp), "file")
        self.assertEqual(foreign_bypass_attempts, ["foreign"])
        self.assertFalse(result["foreign_context_blocked"])
        self.assertEqual(result["handler_effect_count"], 2)

        with TemporaryDirectory() as temp:
            with patch.object(study.HostOwnedEffectBroker, "execute",
                              execute_without_principal_isolation):
                with self.assertRaisesRegex(AssertionError, "broker control regression"):
                    study.run()

    def test_stateledger_blocks_unapproved_mutation_but_uncontrolled_file_does_not(self):
        evidence = study.run()["data_mutation_interventions"]["unapproved"]
        self.assertEqual(evidence["statebus"], {
            "write_blocked": True, "baseline_state_preserved": True,
            "replacement_visible": False,
        })
        self.assertEqual(evidence["file"]["replacement_visible"], True)
        self.assertEqual(evidence["governed_file"], evidence["statebus"])

    def test_stateledger_revision_cas_rejects_stale_writer(self):
        evidence = study.run()["data_mutation_interventions"]["stale"]
        self.assertTrue(evidence["statebus"]["write_blocked"])
        self.assertTrue(evidence["statebus"]["baseline_state_preserved"])
        self.assertTrue(evidence["file"]["replacement_visible"])
        self.assertEqual(evidence["governed_file"], evidence["statebus"])

    def test_authorized_nonstale_update_succeeds_under_all_arms(self):
        evidence = study.run()["data_mutation_interventions"]["approved"]
        for arm in ("statebus", "file", "governed_file"):
            self.assertEqual(evidence[arm], {
                "write_blocked": False, "baseline_state_preserved": False,
                "replacement_visible": True,
            })

    def test_plain_file_scope_does_not_return_foreign_session(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            plain = study.PlainFileProjection(Path(temp) / "snapshot.json")
            plain.set("receiver", "trial", study.INITIAL)
            self.assertEqual(plain.project("receiver", "wrong-session"), {})
            self.assertEqual(plain.project("wrong-receiver", "trial"), {})
            self.assertEqual(plain.project("receiver", "trial"), {study.KEY: study.INITIAL})

    def test_governed_json_same_uid_raw_write_bypasses_wrapper(self):
        """This toy wrapper is not an OS-level access boundary."""
        import json
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            path = Path(temp) / "governed.json"
            guarded = study.GovernedFileProjection(path)
            self.assertEqual(guarded.project(study.TARGET, study.SESSION)[study.KEY], study.INITIAL)
            path.write_text(json.dumps({
                "target": study.TARGET, "session": study.SESSION,
                "state": {study.KEY: study.UNAPPROVED},
            }), encoding="utf-8")
            self.assertEqual(guarded.project(study.TARGET, study.SESSION)[study.KEY], study.UNAPPROVED)


if __name__ == "__main__":
    unittest.main()
