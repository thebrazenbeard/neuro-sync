"""Check the study's externally visible result and selected real mutations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


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
