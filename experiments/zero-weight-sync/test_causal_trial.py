"""Causal falsification controls for *scripted* zero-weight StateBus pilot."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("causal_trial", HERE / "causal_trial.py")
trial = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = trial
spec.loader.exec_module(trial)


class CausalTrialTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.a, self.b = self.root / "A.json", self.root / "B.json"
        trial.pilot.model_artifact(self.a, node="node-A", style="annotated")
        trial.pilot.model_artifact(self.b, node="node-B", style="terse")
        self.b_sha = trial.pilot.canonical_file(self.b)
        self.scenario = trial.Scenario.generate(2416)
        self.registry = self.root / "registry.json"
        self.broker = self.root / "broker.json"
        self.db = self.root / "ledger.sqlite3"
        self.state_path = self.root / "state.json"
        trial.write_json(self.registry, self.scenario.registry_wire())
        self.now = int(time.time())
        self.until = self.now + 300
        self.session = "causal"
        self.target = "shared-runtime"

    def observe(self):
        return trial.a_discover(self.a, self.scenario.donor_view())

    def commit(self, *, observation=None):
        observation = self.observe() if observation is None else observation
        value = trial.grant_via_bus(
            self.db, self.broker, self.scenario, self.a,
            trial.pilot.trusted_source_head(None), self.session, self.target,
            observation=observation, now=self.now, valid_until=self.until)
        trial.write_json(self.state_path, value)
        return value

    def receive(self, transport, x=1031, *, fresh=False, **kwargs):
        args = dict(transport=transport, registry_path=self.registry,
                    broker_path=self.broker, receiver_artifact=self.b,
                    session=self.session, target=self.target, x=x,
                    db_path=self.db, state_path=self.state_path)
        args.update(kwargs)
        if fresh:
            args.pop("now", None)
            return trial.b_fresh_process(**args)
        return trial.receive(**args)

    def test_a_actively_discovers_randomized_hidden_reference(self):
        selected = []
        for seed in (17, 2416, 99, 421):
            scenario = trial.Scenario.generate(seed)
            obs = trial.a_discover(self.a, scenario.donor_view())
            self.assertEqual(obs["selected_ref"], scenario.target_ref)
            self.assertTrue(trial.host_validate_discovery(scenario, obs, self.a))
            self.assertNotIn("target_ref", scenario.registry_wire())
            selected.append(obs["selected_ref"])
        self.assertEqual(len(set(selected)), 4)

    def test_a_is_passed_restricted_view_without_sealed_target_label(self):
        view = self.scenario.donor_view()
        self.assertEqual(set(vars(view)), {"candidates", "signature"})
        self.assertFalse(hasattr(view, "target_ref"))
        with self.assertRaises(TypeError):
            trial.a_discover(self.a, self.scenario)
        self.assertEqual(trial.a_discover(self.a, view)["selected_ref"],
                         self.scenario.target_ref)

    def test_no_a_observation_no_bus_commit(self):
        with self.assertRaises(PermissionError):
            trial.grant_via_bus(self.db, self.broker, self.scenario, self.a,
                                trial.pilot.trusted_source_head(None), self.session,
                                self.target, observation=None, now=self.now,
                                valid_until=self.until)
        self.assertEqual(self.receive("statebus")["reason"], "missing-or-invalid-state")

    def test_bad_a_identity_and_forged_probe_cannot_commit(self):
        trial.pilot.model_artifact(self.a, node="node-impostor", style="annotated")
        with self.assertRaises(ValueError):
            self.observe()
        trial.pilot.model_artifact(self.a, node="node-A", style="annotated")
        fake = {**self.observe(), "probe_outputs": [0, 0, 0]}
        self.assertFalse(trial.host_validate_discovery(self.scenario, fake, self.a))
        with self.assertRaises(PermissionError):
            self.commit(observation=fake)
        self.assertEqual(self.receive("statebus")["reason"], "missing-or-invalid-state")

    def test_bad_reference_and_mismatched_catalog_are_rejected(self):
        obs = {**self.observe(), "selected_ref": list(self.scenario.registry)[0]}
        if obs["selected_ref"] == self.scenario.target_ref:
            obs["selected_ref"] = list(self.scenario.registry)[1]
        self.assertFalse(trial.host_validate_discovery(self.scenario, obs, self.a))
        obs = {**self.observe(), "registry_sha256": "0" * 64}
        self.assertFalse(trial.host_validate_discovery(self.scenario, obs, self.a))

    def test_probe_receipt_bool_false_is_not_numeric_zero(self):
        scenario = trial.Scenario.generate(1850)
        original = trial.a_discover(self.a, scenario.donor_view())
        self.assertEqual(original["probe_outputs"][0], 0)
        forged = {**original, "probe_outputs": [False] + original["probe_outputs"][1:]}
        self.assertFalse(trial.host_validate_discovery(scenario, forged, self.a))

    def test_causal_no_state_and_valid_state_separated(self):
        no_state = self.receive("statebus", fresh=True)
        self.assertFalse(no_state["ok"])
        self.commit()
        correct = self.receive("statebus", fresh=True)
        self.assertTrue(correct["ok"])
        self.assertEqual(correct["value"], self.scenario.evaluate(self.scenario.target_ref, 1031))
        self.assertEqual(self.b_sha, trial.pilot.canonical_file(self.b))

    def test_task_blind_bus_and_b_independent_heldout_probes(self):
        state = self.commit()
        heldout = self.scenario.heldout(10)
        self.assertTrue(set(heldout).isdisjoint(trial.PROBES))
        raw = trial.canonical_json(state)
        self.assertNotIn("probe_outputs", raw)
        self.assertNotIn("target_ref", raw)
        self.assertNotIn("answer", raw.lower())
        self.assertNotIn("target_ref", trial.canonical_json(
            trial.bounded_json(self.registry)))
        outputs = [self.receive("statebus", x=x, fresh=True) for x in heldout]
        self.assertEqual(trial.host_score(self.scenario, self.scenario.target_ref,
                                          heldout, outputs), len(heldout))
        self.assertEqual(self.b_sha, trial.pilot.canonical_file(self.b))

    def test_plain_file_matches_statebus_in_functionality(self):
        self.commit()
        xs = self.scenario.heldout(6)
        bus = [self.receive("statebus", x=x, fresh=True) for x in xs]
        plain = [self.receive("file", x=x, fresh=True) for x in xs]
        self.assertEqual([x["value"] for x in bus], [x["value"] for x in plain])
        self.assertEqual(trial.host_score(self.scenario, self.scenario.target_ref,
                                          xs, bus), len(xs))
        self.assertEqual(trial.host_score(self.scenario, self.scenario.target_ref,
                                          xs, plain), len(xs))

    def test_operator_direct_reference_works_without_a(self):
        """Confounder is exposed: an operator can supply same mapping directly."""
        chosen = {"selected_ref": self.scenario.target_ref,
                  "registry_sha256": trial.digest(self.scenario.registry_wire()),
                  "donor_artifact_sha256": trial.pilot.canonical_file(self.a)}
        state = trial.capability_value(chosen, self.until)
        trial.write_json(self.state_path, state)
        trial.write_json(self.broker, {"active": True, "session": self.session,
                                           "target": self.target, "ref": state["ref"],
                                           "catalog_sha256": state["catalog_sha256"],
                                           "donor_artifact_sha256": state["donor_artifact_sha256"]})
        self.assertEqual(trial.StateLedger(self.db).project(self.target, self.session), {})
        direct = self.receive("file", fresh=True)
        self.assertTrue(direct["ok"])
        self.assertEqual(direct["value"], self.scenario.evaluate(self.scenario.target_ref, 1031))

    def test_decoy_ref_and_spoofed_state_denied_at_use(self):
        valid = self.commit()
        decoy = list(self.scenario.registry)[0]
        if decoy == valid["ref"]:
            decoy = list(self.scenario.registry)[1]
        trial.write_json(self.state_path, {**valid, "ref": decoy})
        self.assertEqual(self.receive("file")["reason"], "no-active-grant")
        trial.write_json(self.state_path, {**valid, "unexpected_answer": 999})
        self.assertEqual(self.receive("file")["reason"], "missing-or-invalid-state")
        trial.write_json(self.state_path, {**valid, "catalog_sha256": "f" * 64})
        self.assertEqual(self.receive("file")["reason"], "registry-mismatch")

    def test_revocation_after_commit_denies_each_transport_on_fresh_b(self):
        self.commit()
        self.assertTrue(self.receive("statebus", fresh=True)["ok"])
        trial.revoke(self.broker)
        self.assertEqual(self.receive("statebus", fresh=True)["reason"], "no-active-grant")
        self.assertEqual(self.receive("file", fresh=True)["reason"], "no-active-grant")

    def test_missing_broker_denies_committed_reference(self):
        self.commit()
        self.broker.unlink()
        self.assertEqual(self.receive("statebus", fresh=True)["reason"], "no-active-grant")

    def test_stale_state_and_wrong_policy_fail(self):
        chosen = self.commit()
        self.assertEqual(self.receive("statebus", now=self.until + 1)["reason"],
                         "expired-or-wrong-policy")
        trial.write_json(self.state_path, {**chosen, "policy": "different"})
        self.assertEqual(self.receive("file")["reason"], "expired-or-wrong-policy")

    def test_cross_session_and_target_isolation(self):
        self.commit()
        self.assertEqual(self.receive("statebus", session="other")["reason"],
                         "missing-or-invalid-state")
        self.assertEqual(self.receive("statebus", target="other")["reason"],
                         "missing-or-invalid-state")

    def test_forged_registry_and_missing_handler_fail(self):
        self.commit()
        changed = trial.bounded_json(self.registry)
        changed[self.scenario.target_ref][0] += 1
        trial.write_json(self.registry, changed)
        self.assertEqual(self.receive("statebus")["reason"], "registry-mismatch")
        self.registry.unlink()
        self.assertEqual(self.receive("statebus")["reason"], "invalid-registry")

    def test_malformed_and_duplicate_json_rejected_before_tool_use(self):
        self.commit()
        self.state_path.write_text('{"ref":"x","ref":"y"}', encoding="utf8")
        self.assertEqual(self.receive("file")["reason"], "invalid-state-file")
        self.state_path.write_text('{not-json', encoding="utf8")
        self.assertEqual(self.receive("file")["reason"], "invalid-state-file")
        self.broker.write_text('{"active":true,"active":true}', encoding="utf8")
        self.assertEqual(self.receive("statebus")["reason"], "no-active-grant")

    def test_malformed_user_input_rejected_by_fresh_process(self):
        self.commit()
        cmd = [sys.executable, str(HERE / "causal_trial.py"), "receive",
               "--transport", "statebus", "--registry", str(self.registry),
               "--broker", str(self.broker), "--receiver", str(self.b),
               "--session", self.session, "--db", str(self.db)]
        for body in ({"x": "1031"}, {"x": True}, {"x": 999999},
                     {"source": "answer=42"}):
            result = subprocess.run(cmd, input=json.dumps(body), text=True,
                                    capture_output=True, timeout=5, check=True)
            self.assertFalse(json.loads(result.stdout)["ok"])

    def test_scoring_rejects_wrong_result_even_with_verified_flag(self):
        xs = self.scenario.heldout(2)
        forged = [{"ok": True, "value": 999999, "verified": True} for _ in xs]
        self.assertEqual(trial.host_score(self.scenario, self.scenario.target_ref,
                                          xs, forged), 0)
        self.assertEqual(trial.host_score(self.scenario, self.scenario.target_ref,
                                          xs, forged[:1]), 0)
        class OneOracle:
            def evaluate(self, _ref, _x):
                return 1
        self.assertEqual(trial.host_score(OneOracle(), "opaque", [7],
                                          [{"ok": True, "value": True}]), 0)

    def test_comparison_demo_discloses_equal_file_baseline(self):
        output = trial.demo(317)
        self.assertEqual(output["no_state_score"], 0)
        self.assertEqual(output["statebus_score"], 8)
        self.assertEqual(output["plain_file_score"], 8)
        self.assertEqual(output["revoked_score"], 0)
        self.assertTrue(output["unchanged_B_artifact"])
        self.assertTrue(output["task_blind_bus"])

    def test_task_blindness_is_structural_not_hash_substring(self):
        # Seed 91 previously produced false negative because integer heldout
        # happened to be a decimal substring of a SHA/ref in the state.
        self.assertTrue(trial.demo(91)["task_blind_bus"])


if __name__ == "__main__":
    unittest.main()
