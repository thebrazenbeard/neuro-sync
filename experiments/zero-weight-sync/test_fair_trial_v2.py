"""Hostile mechanics checks: the same B and tools across N/V/P/H mock arms."""

import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fair_trial_v2", HERE / "fair_trial_v2.py")
fair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fair
spec.loader.exec_module(fair)


class FairTrialV2Tests(unittest.TestCase):
    def test_eight_scenarios_two_independent_families(self):
        seeds = (3, 7, 19, 31, 47, 61, 73, 89)
        for family in ("affine", "quadratic"):
            labels = []
            for seed in seeds:
                s = fair.HostScenario.generate(seed, family)
                self.assertEqual(len(s.tools), 13)
                self.assertNotIn("target_ref", vars(s.donor_view()))
                self.assertTrue(set(s.heldout()).isdisjoint(fair.PROBES))
                labels.append(s.target_ref)
            self.assertEqual(len(set(labels)), len(labels))

    def test_donor_does_not_receive_host_sealed_label(self):
        s = fair.HostScenario.generate(19, "quadratic")
        view = s.donor_view()
        self.assertEqual(set(vars(view)), {"family", "candidates", "signature"})
        self.assertFalse(hasattr(view, "target_ref"))
        with self.assertRaises(TypeError):
            fair.donor_discover(s)
        obs = fair.donor_discover(view)
        self.assertEqual(obs["ref"], s.target_ref)
        self.assertTrue(fair.host_validate(s, obs))
        self.assertFalse(fair.host_validate(s, {**obs, "candidate_count": True}))

    def test_public_handles_cannot_determine_run_private_target(self):
        """Same seed/ref inventory but changing host-private entropy changes target."""
        results = [fair.HostScenario.generate(3, "affine", host_secret=bytes([n]) * 32)
                   for n in range(16)]
        handles = [tuple(s.tools) for s in results]
        self.assertEqual(len(set(handles)), 1)
        self.assertGreater(len({s.target_ref for s in results}), 1)
        self.assertTrue(all("host_secret" not in vars(s.donor_view()) for s in results))

    def test_receiver_uninformed_attempts_same_candidate_tool_interface(self):
        s = fair.HostScenario.generate(7, "affine")
        handles = list(s.tools)
        request = {"handles": handles, "tasks": [7, 17, 39], "state": None}
        result = fair.receiver_decide(request)
        self.assertEqual(len(result["choices"]), 3)
        self.assertEqual({x["decision"] for x in result["choices"]}, {"uninformed-guess"})
        self.assertTrue(all(x["ref"] in handles for x in result["choices"]))
        self.assertEqual(fair.host_effects(s, (7, 17, 39), result)["raw_effects"][0]["x"], 7)

    def test_single_receiver_schema_does_not_include_privileged_paths(self):
        s = fair.HostScenario.generate(7, "quadratic")
        request = {"handles": list(s.tools), "tasks": [7], "state": {"ref": s.target_ref}}
        out, digest = fair.fresh_receiver(request)
        self.assertEqual(out["choices"][0]["ref"], s.target_ref)
        self.assertEqual(len(digest), 64)
        self.assertEqual(set(request), {"handles", "tasks", "state"})
        source = inspect.getsource(fair.fresh_receiver)
        self.assertNotIn("--registry", source)
        self.assertNotIn("--broker", source)
        self.assertNotIn("--db", source)
        receiver_text = (HERE / "fair_receiver_v2.py").read_text(encoding="utf8")
        self.assertNotIn("HostScenario", receiver_text)
        self.assertNotIn("host_secret", receiver_text)
        self.assertNotIn("from fair_trial_v2", receiver_text)

    def test_invalid_inputs_and_bool_integer_confusion_fail_closed(self):
        s = fair.HostScenario.generate(7, "affine")
        base = {"handles": list(s.tools), "tasks": [1], "state": None}
        for bad in ({**base, "tasks": [True]}, {**base, "state": {"ref": "wrong"}},
                    {**base, "registry_path": "/tmp/answer"},
                    {**base, "handles": ["duplicate", "duplicate", "third"]}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    fair.receiver_decide(bad)

    def test_file_arm_never_calls_statebus(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plain.json"
            path.write_text('{"ref":"cap-opaque"}', encoding="utf8")
            with mock.patch.object(fair.StateLedger, "project", side_effect=AssertionError("bus called")):
                self.assertEqual(fair.read_plain_file(path), {"ref": "cap-opaque"})
            with self.assertRaises(ValueError):
                path.write_text('{"ref":"ok","answer":42}', encoding="utf8")
                fair.read_plain_file(path)

    def test_H_operator_baseline_is_evaluated_before_scripted_A(self):
        timestamps = []
        original_fresh, original_donor = fair.fresh_receiver, fair.donor_discover
        def track_fresh(req):
            timestamps.append("B")
            return original_fresh(req)
        def track_donor(view):
            timestamps.append("A")
            return original_donor(view)
        with mock.patch.object(fair, "fresh_receiver", track_fresh), \
                mock.patch.object(fair, "donor_discover", track_donor):
            fair.one_scenario(3, "affine")
        self.assertEqual(timestamps[:3], ["B", "B", "A"])

    def test_same_tasks_and_identical_context_in_valid_arms(self):
        for family in ("affine", "quadratic"):
            item = fair.one_scenario(19, family)
            self.assertTrue(item["P_V_H_input_equal"])
            for arm in ("N", "V", "P", "H"):
                self.assertEqual([x["x"] for x in item["arms"][arm]["raw_effects"]], item["tasks"])
            for arm in ("V", "P", "H"):
                self.assertEqual(item["arms"][arm]["effect_score"], len(item["tasks"]))
                self.assertEqual(item["arms"][arm]["selected_ref_score"], len(item["tasks"]))

    def test_raw_scenario_outputs_include_oracle_and_actual_effects(self):
        item = fair.one_scenario(3, "quadratic")
        result = item["arms"]["N"]
        self.assertEqual(len(result["raw_effects"]), 8)
        self.assertEqual(set(result["raw_effects"][0]), {
            "x", "selected_ref", "decision", "effect", "expected",
            "correct_effect", "correct_reference"})
        self.assertTrue(all(type(t["effect"]) is int for t in result["raw_effects"]))

    def test_sealed_oracle_refuses_issuer_shortcut(self):
        s = fair.HostScenario.generate(3, "affine")
        obs = fair.donor_discover(s.donor_view())
        wrong = next(x for x in s.tools if x != s.target_ref)
        self.assertFalse(fair.host_validate(s, {**obs, "ref": wrong}))

    def test_handler_corruption_does_not_corrupt_frozen_host_oracle(self):
        s = fair.HostScenario.generate(3, "quadratic", host_secret=b"I" * 32)
        tasks = s.heldout()
        expected = tuple(s.oracle(x) for x in tasks)
        wrong = next(ref for ref in s.tools if ref != s.target_ref)
        choices = {"choices": [{"x": x, "ref": wrong, "decision": "uninformed-guess"}
                               for x in tasks]}
        with mock.patch.object(fair, "evaluate", return_value=42):
            # Faulty tool now returns 42 for everything. The independent
            # Horner oracle remains frozen to the actual expected values.
            score = fair.host_effects(s, tasks, choices, frozen_expected=expected)
        self.assertEqual(score["effect_score"], 0)
        self.assertEqual(score["selected_ref_score"], 0)
        self.assertTrue(all(x["effect"] == 42 and x["expected"] != 42
                            for x in score["raw_effects"]))

    def test_host_refuses_malformed_receiver_reference_types(self):
        s = fair.HostScenario.generate(3, "affine", host_secret=b"R" * 32)
        tasks = (3,)
        for ref in ([], {}, None, True, "cap-does-not-exist"):
            with self.subTest(ref=ref):
                forged = {"choices": [{"x": 3, "ref": ref,
                                       "decision": "supplied-reference"}]}
                with self.assertRaisesRegex(ValueError, "invalid receiver tool choice"):
                    fair.host_effects(s, tasks, forged)

    def test_exact_source_and_test_policy_pinned(self):
        oid, manifest = fair.source_provenance()
        self.assertTrue(oid.startswith("sha1:"))
        self.assertEqual(len(manifest), 64)
        target = fair.ROOT / fair.SCOPED_SOURCES[0]
        actual_read = Path.read_bytes
        def dirty(path):
            return b"corrupt evaluator" if path == target else actual_read(path)
        with mock.patch.object(Path, "read_bytes", dirty):
            with self.assertRaisesRegex(ValueError, "differs from committed HEAD"):
                fair.source_provenance()

    def test_synthetic_host_secret_commit_reveal_happens_after_all_receivers(self):
        # Six independent seeds, two families, same host-private entropy.
        result = fair.experiment((3, 7, 19, 31, 47, 61),
                                 host_secret=b"Z" * 32)
        self.assertEqual(result["scenario_count"], 12)
        self.assertEqual(result["tasks_per_arm"], 96)
        self.assertEqual(result["aggregate_effect_scores"]["V"], 96)
        self.assertEqual(result["aggregate_effect_scores"]["P"], 96)
        self.assertEqual(result["aggregate_effect_scores"]["H"], 96)
        self.assertEqual(result["host_secret_revealed_after_trial_hex"], (b"Z" * 32).hex())
        self.assertTrue(all(s["P_V_H_input_equal"] for s in result["scenarios"]))


if __name__ == "__main__":
    unittest.main()
