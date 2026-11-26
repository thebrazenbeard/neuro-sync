"""Offline adversarial controls for multi-association Qwen transfer."""
import math
import unittest
from selective_protocol import generate_suites, assess_scores

class SelectiveProtocolTests(unittest.TestCase):
    def test_deterministic_disjoint_suites(self):
        train, test = generate_suites(109)
        self.assertEqual((train, test), generate_suites(109))
        self.assertNotEqual((train, test), generate_suites(110))
        self.assertEqual(train["protocol"], "selective-associations-v1")
        self.assertEqual(test["protocol"], "selective-associations-v1")
        known = [x for x in train["examples"] if x["kind"] == "known"]
        calibration = [x for x in train["examples"] if x["kind"] == "calibration_unknown"]
        hidden = [x for x in test["examples"] if x["kind"] == "unknown"]
        self.assertGreaterEqual(len(known), 6)
        self.assertGreaterEqual(len(calibration), 2)
        self.assertGreaterEqual(len(hidden), 3)
        self.assertFalse(set(x["id"] for x in hidden) & set(x["id"] for x in train["examples"]))
    def test_training_never_contains_heldout_wording_or_hidden_ids(self):
        train, test = generate_suites(109)
        prompts = [x["prompt"] for x in train["examples"]]
        for example in test["examples"]:
            self.assertNotIn(example["prompt"], prompts)
            if example["kind"] == "unknown":
                self.assertNotIn(example["id"], str(train))
        self.assertEqual(set(train["candidates"]), set(test["candidates"]))

    def _rows(self, suite, strategy):
        out = []
        for item in suite["examples"]:
            selected = strategy(item, suite)
            scores = {key: -10.0 for key in suite["candidates"]}
            scores[selected] = 0.0
            out.append({"id": item["id"], "variant": item["variant"], "scores": scores})
        return out
    def test_perfect_selective_vs_constant_code_bias(self):
        _, suite = generate_suites(109)
        good = self._rows(suite, lambda e, s: e["answer"])
        biased = self._rows(suite, lambda e, s: s["candidates"][0])
        p = assess_scores(suite, good)
        b = assess_scores(suite, biased)
        self.assertEqual(p["exact_accuracy"], 1.0)
        self.assertEqual(p["paraphrase_accuracy"], 1.0)
        self.assertEqual(p["unknown_rejection"], 1.0)
        self.assertEqual(b["unknown_rejection"], 0.0)
        self.assertLess(b["paraphrase_accuracy"], 0.5)
        self.assertFalse(b["qualified"])

    def test_invalid_or_unmatched_rows_rejected(self):
        _, suite = generate_suites(109)
        valid = self._rows(suite, lambda e, s: e["answer"])
        for corrupt in ("missing", "extra", "nonfinite", "bool", "tie"):
            with self.subTest(corrupt=corrupt):
                rows = [{"id": x["id"], "variant": x["variant"], "scores": dict(x["scores"])} for x in valid]
                if corrupt == "missing": rows.pop()
                if corrupt == "extra": rows.append(dict(rows[-1]))
                if corrupt == "nonfinite": rows[0]["scores"][suite["candidates"][0]] = math.inf
                if corrupt == "bool": rows[0]["scores"][suite["candidates"][0]] = True
                if corrupt == "tie":
                    rows[0]["scores"] = {x: 0.0 for x in suite["candidates"]}
                with self.assertRaises(ValueError):
                    assess_scores(suite, rows)
    def test_no_hidden_label_leaks_and_calibration_only(self):
        train, test = generate_suites(109)
        self.assertEqual({x["answer"] for x in train["examples"] if x["kind"] == "calibration_unknown"}, {" UNKNOWN"})
        self.assertTrue(all(x["variant"] == "heldout_unknown" for x in test["examples"] if x["kind"] == "unknown"))
        self.assertTrue(all(x["answer"] == " UNKNOWN" for x in test["examples"] if x["kind"] == "unknown"))
        self.assertTrue(all(x["variant"] in {"exact","paraphrase"} for x in test["examples"] if x["kind"] == "known"))

    def test_compared_gate_needs_improvement_not_only_plausible_outputs(self):
        from selective_protocol import compare_models
        _, suite = generate_suites(109)
        perfect = self._rows(suite, lambda e, s: e["answer"])
        biased = self._rows(suite, lambda e, s: s["candidates"][0])
        qualified = compare_models(suite, biased, perfect)
        self.assertTrue(qualified["screen_pass"])
        self.assertFalse(qualified["generalization_proven"])
        self.assertFalse(compare_models(suite, perfect, perfect)["screen_pass"])
        self.assertFalse(compare_models(suite, biased, biased)["screen_pass"])

    def test_compared_gate_cannot_accept_unrejected_unknowns(self):
        from selective_protocol import compare_models
        _, suite = generate_suites(109)
        biased = self._rows(suite, lambda e, s: s["candidates"][0])
        known_only = self._rows(suite, lambda e, s: e["answer"]
                                if e["kind"] == "known" else s["candidates"][0])
        outcome = compare_models(suite, biased, known_only)
        self.assertFalse(outcome["screen_pass"])
        self.assertEqual(outcome["adapted"]["unknown_rejection"], 0)

    def test_shuffled_label_training_preserves_frequency_but_breaks_association(self):
        from selective_protocol import shuffled_training
        train, _ = generate_suites(109)
        altered = shuffled_training(train)
        known = {e["id"]: e["answer"] for e in train["examples"] if e["kind"] == "known"}
        shifted = {e["id"]: e["answer"] for e in altered["examples"] if e["kind"] == "known"}
        self.assertEqual(set(known), set(shifted))
        self.assertEqual(sorted(known.values()), sorted(shifted.values()))
        self.assertTrue(all(known[key] != shifted[key] for key in known))
        self.assertEqual(
            [e for e in altered["examples"] if e["kind"] == "calibration_unknown"],
            [e for e in train["examples"] if e["kind"] == "calibration_unknown"])
        self.assertEqual(altered, shuffled_training(train))
        self.assertEqual([e["answer"] for e in train["examples"] if e["kind"] == "known"],
                         list(known.values()))

if __name__ == "__main__":
    unittest.main()
