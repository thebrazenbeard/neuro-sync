"""Pure-stdlib evidence for fact persistence through known distinct channels."""

import unittest

from toy_factual_experiment import (
    FACTS, distill_donor_activations, exact_delta_transplant,
    full_copy, quiz, teach_donor_only, untrained_base,
)


class StructuredFactControls(unittest.TestCase):
    def test_five_arms_and_donor_disconnection(self):
        original = untrained_base()
        a = teach_donor_only(original)
        b_before = untrained_base()

        # Text-only discussion has no persistent model parameter write here.
        ephemeral_chat_context = [f"{cue}: {answer}" for cue, answer in FACTS]
        ephemeral_chat_context.clear()
        b_after_text_reset = b_before

        b_full_copy = full_copy(a)
        b_delta = exact_delta_transplant(b_before, a, original)
        b_interaction = distill_donor_activations(b_before, a, learning_rate=0.75)
        self.assertNotEqual(b_interaction.weights, a.weights)  # not a copy
        del a  # donor disconnected before all B quiz readbacks

        for baseline in (b_before, b_after_text_reset):
            self.assertTrue(all(answer is None for answer in quiz(baseline).values()))
            self.assertEqual(baseline.weights, original.weights)
        for receiver in (b_full_copy, b_delta, b_interaction):
            for cue, expected in FACTS:
                self.assertEqual(receiver.answer(cue), expected)
        self.assertEqual(b_before.weights, original.weights)

    def test_wrong_exact_base_fails_closed(self):
        original = untrained_base()
        a = teach_donor_only(original)
        wrong_receiver = distill_donor_activations(untrained_base(), a)
        with self.assertRaisesRegex(ValueError, "base hash"):
            exact_delta_transplant(wrong_receiver, a, original)

    def test_activations_without_receiver_learning_do_not_persist(self):
        original = untrained_base()
        a = teach_donor_only(original)
        b = untrained_base()
        # B may consult live A's activations, but it never updates itself.
        observed_while_connected = {cue: a.answer(cue) for cue, _ in FACTS}
        self.assertTrue(all(observed_while_connected[cue] == answer for cue, answer in FACTS))
        del a
        self.assertTrue(all(answer is None for answer in quiz(b).values()))

    def test_shuffled_activation_pairing_is_not_correct_transfer(self):
        a = teach_donor_only(untrained_base())
        b = distill_donor_activations(untrained_base(), a, shuffle_teacher=True)
        del a
        self.assertEqual(sum(b.answer(cue) == answer for cue, answer in FACTS), 0)


if __name__ == "__main__":
    unittest.main()
