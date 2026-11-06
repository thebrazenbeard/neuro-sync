"""Structured fact quiz: independent B after copy, delta, or activation tutoring.

All names/facts are deliberately synthetic Starfleet-*style* fiction, not
canonical Star Trek facts. This is a tiny associative matrix with a fixed
vocabulary, not a language model. It tests persistence mechanics, not general
reasoning, prompt paraphrases, or a new weight-space training breakthrough.
"""

from __future__ import annotations

from dataclasses import dataclass

from adapter_transplant import (
    SCHEMA, Matrix, PermissionGrant, TransplantManifest,
    apply_delta, matrix_hash, normalize_matrix,
)


CUES = (
    ("USS Lantern", "commander"),
    ("Station Aster", "sector"),
    ("Shuttle Peregrine", "destination"),
)
ANSWERS = ("Captain Evi", "Sector Nacre", "Outpost Sable")
FACTS = tuple(zip(CUES, ANSWERS))
THRESHOLD = 0.5


@dataclass(frozen=True)
class FactQuizModel:
    weights: Matrix

    def scores(self, cue: tuple[str, str]) -> tuple[float, ...]:
        column = CUES.index(cue)
        return tuple(row[column] for row in self.weights)

    def answer(self, cue: tuple[str, str]) -> str | None:
        values = self.scores(cue)
        best = max(values)
        if best <= THRESHOLD or values.count(best) != 1:
            return None
        return ANSWERS[values.index(best)]


def untrained_base() -> FactQuizModel:
    return FactQuizModel(normalize_matrix([[0.0] * len(CUES) for _ in ANSWERS]))


def teach_donor_only(model: FactQuizModel) -> FactQuizModel:
    """Write supervised facts into A, starting from an untouched zero base."""
    updates = [list(row) for row in model.weights]
    for cue, answer in FACTS:
        updates[ANSWERS.index(answer)][CUES.index(cue)] += 1.0
    return FactQuizModel(normalize_matrix(updates))


def full_copy(donor: FactQuizModel) -> FactQuizModel:
    """Obvious positive control: copy every matrix weight."""
    return FactQuizModel(normalize_matrix(donor.weights))


def exact_delta_transplant(
    receiver: FactQuizModel, donor: FactQuizModel, original_base: FactQuizModel
) -> FactQuizModel:
    delta = normalize_matrix([
        [trained - initial for trained, initial in zip(trained_row, initial_row)]
        for trained_row, initial_row in zip(donor.weights, original_base.weights)
    ])
    manifest = TransplantManifest(
        schema=SCHEMA,
        target_id="quiz-receiver-B",
        base_sha256=matrix_hash(original_base.weights),
        delta_sha256=matrix_hash(delta),
        matrix_shape=(len(ANSWERS), len(CUES)),
    )
    applied = apply_delta(
        receiver.weights, delta, manifest,
        PermissionGrant(target_id="quiz-receiver-B", allowed=True),
    )
    return FactQuizModel(applied.after)


def distill_donor_activations(
    receiver: FactQuizModel,
    donor: FactQuizModel,
    *,
    learning_rate: float = 0.75,
    shuffle_teacher: bool = False,
) -> FactQuizModel:
    """Interaction-only arm: B updates from A's scores, not weight copying.

    This is elementary supervised distillation of target activations with a
    B-local update. The donor is never referenced by the returned B model.
    """
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning rate must be in (0, 1]")
    updated = [list(row) for row in receiver.weights]
    for column, cue in enumerate(CUES):
        # Negative control: pair target cue with the next donor cue.
        teacher_cue = CUES[(column + 1) % len(CUES)] if shuffle_teacher else cue
        teacher_scores = donor.scores(teacher_cue)
        for row, teacher_score in enumerate(teacher_scores):
            updated[row][column] += learning_rate * (teacher_score - updated[row][column])
    return FactQuizModel(normalize_matrix(updated))


def quiz(model: FactQuizModel) -> dict[tuple[str, str], str | None]:
    """Blind-to-donor readback; no teacher or source context is an argument."""
    return {cue: model.answer(cue) for cue in CUES}


if __name__ == "__main__":
    original = untrained_base()
    donor_a = teach_donor_only(original)
    receiver_b = untrained_base()
    arms = {
        "B-before": receiver_b,
        "B-text-chat-reset": receiver_b,
        "B-full-copy": full_copy(donor_a),
        "B-exact-delta": exact_delta_transplant(receiver_b, donor_a, original),
        "B-activation-interaction": distill_donor_activations(receiver_b, donor_a),
    }
    del donor_a  # each receiver is read back with A unavailable
    for label, candidate in arms.items():
        answers = quiz(candidate)
        correct = sum(answers[cue] == expected for cue, expected in FACTS)
        print(f"{label}: {correct}/{len(FACTS)}")
