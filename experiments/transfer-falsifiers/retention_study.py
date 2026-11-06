"""Small, honest falsifier: weight copying versus a recipient's prior learning.

Uses the existing categorical SGD learner and its fresh-process evaluator. No
language model or nonlocal model weights are involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1] / "trek-transfer-v0"
sys.path.insert(0, str(SOURCE))
from fixtures import SYNTHETIC_FACTS  # noqa: E402
from learner import TinyLearner, delta_into_exact_base, train, sha256  # noqa: E402
from run import evaluate_in_new_process, one_hot  # noqa: E402


def partitions():
    """Freeze disjoint *queries*, deliberately sharing one subject across roles."""
    donor_facts = tuple(SYNTHETIC_FACTS[i] for i in (0, 2, 3, 5))
    recipient_facts = tuple(SYNTHETIC_FACTS[i] for i in (1, 4))
    donor_keys = {(f.subject, f.relation) for f in donor_facts}
    recipient_keys = {(f.subject, f.relation) for f in recipient_facts}
    if len(donor_facts) != 4 or len(recipient_facts) != 2 or donor_keys & recipient_keys:
        raise AssertionError("invalid disjoint fact partition")
    if not ({f.subject for f in donor_facts} & {f.subject for f in recipient_facts}):
        raise AssertionError("recipient and donor should share a subject to stress interference")
    return donor_facts, recipient_facts


def labeled_targets(model, facts):
    return [one_hot(model.answers.index(f.answer), len(model.answers)) for f in facts]


def trial(*, seed: int = 11, epochs: int = 400, lr: float = 0.15) -> dict:
    """Return measured effects; assert only structural invariants, not a win."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a bounded integer")
    if type(epochs) is not int or not 1 <= epochs <= 2000:
        raise ValueError("epochs must be 1..2000")
    donor_facts, recipient_facts = partitions()
    facts = donor_facts + recipient_facts
    base = TinyLearner(
        [f.subject for f in facts], [f.relation for f in facts],
        [f.answer for f in facts], seed=seed,
    )
    donor = base.clone()
    recipient = base.clone()
    base_sha = base.digest()
    if donor.digest() != recipient.digest() or donor.digest() != base_sha:
        raise AssertionError("initial model weights must be exactly cloned")

    donor_updates = train(donor, donor_facts, labeled_targets(donor, donor_facts),
                          epochs=epochs, lr=lr, seed=77)
    recipient_updates = train(recipient, recipient_facts,
                              labeled_targets(recipient, recipient_facts),
                              epochs=epochs, lr=lr, seed=79)
    donor_sha = donor.digest()
    learned_recipient_sha = recipient.digest()
    if donor_sha == base_sha or learned_recipient_sha == base_sha:
        raise AssertionError("both sides must actually have learned independent data")

    # A real post-clone recipient no longer satisfies the exact-base premise.
    # The existing primitive must refuse to overwrite its independently
    # learned parameters under the label of a harmless delta.
    try:
        delta_into_exact_base(base, donor, recipient)
    except ValueError as exc:
        delta_rejected = True
        delta_error = type(exc).__name__
    else:
        raise AssertionError("exact-base delta improperly accepted learned B")

    teacher_outputs = [donor.distribution(f.subject, f.relation) for f in donor_facts]
    full_copy = donor.clone()
    aligned = recipient.clone()
    wrong = recipient.clone()
    supervised = recipient.clone()
    aligned_updates = train(aligned, donor_facts, teacher_outputs,
                            epochs=epochs, lr=lr, seed=83)
    wrong_updates = train(wrong, donor_facts,
                          teacher_outputs[1:] + teacher_outputs[:1],
                          epochs=epochs, lr=lr, seed=83)
    # C5 is a strong B-side compute-matched comparator, not a proposed
    # deployment path: direct labels that were available to the experimenter.
    # Unlike C3, it DOES receive true answer labels for all donor tasks.
    supervised_updates = train(supervised, donor_facts,
                               labeled_targets(supervised, donor_facts),
                               epochs=epochs, lr=lr, seed=83)
    recipients = {
        "C0_untouched_base": base.clone(),
        "B_retention_only": recipient,
        "C1_full_donor_copy": full_copy,
        "C3_distill_on_learned_B": aligned,
        "C4_wrong_pairing_on_learned_B": wrong,
        "C5_direct_labels_on_learned_B": supervised,
    }
    del donor

    with tempfile.TemporaryDirectory(prefix="ns-transfer-falsifier-") as tmp:
        outcomes = {}
        for name, model in recipients.items():
            outcomes[name] = {
                "donor_taught": evaluate_in_new_process(model, donor_facts, tmp),
                "recipient_retention": evaluate_in_new_process(model, recipient_facts, tmp),
            }

    return {
        "schema_version": 1,
        "evidence_class": "SYNTHETIC_CATEGORICAL_LOCAL_TEST_ONLY",
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "fixture_sha256": sha256([f.__dict__ for f in facts]),
        "donor_taught_count": len(donor_facts),
        "recipient_retention_count": len(recipient_facts),
        "partition_selection": "posthoc_exploratory_shared_subject_after_disjoint_subject_pilot",
        "donor_taught_subject_relation_keys": [
            [f.subject, f.relation] for f in donor_facts],
        "recipient_retention_subject_relation_keys": [
            [f.subject, f.relation] for f in recipient_facts],
        "base_sha256": base_sha,
        "donor_sha256_after_sgd": donor_sha,
        "recipient_sha256_after_own_sgd": learned_recipient_sha,
        "exact_base_delta_rejected_for_learned_recipient": delta_rejected,
        "delta_rejection_type": delta_error,
        "train_updates": {
            "donor": donor_updates["updates"],
            "C0_untouched_base": 0,
            "B_retention_only": recipient_updates["updates"],
            "C1_full_donor_copy": 0,
            "C3_aligned_distillation": aligned_updates["updates"],
            "C4_wrong_pairing": wrong_updates["updates"],
            "C5_direct_label_supervision": supervised_updates["updates"],
        },
        "recipient_only_process_evaluations": outcomes,
        "limitations": [
            "Tiny closed-vocabulary categorical neural learner, not an LLM",
            "Recipient-only holdout facts were learned by B, not generalization without teaching",
            "Donor never trained on recipient-only answer labels; distillation vectors still span all shared answer tokens",
            "Exact-base delta is inapplicable after B learns; no nontrivial weight merge proposed",
            "C3 is ordinary distribution distillation and may forget B's earlier facts",
            "C1 overwrites the recipient; it is a full-copy control, not synchronization",
            "C5 uses true donor-task answer labels and is matched only for B-side SGD update count",
            "Donor generation/training compute is not matched against C5; no total-efficiency claim",
            "No independent external training rights witness",
        ],
    }


def study(seeds: tuple[int, ...] = (11, 23, 47), *, epochs: int = 400) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or len(seeds) > 10:
        raise ValueError("require 1..10 unique seeds")
    trials = [trial(seed=s, epochs=epochs) for s in seeds]
    arms = tuple(trials[0]["recipient_only_process_evaluations"])
    return {
        "schema_version": 1,
        "evidence_class": "SYNTHETIC_CATEGORICAL_LOCAL_TEST_ONLY",
        "trials": trials,
        "mean_correct_by_arm": {
            arm: {task: sum(t["recipient_only_process_evaluations"][arm][task]["correct"]
                             for t in trials) / len(trials)
                  for task in ("donor_taught", "recipient_retention")}
            for arm in arms
        },
        "units": "training seed; separate queries within a seed are correlated",
        "statistical_claim": "posthoc descriptive only; three chosen seeds cannot prove generalization",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(study(epochs=args.epochs), indent=2, sort_keys=True))
